import torch
from modeling.matrix_calculations import (
    build_density_matrix,
    build_density_matrix_1d,
    conditional_from_joint,
)

from pipeline.eval import Evaluator


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, train):
        (
            _ground_truths_train,
            bdecay_train_delta_E,
            bdecay_train_annihilation_angle,
            bdecay_train_arm,
            bdecay_train_lor,
            bg_train_delta_E,
            bg_train_annihilation_angle,
            bg_train_arm,
            bg_train_lor,
            combined_train_delta_E,
            combined_train_annihilation_angle,
            combined_train_arm,
            combined_train_lor,
        ) = train

        _, self.deltaE_angle_bins, self.angle_bins = build_density_matrix(
            combined_train_delta_E,
            combined_train_annihilation_angle,
            spacing_x="log",
            spacing_y="linear",
            log_x_floor=1e-3,
            n_bins_x=25,
            n_bins_y=25,
        )
        _, self.deltaE_arm_bins, self.arm_bins = build_density_matrix(
            combined_train_delta_E,
            combined_train_arm,
            spacing_x="log",
            spacing_y="log",
            log_x_floor=1e-3,
            log_y_floor=1e-3,
            n_bins_x=25,
            n_bins_y=25,
        )
        # LOR score axis: log-spaced with a 1e-3 floor so the degenerate
        # zero point-mass clamps into the bottom bin — that bin carries the
        # ~6:1 bg-ward evidence and must stay its own cell.
        _, self.deltaE_lor_bins, self.lor_bins = build_density_matrix(
            combined_train_delta_E,
            combined_train_lor,
            spacing_x="log",
            spacing_y="log",
            log_x_floor=0.1,
            log_y_floor=1e-3,
            n_bins_x=25,
            n_bins_y=25,
        )

        self.bdecay_joint_deltaE_angle, _, _ = build_density_matrix(
            bdecay_train_delta_E,
            bdecay_train_annihilation_angle,
            x_bins=self.deltaE_angle_bins,
            y_bins=self.angle_bins,
            smoothing=0,
        )
        self.bdecay_joint_deltaE_arm, _, _ = build_density_matrix(
            bdecay_train_delta_E,
            bdecay_train_arm,
            x_bins=self.deltaE_arm_bins,
            y_bins=self.arm_bins,
            smoothing=0,
        )
        self.bg_joint_deltaE_angle, _, _ = build_density_matrix(
            bg_train_delta_E,
            bg_train_annihilation_angle,
            x_bins=self.deltaE_angle_bins,
            y_bins=self.angle_bins,
            smoothing=0,
        )
        self.bg_joint_deltaE_arm, _, _ = build_density_matrix(
            bg_train_delta_E,
            bg_train_arm,
            x_bins=self.deltaE_arm_bins,
            y_bins=self.arm_bins,
            smoothing=0,
        )
        self.bdecay_joint_deltaE_lor, _, _ = build_density_matrix(
            bdecay_train_delta_E,
            bdecay_train_lor,
            x_bins=self.deltaE_lor_bins,
            y_bins=self.lor_bins,
            smoothing=0.5,
        )
        self.bg_joint_deltaE_lor, _, _ = build_density_matrix(
            bg_train_delta_E,
            bg_train_lor,
            x_bins=self.deltaE_lor_bins,
            y_bins=self.lor_bins,
            smoothing=0.5,
        )

        self.bdecay_marginal_deltaE, _ = build_density_matrix_1d(
            bdecay_train_delta_E, x_bins=self.deltaE_angle_bins, smoothing=1.0
        )
        self.bg_marginal_deltaE, _ = build_density_matrix_1d(
            bg_train_delta_E, x_bins=self.deltaE_angle_bins, smoothing=1.0
        )

        self.bdecay_cond_angle = conditional_from_joint(self.bdecay_joint_deltaE_angle, axis=0)
        self.bdecay_cond_arm = conditional_from_joint(self.bdecay_joint_deltaE_arm, axis=0)
        self.bdecay_cond_lor = conditional_from_joint(self.bdecay_joint_deltaE_lor, axis=0)
        self.bg_cond_angle = conditional_from_joint(self.bg_joint_deltaE_angle, axis=0)
        self.bg_cond_arm = conditional_from_joint(self.bg_joint_deltaE_arm, axis=0)
        self.bg_cond_lor = conditional_from_joint(self.bg_joint_deltaE_lor, axis=0)

        # Class prior P(β)/P(bg) must reflect the true population balance, not
        # the subset of events that happened to yield finite features. Counting
        # over isfinite(...) biased the prior toward whichever class less often
        # produced NaN features and made the denominator inconsistent with the
        # eval population the prior is applied to. Use the full class counts.
        self.n_beta_decay = int(bdecay_train_delta_E.numel())
        self.n_bg = int(bg_train_delta_E.numel())

        Evaluator(self).evaluate(train, split_name="train")

        return self
