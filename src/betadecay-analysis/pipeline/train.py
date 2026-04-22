"""
This code should take in the 80% data & the according point source vector from compton
cones from the dataloader.py, and accordingly fill up the matrices in /physics.
Also do inference from the bayesian network for the 80% data to get that too.
"""

import torch
from physics.matrix_calculations import (
    build_density_matrix,
    build_density_matrix_1d,
    conditional_from_joint,
)


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, train):
        (
            _ground_truths_train,
            bdecay_train_delta_E,
            bdecay_train_annihilation_angle,
            bdecay_train_arm,
            bg_train_delta_E,
            bg_train_annihilation_angle,
            bg_train_arm,
            combined_train_delta_E,
            combined_train_annihilation_angle,
            combined_train_arm,
        ) = train

        _, self.deltaE_angle_bins, self.angle_bins = build_density_matrix(
            combined_train_delta_E,
            combined_train_annihilation_angle,
            spacing_x="log",
            spacing_y="linear",
            log_x_floor=1e-3,
            n_bins_x=200,
            n_bins_y=200,
        )
        _, self.deltaE_arm_bins, self.arm_bins = build_density_matrix(
            combined_train_delta_E,
            combined_train_arm,
            spacing_x="log",
            spacing_y="log",
            log_x_floor=1e-3,
            log_y_floor=1e-3,
            n_bins_x=200,
            n_bins_y=200,
        )

        bdecay_joint_deltaE_angle, _, _ = build_density_matrix(
            bdecay_train_delta_E,
            bdecay_train_annihilation_angle,
            x_bins=self.deltaE_angle_bins,
            y_bins=self.angle_bins,
            smoothing=0.0,
        )
        bdecay_joint_deltaE_arm, _, _ = build_density_matrix(
            bdecay_train_delta_E,
            bdecay_train_arm,
            x_bins=self.deltaE_arm_bins,
            y_bins=self.arm_bins,
            smoothing=0.0,
        )
        bg_joint_deltaE_angle, _, _ = build_density_matrix(
            bg_train_delta_E,
            bg_train_annihilation_angle,
            x_bins=self.deltaE_angle_bins,
            y_bins=self.angle_bins,
            smoothing=0.0,
        )
        bg_joint_deltaE_arm, _, _ = build_density_matrix(
            bg_train_delta_E,
            bg_train_arm,
            x_bins=self.deltaE_arm_bins,
            y_bins=self.arm_bins,
            smoothing=0.0,
        )

        self.bdecay_marginal_deltaE, _ = build_density_matrix_1d(
            bdecay_train_delta_E, x_bins=self.deltaE_angle_bins, smoothing=1.0
        )
        self.bg_marginal_deltaE, _ = build_density_matrix_1d(
            bg_train_delta_E, x_bins=self.deltaE_angle_bins, smoothing=1.0
        )

        self.bdecay_cond_angle = conditional_from_joint(bdecay_joint_deltaE_angle, axis=0)
        self.bdecay_cond_arm = conditional_from_joint(bdecay_joint_deltaE_arm, axis=0)
        self.bg_cond_angle = conditional_from_joint(bg_joint_deltaE_angle, axis=0)
        self.bg_cond_arm = conditional_from_joint(bg_joint_deltaE_arm, axis=0)

        self.n_beta_decay = int(torch.isfinite(bdecay_train_delta_E).sum().item())
        self.n_bg = int(torch.isfinite(bg_train_delta_E).sum().item())

        from pipeline.eval import Evaluator

        Evaluator(self).evaluate(train, split_name="train")

        return self
