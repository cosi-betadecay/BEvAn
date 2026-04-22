from modeling.calculate_probablities import predict
from modeling.matrix_calculations import (
    lookup_density_values,
    lookup_density_values_1d,
)


class Evaluator:
    def __init__(self, trainer):
        self.trainer = trainer

    def evaluate(self, data, split_name: str = "eval"):
        (
            ground_truths,
            bdecay_delta_E,
            bdecay_annihilation_angle,
            bdecay_arm,
            bg_delta_E,
            bg_annihilation_angle,
            bg_arm,
            combined_delta_E,
            combined_annihilation_angle,
            combined_arm,
        ) = data
        t = self.trainer

        p_beta_deltaE = lookup_density_values_1d(
            combined_delta_E, t.bdecay_marginal_deltaE, t.deltaE_angle_bins
        )
        p_bg_deltaE = lookup_density_values_1d(combined_delta_E, t.bg_marginal_deltaE, t.deltaE_angle_bins)
        p_beta_angle = lookup_density_values(
            combined_delta_E,
            combined_annihilation_angle,
            t.bdecay_cond_angle,
            t.deltaE_angle_bins,
            t.angle_bins,
        )
        p_bg_angle = lookup_density_values(
            combined_delta_E,
            combined_annihilation_angle,
            t.bg_cond_angle,
            t.deltaE_angle_bins,
            t.angle_bins,
        )
        p_beta_arm = lookup_density_values(
            combined_delta_E,
            combined_arm,
            t.bdecay_cond_arm,
            t.deltaE_arm_bins,
            t.arm_bins,
        )
        p_bg_arm = lookup_density_values(
            combined_delta_E,
            combined_arm,
            t.bg_cond_arm,
            t.deltaE_arm_bins,
            t.arm_bins,
        )

        predict(
            ground_truths,
            p_beta_deltaE,
            p_bg_deltaE,
            p_beta_angle,
            p_bg_angle,
            p_beta_arm,
            p_bg_arm,
            t.n_beta_decay,
            t.n_bg,
            split_name=split_name,
        )
