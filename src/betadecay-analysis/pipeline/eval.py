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
            bdecay_lor,
            bg_delta_E,
            bg_annihilation_angle,
            bg_arm,
            bg_lor,
            combined_delta_E,
            combined_annihilation_angle,
            combined_arm,
            combined_lor,
        ) = data
        t = self.trainer
        double_count = bool(t.cfg.modeling.double_count_deltaE)

        if double_count:
            # Double-counting mode: look up the two 2D joints directly.
            # P(ΔE, angle) and P(ΔE, ARM) — ΔE evidence will appear in both
            # factors of the likelihood ratio.
            p_beta_deltaE_angle = lookup_density_values(
                combined_delta_E,
                combined_annihilation_angle,
                t.bdecay_joint_deltaE_angle,
                t.deltaE_angle_bins,
                t.angle_bins,
            )
            p_bg_deltaE_angle = lookup_density_values(
                combined_delta_E,
                combined_annihilation_angle,
                t.bg_joint_deltaE_angle,
                t.deltaE_angle_bins,
                t.angle_bins,
            )
            p_beta_deltaE_arm = lookup_density_values(
                combined_delta_E,
                combined_arm,
                t.bdecay_joint_deltaE_arm,
                t.deltaE_arm_bins,
                t.arm_bins,
            )
            p_bg_deltaE_arm = lookup_density_values(
                combined_delta_E,
                combined_arm,
                t.bg_joint_deltaE_arm,
                t.deltaE_arm_bins,
                t.arm_bins,
            )
            # The lor term uses the CONDITIONAL P(lor | ΔE), not the joint:
            # the angle and arm joints already give ΔE two votes, and the
            # residual FP class is 511-like in ΔE — a third ΔE vote feeds it.
            p_beta_deltaE_lor = lookup_density_values(
                combined_delta_E,
                combined_lor,
                t.bdecay_cond_lor,
                t.deltaE_lor_bins,
                t.lor_bins,
            )
            p_bg_deltaE_lor = lookup_density_values(
                combined_delta_E,
                combined_lor,
                t.bg_cond_lor,
                t.deltaE_lor_bins,
                t.lor_bins,
            )

            predict(
                ground_truths,
                [
                    (p_beta_deltaE_angle, p_bg_deltaE_angle),  # P(ΔE, angle)
                    (p_beta_deltaE_arm, p_bg_deltaE_arm),  # P(ΔE, ARM)
                    (p_beta_deltaE_lor, p_bg_deltaE_lor),  # P(lor | ΔE)
                ],
                t.n_beta_decay,
                t.n_bg,
                split_name=split_name,
            )
            return

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
        p_beta_lor = lookup_density_values(
            combined_delta_E,
            combined_lor,
            t.bdecay_cond_lor,
            t.deltaE_lor_bins,
            t.lor_bins,
        )
        p_bg_lor = lookup_density_values(
            combined_delta_E,
            combined_lor,
            t.bg_cond_lor,
            t.deltaE_lor_bins,
            t.lor_bins,
        )

        predict(
            ground_truths,
            [
                (p_beta_deltaE, p_bg_deltaE),  # P(ΔE)
                (p_beta_angle, p_bg_angle),  # P(angle | ΔE)
                (p_beta_arm, p_bg_arm),  # P(ARM | ΔE)
                (p_beta_lor, p_bg_lor),  # P(lor | ΔE)
            ],
            t.n_beta_decay,
            t.n_bg,
            split_name=split_name,
        )
