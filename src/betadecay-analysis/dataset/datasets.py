from typing import Any

import omegaconf
import ROOT as M
import torch
from physics.annihilation_detection_utils import (
    build_density_matrix,
    build_density_matrix_1d,
    conditional_from_joint,
)
from physics.event_processing import event_data_processing
from physics.ground_truths import ground_truth_bdecay
from physics.matrix_calculations_factors import annihilation_angle, arm, delta_E
from tqdm import tqdm


class Datasets:
    def __init__(
        self,
        cfg: omegaconf.dictconfig.DictConfig,
        ref_energy: float,
        reader: Any,
        reconstructed_unit_vector: torch.Tensor,
    ):
        self.cfg = cfg
        self.ref_energy = ref_energy
        self.reader = reader
        self.reconstructed_unit_vector = reconstructed_unit_vector
        self.train_percentage = 0.8
        self.eval_percentage = 0.2

    def compute_event_features(self, cfg, ref_energy, reader, reconstructed_unit_vector):
        ground_truths = []

        # Lists beta decay
        bdecay_list_delta_E = []
        bdecay_list_annihilation_angle = []
        bdecay_list_arm = []
        # Lists bg
        bg_list_delta_E = []
        bg_list_annihilation_angle = []
        bg_list_arm = []
        # Lists general
        gen_list_delta_E = []
        gen_list_annihilation_angle = []
        gen_list_arm = []

        for event in tqdm(
            iter(lambda: reader.GetNextEvent(), None),
            desc="Processing events",
            unit=" events",
        ):
            M.SetOwnership(event, True)
            n_hits = event.GetNHTs()

            gt = ground_truth_bdecay(event, ref_energy)
            ground_truths.append(gt)

            energies, positions, sizes = event_data_processing(event)
            if energies is None or positions is None or sizes is None:
                gen_list_delta_E.append(float("nan"))
                gen_list_annihilation_angle.append(float("nan"))
                gen_list_arm.append(float("nan"))
                if gt:
                    bdecay_list_delta_E.append(float("nan"))
                    bdecay_list_annihilation_angle.append(float("nan"))
                    bdecay_list_arm.append(float("nan"))
                else:
                    bg_list_delta_E.append(float("nan"))
                    bg_list_annihilation_angle.append(float("nan"))
                    bg_list_arm.append(float("nan"))
                continue

            _delta_E = delta_E(energies, sizes=sizes)
            _annihilation_angle = annihilation_angle(positions, n_hits=n_hits, sizes=sizes)
            _arm = arm(energies, positions, cfg, reconstructed_unit_vector, sizes=sizes)

            gen_list_delta_E.append(_delta_E)
            gen_list_annihilation_angle.append(_annihilation_angle)
            gen_list_arm.append(_arm)

            if gt:
                bdecay_list_delta_E.append(_delta_E)
                bdecay_list_annihilation_angle.append(_annihilation_angle)
                bdecay_list_arm.append(_arm)
            else:
                bg_list_delta_E.append(_delta_E)
                bg_list_annihilation_angle.append(_annihilation_angle)
                bg_list_arm.append(_arm)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Ground truth
        ground_truths = torch.tensor(ground_truths, dtype=torch.bool)
        # Converting beta decay lists to tensors
        bdecay_tensor_delta_E = torch.tensor(bdecay_list_delta_E, dtype=torch.float32)
        bdecay_tensor_annihilation_angle = torch.tensor(bdecay_list_annihilation_angle, dtype=torch.float32)
        bdecay_tensor_arm = torch.tensor(bdecay_list_arm, dtype=torch.float32)
        # Converting bg lists to tensors
        bg_tensor_delta_E = torch.tensor(bg_list_delta_E, dtype=torch.float32)
        bg_tensor_annihilation_angle = torch.tensor(bg_list_annihilation_angle, dtype=torch.float32)
        bg_tensor_arm = torch.tensor(bg_list_arm, dtype=torch.float32)
        # Converting general lists to tensors
        gen_tensor_delta_E = torch.tensor(gen_list_delta_E, dtype=torch.float32)
        gen_tensor_annihilation_angle = torch.tensor(gen_list_annihilation_angle, dtype=torch.float32)
        gen_tensor_arm = torch.tensor(gen_list_arm, dtype=torch.float32)

        combined_tensor_delta_E = torch.cat([bdecay_tensor_delta_E, bg_tensor_delta_E])
        combined_tensor_annihilation_angle = torch.cat(
            [bdecay_tensor_annihilation_angle, bg_tensor_annihilation_angle]
        )
        combined_tensor_arm = torch.cat([bdecay_tensor_arm, bg_tensor_arm])
        return (
            ground_truths,
            bdecay_tensor_delta_E,
            bdecay_tensor_annihilation_angle,
            bdecay_tensor_arm,
            bg_tensor_delta_E,
            bg_tensor_annihilation_angle,
            bg_tensor_arm,
            gen_tensor_delta_E,
            gen_tensor_annihilation_angle,
            gen_tensor_arm,
            combined_tensor_delta_E,
            combined_tensor_annihilation_angle,
            combined_tensor_arm,
        )

    def split_dataset(
        self,
        bdecay_tensor_delta_E,
        bdecay_tensor_annihilation_angle,
        bdecay_tensor_arm,
        bg_tensor_delta_E,
        bg_tensor_annihilation_angle,
        bg_tensor_arm,
    ):
        generator = torch.Generator().manual_seed(42)

        n_bdecay = bdecay_tensor_delta_E.shape[0]
        n_bg = bg_tensor_delta_E.shape[0]

        bdecay_perm = torch.randperm(n_bdecay, generator=generator)
        bg_perm = torch.randperm(n_bg, generator=generator)

        n_bdecay_train = int(n_bdecay * self.train_percentage)
        n_bg_train = int(n_bg * self.train_percentage)

        bdecay_train_idx = bdecay_perm[:n_bdecay_train]
        bdecay_eval_idx = bdecay_perm[n_bdecay_train:]
        bg_train_idx = bg_perm[:n_bg_train]
        bg_eval_idx = bg_perm[n_bg_train:]

        bdecay_train_delta_E = bdecay_tensor_delta_E[bdecay_train_idx]
        bdecay_train_annihilation_angle = bdecay_tensor_annihilation_angle[bdecay_train_idx]
        bdecay_train_arm = bdecay_tensor_arm[bdecay_train_idx]
        bdecay_eval_delta_E = bdecay_tensor_delta_E[bdecay_eval_idx]
        bdecay_eval_annihilation_angle = bdecay_tensor_annihilation_angle[bdecay_eval_idx]
        bdecay_eval_arm = bdecay_tensor_arm[bdecay_eval_idx]

        bg_train_delta_E = bg_tensor_delta_E[bg_train_idx]
        bg_train_annihilation_angle = bg_tensor_annihilation_angle[bg_train_idx]
        bg_train_arm = bg_tensor_arm[bg_train_idx]
        bg_eval_delta_E = bg_tensor_delta_E[bg_eval_idx]
        bg_eval_annihilation_angle = bg_tensor_annihilation_angle[bg_eval_idx]
        bg_eval_arm = bg_tensor_arm[bg_eval_idx]

        combined_train_delta_E = torch.cat([bdecay_train_delta_E, bg_train_delta_E])
        combined_train_annihilation_angle = torch.cat(
            [bdecay_train_annihilation_angle, bg_train_annihilation_angle]
        )
        combined_train_arm = torch.cat([bdecay_train_arm, bg_train_arm])
        combined_eval_delta_E = torch.cat([bdecay_eval_delta_E, bg_eval_delta_E])
        combined_eval_annihilation_angle = torch.cat(
            [bdecay_eval_annihilation_angle, bg_eval_annihilation_angle]
        )
        combined_eval_arm = torch.cat([bdecay_eval_arm, bg_eval_arm])

        ground_truths_train = torch.cat(
            [
                torch.ones(bdecay_train_delta_E.shape[0], dtype=torch.bool),
                torch.zeros(bg_train_delta_E.shape[0], dtype=torch.bool),
            ]
        )
        ground_truths_eval = torch.cat(
            [
                torch.ones(bdecay_eval_delta_E.shape[0], dtype=torch.bool),
                torch.zeros(bg_eval_delta_E.shape[0], dtype=torch.bool),
            ]
        )

        trainer = (
            ground_truths_train,
            bdecay_train_delta_E,
            bdecay_train_annihilation_angle,
            bdecay_train_arm,
            bg_train_delta_E,
            bg_train_annihilation_angle,
            bg_train_arm,
            combined_train_delta_E,
            combined_train_annihilation_angle,
            combined_train_arm,
        )
        evaluator = (
            ground_truths_eval,
            bdecay_eval_delta_E,
            bdecay_eval_annihilation_angle,
            bdecay_eval_arm,
            bg_eval_delta_E,
            bg_eval_annihilation_angle,
            bg_eval_arm,
            combined_eval_delta_E,
            combined_eval_annihilation_angle,
            combined_eval_arm,
        )
        return trainer, evaluator

    def build_density_matrices(
        self,
        bdecay_tensor_delta_E,
        bdecay_tensor_annihilation_angle,
        bdecay_tensor_arm,
        bg_tensor_delta_E,
        bg_tensor_annihilation_angle,
        bg_tensor_arm,
        combined_tensor_delta_E,
        combined_tensor_annihilation_angle,
        combined_tensor_arm,
    ):
        _, deltaE_angle_bins, angle_bins = build_density_matrix(
            combined_tensor_delta_E,
            combined_tensor_annihilation_angle,
            spacing_x="log",
            spacing_y="linear",
            log_x_floor=1e-3,
            n_bins_x=200,
            n_bins_y=200,
        )
        _, deltaE_arm_bins, arm_bins = build_density_matrix(
            combined_tensor_delta_E,
            combined_tensor_arm,
            spacing_x="log",
            spacing_y="log",
            log_x_floor=1e-3,
            log_y_floor=1e-3,
            n_bins_x=200,
            n_bins_y=200,
        )

        bdecay_joint_deltaE_angle, _, _ = build_density_matrix(
            bdecay_tensor_delta_E,
            bdecay_tensor_annihilation_angle,
            x_bins=deltaE_angle_bins,
            y_bins=angle_bins,
            smoothing=0.0,
        )
        bdecay_joint_deltaE_arm, _, _ = build_density_matrix(
            bdecay_tensor_delta_E,
            bdecay_tensor_arm,
            x_bins=deltaE_arm_bins,
            y_bins=arm_bins,
            smoothing=0.0,
        )
        bg_joint_deltaE_angle, _, _ = build_density_matrix(
            bg_tensor_delta_E,
            bg_tensor_annihilation_angle,
            x_bins=deltaE_angle_bins,
            y_bins=angle_bins,
            smoothing=0.0,
        )
        bg_joint_deltaE_arm, _, _ = build_density_matrix(
            bg_tensor_delta_E,
            bg_tensor_arm,
            x_bins=deltaE_arm_bins,
            y_bins=arm_bins,
            smoothing=0.0,
        )

        bdecay_marginal_deltaE_angle_grid, _ = build_density_matrix_1d(
            bdecay_tensor_delta_E, x_bins=deltaE_angle_bins, smoothing=1.0
        )
        bg_marginal_deltaE_angle_grid, _ = build_density_matrix_1d(
            bg_tensor_delta_E, x_bins=deltaE_angle_bins, smoothing=1.0
        )

        bdecay_cond_angle = conditional_from_joint(bdecay_joint_deltaE_angle, axis=0)
        bdecay_cond_arm = conditional_from_joint(bdecay_joint_deltaE_arm, axis=0)
        bg_cond_angle = conditional_from_joint(bg_joint_deltaE_angle, axis=0)
        bg_cond_arm = conditional_from_joint(bg_joint_deltaE_arm, axis=0)

        return (
            bdecay_marginal_deltaE_angle_grid,
            bg_marginal_deltaE_angle_grid,
            bdecay_cond_angle,
            bdecay_cond_arm,
            bg_cond_angle,
            bg_cond_arm,
        )
