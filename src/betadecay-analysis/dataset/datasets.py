from typing import Any

import omegaconf
import ROOT as M
import torch
from physics.event_processing import event_data_processing
from physics.ground_truths import ground_truth_bdecay
from physics.physics_factors import annihilation_angle, arm, delta_E
from tqdm import tqdm


class Datasets:
    def __init__(
        self,
        cfg: omegaconf.dictconfig.DictConfig,
        ref_energy: float,
        reader_sim: Any,
        reader_tra: Any,
        reconstructed_unit_vector: torch.Tensor,
    ):
        self.cfg = cfg
        self.ref_energy = ref_energy
        self.reader_sim = reader_sim
        self.reader_tra = reader_tra
        self.reconstructed_unit_vector = reconstructed_unit_vector
        self.train_percentage = 0.8
        self.eval_percentage = 0.2

    def compute_event_features(self, cfg, ref_energy, reader_sim, reader_tra, reconstructed_unit_vector):
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

        for event_sim in tqdm(
            iter(lambda: reader_sim.GetNextEvent(), None),
            desc="Processing events",
            unit=" events",
        ):
            M.SetOwnership(event_sim, True)
            n_hits = event_sim.GetNHTs()

            if n_hits == 0:
                continue

            gt = ground_truth_bdecay(event_sim, ref_energy)
            ground_truths.append(gt)

            event_tra = reader_tra.GetNextEvent()
            M.SetOwnership(event_tra, True)

            assert event_sim.GetID() == event_tra.GetId(), (
                f"Mismatched IDs: sim={event_sim.GetID()} tra={event_tra.GetId()}"
            )

            energies, positions, sizes = event_data_processing(event_tra)
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
            _arm = arm(event_tra, reconstructed_unit_vector)

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
