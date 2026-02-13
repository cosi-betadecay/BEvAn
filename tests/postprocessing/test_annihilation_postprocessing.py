import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from physics.postprocessing.annihilation import annihilation_kernel
from utils.synthetic_data_generator import SyntheticDataGenerator


def _event(p0: list[float], p1: list[float]) -> torch.Tensor:
    return torch.tensor([p0, p1], dtype=torch.float32)


def test_annihilation_kernel_peaks_for_back_to_back_tracks():
    event_1 = _event([0, 0, 0], [1, 0, 0])
    event_2 = _event([0, 0, 0], [-1, 0, 0])

    val = annihilation_kernel(event_1, event_2)

    assert torch.isclose(val, torch.tensor(1.0), atol=1e-6)


def test_annihilation_kernel_low_for_parallel_tracks():
    event_1 = _event([0, 0, 0], [1, 0, 0])
    event_2 = _event([0, 0, 0], [1, 0, 0])

    val = annihilation_kernel(event_1, event_2)

    assert val < 1e-20


def test_annihilation_kernel_low_for_orthogonal_tracks():
    event_1 = _event([0, 0, 0], [1, 0, 0])
    event_2 = _event([0, 0, 0], [0, 1, 0])

    val = annihilation_kernel(event_1, event_2)

    assert val < 1e-20


def test_annihilation_kernel_is_symmetric_under_event_swap():
    event_1 = _event([0, 0, 0], [1, 2, 0])
    event_2 = _event([0, 0, 0], [-2, -1, 0])

    val_12 = annihilation_kernel(event_1, event_2)
    val_21 = annihilation_kernel(event_2, event_1)

    assert torch.isclose(val_12, val_21, atol=1e-6)


def test_annihilation_kernel_accepts_batched_or_unbatched_inputs():
    event_1 = _event([0, 0, 0], [1, 0, 0])
    event_2 = _event([0, 0, 0], [-1, 0, 0])

    val_unbatched = annihilation_kernel(event_1, event_2)
    val_batched = annihilation_kernel(event_1.unsqueeze(0), event_2.unsqueeze(0))

    assert torch.isclose(val_unbatched, val_batched, atol=1e-6)


def test_annihilation_kernel_output_is_bounded():
    event_1 = _event([0, 0, 0], [1, 1, 0])
    event_2 = _event([0, 0, 0], [1, -1, 0])

    val = annihilation_kernel(event_1, event_2)

    assert val >= 0.0
    assert val <= 1.0


def test_annihilation_kernel_handles_near_zero_displacements():
    event_1 = _event([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    event_2 = _event([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    val = annihilation_kernel(event_1, event_2)

    assert torch.isfinite(val)
    assert val >= 0.0
    assert val <= 1.0


def test_annihilation_kernel_uses_first_batch_entry():
    event_1 = torch.tensor(
        [
            [[0, 0, 0], [1, 0, 0]],  # anti-parallel with event_2[0] -> high score
            [[0, 0, 0], [0, 1, 0]],  # ignored by current implementation
        ],
        dtype=torch.float32,
    )
    event_2 = torch.tensor(
        [
            [[0, 0, 0], [-1, 0, 0]],
            [[0, 0, 0], [0, 1, 0]],
        ],
        dtype=torch.float32,
    )

    val = annihilation_kernel(event_1, event_2)

    assert torch.isclose(val, torch.tensor(1.0), atol=1e-6)


def test_annihilation_kernel_true_mode_samples_are_finite_and_bounded():
    _, positions_true = SyntheticDataGenerator(max_hits_per_sequence=2, seed=0).generate(
        num_samples=32, mode=1
    )

    vals = torch.stack(
        [
            annihilation_kernel(positions_true[i], positions_true[i + 1])
            for i in range(0, positions_true.shape[0] - 1, 2)
        ]
    )

    assert torch.isfinite(vals).all()
    assert torch.all(vals >= 0.0)
    assert torch.all(vals <= 1.0)


def test_annihilation_kernel_false_mode_samples_are_finite_and_bounded():
    _, positions_false = SyntheticDataGenerator(max_hits_per_sequence=2, seed=0).generate(
        num_samples=32, mode=2
    )

    vals = torch.stack(
        [
            annihilation_kernel(positions_false[i], positions_false[i + 1])
            for i in range(0, positions_false.shape[0] - 1, 2)
        ]
    )

    assert torch.isfinite(vals).all()
    assert torch.all(vals >= 0.0)
    assert torch.all(vals <= 1.0)
