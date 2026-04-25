import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import wandb
from omegaconf import OmegaConf
from physics.physics_factors import (
    all_vector_cosines_blockwise,
    all_vector_cosines_matrix_calculation,
    annihilation_angle,
    arm,
    delta_E,
)


def _cfg(sigma_x: float = 4.0, n_sigma_cos_theta_kin: int = 1):
    return OmegaConf.create(
        {
            "likelihoods": {
                "arm": {
                    "sigma_x": sigma_x,
                    "n_sigma_cos_theta_kin": n_sigma_cos_theta_kin,
                }
            }
        }
    )


# ---------------------------------------------------------------------------
# delta_E
# ---------------------------------------------------------------------------


def test_delta_E_zero_when_sum_matches_reference():
    """ΣE = ref_energy ⇒ Δ = 0."""
    energies = torch.tensor([200.0, 311.0])
    out = delta_E(energies, ref_energy=511.0)
    assert torch.allclose(out, torch.tensor([0.0]), atol=1e-6)


def test_delta_E_returns_abs_difference_from_reference():
    energies = torch.tensor([100.0, 200.0])  # sum = 300
    out = delta_E(energies, ref_energy=511.0)
    assert torch.allclose(out, torch.tensor([211.0]), atol=1e-6)


def test_delta_E_custom_reference_energy():
    energies = torch.tensor([50.0, 50.0])  # sum = 100
    out = delta_E(energies, ref_energy=80.0)
    assert torch.allclose(out, torch.tensor([20.0]), atol=1e-6)


def test_delta_E_2d_input_returns_min_across_batch():
    """delta_E on a (B, N) input returns the min |ΣE − ref| across the
    batch — the "best" candidate split per event."""
    energies = torch.tensor(
        [
            [200.0, 200.0],  # sum 400 → diff 111
            [255.0, 256.0],  # sum 511 → diff   0
            [1000.0, 0.0],  # sum 1000 → diff 489
        ]
    )
    out = delta_E(energies, ref_energy=511.0)
    assert torch.allclose(out, torch.tensor([0.0]), atol=1e-6)


def test_delta_E_with_sizes_ignores_padding():
    """Padded NaN-like entries (beyond each row's size) must be ignored."""
    energies = torch.tensor(
        [
            [200.0, 311.0, 999.0],  # size 2 → use first 2, sum=511, diff=0
            [200.0, 200.0, 200.0],  # size 3 → sum=600, diff=89
        ]
    )
    sizes = torch.tensor([2, 3])
    out = delta_E(energies, ref_energy=511.0, sizes=sizes)
    assert torch.allclose(out, torch.tensor([0.0]), atol=1e-6)


def test_delta_E_with_sizes_invalid_shape_raises():
    energies = torch.tensor([1.0, 2.0, 3.0])  # 1D, not 2D
    with pytest.raises(ValueError, match="Sized delta_E expects"):
        delta_E(energies, sizes=torch.tensor([2]))


def test_delta_E_with_sizes_batch_mismatch_raises():
    energies = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # B=2
    sizes = torch.tensor([2, 2, 2])  # B=3
    with pytest.raises(ValueError, match="batch dimension and sizes length must match"):
        delta_E(energies, sizes=sizes)


# ---------------------------------------------------------------------------
# annihilation_angle
# ---------------------------------------------------------------------------


def test_annihilation_angle_three_collinear_hits_give_cos_one():
    """Three hits on a straight line (i<j indexing) all yield pair-vectors in
    the same direction, so every pair-of-pair-vectors cosine is +1 and the
    minimum is +1."""
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    out = annihilation_angle(positions)
    assert torch.allclose(out, torch.tensor([1.0], dtype=torch.float64), atol=1e-6)


def test_annihilation_angle_anti_parallel_pair_vectors_give_cos_minus_one():
    """Hits at (0,0,0), (1,0,0), (-1,0,0), (0,1,0) produce two pair-vectors
    exactly anti-parallel (h1−h0 and h2−h0), giving a min cosine of −1."""
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    out = annihilation_angle(positions)
    assert torch.allclose(out, torch.tensor([-1.0], dtype=torch.float64), atol=1e-6)


def test_annihilation_angle_returns_finite_for_random_three_hits():
    g = torch.Generator().manual_seed(0)
    positions = torch.randn(3, 3, generator=g, dtype=torch.float64) * 5.0
    out = annihilation_angle(positions)
    assert torch.isfinite(out).item()


def test_annihilation_angle_3d_batched_input():
    """(B, N, 3) inputs must be accepted; returns the global min across the
    batch."""
    g = torch.Generator().manual_seed(1)
    positions = torch.randn(4, 3, 3, generator=g, dtype=torch.float64)
    out = annihilation_angle(positions)
    assert out.numel() == 1
    assert torch.isfinite(out).item()


def test_annihilation_angle_with_sizes_size_below_3_returns_nan_for_that_slot():
    """Events with fewer than 3 hits can't form a pair-of-pairs cosine and
    must report NaN. If ALL slots are NaN, the output is NaN."""
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    sizes = torch.tensor([2])  # only 2 real hits → should NaN out
    out = annihilation_angle(positions, sizes=sizes)
    assert torch.isnan(out).item()


def test_annihilation_angle_with_sizes_takes_min_across_sized_groups():
    """Two events grouped by size=3. The second has perpendicular pair-vectors
    (min cosine = −0.707); the first has parallel pair-vectors (min = +1).
    The batched output must equal the smaller of the two."""
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],  # collinear → cos = 1
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # L-shaped → cos ≈ -0.707
        ],
        dtype=torch.float64,
    )
    sizes = torch.tensor([3, 3])
    out_batched = annihilation_angle(positions, sizes=sizes)
    out_l_shape = annihilation_angle(positions[1], n_hits=3)
    assert torch.isclose(out_batched, out_l_shape, atol=1e-6).item()
    assert out_batched.item() < 0.0, "L-shaped triangle should give a negative min cosine"


def test_annihilation_angle_with_sizes_invalid_shape_raises():
    positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)  # 2D not 3D
    with pytest.raises(ValueError, match="Sized annihilation_angle expects"):
        annihilation_angle(positions, sizes=torch.tensor([1]))


def test_annihilation_angle_2d_input_accepted_without_sizes():
    """A bare (N, 3) tensor is treated as a single event."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    out = annihilation_angle(positions)
    assert out.numel() == 1


# ---------------------------------------------------------------------------
# arm
# ---------------------------------------------------------------------------


def test_arm_returns_finite_for_valid_geometry():
    g = torch.Generator().manual_seed(7)
    energies = torch.tensor([[300.0, 211.0]], dtype=torch.float64)
    positions = torch.randn(1, 2, 3, generator=g, dtype=torch.float64) + 5.0
    v_in = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    out = arm(energies, positions, _cfg(), v_in)
    assert torch.isfinite(out).item()


def test_arm_is_non_negative():
    """ARM is |θ_geo − θ_kin|; always ≥ 0."""
    g = torch.Generator().manual_seed(3)
    energies = torch.rand(1, 2, generator=g, dtype=torch.float64) * 500 + 100
    positions = torch.randn(1, 2, 3, generator=g, dtype=torch.float64) * 5.0
    v_in = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    out = arm(energies, positions, _cfg(), v_in)
    assert out.item() >= 0.0


def test_arm_with_sizes_size_below_2_returns_nan():
    energies = torch.tensor([[100.0, 0.0]], dtype=torch.float64)
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float64)
    sizes = torch.tensor([1])  # < 2 hits
    v_in = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    out = arm(energies, positions, _cfg(), v_in, sizes=sizes)
    assert torch.isnan(out).item()


def test_arm_with_sizes_invalid_shape_raises():
    energies = torch.tensor([100.0, 200.0], dtype=torch.float64)  # 1D not 2D
    positions = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float64)
    v_in = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="Sized arm expects"):
        arm(energies, positions, _cfg(), v_in, sizes=torch.tensor([2]))


def test_arm_with_sizes_takes_min_across_sized_groups():
    """Two events, same size — the output is the min ARM over the batch."""
    energies = torch.tensor(
        [
            [300.0, 211.0],
            [300.0, 211.0],
        ],
        dtype=torch.float64,
    )
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    sizes = torch.tensor([2, 2])
    v_in = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    out_batched = arm(energies, positions, _cfg(), v_in, sizes=sizes)
    out_first = arm(energies[0:1], positions[0:1], _cfg(), v_in)
    out_second = arm(energies[1:2], positions[1:2], _cfg(), v_in)
    assert torch.isclose(out_batched, torch.minimum(out_first, out_second), atol=1e-6).item()


# ---------------------------------------------------------------------------
# all_vector_cosines — the low-level helpers used by annihilation_angle
# ---------------------------------------------------------------------------


def test_all_vector_cosines_three_collinear_hits_give_cos_one():
    """Colinear hits produce parallel pair-vectors ⇒ all pairwise cosines = +1
    ⇒ amin returns +1."""
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    out = all_vector_cosines_matrix_calculation(positions)
    assert torch.allclose(out, torch.tensor([1.0], dtype=torch.float64), atol=1e-6)


def test_all_vector_cosines_finds_anti_parallel_pair():
    """When two pair-vectors are exactly anti-parallel, amin must land at −1."""
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    out = all_vector_cosines_matrix_calculation(positions)
    assert torch.allclose(out, torch.tensor([-1.0], dtype=torch.float64), atol=1e-6)


def test_all_vector_cosines_n_less_than_3_returns_empty():
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    out = all_vector_cosines_matrix_calculation(positions)
    assert out.numel() == 0


def test_all_vector_cosines_rejects_wrong_ndim():
    bad = torch.zeros(4, 3, 3, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="Expected positions shape"):
        all_vector_cosines_matrix_calculation(bad)


def test_all_vector_cosines_blockwise_requires_3d():
    with pytest.raises(ValueError, match="Expected positions shape"):
        all_vector_cosines_blockwise(torch.zeros(3, 3, dtype=torch.float64))


def test_all_vector_cosines_blockwise_matches_full_matrix_calc():
    """The blockwise path must agree with the full-matrix computation on
    the same inputs — blocking is a memory-saving equivalent, not a
    different algorithm."""
    g = torch.Generator().manual_seed(11)
    positions = torch.randn(2, 6, 3, generator=g, dtype=torch.float64) * 3.0
    full = all_vector_cosines_matrix_calculation(positions)
    chunked = all_vector_cosines_blockwise(positions, combo_block_size=1, vec_block_size=2)
    assert torch.isclose(full.amin(), chunked.amin(), atol=1e-9).item()


@pytest.mark.parametrize("n_hits", [3, 5, 8, 12])
def test_annihilation_angle_returns_value_in_valid_cosine_range(n_hits):
    g = torch.Generator().manual_seed(n_hits)
    positions = torch.randn(n_hits, 3, generator=g, dtype=torch.float64) * 10.0
    out = annihilation_angle(positions)
    assert torch.isfinite(out).item()
    assert -1.0001 <= out.item() <= 1.0001


# ---------------------------------------------------------------------------
# Real-data distribution checks + W&B plots
# ---------------------------------------------------------------------------


def test_real_delta_E_distribution_per_class(real_tensors, wandb_run):
    """Plot log-log ΔE histograms for β vs bg. β events should concentrate
    near zero (their energies land on 511 keV); bg spreads toward larger ΔE."""
    beta = real_tensors["bdecay_tensor_delta_E"]
    bg = real_tensors["bg_tensor_delta_E"]
    beta = beta[torch.isfinite(beta) & (beta > 0)].cpu().numpy()
    bg = bg[torch.isfinite(bg) & (bg > 0)].cpu().numpy()
    assert beta.size > 0 and bg.size > 0

    lo = max(min(beta.min(), bg.min()), 1e-3)
    hi = max(beta.max(), bg.max(), lo + 1e-6)
    edges = np.geomspace(lo, hi, 80)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(bg, bins=edges, density=True, alpha=0.5, label=f"bg (n={bg.size})")
    ax.hist(beta, bins=edges, density=True, alpha=0.5, label=f"β (n={beta.size})")
    ax.set_xscale("log")
    ax.set_xlabel("ΔE (keV)")
    ax.set_ylabel("density")
    ax.set_title("delta_E per class")
    ax.legend()
    wandb_run.log(
        {
            "physics_factors/delta_E/hist": wandb.Image(fig),
            "physics_factors/delta_E/beta_median": float(np.median(beta)),
            "physics_factors/delta_E/bg_median": float(np.median(bg)),
        }
    )
    plt.close(fig)

    assert np.median(beta) < np.median(bg), (
        f"β median ΔE ({np.median(beta):.3f}) should be below bg median ({np.median(bg):.3f})"
    )


def test_real_annihilation_angle_distribution_per_class(real_tensors, wandb_run):
    """β topologies should cluster near cos = −1 (back-to-back 511 keV γs);
    bg tops spread across [−1, 1]."""
    beta = real_tensors["bdecay_tensor_annihilation_angle"]
    bg = real_tensors["bg_tensor_annihilation_angle"]
    beta = beta[torch.isfinite(beta)].cpu().numpy()
    bg = bg[torch.isfinite(bg)].cpu().numpy()
    assert beta.size > 0 and bg.size > 0

    edges = np.linspace(-1.0, 1.0, 60)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(bg, bins=edges, density=True, alpha=0.5, label=f"bg (n={bg.size})")
    ax.hist(beta, bins=edges, density=True, alpha=0.5, label=f"β (n={beta.size})")
    ax.axvline(-1.0, color="gray", linestyle="--", linewidth=1, label="back-to-back")
    ax.set_xlabel("min pair cosine")
    ax.set_ylabel("density")
    ax.set_title("annihilation_angle per class")
    ax.legend()
    wandb_run.log(
        {
            "physics_factors/angle/hist": wandb.Image(fig),
            "physics_factors/angle/beta_mean": float(beta.mean()),
            "physics_factors/angle/bg_mean": float(bg.mean()),
        }
    )
    plt.close(fig)

    assert beta.mean() < bg.mean(), (
        f"β mean cosine ({beta.mean():.3f}) should be below bg mean ({bg.mean():.3f}) — "
        "β events should sit closer to the anti-parallel cos = −1 peak."
    )


def test_real_arm_distribution_per_class(real_tensors, wandb_run):
    """ARM measures |θ_geo − θ_kin|. β events should concentrate near zero
    (consistent Compton reconstruction); bg spreads."""
    beta = real_tensors["bdecay_tensor_arm"]
    bg = real_tensors["bg_tensor_arm"]
    beta = beta[torch.isfinite(beta) & (beta > 0)].cpu().numpy()
    bg = bg[torch.isfinite(bg) & (bg > 0)].cpu().numpy()
    assert beta.size > 0 and bg.size > 0

    lo = max(min(beta.min(), bg.min()), 1e-3)
    hi = max(beta.max(), bg.max(), lo + 1e-6)
    edges = np.geomspace(lo, hi, 80)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(bg, bins=edges, density=True, alpha=0.5, label=f"bg (n={bg.size})")
    ax.hist(beta, bins=edges, density=True, alpha=0.5, label=f"β (n={beta.size})")
    ax.set_xscale("log")
    ax.set_xlabel("ARM (rad)")
    ax.set_ylabel("density")
    ax.set_title("arm per class")
    ax.legend()
    wandb_run.log(
        {
            "physics_factors/arm/hist": wandb.Image(fig),
            "physics_factors/arm/beta_median": float(np.median(beta)),
            "physics_factors/arm/bg_median": float(np.median(bg)),
        }
    )
    plt.close(fig)

    assert np.median(beta) < np.median(bg), (
        f"β median ARM ({np.median(beta):.3f}) should be below bg median ({np.median(bg):.3f})"
    )


def test_real_annihilation_reconstruction_deviance_from_back_to_back(real_tensors, wandb_run):
    """Equivalent to the deviance test in tests_specialized but cast in long
    names for tests_physics's real_tensors. Mean deviance from cos = −1
    must be smaller for true β events than for bg events."""
    beta = real_tensors["bdecay_tensor_annihilation_angle"]
    bg = real_tensors["bg_tensor_annihilation_angle"]
    beta = beta[torch.isfinite(beta)].cpu().numpy()
    bg = bg[torch.isfinite(bg)].cpu().numpy()
    dev_beta = np.abs(beta + 1.0)
    dev_bg = np.abs(bg + 1.0)

    mean_beta = float(dev_beta.mean())
    mean_bg = float(dev_bg.mean())

    wandb_run.log(
        {
            "physics_factors/annihilation_deviance/mean_beta": mean_beta,
            "physics_factors/annihilation_deviance/mean_bg": mean_bg,
            "physics_factors/annihilation_deviance/gap": mean_bg - mean_beta,
        }
    )

    assert mean_beta < mean_bg, (
        f"β mean deviance from −1 ({mean_beta:.4f}) should be below bg mean ({mean_bg:.4f})"
    )
