import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import wandb
from mathematics.geometry import theta_geo, theta_kin
from omegaconf import OmegaConf


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
# theta_geo — angle contract
# ---------------------------------------------------------------------------


def test_theta_geo_zero_when_vectors_aligned():
    """v_in ‖ v_out ⇒ cos θ = 1 ⇒ θ = 0."""
    positions = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float64)
    v_in = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    angle, _ = theta_geo(positions, _cfg(), v_in)
    assert torch.allclose(angle, torch.tensor([0.0], dtype=torch.float64), atol=1e-6)


def test_theta_geo_pi_when_vectors_anti_aligned():
    """v_in anti-parallel to v_out ⇒ cos θ = −1 ⇒ θ = π."""
    positions = torch.tensor([[[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]], dtype=torch.float64)
    v_in = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    angle, _ = theta_geo(positions, _cfg(), v_in)
    assert torch.allclose(angle, torch.tensor([math.pi], dtype=torch.float64), atol=1e-6)


def test_theta_geo_pi_over_2_when_vectors_perpendicular():
    """Orthogonal vectors ⇒ cos θ = 0 ⇒ θ = π/2."""
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float64)
    v_in = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    angle, _ = theta_geo(positions, _cfg(), v_in)
    assert torch.allclose(angle, torch.tensor([math.pi / 2], dtype=torch.float64), atol=1e-6)


def test_theta_geo_accepts_2d_positions_as_single_event():
    """A bare (2, 3) input must be promoted to (1, 2, 3) without error."""
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    v_in = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    angle, sigma = theta_geo(positions, _cfg(), v_in)
    assert angle.numel() == 1
    assert sigma.numel() == 1


def test_theta_geo_batched_independence():
    """Stacking events into a batch must not change individual results."""
    a = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    b = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    v_in = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    angle_a, _ = theta_geo(a, _cfg(), v_in)
    angle_b, _ = theta_geo(b, _cfg(), v_in)
    angle_batch, _ = theta_geo(torch.stack([a, b]), _cfg(), v_in)
    assert torch.allclose(angle_batch, torch.cat([angle_a, angle_b]), atol=1e-9)


def test_theta_geo_clamps_cosine_to_valid_arccos_domain():
    """Floating-point drift can push cos θ slightly outside [−1, 1]; the
    function must clamp before arccos so the result is real-valued."""
    g = torch.Generator().manual_seed(0)
    positions = torch.randn(64, 2, 3, generator=g, dtype=torch.float64) * 10.0
    v_in = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    angle, _ = theta_geo(positions, _cfg(), v_in)
    assert torch.all(torch.isfinite(angle))
    assert torch.all(angle >= 0.0) and torch.all(angle <= math.pi)


def test_theta_geo_broadcasts_scalar_v_in_over_batch():
    """A single (3,) source direction must be broadcast across the batch."""
    g = torch.Generator().manual_seed(1)
    positions = torch.randn(50, 2, 3, generator=g, dtype=torch.float64)
    v_in = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    angle, _ = theta_geo(positions, _cfg(), v_in)
    assert angle.shape == (50,)


# ---------------------------------------------------------------------------
# theta_geo — uncertainty contract
# ---------------------------------------------------------------------------


def test_theta_geo_sigma_is_strictly_positive():
    """Per-event σ_θ must be > 0 for any non-degenerate geometry."""
    g = torch.Generator().manual_seed(42)
    positions = torch.randn(30, 2, 3, generator=g, dtype=torch.float64) * 5.0
    v_in = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    _, sigma = theta_geo(positions, _cfg(sigma_x=4.0), v_in)
    assert torch.all(sigma > 0.0)


def test_theta_geo_sigma_scales_with_sigma_x():
    """σ_θ_geo is linear in the position-uncertainty parameter sigma_x —
    doubling sigma_x must double σ_θ for the same geometry."""
    positions = torch.tensor([[[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]], dtype=torch.float64)
    v_in = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    _, sigma_a = theta_geo(positions, _cfg(sigma_x=4.0), v_in)
    _, sigma_b = theta_geo(positions, _cfg(sigma_x=8.0), v_in)
    assert torch.allclose(sigma_b, 2.0 * sigma_a, rtol=1e-6)


def test_theta_geo_sigma_decreases_with_longer_baseline():
    """σ_θ_geo ∝ 1/L for both L_in and L_out — events with longer inter-hit
    distance must report lower uncertainty."""
    cfg = _cfg(sigma_x=4.0)
    v_in = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)

    short = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float64)
    long = torch.tensor([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]], dtype=torch.float64)
    _, sigma_short = theta_geo(short, cfg, v_in)
    _, sigma_long = theta_geo(long, cfg, v_in)
    assert sigma_long.item() < sigma_short.item()


def test_theta_geo_sigma_floor_when_baseline_below_1mm():
    """L_in and L_out are clamped to ≥ 1e-3 inside theta_geo, so σ stays
    finite (and bounded) even when the inter-hit baseline is zero."""
    positions = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float64)
    v_in = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    _, sigma = theta_geo(positions, _cfg(sigma_x=4.0), v_in)
    assert torch.all(torch.isfinite(sigma))


# ---------------------------------------------------------------------------
# theta_kin — angle contract
# ---------------------------------------------------------------------------


def test_theta_kin_in_valid_arccos_range():
    """θ_kin must always lie in [0, π] regardless of input energies — the
    function clamps cos θ before arccos to enforce this."""
    g = torch.Generator().manual_seed(7)
    energies = torch.rand(100, 4, generator=g, dtype=torch.float64) * 1000.0 + 50.0
    angle, _ = theta_kin(energies, _cfg())
    assert torch.all(angle >= 0.0)
    assert torch.all(angle <= math.pi)
    assert torch.all(torch.isfinite(angle))


def test_theta_kin_handles_1d_input_as_single_event():
    """A bare 1D energy tensor is treated as a single event."""
    energies = torch.tensor([300.0, 200.0, 100.0], dtype=torch.float64)
    angle, sigma = theta_kin(energies, _cfg())
    assert angle.numel() == 1
    assert sigma.numel() == 1


def test_theta_kin_batched_independence():
    """Per-event angle computation must not couple events through the batch."""
    e_a = torch.tensor([300.0, 200.0], dtype=torch.float64)
    e_b = torch.tensor([500.0, 100.0], dtype=torch.float64)
    angle_a, _ = theta_kin(e_a, _cfg())
    angle_b, _ = theta_kin(e_b, _cfg())
    angle_batch, _ = theta_kin(torch.stack([e_a, e_b]), _cfg())
    assert torch.allclose(angle_batch, torch.cat([angle_a, angle_b]), atol=1e-9)


def test_theta_kin_matches_compton_formula_directly():
    """For unclamped cos values, θ_kin must equal arccos(1 − 511·(1/E_2 − 1/E_1))."""
    energies = torch.tensor([[700.0, 300.0]], dtype=torch.float64)  # E_1 = 1000, E_2 = 300
    angle, _ = theta_kin(energies, _cfg())
    cos_expected = 1.0 - 511.0 * (1.0 / 300.0 - 1.0 / 1000.0)
    assert -1.0 <= cos_expected <= 1.0
    assert torch.isclose(
        angle.squeeze(),
        torch.tensor(math.acos(cos_expected), dtype=torch.float64),
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# theta_kin — uncertainty contract
# ---------------------------------------------------------------------------


def test_theta_kin_sigma_is_strictly_positive():
    """σ_θ_kin > 0 for any positive energies, by construction."""
    g = torch.Generator().manual_seed(11)
    energies = torch.rand(40, 3, generator=g, dtype=torch.float64) * 800.0 + 100.0
    _, sigma = theta_kin(energies, _cfg())
    assert torch.all(sigma > 0.0)


def test_theta_kin_sigma_increases_at_low_energy():
    """σ_cos_θ ∝ √(1/E_1² + 1/E_2²), so events with lower energies must
    have larger uncertainty (signal-poor → angle-uncertain)."""
    high_e = torch.tensor([[800.0, 600.0]], dtype=torch.float64)
    low_e = torch.tensor([[80.0, 60.0]], dtype=torch.float64)
    _, sigma_high = theta_kin(high_e, _cfg())
    _, sigma_low = theta_kin(low_e, _cfg())
    assert sigma_low.item() > sigma_high.item()


def test_theta_kin_n_sigma_scales_uncertainty_linearly():
    """The likelihood spec's `n_sigma_cos_theta_kin` multiplies σ linearly."""
    energies = torch.tensor([[400.0, 200.0]], dtype=torch.float64)
    _, sigma_1 = theta_kin(energies, _cfg(n_sigma_cos_theta_kin=1))
    _, sigma_3 = theta_kin(energies, _cfg(n_sigma_cos_theta_kin=3))
    assert torch.allclose(sigma_3, 3.0 * sigma_1, rtol=1e-6)


# ---------------------------------------------------------------------------
# Sweep + plot tests (with W&B logging)
# ---------------------------------------------------------------------------


def test_theta_geo_uncertainty_vs_baseline_sweep(wandb_run):
    """σ_θ_geo as a function of baseline length L. Plot σ vs L for both
    L_in and L_out held equal; expect 1/L decay."""
    L_values = np.geomspace(0.5, 50.0, 80)
    sigmas = []
    cfg = _cfg(sigma_x=4.0)
    v_in = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    for L in L_values:
        positions = torch.tensor([[[0.0, 0.0, 0.0], [float(L), 0.0, 0.0]]], dtype=torch.float64)
        _, sigma = theta_geo(positions, cfg, v_in)
        sigmas.append(float(sigma.item()))
    sigmas_arr = np.array(sigmas)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(L_values, sigmas_arr, label="σ_θ_geo (rad)", lw=2)
    ax.plot(L_values, 4.0 * 2.0 / L_values, "--", color="gray", label="2·σ_x / L (theory)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("baseline L (mm)")
    ax.set_ylabel("σ_θ_geo (rad)")
    ax.set_title("theta_geo uncertainty vs baseline length")
    ax.legend()
    wandb_run.log({"geometry/theta_geo_sigma_vs_L": wandb.Image(fig)})
    plt.close(fig)

    assert np.all(np.diff(sigmas_arr) <= 0), "σ_θ_geo must decrease monotonically with L"


def test_theta_kin_uncertainty_vs_energy_sweep(wandb_run):
    """σ_θ_kin as a function of E_1 (with E_2 fixed proportional); expect
    monotonic decrease with rising energy."""
    cfg = _cfg(n_sigma_cos_theta_kin=1)
    E1_values = np.geomspace(60.0, 2000.0, 80)
    sigmas = []
    angles = []
    for E1 in E1_values:
        E2 = 0.5 * E1
        energies = torch.tensor([[E1 - E2, E2]], dtype=torch.float64)
        angle, sigma = theta_kin(energies, cfg)
        sigmas.append(float(sigma.item()))
        angles.append(float(angle.item()))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(E1_values, sigmas, lw=2)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("E_1 (keV)")
    axes[0].set_ylabel("σ_θ_kin (rad)")
    axes[0].set_title("theta_kin uncertainty vs total energy")
    axes[1].plot(E1_values, angles, lw=2)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("E_1 (keV)")
    axes[1].set_ylabel("θ_kin (rad)")
    axes[1].set_title("theta_kin angle vs total energy (E_2 = E_1/2)")
    wandb_run.log({"geometry/theta_kin_vs_energy": wandb.Image(fig)})
    plt.close(fig)

    assert sigmas[0] > sigmas[-1], "σ_θ_kin must decrease with rising E_1"


def test_theta_geo_angle_heatmap_over_polar_sweep(wandb_run):
    """Sweep v_out direction over a unit sphere with v_in fixed and plot
    the resulting θ_geo heatmap. Visual sanity check that the function
    returns the expected angular separation pattern."""
    n_phi, n_theta = 60, 30
    phis = np.linspace(0, 2 * math.pi, n_phi)
    thetas = np.linspace(0.01, math.pi - 0.01, n_theta)
    grid = np.zeros((n_theta, n_phi))
    cfg = _cfg(sigma_x=4.0)
    v_in = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    for i, th in enumerate(thetas):
        for j, ph in enumerate(phis):
            v_out = torch.tensor(
                [
                    math.sin(th) * math.cos(ph),
                    math.sin(th) * math.sin(ph),
                    math.cos(th),
                ],
                dtype=torch.float64,
            )
            positions = torch.stack([torch.zeros(3, dtype=torch.float64), v_out * 5.0])
            angle, _ = theta_geo(positions, cfg, v_in)
            grid[i, j] = float(angle.item())

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.pcolormesh(phis, thetas, grid, cmap="viridis", shading="auto")
    ax.set_xlabel("phi (rad)")
    ax.set_ylabel("theta (rad)")
    ax.set_title("θ_geo over (phi, theta) sweep with v_in = ẑ")
    plt.colorbar(im, ax=ax, label="θ_geo (rad)")
    wandb_run.log({"geometry/theta_geo_heatmap": wandb.Image(fig)})
    plt.close(fig)

    # Co-pole of v_in (θ ≈ 0): θ_geo ≈ 0. Anti-pole (θ ≈ π): θ_geo ≈ π.
    assert grid[0].mean() < 0.5
    assert grid[-1].mean() > math.pi - 0.5


@pytest.mark.parametrize("n_events", [1, 10, 100, 1000])
def test_theta_geo_scales_to_large_batches(n_events):
    """Smoke test: works at a range of batch sizes without OOM / shape issues."""
    g = torch.Generator().manual_seed(13)
    positions = torch.randn(n_events, 2, 3, generator=g, dtype=torch.float64) * 5.0
    v_in = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    angle, sigma = theta_geo(positions, _cfg(), v_in)
    assert angle.shape == (n_events,)
    assert sigma.shape == (n_events,)
    assert torch.all(torch.isfinite(angle))
    assert torch.all(torch.isfinite(sigma))


@pytest.mark.parametrize("n_events,n_hits", [(1, 2), (10, 3), (100, 4), (500, 5)])
def test_theta_kin_scales_to_large_batches(n_events, n_hits):
    """Smoke test for batched + multi-hit energies."""
    g = torch.Generator().manual_seed(17)
    energies = torch.rand(n_events, n_hits, generator=g, dtype=torch.float64) * 500.0 + 50.0
    angle, sigma = theta_kin(energies, _cfg())
    assert angle.shape == (n_events,)
    assert sigma.shape == (n_events,)
    assert torch.all(torch.isfinite(angle))
    assert torch.all(torch.isfinite(sigma))
