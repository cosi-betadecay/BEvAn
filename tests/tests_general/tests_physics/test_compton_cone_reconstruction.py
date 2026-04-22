"""
Unit + real-data tests for physics/compton_cone_reconstruction.py.

The FarFieldImager wraps MEGAlib's MBackprojectionFarField for far-field
point-source localization. The accumulated sky image is exposed both as a
flat (n_pixels,) tensor and as the reshaped (n_theta, n_phi) image.

Test split:
    - Construction & shape contracts (no real data needed beyond a geometry).
    - Reset / accumulator behaviour (synthetic).
    - Real-data sky image: shape, peak direction, image concentration —
      drawn from the cached ``real_tensors["imager"]`` so we don't pay
      for a second backprojection. Two W&B-logged plots: the full sky
      heatmap and the peak's local neighbourhood.
"""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import wandb
from physics.compton_cone_reconstruction import FarFieldImager


def _geo_file() -> str:
    return f"{os.environ['MEGALIB']}/resource/examples/geomega/special/Max.geo.setup"


# ---------------------------------------------------------------------------
# Construction / contract tests (need MEGAlib for geometry but no events)
# ---------------------------------------------------------------------------


def test_constructor_sets_shape_attributes():
    img = FarFieldImager(geometry_file=_geo_file(), n_phi=72, n_theta=36)
    assert img.n_phi == 72
    assert img.n_theta == 36
    assert img.n_pixels == 72 * 36


def test_image_2d_has_n_theta_n_phi_shape():
    img = FarFieldImager(geometry_file=_geo_file(), n_phi=120, n_theta=60)
    arr = img.image_2d()
    assert arr.shape == (60, 120)


def test_accumulated_starts_at_zero():
    img = FarFieldImager(geometry_file=_geo_file(), n_phi=20, n_theta=10)
    assert torch.all(img.accumulated == 0.0)


def test_reset_zeros_accumulator():
    img = FarFieldImager(geometry_file=_geo_file(), n_phi=20, n_theta=10)
    img.accumulated.fill_(7.0)
    img.reset()
    assert torch.all(img.accumulated == 0.0)


def test_unknown_coordinate_system_raises():
    with pytest.raises(ValueError):
        FarFieldImager(geometry_file=_geo_file(), coordinate_system="not-a-system")


def test_galactic_coordinate_system_constructs():
    """Smoke test for the second supported coordinate system."""
    img = FarFieldImager(
        geometry_file=_geo_file(),
        n_phi=36,
        n_theta=18,
        coordinate_system="galactic",
    )
    assert img.image_2d().shape == (18, 36)


def test_invalid_geometry_file_raises():
    with pytest.raises(RuntimeError):
        FarFieldImager(geometry_file="/tmp/does-not-exist.geo.setup")


@pytest.mark.parametrize("n_phi,n_theta", [(36, 18), (90, 45), (360, 180)])
def test_image_2d_shape_for_various_resolutions(n_phi, n_theta):
    img = FarFieldImager(geometry_file=_geo_file(), n_phi=n_phi, n_theta=n_theta)
    assert img.image_2d().shape == (n_theta, n_phi)
    assert img.n_pixels == n_phi * n_theta


# ---------------------------------------------------------------------------
# Peak-direction contract on a synthetic image
# ---------------------------------------------------------------------------


def test_peak_direction_returns_brightest_pixel_centre():
    """Manually paint a single pixel; peak_direction must return that
    pixel's centre coordinates."""
    img = FarFieldImager(geometry_file=_geo_file(), n_phi=360, n_theta=180)
    flat_idx = 45 * 360 + 200
    img.accumulated[flat_idx] = 1.0
    theta, phi = img.peak_direction()
    expected_theta = (45 + 0.5) * math.pi / 180
    expected_phi = (200 + 0.5) * 2.0 * math.pi / 360
    assert math.isclose(theta, expected_theta, abs_tol=1e-9)
    assert math.isclose(phi, expected_phi, abs_tol=1e-9)


def test_peak_direction_within_default_ranges():
    img = FarFieldImager(geometry_file=_geo_file(), n_phi=360, n_theta=180)
    img.accumulated.fill_(0.0)
    img.accumulated[12345] = 1.0
    theta, phi = img.peak_direction()
    assert 0.0 <= theta <= math.pi
    assert 0.0 <= phi <= 2.0 * math.pi


def test_peak_direction_cartesian_returns_unit_vector():
    img = FarFieldImager(geometry_file=_geo_file(), n_phi=360, n_theta=180)
    img.accumulated[1234] = 5.0
    v = img.peak_direction_cartesian()
    assert v.shape == (3,)
    norm = float(torch.linalg.norm(v).item())
    assert math.isclose(norm, 1.0, abs_tol=1e-9)


def test_save_image_numpy_roundtrip(tmp_path):
    """Saved .npy must reload to an identical (n_theta, n_phi) array."""
    img = FarFieldImager(geometry_file=_geo_file(), n_phi=24, n_theta=12)
    img.accumulated.normal_()
    out = tmp_path / "image.npy"
    img.save_image_numpy(out)
    loaded = np.load(str(out))
    assert loaded.shape == (12, 24)
    np.testing.assert_array_equal(loaded, img.image_2d().cpu().numpy())


# ---------------------------------------------------------------------------
# Real-data sky image (cached imager from real_tensors)
# ---------------------------------------------------------------------------


def test_real_imager_accumulates_nonzero_image(real_tensors):
    """After backprojecting the .tra file, the sky image must contain
    non-zero values somewhere — otherwise no Compton events landed."""
    imager = real_tensors["imager"]
    assert torch.any(imager.accumulated > 0.0), "sky image is all zero — backprojection emitted nothing"


def test_real_imager_peak_direction_in_range(real_tensors):
    imager = real_tensors["imager"]
    theta, phi = imager.peak_direction()
    assert 0.0 <= theta <= math.pi
    assert 0.0 <= phi <= 2.0 * math.pi


def test_real_imager_peak_cartesian_is_unit_vector(real_tensors):
    v = real_tensors["reconstructed_unit_vector"]
    assert v.shape == (3,)
    assert math.isclose(float(torch.linalg.norm(v).item()), 1.0, abs_tol=1e-6)


def test_real_imager_image_concentration_around_peak(real_tensors, wandb_run):
    """A localised point source should leave the brightest pixel's local
    neighbourhood holding a noticeable fraction of the total mass — not a
    uniform smear. Plot the sky map and a zoom-in around the peak."""
    imager = real_tensors["imager"]
    img2d = imager.image_2d().cpu().numpy()
    total = img2d.sum()
    assert total > 0.0

    flat_idx = int(np.argmax(img2d))
    theta_idx, phi_idx = divmod(flat_idx, imager.n_phi)
    half = max(2, min(imager.n_theta, imager.n_phi) // 20)

    t_lo, t_hi = max(0, theta_idx - half), min(imager.n_theta, theta_idx + half + 1)
    p_lo, p_hi = max(0, phi_idx - half), min(imager.n_phi, phi_idx + half + 1)
    local_mass = img2d[t_lo:t_hi, p_lo:p_hi].sum()
    local_pix_count = (t_hi - t_lo) * (p_hi - p_lo)
    local_fraction = local_mass / total
    uniform_fraction = local_pix_count / (imager.n_theta * imager.n_phi)

    fig, ax = plt.subplots(figsize=(9, 4))
    extent = [0, 2 * math.pi, math.pi, 0]
    im = ax.imshow(img2d, extent=extent, aspect="auto", cmap="inferno")
    ax.set_xlabel("phi (rad)")
    ax.set_ylabel("theta (rad)")
    ax.set_title("Far-field sky image (Activation.tra)")
    ax.scatter(
        (phi_idx + 0.5) * 2 * math.pi / imager.n_phi,
        (theta_idx + 0.5) * math.pi / imager.n_theta,
        c="cyan",
        marker="x",
        s=80,
        label=f"peak (θ={theta_idx}, φ={phi_idx})",
    )
    ax.legend(loc="upper right")
    plt.colorbar(im, ax=ax, label="accumulated counts")

    fig_zoom, axz = plt.subplots(figsize=(5, 4))
    axz.imshow(img2d[t_lo:t_hi, p_lo:p_hi], cmap="inferno", aspect="auto")
    axz.set_title(f"Zoom around peak (±{half} bins)")
    axz.set_xlabel("Δφ bin")
    axz.set_ylabel("Δθ bin")

    wandb_run.log(
        {
            "compton/sky_image": wandb.Image(fig),
            "compton/sky_image_peak_zoom": wandb.Image(fig_zoom),
            "compton/peak_theta_idx": int(theta_idx),
            "compton/peak_phi_idx": int(phi_idx),
            "compton/local_mass_fraction": float(local_fraction),
            "compton/uniform_local_fraction": float(uniform_fraction),
            "compton/concentration_ratio": float(local_fraction / max(uniform_fraction, 1e-12)),
        }
    )
    plt.close(fig)
    plt.close(fig_zoom)

    assert local_fraction > uniform_fraction, (
        f"local mass fraction near the peak ({local_fraction:.4f}) does not "
        f"exceed the uniform expectation ({uniform_fraction:.4f}); the sky "
        "image looks isotropic — source not localised."
    )


def test_real_imager_image_marginals_nonzero(real_tensors, wandb_run):
    """The φ and θ marginals of the sky image must have full support — the
    accumulator shouldn't be confined to a single row or column."""
    imager = real_tensors["imager"]
    img2d = imager.image_2d().cpu().numpy()
    phi_marginal = img2d.sum(axis=0)
    theta_marginal = img2d.sum(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(np.linspace(0, 2 * math.pi, phi_marginal.size), phi_marginal)
    axes[0].set_xlabel("phi (rad)")
    axes[0].set_ylabel("∑θ counts")
    axes[0].set_title("Sky image φ-marginal")
    axes[1].plot(np.linspace(0, math.pi, theta_marginal.size), theta_marginal)
    axes[1].set_xlabel("theta (rad)")
    axes[1].set_ylabel("∑φ counts")
    axes[1].set_title("Sky image θ-marginal")
    wandb_run.log({"compton/sky_marginals": wandb.Image(fig)})
    plt.close(fig)

    assert (phi_marginal > 0).sum() > 1, "φ-marginal collapsed to a single bin"
    assert (theta_marginal > 0).sum() > 1, "θ-marginal collapsed to a single bin"


def test_real_imager_total_counts_finite_and_positive(real_tensors):
    """The total accumulated mass should be finite and positive — a fast
    regression catch for response-weighting going haywire."""
    imager = real_tensors["imager"]
    total = float(imager.accumulated.sum().item())
    assert total > 0.0
    assert math.isfinite(total)


def test_real_imager_independent_instance_reproduces_peak(real_tensors):
    """Build a fresh imager and re-backproject the .tra file — the peak
    direction should land near the cached imager's peak (within 2°),
    confirming backprojection is deterministic."""
    fresh = FarFieldImager(
        geometry_file=real_tensors["geo_file"],
        n_phi=360,
        n_theta=180,
        coordinate_system="spheric",
    )
    fresh.backproject_file(real_tensors["tra_file"])
    v_fresh = fresh.peak_direction_cartesian()
    v_cached = real_tensors["reconstructed_unit_vector"]
    cos_sep = float(torch.dot(v_fresh, v_cached).item())
    cos_sep = max(-1.0, min(1.0, cos_sep))
    angular_sep = math.acos(cos_sep)
    assert angular_sep < math.radians(2.0), (
        f"angular separation between fresh and cached peak directions is "
        f"{math.degrees(angular_sep):.2f}° — backprojection looks non-deterministic"
    )
