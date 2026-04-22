"""
Session-scoped pytest fixtures shared across tests_physics.

The ``real_tensors`` fixture iterates ``data/Activation.sim`` through the same
event-reading and feature-extraction pipeline the production code uses.
Because it is session-scoped, MEGAlib opens and iterates the file only once
per test run, regardless of how many test functions consume it.

The ``density_matrices`` and ``likelihood_tensors`` fixtures are parametrized
over ``BIN_SIZES`` so every physics test that consumes them runs once per
resolution. This exposes how much of each assertion depends on sparse-bin
memorization vs. genuine feature separation.

Requirements:
    - The MEGALIB environment variable must point to a valid MEGAlib installation.
    - data/Activation.sim must exist relative to the project root (where pytest
      is invoked from).
"""

import gc
import os

import pytest
import torch
import wandb
from dataset.datasets import Datasets
from modeling.matrix_calculations import (
    build_density_matrix,
    build_density_matrix_1d,
    conditional_from_joint,
    lookup_density_values,
    lookup_density_values_1d,
)
from omegaconf import OmegaConf
from physics.compton_cone_reconstruction import FarFieldImager
from utils.reader_extraction import get_reader

# Resolutions swept by density_matrices / likelihood_tensors. 20×20 is coarse
# enough that every bin is well-populated, 1000×1000 is the production
# default. Intermediate 200×200 bridges the two.
BIN_SIZES = [20, 200, 1000]


# Module-level LRU-of-1 caches for the parametrized heavy fixtures. With
# scope="session", pytest would hold all three bin-size instances alive until
# end-of-session, which at n_bins=1000 pushes peak memory into OOM territory.
# Instead we use function-scope fixtures that hit these caches: tests at the
# same n_bins reuse the build, and when pytest moves to the next n_bins the
# previous entry is explicitly evicted and gc'd before the new build starts.
_density_matrices_cache: dict[int, dict] = {}
_likelihood_tensors_cache: dict[int, dict] = {}


def _evict_other_params(cache: dict, current_param) -> None:
    """Drop cached entries for params other than the current one and force a
    garbage collection pass so torch tensors are actually released."""
    stale = [k for k in cache if k != current_param]
    for k in stale:
        del cache[k]
    if stale:
        gc.collect()


@pytest.fixture(scope="session")
def real_tensors():
    """
    Load and process the full Activation.sim file, returning every feature
    tensor the pipeline produces. Session-scoped so MEGAlib iterates the sim
    file exactly once per test run even when downstream fixtures sweep bin
    sizes.
    """
    megalib_root = os.environ["MEGALIB"]
    geo_file = f"{megalib_root}/resource/examples/geomega/special/Max.geo.setup"
    sim_file = "data/Activation.sim"
    tra_file = "data/Activation.tra"

    cfg = OmegaConf.create(
        {
            "likelihoods": {
                "arm": {
                    "sigma_x": 4.0,
                    "n_sigma_cos_theta_kin": 1,
                }
            }
        }
    )

    # Replicates the mainline pipeline in run.py: run the far-field
    # backprojection once per session so ARM has a real reconstructed source
    # direction instead of a hard-coded axis.
    imager = FarFieldImager(
        geometry_file=geo_file,
        n_phi=360,
        n_theta=180,
        coordinate_system="spheric",
    )
    imager.backproject_file(tra_file)
    reconstructed_unit_vector = imager.peak_direction_cartesian()

    reader = get_reader(geo_file, sim_file)
    datasets = Datasets(cfg, 511, reader, reconstructed_unit_vector)

    (
        ground_truths,
        bdecay_dE,
        bdecay_angle,
        bdecay_arm,
        bg_dE,
        bg_angle,
        bg_arm,
        gen_dE,
        gen_angle,
        gen_arm,
        combined_dE,
        combined_angle,
        combined_arm,
    ) = datasets.compute_event_features(cfg, 511, reader, reconstructed_unit_vector)

    return {
        "imager": imager,
        "reconstructed_unit_vector": reconstructed_unit_vector,
        "geo_file": geo_file,
        "tra_file": tra_file,
        "ground_truths": torch.as_tensor(ground_truths, dtype=torch.bool),
        "bdecay_tensor_delta_E": bdecay_dE,
        "bdecay_tensor_annihilation_angle": bdecay_angle,
        "bdecay_tensor_arm": bdecay_arm,
        "bg_tensor_delta_E": bg_dE,
        "bg_tensor_annihilation_angle": bg_angle,
        "bg_tensor_arm": bg_arm,
        "gen_tensor_delta_E": gen_dE,
        "gen_tensor_annihilation_angle": gen_angle,
        "gen_tensor_arm": gen_arm,
        "combined_tensor_delta_E": combined_dE,
        "combined_tensor_annihilation_angle": combined_angle,
        "combined_tensor_arm": combined_arm,
        "n_gen": gen_dE.numel(),
    }


@pytest.fixture(scope="session")
def wandb_run():
    """Single W&B run collecting diagnostics across every bin-size parametrization."""
    run = wandb.init(
        project="cosi-betadecay-tests",
        name="test-density-matrices",
        tags=["tests", "matrix", "density"],
    )
    yield run
    wandb.finish()


def _build_density_matrices(real_tensors: dict, n_bins: int) -> dict:
    """Construct 2D probability matrices for the two feature pairs at a given
    resolution. Mirrors the production pipeline in ``build_density_matrices``.
    ΔE and ARM axes use log spacing; annihilation_angle stays linear."""
    t = real_tensors

    # ---- Angle feature pair: delta_E vs annihilation_angle ----
    _, x_bins_angle, y_bins_angle = build_density_matrix(
        t["combined_tensor_delta_E"],
        t["combined_tensor_annihilation_angle"],
        n_bins_x=n_bins,
        n_bins_y=n_bins,
        spacing_x="log",
        spacing_y="linear",
        log_x_floor=0.1,
    )
    bdecay_angle, _, _ = build_density_matrix(
        t["bdecay_tensor_delta_E"],
        t["bdecay_tensor_annihilation_angle"],
        x_bins=x_bins_angle,
        y_bins=y_bins_angle,
    )
    bg_angle, _, _ = build_density_matrix(
        t["bg_tensor_delta_E"],
        t["bg_tensor_annihilation_angle"],
        x_bins=x_bins_angle,
        y_bins=y_bins_angle,
    )

    # ---- ARM feature pair: delta_E vs ARM ----
    _, x_bins_arm, y_bins_arm = build_density_matrix(
        t["combined_tensor_delta_E"],
        t["combined_tensor_arm"],
        n_bins_x=n_bins,
        n_bins_y=n_bins,
        spacing_x="log",
        spacing_y="log",
        log_x_floor=0.1,
        log_y_floor=1e-3,
    )
    bdecay_arm, _, _ = build_density_matrix(
        t["bdecay_tensor_delta_E"],
        t["bdecay_tensor_arm"],
        x_bins=x_bins_arm,
        y_bins=y_bins_arm,
    )
    bg_arm, _, _ = build_density_matrix(
        t["bg_tensor_delta_E"],
        t["bg_tensor_arm"],
        x_bins=x_bins_arm,
        y_bins=y_bins_arm,
    )

    return {
        "bdecay_angle": bdecay_angle,
        "bg_angle": bg_angle,
        "x_bins_angle": x_bins_angle,
        "y_bins_angle": y_bins_angle,
        "bdecay_arm": bdecay_arm,
        "bg_arm": bg_arm,
        "x_bins_arm": x_bins_arm,
        "y_bins_arm": y_bins_arm,
        "n_bins": n_bins,
    }


@pytest.fixture(scope="function", params=BIN_SIZES, ids=[f"n_bins={n}" for n in BIN_SIZES])
def density_matrices(real_tensors, request):
    """
    2D probability matrices for the two feature pairs at ``request.param`` ×
    ``request.param`` resolution.

    Function-scoped with a module-level LRU-of-1 cache (``_density_matrices_cache``)
    that evicts non-current parametrizations. Within a group of tests at the
    same n_bins, the cache hit avoids rebuild; when pytest transitions to the
    next n_bins, the previous entry is dropped and ``gc.collect()`` runs,
    so peak memory stays bounded to one bin size's worth of matrices instead
    of all three (which is what session-scoped parametrization would give you).
    """
    n_bins = request.param
    _evict_other_params(_density_matrices_cache, n_bins)
    if n_bins not in _density_matrices_cache:
        _density_matrices_cache[n_bins] = _build_density_matrices(real_tensors, n_bins)
    return _density_matrices_cache[n_bins]


def _build_likelihood_tensors(real_tensors: dict, n_bins: int) -> dict:
    """Construct the six per-event likelihood tensors at a given resolution.
    Mirrors ``annihilation_detection_utils.build_density_matrices`` but with
    a configurable bin count (production hardcodes 1000)."""
    t = real_tensors

    _, deltaE_angle_bins, angle_bins = build_density_matrix(
        t["combined_tensor_delta_E"],
        t["combined_tensor_annihilation_angle"],
        n_bins_x=n_bins,
        n_bins_y=n_bins,
        spacing_x="log",
        spacing_y="linear",
        log_x_floor=0.1,
    )
    _, deltaE_arm_bins, arm_bins = build_density_matrix(
        t["combined_tensor_delta_E"],
        t["combined_tensor_arm"],
        n_bins_x=n_bins,
        n_bins_y=n_bins,
        spacing_x="log",
        spacing_y="log",
        log_x_floor=0.1,
        log_y_floor=1e-3,
    )

    bdecay_joint_angle, _, _ = build_density_matrix(
        t["bdecay_tensor_delta_E"],
        t["bdecay_tensor_annihilation_angle"],
        x_bins=deltaE_angle_bins,
        y_bins=angle_bins,
    )
    bdecay_joint_arm, _, _ = build_density_matrix(
        t["bdecay_tensor_delta_E"],
        t["bdecay_tensor_arm"],
        x_bins=deltaE_arm_bins,
        y_bins=arm_bins,
    )
    bg_joint_angle, _, _ = build_density_matrix(
        t["bg_tensor_delta_E"],
        t["bg_tensor_annihilation_angle"],
        x_bins=deltaE_angle_bins,
        y_bins=angle_bins,
    )
    bg_joint_arm, _, _ = build_density_matrix(
        t["bg_tensor_delta_E"],
        t["bg_tensor_arm"],
        x_bins=deltaE_arm_bins,
        y_bins=arm_bins,
    )

    bdecay_marginal_dE, _ = build_density_matrix_1d(t["bdecay_tensor_delta_E"], x_bins=deltaE_angle_bins)
    bg_marginal_dE, _ = build_density_matrix_1d(t["bg_tensor_delta_E"], x_bins=deltaE_angle_bins)

    bdecay_cond_angle = conditional_from_joint(bdecay_joint_angle, axis=0)
    bdecay_cond_arm = conditional_from_joint(bdecay_joint_arm, axis=0)
    bg_cond_angle = conditional_from_joint(bg_joint_angle, axis=0)
    bg_cond_arm = conditional_from_joint(bg_joint_arm, axis=0)

    p_beta_deltaE = lookup_density_values_1d(t["gen_tensor_delta_E"], bdecay_marginal_dE, deltaE_angle_bins)
    p_bg_deltaE = lookup_density_values_1d(t["gen_tensor_delta_E"], bg_marginal_dE, deltaE_angle_bins)
    p_beta_angle = lookup_density_values(
        t["gen_tensor_delta_E"],
        t["gen_tensor_annihilation_angle"],
        bdecay_cond_angle,
        deltaE_angle_bins,
        angle_bins,
    )
    p_bg_angle = lookup_density_values(
        t["gen_tensor_delta_E"],
        t["gen_tensor_annihilation_angle"],
        bg_cond_angle,
        deltaE_angle_bins,
        angle_bins,
    )
    p_beta_arm = lookup_density_values(
        t["gen_tensor_delta_E"],
        t["gen_tensor_arm"],
        bdecay_cond_arm,
        deltaE_arm_bins,
        arm_bins,
    )
    p_bg_arm = lookup_density_values(
        t["gen_tensor_delta_E"],
        t["gen_tensor_arm"],
        bg_cond_arm,
        deltaE_arm_bins,
        arm_bins,
    )

    return {
        "p_beta_deltaE": p_beta_deltaE,
        "p_bg_deltaE": p_bg_deltaE,
        "p_beta_angle": p_beta_angle,
        "p_bg_angle": p_bg_angle,
        "p_beta_arm": p_beta_arm,
        "p_bg_arm": p_bg_arm,
        "n_bins": n_bins,
        "n_gen": t["n_gen"],
    }


@pytest.fixture(scope="function", params=BIN_SIZES, ids=[f"n_bins={n}" for n in BIN_SIZES])
def likelihood_tensors(real_tensors, request):
    """
    Per-event likelihood tensors at ``request.param`` resolution — the
    six factors (P_β(ΔE), P_bg(ΔE), P_β(angle|ΔE), P_bg(angle|ΔE),
    P_β(ARM|ΔE), P_bg(ARM|ΔE)) plus ``n_bins`` and ``n_gen``.

    Function-scoped with module-level LRU-of-1 cache via
    ``_likelihood_tensors_cache``. Same memory-bounding pattern as
    ``density_matrices``: tests at a given n_bins hit the cache; pytest
    transitioning to the next n_bins evicts the old entry and gc.collects
    before the new one is built.
    """
    n_bins = request.param
    _evict_other_params(_likelihood_tensors_cache, n_bins)
    if n_bins not in _likelihood_tensors_cache:
        _likelihood_tensors_cache[n_bins] = _build_likelihood_tensors(real_tensors, n_bins)
    return _likelihood_tensors_cache[n_bins]
