import gc
import os

import pytest
import torch
import wandb
from dataset.datasets import Datasets
from modeling.bayesian_annihilation import BayesianAnnihiliationModel
from modeling.calculate_probablities import R
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

# Resolutions swept by parametrized fixtures across this tree. 20×20 is coarse
# enough that every bin is well-populated, 1000×1000 is the production
# default. 200×200 bridges the two so the curse-of-dimensionality transition
# is visible in the W&B traces.
BIN_SIZES = [20, 200, 1000]


# Module-level LRU-of-1 caches for the parametrized heavy fixtures. With
# scope="session", pytest would hold all three bin-size instances alive until
# end-of-session, which at n_bins=1000 pushes peak memory uncomfortably high.
# Function-scope fixtures hit these caches: tests at the same n_bins reuse
# the build, and when pytest moves to the next n_bins the previous entry is
# evicted and gc'd before the new build starts.
_density_matrices_cache: dict[int, dict] = {}
_likelihood_tensors_cache: dict[int, dict] = {}
_trainer_cache: dict[int, dict] = {}


def _evict_other_params(cache: dict, current_param) -> None:
    stale = [k for k in cache if k != current_param]
    for k in stale:
        del cache[k]
    if stale:
        gc.collect()


@pytest.fixture(scope="session")
def real_tensors():
    """Iterate Activation.sim once and return per-event feature tensors plus
    ground truths. Session-scoped so MEGAlib opens the file exactly once per
    test run."""
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
        "cfg": cfg,
        "datasets": datasets,
        "imager": imager,
        "reconstructed_unit_vector": reconstructed_unit_vector,
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
def train_eval_split(real_tensors):
    """80/20 stratified split of the real-data tensors (the same call the
    production pipeline makes in run.py). Returns the (train, eval) tuple
    pair so tests can drive Trainer/Evaluator end-to-end."""
    datasets = real_tensors["datasets"]
    return datasets.split_dataset(
        real_tensors["bdecay_tensor_delta_E"],
        real_tensors["bdecay_tensor_annihilation_angle"],
        real_tensors["bdecay_tensor_arm"],
        real_tensors["bg_tensor_delta_E"],
        real_tensors["bg_tensor_annihilation_angle"],
        real_tensors["bg_tensor_arm"],
    )


@pytest.fixture(scope="session")
def wandb_run():
    """Single W&B run collecting diagnostics across every test in tests_general
    that doesn't sit under tests_physics (which defines its own run)."""
    run = wandb.init(
        project="cosi-betadecay-tests",
        name="test-tests-general",
        tags=["tests", "general"],
    )
    yield run
    wandb.finish()


def _build_density_matrices(real_tensors: dict, n_bins: int) -> dict:
    """Construct 2D probability matrices for the two feature pairs at a given
    resolution. ΔE and ARM axes use log spacing; annihilation_angle linear."""
    t = real_tensors

    _, x_bins_angle, y_bins_angle = build_density_matrix(
        t["combined_tensor_delta_E"],
        t["combined_tensor_annihilation_angle"],
        n_bins_x=n_bins,
        n_bins_y=n_bins,
        spacing_x="log",
        spacing_y="linear",
        log_x_floor=0.1,
    )
    bdecay_angle_probs, _, _ = build_density_matrix(
        t["bdecay_tensor_delta_E"],
        t["bdecay_tensor_annihilation_angle"],
        x_bins=x_bins_angle,
        y_bins=y_bins_angle,
    )
    bg_angle_probs, _, _ = build_density_matrix(
        t["bg_tensor_delta_E"],
        t["bg_tensor_annihilation_angle"],
        x_bins=x_bins_angle,
        y_bins=y_bins_angle,
    )

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
    bdecay_arm_probs, _, _ = build_density_matrix(
        t["bdecay_tensor_delta_E"],
        t["bdecay_tensor_arm"],
        x_bins=x_bins_arm,
        y_bins=y_bins_arm,
    )
    bg_arm_probs, _, _ = build_density_matrix(
        t["bg_tensor_delta_E"],
        t["bg_tensor_arm"],
        x_bins=x_bins_arm,
        y_bins=y_bins_arm,
    )

    return {
        "bdecay_angle": bdecay_angle_probs,
        "bg_angle": bg_angle_probs,
        "x_bins_angle": x_bins_angle,
        "y_bins_angle": y_bins_angle,
        "bdecay_arm": bdecay_arm_probs,
        "bg_arm": bg_arm_probs,
        "x_bins_arm": x_bins_arm,
        "y_bins_arm": y_bins_arm,
        "n_bins": n_bins,
    }


@pytest.fixture(scope="function", params=BIN_SIZES, ids=[f"n_bins={n}" for n in BIN_SIZES])
def density_matrices(real_tensors, request):
    """2D per-class probability matrices for (ΔE, angle) and (ΔE, ARM) at
    ``request.param`` × ``request.param`` resolution. Function-scoped with an
    LRU-of-1 cache for memory boundedness at high n_bins."""
    n_bins = request.param
    _evict_other_params(_density_matrices_cache, n_bins)
    if n_bins not in _density_matrices_cache:
        _density_matrices_cache[n_bins] = _build_density_matrices(real_tensors, n_bins)
    return _density_matrices_cache[n_bins]


def _build_likelihood_tensors(real_tensors: dict, n_bins: int) -> dict:
    """Construct the six per-event likelihood tensors at a given resolution
    (mirrors the production train.py flow without the wandb side-effects).
    Used by both the likelihood_tensors and trained_model fixtures."""
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
        t["gen_tensor_delta_E"], t["gen_tensor_arm"], bdecay_cond_arm, deltaE_arm_bins, arm_bins
    )
    p_bg_arm = lookup_density_values(
        t["gen_tensor_delta_E"], t["gen_tensor_arm"], bg_cond_arm, deltaE_arm_bins, arm_bins
    )

    n_beta_decay = int(torch.isfinite(t["bdecay_tensor_delta_E"]).sum().item())
    n_bg = int(torch.isfinite(t["bg_tensor_delta_E"]).sum().item())

    ratio = R(p_beta_deltaE, p_bg_deltaE, p_beta_angle, p_bg_angle, p_beta_arm, p_bg_arm)
    model = BayesianAnnihiliationModel(ratio, n_beta_decay, n_bg)

    return {
        "p_beta_deltaE": p_beta_deltaE,
        "p_bg_deltaE": p_bg_deltaE,
        "p_beta_angle": p_beta_angle,
        "p_bg_angle": p_bg_angle,
        "p_beta_arm": p_beta_arm,
        "p_bg_arm": p_bg_arm,
        "ratio": ratio,
        "model": model,
        "n_beta_decay": n_beta_decay,
        "n_bg": n_bg,
        "n_bins": n_bins,
        "n_gen": t["n_gen"],
        "labels": t["ground_truths"].cpu().numpy().astype("int64"),
    }


def _patch_train_module_build_density_matrix(monkeypatch, n_bins: int) -> None:
    """Shared helper used by the ``real_trainer`` fixture to override the
    hard-coded 200 in Trainer.fit. Calls with pre-built x_bins / y_bins pass
    through untouched (those are the second-stage per-class joint builds)."""
    from modeling import matrix_calculations as _mc
    from pipeline import train as _train_mod

    original = _mc.build_density_matrix

    def patched(*args, **kwargs):
        if "x_bins" not in kwargs:
            kwargs["n_bins_x"] = n_bins
        if "y_bins" not in kwargs:
            kwargs["n_bins_y"] = n_bins
        return original(*args, **kwargs)

    monkeypatch.setattr(_train_mod, "build_density_matrix", patched)


@pytest.fixture(params=BIN_SIZES, ids=[f"n_bins={n}" for n in BIN_SIZES])
def real_trainer(real_tensors, train_eval_split, wandb_run, monkeypatch, request):
    """Trainer fitted on the real-data train split at ``request.param`` bins.

    Function-scoped (so pytest's ``monkeypatch`` can take effect) with a
    module-level LRU-of-1 cache (``_trainer_cache``): tests at the same
    n_bins reuse the fitted trainer; when pytest transitions to the next
    n_bins, the previous entry is evicted and gc.collected so peak memory
    stays bounded to one resolution at a time.

    Available to every tests_general sub-directory, so test_train.py and
    test_eval.py share the cache and don't re-fit."""
    from pipeline.train import Trainer

    n_bins = request.param
    _evict_other_params(_trainer_cache, n_bins)
    if n_bins not in _trainer_cache:
        train_tuple, eval_tuple = train_eval_split
        _patch_train_module_build_density_matrix(monkeypatch, n_bins)
        trainer = Trainer(real_tensors["cfg"]).fit(train_tuple)
        _trainer_cache[n_bins] = {
            "trainer": trainer,
            "eval_tuple": eval_tuple,
            "train_tuple": train_tuple,
            "n_bins": n_bins,
        }
    return _trainer_cache[n_bins]


@pytest.fixture(scope="function", params=BIN_SIZES, ids=[f"n_bins={n}" for n in BIN_SIZES])
def likelihood_tensors(real_tensors, request):
    """Per-event likelihood tensors at ``request.param`` resolution plus the
    full Bayesian model and ground-truth labels — everything downstream
    classifier tests need at a fixed bin count.

    Function-scoped with module-level LRU-of-1 cache so tests at the same
    n_bins reuse the build; transitioning to the next n_bins evicts the
    old entry and runs gc.collect()."""
    n_bins = request.param
    _evict_other_params(_likelihood_tensors_cache, n_bins)
    if n_bins not in _likelihood_tensors_cache:
        _likelihood_tensors_cache[n_bins] = _build_likelihood_tensors(real_tensors, n_bins)
    return _likelihood_tensors_cache[n_bins]
