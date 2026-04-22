"""
End-to-end integration tests for the β⁺-annihilation detection pipeline.

The original ``physics/annihilation_detection.py`` module was removed during
the train/eval refactor; its responsibilities now live in ``pipeline/train.py``
(Trainer) and ``pipeline/eval.py`` (Evaluator), which jointly consume the
per-class tensors produced by ``Datasets.compute_event_features`` and the
Compton-cone reconstruction from ``physics/compton_cone_reconstruction.py``.

Rather than re-testing each subsystem here (they have dedicated files), this
file exercises the *whole* annihilation-detection pipeline end-to-end on
real data and surfaces the integrated output — logs a final F1 score per
resolution to W&B so regressions in any of the subsystems surface as a
drop in the integrated metric.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import wandb
from omegaconf import OmegaConf
from pipeline.eval import Evaluator
from pipeline.train import Trainer


BIN_SIZES = [20, 200, 1000]


def _patch_n_bins(monkeypatch, n_bins: int) -> None:
    """Override the hard-coded 200 in Trainer.fit for this test."""
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


def _capture_and_evaluate(trainer, data_tuple, split_name: str) -> dict:
    captured = {}
    real_log = wandb.log

    def capturing_log(payload, *args, **kwargs):
        captured.update(payload)
        return real_log(payload, *args, **kwargs)

    wandb.log = capturing_log
    try:
        Evaluator(trainer).evaluate(data_tuple, split_name=split_name)
    finally:
        wandb.log = real_log
    return captured


@pytest.mark.parametrize("n_bins", BIN_SIZES, ids=[f"n_bins={n}" for n in BIN_SIZES])
def test_annihilation_pipeline_end_to_end(
    real_tensors, wandb_run, monkeypatch, n_bins
):
    """Full pipeline at each bin size: feature extraction → split → fit →
    evaluate. Asserts the eval-F1 beats prevalence; logs train F1, eval F1,
    and the gap."""
    cfg = OmegaConf.create(
        {"likelihoods": {"arm": {"sigma_x": 4.0, "n_sigma_cos_theta_kin": 1}}}
    )
    datasets = real_tensors["datasets"] if "datasets" in real_tensors else None
    if datasets is None:
        # tests_physics/conftest doesn't expose datasets; rebuild via Datasets
        # to stay independent of the fixture variant.
        from dataset.datasets import Datasets

        datasets = Datasets(
            cfg,
            511,
            reader=None,  # unused by split_dataset
            reconstructed_unit_vector=real_tensors["reconstructed_unit_vector"],
        )

    train_tuple, eval_tuple = datasets.split_dataset(
        real_tensors["bdecay_tensor_delta_E"],
        real_tensors["bdecay_tensor_annihilation_angle"],
        real_tensors["bdecay_tensor_arm"],
        real_tensors["bg_tensor_delta_E"],
        real_tensors["bg_tensor_annihilation_angle"],
        real_tensors["bg_tensor_arm"],
    )

    _patch_n_bins(monkeypatch, n_bins)
    trainer = Trainer(cfg).fit(train_tuple)

    train_metrics = _capture_and_evaluate(
        trainer, train_tuple, split_name=f"annihilation_e2e/n_bins={n_bins}/train"
    )
    eval_metrics = _capture_and_evaluate(
        trainer, eval_tuple, split_name=f"annihilation_e2e/n_bins={n_bins}/eval"
    )

    f1_train = train_metrics[f"annihilation_e2e/n_bins={n_bins}/train/f1_score"]
    f1_eval = eval_metrics[f"annihilation_e2e/n_bins={n_bins}/eval/f1_score"]
    precision_eval = eval_metrics[f"annihilation_e2e/n_bins={n_bins}/eval/precision"]
    recall_eval = eval_metrics[f"annihilation_e2e/n_bins={n_bins}/eval/recall"]

    gt_eval = eval_tuple[0]
    prevalence = float(gt_eval.float().mean().item())

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        ["train F1", "eval F1", "prevalence"],
        [f1_train, f1_eval, prevalence],
        color=["tab:blue", "tab:orange", "gray"],
    )
    for bar, v in zip(bars, [f1_train, f1_eval, prevalence]):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("metric")
    ax.set_title(f"Annihilation pipeline end-to-end (n_bins={n_bins})")

    wandb_run.log(
        {
            f"annihilation_e2e/n_bins={n_bins}/train_f1": f1_train,
            f"annihilation_e2e/n_bins={n_bins}/eval_f1": f1_eval,
            f"annihilation_e2e/n_bins={n_bins}/train_minus_eval_f1": f1_train - f1_eval,
            f"annihilation_e2e/n_bins={n_bins}/eval_precision": precision_eval,
            f"annihilation_e2e/n_bins={n_bins}/eval_recall": recall_eval,
            f"annihilation_e2e/n_bins={n_bins}/prevalence": prevalence,
            f"annihilation_e2e/n_bins={n_bins}/summary": wandb.Image(fig),
        }
    )
    plt.close(fig)

    assert f1_eval > prevalence, (
        f"At n_bins={n_bins}: end-to-end eval F1 ({f1_eval:.3f}) did not beat "
        f"prevalence baseline ({prevalence:.3f}) — a full pipeline regression."
    )


def test_annihilation_pipeline_feature_extraction_did_not_drop_events(real_tensors):
    """All three per-event gen tensors and ground_truths must share length —
    the feature-extraction stage is the first place a regression would
    desynchronise the per-event indexing."""
    n_gen = real_tensors["gen_tensor_delta_E"].numel()
    assert real_tensors["gen_tensor_annihilation_angle"].numel() == n_gen
    assert real_tensors["gen_tensor_arm"].numel() == n_gen
    assert real_tensors["ground_truths"].numel() == n_gen


def test_annihilation_pipeline_reconstructed_direction_is_unit_vector(real_tensors):
    """The FarFieldImager's peak_direction_cartesian feeds ARM; if it stops
    being a unit vector, ARM values drift. Guard that invariant here so a
    regression in the imager surfaces as an integration-test failure."""
    v = real_tensors["reconstructed_unit_vector"]
    norm = float(torch.linalg.norm(v).item())
    assert abs(norm - 1.0) < 1e-6, f"reconstructed unit vector has norm {norm:.6f}"


def test_annihilation_pipeline_gen_ground_truths_bool(real_tensors):
    """The Bayesian classifier's inference consumes boolean ground truths —
    if the conversion ever drifts (e.g. back to int64), the class ratio
    computation silently breaks."""
    assert real_tensors["ground_truths"].dtype == torch.bool
