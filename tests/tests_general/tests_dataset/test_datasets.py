import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import wandb
from dataset.datasets import Datasets


def _make_datasets() -> Datasets:
    """Construct a Datasets instance with no MEGAlib dependencies. Only the
    splitting logic uses ``self`` (just to read ``train_percentage``), so the
    other constructor args are fine as ``None`` in unit tests."""
    return Datasets(cfg=None, ref_energy=511.0, reader=None, reconstructed_unit_vector=None)


def _synthetic_class_tensors(n_bdecay: int, n_bg: int, seed: int = 0):
    """Three feature tensors per class, with values offset by class so we can
    later verify which events landed in which split."""
    g = torch.Generator().manual_seed(seed)
    bdecay_dE = torch.rand(n_bdecay, generator=g) * 1.0
    bdecay_angle = torch.rand(n_bdecay, generator=g) * 1.0 - 1.0
    bdecay_arm = torch.rand(n_bdecay, generator=g) * 0.1
    bg_dE = torch.rand(n_bg, generator=g) * 1.0 + 100.0
    bg_angle = torch.rand(n_bg, generator=g) * 2.0 - 1.0
    bg_arm = torch.rand(n_bg, generator=g) * 1.0 + 10.0
    return bdecay_dE, bdecay_angle, bdecay_arm, bg_dE, bg_angle, bg_arm


# ---------------------------------------------------------------------------
# Pure-logic tests on split_dataset
# ---------------------------------------------------------------------------


def test_split_preserves_total_event_count():
    """Train + eval together must contain every input event in each class."""
    ds = _make_datasets()
    n_bdecay, n_bg = 100, 200
    inputs = _synthetic_class_tensors(n_bdecay, n_bg)
    train, evaluator = ds.split_dataset(*inputs)

    bdecay_train_dE, bg_train_dE = train[1], train[4]
    bdecay_eval_dE, bg_eval_dE = evaluator[1], evaluator[4]
    assert bdecay_train_dE.numel() + bdecay_eval_dE.numel() == n_bdecay
    assert bg_train_dE.numel() + bg_eval_dE.numel() == n_bg


def test_split_ratio_is_80_20():
    """train_percentage = 0.8 should put ~80 % of each class into train."""
    ds = _make_datasets()
    n_bdecay, n_bg = 1000, 500
    inputs = _synthetic_class_tensors(n_bdecay, n_bg)
    train, evaluator = ds.split_dataset(*inputs)

    bdecay_train_dE = train[1]
    bg_train_dE = train[4]
    assert bdecay_train_dE.numel() == int(n_bdecay * 0.8)
    assert bg_train_dE.numel() == int(n_bg * 0.8)


def test_split_is_seeded_and_reproducible():
    """Two calls with the same input must return identical splits — the
    seed (42) is hard-coded, so reproducibility is part of the contract."""
    ds = _make_datasets()
    inputs = _synthetic_class_tensors(200, 200)

    train_a, eval_a = ds.split_dataset(*inputs)
    train_b, eval_b = ds.split_dataset(*inputs)

    for a, b in zip(train_a, train_b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), "Train split changed across calls with the same input"
    for a, b in zip(eval_a, eval_b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), "Eval split changed across calls with the same input"


def test_split_train_and_eval_are_disjoint():
    """Every input event must appear in exactly one of train or eval — never
    both, never neither."""
    ds = _make_datasets()

    # Use unique values so set-membership identifies each event individually.
    bdecay_dE = torch.arange(100, dtype=torch.float32)
    bdecay_angle = torch.arange(100, dtype=torch.float32) * -1.0
    bdecay_arm = torch.arange(100, dtype=torch.float32) * 0.01
    bg_dE = torch.arange(200, dtype=torch.float32) + 1000.0
    bg_angle = torch.arange(200, dtype=torch.float32) * -2.0
    bg_arm = torch.arange(200, dtype=torch.float32) * 0.02

    train, evaluator = ds.split_dataset(bdecay_dE, bdecay_angle, bdecay_arm, bg_dE, bg_angle, bg_arm)

    bdecay_train_set = set(train[1].tolist())
    bdecay_eval_set = set(evaluator[1].tolist())
    bg_train_set = set(train[4].tolist())
    bg_eval_set = set(evaluator[4].tolist())

    assert bdecay_train_set.isdisjoint(bdecay_eval_set)
    assert bg_train_set.isdisjoint(bg_eval_set)
    assert bdecay_train_set | bdecay_eval_set == set(bdecay_dE.tolist())
    assert bg_train_set | bg_eval_set == set(bg_dE.tolist())


def test_split_combined_tensor_is_bdecay_then_bg():
    """The combined train/eval tensors are constructed by concatenating
    bdecay-then-bg. Verify by checking the first half matches bdecay."""
    ds = _make_datasets()
    inputs = _synthetic_class_tensors(50, 50)
    train, evaluator = ds.split_dataset(*inputs)

    bdecay_train_dE = train[1]
    bg_train_dE = train[4]
    combined_train_dE = train[7]

    assert torch.equal(combined_train_dE[: bdecay_train_dE.numel()], bdecay_train_dE)
    assert torch.equal(combined_train_dE[bdecay_train_dE.numel() :], bg_train_dE)


def test_split_ground_truths_align_with_combined():
    """ground_truths_train / ground_truths_eval must be 1 for the bdecay
    half and 0 for the bg half, in lockstep with the combined tensors."""
    ds = _make_datasets()
    inputs = _synthetic_class_tensors(100, 50)
    train, evaluator = ds.split_dataset(*inputs)

    bdecay_train_dE = train[1]
    bg_train_dE = train[4]
    gt_train = train[0]

    assert gt_train.dtype == torch.bool
    assert gt_train[: bdecay_train_dE.numel()].all()
    assert not gt_train[bdecay_train_dE.numel() :].any()
    assert gt_train.numel() == bdecay_train_dE.numel() + bg_train_dE.numel()


def test_split_per_class_feature_alignment_preserved():
    """Within a class, the three feature tensors must permute together — so a
    given event's ΔE / angle / ARM stay paired across the split."""
    ds = _make_datasets()

    n_bdecay, n_bg = 80, 60
    bdecay_dE = torch.arange(n_bdecay, dtype=torch.float32)
    bdecay_angle = bdecay_dE * 10.0
    bdecay_arm = bdecay_dE * 100.0
    bg_dE = torch.arange(n_bg, dtype=torch.float32) + 1000.0
    bg_angle = bg_dE * 10.0
    bg_arm = bg_dE * 100.0

    train, _ = ds.split_dataset(bdecay_dE, bdecay_angle, bdecay_arm, bg_dE, bg_angle, bg_arm)

    bdecay_train_dE, bdecay_train_angle, bdecay_train_arm = train[1], train[2], train[3]
    bg_train_dE, bg_train_angle, bg_train_arm = train[4], train[5], train[6]

    assert torch.equal(bdecay_train_angle, bdecay_train_dE * 10.0)
    assert torch.equal(bdecay_train_arm, bdecay_train_dE * 100.0)
    assert torch.equal(bg_train_angle, bg_train_dE * 10.0)
    assert torch.equal(bg_train_arm, bg_train_dE * 100.0)


def test_split_handles_minimal_data():
    """With only a handful of events per class the split must still succeed
    (no crash) and partition them disjointly."""
    ds = _make_datasets()
    bdecay_dE = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    bdecay_angle = -bdecay_dE
    bdecay_arm = bdecay_dE * 0.01
    bg_dE = torch.tensor([10.0, 11.0, 12.0])
    bg_angle = -bg_dE
    bg_arm = bg_dE * 0.01

    train, evaluator = ds.split_dataset(bdecay_dE, bdecay_angle, bdecay_arm, bg_dE, bg_angle, bg_arm)

    assert train[1].numel() == int(5 * 0.8)  # 4
    assert evaluator[1].numel() == 5 - int(5 * 0.8)  # 1
    assert train[4].numel() == int(3 * 0.8)  # 2
    assert evaluator[4].numel() == 3 - int(3 * 0.8)  # 1


def test_split_handles_imbalanced_classes():
    """A 10:1 class imbalance must be preserved by the stratified split."""
    ds = _make_datasets()
    inputs = _synthetic_class_tensors(1000, 100)
    train, evaluator = ds.split_dataset(*inputs)

    bdecay_train_dE = train[1]
    bdecay_eval_dE = evaluator[1]
    bg_train_dE = train[4]
    bg_eval_dE = evaluator[4]

    train_ratio = bdecay_train_dE.numel() / bg_train_dE.numel()
    eval_ratio = bdecay_eval_dE.numel() / bg_eval_dE.numel()
    full_ratio = 1000 / 100

    # Stratification preserves the class ratio in each fold to within a tiny
    # rounding error from the int(n * 0.8) truncation.
    assert abs(train_ratio - full_ratio) < 0.5
    assert abs(eval_ratio - full_ratio) < 0.5


@pytest.mark.parametrize("seed", [42, 0, 1, 7, 100])
def test_split_size_independent_of_input_seed(seed):
    """The seed in split_dataset is hard-coded to 42, so the output FOLD SIZES
    are a deterministic function of input length — not of which random values
    happen to be in the inputs. Sweep different RNG seeds for the input data
    and check sizes stay constant."""
    ds = _make_datasets()
    inputs = _synthetic_class_tensors(500, 250, seed=seed)
    train, evaluator = ds.split_dataset(*inputs)
    assert train[1].numel() == int(500 * 0.8)
    assert evaluator[1].numel() == 500 - int(500 * 0.8)
    assert train[4].numel() == int(250 * 0.8)
    assert evaluator[4].numel() == 250 - int(250 * 0.8)


# ---------------------------------------------------------------------------
# Real-data tests on the full Activation.sim flow
# ---------------------------------------------------------------------------


def test_real_compute_event_features_returns_aligned_tensors(real_tensors):
    """gen_*, ground_truths must all share length; bdecay+bg must equal the
    combined-tensor length (concatenation order)."""
    n_gen = real_tensors["gen_tensor_delta_E"].numel()
    assert real_tensors["gen_tensor_annihilation_angle"].numel() == n_gen
    assert real_tensors["gen_tensor_arm"].numel() == n_gen
    assert real_tensors["ground_truths"].numel() == n_gen

    n_bdecay = real_tensors["bdecay_tensor_delta_E"].numel()
    n_bg = real_tensors["bg_tensor_delta_E"].numel()
    assert real_tensors["combined_tensor_delta_E"].numel() == n_bdecay + n_bg
    assert real_tensors["combined_tensor_annihilation_angle"].numel() == n_bdecay + n_bg
    assert real_tensors["combined_tensor_arm"].numel() == n_bdecay + n_bg


def test_real_split_class_ratios_match_full_dataset(real_tensors, train_eval_split, wandb_run):
    """The 80/20 split is stratified per class, so the β / bg ratio in train
    and in eval must match the full-dataset ratio to within 0.5 %."""
    train, evaluator = train_eval_split

    n_beta_full = int(torch.isfinite(real_tensors["bdecay_tensor_delta_E"]).sum().item())
    n_bg_full = int(torch.isfinite(real_tensors["bg_tensor_delta_E"]).sum().item())

    n_beta_train = int(torch.isfinite(train[1]).sum().item())
    n_bg_train = int(torch.isfinite(train[4]).sum().item())
    n_beta_eval = int(torch.isfinite(evaluator[1]).sum().item())
    n_bg_eval = int(torch.isfinite(evaluator[4]).sum().item())

    full_ratio = n_beta_full / max(n_bg_full, 1)
    train_ratio = n_beta_train / max(n_bg_train, 1)
    eval_ratio = n_beta_eval / max(n_bg_eval, 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["full", "train", "eval"], [full_ratio, train_ratio, eval_ratio])
    for bar, v in zip(bars, [full_ratio, train_ratio, eval_ratio]):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
    ax.set_ylabel("β / bg ratio (finite-count basis)")
    ax.set_title("Stratification check — β/bg ratio across folds")

    wandb_run.log(
        {
            "split/full_ratio_beta_over_bg": full_ratio,
            "split/train_ratio_beta_over_bg": train_ratio,
            "split/eval_ratio_beta_over_bg": eval_ratio,
            "split/n_beta_train": n_beta_train,
            "split/n_bg_train": n_bg_train,
            "split/n_beta_eval": n_beta_eval,
            "split/n_bg_eval": n_bg_eval,
            "split/ratio_bars": wandb.Image(fig),
        }
    )
    plt.close(fig)

    assert abs(train_ratio - full_ratio) / max(full_ratio, 1e-8) < 0.005
    assert abs(eval_ratio - full_ratio) / max(full_ratio, 1e-8) < 0.005


def test_real_split_class_conditional_distributions(real_tensors, train_eval_split, wandb_run):
    """Verify the train & eval per-feature distributions are visually similar
    (no accidental skew). Plot overlaid histograms for ΔE / angle / ARM,
    each per class."""
    train, evaluator = train_eval_split

    feature_pairs = [
        ("deltaE", train[1], evaluator[1], train[4], evaluator[4], "log"),
        ("angle", train[2], evaluator[2], train[5], evaluator[5], "linear"),
        ("arm", train[3], evaluator[3], train[6], evaluator[6], "log"),
    ]
    for name, beta_train, beta_eval, bg_train, bg_eval, scale in feature_pairs:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, (label, train_t, eval_t) in zip(
            axes,
            [
                ("β", beta_train, beta_eval),
                ("bg", bg_train, bg_eval),
            ],
        ):
            tr = train_t[torch.isfinite(train_t)].cpu().numpy()
            ev = eval_t[torch.isfinite(eval_t)].cpu().numpy()
            if scale == "log":
                tr = tr[tr > 0]
                ev = ev[ev > 0]
                lo = max(
                    min(float(tr.min()) if tr.size else 1e-3, float(ev.min()) if ev.size else 1e-3),
                    1e-3,
                )
                hi = max(
                    float(tr.max()) if tr.size else 1.0,
                    float(ev.max()) if ev.size else 1.0,
                    lo + 1e-6,
                )
                edges = np.geomspace(lo, hi, 50)
                ax.set_xscale("log")
            else:
                lo = float(min(tr.min(), ev.min())) if tr.size and ev.size else -1.0
                hi = float(max(tr.max(), ev.max())) if tr.size and ev.size else 1.0
                if lo == hi:
                    hi = lo + 1e-6
                edges = np.linspace(lo, hi, 50)
            ax.hist(tr, bins=edges, density=True, alpha=0.5, label=f"train (n={tr.size})")
            ax.hist(ev, bins=edges, density=True, alpha=0.5, label=f"eval (n={ev.size})")
            ax.set_xlabel(name)
            ax.set_ylabel("density")
            ax.set_title(f"{label} — train vs eval {name}")
            ax.legend()
        wandb_run.log({f"split/{name}/train_vs_eval_hist": wandb.Image(fig)})
        plt.close(fig)


def test_real_compute_event_features_finite_fractions(real_tensors, wandb_run):
    """Surface the fraction of valid (finite) events per feature per class —
    a high NaN fraction in any one is a red flag for the upstream physics."""
    rows = []
    for class_label, prefix in [("β", "bdecay_tensor_"), ("bg", "bg_tensor_")]:
        for feat in ["delta_E", "annihilation_angle", "arm"]:
            t = real_tensors[prefix + feat]
            n_total = t.numel()
            n_finite = int(torch.isfinite(t).sum().item())
            frac = n_finite / max(n_total, 1)
            rows.append((class_label, feat, n_total, n_finite, frac))

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"{c}/{f}" for c, f, *_ in rows]
    fracs = [r[-1] for r in rows]
    bars = ax.bar(labels, fracs)
    for bar, v in zip(bars, fracs):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("finite fraction")
    ax.set_title("Per-class per-feature finite-event fraction")
    plt.xticks(rotation=20, ha="right")

    wandb_run.log(
        {
            "features/finite_fraction_bars": wandb.Image(fig),
            **{f"features/{cls}/{feat}/finite_fraction": frac for cls, feat, _, _, frac in rows},
        }
    )
    plt.close(fig)

    for cls, feat, _, _, frac in rows:
        assert frac > 0.0, f"{cls}/{feat} has zero finite events — pipeline broken upstream"
