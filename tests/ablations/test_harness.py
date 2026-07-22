import math

import harness
import pytest
import torch

from pipeline.datasets import BUCKETS, CLASSES, FEATURES
from pipeline.eval import Evaluator


def make_data_dir(tmp_path, monkeypatch, folders: dict[str, list[str]]):
    """Point harness at a tmp data/ tree holding the given per-folder filenames."""
    data_dir = tmp_path / "data"
    for name, files in folders.items():
        folder = data_dir / name
        folder.mkdir(parents=True)
        for filename in files:
            (folder / filename).write_text("")
    monkeypatch.setattr(harness, "DATA_DIR", data_dir)
    monkeypatch.setattr(harness, "GEOMETRIES", {"good": "geo.setup", "no_tra": "x", "no_prior": "x"})
    return data_dir


def test_discover_datasets_complete_folder_only(tmp_path, monkeypatch, capsys):
    make_data_dir(
        tmp_path,
        monkeypatch,
        {
            "good": ["good.sim", "good.tra", "good_P1.sim", "good_P2.sim"],
            "no_tra": ["no_tra.sim", "no_tra_P1.sim"],
            "no_prior": ["no_prior.sim", "no_prior.tra"],
            "no_geometry": ["no_geometry.sim", "no_geometry.tra", "no_geometry_P1.sim"],
            "misnamed": ["other.sim"],
            "cosima": ["source.txt"],
        },
    )
    datasets = harness.discover_datasets()

    assert [d["name"] for d in datasets] == ["good"]
    good = datasets[0]
    assert good["geo_file"] == "geo.setup"
    assert good["sim_file"].endswith("good/good.sim")
    assert good["tra_file"].endswith("good/good.tra")
    assert [p[-11:] for p in good["prior_sims"]] == ["good_P1.sim", "good_P2.sim"]

    skipped = capsys.readouterr().out
    for fragment in ("no_tra", "no_prior", "no_geometry", "misnamed"):
        assert fragment in skipped
    assert "cosima" not in skipped  # folders with no .sim at all are ignored silently


def test_aggregate_seed_records_mean_std_and_passthrough():
    records = [
        {"f1_score": 0.8, "tp": 1, "weights": {"arm": 1.0}},
        {"f1_score": 0.6, "tp": 3, "weights": {"arm": 2.0}},
    ]
    agg = harness.aggregate_seed_records(records)
    assert agg["f1_score"] == pytest.approx(0.7)
    assert agg["f1_score_std"] == pytest.approx(0.1)
    assert agg["tp"] == pytest.approx(2.0)
    assert agg["weights"] == {"arm": 1.0}  # non-scalars carried from the first seed


def test_aggregate_seed_records_needs_two_finite_for_std():
    single = harness.aggregate_seed_records([{"f1_score": 0.8}])
    assert single["f1_score"] == pytest.approx(0.8)
    assert math.isnan(single["f1_score_std"])
    one_finite = harness.aggregate_seed_records([{"f1_score": float("nan")}, {"f1_score": 0.6}])
    assert one_finite["f1_score"] == pytest.approx(0.6)
    assert math.isnan(one_finite["f1_score_std"])


def test_extract_split_seed_varies_partition(tiny_ds, tiny_cache):
    train_default, _ = harness.extract_split(tiny_ds)
    train_seeded, _ = harness.extract_split(tiny_ds, seed=43)
    assert train_seeded["bdecay"][2]["delta_E"].numel() == train_default["bdecay"][2]["delta_E"].numel()
    assert any(
        not torch.equal(train_seeded[cls][b]["delta_E"], train_default[cls][b]["delta_E"])
        for cls in CLASSES
        for b in BUCKETS
    )


def test_extract_split_writes_and_reuses_cache(tiny_ds, tiny_cache):
    train, eval_data = harness.extract_split(tiny_ds)
    assert (tiny_cache / "tiny_ckd.pt").is_file()
    for cls in CLASSES:
        for b in BUCKETS:
            assert set(train[cls][b]) == set(FEATURES)
    # Bucket 2 holds 6 events per class, split 80/20 by the fixed-seed shuffle.
    assert train["bdecay"][2]["delta_E"].numel() == 4
    assert eval_data["bdecay"][2]["delta_E"].numel() == 2

    def boom(*args, **kwargs):
        raise AssertionError("cache miss: re-extracted despite existing cache")

    original = harness.__dict__["_extract_payload"]
    harness._extract_payload = boom
    try:
        train_again, _ = harness.extract_split(tiny_ds)
    finally:
        harness._extract_payload = original
    assert torch.equal(train_again["bdecay"][2]["delta_E"], train["bdecay"][2]["delta_E"])


def test_extract_keyed_layout_and_sky(tiny_ds, tiny_cache):
    data, keys, sky_image, n_events = harness.extract_keyed(tiny_ds)
    assert sky_image.shape == (180, 360)
    assert n_events == 6  # the six Compton events of the synthetic .tra
    assert sky_image.sum().item() == pytest.approx(6.0)
    for cls in CLASSES:
        for b in BUCKETS:
            assert data[cls][b]["delta_E"].numel() == len(keys[cls][b])
    total = sum(len(keys[cls][b]) for cls in CLASSES for b in BUCKETS)
    assert total == 30  # every non-empty synthetic event lands in exactly one cell


def test_extract_pool_caches_flat_pool(tiny_ds, tiny_cache):
    pool = harness.extract_pool(tiny_ds)
    assert (tiny_cache / "tiny_ckd_pool.pt").is_file()
    assert pool["bucket"].numel() == 30
    assert set(pool) == {"bucket", *FEATURES, "label_score"}
    again = harness.extract_pool(tiny_ds)
    assert torch.equal(again["label_score"], pool["label_score"])


def test_reference_uses_default_trainer_config(tiny_split):
    train, _ = tiny_split
    trainer, config = harness.reference(train)
    assert config == harness.Trainer().config()
    assert set(trainer.models) == set(BUCKETS)


def test_pooled_counts_match_evaluator_totals(tiny_split):
    train, eval_data = tiny_split
    trainer, _ = harness.reference(train)
    evaluator = Evaluator(trainer)
    assert harness.pooled_counts(evaluator, eval_data) == evaluator.evaluate(eval_data)


def test_metric_record_keys(tiny_split):
    train, eval_data = tiny_split
    trainer, _ = harness.reference(train)
    record = harness.metric_record(trainer, eval_data)
    expected = {
        "tp",
        "fp",
        "fn",
        "tn",
        "excluded",
        "precision",
        "recall",
        "fpr",
        "f1_score",
        "best_f1",
        "auc",
    }
    assert expected <= set(record)
    assert record["tp"] + record["fp"] + record["fn"] + record["tn"] + record["excluded"] > 0


def test_save_model_plots_writes_figure_set(tiny_split, tmp_path):
    train, eval_data = tiny_split
    trainer, _ = harness.reference(train)
    out = tmp_path / "reference"
    harness.save_model_plots(trainer, eval_data, out)
    assert (out / "confusion_matrix_eval.png").is_file()
    assert (out / "roc_eval.png").is_file()
    assert (out / "pr_eval.png").is_file()
    assert list(out.glob("density_*.png"))


def test_best_f1_record_on_separable_split(tiny_split):
    train, eval_data = tiny_split
    trainer, _ = harness.reference(train)
    record = harness.best_f1_record(trainer, train, eval_data)
    assert 0.0 <= record["f1_score"] <= 1.0
    assert record["tp"] + record["fn"] == int(sum(eval_data["bdecay"][b]["delta_E"].numel() for b in BUCKETS))


def test_best_f1_record_empty_split_is_nan(tiny_split):
    train, _ = tiny_split
    trainer, _ = harness.reference(train)
    empty = {cls: {b: {f: torch.tensor([]) for f in FEATURES} for b in BUCKETS} for cls in CLASSES}
    record = harness.best_f1_record(trainer, empty, empty)
    assert record["tp"] == 0
    assert math.isnan(record["best_f1"])
    assert math.isnan(record["auc"])


def test_term_factor_dispatch():
    assert harness.term_factor({"kind": "1d", "xfeat": "delta_E"}) == "delta_E"
    assert harness.term_factor({"kind": "2d", "xfeat": "delta_E", "yfeat": "arm"}) == "arm"


def test_all_factors_finite_mask():
    nan = float("nan")
    bucket = {
        "delta_E": torch.tensor([1.0, 1.0, nan]),
        "arm": torch.tensor([0.1, nan, 0.1]),
        "anni": torch.tensor([-0.9, -0.9, -0.9]),
    }
    assert harness.all_factors_finite(bucket).tolist() == [True, False, False]


def test_bins_from_counts_scaling():
    assert harness.bins_from_counts(0, d=1, rho_floor=10.0) == 2
    assert harness.bins_from_counts(1000, d=1, rho_floor=10.0) == 100
    assert harness.bins_from_counts(1000, d=2, rho_floor=10.0) == 10
    assert harness.bins_from_counts(5, d=1, rho_floor=10.0, min_bins=4) == 4


def test_factor_1d_llr_scores_common_population(tiny_split):
    train, eval_data = tiny_split
    scored = harness.factor_1d_llr(train, eval_data, "delta_E")
    assert scored is not None
    llr, labels = scored
    n_eval = eval_data["bdecay"][3]["delta_E"].numel() + eval_data["bg"][3]["delta_E"].numel()
    assert llr.shape == labels.shape == (n_eval,)
    assert labels.sum() == eval_data["bdecay"][3]["delta_E"].numel()


def test_factor_1d_llr_none_when_class_empty(tiny_split):
    train, eval_data = tiny_split
    for f in FEATURES:
        train["bdecay"][3][f] = torch.tensor([])
    assert harness.factor_1d_llr(train, eval_data, "delta_E") is None


def test_bucket_full_scores(tiny_split):
    train, eval_data = tiny_split
    trainer, _ = harness.reference(train)
    scores = harness.bucket_full_scores(trainer, eval_data)
    assert set(scores) == {"f1", "auc"}
    assert 0.0 <= scores["auc"] <= 1.0

    empty = {cls: {b: {f: torch.tensor([]) for f in FEATURES} for b in BUCKETS} for cls in CLASSES}
    nan_scores = harness.bucket_full_scores(trainer, empty)
    assert math.isnan(nan_scores["f1"]) and math.isnan(nan_scores["auc"])


def test_term_llr_columns_zero_for_absent_factors(tiny_split):
    train, _ = tiny_split
    trainer, _ = harness.reference(train)
    x, y = harness.term_llr_columns(Evaluator(trainer), train)
    assert x.shape == (y.numel(), len(harness.FACTORS))
    # Bucket 1 contributes only the delta_E term, so some rows must have zero
    # arm/anni columns while delta_E columns are populated overall.
    assert (x[:, harness.FACTOR_INDEX["arm"]] == 0).any()
    assert (x[:, harness.FACTOR_INDEX["delta_E"]] != 0).any()


def test_term_llr_columns_empty_data(tiny_split):
    train, _ = tiny_split
    trainer, _ = harness.reference(train)
    empty = {cls: {b: {f: torch.tensor([]) for f in FEATURES} for b in BUCKETS} for cls in CLASSES}
    x, y = harness.term_llr_columns(Evaluator(trainer), empty)
    assert x.shape == (0, len(harness.FACTORS))
    assert y.numel() == 0


def test_fit_logistic_separable_sign():
    x = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    y = torch.tensor([False, False, True, True])
    weights, bias = harness.fit_logistic(x, y)
    assert weights.shape == (1,)
    assert weights[0] > 0
    logits = x.double() @ weights + bias
    assert ((logits > 0) == y).all()
