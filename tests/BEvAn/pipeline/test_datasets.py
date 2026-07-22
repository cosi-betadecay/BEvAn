import math

import pytest
import torch
from fake_megalib import SimEvent, SimHit, SimIA

from pipeline.datasets import BUCKETS, CLASSES, FEATURES, Datasets, pool_to_features, split_features

Z_HAT = torch.tensor([0.0, 0.0, 1.0])


class ListReader:
    """Minimal reader satisfying the MFileEventsSim protocol, for synthetic events."""

    def __init__(self, events: list[SimEvent]):
        self._it = iter(events)

    def GetNextEvent(self) -> SimEvent | None:
        return next(self._it, None)

    def Close(self) -> None:
        return None


def _beta_ias() -> list[SimIA]:
    """IAs making every hit with origin id 2 an ANNI secondary deposit."""
    return [SimIA("ANNI", 1, 0), SimIA("COMP", 2, 1)]


def _events() -> list[SimEvent]:
    """Four events covering: bucket 1 (beta), empty (skipped), bucket 2 (bg), bucket 3 (bg)."""
    bucket1 = SimEvent(1, hits=[SimHit((1.0, 1.0, 1.0), 511.0, origins=(2,))], ias=_beta_ias())
    empty = SimEvent(2)
    bucket2 = SimEvent(3, hits=[SimHit((0.0, 0.0, 0.0), 255.5), SimHit((0.0, 1.0, 0.0), 255.5)])
    bucket3 = SimEvent(
        4,
        hits=[
            SimHit((0.0, 0.0, 0.0), 10.0),
            SimHit((1.0, 0.0, 0.0), 511.0),
            SimHit((-1.0, 0.0, 0.0), 511.0),
        ],
    )
    return [bucket1, empty, bucket2, bucket3]


def _dataset() -> Datasets:
    return Datasets(ListReader(_events()), Z_HAT)


def test_feature_bucket_routing():
    ds = _dataset()
    nan = float("nan")
    assert ds.feature_bucket(nan, nan) == 1
    assert ds.feature_bucket(0.1, nan) == 2
    assert ds.feature_bucket(0.1, -0.9) == 3
    assert ds.feature_bucket(nan, -0.9) == 3  # anni alone routes to 3


def test_compute_event_features_routes_and_labels():
    data = _dataset().compute_event_features()

    assert data["bdecay"][1]["delta_E"].tolist() == pytest.approx([0.0])
    assert math.isnan(data["bdecay"][1]["arm"].item())
    assert math.isnan(data["bdecay"][1]["anni"].item())

    assert data["bg"][2]["delta_E"].item() == pytest.approx(0.0)
    assert data["bg"][2]["arm"].item() == pytest.approx(0.0, abs=1e-6)
    assert math.isnan(data["bg"][2]["anni"].item())

    assert data["bg"][3]["anni"].item() == pytest.approx(-1.0)

    empty = [(cls, b) for cls in CLASSES for b in BUCKETS if data[cls][b]["delta_E"].numel() == 0]
    assert len(empty) == 3  # everything except the three populated (class, bucket) cells


def test_compute_event_features_keyed_colocation():
    data, keys = _dataset().compute_event_features_keyed()
    assert keys["bdecay"][1] == [(0, 1)]
    assert keys["bg"][2] == [(0, 3)]
    assert keys["bg"][3] == [(0, 4)]  # the empty event advanced the ID tracking
    for cls in CLASSES:
        for b in BUCKETS:
            for f in FEATURES:
                assert data[cls][b][f].numel() == len(keys[cls][b])


def test_compute_event_pool_relabel_parity():
    features = Datasets(ListReader(_events()), Z_HAT).compute_event_features()
    pool = Datasets(ListReader(_events()), Z_HAT).compute_event_pool()
    relabeled = pool_to_features(pool, n_std=5)
    for cls in CLASSES:
        for b in BUCKETS:
            for f in FEATURES:
                assert torch.equal(features[cls][b][f].isnan(), relabeled[cls][b][f].isnan()), (
                    f"{cls}/{b}/{f}"
                )
                assert torch.allclose(features[cls][b][f], relabeled[cls][b][f], equal_nan=True)


def test_pool_to_features_moves_labels_with_the_window():
    pool = {
        "bucket": torch.tensor([1, 1]),
        "delta_E": torch.tensor([0.5, 1.5]),
        "arm": torch.tensor([float("nan")] * 2),
        "anni": torch.tensor([float("nan")] * 2),
        "label_score": torch.tensor([2.0, 100.0], dtype=torch.float64),
    }
    wide = pool_to_features(pool, n_std=5)  # tolerance ~4.78: score 2 -> beta
    narrow = pool_to_features(pool, n_std=1)  # tolerance ~0.96: nothing passes
    assert wide["bdecay"][1]["delta_E"].tolist() == [0.5]
    assert narrow["bdecay"][1]["delta_E"].numel() == 0
    assert narrow["bg"][1]["delta_E"].numel() == 2


def test_split_features_deterministic_and_partitioning():
    data = {
        cls: {b: {f: torch.arange(10, dtype=torch.float32) + (100 * b) for f in FEATURES} for b in BUCKETS}
        for cls in CLASSES
    }
    train_a, eval_a = split_features(data, train_percentage=0.8)
    train_b, eval_b = split_features(data, train_percentage=0.8)
    for cls in CLASSES:
        for b in BUCKETS:
            assert train_a[cls][b]["delta_E"].numel() == 8
            assert eval_a[cls][b]["delta_E"].numel() == 2
            assert torch.equal(train_a[cls][b]["delta_E"], train_b[cls][b]["delta_E"])
            assert torch.equal(eval_a[cls][b]["delta_E"], eval_b[cls][b]["delta_E"])
            combined = torch.cat([train_a[cls][b]["delta_E"], eval_a[cls][b]["delta_E"]])
            assert torch.equal(combined.sort().values, data[cls][b]["delta_E"])


def test_split_features_seed_varies_partition():
    data = {
        cls: {b: {f: torch.arange(10, dtype=torch.float32) + (100 * b) for f in FEATURES} for b in BUCKETS}
        for cls in CLASSES
    }
    train_default, _ = split_features(data)
    train_42, _ = split_features(data, seed=42)
    train_7, eval_7 = split_features(data, seed=7)
    for cls in CLASSES:
        for b in BUCKETS:
            assert torch.equal(train_default[cls][b]["delta_E"], train_42[cls][b]["delta_E"])
            assert train_7[cls][b]["delta_E"].numel() == 8
            combined = torch.cat([train_7[cls][b]["delta_E"], eval_7[cls][b]["delta_E"]])
            assert torch.equal(combined.sort().values, data[cls][b]["delta_E"])
    assert any(
        not torch.equal(train_7[cls][b]["delta_E"], train_42[cls][b]["delta_E"])
        for cls in CLASSES
        for b in BUCKETS
    )
