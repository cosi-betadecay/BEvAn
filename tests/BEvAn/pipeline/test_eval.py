import math

import pytest
import torch

from pipeline.datasets import BUCKETS, CLASSES
from pipeline.eval import Evaluator, prior_free_scores
from pipeline.train import Trainer


def test_evaluator_perfect_on_separable_data(separable_data):
    n = separable_data["bdecay"][1]["delta_E"].numel()
    counts = Evaluator(Trainer().fit(separable_data)).evaluate(separable_data)
    assert counts["tp"] == 3 * n
    assert counts["tn"] == 3 * n
    assert counts["fp"] == 0
    assert counts["fn"] == 0
    assert counts["excluded"] == 0


def test_evaluator_excludes_nonfinite_bucket_features(separable_data):
    n = separable_data["bdecay"][1]["delta_E"].numel()
    trainer = Trainer().fit(separable_data)
    separable_data["bg"][2]["delta_E"] = torch.cat([separable_data["bg"][2]["delta_E"], torch.tensor([75.0])])
    separable_data["bg"][2]["arm"] = torch.cat([separable_data["bg"][2]["arm"], torch.tensor([float("nan")])])
    separable_data["bg"][2]["anni"] = torch.cat(
        [separable_data["bg"][2]["anni"], torch.tensor([float("nan")])]
    )
    counts = Evaluator(trainer).evaluate(separable_data)
    assert counts["excluded"] == 1
    assert counts["tn"] == 3 * n  # the NaN-arm event is excluded, not misclassified


def test_classify_events_tags_match_labels(separable_data):
    n = separable_data["bdecay"][1]["delta_E"].numel()
    keys = {cls: {b: [(b, 1000 * (cls == "bg") + i) for i in range(n)] for b in BUCKETS} for cls in CLASSES}
    tagged = Evaluator(Trainer().fit(separable_data)).classify_events(separable_data, keys)
    assert tagged["bdecay"] == {k for b in BUCKETS for k in keys["bdecay"][b]}
    assert tagged["bg"] == {k for b in BUCKETS for k in keys["bg"][b]}


def test_classify_events_asserts_on_misaligned_keys(separable_data):
    n = separable_data["bdecay"][1]["delta_E"].numel()
    keys = {cls: {b: [(b, i) for i in range(n - 1)] for b in BUCKETS} for cls in CLASSES}
    with pytest.raises(AssertionError, match="misaligned"):
        Evaluator(Trainer().fit(separable_data)).classify_events(separable_data, keys)


def test_pooled_log_ratio_and_prior_free_scores(separable_data):
    evaluator = Evaluator(Trainer().fit(separable_data))
    pooled = evaluator.pooled_log_ratio(separable_data)
    assert pooled is not None
    scores = prior_free_scores(pooled)
    assert scores["best_f1"] == pytest.approx(1.0)
    assert scores["auc"] == pytest.approx(1.0)


def test_prior_free_scores_none_gives_nan():
    scores = prior_free_scores(None)
    assert math.isnan(scores["best_f1"])
    assert math.isnan(scores["auc"])
