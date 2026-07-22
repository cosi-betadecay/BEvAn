import math

import pytest
import torch

from modeling.metrics import best_f1_threshold, metrics, roc_auc


def test_metrics_hand_values():
    scores = metrics({"tp": 8, "fp": 2, "fn": 4, "tn": 6})
    assert scores["precision"] == pytest.approx(0.8)
    assert scores["recall"] == pytest.approx(8 / 12)
    assert scores["fpr"] == pytest.approx(0.25)
    assert scores["f1_score"] == pytest.approx(2 * 0.8 * (8 / 12) / (0.8 + 8 / 12))


def test_metrics_zero_denominators():
    scores = metrics({"tp": 0, "fp": 0, "fn": 0, "tn": 5})
    assert scores == {"precision": 0.0, "recall": 0.0, "fpr": 0.0, "f1_score": 0.0}


def test_roc_auc_perfect_random_and_ties():
    labels = torch.tensor([False, False, True, True])
    assert roc_auc(torch.tensor([0.0, 1.0, 2.0, 3.0]), labels) == pytest.approx(1.0)
    assert roc_auc(torch.tensor([3.0, 2.0, 1.0, 0.0]), labels) == pytest.approx(0.0)
    assert roc_auc(torch.zeros(4), labels) == pytest.approx(0.5)
    assert math.isnan(roc_auc(torch.tensor([1.0]), torch.tensor([True])))


def test_best_f1_threshold_hand_case():
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    labels = torch.tensor([True, True, False, True])
    best, threshold = best_f1_threshold(scores, labels)
    assert best == pytest.approx(6 / 7)  # cut at 0.6: tp=3, fp=1, fn=0
    assert threshold == pytest.approx(0.6)
    predicted_f1 = metrics(
        {
            "tp": int(((scores >= threshold) & labels).sum()),
            "fp": int(((scores >= threshold) & ~labels).sum()),
            "fn": int(((scores < threshold) & labels).sum()),
            "tn": int(((scores < threshold) & ~labels).sum()),
        }
    )["f1_score"]
    assert predicted_f1 == pytest.approx(best)


def test_best_f1_threshold_no_positives():
    best, threshold = best_f1_threshold(torch.tensor([1.0, 2.0]), torch.tensor([False, False]))
    assert math.isnan(best)
    assert math.isnan(threshold)
