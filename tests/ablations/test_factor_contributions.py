import factor_contributions
import harness
import torch

from pipeline.datasets import FEATURES


def test_run_scores_each_factor_on_the_common_population(tiny_split):
    train, eval_data = tiny_split
    contributions = factor_contributions.run(train, eval_data)
    assert set(contributions) <= set(harness.FACTORS)
    assert contributions  # the synthetic bucket 3 has both classes, so factors score
    for record in contributions.values():
        assert set(record) == {"f1", "auc"}
        assert 0.0 <= record["auc"] <= 1.0
        assert 0.0 <= record["f1"] <= 1.0


def test_run_skips_factors_that_cannot_score(tiny_split):
    train, eval_data = tiny_split
    for f in FEATURES:
        train["bdecay"][3][f] = torch.tensor([])
    assert factor_contributions.run(train, eval_data) == {}
