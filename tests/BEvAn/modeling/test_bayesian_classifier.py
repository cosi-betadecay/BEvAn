import math

import torch

from modeling.bayesian_classifier import BayesianClassifier


def test_decision_rule_at_count_prior():
    log_ratio = torch.tensor([-1.0, 0.0, 1.0])
    assert BayesianClassifier(log_ratio, 10, 10).decision_rule().tolist() == [False, True, True]
    # Prior 3:1 for background moves the cut to log(3).
    threshold = math.log(3.0)
    preds = BayesianClassifier(torch.tensor([threshold - 0.01, threshold + 0.01]), 10, 30).decision_rule()
    assert preds.tolist() == [False, True]


def test_decision_rule_degenerate_priors_and_nan():
    log_ratio = torch.tensor([-5.0, 5.0, float("nan")])
    assert BayesianClassifier(log_ratio, 10, 0).decision_rule().tolist() == [True, True, False]
    assert BayesianClassifier(log_ratio, 0, 10).decision_rule().tolist() == [False, False, False]
