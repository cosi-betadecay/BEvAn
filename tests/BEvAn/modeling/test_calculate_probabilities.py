import torch

from modeling.calculate_probabilities import confusion_counts, log_confusion, log_R


def test_log_R_sums_term_log_ratios() -> None:
    """Sum per-term log density ratios into the total log R."""
    terms = [
        (torch.tensor([0.4, 0.1]), torch.tensor([0.1, 0.4])),
        (torch.tensor([0.2, 0.2]), torch.tensor([0.2, 0.2])),
    ]
    expected = torch.log(torch.tensor([0.4, 0.1]) + 1e-8) - torch.log(torch.tensor([0.1, 0.4]) + 1e-8)
    assert torch.allclose(log_R(terms), expected)


def test_confusion_counts_tallies_and_excludes_nan() -> None:
    """Tally confusion counts and exclude NaN-scored events."""
    truths = torch.tensor([True, True, False, False, True])
    p_beta = torch.tensor([0.9, 0.1, 0.9, 0.1, float("nan")])
    p_bg = torch.tensor([0.1, 0.9, 0.1, 0.9, 0.5])
    counts = confusion_counts(truths, [(p_beta, p_bg)], n_beta_decay=10, n_bg=10)
    assert counts == {"tp": 1, "fp": 1, "fn": 1, "tn": 1, "excluded": 1}


def test_confusion_counts_no_terms_decides_on_prior_alone() -> None:
    """Decide on the count prior alone when no density terms are given."""
    truths = torch.tensor([True, False])
    counts = confusion_counts(truths, [], n_beta_decay=20, n_bg=10)
    assert counts == {"tp": 1, "fp": 1, "fn": 0, "tn": 0, "excluded": 0}


def test_log_confusion_noop_without_wandb_run() -> None:
    """Return None from log_confusion when no wandb run is active."""
    assert log_confusion({"tp": 1, "fp": 0, "fn": 0, "tn": 1, "excluded": 0}) is None
