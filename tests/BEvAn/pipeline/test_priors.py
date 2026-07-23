import math
from pathlib import Path
from typing import ClassVar

import pytest
from fake_megalib import iter_sim_events

from physics.ground_truths import ground_truth_bdecay
from pipeline.priors import (
    apply_prior_counts,
    bucket_priors,
    count_prior_events,
    count_prior_pool,
    pool_prior_counts,
    validate_prior_counts,
    validate_prior_sims,
)


def test_validate_prior_sims_rejects_bad_paths(tmp_path: Path) -> None:
    """Reject empty, wrong-extension, and missing prior-sim paths; accept a valid one."""
    with pytest.raises(ValueError, match="At least one"):
        validate_prior_sims([])
    with pytest.raises(ValueError, match=r"\.sim"):
        validate_prior_sims([str(tmp_path / "priors.tra")])
    with pytest.raises(FileNotFoundError):
        validate_prior_sims([str(tmp_path / "missing.sim")])
    ok = tmp_path / "ok.sim"
    ok.write_text("SE\nID 1 1\nEN\n")
    validate_prior_sims([str(ok)])


def test_count_prior_events_matches_direct_parse(geo_path: Path, sim_path: Path) -> None:
    """Match count_prior_events against a direct per-bucket parse of the sim file."""
    counts = count_prior_events(str(geo_path), [str(sim_path)])

    expected = {b: {"bdecay": 0, "bg": 0} for b in (1, 2, 3)}
    for event in iter_sim_events(sim_path):
        n_hits = event.GetNHTs()
        if n_hits == 0:
            continue
        cls = "bdecay" if ground_truth_bdecay(event) else "bg"
        expected[min(n_hits, 3)][cls] += 1

    assert counts == expected
    assert sum(sum(c.values()) for c in counts.values()) > 0


def test_count_prior_events_sums_over_files(geo_path: Path, sim_path: Path) -> None:
    """Sum prior counts linearly across repeated sim files."""
    single = count_prior_events(str(geo_path), [str(sim_path)])
    double = count_prior_events(str(geo_path), [str(sim_path), str(sim_path)])
    for b in (1, 2, 3):
        for cls in ("bdecay", "bg"):
            assert double[b][cls] == 2 * single[b][cls]


@pytest.mark.parametrize("n_std", [1, 2, 5, 10])
def test_pool_prior_counts_matches_a_fresh_count(geo_path: Path, sim_path: Path, n_std: int) -> None:
    """The pooled prior must reproduce a fresh stream at every window, not just 5."""
    pool = count_prior_pool(str(geo_path), [str(sim_path)])

    expected = {b: {"bdecay": 0, "bg": 0} for b in (1, 2, 3)}
    for event in iter_sim_events(sim_path):
        n_hits = event.GetNHTs()
        if n_hits == 0:
            continue
        cls = "bdecay" if ground_truth_bdecay(event, n_std) else "bg"
        expected[min(n_hits, 3)][cls] += 1

    assert pool_prior_counts(pool, n_std) == expected


def test_pool_prior_counts_at_the_deployed_window_matches_count_prior_events(
    geo_path: Path, sim_path: Path
) -> None:
    """Reproduce count_prior_events from the pool at the deployed window of 5."""
    pool = count_prior_pool(str(geo_path), [str(sim_path)])
    assert pool_prior_counts(pool, 5) == count_prior_events(str(geo_path), [str(sim_path)])


def test_count_prior_pool_skips_zero_hit_events_and_sums_over_files(
    geo_path: Path, sim_path: Path
) -> None:
    """Skip zero-hit events and sum pool rows across repeated files."""
    n_nonempty = sum(1 for e in iter_sim_events(sim_path) if e.GetNHTs() > 0)
    single = count_prior_pool(str(geo_path), [str(sim_path)])
    double = count_prior_pool(str(geo_path), [str(sim_path), str(sim_path)])
    assert single["bucket"].numel() == n_nonempty
    assert single["label_score"].numel() == n_nonempty
    assert double["bucket"].numel() == 2 * n_nonempty


def test_validate_prior_counts_raises_on_empty_bucket() -> None:
    """Raise when a bucket has no prior events at all."""
    counts = {1: {"bdecay": 1, "bg": 1}, 2: {"bdecay": 0, "bg": 0}, 3: {"bdecay": 1, "bg": 1}}
    with pytest.raises(ValueError, match="bucket 2"):
        validate_prior_counts(counts)


def test_validate_prior_counts_warns_on_single_class_bucket() -> None:
    """Warn when a bucket holds only one class of prior events."""
    counts = {1: {"bdecay": 1, "bg": 1}, 2: {"bdecay": 3, "bg": 0}, 3: {"bdecay": 1, "bg": 1}}
    with pytest.warns(UserWarning, match="Bucket 2"):
        validate_prior_counts(counts)


def test_bucket_priors_fractions_and_nan() -> None:
    """Compute per-bucket beta fractions, yielding NaN for an empty bucket."""
    counts = {1: {"bdecay": 1, "bg": 3}, 2: {"bdecay": 0, "bg": 0}, 3: {"bdecay": 2, "bg": 2}}
    priors = bucket_priors(counts)
    assert priors[1] == pytest.approx(0.25)
    assert math.isnan(priors[2])
    assert priors[3] == pytest.approx(0.5)


def test_apply_prior_counts_overwrites_only_fitted_buckets() -> None:
    """Overwrite counts only in buckets that have fitted density terms."""

    class TrainerStub:
        models: ClassVar[dict[int, dict]] = {
            1: {"n_beta": 5, "n_bg": 5, "terms": []},
            2: {"n_beta": 5, "n_bg": 5, "terms": [{"kind": "1d"}]},
            3: {"n_beta": 5, "n_bg": 5, "terms": [{"kind": "2d"}]},
        }

    counts = {b: {"bdecay": 10 * b, "bg": 20 * b} for b in (1, 2, 3)}
    trainer = TrainerStub()
    apply_prior_counts(trainer, counts)
    assert trainer.models[1]["n_beta"] == 5  # terms-less bucket untouched
    assert trainer.models[2]["n_beta"] == 20
    assert trainer.models[2]["n_bg"] == 40
    assert trainer.models[3]["n_beta"] == 30
    assert trainer.models[3]["n_bg"] == 60
