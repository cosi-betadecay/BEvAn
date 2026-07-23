from pathlib import Path

import harness
import timing


def test_run_reports_every_stage(tiny_ds: harness.DatasetPaths, tiny_cache: Path) -> None:
    """Verify timing.run reports every timed stage with positive, ordered measurements."""
    record = timing.run(tiny_ds, repeats=2)
    expected = {
        "n_events",
        "n_eval_events",
        "extract_s",
        "extract_us_per_event",
        "fit_s",
        "score_s",
        "score_events_per_s_batched",
        "extract_over_score_per_event",
        "artifact_kb",
        "n_density_cells",
    }
    assert expected <= set(record)
    assert record["extract_s"] > 0
    assert record["fit_s"] > 0
    assert record["score_s"] > 0
    assert record["n_eval_events"] < record["n_events"]


def test_run_bypasses_the_feature_cache(tiny_ds: harness.DatasetPaths, tiny_cache: Path) -> None:
    """The whole point of the module: it must extract, not read a cached payload.

    Priming the cache first would make a cache-reading implementation look fast and
    correct, so the assertion is that the cache is still absent afterwards.
    """
    timing.run(tiny_ds, repeats=1)
    assert not (tiny_cache / "tiny_ckd.pt").exists()


def test_n_events_counts_both_classes_and_every_bucket(tiny_split: tuple[dict, dict]) -> None:
    """Verify n_events sums events across both classes and every bucket."""
    train, eval_data = tiny_split
    expected = sum(train[cls][b]["delta_E"].numel() for cls in ("bdecay", "bg") for b in harness.BUCKETS)
    assert timing.n_events(train) == expected
    assert timing.n_events(eval_data) > 0


def test_density_cells_match_the_configured_bin_budget(tiny_split: tuple[dict, dict]) -> None:
    """Cell count is the bin budget, not the data: 2 classes x (25 + 8^2 + 2*35^2)."""
    train, _ = tiny_split
    trainer = harness.Trainer().fit(train)
    assert timing.n_density_cells(trainer) == 2 * (25 + 8**2 + 2 * 35**2)


def test_artifact_kb_is_positive_and_matches_a_saved_file(
    tiny_split: tuple[dict, dict], tmp_path: Path
) -> None:
    """Verify artifact_kb reports the trained model's on-disk size in kilobytes."""
    train, _ = tiny_split
    trainer = harness.Trainer().fit(train)
    path = tmp_path / "model.pt"
    trainer.save(path)
    assert timing.artifact_kb(trainer) == path.stat().st_size / 1024


def test_environment_names_the_machine() -> None:
    """Verify environment reports the platform, interpreter, and torch runtime details."""
    env = timing.environment()
    assert {"platform", "processor", "python", "torch", "torch_threads", "cuda_available"} <= set(env)
    assert env["torch_threads"] >= 1
