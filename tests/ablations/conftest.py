import functools
import sys
from pathlib import Path

import pytest

# The ablation modules import each other as top-level names (`import harness`),
# the way the driver runs them from ablations/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ablations"))

import harness


@pytest.fixture()
def tiny_ds(synthetic_dataset) -> harness.DatasetPaths:
    """The synthetic dataset shaped as a discovered ablation dataset entry."""
    ds = synthetic_dataset
    return {
        "name": "tiny",
        "geo_file": ds["geo"],
        "sim_file": ds["sim"],
        "tra_file": ds["tra"],
        "prior_sims": [ds["prior"]],
    }


@pytest.fixture()
def tiny_cache(tmp_path, monkeypatch) -> Path:
    """Redirect the harness extraction cache into tmp_path.

    The cache-dir default is bound to ablations/cache at import time, so the
    extract functions are wrapped rather than the CACHE_DIR constant patched.
    """
    cache = tmp_path / "ablation_cache"
    for name in ("extract_split", "extract_keyed", "extract_pool", "extract_prior_pool"):
        monkeypatch.setattr(harness, name, functools.partial(getattr(harness, name), cache_dir=cache))
    return cache


@pytest.fixture()
def tiny_split(tiny_ds, tiny_cache):
    """The synthetic dataset's cached (train, eval) reference-ordering split."""
    return harness.extract_split(tiny_ds)
