import csv
import os
from datetime import datetime
from pathlib import Path

import hydra
import omegaconf
import torch
from dataset.datasets import Datasets
from dotenv import load_dotenv
from physics.compton_cone_reconstruction import FarFieldImager
from pipeline.eval import Evaluator
from pipeline.train import Trainer
from utils.reader_extraction import get_reader

import wandb

# Directory holding the {name}.sim / {name}.tra pairs, relative to the repo root
# (hydra runs with chdir disabled, so the cwd is the invocation directory).
DATA_DIR = Path("data")
# Where per-run result CSVs are written (one timestamped file per total run).
RESULTS_DIR = Path("results")


def _metrics(counts: dict) -> dict:
    """Derive precision/recall/FPR/F1 from raw confusion counts.

    Matches the definitions in modeling.calculate_probablities.log_confusion.
    """
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "fpr": fpr, "f1_score": f1}


def _auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Prior-free ROC AUC via the tie-aware Mann-Whitney rank statistic.

    Invariant to any monotonic transform of ``scores`` (so the log-ratio gives
    the same answer as the ratio). Returns NaN if either class is empty.
    """
    labels = labels.bool()
    n_pos = int(labels.sum())
    n_neg = labels.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = torch.argsort(scores)
    s_sorted = scores[order]
    _, inv, counts = torch.unique_consecutive(s_sorted, return_inverse=True, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts  # 0-based start index of each tie group
    avg_rank_group = starts.double() + (counts.double() + 1) / 2  # 1-based average rank
    ranks_sorted = avg_rank_group[inv]
    ranks = torch.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted

    r_pos = ranks[labels].sum().item()
    return (r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _best_f1(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Best F1 achievable by sweeping a threshold over the prior-free score.

    Cuts are only allowed between distinct score values (tie-aware). Returns
    NaN if there are no positives.
    """
    labels = labels.bool()
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")

    order = torch.argsort(scores, descending=True)
    s_sorted = scores[order]
    y_sorted = labels[order].double()
    tp_cum = torch.cumsum(y_sorted, 0)
    k = torch.arange(1, s_sorted.numel() + 1, dtype=torch.double)
    f1 = 2 * tp_cum / (k + n_pos)

    # Valid threshold boundaries: last element, or where the next score differs.
    boundary = torch.ones(s_sorted.numel(), dtype=torch.bool)
    boundary[:-1] = s_sorted[:-1] != s_sorted[1:]
    return float(f1[boundary].max().item())


def _prior_free_scores(evaluator, data) -> dict:
    """Prior-free best-F1 and AUC over the pooled per-event likelihood ratio."""
    pooled = evaluator.pooled_log_ratio(data)
    if pooled is None:
        return {"best_f1": float("nan"), "auc": float("nan")}
    scores, labels = pooled
    finite = torch.isfinite(scores)
    scores, labels = scores[finite], labels[finite]
    return {"best_f1": _best_f1(scores, labels), "auc": _auc(scores, labels)}


def _geo_from_header(sim_file: Path) -> str:
    """Extract the geometry setup path recorded in a .sim/.tra header.

    The header stores the absolute path used on the machine that produced the
    file, e.g. ``/home/arya/.../resource/examples/geomega/special/SPILike.geo.setup``.
    We rewrite it to a portable ``$(MEGALIB)/resource/...`` path so MEGAlib
    resolves it on whichever machine runs this.
    """
    with sim_file.open("r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("Geometry"):
                raw = line.split(None, 1)[1].strip()
                idx = raw.find("resource/examples/geomega")
                if idx != -1:
                    return f"$(MEGALIB)/{raw[idx:]}"
                return raw
            # Geometry is in the first handful of header lines; bail out early.
            if line.startswith(("SE", "EN", "TI")):
                break
    raise RuntimeError(f"No Geometry line found in header of {sim_file}")


def _discover_datasets() -> list[dict[str, str]]:
    """Find every {name}.sim that has a matching {name}.tra in DATA_DIR."""
    datasets = []
    for sim_file in sorted(DATA_DIR.glob("*.sim")):
        tra_file = sim_file.with_suffix(".tra")
        if not tra_file.exists():
            print(f"Skipping {sim_file.name}: no matching {tra_file.name}")
            continue
        datasets.append(
            {
                "name": sim_file.stem,
                "geo_file": _geo_from_header(sim_file),
                "sim_file": str(sim_file),
                "tra_file": str(tra_file),
            }
        )
    return datasets


def _run_one(cfg: omegaconf.dictconfig.DictConfig, ds: dict[str, str]) -> dict:
    """Run the exact run.py pipeline for a single (geo, sim, tra) dataset.

    Returns the pooled confusion counts for each split, keyed by split name.
    """
    wandb.init(project=cfg.project_names[0], name=ds["name"], reinit=True)

    imager = FarFieldImager(
        geometry_file=ds["geo_file"],
        n_phi=360,
        n_theta=180,
        coordinate_system="spheric",
    )
    n_events = imager.backproject_file(ds["tra_file"])
    print(f"Accumulated {n_events} Compton events from {ds['tra_file']}")
    unit_vector = imager.peak_direction_cartesian()

    ref_energy = 511.0
    reader_sim, reader_tra = get_reader(ds["geo_file"], ds["sim_file"])

    datasets = Datasets(cfg, ref_energy, reader_sim, reader_tra, unit_vector)
    data = datasets.compute_event_features(cfg, ref_energy, reader_sim, reader_tra, unit_vector)
    train, eval = datasets.split_dataset(data)

    trainer = Trainer(cfg).fit(train)
    evaluator = Evaluator(trainer)
    results = {}
    for split_name, split_data in (("train", train), ("eval", eval)):
        counts = evaluator.evaluate(split_data, split_name=split_name)
        results[split_name] = {**counts, **_prior_free_scores(evaluator, split_data)}

    wandb.finish()
    return results


@hydra.main(
    config_path="configs",
    config_name="config",
    version_base=None,
)
def main(
    cfg: omegaconf.dictconfig.DictConfig,
) -> None:
    """Run annihilation detection extraction over every .sim/.tra/geometry triple.

    Args:
        cfg (omegaconf.dictconfig.DictConfig): Configuration object.
    """
    datasets = _discover_datasets()
    if not datasets:
        raise RuntimeError(f"No .sim/.tra pairs found in {DATA_DIR.resolve()}")

    print(f"Found {len(datasets)} dataset(s): {', '.join(d['name'] for d in datasets)}")
    rows = []
    for ds in datasets:
        print(f"\n{'=' * 60}\nRunning {ds['name']}\n  geo: {ds['geo_file']}\n  sim: {ds['sim_file']}\n  tra: {ds['tra_file']}\n{'=' * 60}")
        totals = _run_one(cfg, ds)
        for split, counts in totals.items():
            rows.append(
                {
                    "dataset": ds["name"],
                    "split": split,
                    "geo_file": ds["geo_file"],
                    "sim_file": ds["sim_file"],
                    "tra_file": ds["tra_file"],
                    **{k: counts[k] for k in ("tp", "fp", "fn", "tn", "excluded")},
                    **_metrics(counts),
                    # Prior-free, threshold-independent separating power: best_f1 is
                    # the best F1 over all score thresholds (ignores the prior),
                    # auc is the ROC AUC of the per-event likelihood ratio.
                    "best_f1": counts["best_f1"],
                    "auc": counts["auc"],
                }
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = RESULTS_DIR / f"{timestamp}.csv"
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved results for {len(rows)} dataset(s) to {out_path.resolve()}")


if __name__ == "__main__":
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")

    if wandb_api_key is None:
        raise RuntimeError("WANDB_API_KEY not found in .env")

    wandb.login(key=wandb_api_key)

    main()
