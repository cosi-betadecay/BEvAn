import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv

import wandb
from dataset.datasets import Datasets
from physics.compton_cone_reconstruction import FarFieldImager
from pipeline.eval import Evaluator
from pipeline.model_selection import select_hyperparams
from pipeline.train import Trainer
from utils.reader_extraction import get_reader

PROJECT = "cosi-betadecay"
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")


def metrics(counts: dict) -> dict:
    """Derive precision/recall/FPR/F1 from raw confusion counts.

    Matches the definitions in modeling.calculate_probablities.log_confusion.

    Args:
        counts (dict): Confusion counts with tp/fp/fn/tn keys.

    Returns:
        dict: Keys precision, recall, fpr, and f1_score.
    """
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "fpr": fpr, "f1_score": f1}


def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Prior-free ROC AUC via the tie-aware Mann-Whitney rank statistic.

    Invariant to any monotonic transform of ``scores`` (so the log-ratio gives
    the same answer as the ratio).

    Args:
        scores (torch.Tensor): Per-event separating scores.
        labels (torch.Tensor): Per-event boolean truth labels (β⁺ = True).

    Returns:
        float: The ROC AUC, or NaN if either class is empty.
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


def best_f1(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Best F1 achievable by sweeping a threshold over the prior-free score.

    Cuts are only allowed between distinct score values (tie-aware).

    Args:
        scores (torch.Tensor): Per-event separating scores.
        labels (torch.Tensor): Per-event boolean truth labels (β⁺ = True).

    Returns:
        float: The best F1 over all thresholds, or NaN if there are no positives.
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


def prior_free_scores(evaluator: Evaluator, data: dict[str, dict[int, dict[str, torch.Tensor]]]) -> dict:
    """Prior-free best-F1 and AUC over the pooled per-event likelihood ratio.

    Args:
        evaluator (Evaluator): Trained evaluator providing the pooled log-ratio.
        data (dict): Nested per-class, per-bucket feature dict for one split.

    Returns:
        dict: Keys best_f1 and auc (both NaN if no events pool).
    """
    pooled = evaluator.pooled_log_ratio(data)
    if pooled is None:
        return {"best_f1": float("nan"), "auc": float("nan")}
    scores, labels = pooled
    finite = torch.isfinite(scores)
    scores, labels = scores[finite], labels[finite]
    return {"best_f1": best_f1(scores, labels), "auc": auc(scores, labels)}


def geo_from_header(sim_file: Path) -> str:
    """Extract the geometry setup path recorded in a .sim/.tra header.

    The header stores the absolute path used on the machine that produced the
    file, e.g. ``/home/arya/.../resource/examples/geomega/special/SPILike.geo.setup``.
    We rewrite it to a portable ``$(MEGALIB)/resource/...`` path so MEGAlib
    resolves it on whichever machine runs this.

    Args:
        sim_file (Path): Path to the .sim or .tra file to read the header from.

    Returns:
        str: The portable geometry setup path.

    Raises:
        RuntimeError: If no Geometry line is found in the header.
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


def discover_datasets() -> list[dict[str, str]]:
    """Find every {name}.sim that has a matching {name}.tra in DATA_DIR.

    Returns:
        list[dict[str, str]]: One dict per dataset with name, geo_file,
            sim_file, and tra_file keys.
    """
    datasets = []
    for sim_file in sorted(DATA_DIR.glob("*.sim")):
        tra_file = sim_file.with_suffix(".tra")
        if not tra_file.exists():
            print(f"Skipping {sim_file.name}: no matching {tra_file.name}")
            continue
        datasets.append(
            {
                "name": sim_file.stem,
                "geo_file": geo_from_header(sim_file),
                "sim_file": str(sim_file),
                "tra_file": str(tra_file),
            }
        )
    return datasets


def run_one(ds: dict[str, str], use_wandb: bool) -> dict:
    """Run the exact run.py pipeline for a single (geo, sim, tra) dataset.

    Args:
        ds (dict[str, str]): Dataset paths with name, geo_file, sim_file, tra_file.
        use_wandb (bool): Whether to use Weights & Biases for logging.

    Returns:
        dict: Pooled confusion counts plus prior-free metrics, keyed by split name.
    """
    if use_wandb:
        wandb.init(project=PROJECT, name=ds["name"], reinit=True)

    imager = FarFieldImager(
        geometry_file=ds["geo_file"],
        n_phi=360,
        n_theta=180,
        coordinate_system="spheric",
    )
    n_events = imager.backproject_file(ds["tra_file"])
    print(f"Accumulated {n_events} Compton events from {ds['tra_file']}")
    unit_vector = imager.peak_direction_cartesian()

    reader_sim = get_reader(ds["geo_file"], ds["sim_file"])

    datasets = Datasets(reader_sim, unit_vector)
    data = datasets.compute_event_features()
    train, eval_data = datasets.split_dataset(data)

    # Select bins + smoothing per dataset (geometries differ wildly in event
    # counts), leak-free on the training split, then refit on the full train set.
    best_config, cv_scores = select_hyperparams(train)
    print(f"  selected {best_config} (CV log-likelihood {cv_scores[(best_config['rho_floor'], best_config['joint_smoothing'], best_config['marginal_smoothing'])]:.4f})")
    if use_wandb:
        wandb.log(best_config)

    trainer = Trainer(**best_config).fit(train)
    evaluator = Evaluator(trainer)
    results = {}
    for split_name, split_data in (("train", train), ("eval", eval_data)):
        counts = evaluator.evaluate(split_data, split_name=split_name)
        results[split_name] = {**counts, **prior_free_scores(evaluator, split_data)}

    if use_wandb:
        wandb.finish()
    return results


def main(use_wandb: bool) -> None:
    """Run annihilation detection extraction over every .sim/.tra/geometry triple.

    Args:
        use_wandb (bool): Whether to use Weights & Biases for logging.

    Raises:
        FileNotFoundError: If the data directory does not exist.
        RuntimeError: If the data directory contains no ``.sim``/``.tra`` pairs.
    """
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR.resolve()}")
    datasets = discover_datasets()
    if not datasets:
        raise RuntimeError(f"No .sim/.tra pairs found in {DATA_DIR.resolve()}")

    print(f"Found {len(datasets)} dataset(s): {', '.join(d['name'] for d in datasets)}")
    rows = []
    for ds in datasets:
        print(
            f"\n{'=' * 60}\nRunning {ds['name']}\n  geo: {ds['geo_file']}\n  sim: {ds['sim_file']}\n  tra: {ds['tra_file']}\n{'=' * 60}"
        )
        totals = run_one(ds, use_wandb)
        for split, counts in totals.items():
            rows.append(
                {
                    "dataset": ds["name"],
                    "split": split,
                    "geo_file": ds["geo_file"],
                    "sim_file": ds["sim_file"],
                    "tra_file": ds["tra_file"],
                    **{k: counts[k] for k in ("tp", "fp", "fn", "tn", "excluded")},
                    **metrics(counts),
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
    parser = argparse.ArgumentParser(description="Run β⁺ detection over every .sim/.tra pair found in data/.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()

    if args.wandb:
        load_dotenv()
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key is None:
            raise RuntimeError("WANDB_API_KEY not found in .env")
        wandb.login(key=wandb_api_key)

    main(args.wandb)
