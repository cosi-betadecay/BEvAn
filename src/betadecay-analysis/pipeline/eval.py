import os

import torch
from dataset.datasets import BUCKETS, _FEATURES
from modeling.calculate_probablities import (
    confusion_counts,
    log_confusion,
    log_threshold_sweep,
    threshold_sweep,
)
from modeling.matrix_calculations import lookup_density_values, lookup_density_values_1d

# Opt-in diagnostic: set BETADECAY_THRESHOLD_SWEEP=1 to print, per bucket, the
# FP/precision/recall/F1 tradeoff as the decision threshold is raised. Read-only;
# changes no logged metric. Empty -> only the highest bucket(s); "all" -> every bucket.
_SWEEP_ENV = os.environ.get("BETADECAY_THRESHOLD_SWEEP", "")


class Evaluator:
    def __init__(self, trainer):
        self.trainer = trainer

    def evaluate(self, data, split_name: str = "eval"):
        t = self.trainer
        total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "excluded": 0}

        for b in BUCKETS:
            counts = self._evaluate_bucket(data, b, t.models[b], split_name=f"{split_name}/n{b}")
            if counts is None:
                continue
            for k in total:
                total[k] += counts[k]
            # Per-bucket diagnostics (no confusion image to avoid clutter).
            log_confusion(counts, split_name=f"{split_name}/n{b}", log_image=False)

        log_confusion(total, split_name=split_name)

    def _evaluate_bucket(self, data, bucket, model, split_name: str = "eval"):
        bd, bg = data["bdecay"][bucket], data["bg"][bucket]
        n_bd, n_bg = bd["delta_E"].numel(), bg["delta_E"].numel()
        if n_bd + n_bg == 0:
            return None

        feats = {f: torch.cat([bd[f], bg[f]]) for f in _FEATURES}
        ground_truths = torch.cat(
            [torch.ones(n_bd, dtype=torch.bool), torch.zeros(n_bg, dtype=torch.bool)]
        )

        # Keep only events whose bucket-relevant features are finite (bucket 1 may
        # hold invalid-processing events with NaN delta_E).
        used = ["delta_E"] + (["arm"] if bucket >= 2 else []) + (["anni"] if bucket >= 3 else [])
        valid = torch.ones(ground_truths.shape[0], dtype=torch.bool)
        for f in used:
            valid = valid & torch.isfinite(feats[f])
        feats = {f: v[valid] for f, v in feats.items()}
        ground_truths = ground_truths[valid]

        terms = []
        for spec in model["terms"]:
            if spec["kind"] == "1d":
                pb = lookup_density_values_1d(feats[spec["xfeat"]], spec["beta"], spec["x_bins"])
                pg = lookup_density_values_1d(feats[spec["xfeat"]], spec["bg"], spec["x_bins"])
            else:
                pb = lookup_density_values(
                    feats[spec["xfeat"]], feats[spec["yfeat"]], spec["beta"], spec["x_bins"], spec["y_bins"]
                )
                pg = lookup_density_values(
                    feats[spec["xfeat"]], feats[spec["yfeat"]], spec["bg"], spec["x_bins"], spec["y_bins"]
                )
            terms.append((pb, pg))

        counts = confusion_counts(ground_truths, terms, model["n_beta"], model["n_bg"])
        # Count the features-not-finite events as excluded for transparency.
        counts["excluded"] += int((~valid).sum().item())

        # Opt-in, read-only: show the FP-vs-F1 tradeoff for this bucket. Default
        # targets only the top bucket (where ~all the FP live); "all" does every one.
        if _SWEEP_ENV and (_SWEEP_ENV.lower() == "all" or bucket == max(BUCKETS)):
            rows = threshold_sweep(ground_truths, terms, model["n_beta"], model["n_bg"])
            log_threshold_sweep(rows, split_name=split_name)

        return counts
