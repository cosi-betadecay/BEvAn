import torch
from dataset.datasets import _FEATURES, BUCKETS
from modeling.calculate_probablities import confusion_counts, log_confusion
from modeling.matrix_calculations import lookup_density_values, lookup_density_values_1d


class Evaluator:
    def __init__(self, trainer):
        self.trainer = trainer

    def evaluate(self, data, split_name: str = "eval"):
        t = self.trainer
        total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "excluded": 0}
        by_bucket = {}

        for b in BUCKETS:
            counts = self._evaluate_bucket(data, b, t.models[b])
            if counts is None:
                continue
            by_bucket[b] = counts
            for k in total:
                total[k] += counts[k]
            # Per-bucket diagnostics (no confusion image to avoid clutter).
            log_confusion(counts, split_name=f"{split_name}/n{b}", log_image=False)

        log_confusion(total, split_name=split_name)
        # Per-bucket confusion is returned so the population correction can
        # calibrate class-conditional rates (train) and apply them (eval).
        return {"by_bucket": by_bucket, "total": total}

    def _evaluate_bucket(self, data, bucket, model):
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
        return counts
