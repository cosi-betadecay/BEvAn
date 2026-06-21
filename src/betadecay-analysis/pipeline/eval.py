import torch
from dataset.datasets import BUCKETS, _FEATURES
from modeling.calculate_probablities import confusion_counts, log_confusion
from modeling.matrix_calculations import lookup_density_values, lookup_density_values_1d


class Evaluator:
    def __init__(self, trainer):
        self.trainer = trainer

    def evaluate(self, data, split_name: str = "eval"):
        t = self.trainer
        total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "excluded": 0}

        for b in BUCKETS:
            counts = self._evaluate_bucket(data, b, t.models[b])
            if counts is None:
                continue
            for k in total:
                total[k] += counts[k]
            # Per-bucket diagnostics (no confusion image to avoid clutter).
            log_confusion(counts, split_name=f"{split_name}/n{b}", log_image=False)

        log_confusion(total, split_name=split_name)

    def _evaluate_bucket(self, data, bucket, model):
        bd, bg = data["bdecay"][bucket], data["bg"][bucket]
        n_bd, n_bg = bd["delta_E"].numel(), bg["delta_E"].numel()
        if n_bd + n_bg == 0:
            return None

        feats = {f: torch.cat([bd[f], bg[f]]) for f in _FEATURES}
        ground_truths = torch.cat(
            [torch.ones(n_bd, dtype=torch.bool), torch.zeros(n_bg, dtype=torch.bool)]
        )

        # Global validity: delta_E must be finite (drops invalid-processing events).
        # Optional factors (e.g. anni in the >=3 bucket) are NOT used to exclude
        # events — a missing factor is dropped per-event below, so every event in a
        # bucket shares one prior/stratum.
        valid = torch.isfinite(feats["delta_E"])
        feats = {f: v[valid] for f, v in feats.items()}
        ground_truths = ground_truths[valid]

        terms = []
        for spec in model["terms"]:
            if spec["kind"] == "1d":
                x = feats[spec["xfeat"]]
                xs = torch.nan_to_num(x)
                pb = lookup_density_values_1d(xs, spec["beta"], spec["x_bins"])
                pg = lookup_density_values_1d(xs, spec["bg"], spec["x_bins"])
                present = torch.isfinite(x)
            else:
                x, y = feats[spec["xfeat"]], feats[spec["yfeat"]]
                xs, ys = torch.nan_to_num(x), torch.nan_to_num(y)
                pb = lookup_density_values(xs, ys, spec["beta"], spec["x_bins"], spec["y_bins"])
                pg = lookup_density_values(xs, ys, spec["bg"], spec["x_bins"], spec["y_bins"])
                present = torch.isfinite(x) & torch.isfinite(y)
            # Drop this factor for events missing its feature(s): ratio -> 1.
            one = pb.new_ones(())
            pb = torch.where(present, pb, one)
            pg = torch.where(present, pg, one)
            terms.append((pb, pg))

        counts = confusion_counts(ground_truths, terms, model["n_beta"], model["n_bg"])
        counts["excluded"] += int((~valid).sum().item())
        return counts
