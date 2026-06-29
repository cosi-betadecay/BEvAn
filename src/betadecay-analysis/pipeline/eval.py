import torch

from dataset.datasets import BUCKETS, FEATURES
from modeling.calculate_probabilities import confusion_counts, log_confusion
from modeling.matrix_calculations import lookup_density_values, lookup_density_values_1d
from modeling.metrics import best_f1_threshold, roc_auc
from pipeline.train import Trainer
from utils.wandb_logging import log_score_curves


class Evaluator:
    """Evaluate a trained per-bucket model on a feature dataset.

    For each hit-multiplicity bucket it builds the per-event likelihood-ratio
    terms, applies the Bayesian decision rule to tally confusion counts, and
    (separately) pools the prior-free log-likelihood ratio across buckets for
    threshold-independent metrics such as AUC and best-F1.
    """

    def __init__(self, trainer: Trainer) -> None:
        """Bind the evaluator to a fitted trainer.

        Args:
            trainer: A fitted :class:`~pipeline.train.Trainer` providing the
                per-bucket models used for scoring.
        """
        self.trainer = trainer

    def evaluate(self, data: dict[str, dict[int, dict[str, torch.Tensor]]], split_name: str = "eval") -> dict:
        """Evaluate every bucket and return the pooled confusion counts.

        Per-bucket counts are summed into a dataset total; both the per-bucket
        and total counts are logged via :func:`log_confusion`.

        Args:
            data: Nested per-class, per-bucket feature dict for this split.
            split_name: Metric namespace prefix for logging.

        Returns:
            The summed ``tp``/``fp``/``fn``/``tn``/``excluded`` counts.
        """
        t = self.trainer
        total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "excluded": 0}

        for b in BUCKETS:
            counts = self.evaluate_bucket(data, b, t.models[b])
            if counts is None:
                continue
            for k in total:
                total[k] += counts[k]
            # Per-bucket diagnostics (no confusion image to avoid clutter).
            log_confusion(counts, split_name=f"{split_name}/n{b}", log_image=False)

        log_confusion(total, split_name=split_name)
        # Same wandb treatment as the confusion matrix, for this split's pooled
        # score: the threshold-free ROC and precision-recall curves (no-op when no
        # run is active).
        log_score_curves(self, data, split_name)

        return total

    def prepare_terms(
        self, data: dict[str, dict[int, dict[str, torch.Tensor]]], bucket: int, model: dict
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor] | None:
        """Shared per-bucket setup: features, finite mask, and per-term densities.

        Concatenates the β and bg features for ``bucket``, drops events whose
        bucket-relevant features are not finite, and looks up each model term's
        ``(p_beta, p_bg)`` densities.

        Args:
            data: Nested per-class, per-bucket feature dict.
            bucket: Hit-multiplicity bucket to prepare.
            model: The bucket's fitted model (``terms`` plus prior counts).

        Returns:
            ``(terms, ground_truths, valid)`` where ``valid`` is the pre-filter
            finite mask, or None for an empty bucket.
        """
        bd, bg = data["bdecay"][bucket], data["bg"][bucket]
        n_bd, n_bg = bd["delta_E"].numel(), bg["delta_E"].numel()
        if n_bd + n_bg == 0:
            return None

        feats = {f: torch.cat([bd[f], bg[f]]) for f in FEATURES}
        ground_truths = torch.cat([torch.ones(n_bd, dtype=torch.bool), torch.zeros(n_bg, dtype=torch.bool)])

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
        return terms, ground_truths, valid

    def evaluate_bucket(
        self, data: dict[str, dict[int, dict[str, torch.Tensor]]], bucket: int, model: dict
    ) -> dict | None:
        """Confusion counts for one bucket, non-finite-feature events excluded.

        Args:
            data: Nested per-class, per-bucket feature dict.
            bucket: Hit-multiplicity bucket to evaluate.
            model: The bucket's fitted model (``terms`` plus prior counts).

        Returns:
            The bucket's ``tp``/``fp``/``fn``/``tn``/``excluded`` counts, or None
            for an empty bucket.
        """
        prepared = self.prepare_terms(data, bucket, model)
        if prepared is None:
            return None
        terms, ground_truths, valid = prepared

        counts = confusion_counts(
            ground_truths, terms, model["n_beta"], model["n_bg"], log_threshold=model.get("log_threshold")
        )
        # Count the features-not-finite events as excluded for transparency.
        counts["excluded"] += int((~valid).sum().item())
        return counts

    def bucket_log_ratio(
        self, data: dict[str, dict[int, dict[str, torch.Tensor]]], bucket: int, model: dict
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Prior-free per-event log-likelihood ratio and truths for one bucket.

        Args:
            data: Nested per-class, per-bucket feature dict.
            bucket: Hit-multiplicity bucket to score.
            model: The bucket's fitted model (``terms`` plus prior counts).

        Returns:
            ``(log_r, ground_truths)``, or None for an empty bucket.
        """
        prepared = self.prepare_terms(data, bucket, model)
        if prepared is None:
            return None
        terms, ground_truths = prepared[:2]

        eps = 1e-8
        log_r = torch.zeros(ground_truths.shape[0])
        for pb, pg in terms:
            log_r = log_r + torch.log(pb + eps) - torch.log(pg + eps)
        return log_r, ground_truths

    def pooled_log_ratio(
        self, data: dict[str, dict[int, dict[str, torch.Tensor]]]
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Concatenate the prior-free log-ratios and truths across all buckets.

        Returns ``(log_r, ground_truths)`` over the whole dataset, or None if
        every bucket is empty.
        """
        log_rs, gts = [], []
        for b in BUCKETS:
            out = self.bucket_log_ratio(data, b, self.trainer.models[b])
            if out is None:
                continue
            log_rs.append(out[0])
            gts.append(out[1])
        if not log_rs:
            return None
        return torch.cat(log_rs), torch.cat(gts)

    def held_out_log_likelihood(self, data: dict[str, dict[int, dict[str, torch.Tensor]]]) -> float:
        """Mean per-event ``log P(D | true class)`` over all buckets.

        Each finite-feature event contributes its own class's density (signal ->
        β term, background -> bg term) summed across that bucket's terms; the
        sum is averaged over events. Prior-free, so it scores density fit alone —
        the cross-validation reward for the bin-size rule. Over-fine bins drop
        this score as held-out events land in noise-dominated cells, so larger is
        better. ``O(N)`` time and space; NaN if no event scores.

        Args:
            data: Nested per-class, per-bucket feature dict for one split.

        Returns:
            The mean per-event held-out log-likelihood.
        """
        eps = 1e-8
        total_ll, total_n = 0.0, 0
        for b in BUCKETS:
            prepared = self.prepare_terms(data, b, self.trainer.models[b])
            if prepared is None:
                continue
            terms, ground_truths = prepared[:2]
            if not terms or ground_truths.numel() == 0:
                continue
            ll = torch.zeros(ground_truths.shape[0])
            for pb, pg in terms:
                ll = ll + torch.where(ground_truths, torch.log(pb + eps), torch.log(pg + eps))
            total_ll += float(ll.sum().item())
            total_n += int(ground_truths.numel())
        return total_ll / total_n if total_n else float("nan")

    def held_out_auc(self, data: dict[str, dict[int, dict[str, torch.Tensor]]]) -> float:
        """Prior-free ROC AUC of the pooled per-event log-ratio over all buckets.

        Unlike :meth:`held_out_log_likelihood` (which rewards calibrated density
        fit and can favour coarse, well-populated bins that separate the classes
        poorly), this scores *separating power* directly — the same quantity F1
        tracks — so it is the cross-validation reward aligned with the detection
        objective. Threshold- and prior-independent. NaN if a split has only one
        class or no events.

        Args:
            data: Nested per-class, per-bucket feature dict for one split.

        Returns:
            The pooled ROC AUC, or NaN if it is undefined for this split.
        """
        pooled = self.pooled_log_ratio(data)
        if pooled is None:
            return float("nan")
        scores, labels = pooled
        finite = torch.isfinite(scores)
        scores, labels = scores[finite], labels[finite]
        if scores.numel() == 0:
            return float("nan")
        return roc_auc(scores, labels)


def prior_free_scores(evaluator: "Evaluator", data: dict[str, dict[int, dict[str, torch.Tensor]]]) -> dict:
    """Prior-free best-F1 and AUC over the pooled per-event likelihood ratio.

    Threshold- and prior-independent separating power: ``best_f1`` is the best F1
    over all score cuts and ``auc`` is the ROC AUC of the pooled log-ratio.

    Args:
        evaluator: Trained evaluator providing the pooled log-ratio.
        data: Nested per-class, per-bucket feature dict for one split.

    Returns:
        Dict with ``best_f1`` and ``auc`` (both NaN if no events pool).
    """
    pooled = evaluator.pooled_log_ratio(data)
    if pooled is None:
        return {"best_f1": float("nan"), "auc": float("nan")}
    scores, labels = pooled
    finite = torch.isfinite(scores)
    scores, labels = scores[finite], labels[finite]
    return {"best_f1": best_f1_threshold(scores, labels)[0], "auc": roc_auc(scores, labels)}
