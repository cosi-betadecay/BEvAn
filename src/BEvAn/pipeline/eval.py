import torch

from modeling.bayesian_classifier import BayesianClassifier
from modeling.calculate_probabilities import confusion_counts, log_confusion, log_R
from modeling.matrix_calculations import lookup_density_values, lookup_density_values_1d
from modeling.metrics import best_f1_threshold, roc_auc
from pipeline.datasets import BUCKETS, CLASSES, FEATURES
from pipeline.train import Trainer


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

    def evaluate(
        self, data: dict[str, dict[int, dict[str, torch.Tensor]]], split_name: str = "eval"
    ) -> dict[str, int]:
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

        counts = confusion_counts(ground_truths, terms, model["n_beta"], model["n_bg"])
        # Count the features-not-finite events as excluded for transparency.
        counts["excluded"] += int((~valid).sum().item())
        return counts

    def classify_events(
        self,
        data: dict[str, dict[int, dict[str, torch.Tensor]]],
        event_keys: dict[str, dict[int, list[tuple[int, int]]]],
    ) -> dict[str, set[tuple[int, int]]]:
        """Tag every scorable event with the deployed decision, keyed by event identity.

        Runs the same per-bucket likelihood ratio and decision rule the confusion
        counts use, but returns *which* events were called β⁺/background — as their
        chunk-scoped ``(chunk, id)`` keys — instead of tallying them. Events whose
        bucket features are not finite get no tag (the same events the confusion
        matrix excludes). Alignment with ``event_keys`` is positional per bucket
        (β⁺ decay events before background, mirroring :meth:`prepare_terms`), so both
        arguments must come from one :meth:`~pipeline.datasets.Datasets.compute_event_features_keyed`
        call.

        Args:
            data: Nested per-class, per-bucket feature dict.
            event_keys: The parallel per-class, per-bucket event-key lists.

        Returns:
            ``{"bdecay": keys, "bg": keys}`` — the tagged event-key sets.
        """
        tagged: dict[str, set[tuple[int, int]]] = {cls: set() for cls in CLASSES}
        for b in BUCKETS:
            model = self.trainer.models[b]
            prepared = self.prepare_terms(data, b, model)
            if prepared is None:
                continue
            terms, _, valid = prepared

            bucket_keys = event_keys["bdecay"][b] + event_keys["bg"][b]
            assert len(bucket_keys) == valid.shape[0], "event_keys misaligned with data"

            log_r = log_R(terms) if terms else torch.zeros(int(valid.sum().item()))
            predictions = BayesianClassifier(log_r, model["n_beta"], model["n_bg"]).decision_rule()
            scored_keys = [k for k, v in zip(bucket_keys, valid.tolist(), strict=True) if v]
            for key, is_beta in zip(scored_keys, predictions.tolist(), strict=True):
                tagged["bdecay" if is_beta else "bg"].add(key)
        return tagged

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

        log_r = log_R(terms) if terms else torch.zeros(ground_truths.shape[0])
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


def prior_free_scores(pooled: tuple[torch.Tensor, torch.Tensor] | None) -> dict[str, float]:
    """Prior-free best-F1 and AUC over the pooled per-event likelihood ratio.

    Threshold- and prior-independent separating power: ``best_f1`` is the best F1
    over all score cuts and ``auc`` is the ROC AUC of the pooled log-ratio.

    Args:
        pooled: A split's ``(scores, labels)`` from
            :meth:`Evaluator.pooled_log_ratio`, or None if no events pooled.

    Returns:
        Dict with ``best_f1`` and ``auc`` (both NaN if no events pool).
    """
    if pooled is None:
        return {"best_f1": float("nan"), "auc": float("nan")}
    scores, labels = pooled
    finite = torch.isfinite(scores)
    scores, labels = scores[finite], labels[finite]
    return {"best_f1": best_f1_threshold(scores, labels)[0], "auc": roc_auc(scores, labels)}
