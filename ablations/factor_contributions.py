from harness import Evaluator, best_f1_threshold, factor_term_scores, roc_auc


def run(trainer: object, data: dict) -> dict[str, dict[str, float]]:
    """Per-factor standalone F1 and AUC from each factor's own pipeline term.

    Args:
        trainer: The fitted champion trainer (its per-bucket density terms are
            scored individually, double-count retained).
        data: The split to evaluate on (the eval split for the paper figure).

    Returns:
        ``{factor: {"f1": ..., "auc": ...}}`` over the factors whose term is present,
        each computed on that term's active event population.
    """
    evaluator = Evaluator(trainer)
    scores = factor_term_scores(evaluator, data)
    return {
        factor: {"auc": roc_auc(s, labels), "f1": best_f1_threshold(s, labels)[0]}
        for factor, (s, labels) in scores.items()
    }
