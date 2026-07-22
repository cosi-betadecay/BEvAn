import torch


class BayesianClassifier:
    """Binary Bayesian classifier over a precomputed log-likelihood ratio.

    Pure Bayes' theorem — no feature or physics computation happens here. Given
    the per-event log-likelihood ratio ``log R = log[P(D | β⁺) / P(D | bg)]``
    (see :func:`modeling.calculate_probabilities.log_R`) and the per-class event
    counts forming the prior, the posterior rule ``P(β⁺ | D) >= P(bg | D)``
    reduces algebraically to a single log-space comparison::

        R * n_beta >= n_bg   <=>   log R >= log(n_bg / n_beta)

    Deciding in log space keeps the ratio from ever being exponentiated, so the
    decision cannot overflow however strongly the terms favour one class.
    """

    def __init__(
        self,
        log_ratio: torch.Tensor,
        n_beta_decay: int,
        n_bg: int,
    ) -> None:
        """Bind the per-event log-likelihood ratio and the class counts for the prior.

        Args:
            log_ratio: Per-event log-likelihood ratio log[P(D | β⁺) / P(D | bg)].
            n_beta_decay: β⁺ event count used to estimate the class prior.
            n_bg: Background event count used to estimate the class prior.
        """
        self.log_ratio = log_ratio
        self.n_beta_decay = n_beta_decay
        self.n_bg = n_bg

    def decision_rule(self) -> torch.Tensor:
        """Apply the Bayesian posterior rule, returning the per-event β⁺/bg prediction.

        Predicts β⁺ when ``log R >= log(n_bg / n_beta)`` — exactly the posterior
        rule ``P(β⁺ | D) >= P(bg | D)`` at the count prior. The threshold is
        computed with ``torch.log`` so a zero class count degrades to ±inf
        (always/never predict β⁺) instead of raising; a NaN log-ratio compares
        False, predicting background (those events are excluded downstream).

        Returns:
            torch.Tensor: Batched boolean classification output.
        """
        counts = torch.tensor([float(self.n_bg), float(self.n_beta_decay)])
        log_prior_ratio = torch.log(counts[0]) - torch.log(counts[1])
        return self.log_ratio >= log_prior_ratio
