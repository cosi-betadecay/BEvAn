import math

import torch


class BayesianClassifier:
    """Binary Bayesian classifier over a precomputed likelihood ratio.

    Pure Bayes' theorem — no feature or physics computation happens here. Given
    the per-event likelihood ratio ``R = P(D | β⁺) / P(D | bg)`` and the per-class
    event counts, it forms the class priors ``P(β⁺)`` / ``P(bg)``, the posteriors
    ``P(β⁺ | D)`` / ``P(bg | D)``, and the decision rule. The ratio is built
    upstream (see :func:`modeling.calculate_probabilities.R`) and the priors come
    from the counts.
    """

    def __init__(
        self,
        ratio: torch.Tensor,
        n_beta_decay: int,
        n_bg: int,
    ) -> None:
        """Bind the per-event likelihood ratio and the class counts for the prior.

        Args:
            ratio: Per-event likelihood ratio P(D | β⁺) / P(D | bg).
            n_beta_decay: β⁺ event count used to estimate the class prior.
            n_bg: Background event count used to estimate the class prior.
        """
        self.ratio = ratio
        # We should compute these two with other datasets
        self.n_beta_decay = n_beta_decay
        self.n_bg = n_bg

    def beta_decay_probability(self) -> float:
        """Compute the prior probability of a β⁺ event.

        Returns:
            float: Prior probability P(β⁺) estimated from the relative frequency of β⁺ events.
        """
        return self.n_beta_decay / (self.n_beta_decay + self.n_bg)

    def background_probability(self) -> float:
        """Compute the prior probability of a background event.

        Returns:
            float: Prior probability P(bg) estimated from the relative frequency of background events.
        """
        return self.n_bg / (self.n_beta_decay + self.n_bg)

    def data_given_beta_decay_probability(self) -> torch.Tensor:
        """Normalized likelihood P(D | β⁺) from the per-event ratio, ``R / (R + 1)``."""
        return self.ratio / (self.ratio + 1)

    def data_given_background_probability(self) -> torch.Tensor:
        """Normalized likelihood P(D | bg) from the per-event ratio, ``1 / (R + 1)``."""
        return 1 / (self.ratio + 1)

    def beta_decay_given_data_probability(self) -> torch.Tensor:
        """Compute the probability of a β⁺ event given the observed data.

        Applies Bayes' theorem:

            P(β⁺ | D) =
                [P(D | β⁺) P(β⁺)] /
                [P(D | β⁺) P(β⁺) + P(D | bg) P(bg)]

        Returns:
            torch.Tensor: Probability tensor P(β⁺ | D)
        """
        numerator = self.data_given_beta_decay_probability() * self.beta_decay_probability()
        denominator = (
            self.data_given_beta_decay_probability() * self.beta_decay_probability()
            + self.data_given_background_probability() * self.background_probability()
        )
        return numerator / denominator

    def background_given_data_probability(self) -> torch.Tensor:
        """Compute the probability of a background event given the observed data.

        Applies Bayes' theorem:

            P(bg | D) =
                [P(D | bg) P(bg)] /
                [P(D | β⁺) P(β⁺) + P(D | bg) P(bg)]

        Returns:
            torch.Tensor: Probability tensor P(bg | D)
        """
        numerator = self.data_given_background_probability() * self.background_probability()
        denominator = (
            self.data_given_beta_decay_probability() * self.beta_decay_probability()
            + self.data_given_background_probability() * self.background_probability()
        )
        return numerator / denominator

    def decision_rule(self, log_threshold: float | None = None) -> torch.Tensor:
        """Apply the decision rule, returning the per-event β⁺/bg prediction.

        With ``log_threshold=None`` the decision is the Bayesian posterior rule
        (predict β⁺ when ``P(β⁺ | D) >= P(bg | D)``), i.e. the operating point set
        by the raw class-count prior. A ``log_threshold`` instead applies a
        calibrated cut on the log-likelihood ratio: predict β⁺ when
        ``log R >= log_threshold`` (equivalently ``R >= exp(log_threshold)``).
        This lets a leak-free F1 calibration shift the operating point off the
        count-prior default without touching the densities.

        Args:
            log_threshold: Calibrated LLR cut, or None for the count-prior posterior rule.

        Returns:
            torch.Tensor: Batched boolean classification output.
        """
        if log_threshold is None:
            return self.beta_decay_given_data_probability() >= self.background_given_data_probability()
        return self.ratio >= math.exp(log_threshold)
