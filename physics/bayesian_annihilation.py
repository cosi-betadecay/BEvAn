import torch


class BayesianAnnihiliationModel:
    """
    Bayesian classifier for identifying 511 keV positron-annihilation events.

    This model combines class-conditional likelihood scores
    P(D | β⁺) and P(D | bg) with prior class probabilities
    P(β⁺) and P(bg) using Bayes' theorem to compute posterior
    probabilities P(β⁺ | D) and P(bg | D).

    The likelihood terms (posterior_beta_decay and posterior_bg)
    are expected to represent class-conditional likelihood scores
    derived from physics-based kernels (e.g., energy and ARM).

    The class priors are estimated from event counts and should
    ideally be computed from representative training datasets.
    """

    def __init__(
        self,
        scores_beta_decay: torch.Tensor | float,
        scores_bg: torch.Tensor | float,
        n_beta_decay: int,
        n_bg: int,
    ):
        self.scores_beta_decay = scores_beta_decay
        self.scores_bg = scores_bg

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

    def data_given_beta_decay_probability(
        self, scores_beta_decay: torch.Tensor | float
    ) -> torch.Tensor | float:
        """Return the likelihood of the observed data under the β⁺ hypothesis.

        Returns:
            torch.Tensor | float: Likelihood score P(D | β⁺), either scalar (single event)
            or tensor (batched events).
        """
        return scores_beta_decay

    def data_given_background_probability(self, scores_bg: torch.Tensor | float) -> torch.Tensor | float:
        """Return the likelihood of the observed data under the background hypothesis.

        Returns:
            torch.Tensor | float: Likelihood score P(D | bg), either scalar (single event)
            or tensor (batched events).
        """
        return scores_bg

    def beta_decay_given_data_probability(
        self, scores_beta_decay: torch.Tensor | float, scores_bg: torch.Tensor | float
    ) -> torch.Tensor | float:
        """
        Compute the posterior probability of a β⁺ event given the observed data.

        Applies Bayes' theorem:

            P(β⁺ | D) =
                [P(D | β⁺) P(β⁺)] /
                [P(D | β⁺) P(β⁺) + P(D | bg) P(bg)]

        Returns:
            torch.Tensor | float: Posterior probability P(β⁺ | D), scalar for single-event
            input or tensor for batched input.
        """
        numerator = self.data_given_beta_decay_probability(scores_beta_decay) * self.beta_decay_probability()
        denominator = (
            self.data_given_beta_decay_probability(scores_beta_decay) * self.beta_decay_probability()
            + self.data_given_background_probability(scores_bg) * self.background_probability()
        )
        return numerator / denominator

    def background_given_data_probability(
        self, scores_beta_decay: torch.Tensor | float, scores_bg: torch.Tensor | float
    ) -> torch.Tensor | float:
        """
        Compute the posterior probability of a background event
        given the observed data.

        Returns:
            torch.Tensor | float: Posterior probability P(bg | D), scalar for single-event
            input or tensor for batched input.
        """
        numerator = self.data_given_background_probability(scores_bg) * self.background_probability()
        denominator = (
            self.data_given_beta_decay_probability(scores_beta_decay) * self.beta_decay_probability()
            + self.data_given_background_probability(scores_bg) * self.background_probability()
        )
        return numerator / denominator

    def inference(self) -> torch.Tensor | bool:
        """
        Perform binary classification based on posterior probabilities.

        Returns:
            torch.Tensor | bool: Classification output, bool for single-event input
            or boolean tensor for batched input.
        """
        return self.beta_decay_given_data_probability(
            self.scores_beta_decay, self.scores_bg
        ) >= self.background_given_data_probability(self.scores_beta_decay, self.scores_bg)


# delta E (1) and 180 degrees flying way (2) if 511 how close the compton cone (3)
