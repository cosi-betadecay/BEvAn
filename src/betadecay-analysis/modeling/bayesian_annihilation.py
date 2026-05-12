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
        ratio: torch.Tensor,
        n_beta_decay: int,
        n_bg: int,
    ):
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
        return self.ratio / (self.ratio + 1)

    def data_given_background_probability(self) -> torch.Tensor:
        return 1 / (self.ratio + 1)

    def beta_decay_given_data_probability(self) -> torch.Tensor:
        """
        Compute the probability of a β⁺ event given the observed data.

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

    def background_given_data_probability(self) -> torch.Tensor | float:
        """
        Compute the probability of a background event given the observed data.

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

    def inference(self) -> torch.Tensor | bool:
        """
        Perform classification based on probabilities.

        Returns:
            torch.Tensor: Batched classification output.
        """
        return self.beta_decay_given_data_probability() >= 0.7 * self.background_given_data_probability()
