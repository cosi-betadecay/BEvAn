"""
Unit tests for the Zoglauer-style density factorization implemented in
physics/annihilation_detection_utils.py:build_density_matrices.

The function derives shared bin edges from the combined (signal + background)
tensor, builds separate 2D probability matrices for beta-decay and
background, factors ΔE out of the two joints via conditional_from_joint, and
looks up per-event likelihood values. It returns six per-event likelihood
tensors:

    P_β(ΔE), P_bg(ΔE),
    P_β(angle | ΔE), P_bg(angle | ΔE),
    P_β(ARM   | ΔE), P_bg(ARM   | ΔE)

All tests run against real feature data extracted from Activation.sim via the
session-scoped ``likelihood_tensors`` fixture defined in ``conftest.py``,
which is parametrized over ``BIN_SIZES = [20, 200, 2000]``. Every test
therefore runs once per resolution.

Requirements:
    - MEGALIB env variable set and data/Activation.sim present (handled by
      the real_tensors / likelihood_tensors fixtures in conftest.py).

Covered tests:
    1. Returns six tensors whose length equals the number of gen events.
    2. All returned likelihood values are non-negative.
    3. NaN entries in the gen tensors produce 0.0 in every output tensor.
    4. True beta-decay events yield higher total likelihoods in the β factors
       than in the bg factors — the core physics contract.
    5. All six output tensors share the same length.
"""

import torch

from physics.annihilation_detection_utils import build_density_matrices

# ===========================================================================
# Helpers
# ===========================================================================

_TENSOR_KEYS = (
    "p_beta_deltaE",
    "p_bg_deltaE",
    "p_beta_angle",
    "p_bg_angle",
    "p_beta_arm",
    "p_bg_arm",
)


def _as_tuple(l):
    return tuple(l[k] for k in _TENSOR_KEYS)


# ===========================================================================
# Tests
# ===========================================================================


def test_returns_six_tensors_of_correct_length(likelihood_tensors):
    """
    build_density_matrices must return exactly six tensors, each with length
    equal to the number of gen events.

    The six outputs are the per-event likelihood values used to build the
    Zoglauer-style likelihood ratio R with ΔE counted once:
        P_β(ΔE), P_bg(ΔE)
        P_β(angle | ΔE), P_bg(angle | ΔE)
        P_β(ARM   | ΔE), P_bg(ARM   | ΔE)

    Every output tensor must have length == n_gen so the downstream Bayesian
    classifier can pair each likelihood value with the correct event.
    """
    tensors = _as_tuple(likelihood_tensors)
    n_gen = likelihood_tensors["n_gen"]

    assert len(tensors) == 6, f"Expected 6 output tensors, got {len(tensors)}"
    for name, tensor in zip(_TENSOR_KEYS, tensors):
        assert tensor.numel() == n_gen, (
            f"{name} has {tensor.numel()} elements; expected {n_gen}"
        )


def test_all_returned_values_are_non_negative(likelihood_tensors):
    """
    All six returned tensors must contain only non-negative values.

    They represent probability densities looked up from normalised histograms.
    Negative probabilities are physically meaningless and would corrupt the
    log-likelihood ratio computed downstream.
    """
    for name in _TENSOR_KEYS:
        tensor = likelihood_tensors[name]
        n_neg = int((tensor < 0).sum().item())
        assert torch.all(tensor >= 0.0), f"{name} contains {n_neg} negative values"


def test_nan_in_gen_tensors_returns_zero(real_tensors):
    """
    NaN entries in any gen tensor must produce 0.0 in the corresponding
    position of every output tensor.

    lookup_density_values and lookup_density_values_1d mark NaN coordinates as
    invalid and fill their output slots with 0.0. This test injects NaN at
    known indices into copies of the real gen tensors and verifies
    build_density_matrices propagates that behaviour in both feature pairs
    (ΔE / angle and ΔE / ARM) and in the ΔE marginals.

    Uses build_density_matrices directly (at its production default
    resolution) rather than the parametrized fixture, because the NaN
    contract is bin-size-independent.
    """
    nan_indices = [0, 10]

    gen_dE = real_tensors["gen_tensor_delta_E"].clone()
    gen_angle = real_tensors["gen_tensor_annihilation_angle"].clone()
    gen_arm = real_tensors["gen_tensor_arm"].clone()

    for idx in nan_indices:
        gen_dE[idx] = float("nan")
        gen_angle[idx] = float("nan")
        gen_arm[idx] = float("nan")

    outputs = build_density_matrices(
        real_tensors["bdecay_tensor_delta_E"],
        real_tensors["bdecay_tensor_annihilation_angle"],
        real_tensors["bdecay_tensor_arm"],
        real_tensors["bg_tensor_delta_E"],
        real_tensors["bg_tensor_annihilation_angle"],
        real_tensors["bg_tensor_arm"],
        gen_dE,
        gen_angle,
        gen_arm,
        real_tensors["combined_tensor_delta_E"],
        real_tensors["combined_tensor_annihilation_angle"],
        real_tensors["combined_tensor_arm"],
    )

    for name, tensor in zip(_TENSOR_KEYS, outputs):
        for idx in nan_indices:
            assert tensor[idx].item() == 0.0, (
                f"{name}[{idx}] = {tensor[idx].item():.6f}; expected 0.0 for NaN gen input"
            )


def test_true_bdecay_events_have_higher_beta_likelihood(real_tensors, likelihood_tensors):
    """
    For events the simulation labels as beta-decay (ground_truths == True),
    the total β-class likelihood must exceed the total bg-class likelihood
    across all three factored sub-spaces: ΔE marginal, angle|ΔE conditional,
    and ARM|ΔE conditional.

    This is the core physics contract of the density factorization — the
    class-conditional densities must favour the correct class on true events.
    It holds across bin sizes (the fixture sweeps 20/200/2000), so this
    assertion is checked at every resolution.
    """
    gt = real_tensors["ground_truths"]
    t = likelihood_tensors

    pairs = [
        ("deltaE", t["p_beta_deltaE"], t["p_bg_deltaE"]),
        ("angle|ΔE", t["p_beta_angle"], t["p_bg_angle"]),
        ("ARM|ΔE", t["p_beta_arm"], t["p_bg_arm"]),
    ]
    for label, p_beta, p_bg in pairs:
        beta_sum = p_beta[gt].sum()
        bg_sum = p_bg[gt].sum()
        assert beta_sum > bg_sum, (
            f"[{label}] True bdecay events: expected β likelihood sum "
            f"({beta_sum:.6f}) > bg likelihood sum ({bg_sum:.6f}) "
            f"at n_bins={t['n_bins']}"
        )


def test_all_six_outputs_have_identical_length(likelihood_tensors):
    """
    All six output tensors must have exactly the same number of elements.

    The downstream likelihood ratio R() operates element-wise across every
    factor simultaneously. Any length mismatch would silently broadcast or
    truncate, producing wrong classification results. Acts as a shape-contract
    safety net against future refactoring.
    """
    lengths = [likelihood_tensors[k].numel() for k in _TENSOR_KEYS]
    assert len(set(lengths)) == 1, (
        f"Output tensors have inconsistent lengths: {lengths}. "
        "All six must be equal for element-wise R() computation."
    )
