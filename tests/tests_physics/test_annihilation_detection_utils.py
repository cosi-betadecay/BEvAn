"""
Unit tests for the build_density_matrices function in
physics/annihilation_detection_utils.py.

build_density_matrices is the density-estimation orchestrator: it derives
shared bin edges from the combined (signal + background) tensor, builds
separate 2D probability matrices for beta-decay and background, then looks
up per-event likelihood values for every event in the gen tensors.

All tests run against real feature data extracted from Activation.sim via the
session-scoped real_tensors fixture in conftest.py.  No synthetic data or
mocks are used.

Requirements:
    - MEGALIB env variable set and data/Activation.sim present (handled by
      the real_tensors fixture in conftest.py).

Covered tests (5):
    1. Returns four tensors whose length equals the length of the gen tensors.
    2. All returned probability values are non-negative.
    3. NaN entries injected into gen tensors produce 0.0 in every output tensor.
    4. True beta-decay events (from ground_truths) yield higher total beta
       likelihoods than background likelihoods — the core physics contract.
    5. All four output tensors share the same length (shape consistency check).
"""

import torch

from physics.annihilation_detection_utils import build_density_matrices


# ===========================================================================
# Helpers
# ===========================================================================


def _run_build(t, gen_dE=None, gen_angle=None, gen_arm=None):
    """
    Call build_density_matrices with the given real_tensors dict, optionally
    overriding the gen tensors (e.g. to inject NaN values).
    """
    return build_density_matrices(
        t["bdecay_tensor_delta_E"],
        t["bdecay_tensor_annihilation_angle"],
        t["bdecay_tensor_arm"],
        t["bg_tensor_delta_E"],
        t["bg_tensor_annihilation_angle"],
        t["bg_tensor_arm"],
        gen_dE if gen_dE is not None else t["gen_tensor_delta_E"],
        gen_angle if gen_angle is not None else t["gen_tensor_annihilation_angle"],
        gen_arm if gen_arm is not None else t["gen_tensor_arm"],
        t["combined_tensor_delta_E"],
        t["combined_tensor_annihilation_angle"],
        t["combined_tensor_arm"],
    )


# ===========================================================================
# Tests
# ===========================================================================


def test_returns_four_tensors_of_correct_length(real_tensors):
    """
    build_density_matrices must return exactly four tensors, each with the
    same number of elements as the gen tensors.

    The four outputs are the per-event likelihood values for:
        n_beta_deltaE_annihilation_angle  — P(event | beta-decay, angle feature)
        n_bg_deltaE_annihilation_angle    — P(event | background, angle feature)
        n_beta_deltaE_arm                 — P(event | beta-decay, ARM feature)
        n_bg_deltaE_arm                   — P(event | background, ARM feature)

    Every output tensor must have length == n_gen so that the downstream
    Bayesian classifier can pair each likelihood value with the correct event.
    """
    result = _run_build(real_tensors)

    assert len(result) == 4, f"Expected 4 output tensors, got {len(result)}"
    for i, tensor in enumerate(result):
        assert tensor.numel() == real_tensors["n_gen"], (
            f"Output tensor {i} has {tensor.numel()} elements; "
            f"expected {real_tensors['n_gen']}"
        )


def test_all_returned_values_are_non_negative(real_tensors):
    """
    All four returned tensors must contain only non-negative values.

    They represent probability densities looked up from normalised histograms.
    Negative probabilities are physically meaningless and would corrupt the
    log-likelihood ratio R() computed downstream.

    This is verified on real Activation.sim data, where the feature
    distributions span the actual dynamic range of the detector.
    """
    n_beta_angle, n_bg_angle, n_beta_arm, n_bg_arm = _run_build(real_tensors)

    for name, tensor in [
        ("n_beta_angle", n_beta_angle),
        ("n_bg_angle", n_bg_angle),
        ("n_beta_arm", n_beta_arm),
        ("n_bg_arm", n_bg_arm),
    ]:
        assert torch.all(tensor >= 0.0), (
            f"{name} contains {(tensor < 0).sum().item()} negative values"
        )


def test_nan_in_gen_tensors_returns_zero(real_tensors):
    """
    NaN entries in any gen tensor must produce 0.0 in the corresponding
    position of all four output tensors.

    lookup_density_values marks NaN coordinates as invalid and fills their
    output slots with 0.0.  This test verifies that build_density_matrices
    propagates that behaviour correctly for both feature pairs
    (delta_E / angle and delta_E / ARM) by injecting NaN at two known
    indices into copies of the real gen tensors.
    """
    nan_indices = [0, 10]

    gen_dE = real_tensors["gen_tensor_delta_E"].clone()
    gen_angle = real_tensors["gen_tensor_annihilation_angle"].clone()
    gen_arm = real_tensors["gen_tensor_arm"].clone()

    for idx in nan_indices:
        gen_dE[idx] = float("nan")
        gen_angle[idx] = float("nan")
        gen_arm[idx] = float("nan")

    n_beta_angle, n_bg_angle, n_beta_arm, n_bg_arm = _run_build(
        real_tensors, gen_dE=gen_dE, gen_angle=gen_angle, gen_arm=gen_arm
    )

    for name, tensor in [
        ("n_beta_angle", n_beta_angle),
        ("n_bg_angle", n_bg_angle),
        ("n_beta_arm", n_beta_arm),
        ("n_bg_arm", n_bg_arm),
    ]:
        for idx in nan_indices:
            assert tensor[idx].item() == 0.0, (
                f"{name}[{idx}] = {tensor[idx].item():.6f}; "
                f"expected 0.0 for NaN gen input"
            )


def test_true_bdecay_events_have_higher_beta_likelihood(real_tensors):
    """
    For events that are truly beta-decay (ground_truths == True), the total
    beta-decay likelihood must exceed the total background likelihood across
    both feature pairs.

    This is the core physics contract of build_density_matrices: the 2D
    probability matrices must capture the correct class-conditional densities
    so that the downstream Bayesian classifier can distinguish signal from
    background.

    The test uses real simulation ground truths from Activation.sim rather
    than synthetic labels, making it a meaningful end-to-end sanity check of
    the entire density-estimation step.

    Expected outcome:
        sum(n_beta_angle[gt]) > sum(n_bg_angle[gt])
        sum(n_beta_arm[gt])   > sum(n_bg_arm[gt])
    where gt is the boolean mask of true beta-decay events.
    """
    gt = real_tensors["ground_truths"]  # bool tensor, True = beta-decay

    n_beta_angle, n_bg_angle, n_beta_arm, n_bg_arm = _run_build(real_tensors)

    beta_angle_signal = n_beta_angle[gt].sum()
    bg_angle_signal = n_bg_angle[gt].sum()
    beta_arm_signal = n_beta_arm[gt].sum()
    bg_arm_signal = n_bg_arm[gt].sum()

    assert beta_angle_signal > bg_angle_signal, (
        f"True bdecay events: expected n_beta_angle sum ({beta_angle_signal:.6f}) "
        f"> n_bg_angle sum ({bg_angle_signal:.6f})"
    )
    assert beta_arm_signal > bg_arm_signal, (
        f"True bdecay events: expected n_beta_arm sum ({beta_arm_signal:.6f}) "
        f"> n_bg_arm sum ({bg_arm_signal:.6f})"
    )


def test_all_four_outputs_have_identical_length(real_tensors):
    """
    All four output tensors must have exactly the same number of elements.

    The downstream likelihood ratio R() operates element-wise across all
    four tensors simultaneously.  Any length mismatch would silently truncate
    or broadcast incorrectly, producing wrong classification results.

    This test acts as a shape-contract safety net against future refactoring.
    """
    outputs = _run_build(real_tensors)

    lengths = [tensor.numel() for tensor in outputs]
    assert len(set(lengths)) == 1, (
        f"Output tensors have inconsistent lengths: {lengths}. "
        "All four must be equal for element-wise R() computation."
    )
