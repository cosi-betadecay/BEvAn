"""
Unit tests for the build_density_matrices function in
physics/annihilation_detection_utils.py.

build_density_matrices is a pure-tensor orchestrator: it builds shared bin
edges from combined (signal + background) data, constructs separate 2D
probability matrices for signal (bdecay) and background, then looks up
per-event likelihood values for every event in the general ("gen") tensors.

Because annihilation_detection_utils.py imports ROOT and wandb at module
level, both are stubbed out via sys.modules before the import so these tests
run in any environment without MEGAlib or a W&B account.

Covered tests (5):
    1. Returns four tensors whose length equals the length of the gen tensors.
    2. All returned probability values are non-negative.
    3. NaN entries in the gen tensors produce 0.0 in every output tensor.
    4. Signal-region events yield higher beta likelihoods than background likelihoods.
    5. All four output tensors share the same length (shape consistency check).
"""

import sys
from unittest.mock import MagicMock

import pytest
import torch

# ---------------------------------------------------------------------------
# Stub unavailable C++ / external dependencies before importing the module
# under test.  Using setdefault() preserves a real installation if present.
# ---------------------------------------------------------------------------
for _dep in ("ROOT", "wandb"):
    sys.modules.setdefault(_dep, MagicMock())

from physics.annihilation_detection_utils import build_density_matrices  # noqa: E402

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def synthetic_tensors():
    """
    Create small, fully deterministic synthetic tensors that mimic the real
    pipeline inputs to build_density_matrices.

    Signal (bdecay) events are concentrated around:
        delta_E          ≈ 1.0  keV  (small deviation from 511 keV)
        annihilation_angle ≈ -1.0    (back-to-back, cosine ≈ -1)
        arm              ≈  0.0  rad  (perfect Compton consistency)

    Background (bg) events are placed far away:
        delta_E          ≈ 20.0 keV
        annihilation_angle ≈  0.0
        arm              ≈  1.0  rad

    The "gen" tensor is a direct concatenation of equal-sized slices of the
    signal and background feature tensors so that the first half corresponds
    to signal events and the second half to background events.

    Returns a dict with keys matching the argument names of
    build_density_matrices plus 'n_gen' for convenience in assertions.
    """
    torch.manual_seed(42)

    n_sig = 60
    n_bg = 120
    n_half = 30  # gen = 30 signal + 30 background

    # --- signal ---
    sig_dE = torch.abs(torch.randn(n_sig) * 0.3 + 1.0)  # ~1 keV
    sig_angle = torch.randn(n_sig) * 0.05 - 1.0  # ~-1
    sig_arm = torch.abs(torch.randn(n_sig) * 0.05)  # ~0

    # --- background ---
    bg_dE = torch.abs(torch.randn(n_bg) * 1.0 + 20.0)  # ~20 keV
    bg_angle = torch.randn(n_bg) * 0.1  # ~0
    bg_arm = torch.abs(torch.randn(n_bg) * 0.2 + 1.0)  # ~1 rad

    # --- combined (shared-bin source) ---
    comb_dE = torch.cat([sig_dE, bg_dE])
    comb_angle = torch.cat([sig_angle, bg_angle])
    comb_arm = torch.cat([sig_arm, bg_arm])

    # --- gen: first half is signal, second half is background ---
    gen_dE = torch.cat([sig_dE[:n_half], bg_dE[:n_half]])
    gen_angle = torch.cat([sig_angle[:n_half], bg_angle[:n_half]])
    gen_arm = torch.cat([sig_arm[:n_half], bg_arm[:n_half]])
    n_gen = gen_dE.numel()

    return dict(
        bdecay_tensor_delta_E=sig_dE,
        bdecay_tensor_annihilation_angle=sig_angle,
        bdecay_tensor_arm=sig_arm,
        bg_tensor_delta_E=bg_dE,
        bg_tensor_annihilation_angle=bg_angle,
        bg_tensor_arm=bg_arm,
        gen_tensor_delta_E=gen_dE,
        gen_tensor_annihilation_angle=gen_angle,
        gen_tensor_arm=gen_arm,
        combined_tensor_delta_E=comb_dE,
        combined_tensor_annihilation_angle=comb_angle,
        combined_tensor_arm=comb_arm,
        n_gen=n_gen,
        n_half=n_half,
    )


# ===========================================================================
# Tests
# ===========================================================================


def test_returns_four_tensors_of_correct_length(synthetic_tensors):
    """
    build_density_matrices must return exactly four tensors, each with the
    same number of elements as the gen tensors.

    The four outputs are the per-event likelihood values for:
        n_beta_deltaE_annihilation_angle  — P(event | beta, angle feature)
        n_bg_deltaE_annihilation_angle    — P(event | bg,   angle feature)
        n_beta_deltaE_arm                 — P(event | beta, ARM feature)
        n_bg_deltaE_arm                   — P(event | bg,   ARM feature)

    They must all have length == len(gen tensors) so that the downstream
    Bayesian classifier can pair each likelihood value with the correct event.
    """
    t = synthetic_tensors
    result = build_density_matrices(
        t["bdecay_tensor_delta_E"],
        t["bdecay_tensor_annihilation_angle"],
        t["bdecay_tensor_arm"],
        t["bg_tensor_delta_E"],
        t["bg_tensor_annihilation_angle"],
        t["bg_tensor_arm"],
        t["gen_tensor_delta_E"],
        t["gen_tensor_annihilation_angle"],
        t["gen_tensor_arm"],
        t["combined_tensor_delta_E"],
        t["combined_tensor_annihilation_angle"],
        t["combined_tensor_arm"],
    )

    assert len(result) == 4, f"Expected 4 output tensors, got {len(result)}"
    for i, tensor in enumerate(result):
        assert tensor.numel() == t["n_gen"], (
            f"Output tensor {i} has {tensor.numel()} elements; expected {t['n_gen']}"
        )


def test_all_returned_values_are_non_negative(synthetic_tensors):
    """
    All four returned tensors must contain only non-negative values.

    They represent probability densities looked up from normalized
    histograms.  Negative probabilities are physically meaningless and
    would corrupt the likelihood ratio computation downstream.
    """
    t = synthetic_tensors
    n_beta_angle, n_bg_angle, n_beta_arm, n_bg_arm = build_density_matrices(
        t["bdecay_tensor_delta_E"],
        t["bdecay_tensor_annihilation_angle"],
        t["bdecay_tensor_arm"],
        t["bg_tensor_delta_E"],
        t["bg_tensor_annihilation_angle"],
        t["bg_tensor_arm"],
        t["gen_tensor_delta_E"],
        t["gen_tensor_annihilation_angle"],
        t["gen_tensor_arm"],
        t["combined_tensor_delta_E"],
        t["combined_tensor_annihilation_angle"],
        t["combined_tensor_arm"],
    )

    for name, tensor in [
        ("n_beta_angle", n_beta_angle),
        ("n_bg_angle", n_bg_angle),
        ("n_beta_arm", n_beta_arm),
        ("n_bg_arm", n_bg_arm),
    ]:
        assert torch.all(tensor >= 0.0), f"{name} contains negative values: {tensor[tensor < 0].tolist()}"


def test_nan_in_gen_tensors_returns_zero(synthetic_tensors):
    """
    NaN entries in any gen tensor must produce 0.0 in the corresponding
    position of all four output tensors.

    lookup_density_values marks NaN coordinates as invalid and fills their
    slots with the initial value of 0.0.  build_density_matrices must
    propagate this behaviour correctly for both feature pairs
    (delta_E / angle and delta_E / ARM).
    """
    t = synthetic_tensors

    # Inject NaN at indices 0 and 5 in every gen tensor
    nan_indices = [0, 5]
    gen_dE = t["gen_tensor_delta_E"].clone()
    gen_angle = t["gen_tensor_annihilation_angle"].clone()
    gen_arm = t["gen_tensor_arm"].clone()
    for idx in nan_indices:
        gen_dE[idx] = float("nan")
        gen_angle[idx] = float("nan")
        gen_arm[idx] = float("nan")

    n_beta_angle, n_bg_angle, n_beta_arm, n_bg_arm = build_density_matrices(
        t["bdecay_tensor_delta_E"],
        t["bdecay_tensor_annihilation_angle"],
        t["bdecay_tensor_arm"],
        t["bg_tensor_delta_E"],
        t["bg_tensor_annihilation_angle"],
        t["bg_tensor_arm"],
        gen_dE,
        gen_angle,
        gen_arm,
        t["combined_tensor_delta_E"],
        t["combined_tensor_annihilation_angle"],
        t["combined_tensor_arm"],
    )

    for name, tensor in [
        ("n_beta_angle", n_beta_angle),
        ("n_bg_angle", n_bg_angle),
        ("n_beta_arm", n_beta_arm),
        ("n_bg_arm", n_bg_arm),
    ]:
        for idx in nan_indices:
            assert tensor[idx].item() == 0.0, (
                f"{name}[{idx}] = {tensor[idx].item():.6f}; expected 0.0 for NaN gen input"
            )


def test_signal_region_has_higher_beta_likelihood(synthetic_tensors):
    """
    For events whose features lie in the signal-concentrated region, the
    beta-decay likelihood must be strictly greater than the background
    likelihood.

    The fixture places signal data near (delta_E ≈ 1, angle ≈ -1, ARM ≈ 0)
    and background data near (delta_E ≈ 20, angle ≈ 0, ARM ≈ 1).  The gen
    tensor's first half contains signal events.  For those events:

        n_beta_angle > n_bg_angle   (signal matrix denser at that location)
        n_beta_arm   > n_bg_arm     (same for ARM feature)

    This validates that the matrices capture the correct class-conditional
    densities, which is the core requirement for the downstream Bayesian
    classifier to work correctly.
    """
    t = synthetic_tensors
    n_half = t["n_half"]

    n_beta_angle, n_bg_angle, n_beta_arm, n_bg_arm = build_density_matrices(
        t["bdecay_tensor_delta_E"],
        t["bdecay_tensor_annihilation_angle"],
        t["bdecay_tensor_arm"],
        t["bg_tensor_delta_E"],
        t["bg_tensor_annihilation_angle"],
        t["bg_tensor_arm"],
        t["gen_tensor_delta_E"],
        t["gen_tensor_annihilation_angle"],
        t["gen_tensor_arm"],
        t["combined_tensor_delta_E"],
        t["combined_tensor_annihilation_angle"],
        t["combined_tensor_arm"],
    )

    # First n_half events are signal events
    sig_beta_angle = n_beta_angle[:n_half].sum()
    sig_bg_angle = n_bg_angle[:n_half].sum()
    sig_beta_arm = n_beta_arm[:n_half].sum()
    sig_bg_arm = n_bg_arm[:n_half].sum()

    assert sig_beta_angle > sig_bg_angle, (
        f"Signal events: expected n_beta_angle ({sig_beta_angle:.6f}) > n_bg_angle ({sig_bg_angle:.6f})"
    )
    assert sig_beta_arm > sig_bg_arm, (
        f"Signal events: expected n_beta_arm ({sig_beta_arm:.6f}) > n_bg_arm ({sig_bg_arm:.6f})"
    )


def test_all_four_outputs_have_identical_length(synthetic_tensors):
    """
    All four output tensors must have exactly the same number of elements.

    The downstream likelihood ratio R() zips the four tensors element-wise;
    any length mismatch would silently truncate or broadcast incorrectly.
    This test acts as a safety net against future refactoring that might
    accidentally break the shape contract.
    """
    t = synthetic_tensors
    outputs = build_density_matrices(
        t["bdecay_tensor_delta_E"],
        t["bdecay_tensor_annihilation_angle"],
        t["bdecay_tensor_arm"],
        t["bg_tensor_delta_E"],
        t["bg_tensor_annihilation_angle"],
        t["bg_tensor_arm"],
        t["gen_tensor_delta_E"],
        t["gen_tensor_annihilation_angle"],
        t["gen_tensor_arm"],
        t["combined_tensor_delta_E"],
        t["combined_tensor_annihilation_angle"],
        t["combined_tensor_arm"],
    )

    lengths = [tensor.numel() for tensor in outputs]
    assert len(set(lengths)) == 1, (
        f"Output tensors have inconsistent lengths: {lengths}. "
        "All four must be the same length for element-wise operations."
    )
