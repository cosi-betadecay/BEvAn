"""Ablation A3 — ARM as delta_E (remove the geometry, keep the energy it carries).

What it is:
    The champion model with the 2D (delta_E, ARM) joint term replaced by an
    explicit 1D P(delta_E) term. ARM is partly a re-encoding of energy: the
    measured scatter angle theta_kin is computed from the deposited energies, so
    the (delta_E, ARM) joint double-counts the energy information. This row keeps
    that energy contribution but strips the angular/geometric part of ARM.

What it tries to prove:
    Whether ARM carries genuine *geometric* information beyond the energy it
    already double-counts. The VM finding is that dropping the (delta_E, ARM)
    joint costs F1 0.8935 -> 0.8681, but n2 is unchanged (ARM's geometry
    contributes ~0) and the loss is all n3 recall — i.e. the apparent ARM signal
    is the delta_E double-count, not the angle. If this row recovers ~0.8935,
    that is confirmed and the paper can state it cleanly.

Why it is important:
    A naive "drop ARM" ablation conflates two effects and a reviewer will catch
    the conflation. This is the scientifically correct decomposition: it isolates
    the geometry from the energy bookkeeping, so the claim "Compton geometry adds
    little here" rests on a fair experiment rather than a misleading term removal.
"""
