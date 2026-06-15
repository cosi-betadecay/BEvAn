# Event Reconstruction: Current Approach vs. Boggs & Jean (2000)

This document compares the current event-reconstruction pipeline (rooted in
[`event_processing.py`](../src/betadecay-analysis/physics/event_processing.py))
with the techniques proposed in Boggs & Jean, *"Event reconstruction in high
resolution Compton telescopes"*, A&AS 145, 311 (2000), and proposes a phased
plan for adopting the parts of the paper that are most relevant to this
project.

---

## 1. What the paper proposes

The paper is fundamentally an **ordering algorithm** for multi-site Compton
events in germanium telescopes, plus a layered set of background-rejection
tests. Its headline technique is **Compton Kinematic Discrimination (CKD)**.

For an event with N ≥ 3 interactions, every scatter angle φ_i can be measured
in two independent ways:

- **Geometric** (from positions): `cos φ_i = r̂'_i · r̂'_{i-1}`
- **Kinematic** (from energies, via the Compton formula):
  `cos φ'_i = 1 + 1/W_{i-1} − 1/W_i`, where
  `W_i = (1/m_e c²) Σ_{j>i} E_j`

CKD enumerates **every permutation** of the interactions, computes the χ²
between these two redundant measurements over the N−2 internal scatters
(paper Eq. 13), and picks the ordering with minimum χ². Events with χ²_min
above some threshold (≈5% probability) are rejected — Compton continuum,
pair production, β-decay-with-emission, etc., all fail to satisfy Compton
kinematics.

For two-site events, **Single Scatter Discrimination (SSD)** replaces CKD:
pick the unique physically-allowed ordering, or fall back to "E_1 > E_2".

Section 6 then layers further rejection tests on top:

- Shield veto (reject events from below the instrument)
- Site-count cuts (reject 1-site, reject ≥8-site)
- Minimum lever arm between first two interactions
- β⁻ characteristic-energy screen (Ge daughter lines)
- Positron-signature screen (any combination summing to 511 keV)
- Backscatter / effective-TOF test

The **output** of all this is, per event: a chosen interaction ordering →
an annulus on the sky from the Compton scatter formula → an accept/reject
decision.

---

## 2. What the current pipeline does

[`event_processing.py`](../src/betadecay-analysis/physics/event_processing.py)
enumerates **all non-empty subsets** of hits (the powerset, padded with NaN
sentinels), not permutations of a single ordering. The three returned tensors
then feed three physics factors in
[`physics_factors.py`](../src/betadecay-analysis/physics/physics_factors.py):

- `delta_E`: |Σ E − 511 keV| over each subset
- `annihilation_angle`: minimum pairwise angle between position vectors
  (back-to-back geometry)
- `arm`: angular resolution measure, comparing the kinematic photon direction
  to a separately-computed geometric direction from MEGAlib's `FarFieldImager`
  cone backprojection

Per event the pipeline takes the **best (minimum)** factor across all subsets,
bins those three features into 2D density matrices for β-decay vs background
([`bayesian_annihilation.py`](../src/betadecay-analysis/modeling/bayesian_annihilation.py)),
and outputs a **single Bayesian classification probability**.

---

## 3. Conceptual differences

| | Paper (CKD/SSD) | Current code |
|---|---|---|
| **Enumeration unit** | Permutations of a fixed hit set | All subsets of hits; no ordering inside a subset |
| **Ranking metric** | χ² between geometric & kinematic cos φ over N−2 redundant scatter angles | Minimum of `delta_E`, `annihilation_angle`, `arm` over all subsets |
| **Use of Compton formula** | Central — explicitly compares cos φ ↔ cos φ' at every internal scatter | Only implicit, inside `arm`, and only at the first scatter |
| **Direction reconstruction** | Falls out of CKD: once order is fixed, r̂'_1 + φ_1 give the annulus | Separate MEGAlib backprojection (`FarFieldImager`) that doesn't know about CKD |
| **Background rejection** | Many discrete, physically motivated cuts (χ², lever arm, characteristic-energy, TOF, shield) | One Bayesian likelihood ratio over (`delta_E`, `angle`, `arm`) |
| **Event size handling** | Branches: SSD for N=2, CKD for N≥3, reject 1-site & ≥8-site outright | Uniform powerset treatment for any N |
| **Final output per event** | Ordered interaction list + sky annulus + accept/reject | Scalar P(β-decay \| features) |

---

## 4. Why this matters for β⁺ identification specifically

Two important wrinkles for this project's physics:

1. The paper treats positron annihilation as **background** (Sect. 6.8) and
   uses the 511-keV combination test to *reject* events. This project does
   the **inverse**: it *selects* β⁺ events by their 511-keV annihilation
   photons. The same combinatorial 511-keV check applies, just used with the
   opposite sign of selection.

2. CKD as written assumes a **single photon** scattering multiple times.
   β⁺ annihilation produces **two back-to-back photons**, each potentially
   Compton-scattering multiple times. The "all subsets" approach in the
   current code is implicitly searching for a valid partition of hits between
   the two photon tracks — that's the right instinct but it isn't combined
   with per-photon Compton ordering.

---

## 5. Where the paper is genuinely stronger

1. **Ordering uses two independent measurements per scatter; the current
   code uses one.** `arm` compares kinematic vs geometric photon direction
   only at the first scatter. CKD checks geometric vs kinematic agreement at
   every internal scatter for N≥3 — strictly more information.

2. **Continuum rejection.** A photon that scatters out of the instrument
   before fully depositing its energy gives nonsensical W_i values and fails
   CKD's χ² check. The current pipeline has no analogous physics-grounded
   filter; partially-deposited events pollute the feature distributions and
   rely on the Bayesian classifier to learn to ignore them.

3. **`arm` would itself improve.** `arm` requires knowing which hit is the
   first scatter. If CKD-chosen ordering replaced MEGAlib's implicit choice,
   the `arm` distribution should tighten for true events and the classifier
   gains discriminating power without any change to its structure.

4. **Layered, interpretable cuts.** The paper's tests (lever arm, ≥8-site
   rejection, characteristic-energy screen) are each tied to a specific
   physical background. A Bayesian classifier on three features may learn
   analogous behavior, but only if those backgrounds happen to be
   representable in the chosen feature space.

---

## 6. Where the current approach has advantages worth keeping

- **The Bayesian classifier handles continuous-valued evidence**, while the
  paper's tests are mostly hard cuts. For β⁺ signature this matters — the
  shape of `delta_E` near 511 keV would be discarded by a hard "is any
  subset within ±X of 511 keV?" cut.
- **Subset enumeration, not just permutation enumeration.** For two-photon
  β⁺ events, considering subsets (partitions between the two photons) is
  necessary. The paper doesn't cover this case.
- **No reliance on TOF.** The paper itself notes GeD timing (~10 ns) is too
  coarse to order interactions; CKD exists precisely because TOF doesn't
  work for Ge. The current pipeline is similarly TOF-free, which matches
  the underlying physics.

---

## 7. Proposed phased adoption

### Tier 1 — Cheap physics cuts (a few hours)

Filters added before subset enumeration. Low risk, almost certainly worth it.

- **Reject events with 1 site or ≥8 sites.** Pair production contaminates
  the ≥8 bin; single-site has no Compton info. Paper shows this alone cuts
  ~35% of background.
- **Minimum lever arm** between the first two interactions. One scalar cut.
- **Filter unphysical orderings**: any trial with |1 + 1/W_{i-1} − 1/W_i| > 1
  is physically impossible. Drop those subsets. Free, removes noise from the
  feature distributions.
- **N=2 SSD rule**: when only two hits, assume E_1 > E_2 (initial scatter
  deposits more). Trivial, ~60–80% accuracy per paper Fig. 5.

No architectural change — these are just filters before the existing
`delta_E` / `annihilation_angle` / `arm` computation.

### Tier 2 — CKD χ² as a feature (a few days)

The real "implement the paper" step.

For each subset, and each permutation of that subset, compute:

- Geometric cos φ_i from positions (already present implicitly in `arm`)
- Kinematic cos φ'_i = 1 + 1/W_{i-1} − 1/W_i from energies
- χ²_min = min over permutations of paper Eq. 13

The denominator needs per-interaction energy and position uncertainties
(δE_i, δr_i). Since `.tra` files come from MEGAlib simulation with known
detector resolutions, pull those from the simulation config or hardcode the
COSI values.

Then **add χ²_min as a fourth feature** to the Bayesian classifier in
[`bayesian_annihilation.py`](../src/betadecay-analysis/modeling/bayesian_annihilation.py).
True β⁺ photopeak events will have χ²_min ≈ 1; Compton continuum, partial
deposits, and β-decay backgrounds will have χ²_min ≫ 1. This is the single
most discriminating quantity in the paper.

Computationally tractable: with the ≥8-site cap from Tier 1, at most 7! =
5040 permutations per subset, vectorizable on GPU the same way the current
code vectorizes over combinations.

**Side benefit**: once CKD picks an ordering, use *that* ordering as the
"first scatter" for the `arm` computation instead of MEGAlib's
`FarFieldImager`. The `arm` distribution should tighten noticeably for true
events.

### Tier 3 — Two-photon partition CKD (a week+, conditional)

The honest gap: CKD assumes one photon scattering. β⁺ gives two back-to-back
photons each scattering multiple times. The principled extension:

- For each subset S of hits, treat S as photon-1's track and the complement
  as photon-2's track.
- Apply CKD to each side independently → χ²_1, χ²_2.
- Combined statistic: χ²_combined = χ²_1 + χ²_2, subject to E(S) ≈ 511 keV
  and E(complement) ≈ 511 keV.
- The subset minimizing χ²_combined is the most likely partition.

This is the right model for the physics here, but combinatorics start to
bite and the χ² normalization (degrees of freedom differ per partition)
needs care. Only worth doing if Tier 2 doesn't get the classifier where it
needs to be.

### What to skip

- **Shield veto, effective TOF, β⁻ characteristic-energy screen** — these
  target backgrounds either already suppressed by the Bayesian classifier or
  specific to the paper's GCT configuration. Don't port them blindly.
- **The paper's β⁺ rejection (Sect. 6.8)** — literally inverted for this
  project. Already covered by `delta_E`.

---

## 8. Bottom line

Tier 1 is a half-day of small-cut work. Tier 2 is the meaningful one:
implementing CKD χ² as a per-event feature is probably 1–3 focused days
including validation, and is likely to be the biggest classifier improvement
available without changing the model itself. Tier 3 is research-grade, only
worth it if Tier 2 doesn't suffice.

Start with Tier 2. If χ²_min separates β⁺ from background cleanly (it
should), the rest is polish.

---

## References

- Boggs, S. E. & Jean, P. (2000). *Event reconstruction in high resolution
  Compton telescopes.* Astron. Astrophys. Suppl. Ser. 145, 311–321.
- Current pipeline entry points:
  - [`event_processing.py`](../src/betadecay-analysis/physics/event_processing.py)
  - [`physics_factors.py`](../src/betadecay-analysis/physics/physics_factors.py)
  - [`compton_cone_reconstruction.py`](../src/betadecay-analysis/physics/compton_cone_reconstruction.py)
  - [`bayesian_annihilation.py`](../src/betadecay-analysis/modeling/bayesian_annihilation.py)
