# Event Reconstruction Theory

Compressed notes from **Boggs & Jean (2000)**, *Event reconstruction in
high resolution Compton telescopes*, A&AS 145, 311–321. Read this file
before forming each iteration's hypothesis. It is the idea backlog.

**Context flip for this project:** Boggs & Jean treat 511 keV β⁺
annihilation as *background* to reject. We want the opposite — 511 keV
is **signal**, all other Compton events are **background**. So when the
paper says "reject events whose W_i = 511 keV," for us that's a *signal
selector*. Every rejection criterion in the paper is potentially a
filter for us, but the sign may flip. Read carefully before applying.

---

## 1. The geometry

A γ-ray of energy E enters and Compton-scatters N times at positions
`r_1, ..., r_N`, depositing E_i at each. Last hit is photoabsorption.

- Direction between scatter i and i+1: `r'_i = (r_{i+1} - r_i) / |...|`.
- Residual photon energy after scatter i (units of m_e c² = 511 keV):
  `W_i = (1/m_e c²) Σ_{j>i} E_j`.
- **Compton formula (Eq 2):** `cos φ_i = 1 + 1/W_{i-1} - 1/W_i`.

Two *independent* ways to measure cos φ_i at every internal scatter:

| Measurement | Formula | Eq |
|---|---|---|
| Geometric (from positions) | `cos φ_i = r'_i · r'_{i-1}` | (9) |
| Kinematic (from energies) | `cos φ'_i = 1 + 1/W_{i-1} - 1/W_i` | (11) |

With N hits, you get `N - 2` redundant pairs. Empty for N = 2 (no
redundancy → CKD doesn't work for 2-site events; use SSD instead).

---

## 2. Compton Kinematic Discrimination (CKD) — the core idea

For every permutation of hits, compute geometric cos φ_i and kinematic
cos φ'_i at each internal scatter, then form

```
χ² = (1 / (N-2)) Σ_{i=2..N-1} (η_i - η'_i)² / (δη_i² + δη'_i²)
```

where η = cos φ (geometric), η' = cos φ' (kinematic). Take the
**permutation with minimum χ²** as the most likely ordering: `χ²_min`.

**Two uses:**
1. **Ordering**: argmin over permutations gives the most likely scatter
   sequence.
2. **Background rejection**: `χ²_min ≫ 1` ⇒ the event is *not* a
   well-behaved multi-scatter photoabsorption. Reject if
   `P(χ²_min, dof=N-2) < 5%` (paper's working threshold; 1%–10% gives
   similar performance).

**Pre-filter inside permutation search:** any trial ordering producing
`|cos φ'_i| ≥ 1` is unphysical → discard before computing χ².

**What CKD rejects naturally:** partially-deposited (Compton continuum),
spatially-unresolved interactions, pair production, β decays.

**Effect on photopeak (paper's GCT model, 3+ sites, 0.2–10 MeV):** ~60–70%
correctly ordered, ~10–20% rejected, ~10–20% mis-imaged.

---

## 3. Single Scatter Discrimination (SSD) for N=2

CKD doesn't work for 2-site events (no redundancy). Instead:

1. Test both orderings against `cos φ_1 < 1`. Drop physically impossible
   ones.
2. If both orderings pass: **prefer the order where E_1 > E_2.** Photons
   that deposit most of their energy in the first scatter are far more
   likely to be photoabsorbed at the second site (empirical result,
   strong above ~0.4 MeV).

Boggs & Jean caution that 2-site events have weak background rejection
and many are backscatters (poor angular resolution from `δφ_{1,E}`).
Be wary of relying on them.

---

## 4. Other rejection criteria (Section 6)

These are the **practical filters** layered on top of CKD/SSD. Most
translate directly into one-line checks inside `event_processing.py`.

### 4.1 Hit-count cuts (§6.2)

- **Reject N = 1** (single site can't be a Compton event).
- **Reject N ≥ 8** (likely pair production / β decay; see Fig 3).
  Current code uses N ≤ 7 already — consistent with this.

### 4.2 Lever-arm cut (§6.9, Fig 8)

Minimum acceptable distance between first and second interaction.
Trade-off: smaller cut = more area, worse angular resolution. Paper
uses 10 cm in Table 2 for the GCT model. For our smaller COSI-like
geometry the cut scales with detector resolution — sweep is appropriate
(current code: 0.5 cm).

### 4.3 Backscatter / Effective TOF (§6.4, §6.5)

After ordering, the first scatter direction `r'_1` tells you if the
photon entered from above or below the instrument, and whether the
initial scatter was forward (cos φ_1 > 0) or backward.

- Reject events whose first scatter appears to originate from **below**
  the instrument (satellite γ, β-decays inside detector, many pair
  productions). Paper: ~95% rejection of below-instrument events.
- Optionally reject **backscatters** (large initial cos φ_1 mismatch).
  Improves angular resolution but cuts effective area.

### 4.4 "Standard φ restriction" (§6.6)

Cut on accepted initial scatter angles to restrict FOV / improve
imaging. Mostly an imaging optimization; secondary for our F1 goal.

### 4.5 Characteristic-energy rejections (§6.7, Table 1)

β⁻ decay daughters emit photons at characteristic energies (e.g.
⁷⁵Ge: 0.265, ⁷¹Zn: 0.512, ⁷⁶As: 0.559, ²⁸Al: 1.779 MeV, etc.).
If any W_i (or any subset-energy combination) matches a characteristic
line within tolerance, reject as β⁻ contamination.

**Sign for us:** ⁷¹Zn at 0.512 MeV is dangerously close to our 511 keV
signal — don't blindly remove it. Treat the other lines as background
markers we can reject; treat lines near 511 keV with care.

### 4.6 Positron signature (§6.8) — **INVERTED for us**

Boggs & Jean *reject* events whose hit combinations sum to 511 keV
(positron annihilation = background for them). For us this is the
**signal selector**: subsets whose total energy is within tolerance of
511 keV are candidate β⁺ annihilation photons. This is essentially
what `delta_E` already encodes — but at the *subset level*, not the
combinatorial-energy level the paper uses.

---

## 5. Performance numbers worth knowing (Table 2)

Percentage of events surviving each cut, applied sequentially, for the
paper's GCT model:

| Cut applied | 0.5 MeV photopeak | β⁻ background | β⁺ background |
|---|---|---|---|
| 2-site events | 65.0% | 65.6% | 77.0% |
| 8+ site events | 64.9% | 62.7% | 74.0% |
| CKD | 51.2% | 17.9% | 16.7% |
| β signatures | 48.4% | 15.1% | 5.2% |
| backscatter / TOF | 35.5% | 6.8% | 1.9% |
| min lever arm (10 cm) | 15.0% | 2.0% | 1.2% |

Read: **CKD is the single biggest discriminator** (factor of ~3–6
improvement in peak-to-Compton ratio). Backscatter/TOF, β-signature,
and lever-arm are smaller but additive multipliers. End-to-end factor
of 5–10× sensitivity improvement vs no reconstruction.

**Implication for our loop:** if any filter is going to move MCC a lot,
it's CKD. Lever-arm and hit-count sweeps are smaller knobs.

---

## 6. Filters to try, ranked by expected impact

1. **CKD χ²_min test** — implement Eq 11–13, take min over permutations,
   reject subsets with `χ²_min > threshold` (start with 5%-tail of
   χ²(N-2)). Highest-leverage filter in the paper.
2. **Per-subset Compton ordering** via argmin χ² (or for N=2, the
   E_1 > E_2 rule). Replaces the "any subset, any order" enumeration
   with an ordered, physics-consistent one.
3. **Unphysical-cosine pre-filter**: drop subsets / orderings where
   any `|cos φ'_i| ≥ 1`. Cheap, no tuning.
4. **Lever-arm sweep**: try 0.1, 0.3, 0.5 (current), 0.75, 1.0, 2.0 cm.
5. **Minimum hit count = 3** (so CKD has at least one redundant
   measurement).
6. **β⁻ characteristic-energy veto**: reject subsets whose ΣE matches
   a Table 1 line within ~3σ — *except* 0.512 MeV (⁷¹Zn).
7. **Below-instrument / backscatter rejection** using first-scatter
   direction sign.
8. **Energy-window prefilter** around 511 keV at the subset level
   (extension of current `delta_E`, but as a hard cut).
9. **"Standard φ restriction"** — initial-scatter-angle FOV cut.

Filters are additive in the paper. In our loop they're tested *one at
a time* (per CLAUDE.md). Keep ones that improve MCC; revert others.

---

## 7. Things the paper does NOT give us

- A closed-form decision boundary for the χ² threshold on *our*
  detector. Has to be tuned (or use the χ² CDF as the paper does).
- Anything about the Bayesian likelihood-ratio classifier — that's
  Zoglauer / our `modeling/` code, downstream of reconstruction.
- Optimal lever-arm value for COSI geometry — the paper's 10 cm is
  GCT-specific.
- A two-photon joint constraint (both annihilation photons summing to
  ~1022 keV). Worth inventing on top of the paper's single-photon CKD.
