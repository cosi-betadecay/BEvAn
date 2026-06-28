# Event Reconstruction (Paper Section 4)

Event reconstruction = translating measured positions/energies of interactions
into the initial gamma-ray direction (the event circle, or event arc with
tracking). For classic two-plane telescopes with time-of-flight this is easy
(temporal order known). For compact telescopes the timing resolution can't
order the hits, so the **sequence must be reconstructed** before the event
circle can be defined. This is **Compton sequencing**.

Two main payoffs: (1) higher efficiency — high-energy photons can scatter >5
times before photoabsorption, and reconstructing those long paths recovers them;
(2) **background rejection** — events with missing energy, passive-material
interactions, or **β⁺ decays inside the instrument** are filtered out, raising
signal-to-background. (This is directly the COSI β⁺ problem the classifier
targets.)

## Pipeline of event identification (Section 4.1)

For a tracking telescope, in order (following Zoglauer):

1. Search for a **pair-event vertex** (characteristic inverted "V" of e⁺e⁻).
2. Search for **high-energy charged-particle straight tracks** (cosmic rays,
   muons) — fit a straight line.
3. Search for **recoil-electron tracks** from Compton interactions.
4. Search for the **Compton interaction sequence**.

Compton multi-scatter hits are *separated, isolated* deposits (no straight
pattern), so the easily-recognizable event types are checked/removed first.

### Recoil-electron track reconstruction (Section 4.1.1)

The recoil electron ionizes a track whose length depends on its momentum and
the material density. Two uses: locate the **first interaction** (for sequencing)
and measure the **electron direction** (to constrain the photon, Sec 3.3).
Tracks are rarely straight (often U-shaped), so reconstruction is hard. Standard
trick (Bethe-Bloch / **Bragg peak**): the electron deposits *more* energy as it
slows, so the **lower-energy track end is the start**. Algorithms used:
figure-of-merit, Bayesian, momentum analysis, graph theory.

## Compton Kinematic Discrimination (CKD) — Section 4.2

**This is the `ordering="ckd"` logic in the repo's `Datasets`.** For an event
with N interactions there are N! possible orderings; CKD (a.k.a. the χ²/mean-
squared-difference approach; Aprile et al., formalized by Boggs & Jean) scores
each. For events with ≥3 interactions, the Compton angle of a central
interaction *l* can be computed **two independent ways**:

**1. Kinematically** from the energies (Eq 17):

```
cos φ_l^kin = 1 − mₑc²/E_{L⁺} + mₑc²/(E_l + E_{L⁺})
```

where E_{L⁺} = total energy of all interactions *after* l (E_l not included).

**2. Geometrically** from the hit positions (Eq 18):

```
cos φ_l^geo = (ḡ_k · ḡ_l) / (|ḡ_k| |ḡ_l|)
```

ḡ_k = incoming direction into l, ḡ_l = outgoing direction from l.

For the **correct** ordering in an ideal instrument these two agree. CKD assigns
a χ² quality factor Q to each of the N! permutations (Eq 19):

```
Q = Σ_{i=2}^{N-1}  (cos φ_i^kin − cos φ_i^geo)²
                   / ( (d cos φ_i^kin)² + (d cos φ_i^geo)² )
```

with the d's from error propagation. **Lowest Q = most probable sequence.**

> Note: φ^kin and φ^geo here are the same quantities as `theta_kin` and
> `theta_geo` in `physics_factors.py`. CKD compares them to *order* hits; the
> ARM feature (Eq 24) takes their *difference* once the source direction is known.

Other approaches: deterministic Klein-Nishina-based sequencing; probabilistic
methods that handle escape (not-fully-absorbed) events; increasingly, machine
learning. Performance: standard CKD correctly reconstructs **50–70% of 3+ site
events**; assuming the first interaction is the largest energy deposit improves
sequencing ~20% for ≳1 MeV.

**Background rejection via CKD:** set a maximum acceptable Q. Photons that don't
deposit all their energy, or pair/β-decay events, fail to reconstruct as "good"
Compton events and are cut. Only statistically-good Compton events are kept.

## Two-site event reconstruction (Section 4.3)

Two-site events (one scatter + one photoabsorption) have no redundant info, so
only a fraction can be ordered. Tools:

**Compton Edge Test** — the max energy deposited in a Compton scatter is the
backscatter value (Eq 7); this is the Compton edge (Eq 8). If a measured energy
exceeds the Compton edge max for the given E₀ = E₁+E₂, that interaction *must*
be the photoabsorption (second). Eq 20:

```
max(E_S) = E_e|₁₈₀ = E₀ (1 − 1/(1 + 2E₀/mₑc²))
```

For low incident energy (2E₀ < mₑc², i.e. E₀ ≲ 256 keV), Eq 21–23 give
`max(E_S) < E₀/2 < (E_A+E_S)/2`, hence **max(E_S) < E_A**: the *smaller* deposit
is the Compton scatter. This ordering holds (via simulation) up to ~400 keV;
above ~400 keV the **first** interaction is likelier to be the larger deposit.

More sophisticated: deterministic Klein-Nishina algorithm including the
photoabsorption probability over the inter-hit distance.

**Drawback:** two-site events have no redundant info to confirm full absorption,
so they carry weaker background rejection. Sensitivity can sometimes improve by
**excluding two-site events** for some observations.

> Repo connection: this is exactly why hit-multiplicity bucketing exists. Bucket
> 3 (≥3 hits) has the back-to-back/annihilation hypothesis and the strongest
> kinematic constraints; bucket 2 (2 hits) has ARM but weaker discrimination.
