# Compton Telescope Performance Parameters (Paper Section 5)

Performance is driven by the precision of the per-interaction energy/position
measurements and of the event reconstruction. The science question sets the
design (a GRB polarimeter needs little energy resolution; a sensitive continuum
imager needs excellent spectral + spatial resolution + background rejection).

## The Compton Data Space (CDS) — Section 5.1, Fig 18–19

Only the first two interactions matter for imaging: the scattered-photon
direction and the scatter angle φ. So analysis uses a reduced **3D data space**:

- **φ** — Compton scatter angle of the first interaction (from energies, Eq 4).
- **χ, ψ** — polar and azimuthal angles of the scattered-photon direction (in
  detector coordinates).

Energy is a 4th (implicit) dimension. This (χ, ψ, φ) space is the **Compton Data
Space (CDS)**, pioneered for COMPTEL. A point source at (χ₀, ψ₀) maps to a
**cone in the CDS** with apex at the source and a **90° opening angle** (because
the measured φ equals, within error, the deviation of the scattered direction
from the source). Each Compton event is a point on this cone; accumulated events
fill the cone walls. The CDS cone is the data-space dual of the image-space event
circle, but gives **much stronger source/background separation** than overlapping
event circles in image space.

The response density along the cone follows Klein-Nishina (small φ near the apex
favored by high-energy events), and **polarization** shows as a sinusoidal
modulation in the azimuthal direction (η ≡ ψ for on-axis sources). Rotating the
CDS so χ,ψ are relative to a known source (Fig 19): the cone becomes a **2D plane
at 45°**, χ → geometric scatter angle φ^geo, ψ → azimuthal η.

## Angular Resolution Measure (ARM) — Section 5.1.1, Fig 20–21

**This is the repo's `arm` feature.** With the source location known, the ARM of
each event (Eq 24):

```
Δφ_ARM = φ^geo − φ^kin
```

- **Kinematic** scatter angle (Eq 25): `φ^kin = arccos[1 − mₑc²(1/E_scat − 1/E₀)]`
- **Geometric** scatter angle (Eq 26): `φ^geo = arccos[(ḡ₀·ḡ₁)/(|ḡ₀||ḡ₁|)]`,
  where ḡ₀ uses the **known source direction** and ḡ₁ is first→second hit.

Δφ_ARM is the shortest angular distance between the source and the event circle.
For an ideal detector Δφ_ARM = 0; measurement errors broaden it. **The FWHM of
the Δφ_ARM distribution is the standard definition of angular resolution.**
Mis-reconstructed events (incomplete absorption) land in the **wings ~±50°**:

- incompletely absorbed recoil electron → **φ^kin < φ^geo**
- incompletely absorbed gamma ray → **φ^kin > φ^geo**

Example: a far-field 511 keV source in the **COSI balloon** detector gives ~6°
ARM resolution.

> This is the physics defense of the `arm` feature: a real 511 keV β⁺ photon with
> correct geometry sits at Δφ_ARM ≈ 0; background sits in the wings. The repo's
> "511-aware" ARM adds an energy push so events off the 511 keV line are also
> shoved away from zero.

## Scatter Plane Deviation (SPD) — Section 5.1.2

For electron-tracking telescopes, the angular resolution is still ARM, but the
electron direction defines the event **arc**. The SPD (Eq 27) measures the arc:

```
Δν_SPD = arccos[ (ḡ₁ × ḡ₀) · (ḡ₁ × ē) ]
```

ḡ₀ = source→first-hit, ḡ₁ = scattered-photon direction, ē = recoil-electron
direction. SPD = angular distance between the source and the calculated photon
origin; it sets the **arc length**. Selecting small-SPD events tightens the
image-space source and strongly cuts background. Electron tracking demonstrated
in gaseous TPCs (SMILE-II: SPD FWHM ~75° for >80 keV recoils from ¹³⁷Cs) and in
multi-layer Si (Kanbach et al.: ~80° for ⁸⁸Y); CCD/CMOS Si achieve ~10–20 μm
spatial resolution for finer tracking.

## Angular-resolution error budget — Section 5.1.3, Eq 28–29

Two independent contributions, added in quadrature (Eq 29):
`dφ_total = √[(dφ^kin)² + (dφ^geo)²]`.

**Energy term** (Eq 28) — propagating energy errors through Eq 4:

```
dφ^kin = (E₀ / sin φ) √[ (1/E²_scat − 1/(E_e+E_scat)²)² dE²_scat
                          + (1/(E_e+E_scat)⁴) dE²_e ]
```

To reach ~1° angular resolution the **energy resolution must be ≲1–2%**; poor
energy resolution hurts most at large scatter angles. Selecting small-φ events
improves resolution.

**Position term** — uncertainty in the first two hit positions tilts the event-
circle axis; estimate `dφ^geo ≲ tan(dx/D)`, dx = spatial resolution, D = inter-
hit distance. (1 mm at 0.5 cm separation → ~11°; at 1 cm → ~6°.) Position error
dominates at higher energies and small separations.

There is a **constant trade-off between angular resolution and efficiency**:
rejecting large-φ or small-separation events sharpens resolution but uses fewer
events. (Fig 24: for a Si/CdTe instrument, energy+Doppler dominate the ARM below
~300 keV; position resolution dominates above.)

## Doppler broadening — the fundamental limit, Section 5.1.4, Fig 25

Klein-Nishina assumes a **free electron at rest**. Real electrons are **bound
with finite momentum**, adding an irreducible uncertainty to the scatter angle —
**Doppler broadening**. This sets a hard floor on angular resolution that **even
an ideal detector cannot beat (~1°)**. Magnitude depends on incident direction,
scatter angle, and **atomic number**:

- **Worst for high-Z** (germanium, CdTe, CZT) — almost ×2 worse than Si.
- **Best for low-Z silicon** (and plastic). Also worst when electron shells are
  completely filled; best for alkali / alkaline-earth metals.

This is a key material trade-off: high-Z gives better stopping power and energy
resolution but worse Doppler-limited angular resolution.

## Sensitivity — Section 5.2, Eq 30

Lowest detectable source flux (the headline metric, Fig 1):

```
F_min(E) = n √(N_S + N_B) / (A_eff T_obs)
         = (n² + n√(n² + 4N_B)) / (2 A_eff T_obs)
```

n = detection significance (3 or 5σ), N_S/N_B = source/background photons,
A_eff = effective area, T_obs = observation time. N_B depends on the size of the
source resolution element — set by ARM, energy bin width, and SPD — so better
energy/position resolution lowers N_B. Standard cut: ARM distance ≤ 1×FWHM.
**Narrow-line sensitivity** (for nuclear lines like 511 keV, ²⁶Al): use only
background within ΔE ≈ 1.5–2×FWHM of the line — so excellent **energy resolution**
directly buys line sensitivity (fewer background photons under the photopeak).
Accurate sensitivity requires detailed simulations (background is largely
instrument activation; effective area depends on event selections). CDS-based
analysis gives lower sensitivity than image space (easier source/background
separation).

## Imaging — Section 5.3, Eq 31, Fig 26–27

Simplest imaging = back-project event circles; overlap density reveals sources
(but extended circles create artifacts → need deconvolution). The imaging
response (Eq 31):

```
D(χ,ψ,φ) = R(χ,ψ,φ; l,b) × I(l,b) + B(χ,ψ,φ)
```

D = measured CDS data, R = instrument response (prob. that a photon from image
bin (l,b) lands in data bin (χ,ψ,φ)), I = sky intensity, B = background. Not
directly invertible → iterative methods. Two classes:

- **List-mode** (e.g. **LM-ML-EM**, List-Mode Maximum-Likelihood Expectation-
  Maximization): per-event, full precision, no predefined response; great for
  bright point sources / terrestrial / quick-look. But can't give absolute flux
  and is biased by un-modeled data-space asymmetries → limited for astrophysics.
- **Binned-mode**: response matrix filled by simulation + calibration; needed for
  accurate astrophysical flux. Algorithms: **Maximum Entropy (MEM)** (lumpy,
  point-source-enhancing), **Richardson-Lucy** (point sources), **MREM**
  (Multi-resolution Regularized EM — smoothest images). Best practice: compare
  methods.

## Polarimetry — Section 5.4, Eq 32–36, Fig 28

From the polarized cross-section (Eq 12), a polarized source modulates the
azimuthal scatter angle η sinusoidally — the **Azimuthal Scatter Angle
Distribution (ASAD)**. The **modulation factor** (Eq 32):

```
μ = A/B = (C_max − C_min)/(C_max + C_min)
```

A = sinusoid amplitude, B = offset. As a function of energy and scatter angle
(Eq 33):

```
μ(E,φ) = sin²φ / (E_scat/E₀ + E₀/E_scat − sin²φ)
```

Maximized at φ ≈ 90° and at **lower energies** (Fig 28). Geometric detector
asymmetries cause false modulation → must normalize by a **simulated unpolarized
ASAD**. Degree of linear polarization (Eq 34): `Π = μ / μ₁₀₀`, where μ₁₀₀ is the
modulation for 100% polarized photons (from simulation).

**Minimum Detectable Polarization (MDP)** (Eq 35), 99% confidence:

```
MDP = (4.29 / (μ₁₀₀ S)) √((S + B) / T_obs)
```

Standard analysis: (1) simulate 100% polarized source → μ₁₀₀; (2) find event
selections minimizing MDP; (3) background-subtract the ASAD; (4) divide by a
simulated unpolarized ASAD to correct geometry; (5) fit the corrected ASAD with
(Eq 36) `C(η) = A cos(2(η − η₀)) + B` → μ = A/B; (6) Π = μ/μ₁₀₀. Pre-flight
polarization calibration is essential (systematics overestimate polarization).
3D position resolution (polar **and** azimuthal angle) + maximum-likelihood
analysis significantly improves polarization sensitivity (demonstrated on the
COSI balloon; POLAR uses the 3ML framework).

## Limitations & MeV backgrounds — Section 5.5, Fig 29

Why the MeV band lags: low Compton cross-section → large detectors hard to
instrument with good resolution; direction constrained only to a circle/arc (not
a direct image); multi-interaction events need heavy reconstruction + simulation
for the response; Doppler broadening caps resolution at ~1°; and **high
background**.

**Background sources** (Section 5.5.1): large absorbing volumes + wide FOV → high
background. Key components:

- **Instrument activation** — cosmic-ray collisions activate the spacecraft/
  detector mass; subsequent **nuclear decays** are a major, sometimes line-
  producing, background. (Directly relevant: the repo's β⁺ signal competes with
  the activation-decay background — many activation isotopes themselves β⁺ decay.)
- **Atmospheric albedo** — cosmic rays hitting the atmosphere produce a broadband
  gamma flux, strongest near the horizon, orders of magnitude above Galactic
  sources.
- **South Atlantic Anomaly (SAA)** passages — drive activation; low-inclination
  (≲5°) low-Earth orbits minimize it.

**Mitigations** (lessons from COMPTEL): **anti-coincidence detectors (ACDs)** to
veto charged-particle events; **low-activation materials** near the detector;
neutron/cosmic-ray **pulse-shape discrimination**; **electron tracking** + event-
quality selections; and accurate pre-launch **simulations** of source + background
response. Even with all this, large data spaces and heavy simulation reliance
remain.
