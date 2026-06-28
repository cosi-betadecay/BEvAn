# Basic Operating Principles of Compton Telescopes (Paper Section 3)

In the 100 keV–tens-of-MeV range, photoabsorption and pair-production
instruments are too inefficient; the only process with reasonable probability is
Compton scattering. But the initial direction/energy **cannot** be derived from
the first scatter alone — the photon must be (ideally) fully absorbed and the
energy + position of **each** interaction recorded.

## The event circle (Fig 8)

Consider a two-site event: one Compton scatter (E₁, r̄₁) followed by
photoabsorption (E₂, r̄₂). Then:

- Initial energy: **E₀ = E₁ + E₂** (total deposited).
- First-scatter Compton angle (Eq 15): `cos φ = 1 − mₑc²/E₂ + mₑc²/(E₁+E₂)`
  (from Eq 4 with E_scat = E₂, E_e = E₁).
- The scattered photon direction r̄₁ → r̄₂ plus the angle φ define a **cone** on
  the sky. Back-projecting that cone gives the **event circle** — the locus of
  possible source positions for that one photon.

Without the recoil-electron direction you cannot tell *where on the circle* the
photon came from. Multiple photons from the same source give **overlapping event
circles** whose intersection reveals the source (basis of imaging, Sec 5.3).

## Three configurations (Fig 11)

1. **Classic double-scatter (COMPTEL-like)** — two separated detector planes:
   D1 = low-Z scatterer, D2 = high-Z absorber. Measures exactly one
   Compton scatter + one photoabsorption (two-site events only). Large
   separation enables **time-of-flight** to order the two hits and reject
   upward-moving background.
2. **Compact** — a single large 3D position-sensitive volume acts as both
   scatterer and absorber; measures **multiple** Compton scatters with **no
   geometric limit on the scatter angle**. Higher efficiency. No time-of-flight
   → needs Compton sequencing (Sec 4) to order hits.
3. **Multi-layer** — many thin position-sensitive detectors (e.g. thin Si);
   adds **electron tracking** (Sec 3.3) on top of the compact advantages.

Modern telescopes are mostly compact/segmented, built around a **3D
position-sensitive detector volume** with (sub)millimetre spatial resolution.

### COMPTEL (the first space Compton telescope)

On NASA's CGRO (1991–2000). D1 = NE 213A liquid scintillator (low-Z, enhances
scattering), D2 = NaI(Tl) (high-Z, enhances absorption); planes ~1.5 m diameter,
1.5 m apart, ~600 kg total. 0.75–30 MeV. Two background-rejection tools:
**time-of-flight** (4.5 ns window selects downward photons) and **pulse-shape
discrimination** in the liquid scintillator (separates photons from neutrons).
Effective area only 10–50 cm² (efficiency <1%); angular resolution a few degrees;
FOV ~1 sr. Produced the first MeV all-sky survey and the ²⁶Al 1.8 MeV map.

## Multi-site reconstruction (Eq 16)

For a three-site event (two scatters + absorption), the first-scatter angle is
still from Eq 4, but the "scattered" energy now includes everything after the
first interaction:

```
cos φ = 1 − mₑc²/(E₂+E₃) + mₑc²/(E₁+E₂+E₃)
```

Generalizes to n sites: E₀ = E₁ + … + Eₙ (must be fully absorbed) and the
scattered-photon energy after the first interaction is E_scat = E₂ + … + Eₙ.
More interactions → more redundant information → better background rejection.

## Electron tracking and the event arc (Section 3.3, Fig 12–13)

If the **recoil-electron direction** is measured, momentum conservation further
constrains the incident photon, reducing the event circle to a short **event
arc**. The kinematics of the recoil electron are treated exactly like the
scattered photon (Sec 2, Eq 9). Key points:

- The electron direction is **less precise** than the scattered-photon direction
  (short penetration depth + Molière scattering), so electron tracking does
  **not generally improve angular resolution** — but it sharpens the point
  spread function in image space and is a powerful **background-rejection** tool
  (fewer photons needed to localize a source).
- Demonstrated first by TIGRE and MEGA (thin Si layers); also in gaseous TPCs.
- Track length is set by electron energy and material density: too short to
  resolve in high-Z (Ge, CdTe, CZT); resolvable in low-Z Si or low-density
  gas. Higher-energy electrons (≳1 MeV) have longer, straighter tracks.
- An electron-tracking telescope can also reconstruct **pair-production** events
  (the e⁺e⁻ vertex), enabling a combined Compton+pair instrument spanning the
  MeV gap (Sec 6.4).

## Dedicated polarimeters (Section 3.4)

A class of Compton telescopes optimized for **polarimetry, not imaging**. From
the polarized cross-section (Eq 12), the amplitude of the azimuthal-scatter (η)
modulation is maximized at scatter angle φ ≈ 90°. A polarized source produces a
sinusoidal η distribution. Simplest design: scatter in a central element, absorb
in a detector at 90°, sample around the azimuth. Energy resolution is **not
critical** here (energy isn't used in the reconstruction), so these instruments
favor simple, high-efficiency, large-FOV scintillators. Optimized for transients
with high signal-to-background (e.g. GRBs). Example: **LEAP** (NASA MoO concept),
two scintillator types for scatter (low-Z) and absorb (high-Z).
