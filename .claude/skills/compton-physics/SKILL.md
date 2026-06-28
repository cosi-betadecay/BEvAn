---
name: compton-physics
description: Physics of Compton telescopes for this repo — Compton kinematics, the Klein-Nishina cross-section, event reconstruction (CKD/Compton sequencing), the Compton Data Space and ARM angular resolution, SPD/electron tracking, sensitivity, polarimetry, Doppler broadening, MeV backgrounds, and the COSI instrument. Sourced entirely from Kierans, Takahashi & Kanbach 2022 ("Compton Telescopes for Gamma-ray Astrophysics", arXiv:2208.07819). Use when reasoning about delta_E / ARM / annihilation-angle features, theta_geo/theta_kin, CKD subset ordering, the 511 keV β⁺ signal, the energy tolerance, or why a feature is physically motivated.
---

# Compton telescope physics for betadecay-analysis

This skill is the physics reference behind the classifier. Every equation,
constant, and concept here comes from one review:

> C. Kierans, T. Takahashi, G. Kanbach, **"Compton Telescopes for Gamma-ray
> Astrophysics"**, Handbook of X-ray and Gamma-ray Astrophysics (2022),
> arXiv:2208.07819. Authors: NASA GSFC / Kavli IPMU Tokyo / MPE Garching.

The paper PDF is local at
`~/Library/CloudStorage/OneDrive-NTNU/COSI/Articles/Compton_Telescopes_for_Gamma-ray_Astrophysics.pdf`.
Equation numbers below (Eq N) match the paper.

## Why this matters here

COSI detects β⁺ (positron) decay: the positron annihilates into **two
back-to-back 511 keV photons**, which Compton-scatter in segmented HPGe
detectors. The classifier's three physics features are direct readouts of the
kinematics in this skill:

| Repo feature (`physics/physics_factors.py`) | Physics in this skill |
|---|---|
| `delta_E` = min \|ΣE − 511\| | 511 keV photopeak = full absorption of one annihilation photon (mₑc² line). See [compton-scattering.md](references/compton-scattering.md). |
| `arm` = (θ_geo − θ_kin)/π, 511-aware | Angular Resolution Measure ΔφARM (Eq 24–26). See [performance.md](references/performance.md). |
| `annihilation_angle` = back-to-back score, pushed toward 1022 keV | Two 511 keV photons emitted anti-parallel; 1022 keV = 2mₑc² total. See [compton-scattering.md](references/compton-scattering.md). |
| CKD subset ordering (`ordering="ckd"`) | Compton Kinematic Discrimination, φ_kin vs φ_geo (Eq 17–19). See [event-reconstruction.md](references/event-reconstruction.md). |

The `arm` and CKD code use `theta_geo` (geometric scatter angle from positions
+ source direction) and `theta_kin` (kinematic scatter angle from energies) —
these are **exactly** φ^geo (Eq 18/26) and φ^kin (Eq 17/25) below.

## Master equation set (the ones the code uses)

Electron rest energy **mₑc² = 511 keV** is the pivot of everything.

- **Compton scatter angle from energies** (Eq 4) — this is `theta_kin`:

  `cos φ = 1 − mₑc² (1/E_scat − 1/E₀)`

  For a fully-absorbed multi-site event E₀ = ΣEᵢ (total deposited) and
  E_scat = E₀ − E₁ (everything after the first interaction). Two-site form
  (Eq 15): `cos φ = 1 − mₑc²/E₂ + mₑc²/(E₁+E₂)`.

- **Geometric scatter angle from positions** (Eq 18/26) — this is `theta_geo`:

  `cos φ^geo = (ḡ₀ · ḡ₁) / (|ḡ₀| |ḡ₁|)`

  ḡ₀ = incoming direction (needs the known/reconstructed **source direction**),
  ḡ₁ = first→second interaction vector. This is why the pipeline back-projects
  the `.tra` to get the source unit vector before computing ARM.

- **Angular Resolution Measure** (Eq 24): `ΔφARM = φ^geo − φ^kin`.
  Zero for a perfectly reconstructed event; its **FWHM is the telescope's
  angular resolution**. Sign carries info: incompletely absorbed recoil
  electron → φ^kin < φ^geo; incompletely absorbed γ → φ^kin > φ^geo.

- **Compton edge / max recoil electron energy** (Eq 7, 8): a photon can
  *never* deposit all its energy in a single Compton scatter. Backscatter
  (φ=180°) gives the most:
  `E_e,max = E₀ (1 − 1/(1 + 2E₀/mₑc²))`, and the Compton edge is
  `E_c = E₀ / (1 + 2E₀/mₑc²)`. Used by the two-site Compton Edge Test (Eq 20–23).

## Key constants

- mₑc² = **511 keV** (electron rest energy; the β⁺ annihilation line)
- 2mₑc² = **1022 keV** (two-photon total; pair-production threshold)
- Compton wavelength h/mₑc = 2.426 × 10⁻¹² m (Eq 3)
- Classical electron radius r₀ = 2.818 × 10⁻¹⁵ m (Eq 10)
- Thomson cross-section σ_T = 8πe²/3mₑc² = 0.665 × 10⁻²⁸ m² (Eq 11)
- Dimensionless photon energy ε = α = E₀/mₑc²

## Reference files (read on demand — full detail + every equation)

- [compton-scattering.md](references/compton-scattering.md) — Section 2: the
  Compton equation, Klein-Nishina (unpolarized Eq 10 + polarized Eq 12), total
  cross-section (Eq 11), Compton continuum/edge, mass attenuation & material
  choice (low-Z scatterer vs high-Z absorber), the three interaction regimes.
- [telescope-principles.md](references/telescope-principles.md) — Section 3:
  event circle, classic double-scatter (COMPTEL) vs compact vs multi-layer,
  three/n-site reconstruction (Eq 16), electron tracking & the event arc,
  dedicated polarimeters.
- [event-reconstruction.md](references/event-reconstruction.md) — Section 4:
  event identification, recoil-electron tracking, **Compton Kinematic
  Discrimination (Eq 17–19)**, two-site reconstruction & the Compton Edge Test
  (Eq 20–23), background rejection via sequencing.
- [performance.md](references/performance.md) — Section 5: the **Compton Data
  Space (χ,ψ,φ)**, ARM (Eq 24–26) & SPD (Eq 27), angular-resolution error
  budget (Eq 28–29), **Doppler broadening** (the ~1° fundamental limit),
  sensitivity (Eq 30), imaging (list-mode vs binned, LM-ML-EM), polarimetry
  (Eq 32–36, modulation factor, MDP), and **MeV backgrounds**.
- [instruments.md](references/instruments.md) — Section 6 + conclusions:
  COMPTEL, Hitomi/SGD, **COSI** (full spec), LXeGRIT, POLAR, MEGA/TIGRE,
  applications, and the open challenges of the field.

## One-paragraph orientation (the "MeV gap")

100 keV–100 MeV is the least-explored band because the photon cross-section is
lowest here and the dominant interaction is Compton scattering, which produces
long-range secondaries. There is no focusing optic, so detectors must be large
(~10 g/cm² interaction depth) and therefore see high backgrounds (atmospheric
albedo + instrument activation). A Compton telescope turns this to advantage:
by measuring the energy and position of each scatter it reconstructs the
photon's origin (to an event circle, or an arc with electron tracking) and
rejects background by demanding events obey Compton kinematics. COMPTEL (1991–
2000) opened the band; COSI (launch 2026) is the next NASA SMEX in it.
