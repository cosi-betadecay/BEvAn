---
name: positron-annihilation-511
description: The astrophysics of the 511 keV positron-annihilation line — the signal this classifier detects. Sourced entirely from Prantzos et al. 2011 (Rev. Mod. Phys. 83, 1001; arXiv:1009.4620). Covers positronium (para/ortho, the positronium fraction, why most annihilations are NOT a clean back-to-back 2γ), the annihilation channels and emitted spectrum, the β⁺-decay source isotopes (²⁶Al, ⁴⁴Ti, ⁵⁶Co, ²²Na — the same activation radioactivities that make COSI's background), positron energies, line widths, energy loss / thermalization / annihilation, and the observed Galactic morphology. Use when reasoning about the 511 keV / 1022 keV signal, the annihilation-angle feature, the energy tolerance, positronium, or what physically produces signal vs background.
---

# The 511 keV positron-annihilation line for BEvAn

This skill is the astrophysics of the **signal the classifier flags**: the
511 keV γ-ray line from e⁺e⁻ annihilation. Everything here is from one review:

> N. Prantzos, C. Boehm, A. M. Bykov, R. Diehl, K. Ferrière, N. Guessoum,
> P. Jean, J. Knödlseder, A. Marcowith, I. V. Moskalenko, A. Strong,
> G. Weidenspointner, **"The 511 keV emission from positron annihilation in
> the Galaxy"**, Rev. Mod. Phys. **83**, 1001 (2011); arXiv:1009.4620.

PDF local at `~/Downloads/1009.4620v1.pdf`. Section/equation numbers (Sec X, Eq N)
below match the paper. This complements [[compton-physics]] (how COSI *detects*
the photons) — this skill is what the photons *are and where they come from*.

## The one fact that matters most for this repo

A positron annihilating with an electron releases the total rest-mass energy
**1022 keV = 2 mₑc²**. But it does **not** reliably come out as two clean
back-to-back 511 keV photons. Most annihilation goes through **positronium (Ps)**,
a bound e⁺e⁻ atom, and 3/4 of Ps is **ortho-positronium**, which decays to a
**three-photon continuum** *below* 511 keV — not a 511 keV pair.

With the observed Galactic positronium fraction **f_Ps ≈ 0.97**, the fraction of
annihilations giving the clean 2-photon 511 keV signal is only

```
I_2γ ∝ 2 − 1.5 f_Ps  ≈  2 − 1.5(0.97) = 0.545   (relative; Eq 2)
→ ~27% of annihilations yield a back-to-back 511 keV pair; ~73% are ortho-Ps 3γ.
```

> **This is the physical reason the repo's `annihilation_angle` / back-to-back
> feature is intrinsically weak** (cf. the project's own finding that back-to-back
> carries little signal). The energy line at 511 keV (`delta_E`) is robust because
> it tags *any* 2γ annihilation photon and the photopeak; the *geometry* of a
> clean anti-parallel pair only exists for a minority of events, and even then
> COSI must catch both 511 keV photons. See [positronium-and-spectrum.md](references/positronium-and-spectrum.md).

## Mapping to the code

| Repo concept | Physics in this skill |
|---|---|
| 511 keV photopeak (`delta_E`, mₑc²) | direct + para-Ps 2γ line at **511.0 ± 0.08 keV**, intrinsically narrow (FWHM ~1–5 keV astrophysical). |
| 1022 keV target (`annihilation_angle`) | **2 mₑc²** total annihilation energy = two 511 keV photons if fully captured. |
| back-to-back pair score | only direct + para-Ps give 2γ; ortho-Ps (75% of Ps) gives a 3γ continuum → most events have no clean pair. |
| signal vs activation background | both come from **β⁺-decay isotopes** (²⁶Al, ⁴⁴Ti, ⁵⁶Co, ²²Na) — see below. |
| energy tolerance (FWHM 2.25 keV) | that is the *detector* resolution; the *astrophysical* line is ~1.3 keV (narrow) + ~5.4 keV (broad). Don't conflate them. |

## β⁺-decay: the source of the positrons (and the background)

A positron is emitted when an unstable nucleus β⁺-decays (p → n), which requires
the parent-daughter mass difference **ΔM > mₑc² = 511 keV**. The astrophysically
important β⁺ emitters (Table VII) are exactly the **activation/nucleosynthesis
radioactivities** that dominate COSI's instrumental and Galactic background:

| Nuclide | Decay chain | e⁺ branching | mean / endpoint e⁺ (keV) | γ-line (keV) | source |
|---|---|---|---|---|---|
| **²⁶Al** | ²⁶Al → ²⁶Mg* | 0.82 | 543 / 1117 | **1809** | massive stars |
| **⁴⁴Ti** | ⁴⁴Ti→⁴⁴Sc→⁴⁴Ca | 0.94 (⁴⁴Sc) | 632 / 1474 | 68, 78, 1157 | supernovae |
| **⁵⁶Co** | ⁵⁶Ni→⁵⁶Co→⁵⁶Fe | 0.19 | 610 / 1459 | 847, 1238, 1771 | SN Ia |
| **²²Na** | ²²Na → ²²Ne* | 0.90 | 216 / 1820 | 1275 | novae |

Key consequence: **β⁺-decay positrons have E ≲ 1 MeV** (mean a few hundred keV).
That low energy is why radioactivity is the favored source — higher-energy
positrons (pulsars, cosmic-ray p-p, ~30 MeV) would over-produce the in-flight
annihilation continuum that COMPTEL/SPI *don't* see. See
[positron-sources.md](references/positron-sources.md).

## What the photons look like (observed spectrum)

- **Line centroid 511.0 ± 0.08 keV** — no net shift → annihilation essentially at
  rest (thermalized positrons), not in flight.
- **Two Gaussian line components**: narrow FWHM ≈ **1.3 keV** (~2/3 of flux) +
  broad FWHM ≈ **5.4 keV** (~1/3). Broad = positronium formed by charge exchange
  with H; narrow = warm ionized medium.
- **Ortho-positronium 3γ continuum** below 511 keV (Fig 1).
- **f_Ps ≈ 0.97 ± 0.02** (Jean et al. 2006) — ~97% of positrons annihilate via Ps.
- Implies positrons annihilate **cold**, in **warm (T ~ 8000 K)** gas, roughly
  equal parts neutral and ionized. See [positronium-and-spectrum.md](references/positronium-and-spectrum.md).

## Reference files (read on demand)

- [positronium-and-spectrum.md](references/positronium-and-spectrum.md) — Sec II.A,
  V.D–V.E: positronium (para/ortho, spins, lifetimes), the three annihilation
  channels, f_Ps (Eq 1–3), the per-channel line widths, the constructed
  annihilation spectrum (Eq 18–20), and the spectral analysis of the SPI line.
- [positron-sources.md](references/positron-sources.md) — Sec IV: β⁺
  radioactivity (²⁶Al, ⁴⁴Ti, ⁵⁶Co/⁵⁶Ni, ²²Na — Table VII), SN Ia / massive stars
  / novae, plus the high-energy & exotic candidates (cosmic rays, pulsars, X-ray
  binaries, the central black hole, dark matter), and the positron-energy
  constraint.
- [interactions-and-annihilation.md](references/interactions-and-annihilation.md)
  — Sec V.A–V.C: positron energy losses (ionization, Coulomb, bremsstrahlung,
  synchrotron, inverse Compton; Eq 10–15), thermalization, annihilation in flight
  (Eq 16–17), and Table X reaction thresholds.
- [observations-morphology.md](references/observations-morphology.md) — Sec II–III:
  detection history, OSSE/SIGMA/INTEGRAL-SPI, the bulge/disk morphology
  (B/D ~1.4, the "bulge dominance"), the ²⁶Al 1.8 MeV map, and the Galactic
  background material (ISM phases, SFR/SN rates, magnetic fields).
- [propagation.md](references/propagation.md) — Sec VI: why positrons may
  annihilate far from their birthplace (collisional vs collisionless transport),
  and why that decouples the 511 keV map from the source map.

## Headline numbers

- Galactic e⁺ annihilation rate ≳ **2 × 10⁴³ e⁺ s⁻¹**; luminosity bulge/disk
  ratio **B/D ≈ 1.4** (bulge-dominated — unlike any other wavelength).
- 511 keV flux at Earth ~10⁻³ ph cm⁻² s⁻¹; annihilation power ~10³⁷ erg s⁻¹
  (~10⁴ L_⊙); ~3 M_⊙ of positrons annihilated over the Galaxy's lifetime.
- First γ-ray line ever detected from outside the solar system (Johnson et al.
  1972; resolved as 511 keV by Leventhal et al. 1978 with a Ge detector).
- The dominant *known* source is ²⁶Al β⁺-decay (~0.4 × 10⁴³ e⁺ s⁻¹, traced by its
  1.8 MeV line); ⁴⁴Ti adds ~0.3 × 10⁴³. SN Ia (⁵⁶Co) remain a plausible but
  unconfirmed major candidate. After 30+ years, the dominant source is still open.
