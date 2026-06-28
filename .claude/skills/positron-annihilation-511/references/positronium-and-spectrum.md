# Positronium and the Annihilation Spectrum (Paper Sec II.A, V.D–V.E)

This is the core annihilation physics — what photons come out and at what energy.

## Annihilation channels

A positron annihilating with an electron releases the total rest-mass energy
**1022 keV = 2 mₑc²**. There are three channels:

1. **Direct annihilation** of a free e⁺e⁻ pair at rest → **two photons of 511 keV
   each, emitted back-to-back** (momentum conservation). This is the clean
   "two-photon line" picture.
2. **Para-positronium (p-Ps)** decay → also **2 photons of 511 keV**.
3. **Ortho-positronium (o-Ps)** decay → **3 photons** forming a **continuum from 0
   up to 511 keV** (Fig 1, Ore & Powell 1949) — *not* a line, *not* a pair.

Higher photon multiplicities (4γ, 5γ) and single-photon annihilation exist but
have negligible branching (~10⁻⁶). Single-photon annihilation requires the e⁺e⁻
to be bound to a third body (e.g. a dust grain).

## Positronium (Ps)

A bound e⁺e⁻ "atom", hydrogen-like with reduced mass = half the electron mass, so
its spectral lines are ~half the H frequencies. Two spin states from the (2S+1)
degeneracy:

| State | Spin | Notation | Formed | Decay | Vacuum lifetime |
|---|---|---|---|---|---|
| **para-Ps** (p-Ps) | S=0 (antiparallel) | ¹S₀ | **1/4** of the time | 2γ, 511 keV each | 1.2 × 10⁻¹⁰ s |
| **ortho-Ps** (o-Ps) | S=1 (parallel) | ³S₁ | **3/4** of the time | 3γ continuum (<511 keV) | 1.4 × 10⁻⁷ s |

(Hyperfine splitting between the states = 8.4 × 10⁻⁴ eV, astrophysically
irrelevant.) The 1/4 : 3/4 ratio is pure spin statistics.

## The positronium fraction f_Ps (Eq 1–3)

Let **f_Ps** = fraction of positrons that annihilate via positronium. Then:

- Ortho-Ps 3-photon continuum intensity (Eq 1):
  ```
  I_3γ ∝ (3/4) · 3 · f_Ps
  ```
  (3/4 of Ps is ortho, each giving 3 photons.)
- Two-photon 511 keV line intensity (Eq 2) — direct annihilation of the
  (1 − f_Ps) non-Ps fraction *plus* the 1/4 of Ps that is para:
  ```
  I_2γ ∝ 2(1 − f_Ps) + (1/4)·2·f_Ps = 2 − 1.5 f_Ps
  ```
- Inverting to measure f_Ps from the line/continuum ratio (Eq 3):
  ```
  f_Ps = (8 I_3γ/I_2γ) / (9 + 6 I_3γ/I_2γ)
  ```

f_Ps is a **diagnostic of the annihilation medium** (temperature, ionization,
composition).

### Why this is the key fact for the classifier

Observed **f_Ps ≈ 0.97**. Plug into Eq 2: I_2γ ∝ 2 − 1.5(0.97) = 0.545, while the
total annihilation weight (2γ + 3γ "events") works out so that the **fraction of
annihilations producing a back-to-back 511 keV pair is only ~27%**; the other
~73% are ortho-Ps three-photon continuum with no clean pair and no 511 keV line
photons. So:

- The **511 keV line** (`delta_E` photopeak) is the robust observable.
- A **clean anti-parallel two-photon geometry** (`annihilation_angle`) exists for a
  minority of annihilations — and COSI must still catch *both* 511 keV photons in
  the detector volume. This is the physical basis for the project's measured
  result that the back-to-back feature carries little discriminating signal.

## Observed line: centroid and width (Sec II.B.4, V.E)

- **Centroid E = 511.0 ± 0.08 keV** — no net Doppler shift → positrons annihilate
  essentially **at rest** (thermalized), not in flight. (Bulk Galactic-rotation
  broadening Δv ≲ 100 km/s → ΔE ≲ 0.17 keV, negligible.)
- The line is fit by **two Gaussians** (Churazov et al. 2005, Jean et al. 2006,
  Table II):
  - **Narrow**: FWHM Γ_n = **1.32 ± 0.35 keV**, ~2/3 of the line flux.
  - **Broad**: FWHM Γ_b = **5.36 ± 1.22 keV**, ~1/3 of the line flux.
  - The broad component matches the broadening expected from **positronium formed
    by charge exchange with H atoms**; the narrow one comes from the **warm
    ionized medium**.
- **f_Ps = 0.97 ± 0.02** (Jean et al. 2006); consistent with OSSE (0.97 ± 0.03)
  and TGRS/WIND (0.94 ± 0.04). ⇒ ~97% of positrons annihilate via Ps.
- Spectral analysis ⇒ positrons annihilate **cold, in warm (T ~ 8000 K) gas**,
  roughly equal parts neutral and ionized; <8% in cold molecular clouds, <0.5% in
  hot gas, <1.2% on grains (Jean et al. 2006).

> **Note for the repo's energy tolerance.** The repo's `calculate_tolerance` uses
> **detector** FWHM = 2.25 keV (HPGe resolution) → σ = 2.25/2.355. The
> *astrophysical* line above (1.3 keV narrow, 5.4 keV broad) is a different,
> physical width; don't conflate the two. In the COSI simulation the relevant
> width for the 511 keV photopeak label is the detector resolution.

## Per-channel line widths (Sec V.D)

Each annihilation process imprints a characteristic 511 keV line width (from the
kinetic energy distribution of the annihilating pair / the Ps), e.g. (T ~ 8000 K):

- **Charge exchange** with H / He: ~1.16 / 1.22 keV.
- **Radiative recombination + direct annihilation**: Γ ≈ 1.1 × (T/10⁴)^½ keV.
- **Direct annihilation with bound electrons** (cold neutral media): ~1.56 keV (H),
  2.50 keV (H₂), 1.71 keV (He).
- **On dust grains**: Ps inside grains gives Γ_Ps,in ≈ 2.0 keV, ejected
  Γ_Ps,out ≈ 1.4 keV; grain Ps always annihilates as 2γ ("pick-off").
- Annihilation thresholds: charge exchange needs the positron above ~6.8 eV (H),
  so it only operates at T ≳ a few thousand K, not in cold/hot phases.

## The constructed annihilation spectrum (Eq 18–20)

Per process, the reaction rate for a Maxwellian positron population (Eq 18):
```
r_p = n ⟨σv⟩ = ∫_{E_T}^∞ (2/√π) √E /(kT)^{3/2} e^{−E/kT} σ(E) v dE
```
The total emission spectrum S(E) (Eq 19) sums, over channels, either a 511 keV
Gaussian G(E,E₀,Γ) (line, various widths Γ) or the ortho-Ps 3γ continuum P_t(E),
weighted by the per-channel fractions f_p = r_p/Σr_p (charge exchange, radiative
recombination, direct annihilation, grains) and by abundances X (H/H₂ ≈ 90%) and
Y (He ≈ 10%). The "global" ISM spectrum (Eq 20) combines the five ISM phases
{molecular, cold, warm neutral, warm ionized, hot} with fractions f_i fit to the
SPI data:
```
S_ISM(E) = I_{e⁺e⁻} · Σ_{i=1}^{5} f_i · S_i(E, x_gr)
```
Jean et al. (2006) best fit: **49% warm neutral + 51% warm ionized**, the rest
small. This is the spectral confirmation that positrons die in warm gas.
