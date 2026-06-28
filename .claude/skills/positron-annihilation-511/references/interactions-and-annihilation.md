# Positron Interactions with Matter and Annihilation (Paper Sec V)

How a positron, once injected at ~MeV energy, slows down and finally annihilates.
The chain (Fig 22): **energy losses → thermalization → annihilation** (after
slowing), with a fraction annihilating **in flight** while still energetic.

## Energy losses (Sec V.A)

A positron loses energy via the same electromagnetic processes as an electron;
which dominates depends on its energy and the local density / B-field / radiation
field (Fig 23, energy-loss rates):

- **Ultra-relativistic (E > 10 GeV)**: inverse Compton on CMB + ISRF, and
  synchrotron. (Eq 10) IC: dE/dt = −2.6×10⁻¹⁴ u_rad γ²β² eV/s. (Eq 11) synchrotron:
  dE/dt = −9.9×10⁻¹⁶ B² γ²β² sin²α eV/s. Synchrotron dominates IC when B > 6.3 μG
  √u_rad (e.g. near pulsars, the Galactic centre).
- **1–10 GeV**: bremsstrahlung (Eq 12–13).
- **Below 1 GeV**: Coulomb scattering off free electrons (ionized gas) and
  ionization/excitation of atoms/molecules (neutral gas) — a quasi-continuous loss
  (Eq 14–15, Bethe-Bloch-like). Coulomb (Eq 14): dE/dt = −7.7×10⁻⁹ (n_e/β)[ln(γ/n_e)
  + 73.6] eV/s.
- **Below ~100 eV**: the positron can pick up an electron (charge exchange) to form
  positronium "in flight"; cross sections must be modeled by Monte Carlo.

Eq 10 and Eq 15 between them estimate the loss rate to good approximation across
the range.

## Thermalization (Sec V.B–C)

When the positron reaches energies near the ambient gas thermal energy it
"thermalizes" — its distribution relaxes to a Maxwellian at the gas temperature T.
The key result: **the thermalization (slowing-down) timescale is much shorter than
the annihilation timescale** (annihilation ~10¹²–10¹⁴ s; slowing ~ much less). So
positrons **annihilate essentially at rest**, justifying Maxwellian-averaged
annihilation rates and explaining the unshifted 511 keV centroid. (A statistical-
physics Fokker-Planck treatment confirms this.)

## Annihilation in flight (Sec V.B)

Before fully slowing, a positron can annihilate **in flight** with a free/bound
electron. The two photons are then **Doppler-shifted**, producing a *continuum*
in the range mₑc²/2 ≲ E_γ ≲ E + mₑc²/2 (E = positron total energy), not the 511 keV
line. Probability of annihilating in flight before reaching energy E (Eq 16):
```
P(E₀,E) = 1 − exp(−n_e ∫_E^{E₀} v(E') σ_a(E') / |dE'/dt| dE')
```
with the annihilation cross-section (Eq 17, Dirac, valid for E > 75 keV):
```
σ_a = (π r_e²/(γ+1)) [ (γ²+4γ+1)/(γ²−1) · ln(γ+√(γ²−1)) − (γ+3)/√(γ²−1) ]
```
For positrons injected ≲ 1 MeV (i.e. β⁺-decay), the in-flight fraction is
**negligible (≲4%)**. For high-energy injection (pulsars, p-p, ~30 MeV) it is
substantial → the in-flight continuum above 1 MeV becomes detectable. **The
observed lack of such a continuum (COMPTEL/SPI) is the constraint that favors
low-energy (radioactive) positron sources** and bounds DM masses to ≲ a few MeV.

## Annihilation processes after thermalization (Sec V.D)

Once thermal, the positron annihilates by one of several processes, listed by
decreasing cross-section (Table X gives reaction thresholds, Fig 27 cross-sections):

1. **Charge exchange** with atoms/molecules (H, He, H₂) → forms positronium.
   Threshold 6.8 eV (H), 17.8 (He), 8.6 (H₂) — so it operates **only at T ≳ a few
   thousand K**, not in cold or fully-ionized media. Highest cross-section where it
   is allowed; gives the broad 511 keV line (~1.16 keV at T=8000 K).
2. **Radiative recombination** with free electrons → Ps.
3. **Direct annihilation with free electrons** (important only in hot gas, T > 10⁵
   K).
4. **Direct annihilation with bound electrons** (dominant in cold neutral media
   where charge exchange is below threshold).
5. **Annihilation on dust grains** (Ps formation inside grains).

The relative fractions set the **positronium fraction f_Ps** and the spectral
shape, which is why the observed spectrum is a sensitive probe of the annihilation
medium (warm, T ~ 8000 K, ~equal neutral/ionized — see
[positronium-and-spectrum.md](positronium-and-spectrum.md)).
