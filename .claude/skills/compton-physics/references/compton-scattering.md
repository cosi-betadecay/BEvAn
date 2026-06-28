# Physics of Compton Scattering (Paper Section 2)

Compton scattering: a gamma ray partially transfers energy to a bound electron,
ejecting it (the recoil electron) and scattering at angle φ with lower energy.
The scattered photon may scatter again (and again) before final photoabsorption.
Only the **complete** measurement of all secondaries (recoil electron + de-
energized scattered photon) recovers the incident photon's energy and direction.

Historical note: discovered by Arthur Compton (Nobel 1927); could not be
explained by coherent (Thomson/Rayleigh) wave scattering — required the photon
(quantum-of-light) picture.

## Conservation laws and the Compton equation

For an incident photon of energy E₀ scattering off an electron assumed **free
and at rest** (this assumption is revisited under Doppler broadening, Sec 5.1.4):

- Energy conservation (Eq 1): `E₀ = E_scat + E_e`
  (E_scat = scattered photon energy, E_e = recoil electron kinetic energy)
- Momentum conservation (Eq 2): `p̄₀ = p̄_scat + p̄_e`
  with |p̄₀| = hν₀/c, |p̄_scat| = hν_scat/c, |p̄_e| = mₑvγ, γ = 1/√(1−β²), β = v/c.

These give the **Compton equation** in wavelength (Eq 3):

```
λ_scat − λ₀ = (h / mₑc) (1 − cos φ)
```

h/mₑc = 2.426 × 10⁻¹² m is the Compton wavelength (the shift at 90°). A photon
**can never lose all its energy** in a single Compton scatter, even backscattered.

Solved for the scatter angle in terms of energies (Eq 4) — **this is `theta_kin`**:

```
cos φ = 1 − mₑc² (1/E_scat − 1/E₀)
```

Scattered photon energy (Eq 5):

```
E_scat = E₀ / (1 + (E₀/mₑc²)(1 − cos φ))
```

So E_scat is uniquely determined by E₀ and φ. The recoil-electron kinetic
energy (Eq 6), the quantity the detector actually ionizes with:

```
E_e = E₀ − E_scat = E₀ · ( E₀(1 − cos φ) / (mₑc² + E₀(1 − cos φ)) )
```

## Compton continuum and Compton edge (Fig 3)

Because φ ranges over all angles, the recoil-electron spectrum is a **continuum**
from 0 (grazing, φ≈0, E_e≈0, E_scat≈E₀) up to a maximum at backscatter:

- Max recoil electron energy (Eq 7), φ = 180°:
  `E_e|₁₈₀ = E₀ (2E₀/(mₑc²+2E₀)) = E₀ (1 − 1/(1 + 2E₀/mₑc²))`
- **Compton edge** (Eq 8), the gap between E₀ and that maximum:
  `E_c = E₀ − E_e|₁₈₀ = E₀ / (1 + 2E₀/mₑc²)`

These define the Compton Edge Test used to order two-site events (Eq 20–23,
see event-reconstruction.md).

## Recoil electron via its own scatter angle Θ (Eq 9)

With Θ the angle between incident photon and recoil electron direction, and
α = E₀/mₑc²:

```
cot(Θ) = (1 + α) tan(φ/2)
E_e = 2E₀ α cos²Θ / ((1+α)² − α² cos²Θ)
```

This is how electron-tracking telescopes use the measured electron direction
to further constrain kinematics.

## Klein-Nishina cross-section (the angular distribution)

Full QED treatment of γ + e⁻ → γ + e⁻ on free electrons (Klein & Nishina 1928).
**Unpolarized** differential cross-section (Eq 10):

```
dσ/dΩ = (r₀²/2) (E_scat/E₀)² ( E_scat/E₀ + E₀/E_scat − sin²φ )
```

r₀ = 2.818 × 10⁻¹⁵ m (classical electron radius). Consequences (Fig 4):

- **Higher-energy photons scatter forward** (small φ favored); the cross-section
  becomes strongly forward-peaked at MeV energies.
- **Lower-energy photons scatter more isotropically** (at 10 keV the magnitude
  varies only ~×2 over all angles).

**Total cross-section** (Eq 11), in Thomson units (σ_T = 8πe²/3mₑc² =
0.665 × 10⁻²⁸ m²), dimensionless energy ε = E₀/mₑc²:

```
σ_C = (3/4) σ_T [ (1+ε)/ε³ ( 2ε(1+ε)/(1+2ε) − ln(1+2ε) )
                  + ln(1+2ε)/(2ε) − (1+3ε)/(1+2ε)² ]
```

As ε → 0 (E₀ ≪ mₑc²) this reduces to the Thomson cross-section σ_T. The total
cross-section scales with the **electron density** of the material (each scatter
is independent → sum over electrons; depends on Z and ρ). Cross-section data:
NIST **XCOM**.

## Polarized cross-section (Eq 12) — basis of polarimetry

Linearly-polarized differential cross-section (Heitler 1954), with η the
azimuthal scatter angle measured relative to the photon's initial electric-field
vector ξ̄:

```
dσ/dΩ = (r₀²/2) (E_scat/E₀)² ( E_scat/E₀ + E₀/E_scat − 2 sin²φ cos²η )
```

Maximized when cos²η = 0, i.e. **photons preferentially scatter at 90° (η=0
direction is suppressed) to the incident E-field vector**. Measuring the
azimuthal distribution of η therefore measures linear polarization — Compton
telescopes are inherently polarimeters (see performance.md, Sec 5.4).

## Mass attenuation and material choice (Eq 13–14, Fig 6–7)

Beam attenuation through thickness x (g/cm²), x = l·ρ (Eq 13):

```
I = I₀ e^(−μx)
```

μ = mass attenuation coefficient (cm²/g). Path length for a given attenuation
(Eq 14). **Worked example:** silicon at 511 keV has μ = 8.67 × 10⁻² cm²/g,
ρ = 2.33 g/cm³, so absorbing/scattering 50% of 511 keV photons needs

```
l = (1/ρμ) ln(I/I₀) = ln(0.5) / (2.33 · 8.67e-2) ≈ 3.4 cm
```

— illustrating why Compton telescopes need **large detector volumes**.

**Three interaction regimes** (Fig 7, as functions of energy and absorber Z):

- **Photoelectric absorption** — dominant at low energy / high Z (steep Z
  dependence). Used to fully absorb the scattered photon (high-Z absorber).
- **Compton scattering** — dominant in the middle band. For **low-Z** materials
  (plastic scintillator ρ≈1, silicon Z=14) Compton dominates ~30 keV–30 MeV;
  for **high-Z** (NaI(Tl) ρ=3.67) only ~300 keV–8 MeV.
- **Pair production** — dominant above ~few MeV (threshold 2mₑc² = 1022 keV).

Design implication: **low/medium-Z materials are the best Compton scatterers**
over a wide range (plastic, Si); **high-Z, high-density materials are the best
absorbers** (NaI(Tl), CsI(Tl), CdTe, CZT). Germanium (used in COSI) is a
high-resolution semiconductor that does both within one large volume.
