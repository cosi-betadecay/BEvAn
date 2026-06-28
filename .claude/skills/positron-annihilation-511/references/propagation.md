# Positron Propagation in the ISM (Paper Sec VI)

The least code-relevant section, but it carries one conceptually important point.

## The core point: the 511 keV map ≠ the source map

The spectral analysis (f_Ps ~ 0.97, line width ΔE/E < 0.01) shows positrons
annihilate at **very low energies** — i.e. they slow down a lot, taking a long
time, during which they can **travel far from where they were born**. Prantzos
(2006) stressed that the morphology of the 511 keV emission therefore does **not
necessarily reflect the source distribution**. This loosens every "source must
match the map" argument:

- e⁺ from **SN Ia** are released in the hot, rarefied ionized medium (large
  scaleheight) and can stream into the halo / far from the disk, **reducing the
  disk 511 keV emissivity** and thus *raising* the effective B/D — potentially
  rescuing disk sources from the bulge-dominance problem.
- If the Galactic halo field has a strong **poloidal component**, positrons
  escaping the disk could be **channeled into the bulge** and annihilate there;
  Prantzos (2006) showed this could explain *both* the total rate (~2 × 10⁴³ s⁻¹)
  *and* the B/D ratio if the SN Ia escape fraction is ~4%.

This is why, despite the bulge-dominated morphology, **dark matter cannot be cleanly
excluded and SN Ia cannot be cleanly required** — propagation smears the spatial
profile.

## Two transport regimes (Sec VI.A–B)

- **Collisional**: Coulomb interactions with gas particles while spiraling along
  field lines. Fig 31: the **stopping distance** D of MeV positrons is much larger
  than the typical size of the hot/warm ISM phases and comparable to the cold
  molecular phase — so in the collisional regime only dense molecular gas readily
  stops them. MeV positrons travel an effective ~S ~ 10 (n/cm⁻³)⁻¹ kpc before
  Coulomb stopping (Jean et al. 2009).
- **Collisionless**: scattering off MHD turbulence in the magnetized plasma
  (resonant cyclotron/Cherenkov interactions with Alfvén / fast-magnetosonic
  waves). Can also **re-accelerate** positrons (stochastic Fermi acceleration,
  Eq 21–24), extending the annihilation region — important in low-density,
  high-B regions like the inner bulge.

The dominant regime is **unknown** because small-scale (~gyroradius) ISM turbulence
is poorly constrained. MHD waves are damped by ion-neutral collisions in neutral
phases (so can't resonate with positrons there) but survive in ionized phases.
Table XII summarizes which transport processes operate in each ISM phase.

## Global models (Sec VI.C)

- **Higdon et al. (2009)**: first "global" model — assumed radioactivity (²⁶Al,
  ⁴⁴Ti, mostly ⁵⁶Co) is the *sole* source, modeled propagation through all ISM
  phases, and reproduced **every** 511 keV observable to <10% (B/D, spectral
  narrow+broad split, even the disk asymmetry, attributed to the spiral-arm
  pattern seen from the Sun). Lingenfelter et al. (2009) argued this lets DM be
  excluded as a major source — but the result hinges on the uncertain SN Ia escape
  fraction and the poorly-known inner-Galaxy turbulence, so it is not definitive.
- A predicted, testable feature: spatial differentiation of the **broad vs narrow**
  511 keV line components (broad in the "middle bulge" via in-flight Ps, narrow in
  the "inner bulge" via thermal Ps).

## Bottom line (relevance to the classifier: low, but good context)

Propagation is about *where* annihilation happens on the sky, not about the
event-level photon kinematics the classifier sees. It matters here only as the
reason the Galactic-scale source debate is unresolved — it does not change what a
single 511 keV annihilation event looks like in the COSI detector. For event-level
physics, see [positronium-and-spectrum.md](positronium-and-spectrum.md) and
[interactions-and-annihilation.md](interactions-and-annihilation.md).
