# Positron Sources (Paper Sec IV)

Where the positrons come from. The candidate sources are constrained by three
observables (Sec IV.D): (1) total annihilation rate ≳ 2 × 10⁴³ e⁺ s⁻¹; (2)
injection energy ≲ a few MeV (from the MeV continuum, else too much in-flight
annihilation); (3) morphology — the bulge/disk ratio B/D ≈ 1.4 (bulge-dominated).

## β⁺ radioactivity from stellar nucleosynthesis (Sec IV.A) — the main known source

A positron is emitted when an unstable nucleus undergoes β⁺-decay (p → n + e⁺ + ν),
which requires the parent–daughter mass difference **ΔM > mₑc² = 511 keV**. Proton-
rich nuclei made in stellar explosions (where proton captures outrun β⁺-decay) and
Fe-peak nuclei made in nuclear statistical equilibrium (T > 4 × 10⁹ K) are the β⁺
emitters. **This is the single class definitely known to inject positrons into the
ISM**, and these are exactly the activation/decay isotopes the COSI simulation and
the classifier's background contain.

### Table VII — the astrophysically important β⁺ emitters

| Nuclide | Decay chain | e⁺ branching ratio | lifetime | γ-lines (keV) | endpoint e⁺ (keV) | mean e⁺ (keV) | Source |
|---|---|---|---|---|---|---|---|
| **⁵⁶Ni** | ⁵⁶Ni → ⁵⁶Co* | EC | 6.075 d | 158, 812 | — | — | SN Ia |
| **⁵⁶Co** | ⁵⁶Co → ⁵⁶Fe* | e⁺ 0.19 | 77.2 d | 847, 1238, 1771 | 1458.9 | 610 | SN Ia |
| **²²Na** | ²²Na → ²²Ne* | e⁺ 0.90 | 2.61 y | 1275 | 1820.2 | 215.9 | Novae |
| **⁴⁴Ti** | ⁴⁴Ti → ⁴⁴Sc* | EC | 59.0 y | 68, 78 | — | — | Supernovae |
| **⁴⁴Sc** | ⁴⁴Sc → ⁴⁴Ca* | e⁺ 0.94 | 3.97 h | 1157 | 1474.0 | 632 | (⁴⁴Ti chain) |
| **²⁶Al** | ²⁶Al → ²⁶Mg* | e⁺ 0.82 | 7.4 × 10⁵ yr | 1809 | 1117.35 | 543.3 | Massive stars |

(EC = electron capture; the daughter de-excitation emits the listed γ-line. The e⁺
branching is the fraction of decays that emit a positron rather than capturing an
electron.)

### What each contributes

- **²⁶Al** (massive stars): the *best-established* source. Long-lived (τ = 7.4 ×
  10⁵ yr), so it diffuses and traces massive-star regions. Its **1809 keV (1.8 MeV)
  line was the first radioactivity detected in the Galaxy** (HEAO-3, then mapped by
  COMPTEL — irregular, clumpy, along the disk, spiral-arm tangents). SPI measures a
  production/decay rate Ṅ₂₆ ≈ 4.3 × 10⁴² s⁻¹ (≈2.7 M_⊙/Myr). With BR 0.82 it gives
  **Ṅ_e⁺ ≈ 0.4 × 10⁴³ e⁺ s⁻¹** (10–17% of the total Galactic positron rate).
- **⁴⁴Ti** (core-collapse SNe, α-rich freeze-out): directly detected in Cas A via
  its lines. Uncertain yields; contributes **~0.3 × 10⁴³ e⁺ s⁻¹**, comparable to
  ²⁶Al. ⁴⁴Ti sources appear rare (need high yields).
- **⁵⁶Co** (SN Ia, from ⁵⁶Ni → ⁵⁶Co → ⁵⁶Fe): SN Ia make ~0.7 M_⊙ of ⁵⁶Ni →
  **5 × 10⁴⁴ e⁺ s⁻¹ produced *inside* SN Ia**. If only **~4% escape** the remnant
  before annihilating, the observed rate is matched → potential Galactic yield
  ~2 × 10⁴³ e⁺ s⁻¹. **But** SN Ia and LMXRBs have a disk-like morphology, giving
  B/D ≤ 0.5, which does *not* match the observed bulge-dominated B/D ≈ 1.4; and
  whether positrons escape SN Ia at all is unsettled (1D models say no, 3D / radial
  magnetic combing may allow it). SN Ia remain an intriguing but unconfirmed
  candidate.
- **²²Na** (novae): short-lived (2.61 y); novae are close to the plane; modest
  contribution (~10⁴¹ e⁺ s⁻¹), two orders below the observed rate.

> **Key takeaway for the classifier.** β⁺-decay positrons are emitted with **mean
> energies of a few hundred keV and endpoints ≲ 1.5 MeV** — i.e. **≲ 1 MeV typical**.
> This low energy is *why* radioactivity is favored: it naturally satisfies the
> "no excess MeV continuum from in-flight annihilation" constraint that rules out
> high-energy sources. In the COSI sim, both the β⁺ signal and the activation
> background are these same nuclei decaying.

## High-energy & compact sources (Sec IV.B) — disfavored by energy/morphology

- **Inelastic p-p collisions** in cosmic rays: π⁺ → μ⁺ → e⁺. Positron spectrum
  peaks at E ~ 30–40 MeV (Fig 15) — **too energetic** (would over-produce >1 MeV
  in-flight continuum). Galactic cosmic-ray secondaries give ~5–10% of the rate.
- **γ–γ pair production** (Eq 8, Breit-Wheeler) and single-photon pair production
  in B ≳ 10¹² G fields (pulsars/magnetars): produce E ≳ 30 MeV positrons — too
  high; morphology (disk) wrong.
- **Pulsars / ms pulsars / magnetars** (Table VIII): pair cascades in the
  magnetosphere; total ~10⁴² e⁺ s⁻¹ but E > 30 MeV → disfavored.
- **X-ray binaries / microquasars**: pair production in the inner disk or jets;
  LMXRBs spatially peak toward the inner Galaxy (~10³⁹ erg/s); could supply up to
  ~10⁴¹–10⁴³ e⁺ s⁻¹ but the positron content/energy of jets is uncertain. Hard
  LMXRBs are the leading "conventional" candidate for the asymmetric *disk* 511 keV
  emission (Weidenspointner et al. 2008a).
- **The central supermassive black hole (Sgr A\*)**: revived by the bulge-dominated
  SPI image; needs episodic past activity (it is currently very weak) and positron
  diffusion to kpc scales to fill the bulge.

## Dark matter & exotica (Sec IV.C)

Light DM (MeV-scale) annihilating/decaying to e⁺e⁻ was a popular "exotic"
explanation after SPI first showed the bulge. Constraints:
- Injection energy must be ≲ a few MeV (in-flight continuum) → m_DM < a few MeV for
  annihilating/decaying DM (does not constrain *de-exciting* DM).
- Morphology: annihilating/de-exciting DM (rate ∝ n²) gives a cuspy, centrally
  peaked profile (possible if the halo is NFW-cuspy); **decaying DM (rate ∝ n) is
  too flat → excluded**. Light **scalar** DM with fermionic interactions remains a
  bare possibility. Other exotica: MeV millicharged particles, Q-balls,
  superconducting cosmic strings, primordial black holes.

## Assessment (Sec IV.D, Table IX)

No single conventional source reproduces both the rate and the bulge-dominated
morphology. Conclusion: either (i) an unknown source dominates, or (ii) a
combination — e.g. ²⁶Al + ⁴⁴Ti + LMXRBs/microquasars for the disk plus the central
SMBH (or DM) for the bulge — possibly aided by **positron propagation** decoupling
the annihilation map from the source map (see [propagation.md](propagation.md)).
After 30+ years the dominant Galactic positron source remains an open question.
