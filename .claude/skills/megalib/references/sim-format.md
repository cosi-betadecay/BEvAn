# .sim file format (authoritative)

Sources: `doc/Cosima.pdf` (official manual, format section pp. 8-14) and the
parsing code in `src/sivan/src/MSimEvent.cxx` (`ParseLine`), `MSimIA.cxx`
(`AddRawInput`/`ToSimString`), `MSimHT.cxx` — all under
`/Users/aryaraeesi/Documents/COSI/megalib`. Where the PDF and source disagree,
the source wins (one such case: the particle ID table, see below).

Applies to **Version 101** (Cosima output, MEGAlib 4.00.00), which is what
`data/Activation.sim` uses. Units everywhere: **cm, keV, seconds**.

## File header keywords

| Keyword | Meaning |
|---|---|
| `Type SIM` / `Version 101` | file type and format version |
| `Geometry <path>` | geometry used by the generating machine (pass your local one explicitly) |
| `Date`, `MEGAlib`, `Seed` | provenance |
| `BeamType`, `SpectralType` | source definition (here: `Activation`) |
| `TB` / `TE` | observation start/end time [s] |
| `TS` | total number of *started* (simulated) events |
| `NF` / `IN` | include/concatenation file references |

## SE event block keywords (MSimEvent::ParseLine)

| Keyword | Meaning |
|---|---|
| `SE` | start of event |
| `ID a b` | triggered event ID, simulated (started) event ID |
| `TI` | event time [s] |
| `ED` | total energy **deposited in sensitive material** [keV] |
| `EC` | total energy of particles that **escaped** the world volume [keV] |
| `NS` | energy lost in **non-sensitive (passive) material** [keV] |
| `PM <material> <E>` | energy in a specific passive material |
| `GR x y z E` | guard-ring deposit of strip detectors |
| `XE x y z E` | total detector energy at detector-center position |
| `DR x y z dx dy dz E` | electron direction info (Strip3DDirectional) |
| `VT E1..E4` | veto detector energies (>100 keV ⇒ vetoed) |
| `BD <flags>` | bad-event flags |
| `CC <text>` | comment |
| `EN` | end marker |

Data check (Activation.sim event 1): `ED 10.33` = the HTsim energy;
`EC 222.31` = the energy of the escaping neutrino (`IA ESCP`). Confirmed.

## IA line — 24 semicolon-separated fields after `IA <PROC>`

```
IA PROC ID; Origin; DetType; Time; X; Y; Z;
   MotherPID; MDirX; MDirY; MDirZ; MPolX; MPolY; MPolZ; MEnergy;
   SecPID;    SDirX; SDirY; SDirZ; SPolX; SPolY; SPolZ; SEnergy
```

- **ID** — 1-based interaction ID, unique within the event.
- **Origin** — IA ID of the interaction that created the incoming (mother)
  particle; 0 for the primary (`INIT`).
- **DetType** — detector type at the interaction point (see table in
  cosima-geomega.md; 0 = outside any detector).
- **Mother block** = the incoming particle *after* the interaction
  (PID, direction, polarization, remaining kinetic energy).
- **Secondary block** = the newly created particle (PID, emission direction,
  polarization, kinetic energy).
- Directions are unit-vector components; polarization meaningful for photons.

Parsing: `MSimIA::AddRawInput`, MSimIA.cxx:110-158. Serialization:
`ToSimString`, MSimIA.cxx:192-286.

### Process codes

`INIT` (primary creation), `DECA` (radioactive/particle decay), `ANNI`
(e⁺e⁻ annihilation), `COMP` (Compton), `PHOT` (photoeffect), `BREM`
(bremsstrahlung), `RAYL` (Rayleigh — no secondary recorded), `PAIR`
(pair production), `IONI` (ionization), `INEL` (inelastic hadronic),
`CAPT` (capture), `ESCP` (left the world volume — secondary PID 0),
`ENTR`/`EXIT` (watched-volume crossing), `BLAK` (black absorber).

### ANNI lines (the annihilation ground truth)

One annihilation at rest ⇒ **two consecutive ANNI lines** with: same
position (= the vertex), same OriginID (the DECA that made the e⁺), mother
= PID 2 with zero direction/energy (annihilated at rest), secondary = PID 1
(γ) with **exactly negated unit directions** and ~510.999 keV each.
In-flight annihilation also occurs (secondary energies ≠ 511, directions
not anti-parallel in the lab frame) — treat 180°/511+511 as the dominant
mode, not a law. Truth exists only in simulation: use for development and
validation metrics, never as a runtime feature input.

## Particle IDs (authoritative: MCSource.cc:118-175 — "Don't change this list because the ID is written to the Sim file")

| ID | Particle | ID | Particle |
|---|---|---|---|
| 1 | gamma | 12 | **electron neutrino ν_e** |
| 2 | positron e⁺ | 13 | anti-electron neutrino |
| 3 | electron e⁻ | 14/15 | ν_μ / anti-ν_μ |
| 4 | proton | 16/17 | ν_τ / anti-ν_τ |
| 5 | anti-proton | 18 | deuteron |
| 6 | neutron | 19 | triton |
| 7 | anti-neutron | 20 | ³He |
| 8/9 | μ⁺ / μ⁻ | 21 | α |
| 10/11 | τ⁺ / τ⁻ | 22 | generic ion |

(23+ are mesons/hyperons, rarely relevant here.)
**Nuclei encode as 1000·Z + A**: 29061 = Cu-61, 28061 = Ni-61,
32071 = Ge-71, 31071 = Ga-71 (all present in Activation.sim — Ge activation
chains). NOTE: the table in Cosima.pdf p.14 was found to disagree with the
source for IDs ≥ 4 (the PDF-derived reading had 12 = proton); the source
constants above are correct and match the data (in β⁺/EC decays the
escaping PID-12 particle carries the `EC` energy ⇒ it is the neutrino).

## HTsim line (hits — what the detector "measures")

```
HTsim DetType; X; Y; Z; Energy; Time; OriginIA1; OriginIA2; ...
```

First field is the **detector type**, then position [cm] (voxel/strip
center after discretization), deposited energy [keV] (after noising), time
[s], then the list of IA IDs whose particles contributed energy to this
hit — the link from measurement back to truth (`MSimHT::IsOrigin(id)`).

Variants: version 100 `HTsim` had no time; versions 200/201 use keyword
`HT` with uncertainty fields. Version 101 = `HTsim` with time (this data).
Parsing: `MSimHT::AddRawInput`, MSimHT.cxx:169-251.
