---
name: geometry-complexity
description: "How simple/realistic each of the 4 in-domain MEGAlib geometries actually is — Max/GeACT idealized, SPILike intermediate, NCT a realistic mass model. For the generalization claim's framing."
metadata: 
  node_type: memory
  type: project
  originSessionId: d5f04a0b-fcd4-43de-b51b-e29ad85e85da
---

2026-06-25 (read the actual .geo.setup files in /Users/aryaraeesi/Documents/COSI/megalib/resource/examples/geomega/special/). Q from user: "are the geometries we use simple?" Answer = MIXED, and it matters for the "generalizes across geometries" paper claim.

- **Max.geo.setup** — GENUINELY SIMPLE. Header literally: "Very simple Germanium ACT - similar to NCT". 5 volumes, 2 materials (Al, Ge). 4 idealized Ge wafers in Al cryostats. This is the train reference / champion-0.89 dataset. Do NOT call it realistic.
- **GeACT.geo.setup** — SIMPLE/idealized. 6 volumes, 2 materials (Al, Ge). Clean strip array, no passive clutter.
- **SPILike.geo.setup** — INTERMEDIATE. "SPI like coded mask telescope". 12 volume decls but 124 .Copy (many repeated detector elements), BGO anticoincidence shield (ACSRing + ACSDisk), hexagonal Ge (PGON). Materials BGO+Ge only — real instrument concept, idealized materials.
- **NCT** = `Simple_Ge_NCT_try1_040721.geo.setup` (NCT = Nuclear Compton Telescope, the flown balloon instrument) — REALISTIC MASS MODEL despite "Simple_" in filename. 22 volumes, 9 distinct materials (Copper coldfinger, Alu6061, GePassive dead layers, Indium thermal joints, Steel_18_8 screws, ro4003/roTMM3 PCB boards, Al strips 400nm). Built from "Jason Bowen's MGEANT mass model". This is the REALISM ANCHOR — proves the method survives real passive material (scores 0.876).

Framing guidance for the paper: span is real along DETECTOR CONFIG (count 4→125, segmentation 40×40/78×78/calorimeter, thickness, shield, layout) — that claim is solid. Span along REALISM/passive-mass is UNEVEN (2 idealized + 1 intermediate + 1 mass model). Honest claim: "idealized Ge stacks (Max, GeACT) → coded-mask+shield (SPILike) → detailed instrument mass model (NCT)". Lean on NCT as realism anchor; don't oversell Max/GeACT. The fullest realistic model (COSI_Balloon) is where it FAILS — limiting factor is background contamination, not geometry (see [[dataset-fitness-clean511]]). This realism unevenness is likely why advisor was measured, not wowed. Relates to [[betadecay-classifier-state]], [[energy-specificity-vs-generic-geometry]].
