# GEOMEGA — geometry & detector description

Program: `geomega`. Library: `libGeomega`. Classes prefixed `MD`. Source:
`src/geomega/{inc,src}`. Defines the instrument model that every other module loads.
For the `.geo.setup` *syntax*, see `file-formats.md §1`; this file is the conceptual model
and the program usage.

> **For this repo specifically**, `references/cosima-geomega.md` documents the actual
> geometry the data was simulated on (`Max.geo.setup` — a 4-layer Strip3D germanium stack)
> and the **numeric detector-type IDs** (verified from `MDDetector.cxx`; `3 = Strip3D` is
> this dataset). Read it for the dataset's geometry and the HTsim detector-type table.

## Object model

- **`MDGeometry`** (`MDGeometry.cxx`) — the master class. Holds the volume tree
  (`MDVolume`), materials (`MDMaterial`), detectors (`MDDetector`), shapes (`MDShape`),
  triggers (`MDTriggerUnit`). `ScanSetupFile()` parses `.geo.setup` (handles `Include`,
  `Constant`, `{}` math, GDML import). Builds a ROOT `TGeoManager` for visualization/queries.
- **`MDGeometryQuest`** — the *queryable* geometry used at analysis time: "what
  volume/material is at point P?", "is P sensitive?", noising of hits, resolution lookups,
  absorption probabilities, Compton intersection geometry. Modules subclass it
  (`MGeometryRevan` in revan).
- **`MDVolume`** — node in the hierarchy: shape + material + position/rotation + mother +
  optional detector assignment + visualization. Supports cloned placements.
- **`MDMaterial`** (+ `MDMaterialComponent`) — composition, density, radiation length,
  cross sections (photo/Compton/pair/Rayleigh). Cross-section files can be generated.
- **`MDShape`** subclasses → ROOT `TGeo*`: `MDShapeBRIK` (box), `TUBS` (tube), `SPHE`,
  `CONE`/`CONS`, `TRAP`, `TRD1/TRD2`, `GTRA`, `PCON`, `PGON`, plus boolean
  `MDShapeUnion`/`Subtraction`/`Intersection`.

## Detector types (`MDDetector` subclasses)

| Keyword / class | Use |
|-----------------|-----|
| `MDStrip2D` | 2D silicon strip (cross-strip) |
| `MDStrip3D` | strip with depth (3D position) — e.g. COSI Ge |
| `MDStrip3DDirectional` | directional/tracking strip |
| `MDCalorimeter` | bar/voxel calorimeter |
| `MDVoxel3D` | 3D voxel detector |
| `MDAngerCamera` | position-sensitive scintillator (Anger logic) |
| `MDDriftChamber` | gas drift chamber / TPC |
| `MDScintillator` | simple scintillator |
| `MDACS` | anticoincidence shield (veto) |

Each detector defines how truth deposits become measured hits: `Discretize()` (snap to
strip/voxel grid), noising via energy/depth/time **resolution** curves, **noise** and
**trigger thresholds**, failure/overflow handling, optional guard rings (`MDGuardRing`).
Properties are set in the `.geo.setup` (`file-formats.md §1`); the authoritative property
list per type is the matching `src/geomega/inc/MD<Type>.h`.

## Triggers

`MDTriggerUnit` aggregates `MDTrigger` rules. A rule is either a **trigger** (`Veto false`)
or a **veto** (`Veto true`), and fires `TriggerByDetector`/`TriggerByChannel` with a
minimum hit multiplicity per named detector / detector type / guard ring. `MDTriggerBasic`
(threshold) and `MDTriggerMap` (spatial map) are the concrete forms. Triggering decides
which simulated events are written and how measured events qualify.

## Running geomega

```bash
geomega -g Detector.geo.setup            # open GUI, then View/Geometry to render
geomega -g Detector.geo.setup -a         # auto-display on startup
geomega -g Detector.geo.setup -s Wafer   # treat volume 'Wafer' as the world (zoom in)
geomega -g Detector.geo.setup -n --create-cross-sections   # batch: generate xsection files
```

Common needs (confirm flags with `geomega --help` / `Usage()` in
`src/geomega/src/MInterfaceGeomega.cxx`):
- **Render / inspect** the geometry (GUI; or `-a`).
- **Check for overlaps** — geomega validates the tree and reports overlapping/protruding
  volumes; `.source`/cosima also has `CheckForOverlaps`. Overlaps are the #1 geometry bug.
- **Generate cross-section files** (`--create-cross-sections`) needed by cosima/revan.
- **GDML import/export** is supported via `MDGeometry` (CAD interchange).

## Authoring tips

- The top volume must be `WorldVolume` with `.Mother 0`; everything nests inside it.
- Coordinates/extents are in **cm**; `BRIK`/`BOX` take **half-extents**.
- Build incrementally and re-open in geomega to catch overlaps early; use
  `.Visibility`/`.Color` to debug placement.
- Resolutions are given as one line per calibration energy
  (`EnergyResolution Gaus Ein Epeak sigma`); MEGAlib interpolates between them.
- Reuse the included materials file (`resource/examples/geomega/materials/Materials.geo`)
  rather than redefining standard materials.
- Reference geometries to learn from: `resource/examples/geomega/cosiballoon/` (COSI),
  `simplifiedprototype/`, `detailedprototype/`, `GRI/`, `GRIPS/`, `mpesatellitebaseline/`.
