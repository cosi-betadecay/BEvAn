# MIMREC — image reconstruction & high-level analysis

Program: `mimrec`. Library: `libMimrec`. Classes prefixed `MI`/`MImage`. Source:
`src/mimrec/{inc,src}`. Input: `.tra` (reconstructed events from revan). The scientific
endpoint: sky images plus spectra, ARM, polarization, sensitivity, light curves.

## Image reconstruction (the core)

MEGAlib does **list-mode maximum-likelihood Compton image reconstruction** — each event is
treated individually (its Compton cone / response is back-projected), and the image is found
by iterative expectation-maximization deconvolution.

- **`MImager`** — orchestrator (coordinate system, response, threading, memory).
- **`MLMLAlgorithms`** base → **`MLMLClassicEM`** (Classic EM) and **`MLMLOSEM`**
  (Ordered-Subset EM — faster convergence; pick number of subsets). Each iteration does a
  forward projection (convolve current image through the response → predicted counts) then a
  backward projection (deconvolve observed/predicted correction).
- **Backprojection**: `MBackprojectionFarField` (astrophysics: spherical / galactic at
  effectively infinite distance) vs `MBackprojectionNearField` (2D/3D Cartesian, nearby
  sources / lab / medical). Per-event response slices stored as `MBPData*`
  (dense/sparse, optional 1-byte compression) to bound memory.
- **Response models** (how each event's PSF is computed): `MResponseGaussian`
  (analytic ARM/SPD widths — fast, default), `MResponseGaussianByUncertainties`
  (per-event uncertainties), `MResponseConeShapes` (detailed shapes from simulation),
  `MResponsePRM` (parametrized matrices). The Gaussian model needs no `.rsp` file; detailed
  models do.

## Coordinate systems

`MCoordinateSystem`: `Galactic`, `Spheric`, `Cartesian2D`, `Cartesian3D`. Far-field images
are displayed in galactic or spherical (Hammer projection via `MImageSpheric`/
`MImageGalactic`); near-field in Cartesian. Set in the `.mimrec.cfg`.

## High-level analysis tools (also on the CLI / GUI)

- **ARM (Angular Resolution Measure)** — `MARMFitter`: the signature Compton-telescope
  resolution metric; fits the ARM distribution (Gaussian / Lorentz / generalized-normal /
  composite families), reports FWHM + containment, bootstrap uncertainties.
- **Spectra** — energy spectra of selected events.
- **Polarization** — modulation curve / polarization fraction & angle.
- **Sensitivity / exposure / effective area** — `MExposure`, `MEfficiency` (needs an
  efficiency response).
- **Light curves**, scatter-angle distributions, Compton sequence length, interaction
  distances, location of first interaction.
- **Event selection** — `MEventSelector`: filter by energy windows, time, detector, ARM
  window, earth-horizon, quality flags, sequence length, etc. The same selector concept is
  shared with revan/response.

## Running mimrec

The `.mimrec.cfg` stores geometry, data file, algorithm, response, and event selections
(easiest to create via the GUI). Confirm flags in
`src/mimrec/src/MInterfaceMimrec.cxx::Usage()`.

```bash
# GUI
mimrec -c CrabOnly.mimrec.cfg

# Batch: reconstruct an image headless
mimrec -g Det.geo.setup -c CrabOnly.mimrec.cfg -f Crab.tra -n -i -o Image.root

# Other batch products
mimrec -c X.mimrec.cfg -f Crab.tra -n -s          # energy spectrum
mimrec -c X.mimrec.cfg -f Crab.tra -n -a          # ARM (gamma)
mimrec -c X.mimrec.cfg -f Crab.tra -n -p          # polarization
mimrec -c X.mimrec.cfg -f Crab.tra -n -l          # light curve
mimrec -c X.mimrec.cfg -f Crab.tra -n -x -o Sel.tra   # extract selected events to a new .tra
mimrec -c X.mimrec.cfg -f Crab.tra -n -e          # list available event selections

# Override a config value inline (dotted path, repeatable)
mimrec -c X.mimrec.cfg -f Crab.tra -n -a -C TestPositions.DistanceTrans=10
```

Key options (verify): `-g`/`-f`/`-c`/`-n`/`-o`/`-C` as in revan, plus the batch-analysis
selectors `-i` image, `-s` spectrum, `-a` ARM, `-p` polarization, `-l` light curve,
`-x` extract events, `-e` list selections, `--standard-analysis-spherical E theta phi`.

## Notes

- **Iterations matter**: too few = blurry/under-deconvolved, too many = noise amplification.
  OSEM with a few subsets reaches a given quality in fewer iterations than Classic EM.
- The Gaussian response is the quick path; switch to ConeShapes / matrix responses (built by
  `responsecreator`, `sivan-response.md`) for quantitative work — they must match the
  geometry used for the data.
- Tight **event selection** (ARM window, energy, quality) cleans images dramatically; start
  from an example `.mimrec.cfg` and adjust.
- For sensitivity/effective-area studies see `resource/examples/mimrec/EffectiveArea.mimrec.cfg`
  and `resource/examples/advanced/{AllSky,BinnedImaging,InducedPolarization,LightCurves}`.
