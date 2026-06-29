# COSI β⁺ Classifier — "Tagging the Positron Sky"

A physics-based likelihood classifier that tags each simulated COSI event as a
β⁺ / positron-annihilation signal (back-to-back 511 keV photons) or background,
using the event's Compton kinematics rather than a black-box model.

The model is an **interpretable, physics-based naive-Bayes likelihood-ratio
classifier** — per-bucket histogram densities over physics features
(Δ-energy, 511-aware ARM, annihilation angle), routed by hit multiplicity.

```{toctree}
:maxdepth: 2
:caption: Contents

api/modules
```

## Pipeline at a glance

The pipeline is a cascade:

> read `.sim`/`.tra` → features → fit per-bucket densities → likelihood ratio →
> decision → eval

| Package | Responsibility |
| --- | --- |
| {mod}`physics` | Compton-geometry features, event processing, 511 keV ground truth |
| {mod}`modeling` | Histogram densities, likelihood ratio, Bayesian decision, metrics |
| {mod}`pipeline` | Datasets, train/eval orchestration, bin-size + threshold selection |
| {mod}`utils` | `.sim`/`.tra` readers, plots, W&B, local CSV results |

## Indices

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
