"""Sphinx configuration for the COSI β⁺ classifier documentation.

autodoc imports the project's modules to read their docstrings, so this file
has to reproduce the import environment the code actually runs in:

* ``src/BEvAn/`` is the import root (modules import each other
  flatly, e.g. ``from utils.megalib_types import MSimEvent``), exactly as
  ``[tool.pytest.ini_options].pythonpath`` declares — so it goes on sys.path.
* ``ablations/`` is on sys.path too: the ablation modules import each other as
  top-level names (``import harness``), the way ``ablations/main.py`` runs them.
* ``tests/`` is on sys.path so the suite's helpers (``fake_megalib``) resolve,
  and the repo root so the mirrored test tree is importable as the implicit
  namespace packages ``tests.BEvAn.*`` / ``tests.ablations.*`` that
  ``sphinx-apidoc --implicit-namespaces`` names them.
* ROOT / MEGAlib is not importable everywhere docs are built (e.g. this Mac),
  yet several modules do ``import ROOT as M`` at import time. It is mocked so
  the build never depends on a MEGAlib install.
"""

import os
import sys

# src/BEvAn/ is the package root (see pyproject pytest pythonpath); the rest
# mirror the sys.path tweaks ablations/main.py and tests/conftest.py make.
sys.path.insert(0, os.path.abspath("../../src/BEvAn"))
sys.path.insert(0, os.path.abspath("../../ablations"))
sys.path.insert(0, os.path.abspath("../../tests"))
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "BEvAn: β⁺ Decay Event Analyzer"
copyright = "2026, The Compton Spectrometer and Imager"
author = "Arya Raeesi"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",  # pull docstrings from the modules
    "sphinx.ext.napoleon",  # parse Google-style Args:/Returns: docstrings
    "sphinx.ext.autosummary",  # summary tables of modules/members
    "sphinx.ext.viewcode",  # "[source]" links to highlighted source
    "sphinx.ext.intersphinx",  # cross-link to numpy/torch/python docs
    "myst_parser",  # author pages in Markdown (MyST)
]

# Modules that aren't installable in every doc-build environment. autodoc
# replaces these with stand-ins so importing the code never fails the build.
autodoc_mock_imports = ["ROOT"]

# ablations/harness.py re-exports pipeline symbols (Trainer, Evaluator, ...) via
# __all__ for its sibling modules. Honouring __all__ would document them a second
# time under ablations.harness and make every :class:`Trainer` reference
# ambiguous, so members come from what each module actually defines instead.
autodoc_default_options = {"ignore-module-all": True}

autosummary_generate = True
autodoc_typehints = "description"  # render type hints in the body, not signatures
autodoc_member_order = "bysource"

napoleon_google_docstring = True
napoleon_numpy_docstring = False
# Render Attributes: sections as :ivar: field lists rather than standalone
# .. attribute:: directives, so they don't register duplicate cross-ref targets
# that collide with autodoc's own documentation of the same attributes.
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# MyST: allow Markdown and reST side by side.
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]

templates_path = ["_templates"]
exclude_patterns = []

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_static_path = ["_static"]
html_title = "BEvAn API Documentation"
