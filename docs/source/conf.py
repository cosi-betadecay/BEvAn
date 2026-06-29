"""Sphinx configuration for the COSI β⁺ classifier documentation.

autodoc imports the project's modules to read their docstrings, so this file
has to reproduce the import environment the code actually runs in:

* ``src/betadecay-analysis/`` is the import root (modules import each other
  flatly, e.g. ``from utils.megalib_types import MSimEvent``), exactly as
  ``[tool.pytest.ini_options].pythonpath`` declares — so it goes on sys.path.
* ROOT / MEGAlib is not importable everywhere docs are built (e.g. this Mac),
  yet several modules do ``import ROOT as M`` at import time. It is mocked so
  the build never depends on a MEGAlib install.
"""

import os
import sys

# src/betadecay-analysis/ is the package root (see pyproject pytest pythonpath).
sys.path.insert(0, os.path.abspath("../../src/betadecay-analysis"))

# -- Project information -----------------------------------------------------

project = "COSI β⁺ Classifier"
copyright = "2026, COSI betadecay-analysis"
author = "COSI betadecay-analysis"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",      # pull docstrings from the modules
    "sphinx.ext.napoleon",     # parse Google-style Args:/Returns: docstrings
    "sphinx.ext.autosummary",  # summary tables of modules/members
    "sphinx.ext.viewcode",     # "[source]" links to highlighted source
    "sphinx.ext.intersphinx",  # cross-link to numpy/torch/python docs
    "myst_parser",             # author pages in Markdown (MyST)
]

# Modules that aren't installable in every doc-build environment. autodoc
# replaces these with stand-ins so importing the code never fails the build.
autodoc_mock_imports = ["ROOT"]

autosummary_generate = True
autodoc_typehints = "description"  # render type hints in the body, not signatures
autodoc_member_order = "bysource"

napoleon_google_docstring = True
napoleon_numpy_docstring = False

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
html_title = "COSI β⁺ Classifier"
