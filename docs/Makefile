# Minimal makefile for Sphinx documentation.
# Run from the docs/ directory, e.g. `make html`. Use uv so the docs
# dependency group is on the path: `uv run make html` from repo root,
# or activate the venv first.

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all: route unknown targets (html, clean, …) to sphinx-build.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
