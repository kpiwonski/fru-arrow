# Minimal Sphinx Makefile

SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help clean html apidoc live

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

clean:
	rm -rf "$(BUILDDIR)"

html:
	$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

# Generate API docs (adjust package path!)
apidoc:
	sphinx-apidoc --separate -o "$(SOURCEDIR)/api" ../pyfru

# Auto-rebuild (requires: pip install sphinx-autobuild)
live:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html"
