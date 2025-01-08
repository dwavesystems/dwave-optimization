# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = dwave
SOURCEDIR     = .
BUILDDIR      = build

DOXYGEN_OUTPUT_DIRECTORY=$(BUILDDIR)/doxygen/ 

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

clean:
	-rm -rf $(BUILDDIR)
	-rm -rf reference/generated

cpp:
	# Note that sphinx's conf.py expects the build directory to be build/
	# and unforuntately there isn't an easy way to read the BUILDDIR from the
	# conf.py.
	# We could do additional env variables, but this is mostly for the benefit
	# of the SDK, so we'll leave it be for now.
	mkdir $(BUILDDIR) -p
	DOXYGEN_OUTPUT_DIRECTORY=$(DOXYGEN_OUTPUT_DIRECTORY) doxygen Doxyfile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
