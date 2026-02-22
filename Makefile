.PHONY: clean build build-verbose check check-fast install document attrs
VERSION := $(shell grep "^Version:" DESCRIPTION | sed 's/Version: //')
PKGNAME := $(shell grep "^Package:" DESCRIPTION | sed 's/Package: //')
TARBALL := $(PKGNAME)_$(VERSION).tar.gz
LOGDIR := .claude
HOMEBREW_BIN := /opt/homebrew/bin

clean:
	find src -name "*.o" -delete 2>/dev/null || true
	find src -name "*.so" -delete 2>/dev/null || true
	rm -f src/*.dll src/RcppExports.cpp
	rm -rf $(PKGNAME).Rcheck
	rm -f $(TARBALL)
	rm -f $(LOGDIR)/*.log

attrs:
	@mkdir -p $(LOGDIR)
	@echo "Running Rcpp::compileAttributes() when applicable..."
	@if grep -q "Rcpp" DESCRIPTION; then \
		R -q -e "Rcpp::compileAttributes()" > $(LOGDIR)/$(PKGNAME)_rcppattrs.log 2>&1; \
		echo "RcppExports regenerated (log: $(LOGDIR)/$(PKGNAME)_rcppattrs.log)"; \
	else \
		echo "No Rcpp usage detected; skipping compileAttributes()." | tee $(LOGDIR)/$(PKGNAME)_rcppattrs.log; \
	fi

document: attrs
	@mkdir -p $(LOGDIR)
	@echo "Running devtools::document()..."
	@R -q -e "devtools::document()" > $(LOGDIR)/$(PKGNAME)_document.log 2>&1
	@echo "Documentation generated (log: $(LOGDIR)/$(PKGNAME)_document.log)"

build: clean document
	@mkdir -p $(LOGDIR)
	@echo "Building package..."
	@cd .. && R CMD build $(PKGNAME) > $(PKGNAME)/$(LOGDIR)/$(PKGNAME)_build.log 2>&1
	@echo "Package built successfully (log: $(LOGDIR)/$(PKGNAME)_build.log)"

build-verbose: clean document
	cd .. && R CMD build $(PKGNAME)

build-log: clean document
	@mkdir -p $(LOGDIR)
	cd .. && R CMD build $(PKGNAME) > $(PKGNAME)/$(LOGDIR)/$(PKGNAME)_build.log 2>&1
	@echo "Build output saved to $(LOGDIR)/$(PKGNAME)_build.log"

check: build
	cd .. && PATH="$(HOMEBREW_BIN):$$PATH" R_TIDYCMD="$(HOMEBREW_BIN)/tidy" R CMD check $(TARBALL) --as-cran

check-fast: build
	cd .. && PATH="$(HOMEBREW_BIN):$$PATH" R_TIDYCMD="$(HOMEBREW_BIN)/tidy" R CMD check $(TARBALL) --as-cran --no-examples --no-tests --no-manual

install: build
	cd .. && R CMD INSTALL $(TARBALL)
