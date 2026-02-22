# malo

`malo` provides legacy 1D non-linear regression and model-averaging APIs
extracted from `gflow`.

## Installation

```bash
R -q -e 'remotes::install_local("malo", dependencies=TRUE, upgrade="never")'
```

## OpenMP Requirement

`malo` default install profile (`cran-safe`) is portable and does not require
OpenMP. If OpenMP is available in the active R toolchain, it will still be used.

For performance-critical workflows that must fail fast when OpenMP is missing,
install with `dev` profile:

```bash
R -q -e 'Sys.setenv(MALO_BUILD_PROFILE="dev"); remotes::install_local("malo", dependencies=TRUE, upgrade="never")'
```

Detailed OS-specific setup instructions are in `INSTALL.md`.

- Linux: GCC-based toolchains usually work out of the box.
- macOS: users must configure an OpenMP-capable toolchain (for example Homebrew GCC
  or LLVM + `libomp`).
- Windows: users need OpenMP-enabled Rtools toolchain setup.

If OpenMP is not configured, installation in `dev` profile will fail with a clear
error message.
