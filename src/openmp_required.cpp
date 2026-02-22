#if !defined(_OPENMP) && !defined(MALO_ALLOW_NO_OPENMP)
#error "malo dev profile requires OpenMP. Configure an OpenMP-enabled toolchain or set MALO_BUILD_PROFILE=cran-safe."
#endif

extern "C" void malo_openmp_requirement_sentinel(void) {}
