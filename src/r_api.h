#ifndef FASTPLS_R_API_H
#define FASTPLS_R_API_H

#ifndef R_NO_REMAP
#define R_NO_REMAP
#endif
#include <Rinternals.h>

extern "C" {
SEXP _fastPLS_has_cuda();
SEXP _fastPLS_has_metal();
SEXP _fastPLS_spearman_correlation_cpp(SEXP observed, SEXP predicted);
}

#endif
