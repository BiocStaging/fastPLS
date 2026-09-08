#ifndef FASTPLS_R_API_H
#define FASTPLS_R_API_H

#ifndef R_NO_REMAP
#define R_NO_REMAP
#endif
#include <Rinternals.h>

extern "C" {
SEXP _fastPLS_has_cuda();
SEXP _fastPLS_has_metal();
SEXP _fastPLS_lda_cuda_native_available();
SEXP _fastPLS_rsvd_audit_reset_debug();
SEXP _fastPLS_rsvd_audit_summary_debug();
SEXP _fastPLS_spearman_correlation_cpp(SEXP observed, SEXP predicted);
SEXP _fastPLS_float32_argmax_cpp(SEXP scores);
SEXP _fastPLS_float32_topk_cpp(SEXP scores, SEXP top);
SEXP _fastPLS_float32_sweep_cols_cpp(SEXP matrix, SEXP statistics,
                                     SEXP operation);
SEXP _fastPLS_float32_standardize_cpp(SEXP matrix, SEXP center, SEXP scale);
SEXP _fastPLS_center_kernel_train_float32_cpp(SEXP kernel);
SEXP _fastPLS_center_kernel_test_float32_cpp(SEXP kernel,
                                             SEXP training_means,
                                             SEXP training_grand_mean);
SEXP _fastPLS_center_kernel_train_cpp(SEXP kernel);
SEXP _fastPLS_center_kernel_test_cpp(SEXP kernel, SEXP training_means,
                                     SEXP training_grand_mean);
SEXP _fastPLS_label_crossprod_scaled_cpp(SEXP predictors, SEXP labels,
                                         SEXP class_count, SEXP scaling);
}

#endif
