// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore

#ifdef FASTPLS_CORE_ONLY

#include "r_api.h"

#include <R_ext/Rdynload.h>

#define FASTPLS_CALL(name, count) \
  {#name, reinterpret_cast<DL_FUNC>(&name), count}

static const R_CallMethodDef call_entries[] = {
  FASTPLS_CALL(_fastPLS_cuda_matrix_multiply, 2),
  FASTPLS_CALL(_fastPLS_cuda_resident_project_cpp, 3),
  FASTPLS_CALL(_fastPLS_cuda_resident_response_sums_cpp, 5),
  FASTPLS_CALL(_fastPLS_cuda_resident_simpls_fit_cpp, 17),
  FASTPLS_CALL(_fastPLS_cuda_resident_simpls_cv_classification_cpp, 13),
  FASTPLS_CALL(_fastPLS_cuda_resident_simpls_cv_regression_cpp, 11),
  FASTPLS_CALL(_fastPLS_cuda_resident_export_cpp, 4),
  FASTPLS_CALL(_fastPLS_cuda_resident_compact_cpp, 2),
  FASTPLS_CALL(_fastPLS_cuda_resident_classify_path_cpp, 5),
  FASTPLS_CALL(_fastPLS_cuda_resident_classify_response_path_cpp, 5),
  FASTPLS_CALL(_fastPLS_cuda_resident_predict_path_cpp, 4),
  FASTPLS_CALL(_fastPLS_has_cuda, 0),
  FASTPLS_CALL(_fastPLS_has_metal, 0),
  FASTPLS_CALL(_fastPLS_simpls_cache_predictor_crossprod, 3),
  FASTPLS_CALL(_fastPLS_set_cpu_threads, 1),
  FASTPLS_CALL(_fastPLS_rsvd_audit_reset_debug, 0),
  FASTPLS_CALL(_fastPLS_rsvd_audit_summary_debug, 0),
  FASTPLS_CALL(_fastPLS_fastsvd_core_cpp, 6),
  FASTPLS_CALL(_fastPLS_fastsvd_float32_core_cpp, 6),
  FASTPLS_CALL(_fastPLS_cv_folds_core_cpp, 4),
  FASTPLS_CALL(_fastPLS_pls_cv_classification_core_cpp, 13),
  FASTPLS_CALL(_fastPLS_pls_cv_classification_float32_core_cpp, 13),
  FASTPLS_CALL(_fastPLS_pls_cv_classification_float32_metal_core_cpp, 18),
  FASTPLS_CALL(_fastPLS_pls_cv_opls_classification_core_cpp, 13),
  FASTPLS_CALL(_fastPLS_pls_cv_opls_classification_float32_core_cpp, 13),
  FASTPLS_CALL(_fastPLS_pls_cv_kernel_classification_core_cpp, 16),
  FASTPLS_CALL(_fastPLS_pls_cv_kernel_classification_float32_core_cpp, 16),
  FASTPLS_CALL(_fastPLS_pls_cv_regression_core_cpp, 11),
  FASTPLS_CALL(_fastPLS_pls_cv_regression_float32_core_cpp, 11),
  FASTPLS_CALL(_fastPLS_pls_cv_regression_float32_metal_core_cpp, 16),
  FASTPLS_CALL(_fastPLS_pls_cv_opls_regression_core_cpp, 11),
  FASTPLS_CALL(_fastPLS_pls_cv_opls_regression_float32_core_cpp, 11),
  FASTPLS_CALL(_fastPLS_pls_cv_kernel_regression_core_cpp, 14),
  FASTPLS_CALL(_fastPLS_pls_cv_kernel_regression_float32_core_cpp, 14),
  FASTPLS_CALL(_fastPLS_lda_train_prefix_cpp, 5),
  FASTPLS_CALL(_fastPLS_lda_train_moments_prefix_cpp, 5),
  FASTPLS_CALL(_fastPLS_lda_project_train_prefix_cpp, 7),
  FASTPLS_CALL(_fastPLS_lda_predict_cpp, 2),
  FASTPLS_CALL(_fastPLS_lda_predict_labels_cpp, 2),
  FASTPLS_CALL(_fastPLS_lda_project_predict_labels_cpp, 4),
  FASTPLS_CALL(_fastPLS_spearman_correlation_cpp, 2),
  FASTPLS_CALL(_fastPLS_evaluate_regression_core_cpp, 5),
  FASTPLS_CALL(_fastPLS_evaluate_regression_by_column_cpp, 5),
  FASTPLS_CALL(_fastPLS_evaluate_classification_core_cpp, 6),
  FASTPLS_CALL(_fastPLS_evaluate_is_onehot_cpp, 1),
  FASTPLS_CALL(_fastPLS_evaluate_class_labels_cpp, 2),
  FASTPLS_CALL(_fastPLS_vip_core_cpp, 1),
  FASTPLS_CALL(_fastPLS_fastcor_core_cpp, 4),
  FASTPLS_CALL(_fastPLS_float32_argmax_cpp, 1),
  FASTPLS_CALL(_fastPLS_float32_topk_cpp, 2),
  FASTPLS_CALL(_fastPLS_lda_train_prefix_float32_cpp, 4),
  FASTPLS_CALL(_fastPLS_lda_predict_float32_cpp, 3),
  FASTPLS_CALL(_fastPLS_float32_sweep_cols_cpp, 3),
  FASTPLS_CALL(_fastPLS_float32_standardize_cpp, 3),
  FASTPLS_CALL(_fastPLS_center_kernel_train_float32_cpp, 1),
  FASTPLS_CALL(_fastPLS_center_kernel_test_float32_cpp, 3),
  FASTPLS_CALL(_fastPLS_center_kernel_train_cpp, 1),
  FASTPLS_CALL(_fastPLS_cpu_float32_matrix_multiply_cpp, 4),
  FASTPLS_CALL(_fastPLS_metal_float32_matrix_multiply_cpp, 4),
  FASTPLS_CALL(_fastPLS_kernel_matrix_float32_cpp, 7),
  FASTPLS_CALL(_fastPLS_opls_apply_filter_float32_cpp, 6),
  FASTPLS_CALL(_fastPLS_opls_apply_filter_cpp, 5),
  FASTPLS_CALL(_fastPLS_kernel_matrix_cpp, 6),
  FASTPLS_CALL(_fastPLS_center_kernel_test_cpp, 3),
  FASTPLS_CALL(_fastPLS_opls_filter_core_cpp, 4),
  FASTPLS_CALL(_fastPLS_opls_filter_labels_core_cpp, 5),
  FASTPLS_CALL(_fastPLS_opls_filter_float32_core_cpp, 7),
  FASTPLS_CALL(_fastPLS_opls_filter_float32_labels_core_cpp, 5),
  FASTPLS_CALL(_fastPLS_opls_filter_float32_backend_core_cpp, 8),
  FASTPLS_CALL(_fastPLS_opls_filter_float32_labels_backend_core_cpp, 9),
  FASTPLS_CALL(_fastPLS_pls_labels_core_cpp, 10),
  FASTPLS_CALL(_fastPLS_pls_simpls_labels_core_cpp, 10),
  FASTPLS_CALL(_fastPLS_pls_matrix_core_cpp, 10),
  FASTPLS_CALL(_fastPLS_pls_matrix_core_xprod_cpp, 10),
  FASTPLS_CALL(_fastPLS_pls_float32_matrix_backend_core_cpp, 11),
  FASTPLS_CALL(_fastPLS_pls_labels_core_predict_cpp, 3),
  FASTPLS_CALL(_fastPLS_pls_class_predict_topk_core_cpp, 5),
  FASTPLS_CALL(_fastPLS_pls_float32_class_predict_compact_cpp, 4),
  FASTPLS_CALL(_fastPLS_lda_project_train_prefix_float32_cpp, 5),
  FASTPLS_CALL(_fastPLS_pls_float32_labels_backend_core_cpp, 12),
  {nullptr, nullptr, 0}
};

extern "C" void R_init_fastPLS(DllInfo* dll) {
  R_registerRoutines(dll, nullptr, call_entries, nullptr, nullptr);
  R_useDynamicSymbols(dll, FALSE);
}

#undef FASTPLS_CALL

#endif
