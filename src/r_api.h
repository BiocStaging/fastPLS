#ifndef FASTPLS_R_API_H
#define FASTPLS_R_API_H

#ifndef R_NO_REMAP
#define R_NO_REMAP
#endif
#include <Rinternals.h>

extern "C" {
SEXP _fastPLS_cuda_resident_project_cpp(SEXP object, SEXP predictors,
                                        SEXP components);
SEXP _fastPLS_cuda_resident_response_sums_cpp(SEXP object, SEXP predictors,
                                              SEXP response, SEXP labels,
                                              SEXP components);
SEXP _fastPLS_cuda_resident_simpls_fit_cpp(
  SEXP predictors, SEXP response, SEXP labels, SEXP classes, SEXP precision,
  SEXP components, SEXP scaling, SEXP oversample, SEXP power, SEXP seed,
  SEXP retain_scores, SEXP method, SEXP north, SEXP kernel, SEXP gamma,
  SEXP degree, SEXP coefficient
);
SEXP _fastPLS_cuda_resident_export_cpp(SEXP object, SEXP loadings,
                                       SEXP variance, SEXP scores);
SEXP _fastPLS_cuda_resident_compact_cpp(SEXP object, SEXP prepare_lda);
SEXP _fastPLS_cuda_resident_classify_cpp(SEXP object, SEXP predictors,
                                         SEXP components, SEXP classifier,
                                         SEXP top);
SEXP _fastPLS_cuda_resident_classify_path_cpp(
  SEXP object, SEXP predictors, SEXP components, SEXP classifier, SEXP top
);
SEXP _fastPLS_cuda_resident_classify_response_path_cpp(
  SEXP object, SEXP predictors, SEXP components, SEXP classifier, SEXP top
);
SEXP _fastPLS_cuda_resident_simpls_predict_cpp(
  SEXP object, SEXP predictors, SEXP components, SEXP classifier
);
SEXP _fastPLS_cuda_resident_predict_path_cpp(
  SEXP object, SEXP predictors, SEXP components, SEXP classifier
);
SEXP _fastPLS_cuda_matrix_multiply(SEXP left, SEXP right);
SEXP _fastPLS_has_cuda();
SEXP _fastPLS_has_metal();
SEXP _fastPLS_cpu_backend_description();
SEXP _fastPLS_lda_cuda_native_available();
SEXP _fastPLS_rsvd_audit_reset_debug();
SEXP _fastPLS_rsvd_audit_summary_debug();
SEXP _fastPLS_lda_train_prefix_cpp(SEXP scores, SEXP labels,
                                    SEXP class_count, SEXP components,
                                    SEXP ridge);
SEXP _fastPLS_lda_train_moments_prefix_cpp(
  SEXP gram, SEXP class_sums, SEXP counts, SEXP sample_count,
  SEXP components
);
SEXP _fastPLS_lda_project_train_prefix_cpp(
  SEXP predictors, SEXP projection, SEXP offset, SEXP labels,
  SEXP class_count, SEXP components, SEXP ridge
);
SEXP _fastPLS_lda_predict_cpp(SEXP scores, SEXP model);
SEXP _fastPLS_lda_predict_labels_cpp(SEXP scores, SEXP model);
SEXP _fastPLS_lda_project_predict_labels_cpp(
  SEXP predictors, SEXP projection, SEXP offset, SEXP model
);
SEXP _fastPLS_spearman_correlation_cpp(SEXP observed, SEXP predicted);
SEXP _fastPLS_float32_argmax_cpp(SEXP scores);
SEXP _fastPLS_float32_topk_cpp(SEXP scores, SEXP top);
SEXP _fastPLS_float32_sweep_cols_cpp(SEXP matrix, SEXP statistics,
                                     SEXP operation);
SEXP _fastPLS_float32_standardize_cpp(SEXP matrix, SEXP center, SEXP scale);
SEXP _fastPLS_cpu_float32_matrix_multiply_cpp(
  SEXP left, SEXP right, SEXP transpose_left, SEXP transpose_right
);
SEXP _fastPLS_metal_float32_matrix_multiply_cpp(
  SEXP left, SEXP right, SEXP transpose_left, SEXP transpose_right
);
SEXP _fastPLS_kernel_matrix_float32_cpp(
  SEXP left, SEXP right, SEXP kernel, SEXP gamma, SEXP degree,
  SEXP offset, SEXP backend
);
SEXP _fastPLS_opls_apply_filter_float32_cpp(
  SEXP matrix, SEXP center, SEXP scale, SEXP weights, SEXP loadings,
  SEXP backend
);
SEXP _fastPLS_opls_apply_filter_cpp(
  SEXP matrix, SEXP center, SEXP scale, SEXP weights, SEXP loadings
);
SEXP _fastPLS_lda_train_prefix_float32_cpp(SEXP scores, SEXP labels,
                                            SEXP class_count,
                                            SEXP components);
SEXP _fastPLS_lda_predict_float32_cpp(SEXP scores, SEXP model,
                                      SEXP return_scores);
SEXP _fastPLS_center_kernel_train_float32_cpp(SEXP kernel);
SEXP _fastPLS_center_kernel_test_float32_cpp(SEXP kernel,
                                             SEXP training_means,
                                             SEXP training_grand_mean);
SEXP _fastPLS_kernel_matrix_cpp(SEXP left, SEXP right, SEXP kernel,
                                SEXP gamma, SEXP degree, SEXP offset);
SEXP _fastPLS_center_kernel_train_cpp(SEXP kernel);
SEXP _fastPLS_center_kernel_test_cpp(SEXP kernel, SEXP training_means,
                                     SEXP training_grand_mean);
SEXP _fastPLS_label_crossprod_scaled_cpp(SEXP predictors, SEXP labels,
                                         SEXP class_count, SEXP scaling);
SEXP _fastPLS_pls_labels_core_cpp(
  SEXP predictors, SEXP labels, SEXP class_count, SEXP components,
  SEXP scaling, SEXP fit, SEXP oversample, SEXP power, SEXP seed
);
SEXP _fastPLS_pls_simpls_labels_core_cpp(
  SEXP predictors, SEXP labels, SEXP class_count, SEXP components,
  SEXP scaling, SEXP fit, SEXP oversample, SEXP power, SEXP seed
);
SEXP _fastPLS_pls_matrix_core_cpp(
  SEXP predictors, SEXP responses, SEXP components, SEXP scaling,
  SEXP fit, SEXP method, SEXP oversample, SEXP power, SEXP seed
);
SEXP _fastPLS_pls_matrix_core_xprod_cpp(
  SEXP predictors, SEXP responses, SEXP components, SEXP scaling,
  SEXP fit, SEXP method, SEXP oversample, SEXP power, SEXP seed
);
SEXP _fastPLS_pls_float32_matrix_core_cpp(
  SEXP predictors, SEXP responses, SEXP components, SEXP scaling,
  SEXP fit, SEXP method, SEXP oversample, SEXP power, SEXP seed
);
SEXP _fastPLS_pls_labels_core_predict_cpp(
  SEXP model, SEXP predictors, SEXP project
);
SEXP _fastPLS_pls_float32_labels_core_cpp(
  SEXP predictors, SEXP labels, SEXP class_count, SEXP components,
  SEXP scaling, SEXP fit, SEXP method, SEXP oversample, SEXP power, SEXP seed
);
}

#endif
