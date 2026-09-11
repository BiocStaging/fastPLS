# Hand-written R C-API entry points used by the dependency-free core.

set_cpu_threads_cpp <- function(threads) {
    .Call(
        "_fastPLS_set_cpu_threads", as.integer(threads), PACKAGE = "fastPLS"
    )
}

simpls_cache_predictor_crossprod_cpp <- function(
    samples, predictors, components
) {
    .Call(
        "_fastPLS_simpls_cache_predictor_crossprod",
        as.integer(samples),
        as.integer(predictors),
        as.integer(components),
        PACKAGE = "fastPLS"
    )
}

fastsvd_core_cpp <- function(
    matrix, components, oversample, power, seed, left_only = FALSE
) {
    .Call(
        "_fastPLS_fastsvd_core_cpp", matrix, components, oversample, power,
        seed, left_only, PACKAGE = "fastPLS"
    )
}

fastsvd_float32_core_cpp <- function(
    matrix, components, oversample, power, seed, left_only = FALSE
) {
    .Call(
        "_fastPLS_fastsvd_float32_core_cpp", matrix, components,
        oversample, power, seed, left_only, PACKAGE = "fastPLS"
    )
}

cv_folds_core_cpp <- function(
    groups, labels = NULL, class_count = 0L, folds = 10L
) {
    .Call(
        "_fastPLS_cv_folds_core_cpp", groups, labels, class_count, folds,
        PACKAGE = "fastPLS"
    )
}

pls_double_cv_core_cpp <- function(
    predictors, response, class_count, outer_folds, inner_folds, components,
    scaling, method, classifier_metric, selection_metric, north, kernel,
    gamma, degree, coef0, oversample, power, seed, backend, classification
) {
    .Call(
        "_fastPLS_pls_double_cv_core_cpp", predictors, response, class_count,
        outer_folds, inner_folds, components, scaling, method,
        classifier_metric, selection_metric, north, kernel, gamma, degree,
        coef0, oversample, power, seed, backend, classification,
        PACKAGE = "fastPLS"
    )
}

pls_cv_classification_core_cpp <- function(
    predictors, labels, class_count, folds, components, scaling, method,
    classifier, oversample, power, seed, store_predictions = TRUE,
    store_scores = TRUE
) {
    .Call(
        "_fastPLS_pls_cv_classification_core_cpp", predictors, labels,
        class_count, folds, components, scaling, method, classifier,
        oversample, power, seed, store_predictions, store_scores,
        PACKAGE = "fastPLS"
    )
}

pls_cv_classification_float32_core_cpp <- function(
    predictors, labels, class_count, folds, components, scaling, method,
    classifier, oversample, power, seed, store_predictions = TRUE,
    store_scores = TRUE
) {
    result <- .Call(
        "_fastPLS_pls_cv_classification_float32_core_cpp", predictors,
        labels, class_count, folds, components, scaling, method, classifier,
        oversample, power, seed, store_predictions, store_scores,
        PACKAGE = "fastPLS"
    )
    result
}

pls_cv_classification_float32_metal_core_cpp <- function(
    predictors, labels, class_count, folds, components, scaling, method,
    classifier, north, kernel, gamma, degree, coef0, oversample, power, seed,
    store_predictions = TRUE, store_scores = TRUE
) {
    .Call(
        "_fastPLS_pls_cv_classification_float32_metal_core_cpp",
        predictors, labels, class_count, folds, components, scaling, method,
        classifier, north, kernel, gamma, degree, coef0, oversample, power,
        seed, store_predictions, store_scores, PACKAGE = "fastPLS"
    )
}

cuda_resident_simpls_cv_classification_cpp <- function(
    predictors, labels, class_count, folds, components, scaling, classifier,
    oversample, power, seed, store_predictions = TRUE, store_scores = TRUE,
    method = 3L
) {
    .Call(
        "_fastPLS_cuda_resident_simpls_cv_classification_cpp",
        predictors, labels, class_count, folds, components, scaling,
        classifier, oversample, power, seed, store_predictions, store_scores,
        method,
        PACKAGE = "fastPLS"
    )
}

cuda_resident_simpls_cv_regression_cpp <- function(
    predictors, responses, folds, components, scaling, metric, oversample,
    power, seed, store_predictions = TRUE, method = 3L
) {
    .Call(
        "_fastPLS_cuda_resident_simpls_cv_regression_cpp",
        predictors, responses, folds, components, scaling, metric,
        oversample, power, seed, store_predictions, method,
        PACKAGE = "fastPLS"
    )
}

pls_cv_opls_classification_core_cpp <- function(
    predictors, labels, class_count, folds, components, scaling, classifier,
    north, oversample, power, seed, store_predictions = TRUE,
    store_scores = TRUE
) {
    .Call(
        "_fastPLS_pls_cv_opls_classification_core_cpp", predictors, labels,
        class_count, folds, components, scaling, classifier, north,
        oversample, power, seed, store_predictions, store_scores,
        PACKAGE = "fastPLS"
    )
}

pls_cv_opls_classification_float32_core_cpp <- function(
    predictors, labels, class_count, folds, components, scaling, classifier,
    north, oversample, power, seed, store_predictions = TRUE,
    store_scores = TRUE
) {
    .Call(
        "_fastPLS_pls_cv_opls_classification_float32_core_cpp", predictors,
        labels, class_count, folds, components, scaling, classifier, north,
        oversample, power, seed, store_predictions, store_scores,
        PACKAGE = "fastPLS"
    )
}

pls_cv_kernel_classification_core_cpp <- function(
    predictors, labels, class_count, folds, components, scaling, classifier,
    kernel, gamma, degree, coef0, oversample, power, seed,
    store_predictions = TRUE, store_scores = TRUE
) {
    .Call(
        "_fastPLS_pls_cv_kernel_classification_core_cpp", predictors, labels,
        class_count, folds, components, scaling, classifier, kernel, gamma,
        degree, coef0, oversample, power, seed, store_predictions,
        store_scores, PACKAGE = "fastPLS"
    )
}

pls_cv_kernel_classification_float32_core_cpp <- function(
    predictors, labels, class_count, folds, components, scaling, classifier,
    kernel, gamma, degree, coef0, oversample, power, seed,
    store_predictions = TRUE, store_scores = TRUE
) {
    .Call(
        "_fastPLS_pls_cv_kernel_classification_float32_core_cpp", predictors,
        labels, class_count, folds, components, scaling, classifier, kernel,
        gamma, degree, coef0, oversample, power, seed, store_predictions,
        store_scores, PACKAGE = "fastPLS"
    )
}

pls_cv_regression_core_cpp <- function(
    predictors, responses, folds, components, scaling, method, metric,
    oversample, power, seed, store_predictions = TRUE
) {
    .Call(
        "_fastPLS_pls_cv_regression_core_cpp", predictors, responses, folds,
        components, scaling, method, metric, oversample, power, seed,
        store_predictions, PACKAGE = "fastPLS"
    )
}

pls_cv_regression_float32_core_cpp <- function(
    predictors, responses, folds, components, scaling, method, metric,
    oversample, power, seed, store_predictions = TRUE
) {
    result <- .Call(
        "_fastPLS_pls_cv_regression_float32_core_cpp", predictors,
        responses, folds, components, scaling, method, metric, oversample,
        power, seed, store_predictions, PACKAGE = "fastPLS"
    )
    result
}

pls_cv_regression_float32_metal_core_cpp <- function(
    predictors, responses, folds, components, scaling, method, metric,
    north, kernel, gamma, degree, coef0, oversample, power, seed,
    store_predictions = TRUE
) {
    .Call(
        "_fastPLS_pls_cv_regression_float32_metal_core_cpp",
        predictors, responses, folds, components, scaling, method, metric,
        north, kernel, gamma, degree, coef0, oversample, power, seed,
        store_predictions, PACKAGE = "fastPLS"
    )
}

pls_cv_opls_regression_core_cpp <- function(
    predictors, responses, folds, components, scaling, metric, north,
    oversample, power, seed, store_predictions = TRUE
) {
    .Call(
        "_fastPLS_pls_cv_opls_regression_core_cpp", predictors, responses,
        folds, components, scaling, metric, north, oversample, power, seed,
        store_predictions, PACKAGE = "fastPLS"
    )
}

pls_cv_opls_regression_float32_core_cpp <- function(
    predictors, responses, folds, components, scaling, metric, north,
    oversample, power, seed, store_predictions = TRUE
) {
    .Call(
        "_fastPLS_pls_cv_opls_regression_float32_core_cpp", predictors,
        responses, folds, components, scaling, metric, north, oversample,
        power, seed, store_predictions, PACKAGE = "fastPLS"
    )
}

pls_cv_kernel_regression_core_cpp <- function(
    predictors, responses, folds, components, scaling, metric, kernel,
    gamma, degree, coef0, oversample, power, seed, store_predictions = TRUE
) {
    .Call(
        "_fastPLS_pls_cv_kernel_regression_core_cpp", predictors, responses,
        folds, components, scaling, metric, kernel, gamma, degree, coef0,
        oversample, power, seed, store_predictions, PACKAGE = "fastPLS"
    )
}

pls_cv_kernel_regression_float32_core_cpp <- function(
    predictors, responses, folds, components, scaling, metric, kernel,
    gamma, degree, coef0, oversample, power, seed, store_predictions = TRUE
) {
    .Call(
        "_fastPLS_pls_cv_kernel_regression_float32_core_cpp", predictors,
        responses, folds, components, scaling, metric, kernel, gamma, degree,
        coef0, oversample, power, seed, store_predictions, PACKAGE = "fastPLS"
    )
}

pls_labels_core_cpp <- function(
    predictors, labels, class_count, components, scaling, fit,
    oversample, power, seed, store_scores = TRUE
) {
    .Call(
        "_fastPLS_pls_labels_core_cpp", predictors, labels, class_count,
        components, scaling, fit, store_scores, oversample, power, seed,
        PACKAGE = "fastPLS"
    )
}

pls_labels_core_predict_cpp <- function(model, predictors, project = FALSE) {
    .Call(
        "_fastPLS_pls_labels_core_predict_cpp", model, predictors, project,
        PACKAGE = "fastPLS"
    )
}

pls_class_predict_topk_core_cpp <- function(
    model, predictors, top = 1L, project = FALSE, block_size = 4096L
) {
    .Call(
        "_fastPLS_pls_class_predict_topk_core_cpp", model, predictors,
        as.integer(top), isTRUE(project), as.integer(block_size),
        PACKAGE = "fastPLS"
    )
}

pls_float32_class_predict_compact_cpp <- function(
    model, predictors, use_lda = FALSE, block_size = 4096L
) {
    .Call(
        "_fastPLS_pls_float32_class_predict_compact_cpp", model, predictors,
        isTRUE(use_lda), as.integer(block_size), PACKAGE = "fastPLS"
    )
}

lda_project_train_prefix_float32_cpp <- function(
    model, predictors, labels, class_count, components
) {
    .Call(
        "_fastPLS_lda_project_train_prefix_float32_cpp", model, predictors,
        labels, class_count, components, PACKAGE = "fastPLS"
    )
}

pls_simpls_labels_core_cpp <- function(
    predictors, labels, class_count, components, scaling, fit,
    oversample, power, seed, store_scores = TRUE
) {
    .Call(
        "_fastPLS_pls_simpls_labels_core_cpp", predictors, labels,
        class_count, components, scaling, fit, store_scores,
        oversample, power, seed,
        PACKAGE = "fastPLS"
    )
}

pls_matrix_core_cpp <- function(
    predictors, responses, components, scaling, fit, method,
    oversample, power, seed, store_scores = TRUE
) {
    .Call(
        "_fastPLS_pls_matrix_core_cpp", predictors, responses, components,
        scaling, fit, store_scores, method, oversample, power, seed,
        PACKAGE = "fastPLS"
    )
}

pls_matrix_core_xprod_cpp <- function(
    predictors, responses, components, scaling, fit, method,
    oversample, power, seed, store_scores = TRUE
) {
    .Call(
        "_fastPLS_pls_matrix_core_xprod_cpp", predictors, responses,
        components, scaling, fit, store_scores, method, oversample, power,
        seed,
        PACKAGE = "fastPLS"
    )
}

pls_float32_matrix_backend_core_cpp <- function(
    predictors, responses, components, scaling, fit, method,
    oversample, power, seed, backend, store_scores = TRUE
) {
    .Call(
        "_fastPLS_pls_float32_matrix_backend_core_cpp", predictors,
        responses, components, scaling, fit, store_scores, method,
        oversample, power, seed, backend, PACKAGE = "fastPLS"
    )
}

pls_float32_labels_backend_core_cpp <- function(
    predictors, labels, class_count, components, scaling, fit, method,
    oversample, power, seed, backend, store_scores = TRUE
) {
    .Call(
        "_fastPLS_pls_float32_labels_backend_core_cpp", predictors, labels,
        class_count, components, scaling, fit, store_scores, method,
        oversample, power, seed, backend, PACKAGE = "fastPLS"
    )
}

lda_train_prefix_cpp <- function(
    scores, labels, class_count, components, ridge
) {
    .Call(
        "_fastPLS_lda_train_prefix_cpp", scores, labels, class_count,
        components, ridge, PACKAGE = "fastPLS"
    )
}

lda_train_moments_prefix_cpp <- function(
    gram, class_sums, counts, sample_count, components
) {
    .Call(
        "_fastPLS_lda_train_moments_prefix_cpp", gram, class_sums, counts,
        sample_count, components, PACKAGE = "fastPLS"
    )
}

lda_project_train_prefix_cpp <- function(
    predictors, projection, offset, labels, class_count, components, ridge
) {
    .Call(
        "_fastPLS_lda_project_train_prefix_cpp", predictors, projection,
        offset, labels, class_count, components, ridge, PACKAGE = "fastPLS"
    )
}

lda_predict_cpp <- function(scores, model) {
    .Call("_fastPLS_lda_predict_cpp", scores, model, PACKAGE = "fastPLS")
}

lda_predict_labels_cpp <- function(scores, model) {
    .Call(
        "_fastPLS_lda_predict_labels_cpp", scores, model,
        PACKAGE = "fastPLS"
    )
}

lda_project_predict_labels_cpp <- function(
    predictors, projection, offset, model
) {
    .Call(
        "_fastPLS_lda_project_predict_labels_cpp", predictors, projection,
        offset, model, PACKAGE = "fastPLS"
    )
}

lda_train_prefix_float32_cpp <- function(
    scores, labels, class_count, components
) {
    .Call(
        "_fastPLS_lda_train_prefix_float32_cpp", scores, labels,
        class_count, components, PACKAGE = "fastPLS"
    )
}

lda_predict_float32_cpp <- function(scores, model, return_scores = TRUE) {
    .Call(
        "_fastPLS_lda_predict_float32_cpp", scores, model, return_scores,
        PACKAGE = "fastPLS"
    )
}

cpu_float32_matrix_multiply_cpp <- function(
    left, right, transpose_left = FALSE, transpose_right = FALSE
) {
    .Call(
        "_fastPLS_cpu_float32_matrix_multiply_cpp", left, right,
        transpose_left, transpose_right, PACKAGE = "fastPLS"
    )
}

metal_float32_matrix_multiply_cpp <- function(
    left, right, transpose_left = FALSE, transpose_right = FALSE
) {
    .Call(
        "_fastPLS_metal_float32_matrix_multiply_cpp", left, right,
        transpose_left, transpose_right, PACKAGE = "fastPLS"
    )
}

kernel_matrix_float32_cpp <- function(
    left, right, kernel, gamma, degree, offset, backend
) {
    .Call(
        "_fastPLS_kernel_matrix_float32_cpp", left, right, kernel, gamma,
        degree, offset, backend, PACKAGE = "fastPLS"
    )
}

opls_apply_filter_float32_cpp <- function(
    matrix, center, scale, weights, loadings, backend
) {
    .Call(
        "_fastPLS_opls_apply_filter_float32_cpp", matrix, center, scale,
        weights, loadings, backend, PACKAGE = "fastPLS"
    )
}

opls_apply_filter_cpp <- function(matrix, center, scale, weights, loadings) {
    .Call(
        "_fastPLS_opls_apply_filter_cpp", matrix, center, scale, weights,
        loadings, PACKAGE = "fastPLS"
    )
}

opls_filter_core_cpp <- function(predictors, responses, north, scaling) {
    .Call(
        "_fastPLS_opls_filter_core_cpp", predictors, responses, north, scaling,
        PACKAGE = "fastPLS"
    )
}

opls_filter_labels_core_cpp <- function(
    predictors, labels, class_count, north, scaling
) {
    .Call(
        "_fastPLS_opls_filter_labels_core_cpp", predictors, labels,
        class_count, north, scaling, PACKAGE = "fastPLS"
    )
}

opls_filter_float32_core_cpp <- function(
    predictors, responses, north, scaling, oversample, power, seed
) {
    .Call(
        "_fastPLS_opls_filter_float32_core_cpp", predictors, responses,
        north, scaling, oversample, power, seed, PACKAGE = "fastPLS"
    )
}

opls_filter_float32_labels_core_cpp <- function(
    predictors, labels, class_count, north, scaling
) {
    .Call(
        "_fastPLS_opls_filter_float32_labels_core_cpp", predictors, labels,
        class_count, north, scaling, PACKAGE = "fastPLS"
    )
}

opls_filter_float32_backend_core_cpp <- function(
    predictors, responses, north, scaling, backend, oversample, power, seed
) {
    .Call(
        "_fastPLS_opls_filter_float32_backend_core_cpp", predictors,
        responses, north, scaling, backend, oversample, power, seed,
        PACKAGE = "fastPLS"
    )
}

opls_filter_float32_labels_backend_core_cpp <- function(
    predictors, labels, class_count, north, scaling, backend, oversample,
    power, seed
) {
    .Call(
        "_fastPLS_opls_filter_float32_labels_backend_core_cpp", predictors,
        labels, class_count, north, scaling, backend, oversample, power, seed,
        PACKAGE = "fastPLS"
    )
}

cuda_resident_project_cpp <- function(object, X, ncomp) {
    .Call(
        "_fastPLS_cuda_resident_project_cpp",
        object, X, ncomp,
        PACKAGE = "fastPLS"
    )
}

cuda_resident_response_sums_cpp <- function(object, X, Y, labels, ncomp) {
    .Call(
        "_fastPLS_cuda_resident_response_sums_cpp",
        object, X, Y, labels, ncomp,
        PACKAGE = "fastPLS"
    )
}

cuda_resident_simpls_fit_cpp <- function(
    X, Y, labels, classes, precision, ncomp, scaling, oversample, power, seed,
    retain_scores = TRUE, method = 3L, north = 1L, kernel = 2L, gamma = 1,
    degree = 3L, coef0 = 1
) {
    .Call(
        "_fastPLS_cuda_resident_simpls_fit_cpp",
        X, Y, labels, classes, precision, ncomp, scaling, oversample, power,
        seed, retain_scores, method, north, kernel, gamma, degree, coef0,
        PACKAGE = "fastPLS"
    )
}

cuda_resident_export_cpp <- function(
    object, loadings = FALSE, variance = FALSE, scores = TRUE
) {
    .Call(
        "_fastPLS_cuda_resident_export_cpp",
        object, loadings, variance, scores,
        PACKAGE = "fastPLS"
    )
}

cuda_resident_compact_cpp <- function(object, prepare_lda = FALSE) {
    invisible(.Call(
        "_fastPLS_cuda_resident_compact_cpp",
        object, prepare_lda,
        PACKAGE = "fastPLS"
    ))
}

cuda_resident_classify_path_cpp <- function(
    object, X, ncomp, classifier, top
) {
    .Call(
        "_fastPLS_cuda_resident_classify_path_cpp",
        object, X, ncomp, classifier, top,
        PACKAGE = "fastPLS"
    )
}

cuda_resident_classify_response_path_cpp <- function(
    object, X, ncomp, classifier, top
) {
    .Call(
        "_fastPLS_cuda_resident_classify_response_path_cpp",
        object, X, ncomp, classifier, top,
        PACKAGE = "fastPLS"
    )
}

cuda_resident_predict_path_cpp <- function(
    object, X, ncomp, classifier = 0L
) {
    .Call(
        "_fastPLS_cuda_resident_predict_path_cpp",
        object, X, ncomp, classifier,
        PACKAGE = "fastPLS"
    )
}

has_cuda <- function() {
    .Call("_fastPLS_has_cuda", PACKAGE = "fastPLS")
}

has_metal <- function() {
    .Call("_fastPLS_has_metal", PACKAGE = "fastPLS")
}

blas_backend_cpp <- function() {
    .Call("_fastPLS_blas_backend_cpp", PACKAGE = "fastPLS")
}

rsvd_audit_reset_debug <- function() {
    invisible(.Call("_fastPLS_rsvd_audit_reset_debug", PACKAGE = "fastPLS"))
}

rsvd_audit_summary_debug <- function() {
    .Call("_fastPLS_rsvd_audit_summary_debug", PACKAGE = "fastPLS")
}

spearman_correlation_cpp <- function(observed, predicted) {
    .Call(
        "_fastPLS_spearman_correlation_cpp",
        observed,
        predicted,
        PACKAGE = "fastPLS"
    )
}

evaluate_regression_core_cpp <- function(
    observed, predicted, training = NULL,
    relative_epsilon = .Machine$double.eps, na.rm = TRUE
) {
    .Call(
        "_fastPLS_evaluate_regression_core_cpp",
        observed, predicted, training, relative_epsilon, na.rm,
        PACKAGE = "fastPLS"
    )
}

evaluate_regression_by_column_cpp <- function(
    observed, predicted, training = NULL,
    relative_epsilon = .Machine$double.eps, na.rm = TRUE
) {
    .Call(
        "_fastPLS_evaluate_regression_by_column_cpp",
        observed, predicted, training, relative_epsilon, na.rm,
        PACKAGE = "fastPLS"
    )
}

evaluate_classification_core_cpp <- function(
    observed, predicted, class_count, scores = NULL,
    score_observed = integer(), top_k = integer()
) {
    .Call(
        "_fastPLS_evaluate_classification_core_cpp",
        observed, predicted, as.integer(class_count), scores,
        as.integer(score_observed), as.integer(top_k), PACKAGE = "fastPLS"
    )
}

evaluate_is_onehot_cpp <- function(values) {
    .Call("_fastPLS_evaluate_is_onehot_cpp", values, PACKAGE = "fastPLS")
}

evaluate_class_labels_cpp <- function(values, reference_levels = NULL) {
    .Call(
        "_fastPLS_evaluate_class_labels_cpp",
        values, reference_levels, PACKAGE = "fastPLS"
    )
}

vip_core_cpp <- function(model) {
    .Call("_fastPLS_vip_core_cpp", model, PACKAGE = "fastPLS")
}

fastcor_core_cpp <- function(a, b = NULL, byrow = TRUE, diag = TRUE) {
    .Call(
        "_fastPLS_fastcor_core_cpp", a, b, byrow, diag,
        PACKAGE = "fastPLS"
    )
}

float32_argmax_cpp <- function(scoresSEXP) {
    .Call("_fastPLS_float32_argmax_cpp", scoresSEXP, PACKAGE = "fastPLS")
}

float32_topk_cpp <- function(scoresSEXP, top) {
    .Call(
        "_fastPLS_float32_topk_cpp",
        scoresSEXP,
        top,
        PACKAGE = "fastPLS"
    )
}

float32_sweep_cols_cpp <- function(XSEXP, rowSEXP, operation) {
    .Call(
        "_fastPLS_float32_sweep_cols_cpp",
        XSEXP,
        rowSEXP,
        operation,
        PACKAGE = "fastPLS"
    )
}

float32_standardize_cpp <- function(XSEXP, centerSEXP, scaleSEXP) {
    .Call(
        "_fastPLS_float32_standardize_cpp",
        XSEXP,
        centerSEXP,
        scaleSEXP,
        PACKAGE = "fastPLS"
    )
}

center_kernel_train_float32_cpp <- function(KSEXP) {
    .Call(
        "_fastPLS_center_kernel_train_float32_cpp",
        KSEXP,
        PACKAGE = "fastPLS"
    )
}

center_kernel_test_float32_cpp <- function(
    KtestSEXP,
    trainColMeansSEXP,
    train_grand_mean
) {
    .Call(
        "_fastPLS_center_kernel_test_float32_cpp",
        KtestSEXP,
        trainColMeansSEXP,
        train_grand_mean,
        PACKAGE = "fastPLS"
    )
}

center_kernel_train_cpp <- function(K) {
    .Call("_fastPLS_center_kernel_train_cpp", K, PACKAGE = "fastPLS")
}

kernel_matrix_cpp <- function(X1, X2, kernel, gamma, degree, coef0) {
    .Call(
        "_fastPLS_kernel_matrix_cpp",
        X1,
        X2,
        kernel,
        gamma,
        degree,
        coef0,
        PACKAGE = "fastPLS"
    )
}

center_kernel_test_cpp <- function(Ktest, train_col_means, train_grand_mean) {
    .Call(
        "_fastPLS_center_kernel_test_cpp",
        Ktest,
        train_col_means,
        train_grand_mean,
        PACKAGE = "fastPLS"
    )
}

transformy <- function(y) {
    labels <- as.integer(y)
    if (!length(labels) || anyNA(labels) || any(labels < 1L)) {
        stop("classification labels must be positive, non-missing integers")
    }
    response <- matrix(0, nrow = length(labels), ncol = max(labels))
    response[cbind(seq_along(labels), labels)] <- 1
    response
}
