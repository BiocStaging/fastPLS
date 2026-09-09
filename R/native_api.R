# Hand-written R C-API entry points used during the dependency-free migration.

cpu_backend_description_cpp <- function() {
    .Call("_fastPLS_cpu_backend_description", PACKAGE = "fastPLS")
}

pls_float32_labels_core_cpp <- function(
    predictors, labels, class_count, components, scaling, fit, method,
    oversample, power, seed
) {
    .Call(
        "_fastPLS_pls_float32_labels_core_cpp", predictors, labels,
        class_count, components, scaling, fit, method, oversample, power, seed,
        PACKAGE = "fastPLS"
    )
}

lda_train_prefix_cpp <- function(scores, labels, class_count, components,
                                 ridge) {
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

cuda_resident_classify_cpp <- function(object, X, ncomp, classifier, top) {
    .Call(
        "_fastPLS_cuda_resident_classify_cpp",
        object, X, ncomp, classifier, top,
        PACKAGE = "fastPLS"
    )
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

cuda_resident_simpls_predict_cpp <- function(
    object, X, ncomp, classifier = 0L
) {
    .Call(
        "_fastPLS_cuda_resident_simpls_predict_cpp",
        object, X, ncomp, classifier,
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

lda_cuda_native_available <- function() {
    .Call("_fastPLS_lda_cuda_native_available", PACKAGE = "fastPLS")
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

label_crossprod_scaled_cpp <- function(
    XtrainSEXP,
    y,
    n_classes,
    scaling
) {
    .Call(
        "_fastPLS_label_crossprod_scaled_cpp",
        XtrainSEXP,
        y,
        n_classes,
        scaling,
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

RQ <- function(yData, yPred) {
    yData <- as.matrix(yData)
    yPred <- as.matrix(yPred)
    if (!identical(dim(yData), dim(yPred))) {
        stop("R2 matrices must have matching dimensions")
    }
    centered <- sweep(yData, 2L, colMeans(yData), "-")
    1 - sum((yData - yPred)^2) / sum(centered^2)
}
