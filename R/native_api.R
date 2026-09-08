# Hand-written R C-API entry points used during the dependency-free migration.

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
