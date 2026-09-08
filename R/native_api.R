# Hand-written R C-API entry points used during the dependency-free migration.

has_cuda <- function() {
    .Call("_fastPLS_has_cuda", PACKAGE = "fastPLS")
}

has_metal <- function() {
    .Call("_fastPLS_has_metal", PACKAGE = "fastPLS")
}

spearman_correlation_cpp <- function(observed, predicted) {
    .Call(
        "_fastPLS_spearman_correlation_cpp",
        observed,
        predicted,
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
