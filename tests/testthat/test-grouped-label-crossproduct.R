test_that("grouped-label BLAS products preserve SIMPLS predictions", {
    set.seed(71)
    labels <- factor(rep(letters[1:4], each = 120L))
    X <- matrix(rnorm(length(labels) * 24L), ncol = 24L)
    Xtest <- matrix(rnorm(80L * 24L), ncol = 24L)
    shuffled <- sample.int(nrow(X))

    grouped_fit <- pls(
        X, labels, ncomp = 1:6, method = "simpls", backend = "cpu",
        svd.method = "rsvd", seed = 19, oversample = 12, power = 3,
        return_variance = FALSE
    )
    shuffled_fit <- pls(
        X[shuffled, , drop = FALSE], labels[shuffled], ncomp = 1:6,
        method = "simpls", backend = "cpu", svd.method = "rsvd", seed = 19,
        oversample = 12, power = 3, return_variance = FALSE
    )

    grouped_prediction <- predict(grouped_fit, Xtest)$Ypred[[6L]]
    shuffled_prediction <- predict(shuffled_fit, Xtest)$Ypred[[6L]]
    expect_identical(grouped_prediction, shuffled_prediction)
})

test_that("grouped-label float32 products preserve class predictions", {
    skip_if_not_installed("float")
    set.seed(72)
    labels <- factor(rep(letters[1:3], each = 100L))
    X <- matrix(rnorm(length(labels) * 18L), ncol = 18L)
    Xtest <- matrix(rnorm(60L * 18L), ncol = 18L)
    shuffled <- sample.int(nrow(X))

    fit_once <- function(index) {
        suppressWarnings(pls(
            float::fl(X[index, , drop = FALSE]), labels[index], ncomp = 1:5,
            method = "simpls", backend = "cpu", svd.method = "rsvd",
            seed = 23, oversample = 10, power = 3, return_variance = FALSE
        ))
    }
    grouped_prediction <- predict(
        fit_once(seq_len(nrow(X))), float::fl(Xtest)
    )$Ypred[[5L]]
    shuffled_prediction <- predict(
        fit_once(shuffled), float::fl(Xtest)
    )$Ypred[[5L]]
    expect_identical(grouped_prediction, shuffled_prediction)
})
