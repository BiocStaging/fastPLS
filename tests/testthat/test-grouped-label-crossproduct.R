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

test_that("float32 class-sum fallback preserves preprocessing statistics", {
    skip_if_not_installed("float")
    set.seed(73)
    per_class <- 600L
    labels <- factor(rep(letters[1:4], each = per_class))
    X <- matrix(rnorm(length(labels) * 12L), ncol = 12L)
    X <- sweep(X, 2L, seq(2, 24, by = 2), "+")
    Xtest <- matrix(rnorm(80L * 12L), ncol = 12L)
    Xtest <- sweep(Xtest, 2L, seq(2, 24, by = 2), "+")
    interleaved <- as.vector(vapply(
        seq_len(per_class),
        function(index) index + (0:3) * per_class,
        integer(4L)
    ))

    fit_once <- function(index) {
        suppressWarnings(pls(
            float::fl(X[index, , drop = FALSE]), labels[index], ncomp = 1:4,
            method = "simpls", backend = "cpu", svd.method = "rsvd",
            seed = 29, oversample = 10, power = 3,
            return_variance = FALSE
        ))
    }
    grouped_fit <- fit_once(seq_len(nrow(X)))
    interleaved_fit <- fit_once(interleaved)

    expect_equal(
        float::dbl(grouped_fit$mX),
        float::dbl(interleaved_fit$mX),
        tolerance = 2e-5
    )
    expect_identical(
        predict(grouped_fit, float::fl(Xtest))$Ypred[[4L]],
        predict(interleaved_fit, float::fl(Xtest))$Ypred[[4L]]
    )
})

test_that("compact float32 class prediction matches retained-score paths", {
    skip_if_not_installed("float")
    predictors <- float::fl(as.matrix(iris[, seq_len(4L)]))
    labels <- factor(iris$Species)

    for (method in c("simpls", "plssvd")) {
        for (classifier in c("argmax", "lda")) {
            fit <- suppressWarnings(pls(
                predictors, labels, ncomp = 1:3, method = method,
                classifier = classifier, backend = "cpu",
                svd.method = "rsvd", oversample = 32L, power = 5L,
                seed = 123L, return_variance = FALSE
            ))
            compact <- predict(fit, predictors, backend = "cpu")$Ypred
            retained <- predict(
                fit, predictors, backend = "cpu", raw_scores = TRUE
            )$Ypred
            expect_identical(
                lapply(compact, as.character),
                lapply(retained, as.character)
            )
        }
    }
})
