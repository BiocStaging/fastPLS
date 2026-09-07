test_that("compiled CPU prediction reuses scores for rSVD and IRLBA paths", {
    set.seed(188)
    X <- matrix(rnorm(100 * 24), 100, 24)
    Y <- matrix(rnorm(100 * 12), 100, 12)
    test <- X[81:100, , drop = FALSE]
    original <- test
    for (family in c("simpls", "plssvd")) {
        for (solver in "rsvd") {
            fit <- suppressWarnings(pls(X[1:80, ], Y[1:80, ],
                ncomp = c(1L, 3L, 6L), method = family, svd.method = solver,
                backend = "cpu", scaling = "autoscaling", seed = 11))
            raw <- fastPLS:::.fastpls_restore_internal_output_fields(fit)
            raw$B <- NULL
            raw$predict_latent_ok <- TRUE
            # Request only the first two prefixes, but retain all score columns
            # when proj=TRUE, as required by the existing prediction contract.
            raw$ncomp <- raw$ncomp[1:2]
            if (!is.null(raw$W_latent)) {
                raw$W_latent <- raw$W_latent[, , 1:2, drop = FALSE]
            }
            if (!is.null(raw$C_latent)) {
                raw$C_latent <- raw$C_latent[, , 1:2, drop = FALSE]
            }
            for (weights in c("stored", "factorized")) {
                model <- raw
                if (weights == "factorized") model$W_latent <- NULL
                for (projection in c(FALSE, TRUE)) {
                    actual <- fastPLS:::pls_predict(model, test, projection)
                    for (i in seq_along(model$ncomp)) {
                        prefix <- model
                        prefix$ncomp <- model$ncomp[[i]]
                        for (field in c("W_latent", "C_latent")) {
                            if (!is.null(model[[field]])) {
                                prefix[[field]] <- model[[field]][, , i,
                                    drop = FALSE]
                            }
                        }
                        expected <- fastPLS:::pls_predict(prefix, test, FALSE)
                        expect_equal(actual$Ypred[, , i], expected$Ypred[, , 1],
                            tolerance = 2e-12)
                    }
                    centered <- sweep(test, 2, as.numeric(model$mX), "-")
                    centered <- sweep(centered, 2, as.numeric(model$vX), "/")
                    if (projection) {
                        expect_equal(actual$Ttest, centered %*% model$R,
                            tolerance = 2e-12)
                    } else {
                        expect_equal(dim(actual$Ttest), c(0L, 0L))
                    }
                    expect_identical(test, original)
                }
            }
        }
    }
})

test_that("invalid latent counts do not bypass the existing coefficient fallback", {
    model <- list(m = 1L, ncomp = 3L, mX = matrix(0, 1, 2),
        vX = matrix(1, 1, 2), mY = matrix(0, 1, 1),
        R = matrix(c(1, 0), 2, 1), Q = matrix(1, 1, 1),
        B = array(c(1, 2), c(2, 1, 1)), pls_method = "simpls",
        predict_latent_ok = TRUE)
    X <- matrix(as.numeric(seq_len(8)), 4, 2)
    result <- fastPLS:::pls_predict(model, X, TRUE)
    expect_equal(drop(result$Ypred), drop(X %*% c(1, 2)))
    expect_equal(result$Ttest, X[, 1, drop = FALSE])
    model$B <- NULL
    expect_error(fastPLS:::pls_predict(model, X, TRUE),
        "compact latent prediction was not available")
})
