test_that("blocked CPU prediction shares immutable response weights", {
    set.seed(703)
    X <- matrix(rnorm(80 * 20), 80, 20)
    Y <- matrix(rnorm(80 * 10), 80, 10)
    test <- X[61:80, , drop = FALSE]
    for (family in c("simpls", "plssvd")) {
        for (solver in "rsvd") {
            fitted <- suppressWarnings(pls(X[1:60, ], Y[1:60, ],
                ncomp = c(1L, 3L, 5L), method = family, svd.method = solver,
                backend = "cpu", scaling = "autoscaling", seed = 7))
            raw <- fastPLS:::.fastpls_restore_internal_output_fields(fitted)
            raw$B <- NULL
            raw$predict_latent_ok <- TRUE
            for (representation in c("stored", "factorized")) {
                model <- raw
                if (representation == "factorized") model$W_latent <- NULL
                original <- serialize(model, NULL)
                expected <- fastPLS:::pls_predict(model, test, TRUE)
                for (block in c(1L, 7L, 4096L)) {
                    for (projection in c(FALSE, TRUE)) {
                        actual <- fastPLS:::pls_predict_flash_cpu(
                            model, test, projection, block)
                        expect_equal(actual$Ypred, expected$Ypred,
                            tolerance = 2e-12)
                        if (projection) {
                            expect_equal(actual$Ttest, expected$Ttest,
                                tolerance = 2e-12)
                        } else {
                            expect_equal(dim(actual$Ttest), c(0L, 0L))
                        }
                        expect_identical(serialize(model, NULL), original)
                    }
                }
                # Native prediction preserves caller-specified prefix ordering.
                order <- c(3L, 1L, 3L)
                model$ncomp <- model$ncomp[order]
                for (field in c("W_latent", "C_latent")) {
                    if (!is.null(model[[field]])) {
                        model[[field]] <- model[[field]][, , order, drop = FALSE]
                    }
                }
                actual <- fastPLS:::pls_predict_flash_cpu(model, test, TRUE, 7L)
                expect_equal(actual$Ypred, expected$Ypred[, , order, drop = FALSE],
                    tolerance = 2e-12)
                model$ncomp[[1L]] <- ncol(model$R) + 1L
                expect_error(fastPLS:::pls_predict_flash_cpu(model, test, FALSE, 7L),
                    "ncomp exceeds latent rank")
            }
        }
    }
})
