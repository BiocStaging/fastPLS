test_that("operation-split Metal compact factors predict every prefix", {
    skip_if_not(has_metal(), "Metal backend is not available")
    set.seed(915)
    X <- float::fl(matrix(rnorm(60 * 12), 60, 12))
    Y <- float::fl(matrix(rnorm(60 * 8), 60, 8))
    components <- c(1L, 3L, 6L)
    for (family in c("simpls", "plssvd")) {
        for (fitted in c(FALSE, TRUE)) {
            model <- suppressWarnings(pls(X, Y, ncomp = components,
                method = family, backend = "metal", fit = fitted,
                return_variance = FALSE, seed = 15))
            raw <- fastPLS:::.fastpls_restore_internal_output_fields(model)
            expect_false(is.matrix(raw$B) || length(dim(raw$B)) == 3L)
            expect_null(raw$resident_state)
            expect_identical(
                raw$execution_route,
                "CPU/Metal hybrid (operation split)"
            )
            repeated <- predict(model, X, backend = "metal")
            repeated_again <- predict(model, X, backend = "metal")
            expect_equal(repeated$Ypred, repeated_again$Ypred, tolerance = 0)
            if (fitted) {
                expect_true(all(is.finite(model$R2Y)))
                expect_false(is.null(model$Yfit))
            }
        }
    }
})

test_that("operation-split Metal class paths match independent fits", {
    skip_if_not(has_metal(), "Metal backend is not available")
    set.seed(919)
    X <- float::fl(matrix(rnorm(90 * 14), 90, 14))
    labels <- factor(rep(c("a", "b", "c"), each = 30))
    Xtest <- float::fl(matrix(rnorm(21 * 14), 21, 14))
    components <- c(1L, 3L, 5L)

    for (classifier in c("argmax", "lda")) {
        model <- pls(
            X, labels, ncomp = components, method = "simpls",
            backend = "metal", classifier = classifier,
            return_variance = FALSE, seed = 19
        )
        for (index in seq_along(components)) {
            independent <- pls(
                X, labels, Xtest, ncomp = components[[index]],
                method = "simpls", backend = "metal",
                classifier = classifier, return_variance = FALSE, seed = 19
            )
            expected <- predict(
                model, Xtest, backend = "metal"
            )$Ypred[[index]]
            expect_identical(
                as.character(independent$Ypred[[1L]]),
                as.character(expected)
            )
        }
    }
})
