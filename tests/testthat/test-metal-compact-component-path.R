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
            repeated <- predict(model, X)
            repeated_again <- predict(model, X)
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
            expected <- predict(model, Xtest)$Ypred[[index]]
            expect_identical(
                as.character(independent$Ypred[[1L]]),
                as.character(expected)
            )
        }
    }
})

test_that("Metal large PLS-SVD paths omit the response-weight cube", {
    skip_if_not(has_metal(), "Metal backend is not available")
    set.seed(916)
    rank <- 32L
    components <- seq(2L, rank, by = 3L)
    prep <- list(X = matrix(rnorm(50 * 400), 50, 400), n = 50L,
        p = 400L, m = 14000L, mX = matrix(0, 1, 400),
        vX = matrix(1, 1, 400), mY = matrix(0, 1, 14000))
    decomposition <- list(R = matrix(rnorm(400 * rank), 400, rank),
        Q = matrix(rnorm(prep$m * rank), prep$m, rank),
        singular = seq(rank, 1), rank = rank, implicit = FALSE)
    expect_gt(rank * prep$m * length(components) * 8, 32 * 1024^2)
    path <- fastPLS:::.metal_plssvd_path(prep, decomposition, components, FALSE)
    expect_null(path$W)
    expect_null(path$B)
    model <- fastPLS:::.metal_plssvd_model(prep, decomposition, path, components)
    expect_false("W_latent" %in% names(model))
    expect_equal(dim(model$C_latent), c(rank, rank, length(components)))
    expect_lt(as.numeric(object.size(model)), 8 * 1024^2)
})

test_that("Metal coefficient-only models remain predictable", {
    skip_if_not(has_metal(), "Metal backend is not available")
    X <- matrix(seq_len(8), 4, 2)
    B <- array(seq_len(8) / 10, c(2, 2, 2))
    model <- list(B = B, ncomp = c(1L, 3L), p = 2L, m = 2L,
        mX = matrix(0, 1, 2), vX = matrix(1, 1, 2), mY = matrix(0, 1, 2))
    actual <- fastPLS:::.pls_predict_metal(model, X)
    for (i in 1:2) {
        expect_equal(actual$Ypred[, , i], fastPLS:::.metal_mm(X, B[, , i]),
            tolerance = 0)
    }
    expect_error(fastPLS:::.pls_predict_metal(model[names(model) != "B"], X),
        "requires compact factors or coefficients")
})
