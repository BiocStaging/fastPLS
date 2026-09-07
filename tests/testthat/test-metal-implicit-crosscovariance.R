test_that("Metal uses an implicit cross-covariance for large regression shapes", {
    skip_if_not(has_metal(), "Metal backend is not available")

    set.seed(812)
    n <- 48L
    p <- 128L
    q <- 128L
    x <- matrix(rnorm(n * p), n, p)
    coefficients <- matrix(rnorm(p * q), p, q)
    y <- x %*% coefficients + matrix(rnorm(n * q, sd = 0.05), n, q)

    for (method in c("simpls", "plssvd")) {
        fit <- pls(
            float::fl(x), float::fl(y),
            ncomp = c(2L, 5L), method = method, backend = "metal",
            fit = TRUE, proj = TRUE, return_variance = FALSE, seed = 31
        )
        controls <- attr(fit, "fastPLS_internal")$resident_controls

        expect_true(isTRUE(controls$implicit_crosscovariance), info = method)
        expect_false(isTRUE(controls$host_assisted_components), info = method)
        expect_identical(fit$diagnostics$residency$component_updates, "metal")
        expect_equal(dim(fit$Ttrain), c(n, 5L), info = method)
        expect_true(all(is.finite(fit$R2Y)), info = method)

        predicted <- predict(fit, float::fl(x))
        expect_equal(length(predicted$Ypred), 2L, info = method)
        expect_equal(dim(predicted$Ypred[[2L]]), c(n, q), info = method)
        expect_true(
            all(is.finite(float::dbl(predicted$Ypred[[2L]]))),
            info = method
        )
    }
})

test_that("Metal keeps small explicit cross-covariance fits resident", {
    skip_if_not(has_metal(), "Metal backend is not available")

    set.seed(813)
    x <- matrix(rnorm(120 * 12), 120, 12)
    y <- matrix(rnorm(120 * 3), 120, 3)
    fit <- pls(
        float::fl(x), float::fl(y),
        ncomp = 3L, method = "simpls", backend = "metal",
        return_variance = FALSE, seed = 32
    )

    internal <- attr(fit, "fastPLS_internal")
    expect_false(is.null(internal$resident_state))
    expect_false(isTRUE(internal$resident_controls$implicit_crosscovariance))
    expect_false(isTRUE(internal$resident_controls$host_assisted_components))
    expect_identical(internal$predict_backend, "metal_resident")
    expect_identical(fit$diagnostics$residency$route, "resident metal")
})
