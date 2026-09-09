dense_regression_fixture <- function() {
    set.seed(9551)
    X <- matrix(rnorm(120L * 20L), 120L, 20L)
    beta <- matrix(rnorm(20L * 4L), 20L, 4L)
    Y <- X %*% beta + matrix(rnorm(120L * 4L, sd = 0.05), 120L, 4L)
    Xtest <- matrix(rnorm(37L * 20L), 37L, 20L)
    list(X = X, Y = Y, Xtest = Xtest)
}

test_that("dependency-free dense PLS preserves legacy regression paths", {
    task <- dense_regression_fixture()
    components <- 1:3
    rsvd <- fastPLS:::.svd_method_id("rsvd")
    for (method in c(plssvd = 1L, simpls = 3L)) {
        core <- fastPLS:::pls_matrix_core_cpp(
            task$X, task$Y, components, 1L, TRUE, unname(method),
            32L, 5L, 9551L
        )
        legacy <- if (unname(method) == 1L) {
            fastPLS:::pls_model1(
                task$X, task$Y, components, 1L, TRUE, rsvd,
                32L, 5L, 0, 9551L
            )
        } else {
            fastPLS:::pls_model2_fast(
                task$X, task$Y, components, 1L, TRUE, rsvd,
                32L, 5L, 0, 9551L
            )
        }
        core_prediction <- fastPLS:::pls_labels_core_predict_cpp(
            core, task$Xtest, FALSE
        )$Ypred
        legacy_prediction <- fastPLS:::pls_predict(
            legacy, task$Xtest, FALSE
        )$Ypred
        expect_equal(core$mY, legacy$mY, tolerance = 1e-12)
        expect_equal(
            as.numeric(core$R2Y), as.numeric(legacy$R2Y), tolerance = 1e-8
        )
        for (index in seq_along(components)) {
            expect_equal(
                core$Yfit[, , index], legacy$Yfit[, , index], tolerance = 1e-8
            )
        }
        expect_equal(core_prediction, legacy_prediction, tolerance = 1e-8)
    }
})

test_that("public dense CPU rSVD uses the standalone core", {
    task <- dense_regression_fixture()
    for (method in c("plssvd", "simpls")) {
        fit <- pls(
            task$X, task$Y, task$Xtest,
            ncomp = 1:3,
            method = method,
            backend = "cpu",
            svd.method = "rsvd",
            fit = FALSE,
            return_variance = FALSE,
            oversample = 32L,
            power = 5L,
            seed = 9551L
        )
        expect_identical(fit$xprod_mode, "float64_dense_crosscov")
        expect_equal(dim(fit$Ypred), c(nrow(task$Xtest), ncol(task$Y), 3L))
    }
})

test_that("float32 dense CPU rSVD uses the same standalone core", {
    skip_if_not_installed("float")
    skip_on_os("windows")
    task <- dense_regression_fixture()
    X <- float::fl(task$X)
    Y <- float::fl(task$Y)
    components <- 1:3
    for (method in c(plssvd = 1L, simpls = 3L)) {
        core <- fastPLS:::pls_float32_matrix_core_cpp(
            X, Y, components, 1L, TRUE, unname(method), 32L, 5L, 9551L
        )
        legacy <- fastPLS:::pls_float32_cpu_cpp(
            X, Y, components, 1L, TRUE, unname(method), 0L, 3L,
            32L, 5L, 9551L
        )
        core_values <- fastPLS:::.float32_bits_list_to_float(core$Yfit)
        legacy_values <- fastPLS:::.float32_bits_list_to_float(legacy$Yfit)
        expect_equal(
            as.numeric(core$R2Y), as.numeric(legacy$R2Y), tolerance = 2e-5
        )
        for (name in names(core_values)) {
            expect_equal(
                float::dbl(core_values[[name]]),
                float::dbl(legacy_values[[name]]),
                tolerance = 2e-5
            )
        }

        fit <- pls(
            X, Y, ncomp = components, method = names(method),
            backend = "cpu", svd.method = "rsvd", fit = FALSE,
            return_variance = FALSE, oversample = 32L, power = 5L,
            seed = 9551L
        )
        expect_identical(fit$xprod_mode, "float32_dense_crosscov")
    }
})

test_that("implicit PLS-SVD avoids the dense cross-covariance", {
    task <- dense_regression_fixture()
    components <- 1:3
    explicit <- fastPLS:::pls_matrix_core_cpp(
        task$X, task$Y, components, 1L, TRUE, 1L, 32L, 5L, 9551L
    )
    implicit <- fastPLS:::pls_matrix_core_xprod_cpp(
        task$X, task$Y, components, 1L, TRUE, 1L, 32L, 5L, 9551L
    )
    explicit_prediction <- fastPLS:::pls_labels_core_predict_cpp(
        explicit, task$Xtest, FALSE
    )$Ypred
    implicit_prediction <- fastPLS:::pls_labels_core_predict_cpp(
        implicit, task$Xtest, FALSE
    )$Ypred
    expect_equal(implicit$R2Y, explicit$R2Y, tolerance = 1e-8)
    expect_equal(implicit_prediction, explicit_prediction, tolerance = 1e-8)

    old <- Sys.getenv(
        c("FASTPLS_ABLATION_MODE", "FASTPLS_ABLATION_XPROD"),
        unset = NA_character_
    )
    on.exit({
        present <- !is.na(old)
        if (any(present)) do.call(Sys.setenv, as.list(old[present]))
        if (any(!present)) Sys.unsetenv(names(old)[!present])
    }, add = TRUE)
    Sys.setenv(FASTPLS_ABLATION_MODE = "1", FASTPLS_ABLATION_XPROD = "1")
    fit <- pls(
        task$X, task$Y, task$Xtest, ncomp = components,
        method = "plssvd", backend = "cpu", svd.method = "rsvd",
        fit = FALSE, return_variance = FALSE, oversample = 32L,
        power = 5L, seed = 9551L
    )
    expect_identical(fit$xprod_mode, "float64_implicit_crosscov")
    expect_equal(fit$Ypred, implicit_prediction, tolerance = 1e-8)
})
