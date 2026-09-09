compact_label_fixture <- function() {
    set.seed(9501)
    n <- 96L
    p <- 18L
    labels <- factor(rep(c("control", "case-a", "case-b"), each = n / 3L))
    X <- matrix(rnorm(n * p), n, p)
    X[labels == "case-a", 1:4] <- X[labels == "case-a", 1:4] + 0.8
    X[labels == "case-b", 5:8] <- X[labels == "case-b", 5:8] - 0.7
    Xtest <- matrix(rnorm(33L * p), 33L, p)
    Y <- stats::model.matrix(~ labels - 1)
    list(X = X, Xtest = Xtest, Y = Y, labels = labels)
}

compact_fitted_slice <- function(model, index) {
    if (is.list(model$Yfit)) {
        return(model$Yfit[[index]])
    }
    model$Yfit[, , index, drop = FALSE][, , 1L]
}

test_that("compact labels preserve double PLS-SVD and SIMPLS mathematics", {
    task <- compact_label_fixture()
    labels <- as.integer(task$labels)

    for (method in c(plssvd = 1L, simpls = 3L)) {
        components <- if (identical(unname(method), 1L)) 1:2 else 1:3
        compact <- if (identical(unname(method), 1L)) {
            fastPLS:::pls_labels_core_cpp(
                task$X, labels, nlevels(task$labels), components, 1L, TRUE,
                32L, 5L, 41L
            )
        } else {
            fastPLS:::pls_simpls_labels_core_cpp(
                task$X, labels, nlevels(task$labels), components, 1L, TRUE,
                32L, 5L, 41L
            )
        }
        dense <- fastPLS:::pls_matrix_core_cpp(
            task$X, task$Y, components, 1L, TRUE, unname(method),
            32L, 5L, 41L
        )

        compact_prediction <- fastPLS:::pls_labels_core_predict_cpp(
            compact, task$Xtest, FALSE
        )$Ypred
        dense_prediction <- fastPLS:::pls_labels_core_predict_cpp(
            dense, task$Xtest, FALSE
        )$Ypred
        expect_equal(compact$mY, dense$mY, tolerance = 1e-13)
        expect_equal(compact$R2Y, dense$R2Y, tolerance = 1e-11)
        for (index in seq_along(components)) {
            expect_equal(
                compact_fitted_slice(compact, index),
                compact_fitted_slice(dense, index),
                tolerance = 1e-10
            )
        }
        expect_equal(compact_prediction, dense_prediction, tolerance = 1e-10)
    }
})

test_that("compact labels preserve the OPLS filtered design", {
    task <- compact_label_fixture()
    compact <- fastPLS:::opls_filter_labels_core_cpp(
        task$X, as.integer(task$labels), nlevels(task$labels), 2L, 1L
    )
    dense <- fastPLS:::opls_filter_core_cpp(task$X, task$Y, 2L, 1L)

    expect_identical(compact$north, dense$north)
    expect_equal(compact$X, dense$X, tolerance = 1e-10)
    expect_equal(compact$W_orth, dense$W_orth, tolerance = 1e-10)
    expect_equal(compact$P_orth, dense$P_orth, tolerance = 1e-10)
})

test_that("factor and character labels use the same compact public route", {
    task <- compact_label_fixture()
    character_labels <- as.character(task$labels)
    factor_fit <- pls(
        task$X, task$labels, ncomp = 1:3, method = "simpls",
        backend = "cpu", svd.method = "rsvd", seed = 73L,
        return_variance = FALSE
    )
    character_fit <- pls(
        task$X, character_labels, ncomp = 1:3, method = "simpls",
        backend = "cpu", svd.method = "rsvd", seed = 73L,
        return_variance = FALSE
    )

    factor_prediction <- predict(factor_fit, task$Xtest)$Ypred
    character_prediction <- predict(character_fit, task$Xtest)$Ypred
    expect_identical(character_prediction, factor_prediction)
    expect_equal(character_fit$R2Y, factor_fit$R2Y, tolerance = 1e-13)
})

test_that("dependency-free double PLS-SVD preserves compact predictions", {
    task <- compact_label_fixture()
    components <- c(1L, 2L)
    labels <- as.integer(task$labels)
    core <- fastPLS:::pls_labels_core_cpp(
        task$X, labels, nlevels(task$labels), components, 1L, TRUE,
        32L, 5L, 9501L
    )
    dense <- fastPLS:::pls_matrix_core_cpp(
        task$X, task$Y, components, 1L, TRUE, 1L, 32L, 5L, 9501L
    )

    core_prediction <- fastPLS:::pls_labels_core_predict_cpp(
        core, task$Xtest, FALSE
    )$Ypred
    dense_prediction <- fastPLS:::pls_labels_core_predict_cpp(
        dense, task$Xtest, FALSE
    )$Ypred
    expect_equal(core$mY, dense$mY, tolerance = 1e-13)
    expect_equal(
        as.numeric(core$R2Y), as.numeric(dense$R2Y), tolerance = 1e-10
    )
    for (index in seq_along(components)) {
        expect_equal(
            compact_fitted_slice(core, index),
            compact_fitted_slice(dense, index),
            tolerance = 1e-9
        )
    }
    expect_equal(core_prediction, dense_prediction, tolerance = 1e-9)

    public <- pls(
        task$X, task$labels, task$Xtest,
        ncomp = components,
        method = "plssvd",
        backend = "cpu",
        svd.method = "rsvd",
        fit = FALSE,
        return_variance = FALSE,
        oversample = 32L,
        power = 5L,
        seed = 9501L
    )
    expect_identical(public$xprod_mode, "float64_label_class_sums")
    expect_named(public$W_latent, paste0("ncomp=", components))
})

test_that("dependency-free double SIMPLS preserves compact predictions", {
    task <- compact_label_fixture()
    components <- 1:3
    labels <- as.integer(task$labels)
    core <- fastPLS:::pls_simpls_labels_core_cpp(
        task$X, labels, nlevels(task$labels), components, 1L, TRUE,
        32L, 5L, 9501L
    )
    dense <- fastPLS:::pls_matrix_core_cpp(
        task$X, task$Y, components, 1L, TRUE, 3L, 32L, 5L, 9501L
    )

    core_prediction <- fastPLS:::pls_labels_core_predict_cpp(
        core, task$Xtest, FALSE
    )$Ypred
    dense_prediction <- fastPLS:::pls_labels_core_predict_cpp(
        dense, task$Xtest, FALSE
    )$Ypred
    expect_equal(core$mY, dense$mY, tolerance = 1e-13)
    expect_equal(
        as.numeric(core$R2Y), as.numeric(dense$R2Y), tolerance = 1e-8
    )
    for (index in seq_along(components)) {
        expect_equal(
            compact_fitted_slice(core, index),
            compact_fitted_slice(dense, index),
            tolerance = 1e-8
        )
    }
    expect_equal(core_prediction, dense_prediction, tolerance = 1e-8)

    public <- pls(
        task$X, task$labels, task$Xtest,
        ncomp = components,
        method = "simpls",
        backend = "cpu",
        svd.method = "rsvd",
        fit = FALSE,
        return_variance = FALSE,
        oversample = 32L,
        power = 5L,
        seed = 9501L
    )
    expect_identical(
        public$xprod_mode, "float64_label_class_sums_blocked"
    )
})
