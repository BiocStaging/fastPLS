test_that("metal is the fixed operation-split float32 PLS backend", {
    expect_identical(
        fastPLS:::.fastpls_validate_backend("metal"),
        "metal"
    )
    expect_identical(fastPLS:::.float32_backend_id("metal"), 3L)
    expect_identical(fastPLS:::.float32_product_backend_id("metal"), 2L)
    expect_identical(
        fastPLS:::.resolve_classifier_for_backend("lda", "metal"),
        "lda_cpp"
    )
    expect_error(
        fastPLS:::.fastpls_validate_backend("metal_hybrid"),
        "must be one of"
    )
})

test_that("metal refuses float64 and never falls back", {
    skip_if_not(has_metal())
    set.seed(710)
    X <- matrix(rnorm(60 * 8), 60, 8)
    y <- factor(rep(letters[1:3], each = 20))
    expect_error(
        pls(X, y, ncomp = 2, backend = "metal"),
        "does not provide native float64"
    )
})

test_that("metal fits every PLS family with a fixed operation split", {
    skip_if_not_installed("float")
    skip_if_not(has_metal())
    set.seed(711)
    X <- float::fl(matrix(rnorm(72 * 10), 72, 10))
    y <- factor(rep(letters[1:3], each = 24))
    Xtest <- X[seq_len(12), , drop = FALSE]
    ytest <- y[seq_len(12)]

    for (family in c("plssvd", "simpls", "opls")) {
        fit <- suppressWarnings(pls(
            X, y, Xtest, ytest, ncomp = 2:3, method = family,
            backend = "metal", classifier = "lda",
            return_variance = FALSE
        ))
        expect_length(fit$accuracy, 2L)
        expect_true(all(is.finite(fit$accuracy)))
        expect_identical(
            fit$diagnostics$metal_operation_split$policy,
            "fixed_operation_split"
        )
    }

    route_fit <- suppressWarnings(pls(
        X, y, ncomp = 2, method = "simpls",
        backend = "metal", return_variance = FALSE
    ))
    internal <- attr(route_fit, "fastPLS_internal")
    expect_identical(
        internal$execution_route,
        "CPU/Metal hybrid (operation split)"
    )
    expect_null(internal$resident_state)
    expect_match(
        route_fit$diagnostics$metal_operation_split$metal_operations,
        "training sample matrix"
    )

    for (kernel in c("linear", "rbf", "poly")) {
        fit <- suppressWarnings(pls(
            X[seq_len(48), , drop = FALSE], y[seq_len(48)],
            Xtest, ytest, ncomp = 2, method = "kernelpls",
            kernel = kernel, backend = "metal",
            return_variance = FALSE
        ))
        expect_true(is.finite(fit$accuracy[[1L]]))
    }
})

test_that("operation-split metal supports grouped cross-validation", {
    skip_if_not_installed("float")
    skip_if_not(has_metal())
    set.seed(712)
    X <- float::fl(matrix(rnorm(60 * 8), 60, 8))
    y <- factor(rep(letters[1:3], each = 20))
    groups <- rep(seq_len(30), each = 2)
    fit <- suppressWarnings(pls.single.cv(
        X, y, constrain = groups, ncomp = 1:2, kfold = 3,
        backend = "metal", fit = FALSE
    ))
    expect_identical(fit$backend, "metal")
    expect_true(fit$best_ncomp %in% 1:2)
    expect_identical(fit$fold[seq(1, 60, by = 2)], fit$fold[seq(2, 60, by = 2)])

    nested <- suppressWarnings(pls.double.cv(
        X, y, constrain = groups, ncomp = 1:2,
        kfold_inner = 2, kfold_outer = 2, runn = 1,
        backend = "metal", seed = 10
    ))
    expect_identical(nested$backend, "metal")
    expect_true(all(is.finite(nested$Q2Y)))
})
