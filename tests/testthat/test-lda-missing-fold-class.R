test_that("compiled LDA CV compacts classes absent from a training fold", {
    set.seed(104)
    X <- matrix(rnorm(31 * 8), 31, 8)
    y <- factor(c("rare", rep("class_b", 15), rep("class_c", 15)))

    backends <- c(
        "cpu",
        if (isTRUE(has_cuda())) "cuda",
        if (isTRUE(has_metal())) "metal"
    )

    for (backend in backends) {
        methods <- if (backend == "cpu") {
            c("plssvd", "simpls", "opls", "kernelpls")
        } else {
            c("plssvd", "simpls", "kernelpls")
        }
        X_backend <- if (backend == "metal") float::fl(X) else X
        for (method in methods) {
            result <- suppressWarnings(pls.single.cv(
                X_backend,
                y,
                ncomp = 1:2,
                kfold = 5,
                method = method,
                backend = backend,
                svd.method = "rsvd",
                classifier = "lda",
                fit = FALSE,
                seed = 104
            ))

            expect_true(all(is.finite(result$accuracy)))
            expect_length(result$Ypred_optim, nrow(X))
            expect_true(all(result$Ypred_optim %in% levels(y)))
        }
    }
})

test_that("PLS-LDA drops unused factor levels before fitting", {
    set.seed(105)
    X <- matrix(rnorm(24 * 6), 24, 6)
    y <- factor(
        rep(c("class_a", "class_b"), each = 12),
        levels = c("class_a", "class_b", "unused")
    )

    fit <- pls(
        X,
        y,
        ncomp = 1:2,
        method = "simpls",
        backend = "cpu",
        svd.method = "rsvd",
        classifier = "lda",
        seed = 105
    )

    expect_equal(fit$lev, c("class_a", "class_b"))
    expect_true(all(fit$Ypred[[2L]] %in% fit$lev))
})

test_that("nested LDA CV retains rare-class holdouts without failing", {
    set.seed(106)
    X <- matrix(rnorm(31 * 6), 31, 6)
    y <- factor(c("rare", rep("class_b", 15), rep("class_c", 15)))

    result <- pls.double.cv(
        X,
        y,
        ncomp = 1:2,
        kfold_inner = 3,
        kfold_outer = 5,
        runn = 1,
        method = "simpls",
        backend = "cpu",
        svd.method = "rsvd",
        classifier = "lda",
        seed = 106
    )

    expect_true(all(is.finite(result$accuracy)))
    expect_length(result$Ypred, nrow(X))
    expect_true(all(result$Ypred %in% levels(y)))
})
