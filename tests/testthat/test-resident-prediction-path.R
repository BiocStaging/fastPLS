test_that("resident CUDA predicts a complete component path consistently", {
    skip_if_not(fastPLS::has_cuda())

    set.seed(721)
    X <- matrix(rnorm(180 * 24), 180, 24)
    y <- factor(rep(letters[1:4], length.out = nrow(X)))
    path <- c(1L, 3L, 5L)

    for (float32 in c(FALSE, TRUE)) {
        input <- if (float32) float::fl(X) else X
        fit <- pls(
            input, y, ncomp = path, method = "simpls",
            backend = "cuda", fit = FALSE, return_variance = FALSE,
            seed = 11
        )
        object <- fastPLS:::.fastpls_restore_internal_output_fields(fit)
        bits <- fastPLS:::.resident_cuda_input(
            input[seq_len(25), ], object$precision, "newdata"
        )
        actual <- fastPLS:::cuda_resident_predict_path_cpp(
            object$resident_state, bits, path, 0L
        )

        for (index in seq_along(path)) {
            expected <- fastPLS:::cuda_resident_predict_path_cpp(
                object$resident_state, bits, path[[index]], 0L
            )
            expected <- fastPLS:::.resident_cuda_summary(
                expected[, , 1L, drop = TRUE], object$precision
            )
            observed <- fastPLS:::.resident_cuda_summary(
                actual[, , index], object$precision
            )
            expect_equal(observed, expected, tolerance = if (float32) {
                2e-6
            } else {
                1e-12
            })
        }
    }
})

test_that("resident CUDA reuses one projection for CV labels and responses", {
    skip_if_not(fastPLS::has_cuda())

    set.seed(724)
    X <- matrix(rnorm(220 * 28), 220, 28)
    y <- factor(rep(letters[1:5], length.out = nrow(X)))
    path <- c(1L, 2L, 4L)
    for (float32 in c(FALSE, TRUE)) {
        input <- if (float32) float::fl(X) else X
        fit <- pls(
            input, y, ncomp = path, method = "simpls",
            backend = "cuda", classifier = "lda", fit = FALSE,
            return_variance = FALSE, seed = 16
        )
        object <- fastPLS:::.fastpls_restore_internal_output_fields(fit)
        bits <- fastPLS:::.resident_cuda_input(
            input[seq_len(37), ], object$precision, "newdata"
        )
        combined <- fastPLS:::cuda_resident_classify_response_path_cpp(
            object$resident_state, bits, path, 1L, 1L
        )
        labels <- fastPLS:::cuda_resident_classify_path_cpp(
            object$resident_state, bits, path, 1L, 1L
        )
        responses <- fastPLS:::cuda_resident_predict_path_cpp(
            object$resident_state, bits, path, 0L
        )
        expect_identical(combined$labels, labels)
        for (index in seq_along(path)) {
            expect_equal(
                fastPLS:::.resident_cuda_summary(
                    combined$predictions[, , index], object$precision
                ),
                fastPLS:::.resident_cuda_summary(
                    responses[, , index], object$precision
                ),
                tolerance = if (float32) 2e-6 else 1e-12
            )
        }
    }
})

test_that("constrained Metal CV retains groups and distinct LDA labels", {
    skip_if_not(fastPLS::has_metal())

    set.seed(723)
    X64 <- matrix(rnorm(240 * 18), 240, 18)
    y <- factor(max.col(X64[, seq_len(4)] +
        matrix(rnorm(240 * 4, sd = 0.4), 240, 4)))
    groups <- rep(seq_len(80), each = 3L)
    common <- list(
        Xdata = float::fl(X64), Ydata = y, constrain = groups,
        ncomp = c(1L, 3L), kfold = 4L, method = "simpls",
        backend = "metal", fit = FALSE, seed = 15
    )
    argmax <- do.call(pls.single.cv, c(common, list(classifier = "argmax")))
    lda <- do.call(pls.single.cv, c(common, list(classifier = "lda")))

    expect_true(all(vapply(
        split(argmax$fold, groups),
        function(value) length(unique(value)) == 1L,
        logical(1L)
    )))
    expect_equal(argmax$Q2Y, lda$Q2Y, tolerance = 2e-6)
    expect_true(any(
        as.character(argmax$pred[[1L]]) != as.character(lda$pred[[1L]])
    ))
})

test_that("constrained CUDA CV retains groups and distinct LDA labels", {
    skip_if_not(fastPLS::has_cuda())

    set.seed(726)
    X64 <- matrix(rnorm(240 * 18), 240, 18)
    y <- factor(max.col(X64[, seq_len(4)] +
        matrix(rnorm(240 * 4, sd = 0.4), 240, 4)))
    groups <- rep(seq_len(80), each = 3L)
    common <- list(
        Xdata = float::fl(X64), Ydata = y, constrain = groups,
        ncomp = c(1L, 3L), kfold = 4L, method = "simpls",
        backend = "cuda", fit = FALSE, seed = 15
    )
    argmax <- do.call(pls.single.cv, c(common, list(classifier = "argmax")))
    lda <- do.call(pls.single.cv, c(common, list(classifier = "lda")))

    expect_true(all(vapply(
        split(argmax$fold, groups),
        function(value) length(unique(value)) == 1L,
        logical(1L)
    )))
    expect_equal(argmax$Q2Y, lda$Q2Y, tolerance = 2e-6)
    expect_true(any(
        as.character(argmax$pred[[1L]]) != as.character(lda$pred[[1L]])
    ))
})

test_that("resident CUDA CV is deterministic and reports full residency", {
    skip_if_not(isTRUE(fastPLS::has_cuda()), "CUDA backend unavailable")
    set.seed(812)
    X <- float::fl(matrix(rnorm(180 * 24), 180, 24))
    y <- factor(rep(LETTERS[1:3], each = 60))
    resident <- pls.single.cv(
        X, y, ncomp = c(2L, 5L), kfold = 3L, method = "simpls",
        backend = "cuda", classifier = "lda", fit = FALSE, seed = 19L
    )
    repeated <- pls.single.cv(
        X, y, ncomp = c(2L, 5L), kfold = 3L, method = "simpls",
        backend = "cuda", classifier = "lda", fit = FALSE, seed = 19L
    )

    expect_identical(resident$fold, repeated$fold)
    expect_identical(
        lapply(resident$pred, as.character),
        lapply(repeated$pred, as.character)
    )
    expect_equal(resident$Q2Y, repeated$Q2Y, tolerance = 1e-6)
    expect_identical(resident$residency$fold_gather, "resident cuda")
    expect_identical(resident$residency$fallback, "none")
})

test_that("resident CUDA regression CV is deterministic", {
    skip_if_not(isTRUE(fastPLS::has_cuda()), "CUDA backend unavailable")
    set.seed(913)
    X64 <- matrix(rnorm(156 * 27), 156, 27)
    coefficients <- matrix(rnorm(27 * 3), 27, 3)
    Y64 <- X64 %*% coefficients + matrix(rnorm(156 * 3, sd = 0.25), 156, 3)
    X <- float::fl(X64)
    Y <- float::fl(Y64)
    common <- list(
        Xdata = X, Ydata = Y, ncomp = c(2L, 5L), kfold = 3L,
        method = "simpls", backend = "cuda", fit = FALSE, seed = 23L
    )

    resident <- do.call(pls.single.cv, common)
    repeated <- do.call(pls.single.cv, common)

    expect_identical(resident$fold, repeated$fold)
    expect_identical(resident$best_ncomp, repeated$best_ncomp)
    expect_equal(resident$Ypred, repeated$Ypred, tolerance = 2e-6)
    expect_equal(resident$Q2Y, repeated$Q2Y, tolerance = 2e-6)
    expect_equal(resident$RMSD, repeated$RMSD, tolerance = 1e-6)
    expect_identical(resident$residency$fold_gather, "resident cuda")
    expect_identical(resident$residency$fallback, "none")
})

test_that("PLS-SVD CUDA CV residency is limited to wide regression", {
    route <- fastPLS:::.cuda_resident_cv_route
    expect_true(route("cuda", "simpls", TRUE, 100L, 3L, 4L))
    expect_false(route("cuda", "plssvd", TRUE, 1000000L, 1000L, 4L))
    expect_false(route("cuda", "plssvd", FALSE, 1000L, 100L, 4L))
    expect_true(route("cuda", "plssvd", FALSE, 1200L, 28355L, 4L))
    expect_false(route("cpu", "plssvd", FALSE, 1200L, 28355L, 4L))
})

test_that("resident PLS-SVD CUDA CV kernels are deterministic", {
    skip_if_not(isTRUE(fastPLS::has_cuda()), "CUDA backend unavailable")
    set.seed(914)
    X64 <- matrix(rnorm(192 * 30), 192, 30)
    y <- factor(max.col(
        X64[, seq_len(4)] + matrix(rnorm(192 * 4, sd = 0.3), 192, 4)
    ))
    X <- float::fl(X64)
    folds <- rep(seq_len(4L), length.out = nrow(X64))
    predictors <- fastPLS:::.resident_cuda_input(X, "float32", "Xdata")
    classification <- fastPLS:::cuda_resident_simpls_cv_classification_cpp(
        predictors = predictors,
        labels = as.integer(y),
        class_count = nlevels(y),
        folds = folds,
        components = c(1L, 3L),
        scaling = 1L,
        classifier = 1L,
        oversample = 20L,
        power = 2L,
        seed = 29L,
        method = 1L
    )
    repeated <- fastPLS:::cuda_resident_simpls_cv_classification_cpp(
        predictors = predictors,
        labels = as.integer(y),
        class_count = nlevels(y),
        folds = folds,
        components = c(1L, 3L),
        scaling = 1L,
        classifier = 1L,
        oversample = 20L,
        power = 2L,
        seed = 29L,
        method = 1L
    )

    expect_identical(classification$fold, repeated$fold)
    expect_identical(classification$class_pred, repeated$class_pred)
    expect_equal(classification$Ypred, repeated$Ypred, tolerance = 2e-6)
    expect_equal(classification$metric_value, repeated$metric_value,
        tolerance = 1e-12)

    coefficients <- matrix(rnorm(30 * 6), 30, 6)
    Y <- float::fl(
        X64 %*% coefficients + matrix(rnorm(192 * 6, sd = 0.2), 192, 6)
    )
    responses <- fastPLS:::.resident_cuda_input(Y, "float32", "Ydata")
    regression <- fastPLS:::cuda_resident_simpls_cv_regression_cpp(
        predictors = predictors,
        responses = responses,
        folds = folds,
        components = c(2L, 4L),
        scaling = 1L,
        metric = 4L,
        oversample = 20L,
        power = 2L,
        seed = 31L,
        method = 1L
    )
    regression_repeated <- fastPLS:::cuda_resident_simpls_cv_regression_cpp(
        predictors = predictors,
        responses = responses,
        folds = folds,
        components = c(2L, 4L),
        scaling = 1L,
        metric = 4L,
        oversample = 20L,
        power = 2L,
        seed = 31L,
        method = 1L
    )

    expect_identical(regression$fold, regression_repeated$fold)
    expect_equal(regression$Ypred, regression_repeated$Ypred, tolerance = 2e-6)
    expect_equal(regression$Q2Y, regression_repeated$Q2Y, tolerance = 2e-6)
    expect_equal(regression$RMSD, regression_repeated$RMSD, tolerance = 1e-6)
})

test_that("CPU float64 classification Q2 is independent of prediction head", {
    set.seed(431)
    X <- matrix(rnorm(96 * 14), 96, 14)
    labels <- factor(rep(c("a", "b", "c"), each = 32))
    train <- seq_len(72)
    test <- setdiff(seq_len(nrow(X)), train)

    for (method in c("plssvd", "simpls", "opls", "kernelpls")) {
        extra <- if (identical(method, "kernelpls")) {
            list(kernel = "rbf")
        } else {
            list()
        }
        common <- c(list(
            Xtrain = X[train, , drop = FALSE],
            Ytrain = labels[train],
            Xtest = X[test, , drop = FALSE],
            Ytest = labels[test],
            ncomp = 1:2,
            method = method,
            backend = "cpu"
        ), extra)
        argmax <- do.call(pls, c(common, list(classifier = "argmax")))
        lda <- do.call(pls, c(common, list(classifier = "lda")))

        expect_true(all(is.finite(lda$Q2Y)), info = method)
        expect_equal(lda$Q2Y, argmax$Q2Y, tolerance = 1e-10, info = method)

        predicted <- predict(
            lda,
            X[test, , drop = FALSE],
            Ytest = labels[test]
        )
        expect_true(all(is.finite(predicted$Q2Y)), info = method)
        expect_equal(predicted$Q2Y, argmax$Q2Y,
            tolerance = 1e-10, info = method)
    }
})
