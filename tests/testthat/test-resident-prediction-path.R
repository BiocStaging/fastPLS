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
