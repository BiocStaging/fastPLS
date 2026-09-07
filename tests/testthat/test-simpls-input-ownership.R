test_that("SIMPLS fitting and prediction leave shared input matrices unchanged", {
    set.seed(73)
    X <- matrix(rnorm(96 * 12), 96, 12) + 2
    Y <- cbind(X[, 1] + 3, X[, 2] - 4, X[, 3] + X[, 4])
    Xtest <- X[1:15, , drop = FALSE]
    original <- serialize(list(X, Y, Xtest), NULL)
    available <- c(cpu = TRUE, cuda = has_cuda())

    for (backend in names(available)[available]) {
        for (scaling in c("none", "centering", "autoscaling")) {
            fit <- pls(
                X, Y, ncomp = 3, scaling = scaling, backend = backend,
                svd.method = "rsvd", seed = 17, return_variance = FALSE
            )
            prediction <- predict(fit, Xtest, backend = backend)$Ypred
            expect_true(all(is.finite(prediction)))
            expect_identical(serialize(list(X, Y, Xtest), NULL), original)
        }
    }
})

test_that("CUDA classification batches fit inside a narrow sketch", {
    skip_if_not(has_cuda(), "CUDA backend is unavailable")
    set.seed(74)
    X <- matrix(rnorm(5000 * 1000), 5000, 1000)
    labels <- rep(seq_len(100L), length.out = nrow(X))
    fit <- fastPLS:::.with_fastpls_fast_options(
        fastPLS:::pls_model2_fast_gpu_labels(
            X, labels, 100L, 50L, 3L, FALSE,
            fastPLS:::.svd_method_id("cuda_rsvd"), 0L, 2L, 0, 17L
        )
    )
    expect_equal(ncol(fit$R), 50L)
    expect_true(all(is.finite(fit$R)))
    expect_true(all(is.finite(fit$Q)))
})
