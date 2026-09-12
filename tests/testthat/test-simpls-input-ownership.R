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
                X, Y, ncomp = 3, scaling = scaling, backend = backend, seed = 17, return_variance = FALSE
            )
            prediction <- predict(fit, Xtest, backend = backend)$Ypred
            expect_true(all(is.finite(prediction)))
            expect_identical(serialize(list(X, Y, Xtest), NULL), original)
        }
    }
})
