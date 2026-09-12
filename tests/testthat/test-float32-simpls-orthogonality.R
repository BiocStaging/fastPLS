test_that("float32 SIMPLS retains score orthogonality on ill-conditioned paths", {
    skip_on_os("windows")
    set.seed(628)
    n <- 120L
    latent <- sweep(matrix(rnorm(n * 40L), n, 40L), 2L,
        10^seq(0, -3, length.out = 40L), "*")
    X <- float::fl(latent %*% matrix(rnorm(40L * 180L), 40L, 180L) +
        matrix(rnorm(n * 180L, sd = 1e-4), n, 180L))
    Y <- float::fl(latent %*% matrix(rnorm(40L * 110L), 40L, 110L) +
        matrix(rnorm(n * 110L, sd = 1e-4), n, 110L))
    backends <- c("cpu", if (has_metal()) "metal", if (has_cuda()) "cuda")
    for (backend in backends) {
        solvers <- if (backend == "cuda") "rsvd" else "rsvd"
        for (solver in solvers) {
            model <- suppressWarnings(pls(X, Y, ncomp = 30L, backend = backend, seed = 29,
                fit = TRUE, return_variance = FALSE))
            internal <- fastPLS:::.fastpls_restore_internal_output_fields(model)
            scores <- float::dbl(internal$Ttrain)
            gram <- crossprod(scores)
            expect_true(all(is.finite(scores)))
            error <- max(abs(gram - diag(ncol(scores))))
            expect_true(error < 5e-3, info = paste(backend, solver, error))
            expect_identical(as.integer(internal$ncomp), 30L)
            prediction <- float::dbl(
                predict(model, X, backend = backend)$Ypred[[1L]]
            )
            fitted <- float::dbl(model$Yfit[[1L]])
            relative_error <- sqrt(sum((prediction - fitted)^2) / sum(fitted^2))
            expect_true(relative_error < 1e-5,
                info = paste(backend, solver, relative_error))
        }
    }
})
