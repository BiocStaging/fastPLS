test_that("CPU rSVD recovery rechecks slow spectra without another solver", {
    set.seed(614)
    U <- qr.Q(qr(matrix(rnorm(160L * 100L), 160L, 100L)))
    V <- qr.Q(qr(matrix(rnorm(125L * 100L), 125L, 100L)))
    singular <- 1 - 0.005 * (0:99)
    A <- U %*% (singular * t(V))
    precisions <- if (.Platform$OS.type == "windows") "float64" else
        c("float64", "float32")
    for (precision in precisions) {
        X <- if (precision == "float32") float::fl(A) else A
        for (seed in c(1L, 7L)) {
            out <- suppressWarnings(fastsvd(X, ncomp = 6L, backend = "cpu",
                method = "rsvd", oversample = 0L, power = 0L, seed = seed))
            audit <- out$diagnostics$rsvd_case_audit
            expect_true(audit$certified)
            expect_false(audit$deterministic_fallback)
            expect_gt(audit$attempts, 3L)
            expect_gt(audit$effective_oversample, 48L)
            expect_gte(audit$effective_power, 6L)
            expect_false(identical(as.numeric(audit$effective_seed), as.numeric(seed)))
            expect_lte(audit$triplet_residual, 1e-2)
            expect_lte(audit$omitted_direction_ratio, 1.01)
            expect_length(out$d, 6L)
            expect_equal(as.numeric(out$d), singular[1:6], tolerance = 2e-5)
            u <- if (precision == "float32") float::dbl(out$u) else out$u
            v <- if (precision == "float32") float::dbl(out$v) else out$v
            tested <- if (precision == "float32") float::dbl(X) else X
            residual <- vapply(seq_len(6L), function(j) max(
                sqrt(sum((tested %*% v[, j] - out$d[j] * u[, j])^2)),
                sqrt(sum((crossprod(tested, u[, j]) - out$d[j] * v[, j])^2))
                ) / out$d[1L], numeric(1))
            expect_lte(max(residual), 2e-5)
        }
    }
})
