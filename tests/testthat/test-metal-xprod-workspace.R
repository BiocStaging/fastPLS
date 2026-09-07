test_that("resident Metal cross-products retain values, precision and ownership", {
    skip_if_not(has_metal(), "Metal backend not available")
    set.seed(947)
    X <- matrix(rnorm(73L * 41L), 73L, 41L)
    Y <- matrix(rnorm(73L * 57L), 73L, 57L)
    initial <- serialize(list(X, Y), NULL)
    rng <- .Random.seed
    workspace <- fastPLS:::metal_xprod_workspace_cpp(X, Y)
    on.exit(fastPLS:::metal_xprod_workspace_release_cpp(workspace), add = TRUE)
    expect_identical(.Random.seed, rng)
    for (width in c(1L, 4L, 17L, 4L, 0L)) {
        B <- matrix(sin(seq_len(ncol(Y) * width)), ncol(Y), width)
        C <- matrix(cos(seq_len(ncol(X) * width)), ncol(X), width)
        forward <- fastPLS:::.metal_xprod_multiply(X, Y, B)
        reverse <- fastPLS:::.metal_xprod_transpose_multiply(X, Y, C)
        expect_equal(fastPLS:::metal_xprod_workspace_multiply_cpp(workspace, B, FALSE),
            forward, tolerance = 0)
        expect_equal(fastPLS:::metal_xprod_workspace_multiply_cpp(workspace, C, TRUE),
            reverse, tolerance = 0)
    }
    expect_identical(serialize(list(X, Y), NULL), initial)
    expect_identical(.Random.seed, rng)
    expect_error(fastPLS:::metal_xprod_workspace_multiply_cpp(workspace,
        matrix(0, 3L, 1L), FALSE), "non-conformable")
    other <- fastPLS:::metal_xprod_workspace_cpp(X + 1, Y - 2)
    on.exit(fastPLS:::metal_xprod_workspace_release_cpp(other), add = TRUE)
    B <- matrix(1, ncol(Y), 3L)
    expect_equal(fastPLS:::metal_xprod_workspace_multiply_cpp(other, B, FALSE),
        fastPLS:::.metal_xprod_multiply(X + 1, Y - 2, B), tolerance = 0)
    expect_equal(fastPLS:::metal_xprod_workspace_multiply_cpp(workspace, B, FALSE),
        fastPLS:::.metal_xprod_multiply(X, Y, B), tolerance = 0)
    fastPLS:::metal_xprod_workspace_release_cpp(workspace)
    expect_error(fastPLS:::metal_xprod_workspace_multiply_cpp(workspace, B, FALSE),
        "released")
    expect_error(fastPLS:::metal_xprod_workspace_release_cpp(NULL), "Invalid")
    expect_error(fastPLS:::metal_xprod_workspace_cpp(X[-1, ], Y), "matching rows")
})

test_that("Metal matrix-free rSVD retains its seed and direction calculation", {
    skip_if_not(has_metal(), "Metal backend not available")
    set.seed(948)
    X <- matrix(rnorm(83L * 97L), 83L, 97L)
    Y <- matrix(rnorm(83L * 121L), 83L, 121L)
    for (power in c(0L, 2L)) {
        for (seed in c(56L, 57L)) {
            set.seed(seed)
            omega <- matrix(rnorm(ncol(Y) * 12L), ncol(Y), 12L)
            values <- fastPLS:::.metal_xprod_multiply(X, Y, omega)
            for (iteration in seq_len(power)) {
                left <- qr.Q(qr(values))
                right <- qr.Q(qr(fastPLS:::.metal_xprod_transpose_multiply(X, Y, left)))
                values <- fastPLS:::.metal_xprod_multiply(X, Y, right)
            }
            basis <- qr.Q(qr(values))
            reduced <- t(fastPLS:::.metal_xprod_transpose_multiply(X, Y, basis))
            small <- svd(reduced, nu = 5L, nv = 5L)
            rng <- .Random.seed
            actual <- fastPLS:::.truncated_rsvd_metal_xprod(X, Y, 5L, 7L, power, seed)
            reference <- basis %*% small$u
            # Blocked LAPACK QR and LINPACK QR can rotate/flip bases. Compare
            # the reconstructed operator and score subspace, not vector signs.
            reconstructed <- actual$U %*% (actual$s * actual$Vt)
            expected <- reference %*% (small$d[1:5] * t(small$v))
            expect_lt(norm(reconstructed - expected, "F") / norm(expected, "F"), 1e-5)
            expect_lt(norm(tcrossprod(actual$U) - tcrossprod(reference), "F"), 1e-5)
            expect_equal(actual$s, small$d[1:5], tolerance = 1e-6)
            expect_identical(.Random.seed, rng)
        }
    }
})

test_that("compiled Metal matrix-free decomposition handles deficient shapes", {
    skip_if_not(has_metal(), "Metal backend not available")
    set.seed(950)
    for (shape in list(c(13L, 31L, 19L), c(31L, 7L, 5L))) {
        X <- matrix(rnorm(shape[1L] * shape[2L]), shape[1L], shape[2L])
        Y <- matrix(rnorm(shape[1L] * shape[3L]), shape[1L], shape[3L])
        X[, 2L] <- X[, 1L]
        Y[, 2L] <- Y[, 1L]
        for (left_only in c(FALSE, TRUE)) {
            result <- fastPLS:::.truncated_rsvd_metal_xprod(X, Y, 3L, 4L, 2L, 4L, left_only)
            expect_true(all(is.finite(result$U)))
            expect_equal(crossprod(result$U), diag(3), tolerance = 1e-6)
            expect_length(result$s, 3L)
            expect_identical(is.null(result$Vt), left_only)
        }
    }
    expect_error(fastPLS:::metal_xprod_rsvd_cpp(X, Y,
        matrix(1, ncol(Y), 2), 3L, 0L, FALSE), "Invalid")
    X[1, 1] <- NaN
    expect_error(fastPLS:::.truncated_rsvd_metal_xprod(X, Y, 3L), "QR factorization")
})

test_that("unavailable Metal workspace allocation fails without a CPU fallback", {
    skip_if(has_metal(), "Metal is available")
    expect_error(fastPLS:::metal_xprod_workspace_cpp(matrix(1, 2, 3),
        matrix(1, 2, 4)), "no CPU fallback")
})
