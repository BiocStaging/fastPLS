test_that("compiled float32 column operations preserve arithmetic and inputs", {
    set.seed(941)
    for (shape in list(c(1L, 17L), c(17L, 1L), c(7L, 11L))) {
        X <- float::fl(matrix(rnorm(prod(shape)), shape[1L], shape[2L]))
        initial <- serialize(X, NULL)
        stats <- runif(ncol(X), 0.1, 2)
        rng <- .Random.seed
        for (row in list(stats, float::fl(stats), float::fl(t(stats)))) {
            row_initial <- serialize(row, NULL)
            expanded <- float::fl(matrix(rep(as.numeric(row), each = nrow(X)), nrow(X)))
            for (op in c("-", "/", "+")) {
                expected <- switch(op, "-" = X - expanded, "/" = X / expanded, "+" = X + expanded)
                actual <- fastPLS:::.float32_sweep_cols(X, row, op)
                expect_identical(actual@Data, expected@Data)
                expect_identical(serialize(X, NULL), initial)
                expect_identical(serialize(row, NULL), row_initial)
            }
        }
        expect_identical(.Random.seed, rng)
    }
})

test_that("fused float32 standardization preserves separate rounding", {
    set.seed(942)
    for (shape in list(c(1L, 19L), c(19L, 1L), c(7L, 11L), c(0L, 3L))) {
        X <- float::fl(matrix(rnorm(prod(shape)), shape[1L], shape[2L]))
        mu <- rnorm(ncol(X))
        sd <- runif(ncol(X), 0.1, 2)
        initial <- serialize(X, NULL)
        rng <- .Random.seed
        for (center in list(mu, float::fl(mu), float::fl(t(mu)))) {
            for (scale in list(sd, float::fl(sd), float::fl(t(sd)))) {
                expected <- fastPLS:::.float32_sweep_cols(
                    fastPLS:::.float32_sweep_cols(X, center, "-"), scale, "/")
                actual <- fastPLS:::.float32_standardize(X, center, scale)
                expect_identical(actual@Data, expected@Data)
                expect_identical(serialize(X, NULL), initial)
            }
        }
        expect_identical(.Random.seed, rng)
    }
    X <- float::fl(matrix(c(NA, NaN, Inf, -Inf, 0, -0, 1, -1, 1e-38), 3L))
    mu <- float::fl(c(1, 0, 1e-30))
    sd <- float::fl(c(0, Inf, 1e-10))
    expected <- fastPLS:::.float32_sweep_cols(
        fastPLS:::.float32_sweep_cols(X, mu, "-"), sd, "/")
    expect_identical(float::dbl(fastPLS:::.float32_standardize(X, mu, sd)),
        float::dbl(expected))
    expect_error(fastPLS:::.float32_standardize(X, c(1, 2), sd), "ncol")
    expect_error(fastPLS:::.float32_standardize(X, mu, 1), "ncol")
    expect_error(fastPLS:::float32_standardize_cpp(X, 1:3, sd), "float32")
})

test_that("float32 zero workspaces are allocated directly as zero bits", {
    for (shape in list(c(0L, 3L), c(3L, 0L), c(1L, 1L), c(7L, 11L))) {
        actual <- fastPLS:::.float32_zeros(shape[1L], shape[2L])
        expect_identical(actual@Data, matrix(0L, shape[1L], shape[2L]))
        expect_identical(float::dbl(actual), matrix(0, shape[1L], shape[2L]))
    }
})

test_that("float32 broadcasting handles nonfinite values and dimension errors", {
    X <- float::fl(matrix(c(NA, NaN, Inf, -Inf, 0, -0, 1, -1, 1e-38), nrow = 3L))
    row <- float::fl(c(1, 0, 1e-30))
    expanded <- float::fl(matrix(rep(as.numeric(row), each = 3L), 3L))
    for (op in c("-", "/", "+")) {
        expected <- switch(op, "-" = X - expanded, "/" = X / expanded, "+" = X + expanded)
        actual <- fastPLS:::.float32_sweep_cols(X, row, op)
        expect_identical(float::dbl(actual), float::dbl(expected))
    }
    expect_error(fastPLS:::.float32_sweep_cols(X, c(1, 2)), "ncol")
    expect_error(fastPLS:::float32_sweep_cols_cpp(X, row, 3L), "Unknown")
    expect_error(fastPLS:::float32_sweep_cols_cpp(matrix(1), row, 0L), "float32")
    expect_identical(dim(fastPLS:::.float32_sweep_cols(
        float::fl(matrix(numeric(), 0L, 3L)), row)), c(0L, 3L))
})
