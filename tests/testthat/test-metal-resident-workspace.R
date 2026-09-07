test_that("Metal resident workspaces are isolated across dimensions and fits", {
    skip_if_not(has_metal(), "Metal backend not available")
    set.seed(912)
    X <- scale(matrix(rnorm(100 * 32), 100, 32), scale = FALSE)
    Y <- scale(matrix(rnorm(100 * 8), 100, 8), scale = FALSE)
    input <- serialize(list(X, Y), NULL)
    first <- fastPLS:::metal_simpls_resident_cpp(X, Y, 5L, 2L, 71L)
    smaller <- fastPLS:::metal_simpls_resident_cpp(
        X[, 1:12], Y[, 1:3], 3L, 2L, 72L
    )
    second <- fastPLS:::metal_simpls_resident_cpp(X, Y, 5L, 2L, 71L)
    expect_identical(serialize(list(X, Y), NULL), input)
    expect_equal(first, second, tolerance = 0)
    expect_equal(ncol(first$R), 5L)
    expect_equal(ncol(smaller$R), 3L)
    expect_true(all(is.finite(first$Q)))
})
