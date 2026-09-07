test_that("Metal products refresh cached values and handle shape changes", {
    skip_if_not(has_metal(), "Metal backend not available")
    set.seed(128)
    multiply <- fastPLS:::metal_matrix_multiply_cpp
    cross <- fastPLS:::metal_crossprod_cpp
    A <- matrix(rnorm(11 * 7), 11, 7)
    B <- matrix(rnorm(7 * 5), 7, 5)
    input <- serialize(list(A, B), NULL)
    first <- multiply(A, B)
    expect_equal(first, A %*% B, tolerance = 1e-5)
    expect_equal(multiply(A + 2, B - 3), (A + 2) %*% (B - 3), tolerance = 1e-5)
    for (columns in 1:12) {
        next_B <- matrix(rnorm(7 * columns), 7, columns)
        expect_equal(multiply(A, next_B), A %*% next_B, tolerance = 1e-5)
    }
    expect_equal(cross(A, A), crossprod(A), tolerance = 1e-5)
    expect_identical(multiply(A, B), first)
    expect_identical(serialize(list(A, B), NULL), input)
    expect_error(multiply(A, B[-1, ]), "non-conformable")
    expect_identical(multiply(A, B), first)
    invisible(multiply(matrix(NaN, 11, 7), B))
    expect_equal(multiply(matrix(0, 11, 7), B), matrix(0, 11, 5))
    expect_equal(multiply(matrix(numeric(), 11, 0), matrix(numeric(), 0, 5)),
                 matrix(0, 11, 5))
    tail_A <- matrix(rnorm(37 * 43), 37, 43)
    tail_B <- matrix(rnorm(43 * 35), 43, 35)
    expect_equal(multiply(tail_A, tail_B), tail_A %*% tail_B, tolerance = 1e-5)
})

test_that("Metal float32 workspaces preserve transpose and precision dispatch", {
    skip_if_not(has_metal(), "Metal backend not available")
    set.seed(129)
    for (left in c(FALSE, TRUE)) {
        for (right in c(FALSE, TRUE)) {
            X <- matrix(rnorm(9 * 5), 9, 5)
            Y <- matrix(rnorm(5 * 7), 5, 7)
            A <- float::fl(if (left) t(X) else X)
            B <- float::fl(if (right) t(Y) else Y)
            input <- serialize(list(A, B), NULL)
            product <- function(A, B) {
                out <- fastPLS:::metal_float32_matrix_multiply_cpp(
                    A, B, transpose_left = left, transpose_right = right
                )
                fastPLS:::.float32_from_bits(out$C)
            }
            first <- product(A, B)
            expect_s4_class(first, "float32")
            expect_equal(float::dbl(first), X %*% Y, tolerance = 1e-5)
            zero <- float::fl(matrix(0, nrow(A), ncol(A)))
            expect_equal(float::dbl(product(zero, B)), matrix(0, 9, 7))
            expect_equal(float::dbl(product(A, B)), float::dbl(first), tolerance = 0)
            expect_identical(serialize(list(A, B), NULL), input)
        }
    }
})
