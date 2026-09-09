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
