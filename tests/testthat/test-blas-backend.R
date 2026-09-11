test_that("the compiled CPU numerical library is reported", {
    backend <- fastPLS_blas()

    expect_type(backend, "character")
    expect_length(backend, 1L)
    expect_true(backend %in% c("Accelerate", "OpenBLAS", "R BLAS/LAPACK"))
})
