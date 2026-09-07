test_that("nonlinear kernel memory is rejected before allocation", {
    expect_error(
        fastPLS:::.kernel_pls_memory_guard(1000000L, 4, "test kernel"),
        "n-by-n Gram matrix"
    )
    expect_silent(
        fastPLS:::.kernel_pls_memory_guard(100L, 8, "test kernel")
    )
})
