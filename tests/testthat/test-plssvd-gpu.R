test_that("pls backend='cuda' requires CUDA or returns fastPLS output", {
  set.seed(20260419)
  X <- matrix(rnorm(90 * 16), nrow = 90, ncol = 16)
  y <- factor(sample(letters[1:6], 90, replace = TRUE))
  idx <- sample(seq_len(90), 20)

  if (!has_cuda()) {
    expect_error(
      pls(
        X[-idx, , drop = FALSE], y[-idx],
        X[idx, , drop = FALSE], y[idx],
        ncomp = 1:3, method = "plssvd", backend = "cuda"
      ),
      "CUDA-enabled"
    )
  } else {
    gpu_fit <- pls(
      X[-idx, , drop = FALSE],
      y[-idx],
      X[idx, , drop = FALSE],
      y[idx],
      ncomp = 1:3,
      method = "plssvd",
      backend = "cuda",
      fit = TRUE,
      seed = 77L
    )
    cpu_fit <- pls(
      X[-idx, , drop = FALSE],
      y[-idx],
      X[idx, , drop = FALSE],
      y[idx],
      ncomp = 1:3,
      method = "plssvd",
      svd.method = "cpu_rsvd",
      fit = TRUE,
      seed = 77L
    )

    expect_s3_class(gpu_fit, "fastPLS")
    expect_null(gpu_fit$B)
    expect_equal(dim(gpu_fit$R), dim(cpu_fit$R))
    expect_true(all(is.finite(gpu_fit$R)))
    expect_true(is.data.frame(gpu_fit$Ypred))
    expect_equal(mean(gpu_fit$Ypred[[3]] == y[idx]), mean(cpu_fit$Ypred[[3]] == y[idx]), tolerance = 0.1)
  }
})

test_that("the direct CUDA matrix-product ABI matches CPU multiplication", {
  skip_if_not(has_cuda(), "CUDA backend is unavailable")
  left <- matrix(as.double(seq_len(12)), 3, 4)
  right <- matrix(as.double(seq_len(20)), 4, 5)
  multiply <- get(
    ".cuda_matmul", envir = asNamespace("fastPLS"), inherits = FALSE
  )
  expect_equal(multiply(left, right), left %*% right, tolerance = 1e-12)
})
