test_that("kernel PLS C++ wrapper predicts classification labels", {
  set.seed(2201)
  X <- matrix(rnorm(72 * 10), nrow = 72, ncol = 10)
  y <- factor(sample(c("A", "B", "C"), 72, replace = TRUE))
  idx <- seq_len(12)

  fit_cpp <- pls(
    X[-idx, , drop = FALSE],
    y[-idx],
    X[idx, , drop = FALSE],
    y[idx],
    ncomp = 1:2,
    method = "kernelpls",
    backend = "cpp",
    kernel = "rbf",
    svd.method = "cpu_rsvd"
  )

  expect_s3_class(fit_cpp, "fastPLSKernel")
  expect_true(is.data.frame(fit_cpp$Ypred))
  expect_equal(nrow(fit_cpp$Ypred), length(idx))
})

test_that("linear kernel PLS retains internal fields for later double prediction", {
  set.seed(2205)
  X <- matrix(rnorm(84 * 9), nrow = 84, ncol = 9)
  y <- factor(sample(c("A", "B", "C"), 84, replace = TRUE))
  idx <- seq_len(14)

  fit <- pls(
    X[-idx, , drop = FALSE], y[-idx], X[idx, , drop = FALSE], y[idx],
    ncomp = 1:2, method = "kernelpls", kernel = "linear",
    backend = "cpu", svd.method = "rsvd", seed = 2205
  )

  # Supplying Xtest at fit time and calling predict() again must both work.
  expect_true(is.data.frame(fit$Ypred))
  expect_equal(nrow(fit$Ypred), length(idx))
  again <- predict(fit, X[idx, , drop = FALSE], Ytest = y[idx])
  expect_true(is.data.frame(again$Ypred))
  expect_equal(nrow(again$Ypred), length(idx))
})

test_that("kernelpls high-level wrapper dispatches to simpls", {
  set.seed(2203)
  X <- matrix(rnorm(60 * 8), nrow = 60, ncol = 8)
  y <- factor(sample(c("low", "high"), 60, replace = TRUE))
  idx <- seq_len(10)

  fit_fast <- pls(
    X[-idx, , drop = FALSE],
    y[-idx],
    X[idx, , drop = FALSE],
    y[idx],
    ncomp = 1:2,
    method = "kernelpls",
    backend = "cpp",
    kernel = "rbf",
    svd.method = "cpu_rsvd"
  )

  expect_s3_class(fit_fast, "fastPLSKernel")
  expect_identical(attr(fit_fast$inner_model, "fastPLS_internal")$pls_method, "simpls")
})

test_that("OPLS C++ wrapper predicts regression matrices", {
  set.seed(2202)
  X <- matrix(rnorm(70 * 11), nrow = 70, ncol = 11)
  Y <- cbind(rnorm(70), rnorm(70))
  idx <- seq_len(10)

  fit_cpp <- pls(
    X[-idx, , drop = FALSE],
    Y[-idx, , drop = FALSE],
    X[idx, , drop = FALSE],
    Y[idx, , drop = FALSE],
    ncomp = 1:2,
    method = "opls",
    backend = "cpp",
    north = 1L,
    svd.method = "cpu_rsvd"
  )

  expect_s3_class(fit_cpp, "fastPLSOpls")
  expect_true(is.array(fit_cpp$Ypred))
  expect_equal(dim(fit_cpp$Ypred), c(length(idx), ncol(Y), 2L))
})

test_that("opls high-level wrapper dispatches to simpls", {
  set.seed(2204)
  X <- matrix(rnorm(64 * 9), nrow = 64, ncol = 9)
  Y <- matrix(rnorm(64), ncol = 1)
  idx <- seq_len(12)

  fit <- pls(
    X[-idx, , drop = FALSE],
    Y[-idx, , drop = FALSE],
    X[idx, , drop = FALSE],
    Y[idx, , drop = FALSE],
    ncomp = 1:2,
    method = "opls",
    backend = "cpp",
    north = 1L,
    svd.method = "cpu_rsvd"
  )

  expect_s3_class(fit, "fastPLSOpls")
  expect_identical(attr(fit$inner_model, "fastPLS_internal")$pls_method, "simpls")
  expect_equal(dim(fit$Ypred), c(length(idx), ncol(Y), 2L))
})

test_that("double OPLS filtering uses the standalone matrix boundary", {
  values <- matrix(seq_len(20), nrow = 5L, ncol = 4L) / 7
  center <- c(0.2, -0.1, 0.4, 0.3)
  scale <- c(1.1, 0.8, 1.4, 0.9)
  weights <- matrix(c(0.3, -0.2, 0.4, 0.1), ncol = 1L)
  loadings <- matrix(c(0.2, 0.3, -0.1, 0.25), ncol = 1L)
  expected <- sweep(values, 2L, center, "-")
  expected <- sweep(expected, 2L, scale, "/")
  expected <- expected - (expected %*% weights) %*% t(loadings)

  expect_equal(
    fastPLS:::opls_apply_filter_cpp(
      values, center, scale, weights, loadings
    ),
    expected,
    tolerance = 1e-12
  )
  expect_error(
    fastPLS:::opls_apply_filter_cpp(
      values, center[-1L], scale, weights, loadings
    ),
    "stored OPLS preprocessing"
  )
})

test_that("standalone OPLS fitting preserves the retained CPU estimator", {
  set.seed(2210)
  X <- matrix(rnorm(90 * 13), 90, 13)
  Y <- cbind(
    X[, 1] - 0.4 * X[, 2] + rnorm(90, sd = 0.1),
    X[, 3] + 0.2 * X[, 4] + rnorm(90, sd = 0.1)
  )

  for (scaling in 1:3) {
    retained <- fastPLS:::opls_filter_cpp(X, Y, 2L, scaling)
    core <- fastPLS:::opls_filter_core_cpp(X, Y, 2L, scaling)
    expect_identical(core$north, retained$north)
    expect_equal(core$X, retained$X, tolerance = 1e-11)
    expect_equal(abs(core$W_orth), abs(retained$W_orth), tolerance = 1e-11)
    expect_equal(abs(core$P_orth), abs(retained$P_orth), tolerance = 1e-11)
  }
})
