test_that("Metal PLS backend fits core method families when available", {
  skip_if_not(fastPLS::has_metal(), "Metal backend is not available")

  set.seed(99)
  n <- 70
  p <- 12
  X <- matrix(rnorm(n * p), n, p)
  y <- matrix(X[, 1] - 0.5 * X[, 2] + rnorm(n, sd = 0.2), ncol = 1)
  X32 <- float::fl(X)
  y32 <- float::fl(y)
  cls <- factor(ifelse(y[, 1] > median(y[, 1]), "a", "b"))
  test <- seq(1, n, by = 5)

  for (method in c("plssvd", "simpls", "kernelpls")) {
    ncomp_test <- if (identical(method, "plssvd")) 1L else 1:2
    fit <- fastPLS::pls(
      X32[-test, , drop = FALSE],
      y32[-test, , drop = FALSE],
      X32[test, , drop = FALSE],
      y32[test, , drop = FALSE],
      ncomp = ncomp_test,
      method = method,
      backend = "metal",
      kernel = "linear",
      north = 1,
      return_variance = FALSE,
      seed = 99
    )
    expect_true(inherits(fit, "fastPLS"))
    expect_identical(
      attr(fit, "fastPLS_internal")$predict_backend,
      "float32_cpp"
    )
    expect_identical(
      attr(fit, "fastPLS_internal")$execution_route,
      "CPU/Metal hybrid (operation split)"
    )
    expect_true(all(is.finite(fit$Q2Y)))
  }

  fit_opls <- pls(
    X32, y32, ncomp = 1, method = "opls", backend = "metal",
    return_variance = FALSE
  )
  fit_kernel <- pls(
    X32, y32, ncomp = 1, method = "kernelpls", kernel = "rbf",
    backend = "metal", return_variance = FALSE
  )
  expect_identical(
    fit_opls$diagnostics$residency$route,
    "CPU/Metal hybrid (operation split)"
  )
  expect_identical(
    fit_kernel$diagnostics$residency$route,
    "CPU/Metal hybrid (operation split)"
  )

  fit_cls <- fastPLS::pls(
    X32[-test, , drop = FALSE],
    cls[-test],
    X32[test, , drop = FALSE],
    cls[test],
    ncomp = 1:2,
    method = "simpls",
    backend = "metal",
    return_variance = FALSE,
    seed = 100
  )
  expect_true(inherits(fit_cls, "fastPLS"))
  expect_identical(
    attr(fit_cls, "fastPLS_internal")$execution_route,
    "CPU/Metal hybrid (operation split)"
  )
  expect_identical(fit_cls$diagnostics$residency$prediction, "cpu")
  expect_true(is.data.frame(fit_cls$Ypred))
})

test_that("Metal backend is available through public CV helpers", {
  skip_if_not(fastPLS::has_metal(), "Metal backend is not available")

  set.seed(101)
  n <- 36
  p <- 8
  X <- matrix(rnorm(n * p), n, p)
  y_signal <- 0.6 * X[, 1] - 0.4 * X[, 2] + rnorm(n, sd = 0.2)
  y_num <- cbind(y_signal, 0.3 * X[, 3] + 0.2 * X[, 4] + rnorm(n, sd = 0.2))
  X32 <- float::fl(X)
  y_num32 <- float::fl(y_num)
  y_cls <- factor(ifelse(y_signal > median(y_signal), "hi", "lo"))

  fixed <- fastPLS::pls.single.cv(
    Xdata = X32,
    Ydata = y_cls,
    ncomp = 1:2,
    kfold = 2,
    method = "simpls",
    backend = "metal",
    seed = 101
  )
  expect_identical(fixed$backend, "metal")
  expect_equal(nrow(fixed$selection_metrics), 2L)
  expect_true(all(is.finite(fixed$selection_metrics$metric_value)))
  expect_equal(length(fixed$metrics$cross_validated), 2L)

  opt <- fastPLS::pls.single.cv(
    Xdata = X32,
    Ydata = y_num32,
    ncomp = 1:2,
    kfold = 2,
    method = "plssvd",
    backend = "metal",
    seed = 102
  )
  expect_identical(opt$backend, "metal")
  expect_true(opt$best_ncomp %in% 1:2)

  nested <- fastPLS::pls.double.cv(
    Xdata = X32,
    Ydata = y_cls,
    ncomp = 1:2,
    runn = 1,
    kfold_inner = 2,
    kfold_outer = 2,
    method = "simpls",
    backend = "metal",
    seed = 103
  )
  expect_identical(nested$backend, "metal")
  expect_true(is.finite(nested$Q2Y))
  expect_null(nested$medianQ2Y)
})
