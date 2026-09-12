with_fastpls_plssvd_optimized <- function(value, code) {
  old <- Sys.getenv("FASTPLS_PLSSVD_OPTIMIZED", unset = NA_character_)
  on.exit({
    if (length(old) != 1L || is.na(old)) {
      Sys.unsetenv("FASTPLS_PLSSVD_OPTIMIZED")
    } else {
      Sys.setenv(FASTPLS_PLSSVD_OPTIMIZED = old)
    }
  }, add = TRUE)
  Sys.setenv(FASTPLS_PLSSVD_OPTIMIZED = as.character(as.integer(value)))
  force(code)
}

test_that("optimized plssvd returns valid fit structure", {
  set.seed(20260321)
  X <- matrix(rnorm(110 * 16), nrow = 110, ncol = 16)
  y <- factor(sample(c("A", "B", "C", "D"), 110, replace = TRUE))
  idx <- sample(seq_len(110), 24)

  fit <- with_fastpls_plssvd_optimized(1, fastPLS::pls(
    X[-idx, , drop = FALSE],
    y[-idx],
    X[idx, , drop = FALSE],
    y[idx],
    ncomp = 1:3,
    method = "plssvd",
    fit = TRUE,
    seed = 77L
  ))

  expect_s3_class(fit, "fastPLS")
  expect_equal(dim(fit$B), c(ncol(X), nlevels(y), 3L))
  expect_equal(dim(fit$R), c(ncol(X), 3L))
  expect_equal(dim(fit$Q), c(nlevels(y), 3L))
  expect_true(is.data.frame(fit$Ypred))
  expect_equal(ncol(fit$Ypred), 3L)
})

test_that("optimized plssvd stays numerically close to the legacy path on a controlled regression task", {
  set.seed(20260321)
  n <- 140
  z1 <- rnorm(n)
  z2 <- rnorm(n)
  X <- cbind(
    outer(z1, seq_len(10), `*`) + matrix(rnorm(n * 10, sd = 0.07), n, 10),
    outer(z2, seq_len(10), `*`) + matrix(rnorm(n * 10, sd = 0.07), n, 10)
  )
  Y <- cbind(
    1.4 * z1 - 0.5 * z2 + rnorm(n, sd = 0.04),
    -0.8 * z1 + 1.7 * z2 + rnorm(n, sd = 0.04)
  )
  idx <- sample(seq_len(n), 30)

  fit_args <- list(
    Xtrain = X[-idx, , drop = FALSE],
    Ytrain = Y[-idx, , drop = FALSE],
    Xtest = X[idx, , drop = FALSE],
    Ytest = Y[idx, , drop = FALSE],
    ncomp = c(1L, 2L),
    method = "plssvd",
    fit = TRUE,
    seed = 123L
  )

  baseline <- with_fastpls_plssvd_optimized(0, do.call(fastPLS::pls, fit_args))
  optimized <- with_fastpls_plssvd_optimized(1, do.call(fastPLS::pls, fit_args))

  expect_equal(dim(baseline$B), dim(optimized$B))
  expect_equal(optimized$B, baseline$B, tolerance = 1e-8)
  expect_equal(optimized$Yfit, baseline$Yfit, tolerance = 1e-8)
})
