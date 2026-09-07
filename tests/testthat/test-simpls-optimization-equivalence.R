test_that("SIMPLS execution optimizations preserve fixed-seed predictions", {
  set.seed(21)
  X <- matrix(rnorm(320 * 30), 320, 30)
  Y <- matrix(rnorm(320 * 4), 320, 4)
  Xtest <- matrix(rnorm(60 * 30), 60, 30)

  keys <- c(
    "FASTPLS_ABLATION_MODE", "FASTPLS_ABLATION_XPROD",
    "FASTPLS_FAST_OPTIMIZED", "FASTPLS_INCREMENTAL_COEFFICIENTS",
    "FASTPLS_FAST_DEFLCACHE", "FASTPLS_STORE_B"
  )
  old <- Sys.getenv(keys, unset = NA_character_)
  on.exit({
    present <- !is.na(old)
    if (any(present)) do.call(Sys.setenv, as.list(old[present]))
    if (any(!present)) Sys.unsetenv(keys[!present])
  }, add = TRUE)

  run_fit <- function(optimized, incremental, deflcache, store_B, xprod) {
    Sys.setenv(
      FASTPLS_ABLATION_MODE = "1",
      FASTPLS_ABLATION_XPROD = xprod,
      FASTPLS_FAST_OPTIMIZED = optimized,
      FASTPLS_INCREMENTAL_COEFFICIENTS = incremental,
      FASTPLS_FAST_DEFLCACHE = deflcache,
      FASTPLS_STORE_B = store_B
    )
    fit <- pls(
      X, Y, ncomp = 1:12, method = "simpls", backend = "cpu",
      svd.method = "rsvd", return_variance = FALSE, seed = 123
    )
    predict(fit, Xtest)$Ypred[[12L]]
  }

  pairs <- list(
    cached_XtX = list(
      c("0", "1", "1", "always", "0"),
      c("1", "1", "1", "always", "0")
    ),
    incremental_coefficients = list(
      c("1", "0", "1", "always", "0"),
      c("1", "1", "1", "always", "0")
    ),
    cached_deflation_products = list(
      c("1", "1", "0", "always", "0"),
      c("1", "1", "1", "always", "0")
    ),
    compact_prediction = list(
      c("1", "1", "1", "always", "0"),
      c("1", "1", "1", "never", "0")
    ),
    matrix_free = list(
      c("1", "1", "1", "never", "0"),
      c("1", "1", "1", "never", "1")
    )
  )
  for (pair in pairs) {
    reference <- do.call(run_fit, as.list(pair[[1L]]))
    optimized <- do.call(run_fit, as.list(pair[[2L]]))
    expect_equal(optimized, reference, tolerance = 1e-10)
  }
})
