test_that("pls does not expose backend-control metadata in fitted objects", {
  old_store_b <- Sys.getenv("FASTPLS_STORE_B", unset = NA_character_)
  on.exit({
    if (is.na(old_store_b)) Sys.unsetenv("FASTPLS_STORE_B") else Sys.setenv(FASTPLS_STORE_B = old_store_b)
  }, add = TRUE)

  Sys.setenv(FASTPLS_STORE_B = "0")
  set.seed(11)
  X <- matrix(rnorm(60 * 8), 60, 8)
  y <- factor(sample(letters[1:3], 60, replace = TRUE))

  fit <- pls(
    X,
    y,
    ncomp = 2,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    return_variance = FALSE
  )

  expect_s3_class(fit, "fastPLS")
  expect_null(fit$backend_control)
  expect_false("env" %in% names(fit))
  expect_false("fastPLS_version" %in% names(fit))
  expect_false("timestamp" %in% names(fit))
})
