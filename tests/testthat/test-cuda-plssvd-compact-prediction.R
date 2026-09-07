test_that("resident CUDA PLS-SVD stays compact across storage hints", {
  skip_if_not(has_cuda(), "CUDA backend not available")

  old_store_b <- Sys.getenv("FASTPLS_STORE_B", unset = NA_character_)
  on.exit({
    if (is.na(old_store_b)) {
      Sys.unsetenv("FASTPLS_STORE_B")
    } else {
      Sys.setenv(FASTPLS_STORE_B = old_store_b)
    }
  }, add = TRUE)

  set.seed(902)
  Xtrain <- matrix(rnorm(80 * 12), 80, 12)
  Ytrain <- matrix(rnorm(80 * 3), 80, 3)
  Xtest <- matrix(rnorm(15 * 12), 15, 12)

  Sys.setenv(FASTPLS_STORE_B = "always")
  stored <- pls(
    Xtrain, Ytrain,
    ncomp = 1:2,
    method = "plssvd",
    backend = "cuda",
    svd.method = "rsvd",
    seed = 902,
    return_variance = FALSE
  )

  Sys.setenv(FASTPLS_STORE_B = "never")
  compact <- pls(
    Xtrain, Ytrain,
    ncomp = 1:2,
    method = "plssvd",
    backend = "cuda",
    svd.method = "rsvd",
    seed = 902,
    return_variance = FALSE
  )

  expect_null(stored$B)
  expect_null(compact$B)
  expect_equal(
    predict(stored, Xtest)$Ypred,
    predict(compact, Xtest)$Ypred,
    tolerance = 1e-8
  )
})
