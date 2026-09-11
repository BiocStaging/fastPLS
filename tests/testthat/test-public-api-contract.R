test_that("the publication API exports only the documented functions", {
  expected <- c(
    "evaluate", "fastcor", "fastPLS_backend", "fastPLS_blas", "fastsvd",
    "has_cuda", "has_metal", "pls", "pls.double.cv", "pls.single.cv",
    "plot.permutation", "ViP"
  )

  expect_setequal(getNamespaceExports("fastPLS"), expected)
  expect_false("pca" %in% getNamespaceExports("fastPLS"))
  expect_false(exists("predict.fastPLSPCA", envir = asNamespace("fastPLS"), inherits = FALSE))
})

test_that("deprecated lda_ridge warns and cannot change the fitted estimator", {
  idx <- c(1:12, 51:62, 101:112)
  X <- as.matrix(iris[idx, 1:4])
  y <- factor(iris[idx, 5])
  base <- pls(
    X, y, X, y, ncomp = 2, method = "simpls", svd.method = "rsvd",
    classifier = "lda", backend = "cpu", return_variance = FALSE
  )
  expect_warning(
    deprecated <- pls(
      X, y, X, y, ncomp = 2, method = "simpls", svd.method = "rsvd",
      classifier = "lda", lda_ridge = 0.5, backend = "cpu",
      return_variance = FALSE
    ),
    "deprecated and ignored"
  )

  expect_identical(deprecated$Ypred, base$Ypred)
  expect_equal(deprecated$accuracy, base$accuracy)
})

test_that("cross-validation does not tune deprecated lda_ridge", {
  idx <- c(1:10, 51:60, 101:110)
  X <- as.matrix(iris[idx, 1:4])
  y <- factor(iris[idx, 5])
  expect_warning(
    cv <- pls.single.cv(
      X, y, ncomp = 1:2, kfold = 3, method = "simpls",
      svd.method = "rsvd", classifier = "lda", lda_ridge = 0.25,
      backend = "cpu", fit = FALSE, seed = 12
    ),
    "deprecated and ignored"
  )

  expect_false("lda_ridge" %in% names(cv$best_parameters))
  expect_false("lda_ridge" %in% names(cv$tuning_config))
})

test_that("double CV deprecates lda_ridge without propagating it", {
  idx <- c(1:6, 51:56, 101:106)
  X <- as.matrix(iris[idx, 1:4])
  y <- factor(iris[idx, 5])
  expect_warning(
    cv <- pls.double.cv(
      X, y, ncomp = 1, kfold_inner = 2, kfold_outer = 2, runn = 1,
      method = "simpls", svd.method = "rsvd", classifier = "lda",
      lda_ridge = 0.1, backend = "cpu", seed = 13
    ),
    "deprecated and ignored"
  )

  expect_false("lda_ridge" %in% names(cv$best_parameters[[1L]][[1L]]))
})
