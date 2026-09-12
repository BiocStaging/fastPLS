test_that("the publication API exports only the documented functions", {
  expected <- c(
    "evaluate", "fastcor", "fastPLS_blas", "fastsvd",
    "has_cuda", "has_metal", "pls", "pls.double.cv", "pls.single.cv",
    "plot.permutation", "ViP"
  )

  expect_setequal(getNamespaceExports("fastPLS"), expected)
  expect_false(exists("fastPLS_backend", envir = asNamespace("fastPLS"), inherits = FALSE))
  expect_false("pca" %in% getNamespaceExports("fastPLS"))
  expect_false(exists("predict.fastPLSPCA", envir = asNamespace("fastPLS"), inherits = FALSE))
})

test_that("obsolete LDA and matrix-route controls are not public inputs", {
  functions <- list(pls, pls.single.cv, pls.double.cv)
  for (fun in functions) {
    expect_false("lda_ridge" %in% names(formals(fun)))
  }
  expect_false("xprod" %in% names(formals(pls.single.cv)))
  expect_false("xprod" %in% names(formals(pls.double.cv)))

  index <- c(seq_len(10), 51:60, 101:110)
  X <- as.matrix(iris[index, seq_len(4)])
  y <- droplevels(iris$Species[index])
  expect_error(pls(X, y, lda_ridge = 0.1), "Unknown entry")
  expect_error(pls(X, y, xprod = FALSE), "Unknown entry")
  expect_error(pls.single.cv(X, y, lda_ridge = 0.1), "Unknown entry")
  expect_error(pls.single.cv(X, y, xprod = FALSE), "Unknown entry")
  expect_error(pls.double.cv(X, y, lda_ridge = 0.1), "Unknown entry")
  expect_error(pls.double.cv(X, y, xprod = FALSE), "Unknown entry")
})

test_that("CV uses selection as its public tuning argument", {
  for (fun in list(pls.single.cv, pls.double.cv)) {
    expect_true("selection" %in% names(formals(fun)))
    expect_false("selection_metric" %in% names(formals(fun)))
  }

  index <- c(seq_len(10), 51:60, 101:110)
  X <- as.matrix(iris[index, seq_len(4)])
  y <- droplevels(iris$Species[index])
  expect_error(
    pls.single.cv(X, y, selection_metric = "accuracy"),
    "Unknown entry"
  )
  expect_error(
    pls.double.cv(X, y, selection_metric = "accuracy"),
    "Unknown entry"
  )
})

test_that("prediction block sizing is internal", {
  expect_false("flash.block_size" %in% names(formals(predict.fastPLS)))
  expect_false("top5" %in% names(formals(predict.fastPLS)))
  expect_null(formals(predict.fastPLS)$top)

  X <- as.matrix(mtcars[, c("disp", "hp", "wt", "qsec")])
  fit <- pls(X, mtcars$mpg, ncomp = 2, backend = "cpu")
  expect_error(
    predict(fit, X[seq_len(2), , drop = FALSE], flash.block_size = 16L),
    "Unknown argument.*flash.block_size"
  )
  expect_error(
    predict(fit, X[seq_len(2), , drop = FALSE], top5 = TRUE),
    "Unknown argument.*top5"
  )
})

test_that("top is classification-specific", {
  X <- as.matrix(mtcars[, c("disp", "hp", "wt", "qsec")])
  fit <- pls(X, mtcars$mpg, ncomp = 2, backend = "cpu")

  expect_no_warning(predict(fit, X[seq_len(2), , drop = FALSE]))
  expect_no_warning(predict(fit, X[seq_len(2), , drop = FALSE], top = NULL))
  expect_warning(
    predict(fit, X[seq_len(2), , drop = FALSE], top = 5L),
    "top is ignored for regression models"
  )
})
