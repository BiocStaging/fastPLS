test_that("pls supports the CPU method, solver, and task grid", {
  set.seed(20260318)
  X <- matrix(rnorm(72 * 10), nrow = 72, ncol = 10)
  y_reg <- matrix(rnorm(72 * 2), ncol = 2)
  y_cls <- factor(sample(c("A", "B", "C"), 72, replace = TRUE))
  idx <- sample(seq_len(72), 18)

  back <- "cpu_rsvd"

  for (m in c("plssvd", "simpls", "opls", "kernelpls")) {
    for (s in back) {
      fit_reg <- pls(
        X[-idx, , drop = FALSE],
        y_reg[-idx, , drop = FALSE],
        X[idx, , drop = FALSE],
        y_reg[idx, , drop = FALSE],
        ncomp = 1:2,
        method = m,
        fit = TRUE
      )
      expect_s3_class(fit_reg, "fastPLS")
      expect_true("Ypred" %in% names(fit_reg))
      expect_true(all(is.finite(unlist(fit_reg$Ypred))))

      fit_cls <- pls(
        X[-idx, , drop = FALSE],
        y_cls[-idx],
        X[idx, , drop = FALSE],
        y_cls[idx],
        ncomp = 1:2,
        method = m,
        fit = TRUE
      )
      expect_s3_class(fit_cls, "fastPLS")
      expect_true(is.data.frame(fit_cls$Ypred))
      expect_equal(nrow(fit_cls$Ypred), length(idx))

      fit_lda <- pls(
        X[-idx, , drop = FALSE],
        y_cls[-idx],
        X[idx, , drop = FALSE],
        y_cls[idx],
        ncomp = 1:2,
        method = m,
        classifier = "lda",
        fit = TRUE
      )
      expect_s3_class(fit_lda, "fastPLS")
      expect_equal(nrow(fit_lda$Ypred), length(idx))
    }
  }
})

test_that("pls.single.cv and pls.double.cv support accelerated simpls", {
  set.seed(20260318)
  X <- matrix(rnorm(54 * 8), nrow = 54, ncol = 8)
  y_reg <- matrix(rnorm(54 * 2), ncol = 2)
  y_cls <- factor(sample(c("L", "M", "H"), 54, replace = TRUE))

  back <- "cpu_rsvd"

  for (s in back) {
    for (m in c("plssvd", "simpls", "opls", "kernelpls")) {
      cv_reg <- pls.single.cv(
        Xdata = X,
        Ydata = y_reg,
        ncomp = 1:2,
        kfold = 3,
        method = m
      )
      expect_true(is.list(cv_reg))
      expect_true("Q2Y" %in% names(cv_reg))

      cv_cls <- pls.single.cv(
        Xdata = X,
        Ydata = y_cls,
        ncomp = 1:2,
        kfold = 3,
        method = m
      )
      expect_true(is.list(cv_cls))
      expect_true("Q2Y" %in% names(cv_cls))

      dcv_reg <- pls.double.cv(
        Xdata = X,
        Ydata = y_reg,
        ncomp = 1:2,
        runn = 1,
        kfold_inner = 3,
        kfold_outer = 3,
        method = m
      )
      expect_true(is.list(dcv_reg))
      expect_true("Q2Y" %in% names(dcv_reg))

      dcv_cls <- pls.double.cv(
        Xdata = X,
        Ydata = y_cls,
        ncomp = 1:2,
        runn = 1,
        kfold_inner = 3,
        kfold_outer = 3,
        method = m
      )
      expect_true(is.list(dcv_cls))
      expect_true("Ypred" %in% names(dcv_cls))
    }
  }
})

test_that("unsupported backend labels are rejected", {
  set.seed(20260321)
  X <- matrix(rnorm(24 * 6), nrow = 24, ncol = 6)
  Y <- matrix(rnorm(24), ncol = 1)
  expect_error(
    pls(X, Y, ncomp = 1, backend = "r"),
    "must be one of"
  )
})

test_that("public PLS functions use rSVD without a solver argument", {
  expect_false("method" %in% names(formals(fastsvd)))
  expect_false("svd.method" %in% names(formals(pls)))
  expect_false("svd.method" %in% names(formals(pls.single.cv)))
  expect_false("svd.method" %in% names(formals(pls.double.cv)))
})

test_that("the fixed public PLS solver remains reproducible", {
  set.seed(20260815)
  X <- matrix(rnorm(48 * 7), nrow = 48)
  Y <- matrix(rnorm(48 * 2), ncol = 2)

  svd_default <- fastsvd(X, ncomp = 2, backend = "cpu", seed = 19)
  svd_repeat <- fastsvd(X, ncomp = 2, backend = "cpu", seed = 19)
  expect_equal(svd_default$d, svd_repeat$d, tolerance = 1e-12)
  expect_identical(svd_default$method, "rsvd")

  fit_default <- pls(X, Y, X, Y, ncomp = 1:2, backend = "cpu", seed = 19)
  fit_explicit <- pls(
    X, Y, X, Y, ncomp = 1:2, backend = "cpu", seed = 19
  )
  expect_equal(fit_default$Ypred, fit_explicit$Ypred, tolerance = 1e-12)

  single_default <- pls.single.cv(
    X, Y, ncomp = 1:2, kfold = 3, backend = "cpu", seed = 19
  )
  single_explicit <- pls.single.cv(
    X, Y, ncomp = 1:2, kfold = 3, backend = "cpu", seed = 19
  )
  expect_identical(single_default$best_ncomp, single_explicit$best_ncomp)
  expect_equal(single_default$Q2Y, single_explicit$Q2Y, tolerance = 1e-12)
})
