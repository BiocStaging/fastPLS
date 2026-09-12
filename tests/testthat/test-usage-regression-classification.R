test_that("pls and predict support regression workflow", {
  set.seed(1001)
  X <- matrix(rnorm(80 * 10), nrow = 80, ncol = 10)
  Y <- matrix(rnorm(80 * 2), nrow = 80, ncol = 2)
  idx <- sample(seq_len(80), 16)

  fit <- pls(
    X[-idx, , drop = FALSE],
    Y[-idx, , drop = FALSE],
    X[idx, , drop = FALSE],
    Y[idx, , drop = FALSE],
    ncomp = 1:3,
    method = "simpls",
    fit = TRUE
  )

  expect_s3_class(fit, "fastPLS")
  expect_equal(dim(fit$B), c(ncol(X), ncol(Y), 3))
  expect_true(is.array(fit$Ypred))
  expect_equal(dim(fit$Ypred), c(length(idx), ncol(Y), 3))
  expect_length(fit$Q2Y, 3L)
  expect_length(fit$variance_explained, 3L)
  expect_true(all(is.finite(fit$variance_explained)))
  expect_equal(fit$x_variance_explained, fit$variance_explained)
  lean_fit <- pls(X, Y, ncomp = 3, method = "simpls", return_variance = FALSE)
  expect_null(lean_fit$variance_explained)
  expect_null(lean_fit$x_variance_explained)

  pr <- predict(fit, X[idx, , drop = FALSE], Ytest = Y[idx, , drop = FALSE], proj = TRUE)
  expect_true(is.array(pr$Ypred))
  expect_length(pr$Q2Y, 3L)
  expect_true(is.matrix(pr$Ttest))
  expect_identical(pr$metrics$task, "regression")
  expect_equal(nrow(pr$metrics$metrics), 3L)
  expect_equal(
    unname(pr$metrics$metrics$Q2),
    unname(pr$Q2Y),
    tolerance = 1e-12
  )
})

test_that("pls and predict support classification workflow", {
  set.seed(1002)
  X <- matrix(rnorm(90 * 12), nrow = 90, ncol = 12)
  y <- factor(sample(c("A", "B", "C"), 90, replace = TRUE))
  idx <- sample(seq_len(90), 20)

  fit <- pls(
    X[-idx, , drop = FALSE],
    y[-idx],
    X[idx, , drop = FALSE],
    y[idx],
    ncomp = 1:2,
    method = "plssvd",
    fit = TRUE
  )

  expect_s3_class(fit, "fastPLS")
  expect_true(is.data.frame(fit$Ypred))
  expect_equal(nrow(fit$Ypred), length(idx))
  expect_equal(ncol(fit$Ypred), 2L)
  expect_true(is.data.frame(fit$Yfit))
  expect_equal(levels(fit$Ypred[[1]]), levels(y))
  expect_length(fit$variance_explained, 2L)
  expect_true(all(is.finite(fit$variance_explained)))

  pr <- predict(fit, X[idx, , drop = FALSE], Ytest = y[idx], proj = FALSE)
  expect_true(is.data.frame(pr$Ypred))
  expect_length(pr$Q2Y, 2L)
  expect_identical(pr$metrics$task, "classification")
  expect_equal(nrow(pr$metrics$metrics), 2L)
  expect_equal(
    unname(pr$metrics$metrics$accuracy),
    unname(pr$accuracy)
  )
})

test_that("classification Yfit uses the same decoding as predict", {
  X <- as.matrix(iris[, 1:4])
  y <- iris$Species

  fit_without_yfit <- pls(
    Xtrain = X,
    Ytrain = y,
    ncomp = 3,
    method = "simpls",
    backend = "cpu",
    classifier = "argmax",
    seed = 123
  )
  expect_true(all(is.na(fit_without_yfit$R2Y)))

  fit <- pls(
    Xtrain = X,
    Ytrain = y,
    ncomp = 3,
    method = "simpls",
    backend = "cpu",
    classifier = "argmax",
    seed = 123,
    fit = TRUE
  )
  pred <- predict(fit, X)

  expect_true(is.data.frame(fit$Yfit))
  expect_identical(as.character(fit$Yfit[[1]]), as.character(pred$Ypred[[1]]))
  expect_true("setosa" %in% as.character(fit$Yfit[[1]]))
})

test_that("pls.single.cv and pls.double.cv run in both contexts", {
  set.seed(1004)
  X <- matrix(rnorm(60 * 8), nrow = 60, ncol = 8)
  Yreg <- matrix(rnorm(60), ncol = 1)
  ycls <- factor(sample(c("L", "M", "H"), 60, replace = TRUE))

  cv_reg <- pls.single.cv(
    Xdata = X,
    Ydata = Yreg,
    ncomp = 1:2,
    kfold = 3,
    method = "simpls"
  )
  expect_true(is.list(cv_reg))
  expect_true("Q2Y" %in% names(cv_reg))

  cv_cls <- pls.single.cv(
    Xdata = X,
    Ydata = ycls,
    ncomp = 1:2,
    kfold = 3,
    method = "simpls"
  )
  expect_true(is.list(cv_cls))
  expect_true("Q2Y" %in% names(cv_cls))

  dcv_reg <- pls.double.cv(
    Xdata = X,
    Ydata = Yreg,
    ncomp = 1:2,
    runn = 2,
    kfold_inner = 3,
    kfold_outer = 3,
    method = "simpls"
  )
  expect_true(is.list(dcv_reg))
  expect_true("Q2Y" %in% names(dcv_reg))
  expect_true("R2Y" %in% names(dcv_reg))
  expect_length(dcv_reg$Q2Y, 2L)
  expect_length(dcv_reg$R2Y, 2L)

  dcv_cls <- pls.double.cv(
    Xdata = X,
    Ydata = ycls,
    ncomp = 1:2,
    runn = 2,
    kfold_inner = 3,
    kfold_outer = 3,
    method = "simpls"
  )
  expect_true(is.list(dcv_cls))
  expect_true(is.factor(dcv_cls$Ypred))
  expect_true("acc_tot" %in% names(dcv_cls))
})

test_that("compiled CV reports the prediction backend", {
  set.seed(10041)
  X <- matrix(rnorm(48 * 7), nrow = 48, ncol = 7)
  y <- factor(sample(c("A", "B", "C"), 48, replace = TRUE))

  cpu_cv <- pls.single.cv(
    Xdata = X,
    Ydata = y,
    ncomp = 2,
    kfold = 3,
    method = "plssvd",
    backend = "cpu",
    seed = 123L
  )
  expect_identical(cpu_cv$backend, "cpu")
  expect_identical(cpu_cv$prediction_backend, "cpu")

  cpu_opt <- pls.single.cv(
    Xdata = X,
    Ydata = y,
    ncomp = 1:2,
    kfold = 3,
    method = "plssvd",
    backend = "cpu",
    seed = 123L
  )
  expect_identical(cpu_opt$backend, "cpu")
  expect_length(cpu_opt$best_ncomp, 1L)

  cpu_double <- pls.double.cv(
    Xdata = X,
    Ydata = y,
    ncomp = 1:2,
    runn = 1,
    kfold_inner = 3,
    kfold_outer = 3,
    method = "plssvd",
    backend = "cpu",
    seed = 123L
  )
  expect_identical(cpu_double$backend, "cpu")
  expect_true(is.factor(cpu_double$Ypred))

  skip_if_not(has_cuda(), "CUDA backend unavailable")
  cuda_cv <- pls.single.cv(
    Xdata = X,
    Ydata = y,
    ncomp = 2,
    kfold = 3,
    method = "plssvd",
    backend = "cuda",
    seed = 123L
  )
  expect_identical(cuda_cv$backend, "cuda")
  expect_identical(cuda_cv$residency$fold_prediction, "resident cuda")
  expect_identical(cuda_cv$residency$fallback, "none")

  cuda_opt <- pls.single.cv(
    Xdata = X,
    Ydata = y,
    ncomp = 1:2,
    kfold = 3,
    method = "plssvd",
    backend = "cuda",
    seed = 123L
  )
  expect_identical(cuda_opt$backend, "cuda")
  expect_length(cuda_opt$best_ncomp, 1L)
})

test_that("SVD utilities and helper functions are usable in practice", {
  set.seed(1005)
  A <- matrix(rnorm(70 * 14), nrow = 70, ncol = 14)

  sr <- fastsvd(A, ncomp = 4, backend = "cpu")
  expect_true(is.list(sr))
  expect_true(all(c("u", "d", "v", "backend", "method", "svd.method", "elapsed") %in% names(sr)))
  expect_identical(sr$backend, "cpu")
  expect_identical(sr$method, "rsvd")
  expect_identical(sr$svd.method, "cpu_rsvd")
  expect_equal(ncol(sr$u), 4L)
  expect_equal(length(sr$d), 4L)

  y <- factor(sample(c("a", "b", "c"), nrow(A), replace = TRUE))

  cls_model <- pls(A, y, ncomp = 1:2, method = "plssvd")
  cls_pred <- predict(cls_model, A[1:5, , drop = FALSE])
  expect_true(is.data.frame(cls_pred$Ypred))
  expect_equal(nrow(cls_pred$Ypred), 5L)

  plot_model <- pls(
    A[1:60, , drop = FALSE],
    y[1:60],
    A[61:70, , drop = FALSE],
    y[61:70],
    ncomp = 1:2,
    method = "opls",
    fit = TRUE,
    proj = TRUE
  )
  expect_length(plot_model$variance_explained, 2L)
  expect_true(all(is.finite(plot_model$variance_explained)))
  png_file <- tempfile(fileext = ".png")
  grDevices::png(png_file)
  expect_silent(plot(plot_model, score.set = "train", groups = y[1:60]))
  expect_silent(plot(plot_model, score.set = "test", groups = plot_model$Ypred[[length(plot_model$Ypred)]]))
  grDevices::dev.off()
  unlink(png_file)

  C1 <- fastcor(A, byrow = FALSE, diag = FALSE)
  expect_true(is.matrix(C1))
  expect_equal(dim(C1), c(ncol(A), ncol(A)))

  C2 <- fastcor(A[, 1:5, drop = FALSE], A[, 1:5, drop = FALSE], byrow = FALSE, diag = TRUE)
  expect_true(is.numeric(C2))
  expect_equal(length(C2), 5L)

  model_uni <- pls(A, matrix(rnorm(nrow(A)), ncol = 1), ncomp = 1:3, method = "simpls", fit = TRUE)
  expect_identical(dim(model_uni$P), c(0L, 0L))
  vip_uni <- ViP(model_uni)
  expect_true(is.matrix(vip_uni))

  model_multi <- pls(A, matrix(rnorm(nrow(A) * 2), ncol = 2), ncomp = 1:3, method = "simpls", fit = TRUE)
  expect_identical(dim(model_multi$P), c(0L, 0L))
  vip_multi <- ViP(model_multi)
  expect_true(is.list(vip_multi))
  expect_equal(length(vip_multi), 2L)

  model_loadings <- pls(
    A,
    matrix(rnorm(nrow(A) * 2), ncol = 2),
    ncomp = 1:3,
    method = "simpls",
    fit = TRUE,
    return_loadings = TRUE
  )
  expect_equal(dim(model_loadings$P), c(ncol(A), 3L))
  expect_true(all(is.finite(model_loadings$P)))
})

test_that("legacy KODAMA-specific helpers are not user-accessible", {
  expect_error(fastPLS::kodama_cpp_simpls_rsvd, "not an exported object")
  expect_error(fastPLS::kodama_cuda_simpls_rsvd, "not an exported object")
  expect_false(exists("kodama_cpp_simpls_rsvd", envir = asNamespace("fastPLS"), inherits = FALSE))
  expect_false(exists("kodama_cuda_simpls_rsvd", envir = asNamespace("fastPLS"), inherits = FALSE))
})
