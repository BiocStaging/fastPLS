test_that("projected C++ LDA agrees with explicit-score C++ LDA", {
  set.seed(20260503)
  Xtrain <- matrix(rnorm(90 * 14), nrow = 90, ncol = 14)
  R <- matrix(rnorm(14 * 6), nrow = 14, ncol = 6)
  offset <- rnorm(6)
  Ttrain <- sweep(Xtrain %*% R, 2L, offset, "-", check.margin = FALSE)
  y <- rep(seq_len(5), length.out = 90)
  ncomp <- c(2L, 6L)

  explicit <- fastPLS:::lda_train_prefix_cpp(Ttrain, y, 5L, ncomp, 1e-8)
  projected <- fastPLS:::lda_project_train_prefix_cpp(Xtrain, R, offset, y, 5L, ncomp, 1e-8)

  for (k in as.character(ncomp)) {
    expect_equal(projected[[k]]$backend, "cpp_project")
    expect_equal(projected[[k]]$means, explicit[[k]]$means, tolerance = 1e-12)
    expect_equal(projected[[k]]$linear, explicit[[k]]$linear, tolerance = 1e-12)
    expect_equal(projected[[k]]$constants, explicit[[k]]$constants, tolerance = 1e-12)

    kk <- as.integer(k)
    pred_explicit <- fastPLS:::lda_predict_labels_cpp(
      Ttrain[, seq_len(kk), drop = FALSE],
      explicit[[k]]
    )
    pred_projected <- fastPLS:::lda_project_predict_labels_cpp(
      Xtrain,
      R[, seq_len(kk), drop = FALSE],
      offset[seq_len(kk)],
      projected[[k]]
    )
    expect_equal(pred_projected, pred_explicit)
  }
})

test_that("native CUDA LDA agrees with C++ LDA", {
  skip_if_not(has_cuda())
  skip_if_not(exists("lda_cuda_native_available", envir = asNamespace("fastPLS"), inherits = FALSE))
  skip_if_not(fastPLS:::lda_cuda_native_available())

  set.seed(20260503)
  Ttrain <- matrix(rnorm(72 * 8), nrow = 72, ncol = 8)
  y <- rep(seq_len(6), each = 12)
  ncomp <- c(3L, 8L)

  cpp <- fastPLS:::lda_train_prefix_cpp(Ttrain, y, 6L, ncomp, 1e-8)
  cuda <- fastPLS:::lda_train_prefix_cuda(Ttrain, y, 6L, ncomp, 1e-8)

  for (k in as.character(ncomp)) {
    expect_equal(cuda[[k]]$backend, "cuda_native")
    expect_equal(cuda[[k]]$means, cpp[[k]]$means, tolerance = 1e-10)
    expect_equal(cuda[[k]]$linear, cpp[[k]]$linear, tolerance = 1e-10)
    expect_equal(cuda[[k]]$constants, cpp[[k]]$constants, tolerance = 1e-10)

    kk <- as.integer(k)
    Tsub <- Ttrain[, seq_len(kk), drop = FALSE]
    pred_cpp <- fastPLS:::lda_predict_labels_cpp(Tsub, cpp[[k]])
    pred_cuda <- fastPLS:::lda_predict_labels_cuda(Tsub, cuda[[k]])
    expect_equal(pred_cuda, pred_cpp)
  }
})

test_that("projected native CUDA LDA agrees with explicit-score C++ LDA", {
  skip_if_not(has_cuda())
  skip_if_not(exists("lda_cuda_native_available", envir = asNamespace("fastPLS"), inherits = FALSE))
  skip_if_not(fastPLS:::lda_cuda_native_available())

  set.seed(20260503)
  Xtrain <- matrix(rnorm(80 * 12), nrow = 80, ncol = 12)
  R <- matrix(rnorm(12 * 7), nrow = 12, ncol = 7)
  offset <- rnorm(7)
  Ttrain <- sweep(Xtrain %*% R, 2L, offset, "-", check.margin = FALSE)
  y <- rep(seq_len(5), length.out = 80)
  ncomp <- c(3L, 7L)

  cpp <- fastPLS:::lda_train_prefix_cpp(Ttrain, y, 5L, ncomp, 1e-8)
  cuda <- fastPLS:::lda_project_train_prefix_cuda(Xtrain, R, offset, y, 5L, ncomp, 1e-8)

  for (k in as.character(ncomp)) {
    expect_equal(cuda[[k]]$backend, "cuda_native_project")
    expect_equal(cuda[[k]]$means, cpp[[k]]$means, tolerance = 1e-10)
    expect_equal(cuda[[k]]$linear, cpp[[k]]$linear, tolerance = 1e-10)
    expect_equal(cuda[[k]]$constants, cpp[[k]]$constants, tolerance = 1e-10)

    kk <- as.integer(k)
    pred_cpp <- fastPLS:::lda_predict_labels_cpp(Ttrain[, seq_len(kk), drop = FALSE], cpp[[k]])
    pred_cuda <- fastPLS:::lda_project_predict_cuda(
      Xtrain,
      R[, seq_len(kk), drop = FALSE],
      offset[seq_len(kk)],
      cuda[[k]],
      FALSE
    )$pred
    expect_equal(pred_cuda, pred_cpp)
  }
})

test_that("pls classifier='lda' with backend='cuda' preserves CUDA LDA predictions", {
  skip_if_not(has_cuda())
  skip_if_not(exists("lda_cuda_native_available", envir = asNamespace("fastPLS"), inherits = FALSE))
  skip_if_not(fastPLS:::lda_cuda_native_available())

  set.seed(20260503)
  X <- matrix(rnorm(120 * 18), nrow = 120, ncol = 18)
  y <- factor(rep(letters[1:5], length.out = 120))
  idx <- sample(seq_len(nrow(X)), 30)

  fit_cpp <- pls(
    X[-idx, , drop = FALSE],
    y[-idx],
    ncomp = 4,
    method = "simpls",
    backend = "cuda",
    classifier = "lda_cpp",
    fit = FALSE,
    proj = FALSE,
    seed = 123L
  )
  fit_cuda <- pls(
    X[-idx, , drop = FALSE],
    y[-idx],
    ncomp = 4,
    method = "simpls",
    backend = "cuda",
    classifier = "lda",
    fit = FALSE,
    proj = FALSE,
    seed = 123L
  )

  pred_cpp <- predict(fit_cpp, X[idx, , drop = FALSE])$Ypred[[1]]
  pred_cuda <- predict(fit_cuda, X[idx, , drop = FALSE])$Ypred[[1]]
  expect_equal(pred_cuda, pred_cpp)
})

test_that("CUDA SIMPLS-LDA retains the requested SIMPLS estimator when dense Y fits", {
  skip_if_not(has_cuda())
  skip_if_not(exists("lda_cuda_native_available", envir = asNamespace("fastPLS"), inherits = FALSE))
  skip_if_not(fastPLS:::lda_cuda_native_available())

  set.seed(20260725)
  X <- matrix(rnorm(150 * 14), nrow = 150, ncol = 14)
  y <- factor(rep(letters[1:5], each = 30))
  idx <- sample(seq_len(nrow(X)), 30)

  fit <- pls(
    X[-idx, , drop = FALSE], y[-idx],
    Xtest = X[idx, , drop = FALSE], Ytest = y[idx],
    ncomp = 4, method = "simpls", backend = "cuda",
    classifier = "lda", fit = FALSE, return_variance = FALSE, seed = 123L
  )
  internal <- attr(fit, "fastPLS_internal", exact = TRUE)

  expect_equal(if (is.null(internal$requested_pls_method)) "simpls" else internal$requested_pls_method,
               "simpls")
  expect_equal(internal$pls_method, "simpls")
  expect_null(internal$method_substitution_reason)
  expect_true(is.finite(tail(fit$accuracy, 1L)))
})

test_that("resident CUDA PLS-LDA is invariant to the retired fusion hint", {
  skip_if_not(has_cuda())
  skip_if_not(exists("lda_cuda_native_available", envir = asNamespace("fastPLS"), inherits = FALSE))
  skip_if_not(fastPLS:::lda_cuda_native_available())
  skip_if_not(exists("pls_lda_gpu_native", envir = asNamespace("fastPLS"), inherits = FALSE))

  old <- Sys.getenv("FASTPLS_FUSED_CUDA_LDA", unset = NA_character_)
  on.exit({
    if (is.na(old)) {
      Sys.unsetenv("FASTPLS_FUSED_CUDA_LDA")
    } else {
      Sys.setenv(FASTPLS_FUSED_CUDA_LDA = old)
    }
  }, add = TRUE)

  set.seed(20260503)
  X <- matrix(rnorm(120 * 18), nrow = 120, ncol = 18)
  y <- factor(rep(letters[1:5], length.out = 120))
  idx <- sample(seq_len(nrow(X)), 30)

  fit_standard <- pls(
    X[-idx, , drop = FALSE],
    y[-idx],
    ncomp = 4,
    method = "plssvd",
    backend = "cuda",
    classifier = "lda",
    fit = FALSE,
    proj = FALSE,
    seed = 123L
  )
  pred_standard <- predict(fit_standard, X[idx, , drop = FALSE])$Ypred[[1]]

  Sys.setenv(FASTPLS_FUSED_CUDA_LDA = "1")
  fit_fused <- pls(
    X[-idx, , drop = FALSE],
    y[-idx],
    Xtest = X[idx, , drop = FALSE],
    ncomp = 4,
    method = "plssvd",
    backend = "cuda",
    classifier = "lda",
    fit = FALSE,
    proj = FALSE,
    seed = 123L
  )

  expect_equal(
    attr(fit_fused, "fastPLS_internal")$predict_backend,
    "cuda_resident"
  )
  expect_equal(fit_fused$diagnostics$residency$lda, "cuda")
  expect_equal(fit_fused$Ypred[[1]], pred_standard)
})

test_that("resident CUDA LDA is the default implementation", {
  skip_if_not(has_cuda())
  skip_if_not(exists("lda_cuda_native_available", envir = asNamespace("fastPLS"), inherits = FALSE))
  skip_if_not(fastPLS:::lda_cuda_native_available())

  old <- Sys.getenv("FASTPLS_FUSED_CUDA_LDA", unset = NA_character_)
  old_opt <- getOption("fastPLS.fused_cuda_lda", NULL)
  on.exit({
    if (is.na(old)) {
      Sys.unsetenv("FASTPLS_FUSED_CUDA_LDA")
    } else {
      Sys.setenv(FASTPLS_FUSED_CUDA_LDA = old)
    }
    options(fastPLS.fused_cuda_lda = old_opt)
  }, add = TRUE)
  Sys.unsetenv("FASTPLS_FUSED_CUDA_LDA")
  options(fastPLS.fused_cuda_lda = FALSE)

  set.seed(20260503)
  X <- matrix(rnorm(100 * 16), nrow = 100, ncol = 16)
  y <- factor(rep(letters[1:5], length.out = 100))
  idx <- sample(seq_len(nrow(X)), 20)

  fit_default <- pls(
    X[-idx, , drop = FALSE],
    y[-idx],
    Xtest = X[idx, , drop = FALSE],
    ncomp = 4,
    method = "plssvd",
    backend = "cuda",
    classifier = "lda",
    fit = FALSE,
    proj = FALSE,
    seed = 123L
  )

  expect_equal(fit_default$diagnostics$residency$lda, "cuda")
  expect_equal(
    attr(fit_default, "fastPLS_internal")$predict_backend,
    "cuda_resident"
  )
})
