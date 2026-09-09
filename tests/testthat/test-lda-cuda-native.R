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

test_that("public resident CUDA LDA agrees with the CPU workflow", {
  skip_if_not(has_cuda())

  set.seed(20260503)
  X <- matrix(rnorm(120 * 18), nrow = 120, ncol = 18)
  y <- factor(rep(letters[1:5], each = 24))
  X[, 1:5] <- X[, 1:5] + 3 * model.matrix(~ y - 1)
  idx <- sample(seq_len(nrow(X)), 30)

  fit_cpu <- pls(
    X[-idx, , drop = FALSE],
    y[-idx],
    ncomp = 4,
    method = "simpls",
    backend = "cpu",
    classifier = "lda",
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

  pred_cpu <- predict(fit_cpu, X[idx, , drop = FALSE])$Ypred[[1]]
  pred_cuda <- predict(fit_cuda, X[idx, , drop = FALSE])$Ypred[[1]]
  expect_gte(mean(pred_cuda == pred_cpu), 0.99)
})

test_that("CUDA SIMPLS-LDA retains the requested SIMPLS estimator when dense Y fits", {
  skip_if_not(has_cuda())

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

test_that("resident CUDA LDA is the default implementation", {
  skip_if_not(has_cuda())

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
