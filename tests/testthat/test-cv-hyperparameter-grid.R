test_that("pls.single.cv tunes prediction hyperparameters", {
  idx <- c(1:12, 51:62, 101:112)
  X <- as.matrix(iris[idx, 1:4])
  y <- factor(iris[idx, 5])

  opt <- pls.single.cv(
    X,
    y,
    ncomp = 1:2,
    kfold = 3,
    method = "kernelpls",
    backend = "cpu",
    kernel = c("linear", "rbf"),
    gamma = c(0.1, 1),
    seed = 1
  )

  expect_true(opt$best_ncomp %in% 1:2)
  expect_true(all(c("kernel", "gamma", "ncomp") %in% names(opt$best_parameters)))
  expect_false("method" %in% names(opt$best_parameters))
  expect_true(nrow(opt$tuning_summary) >= 2L)
  expect_true(all(c("kernel", "gamma", "best_metric_value") %in% names(opt$tuning_summary)))
})

test_that("pls.double.cv uses inner selected hyperparameters", {
  idx <- c(1:10, 51:60, 101:110)
  X <- as.matrix(iris[idx, 1:4])
  y <- factor(iris[idx, 5])

  nested <- pls.double.cv(
    X,
    y,
    ncomp = 1:2,
    runn = 1,
    kfold_inner = 2,
    kfold_outer = 2,
    method = "kernelpls",
    backend = "cpu",
    kernel = c("linear", "rbf"),
    gamma = c(0.1, 1),
    seed = 1
  )

  expect_length(nested$results, 1L)
  expect_true(all(nested$results[[1]]$best_ncomp %in% 1:2))
  expect_true(all(vapply(nested$results[[1]]$best_parameters, function(x) {
    is.list(x) && all(c("kernel", "gamma", "ncomp") %in% names(x)) &&
      !("method" %in% names(x))
  }, logical(1L))))
})

test_that("pls.single.cv reports only optimized best parameters", {
  idx <- c(1:12, 51:62, 101:112)
  X <- as.matrix(iris[idx, 1:4])
  y <- factor(iris[idx, 5])

  opt <- pls.single.cv(
    X,
    y,
    ncomp = 2:4,
    kfold = 3,
    classifier = c("argmax", "lda"),
    seed = 1
  )

  expect_setequal(names(opt$best_parameters), c("ncomp", "classifier"))
  expect_true(opt$best_parameters$ncomp %in% 2:4)
  expect_true(opt$best_parameters$classifier %in% c("argmax", "lda"))
})

test_that("pls.single.cv tuning_config omits irrelevant classifier controls", {
  idx <- c(1:12, 51:62, 101:112)
  X <- as.matrix(iris[idx, 1:4])
  y <- factor(iris[idx, 5])

  argmax <- pls.single.cv(
    X,
    y,
    ncomp = 2,
    kfold = 3,
    classifier = "argmax",
    seed = 2
  )
  expect_true("classifier" %in% names(argmax$tuning_config))
  expect_false(any(c("lda_ridge", "k", "tau", "alpha", "top_m", "cknn_memory") %in% names(argmax$tuning_config)))

  lda <- pls.single.cv(
    X,
    y,
    ncomp = 2,
    kfold = 3,
    classifier = "lda",
    seed = 2
  )
  expect_false(any(c("lda_ridge", "k", "tau", "alpha", "top_m", "cknn_memory") %in% names(lda$tuning_config)))
})

test_that("pls refits and predicts from a pls.single.cv result", {
  set.seed(2106)
  test_idx <- sample(seq_len(nrow(iris)), 30)
  Xtrain <- as.matrix(iris[-test_idx, 1:4])
  Ytrain <- factor(iris[-test_idx, 5])
  Xtest <- as.matrix(iris[test_idx, 1:4])
  Ytest <- factor(iris[test_idx, 5], levels = levels(Ytrain))

  opt <- pls.single.cv(
    Xtrain,
    Ytrain,
    ncomp = 1:3,
    kfold = 3,
    method = "simpls",
    backend = "cpu",
    classifier = c("argmax", "lda"),
    seed = 2106
  )

  expect_s3_class(opt, "fastPLSCV")
  expect_true(all(c("Xdata", "Ydata") %in% names(attr(opt, "fit_data"))))

  fit_named <- pls(opt, Xtest = Xtest, Ytest = Ytest, return_variance = FALSE)
  fit_positional <- pls(opt, Xtest, Ytest = Ytest, return_variance = FALSE)

  expect_s3_class(fit_named, "fastPLS")
  expect_equal(as.integer(attr(fit_named, "fastPLS_internal")$ncomp), as.integer(opt$best_ncomp))
  expect_equal(fit_named$cv_best_parameters, opt$best_parameters)
  expect_true(is.data.frame(fit_named$Ypred))
  expect_equal(nrow(fit_named$Ypred), nrow(Xtest))
  expect_equal(fit_named$Ypred, fit_positional$Ypred)
})

test_that("pls refits regression models selected by pls.single.cv", {
  set.seed(2107)
  X <- matrix(rnorm(60 * 6), nrow = 60, ncol = 6)
  y <- X[, 1] - 0.5 * X[, 2] + rnorm(60, sd = 0.2)
  test_idx <- seq(1, 60, by = 3)

  opt <- pls.single.cv(
    X[-test_idx, , drop = FALSE],
    y[-test_idx],
    ncomp = 1:3,
    kfold = 3,
    method = "simpls",
    backend = "cpu",
    seed = 2107,
    fit = FALSE
  )
  fit <- pls(
    opt,
    Xtest = X[test_idx, , drop = FALSE],
    Ytest = y[test_idx],
    return_variance = FALSE
  )

  expect_s3_class(fit, "fastPLS")
  expect_equal(as.integer(attr(fit, "fastPLS_internal")$ncomp), as.integer(opt$best_ncomp))
  expect_true(is.array(fit$Ypred))
  expect_equal(dim(fit$Ypred)[1], length(test_idx))
  expect_true(any(is.finite(fit$Ypred)))
  expect_true(any(is.finite(fit$Q2Y)))
})

test_that("OPLS component selection returns fitted R2 output", {
  idx <- c(1:12, 51:62, 101:112)
  X <- as.matrix(iris[idx, 1:4])
  y <- factor(iris[idx, 5])

  fit <- pls.single.cv(
    X,
    y,
    ncomp = 1:2,
    kfold = 3,
    method = "opls",
    backend = "cpu",
    classifier = "argmax",
    fit = TRUE,
    seed = 7
  )

  expect_true(fit$best_ncomp %in% 1:2)
  expect_length(fit$R2Y, 2L)
  expect_true(all(is.finite(fit$R2Y)))
  expect_length(fit$Yfit, 2L)
  expect_true(all(is.finite(fit$accuracy)))
})

test_that("joint argmax and LDA tuning reuses an identical PLS score path", {
  set.seed(99)
  X <- matrix(rnorm(240 * 28), 240, 28)
  y <- factor(rep(letters[1:4], each = 60))
  common <- list(
    Xdata = X,
    Ydata = y,
    ncomp = c(2L, 5L),
    kfold = 4L,
    method = "simpls",
    backend = "cpu",
    fit = FALSE,
    seed = 17L
  )

  argmax <- do.call(pls.single.cv, c(common, list(classifier = "argmax")))
  lda <- do.call(pls.single.cv, c(common, list(classifier = "lda")))
  joint <- do.call(
    pls.single.cv,
    c(common, list(classifier = c("argmax", "lda")))
  )
  by_head <- setNames(
    joint$tuning_results,
    vapply(
      joint$tuning_results,
      function(value) value$tuning_config$classifier,
      character(1L)
    )
  )

  expect_identical(
    lapply(by_head$argmax$pred, as.character),
    lapply(argmax$pred, as.character)
  )
  expect_identical(
    lapply(by_head$lda$pred, as.character),
    lapply(lda$pred, as.character)
  )
  expect_equal(
    by_head$argmax$selection_metrics,
    argmax$selection_metrics,
    tolerance = 0
  )
  expect_equal(
    by_head$lda$selection_metrics,
    lda$selection_metrics,
    tolerance = 0
  )
  expect_identical(by_head$argmax$best_ncomp, argmax$best_ncomp)
  expect_identical(by_head$lda$best_ncomp, lda$best_ncomp)
})

test_that("joint classifier reuse agrees on available float32 backends", {
  set.seed(101)
  X <- float::fl(matrix(rnorm(120 * 12), 120, 12))
  y <- factor(rep(LETTERS[1:3], each = 40))
  backends <- "cpu"
  if (isTRUE(has_cuda())) backends <- c(backends, "cuda")
  if (isTRUE(has_metal())) backends <- c(backends, "metal")

  for (backend in backends) {
    common <- list(
      Xdata = X,
      Ydata = y,
      ncomp = c(1L, 3L),
      kfold = 3L,
      method = "simpls",
      backend = backend,
      fit = FALSE,
      seed = 23L
    )
    argmax <- do.call(
      pls.single.cv,
      c(common, list(classifier = "argmax"))
    )
    lda <- do.call(pls.single.cv, c(common, list(classifier = "lda")))
    joint <- do.call(
      pls.single.cv,
      c(common, list(classifier = c("argmax", "lda")))
    )
    by_head <- setNames(
      joint$tuning_results,
      vapply(
        joint$tuning_results,
        function(value) value$tuning_config$classifier,
        character(1L)
      )
    )

    expect_identical(
      lapply(by_head$argmax$pred, as.character),
      lapply(argmax$pred, as.character),
      info = backend
    )
    expect_identical(
      lapply(by_head$lda$pred, as.character),
      lapply(lda$pred, as.character),
      info = backend
    )
    expect_equal(
      by_head$argmax$selection_metrics,
      argmax$selection_metrics,
      tolerance = 0,
      info = backend
    )
    expect_equal(
      by_head$lda$selection_metrics,
      lda$selection_metrics,
      tolerance = 0,
      info = backend
    )
  }
})

test_that("joint classifier reuse preserves every PLS family", {
  index <- c(1:20, 51:70, 101:120)
  X <- as.matrix(iris[index, 1:4])
  y <- droplevels(iris[index, 5])

  for (method in c("plssvd", "simpls", "opls", "kernelpls")) {
    common <- list(
      Xdata = X,
      Ydata = y,
      ncomp = 1:2,
      kfold = 3,
      method = method,
      backend = "cpu",
      fit = FALSE,
      seed = 17
    )
    argmax <- do.call(
      pls.single.cv,
      c(common, list(classifier = "argmax"))
    )
    lda <- do.call(pls.single.cv, c(common, list(classifier = "lda")))
    joint <- do.call(
      pls.single.cv,
      c(common, list(classifier = c("argmax", "lda")))
    )
    by_head <- setNames(
      joint$tuning_results,
      vapply(
        joint$tuning_results,
        function(value) value$tuning_config$classifier,
        character(1L)
      )
    )

    expect_identical(
      lapply(by_head$argmax$pred, as.character),
      lapply(argmax$pred, as.character),
      info = method
    )
    expect_identical(
      lapply(by_head$lda$pred, as.character),
      lapply(lda$pred, as.character),
      info = method
    )
    expect_identical(by_head$argmax$best_ncomp, argmax$best_ncomp)
    expect_identical(by_head$lda$best_ncomp, lda$best_ncomp)
  }
})

test_that("SIMPLS-LDA fold moments preserve the cross-validation result", {
  previous <- Sys.getenv("FASTPLS_CV_FOLD_GRAM_CACHE", unset = NA_character_)
  on.exit({
    if (is.na(previous)) {
      Sys.unsetenv("FASTPLS_CV_FOLD_GRAM_CACHE")
    } else {
      Sys.setenv(FASTPLS_CV_FOLD_GRAM_CACHE = previous)
    }
  }, add = TRUE)

  set.seed(311)
  X <- matrix(rnorm(240 * 24), 240, 24)
  y <- factor(rep(LETTERS[1:4], each = 60))
  common <- list(
    Xdata = X,
    Ydata = y,
    ncomp = c(5L, 12L, 20L),
    kfold = 4L,
    method = "simpls",
    backend = "cpu",
    classifier = "lda",
    selection = "accuracy",
    fit = FALSE,
    seed = 47L
  )

  Sys.setenv(FASTPLS_CV_FOLD_GRAM_CACHE = "0")
  explicit_scores <- do.call(pls.single.cv, common)
  Sys.setenv(FASTPLS_CV_FOLD_GRAM_CACHE = "1")
  sufficient_moments <- do.call(pls.single.cv, common)

  agreement <- vapply(seq_along(explicit_scores$pred), function(index) {
    mean(
      as.character(explicit_scores$pred[[index]]) ==
        as.character(sufficient_moments$pred[[index]])
    )
  }, numeric(1L))
  expect_true(all(agreement >= 0.995))
  expect_equal(
    explicit_scores$accuracy,
    sufficient_moments$accuracy,
    tolerance = 0.005
  )
  expect_identical(
    explicit_scores$best_ncomp,
    sufficient_moments$best_ncomp
  )
  expect_identical(explicit_scores$fold, sufficient_moments$fold)
})
