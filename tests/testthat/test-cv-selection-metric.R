test_that("pls.single.cv can optimize explicit regression metrics", {
  set.seed(2101)
  X <- matrix(rnorm(48 * 7), nrow = 48, ncol = 7)
  y <- matrix(0.6 * X[, 1] - 0.3 * X[, 2] + rnorm(48, sd = 0.2), ncol = 1)

  opt_r2 <- pls.single.cv(
    Xdata = X,
    Ydata = y,
    ncomp = 1:2,
    kfold = 3,
    method = "simpls",
    backend = "cpu",
    seed = 11,
    selection = "R2Y",
    fit = FALSE
  )
  expect_identical(opt_r2$selection_metric, "R2Y")
  expect_identical(opt_r2$best_metric_name, "R2Y")
  expect_true(opt_r2$best_ncomp %in% 1:2)
  expect_equal(opt_r2$selection_metrics$metric_value, unname(opt_r2$R2Y))
  expect_false(is.null(opt_r2$Yfit))

  opt_rmsd <- pls.single.cv(
    Xdata = X,
    Ydata = y,
    ncomp = 1:2,
    kfold = 3,
    method = "simpls",
    backend = "cpu",
    seed = 11,
    selection = "RMSD"
  )
  expect_identical(opt_rmsd$selection_metric, "RMSD")
  expect_identical(opt_rmsd$best_metric_name, "RMSD")
  expect_true(opt_rmsd$best_ncomp %in% 1:2)
  expect_false(is.null(opt_rmsd$Ypred))
  expect_false(is.null(opt_rmsd$Ypred_optim))
})

test_that("R2Y and Q2Y selection never substitute for one another", {
  q2_only <- data.frame(metric_name = "Q2Y", metric_value = 0.4)
  r2_only <- data.frame(metric_name = "R2Y", metric_value = 0.5)

  expect_error(
    fastPLS:::.cv_best_index(q2_only, "R2Y"),
    "selection = 'R2Y' is unavailable"
  )
  expect_error(
    fastPLS:::.cv_best_index(r2_only, "Q2Y"),
    "selection = 'Q2Y' is unavailable"
  )
})

test_that("single CV metric paths are named by component count", {
  set.seed(21011)
  X <- matrix(rnorm(54 * 6), nrow = 54, ncol = 6)
  y <- factor(rep(c("A", "B", "C"), each = 18))
  fit <- pls.single.cv(
    X, y, ncomp = 1:2, kfold = 3,
    method = "simpls", backend = "cpu", seed = 13
  )

  expected <- c("ncomp=1", "ncomp=2")
  expect_identical(names(fit$accuracy), expected)
  expect_identical(names(fit$balanced_accuracy), expected)
  expect_identical(names(fit$Q2Y), expected)
  expect_identical(names(fit$RMSD), expected)
})

test_that("nested R2Y permutation uses the fitted endpoint", {
  set.seed(21012)
  X <- matrix(rnorm(48 * 5), nrow = 48, ncol = 5)
  y <- 0.7 * X[, 1] - 0.2 * X[, 2] + rnorm(48, sd = 0.3)
  fit <- pls.double.cv(
    X, y, ncomp = 1:2, kfold_inner = 3, kfold_outer = 3,
    method = "simpls", backend = "cpu", selection = "R2Y",
    perm.test = TRUE, times = 2, seed = 17
  )

  heldout <- vapply(fit$results, `[[`, numeric(1L), "metric_value")
  expect_identical(fit$permutation_metric, "R2Y")
  expect_equal(fit$permutation_observed, median(heldout))
  expect_equal(heldout, fit$R2Y)

  mock <- list(
    results = list(list(metric_name = "R2Y", metric_value = 0.25)),
    R2Y = 0.95
  )
  expect_equal(fastPLS:::.double_cv_metric_values(mock, "R2Y"), 0.25)
})

test_that("a single selected regression path reuses its prediction cube", {
  prediction <- array(seq_len(24), dim = c(4L, 6L, 1L))
  selected <- fastPLS:::.cv_extract_prediction_at(
    list(Ypred = prediction),
    1L
  )

  expect_identical(selected, prediction)
})

test_that("classification R2Y selection enables the fitted path", {
  set.seed(2105)
  X <- matrix(rnorm(36 * 5), nrow = 36, ncol = 5)
  y <- factor(rep(c("A", "B"), each = 18))

  fit <- pls.single.cv(
      Xdata = X,
      Ydata = y,
      ncomp = 1:2,
      kfold = 3,
      method = "simpls",
      backend = "cpu",
      seed = 15,
      selection = "R2Y",
      fit = FALSE
  )
  expect_identical(fit$selection_metric, "R2Y")
  expect_equal(fit$selection_values, unname(fit$R2Y))
  expect_false(is.null(fit$Yfit))
})

test_that("classification CV selects by accuracy and nested CV forwards the rule", {
  set.seed(2102)
  X <- matrix(rnorm(54 * 6), nrow = 54, ncol = 6)
  y <- factor(rep(c("A", "B", "C"), each = 18))
  X[, 1] <- X[, 1] + as.numeric(y)

  opt <- pls.single.cv(
    Xdata = X,
    Ydata = y,
    ncomp = 1:2,
    kfold = 3,
    method = "plssvd",
    backend = "cpu",
    seed = 12,
    selection = "accuracy"
  )
  expect_identical(opt$selection_metric, "accuracy")
  expect_identical(opt$best_metric_name, "accuracy")
  expect_true(opt$best_ncomp %in% 1:2)
  expect_false(is.null(opt$class_pred))
  expect_false(is.null(opt$Ypred_optim))

  nested <- pls.double.cv(
    Xdata = X,
    Ydata = y,
    ncomp = 1:2,
    runn = 1,
    kfold_inner = 3,
    kfold_outer = 3,
    method = "plssvd",
    backend = "cpu",
    seed = 13,
    selection = "accuracy"
  )
  expect_identical(nested$selection_metric, "accuracy")
  expect_true(all(vapply(nested$results[[1]]$inner, function(x) {
    is.null(x) || identical(x$selection_metric, "accuracy")
  }, logical(1))))
})

test_that("balanced accuracy drives both classification selection and permutation", {
  set.seed(21021)
  y <- factor(c(rep("major", 30), rep("minor_a", 9), rep("minor_b", 6)))
  X <- matrix(rnorm(length(y) * 6), nrow = length(y), ncol = 6)
  X[, 1] <- X[, 1] + ifelse(y == "minor_a", 1.5, 0)
  X[, 2] <- X[, 2] + ifelse(y == "minor_b", 2, 0)

  single <- pls.single.cv(
    Xdata = X,
    Ydata = y,
    ncomp = 1:2,
    kfold = 3,
    method = "plssvd",
    backend = "cpu",
    seed = 121,
    selection = "bacc",
    fit = FALSE
  )
  expect_identical(single$selection_metric, "balanced_accuracy")
  expect_identical(single$best_metric_name, "balanced_accuracy")
  expect_true(all(single$selection_metrics$metric_name == "balanced_accuracy"))
  expect_equal(single$balanced_accuracy, single$selection_values)

  nested <- pls.double.cv(
    Xdata = X,
    Ydata = y,
    ncomp = 1:2,
    runn = 1,
    kfold_inner = 2,
    kfold_outer = 2,
    method = "plssvd",
    backend = "cpu",
    classifier = "lda",
    seed = 122,
    selection = "balanced_accuracy",
    perm.test = TRUE,
    times = 2
  )
  expect_identical(nested$selection_metric, "balanced_accuracy")
  expect_identical(nested$permutation_metric, "balanced_accuracy")
  expect_length(nested$balanced_accuracy, 1L)
  expect_length(nested$permutation_sampled, 2L)
  expect_equal(
    nested$permutation_observed,
    median(nested$balanced_accuracy, na.rm = TRUE)
  )
  expect_identical(nested$metrics$permutation$metric, "balanced_accuracy")
  expect_null(nested$Q2Ysampled)
})

test_that("SIMPLS CV always stores prediction scores", {
  set.seed(2103)
  X <- matrix(rnorm(60 * 8), nrow = 60, ncol = 8)
  y <- matrix(0.7 * X[, 1] - 0.5 * X[, 3] + rnorm(60, sd = 0.25), ncol = 1)

  cv <- pls.single.cv(
    Xdata = X,
    Ydata = y,
    ncomp = 1:3,
    kfold = 3,
    method = "simpls",
    backend = "cpu",
    seed = 21,
    selection = "rmsd"
  )

  expect_false(is.null(cv$Ypred))
  expect_equal(dim(cv$Ypred), c(nrow(X), ncol(y), 3L))
  expect_equal(cv$best_ncomp, cv$ncomp[[cv$best_index]])
})

test_that("regression CV reports distinct training R2 and held-out Q2", {
  set.seed(2104)
  X <- matrix(rnorm(80 * 12), nrow = 80, ncol = 12)
  beta <- matrix(rnorm(12 * 2), nrow = 12, ncol = 2)
  Y <- X %*% beta + matrix(rnorm(80 * 2, sd = 2), nrow = 80, ncol = 2)

  scalar_fit <- pls(
    Xtrain = X,
    Ytrain = Y,
    ncomp = 3,
    method = "simpls",
    backend = "cpu",
    fit = TRUE,
    return_variance = FALSE,
    seed = 31
  )
  path_fit <- pls(
    Xtrain = X,
    Ytrain = Y,
    ncomp = 1:3,
    method = "simpls",
    backend = "cpu",
    fit = TRUE,
    return_variance = FALSE,
    seed = 31
  )
  expect_equal(scalar_fit$R2Y[[1]], path_fit$R2Y[[3]], tolerance = 1e-10)

  single <- pls.single.cv(
    Xdata = X,
    Ydata = Y,
    ncomp = 1:3,
    kfold = 4,
    method = "simpls",
    backend = "cpu",
    seed = 32,
    selection = "Q2Y"
  )
  expect_false(isTRUE(all.equal(single$Q2Y, single$R2Y)))
  expect_false(isTRUE(all.equal(single$Q2Y, single$RMSD)))
  expect_false(isTRUE(all.equal(single$R2Y, single$RMSD)))

  nested <- pls.double.cv(
    Xdata = X,
    Ydata = Y,
    ncomp = 1:3,
    runn = 1,
    kfold_inner = 3,
    kfold_outer = 3,
    method = "simpls",
    backend = "cpu",
    seed = 33,
    selection = "Q2Y"
  )
  expect_false(isTRUE(all.equal(nested$Q2Y, nested$R2Y)))
  expect_false(isTRUE(all.equal(nested$Q2Y, nested$RMSD)))
  expect_false(isTRUE(all.equal(nested$R2Y, nested$RMSD)))
})

test_that("pls.single.cv can skip the extra full-data fit", {
  set.seed(21045)
  X <- matrix(rnorm(48 * 7), nrow = 48, ncol = 7)
  beta <- matrix(rnorm(7 * 2), nrow = 7, ncol = 2)
  Y <- X %*% beta + matrix(rnorm(48 * 2, sd = 0.25), nrow = 48, ncol = 2)

  with_fit <- pls.single.cv(
    Xdata = X,
    Ydata = Y,
    ncomp = 1:3,
    kfold = 4,
    method = "simpls",
    backend = "cpu",
    seed = 21045,
    fit = TRUE
  )
  without_fit <- pls.single.cv(
    Xdata = X,
    Ydata = Y,
    ncomp = 1:3,
    kfold = 4,
    method = "simpls",
    backend = "cpu",
    seed = 21045,
    fit = FALSE
  )

  expect_true(any(is.finite(with_fit$R2Y)))
  expect_false(is.null(with_fit$Yfit))
  expect_true(all(is.na(without_fit$R2Y)))
  expect_null(without_fit$Yfit)
  expect_true(any(is.finite(without_fit$Q2Y)))
  expect_true(any(is.finite(without_fit$RMSD)))
  expect_equal(without_fit$best_ncomp, with_fit$best_ncomp)
})

test_that("classification CV keeps held-out accuracy separate from training R2", {
  X <- as.matrix(iris[, 1:4])
  y <- factor(iris[, 5])

  with_fit <- pls.single.cv(
    Xdata = X,
    Ydata = y,
    ncomp = 2,
    kfold = 3,
    method = "simpls",
    backend = "cpu",
    seed = 21046,
    fit = TRUE
  )
  without_fit <- pls.single.cv(
    Xdata = X,
    Ydata = y,
    ncomp = 2,
    kfold = 3,
    method = "simpls",
    backend = "cpu",
    seed = 21046,
    fit = FALSE
  )
  full_fit <- pls(
    Xtrain = X,
    Ytrain = y,
    ncomp = 2,
    method = "simpls",
    backend = "cpu",
    fit = TRUE,
    return_variance = FALSE,
    seed = 21046
  )

  expect_true(is.finite(with_fit$Q2Y))
  expect_true(is.finite(with_fit$R2Y))
  expect_true(is.finite(with_fit$accuracy))
  expect_false(is.null(with_fit$Yfit))
  expect_equal(as.numeric(with_fit$R2Y), as.numeric(full_fit$R2Y), tolerance = 1e-10)
  expect_false(isTRUE(all.equal(with_fit$Q2Y, with_fit$R2Y)))
  expect_false(isTRUE(all.equal(with_fit$Q2Y, with_fit$accuracy)))
  expect_true(all(is.na(without_fit$R2Y)))
  expect_null(without_fit$Yfit)
  expect_equal(without_fit$Q2Y, with_fit$Q2Y)
  expect_equal(without_fit$accuracy, with_fit$accuracy)
})

test_that("classification double CV reports Q2, R2, and accuracy separately", {
  X <- as.matrix(iris[, 1:4])
  y <- factor(iris[, 5])

  nested <- pls.double.cv(
    Xdata = X,
    Ydata = y,
    ncomp = 1:2,
    runn = 1,
    kfold_inner = 2,
    kfold_outer = 3,
    method = "simpls",
    backend = "cpu",
    seed = 21047
  )

  expect_true(is.finite(nested$Q2Y))
  expect_true(is.finite(nested$R2Y))
  expect_true(is.finite(nested$accuracy))
  expect_false(isTRUE(all.equal(nested$Q2Y, nested$R2Y)))
  expect_false(isTRUE(all.equal(nested$Q2Y, nested$accuracy)))
})

test_that("double CV omits repeated-run summaries for a single run", {
  set.seed(21048)
  X <- matrix(rnorm(36 * 5), nrow = 36, ncol = 5)
  y <- drop(X[, 1] - 0.5 * X[, 2] + rnorm(36, sd = 0.2))

  single_run <- pls.double.cv(
    Xdata = X,
    Ydata = y,
    ncomp = 1:2,
    runn = 1,
    kfold_inner = 2,
    kfold_outer = 2,
    method = "simpls",
    backend = "cpu",
    seed = 21048
  )
  repeated <- pls.double.cv(
    Xdata = X,
    Ydata = y,
    ncomp = 1:2,
    runn = 2,
    kfold_inner = 2,
    kfold_outer = 2,
    method = "simpls",
    backend = "cpu",
    seed = 21048
  )

  summary_names <- c("medianR2Y", "CI95R2Y", "medianQ2Y", "CI95Q2Y", "medianRMSD", "CI95RMSD")
  expect_false(any(summary_names %in% names(single_run)))
  expect_true(all(summary_names %in% names(repeated)))
})

test_that("RMSD selection does not overwrite Q2Y", {
  set.seed(2105)
  X <- matrix(rnorm(72 * 9), nrow = 72, ncol = 9)
  beta <- matrix(rnorm(9 * 3), nrow = 9, ncol = 3)
  Y <- X %*% beta + matrix(rnorm(72 * 3, sd = 1.5), nrow = 72, ncol = 3)

  single <- pls.single.cv(
    Xdata = X,
    Ydata = Y,
    ncomp = 1:3,
    kfold = 4,
    method = "simpls",
    backend = "cpu",
    seed = 34,
    selection = "RMSD"
  )
  expect_identical(single$best_metric_name, "RMSD")
  expect_false(isTRUE(all.equal(single$Q2Y, single$RMSD)))
  expect_equal(single$selection_values, single$RMSD, tolerance = 1e-10)

  nested <- pls.double.cv(
    Xdata = X,
    Ydata = Y,
    ncomp = 1:3,
    runn = 1,
    kfold_inner = 3,
    kfold_outer = 3,
    method = "simpls",
    backend = "cpu",
    seed = 35,
    selection = "RMSD"
  )
  expect_identical(nested$metric_name[[1]], "RMSD")
  expect_false(isTRUE(all.equal(nested$Q2Y, nested$RMSD)))
  expect_equal(nested$results[[1]]$metric_value, nested$RMSD[[1]], tolerance = 1e-10)
})

test_that("selection metrics are task specific and unambiguous", {
  set.seed(2106)
  X <- matrix(rnorm(42 * 5), nrow = 42)
  y_reg <- X[, 1] + rnorm(42)
  y_cls <- factor(rep(c("A", "B"), each = 21))

  expect_error(
    pls.single.cv(X, y_reg, ncomp = 1:2, kfold = 3,
      selection = "accuracy"),
    "not valid for regression"
  )
  expect_error(
    pls.double.cv(X, y_cls, ncomp = 1:2, kfold_inner = 2,
      kfold_outer = 2, selection = "RMSD"),
    "not valid for classification"
  )
  expect_error(
    pls.single.cv(X, y_reg, ncomp = 1:2, kfold = 3, selection = "r2"),
    "use 'R2Y'"
  )
  expect_error(
    pls.single.cv(X, y_reg, ncomp = 1:2, kfold = 3, selection = "q2"),
    "use 'Q2Y'"
  )
  expect_error(
    pls.single.cv(X, y_reg, ncomp = 1:2, kfold = 3,
      selection = "MRE_percent"),
    "signed and has no unambiguous optimization direction"
  )
  expect_error(
    pls.single.cv(X, y_reg, ncomp = 1:2, kfold = 3, selection = "RMSE"),
    "duplicates RMSD"
  )
})

test_that("multivariate regression can select evaluate metrics", {
  set.seed(2107)
  X <- matrix(rnorm(54 * 7), nrow = 54)
  Y <- cbind(
    X[, 1] + rnorm(54, sd = 0.2),
    2 * X[, 2] + rnorm(54, sd = 0.3),
    X[, 3] - X[, 4] + rnorm(54, sd = 0.2)
  )
  single <- pls.single.cv(
    X, Y, ncomp = 1:2, kfold = 3, fit = FALSE, selection = "MAE"
  )
  expect_identical(single$selection_metric, "MAE")
  expect_true(all(single$selection_metrics$metric_name == "MAE"))
  expect_equal(
    single$best_index,
    which.min(single$selection_metrics$metric_value)
  )

  nested <- pls.double.cv(
    X, Y, ncomp = 1:2, kfold_inner = 2, kfold_outer = 2,
    selection = "Pearson_r"
  )
  expect_identical(nested$selection_metric, "Pearson_r")
  expect_identical(nested$metric_name[[1L]], "Pearson_r")
  expect_true(is.finite(nested$results[[1L]]$metric_value))
})
