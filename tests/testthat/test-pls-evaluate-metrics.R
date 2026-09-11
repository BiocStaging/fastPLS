test_that("pls includes complete evaluate metrics for classification", {
  idx <- c(1:12, 51:62, 101:112)
  X <- as.matrix(iris[idx, 1:4])
  y <- factor(iris[idx, 5])

  fit <- pls(
    X, y, X, y,
    ncomp = 1:2,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    fit = TRUE,
    return_variance = FALSE
  )

  expect_named(fit$metrics, c("definitions", "fitted", "test"))
  expect_named(fit$metrics$test[["ncomp=1"]],
               c("task", "metrics", "metric_definitions", "per_class", "confusion", "topk"))
  expect_true("lift_accuracy" %in% names(fit$metrics$test[["ncomp=1"]]$metrics))
})

test_that("float32 classification decodes fitted labels for metrics", {
  idx <- c(1:12, 51:62, 101:112)
  X <- float::fl(as.matrix(iris[idx, 1:4]))
  y <- factor(iris[idx, 5])

  fit <- suppressWarnings(pls(
    X, y, X, y,
    ncomp = 1:2,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    fit = TRUE,
    return_variance = FALSE
  ))

  expect_length(fit$metrics$fitted, 2L)
  expect_length(fit$metrics$test, 2L)
  expect_true(all(vapply(fit$Yfit, is.factor, logical(1L))))
  expect_true(all(vapply(fit$metrics$fitted, function(value) {
    is.finite(value$metrics$accuracy[[1L]])
  }, logical(1L))))
})

test_that("special PLS-family wrappers retain component-wise test metrics", {
  set.seed(205)
  X <- matrix(rnorm(48 * 7), 48, 7)
  Y <- cbind(
    X[, 1] - 0.3 * X[, 3] + rnorm(48, sd = 0.1),
    X[, 2] + 0.2 * X[, 4] + rnorm(48, sd = 0.1)
  )
  train <- 1:36
  test <- 37:48

  configurations <- list(
    opls = list(method = "opls", north = 1L),
    kernel_linear = list(method = "kernelpls", kernel = "linear"),
    kernel_rbf = list(
      method = "kernelpls", kernel = "rbf", gamma = 1 / ncol(X)
    ),
    kernel_poly = list(
      method = "kernelpls", kernel = "poly", degree = 2L, coef0 = 1
    )
  )

  for (configuration in configurations) {
    fit <- do.call(pls, c(list(
      Xtrain = X[train, ], Ytrain = Y[train, ],
      Xtest = X[test, ], Ytest = Y[test, ],
      ncomp = 1:2,
      backend = "cpu",
      svd.method = "rsvd",
      fit = TRUE,
      return_variance = FALSE,
      seed = 9
    ), configuration))
    expect_length(fit$metrics$fitted, 2L)
    expect_length(fit$metrics$test, 2L)
    expect_true(
      all(vapply(fit$metrics$test, function(value) {
        is.data.frame(value$metrics) && is.finite(value$metrics$RMSD[[1L]])
      }, logical(1L)))
    )
  }
})

test_that("single and double CV expose evaluate metrics separately", {
  idx <- c(1:12, 51:62, 101:112)
  X <- as.matrix(iris[idx, 1:4])
  y <- factor(iris[idx, 5])

  single <- pls.single.cv(
    X, y, ncomp = 1:2, kfold = 3,
    method = "simpls", backend = "cpu", svd.method = "rsvd", seed = 1
  )
  expect_named(single$metrics, c("definitions", "cross_validated", "fitted"))
  expect_true("balanced_accuracy" %in%
    names(single$metrics$cross_validated[["ncomp=1"]]$metrics))
  expect_true(is.data.frame(single$selection_metrics))

  nested <- pls.double.cv(
    X, y, ncomp = 1:2, runn = 1,
    kfold_inner = 2, kfold_outer = 2,
    method = "simpls", backend = "cpu", svd.method = "rsvd", seed = 1
  )
  expect_named(nested$metrics, c("definitions", "cross_validated", "aggregate"))
  expect_true("macro_f1" %in% names(nested$metrics$aggregate$metrics))
})

test_that("PLS metric paths honor bycol and retain permutation metrics", {
  set.seed(2)
  X <- matrix(rnorm(60), nrow = 15, ncol = 4)
  Y <- cbind(X[, 1] + rnorm(15, sd = 0.1), X[, 2] + rnorm(15, sd = 0.1))

  aggregate <- pls(
    X, Y, X, Y, ncomp = 1, method = "simpls", backend = "cpu",
    svd.method = "rsvd", return_variance = FALSE
  )
  detailed <- pls(
    X, Y, X, Y, ncomp = 1, method = "simpls", backend = "cpu",
    svd.method = "rsvd", bycol = TRUE, return_variance = FALSE
  )
  expect_null(aggregate$metrics$test[["ncomp=1"]]$per_response)
  expect_equal(nrow(detailed$metrics$test[["ncomp=1"]]$per_response), ncol(Y))

  single <- pls.single.cv(
    X, Y, ncomp = 1, kfold = 2, method = "simpls", backend = "cpu",
    svd.method = "rsvd", bycol = TRUE, seed = 1
  )
  expect_equal(nrow(single$metrics$cross_validated[["ncomp=1"]]$per_response), ncol(Y))

  nested <- pls.double.cv(
    X, Y, ncomp = 1, runn = 1, kfold_inner = 2, kfold_outer = 2,
    method = "simpls", backend = "cpu", svd.method = "rsvd",
    bycol = TRUE, seed = 1
  )
  expect_equal(nrow(nested$metrics$aggregate$per_response), ncol(Y))

  permuted <- pls(
    X, Y, X, Y, ncomp = 1, method = "simpls", backend = "cpu",
    svd.method = "rsvd", perm.test = TRUE, times = 2,
    return_variance = FALSE
  )
  expect_true(all(
    c("results", "p_value", "requested", "completed", "failed") %in%
      names(permuted$metrics$permutation)
  ))
})

test_that("single-split permutation p-values are calculated per component", {
  set.seed(204)
  X <- matrix(rnorm(72), nrow = 18, ncol = 4)
  Y <- cbind(X[, 1] + rnorm(18, sd = 0.2), X[, 2] + rnorm(18, sd = 0.2))

  fit <- pls(
    X, Y, X, Y, ncomp = 1:2, method = "simpls", backend = "cpu",
    svd.method = "rsvd", perm.test = TRUE, times = 4, seed = 17,
    return_variance = FALSE
  )

  perm_q2 <- subset(
    fit$permutation,
    type == "permutation" & metric == "Q2"
  )
  component_values <- sort(unique(perm_q2$ncomp))
  expected <- vapply(seq_along(fit$Q2Y), function(j) {
    values <- perm_q2$value[perm_q2$ncomp == component_values[[j]]]
    valid <- is.finite(values)
    (sum(values[valid] >= fit$Q2Y[[j]]) + 1) / (sum(valid) + 1)
  }, numeric(1L))
  names(expected) <- names(fit$Q2Y)

  expect_equal(fit$pval, expected)
  expect_true(all(fit$pval > 0))
  expect_identical(names(fit$pval), names(fit$Q2Y))
})

test_that("compiled aggregate regression evaluation matches reference formulas", {
  set.seed(219)
  observed <- matrix(rnorm(105), 21, 5)
  predicted <- observed + matrix(rnorm(105, sd = 0.2), 21, 5)
  training <- matrix(rnorm(155), 31, 5)
  observed[2, 3] <- NA_real_
  predicted[5, 4] <- Inf

  keep <- is.finite(observed) & is.finite(predicted)
  observed_complete <- observed[keep]
  predicted_complete <- predicted[keep]
  error <- predicted_complete - observed_complete
  relative <- abs(error / observed_complete) * 100
  sse <- sum(error^2)
  observed_tss <- sum(vapply(seq_len(ncol(observed)), function(column) {
    column_keep <- keep[, column]
    values <- observed[column_keep, column]
    sum((values - mean(values))^2)
  }, numeric(1)))
  training_tss <- sum(vapply(seq_len(ncol(observed)), function(column) {
    column_keep <- keep[, column]
    sum((observed[column_keep, column] - mean(training[, column]))^2)
  }, numeric(1)))
  rmsd <- sqrt(mean(error^2))
  expected <- c(
    n = length(error), R2 = 1 - sse / observed_tss,
    Q2 = 1 - sse / training_tss, RMSD = rmsd, RMSE = rmsd,
    MAE = mean(abs(error)), bias = mean(error),
    MRE_percent = median(relative), MAPE_percent = mean(relative),
    RPD = stats::sd(observed_complete) / rmsd,
    Pearson_r = stats::cor(observed_complete, predicted_complete),
    Spearman_r = stats::cor(
      observed_complete, predicted_complete, method = "spearman"
    )
  )
  actual <- unlist(evaluate(
    observed, predicted, ytrain = training, bycol = FALSE
  )$metrics[1, ], use.names = TRUE)

  expect_equal(actual, expected, tolerance = 1e-12)
})
