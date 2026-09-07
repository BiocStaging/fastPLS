test_that("independent-test Q2 uses the training-response mean", {
  observed <- matrix(c(2, 4, 6, 8), ncol = 1)
  predicted <- matrix(c(2, 3, 7, 8), ncol = 1)
  training <- matrix(c(0, 2, 4), ncol = 1)

  expected <- 1 - sum((observed - predicted)^2) /
    sum((observed - mean(training))^2)
  result <- evaluate(observed, predicted, ytrain = training)

  expect_equal(result$metrics$Q2, expected)
  expect_match(result$metric_definitions$Q2, "training-response mean")
  expect_false(isTRUE(all.equal(result$metrics$Q2, result$metrics$R2)))
})

test_that("fold-aware Q2 uses each fold training mean", {
  observed <- matrix(c(1, 2, 10, 11), ncol = 1)
  predicted <- matrix(c(1.5, 2.5, 9.5, 10.5), ncol = 1)
  fold <- c(0L, 0L, 1L, 1L)

  press <- sum((observed - predicted)^2)
  tss <- sum((observed[fold == 0L] - mean(observed[fold == 1L]))^2) +
    sum((observed[fold == 1L] - mean(observed[fold == 0L]))^2)
  expected <- 1 - press / tss

  expect_equal(
    fastPLS:::.fastpls_fold_q2_path(observed, predicted, fold),
    expected
  )
})

test_that("pls independent-test Q2 uses the fitted training mean", {
  set.seed(41)
  X <- matrix(rnorm(80 * 6), 80, 6)
  y <- matrix(3 + X[, 1] - 0.4 * X[, 2] + rnorm(80, sd = 0.2), ncol = 1)
  train <- seq_len(60)
  test <- setdiff(seq_len(80), train)
  fit <- pls(
    X[train, ], y[train, , drop = FALSE],
    X[test, ], y[test, , drop = FALSE],
    ncomp = 2, method = "simpls", svd.method = "rsvd",
    backend = "cpu", return_variance = FALSE
  )
  pred <- if (length(dim(fit$Ypred)) == 3L) {
    matrix(fit$Ypred[, , 1L], ncol = 1L)
  } else if (is.list(fit$Ypred)) {
    as.matrix(fit$Ypred[[1L]])
  } else {
    as.matrix(fit$Ypred)
  }
  expected <- 1 - sum((y[test, ] - pred)^2) /
    sum((y[test, ] - mean(y[train, ]))^2)

  expect_equal(unname(fit$Q2Y[[1L]]), expected, tolerance = 1e-10)
  expect_match(fit$metrics$definitions$Q2Y, "training-response mean")
})

test_that("single CV reports fold-specific Q2", {
  set.seed(42)
  X <- matrix(rnorm(48 * 5), 48, 5)
  y <- matrix(2 + X[, 1] + rnorm(48, sd = 0.3), ncol = 1)
  cv <- pls.single.cv(
    X, y, ncomp = 1:2, kfold = 4, method = "simpls",
    svd.method = "rsvd", backend = "cpu", fit = FALSE, seed = 7
  )
  expected <- fastPLS:::.fastpls_fold_q2_path(y, cv$Ypred, cv$fold)

  expect_equal(unname(cv$Q2Y), expected, tolerance = 1e-10)
  expect_match(cv$metrics$definitions$Q2Y, "fold-training response mean")
  expect_true(all(is.na(cv$R2Y)))
})

test_that("double CV reports outer-fold-specific Q2", {
  set.seed(43)
  X <- matrix(rnorm(42 * 5), 42, 5)
  y <- matrix(1.5 + 0.8 * X[, 1] - 0.3 * X[, 2] + rnorm(42, sd = 0.25), ncol = 1)
  cv <- pls.double.cv(
    X, y, ncomp = 1:2, kfold_outer = 3, kfold_inner = 3,
    runn = 1, method = "simpls", svd.method = "rsvd",
    backend = "cpu", seed = 9
  )
  run <- cv$results[[1L]]
  expected <- fastPLS:::.fastpls_fold_q2_path(y, run$Ypred, run$fold)

  expect_equal(unname(cv$Q2Y[[1L]]), expected, tolerance = 1e-10)
  expect_match(cv$metrics$definitions$Q2Y, "outer-training response mean")
})

test_that("internal Q2 calculation requires an explicit reference", {
  observed <- matrix(1:4, ncol = 1)
  predicted <- observed + 0.1

  expect_error(
    fastPLS:::.cv_metric_from_matrix(observed, predicted, metric = "q2"),
    "explicit training-response reference"
  )
})

test_that("classification Q2 is labelled as a dummy-response metric", {
  definitions <- fastPLS:::.fastpls_metric_definitions(
    "single_cv", classification = TRUE
  )
  expect_match(definitions$Q2Y, "dummy-response Q2")
  expect_match(definitions$Q2Y, "not classification accuracy")
})
