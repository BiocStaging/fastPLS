test_that("LOOCV leaves out one constraint group at a time", {
  idx <- c(1:8, 51:58, 101:108)
  X <- as.matrix(iris[idx, 1:4])
  y <- factor(iris[idx, 5])
  constrain <- rep(seq_len(nrow(X) / 2L), each = 2L)

  cv <- pls.single.cv(
    X,
    y,
    constrain = constrain,
    ncomp = 1,
    kfold = "loocv",
    method = "simpls",
    backend = "cpu",
    seed = 11
  )

  expect_equal(length(unique(cv$fold)), length(unique(constrain)))
  group_folds <- split(cv$fold, constrain)
  expect_true(all(vapply(group_folds, function(z) length(unique(z)) == 1L, logical(1))))
  fold_groups <- split(constrain, cv$fold)
  expect_true(all(vapply(fold_groups, function(z) length(unique(z)) == 1L, logical(1))))
})

test_that("numeric kfold at group count also means constrained LOOCV", {
  set.seed(12)
  X <- matrix(rnorm(24 * 5), 24, 5)
  y <- factor(rep(letters[1:3], each = 8))
  constrain <- rep(seq_len(12), each = 2L)

  opt <- pls.single.cv(
    X,
    y,
    constrain = constrain,
    ncomp = 1:2,
    kfold = nrow(X),
    method = "simpls",
    backend = "cpu",
    seed = 12
  )

  expect_equal(length(unique(opt$fold)), length(unique(constrain)))
  group_folds <- split(opt$fold, constrain)
  expect_true(all(vapply(group_folds, function(z) length(unique(z)) == 1L, logical(1))))
  fold_groups <- split(constrain, opt$fold)
  expect_true(all(vapply(fold_groups, function(z) length(unique(z)) == 1L, logical(1))))
})

test_that("LOOCV grouped splitting also works for regression responses", {
  set.seed(14)
  X <- matrix(rnorm(20 * 6), 20, 6)
  Y <- cbind(rnorm(20), rnorm(20))
  constrain <- rep(seq_len(10), each = 2L)

  cv <- pls.single.cv(
    X,
    Y,
    constrain = constrain,
    ncomp = 1,
    kfold = "loocv",
    method = "plssvd",
    backend = "cpu",
    seed = 14
  )

  expect_equal(length(unique(cv$fold)), length(unique(constrain)))
  group_folds <- split(cv$fold, constrain)
  expect_true(all(vapply(group_folds, function(z) length(unique(z)) == 1L, logical(1))))
  fold_groups <- split(constrain, cv$fold)
  expect_true(all(vapply(fold_groups, function(z) length(unique(z)) == 1L, logical(1))))
})

test_that("double CV accepts LOOCV for the outer grouped split", {
  idx <- c(1:8, 51:58, 101:108)
  X <- as.matrix(iris[idx, 1:4])
  y <- factor(iris[idx, 5])
  constrain <- rep(seq_len(nrow(X) / 2L), each = 2L)

  dcv <- pls.double.cv(
    X,
    y,
    constrain = constrain,
    ncomp = 1:2,
    runn = 1,
    kfold_inner = 2,
    kfold_outer = "loocv",
    method = "simpls",
    backend = "cpu",
    seed = 13
  )

  expect_equal(length(unique(dcv$results[[1]]$fold)), length(unique(constrain)))
  fold_groups <- split(constrain, dcv$results[[1]]$fold)
  expect_true(all(vapply(fold_groups, function(z) length(unique(z)) == 1L, logical(1))))
})
