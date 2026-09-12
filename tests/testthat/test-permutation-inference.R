test_that("Monte Carlo permutation p-values use correction and valid fits", {
  expect_equal(
    fastPLS:::.fastpls_permutation_pvalue(c(0.2, NA, 0.5), 0.4),
    2 / 3
  )
  expect_equal(
    fastPLS:::.fastpls_permutation_pvalue(c(0.2, NA, 0.5), 0.4, lower_tail = TRUE),
    2 / 3
  )
  expect_true(is.na(
    fastPLS:::.fastpls_permutation_pvalue(c(NA_real_, NA_real_), 0.4)
  ))
})

test_that("constraint-block permutations preserve complete equal-size groups", {
  constrain <- rep(c("p1", "p2", "p3", "p4"), c(2, 2, 3, 3))
  indices <- fastPLS:::.fastpls_permutation_indices(constrain, times = 5, seed = 91)
  groups <- split(seq_along(constrain), constrain)

  expect_length(indices, 5L)
  for (idx in indices) {
    expect_equal(sort(idx), seq_along(constrain))
    expect_false(identical(idx, seq_along(constrain)))
    for (target in groups) {
      source_groups <- unique(constrain[idx[target]])
      expect_length(source_groups, 1L)
      expect_equal(length(target), sum(constrain == source_groups))
    }
  }
})

test_that("grouped nested permutation reports its inferential contract", {
  set.seed(92)
  constrain <- rep(seq_len(12), each = 2)
  y_group <- factor(rep(c("A", "B"), each = 6))
  y <- y_group[constrain]
  X <- matrix(rnorm(24 * 5), nrow = 24)
  X[, 1] <- X[, 1] + ifelse(y == "B", 0.7, -0.7)

  fit <- pls.double.cv(
    Xdata = X,
    Ydata = y,
    ncomp = 1,
    constrain = constrain,
    kfold_inner = 2,
    kfold_outer = 2,
    method = "simpls",
    backend = "cpu",
    classifier = "argmax",
    selection_metric = "balanced_accuracy",
    perm.test = TRUE,
    times = 2,
    seed = 93
  )

  expect_identical(
    fit$permutation_unit,
    "constraint groups within equal-size exchangeability strata"
  )
  expect_true(fit$permutation_group_sizes_preserved)
  expect_true(fit$permutation_class_frequencies_preserved)
  expect_identical(fit$permutation_folds, "fixed across observed and null fits")
  expect_identical(
    fit$permutation_solver_seed,
    "fixed across observed and null fits"
  )
  expect_equal(fit$permutation_completed + fit$permutation_failed, 2L)
  expect_gte(fit$p.value, 1 / (fit$permutation_completed + 1))
  expect_identical(fit$metrics$permutation$completed, fit$permutation_completed)
})
