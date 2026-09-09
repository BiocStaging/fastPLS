test_that("top-k classification prediction preserves argmax by default", {
  set.seed(20260511)
  X <- matrix(rnorm(96 * 10), nrow = 96, ncol = 10)
  y <- factor(sample(paste0("C", seq_len(6)), 96, replace = TRUE))
  idx <- sample(seq_len(96), 18)

  fit <- pls(
    X[-idx, , drop = FALSE],
    y[-idx],
    ncomp = 1:3,
    method = "plssvd",
    svd.method = "cpu_rsvd",
    seed = 123L
  )

  fast <- predict(fit, X[idx, , drop = FALSE])
  full <- predict(fit, X[idx, , drop = FALSE], raw_scores = TRUE)
  expect_equal(fast$Ypred, full$Ypred)

  top5 <- predict(fit, X[idx, , drop = FALSE], top5 = TRUE)
  expect_true(is.list(top5$Ypred_top))
  expect_equal(dim(top5$Ypred_top[[1]]), c(length(idx), 5L))
  expect_equal(dim(top5$Ypred_top_score[[1]]), c(length(idx), 5L))
})

test_that("standalone SIMPLS top-k prediction matches the full score path", {
  set.seed(20260909)
  X <- matrix(rnorm(140 * 18), nrow = 140, ncol = 18)
  y <- factor(sample(paste0("C", seq_len(7)), 140, replace = TRUE))
  train <- seq_len(110)
  test <- X[-train, , drop = FALSE]
  fit <- pls(
    X[train, , drop = FALSE], y[train], ncomp = c(1L, 3L, 5L),
    method = "simpls", backend = "cpu", seed = 91L
  )
  model <- fastPLS:::.fastpls_restore_internal_output_fields(fit)
  compact <- fastPLS:::.class_topk_predict(
    model, test, top = 3L, proj = TRUE, backend = "cpp"
  )
  full <- fastPLS:::pls_labels_core_predict_cpp(model, test, TRUE)
  expected <- fastPLS:::.class_topk_from_score_cube(
    full$Ypred, model$lev, model$ncomp, top = 3L
  )

  expect_identical(compact$Ypred, expected$Ypred)
  expect_identical(compact$Ypred_top, expected$Ypred_top)
  expect_equal(compact$Ypred_top_score, expected$Ypred_top_score,
               tolerance = 2e-12)
  expect_equal(compact$Ttest, full$Ttest, tolerance = 2e-12)
  expect_identical(compact$predict_backend, "core_topk")
})
