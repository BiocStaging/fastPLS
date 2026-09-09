test_that("PLS-LDA handles singular pooled covariance with the fixed ladder", {
  scores <- cbind(
    rep(c(-1, 0, 1), each = 4),
    rep(c(-1, 0, 1), each = 4),
    rep(1, 12)
  )
  labels <- rep(1:3, each = 4)

  models <- fastPLS:::lda_train_prefix_cpp(
    scores, labels, 3L, c(1L, 3L), 0.5
  )
  model <- models[["3"]]

  expect_equal(dim(model$linear), c(3L, 3L))
  expect_equal(dim(model$inv_cov), c(0L, 0L))
  expect_true(model$ridge_relative %in% c(1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2))
  expect_true(all(is.finite(model$linear)))
  expect_equal(
    fastPLS:::lda_predict_cpp(scores, model)$pred,
    labels
  )
})

test_that("PLS-LDA pooled covariance and discriminant match their definitions", {
  set.seed(123)
  scores <- matrix(rnorm(90), nrow = 30, ncol = 3)
  labels <- rep(1:3, each = 10)
  model <- fastPLS:::lda_train_prefix_cpp(
    scores, labels, 3L, 3L, 1e-8
  )[[1L]]

  means <- do.call(rbind, lapply(1:3, function(cls) {
    colMeans(scores[labels == cls, , drop = FALSE])
  }))
  pooled <- crossprod(scores)
  for (cls in 1:3) {
    pooled <- pooled - 10 * tcrossprod(means[cls, ])
  }
  pooled <- pooled / (nrow(scores) - 3)
  expected_linear <- solve(
    pooled + diag(model$ridge, ncol(scores)), t(means)
  )
  expected_linear <- t(expected_linear)
  expected_constants <- vapply(1:3, function(cls) {
    -0.5 * sum(means[cls, ] * expected_linear[cls, ]) + log(1 / 3)
  }, numeric(1L))

  expect_equal(model$means, means, tolerance = 1e-12)
  expect_equal(model$linear, expected_linear, tolerance = 1e-9)
  expect_equal(as.numeric(model$constants), expected_constants, tolerance = 1e-9)
  expected_scores <- scores %*% t(expected_linear) +
    matrix(expected_constants, nrow(scores), 3L, byrow = TRUE)
  expect_equal(
    fastPLS:::lda_predict_cpp(scores, model)$scores,
    expected_scores,
    tolerance = 1e-9
  )
})

test_that("streamed score moments use the same compiled LDA solve", {
  set.seed(126)
  scores <- matrix(rnorm(300), nrow = 60, ncol = 5)
  labels <- rep(1:3, each = 20)
  order <- sample(seq_len(nrow(scores)))
  scores <- scores[order, , drop = FALSE]
  labels <- labels[order]
  counts <- as.numeric(tabulate(labels, nbins = 3L))
  class_sums <- rowsum(scores, labels, reorder = TRUE)
  class_sums <- class_sums[as.character(seq_len(3L)), , drop = FALSE]
  components <- c(2L, 5L)

  direct <- fastPLS:::lda_train_prefix_cpp(
    scores, labels, 3L, components, 1e-8
  )
  moments <- fastPLS:::lda_train_moments_prefix_cpp(
    crossprod(scores), class_sums, counts, nrow(scores), components
  )

  for (key in names(direct)) {
    expect_equal(moments[[key]]$linear, direct[[key]]$linear,
                 tolerance = 1e-10)
    expect_equal(moments[[key]]$constants, direct[[key]]$constants,
                 tolerance = 1e-10)
    expect_equal(moments[[key]]$ridge_relative,
                 direct[[key]]$ridge_relative)
  }
})

test_that("streamed projected LDA preserves compact class order", {
  set.seed(127)
  X <- matrix(rnorm(360), nrow = 60, ncol = 6)
  R <- matrix(rnorm(30), nrow = 6, ncol = 5)
  labels <- rep(1:3, each = 20)
  order <- c(41L, 1L, 21L, sample(setdiff(seq_len(60), c(41L, 1L, 21L))))
  X <- X[order, , drop = FALSE]
  labels <- labels[order]
  offset <- seq(0.01, 0.05, length.out = 5)
  scores <- sweep(X %*% R, 2L, offset, "-")
  components <- c(2L, 5L)

  direct <- fastPLS:::lda_train_prefix_cpp(
    scores, labels, 3L, components, 1e-8
  )
  streamed <- fastPLS:::.lda_train_projected_stream(
    X, R, offset, labels, 3L, components,
    block_size = nrow(X), backend = "cpu"
  )

  for (key in names(direct)) {
    expect_equal(streamed[[key]]$linear, direct[[key]]$linear,
                 tolerance = 1e-10)
    expect_equal(streamed[[key]]$constants, direct[[key]]$constants,
                 tolerance = 1e-10)
  }
})

test_that("float32 CPU PLS-LDA agrees with the double path", {
  skip_if_not_installed("float")
  skip_on_os("windows")
  set.seed(124)
  scores <- matrix(rnorm(240), nrow = 60, ncol = 4)
  labels <- rep(1:3, each = 20)
  model64 <- fastPLS:::lda_train_prefix_cpp(
    scores, labels, 3L, c(2L, 4L), 1e-8
  )
  model32 <- fastPLS:::lda_train_prefix_float32_cpp(
    float::fl(scores), labels, 3L, c(2L, 4L)
  )

  pred64 <- fastPLS:::lda_predict_cpp(scores, model64[["4"]])$pred
  raw32 <- fastPLS:::lda_predict_float32_cpp(
    float::fl(scores), model32[["4"]]
  )
  scores32 <- methods::new("float32", Data = raw32$scores)

  expect_identical(raw32$pred, pred64)
  expect_equal(fastPLS:::.float32_to_numeric_matrix(scores32),
               fastPLS:::lda_predict_cpp(scores, model64[["4"]])$scores,
               tolerance = 2e-4)
  expect_identical(model32[["4"]]$precision, "float32")
})

test_that("float32 CPU PLS-LDA regularizes singular and near-singular covariance", {
  skip_if_not_installed("float")
  skip_on_os("windows")
  base <- rep(c(-1, 0, 1), each = 8)
  scores <- cbind(base, base, base + rep(c(0, 1e-7), 12), 1)
  labels <- rep(1:3, each = 8)
  scores32 <- float::fl(scores)

  model <- fastPLS:::lda_train_prefix_float32_cpp(
    scores32, labels, 3L, c(2L, 4L)
  )[["4"]]
  pred <- fastPLS:::lda_predict_float32_cpp(scores32, model)

  ridge_grid <- c(1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
  expect_lt(min(abs(model$ridge_relative - ridge_grid)), 1e-9)
  expect_true(all(is.finite(fastPLS:::.float32_to_numeric_matrix(
    methods::new("float32", Data = model$linear)
  ))))
  expect_identical(pred$pred, labels)
})
