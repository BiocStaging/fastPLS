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

test_that("label-aware PLSSVD model avoids dense response storage", {
  set.seed(20260514)
  X <- matrix(rnorm(80 * 12), nrow = 80, ncol = 12)
  y <- factor(sample(paste0("K", seq_len(5)), 80, replace = TRUE))
  idx <- sample(seq_len(80), 15)
  fit_fun <- get(".plssvd_label_aware_stream_model", envir = asNamespace("fastPLS"))
  fit <- fit_fun(
    X[-idx, , drop = FALSE],
    y[-idx],
    ncomp = 1:3,
    scaling = 1L,
    backend = "cpp",
    block_size = 13L
  )
  expect_true(isTRUE(fit$classification))
  expect_null(fit[["B"]])
  expect_equal(fit$xprod_mode, "label_aware_stream")

  pred <- predict(fit, X[idx, , drop = FALSE], top5 = TRUE)
  expect_equal(dim(pred$Ypred_top[[1]]), c(length(idx), 5L))
  expect_true(all(as.character(pred$Ypred[[1]]) %in% levels(y)))
})

test_that("label-aware PLSSVD class sums are invariant to first-seen class order", {
  set.seed(20260720)
  y <- factor(rep(paste0("K", 1:3), each = 12))
  X <- matrix(rnorm(36 * 7), nrow = 36, ncol = 7)
  X[, 1] <- as.integer(y) * 3 + rnorm(36, sd = 0.05)
  order <- c(25L, 1L, 13L, sample(setdiff(seq_len(36), c(25L, 1L, 13L))))
  X <- X[order, , drop = FALSE]
  y <- y[order]

  fit <- fastPLS:::.plssvd_label_aware_stream_model(
    X, y, ncomp = 1:2, scaling = 1L, backend = "cpp",
    block_size = nrow(X)
  )
  Xc <- sweep(X, 2L, colMeans(X), "-")
  Y <- fastPLS:::transformy(y)
  Yc <- sweep(Y, 2L, colMeans(Y), "-")
  reference <- svd(crossprod(Xc, Yc), nu = 2L, nv = 2L)

  alignment <- abs(crossprod(fit$R[, 1:2, drop = FALSE],
                             reference$u[, 1:2, drop = FALSE]))
  expect_equal(diag(alignment), rep(1, 2), tolerance = 1e-8)
})
