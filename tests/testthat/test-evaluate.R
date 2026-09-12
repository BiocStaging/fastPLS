test_that("evaluate computes classification metrics", {
  observed <- factor(c("a", "a", "b", "b", "c"))
  predicted <- factor(c("a", "b", "b", "b", "c"), levels = levels(observed))

  res <- evaluate(observed, predicted)

  expect_identical(res$task, "classification")
  expect_equal(res$metrics$accuracy, 4 / 5)
  expect_equal(res$metrics$no_information_rate, 2 / 5)
  expect_equal(res$metrics$lift_accuracy, 2)
  expect_true("macro_f1" %in% names(res$metrics))
  expect_equal(sum(res$confusion), 5)
  expect_false("notes" %in% names(res))
})

test_that("evaluate computes top-k accuracy from score matrices", {
  observed <- factor(c("a", "b", "c"))
  scores <- matrix(
    c(
      0.8, 0.1, 0.1,
      0.4, 0.5, 0.1,
      0.6, 0.3, 0.1
    ),
    nrow = 3,
    byrow = TRUE,
    dimnames = list(NULL, c("a", "b", "c"))
  )

  res <- evaluate(observed, scores)

  expect_equal(res$metrics$accuracy, 2 / 3)
  expect_equal(res$topk$k, 1:3)
  expect_equal(res$topk$accuracy[res$topk$k == 3L], 1)
})

test_that("evaluate infers ranks from ranked label matrices", {
  observed <- factor(c("a", "b", "c"))
  ranked <- matrix(c(
    "a", "b", "c",
    "a", "b", "c",
    "b", "a", "c"
  ), nrow = 3, byrow = TRUE)

  result <- evaluate(observed, ranked)

  expect_equal(result$metrics$accuracy, 1 / 3)
  expect_equal(result$topk$k, 1:3)
  expect_equal(result$topk$accuracy, c(1 / 3, 2 / 3, 1))
})

test_that("evaluate accepts complete classification prediction results", {
  set.seed(41)
  X <- as.matrix(iris[, seq_len(4)])
  y <- iris$Species
  fit <- pls(X[-seq_len(15), ], y[-seq_len(15)], ncomp = 1:2,
    method = "plssvd", backend = "cpu", seed = 8)
  prediction <- predict(fit, X[seq_len(15), ], top = 2)

  result <- evaluate(y[seq_len(15)], prediction)

  expect_identical(result$task, "classification")
  expect_equal(rownames(result$metrics), c("ncomp=1", "ncomp=2"))
  expect_named(result$by_component, c("ncomp=1", "ncomp=2"))
  expect_equal(result$by_component[[1]]$topk$k, 1:2)
})

test_that("evaluate accepts complete regression prediction results", {
  X <- as.matrix(mtcars[, c("disp", "hp", "wt", "qsec")])
  y <- mtcars$mpg
  fit <- pls(X[-seq_len(6), ], y[-seq_len(6)], ncomp = 1:2,
    method = "simpls", backend = "cpu", seed = 12)
  prediction <- predict(fit, X[seq_len(6), ])

  result <- evaluate(y[seq_len(6)], prediction, ytrain = y[-seq_len(6)])

  expect_identical(result$task, "regression")
  expect_equal(nrow(result$metrics), 2L)
  expect_true(all(is.finite(result$metrics$RMSD)))
})

test_that("evaluate no longer exposes task or top_k controls", {
  expect_false("task" %in% names(formals(evaluate)))
  expect_false("top_k" %in% names(formals(evaluate)))
})

test_that("evaluate computes regression and spectral metrics", {
  observed <- matrix(c(1, 2, 3, 2, 4, 6), nrow = 3, ncol = 2)
  predicted <- observed + 0.1
  train <- observed + 0.5

  res <- evaluate(observed, predicted, ytrain = train)

  expect_identical(res$task, "regression")
  expect_true(all(c("R2", "Q2", "RMSD", "MRE_percent", "RPD") %in% names(res$metrics)))
  expect_equal(nrow(res$per_response), 2)
  expect_false(isTRUE(all.equal(res$metrics$R2, res$metrics$Q2)))
  expect_false("notes" %in% names(res))
})

test_that("multivariate R2 and Q2 use response-specific reference means", {
  observed <- cbind(c(1, 2, 3), c(101, 102, 103))
  predicted <- observed + 1
  training <- cbind(c(0, 1, 2), c(100, 101, 102))

  result <- evaluate(observed, predicted, ytrain = training)
  expected_r2 <- 1 - sum((observed - predicted)^2) /
    sum(sweep(observed, 2L, colMeans(observed), "-")^2)
  expected_q2 <- 1 - sum((observed - predicted)^2) /
    sum(sweep(observed, 2L, colMeans(training), "-")^2)

  expect_equal(result$metrics$R2, expected_r2)
  expect_equal(result$metrics$Q2, expected_q2)
  expect_equal(result$metrics$Q2, 0.4)
  expect_match(result$metric_definitions$Q2, "each response")
})

test_that("evaluate can omit response-wise regression metrics", {
  observed <- matrix(c(1, 2, 3, 2, 4, 6), nrow = 3, ncol = 2)
  predicted <- observed + 0.1

  res <- evaluate(observed, predicted, bycol = FALSE)

  expect_null(res$per_response)
  expect_true(all(c("R2", "Q2", "RMSD", "MAE") %in% names(res$metrics)))
  expect_true(is.na(res$metrics$Q2))
})

test_that("evaluate only reports notes when they are informative", {
  observed <- c(1, 2, 3)
  predicted <- observed + 0.1

  res <- evaluate(observed, predicted)

  expect_true("notes" %in% names(res))
  expect_match(res$notes[[1]], "Q2 was not computed")
  expect_true(is.na(res$metrics$Q2))
  expect_match(res$metric_definitions$Q2, "requires ytrain")
})
