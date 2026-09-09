legacy_predict_core <- function(model, Xtest, proj = FALSE) {
  restore_internal <- get(".fastpls_restore_internal_output_fields", envir = asNamespace("fastPLS"))
  model <- restore_internal(model)
  m <- model$m
  w <- nrow(Xtest)
  length_ncomp <- length(model$ncomp)
  Xwork <- Xtest
  Xwork <- sweep(Xwork, 2, as.numeric(model$mX), FUN = "-")
  Xwork <- sweep(Xwork, 2, as.numeric(model$vX), FUN = "/")
  mY <- as.numeric(model$mY)
  B <- model$B
  RR <- model$R

  Ypred <- array(0, dim = c(w, m, length_ncomp))
  for (a in seq_len(length_ncomp)) {
    Ypred[, , a] <- Xwork %*% B[, , a]
    Ypred[, , a] <- sweep(Ypred[, , a, drop = FALSE], 2, mY, FUN = "+")
  }

  Ttest <- matrix(numeric(0), 0, 0)
  if (proj) {
    Ttest <- Xwork %*% RR
  }

  list(Ypred = Ypred, Ttest = Ttest)
}

test_that("optimized prediction core preserves regression outputs", {
  set.seed(20260323)
  X <- matrix(rnorm(220 * 24), nrow = 220, ncol = 24)
  Y <- cbind(
    0.7 * X[, 1] - 0.4 * X[, 2] + rnorm(220, sd = 0.15),
    -0.2 * X[, 3] + 0.5 * X[, 4] + rnorm(220, sd = 0.15)
  )
  idx <- sample(seq_len(nrow(X)), 40)

  fit <- fastPLS::pls(
    X[-idx, , drop = FALSE],
    Y[-idx, , drop = FALSE],
    X[idx, , drop = FALSE],
    Y[idx, , drop = FALSE],
    ncomp = c(1L, 3L, 5L),
    method = "simpls",
    svd.method = "cpu_rsvd",
    fit = TRUE,
    seed = 123L
  )

  model <- unclass(fit)
  Xtest <- X[idx, , drop = FALSE]
  optimized <- fastPLS:::pls_labels_core_predict_cpp(
    get(".fastpls_restore_internal_output_fields", envir = asNamespace("fastPLS"))(model),
    Xtest,
    TRUE
  )
  legacy <- legacy_predict_core(model, Xtest, TRUE)

  expect_equal(dim(optimized$Ypred), dim(legacy$Ypred))
  expect_equal(dim(optimized$Ttest), dim(legacy$Ttest))
  expect_equal(optimized$Ypred, legacy$Ypred, tolerance = 1e-12)
  expect_equal(optimized$Ttest, legacy$Ttest, tolerance = 1e-12)
})
