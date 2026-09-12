test_that("plot.permutation renders stored PLS permutation diagnostics", {
  set.seed(21055)
  X <- matrix(rnorm(28 * 5), nrow = 28, ncol = 5)
  y <- drop(X[, 1] * 0.4 + rnorm(28, sd = 0.5))
  fit <- pls(
    Xtrain = X,
    Ytrain = y,
    Xtest = X,
    Ytest = y,
    ncomp = 2,
    method = "simpls",
    backend = "cpu",
    fit = TRUE,
    perm.test = TRUE,
    times = 3,
    seed = 21055,
    return_variance = FALSE
  )

  expect_s3_class(fit, "fastPLS")
  expect_true(is.data.frame(fit$permutation))
  expect_true(all(c("type", "permutation", "ncomp", "metric", "cor", "value") %in% names(fit$permutation)))
  expect_true(all(c("R2", "Q2") %in% fit$permutation$metric))
  expect_true(any(fit$permutation$type == "observed"))

  pdf_file <- tempfile(fileext = ".pdf")
  grDevices::pdf(pdf_file)
  plotted <- plot.permutation(fit, ncomp = 2)
  grDevices::dev.off()
  expect_true(file.exists(pdf_file))
  expect_true(is.data.frame(plotted))
  expect_true(nrow(plotted) > 0)
})
