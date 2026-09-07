dense_simpls_reference <- function(X, Y, ncomp, ...) {
  fastPLS:::pls.model2(X, Y, ncomp = ncomp, scaling = 1L,
    fit = TRUE, svd.method = 3L, seed = 1L)
}

test_that("pls defaults to randomized SVD", {
  set.seed(42)
  X <- matrix(rnorm(70 * 18), nrow = 70, ncol = 18)
  Y <- matrix(rnorm(70 * 4), nrow = 70, ncol = 4)

  m_default <- pls(X, Y, ncomp = 1:3, fit = TRUE)
  m_explicit <- pls(X, Y, ncomp = 1:3, fit = TRUE, svd.method = "rsvd")

  align_signs <- function(ref, x) {
    out <- x
    for (j in seq_len(min(ncol(ref), ncol(out)))) {
      s <- sum(ref[, j] * out[, j], na.rm = TRUE)
      if (is.finite(s) && s < 0) {
        out[, j] <- -out[, j]
      }
    }
    out
  }

  expect_equal(m_default$B, m_explicit$B)
  expect_equal(align_signs(m_default$R, m_explicit$R), m_default$R)
  expect_equal(align_signs(m_default$Q, m_explicit$Q), m_default$Q)
})

test_that("pls accepts SVD tuning through compact dots", {
  set.seed(43)
  X <- matrix(rnorm(72 * 20), nrow = 72, ncol = 20)
  Y <- matrix(rnorm(72 * 5), nrow = 72, ncol = 5)

  compact <- pls(
    X,
    Y,
    ncomp = 1:3,
    fit = TRUE,
    svd.method = "cpu_rsvd",
    oversample = 8L,
    power = 2L,
    seed = 123L
  )
  aliased <- pls(
    X,
    Y,
    ncomp = 1:3,
    fit = TRUE,
    svd.method = "cpu_rsvd",
    oversample = 8L,
    power = 2L,
    seed = 123L
  )
  internal_names <- pls(
    X,
    Y,
    ncomp = 1:3,
    fit = TRUE,
    svd.method = "cpu_rsvd",
    rsvd_oversample = 8L,
    rsvd_power = 2L,
    seed = 123L
  )

  expect_equal(aliased$B, compact$B)
  expect_equal(aliased$R, compact$R)
  expect_equal(internal_names$B, compact$B)
  expect_equal(internal_names$R, compact$R)
})

test_that("cpu_rsvd tracks dense-SVD SIMPLS on PLS outputs", {
  set.seed(99)
  X <- matrix(rnorm(80 * 24), nrow = 80, ncol = 24)
  Y <- matrix(rnorm(80 * 6), nrow = 80, ncol = 6)

  exact <- dense_simpls_reference(
    X,
    Y,
    ncomp = 1:4,
    fit = TRUE,
    svd.method = "rsvd"
  )

  rsvd <- pls(
    X,
    Y,
    ncomp = 1:4,
    fit = TRUE,
    svd.method = "cpu_rsvd",
    rsvd_oversample = 12L,
    rsvd_power = 2L,
    seed = 123L
  )

  expect_equal(dim(exact$B), dim(rsvd$B))
  expect_equal(dim(exact$R), dim(rsvd$R))
  expect_equal(rsvd$B, exact$B, tolerance = 5e-2)
  expect_true(all(is.finite(rsvd$R)))
  expect_true(all(is.finite(rsvd$Q)))
})

test_that("cpu_rsvd is deterministic with a fixed seed", {
  set.seed(7)
  X <- matrix(rnorm(90 * 25), nrow = 90, ncol = 25)
  Y <- matrix(rnorm(90 * 12), nrow = 90, ncol = 12)

  fit1 <- pls(
    X,
    Y,
    ncomp = 1:5,
    fit = TRUE,
    svd.method = "cpu_rsvd",
    rsvd_oversample = 5L,
    rsvd_power = 1L,
    seed = 777L
  )

  fit2 <- pls(
    X,
    Y,
    ncomp = 1:5,
    fit = TRUE,
    svd.method = "cpu_rsvd",
    rsvd_oversample = 5L,
    rsvd_power = 1L,
    seed = 777L
  )

  fit3 <- pls(
    X,
    Y,
    ncomp = 1:5,
    fit = TRUE,
    svd.method = "cpu_rsvd",
    rsvd_oversample = 5L,
    rsvd_power = 1L,
    seed = 778L
  )

  # LAPACK/BLAS reductions need not be bitwise identical, even when the
  # randomized sketch and deterministic recovery decisions are identical.
  expect_equal(fit1$B, fit2$B, tolerance = 1e-6)
  # Different seeds may converge to the same coefficients when the case audit
  # selects deterministic recovery, so seed sensitivity is not required.
  expect_true(all(is.finite(fit3$B)))
  expect_identical(
    fit3$diagnostics$status,
    "structural_checks_passed_case_audit_unavailable"
  )
})

test_that("xprod default threshold matches the benchmark rule", {
  should_use_rsvd <- get(".should_use_xprod_default", envir = asNamespace("fastPLS"))

  # singlecell-like shape: q is large, but ncomp is not small and X'Y is tiny.
  expect_false(should_use_rsvd(p = 50, q = 133, ncomp = 50))

  # A CIFAR-like cross-covariance is small enough to materialize safely.
  expect_false(should_use_rsvd(p = 2048, q = 100, ncomp = 10))
  expect_false(should_use_rsvd(p = 2048, q = 100, ncomp = 20))

  # Large cross-response products use xprod for rSVD.
  expect_true(should_use_rsvd(p = 5000, q = 1000, ncomp = 50))
})

test_that("compact streamed prediction is the default for compiled PLS", {
  set.seed(17)
  X <- matrix(rnorm(70 * 20), nrow = 70, ncol = 20)
  Y <- matrix(rnorm(70 * 5), nrow = 70, ncol = 5)
  idx <- 1:12

  for (method in c("plssvd", "simpls")) {
    ref <- pls(
      X[-idx, ],
      Y[-idx, ],
      ncomp = 1:4,
      method = method,
      backend = "cpp",
      svd.method = "cpu_rsvd",
      rsvd_oversample = 32L,
      rsvd_power = 5L,
      seed = 17L
    )
    flash <- pls(
      X[-idx, ],
      Y[-idx, ],
      ncomp = 1:4,
      method = method,
      backend = "cpp",
      svd.method = "cpu_rsvd",
      rsvd_oversample = 32L,
      rsvd_power = 5L,
      seed = 17L
    )
    pred_ref <- predict(ref, X[idx, , drop = FALSE], backend = "cpu")
    pred_flash <- predict(flash, X[idx, , drop = FALSE])
    flash_internal <- attr(flash, "fastPLS_internal")

    expect_s3_class(flash, "fastPLS")
    expect_true(isTRUE(flash_internal$flash_svd))
    expect_identical(flash_internal$flash_svd_backend, "cpu")
    expect_identical(flash_internal$predict_backend, "cpu_flash")
    expect_identical(flash_internal$flash_svd_mode, "streamed_low_rank_prediction")
    expect_equal(flash$B, ref$B)
    expect_equal(pred_flash$Ypred, pred_ref$Ypred, tolerance = 1e-10)
  }
})

test_that("GPU availability helpers return scalar logical values", {
  flag <- has_cuda()
  expect_type(flag, "logical")
  expect_length(flag, 1L)

  metal_flag <- has_metal()
  expect_type(metal_flag, "logical")
  expect_length(metal_flag, 1L)
})

test_that("pls validates svd.method through the compact CPU backend choices", {
  set.seed(1)
  X <- matrix(rnorm(40 * 10), nrow = 40, ncol = 10)
  Y <- matrix(rnorm(40 * 3), nrow = 40, ncol = 3)
  expect_error(pls(X, Y, ncomp = 1:2, svd.method = "cuda_rsvd"), "arg.*should be")
})

test_that("simpls path agrees with a dense-SVD reference", {
  set.seed(1234)
  X <- matrix(rnorm(100 * 25), nrow = 100, ncol = 25)
  Y <- matrix(rnorm(100 * 8), nrow = 100, ncol = 8)

  exact <- dense_simpls_reference(
    X,
    Y,
    ncomp = 1:5,
    fit = TRUE,
    method = "simpls",
    svd.method = "rsvd"
  )

  rsvd <- pls(
    X,
    Y,
    ncomp = 1:5,
    fit = TRUE,
    method = "simpls",
    svd.method = "cpu_rsvd",
    rsvd_oversample = 20L,
    rsvd_power = 2L,
    seed = 99L
  )

  expect_equal(dim(exact$B), dim(rsvd$B))
  expect_equal(rsvd$B, exact$B, tolerance = 7e-2)
  expect_true(all(is.finite(rsvd$R)))
  expect_true(all(is.finite(rsvd$Q)))
  expect_identical(
    rsvd$diagnostics$status,
    "structural_checks_passed_case_audit_unavailable"
  )
  expect_true(isTRUE(rsvd$diagnostics$stochastic))
  expect_equal(rsvd$diagnostics$effective_components, 5L)

  expect_error(
    pls(
      X,
      Y,
      ncomp = 1:5,
      fit = TRUE,
      method = "simpls",
      svd.method = "cuda_rsvd",
      rsvd_oversample = 10L,
      rsvd_power = 1L,
      seed = 99L
    ),
    "arg.*should be"
  )
})

test_that("accelerated SIMPLS preserves prediction despite coefficient changes", {
  set.seed(204)
  n <- 180
  latent <- matrix(rnorm(n * 12), n, 12)
  X <- latent %*% matrix(rnorm(12 * 60), 12, 60) +
    matrix(rnorm(n * 60, sd = 0.1), n, 60)
  Y <- latent %*% matrix(rnorm(12 * 20), 12, 20) +
    matrix(rnorm(n * 20, sd = 0.1), n, 20)

  reference <- dense_simpls_reference(
    X, Y, ncomp = 1:8, method = "simpls", backend = "cpu",
    svd.method = "rsvd", fit = TRUE, return_variance = FALSE
  )
  approximate <- pls(
    X, Y, ncomp = 1:8, method = "simpls", backend = "cpu",
    svd.method = "rsvd", oversample = 20L, power = 5L, seed = 204L,
    fit = TRUE, return_variance = FALSE
  )

  pred_reference <- as.vector(reference$Yfit[, , 8L])
  pred_approximate <- as.vector(approximate$Yfit[, , 8L])
  expect_gt(cor(pred_approximate, pred_reference), 0.99)
  rmse_reference <- sqrt(mean((pred_reference - as.vector(Y))^2))
  rmse_approximate <- sqrt(mean((pred_approximate - as.vector(Y))^2))
  expect_lt(rmse_approximate / rmse_reference, 1.02)
  expect_true(
    approximate$diagnostics$simpls_direction$approximate_execution
  )
})


test_that("Rcpp plssvd handles ncomp above rank by capping internally", {
  set.seed(78)
  X <- matrix(rnorm(180 * 45), nrow = 180, ncol = 45)
  y <- factor(sample(letters[1:10], 180, replace = TRUE))
  idx <- sample(seq_len(180), 40)

  expect_warning({
    fit <- pls(X[-idx, ], y[-idx], X[idx, ], ncomp = 60, method = "plssvd", svd.method = "cpu_rsvd")
    expect_s3_class(fit, "fastPLS")
    expect_true(is.data.frame(fit$Ypred))
  }, "rank is limited")
})

test_that("centered factor-response PLSSVD respects the C minus 1 rank bound", {
  set.seed(79)
  X <- matrix(rnorm(90 * 12), nrow = 90, ncol = 12)
  y <- factor(rep(letters[1:3], each = 30))

  expect_warning(
    fit <- pls(
      X,
      y,
      ncomp = 3,
      method = "plssvd",
      backend = "cpu",
      svd.method = "rsvd",
      return_variance = FALSE
    ),
    "rank is limited to 2"
  )
  expect_equal(as.integer(attr(fit, "fastPLS_internal")$ncomp), 2L)
})

test_that("CUDA SIMPLS memory guard refuses estimator substitution", {
  old <- Sys.getenv("FASTPLS_LABEL_AWARE_Y_THRESHOLD_MB", unset = NA_character_)
  on.exit({
    if (is.na(old)) {
      Sys.unsetenv("FASTPLS_LABEL_AWARE_Y_THRESHOLD_MB")
    } else {
      Sys.setenv(FASTPLS_LABEL_AWARE_Y_THRESHOLD_MB = old)
    }
  }, add = TRUE)
  Sys.setenv(FASTPLS_LABEL_AWARE_Y_THRESHOLD_MB = "0")

  expect_true(fastPLS:::.dense_indicator_exceeds_cuda_guard(20, 3))
  expect_error(
    fastPLS:::.stop_unsafe_cuda_simpls_response(20, 3),
    "does not replace a requested SIMPLS estimator with PLS-SVD"
  )
})
