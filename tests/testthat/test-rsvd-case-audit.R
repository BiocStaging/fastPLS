test_that("CPU rSVD certifies difficult spectra across seeds", {
  singular_values <- exp(seq(log(20), log(0.05), length.out = 60L))
  set.seed(991)
  U <- qr.Q(qr(matrix(rnorm(140L * 60L), 140L, 60L)))
  V <- qr.Q(qr(matrix(rnorm(90L * 60L), 90L, 60L)))
  A <- U %*% (singular_values * t(V))

  for (seed in c(1L, 7L, 19L, 43L, 123L)) {
    out <- fastsvd(A, ncomp = 30L, seed = seed)
    expect_true(out$diagnostics$rsvd_case_audit$performed)
    expect_true(out$diagnostics$rsvd_case_audit$certified)
    expect_lte(out$diagnostics$rsvd_case_audit$triplet_residual, 1e-2)
    expect_lte(out$diagnostics$rsvd_case_audit$omitted_direction_ratio, 1.01)
  }
})

test_that("rank-deficient CPU rSVD remains finite and audited", {
  set.seed(992)
  L <- matrix(rnorm(180L * 12L), 180L, 12L)
  R <- matrix(rnorm(12L * 75L), 12L, 75L)
  out <- fastsvd(L %*% R, ncomp = 10L, seed = 11L)

  expect_true(all(is.finite(out$d)))
  expect_true(out$diagnostics$rsvd_case_audit$certified)
})

test_that("near-tied retained boundaries do not force deterministic recovery", {
  set.seed(995)
  U <- qr.Q(qr(matrix(rnorm(120L * 20L), 120L, 20L)))
  V <- qr.Q(qr(matrix(rnorm(80L * 20L), 80L, 20L)))
  singular_values <- c(rep(10, 6L), seq(9, 1, length.out = 14L))
  A <- U %*% (singular_values * t(V))

  out <- fastsvd(A, ncomp = 5L, seed = 23L)

  expect_true(out$diagnostics$rsvd_case_audit$certified)
  expect_false(out$diagnostics$rsvd_case_audit$deterministic_fallback)
  expect_lte(out$diagnostics$rsvd_case_audit$triplet_residual, 1e-2)
  expect_lte(out$diagnostics$rsvd_case_audit$omitted_direction_ratio, 1.01)
})

test_that("matrix-free accelerated SIMPLS reports structural diagnostics", {
  old_mode <- Sys.getenv("FASTPLS_ABLATION_MODE", unset = NA_character_)
  old_xprod <- Sys.getenv("FASTPLS_ABLATION_XPROD", unset = NA_character_)
  on.exit({
    if (is.na(old_mode)) Sys.unsetenv("FASTPLS_ABLATION_MODE") else Sys.setenv(FASTPLS_ABLATION_MODE = old_mode)
    if (is.na(old_xprod)) Sys.unsetenv("FASTPLS_ABLATION_XPROD") else Sys.setenv(FASTPLS_ABLATION_XPROD = old_xprod)
  }, add = TRUE)
  Sys.setenv(FASTPLS_ABLATION_MODE = "1", FASTPLS_ABLATION_XPROD = "1")

  set.seed(993)
  X <- matrix(rnorm(220L * 90L), 220L, 90L)
  Y <- matrix(rnorm(220L * 45L), 220L, 45L)
  fit <- pls(X, Y, ncomp = 1:4, method = "simpls", seed = 17L)

  expect_identical(
    fit$diagnostics$status,
    "structural_checks_passed_case_audit_unavailable"
  )
  expect_identical(fit$diagnostics$rsvd$case_audit$solves, 0L)
  expect_true(fit$diagnostics$simpls_direction$approximate_execution)
})

test_that("float32 CPU rSVD performs a case-specific audit", {
  skip_if_not_installed("float")
  skip_on_os("windows")
  set.seed(994)
  A <- float::fl(matrix(rnorm(100L * 45L), 100L, 45L))
  out <- fastsvd(A, ncomp = 8L, backend = "cpu")

  expect_true(out$diagnostics$rsvd_case_audit$performed)
  expect_true(out$diagnostics$rsvd_case_audit$certified)
  expect_true(all(is.finite(out$d)))
})
