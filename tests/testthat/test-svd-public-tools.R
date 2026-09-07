test_that("fastsvd returns decomposition outputs from public backends", {
  set.seed(1)
  A <- matrix(rnorm(120 * 20), 120, 20)
  out <- suppressWarnings(fastsvd(A, ncomp = 5, backend = "cpu"))
  expect_true(is.list(out))
  expect_true(all(c(
    "u", "d", "v", "backend", "method", "svd.method", "elapsed",
    "diagnostics"
  ) %in% names(out)))
  expect_identical(out$backend, "cpu")
  expect_identical(out$method, "rsvd")
  expect_identical(out$svd.method, "cpu_rsvd")
  expect_equal(ncol(out$u), 5)
  expect_equal(length(out$d), 5)
  expect_equal(ncol(out$v), 5)
  expect_true(out$diagnostics$status %in% c(
    "rsvd_case_audit_passed",
    "rsvd_case_audit_passed_with_deterministic_recovery"
  ))
  expect_true(out$diagnostics$rsvd_case_audit$performed)
  expect_true(out$diagnostics$rsvd_case_audit$certified)
  expect_true(is.finite(out$diagnostics$max_relative_triplet_residual))
  if (identical(out$diagnostics$status, "failed_large_triplet_residual")) {
    expect_gt(
      out$diagnostics$max_relative_triplet_residual,
      out$diagnostics$residual_failure_threshold
    )
  }
})

test_that("unsupported fastsvd method labels fail explicitly", {
  set.seed(11)
  A <- matrix(rnorm(80 * 12), 80, 12)
  expect_error(fastsvd(A, ncomp = 4, backend = "cpu", method = "unsupported"), "rsvd.*only")
  expect_error(fastsvd(A, ncomp = 4, backend = "cpu", method = "full"), "rsvd.*only")
  expect_error(fastsvd(A, work = 10L), "unused argument")
})

test_that("fastsvd maps backend and method to the intended internal SVD", {
  set.seed(12)
  A <- matrix(rnorm(70 * 10), 70, 10)

  expect_error(fastsvd(A, ncomp = 4, backend = "cpu", method = "irlba"), "rsvd.*only")

  cpu_rsvd <- fastsvd(A, ncomp = 4, backend = "cpu", method = "rsvd")
  expect_identical(cpu_rsvd$svd.method, "cpu_rsvd")

  expect_error(
    fastsvd(A, ncomp = 4, backend = "cuda", method = "irlba"),
    "rsvd.*only|No CPU fallback"
  )
  expect_error(
    fastsvd(A, ncomp = 4, backend = "metal", method = "irlba"),
    "rsvd.*only|No CPU fallback"
  )
})

test_that("small rSVD inputs use the dense full-subspace calculation", {
  set.seed(42)
  A <- matrix(rnorm(40 * 5), 40, 5)
  ref <- base::svd(A, nu = 3, nv = 3)

  for (method in "rsvd") {
    out <- suppressWarnings(fastsvd(
      A,
      ncomp = 3,
      backend = "cpu",
      method = method,
      oversample = 0L,
      power = 0L,
      seed = 99L
    ))
    expect_equal(out$d, ref$d[1:3], tolerance = 1e-8)
    expect_equal(abs(out$u), abs(ref$u[, 1:3, drop = FALSE]), tolerance = 1e-6)
    expect_equal(abs(out$v), abs(ref$v[, 1:3, drop = FALSE]), tolerance = 1e-6)
  }
})

test_that("fastPLS does not mask base svd", {
  expect_identical(svd, base::svd)
})
