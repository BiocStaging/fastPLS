test_that("float32 capability assessment is shape based", {
  safe <- fastPLS:::.float32_capability_assessment(
    method = "plssvd",
    backend = "cpu",
    svd_method = "cpu_rsvd",
    q = 22L,
    ncomp = 20L,
    classification = TRUE,
    os_type = "unix"
  )
  expect_identical(safe$status, "validated")
  expect_identical(safe$action, "allow")
  expect_length(safe$warnings, 0L)

  risky <- fastPLS:::.float32_capability_assessment(
    method = "simpls",
    backend = "cuda",
    svd_method = "cpu_rsvd",
    q = 28355L,
    ncomp = 100L,
    classification = FALSE,
    os_type = "unix"
  )
  expect_identical(risky$status, "experimental")
  expect_identical(risky$action, "warn")
  expect_match(risky$warnings, "numerical-risk")

  plssvd <- fastPLS:::.float32_capability_assessment(
    method = "plssvd",
    backend = "cuda",
    svd_method = "cpu_rsvd",
    q = 28355L,
    ncomp = 100L,
    classification = FALSE,
    os_type = "unix"
  )
  expect_identical(plssvd$status, "experimental")
  expect_match(plssvd$warnings, "performance-risk")
})

test_that("validated float32 classification is quiet and nonlinear kernels are flagged", {
  classification <- fastPLS:::.float32_capability_assessment(
    method = "simpls",
    backend = "cpu",
    svd_method = "cpu_rsvd",
    q = 100L,
    ncomp = 50L,
    classification = TRUE,
    os_type = "unix"
  )
  expect_identical(classification$status, "validated")
  expect_identical(classification$action, "allow")
  expect_length(classification$warnings, 0L)

  nonlinear <- fastPLS:::.float32_capability_assessment(
    method = "kernelpls",
    backend = "metal",
    svd_method = "cpu_rsvd",
    q = 1L,
    ncomp = 5L,
    classification = FALSE,
    kernel = "rbf",
    os_type = "unix"
  )
  expect_identical(nonlinear$status, "experimental")
  expect_identical(nonlinear$action, "warn")
  expect_length(nonlinear$errors, 0L)
  expect_true(any(grepl("n-by-n Gram matrix", nonlinear$warnings)))
})

test_that("float32 unavailable and hybrid routes are explicit", {
  cuda_irlba <- fastPLS:::.float32_capability_assessment(
    method = "plssvd",
    backend = "cuda",
    svd_method = "irlba",
    q = 3L,
    ncomp = 2L,
    classification = TRUE,
    os_type = "unix"
  )
  expect_identical(cuda_irlba$status, "unavailable")
  expect_identical(cuda_irlba$action, "error")
  expect_match(cuda_irlba$errors, "supports rSVD only")

  metal_lda <- fastPLS:::.float32_capability_assessment(
    method = "simpls",
    backend = "metal",
    svd_method = "cpu_rsvd",
    q = 3L,
    ncomp = 2L,
    classification = TRUE,
    classifier = "lda",
    os_type = "unix"
  )
  expect_identical(metal_lda$status, "validated")
  expect_identical(metal_lda$execution, "device_accelerated")
  expect_false(any(grepl("LDA is hybrid", metal_lda$warnings)))

  for (cfg in list(
    list(method = "opls", kernel = "linear", classifier = "argmax"),
    list(method = "kernelpls", kernel = "rbf", classifier = "argmax"),
    list(method = "simpls", kernel = "linear", classifier = "lda")
  )) {
    windows_route <- fastPLS:::.float32_capability_assessment(
      method = cfg$method,
      backend = "cpu",
      svd_method = "cpu_rsvd",
      q = if (identical(cfg$classifier, "lda")) 3L else 1L,
      ncomp = 2L,
      classification = identical(cfg$classifier, "lda"),
      kernel = cfg$kernel,
      classifier = cfg$classifier,
      os_type = "windows"
    )
    expect_identical(windows_route$status, "experimental")
    expect_identical(windows_route$action, "warn")
    expect_identical(windows_route$execution, "portable_cpu")
    expect_true(any(grepl("OPLS, nonlinear kernel PLS", windows_route$warnings)))
  }
})

test_that("float32 risk warnings are emitted once per route", {
  state <- fastPLS:::.float32_warning_state
  rm(list = ls(envir = state, all.names = TRUE), envir = state)
  on.exit(rm(list = ls(envir = state, all.names = TRUE), envir = state))

  Y <- matrix(0, nrow = 1L, ncol = 10000L)
  expect_warning(
    fastPLS:::.warn_float32_capability(
      method = "simpls",
      backend = "cuda",
      svd_method = "cpu_rsvd",
      Ytrain = Y,
      ncomp = 50L,
      os_type = "unix"
    ),
    "numerical-risk"
  )
  expect_silent(
    fastPLS:::.warn_float32_capability(
      method = "simpls",
      backend = "cuda",
      svd_method = "cpu_rsvd",
      Ytrain = Y,
      ncomp = 50L,
      os_type = "unix"
    )
  )
})
