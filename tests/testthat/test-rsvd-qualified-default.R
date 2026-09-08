test_that("PLS and standalone SVD use their documented rSVD defaults", {
  defaults <- fastPLS:::.svd_control_defaults()
  expect_identical(defaults$rsvd_oversample, 32L)
  expect_identical(defaults$rsvd_power, 5L)
  expect_identical(formals(fastsvd)$oversample, 32L)
  expect_identical(formals(fastsvd)$power, 5L)

  resolved <- fastPLS:::.resolve_svd_control(context = "test")
  expect_identical(resolved$rsvd_oversample, 32L)
  expect_identical(resolved$rsvd_power, 5L)
})

test_that("backend controls distinguish panel evidence from general certification", {
  base <- fastPLS:::.resolve_svd_control(context = "test")

  cpu <- fastPLS:::.apply_backend_rsvd_controls(base, "cpu", "test")
  expect_identical(cpu$rsvd_oversample, 32L)
  expect_identical(cpu$rsvd_power, 5L)
  expect_true(cpu$rsvd_qualification$qualified_on_prespecified_panel)
  expect_true(cpu$rsvd_qualification$met_prespecified_panel)
  expect_false(cpu$rsvd_qualification$general_use_certified)

  cuda <- fastPLS:::.apply_backend_rsvd_controls(base, "cuda", "test")
  expect_identical(cuda$rsvd_oversample, 32L)
  expect_identical(cuda$rsvd_power, 5L)
  expect_true(cuda$rsvd_qualification$qualified_on_prespecified_panel)
  expect_false(cuda$rsvd_qualification$general_use_certified)

  explicit_weak <- fastPLS:::.resolve_svd_control(
    dots = list(oversample = 10L, power = 1L),
    context = "test"
  )
  expect_warning(
    cuda_weak <- fastPLS:::.apply_backend_rsvd_controls(
      explicit_weak, "cuda", "test"
    ),
    "raised CUDA rSVD controls"
  )
  expect_identical(cuda_weak$rsvd_oversample, 32L)
  expect_identical(cuda_weak$rsvd_power, 5L)

  metal <- fastPLS:::.apply_backend_rsvd_controls(base, "metal", "test")
  expect_identical(metal$rsvd_oversample, 32L)
  expect_identical(metal$rsvd_power, 5L)
  expect_true(metal$rsvd_qualification$qualified_on_prespecified_panel)
})

test_that("accelerated SIMPLS stabilizes ordinary shapes automatically", {
  set.seed(2201)
  X <- matrix(rnorm(80 * 12), 80, 12)
  Y <- matrix(rnorm(80 * 3), 80, 3)

  fit_default <- pls(X, Y, ncomp = 2, method = "simpls")
  expect_identical(fit_default$diagnostics$rsvd$oversample, 32L)
  expect_identical(fit_default$diagnostics$rsvd$power, 5L)
  expect_identical(
    fit_default$diagnostics$rsvd$control_profile,
    "ordinary_fast"
  )
  expect_true(fit_default$diagnostics$simpls_direction$approximate_execution)

  y <- factor(rep(c("a", "b"), each = 40L))
  fit_class <- pls(X, y, ncomp = 2, method = "simpls")
  expect_identical(fit_class$diagnostics$rsvd$oversample, 32L)
  expect_identical(fit_class$diagnostics$rsvd$power, 5L)

  expect_no_warning(
    fit_explicit <- pls(
      X, Y, ncomp = 2, method = "simpls",
      oversample = 6L, power = 1L
    )
  )
  expect_identical(fit_explicit$diagnostics$rsvd$oversample, 6L)
  expect_identical(fit_explicit$diagnostics$rsvd$power, 1L)
  expect_identical(fit_explicit$diagnostics$rsvd$control_profile, "explicit")
})

test_that("automatic SIMPLS controls strengthen numerically difficult shapes", {
  control <- fastPLS:::.resolve_svd_control(context = "test")
  control$svd.method <- "cpu_rsvd"

  for (family in c("simpls", "opls", "kernelpls")) {
    high_response <- fastPLS:::.apply_fast_simpls_shape_controls(
      control,
      family,
      matrix(0, 100, 20),
      matrix(0, 100, 64)
    )
    expect_identical(high_response$rsvd_oversample, 48L)
    expect_identical(high_response$rsvd_power, 6L)
    expect_identical(high_response$rsvd_profile, "high_response_stable")

    sparse_classes <- fastPLS:::.apply_fast_simpls_shape_controls(
      control,
      family,
      matrix(0, 500, 20),
      factor(rep(seq_len(50), each = 10))
    )
    expect_identical(sparse_classes$rsvd_oversample, 64L)
    expect_identical(sparse_classes$rsvd_power, 7L)
    expect_identical(
      sparse_classes$rsvd_profile,
      "sparse_high_class_stable"
    )
  }
})

test_that("massive SIMPLS shapes retain the fast default controls", {
  control <- fastPLS:::.resolve_svd_control(context = "test")
  control$svd.method <- "cpu_rsvd"
  X <- matrix(0, 2, 13000)
  Y <- matrix(0, 2, 28355)
  resolved <- fastPLS:::.apply_fast_simpls_shape_controls(
    control,
    "simpls",
    X,
    Y
  )
  expect_identical(resolved$rsvd_oversample, 12L)
  expect_identical(resolved$rsvd_power, 1L)
  expect_identical(resolved$rsvd_profile, "massive_crosscovariance")
})

test_that("qualification metadata describes the executed shape profile", {
  X <- matrix(0, 2, 13000)
  Y <- matrix(0, 2, 28355)
  context <- fastPLS:::.pls_context(
    X, Y, NULL, NULL, "simpls", "rsvd", list(),
    "cpu", "argmax", "centering"
  )

  expect_identical(context$control$rsvd_oversample, 12L)
  expect_identical(context$control$rsvd_power, 1L)
  expect_identical(context$control$rsvd_profile, "massive_crosscovariance")
  expect_identical(
    context$control$rsvd_qualification$oversample,
    context$control$rsvd_oversample
  )
  expect_identical(
    context$control$rsvd_qualification$power,
    context$control$rsvd_power
  )
})

test_that("reported rSVD screening criteria match the current qualification", {
  set.seed(812)
  X <- matrix(rnorm(80 * 14), 80, 14)
  Y <- matrix(rnorm(80 * 3), 80, 3)
  fit <- pls(
    X, Y, ncomp = 3, method = "simpls", backend = "cpu",
    svd.method = "rsvd", seed = 19, return_variance = FALSE
  )

  criteria <- fit$diagnostics$rsvd$validation_failure_criteria
  expect_identical(criteria$prediction_relative_error_above, 0.01)
  expect_identical(criteria$score_relative_error_above, 0.01)
  expect_identical(criteria$prediction_correlation_below, 0.995)
  expect_identical(criteria$score_correlation_below, 0.995)
  expect_identical(criteria$latent_subspace_angle_degrees_above, 0.1)
  expect_identical(criteria$classification_label_agreement_below, 0.995)
  expect_identical(criteria$predictive_metric_absolute_difference_above, 0.005)
})
