test_that("SIMPLS reports the release direction-refresh rule", {
  set.seed(41)
  X <- matrix(rnorm(72 * 14), 72, 14)
  Y <- cbind(
    0.7 * X[, 1] - 0.2 * X[, 3] + rnorm(72, sd = 0.05),
    -0.4 * X[, 2] + 0.3 * X[, 5] + rnorm(72, sd = 0.05)
  )

  fit <- pls(
    X,
    Y,
    ncomp = 1:3,
    method = "simpls",
    backend = "cpu",
    svd.method = "rsvd",
    return_variance = FALSE,
    seed = 17
  )

  rule <- fit$diagnostics$simpls_direction
  expect_identical(rule$rule, "fresh_oversampled_sketch_per_component")
  expect_identical(rule$directions_per_solve, 1L)
  expect_false(rule$candidate_block_refresh)
  expect_identical(rule$seed_rule, "seed_plus_component_index")
  expect_true("cached_rank_one_deflation_product" %in% rule$active_optimizations)
  expect_true(rule$approximate_execution)
})


test_that("rejected refresh environment variables no longer affect SIMPLS", {
  set.seed(42)
  X <- matrix(rnorm(64 * 12), 64, 12)
  Y <- cbind(X[, 1] + rnorm(64, sd = 0.1), X[, 2] + rnorm(64, sd = 0.1))

  fit_reference <- pls(
    X, Y, ncomp = 1:3, method = "simpls", backend = "cpu",
    svd.method = "rsvd", return_variance = FALSE, seed = 19
  )

  old <- Sys.getenv(
    c("FASTPLS_FAST_INCREMENTAL", "FASTPLS_FAST_ADAPTIVE_RSVD"),
    unset = NA_character_
  )
  on.exit({
    for (name in names(old)) {
      if (is.na(old[[name]])) Sys.unsetenv(name) else do.call(Sys.setenv, setNames(list(old[[name]]), name))
    }
  }, add = TRUE)
  Sys.setenv(FASTPLS_FAST_INCREMENTAL = "1", FASTPLS_FAST_ADAPTIVE_RSVD = "1")

  fit_obsolete_env <- pls(
    X, Y, ncomp = 1:3, method = "simpls", backend = "cpu",
    svd.method = "rsvd", return_variance = FALSE, seed = 19
  )

  expect_equal(fit_obsolete_env$R, fit_reference$R, tolerance = 0)
  expect_equal(fit_obsolete_env$Q, fit_reference$Q, tolerance = 0)
  expect_identical(
    fit_obsolete_env$diagnostics$simpls_direction$rule,
    "fresh_oversampled_sketch_per_component"
  )
})

test_that("available accelerator dispatches expose their SIMPLS rule", {
  set.seed(43)
  X <- matrix(rnorm(48 * 10), 48, 10)
  Y <- cbind(X[, 1] + rnorm(48, sd = 0.1), X[, 2] + rnorm(48, sd = 0.1))

  available <- c(cpu = TRUE, cuda = has_cuda(), metal = has_metal())
  for (backend in names(available)[available]) {
    X_backend <- if (backend == "metal") float::fl(X) else X
    Y_backend <- if (backend == "metal") float::fl(Y) else Y
    fit <- suppressWarnings(pls(
      X_backend, Y_backend, ncomp = 1:2, method = "simpls", backend = backend,
      svd.method = "rsvd", return_variance = FALSE, seed = 23
    ))
    expect_equal(ncol(fit$R), 2L, info = backend)
    expect_equal(ncol(fit$Q), 2L, info = backend)
    expect_true(all(is.finite(as.matrix(fit$R))), info = backend)
    expect_true(all(is.finite(as.matrix(fit$Q))), info = backend)
    rule <- fit$diagnostics$simpls_direction
    expect_identical(rule$directions_per_solve, 1L, info = backend)
    expect_false(rule$candidate_block_refresh, info = backend)
    expect_true(rule$fresh_start, info = backend)
    resident_gpu <- backend %in% c("cuda", "metal") &&
      !is.null(attr(fit, "fastPLS_internal")$resident_state)
    expect_identical(is.na(rule$refresh_width), !resident_gpu, info = backend)
    expect_identical(is.na(rule$refresh_iterations), !resident_gpu,
      info = backend)
    expect_identical(rule$seed_rule, "seed_plus_component_index", info = backend)
    expected_rule <- "fresh_oversampled_sketch_per_component"
    expect_identical(rule$rule, expected_rule)
  }
})

test_that("Metal source does not reuse a preceding component direction", {
  source_path <- testthat::test_path("..", "..", "src", "svd_metal_backend.mm.in")
  skip_if_not(file.exists(source_path), "source tree unavailable after installation")
  source <- paste(readLines(source_path, warn = FALSE), collapse = "\n")
  expect_false(grepl("has_rr_prev|rr_prev", source))
  expect_match(source, "A fresh direction avoids propagating approximation")
})

test_that("classification diagnostics report bounded batches on every backend", {
  for (backend in c("cpu", "cuda", "metal")) {
    rule <- fastPLS:::.simpls_direction_diagnostics(
      randomized = TRUE,
      backend = backend,
      classification = TRUE,
      training_samples = 50000L,
      response_dimension = 100L,
      predictor_dimension = 768L,
      requested_components = 100L
    )
    expect_identical(
      rule$rule,
      paste0("batched_", backend, "_candidate_block"),
      info = backend
    )
    expect_identical(rule$directions_per_solve, 64L, info = backend)
    expect_true(rule$candidate_block_refresh, info = backend)
  }
})

test_that("massive cross-covariance routing accounts for input precision", {
  X <- matrix(0, 2L, 10000L)
  Y <- matrix(0, 2L, 8000L)
  double <- fastPLS:::.fast_simpls_shape_profile(X, Y, float32 = FALSE)
  single <- fastPLS:::.fast_simpls_shape_profile(X, Y, float32 = TRUE)

  expect_identical(double$profile, "massive_crosscovariance")
  expect_identical(single$profile, "high_response_stable")
})

test_that("massive implicit SIMPLS is reproducible and backend concordant", {
  skip_on_os("windows")
  set.seed(918)
  ntrain <- 48L
  ntest <- 12L
  predictor_count <- 9000L
  response_count <- 15000L
  latent_count <- 4L
  latent <- matrix(
    rnorm((ntrain + ntest) * latent_count),
    ntrain + ntest, latent_count
  )
  predictors <- latent %*% matrix(
    rnorm(latent_count * predictor_count, sd = 0.25),
    latent_count, predictor_count
  )
  responses <- latent %*% matrix(
    rnorm(latent_count * response_count, sd = 0.08),
    latent_count, response_count
  )
  Xtrain <- float::fl(predictors[seq_len(ntrain), , drop = FALSE])
  Ytrain <- float::fl(responses[seq_len(ntrain), , drop = FALSE])
  Xtest <- float::fl(
    predictors[ntrain + seq_len(ntest), , drop = FALSE]
  )

  fit_once <- function(backend) {
    pls(
      Xtrain, Ytrain,
      ncomp = 1:6,
      method = "simpls",
      backend = backend,
      scaling = "none",
      fit = FALSE,
      proj = FALSE,
      return_variance = FALSE,
      return_loadings = FALSE,
      oversample = 12L,
      power = 1L,
      seed = 7L
    )
  }
  first <- fit_once("cpu")
  second <- fit_once("cpu")
  first_prediction <- predict(first, Xtest)$Ypred[[6L]]
  second_prediction <- predict(second, Xtest)$Ypred[[6L]]

  expect_identical(
    first$diagnostics$simpls_direction$rule,
    "fresh_cpu_rank_one_refresh"
  )
  expect_equal(
    float::dbl(first_prediction),
    float::dbl(second_prediction),
    tolerance = 0
  )

  if (isTRUE(has_metal())) {
    metal <- fit_once("metal")
    metal_prediction <- predict(metal, Xtest, backend = "metal")$Ypred[[6L]]
    expect_identical(
      metal$diagnostics$simpls_direction$rule,
      "fresh_metal_rank_one_refresh"
    )
    expect_equal(
      float::dbl(metal_prediction),
      float::dbl(first_prediction),
      tolerance = 1e-6
    )
  }
})

test_that("public diagnostics name componentwise and block randomized SIMPLS", {
  set.seed(831)
  ordinary <- pls(
    matrix(rnorm(80 * 12), 80, 12),
    matrix(rnorm(80 * 2), 80, 2),
    ncomp = 3,
    method = "simpls",
    return_variance = FALSE,
    seed = 17
  )
  expect_identical(
    ordinary$diagnostics$algorithm_variant,
    "componentwise_randomized_simpls"
  )

  direction <- fastPLS:::.simpls_direction_diagnostics(
    randomized = TRUE,
    backend = "cpu",
    classification = TRUE,
    training_samples = 50000L,
    predictor_dimension = 768L,
    response_dimension = 100L,
    requested_components = 100L
  )
  candidate <- list(diagnostics = list(simpls_direction = direction))
  expect_identical(
    fastPLS:::.fastpls_algorithm_variant(
      candidate,
      list(method = "simpls"),
      list(kernel = "linear")
    ),
    "block_randomized_simpls"
  )
})

test_that("CUDA moderate classification retains per-component refresh", {
  rule <- fastPLS:::.simpls_direction_diagnostics(
    randomized = TRUE,
    backend = "cuda",
    classification = TRUE,
    training_samples = 5000L,
    predictor_dimension = 128L,
    response_dimension = 50L,
    requested_components = 50L
  )
  expect_identical(rule$rule, "fresh_oversampled_sketch_per_component")
  expect_false(rule$candidate_block_refresh)
})

test_that("batch diagnostics respect shape and the shared block limit", {
  args <- list(
    randomized = TRUE, backend = "cpu", classification = TRUE,
    training_samples = 2000L, predictor_dimension = 3000L,
    response_dimension = 100L, requested_components = 100L
  )
  cpu <- do.call(fastPLS:::.simpls_direction_diagnostics, args)
  expect_true(cpu$candidate_block_refresh)
  args$backend <- "cuda"
  cuda_small <- do.call(fastPLS:::.simpls_direction_diagnostics, args)
  expect_true(cuda_small$candidate_block_refresh)
  expect_identical(cuda_small$directions_per_solve, 64L)
  args$training_samples <- 5000L
  args$oversample <- 0L
  cuda_narrow <- do.call(fastPLS:::.simpls_direction_diagnostics, args)
  expect_identical(cuda_narrow$directions_per_solve, 64L)
  expect_true(cuda_narrow$candidate_block_refresh)
  args$oversample <- 3L
  cuda_four <- do.call(fastPLS:::.simpls_direction_diagnostics, args)
  expect_identical(cuda_four$directions_per_solve, 64L)
  args$requested_components <- 20L
  cuda_prefix <- do.call(fastPLS:::.simpls_direction_diagnostics, args)
  expect_true(cuda_prefix$candidate_block_refresh)
  expect_identical(cuda_prefix$directions_per_solve, 20L)
})

test_that("CUDA massive cross-covariance diagnostics report block refresh", {
  rule <- fastPLS:::.simpls_direction_diagnostics(
    randomized = TRUE,
    backend = "cuda",
    classification = FALSE,
    training_samples = 1200L,
    predictor_dimension = 13000L,
    response_dimension = 28355L,
    requested_components = 165L,
    power = 2L
  )
  expect_identical(rule$rule, "batched_cuda_candidate_block")
  expect_identical(rule$refresh_width, 8L)
  expect_identical(rule$directions_per_solve, 8L)
  expect_identical(rule$refresh_iterations, 2L)
  expect_true(rule$fresh_start)
})

test_that("CPU massive cross-covariance diagnostics report rank-one refresh", {
  rule <- fastPLS:::.simpls_direction_diagnostics(
    randomized = TRUE,
    backend = "cpu",
    classification = FALSE,
    training_samples = 1200L,
    predictor_dimension = 13000L,
    response_dimension = 28355L,
    requested_components = 50L,
    power = 2L
  )
  expect_identical(rule$rule, "fresh_cpu_rank_one_refresh")
  expect_identical(rule$refresh_width, 1L)
  expect_identical(rule$directions_per_solve, 1L)
  expect_identical(rule$refresh_iterations, 2L)
  expect_identical(rule$seed_rule, "seed_plus_component_index")
  expect_true(rule$fresh_start)
})

test_that("Metal massive SIMPLS diagnostics report rank-one refresh", {
  rule <- fastPLS:::.simpls_direction_diagnostics(
    randomized = TRUE,
    backend = "metal",
    classification = FALSE,
    training_samples = 1200L,
    predictor_dimension = 13000L,
    response_dimension = 28355L,
    requested_components = 50L,
    route_mode = "metal_resident_rank_one_simpls",
    power = 1L
  )
  expect_identical(rule$rule, "fresh_metal_rank_one_refresh")
  expect_identical(rule$refresh_width, 1L)
  expect_identical(rule$directions_per_solve, 1L)
  expect_identical(rule$refresh_iterations, 1L)
  expect_identical(
    rule$seed_rule,
    "single_seed_fresh_component_sequence"
  )
  expect_true(rule$fresh_start)
})

test_that("CUDA rSVD repeats exactly when the seed is fixed", {
  skip_if_not(isTRUE(has_cuda()), "CUDA backend is unavailable")
  set.seed(20260902)
  X <- matrix(rnorm(180 * 24), 180, 24)
  Y <- matrix(rnorm(180 * 5), 180, 5)
  fit_once <- function() {
    pls(
      X, Y,
      ncomp = 4L,
      method = "simpls",
      backend = "cuda",
      svd.method = "rsvd",
      seed = 77L,
      fit = TRUE,
      return_variance = FALSE
    )
  }
  first <- fit_once()
  second <- fit_once()
  expect_equal(first$Yfit, second$Yfit, tolerance = 0)
})

test_that("massive CUDA diagnostics report the executed candidate block", {
  direction <- fastPLS:::.simpls_direction_diagnostics(
    randomized = TRUE,
    backend = "cuda",
    classification = FALSE,
    training_samples = 1200L,
    predictor_dimension = 13000L,
    response_dimension = 28355L,
    requested_components = 50L,
    route_mode = "cuda_resident_candidate_block_simpls",
    power = 6L
  )
  expect_identical(direction$rule, "batched_cuda_candidate_block")
  expect_identical(direction$directions_per_solve, 8L)
  expect_true(direction$fresh_start)
  expect_identical(direction$refresh_iterations, 6L)
})

test_that("SIMPLS-family backends attach fresh-start diagnostics", {
  set.seed(44)
  X <- matrix(rnorm(54 * 9), 54, 9)
  Y <- cbind(X[, 1] + rnorm(54, sd = 0.1), X[, 2] + rnorm(54, sd = 0.1))

  available <- c(cpu = TRUE, cuda = has_cuda(), metal = has_metal())
  for (backend in names(available)[available]) {
    methods <- if (backend == "cpu") {
      c("simpls", "opls", "kernelpls")
    } else {
      c("simpls", "kernelpls")
    }
    for (method in methods) {
      X_backend <- if (backend == "metal") float::fl(X) else X
      Y_backend <- if (backend == "metal") float::fl(Y) else Y
      kernel <- if (backend == "cpu") "rbf" else "linear"
      fit <- suppressWarnings(pls(
        X_backend, Y_backend,
        ncomp = 1:2,
        method = method,
        kernel = kernel,
        backend = backend,
        svd.method = "rsvd",
        return_variance = FALSE,
        seed = 29
      ))
      direction <- fit$diagnostics$simpls_direction
      resident_metal <- backend == "metal" &&
        !is.null(attr(fit, "fastPLS_internal")$resident_state)
      resident_cuda <- backend == "cuda" &&
        !is.null(attr(fit, "fastPLS_internal")$resident_state)
      expected_optimizations <- if (resident_metal || resident_cuda) {
        resident_precision <- if (backend == "metal") "float32" else "float64"
        c("cached_rank_one_deflation_product", "persistent_device_workspace",
          "compact_prediction",
          paste0(backend, "_resident_", resident_precision, "_buffers"))
      } else if (backend == "metal") {
        c("cached_rank_one_deflation_product", "compact_prediction",
          "float32_buffers")
      } else {
        c("cached_rank_one_deflation_product",
          "incremental_coefficient_path",
          "conditional_crossproduct_cache", "compact_prediction")
      }
      expect_identical(
        direction$rule,
        if (resident_metal) {
          "fresh_randomized_direction_per_component"
        } else {
          "fresh_oversampled_sketch_per_component"
        },
        info = paste(method, backend)
      )
      expect_true(direction$fresh_start, info = paste(method, backend))
      expect_identical(
        direction$directions_per_solve,
        1L,
        info = paste(method, backend)
      )
      expect_true(
        all(expected_optimizations %in% direction$active_optimizations),
        info = paste(method, backend)
      )
      expect_identical(
        fit$diagnostics$rsvd$backend,
        backend,
        info = paste(method, backend)
      )
    }
  }
})
