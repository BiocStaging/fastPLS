test_that("SIMPLS phase timing is opt-in and hidden from public output", {
  set.seed(41)
  X <- matrix(rnorm(80 * 12), 80, 12)
  y <- factor(rep(letters[1:4], each = 20))

  old <- Sys.getenv("FASTPLS_BENCH_PHASE_TIMING", unset = NA_character_)
  on.exit({
    if (is.na(old)) {
      Sys.unsetenv("FASTPLS_BENCH_PHASE_TIMING")
    } else {
      Sys.setenv(FASTPLS_BENCH_PHASE_TIMING = old)
    }
  }, add = TRUE)

  Sys.unsetenv("FASTPLS_BENCH_PHASE_TIMING")
  ordinary <- pls(
    X, y, ncomp = 1:3, method = "simpls", backend = "cpu",
    svd.method = "rsvd", return_variance = FALSE
  )
  expect_false("benchmark_phase_timing" %in% names(ordinary))
  expect_null(attr(ordinary, "fastPLS_internal", exact = TRUE)$benchmark_phase_timing)

  Sys.setenv(FASTPLS_BENCH_PHASE_TIMING = "1")
  measured <- pls(
    X, y, ncomp = 1:3, method = "simpls", backend = "cpu",
    svd.method = "rsvd", return_variance = FALSE
  )
  expect_false("benchmark_phase_timing" %in% names(measured))
  timing <- attr(measured, "fastPLS_internal", exact = TRUE)$benchmark_phase_timing
  expect_named(timing, c(
    "preprocess_crosscov_sec", "response_crosscov_sec", "crossprod_cache_sec",
    "right_gram_sec", "estimator_sec", "direction_sec", "component_update_sec",
    "coefficient_path_sec",
    "fitted_values_sec", "model_assembly_sec", "cpp_total_sec"
  ))
  # The encompassing timer also includes the work between subphase timers.
  expect_gte(timing$preprocess_crosscov_sec + 1e-12,
             timing$response_crosscov_sec + timing$crossprod_cache_sec +
               timing$right_gram_sec)
  expect_equal(timing$estimator_sec,
               timing$direction_sec + timing$component_update_sec)
  expect_true(all(is.finite(unlist(timing))))
  expect_true(all(unlist(timing) >= 0))
  expect_gte(timing$cpp_total_sec, timing$model_assembly_sec)
})
