test_that("fastPLS backend precedence is explicit, option, environment, CPU", {
  old_option <- getOption("backend", NULL)
  old_env <- Sys.getenv("FASTPLS_BACKEND", unset = NA_character_)
  on.exit({
    options(backend = old_option)
    if (is.na(old_env)) Sys.unsetenv("FASTPLS_BACKEND") else Sys.setenv(FASTPLS_BACKEND = old_env)
  }, add = TRUE)
  options(backend = NULL); Sys.unsetenv("FASTPLS_BACKEND")
  expect_identical(fastPLS_backend(), "cpu")
  Sys.setenv(FASTPLS_BACKEND = "metal")
  expect_identical(fastPLS:::.fastpls_resolve_backend(NULL), "metal")
  options(backend = "cuda")
  expect_identical(fastPLS:::.fastpls_resolve_backend(NULL), "cuda")
  expect_identical(fastPLS:::.fastpls_resolve_backend("cpu"), "cpu")
  expect_error(fastPLS:::.fastpls_resolve_backend("auto"), "must be one of")
})

test_that("generic backend option controls fastPLS and explicit values win", {
  old_option <- getOption("backend", NULL)
  old_env <- Sys.getenv("FASTPLS_BACKEND", unset = NA_character_)
  on.exit({
    options(backend = old_option)
    if (is.na(old_env)) Sys.unsetenv("FASTPLS_BACKEND") else Sys.setenv(FASTPLS_BACKEND = old_env)
  }, add = TRUE)
  Sys.setenv(FASTPLS_BACKEND = "metal")
  options(backend = "cuda")
  expect_identical(fastPLS:::.fastpls_resolve_backend(NULL), "cuda")
  expect_identical(fastPLS:::.fastpls_resolve_backend("cpu"), "cpu")
})

test_that("CPU core option is validated and applied to thread runtimes", {
  old_cores <- getOption("cores", NULL)
  variables <- c("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS",
                 "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
  old_environment <- Sys.getenv(variables, unset = NA_character_)
  on.exit({
    options(cores = old_cores)
    for (variable in variables) {
      value <- old_environment[[variable]]
      if (is.na(value)) Sys.unsetenv(variable) else do.call(Sys.setenv,
        stats::setNames(list(value), variable))
    }
  }, add = TRUE)
  options(cores = 3L)
  expect_identical(fastPLS:::.fastpls_apply_cpu_cores(), 3L)
  expect_true(all(Sys.getenv(variables) == "3"))
  expect_type(fastPLS:::set_cpu_threads_cpp(2L), "character")
  expect_error(
    fastPLS:::set_cpu_threads_cpp(0L),
    "positive integer"
  )
  options(cores = 1.5)
  expect_error(fastPLS:::.fastpls_apply_cpu_cores(), "positive integer")
})

test_that("public fitting functions defer omitted backends", {
  expect_null(formals(fastsvd)$backend)
  expect_null(formals(pls)$backend)
  expect_null(formals(pls.single.cv)$backend)
  expect_null(formals(pls.double.cv)$backend)
  expect_null(formals(getS3method("predict", "fastPLS"))$backend)
})

test_that("backend availability guard never substitutes CPU", {
  expect_identical(
    fastPLS:::.fastpls_require_backend_available("cpu", available = TRUE),
    "cpu"
  )
  expect_error(
    fastPLS:::.fastpls_require_backend_available(
      "cuda",
      "Test operation",
      available = FALSE
    ),
    "backend='cuda'.*No CPU fallback"
  )
  expect_error(
    fastPLS:::.fastpls_require_backend_available(
      "metal",
      "Test operation",
      available = FALSE
    ),
    "backend='metal'.*No CPU fallback"
  )
})

test_that("the backend setter rejects unavailable accelerators immediately", {
  unavailable <- if (!isTRUE(has_cuda())) {
    "cuda"
  } else if (!isTRUE(has_metal())) {
    "metal"
  } else {
    skip("Both optional accelerator backends are available")
  }
  old_option <- getOption("backend", NULL)
  on.exit(options(backend = old_option), add = TRUE)

  expect_error(
    fastPLS_backend(unavailable),
    "No CPU fallback"
  )
  expect_identical(getOption("backend", NULL), old_option)
})

test_that("the backend getter rejects an unavailable configured accelerator", {
  unavailable <- if (!isTRUE(has_cuda())) {
    "cuda"
  } else if (!isTRUE(has_metal())) {
    "metal"
  } else {
    skip("Both optional accelerator backends are available")
  }
  old_option <- getOption("backend", NULL)
  on.exit(options(backend = old_option), add = TRUE)
  options(backend = unavailable)

  expect_error(fastPLS_backend(), "No CPU fallback")
  expect_identical(getOption("backend"), unavailable)
})

test_that("each unavailable accelerator stops public operations", {
  X <- matrix(rnorm(40), 10, 4)
  y <- rnorm(10)
  fit <- pls(X, y, ncomp = 1, backend = "cpu", return_variance = FALSE)

  for (requested in c("cuda", "metal")) {
    available <- if (identical(requested, "cuda")) has_cuda() else has_metal()
    if (isTRUE(available)) {
      next
    }
    expect_error(
      pls(X, y, ncomp = 1, backend = requested),
      "No CPU fallback",
      info = paste("pls backend", requested)
    )
    expect_error(
      fastsvd(X, ncomp = 1, backend = requested),
      "No CPU fallback",
      info = paste("fastsvd backend", requested)
    )
    expect_error(
      predict(fit, X[1:2, , drop = FALSE], backend = requested),
      "No CPU fallback",
      info = paste("prediction backend", requested)
    )
    expect_error(
      pls.single.cv(X, y, ncomp = 1, kfold = 2, backend = requested),
      "No CPU fallback",
      info = paste("single CV backend", requested)
    )
    expect_error(
      pls.double.cv(
        X,
        y,
        ncomp = 1,
        kfold_inner = 2,
        kfold_outer = 2,
        backend = requested
      ),
      "No CPU fallback",
      info = paste("double CV backend", requested)
    )
  }
})

test_that("unavailable configured backends stop public operations", {
  unavailable <- if (!isTRUE(has_cuda())) {
    "cuda"
  } else if (!isTRUE(has_metal())) {
    "metal"
  } else {
    skip("Both optional accelerator backends are available")
  }
  old_option <- getOption("backend", NULL)
  on.exit(options(backend = old_option), add = TRUE)
  options(backend = unavailable)
  X <- matrix(rnorm(40), 10, 4)
  y <- rnorm(10)

  expect_error(
    pls(X, y, ncomp = 1),
    "No CPU fallback"
  )
  expect_error(
    fastsvd(X, ncomp = 1),
    "No CPU fallback"
  )
  expect_error(
    pls.single.cv(X, y, ncomp = 1, kfold = 2),
    "No CPU fallback"
  )
  expect_error(
    pls.double.cv(
      X,
      y,
      ncomp = 1,
      kfold_inner = 2,
      kfold_outer = 2
    ),
    "No CPU fallback"
  )
})

test_that("backend availability is checked before input conversion", {
  unavailable <- if (!isTRUE(has_cuda())) {
    "cuda"
  } else if (!isTRUE(has_metal())) {
    "metal"
  } else {
    skip("Both optional accelerator backends are available")
  }
  malformed <- list(not = "a matrix")
  X <- matrix(rnorm(40), 10, 4)
  y <- rnorm(10)
  fit <- pls(X, y, ncomp = 1, backend = "cpu", return_variance = FALSE)

  expect_error(
    pls(malformed, y, ncomp = 1, backend = unavailable),
    "No CPU fallback"
  )
  expect_error(
    fastsvd(malformed, ncomp = 1, backend = unavailable),
    "No CPU fallback"
  )
  expect_error(
    predict(fit, malformed, backend = unavailable),
    "No CPU fallback"
  )
})

test_that("environment-selected unavailable backends stop CV", {
  unavailable <- if (!isTRUE(has_cuda())) {
    "cuda"
  } else if (!isTRUE(has_metal())) {
    "metal"
  } else {
    skip("Both optional accelerator backends are available")
  }
  old_option <- getOption("backend", NULL)
  old_env <- Sys.getenv("FASTPLS_BACKEND", unset = NA_character_)
  on.exit({
    options(backend = old_option)
    if (is.na(old_env)) {
      Sys.unsetenv("FASTPLS_BACKEND")
    } else {
      Sys.setenv(FASTPLS_BACKEND = old_env)
    }
  }, add = TRUE)
  options(backend = NULL)
  do.call(Sys.setenv, list(FASTPLS_BACKEND = unavailable))
  X <- matrix(rnorm(40), 10, 4)
  y <- rnorm(10)

  expect_error(
    pls.single.cv(X, y, ncomp = 1, kfold = 2),
    "No CPU fallback"
  )
  expect_error(
    pls.double.cv(
      X,
      y,
      ncomp = 1,
      kfold_inner = 2,
      kfold_outer = 2
    ),
    "No CPU fallback"
  )
})

test_that("prediction and CV reject unavailable accelerator requests", {
  unavailable <- if (!isTRUE(has_cuda())) {
    "cuda"
  } else if (!isTRUE(has_metal())) {
    "metal"
  } else {
    skip("Both optional accelerator backends are available")
  }
  prediction_backend <- unavailable
  X <- matrix(rnorm(60), 15, 4)
  y <- rnorm(15)
  fit <- pls(X, y, ncomp = 1, backend = "cpu", return_variance = FALSE)

  expect_error(
    predict(fit, X[1:2, , drop = FALSE], backend = prediction_backend),
    "No CPU fallback"
  )
  expect_error(
    pls.single.cv(X, y, ncomp = 1, kfold = 3, backend = unavailable),
    "No CPU fallback"
  )
  expect_error(
    pls.double.cv(
      X,
      y,
      ncomp = 1,
      kfold_inner = 2,
      kfold_outer = 2,
      backend = c("cpu", unavailable)
    ),
    "No CPU fallback"
  )
})

test_that("model-aware prediction does not fall back from an unavailable backend", {
  unavailable <- if (!isTRUE(has_cuda())) {
    "cuda"
  } else if (!isTRUE(has_metal())) {
    "metal"
  } else {
    skip("Both optional accelerator backends are available")
  }
  stored <- if (identical(unavailable, "cuda")) "cuda_flash" else "metal"
  model <- list(
    predict_backend = stored,
    flash_svd = TRUE,
    ncomp = 1L,
    m = 1L
  )

  expect_error(
    fastPLS:::.prediction_route(model, matrix(0, 1, 1), "auto", NULL),
    "No CPU fallback"
  )
})

test_that("omitted prediction backend follows the session configuration", {
  old_option <- getOption("backend", NULL)
  on.exit(options(backend = old_option), add = TRUE)
  model <- list(predict_backend = "cpu_flash", ncomp = 1L, m = 1L)
  X <- matrix(0, 1L, 1L)

  options(backend = "cpu")
  expect_identical(
    fastPLS:::.prediction_route(model, X, NULL, NULL)$selected,
    "cpu"
  )

  expect_identical(
    fastPLS:::.prediction_route(model, X, "auto", NULL)$selected,
    "cpu"
  )
})

test_that("family prediction wrappers reject unavailable accelerators early", {
  unavailable <- if (!isTRUE(has_cuda())) {
    "cuda"
  } else if (!isTRUE(has_metal())) {
    "metal"
  } else {
    skip("Both optional accelerator backends are available")
  }
  X <- matrix(rnorm(120), 30, 4)
  y <- rnorm(30)
  kernel_fit <- pls(
    X,
    y,
    ncomp = 1,
    method = "kernelpls",
    kernel = "rbf",
    backend = "cpu",
    return_variance = FALSE
  )
  opls_fit <- pls(
    X,
    y,
    ncomp = 1,
    method = "opls",
    backend = "cpu",
    return_variance = FALSE
  )

  expect_error(
    predict(kernel_fit, X[1:2, , drop = FALSE], backend = unavailable),
    "No CPU fallback"
  )
  expect_error(
    predict(opls_fit, X[1:2, , drop = FALSE], backend = unavailable),
    "No CPU fallback"
  )
})

test_that("all public PLS families reject every unavailable accelerator", {
  unavailable <- c(
    if (!isTRUE(has_cuda())) "cuda",
    if (!isTRUE(has_metal())) "metal"
  )
  if (!length(unavailable)) {
    skip("Both optional accelerator backends are available")
  }
  X <- matrix(rnorm(72), 18, 4)
  y_reg <- rnorm(18)
  y_cls <- factor(rep(c("a", "b", "c"), each = 6))
  X32 <- float::fl(X)
  y32 <- float::fl(matrix(y_reg, ncol = 1L))

  for (requested in unavailable) {
    for (family in c("plssvd", "simpls", "opls", "kernelpls")) {
      for (response in list(y_reg, y_cls)) {
        expect_error(
          pls(
            X,
            response,
            ncomp = 1,
            method = family,
            backend = requested
          ),
          "No CPU fallback",
          info = paste(requested, family, class(response)[1L])
        )
      }
      expect_error(
        pls(
          X32,
          y32,
          ncomp = 1,
          method = family,
          backend = requested
        ),
        "No CPU fallback",
        info = paste(requested, family, "float32")
      )
    }

    expect_error(
      pls.single.cv(
        X,
        y_cls,
        ncomp = 1,
        kfold = 2,
        method = c("plssvd", "simpls", "opls", "kernelpls"),
        classifier = c("argmax", "lda"),
        backend = requested
      ),
      "No CPU fallback"
    )
    expect_error(
      pls.double.cv(
        X,
        y_cls,
        ncomp = 1,
        kfold_inner = 2,
        kfold_outer = 2,
        method = c("plssvd", "simpls", "opls", "kernelpls"),
        classifier = c("argmax", "lda"),
        backend = requested
      ),
      "No CPU fallback"
    )
  }
})

test_that("every PLS family and classifier rejects an unavailable backend", {
  unavailable <- if (!isTRUE(has_cuda())) {
    "cuda"
  } else if (!isTRUE(has_metal())) {
    "metal"
  } else {
    skip("Both optional accelerator backends are available")
  }
  X <- matrix(rnorm(120), 30, 4)
  regression <- rnorm(30)
  classification <- factor(rep(c("a", "b"), each = 15))

  for (method in c("plssvd", "simpls", "opls", "kernelpls")) {
    expect_error(
      pls(
        X,
        regression,
        ncomp = 1,
        method = method,
        backend = unavailable,
        kernel = "rbf",
        return_variance = FALSE
      ),
      "No CPU fallback"
    )
    for (classifier in c("argmax", "lda")) {
      expect_error(
        pls(
          X,
          classification,
          ncomp = 1,
          method = method,
          classifier = classifier,
          backend = unavailable,
          kernel = "rbf",
          return_variance = FALSE
        ),
        "No CPU fallback"
      )
    }
  }
})

test_that("internal prediction routes are converted to public backends", {
  expect_identical(
    fastPLS:::.model_public_backend(list(predict_backend = "cuda_flash")),
    "cuda"
  )
  expect_identical(
    fastPLS:::.model_public_backend(list(predict_backend = "float32_cuda")),
    "cuda"
  )
  expect_identical(
    fastPLS:::.model_public_backend(list(predict_backend = "metal")),
    "metal"
  )
  expect_identical(
    fastPLS:::.model_public_backend(list(predict_backend = "cpu_flash")),
    "cpu"
  )
})

test_that("invalid explicit backend vectors are not silently ignored", {
  old_option <- getOption("backend", NULL)
  on.exit(options(backend = old_option), add = TRUE)
  options(backend = "cpu")
  expect_error(
    fastPLS:::.fastpls_resolve_backend(c("cuda", "cpu")),
    "must be one of"
  )
})
