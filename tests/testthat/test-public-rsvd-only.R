test_that("public PLS APIs reject removed solvers and controls", {
    X <- as.matrix(iris[, 1:4])
    y <- iris$Species
    for (fun in list(pls, pls.single.cv, pls.double.cv)) {
        expect_identical(formals(fun)$svd.method, "rsvd")
        expect_error(fun(X, y, svd.method = "irlba"), "svd.method")
        expect_error(fun(X, y, svd.method = c("rsvd", "irlba")), "svd.method")
        expect_error(fun(X, y, work = 20L), "Unknown entr")
        expect_error(fun(X, y, svds_tol = 0.1), "Unknown entr")
    }
})

test_that("removed float32 IRLBA entry points cannot be invoked", {
    expect_false(exists("metal_float32_irlba_cpp",
        envir = asNamespace("fastPLS"), inherits = FALSE))
    skip_on_os("windows")
    X <- float::fl(as.matrix(iris[, 1:4]))
    expect_error(fastPLS:::fastsvd_float32_cpp(
        X, 2L, 0L, 1L, 32L, 5L, 1L, FALSE), "rSVD only")
    if (has_metal()) {
        expect_error(fastPLS:::fastsvd_float32_cpp(
            X, 2L, 2L, 1L, 32L, 5L, 1L, FALSE), "rSVD only")
    }
})

test_that("compiled float64 routes reject removed IRLBA requests", {
    X <- as.matrix(iris[, 1:4])
    Y <- X[, 1:2, drop = FALSE]
    for (fun in list(fastPLS:::pls_model1, fastPLS:::pls_model2,
        fastPLS:::pls_model2_fast)) {
        expect_error(fun(X, Y, 1L, 1L, FALSE, 1L, 32L, 5L, 0, 1L),
            "IRLBA is not part")
    }
    for (fun in list(fastPLS:::pls_model1_rsvd_xprod_precision,
        fastPLS:::pls_model2_fast_rsvd_xprod_precision)) {
        expect_error(fun(X, Y, 1L, 1L, FALSE, 32L, 5L, 0, 1L, 5L),
            "IRLBA is not part")
    }
})

test_that("float32 capability records no available IRLBA route", {
    for (backend in c("cpu", "cuda", "metal")) {
        route <- fastPLS:::.float32_capability_assessment(
            "simpls", backend, "irlba", 3L, 2L, os_type = "unix")
        expect_identical(route$status, "unavailable")
        expect_identical(route$action, "error")
    }
})

test_that("IRLBA environment and public control registries are removed", {
    ns <- asNamespace("fastPLS")
    expect_false(exists(".with_irlba_options", envir = ns, inherits = FALSE))
    for (registry in list(fastPLS:::.backend_control_env_defaults,
        fastPLS:::.backend_control_env_groups,
        fastPLS:::.svd_control_defaults())) {
        expect_false(any(grepl("irlba", names(registry), ignore.case = TRUE)))
    }
    expect_setequal(names(fastPLS:::.svd_direct_aliases()),
        c("oversample", "power"))
    expect_identical(fastPLS:::.svd_methods_public, "rsvd")
    expect_true(all(fastPLS:::.svd_methods()$method == "rsvd"))
    expect_error(fastPLS:::.svd_method_id("irlba"), "arg.*should be")
})

test_that("private helpers no longer accept unused IRLBA controls", {
    ns <- asNamespace("fastPLS")
    objects <- mget(ls(ns, all.names = TRUE), envir = ns, inherits = FALSE)
    helpers <- Filter(is.function, objects)
    obsolete <- vapply(helpers, function(fun) {
        any(grepl("^irlba_", names(formals(fun))))
    }, logical(1))
    expect_length(names(helpers)[obsolete], 0L)
    expect_false(exists(".should_use_xprod_irlba_default", ns,
        inherits = FALSE))
    expect_error(fastPLS:::.float32_svd_id("irlba"), "rSVD only")
    expect_error(fastPLS:::.compiled_cv_solver("cpp", "irlba"),
        "arg.*should be")
    expect_false(grepl("IRLBA", fastPLS:::.solver_diagnostic_guidance(
        list(randomized = FALSE))))
})
