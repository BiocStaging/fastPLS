test_that("public PLS APIs do not expose an SVD selector", {
    X <- as.matrix(iris[, 1:4])
    y <- iris$Species
    for (fun in list(pls, pls.single.cv, pls.double.cv)) {
        expect_false("svd.method" %in% names(formals(fun)))
        expect_error(fun(X, y, svd.method = "rsvd"), "has been removed")
        expect_error(fun(X, y, svd.method = "irlba"), "svd.method")
        expect_error(fun(X, y, svd.method = c("rsvd", "irlba")), "svd.method")
        expect_error(fun(X, y, work = 20L), "Unknown entr")
        expect_error(fun(X, y, svds_tol = 0.1), "Unknown entr")
    }
})

test_that("fastsvd exposes only the native rSVD route", {
    X <- as.matrix(iris[, 1:4])
    expect_false("method" %in% names(formals(fastsvd)))
    expect_error(fastsvd(X, method = "rsvd"), "unused argument")
    expect_error(fastsvd(X, method = "irlba"), "unused argument")
})

test_that("removed float32 solver entry points are absent", {
    expect_false(exists("metal_float32_irlba_cpp",
        envir = asNamespace("fastPLS"), inherits = FALSE))
    expect_false(exists("fastsvd_float32_cpp",
        envir = asNamespace("fastPLS"), inherits = FALSE))
})

test_that("legacy compiled model entry points are absent", {
    ns <- asNamespace("fastPLS")
    removed <- c(
        "pls_model1", "pls_model2", "pls_model2_fast",
        "pls_model1_rsvd_xprod_precision",
        "pls_model2_fast_rsvd_xprod_precision"
    )
    expect_false(any(vapply(removed, exists, logical(1), envir = ns,
        inherits = FALSE)))
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
    registry <- fastPLS:::.svd_control_defaults()
    expect_false(any(grepl("irlba", names(registry), ignore.case = TRUE)))
    expect_setequal(names(fastPLS:::.svd_direct_aliases()),
        c("oversample", "power"))
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
    expect_false(grepl("IRLBA", fastPLS:::.solver_diagnostic_guidance(
        list(randomized = FALSE))))
})
