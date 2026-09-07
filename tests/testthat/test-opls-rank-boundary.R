opls_rank_task <- function() {
    set.seed(901)
    X <- matrix(rnorm(72 * 8), 72, 8)
    X <- sweep(X, 2L, seq_len(ncol(X)), "*")
    list(X = X, Y = cbind(X[, 1] + X[, 3], X[, 2] - X[, 4]),
        y = factor(rep(1:3, length.out = nrow(X))))
}

test_that("OPLS rank guard accounts for centering and removed directions", {
    check <- fastPLS:::.opls_require_predictive_rank
    X <- matrix(0, 6, 8)
    expect_identical(check(4L, X, 1L, TRUE), 4L)
    expect_identical(check(5L, X, 1L, FALSE), 5L)
    expect_error(check(5L, X, 1L, TRUE), "at most 4 remain")
    expect_error(check(6L, X, 1L, FALSE), "at most 5 remain")
})

for (rank_backend in c("cpu", "cuda", "metal")) {
    local({
        backend <- rank_backend
        test_that(paste("OPLS rejects invalid rank on", backend), {
            if (backend == "cuda") skip_if_not(has_cuda())
            if (backend == "metal") skip_if_not(has_metal())
            task <- opls_rank_task()
            solvers <- if (backend == "cpu") "rsvd" else "rsvd"
            for (solver in solvers) {
                for (precision in c("float64", "float32")) {
                    if (.Platform$OS.type == "windows" &&
                        precision == "float32" &&
                        (backend != "cpu" || solver != "rsvd")) next
                    X <- if (precision == "float32") float::fl(task$X) else task$X
                    if (backend == "metal" && precision == "float64") {
                        expect_error(
                            pls(X, task$Y, method = "opls", north = 1L,
                                ncomp = 1:3, backend = backend,
                                svd.method = solver, seed = 19L),
                            "does not provide native float64"
                        )
                        next
                    }
                    for (Y in list(task$Y, task$y)) {
                        if (precision == "float32" && is.matrix(Y)) Y <- float::fl(Y)
                        for (head in c("argmax", "lda")) {
                            if (is.matrix(Y) || inherits(Y, "float32")) {
                                if (head == "lda") next
                            }
                            expect_error(suppressWarnings(pls(X, Y,
                                method = "opls", north = 1L, ncomp = 8L,
                                backend = backend, svd.method = solver,
                                classifier = head, seed = 19L)),
                                "OPLS requested 8 predictive components, but at most 7 remain")
                            fit <- suppressWarnings(pls(X, Y,
                                method = "opls", north = 1L, ncomp = 1:3,
                                backend = backend, svd.method = solver,
                                classifier = head, seed = 19L))
                            model <- fastPLS:::.fastpls_restore_internal_output_fields(fit)
                            expect_equal(as.integer(model$ncomp), 1:3)
                        }
                    }
                }
            }
        })
        test_that(paste("compiled OPLS CV guards each fold rank on", backend), {
            if (backend == "cuda") skip_if_not(has_cuda())
            if (backend == "metal") skip_if_not(has_metal())
            task <- opls_rank_task()
            if (backend %in% c("cuda", "metal")) {
                X <- if (backend == "metal") float::fl(task$X) else task$X
                Y <- if (backend == "metal") float::fl(task$Y) else task$Y
                expect_error(
                    pls.single.cv(
                        X, Y, method = "opls", north = 1L, ncomp = 8L,
                        kfold = 3L, backend = backend, seed = 12L
                    ),
                    "OPLS requested 8 predictive components"
                )
                return(invisible())
            }
            for (classifier in 0:1) {
                set.seed(12)
                expect_error(fastPLS:::pls_cv_predict_compiled(
                    Xdata = task$X,
                    Ydata = matrix(as.double(task$y), ncol = 1),
                    constrain = seq_len(nrow(task$X)), ncomp = 8L,
                    scaling = 1L, kfold = 3L, method = 4L,
                    backend = match(backend, c("cpu", "cuda", "metal")) - 1L,
                    svd_method = fastPLS:::.svd_method_id(
                        if (backend == "cuda") "cuda_rsvd" else "cpu_rsvd"),
                    rsvd_oversample = 32L, rsvd_power = 5L,
                    svds_tol = 0, seed = 12L, classification = TRUE,
                    n_response = 3L, xprod = FALSE, opls_north = 1L,
                    return_scores = TRUE, class_codes = matrix(numeric(), 0, 0),
                    classifier = classifier, lda_ridge = 0,
                    store_predictions = TRUE, metric_id = 4L),
                    "OPLS requested 8 predictive components, but at most 7 remain")
            }
        })
    })
}
