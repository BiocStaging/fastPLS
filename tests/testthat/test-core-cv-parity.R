legacy_cv_call <- function(X, Y, groups, components, method,
                           classification, classes, classifier, seed,
                           return_scores = FALSE) {
    set.seed(seed)
    fastPLS:::pls_cv_predict_compiled(
        Xdata = X,
        Ydata = Y,
        constrain = groups,
        ncomp = components,
        scaling = 1L,
        kfold = 4L,
        method = method,
        backend = 0L,
        svd_method = fastPLS:::.svd_method_id("cpu_rsvd"),
        rsvd_oversample = 16L,
        rsvd_power = 3L,
        svds_tol = 0,
        seed = seed,
        classification = classification,
        n_response = classes,
        xprod = FALSE,
        opls_north = 0L,
        return_scores = return_scores,
        class_codes = matrix(numeric(), 0, 0),
        classifier = classifier,
        lda_ridge = 0,
        store_predictions = TRUE,
        metric_id = if (classification) 1L else 4L
    )
}

test_that("standalone core CV preserves linear classification workflows", {
    set.seed(2026)
    labels <- rep(1:3, each = 40)
    X <- matrix(rnorm(120 * 15), 120, 15)
    X[, 1:3] <- X[, 1:3] + 4 * model.matrix(~ factor(labels) - 1)
    groups <- rep(seq_len(60), each = 2)
    components <- 1:2

    for (method in c(1L, 3L)) {
        for (classifier in c(0L, 1L)) {
            legacy <- legacy_cv_call(
                X, matrix(as.double(labels), ncol = 1), groups, components,
                method, TRUE, 3L, classifier, 41L,
                return_scores = classifier == 0L
            )
            core <- fastPLS:::pls_cv_classification_core_cpp(
                X, labels, 3L, legacy$fold, components, 1L, method,
                classifier, 16L, 3L, 41L, TRUE, classifier == 0L
            )
            expect_identical(core$fold, as.integer(legacy$fold))
            expect_identical(core$status, as.integer(legacy$status))
            expect_identical(dim(core$class_pred), dim(legacy$class_pred))
            expect_identical(
                as.integer(core$class_pred), as.integer(legacy$class_pred)
            )
            expect_equal(
                core$metric_value, legacy$metrics$metric_value,
                tolerance = 1e-12
            )
            if (classifier == 0L) {
                expect_equal(core$Ypred, legacy$Ypred, tolerance = 1e-10)
            }
        }
    }
})

test_that("standalone core CV preserves linear regression workflows", {
    set.seed(911)
    X <- matrix(rnorm(96 * 12), 96, 12)
    Y <- cbind(
        X[, 1] - 0.4 * X[, 2] + rnorm(96, sd = 0.2),
        X[, 3] + 0.3 * X[, 5] + rnorm(96, sd = 0.2),
        X[, 4] - X[, 6] + rnorm(96, sd = 0.2)
    )
    groups <- rep(seq_len(48), each = 2)
    components <- 1:3

    for (method in c(1L, 3L)) {
        legacy <- legacy_cv_call(
            X, Y, groups, components, method, FALSE, ncol(Y), 0L, 73L
        )
        core <- fastPLS:::pls_cv_regression_core_cpp(
            X, Y, legacy$fold, components, 1L, method, 4L,
            16L, 3L, 73L, TRUE
        )
        expect_identical(core$fold, as.integer(legacy$fold))
        expect_identical(core$status, as.integer(legacy$status))
        expect_equal(
            core$metric_value, legacy$metrics$metric_value,
            tolerance = 1e-10
        )
        for (index in seq_along(components)) {
            expect_equal(
                core$Ypred[, , index], legacy$Ypred[, , index],
                tolerance = 1e-9
            )
        }
    }
})

test_that("float32 core CV follows the float64 linear workflows", {
    set.seed(622)
    labels <- rep(1:3, each = 32)
    X <- matrix(rnorm(96 * 14), 96, 14)
    X[, 1:3] <- X[, 1:3] + 4 * model.matrix(~ factor(labels) - 1)
    Y <- cbind(
        X[, 1] - 0.2 * X[, 5] + rnorm(96, sd = 0.1),
        X[, 2] + 0.4 * X[, 7] + rnorm(96, sd = 0.1)
    )
    groups <- rep(seq_len(48), each = 2)
    set.seed(29)
    folds <- fastPLS:::cv_folds_core_cpp(groups, labels, 3L, 4L)
    X32 <- float::fl(X)

    for (method in c(1L, 3L)) {
        components <- if (method == 1L) 1:2 else 1:3
        for (classifier in c(0L, 1L)) {
            reference <- fastPLS:::pls_cv_classification_core_cpp(
                X, labels, 3L, folds, components, 1L, method,
                classifier, 16L, 3L, 29L, TRUE, TRUE
            )
            candidate <- fastPLS:::pls_cv_classification_float32_core_cpp(
                X32, labels, 3L, folds, components, 1L, method,
                classifier, 16L, 3L, 29L, TRUE, TRUE
            )
            expect_identical(candidate$fold, reference$fold)
            expect_identical(candidate$status, reference$status)
            expect_equal(candidate$metric_value, reference$metric_value)
            expect_identical(candidate$class_pred, reference$class_pred)
            expect_equal(candidate$Ypred, reference$Ypred, tolerance = 1e-4)
        }

        reference <- fastPLS:::pls_cv_regression_core_cpp(
            X, Y, folds, components, 1L, method, 4L,
            16L, 3L, 29L, TRUE
        )
        candidate <- fastPLS:::pls_cv_regression_float32_core_cpp(
            X32, float::fl(Y), folds, components, 1L, method, 4L,
            16L, 3L, 29L, TRUE
        )
        expect_identical(candidate$fold, reference$fold)
        expect_identical(candidate$status, reference$status)
        expect_equal(candidate$metric_value, reference$metric_value,
            tolerance = 5e-4)
        expect_equal(candidate$Ypred, reference$Ypred, tolerance = 5e-4)
    }
})

test_that("public CPU CV dispatches float32 linear inputs through the core", {
    X <- scale(as.matrix(iris[, -5]))
    y <- iris$Species
    arguments <- list(
        Ydata = y, ncomp = 1:2, kfold = 3L, scaling = "none",
        method = "simpls", backend = "cpu", classifier = "argmax",
        oversample = 16L, power = 3L, seed = 17L, fit = FALSE
    )
    reference <- do.call(pls.single.cv, c(list(Xdata = X), arguments))
    candidate <- do.call(
        pls.single.cv,
        c(list(Xdata = float::fl(X)), arguments)
    )

    expect_identical(candidate$best_ncomp, reference$best_ncomp)
    expect_equal(candidate$accuracy, reference$accuracy)
    expect_equal(candidate$Q2Y, reference$Q2Y, tolerance = 1e-4)
    expect_true(all(vapply(
        seq_along(reference$pred),
        function(index) identical(
            as.character(candidate$pred[[index]]),
            as.character(reference$pred[[index]])
        ),
        logical(1)
    )))
})
