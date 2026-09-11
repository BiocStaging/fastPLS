test_that("standalone core CV is deterministic for grouped classification", {
    set.seed(2026)
    labels <- rep(1:3, each = 40)
    X <- matrix(rnorm(120 * 15), 120, 15)
    X[, 1:3] <- X[, 1:3] + 4 * model.matrix(~ factor(labels) - 1)
    groups <- rep(seq_len(60), each = 2)
    components <- 1:2
    set.seed(41)
    folds <- fastPLS:::cv_folds_core_cpp(groups, labels, 3L, 4L)
    expect_true(all(vapply(split(folds, groups), function(x) {
        length(unique(x)) == 1L
    }, logical(1))))

    for (method in c(1L, 3L)) {
        for (classifier in c(0L, 1L)) {
            core <- fastPLS:::pls_cv_classification_core_cpp(
                X, labels, 3L, folds, components, 1L, method,
                classifier, 16L, 3L, 41L, TRUE, classifier == 0L
            )
            repeated <- fastPLS:::pls_cv_classification_core_cpp(
                X, labels, 3L, folds, components, 1L, method,
                classifier, 16L, 3L, 41L, TRUE, classifier == 0L
            )
            expect_identical(core$fold, as.integer(folds))
            expect_true(all(core$status == 1L))
            expect_equal(dim(core$class_pred), c(nrow(X), length(components)))
            expect_identical(core$class_pred, repeated$class_pred)
            expect_equal(core$metric_value, repeated$metric_value)
            expect_true(all(is.finite(core$metric_value)))
            expect_identical(
                core$native_best_index,
                as.integer(which.max(core$metric_value))
            )
            expect_identical(
                core$native_best_ncomp,
                components[[core$native_best_index]]
            )
            if (classifier == 0L) {
                expect_length(core$Q2Y, length(components))
                expect_true(all(is.finite(core$Q2Y)))
            }
        }
    }
})

test_that("standalone core CV is deterministic for grouped regression", {
    set.seed(911)
    X <- matrix(rnorm(96 * 12), 96, 12)
    Y <- cbind(
        X[, 1] - 0.4 * X[, 2] + rnorm(96, sd = 0.2),
        X[, 3] + 0.3 * X[, 5] + rnorm(96, sd = 0.2),
        X[, 4] - X[, 6] + rnorm(96, sd = 0.2)
    )
    groups <- rep(seq_len(48), each = 2)
    components <- 1:3
    set.seed(73)
    folds <- fastPLS:::cv_folds_core_cpp(groups, NULL, 0L, 4L)

    for (method in c(1L, 3L)) {
        core <- fastPLS:::pls_cv_regression_core_cpp(
            X, Y, folds, components, 1L, method, 4L,
            16L, 3L, 73L, TRUE
        )
        repeated <- fastPLS:::pls_cv_regression_core_cpp(
            X, Y, folds, components, 1L, method, 4L,
            16L, 3L, 73L, TRUE
        )
        expect_identical(core$fold, as.integer(folds))
        expect_true(all(core$status == 1L))
        expect_equal(core$metric_value, repeated$metric_value)
        expect_equal(core$Ypred, repeated$Ypred)
        expect_true(all(is.finite(core$metric_value)))
        expect_identical(
            core$native_best_index,
            as.integer(which.min(core$metric_value))
        )
        expect_identical(
            core$native_best_ncomp,
            components[[core$native_best_index]]
        )
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
    expect_null(candidate$native_best_index)
    expect_null(candidate$native_best_ncomp)
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

test_that("standalone OPLS CV preserves float32 workflows", {
    set.seed(88)
    labels <- rep(1:3, each = 30)
    X <- matrix(rnorm(90 * 12), 90, 12)
    X[, 1:3] <- X[, 1:3] + 3 * model.matrix(~ factor(labels) - 1)
    Y <- cbind(
        X[, 1] - 0.5 * X[, 2] + rnorm(90, sd = 0.2),
        X[, 3] + 0.3 * X[, 4] + rnorm(90, sd = 0.2)
    )
    groups <- rep(seq_len(45), each = 2)
    components <- 1:2
    set.seed(41)
    folds <- fastPLS:::cv_folds_core_cpp(groups, labels, 3L, 4L)

    core <- fastPLS:::pls_cv_opls_classification_core_cpp(
        X, labels, 3L, folds, components, 1L, 0L, 1L,
        16L, 3L, 41L, TRUE, TRUE
    )
    core32 <- fastPLS:::pls_cv_opls_classification_float32_core_cpp(
        float::fl(X), labels, 3L, folds, components, 1L, 0L, 1L,
        16L, 3L, 41L, TRUE, TRUE
    )
    expect_identical(core32$class_pred, core$class_pred)
    expect_equal(core32$Ypred, core$Ypred, tolerance = 1e-4)

    set.seed(19)
    regression_folds <- fastPLS:::cv_folds_core_cpp(groups, NULL, 0L, 4L)
    core_regression <- fastPLS:::pls_cv_opls_regression_core_cpp(
        X, Y, regression_folds, components, 1L, 4L, 1L,
        16L, 3L, 19L, TRUE
    )
    core_regression32 <-
        fastPLS:::pls_cv_opls_regression_float32_core_cpp(
            float::fl(X), float::fl(Y), regression_folds, components,
            1L, 4L, 1L, 16L, 3L, 19L, TRUE
        )
    expect_equal(
        core_regression32$metric_value,
        core_regression$metric_value,
        tolerance = 1e-4
    )
    expect_equal(
        core_regression32$Ypred, core_regression$Ypred,
        tolerance = 1e-4
    )
})

kernel_fold_reference <- function(X, Y, folds, components, kernel, gamma,
                                  degree, coef0, classifier, seed) {
    classification <- is.factor(Y)
    predictions <- if (classification) {
        matrix(NA_integer_, nrow(X), length(components))
    } else {
        array(NA_real_, c(nrow(X), ncol(Y), length(components)))
    }
    for (fold in sort(unique(folds))) {
        test <- which(folds == fold)
        train <- which(folds != fold)
        fitted <- pls(
            X[train, , drop = FALSE],
            if (classification) Y[train] else Y[train, , drop = FALSE],
            X[test, , drop = FALSE],
            ncomp = components,
            scaling = "centering",
            method = "kernelpls",
            backend = "cpu",
            kernel = kernel,
            gamma = gamma,
            degree = degree,
            coef0 = coef0,
            classifier = classifier,
            fit = FALSE,
            return_variance = FALSE,
            oversample = 16L,
            power = 3L,
            seed = seed + fold - 1L
        )
        if (classification) {
            for (index in seq_along(components)) {
                predictions[test, index] <- as.integer(fitted$Ypred[[index]])
            }
        } else {
            for (index in seq_along(components)) {
                predictions[test, , index] <- fitted$Ypred[, , index]
            }
        }
    }
    predictions
}

test_that("standalone nonlinear kernel CV preserves independent fold fits", {
    set.seed(703)
    labels <- rep(1:3, each = 24)
    factor_labels <- factor(labels)
    X <- matrix(rnorm(72 * 9), 72, 9)
    X[, 1:3] <- X[, 1:3] + 2.5 * model.matrix(~ factor(labels) - 1)
    Y <- cbind(
        sin(X[, 1]) + 0.2 * X[, 4],
        X[, 2]^2 - 0.3 * X[, 5]
    )
    groups <- rep(seq_len(36), each = 2)
    components <- 1:2
    set.seed(37)
    folds <- fastPLS:::cv_folds_core_cpp(groups, labels, 3L, 3L)

    for (kernel in c("rbf", "poly")) {
        kernel_id <- if (kernel == "rbf") 2L else 3L
        gamma <- if (kernel == "rbf") 0.2 else 0.1
        for (classifier in c("argmax", "lda")) {
            classifier_id <- if (classifier == "argmax") 0L else 1L
            reference <- kernel_fold_reference(
                X, factor_labels, folds, components, kernel, gamma,
                2L, 0.5, classifier, 37L
            )
            core <- fastPLS:::pls_cv_kernel_classification_core_cpp(
                X, labels, 3L, folds, components, 1L, classifier_id,
                kernel_id, gamma, 2L, 0.5, 16L, 3L, 37L, TRUE, TRUE
            )
            expect_identical(core$class_pred, reference)
            expect_equal(
                core$metric_value,
                colMeans(reference == labels),
                tolerance = 1e-12
            )
        }

        reference <- kernel_fold_reference(
            X, Y, folds, components, kernel, gamma, 2L, 0.5,
            "argmax", 37L
        )
        core <- fastPLS:::pls_cv_kernel_regression_core_cpp(
            X, Y, folds, components, 1L, 4L, kernel_id, gamma, 2L, 0.5,
            16L, 3L, 37L, TRUE
        )
        expect_equal(core$Ypred, reference, tolerance = 1e-9)
        expect_equal(
            core$metric_value,
            vapply(
                seq_along(components),
                function(index) sqrt(mean((reference[, , index] - Y)^2)),
                numeric(1)
            ),
            tolerance = 1e-10
        )
    }
})

test_that("float32 and public nonlinear kernel CV use the standalone core", {
    set.seed(181)
    labels <- rep(1:3, each = 20)
    y <- factor(labels)
    X <- matrix(rnorm(60 * 8), 60, 8)
    X[, 1:3] <- X[, 1:3] + 2 * model.matrix(~ factor(labels) - 1)
    Y <- cbind(cos(X[, 1]), X[, 2] * X[, 3])
    groups <- rep(seq_len(30), each = 2)
    components <- 1:2
    set.seed(11)
    folds <- fastPLS:::cv_folds_core_cpp(groups, labels, 3L, 3L)

    class64 <- fastPLS:::pls_cv_kernel_classification_core_cpp(
        X, labels, 3L, folds, components, 1L, 1L, 2L, 0.2, 3L, 1,
        16L, 3L, 11L, TRUE, TRUE
    )
    class32 <- fastPLS:::pls_cv_kernel_classification_float32_core_cpp(
        float::fl(X), labels, 3L, folds, components, 1L, 1L, 2L, 0.2,
        3L, 1, 16L, 3L, 11L, TRUE, TRUE
    )
    expect_identical(class32$class_pred, class64$class_pred)
    expect_equal(class32$Ypred, class64$Ypred, tolerance = 2e-4)

    regression64 <- fastPLS:::pls_cv_kernel_regression_core_cpp(
        X, Y, folds, components, 1L, 4L, 2L, 0.2, 3L, 1,
        16L, 3L, 11L, TRUE
    )
    regression32 <- fastPLS:::pls_cv_kernel_regression_float32_core_cpp(
        float::fl(X), float::fl(Y), folds, components, 1L, 4L, 2L, 0.2,
        3L, 1, 16L, 3L, 11L, TRUE
    )
    expect_equal(regression32$metric_value, regression64$metric_value,
        tolerance = 2e-4)
    expect_equal(regression32$Ypred, regression64$Ypred, tolerance = 2e-4)

    arguments <- list(
        Ydata = y, constrain = groups, ncomp = components, kfold = 3L,
        method = "kernelpls", kernel = "rbf", gamma = 0.2,
        backend = "cpu", classifier = "lda", fit = FALSE,
        oversample = 16L, power = 3L, seed = 11L
    )
    public64 <- do.call(pls.single.cv, c(list(Xdata = X), arguments))
    public32 <- do.call(
        pls.single.cv,
        c(list(Xdata = float::fl(X)), arguments)
    )
    expect_identical(public32$best_ncomp, public64$best_ncomp)
    expect_equal(public32$accuracy, public64$accuracy)
    expect_true(all(vapply(
        seq_along(public64$pred),
        function(index) identical(
            as.character(public32$pred[[index]]),
            as.character(public64$pred[[index]])
        ),
        logical(1)
    )))

    regression_arguments <- list(
        constrain = groups, ncomp = components, kfold = 3L,
        method = "kernelpls", kernel = "poly", gamma = 0.1,
        degree = 2L, coef0 = 0.5, backend = "cpu", fit = FALSE,
        oversample = 16L, power = 3L, seed = 11L
    )
    public_regression64 <- do.call(
        pls.single.cv,
        c(list(Xdata = X, Ydata = Y), regression_arguments)
    )
    public_regression32 <- do.call(
        pls.single.cv,
        c(list(Xdata = float::fl(X), Ydata = float::fl(Y)),
            regression_arguments)
    )
    expect_identical(
        public_regression32$best_ncomp,
        public_regression64$best_ncomp
    )
    expect_equal(
        public_regression32$RMSD,
        public_regression64$RMSD,
        tolerance = 2e-4
    )
})
