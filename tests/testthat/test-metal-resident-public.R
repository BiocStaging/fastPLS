test_that("public float32 SIMPLS uses the explicit Metal operator", {
    skip_if_not(has_metal(), "Metal backend is not available")

    set.seed(41)
    x <- matrix(rnorm(180 * 24), 180, 24)
    coefficients <- matrix(rnorm(24 * 3), 24, 3)
    y <- x %*% coefficients + matrix(rnorm(180 * 3, sd = 0.1), 180, 3)
    train <- seq_len(140)
    test <- setdiff(seq_len(180), train)

    fit <- pls(
        float::fl(x[train, , drop = FALSE]),
        float::fl(y[train, , drop = FALSE]),
        float::fl(x[test, , drop = FALSE]),
        float::fl(y[test, , drop = FALSE]),
        ncomp = c(2, 5, 8), method = "simpls", backend = "metal",
        fit = TRUE, proj = TRUE,
        return_variance = TRUE, seed = 13
    )

    expect_s3_class(fit, "fastPLS")
    expect_length(fit$Yfit, 3L)
    expect_length(fit$Ypred, 3L)
    expect_equal(dim(fit$Ttest), c(length(test), 8L))
    expect_equal(dim(fit$P), c(0L, 0L))
    expect_true(all(is.finite(fit$R2Y)))
    expect_true(all(is.finite(fit$Q2Y)))

    predicted <- predict(
        fit,
        float::fl(x[test, , drop = FALSE]),
        float::fl(y[test, , drop = FALSE]),
        proj = TRUE
    )
    expect_equal(
        float::dbl(predicted$Ypred[[3L]]),
        float::dbl(fit$Ypred[[3L]]),
        tolerance = 2e-5
    )
})

test_that("Metal classification preserves labels and top ranks", {
    skip_if_not(has_metal(), "Metal backend is not available")

    set.seed(52)
    x <- matrix(rnorm(240 * 20), 240, 20)
    y <- factor(rep(c("alpha", "beta", "gamma"), each = 80))
    x[y == "alpha", 1:3] <- x[y == "alpha", 1:3] + 2
    x[y == "beta", 4:6] <- x[y == "beta", 4:6] + 2
    x[y == "gamma", 7:9] <- x[y == "gamma", 7:9] + 2

    fit <- pls(
        float::fl(x[seq_len(180), , drop = FALSE]), y[seq_len(180)],
        float::fl(x[181:240, , drop = FALSE]), y[181:240],
        ncomp = c(2, 4, 6), method = "simpls", classifier = "argmax",
        backend = "metal", fit = TRUE, proj = TRUE, seed = 17
    )
    predicted <- predict(
        fit, float::fl(x[181:240, , drop = FALSE]), y[181:240], top = 2L
    )

    expect_true(all(is.finite(fit$accuracy)))
    expect_length(predicted$Ypred_top, 3L)
    expect_identical(levels(predicted$Ypred[[1L]]), levels(y))
})

test_that("Metal score projection and float32 LDA preserve the discriminant", {
    skip_if_not(has_metal(), "Metal backend is not available")

    set.seed(702)
    n <- 360L
    p <- 30L
    classes <- 4L
    x <- matrix(rnorm(n * p), n, p)
    y <- factor(rep(paste0("class", seq_len(classes)), each = n / classes))
    for (class_index in seq_len(classes)) {
        rows <- y == paste0("class", class_index)
        columns <- (3L * class_index - 2L):(3L * class_index)
        x[rows, columns] <- x[rows, columns] + 1.5
    }
    train <- unlist(lapply(split(seq_len(n), y), function(index) index[1:70]))
    test <- setdiff(seq_len(n), train)
    components <- c(2L, 5L, 8L)

    fit <- pls(
        float::fl(x[train, , drop = FALSE]), y[train],
        float::fl(x[test, , drop = FALSE]), y[test],
        ncomp = components, method = "simpls", classifier = "lda",
        backend = "metal", fit = TRUE, proj = TRUE,
        return_variance = FALSE, seed = 29
    )
    metal <- predict(
        fit, float::fl(x[test, , drop = FALSE]), y[test], raw_scores = TRUE
    )
    models <- fastPLS:::lda_train_prefix_float32_cpp(
        fastPLS:::.as_float32_matrix(fit$Ttrain, "Ttrain"), as.integer(y[train]),
        classes, components
    )

    for (index in seq_along(components)) {
        component <- components[[index]]
        scores <- fastPLS:::.as_float32_matrix(
            fit$Ttest[, seq_len(component), drop = FALSE], "Ttest"
        )
        cpu <- fastPLS:::lda_predict_float32_cpp(
            scores, models[[as.character(component)]], TRUE
        )
        cpu_scores <- float::dbl(fastPLS:::.float32_from_bits(cpu$scores))
        metal_scores <- float::dbl(metal$LDA_scores[[index]])
        expect_equal(
            unname(metal_scores), unname(cpu_scores), tolerance = 2e-4
        )
        expect_identical(
            as.character(fit$Ypred[[index]]),
            levels(y)[cpu$pred]
        )
    }
})

test_that("Metal PLS-SVD reuses its compact prefix factors", {
    skip_if_not(has_metal(), "Metal backend is not available")

    set.seed(811)
    x <- matrix(rnorm(220 * 28), 220, 28)
    coefficients <- matrix(rnorm(28 * 6), 28, 6)
    y <- x %*% coefficients + matrix(rnorm(220 * 6, sd = 0.2), 220, 6)
    train <- seq_len(170)
    test <- 171:220
    components <- c(2L, 4L, 6L)
    fit <- pls(
        float::fl(x[train, ]), float::fl(y[train, ]),
        float::fl(x[test, ]), float::fl(y[test, ]),
        ncomp = components, method = "plssvd", backend = "metal",
        fit = TRUE, proj = TRUE,
        return_variance = FALSE, power = 5, seed = 31
    )

    r <- float::dbl(fit$R)
    ttrain <- float::dbl(fit$Ttrain)
    mean_x <- as.vector(float::dbl(fit$mX))
    scale_x <- as.vector(float::dbl(fit$vX))
    mean_y <- as.vector(float::dbl(fit$mY))
    xtrain <- sweep(sweep(x[train, ], 2, mean_x), 2, scale_x, "/")
    ytrain <- sweep(y[train, ], 2, mean_y)
    cross_covariance <- crossprod(xtrain, ytrain)
    xtest <- sweep(sweep(x[test, ], 2, mean_x), 2, scale_x, "/")

    for (index in seq_along(components)) {
        component <- components[[index]]
        selected <- seq_len(component)
        weights <- solve(
            crossprod(ttrain[, selected, drop = FALSE]),
            crossprod(r[, selected, drop = FALSE], cross_covariance)
        )
        expected <- xtest %*% r[, selected, drop = FALSE] %*% weights
        expected <- sweep(expected, 2, mean_y, "+")
        expect_equal(
            float::dbl(fit$Ypred[[index]]), expected,
            tolerance = 3e-4
        )
    }
})

test_that("Metal never returns a hybrid PLS model", {
    skip_if_not(has_metal(), "Metal backend is not available")
    set.seed(832)
    X <- float::fl(matrix(rnorm(90 * 12), 90, 12))
    Y <- float::fl(matrix(rnorm(90 * 3), 90, 3))

    for (method in c("simpls", "plssvd", "opls", "kernelpls")) {
        fit <- pls(
            X, Y, ncomp = 2, method = method, kernel = "linear",
            backend = "metal", return_variance = FALSE, seed = 19
        )
        residency <- fit$diagnostics$residency
        expect_identical(residency$route, "resident metal")
        expect_true(all(unlist(residency[c(
            "preprocessing", "cross_products", "decomposition",
            "component_updates", "prediction",
            "response_sums_of_squares"
        )]) == "metal"))
    }
    nonlinear <- suppressWarnings(pls(
        X, Y, ncomp = 2, method = "kernelpls", kernel = "rbf",
        backend = "metal", return_variance = FALSE
    ))
    expect_identical(
        nonlinear$diagnostics$residency$route,
        "resident metal"
    )
})

test_that("resident Metal OPLS and nonlinear kernel predictions agree with CPU", {
    skip_if_not(has_metal(), "Metal backend is not available")
    set.seed(901)
    n <- 180L
    p <- 16L
    X <- matrix(rnorm(n * p), n, p)
    coefficients <- matrix(rnorm(p * 3L), p, 3L)
    Y <- X %*% coefficients + matrix(rnorm(n * 3L, sd = 0.05), n, 3L)
    train <- seq_len(140L)
    test <- 141:180

    for (method in c("opls", "kernelpls")) {
        kernel <- if (method == "kernelpls") "rbf" else "linear"
        arguments <- list(
            float::fl(X[train, , drop = FALSE]),
            float::fl(Y[train, , drop = FALSE]),
            float::fl(X[test, , drop = FALSE]),
            float::fl(Y[test, , drop = FALSE]),
            ncomp = 3L, method = method, kernel = kernel,
            return_variance = FALSE, seed = 29L
        )
        cpu <- suppressWarnings(do.call(pls, c(arguments, list(backend = "cpu"))))
        gpu <- suppressWarnings(do.call(pls, c(arguments, list(backend = "metal"))))
        cpu_prediction <- float::dbl(cpu$Ypred[[1L]])
        gpu_prediction <- float::dbl(gpu$Ypred[[1L]])
        relative_error <- sqrt(
            sum((cpu_prediction - gpu_prediction)^2) /
                sum(cpu_prediction^2)
        )
        expect_lt(relative_error, 2e-5)
        expect_gt(cor(c(cpu_prediction), c(gpu_prediction)), 0.999999)
        expect_identical(gpu$diagnostics$residency$route, "resident metal")
    }
})

test_that("Metal rejects float64 and retains Metal as the prediction default", {
    skip_if_not(has_metal(), "Metal backend is not available")

    set.seed(63)
    x <- matrix(rnorm(60 * 8), 60, 8)
    y <- matrix(rnorm(60), 60, 1)
    expect_error(
        pls(x, y, ncomp = 2, backend = "metal"),
        "does not provide native float64"
    )

    fit <- pls(
        float::fl(x), float::fl(y), ncomp = 2,
        backend = "metal", return_variance = FALSE
    )
    default_prediction <- predict(fit, float::fl(x))$Ypred[[1L]]
    expect_error(
        predict(fit, float::fl(x), backend = "cpu"),
        "No CPU fallback"
    )
})

test_that("Metal CV uses resident fold models and returns finite PLS scores", {
    skip_if_not(has_metal(), "Metal backend is not available")

    set.seed(914)
    x <- float::fl(matrix(rnorm(120 * 16), 120, 16))
    y <- factor(rep(c("alpha", "beta", "gamma"), each = 40))
    cv <- pls.single.cv(
        x, y, ncomp = 1:3, kfold = 3, method = "simpls",
        classifier = "lda", backend = "metal", fit = FALSE, seed = 27
    )

    expect_false(anyNA(cv$Yscore))
    expect_true(all(is.finite(cv$Q2Y)))
    expect_false(all(cv$Q2Y == 1))
    expect_identical(cv$residency$fold_model_fit, "resident metal")
    expect_identical(cv$residency$fold_prediction, "resident metal")
    expect_identical(cv$residency$fallback, "none")
})
