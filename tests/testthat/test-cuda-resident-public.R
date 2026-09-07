test_that("resident CUDA public fits retain both precisions and prediction semantics", {
    skip_if_not(isTRUE(has_cuda()), "CUDA runtime is unavailable")
    set.seed(23)
    X <- matrix(rnorm(64 * 7), 64, 7)
    Y <- matrix(rnorm(64 * 3), 64, 3)
    labels <- factor(rep(c("a", "b", "c"), length.out = 64))
    Xtest <- matrix(rnorm(11 * 7), 11, 7)
    for (method in c("simpls", "plssvd")) {
        for (precision in c("double", "float32")) {
            for (task in c("regression", "argmax", "lda")) {
                floating <- precision == "float32"
                x <- if (floating) float::fl(X) else X
                xt <- if (floating) float::fl(Xtest) else Xtest
                y <- if (task != "regression") labels else if (floating) float::fl(Y) else Y
                yt <- if (task != "regression") labels[1:11] else if (floating) float::fl(Y[1:11, ]) else Y[1:11, ]
                model <- pls(x, y, xt, yt, ncomp = 1:2, backend = "cuda",
                    method = method, classifier = if (task == "lda") "lda" else "argmax",
                    fit = TRUE, proj = TRUE, return_loadings = TRUE,
                    scaling = "autoscaling", seed = 17, oversample = 32, power = 5)
                expect_identical(model$diagnostics$residency$decomposition, "cuda")
                expect_identical(
                    model$diagnostics$resident_controls$refresh_block_limit,
                    0L
                )
                expect_false("resident_state" %in% names(model))
                expect_true(all(is.finite(model$R2Y)))
                expect_true(all(is.finite(model$Q2Y)))
                expect_identical(dim(model$P), c(7L, 2L))
                expect_identical(dim(model$Ttest), c(11L, 2L))
                predicted <- predict(model, xt, yt, backend = "cuda", proj = TRUE)
                expect_equal(predicted$Ypred, model$Ypred)
                expect_equal(predicted$Q2Y, model$Q2Y)
                if (floating) expect_s4_class(model$R, "float32")
                expect_error(predict(model, xt, backend = "cpu"), "No CPU fallback")
            }
        }
    }
})

test_that("CUDA never returns a hybrid PLS model", {
    skip_if_not(isTRUE(has_cuda()), "CUDA runtime is unavailable")
    set.seed(831)
    X <- matrix(rnorm(80 * 10), 80, 10)
    Y <- matrix(rnorm(80 * 2), 80, 2)

    for (floating in c(FALSE, TRUE)) {
        x <- if (floating) float::fl(X) else X
        y <- if (floating) float::fl(Y) else Y
        for (method in c("simpls", "plssvd", "opls", "kernelpls")) {
            fit <- pls(
                x, y, ncomp = 2, method = method, kernel = "linear",
                backend = "cuda", return_variance = FALSE, seed = 17
            )
            residency <- fit$diagnostics$residency
            expect_identical(residency$route, "resident cuda")
            expect_true(all(unlist(residency[c(
                "preprocessing", "cross_products", "decomposition",
                "component_updates", "prediction",
                "response_sums_of_squares"
            )]) == "cuda"))
        }
        nonlinear <- pls(
            x, y, ncomp = 2, method = "kernelpls", kernel = "rbf",
            backend = "cuda", return_variance = FALSE
        )
        expect_identical(
            nonlinear$diagnostics$residency$route,
            "resident cuda"
        )
    }
})

test_that("resident CUDA class paths match independent prefix prediction", {
    skip_if_not(isTRUE(has_cuda()), "CUDA runtime is unavailable")
    set.seed(832)
    components <- c(1L, 3L, 5L)
    X <- float::fl(matrix(rnorm(90 * 14), 90, 14))
    labels <- factor(rep(c("a", "b", "c"), each = 30))
    Xtest <- float::fl(matrix(rnorm(21 * 14), 21, 14))

    for (classifier in c("argmax", "lda")) {
        model <- pls(
            X, labels, ncomp = components, method = "simpls",
            backend = "cuda", classifier = classifier,
            return_variance = FALSE, seed = 19
        )
        raw <- fastPLS:::.fastpls_restore_internal_output_fields(model)
        bits <- fastPLS:::.resident_cuda_input(Xtest, "float32", "Xtest")
        lda <- as.integer(identical(classifier, "lda"))
        path <- fastPLS:::cuda_resident_classify_path_cpp(
            raw$resident_state, bits, components, lda, 3L
        )
        expect_identical(dim(path), c(21L, 3L, length(components)))
        for (index in seq_along(components)) {
            independent <- fastPLS:::cuda_resident_classify_cpp(
                raw$resident_state, bits, components[[index]], lda, 3L
            )
            expect_equal(path[, , index], independent, tolerance = 0)
        }
    }
})

test_that("resident CUDA materializes training scores only for fitted output", {
    skip_if_not(isTRUE(has_cuda()), "CUDA runtime is unavailable")
    set.seed(833)
    X <- float::fl(matrix(rnorm(96 * 18), 96, 18))
    labels <- factor(rep(letters[1:3], length.out = 96))

    compact <- pls(
        X, labels, ncomp = 1:3, backend = "cuda",
        method = "simpls", fit = FALSE, return_variance = FALSE
    )
    expect_null(compact$Ttrain)

    fitted <- pls(
        X, labels, ncomp = 1:3, backend = "cuda",
        method = "simpls", fit = TRUE, return_variance = FALSE
    )
    expect_equal(dim(fitted$Ttrain), c(nrow(X), 3L))
})

test_that("resident CUDA OPLS and nonlinear kernel predictions agree with CPU", {
    skip_if_not(isTRUE(has_cuda()), "CUDA runtime is unavailable")
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
        for (precision in c("double", "float32")) {
            convert <- if (precision == "float32") float::fl else identity
            arguments <- list(
                convert(X[train, , drop = FALSE]),
                convert(Y[train, , drop = FALSE]),
                convert(X[test, , drop = FALSE]),
                convert(Y[test, , drop = FALSE]),
                ncomp = 3L, method = method, kernel = kernel,
                return_variance = FALSE, seed = 29L
            )
            cpu <- suppressWarnings(do.call(pls, c(arguments, list(backend = "cpu"))))
            gpu <- suppressWarnings(do.call(pls, c(arguments, list(backend = "cuda"))))
            extract <- function(model) {
                if (is.array(model$Ypred)) {
                    return(model$Ypred[, , 1L, drop = TRUE])
                }
                float::dbl(model$Ypred[[1L]])
            }
            cpu_prediction <- extract(cpu)
            gpu_prediction <- extract(gpu)
            relative_error <- sqrt(
                sum((cpu_prediction - gpu_prediction)^2) /
                    sum(cpu_prediction^2)
            )
            tolerance <- if (precision == "float32") 1e-3 else 1e-10
            expect_lt(relative_error, tolerance)
            expect_gt(cor(c(cpu_prediction), c(gpu_prediction)), 0.99999)
            expect_identical(gpu$diagnostics$residency$route, "resident cuda")
        }
    }
})
