cv_cache_state <- function(enabled) {
    variables <- c(
        "FASTPLS_CV_FOLD_GRAM_CACHE",
        "FASTPLS_CV_FOLD_CROSSCOV_CACHE",
        "FASTPLS_CV_CLASS_SUM_CACHE"
    )
    previous <- Sys.getenv(variables, unset = NA_character_)
    value <- if (enabled) "1" else "0"
    do.call(Sys.setenv, as.list(stats::setNames(rep(value, 3L), variables)))
    function() {
        for (index in seq_along(variables)) {
            if (is.na(previous[[index]])) {
                Sys.unsetenv(variables[[index]])
            } else {
                do.call(Sys.setenv, stats::setNames(
                    list(previous[[index]]), variables[[index]]
                ))
            }
        }
    }
}

test_that("fold sufficient statistics preserve grouped SIMPLS CV", {
    set.seed(301)
    X <- matrix(rnorm(480L * 24L), 480L, 24L)
    signal <- X[, 1L] - 0.5 * X[, 2L] + 0.2 * X[, 3L]
    y <- factor(cut(signal, breaks = c(-Inf, -0.5, 0.5, Inf)))
    groups <- rep(seq_len(240L), each = 2L)

    run <- function(enabled, classifier) {
        restore <- cv_cache_state(enabled)
        on.exit(restore())
        pls.single.cv(
            X, y, constrain = groups, ncomp = c(10L, 20L), kfold = 5L,
            method = "simpls", backend = "cpu", classifier = classifier,
            seed = 17L, fit = FALSE
        )
    }

    for (classifier in c("argmax", "lda")) {
        ordinary <- run(FALSE, classifier)
        cached <- run(TRUE, classifier)
        expect_identical(cached$fold, ordinary$fold)
        expect_identical(cached$best_ncomp, ordinary$best_ncomp)
        expect_equal(cached$best_metric_value, ordinary$best_metric_value,
            tolerance = 1e-12)
        expect_identical(
            lapply(cached$pred, as.character),
            lapply(ordinary$pred, as.character)
        )
    }
})

test_that("fold sufficient statistics preserve multivariate regression CV", {
    set.seed(302)
    X <- matrix(rnorm(400L * 20L), 400L, 20L)
    coefficients <- matrix(rnorm(20L * 5L), 20L, 5L)
    Y <- X %*% coefficients + matrix(rnorm(400L * 5L, sd = 0.1), 400L, 5L)
    groups <- rep(seq_len(200L), each = 2L)

    run <- function(enabled, method, ncomp) {
        restore <- cv_cache_state(enabled)
        on.exit(restore())
        pls.single.cv(
            X, Y, constrain = groups, ncomp = ncomp, kfold = 5L,
            method = method, backend = "cpu", seed = 23L, fit = FALSE
        )
    }

    for (case in list(
        list(method = "simpls", ncomp = c(10L, 20L)),
        list(method = "kernelpls", ncomp = c(10L, 20L)),
        list(method = "plssvd", ncomp = c(2L, 4L))
    )) {
        ordinary <- run(FALSE, case$method, case$ncomp)
        cached <- run(TRUE, case$method, case$ncomp)
        expect_identical(cached$fold, ordinary$fold)
        expect_identical(cached$best_ncomp, ordinary$best_ncomp)
        expect_equal(cached$best_metric_value, ordinary$best_metric_value,
            tolerance = 1e-12)
        expect_equal(cached$Ypred, ordinary$Ypred, tolerance = 1e-10)
    }
})
