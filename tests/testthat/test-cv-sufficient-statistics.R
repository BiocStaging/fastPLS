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

test_that("fold sufficient statistics preserve grouped PLS-SVD CV", {
    set.seed(304)
    X <- matrix(rnorm(600L * 24L), 600L, 24L)
    y <- factor(rep(seq_len(30L), each = 20L))
    groups <- rep(seq_len(300L), each = 2L)

    run <- function(enabled, classifier) {
        restore <- cv_cache_state(enabled)
        on.exit(restore())
        pls.single.cv(
            X, y, constrain = groups, ncomp = c(10L, 20L), kfold = 5L,
            method = "plssvd", backend = "cpu", classifier = classifier,
            seed = 29L, fit = FALSE
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

test_that("compiled implicit CV preserves the public large-response path", {
    skip_on_os("windows")
    set.seed(303)
    observations <- 12L
    predictors <- 9000L
    responses <- 15000L
    latent <- matrix(rnorm(observations * 2L), observations, 2L)
    X <- float::fl(latent %*% matrix(rnorm(2L * predictors), 2L, predictors))
    Y <- float::fl(latent %*% matrix(rnorm(2L * responses), 2L, responses))

    common <- list(
        Xdata = X,
        Ydata = Y,
        ncomp = c(1L, 2L),
        kfold = 2L,
        scaling = "centering",
        svd.method = "rsvd",
        seed = 19L,
        rsvd_oversample = 12L,
        rsvd_power = 1L,
        store_predictions = TRUE,
        selection_metric = "rmsd"
    )

    for (method in c("plssvd", "simpls")) {
        compiled <- suppressWarnings(do.call(
            fastPLS:::.pls_cv_compiled,
            c(common, list(method = method, backend = "cpp"))
        ))
        fold_local <- suppressWarnings(do.call(
            fastPLS:::.pls_cv_via_pls,
            c(common, list(method = method, backend = "cpu"))
        ))

        expect_true(compiled$xprod, info = method)
        expect_identical(compiled$fold, fold_local$fold, info = method)
        expect_lte(
            max(abs(
                compiled$metrics$metric_value -
                    fold_local$metrics$metric_value
            )),
            5e-7
        )
        expect_equal(
            compiled$Ypred,
            fold_local$Ypred,
            tolerance = 2e-6,
            info = method
        )
    }
})

test_that("Apple sample-Gram centering preserves wide-response SIMPLS CV", {
    skip_if_not(identical(Sys.info()[["sysname"]], "Darwin"))
    set.seed(305)
    observations <- 20L
    predictors <- 9000L
    responses <- 15000L
    latent <- matrix(rnorm(observations * 6L), observations, 6L)
    X <- float::fl(
        latent %*% matrix(rnorm(6L * predictors), 6L, predictors)
    )
    Y <- float::fl(
        latent %*% matrix(rnorm(6L * responses), 6L, responses)
    )

    run <- function(enabled) {
        previous <- Sys.getenv(
            "FASTPLS_CV_SAMPLE_RESPONSE_GRAM", unset = NA_character_
        )
        on.exit({
            if (is.na(previous)) {
                Sys.unsetenv("FASTPLS_CV_SAMPLE_RESPONSE_GRAM")
            } else {
                Sys.setenv(FASTPLS_CV_SAMPLE_RESPONSE_GRAM = previous)
            }
        })
        Sys.setenv(
            FASTPLS_CV_SAMPLE_RESPONSE_GRAM = if (enabled) "1" else "0"
        )
        pls.single.cv(
            X, Y, ncomp = c(3L, 5L), kfold = 2L,
            method = "simpls", backend = "cpu", scaling = "centering",
            svd.method = "rsvd", rsvd_oversample = 12L,
            rsvd_power = 2L, seed = 31L, fit = FALSE
        )
    }

    implicit <- run(FALSE)
    cached <- run(TRUE)
    expect_identical(cached$fold, implicit$fold)
    expect_identical(cached$best_ncomp, implicit$best_ncomp)
    expect_lte(
        max(abs(cached$RMSD - implicit$RMSD)),
        5e-7
    )
    expect_equal(cached$Ypred, implicit$Ypred, tolerance = 3e-6)
})
