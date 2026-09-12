test_that("compiled Spearman preserves average ties and complete pairs", {
    set.seed(901)
    cases <- list(
        list(rnorm(9001), rnorm(9001)),
        list(sample(-9:9, 9001, TRUE), sample(-20:20, 9001, TRUE)),
        list(rep(c(-Inf, -1, -0, 0, 1, Inf, NA, NaN), 1100),
            rep(c(NA, Inf, 1, 0, -0, -1, -Inf, NaN), 1100)),
        list(seq_len(10001), rev(seq_len(10001))),
        list(rnorm(1800000), rnorm(1800000)),
        list(rep(1, 9000), seq_len(9000)),
        list(rep(c(1e300, 1e-300, -1e300, -1e-300), 2200),
            rep(c(2, 3, 3, 1), 2200))
    )
    for (values in cases) {
        x <- values[[1L]]
        y <- values[[2L]]
        expected <- suppressWarnings(cor(x, y, method = "spearman",
            use = "complete.obs"))
        expect_equal(fastPLS:::spearman_correlation_cpp(x, y), expected,
            tolerance = 1e-14)
    }
    expect_true(is.na(fastPLS:::spearman_correlation_cpp(1, 2)))
    expect_error(fastPLS:::spearman_correlation_cpp(c(NA, NaN), c(1, 2)),
        "no complete element pairs")
    expect_error(fastPLS:::spearman_correlation_cpp(1, 1:2), "equal length")
})

test_that("large regression metrics retain their statistical definitions", {
    set.seed(802)
    y <- matrix(rnorm(24000), 400, 60)
    prediction <- y + matrix(rnorm(24000, sd = 0.1), 400, 60)
    training <- y + 1
    for (bycol in c(FALSE, TRUE)) {
        actual <- evaluate(y, prediction, ytrain = training, bycol = bycol)
        expect_equal(actual$metrics$Spearman_r,
            cor(as.vector(y), as.vector(prediction), method = "spearman"),
            tolerance = 1e-14)
        expect_equal(actual$metrics$RMSD, sqrt(mean((y - prediction)^2)))
        if (bycol) {
            expected <- vapply(seq_len(ncol(y)), function(j) {
                cor(y[, j], prediction[, j], method = "spearman")
            }, numeric(1))
            expect_equal(actual$per_response$Spearman_r, expected)
        }
    }
})

test_that("CV metric outputs use the same complete-pair Spearman statistic", {
    set.seed(318)
    X <- matrix(rnorm(64 * 12), 64, 12)
    Y <- X %*% matrix(rnorm(12 * 80), 12, 80) +
        matrix(rnorm(64 * 80), 64, 80)
    for (solver in "rsvd") {
        single <- pls.single.cv(X, Y, ncomp = 1:2, kfold = 2,
            backend = "cpu", seed = 11)
        for (index in seq_along(single$ncomp)) {
            prediction <- fastPLS:::.fastpls_component_prediction(
                single$pred, index, single$ncomp, FALSE)
            expected <- cor(as.vector(Y), as.vector(as.matrix(prediction)),
                method = "spearman")
            expect_equal(single$metrics$cross_validated[[index]]$metrics$Spearman_r,
                expected, tolerance = 1e-14)
        }
        nested <- pls.double.cv(X, Y, ncomp = 1:2,
            kfold_inner = 2, kfold_outer = 2, runn = 1,
            backend = "cpu", seed = 11)
        expected <- cor(as.vector(Y), as.vector(as.matrix(nested$Ypred)),
            method = "spearman")
        expect_equal(nested$metrics$aggregate$metrics$Spearman_r, expected,
            tolerance = 1e-14)
    }
})
