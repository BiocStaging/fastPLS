test_that("public PLS functions reject invalid component counts", {
    set.seed(811)
    X <- matrix(rnorm(60 * 8), 60, 8)
    y <- factor(rep(letters[1:3], each = 20))
    invalid <- list(0, -1, 1.5, NA_real_, Inf, numeric())

    for (value in invalid) {
        expect_error(
            pls(X, y, ncomp = value),
            "ncomp must contain positive integers"
        )
        expect_error(
            pls.single.cv(X, y, ncomp = value, kfold = 3),
            "ncomp must contain positive integers"
        )
        expect_error(
            pls.double.cv(
                X,
                y,
                ncomp = value,
                kfold_inner = 2,
                kfold_outer = 2
            ),
            "ncomp must contain positive integers"
        )
    }
})

test_that("integer-valued numeric component counts remain supported", {
    set.seed(812)
    X <- matrix(rnorm(60 * 8), 60, 8)
    y <- factor(rep(letters[1:3], each = 20))
    fit <- pls(X, y, ncomp = c(1, 2), return_variance = FALSE)

    expect_identical(
        as.integer(attr(fit, "fastPLS_internal")$ncomp),
        c(1L, 2L)
    )
})

test_that("cross-validation controls reject invalid integer values", {
    set.seed(813)
    X <- matrix(rnorm(60 * 8), 60, 8)
    y <- factor(rep(letters[1:3], each = 20))

    expect_error(
        pls.single.cv(X, y, ncomp = 1:2, kfold = 1),
        "kfold must contain one integer not smaller than 2"
    )
    expect_error(
        pls.single.cv(X, y, ncomp = 1:2, kfold = 2.5),
        "kfold must contain one integer"
    )
    expect_error(
        pls.double.cv(
            X, y, ncomp = 1:2, runn = 0,
            kfold_inner = 2, kfold_outer = 2
        ),
        "runn must contain one integer not smaller than 1"
    )
    expect_error(
        pls.double.cv(
            X, y, ncomp = 1:2, runn = 1.5,
            kfold_inner = 2, kfold_outer = 2
        ),
        "runn must contain one integer"
    )
})

test_that("family-specific integer controls are validated before fitting", {
    set.seed(814)
    X <- matrix(rnorm(60 * 8), 60, 8)
    y <- factor(rep(letters[1:3], each = 20))

    expect_error(
        pls(X, y, method = "opls", north = -1),
        "north must contain one integer not smaller than 0"
    )
    expect_error(
        pls(X, y, method = "opls", north = 1.5),
        "north must contain one integer"
    )
    expect_error(
        pls(X, as.numeric(y), method = "kernelpls", kernel = "poly",
            degree = 0),
        "degree must contain one integer not smaller than 1"
    )
})

test_that("response and predictor dimensions fail early and clearly", {
    set.seed(815)
    X <- matrix(rnorm(60 * 8), 60, 8)
    y <- factor(rep(letters[1:3], each = 20))

    expect_error(pls(X, y[-1]), "Ytrain must contain one row per")
    expect_error(pls(X, y, X[1:5, -1]), "same number of columns")
    expect_error(pls(X, y, Ytest = y[1:5]), "without Xtest")
    bad <- y
    bad[[1L]] <- NA
    expect_error(pls(X, bad), "missing class labels")
    expect_error(pls(X, factor(rep("a", 60))), "at least two observed")
})

test_that("character class labels follow factor classification paths", {
    set.seed(816)
    X <- matrix(rnorm(60 * 8), 60, 8)
    y <- rep(letters[1:3], each = 20)

    direct <- pls(
        X[-c(1, 21, 41), ], y[-c(1, 21, 41)],
        X[c(1, 21, 41), ], y[c(1, 21, 41)],
        ncomp = 1:2, classifier = "lda", return_variance = FALSE
    )
    expect_true(all(vapply(direct$Ypred, is.factor, logical(1L))))
    expect_true(all(is.finite(direct$accuracy)))

    single <- pls.single.cv(
        X, y, ncomp = 1:2, kfold = 3, fit = FALSE, seed = 1
    )
    expect_true(single$classification)
    expect_true(all(is.finite(single$accuracy)))

    double <- pls.double.cv(
        X, y, ncomp = 1:2, kfold_inner = 3, kfold_outer = 3,
        runn = 1, seed = 1
    )
    expect_true(is.factor(double$Ypred))
    expect_true(all(is.finite(double$accuracy)))

    unseen <- pls(
        X, y, X[1:2, ], c("a", "unseen"),
        ncomp = 1, return_variance = FALSE
    )
    expected <- mean(
        as.character(unseen$Ypred[[1L]]) == c("a", "unseen")
    )
    expect_equal(unname(unseen$accuracy), expected)
    expect_lte(unname(unseen$accuracy), 0.5)
})
