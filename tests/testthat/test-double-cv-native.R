test_that("compiled nested classification covers every family and head", {
    index <- c(seq_len(12), 51:62, 101:112)
    X <- as.matrix(iris[index, seq_len(4)])
    y <- droplevels(iris$Species[index])

    for (method in c("plssvd", "simpls", "opls", "kernelpls")) {
        for (classifier in c("argmax", "lda")) {
            arguments <- list(
                Xdata = X,
                Ydata = y,
                ncomp = 1:2,
                kfold_inner = 3,
                kfold_outer = 3,
                runn = 2,
                seed = 11,
                method = method,
                backend = "cpu",
                classifier = classifier,
                kernel = if (method == "kernelpls") "rbf" else "linear"
            )
            native <- do.call(pls.double.cv, arguments)
            expect_type(native, "list")
            expect_length(native$results, 2L)
            expect_length(native$Ypred, nrow(X))
            expect_true(all(native$results[[1L]]$best_ncomp %in% 1:2))
            expect_true(all(is.finite(native$accuracy)))
            expect_true(all(is.finite(native$balanced_accuracy)))
            expect_true(all(is.finite(native$Q2Y)))
            expect_true(all(is.finite(native$R2Y)))
        }
    }
})

test_that("compiled nested regression covers every family", {
    X <- as.matrix(mtcars[, c("disp", "hp", "wt")])
    Y <- cbind(mpg = mtcars$mpg, qsec = mtcars$qsec)

    for (method in c("plssvd", "simpls", "opls", "kernelpls")) {
        arguments <- list(
            Xdata = X,
            Ydata = Y,
            ncomp = 1:2,
            kfold_inner = 3,
            kfold_outer = 3,
            runn = 2,
            seed = 17,
            method = method,
            backend = "cpu",
            kernel = if (method == "kernelpls") "rbf" else "linear"
        )
        native <- do.call(pls.double.cv, arguments)
        expect_type(native, "list")
        expect_equal(dim(native$Ypred), dim(Y))
        expect_true(all(native$results[[1L]]$best_ncomp %in% 1:2))
        expect_true(all(is.finite(native$Q2Y)))
        expect_true(all(is.finite(native$R2Y)))
        expect_true(all(is.finite(native$RMSD)))
    }
})

test_that("compiled nested CV preserves float32 dispatch", {
    index <- c(seq_len(10), 51:60, 101:110)
    X <- scale(as.matrix(iris[index, seq_len(4)]))
    y <- droplevels(iris$Species[index])
    arguments <- list(
        Ydata = y,
        ncomp = 1:2,
        kfold_inner = 3,
        kfold_outer = 3,
        runn = 1,
        seed = 23,
        method = "simpls",
        backend = "cpu",
        classifier = "lda"
    )
    double <- do.call(pls.double.cv, c(list(Xdata = X), arguments))
    single <- do.call(
        pls.double.cv,
        c(list(Xdata = float::fl(X)), arguments)
    )

    expect_identical(single$bcomp, double$bcomp)
    expect_identical(single$Ypred, double$Ypred)
    expect_equal(single$Q2Y, double$Q2Y, tolerance = 1e-4)
    expect_equal(single$R2Y, double$R2Y, tolerance = 1e-4)
})

test_that("nested CV does not expose the retired xprod control", {
    expect_false("xprod" %in% names(formals(pls.double.cv)))
    index <- c(seq_len(10), 51:60, 101:110)
    expect_error(
        pls.double.cv(
            as.matrix(iris[index, seq_len(4)]),
            droplevels(iris$Species[index]),
            ncomp = 1,
            kfold_inner = 2,
            kfold_outer = 2,
            xprod = FALSE
        ),
        "Unknown entry"
    )
})
