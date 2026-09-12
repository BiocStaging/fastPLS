test_that("native fastcor preserves the former matrix calculation", {
    set.seed(1201)
    a <- matrix(rnorm(84), 12, 7)
    b <- matrix(rnorm(84), 12, 7)

    reference <- function(left, right = NULL, byrow = TRUE, diagonal = TRUE) {
        if (!byrow) left <- t(left)
        left <- left - rowMeans(left)
        left <- left / sqrt(rowSums(left * left))
        if (is.null(right)) return(tcrossprod(left))
        if (!byrow) right <- t(right)
        right <- right - rowMeans(right)
        right <- right / sqrt(rowSums(right * right))
        if (diagonal) rowSums(left * right) else tcrossprod(left, right)
    }

    expect_equal(fastcor(a), reference(a), tolerance = 1e-12)
    expect_equal(
        fastcor(a, b, diag = TRUE),
        reference(a, b, diagonal = TRUE),
        tolerance = 1e-12
    )
    expect_equal(
        fastcor(a, b, diag = FALSE),
        reference(a, b, diagonal = FALSE),
        tolerance = 1e-12
    )
    expect_equal(
        fastcor(a, byrow = FALSE),
        reference(a, byrow = FALSE),
        tolerance = 1e-12
    )

    rownames(a) <- paste0("sample", seq_len(nrow(a)))
    expect_identical(rownames(fastcor(a)), rownames(a))
})

test_that("native VIP preserves component-wise trajectories", {
    set.seed(1202)
    x <- matrix(rnorm(120 * 9), 120, 9)
    y <- cbind(x[, 1] - x[, 3], x[, 2] + 0.5 * x[, 4])
    model <- pls(x, y, ncomp = 1:4, fit = TRUE)

    reference <- function(response) {
        score_ss <- colSums(model$Ttrain^2)
        explained <- model$Q[response, ]^2 * score_ss
        weight_ss <- colSums(model$R^2)
        weighted <- sweep(model$R^2, 2, explained / weight_ss, "*")
        sqrt(nrow(weighted) * apply(weighted, 1, cumsum) /
            cumsum(explained))
    }

    observed <- ViP(model)
    expect_length(observed, 2L)
    expect_equal(observed[[1]], reference(1), tolerance = 1e-12)
    expect_equal(observed[[2]], reference(2), tolerance = 1e-12)
})

test_that("native evaluation handles missing and response-wise values", {
    observed <- cbind(a = c(1, 2, NA, 4), b = c(10, 12, 14, 16))
    predicted <- cbind(a = c(1.1, 1.9, 3, 4.2), b = c(9, 13, 14, 15))
    training <- cbind(a = 0:4, b = seq(8, 16, 2))

    result <- evaluate(observed, predicted, ytrain = training, bycol = TRUE)
    expect_equal(result$metrics$n, 7)
    expect_equal(result$per_response$n, c(3, 4))
    expect_true(all(is.finite(result$per_response$RMSD)))
    expect_equal(result$per_response$response, c("a", "b"))
})

test_that("float32 CV returns the native aggregate regression metrics", {
    set.seed(1203)
    x_double <- matrix(rnorm(120L * 12L), 120L, 12L)
    y_double <- x_double %*% matrix(rnorm(12L * 4L), 12L, 4L) +
        matrix(rnorm(120L * 4L, sd = 0.1), 120L, 4L)
    x <- float::fl(x_double)
    y <- float::fl(y_double)
    fit <- pls.single.cv(
        x, y, ncomp = c(2L, 4L), kfold = 5L,
        method = "plssvd", backend = "cpu", fit = FALSE,
        bycol = FALSE, seed = 7L
    )
    observed <- float::dbl(y)
    for (index in seq_along(fit$ncomp)) {
        reference <- evaluate(
            observed,
            fit$pred[, , index],
            ytrain = observed,
            bycol = FALSE
        )$metrics
        reference$Q2 <- fit$Q2Y[[index]]
        expect_equal(
            fit$metrics$cross_validated[[index]]$metrics,
            reference,
            tolerance = 1e-12
        )
    }
    expect_null(fit$native_evaluation)
})
