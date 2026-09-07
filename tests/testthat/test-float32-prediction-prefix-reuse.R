test_that("float32 shared projection agrees with separate requested prefixes", {
    skip_on_os("windows")
    set.seed(933)
    X <- float::fl(matrix(rnorm(72L * 13L), 72L, 13L))
    response <- float::fl(matrix(rnorm(72L * 8L), 72L, 8L))
    components <- c(1L, 3L, 5L)
    restrict_prefix <- function(model, index) {
        model <- fastPLS:::.fastpls_restore_internal_output_fields(model)
        model$ncomp <- components[index]
        if (!is.null(model$W_latent)) model$W_latent <- model$W_latent[index]
        if (!is.null(model$inner_model)) {
            model$inner_model <- restrict_prefix(model$inner_model, index)
        }
        model
    }
    for (family in c("simpls", "plssvd", "opls", "kernelpls")) {
        fit <- suppressWarnings(pls(X[1:60, ], response[1:60, ],
            ncomp = components, method = family, seed = 33,
            return_variance = FALSE, backend = "cpu"))
        combined <- predict(fit, X[61:72, ], proj = TRUE)
        for (index in seq_along(components)) {
            single <- restrict_prefix(fit, index)
            predicted <- predict(single, X[61:72, ], proj = TRUE)
            actual <- float::dbl(combined$Ypred[[index]])
            expected <- float::dbl(predicted$Ypred[[1L]])
            error <- sqrt(sum((actual - expected)^2) /
                max(sum(expected^2), .Machine$double.eps))
            expect_true(error < 1e-5, info = paste(family, components[index], error))
        }
    }
})
