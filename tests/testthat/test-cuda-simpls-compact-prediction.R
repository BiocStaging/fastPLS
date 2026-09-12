test_that("resident CUDA SIMPLS remains compact across storage hints", {
    skip_if_not(has_cuda(), "CUDA backend not available")
    old <- Sys.getenv("FASTPLS_STORE_B", unset = NA_character_)
    on.exit({
        if (is.na(old)) {
            Sys.unsetenv("FASTPLS_STORE_B")
        } else {
            Sys.setenv(FASTPLS_STORE_B = old)
        }
    }, add = TRUE)

    set.seed(902)
    X <- matrix(rnorm(120 * 24), 120, 24)
    Y <- matrix(rnorm(120 * 5), 120, 5)
    Xtest <- matrix(rnorm(20 * 24), 20, 24)
    fit_once <- function(storage) {
        Sys.setenv(FASTPLS_STORE_B = storage)
        pls(
            X, Y, ncomp = 1:4, method = "simpls", backend = "cuda", seed = 902, fit = TRUE,
            return_variance = FALSE
        )
    }

    dense <- fit_once("always")
    compact <- fit_once("never")
    dense_again <- fit_once("always")
    expect_null(dense$B)
    expect_null(compact$B)
    expect_equal(compact$R, dense$R, tolerance = 0)
    expect_equal(compact$Q, dense$Q, tolerance = 0)
    expect_equal(compact$Yfit, dense$Yfit, tolerance = 0)
    expect_null(dense_again$B)
    expect_equal(
        predict(compact, Xtest, backend = "cuda")$Ypred,
        predict(dense, Xtest, backend = "cuda")$Ypred,
        tolerance = 1e-8
    )
})
