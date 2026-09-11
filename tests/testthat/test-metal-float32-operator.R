test_that("Metal float32 deflation updates the operation-split operator", {
    skip_if_not(has_metal())
    set.seed(421)
    X <- float::fl(matrix(rnorm(120 * 47), 120, 47))
    Y <- float::fl(matrix(rnorm(120 * 37), 120, 37))
    for (solver in "rsvd") {
        call_fit <- function(x, components) {
            suppressWarnings(pls(
                x, Y, Xtest = X[101:120, , drop = FALSE],
                ncomp = components, svd.method = solver, backend = "metal",
                oversample = 3, power = 2, fit = TRUE, seed = 51
            ))
        }
        original <- call_fit(X, c(2L, 4L))
        if (solver == "rsvd") {
            expect_identical(original$diagnostics$rsvd$backend, "metal")
            expect_identical(original$diagnostics$simpls_direction$rule,
                "fresh_oversampled_sketch_per_component")
        }
        changed <- call_fit(X * float::fl(2), 4L)
        repeated <- call_fit(X, c(2L, 4L))
        prefix <- call_fit(X, 2L)
        expect_identical(original$Ypred, repeated$Ypred)
        expect_equal(original$Ypred[[1L]], prefix$Ypred[[1L]], tolerance = 0)
        expect_false(isTRUE(all.equal(original$Ypred[[2L]], changed$Ypred[[1L]])))
        expect_true(all(vapply(original$Ypred, function(x) {
            all(is.finite(float::dbl(x)))
        }, logical(1))))
    }
})

test_that("float32 diagnostics report each massive backend route", {
    for (backend in c("cpu", "cuda", "metal")) {
        actual <- fastPLS:::.simpls_direction_diagnostics(
            TRUE, backend, training_samples = 1200L,
            predictor_dimension = 13000L, response_dimension = 28355L,
            requested_components = 165L, power = 2L, oversample = 12L,
            precision = "float32"
        )
        if (identical(backend, "cuda")) {
            expect_identical(actual$rule, "batched_cuda_candidate_block")
            expect_true(actual$candidate_block_refresh)
            expect_identical(actual$refresh_width, 8L)
        } else {
            expect_identical(
                actual$rule,
                paste0("fresh_", backend, "_rank_one_refresh")
            )
            expect_false(actual$candidate_block_refresh)
            expect_identical(actual$refresh_width, 1L)
        }
        expect_true(actual$fresh_start)
        expect_false(
            "conditional_crossproduct_cache" %in%
                actual$active_optimizations
        )
    }
})

test_that("Metal float32 class products reuse no stale labels", {
    skip_if_not(has_metal())
    set.seed(422)
    X <- float::fl(matrix(rnorm(140 * 49), 140, 49))
    y <- factor(rep(seq_len(7), 20))
    call_fit <- function(labels) {
        suppressWarnings(pls(X, labels, Xtest = X, ncomp = 1:4,
            backend = "metal", oversample = 2, power = 2, seed = 52))
    }
    original <- call_fit(y)
    changed <- call_fit(rev(y))
    repeated <- call_fit(y)
    expect_identical(original$Ypred, repeated$Ypred)
    expect_false(identical(original$Ypred, changed$Ypred))
})
