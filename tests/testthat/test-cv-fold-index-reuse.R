ordered_draw_reference <- function(values) {
    result <- integer(length(values))
    for (i in seq_along(result)) {
        position <- floor(runif(1) * length(values)) + 1L
        result[[i]] <- values[[position]]
        values <- values[-position]
    }
    result
}

legacy_fold_reference <- function(groups, labels, kfold) {
    levels <- sort(unique(groups))
    mapped <- match(groups, levels)
    count <- length(levels)
    if (kfold < 0 || kfold >= count) return(as.integer(mapped))
    group_fold <- integer(count)
    if (!is.null(labels)) {
        first_label <- labels[match(levels, groups)]
        for (label in seq_len(max(labels))) {
            indices <- which(first_label == label)
            if (!length(indices)) next
            order <- ordered_draw_reference(seq_along(indices))
            group_fold[indices[order]] <- (seq_along(indices) - 1L) %% kfold
        }
    } else {
        group_fold <- (ordered_draw_reference(seq_len(count)) - 1L) %% kfold
    }
    as.integer(group_fold[mapped] + 1L)
}

test_that("compiled CV preserves ordered fold draws and input ownership", {
    set.seed(75)
    X <- matrix(rnorm(72 * 8), 72, 8)
    labels <- rep(1:3, length.out = nrow(X))
    Yreg <- cbind(X[, 1], X[, 3])
    groupings <- list(
        seq_len(nrow(X)),
        rep(c(91L, -4L, 1024L, 6L, 2L, 14L, 88L, 17L, 30L), 8L)
    )
    for (groups in groupings) {
        for (classification in c(FALSE, TRUE)) {
            Y <- if (classification) matrix(as.double(labels), ncol = 1) else Yreg
            original <- serialize(list(X, Y), NULL)
            for (fold_count in c(3L, 5L, -1L)) {
                for (seed in c(7L, 912L)) {
                    set.seed(seed)
                    expected <- legacy_fold_reference(
                        groups, if (classification) labels else NULL, fold_count
                    )
                    set.seed(seed)
                    core_fold <- fastPLS:::cv_folds_core_cpp(
                        groups = groups,
                        labels = if (classification) labels else NULL,
                        class_count = if (classification) 3L else 0L,
                        folds = fold_count
                    )
                    expect_identical(as.integer(core_fold), expected)
                    set.seed(seed)
                    result <- fastPLS:::pls_cv_predict_compiled(
                        Xdata = X, Ydata = Y, constrain = groups, ncomp = 1:2,
                        scaling = 1L, kfold = fold_count, method = 3L,
                        backend = 0L,
                        svd_method = fastPLS:::.svd_method_id("cpu_rsvd"),
                        rsvd_oversample = 32L, rsvd_power = 5L,
                        svds_tol = 0, seed = seed,
                        classification = classification,
                        n_response = if (classification) 3L else 2L,
                        xprod = FALSE, opls_north = 0L, return_scores = TRUE,
                        class_codes = matrix(numeric(), 0, 0), classifier = 0L,
                        lda_ridge = 0, store_predictions = TRUE, metric_id = 4L
                    )
                    expect_identical(as.integer(result$fold), expected)
                    expect_identical(serialize(list(X, Y), NULL), original)
                    expect_true(all(is.finite(result$Ypred)))
                }
            }
        }
    }
})
