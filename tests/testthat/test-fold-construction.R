legacy_single_cv_folds <- function(Ydata, constrain, kfold, seed) {
    constrain <- as.integer(as.factor(constrain))
    groups <- sort(unique(constrain))
    group_fold <- integer(length(groups))
    names(group_fold) <- as.character(groups)
    fastPLS:::.fastpls_set_seed(seed)
    if (is.factor(Ydata) || is.character(Ydata)) {
        first_group_class <- vapply(
            groups,
            function(group) {
                as.character(Ydata[which(constrain == group)[1L]])
            },
            character(1L)
        )
        for (class in unique(first_group_class)) {
            index <- which(first_group_class == class)
            index <- sample(index, length(index))
            group_fold[index] <- (seq_along(index) - 1L) %% kfold
        }
    } else {
        index <- sample(seq_along(groups), length(groups))
        group_fold[index] <- (seq_along(index) - 1L) %% kfold
    }
    as.integer(group_fold[as.character(constrain)])
}

test_that("linear fold construction preserves legacy seeded assignments", {
    constrain <- rep(c("patient-c", "patient-a", "patient-d", "patient-b"),
        c(3L, 2L, 4L, 3L))
    response <- factor(rep(c("case", "control", "case", "control"),
        c(3L, 2L, 4L, 3L)))

    expected <- legacy_single_cv_folds(response, constrain, 3L, 19L)
    observed <- fastPLS:::.make_single_cv_folds(
        response, constrain, 3L, 19L
    )

    expect_identical(observed, expected)
    expect_true(all(vapply(split(observed, constrain), function(value) {
        length(unique(value)) == 1L
    }, logical(1L))))
})

test_that("linear fold construction preserves unstratified assignments", {
    constrain <- rep(c(10L, 30L, 20L, 40L), each = 3L)
    response <- matrix(seq_along(constrain), ncol = 1L)

    expect_identical(
        fastPLS:::.make_single_cv_folds(response, constrain, 3L, 31L),
        legacy_single_cv_folds(response, constrain, 3L, 31L)
    )
})
