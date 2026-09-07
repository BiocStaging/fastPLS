.resident_metal_input <- function(x, name) {
    value <- .as_float32_matrix(x, name)
    bits <- value@Data
    if (is.null(dim(bits))) {
        bits <- matrix(bits, nrow = nrow(value), ncol = ncol(value))
    }
    bits
}

.resident_metal_output <- function(x) {
    .float32_from_bits(x)
}

.resident_metal_summary <- function(x) {
    float::dbl(.resident_metal_output(x))
}

.resident_metal_cv_classification_path <- function(object, newdata) {
    .fastpls_require_backend_available("metal", "Resident Metal CV prediction")
    x <- .resident_metal_input(newdata, "newdata")
    ncomp <- as.integer(object$ncomp)
    effective <- sort(unique(ncomp))
    path_index <- match(ncomp, effective)
    lda <- as.integer(.is_lda_classifier(object$classification_rule))
    combined <- metal_resident_classify_response_path_cpp(
        object$resident_state, x, effective, 1L, lda
    )
    ranked <- combined$labels[, , path_index, drop = FALSE]
    component_names <- paste0("ncomp=", ncomp)
    predicted <- lapply(seq_along(ncomp), function(index) {
        factor(object$lev[ranked[, 1L, index]], levels = object$lev)
    })
    names(predicted) <- component_names
    responses_unique <- lapply(seq_along(effective), function(index) {
        .resident_metal_output(
            combined$predictions[, , index, drop = TRUE]
        )
    })
    responses <- responses_unique[path_index]
    names(responses) <- component_names
    list(
        Ypred = as.data.frame(predicted, check.names = FALSE),
        Ypred_scores = responses
    )
}

.resident_metal_predict <- function(object, newdata, Ytest = NULL,
    proj = FALSE, top = 1L, raw_scores = FALSE,
    include_classes = TRUE) {
    .fastpls_require_backend_available("metal", "Resident Metal prediction")
    lda <- as.integer(
        isTRUE(object$classification) &&
            .is_lda_classifier(object$classification_rule)
    )
    x <- .resident_metal_input(newdata, "newdata")
    state <- object$resident_state
    ncomp <- as.integer(object$ncomp)
    component_names <- paste0("ncomp=", ncomp)
    classification <- isTRUE(object$classification)
    result <- list()
    if (classification && isTRUE(include_classes)) {
        top <- min(as.integer(top), length(object$lev))
        ranked <- metal_resident_classify_path_cpp(
            state, x, ncomp, top, lda
        )
        predicted <- lapply(seq_along(ncomp), function(index) {
            factor(object$lev[ranked[, 1L, index]], levels = object$lev)
        })
        names(predicted) <- component_names
        result$Ypred <- as.data.frame(predicted, check.names = FALSE)
        result$Ypred_index <- matrix(
            ranked[, 1L, ], nrow(x), length(ncomp),
            dimnames = list(NULL, component_names)
        )
        if (top > 1L) {
            result$Ypred_top <- stats::setNames(
                lapply(seq_along(ncomp), function(index) {
                    matrix(
                        object$lev[ranked[, , index]], nrow(x), top,
                        dimnames = list(NULL, paste0("rank", seq_len(top)))
                    )
                }),
                component_names
            )
        }
    }
    if (!classification || raw_scores) {
        effective_path <- sort(unique(ncomp))
        path_index <- match(ncomp, effective_path)
        path <- metal_resident_predict_path_cpp(
            state, x, effective_path, lda
        )
        responses_unique <- lapply(seq_along(effective_path), function(index) {
            .resident_metal_output(path[, , index, drop = TRUE])
        })
        responses <- responses_unique[path_index]
        names(responses) <- component_names
        result[[if (!classification) {
            "Ypred"
        } else if (lda == 1L) {
            "LDA_scores"
        } else {
            "Ypred_scores"
        }]] <- responses
    }
    if (proj) {
        result$Ttest <- .resident_metal_output(
            metal_resident_project_cpp(state, x, max(ncomp))
        )
    }
    if (!is.null(Ytest)) {
        labels <- if (classification) match(as.character(Ytest), object$lev) else NULL
        if (classification && anyNA(labels)) {
            stop("Ytest contains unknown class labels.", call. = FALSE)
        }
        response <- if (classification) {
            NULL
        } else {
            .resident_metal_input(Ytest, "Ytest")
        }
        sums <- lapply(ncomp, function(component) {
            .resident_metal_summary(
                metal_resident_response_sums_cpp(
                    state, x, response, labels, component
                )
            )
        })
        result$Q2Y <- vapply(sums, function(value) {
            denominator <- sum(value[2L, ])
            if (denominator > 0) {
                1 - sum(value[1L, ]) / denominator
            } else {
                NA_real_
            }
        }, numeric(1))
        if (classification) {
            result$accuracy <- vapply(
                result$Ypred,
                function(value) mean(value == Ytest),
                numeric(1)
            )
            if (top > 1L) {
                result$top_k_accuracy <- vapply(
                    result$Ypred_top,
                    function(value) {
                        mean(rowSums(value == as.character(Ytest)) > 0L)
                    },
                    numeric(1)
                )
            }
        }
    }
    .fastpls_public_predict_output(result, ncomp)
}

.pls_fit_resident_metal <- function(context, config) {
    .fastpls_require_backend_available("metal", "Resident Metal fitting")
    if (!isTRUE(context$float32)) {
        stop(
            "Apple Metal does not provide native float64 arithmetic. Use float32 input or another backend; no CPU fallback is performed.",
            call. = FALSE
        )
    }
    if (!context$method %in% c("simpls", "plssvd", "opls", "kernelpls")) {
        stop(
            "The requested method is unavailable in the resident Metal implementation. No hybrid or CPU fallback is performed.",
            call. = FALSE
        )
    }
    kernel <- config$kernel %||% "linear"
    if (identical(context$method, "kernelpls") && identical(kernel, "linear")) {
        stop(
            "Use method = 'simpls' for a linear Metal model; resident Metal kernel PLS implements nonlinear RBF and polynomial kernels.",
            call. = FALSE
        )
    }
    if (isTRUE(config$perm.test)) {
        stop(
            "Permutation testing is not yet connected to resident Metal fitting. No CPU fallback is performed.",
            call. = FALSE
        )
    }
    control <- context$control
    x <- .resident_metal_input(context$Xtrain, "Xtrain")
    classification <- isTRUE(context$classification)
    labels <- NULL
    levels <- NULL
    y <- NULL
    if (classification) {
        response <- droplevels(as.factor(context$Ytrain))
        levels <- levels(response)
        labels <- as.integer(response)
        responses <- length(levels)
    } else {
        y <- .resident_metal_input(context$Ytrain, "Ytrain")
        responses <- ncol(y)
    }
    ncomp <- as.integer(config$ncomp)
    if (identical(context$method, "opls")) {
        .opls_require_predictive_rank(
            ncomp,
            x,
            as.integer(config$north %||% 1L),
            context$scal != 3L
        )
    }
    if (identical(context$method, "plssvd")) {
        ncomp <- .cap_plssvd_ncomp(
            ncomp, nrow(x) - 1L, ncol(x), responses,
            factor_response = classification
        )$ncomp
    }
    method_code <- switch(
        context$method,
        plssvd = 1L,
        simpls = 3L,
        opls = 4L,
        kernelpls = 5L
    )
    kernel_code <- .kernel_pls_kernel_id(kernel)
    gamma <- if (identical(context$method, "kernelpls")) {
        .kernel_pls_gamma(config$gamma, x)
    } else {
        1
    }
    state <- metal_resident_simpls_fit_cpp(
        x, y, labels, responses, max(ncomp), context$scal,
        control$rsvd_oversample, control$rsvd_power, control$seed,
        method_code, as.integer(config$north %||% 1L), kernel_code, gamma,
        as.integer(config$degree %||% 3L), as.numeric(config$coef0 %||% 1)
    )
    fields <- list(
        R = .resident_metal_output(metal_resident_export_cpp(state, 0L)),
        Q = .resident_metal_output(metal_resident_export_cpp(state, 1L)),
        mX = .resident_metal_output(metal_resident_export_cpp(state, 3L)),
        vX = .resident_metal_output(metal_resident_export_cpp(state, 4L)),
        mY = .resident_metal_output(metal_resident_export_cpp(state, 5L))
    )
    if (isTRUE(config$fit)) {
        fields$Ttrain <- .resident_metal_output(
            metal_resident_export_cpp(state, 2L)
        )
    }
    if (isTRUE(config$return_loadings)) {
        fields$P <- .resident_metal_output(
            metal_resident_export_cpp(state, 6L)
        )
    } else {
        fields$P <- matrix(numeric(), 0L, 0L)
    }
    model <- c(fields, list(
        resident_state = state,
        resident_backend = "metal",
        gpu_resident = TRUE,
        ncomp = ncomp,
        p = ncol(x),
        m = responses,
        lev = levels,
        precision = "float32",
        classification = classification,
        classification_rule = context$classifier,
        pls_method = context$method,
        north = if (identical(context$method, "opls")) {
            as.integer(config$north %||% 1L)
        } else NULL,
        kernel = if (identical(context$method, "kernelpls")) kernel else NULL,
        gamma = if (identical(context$method, "kernelpls")) gamma else NULL,
        degree = if (identical(kernel, "poly")) {
            as.integer(config$degree %||% 3L)
        } else NULL,
        coef0 = if (identical(kernel, "poly")) {
            as.numeric(config$coef0 %||% 1)
        } else NULL,
        predict_backend = "metal_resident",
        xprod_mode = if (isTRUE(state$implicit_crosscovariance)) {
            "metal_implicit_resident"
        } else {
            "metal_explicit_resident"
        },
        B_stored = FALSE,
        compact_prediction = TRUE,
        predict_latent_ok = TRUE,
        R2Y = rep(NA_real_, length(ncomp))
    ))
    model$resident_controls <- list(
        requested_oversample = control$rsvd_oversample,
        requested_power = control$rsvd_power,
        effective_oversample = state$effective_oversample,
        effective_power = state$effective_power,
        refresh_block = state$refresh_block,
        implicit_crosscovariance = isTRUE(state$implicit_crosscovariance),
        predictor_crossprod_cache = isTRUE(state$predictor_crossprod_cache),
        host_assisted_components = isTRUE(state$host_assisted_components)
    )
    model$execution_route <- "resident Metal"
    if (isTRUE(config$return_variance)) {
        sums <- as.vector(.resident_metal_summary(
            metal_resident_export_cpp(state, 7L)
        ))
        total <- tail(sums, 1L)
        component_sums <- head(sums, -1L)
        if (is.finite(total) && total > 0) {
            model$variance <- .fastpls_named_components(
                component_sums / max(1L, nrow(x) - 1L), "LV"
            )
            model$variance_explained <- .fastpls_named_components(
                component_sums / total, "LV"
            )
            model$cumulative_variance_explained <- .fastpls_named_components(
                cumsum(component_sums) / total, "LV"
            )
            model$x_variance <- model$variance
            model$x_variance_explained <- model$variance_explained
            model$x_cumulative_variance_explained <-
                model$cumulative_variance_explained
            model$x_variance_total <- total / max(1L, nrow(x) - 1L)
        }
    }
    metal_resident_compact_cpp(
        state,
        prepare_lda = classification &&
            .is_lda_classifier(context$classifier)
    )
    class(model) <- "fastPLS"
    if (isTRUE(config$fit)) {
        fitted <- .resident_metal_predict(
            model, context$Xtrain, context$Ytrain
        )
        model$Yfit <- fitted$Ypred
        model$R2Y <- fitted$Q2Y
    }
    if (!is.null(context$Xtest)) {
        predicted <- .resident_metal_predict(
            model, context$Xtest, context$Ytest, proj = config$proj
        )
        model[names(predicted)] <- predicted
    }
    model
}
