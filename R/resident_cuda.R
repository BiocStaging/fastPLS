.resident_cuda_input <- function(x, precision, name) {
    if (identical(precision, "float32")) {
        return(.as_float32_matrix(x, name)@Data)
    }
    x <- as.matrix(x)
    storage.mode(x) <- "double"
    x
}

.resident_cuda_output <- function(x, precision) {
    if (identical(precision, "float32")) .float32_from_bits(x) else x
}

.resident_cuda_summary <- function(x, precision) {
    if (identical(precision, "float32")) float::dbl(.float32_from_bits(x)) else x
}

.resident_cuda_cv_classification_path <- function(object, newdata) {
    .fastpls_require_backend_available("cuda", "Resident CUDA CV prediction")
    precision <- object$precision
    x <- .resident_cuda_input(newdata, precision, "newdata")
    ncomp <- as.integer(object$ncomp)
    effective <- sort(unique(ncomp))
    path_index <- match(ncomp, effective)
    lda <- as.integer(.is_lda_classifier(object$classification_rule))
    combined <- cuda_resident_classify_response_path_cpp(
        object$resident_state, x, effective, lda, 1L
    )
    ranked <- combined$labels[, , path_index, drop = FALSE]
    component_names <- paste0("ncomp=", ncomp)
    predicted <- lapply(seq_along(ncomp), function(index) {
        factor(object$lev[ranked[, 1L, index]], levels = object$lev)
    })
    names(predicted) <- component_names
    responses_unique <- lapply(seq_along(effective), function(index) {
        .resident_cuda_output(
            combined$predictions[, , index, drop = TRUE], precision
        )
    })
    responses <- responses_unique[path_index]
    names(responses) <- component_names
    if (identical(precision, "double")) {
        responses <- array(
            unlist(responses, use.names = FALSE),
            c(nrow(x), object$m, length(ncomp)),
            dimnames = list(NULL, object$lev, component_names)
        )
    }
    list(
        Ypred = as.data.frame(predicted, check.names = FALSE),
        Ypred_scores = responses
    )
}

.resident_cuda_predict <- function(object, newdata, Ytest = NULL,
    proj = FALSE, top = 1L, raw_scores = FALSE,
    include_classes = TRUE) {
    .fastpls_require_backend_available("cuda", "Resident CUDA prediction")
    precision <- object$precision
    x <- .resident_cuda_input(newdata, precision, "newdata")
    state <- object$resident_state
    ncomp <- as.integer(object$ncomp)
    component_names <- paste0("ncomp=", ncomp)
    classification <- isTRUE(object$classification)
    lda <- as.integer(classification && .is_lda_classifier(object$classification_rule))
    top <- min(as.integer(top), if (classification) length(object$lev) else 1L)
    result <- list()
    if (classification && isTRUE(include_classes)) {
        # Rank-limited PLS-SVD folds can map several requested prefixes to the
        # same fitted component. Evaluate each effective prefix once, then
        # restore the requested path without sending duplicates to CUDA.
        effective_path <- sort(unique(ncomp))
        path_index <- match(ncomp, effective_path)
        ranked_unique <- cuda_resident_classify_path_cpp(
            state, x, effective_path, lda, top
        )
        ranked <- ranked_unique[, , path_index, drop = FALSE]
        # These operations only label returned class indices, not model scores.
        predicted <- lapply(seq_along(ncomp), function(j) {
            factor(object$lev[ranked[, 1L, j]], levels = object$lev)
        })
        names(predicted) <- component_names
        result$Ypred <- as.data.frame(predicted, check.names = FALSE)
        result$Ypred_index <- matrix(ranked[, 1L, ], nrow(x), length(ncomp),
            dimnames = list(NULL, component_names))
        if (top > 1L) {
            result$Ypred_top <- stats::setNames(lapply(seq_along(ncomp), function(j) {
                matrix(object$lev[ranked[, , j]], nrow(x), top,
                    dimnames = list(NULL, paste0("rank", seq_len(top))))
            }), component_names)
        }
    }
    if (!classification || raw_scores) {
        effective_path <- sort(unique(ncomp))
        path_index <- match(ncomp, effective_path)
        path <- cuda_resident_predict_path_cpp(
            state, x, effective_path, lda
        )
        responses_unique <- lapply(seq_along(effective_path), function(index) {
            .resident_cuda_output(path[, , index, drop = TRUE], precision)
        })
        responses <- responses_unique[path_index]
        names(responses) <- component_names
        if (identical(precision, "double")) {
            responses <- array(unlist(responses, use.names = FALSE),
                c(nrow(x), object$m, length(ncomp)),
                dimnames = list(NULL, if (classification) object$lev else NULL, component_names))
        }
        result[[if (!classification) "Ypred" else if (lda == 1L) "LDA_scores" else "Ypred_scores"]] <- responses
    }
    if (proj) {
        result$Ttest <- .resident_cuda_output(
            cuda_resident_project_cpp(state, x, max(ncomp)), precision)
    }
    if (!is.null(Ytest)) {
        labels <- if (classification) match(as.character(Ytest), object$lev) else NULL
        if (classification && anyNA(labels)) stop("Ytest contains unknown class labels.", call. = FALSE)
        response <- if (classification) NULL else .resident_cuda_input(Ytest, precision, "Ytest")
        sums <- lapply(ncomp, function(a) {
            .resident_cuda_summary(cuda_resident_response_sums_cpp(state, x,
                response, labels, a), precision)
        })
        result$Q2Y <- vapply(sums, function(s) {
            denominator <- sum(s[2L, ])
            if (denominator > 0) 1 - sum(s[1L, ]) / denominator else NA_real_
        }, numeric(1))
        if (classification) {
            result$accuracy <- vapply(result$Ypred, function(y) mean(y == Ytest), numeric(1))
            if (top > 1L) {
                result$top_k_accuracy <- vapply(result$Ypred_top,
                    function(y) mean(rowSums(y == as.character(Ytest)) > 0L), numeric(1))
            }
        }
    }
    .fastpls_public_predict_output(result, ncomp)
}

.pls_fit_resident_cuda <- function(context, config) {
    .fastpls_require_backend_available("cuda", "Resident CUDA fitting")
    if (isTRUE(config$perm.test)) {
        stop("Permutation testing is not yet connected to resident CUDA fitting. No CPU fallback is performed.", call. = FALSE)
    }
    ctl <- context$control
    precision <- if (context$float32) "float32" else "double"
    x <- .resident_cuda_input(context$Xtrain, precision, "Xtrain")
    classification <- isTRUE(context$classification)
    labels <- NULL
    levels <- NULL
    y <- NULL
    if (classification) {
        response <- droplevels(as.factor(context$Ytrain))
        levels <- levels(response)
        labels <- as.integer(response)
        q <- length(levels)
    } else {
        y <- .resident_cuda_input(context$Ytrain, precision, "Ytrain")
        q <- ncol(y)
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
        ncomp <- .cap_plssvd_ncomp(ncomp, nrow(x) - 1L, ncol(x), q,
            factor_response = classification)$ncomp
    }
    kernel <- config$kernel %||% "linear"
    gamma <- if (identical(context$method, "kernelpls") &&
        !identical(kernel, "linear")) {
        .kernel_pls_gamma(config$gamma, x)
    } else {
        1
    }
    method_id <- switch(
        context$method,
        plssvd = 1L,
        simpls = 3L,
        opls = 4L,
        kernelpls = 5L
    )
    state <- cuda_resident_simpls_fit_cpp(x, y, labels, q,
        if (context$float32) 32L else 64L, max(ncomp), context$scal,
        ctl$rsvd_oversample, ctl$rsvd_power, ctl$seed,
        config$fit || config$return_loadings || config$return_variance,
        method_id,
        as.integer(config$north %||% 1L),
        .kernel_pls_kernel_id(kernel),
        gamma,
        as.integer(config$degree %||% 3L),
        as.numeric(config$coef0 %||% 1))
    cv_internal <- isTRUE(config$cv_internal)
    fields <- if (cv_internal) list() else {
        cuda_resident_export_cpp(
            state,
            config$return_loadings,
            config$return_variance,
            config$fit
        )
    }
    ss <- fields$predictor_ss
    fields$predictor_ss <- NULL
    if (context$float32 && length(fields)) {
        fields <- lapply(fields, .float32_from_bits)
    }
    model <- c(fields, list(resident_state = state, ncomp = ncomp,
        p = ncol(x), m = q, lev = levels, precision = precision,
        resident_backend = "cuda", gpu_resident = TRUE,
        classification = classification, classification_rule = context$classifier,
        pls_method = context$method, predict_backend = "cuda_resident",
        B_stored = FALSE, compact_prediction = TRUE, predict_latent_ok = TRUE,
        R2Y = rep(NA_real_, length(ncomp))))
    if (identical(context$method, "opls")) {
        model$north <- as.integer(config$north %||% 1L)
        model$opls_engine <- "cuda_resident"
    }
    if (identical(context$method, "kernelpls")) {
        model$kernel <- kernel
        model$kernel_id <- .kernel_pls_kernel_id(kernel)
        model$gamma <- gamma
        model$degree <- as.integer(config$degree %||% 3L)
        model$coef0 <- as.numeric(config$coef0 %||% 1)
        model$kernel_engine <- "cuda_resident"
    }
    model$resident_controls <- list(
        requested_oversample = ctl$rsvd_oversample,
        requested_power = ctl$rsvd_power,
        effective_oversample = state$effective_oversample,
        effective_power = state$effective_power,
        refresh_block = state$refresh_block,
        refresh_block_limit = state$refresh_block_limit,
        implicit_crosscovariance = isTRUE(state$implicit_crosscovariance),
        predictor_crossprod_cache = isTRUE(state$predictor_crossprod_cache)
    )
    if (cv_internal) model$cv_internal <- TRUE
    model$execution_route <- "resident CUDA"
    if (!config$return_loadings && !cv_internal) {
        model$P <- matrix(numeric(), 0L, 0L)
    }
    if (!is.null(ss)) {
        ss <- as.vector(.resident_cuda_summary(ss, precision))
        total <- utils::tail(ss, 1L)
        component_ss <- utils::head(ss, -1L)
        if (is.finite(total) && total > 0) {
            model$variance <- .fastpls_named_components(component_ss / max(1L, nrow(x) - 1L), "LV")
            model$variance_explained <- .fastpls_named_components(component_ss / total, "LV")
            model$cumulative_variance_explained <- cumsum(model$variance_explained)
            model$variance_total <- total / max(1L, nrow(x) - 1L)
            model$variance_basis <- "X"
            for (name in c("variance", "variance_explained", "cumulative_variance_explained", "variance_total")) {
                model[[paste0("x_", name)]] <- model[[name]]
            }
        }
    }
    cuda_resident_compact_cpp(
        state,
        prepare_lda = classification &&
            .is_lda_classifier(context$classifier)
    )
    class(model) <- "fastPLS"
    if (config$fit) {
        fitted <- .resident_cuda_predict(model, context$Xtrain, context$Ytrain)
        model$Yfit <- fitted$Ypred
        model$R2Y <- fitted$Q2Y
    }
    if (!is.null(context$Xtest)) {
        predicted <- .resident_cuda_predict(model, context$Xtest, context$Ytest, proj = config$proj)
        model[names(predicted)] <- predicted
    }
    model
}
