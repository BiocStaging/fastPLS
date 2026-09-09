`%||%` <- function(x, y) {
    if (is.null(x)) y else x
}

.fastpls_quiet <- function(expr) {
    withCallingHandlers(
        expr,
        warning = function(condition) invokeRestart("muffleWarning")
    )
}

.cap_plssvd_ncomp <- function(
    ncomp,
    nrows_x,
    ncols_x,
    ncols_y,
    factor_response = FALSE,
    warn = TRUE
) {
    ncomp <- as.integer(ncomp)
    response_rank_bound <- as.integer(ncols_y)
    if (isTRUE(factor_response)) {
        # Centered C-column indicator responses have at most C - 1 independent
        # columns.
        response_rank_bound <- response_rank_bound - 1L
    }
    max_plssvd_rank <- min(
        as.integer(nrows_x),
        as.integer(ncols_x),
        response_rank_bound
    )
    if (max_plssvd_rank < 1L) {
        stop("plssvd rank is < 1")
    }
    over <- max(ncomp, na.rm = TRUE) > max_plssvd_rank
    if (isTRUE(over) && isTRUE(warn)) {
        message_format <- paste0(
            "plssvd rank is limited to %d; requested ncomp above this ",
            "value will use %d components internally"
        )
        warning(
            sprintf(
                message_format,
                max_plssvd_rank,
                max_plssvd_rank
            ),
            call. = FALSE
        )
    }
    ncomp <- pmin(pmax(ncomp, 1L), max_plssvd_rank)
    list(ncomp = ncomp, max_rank = max_plssvd_rank, capped = isTRUE(over))
}

.restore_env_scalar <- function(name, value) {
    stopifnot(length(name) == 1L, nzchar(name))
    if (length(value) != 1L || is.na(value)) {
        Sys.unsetenv(name)
    } else {
        val <- list(as.character(value))
        names(val) <- name
        do.call(Sys.setenv, val)
    }
}

.opls_require_predictive_rank <- function(ncomp, X, removed, centered) {
    available <- min(nrow(X) - as.integer(centered), ncol(X)) - removed
    requested <- max(as.integer(ncomp))
    if (requested > available) {
        message_format <- paste0(
            "OPLS requested %d predictive components, but at most %d remain ",
            "after removing %d orthogonal components; reduce ncomp or north."
        )
        stop(sprintf(message_format, requested, available, removed),
            call. = FALSE)
    }
    invisible(available)
}

.backend_control_env_defaults <- c(
    FASTPLS_STORE_B = "auto",
    FASTPLS_STORE_B_MAX_MB = "256",
    FASTPLS_PREDICT_LATENT_MIN_B_MB = "256",
    FASTPLS_COMPACT_CLASS_BLOCK_SIZE = "4096",
    FASTPLS_LABEL_AWARE_Y_THRESHOLD_MB = "512",
    FASTPLS_LABEL_AWARE_BLOCK_SIZE = "8192",
    FASTPLS_PLSSVD_SMALL_EXACT_MAX_RANK = "32",
    FASTPLS_PLSSVD_OPTIMIZED = "1",
    FASTPLS_LEADING_LEFT_MAX_ITERS = NA_character_,
    FASTPLS_FAST_CENTER_T = "0",
    FASTPLS_FAST_REORTH_V = "0",
    FASTPLS_FAST_DEFLCACHE = "1",
    FASTPLS_FAST_OPTIMIZED = "1",
    FASTPLS_INCREMENTAL_COEFFICIENTS = "1",
    FASTPLS_FAST_CROSSPROD_MIN_NCOMP = "20",
    FASTPLS_FAST_CROSSPROD_MAX_P = "512",
    FASTPLS_FAST_CROSSPROD_MIN_N_TO_P_RATIO = "8",
    FASTPLS_RETURN_TTRAIN = "0",
    FASTPLS_GPU_DEVICE_STATE = "0",
    FASTPLS_GPU_QR = "1",
    FASTPLS_GPU_EIG = "1",
    FASTPLS_GPU_FINALIZE_THRESHOLD = "32",
    FASTPLS_GPU_SIMPLS_XPROD = "0",
    FASTPLS_CUDA_WORKSPACE_STREAMS = "0",
    FASTPLS_FAST_GPU_MIN_M = "512",
    FASTPLS_FAST_GPU_MIN_N = "16",
    FASTPLS_FAST_GPU_MIN_WORK = "200000",
    FASTPLS_CUDA_RSVD_RESIDENT_PUBLIC = "0",
    FASTPLS_CUDA_RSVD_RESIDENT_MIN_L = "48",
    FASTPLS_CUDA_RSVD_RESIDENT_MIN_WORK = "1000000",
    FASTPLS_METAL_MIN_FLOPS = "200000000",
    FASTPLS_METAL_EXACT_MAX_RANK = "256",
    FASTPLS_METAL_EXPERIMENTAL_ITERATIVE = "false",
    FASTPLS_METAL_RESIDENT_SIMPLS = "true",
    FASTPLS_RETURN_LDA_SCORES = "false",
    FASTPLS_FUSED_CUDA_LDA = "0"
)

.backend_control_env_groups <- c(
    FASTPLS_STORE_B = "storage",
    FASTPLS_STORE_B_MAX_MB = "storage",
    FASTPLS_PREDICT_LATENT_MIN_B_MB = "prediction",
    FASTPLS_COMPACT_CLASS_BLOCK_SIZE = "prediction",
    FASTPLS_LABEL_AWARE_Y_THRESHOLD_MB = "response",
    FASTPLS_LABEL_AWARE_BLOCK_SIZE = "response",
    FASTPLS_PLSSVD_SMALL_EXACT_MAX_RANK = "plssvd",
    FASTPLS_PLSSVD_OPTIMIZED = "plssvd",
    FASTPLS_LEADING_LEFT_MAX_ITERS = "plssvd",
    FASTPLS_FAST_CENTER_T = "simpls",
    FASTPLS_FAST_REORTH_V = "simpls",
    FASTPLS_FAST_DEFLCACHE = "simpls",
    FASTPLS_FAST_OPTIMIZED = "simpls",
    FASTPLS_INCREMENTAL_COEFFICIENTS = "simpls",
    FASTPLS_FAST_CROSSPROD_MIN_NCOMP = "simpls",
    FASTPLS_FAST_CROSSPROD_MAX_P = "simpls",
    FASTPLS_FAST_CROSSPROD_MIN_N_TO_P_RATIO = "simpls",
    FASTPLS_RETURN_TTRAIN = "simpls",
    FASTPLS_GPU_DEVICE_STATE = "gpu",
    FASTPLS_GPU_QR = "gpu",
    FASTPLS_GPU_EIG = "gpu",
    FASTPLS_GPU_FINALIZE_THRESHOLD = "gpu",
    FASTPLS_GPU_SIMPLS_XPROD = "gpu",
    FASTPLS_CUDA_WORKSPACE_STREAMS = "cuda",
    FASTPLS_FAST_GPU_MIN_M = "cuda",
    FASTPLS_FAST_GPU_MIN_N = "cuda",
    FASTPLS_FAST_GPU_MIN_WORK = "cuda",
    FASTPLS_CUDA_RSVD_RESIDENT_PUBLIC = "cuda",
    FASTPLS_CUDA_RSVD_RESIDENT_MIN_L = "cuda",
    FASTPLS_CUDA_RSVD_RESIDENT_MIN_WORK = "cuda",
    FASTPLS_METAL_MIN_FLOPS = "metal",
    FASTPLS_METAL_EXACT_MAX_RANK = "metal",
    FASTPLS_METAL_EXPERIMENTAL_ITERATIVE = "metal",
    FASTPLS_METAL_RESIDENT_SIMPLS = "metal",
    FASTPLS_RETURN_LDA_SCORES = "classifier",
    FASTPLS_FUSED_CUDA_LDA = "classifier"
)

.backend_control_package_version <- function() {
    desc <- tryCatch(
        utils::packageDescription("fastPLS", fields = "Version"),
        error = function(e) NA_character_
    )
    if (length(desc) == 1L && !is.na(desc)) {
        as.character(desc)
    } else {
        NA_character_
    }
}

.backend_control_snapshot <- function(context = NULL, overrides = NULL) {
    names_env <- names(.backend_control_env_defaults)
    raw <- Sys.getenv(names_env, unset = NA_character_)
    defaults <- .backend_control_env_defaults
    values <- raw
    missing <- is.na(values)
    values[missing] <- defaults[missing]
    groups <- unname(.backend_control_env_groups[names_env])
    groups[is.na(groups)] <- "other"
    env <- data.frame(
        name = names_env,
        group = groups,
        value = unname(values),
        default = unname(defaults),
        overridden = !is.na(raw),
        stringsAsFactors = FALSE
    )
    out <- list(
        context = context %||% NA_character_,
        timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z"),
        fastPLS_version = .backend_control_package_version(),
        env = env,
        overrides = overrides %||% list()
    )
    class(out) <- "fastPLSBackendControl"
    out
}

.with_backend_env <- function(expr, values) {
    if (is.null(values) || length(values) == 0L) {
        return(force(expr))
    }
    if (is.null(names(values)) || any(!nzchar(names(values)))) {
        stop("backend-control environment values must be named", call. = FALSE)
    }
    values <- as.list(values)
    old <- Sys.getenv(names(values), unset = NA_character_)
    on.exit(
        {
            for (nm in names(old)) {
                .restore_env_scalar(nm, old[[nm]])
            }
        },
        add = TRUE
    )
    for (nm in names(values)) {
        .restore_env_scalar(nm, values[[nm]])
    }
    force(expr)
}

.attach_backend_control <- function(model, backend_control = NULL) {
    model
}

.with_fastpls_fast_options <- function(expr, return_ttrain = FALSE) {
    if (identical(Sys.getenv("FASTPLS_ABLATION_MODE", "0"), "1")) {
        return(force(expr))
    }
    .with_backend_env(
        expr,
        c(
            FASTPLS_FAST_CENTER_T = "0",
            FASTPLS_FAST_REORTH_V = "0",
            FASTPLS_FAST_DEFLCACHE = "1",
            FASTPLS_RETURN_TTRAIN = if (isTRUE(return_ttrain)) "1" else "0"
        )
    )
}

.with_gpu_native_options <- function(
    expr,
    gpu_device_state = FALSE,
    gpu_qr = TRUE,
    gpu_eig = TRUE,
    gpu_finalize_threshold = 32L
) {
    .with_backend_env(
        expr,
        c(
        FASTPLS_GPU_DEVICE_STATE = if (isTRUE(gpu_device_state)) "1" else "0",
            FASTPLS_GPU_QR = if (isTRUE(gpu_qr)) "1" else "0",
            FASTPLS_GPU_EIG = if (isTRUE(gpu_eig)) "1" else "0",
            FASTPLS_GPU_FINALIZE_THRESHOLD = as.character(as.integer(
                gpu_finalize_threshold
            ))
        )
    )
}

.with_simpls_gpu_xprod <- function(expr) {
    .with_backend_env(expr, c(FASTPLS_GPU_SIMPLS_XPROD = "1"))
}

.enable_flash_prediction <- function(
    model,
    backend = c("cpu", "cuda"),
    block_size = 4096L
) {
    backend <- match.arg(backend)
    model$predict_backend <- if (identical(backend, "cuda")) {
        "cuda_flash"
    } else {
        "cpu_flash"
    }
    model$flash_svd <- TRUE
    model$flash_svd_backend <- backend
    model$flash_svd_mode <- "streamed_low_rank_prediction"
    model$flash_block_size <- as.integer(block_size)
    model
}

.attach_train_scores <- function(model, Xtrain) {
    if (is.null(model$R) || length(model$R) == 0L) {
        return(model)
    }
    if (
        !is.null(model$Ttrain) &&
            length(model$Ttrain) > 0L &&
            all(dim(model$Ttrain) > 0L)
    ) {
        return(model)
    }
    model$Ttrain <- .fastpls_latent_scores(
        model,
        Xtrain,
        ncomp = max(model$ncomp),
        backend = "cpu"
    )
    model
}

.maybe_attach_x_loadings <- function(model, Xtrain, return_loadings = FALSE) {
    if (!is.null(model$resident_state)) {
        return(model)
    }
    empty_p <- matrix(numeric(0), 0L, 0L)
    if (is.null(model) || !is.list(model)) {
        return(model)
    }
    if (!isTRUE(return_loadings)) {
        model$P <- empty_p
        if (!is.null(model$inner_model) && is.list(model$inner_model)) {
            model$inner_model <- .maybe_attach_x_loadings(model$inner_model,
                Xtrain,
                FALSE)
        }
        return(model)
    }
    if (is.null(model$R) || length(model$R) == 0L || is.null(model$ncomp)) {
        if (!is.null(model$inner_model) && is.list(model$inner_model)) {
            model$inner_model <- .maybe_attach_x_loadings(model$inner_model,
                Xtrain,
                TRUE)
        }
        return(model)
    }
    R <- as.matrix(model$R); Xtrain <- as.matrix(Xtrain)
    if (nrow(R) != ncol(Xtrain)) {
        return(model)
    }
    k <- min(max(as.integer(model$ncomp), na.rm = TRUE), ncol(R))
    if (!is.finite(k) || is.na(k) || k < 1L) {
        return(model)
    }
    Xscaled <- .fastpls_scaled_by_model(model, Xtrain)
    scores <- .fastpls_score_matrix(model, "Ttrain")
    if (is.null(scores) || ncol(scores) < k || nrow(scores) != nrow(Xscaled)) {
        scores <- .fastpls_latent_scores(model, Xtrain, ncomp = k,
            backend = "cpu")
    }
    scores <- as.matrix(scores)[, seq_len(k), drop = FALSE]
    denom <- colSums(scores * scores)
    ok <- is.finite(denom) & denom > 0
    P <- matrix(0, nrow = ncol(Xscaled), ncol = k)
    if (any(ok)) {
        P[, ok] <- sweep(crossprod(Xscaled, scores[, ok, drop = FALSE]), 2L,
            denom[ok],
            "/", check.margin = FALSE)
    }
    rownames(P) <- colnames(Xtrain)
    colnames(P) <- paste0("LV", seq_len(k))
    model$P <- P
    model
}

.fastpls_named_components <- function(x, prefix) {
    names(x) <- paste0(prefix, seq_along(x))
    x
}

.pls_variance_scores <- function(model, Xtrain, Xscaled, k) {
    scores <- .fastpls_score_matrix(model, "Ttrain")
    if (is.null(scores) || ncol(scores) < k || nrow(scores) != nrow(Xscaled)) {
        scores <- .fastpls_latent_scores(
            model,
            Xtrain,
            ncomp = k,
            backend = "cpu"
        )
    }
    as.matrix(scores)[, seq_len(k), drop = FALSE]
}

.pls_orthogonal_score_ss <- function(Xscaled, scores) {
    score_gram <- crossprod(scores)
    score_norms <- diag(score_gram)
    offdiag <- score_gram
    diag(offdiag) <- 0
    gram_scale <- max(abs(score_norms), 1)
    orthogonal <- all(is.finite(score_norms)) &&
        all(score_norms > 0) &&
    max(abs(offdiag), na.rm = TRUE) <= sqrt(.Machine$double.eps) * gram_scale
    if (!orthogonal) {
        return(NULL)
    }
    XtT <- crossprod(Xscaled, scores)
    explained_ss <- colSums(XtT * XtT) / score_norms
    explained_ss[!is.finite(explained_ss) | explained_ss < 0] <- 0
    explained_ss
}

.pls_sequential_score_ss <- function(Xscaled, scores) {
    explained_ss <- numeric(ncol(scores))
    residual <- Xscaled
    for (j in seq_len(ncol(scores))) {
        tj <- scores[, j, drop = FALSE]
        denom <- drop(crossprod(tj))
        if (!is.finite(denom) || denom <= 0) {
            next
        }
        before <- sum(residual * residual)
        pj <- crossprod(residual, tj) / denom
        residual <- residual - tj %*% t(pj)
        gain <- before - sum(residual * residual)
        explained_ss[j] <- if (is.finite(gain) && gain > 0) gain else 0
    }
    explained_ss
}

.pls_x_variance_explained <- function(model, Xtrain) {
    if (is.null(model$R) || length(model$R) == 0L || is.null(model$ncomp)) {
        return(NULL)
    }
    k <- min(
        max(as.integer(model$ncomp), na.rm = TRUE),
        ncol(as.matrix(model$R))
    )
    if (!is.finite(k) || is.na(k) || k < 1L) {
        return(NULL)
    }
    Xscaled <- .fastpls_scaled_by_model(model, Xtrain)
    total_ss <- sum(Xscaled * Xscaled)
    if (!is.finite(total_ss) || total_ss <= 0) {
        return(NULL)
    }
    scores <- .pls_variance_scores(model, Xtrain, Xscaled, k)
    explained_ss <- .pls_orthogonal_score_ss(Xscaled, scores)
    if (is.null(explained_ss)) {
        explained_ss <- .pls_sequential_score_ss(Xscaled, scores)
    }

    denom_df <- max(1, nrow(Xscaled) - 1L)
    variance <- explained_ss / denom_df
    variance_explained <- explained_ss / total_ss
    variance <- .fastpls_named_components(variance, "LV")
    variance_explained <- .fastpls_named_components(variance_explained, "LV")
    cumulative <- .fastpls_named_components(cumsum(variance_explained), "LV")
    list(
        variance = variance,
        variance_explained = variance_explained,
        cumulative_variance_explained = cumulative,
        variance_total = total_ss / denom_df,
        variance_basis = "X"
    )
}

.maybe_attach_pls_variance_explained <- function(
    model,
    Xtrain,
    return_variance = TRUE
) {
    if (!isTRUE(return_variance)) {
        return(model)
    }
    .attach_pls_variance_explained(model, Xtrain)
}

.attach_pls_variance_explained <- function(model, Xtrain) {
    stats <- try(.pls_x_variance_explained(model, Xtrain), silent = TRUE)
    if (inherits(stats, "try-error") || is.null(stats)) {
        return(model)
    }
    model$variance <- stats$variance
    model$variance_explained <- stats$variance_explained
    model$cumulative_variance_explained <- stats$cumulative_variance_explained
    model$variance_total <- stats$variance_total
    model$variance_basis <- stats$variance_basis
    model$x_variance <- stats$variance
    model$x_variance_explained <- stats$variance_explained
    model$x_cumulative_variance_explained <- stats$cumulative_variance_explained
    model$x_variance_total <- stats$variance_total
    model
}

.inherit_inner_variance_explained <- function(model, inner) {
    fields <- c(
        "variance",
        "variance_explained",
        "cumulative_variance_explained",
        "variance_total",
        "variance_basis",
        "x_variance",
        "x_variance_explained",
        "x_cumulative_variance_explained",
        "x_variance_total"
    )
    for (field in fields) {
        if (!is.null(inner[[field]])) {
            model[[field]] <- inner[[field]]
        }
    }
    model
}

.inherit_inner_fit_outputs <- function(model, inner) {
    fields <- c(
        "ncomp",
        "Yfit",
        "R2Y",
        "classification",
        "lev",
        "classification_rule",
        "precision"
    )
    for (field in fields) {
        if (!is.null(inner[[field]])) {
            model[[field]] <- inner[[field]]
        }
    }
    model
}

.classifier_public_choices <- c("argmax", "lda")
.classifier_internal_choices <- c(
    "argmax",
    "lda_cpp",
    "lda_cuda",
    "lda_metal"
)

.fixed_lda_relative_ridge <- 1e-8

.resolve_deprecated_lda_ridge <- function(value, supplied, context) {
    if (isTRUE(supplied) && !is.null(value)) {
        message_format <- paste0(
            "%s argument 'lda_ridge' is deprecated and ignored; PLS-LDA ",
            "uses the fixed scale-normalized Cholesky fallback sequence."
        )
        warning(
            sprintf(message_format, context),
            call. = FALSE
        )
    }
    .fixed_lda_relative_ridge
}

.normalize_classifier_public <- function(classifier) {
    if (length(classifier) > 1L) {
        classifier <- classifier[1L]
    }
    classifier <- as.character(classifier)
    match.arg(classifier, .classifier_public_choices)
}

.normalize_classifier <- function(classifier) {
    if (length(classifier) > 1L) {
        classifier <- classifier[1L]
    }
    classifier <- as.character(classifier)
    if (classifier %in% .classifier_public_choices) {
        classifier <- switch(classifier, argmax = "argmax", lda = "lda_cpp")
    }
    match.arg(classifier, .classifier_internal_choices)
}

.resolve_classifier_for_backend <- function(classifier, backend) {
    if (length(classifier) > 1L) {
        classifier <- classifier[1L]
    }
    classifier <- as.character(classifier)
    if (classifier %in% .classifier_internal_choices) {
        return(.normalize_classifier(classifier))
    }
    classifier <- .normalize_classifier_public(classifier)
    backend <- .normalize_public_backend(backend)
    switch(
        classifier,
        argmax = "argmax",
        lda = switch(
            backend,
            cpu = "lda_cpp",
            cuda = "lda_cuda",
            metal = "lda_cpp"
        )
    )
}

.is_lda_classifier <- function(classifier) {
!is.null(classifier) && classifier %in% c("lda_cpp", "lda_cuda", "lda_metal")
}

.resolve_top_k <- function(top = 1L, top5 = FALSE) {
    top <- as.integer(top)[1L]
    if (!is.finite(top) || is.na(top) || top < 1L) {
        stop("top must be a positive integer", call. = FALSE)
    }
    if (isTRUE(top5)) {
        top <- max(top, 5L)
    }
    top
}

.class_topk_to_labels <- function(top_index, top_score, lev, ncomp) {
    dims <- dim(top_index)
    labels <- array(lev[as.integer(top_index)], dim = dims)
    top1 <- as.data.frame(matrix(
        labels[, 1L, ],
        nrow = dims[1L],
        ncol = dims[3L]
    ))
    colnames(top1) <- paste("ncomp=", ncomp, sep = "")
    for (j in seq_along(top1)) {
        top1[[j]] <- factor(top1[[j]], levels = lev)
    }
    out <- list(Ypred = top1)
out$Ypred_index <- matrix(top_index[, 1L, ], nrow = dims[1L], ncol = dims[3L])
    colnames(out$Ypred_index) <- paste("ncomp=", ncomp, sep = "")
    if (dims[2L] > 1L) {
        top_list <- vector("list", dims[3L])
        score_list <- vector("list", dims[3L])
        names(top_list) <- names(score_list) <- paste("ncomp=", ncomp, sep = "")
        for (a in seq_len(dims[3L])) {
            top_list[[a]] <- matrix(
                labels[, , a],
                nrow = dims[1L],
                ncol = dims[2L],
                dimnames = list(NULL, paste0("rank", seq_len(dims[2L])))
            )
            score_list[[a]] <- matrix(
                top_score[, , a],
                nrow = dims[1L],
                ncol = dims[2L],
                dimnames = list(NULL, paste0("rank", seq_len(dims[2L])))
            )
        }
        out$Ypred_top <- top_list
        out$Ypred_top_score <- score_list
    }
    out
}

.class_topk_from_score_cube <- function(score_cube, lev, ncomp, top = 1L) {
    dims <- dim(score_cube)
    top <- min(as.integer(top)[1L], dims[2L])
    top_index <- array(NA_integer_, dim = c(dims[1L], top, dims[3L]))
    top_score <- array(NA_real_, dim = c(dims[1L], top, dims[3L]))
    for (a in seq_len(dims[3L])) {
        score <- score_cube[, , a, drop = FALSE][, , 1L]
        if (top == 1L) {
            idx <- max.col(score, ties.method = "first")
            top_index[, 1L, a] <- idx
            top_score[, 1L, a] <- score[cbind(seq_len(nrow(score)), idx)]
        } else {
            for (i in seq_len(nrow(score))) {
                idx <- order(score[i, ], decreasing = TRUE)[seq_len(top)]
                top_index[i, , a] <- idx
                top_score[i, , a] <- score[i, idx]
            }
        }
    }
    .class_topk_to_labels(top_index, top_score, lev, ncomp)
}

.class_topk_predict <- function(
    model,
    Xtest,
    top = 1L,
    proj = FALSE,
    backend = c("cpp", "cuda")
) {
    backend <- match.arg(backend)
    if (identical(backend, "cuda") && !isTRUE(has_cuda())) {
        stop(
            "CUDA class prediction requires an available CUDA backend. ",
            "No CPU fallback is performed.",
            call. = FALSE
        )
    }
    block_size <- model$flash_block_size
    if (is.null(block_size) || !length(block_size) || is.na(block_size)) {
        block_size <- 4096L
    }
    out <- if (identical(backend, "cuda")) {
        pls_class_predict_topk_cuda(
            model,
            as.matrix(Xtest),
            as.integer(top),
            isTRUE(proj)
        )
    } else {
        pls_class_predict_topk_cpp(
            model,
            as.matrix(Xtest),
            as.integer(top),
            isTRUE(proj),
            as.integer(block_size)
        )
    }
    res <- .class_topk_to_labels(
        out$top_index,
        out$top_score,
        model$lev,
        model$ncomp
    )
    if (isTRUE(proj)) {
        res$Ttest <- out$Ttest
    }
    if (!is.null(out$predict_backend)) {
        res$predict_backend <- out$predict_backend
    }
    res
}

.fastpls_block_size <- function(option_name, env_name, default = 4096L) {
    value <- getOption(option_name, NULL)
    if (is.null(value)) {
        value <- Sys.getenv(env_name, unset = as.character(default))
    }
    value <- .fastpls_quiet(as.integer(value)[1L])
    if (!is.finite(value) || is.na(value) || value < 1L) {
        value <- as.integer(default)
    }
    value
}

.dense_indicator_exceeds_cuda_guard <- function(n, q) {
    dense_y_mb <- as.numeric(n) * as.numeric(q) * 8 / 1024^2
    threshold <- .fastpls_quiet(
        as.numeric(Sys.getenv("FASTPLS_LABEL_AWARE_Y_THRESHOLD_MB", "512"))[1L]
    )
    if (!is.finite(threshold) || threshold < 0) {
        threshold <- 512
    }
    isTRUE(dense_y_mb >= threshold)
}

.stop_unsafe_cuda_simpls_response <- function(n, q) {
    dense_y_mb <- as.numeric(n) * as.numeric(q) * 8 / 1024^2
    stop(
        sprintf(
            "%s%s%s%s",
            sprintf(
        "CUDA SIMPLS would require an approximately %.1f MB dense indicator ",
                dense_y_mb
            ),
        "response. fastPLS does not replace a requested SIMPLS estimator with ",
            "PLS-SVD. Request method = 'plssvd' explicitly or use a smaller ",
            "response representation."
        ),
        call. = FALSE
    )
}

.rowsum_compact_codes <- function(x, codes, n_groups) {
    sums <- rowsum(x, group = as.integer(codes), reorder = FALSE)
    out <- matrix(0, nrow = n_groups, ncol = ncol(x))
    positions <- .fastpls_quiet(as.integer(rownames(sums)))
    valid <- !is.na(positions) & positions >= 1L & positions <= n_groups
    out[positions[valid], ] <- sums[valid, , drop = FALSE]
    out
}

.stream_predictor_stats <- function(X, scaling, block_size) {
    n <- nrow(X)
    p <- ncol(X)
    sums <- sums_sq <- numeric(p)
    for (start in seq(1L, n, by = block_size)) {
        rows <- start:min(n, start + block_size - 1L)
        block <- X[rows, , drop = FALSE]
        sums <- sums + colSums(block)
        if (scaling == 2L) sums_sq <- sums_sq + colSums(block * block)
    }
    mean <- if (scaling < 3L) sums / n else rep(0, p)
    scale <- rep(1, p)
    if (scaling == 2L) {
        scale <- sqrt(pmax(sums_sq - n * mean * mean, 0) / max(1L, n - 1L))
        scale[!is.finite(scale) | scale == 0] <- 1
    }
    list(mean = mean, scale = scale)
}

.stream_scaled_block <- function(X, rows, scaling, stats) {
    block <- X[rows, , drop = FALSE]
    if (scaling < 3L) {
        block <- sweep(block, 2L, stats$mean, "-")
    }
    if (scaling == 2L) {
        block <- sweep(block, 2L, stats$scale, "/")
    }
    block
}

.stream_class_crossproduct <- function(
    X,
    labels,
    classes,
    scaling,
    stats,
    block_size
) {
    result <- matrix(0, nrow = ncol(X), ncol = classes)
    for (start in seq(1L, nrow(X), by = block_size)) {
        rows <- start:min(nrow(X), start + block_size - 1L)
        block <- .stream_scaled_block(X, rows, scaling, stats)
        result <- result +
            t(.rowsum_compact_codes(
                block,
                labels[rows],
                classes
            ))
    }
    result
}

.stream_score_gram <- function(X, R, scaling, stats, block_size) {
    gram <- matrix(0, nrow = ncol(R), ncol = ncol(R))
    for (start in seq(1L, nrow(X), by = block_size)) {
        rows <- start:min(nrow(X), start + block_size - 1L)
        scores <- .stream_scaled_block(X, rows, scaling, stats) %*% R
        gram <- gram + crossprod(scores)
    }
    gram
}

.plssvd_stream_latent_path <- function(gram, singular, Q, ncomp) {
    rank <- nrow(gram)
    classes <- nrow(Q)
    slices <- length(ncomp)
    coefficients <- array(0, c(rank, rank, slices))
    weights <- array(0, c(rank, classes, slices))
    for (index in seq_along(ncomp)) {
        k <- ncomp[[index]]
        gram_k <- gram[seq_len(k), seq_len(k), drop = FALSE]
        diagonal <- diag(singular[seq_len(k)], nrow = k)
        ridge <- 1e-10 * mean(diag(gram_k))
        if (!is.finite(ridge) || ridge <= 0) {
            ridge <- 1e-10
        }
        value <- tryCatch(
            solve(gram_k + diag(ridge, k), diagonal),
            error = function(e) qr.solve(gram_k + diag(ridge, k), diagonal)
        )
        coefficients[seq_len(k), seq_len(k), index] <- value
        weights[seq_len(k), , index] <-
            value %*% t(Q[, seq_len(k), drop = FALSE])
    }
    list(coefficients = coefficients, weights = weights)
}

.plssvd_stream_model <- function(
    R,
    Q,
    path,
    stats,
    response_mean,
    ncomp,
    levels,
    backend,
    block_size
) {
    model <- list(
        C_latent = path$coefficients,
        W_latent = path$weights,
        Q = Q,
        Ttrain = matrix(numeric(0), 0L, ncol(R)),
        R = R,
        mX = matrix(stats$mean, nrow = 1L),
        vX = matrix(stats$scale, nrow = 1L),
        mY = matrix(response_mean, nrow = 1L),
        p = nrow(R),
        m = nrow(Q),
        ncomp = ncomp,
        Yfit = array(numeric(0), c(0L, 0L, 0L)),
        R2Y = rep(NA_real_, length(ncomp)),
        pls_method = "plssvd",
        classification = TRUE,
        lev = levels,
        predict_latent_ok = TRUE,
        xprod_default = TRUE,
        xprod_mode = "label_aware_stream",
        B_stored = FALSE,
        compact_prediction = TRUE,
        flash_svd = TRUE,
        flash_svd_backend = backend,
        predict_backend = if (backend == "cuda") "cuda_flash" else "cpu_flash",
        flash_block_size = block_size
    )
    class(model) <- "fastPLS"
    .attach_backend_control(model)
}

.plssvd_label_aware_stream_model <- function(Xtrain, y_train, ncomp,
    scaling = 1L,
    backend = c("cpp", "cuda"), block_size = NULL) {
    backend <- match.arg(backend)
    Xtrain <- as.matrix(Xtrain)
    y_train <- factor(y_train)
    n <- nrow(Xtrain)
    p <- ncol(Xtrain)
    classes <- nlevels(y_train)
    if (n < 1L || p < 1L || classes < 2L) {
        stop("label-aware PLS-SVD requires X and at least two classes",
            call. = FALSE)
    }
    ncomp <- .cap_plssvd_ncomp(ncomp, n, p, classes, factor_response = TRUE,
        warn = TRUE)$ncomp
    if (is.null(block_size)) {
        block_size <- .fastpls_block_size("fastPLS.label_aware_block_size",
            "FASTPLS_LABEL_AWARE_BLOCK_SIZE",
            8192L)
    }
    scaling <- as.integer(scaling)
    labels <- as.integer(y_train)
    stats <- .stream_predictor_stats(Xtrain, scaling, block_size)
    response_mean <- tabulate(labels, nbins = classes) / n
    class_product <- .stream_class_crossproduct(Xtrain, labels, classes,
        scaling,
        stats, block_size)
    S <- class_product - tcrossprod(rowSums(class_product), response_mean)
    decomposition <- svd(S, nu = max(ncomp), nv = max(ncomp))
    R <- decomposition$u[, seq_len(max(ncomp)), drop = FALSE]
    Q <- decomposition$v[, seq_len(max(ncomp)), drop = FALSE]
    gram <- .stream_score_gram(Xtrain, R, scaling, stats, block_size)
    path <- .plssvd_stream_latent_path(gram, decomposition$d, Q, ncomp)
    .plssvd_stream_model(R, Q, path, stats, response_mean, ncomp,
        levels(y_train),
        backend, block_size)
}

.plssvd_label_aware_scores_fast_model <- function(Xtrain, y_train, ncomp,
    scaling = 1L) {
    Xtrain <- as.matrix(Xtrain)
    y_train <- factor(y_train)
    lev <- levels(y_train)
    y_code <- as.integer(y_train)
    n <- nrow(Xtrain)
    p <- ncol(Xtrain)
    m <- length(lev)
    cap <- .cap_plssvd_ncomp(ncomp, n, p, m, factor_response = TRUE,
        warn = TRUE)
    ncomp <- as.integer(cap$ncomp)
    max_rank <- max(ncomp)
    stats <- label_crossprod_scaled_cpp(Xtrain, y_code, m, as.integer(scaling))
    sv <- svd(as.matrix(stats$S), nu = max_rank, nv = 0)
    R <- sv$u[, seq_len(max_rank), drop = FALSE]
    model <- list(P = matrix(numeric(0), 0L, 0L), Q = matrix(numeric(0), m,
        max_rank),
    Ttrain = matrix(numeric(0), 0L, 0L), R = R,
    mX = matrix(as.numeric(stats$mX),
        nrow = 1L), vX = matrix(as.numeric(stats$vX), nrow = 1L),
    mY = matrix(as.numeric(stats$mY),
        nrow = 1L), p = p, m = m, ncomp = ncomp, Yfit = array(numeric(0),
        dim = c(0L,
            0L, 0L)), R2Y = rep(NA, length(ncomp)), pls_method = "plssvd",
    classification = TRUE,
    lev = lev, predict_latent_ok = TRUE, xprod_default = TRUE,
    xprod_mode = "label_aware_class_sums",
    B_stored = FALSE, compact_prediction = TRUE, flash_svd = TRUE,
    flash_svd_backend = "cuda",
    predict_backend = "cuda_flash")
    class(model) <- "fastPLS"
    .attach_backend_control(model)
}

.fastpls_scaled_by_model <- function(object, X) {
    X <- as.matrix(X)
    if (!is.null(object$mX) && length(object$mX) == ncol(X)) {
        X <- sweep(X, 2L, as.numeric(object$mX), "-", check.margin = FALSE)
    }
    if (!is.null(object$vX) && length(object$vX) == ncol(X)) {
        scale <- as.numeric(object$vX)
        scale[!is.finite(scale) | scale == 0] <- 1
        X <- sweep(X, 2L, scale, "/", check.margin = FALSE)
    }
    X
}

.fastpls_score_multiply <- function(X, projection, backend) {
    if (identical(backend, "cuda")) {
        if (!.cuda_matmul_available()) {
            stop(
                "CUDA score projection is unavailable. ",
                "No CPU fallback is performed.",
                call. = FALSE
            )
        }
        return(.cuda_matmul(X, projection))
    }
    if (identical(backend, "metal")) {
        .fastpls_require_backend_available(
            "metal",
            "Latent-score projection"
        )
        return(.metal_mm(X, projection))
    }
    X %*% projection
}

.fastpls_apply_score_offset <- function(scores, offset) {
    if (!is.null(offset) && any(offset != 0)) {
        return(sweep(
            scores,
            2L,
            as.numeric(offset),
            "-",
            check.margin = FALSE
        ))
    }
    scores
}

.fastpls_cached_projection <- function(object, X, k) {
    cached <- object$R_predict
    if (is.null(cached) || length(cached) == 0L) {
        return(NULL)
    }
    cached <- as.matrix(cached)
    if (ncol(cached) < k || nrow(cached) != ncol(X)) {
        return(NULL)
    }
    list(
        projection = cached[, seq_len(k), drop = FALSE],
        offset = if (length(object$R_offset) >= k) {
            as.numeric(object$R_offset)[seq_len(k)]
        } else {
            NULL
        }
    )
}

.fastpls_latent_scores <- function(
    object,
    X,
    ncomp = max(object$ncomp),
    backend = c("cpu", "cuda", "metal")
) {
    backend <- match.arg(backend)
    if (is.null(object$R) || length(object$R) == 0L) {
        stop(
            "LDA classification requires latent projection matrix R",
            call. = FALSE
        )
    }
    R <- as.matrix(object$R)
    k <- min(as.integer(ncomp), ncol(R))
    if (!is.finite(k) || is.na(k) || k < 1L) {
        stop(
            "LDA classification requires at least one latent component",
            call. = FALSE
        )
    }
    X <- as.matrix(X)
    cached <- .fastpls_cached_projection(object, X, k)
    if (!is.null(cached)) {
        scores <- .fastpls_score_multiply(X, cached$projection, backend)
        return(.fastpls_apply_score_offset(scores, cached$offset))
    }
    R <- R[, seq_len(k), drop = FALSE]
    if (!is.null(object$vX) && length(object$vX) == nrow(R)) {
        scale <- as.numeric(object$vX)
        scale[!is.finite(scale) | scale == 0] <- 1
        R <- sweep(R, 1L, scale, "/", check.margin = FALSE)
    }
    offset <- NULL
    if (!is.null(object$mX) && length(object$mX) == nrow(R)) {
        offset <- drop(as.numeric(object$mX) %*% R)
    }
    scores <- .fastpls_score_multiply(X, R, backend)
    .fastpls_apply_score_offset(scores, offset)
}

.attach_latent_projection_cache <- function(model, ncomp = max(model$ncomp)) {
    if (is.null(model$R) || length(model$R) == 0L) {
        return(model)
    }
    R <- as.matrix(model$R)
    k <- min(as.integer(ncomp), ncol(R))
    if (!is.finite(k) || is.na(k) || k < 1L) {
        return(model)
    }
    R <- R[, seq_len(k), drop = FALSE]
    if (!is.null(model$vX) && length(model$vX) == nrow(R)) {
        scale <- as.numeric(model$vX)
        scale[!is.finite(scale) | scale == 0] <- 1
        R <- sweep(R, 1L, scale, "/", check.margin = FALSE)
    }
    offset <- rep(0, k)
    if (!is.null(model$mX) && length(model$mX) == nrow(R)) {
        offset <- drop(as.numeric(model$mX) %*% R)
    }
    model$R_predict <- R
    model$R_offset <- offset
    model
}

.lda_stream_inputs <- function(Xtrain, R, offset, y_codes, n_classes, ncomp) {
    Xtrain <- as.matrix(Xtrain)
    R <- as.matrix(R)
    n <- nrow(Xtrain)
    p <- ncol(Xtrain)
    if (n < 1L || p < 1L || nrow(R) != p || ncol(R) < 1L) {
        stop(
            "streamed LDA training received incompatible dimensions",
            call. = FALSE
        )
    }
    y_codes <- as.integer(y_codes)
    if (
        length(y_codes) != n ||
            anyNA(y_codes) ||
            any(y_codes < 1L | y_codes > n_classes)
    ) {
        stop(
            "streamed LDA training requires labels encoded as 1..n_classes",
            call. = FALSE
        )
    }
    ncomp <- as.integer(ncomp)
    kmax <- max(ncomp, na.rm = TRUE)
    if (!is.finite(kmax) || is.na(kmax) || kmax < 1L || kmax > ncol(R)) {
    stop("streamed LDA component counts must be in 1..ncol(R)", call. = FALSE)
    }
    offset <- c(as.numeric(offset), rep(0, kmax))[seq_len(kmax)]
    list(
        X = Xtrain,
        R = R[, seq_len(kmax), drop = FALSE],
        offset = offset,
        labels = y_codes,
        n = n,
        kmax = kmax
    )
}

.lda_stream_moments <- function(input, n_classes, block_size, backend) {
    class_sums <- matrix(0, nrow = n_classes, ncol = input$kmax)
    gram <- matrix(0, nrow = input$kmax, ncol = input$kmax)
    multiply <- switch(
        backend,
        cuda = .cuda_matmul,
        metal = .metal_mm,
        cpu = function(x, y) x %*% y
    )
    for (start in seq(1L, input$n, by = block_size)) {
        rows <- start:min(input$n, start + block_size - 1L)
        scores <- multiply(input$X[rows, , drop = FALSE], input$R)
        if (any(input$offset != 0)) {
            scores <- sweep(scores, 2L, input$offset, "-")
        }
        gram <- gram + crossprod(scores)
        class_sums <- class_sums +
            .rowsum_compact_codes(
                scores,
                input$labels[rows],
                n_classes
            )
        if ((start %/% block_size) %% 16L == 0L) gc(FALSE)
    }
    list(gram = gram, class_sums = class_sums)
}

.require_lda_compute_backend <- function(backend, context) {
    if (backend == "cuda" && !.cuda_matmul_available()) {
        stop(
            context,
            " requested CUDA, but CUDA matrix multiplication is unavailable. ",
            "No CPU fallback is performed.",
            call. = FALSE
        )
    }
    if (backend == "metal") {
        .fastpls_require_backend_available("metal", context)
    }
    invisible(backend)
}

.lda_train_projected_stream <- function(
    Xtrain,
    R,
    offset,
    y_codes,
    n_classes,
    ncomp,
    ridge = 1e-8,
    block_size = NULL,
    backend = c("cpu", "cuda", "metal")
) {
    backend <- match.arg(backend)
    input <- .lda_stream_inputs(
        Xtrain,
        R,
        offset,
        y_codes,
        n_classes,
        ncomp
    )
    if (is.null(block_size)) {
        block_size <- .fastpls_block_size(
            "fastPLS.label_aware_block_size",
            "FASTPLS_LABEL_AWARE_BLOCK_SIZE",
            default = 8192L
        )
    }
    block_size <- max(1L, as.integer(block_size)[1L])
    .require_lda_compute_backend(backend, "Streamed LDA training")
    counts <- tabulate(input$labels, nbins = n_classes)
    if (any(counts <= 0L)) {
        stop("streamed LDA training received an empty class", call. = FALSE)
    }
    moments <- .lda_stream_moments(input, n_classes, block_size, backend)
    unique_ncomp <- sort(unique(pmax(1L, pmin(ncomp, input$kmax))))
    models <- lda_train_moments_prefix_cpp(
        moments$gram,
        moments$class_sums,
        as.numeric(counts),
        input$n,
        unique_ncomp
    )
    names(models) <- as.character(unique_ncomp)
    models
}

.fastpls_lda_predict_cuda <- function(Ttest, lda) {
    if (
        !.cuda_matmul_available() ||
            !exists(
                "lda_predict_cuda",
                envir = asNamespace("fastPLS"),
                inherits = FALSE
            )
    ) {
        stop(
            "CUDA LDA prediction is unavailable. ",
            "No CPU fallback is performed.",
            call. = FALSE
        )
    }
    get("lda_predict_cuda", envir = asNamespace("fastPLS"), inherits = FALSE)(
        as.matrix(Ttest),
        lda
    )
}

.fastpls_lda_project_predict_cuda <- function(
    Xtest,
    R,
    offset,
    lda,
    return_scores = FALSE
) {
    if (
        !.cuda_matmul_available() ||
            !exists(
                "lda_project_predict_cuda",
                envir = asNamespace("fastPLS"),
                inherits = FALSE
            )
    ) {
        stop(
            "CUDA projected LDA prediction is unavailable. ",
            "No CPU fallback is performed.",
            call. = FALSE
        )
    }
    get(
        "lda_project_predict_cuda",
        envir = asNamespace("fastPLS"),
        inherits = FALSE
    )(
        as.matrix(Xtest),
        as.matrix(R),
        as.numeric(offset),
        lda,
        isTRUE(return_scores)
    )
}

.fastpls_lda_project_predict_cpp <- function(Xtest, R, offset, lda) {
    if (
        !exists(
            "lda_project_predict_labels_cpp",
            envir = asNamespace("fastPLS"),
            inherits = FALSE
        )
    ) {
        Ttest <- sweep(
            as.matrix(Xtest) %*% as.matrix(R),
            2L,
            as.numeric(offset),
            "-",
            check.margin = FALSE
        )
        return(lda_predict_labels_cpp(Ttest, lda))
    }
    get(
        "lda_project_predict_labels_cpp",
        envir = asNamespace("fastPLS"),
        inherits = FALSE
    )(
        as.matrix(Xtest),
        as.matrix(R),
        as.numeric(offset),
        lda
    )
}

.resolve_lda_backend <- function(model, classifier) {
    if (classifier == "lda_cuda" && !.cuda_matmul_available()) {
        stop(
            "CUDA LDA was requested, but its native backend is unavailable. ",
            "No CPU fallback is performed.",
            call. = FALSE
        )
    }
    if (classifier == "lda_metal" && !isTRUE(has_metal())) {
        .fastpls_require_backend_available("metal", "LDA fitting")
    }
    model$classification_rule <- classifier
    model$lda_backend <- classifier
    list(model = model, classifier = classifier)
}

.lda_component_grid <- function(model, available) {
    values <- pmax(1L, pmin(as.integer(model$ncomp), available))
    sort(unique(values))
}

.attach_lda_models <- function(model, models, components, ridge, backend) {
    names(models) <- as.character(components)
    model$lda <- list(
        ncomp = components,
        models = models,
        ridge = as.numeric(ridge)[1L],
        train_backend = backend
    )
    model
}

.train_lda_from_scores <- function(
    model,
    scores,
    labels,
    ridge,
    backend,
    train_fun = lda_train_prefix_cpp,
    retain_scores = FALSE
) {
    components <- .lda_component_grid(model, ncol(scores))
    models <- train_fun(
        scores[, seq_len(max(components)), drop = FALSE],
        labels,
        length(model$lev),
        components,
        as.numeric(ridge)[1L]
    )
    model <- .attach_lda_models(
        model,
        models,
        components,
        ridge,
        backend
    )
    if (retain_scores) {
        model$Ttrain <- scores[, seq_len(max(components)), drop = FALSE]
    }
    model
}

.try_stream_lda <- function(model, Xtrain, labels, ridge, backend) {
    if (is.null(model$R_predict) || is.null(model$R_offset)) {
        return(NULL)
    }
    components <- .lda_component_grid(model, ncol(model$R_predict))
    score_mb <- nrow(Xtrain) * max(components) * 8 / 1024^2
    if (!backend %in% c("cuda", "metal") && score_mb < 512) {
        return(NULL)
    }
    models <- .lda_train_projected_stream(
        Xtrain,
        model$R_predict[, seq_len(max(components)), drop = FALSE],
        as.numeric(model$R_offset)[seq_len(max(components))],
        labels,
        length(model$lev),
        components,
        ridge = ridge,
        backend = backend
    )
    .attach_lda_models(
        model,
        models,
        components,
        ridge,
        paste0(backend, "_stream_project")
    )
}

.lda_projection_trainer <- function(classifier) {
    namespace <- asNamespace("fastPLS")
    if (classifier == "lda_cuda") {
        available <- .cuda_matmul_available() &&
            exists("lda_project_train_prefix_cuda", namespace, inherits = FALSE)
        if (!available) {
            stop(
                "CUDA projected LDA fitting is unavailable. ",
                "No CPU fallback is performed.",
                call. = FALSE
            )
        }
        return(list(
            fun = get("lda_project_train_prefix_cuda", namespace),
            backend = "cuda_project"
        ))
    }
    if (exists("lda_project_train_prefix_cpp", namespace, inherits = FALSE)) {
        return(list(
            fun = get("lda_project_train_prefix_cpp", namespace),
            backend = "cpp_project"
        ))
    }
    NULL
}

.try_projected_lda <- function(model, Xtrain, labels, ridge, classifier) {
    trainer <- .lda_projection_trainer(classifier)
if (is.null(trainer) || is.null(model$R_predict) || is.null(model$R_offset)) {
        return(NULL)
    }
    projection <- as.matrix(model$R_predict)
    components <- .lda_component_grid(model, ncol(projection))
    models <- trainer$fun(
        as.matrix(Xtrain),
        projection[, seq_len(max(components)), drop = FALSE],
        as.numeric(model$R_offset)[seq_len(max(components))],
        labels,
        length(model$lev),
        components,
        as.numeric(ridge)[1L]
    )
    .attach_lda_models(
        model,
        models,
        components,
        ridge,
        trainer$backend
    )
}

.fallback_lda_fit <- function(model, Xtrain, labels, ridge, classifier) {
    scores <- model$Ttrain
    if (
        is.null(scores) ||
            !length(scores) ||
            ncol(as.matrix(scores)) < max(model$ncomp)
    ) {
        scores <- .fastpls_latent_scores(
            model,
            Xtrain,
            max(model$ncomp),
            "cpu"
        )
        model$Ttrain <- scores
    }
    namespace <- asNamespace("fastPLS")
    if (classifier == "lda_cuda" &&
        !exists("lda_train_prefix_cuda", namespace, inherits = FALSE)) {
        stop(
            "CUDA LDA fitting is unavailable. No CPU fallback is performed.",
            call. = FALSE
        )
    }
    train_fun <- if (classifier == "lda_cuda") {
        get("lda_train_prefix_cuda", namespace)
    } else {
        lda_train_prefix_cpp
    }
    .train_lda_from_scores(
        model,
        as.matrix(scores),
        labels,
        ridge,
        classifier,
        train_fun = train_fun
    )
}

.attach_lda_after_stream <- function(model, Xtrain, labels, classifier,
    lda_ridge) {
    if (classifier == "lda_metal") {
        scores <- .fastpls_latent_scores(model, Xtrain, max(model$ncomp),
            "metal")
        return(.train_lda_from_scores(model, scores, labels, lda_ridge,
            "metal_project_cpp_lda",
            retain_scores = TRUE))
    }
    cuda_scores <- classifier == "lda_cpp" &&
        identical(model$flash_svd_backend,
            "cuda") && .cuda_matmul_available()
    if (cuda_scores) {
        scores <- .fastpls_latent_scores(model, Xtrain, max(model$ncomp),
            "cuda")
        return(.train_lda_from_scores(model, scores, labels, lda_ridge,
            "cpp_on_cuda_scores"))
    }
    projected <- .try_projected_lda(model, Xtrain, labels, lda_ridge,
        classifier)
    if (!is.null(projected)) {
        return(projected)
    }
    .fallback_lda_fit(model, Xtrain, labels, lda_ridge, classifier)
}

.attach_lda_classifier <- function(
    model,
    Xtrain,
    Ytrain,
    classifier = "argmax",
    lda_ridge = 1e-8
) {
    classifier <- .resolve_classifier_for_backend(classifier, "cpu")
    resolved <- .resolve_lda_backend(model, classifier)
    model <- resolved$model
    classifier <- resolved$classifier
    if (!isTRUE(model$classification) || classifier == "argmax") {
        return(model)
    }
    if (!is.factor(Ytrain)) {
        stop("Classification head requires factor Ytrain", call. = FALSE)
    }
    model <- .attach_latent_projection_cache(model)
    labels <- as.integer(factor(Ytrain, levels = model$lev))
    if (anyNA(labels)) {
        stop("LDA labels are outside the training levels", call. = FALSE)
    }
    backend <- switch(classifier, lda_cuda = "cuda", lda_metal = "metal", "cpu")
    streamed <- .try_stream_lda(
        model,
        Xtrain,
        labels,
        lda_ridge,
        backend
    )
    if (!is.null(streamed)) {
        return(streamed)
    }

    .attach_lda_after_stream(
        model,
        Xtrain,
        labels,
        classifier,
        lda_ridge
    )
}

.fastpls_return_lda_scores <- function() {
    opt <- getOption("fastPLS.return_lda_scores", NULL)
    if (!is.null(opt)) {
        return(isTRUE(opt))
    }
    env <- tolower(Sys.getenv("FASTPLS_RETURN_LDA_SCORES", "false"))
    env %in% c("1", "true", "yes", "y")
}

.fastpls_one_hot_labels <- function(y, lev) {
    y <- as.character(y)
    idx <- match(y, lev)
    out <- matrix(0, nrow = length(y), ncol = length(lev))
    colnames(out) <- lev
    ok <- !is.na(idx)
    if (any(ok)) {
        out[cbind(which(ok), idx[ok])] <- 1
    }
    out
}

.fastpls_accuracy_from_class_labels <- function(lev, Ytest, Ypredlab) {
    vapply(
        seq_along(Ypredlab),
        function(i) {
            pred <- factor(as.character(Ypredlab[[i]]), levels = lev)
            obs <- factor(as.character(Ytest), levels = lev)
            mean(pred == obs, na.rm = TRUE)
        },
        numeric(1)
    )
}

.fastpls_ncomp_names <- function(ncomp) {
    paste0("ncomp=", as.integer(ncomp))
}

.fastpls_q2_from_reference <- function(observed, predicted, reference_mean) {
    observed <- as.matrix(observed)
    predicted <- as.matrix(predicted)
    if (!all(dim(observed) == dim(predicted))) {
        stop(
            "observed and predicted must have the same dimensions for Q2.",
            call. = FALSE
        )
    }
    reference_mean <- as.numeric(reference_mean)
    if (length(reference_mean) != ncol(observed)) {
        stop(
            "reference_mean must contain one value per response column.",
            call. = FALSE
        )
    }
    press <- sum((observed - predicted)^2, na.rm = TRUE)
    tss <- sum(sweep(observed, 2L, reference_mean, "-")^2, na.rm = TRUE)
    if (is.finite(tss) && tss > 0) 1 - press / tss else NA_real_
}

.fastpls_fold_q2_path <- function(Ytrue, Ypred, fold) {
    Ytrue <- .float32_to_numeric_matrix(Ytrue)
    dims <- dim(Ypred)
    if (is.null(dims)) {
        Ypred <- matrix(Ypred, nrow = nrow(Ytrue), ncol = ncol(Ytrue))
        dims <- dim(Ypred)
    }
    if (length(dims) == 2L) {
        Ypred <- array(Ypred, dim = c(dims, 1L))
        dims <- dim(Ypred)
    }
    if (length(dims) != 3L || dims[[1L]] != nrow(Ytrue) ||
        dims[[2L]] != ncol(Ytrue) ||
        length(fold) != nrow(Ytrue)) {
        stop(
            "Fold-aware Q2 requires matched responses, predictions, and folds.",
            call. = FALSE
        )
    }
    fold <- as.integer(factor(fold, levels = unique(fold)))
    vapply(seq_len(dims[[3L]]), function(component) {
        press <- 0
        tss <- 0
        pred <- matrix(Ypred[, , component], nrow = dims[[1L]],
            ncol = dims[[2L]])
        for (f in sort(unique(fold))) {
            test <- fold == f
            train <- !test
            if (!any(test) || !any(train)) {
                next
            }
            center <- colMeans(Ytrue[train, , drop = FALSE], na.rm = TRUE)
            press <- press + sum((Ytrue[test, , drop = FALSE] - pred[test, ,
                drop = FALSE])^2,
            na.rm = TRUE)
            tss <- tss + sum(sweep(Ytrue[test, , drop = FALSE], 2L, center,
                "-")^2,
            na.rm = TRUE)
        }
        if (is.finite(tss) && tss > 0)
            1 - press / tss
        else NA
    }, numeric(1L))
}

.fastpls_metric_definitions <- function(
    context = c("pls", "single_cv", "double_cv"), classification = FALSE
) {
    context <- match.arg(context); r2 <- if (classification) {
        paste0(
            "Training-set dummy-response R2; denominator uses training class ",
            "proportions. This is not classification accuracy."
        )
    } else {
        paste0(
        "Training-set R2; denominator uses the mean of responses fitted on ",
            "the complete training set."
        )
    }
    q2 <- switch(
        context,
        pls = if (classification) {
            paste0(
            "Independent-test dummy-response Q2; denominator uses training ",
                "class proportions. This is not classification accuracy."
            )
        } else {
            "Independent-test Q2; denominator uses the training-response mean."
        },
        single_cv = if (classification) {
            paste0(
        "Cross-validated dummy-response Q2; each held-out fold is centered ",
                "on its fold-training class proportions. This is not ",
                "classification accuracy."
            )
        } else {
            paste0(
                "Cross-validated Q2; each held-out fold is centered on its ",
                "corresponding fold-training response mean."
            )
        },
        double_cv = if (classification) {
            paste0(
            "Mean outer-fold independent-test dummy-response Q2; each test ",
            "fold uses outer-training class proportions. This is not accuracy."
            )
        } else {
            paste0(
        "Outer cross-validated Q2; each held-out outer fold is centered on ",
                "its corresponding outer-training response mean."
            )
        }
    )
    list(R2Y = r2, Q2Y = q2)
}

.fastpls_name_metric_path <- function(x, ncomp) {
    if (is.null(x)) {
        return(x)
    }
    if (length(x) != length(ncomp)) {
        return(x)
    }
    out <- as.numeric(x)
    names(out) <- .fastpls_ncomp_names(ncomp)
    out
}

.fastpls_name_pls_metric_paths <- function(x, ncomp = NULL) {
    if (is.null(ncomp)) {
        ncomp <- x$ncomp
    }
    if (is.null(ncomp)) {
        return(x)
    }
    for (field in c("accuracy", "Q2Y", "R2Y")) {
        if (!is.null(x[[field]])) {
            x[[field]] <- .fastpls_name_metric_path(x[[field]], ncomp)
        }
    }
    x
}

.fastpls_component_prediction <- function(x, index, ncomp, classification) {
    if (is.null(x)) {
        return(NULL)
    }
    key <- .fastpls_ncomp_names(ncomp)[[index]]
    if (is.list(x) && !is.data.frame(x)) {
        if (!is.null(names(x)) && key %in% names(x)) {
            return(x[[key]])
        }
        if (length(x) == length(ncomp)) {
            return(x[[index]])
        }
    }
    if (is.array(x) && length(dim(x)) == 3L && dim(x)[3L] >= index) {
        return(x[, , index, drop = TRUE])
    }
    if (
        isTRUE(classification) &&
            (is.data.frame(x) || is.matrix(x)) &&
            ncol(x) == length(ncomp)
    ) {
        return(x[, index])
    }
    if (length(ncomp) == 1L) {
        return(x)
    }
    NULL
}

.fastpls_evaluate_component_path <- function(observed, predicted, ncomp,
    ytrain = NULL,
    bycol = FALSE) {
    if (is.null(observed) || is.null(predicted) || !length(ncomp)) {
        return(NULL)
    }
    classification <- is.factor(observed) || is.character(observed)
    observed_eval <- if (classification) {
        observed
    }
    else {
        .float32_to_numeric_matrix(observed)
    }
    ytrain_eval <- if (classification || is.null(ytrain)) {
        NULL
    }
    else {
        .float32_to_numeric_matrix(ytrain)
    }
    out <- lapply(seq_along(ncomp), function(i) {
        predicted_i <- .fastpls_component_prediction(predicted, i, ncomp,
            classification)
        if (is.null(predicted_i)) {
            return(NULL)
        }
        tryCatch(evaluate(observed = observed_eval,
            predicted = if (classification) {
                predicted_i
            }
            else {
                .float32_to_numeric_matrix(predicted_i)
            }, task = if (classification)
                "classification"
            else "regression", ytrain = ytrain_eval, bycol = isTRUE(bycol)),
        error = function(e) {
            list(task = if (classification) "classification" else "regression",
                error = conditionMessage(e))
        })
    })
    names(out) <- .fastpls_ncomp_names(ncomp)
    out
}

.fastpls_attach_pls_metrics <- function(
    model,
    Ytrain,
    Ytest = NULL,
    bycol = FALSE
) {
    ncomp <- as.integer(model$ncomp)
    classification <- is.factor(Ytrain) || is.character(Ytrain)
    model$metrics <- list(
        definitions = .fastpls_metric_definitions("pls", classification),
        fitted = .fastpls_evaluate_component_path(
            observed = Ytrain,
            predicted = model$Yfit,
            ncomp = ncomp,
            ytrain = Ytrain,
            bycol = bycol
        ),
        test = .fastpls_evaluate_component_path(
            observed = Ytest,
            predicted = model$Ypred,
            ncomp = ncomp,
            ytrain = Ytrain,
            bycol = bycol
        )
    )
    if (!is.null(model$permutation)) {
        model$metrics$permutation <- list(
            results = model$permutation,
            p_value = model$pval,
            unit = model$permutation_unit,
            group_sizes_preserved = model$permutation_group_sizes_preserved,
            class_frequencies_preserved =
                model$permutation_class_frequencies_preserved,
            folds = model$permutation_folds,
            solver_seed = model$permutation_solver_seed,
            requested = model$permutation_requested,
            completed = model$permutation_completed,
            failed = model$permutation_failed,
            completed_by_component = model$permutation_completed_by_component,
            failed_by_component = model$permutation_failed_by_component,
            errors = model$permutation_errors
        )
    }
    model
}

.fastpls_attach_single_cv_metrics <- function(res, Ydata, fit, bycol = FALSE) {
    classification <- is.factor(Ydata) || is.character(Ydata)
    res$metrics <- list(
        definitions = .fastpls_metric_definitions("single_cv", classification),
        cross_validated = .fastpls_evaluate_component_path(
            observed = Ydata,
            predicted = res$pred,
            ncomp = as.integer(res$ncomp),
            ytrain = Ydata,
            bycol = bycol
        ),
        fitted = if (isTRUE(fit)) {
            .fastpls_evaluate_component_path(
                observed = Ydata,
                predicted = res$Yfit,
                ncomp = as.integer(res$ncomp),
                ytrain = Ydata,
                bycol = bycol
            )
        } else {
            NULL
        }
    )
    if (
    !classification && length(res$Q2Y) == length(res$metrics$cross_validated)
    ) {
        for (i in seq_along(res$metrics$cross_validated)) {
            if (!is.null(res$metrics$cross_validated[[i]]$metrics)) {
                res$metrics$cross_validated[[i]]$metrics$Q2 <- res$Q2Y[[i]]
                res$metrics$cross_validated[[i]]$q2_definition <-
                    "cross_validated_fold_training_mean"
            }
        }
    }
    res
}

.fastpls_double_cv_evaluate <- function(
    predicted,
    Ydata,
    classification,
    bycol
) {
    tryCatch(
        evaluate(
            observed = Ydata,
            predicted = predicted,
            task = if (classification) "classification" else "regression",
            ytrain = if (classification) NULL else Ydata,
            bycol = isTRUE(bycol)
        ),
        error = function(e) {
            list(
                task = if (classification) {
                    "classification"
                } else {
                    "regression"
                },
                error = conditionMessage(e)
            )
        }
    )
}

.fastpls_double_cv_permutation_metrics <- function(res) {
    if (!is.null(res$permutation_sampled)) {
        return(list(
            metric = res$permutation_metric,
            observed = res$permutation_observed,
            permuted = res$permutation_sampled,
            p_value = res$p.value,
            unit = res$permutation_unit,
            group_sizes_preserved = res$permutation_group_sizes_preserved,
    class_frequencies_preserved = res$permutation_class_frequencies_preserved,
            folds = res$permutation_folds,
            solver_seed = res$permutation_solver_seed,
            requested = res$permutation_requested,
            completed = res$permutation_completed,
            failed = res$permutation_failed,
            errors = res$permutation_errors
        ))
    }
    if (!is.null(res$Q2Ysampled)) {
        return(list(
            metric = "Q2Y",
            observed = res$Q2Y,
            permuted = res$Q2Ysampled,
            p_value = res$p.value
        ))
    }
    NULL
}

.fastpls_attach_double_cv_metrics <- function(res, Ydata, bycol = FALSE) {
    classification <- is.factor(Ydata) || is.character(Ydata)
    run_metrics <- lapply(res$results, function(run) {
        predicted <- run$pred %||% run$Ypred
        .fastpls_double_cv_evaluate(predicted, Ydata, classification, bycol)
    })
    names(run_metrics) <- paste0("run=", seq_along(run_metrics))
    aggregate <- .fastpls_double_cv_evaluate(
        res$Ypred,
        Ydata,
        classification,
        bycol
    )
    res$metrics <- list(
        definitions = .fastpls_metric_definitions("double_cv", classification),
        cross_validated = run_metrics,
        aggregate = aggregate
    )
    permutation <- .fastpls_double_cv_permutation_metrics(res)
    if (!is.null(permutation)) {
        res$metrics$permutation <- permutation
    }
    res
}

.fastpls_hidden_output_fields <- c(
    "p",
    "m",
    "ncomp",
    "B_stored",
    "compact_prediction",
    "pls_method",
    "requested_pls_method",
    "method_substitution_reason",
    "predict_latent_ok",
    "xprod_default",
    "predict_backend",
    "flash_svd",
    "flash_svd_backend",
    "flash_svd_mode",
    "flash_block_size",
    "classification",
    "classification_rule",
    "lda_backend",
    "R_predict",
    "R_offset",
    "precision",
    "resident_state",
    "resident_backend",
    "gpu_resident",
    "resident_controls",
    "execution_route",
    "benchmark_phase_timing",
    "rsvd_effective_oversample",
    "rsvd_effective_power"
)

.fastpls_hide_internal_output_fields <- function(x) {
    present <- intersect(.fastpls_hidden_output_fields, names(x))
    if (!length(present)) {
        return(x)
    }
    cls <- class(x)
    internal <- attr(x, "fastPLS_internal", exact = TRUE)
    if (is.null(internal)) {
        internal <- list()
    }
    internal[present] <- x[present]
    x[present] <- NULL
    attr(x, "fastPLS_internal") <- internal
    class(x) <- cls
    x
}

.fastpls_restore_internal_output_fields <- function(x) {
    internal <- attr(x, "fastPLS_internal", exact = TRUE)
    if (is.null(internal) || !length(internal)) {
        return(x)
    }
    cls <- class(x)
    missing <- setdiff(names(internal), names(x))
    if (length(missing)) {
        x[missing] <- internal[missing]
    }
    class(x) <- cls
    x
}

print.fastPLS <- function(x, ...) {
    out <- .fastpls_hide_internal_output_fields(x)
    attr(out, "fastPLS_internal") <- NULL
    class(out) <- setdiff(class(out), "fastPLS")
    print(out, ...)
    invisible(x)
}

.fastpls_public_pls_output <- function(x, ncomp = NULL) {
    x <- .fastpls_name_pls_metric_paths(x, ncomp)
    .fastpls_hide_internal_output_fields(x)
}

.solver_diagnostic_context <- function(model, solver) {
    latent <- if (is.list(model$inner_model)) model$inner_model else model
    requested <- .fastpls_quiet(max(as.integer(latent$ncomp), na.rm = TRUE))
    if (!is.finite(requested)) {
        requested <- NA_integer_
    }
    factors <- Filter(Negate(is.null), list(R = latent$R, Q = latent$Q))
    finite <- if (length(factors)) {
        all(vapply(
            factors,
            function(value) {
            isTRUE(tryCatch(all(is.finite(value)), error = function(e) FALSE))
            },
            logical(1L)
        ))
    } else {
        NA
    }
    effective <- NA_integer_
    if (!is.null(latent$R) && length(dim(latent$R)) == 2L) {
        norms <- tryCatch(
            sqrt(colSums(latent$R * latent$R)),
            error = function(e) numeric()
        )
        effective <- sum(is.finite(norms) & norms > sqrt(.Machine$double.eps))
    }
    randomized <- solver %in% c("cpu_rsvd", "cuda_rsvd", "metal_rsvd")
    audit <- if (randomized) {
        tryCatch(rsvd_audit_summary_debug(), error = function(e) NULL)
    } else {
        NULL
    }
    audited <- !is.null(audit) &&
        audit$solves > 0L &&
        audit$certified == audit$solves &&
        identical(audit$failures, 0L)
    failed <- isFALSE(finite) ||
        (is.finite(requested) && is.finite(effective) && effective < requested)
    list(
        latent = latent,
        requested = requested,
        finite = finite,
        effective = effective,
        randomized = randomized,
        audit = audit,
        audited = audited,
        failed = failed
    )
}

.solver_diagnostic_status <- function(context) {
    if (context$failed) {
        return("failed_structural_check")
    }
    if (!context$randomized) {
        return("deterministic_solver_basic_checks_passed")
    }
    if (!context$audited) {
        return("structural_checks_passed_case_audit_unavailable")
    }
    if (context$audit$deterministic_fallbacks > 0L) {
        "case_audit_passed_with_deterministic_recovery"
    } else {
        "case_audit_passed"
    }
}

.solver_diagnostic_guidance <- function(context) {
    if (!context$randomized) {
        return("This internal reference uses a dense decomposition.")
    }
    if (context$audited) {
        return(paste(
            "Every randomized decomposition met the case-specific checks;",
            "strengthened retries or deterministic recovery are recorded."
        ))
    }
    paste(
        "No case-specific rSVD certificate is available.",
        "Confirm important results across seeds or with an independent",
        "high-accuracy decomposition."
    )
}

.rsvd_diagnostic_record <- function(context, backend, oversample, power, seed) {
    list(
        backend = backend,
        oversample = as.integer(oversample)[1L],
        power = as.integer(power)[1L],
        seed = as.integer(seed)[1L],
        case_audit = context$audit,
        numerical_scope = paste(
            "The recorded checks describe this fit; they do not certify",
            "all matrices with the same controls."
        ),
        validation_failure_criteria = list(
            prediction_relative_error_above = 0.01,
            score_relative_error_above = 0.01,
            prediction_correlation_below = 0.995,
            score_correlation_below = 0.995,
            latent_subspace_angle_degrees_above = 0.1,
            classification_label_agreement_below = 0.995,
            predictive_metric_absolute_difference_above = 0.005
        ),
        setting_guidance = paste(
            "rSVD is approximate; compare important conclusions across",
            "seeds and, when needed, with an independent high-accuracy",
            "decomposition."
        )
    )
}

.simpls_batch_enabled <- function(
    randomized, backend, classification, training_samples,
    predictor_dimension, response_dimension, requested_components
) {
    classification_block <- isTRUE(classification) &&
        is.finite(response_dimension) && response_dimension <= 2048L
    regression_block <- !isTRUE(classification) &&
        is.finite(predictor_dimension) && is.finite(response_dimension) &&
        as.double(predictor_dimension) * as.double(response_dimension) >
            64 * 1024^2
    isTRUE(randomized) && (classification_block || regression_block) &&
        backend %in% c("cpu", "cuda", "metal") &&
        is.finite(training_samples) &&
        training_samples >= 1L && is.finite(response_dimension) &&
        response_dimension > 1L &&
        is.finite(predictor_dimension) &&
        as.double(training_samples) * as.double(predictor_dimension) *
            as.double(response_dimension) >= 5e8 &&
        is.finite(requested_components) && requested_components >= 4L
}

.simpls_batch_width <- function(
    backend, requested_components, predictor_dimension = NA_integer_,
    response_dimension = NA_integer_, oversample = NA_integer_,
    classification = FALSE
) {
    limit <- if (isTRUE(classification)) 64L else 8L
    bounds <- c(limit, requested_components, predictor_dimension,
                response_dimension)
    as.integer(min(bounds[is.finite(bounds)]))
}

.simpls_refresh_rule <- function(
    randomized, batched, backend, fresh_rank_one_backend = NA_character_
) {
    if (batched) {
        return(paste0("batched_", backend, "_candidate_block"))
    }
    if (!is.na(fresh_rank_one_backend)) {
        return(paste0("fresh_", fresh_rank_one_backend, "_rank_one_refresh"))
    }
    if (randomized) {
        return("fresh_oversampled_sketch_per_component")
    }
    "fresh_per_component"
}

.simpls_diagnostic_scalar <- function(value) {
    value <- as.integer(value)
    if (any(is.finite(value))) max(value[is.finite(value)]) else NA_integer_
}

.simpls_rank_one_backend <- function(
    randomized, backend, batched, predictor_dimension, response_dimension,
    requested_components, route_mode, precision = "float64"
) {
    element_bytes <- if (identical(precision, "float32")) 4 else 8
    large_crosscov <- isTRUE(randomized) && !batched &&
        is.finite(requested_components) && requested_components >= 1L &&
        is.finite(predictor_dimension) && is.finite(response_dimension) &&
        as.double(predictor_dimension) * as.double(response_dimension) *
            element_bytes >
            512 * 1024^2
    if (large_crosscov &&
        backend %in% c("cpu", "cuda", "metal")) {
        return(backend)
    }
    metal <- isTRUE(randomized) && identical(backend, "metal") &&
        identical(route_mode, "metal_resident_rank_one_simpls")
    if (metal) return("metal")
    NA_character_
}

.simpls_active_optimizations <- function(rank_one_backend, float32 = FALSE) {
    if (float32) {
        return(c("cached_rank_one_deflation_product",
            "incremental_fitted_path_when_requested", "compact_prediction",
            "float32_buffers",
            if (!is.na(rank_one_backend)) paste0(
                "fresh_", rank_one_backend, "_rank_one_refresh"
            )))
    }
    c(
        "cached_rank_one_deflation_product",
        "incremental_coefficient_path",
        "incremental_fitted_path_when_requested",
        "conditional_crossproduct_cache",
        "compact_prediction",
        "implicit_cross_covariance_when_selected",
        if (!is.na(rank_one_backend)) paste0(
            "fresh_", rank_one_backend, "_rank_one_refresh"
        )
    )
}

.simpls_refresh_iterations <- function(rank_one_backend, power) {
    if (rank_one_backend %in% c("cpu", "cuda", "metal") &&
        is.finite(power)) {
        return(max(1L, power))
    }
    NA_integer_
}

.simpls_seed_rule <- function(randomized, rank_one_backend) {
    if (identical(rank_one_backend, "metal")) {
        return("single_seed_fresh_component_sequence")
    }
    if (randomized) "seed_plus_component_index" else NA_character_
}

.simpls_direction_diagnostics <- function(
    randomized, backend, classification = FALSE,
    training_samples = NA_integer_, response_dimension = NA_integer_,
    predictor_dimension = NA_integer_, requested_components = NA_integer_,
    route_mode = NA_character_, power = NA_integer_, oversample = NA_integer_,
    precision = "float64"
) {
    training_samples <- .simpls_diagnostic_scalar(training_samples)
    response_dimension <- .simpls_diagnostic_scalar(response_dimension)
    predictor_dimension <- .simpls_diagnostic_scalar(predictor_dimension)
    requested_components <- .simpls_diagnostic_scalar(requested_components)
    route_mode <- as.character(route_mode)[1L]
    power <- .simpls_diagnostic_scalar(power)
    float32 <- identical(precision, "float32")
    batched <- .simpls_batch_enabled(
        randomized, backend, classification, training_samples,
        predictor_dimension, response_dimension, requested_components
    )
    block_width <- .simpls_batch_width(
        backend, requested_components, predictor_dimension,
        response_dimension, oversample, classification
    )
    batched <- batched && block_width > 1L
    fresh_rank_one_backend <- .simpls_rank_one_backend(
        randomized, backend, batched, predictor_dimension,
        response_dimension, requested_components, route_mode, precision
    )
    fresh_rank_one <- !is.na(fresh_rank_one_backend)
    list(
        rule = .simpls_refresh_rule(
            randomized, batched, backend, fresh_rank_one_backend
        ),
        directions_per_solve = if (batched) block_width else 1L,
        candidate_block_refresh = batched,
        fresh_start = TRUE,
        refresh_width = if (batched) {
            block_width
        } else if (fresh_rank_one) {
            1L
        } else {
            NA_integer_
        },
        refresh_iterations = if (batched && is.finite(power)) {
            max(1L, power)
        } else {
            .simpls_refresh_iterations(fresh_rank_one_backend, power)
        },
        seed_rule = if (batched) {
            "seed_plus_candidate_block_start"
        } else {
            .simpls_seed_rule(randomized, fresh_rank_one_backend)
        },
        active_optimizations = c(
            .simpls_active_optimizations(fresh_rank_one_backend, float32),
            if (batched) "candidate_block_refresh"
        ),
        approximate_execution = isTRUE(randomized),
        abandoned_optimizations = "adaptive_refresh_policy"
    )
}

.solver_diagnostics_base <- function(context) {
    list(
        solver = if (context$randomized) "rsvd" else "exact",
        stochastic = context$randomized,
        status = .solver_diagnostic_status(context),
        finite_latent_factors = context$finite,
        requested_components = context$requested,
        effective_components = context$effective,
        approximation_audited = !context$randomized || context$audited,
        guidance = .solver_diagnostic_guidance(context)
    )
}

.attach_simpls_direction_diagnostics <- function(
    model, context, family, backend, classification, training_samples, power
) {
    if (!family %in% c("simpls", "simpls_fast", "opls", "kernelpls")) {
        return(model)
    }
    latent <- context$latent
    response_dimension <- model[["m", exact = TRUE]]
    if (is.null(response_dimension)) {
        response_dimension <- latent[["m", exact = TRUE]]
    }
    if (is.null(response_dimension)) {
        response_dimension <- length(
            model[["mY", exact = TRUE]] %||%
                latent[["mY", exact = TRUE]]
        )
    }
    predictor_dimension <- model[["p", exact = TRUE]]
    if (is.null(predictor_dimension)) {
        predictor_dimension <- latent[["p", exact = TRUE]]
    }
    if (is.null(predictor_dimension)) {
        rotations <- latent[["R", exact = TRUE]]
        predictor_dimension <- if (length(dim(rotations)) == 2L) {
            nrow(rotations)
        } else {
            NA_integer_
        }
    }
    model$diagnostics$simpls_direction <- .simpls_direction_diagnostics(
        context$randomized, backend,
        classification = classification,
        training_samples = training_samples,
        response_dimension = response_dimension,
        predictor_dimension = predictor_dimension,
        requested_components = context$requested,
        route_mode = latent[["xprod_mode", exact = TRUE]],
        power = power,
        oversample = model$diagnostics$rsvd$oversample %||% NA_integer_,
        precision = if (inherits(latent$R, "float32")) "float32" else "float64"
    )
    model
}

.enforce_solver_diagnostics <- function(context) {
    if (context$failed) {
        stop(
            "PLS fit failed structural checks: non-finite latent factors or ",
            "fewer effective components than requested.",
            call. = FALSE
        )
    }
    invisible(NULL)
}

.fastpls_attach_solver_diagnostics <- function(
    model,
    svd.method,
    oversample,
    power,
    seed,
    pls_family = NULL,
    classification = FALSE,
    training_samples = NA_integer_,
    execution_backend = NULL
) {
    solver <- .normalize_svd_method(svd.method)
    backend <- execution_backend %||% switch(
        solver,
        cpu_rsvd = "cpu",
        cuda_rsvd = "cuda",
        metal_rsvd = "metal",
        "cpu"
    )
    context <- .solver_diagnostic_context(model, solver)
    model$diagnostics <- .solver_diagnostics_base(context)
    if (context$randomized) {
        model$diagnostics$rsvd <- .rsvd_diagnostic_record(
            context,
            backend,
            oversample,
            power,
            seed
        )
    }
    family <- as.character(
        pls_family %||% context$latent$pls_method %||% model$pls_method %||% ""
    )[1L]
    model <- .attach_simpls_direction_diagnostics(
        model, context, family, backend, classification, training_samples,
        power
    )
    .enforce_solver_diagnostics(context)
    model
}

.fastpls_public_predict_output <- function(x, ncomp = NULL) {
    if (!is.null(ncomp)) {
        x <- .fastpls_name_pls_metric_paths(x, ncomp)
    }
    x[c("predict_backend", "direct")] <- NULL
    x
}

.float32_prediction_input <- function(object, newdata) {
    X <- .as_float32_matrix(newdata, "newdata")
    .float32_standardize(X, object$mX, object$vX)
}

.float32_multiply <- function(left, right, backend = "cpu") {
    if (identical(backend, "metal")) {
        value <- metal_float32_matrix_multiply_cpp(left, right)
        return(.float32_from_bits(value$C))
    }
    value <- cpu_float32_matrix_multiply_cpp(left, right)
    .float32_from_bits(value$C)
}

.float32_score_cube <- function(rows, levels, slices, needed) {
    if (!needed) {
        return(NULL)
    }
    array(
        NA_real_,
        dim = c(rows, length(levels), slices),
        dimnames = list(NULL, levels, NULL)
    )
}

.float32_prediction_block_size <- function(object, rows) {
    block_size <- object$flash_block_size
    if (is.null(block_size) || !length(block_size) || is.na(block_size)) {
        block_size <- 4096L
    }
    max(1L, min(as.integer(block_size)[1L], as.integer(rows)[1L]))
}

.float32_topk_result <- function(top_index, top_score, object, Ytest) {
    result <- .class_topk_to_labels(
        top_index,
        top_score,
        object$lev,
        object$ncomp
    )
    result$Q2Y <- NULL
    if (!is.null(Ytest)) {
        result$accuracy <- .fastpls_accuracy_from_class_labels(
            object$lev,
            Ytest,
            result$Ypred
        )
        result$Q2Y <- rep(NA_real_, length(object$ncomp))
    }
    .fastpls_name_pls_metric_paths(result, object$ncomp)
}

.float32_lda_topk_prediction <- function(object, Xtest, Ytest, top,
    backend = "cpu") {
    ncomp <- as.integer(object$ncomp)
    keep <- min(as.integer(top)[1L], length(object$lev))
    top_index <- array(
        NA_integer_,
        dim = c(nrow(Xtest), keep, length(ncomp))
    )
    top_score <- array(
        NA_real_,
        dim = c(nrow(Xtest), keep, length(ncomp))
    )
    block_size <- .float32_prediction_block_size(object, nrow(Xtest))
    predict_fun <- .float32_lda_predict_fun(object)
    max_components <- max(ncomp)

    for (start in seq.int(1L, nrow(Xtest), by = block_size)) {
        rows <- start:min(nrow(Xtest), start + block_size - 1L)
        block_scores <- .float32_multiply(
            Xtest[rows, , drop = FALSE],
            object$R[, seq_len(max_components), drop = FALSE],
            backend
        )
        for (index in seq_along(ncomp)) {
            k <- ncomp[[index]]
            lda <- object$lda$models[[as.character(k)]]
            if (is.null(lda)) {
                stop(
                    "No fitted float32 LDA classifier for ncomp=",
                    k,
                    call. = FALSE
                )
            }
            value <- predict_fun(
                block_scores[, seq_len(k), drop = FALSE],
                lda,
                TRUE
            )
            scores <- if (.is_float32(value$scores)) {
                value$scores
            } else {
                .float32_from_bits(value$scores)
            }
            ranked <- float32_topk_cpp(scores, top)
            top_index[rows, , index] <- ranked$top_index
            top_score[rows, , index] <- ranked$top_score
        }
    }
    .float32_topk_result(top_index, top_score, object, Ytest)
}

.float32_argmax_topk_prediction <- function(object, Xtest, Ytest, top,
    backend = "cpu") {
    ncomp <- as.integer(object$ncomp)
    keep <- min(as.integer(top)[1L], length(object$lev))
    top_index <- array(
        NA_integer_,
        dim = c(nrow(Xtest), keep, length(ncomp))
    )
    top_score <- array(
        NA_real_,
        dim = c(nrow(Xtest), keep, length(ncomp))
    )
    block_size <- .float32_prediction_block_size(object, nrow(Xtest))
    incremental <- !identical(object$pls_method, "plssvd") &&
        !is.null(object$Q) &&
        length(ncomp) > 0L &&
        all(diff(ncomp) > 0L)

    for (start in seq.int(1L, nrow(Xtest), by = block_size)) {
        rows <- start:min(nrow(Xtest), start + block_size - 1L)
        Xblock <- Xtest[rows, , drop = FALSE]
        all_scores <- .float32_multiply(
            Xblock,
            object$R[, seq_len(max(ncomp)), drop = FALSE],
            backend
        )
        if (incremental) {
            response <- .float32_zeros(length(rows), length(object$lev))
            previous <- 0L
        }
        for (index in seq_along(ncomp)) {
            k <- ncomp[[index]]
            response_k <- if (incremental) {
                columns <- seq.int(previous + 1L, k)
                response <- response + .float32_multiply(
                    all_scores[, columns, drop = FALSE],
                    t(object$Q[, columns, drop = FALSE]),
                    backend
                )
                previous <- k
                .float32_sweep_cols(response, object$mY, "+")
            } else {
                .float32_response_prediction(
                    object, Xblock, k, all_scores, backend
                )$response
            }
            ranked <- float32_topk_cpp(response_k, top)
            top_index[rows, , index] <- ranked$top_index
            top_score[rows, , index] <- ranked$top_score
        }
    }
    .float32_topk_result(top_index, top_score, object, Ytest)
}

.float32_ranked_output <- function(
    predicted,
    scores,
    object,
    top,
    raw_scores,
    score_name
) {
    result <- list(Ypred = predicted, Q2Y = NULL)
    if (!is.null(scores) && raw_scores) {
        result[[score_name]] <- scores
    }
    if (!is.null(scores) && top > 1L) {
        ranked <- .class_topk_from_score_cube(
            scores,
            object$lev,
            object$ncomp,
            top
        )
        result[names(ranked)] <- ranked
    }
    result
}

.float32_lda_predict_fun <- function(object) {
    cuda_model <- identical(object$lda$train_backend, "float32_cuda_lda")
    cuda <- cuda_model && isTRUE(has_cuda()) && exists(
            "lda_predict_float32_cuda",
            envir = asNamespace("fastPLS"),
            inherits = FALSE
        )
    if (cuda_model && !cuda) {
        stop(
            "This model requires float32 CUDA LDA prediction, but CUDA is ",
            "unavailable. No CPU fallback is performed.",
            call. = FALSE
        )
    }
    if (cuda) {
        return(lda_predict_float32_cuda)
    }
    if (object$lda$train_backend == "float32_portable_lda") {
        return(.float32_portable_lda_predict)
    }
    lda_predict_float32_cpp
}

.float32_lda_prediction <- function(object, Xtest, Ytest, proj, top,
    raw_scores, backend = "cpu") {
    if (top > 1L && !raw_scores && !proj) {
        result <- .float32_lda_topk_prediction(
            object, Xtest, Ytest, top, backend
        )
        if (!is.null(Ytest)) {
            all_scores <- .float32_multiply(
                Xtest,
                object$R[, seq_len(max(object$ncomp)), drop = FALSE],
                backend
            )
            result$Q2Y <- vapply(
                as.integer(object$ncomp),
                function(k) {
                    response <- .float32_response_prediction(
                        object, Xtest, k, all_scores, backend
                    )$response
                    .float32_classification_q2(object, Ytest, response)
                },
                numeric(1L)
            )
        }
        return(.fastpls_name_pls_metric_paths(result, object$ncomp))
    }
    ncomp <- as.integer(object$ncomp)
    predicted <- as.data.frame(matrix(nrow = nrow(Xtest), ncol = length(ncomp)))
    names(predicted) <- .fastpls_ncomp_names(ncomp)
    scores <- .float32_score_cube(nrow(Xtest), object$lev, length(ncomp),
        raw_scores ||
            top > 1L)
    all_scores <- .float32_multiply(
        Xtest,
        object$R[, seq_len(max(ncomp)), drop = FALSE],
        backend
    )
    q2 <- rep(NA_real_, length(ncomp))
    predict_fun <- .float32_lda_predict_fun(object)
    for (index in seq_along(ncomp)) {
        k <- ncomp[[index]]
        lda <- object$lda$models[[as.character(k)]]
        if (is.null(lda)) {
            stop("No fitted float32 LDA classifier for ncomp=", k,
                call. = FALSE)
        }
        value <- predict_fun(all_scores[, seq_len(k), drop = FALSE], lda,
            !is.null(scores))
        predicted[[index]] <- factor(object$lev[as.integer(value$pred)],
            levels = object$lev)
        if (!is.null(scores)) {
            score <- if (.is_float32(value$scores)) {
                value$scores
            }
            else {
                .float32_from_bits(value$scores)
            }
            scores[, , index] <- .float32_to_numeric_matrix(score)
        }
        if (!is.null(Ytest)) {
            response <- .float32_response_prediction(
                object, Xtest, k, all_scores, backend
            )$response
            q2[[index]] <- .float32_classification_q2(
                object, Ytest, response
            )
        }
    }
    result <- .float32_ranked_output(predicted, scores, object, top,
        raw_scores,
        "LDA_scores")
    if (!is.null(Ytest)) {
        result$accuracy <- .fastpls_accuracy_from_class_labels(object$lev,
            Ytest,
            result$Ypred)
        result$Q2Y <- q2
    }
    if (proj) {
        result$Ttest <- all_scores
    }
    .fastpls_name_pls_metric_paths(result, ncomp)
}

.float32_response_prediction <- function(object, Xtest, k, scores = NULL,
    backend = "cpu") {
    scores <- if (is.null(scores)) {
        .float32_multiply(
            Xtest,
            object$R[, seq_len(k), drop = FALSE],
            backend
        )
    } else {
        scores[, seq_len(k), drop = FALSE]
    }
    predicted <- if (object$pls_method == "plssvd") {
        .float32_multiply(
            scores,
            object$W_latent[[.fastpls_ncomp_names(k)]],
            backend
        )
    } else {
        .float32_multiply(
            scores,
            t(object$Q[, seq_len(k), drop = FALSE]),
            backend
        )
    }
    list(
        scores = scores,
        response = .float32_sweep_cols(predicted, object$mY, "+")
    )
}

.float32_classification_q2 <- function(object, Ytest, response) {
    if (is.null(Ytest)) {
        return(NA_real_)
    }
    observed <- float::fl(.fastpls_one_hot_labels(Ytest, object$lev))
    .float32_q2_from_reference(observed, response, object$mY)
}

.float32_argmax_prediction <- function(object, Xtest, Ytest, top, raw_scores,
    proj = FALSE, backend = "cpu") {
    if (top > 1L && !raw_scores) {
        result <- .float32_argmax_topk_prediction(
            object, Xtest, Ytest, top, backend
        )
        if (!is.null(Ytest)) {
            all_scores <- .float32_multiply(
                Xtest,
                object$R[, seq_len(max(object$ncomp)), drop = FALSE],
                backend
            )
            result$Q2Y <- vapply(
                as.integer(object$ncomp),
                function(k) {
                    response <- .float32_response_prediction(
                        object, Xtest, k, all_scores, backend
                    )$response
                    .float32_classification_q2(object, Ytest, response)
                },
                numeric(1L)
            )
        }
        return(.fastpls_name_pls_metric_paths(result, object$ncomp))
    }
    ncomp <- as.integer(object$ncomp)
    predicted <- as.data.frame(matrix(nrow = nrow(Xtest), ncol = length(ncomp)))
    names(predicted) <- .fastpls_ncomp_names(ncomp)
    cube <- .float32_score_cube(
        nrow(Xtest),
        object$lev,
        length(ncomp),
        raw_scores || top > 1L
    )
    all_scores <- .float32_multiply(
        Xtest,
        object$R[, seq_len(max(ncomp)), drop = FALSE],
        backend
    )
    q2 <- rep(NA_real_, length(ncomp))
    for (index in seq_along(ncomp)) {
        value <- .float32_response_prediction(
            object, Xtest, ncomp[[index]], all_scores, backend
        )
        numeric_scores <- if (is.null(cube) && .Platform$OS.type != "windows") {
            NULL
        } else {
            .float32_to_numeric_matrix(value$response)
        }
        labels <- if (is.null(numeric_scores)) {
            float32_argmax_cpp(value$response)
        } else {
            max.col(numeric_scores, ties.method = "first")
        }
        predicted[[index]] <- factor(object$lev[labels], levels = object$lev)
        if (!is.null(cube)) cube[, , index] <- numeric_scores
        if (!is.null(Ytest)) {
            q2[[index]] <- .float32_classification_q2(
                object, Ytest, value$response
            )
        }
    }
    result <- .float32_ranked_output(
        predicted,
        cube,
        object,
        top,
        raw_scores,
        "Yscore"
    )
    if (!is.null(Ytest)) {
        result$accuracy <- .fastpls_accuracy_from_class_labels(
            object$lev,
            Ytest,
            result$Ypred
        )
        result$Q2Y <- q2
    }
    if (proj) {
        result$Ttest <- all_scores
    }
    .fastpls_name_pls_metric_paths(result, ncomp)
}

.float32_regression_prediction <- function(object, Xtest, Ytest, proj,
    backend = "cpu") {
    ncomp <- as.integer(object$ncomp)
    predicted <- vector("list", length(ncomp))
    q2 <- rep(NA_real_, length(ncomp))
    observed <- if (is.null(Ytest)) {
        NULL
    } else {
        .as_float32_matrix(Ytest, "Ytest")
    }
    all_scores <- .float32_multiply(
        Xtest,
        object$R[, seq_len(max(ncomp)), drop = FALSE],
        backend
    )
    for (index in seq_along(ncomp)) {
        value <- .float32_response_prediction(
            object, Xtest, ncomp[[index]], all_scores, backend
        )
        predicted[[index]] <- value$response
        if (!is.null(observed)) {
            q2[[index]] <- .float32_q2_from_reference(
                observed,
                value$response,
                object$mY
            )
        }
    }
    names(predicted) <- .fastpls_ncomp_names(ncomp)
    result <- list(Ypred = predicted, Q2Y = if (is.null(Ytest)) NULL else q2)
    if (proj) {
        result$Ttest <- all_scores
    }
    .fastpls_name_pls_metric_paths(result, ncomp)
}

.predict_fastpls_float32 <- function(
    object,
    newdata,
    Ytest = NULL,
    proj = FALSE,
    top = 1L,
    top5 = FALSE,
    raw_scores = FALSE,
    backend = "cpu"
) {
    .require_float_package()
    top <- .resolve_top_k(top, top5)
    Xtest <- .float32_prediction_input(object, newdata)
    if (!isTRUE(object$classification)) {
        return(.float32_regression_prediction(
            object, Xtest, Ytest, proj, backend
        ))
    }
    if (.is_lda_classifier(object$classification_rule %||% "argmax")) {
        return(.float32_lda_prediction(
            object,
            Xtest,
            Ytest,
            proj,
            top,
            raw_scores,
            backend
        ))
    }
    .float32_argmax_prediction(
        object, Xtest, Ytest, top, raw_scores, proj, backend
    )
}

.fastpls_permutation_cor <- function(Y, idx) {
    Y <- as.matrix(Y)
    if (nrow(Y) != length(idx)) {
        return(NA_real_)
    }
    y0 <- as.numeric(Y)
    yp <- as.numeric(Y[idx, , drop = FALSE])
    ok <- is.finite(y0) & is.finite(yp)
    if (sum(ok) < 2L || stats::sd(y0[ok]) == 0 || stats::sd(yp[ok]) == 0) {
        return(NA_real_)
    }
    as.numeric(stats::cor(y0[ok], yp[ok]))
}

.fastpls_permutation_indices <- function(constrain, times, seed) {
    constrain <- as.integer(as.factor(constrain))
    times <- as.integer(times)[1L]
    if (is.na(times) || times < 1L) {
        stop("times must be a positive integer.", call. = FALSE)
    }
    groups <- split(seq_along(constrain), constrain)
    group_sizes <- lengths(groups)
    strata <- split(seq_along(groups), group_sizes)
    exchangeable <- vapply(strata, length, integer(1L)) > 1L
    if (!any(exchangeable)) {
        stop(
            sprintf(
                "%s %s",
                "No non-trivial exchangeability-block permutation is possible:",
            "at least two constraint groups must have the same number of rows."
            ),
            call. = FALSE
        )
    }

    .with_fastpls_seed(seed, {
        lapply(seq_len(times), function(i) {
            source_group <- seq_along(groups)
            repeat {
                for (stratum in strata[exchangeable]) {
                    source_group[stratum] <- sample(
                        stratum,
                        length(stratum),
                        replace = FALSE
                    )
                }
                if (!identical(source_group, seq_along(groups))) break
            }
            idx <- seq_along(constrain)
            for (target in seq_along(groups)) {
                idx[groups[[target]]] <- groups[[source_group[[target]]]]
            }
            as.integer(idx)
        })
    })
}

.fastpls_permutation_pvalue <- function(
    permuted,
    observed,
    lower_tail = FALSE
) {
    valid <- is.finite(permuted)
    completed <- sum(valid)
    if (!completed || !is.finite(observed)) {
        return(NA_real_)
    }
    extreme <- if (isTRUE(lower_tail)) {
        sum(permuted[valid] <= observed)
    } else {
        sum(permuted[valid] >= observed)
    }
    (extreme + 1) / (completed + 1)
}

.cuda_fused_lda_enabled <- function(Xtest, fit, proj, Ytrain) {
    enabled <- isTRUE(getOption("fastPLS.fused_cuda_lda", FALSE)) ||
        tolower(Sys.getenv("FASTPLS_FUSED_CUDA_LDA", "0")) %in%
            c("1", "true", "yes", "y")
    !is.null(Xtest) &&
        enabled &&
        !fit &&
        !proj &&
        is.factor(Ytrain) &&
        has_cuda() &&
        exists(
            "pls_lda_gpu_native",
            asNamespace("fastPLS"),
            inherits = FALSE
        )
}

.cuda_fused_lda_finalize <- function(
    model,
    method_name,
    levels,
    use_xprod,
    Ytest
) {
    model$classification <- TRUE
    model$lev <- levels
    model$pls_method <- method_name
    model$predict_latent_ok <- TRUE
    model$xprod_default <- isTRUE(use_xprod)
    model <- .enable_flash_prediction(model, "cuda")
    model$predict_backend <- "cuda_fused_lda"
    model$flash_svd_mode <- "fused_pls_lda"
    codes <- model$pred_codes
    model$pred_codes <- NULL
    if (!is.null(codes)) {
        codes <- as.matrix(codes)
        predicted <- .fastpls_prediction_frame(nrow(codes), model$ncomp)
        for (index in seq_len(ncol(codes))) {
            predicted[[index]] <- factor(
                levels[as.integer(codes[, index])],
                levels = levels
            )
        }
        model$Ypred <- predicted
        if (!is.null(Ytest)) {
            model$accuracy <- .fastpls_accuracy_from_class_labels(
                levels,
                Ytest,
                predicted
            )
            model$Q2Y <- rep(NA_real_, length(model$ncomp))
        }
    }
    class(model) <- "fastPLS"
    .attach_backend_control(model)
}

.cuda_fused_lda_call <- function(
    Xtrain,
    Ytrain,
    labels,
    Xtest,
    ncomp,
    levels,
    method,
    scaling,
    xprod,
    controls
) {
    pls_lda_gpu_native(
        as.matrix(Xtrain),
        as.matrix(Ytrain),
        as.integer(labels),
        as.matrix(Xtest),
        as.integer(ncomp),
        length(levels),
        as.integer(method),
        as.integer(scaling),
        isTRUE(xprod),
        isTRUE(controls$fit),
        as.integer(controls$oversample),
        as.integer(controls$power),
        as.numeric(controls$tolerance)[1L],
        as.integer(controls$seed)[1L],
        as.numeric(controls$ridge)[1L]
    )
}

.cuda_fused_lda_execute <- function(
    fit_expr,
    method_id,
    use_xprod_default,
    gpu_device_state,
    gpu_qr,
    gpu_eig,
    gpu_finalize_threshold
) {
    tryCatch(
        .with_gpu_native_options(
            if (
                identical(as.integer(method_id), 3L) &&
                    isTRUE(use_xprod_default)
            ) {
                .with_simpls_gpu_xprod(fit_expr())
            } else {
                fit_expr()
            },
            gpu_device_state = gpu_device_state,
            gpu_qr = gpu_qr,
            gpu_eig = gpu_eig,
            gpu_finalize_threshold = gpu_finalize_threshold
        ),
        error = function(error) {
            warning(
                "Native CUDA PLS+LDA failed; using the standard CUDA path: ",
                conditionMessage(error),
                call. = FALSE
            )
            NULL
        }
    )
}

.try_cuda_native_lda_fit_predict <- function(method_id, method_name, Xtrain,
    Ytrain,
    Ytrain_original, Xtest, Ytest, ncomp, scaling_id, use_xprod_default, fit,
    proj,
    rsvd_oversample, rsvd_power, svds_tol, seed, lda_ridge, lev,
    gpu_device_state = FALSE,
    gpu_qr = TRUE, gpu_eig = TRUE, gpu_finalize_threshold = 32L) {
    if (!.cuda_fused_lda_enabled(Xtest, fit, proj, Ytrain_original)) {
        return(NULL)
    }
    y_codes <- as.integer(factor(Ytrain_original, levels = lev))
    if (anyNA(y_codes)) {
        return(NULL)
    }
    controls <- list(fit = fit, oversample = rsvd_oversample,
        power = rsvd_power,
        tolerance = svds_tol, seed = seed, ridge = lda_ridge)
    fit_expr <- function() {
        .cuda_fused_lda_call(Xtrain, Ytrain, y_codes, Xtest, ncomp, lev,
            method_id,
            scaling_id, use_xprod_default, controls)
    }
    model <- .cuda_fused_lda_execute(fit_expr, method_id, use_xprod_default,
        gpu_device_state,
        gpu_qr, gpu_eig, gpu_finalize_threshold)
    if (is.null(model)) {
        return(NULL)
    }
    .cuda_fused_lda_finalize(model, method_name, lev, use_xprod_default, Ytest)
}

.fastpls_use_direct_lda <- function(Xtest, k, n_classes) {
    n <- nrow(Xtest)
    p <- ncol(Xtest)
    latent_ops <- as.numeric(n) * k * (as.numeric(p) + n_classes)
    direct_ops <- as.numeric(n) * p * n_classes
    is.finite(latent_ops) &&
        is.finite(direct_ops) &&
        direct_ops < 0.5 * latent_ops
}

.fastpls_lda_direct_predict <- function(object, Xtest, ncomp_eff,
    use_cuda = FALSE,
    use_metal = FALSE, return_scores = FALSE) {
    backend <- if (isTRUE(use_cuda)) "cuda" else
        if (isTRUE(use_metal)) "metal" else "cpu"
    .require_lda_compute_backend(backend, "LDA prediction")
    if (length(unique(ncomp_eff)) != 1L || is.null(object$R_predict) ||
        is.null(object$R_offset) ||
        is.null(object$lda) || is.null(object$lda$models)) {
        return(NULL)
    }
    k <- as.integer(ncomp_eff[[1L]])
    lda <- object$lda$models[[as.character(k)]]
    if (is.null(lda) || is.null(lda$linear) || is.null(lda$constants)) {
        return(NULL)
    }
    R_predict <- as.matrix(object$R_predict)
    Xtest <- as.matrix(Xtest)
    linear <- as.matrix(lda$linear)
    constants <- as.numeric(lda$constants)
    if (k < 1L || ncol(R_predict) < k || nrow(R_predict) != ncol(Xtest) ||
        ncol(linear) !=
            k || length(constants) != nrow(linear)) {
        return(NULL)
    }
    n_classes <- nrow(linear)
    if (!.fastpls_use_direct_lda(Xtest, k, n_classes)) {
        return(NULL)
    }
    Rk <- R_predict[, seq_len(k), drop = FALSE]
    W <- Rk %*% t(linear)
    offset <- as.numeric(object$R_offset)[seq_len(k)]
    constants <- constants - drop(offset %*% t(linear))
    scores <- if (isTRUE(use_cuda)) {
        .cuda_matmul(Xtest, W)
    }
    else if (isTRUE(use_metal)) {
        .metal_mm(Xtest, W)
    }
    else {
        Xtest %*% W
    }
    scores <- sweep(scores, 2L, constants, "+", check.margin = FALSE)
    pred <- max.col(scores, ties.method = "first")
    list(pred = pred, scores = if (isTRUE(return_scores)) scores else NULL,
        direct = TRUE)
}

.fastpls_prediction_frame <- function(n, ncomp) {
    output <- as.data.frame(matrix(nrow = n, ncol = length(ncomp)))
    colnames(output) <- paste0("ncomp=", ncomp)
    output
}

.lda_cuda_project_predict <- function(object, Xtest, ncomp_eff,
    scores = FALSE) {
    if (!identical(object$classification_rule, "lda_cuda") ||
        !.cuda_matmul_available() ||
        !exists("lda_project_predict_cuda", envir = asNamespace("fastPLS"),
            inherits = FALSE) ||
        is.null(object$R_predict) || is.null(object$R_offset) ||
        is.null(object$lda) ||
        is.null(object$lda$models)) {
        return(NULL)
    }
    Xtest <- as.matrix(Xtest)
    R_predict <- as.matrix(object$R_predict)
    if (nrow(R_predict) != ncol(Xtest)) {
        return(NULL)
    }
    ncomp_eff <- pmax(as.integer(ncomp_eff), 1L)
    if (any(!is.finite(ncomp_eff)) || max(ncomp_eff) > ncol(R_predict)) {
        return(NULL)
    }
    Ypredlab <- .fastpls_prediction_frame(nrow(Xtest), object$ncomp)
    score_cube <- if (isTRUE(scores)) {
        array(NA, dim = c(nrow(Xtest), length(object$lev),
            length(object$ncomp)),
        dimnames = list(NULL, object$lev, NULL))
    }
    else {
        NULL
    }
    for (i in seq_along(object$ncomp)) {
        k <- ncomp_eff[i]
        lda <- object$lda$models[[as.character(k)]]
        if (is.null(lda)) {
            return(NULL)
        }
        pred <- .fastpls_lda_project_predict_cuda(Xtest, R_predict[,
            seq_len(k),
            drop = FALSE], as.numeric(object$R_offset)[seq_len(k)], lda,
        return_scores = scores)
        Ypredlab[, i] <- factor(object$lev[as.integer(pred$pred)],
            levels = object$lev)
        if (isTRUE(scores)) {
            score_cube[, , i] <- as.matrix(pred$scores)
        }
    }
    list(Ypred = Ypredlab, lda_scores = score_cube, Ttest = NULL,
        direct = "cuda_project")
}

.lda_prediction_context <- function(object, return_scores) {
    if (is.null(object$lda) || is.null(object$lda$models)) {
        stop("The model does not contain fitted LDA parameters", call. = FALSE)
    }
    if (object$classification_rule == "lda_cuda" &&
        !.cuda_matmul_available()) {
        stop(
            "This model requires CUDA LDA prediction, but CUDA is ",
            "unavailable. No CPU fallback is performed.",
            call. = FALSE
        )
    }
    if (object$classification_rule == "lda_metal") {
        .fastpls_require_backend_available("metal", "This model")
    }
    components <- pmax(
        1L,
        pmin(as.integer(object$ncomp), max(object$lda$ncomp, na.rm = TRUE))
    )
    list(
        components = components,
        max = max(components),
        cuda = object$classification_rule == "lda_cuda",
        metal = object$classification_rule == "lda_metal",
        return_scores = isTRUE(return_scores)
    )
}

.lda_cpp_project_prediction <- function(object, Xtest, components) {
    eligible <- object$classification_rule == "lda_cpp" &&
        !identical(object$flash_svd_backend, "cuda") &&
        !is.null(object$R_predict) &&
        !is.null(object$R_offset)
    if (!eligible) {
        return(NULL)
    }
    X <- as.matrix(Xtest)
    projection <- as.matrix(object$R_predict)
    if (nrow(projection) != ncol(X) || max(components) > ncol(projection)) {
        return(NULL)
    }
    predicted <- .fastpls_prediction_frame(nrow(X), object$ncomp)
    for (index in seq_along(components)) {
        k <- components[[index]]
        lda <- object$lda$models[[as.character(k)]]
        if (is.null(lda)) {
            return(NULL)
        }
        labels <- .fastpls_lda_project_predict_cpp(
            X,
            projection[, seq_len(k), drop = FALSE],
            as.numeric(object$R_offset)[seq_len(k)],
            lda
        )
        predicted[[index]] <- factor(
            object$lev[as.integer(labels)],
            levels = object$lev
        )
    }
    list(
        Ypred = predicted,
        lda_scores = NULL,
        Ttest = NULL,
        direct = "cpp_project"
    )
}

.lda_repeat_direct_prediction <- function(
    object,
    Xtest,
    direct,
    return_scores
) {
    predicted <- .fastpls_prediction_frame(nrow(as.matrix(Xtest)), object$ncomp)
    for (index in seq_along(object$ncomp)) {
        predicted[[index]] <- factor(
            object$lev[as.integer(direct$pred)],
            levels = object$lev
        )
    }
    scores <- if (return_scores) {
        array(
            as.matrix(direct$scores),
            c(nrow(as.matrix(Xtest)), length(object$lev), length(object$ncomp)),
            dimnames = list(NULL, object$lev, NULL)
        )
    } else {
        NULL
    }
    list(Ypred = predicted, lda_scores = scores, Ttest = NULL, direct = TRUE)
}

.lda_prediction_scores <- function(object, Xtest, Ttest, context) {
    if (
    !is.null(Ttest) && length(Ttest) && ncol(as.matrix(Ttest)) >= context$max
    ) {
        return(as.matrix(Ttest)[, seq_len(context$max), drop = FALSE])
    }
    backend <- if (
        (context$cuda ||
            identical(object$flash_svd_backend, "cuda")) &&
            .cuda_matmul_available()
    ) {
        "cuda"
    } else if (context$metal) {
        "metal"
    } else {
        "cpu"
    }
    .fastpls_latent_scores(object, Xtest, context$max, backend)
}

.lda_component_prediction <- function(scores, lda, context) {
    if (context$return_scores && context$cuda) {
        return(.fastpls_lda_predict_cuda(scores, lda))
    }
    if (context$return_scores) {
        return(lda_predict_cpp(scores, lda))
    }
    namespace <- asNamespace("fastPLS")
    if (context$cuda) {
        if (!exists("lda_predict_labels_cuda", namespace, inherits = FALSE)) {
            stop(
                "CUDA LDA label prediction is unavailable. ",
                "No CPU fallback is performed.",
                call. = FALSE
            )
        }
        return(get("lda_predict_labels_cuda", namespace)(scores, lda))
    }
    if (exists("lda_predict_labels_cpp", namespace, inherits = FALSE)) {
        return(get("lda_predict_labels_cpp", namespace)(scores, lda))
    }
    lda_predict_cpp(scores, lda)$pred
}

.lda_score_predictions <- function(object, scores, context) {
    predicted <- .fastpls_prediction_frame(nrow(scores), object$ncomp)
    cube <- if (context$return_scores) {
        array(
            NA_real_,
            c(nrow(scores), length(object$lev), length(object$ncomp)),
            dimnames = list(NULL, object$lev, NULL)
        )
    } else {
        NULL
    }
    for (index in seq_along(context$components)) {
        k <- context$components[[index]]
        lda <- object$lda$models[[as.character(k)]]
        if (is.null(lda)) {
            stop("No fitted LDA classifier for ncomp=", k, call. = FALSE)
        }
        value <- .lda_component_prediction(
            scores[, seq_len(k), drop = FALSE],
            lda,
            context
        )
        labels <- if (context$return_scores) value$pred else value
        predicted[[index]] <- factor(
            object$lev[as.integer(labels)],
            levels = object$lev
        )
        if (context$return_scores) cube[, , index] <- as.matrix(value$scores)
    }
    list(Ypred = predicted, lda_scores = cube, Ttest = scores)
}

.fastpls_lda_predictions <- function(
    object,
    Xtest,
    Ttest = NULL,
    return_scores = .fastpls_return_lda_scores(),
    keep_ttest = FALSE
) {
    context <- .lda_prediction_context(object, return_scores)
    if (context$cuda && is.null(Ttest) && !keep_ttest) {
        value <- .lda_cuda_project_predict(
            object,
            Xtest,
            context$components,
            context$return_scores
        )
        if (!is.null(value)) {
            return(value)
        }
    }
    if (is.null(Ttest) && !keep_ttest && !context$return_scores) {
        value <- .lda_cpp_project_prediction(
            object,
            Xtest,
            context$components
        )
        if (!is.null(value)) {
            return(value)
        }
    }
    if (is.null(Ttest) && !keep_ttest) {
        direct <- .fastpls_lda_direct_predict(
            object,
            Xtest,
            context$components,
            context$cuda,
            context$metal,
            context$return_scores
        )
        if (!is.null(direct)) {
            return(.lda_repeat_direct_prediction(
                object,
                Xtest,
                direct,
                context$return_scores
            ))
        }
    }
    scores <- .lda_prediction_scores(object, Xtest, Ttest, context)
    .lda_score_predictions(object, scores, context)
}

.should_use_cpu_flash_prediction <- function(object, Xtest) {
    if (
    !isTRUE(object$flash_svd) || !identical(object$predict_backend, "cpu_flash")
    ) {
        return(FALSE)
    }
    if (is.null(object$B)) {
        return(TRUE)
    }
    p <- .fastpls_quiet(as.numeric(ncol(Xtest)))
    m <- .fastpls_quiet(as.numeric(object$m))
    k <- .fastpls_quiet(max(as.integer(object$ncomp), na.rm = TRUE))
    if (
        !is.finite(p) ||
            !is.finite(m) ||
            !is.finite(k) ||
            p <= 0 ||
            m <= 0 ||
            k <= 0
    ) {
        return(FALSE)
    }
    dense_b_mb <- p * m * 8 / 1024^2
    min_b_mb <- .fastpls_quiet(
        as.numeric(Sys.getenv("FASTPLS_PREDICT_LATENT_MIN_B_MB", "256"))
    )
    if (!is.finite(min_b_mb) || min_b_mb < 0) {
        min_b_mb <- 256
    }
    if (dense_b_mb >= min_b_mb) {
        return(TRUE)
    }
    # For small response dimension, dense X %*% B is often faster than X %*% R_k
    # %*% W_k.
    k <= m
}

.normalize_svd_method <- function(method) {
    if (length(method) > 1L) {
        method <- method[[1L]]
    }
    method <- as.character(method)
    aliases <- c(
        rsvd = "cpu_rsvd",
        cuda = "cuda_rsvd"
    )
    if (method %in% names(aliases)) {
        return(unname(aliases[[method]]))
    }
    method
}

.normalize_public_backend <- function(backend) {
    if (
        !is.null(backend) &&
            length(backend) == 1L &&
            identical(tolower(as.character(backend)), "cpp")
    ) {
        backend <- "cpu"
    }
    backend <- .fastpls_resolve_backend(backend)
    if (length(backend) > 1L) {
        backend <- backend[[1L]]
    }
    backend <- as.character(backend)
    if (identical(backend, "cpp")) {
        backend <- "cpu"
    }
    backend <- match.arg(
        backend,
        c("cpu", "cuda", "metal")
    )
    if (identical(backend, "cpu")) {
        .fastpls_apply_cpu_cores()
    }
    backend
}

.compiled_backend <- function(backend) {
    backend <- .normalize_public_backend(backend)
    if (identical(backend, "cpu")) "cpp" else backend
}

.backend_svd_method <- function(svd.method, backend) {
    svd.method <- .normalize_svd_method(svd.method)
    backend <- .normalize_public_backend(backend)
    if (identical(svd.method, "cpu_rsvd")) {
        return(switch(
            backend,
            cpu = "cpu_rsvd",
            cuda = "cuda_rsvd",
            metal = "metal_rsvd"
        ))
    }
    svd.method
}

.svd_control_defaults <- function() {
    list(
        svd.method = "rsvd",
        rsvd_oversample = 32L,
        rsvd_power = 5L,
        svds_tol = 0,
        seed = 1L
    )
}

.normalize_svd_parameter_list <- function(x, accepted, aliases, label) {
    if (is.null(x)) {
        x <- list()
    }
    if (!is.list(x) || is.data.frame(x)) {
        stop(sprintf("%s must be a named list.", label), call. = FALSE)
    }
    if (length(x) && (is.null(names(x)) || any(!nzchar(names(x))))) {
        stop(sprintf("All entries in %s must be named.", label), call. = FALSE)
    }
    if (length(x)) {
        nm <- names(x)
        hit <- match(nm, names(aliases), nomatch = 0L)
        nm[hit > 0L] <- unname(aliases[hit])
        names(x) <- nm
    }
    duplicated_names <- unique(names(x)[duplicated(names(x))])
    if (length(duplicated_names)) {
        duplicated_text <- paste(duplicated_names, collapse = ", ")
        stop(
            sprintf(
                "SVD control value%s supplied more than once in %s: %s",
                if (length(duplicated_names) == 1L) "" else "s",
                label,
                duplicated_text
            ),
            call. = FALSE
        )
    }
    unknown <- setdiff(names(x), accepted)
    if (length(unknown)) {
        unknown_text <- paste(unknown, collapse = ", ")
        stop(
            sprintf(
                "Unknown entr%s in %s: %s",
                if (length(unknown) == 1L) "y" else "ies",
                label,
                unknown_text
            ),
            call. = FALSE
        )
    }
    x
}

.svd_direct_aliases <- function() {
    c(
        oversample = "rsvd_oversample",
        power = "rsvd_power"
    )
}

.svd_control_from_dots <- function(dots) {
    if (!is.list(dots)) {
        dots <- list()
    }
    list(dots = dots)
}

.check_duplicate_svd_controls <- function(sources, context) {
    supplied <- unlist(sources, use.names = FALSE)
    duplicated <- unique(supplied[duplicated(supplied)])
    if (length(duplicated)) {
        stop(
            sprintf(
                "SVD control value%s supplied more than once in %s: %s",
                if (length(duplicated) == 1L) "" else "s",
                context,
                paste(duplicated, collapse = ", ")
            ),
            call. = FALSE
        )
    }
    unique(supplied)
}

.coerce_svd_control <- function(control) {
    control$svd.method <- as.character(control$svd.method)[1L]
    if (identical(control$svd.method, "rsvd")) {
        control$svd.method <- "cpu_rsvd"
    }
    integer_fields <- c(
        "rsvd_oversample",
        "rsvd_power",
        "seed"
    )
    numeric_fields <- "svds_tol"
    control[integer_fields] <- lapply(
        control[integer_fields],
        function(x) as.integer(x)[1L]
    )
    control[numeric_fields] <- lapply(
        control[numeric_fields],
        function(x) as.numeric(x)[1L]
    )
    control
}

.resolve_svd_control <- function(
    svd.method = NULL,
    dots = list(),
    context = "pls()"
) {
    if (!is.list(dots)) {
        dots <- list()
    }

    defaults <- .svd_control_defaults()
    accepted <- setdiff(names(defaults), "svds_tol")
    dots <- .normalize_svd_parameter_list(
        dots,
        accepted = accepted,
        aliases = .svd_direct_aliases(),
        label = sprintf("... in %s", context)
    )
    direct <- list()
    if (!is.null(svd.method)) {
        direct$svd.method <- svd.method
    }

    direct_dots <- names(dots)
    supplied <- .check_duplicate_svd_controls(
        list(direct = names(direct), dots = direct_dots),
        context
    )

    out <- defaults
    if (length(direct)) {
        out[names(direct)] <- direct
    }
    if (length(direct_dots)) {
        out[direct_dots] <- dots[direct_dots]
    }
    if (!length(out$svd.method) || anyNA(out$svd.method) ||
        any(!out$svd.method %in% c("rsvd", "cpu_rsvd",
            "cuda_rsvd", "metal_rsvd"))) {
        stop(context, " supports svd.method = 'rsvd' only.", call. = FALSE)
    }
    out <- .coerce_svd_control(out)
    if (!out$svd.method %in% c("cpu_rsvd", "cuda_rsvd", "metal_rsvd")) {
        stop(context, " supports svd.method = 'rsvd' only.", call. = FALSE)
    }
    out$supplied <- supplied
    out
}

.rsvd_configuration_qualification <- function(backend, oversample, power) {
    backend <- .normalize_public_backend(backend)
    oversample <- as.integer(oversample)[1L]
    power <- as.integer(power)[1L]

    qualified <- switch(
        backend,
        cpu = oversample >= 32L && power >= 5L,
        cuda = oversample >= 32L && power >= 5L,
        metal = oversample >= 32L && power >= 5L
    )
    qualification_panel <- switch(
        backend,
        cpu = "multi-seed controlled CPU rSVD panel",
        cuda = paste(
            "multi-seed controlled CUDA rSVD panel"
        ),
        metal = "operation-split CPU/Metal rSVD panel"
    )
    list(
        backend = backend,
        oversample = oversample,
        power = power,
        general_use_certified = FALSE,
        qualified_on_prespecified_panel = isTRUE(qualified),
        met_prespecified_panel = isTRUE(qualified),
        qualification_panel = qualification_panel,
        interpretation = paste(
            "Panel agreement is numerical validation evidence,",
            "not a guarantee for",
            "a new matrix. Reliability for an individual fit requires the",
            "case-specific residual audit recorded in model diagnostics."
        )
    )
}

.apply_cuda_rsvd_floor <- function(control, context) {
    requested_oversample <- control$rsvd_oversample
    requested_power <- control$rsvd_power
    control$rsvd_oversample <- max(32L, requested_oversample)
    control$rsvd_power <- max(5L, requested_power)
    if (requested_oversample < 32L || requested_power < 5L) {
        message_format <- paste0(
            "%s raised CUDA rSVD controls from oversample=%d, power=%d to ",
            "the safety floor oversample=%d, power=%d."
        )
        warning(
            sprintf(
                message_format,
                context,
                requested_oversample,
                requested_power,
                control$rsvd_oversample,
                control$rsvd_power
            ),
            call. = FALSE
        )
    }
    control
}

.warn_unqualified_rsvd <- function(qualification, context) {
    if (isTRUE(qualification$qualified_on_prespecified_panel)) {
        return(invisible(NULL))
    }
    if (identical(qualification$backend, "metal")) {
        message_format <- paste0(
            "%s is using Metal rSVD with oversample=%d and power=%d; ",
            "no prespecified Metal qualification panel is available. ",
            "Structural diagnostics do not establish agreement with ",
            "an independent decomposition. Compare important results ",
            "across seeds and against a high-accuracy reference."
        )
        warning(
            sprintf(
                message_format,
                context,
                qualification$oversample,
                qualification$power
            ),
            call. = FALSE
        )
        return(invisible(NULL))
    }
    message_format <- paste0(
        "%s is using an rSVD configuration that did not meet the ",
        "prespecified %s: backend='%s', oversample=%d, power=%d. ",
        "Structural diagnostics do not establish agreement with ",
        "an independent decomposition. Use controls that met the backend ",
        "panel, ",
        "and require the fit-level residual audit or confirm the result ",
        "across seeds and against a high-accuracy reference."
    )
    warning(
        sprintf(
            message_format,
            context,
            qualification$qualification_panel,
            qualification$backend,
            qualification$oversample,
            qualification$power
        ),
        call. = FALSE
    )
}

.accelerated_simpls_family <- function(pls_family) {
    as.character(pls_family %||% "")[1L] %in%
        c("simpls", "opls", "kernelpls")
}

.fast_simpls_response_dimension <- function(Ydata) {
    if (is.factor(Ydata) || is.character(Ydata)) {
        return(length(unique(as.character(Ydata))))
    }
    dimensions <- dim(Ydata)
    if (is.null(dimensions)) 1L else as.integer(dimensions[[2L]])
}

.fast_simpls_shape_profile <- function(Xdata, Ydata, float32 = FALSE) {
    predictor_dimension <- ncol(Xdata)
    response_dimension <- .fast_simpls_response_dimension(Ydata)
    training_samples <- nrow(Xdata)
    classification <- is.factor(Ydata) || is.character(Ydata)
    element_bytes <- if (isTRUE(float32)) 4 else 8
    crosscov_bytes <- as.double(predictor_dimension) *
        as.double(response_dimension) * element_bytes
    massive_crosscov <- !is.finite(crosscov_bytes) ||
        crosscov_bytes > 512 * 1024^2
    sparse_high_class <- classification && response_dimension >= 32L &&
        is.finite(training_samples) && training_samples > 0L &&
        training_samples / response_dimension <= 20
    high_response <- !classification && response_dimension >= 64L &&
        is.finite(training_samples) && training_samples > 0L &&
        response_dimension / training_samples >= 0.2

    if (massive_crosscov) {
        return(list(
            oversample = 12L,
            power = 1L,
            profile = "massive_crosscovariance"
        ))
    }
    if (sparse_high_class) {
        return(list(
            oversample = 64L,
            power = 7L,
            profile = "sparse_high_class_stable"
        ))
    }
    if (high_response) {
        return(list(
            oversample = 48L,
            power = 6L,
            profile = "high_response_stable"
        ))
    }
    list(oversample = 32L, power = 5L, profile = "ordinary_fast")
}

.apply_fast_simpls_shape_controls <- function(
    control,
    pls_family,
    Xdata,
    Ydata
) {
    solver <- .normalize_svd_method(control$svd.method)
    if (!.accelerated_simpls_family(pls_family) ||
            !solver %in% c("cpu_rsvd", "cuda_rsvd", "metal_rsvd")) {
        control$rsvd_profile <- "not_applicable"
        return(control)
    }

    control$rsvd_requested_oversample <- control$rsvd_oversample
    control$rsvd_requested_power <- control$rsvd_power
    explicitly_controlled <- any(
        c("rsvd_oversample", "rsvd_power") %in%
            (control$supplied %||% character())
    )
    if (explicitly_controlled) {
        control$rsvd_profile <- "explicit"
        return(control)
    }

    profile <- .fast_simpls_shape_profile(
        Xdata,
        Ydata,
        float32 = .is_float32(Xdata) || .is_float32(Ydata)
    )
    control$rsvd_oversample <- as.integer(profile[["oversample"]])
    control$rsvd_power <- as.integer(profile[["power"]])
    control$rsvd_profile <- unname(profile[["profile"]])
    control
}

.apply_pls_rsvd_controls <- function(
    control,
    backend,
    context,
    pls_family,
    classification,
    Xdata,
    Ydata
) {
    control <- .apply_fast_simpls_shape_controls(
        control,
        pls_family,
        Xdata,
        Ydata
    )
    .apply_backend_rsvd_controls(
        control,
        backend,
        context,
        pls_family = pls_family,
        classification = classification
    )
}

.apply_backend_rsvd_controls <- function(
    control,
    backend,
    context,
    pls_family = NULL,
    classification = FALSE
) {
    backend <- .normalize_public_backend(backend)
    solver <- .backend_svd_method(control$svd.method, backend)
    if (!solver %in% c("cpu_rsvd", "cuda_rsvd", "metal_rsvd")) {
        control$rsvd_qualification <- NULL
        return(control)
    }
    if (.accelerated_simpls_family(pls_family)) {
        qualification <- .rsvd_configuration_qualification(
            backend,
            control$rsvd_oversample,
            control$rsvd_power
        )
        qualification$execution_profile <-
            "accelerated_randomized_simpls"
        qualification$estimator_interpretation <- paste(
            "Approximate high-speed SIMPLS execution; it is not claimed to",
            "reproduce a deterministic de Jong SIMPLS fit."
        )
        control$rsvd_qualification <- qualification
        return(control)
    }
    if (identical(backend, "cuda")) {
        control <- .apply_cuda_rsvd_floor(control, context)
    } else if (identical(backend, "metal")) {
        control$rsvd_oversample <- max(32L, control$rsvd_oversample)
        control$rsvd_power <- max(5L, control$rsvd_power)
    }
    qualification <- .rsvd_configuration_qualification(
        backend,
        control$rsvd_oversample,
        control$rsvd_power
    )
    control$rsvd_qualification <- qualification
    .warn_unqualified_rsvd(qualification, context)
    control
}

.should_use_xprod_default <- function(p, q, ncomp) {
    p <- as.numeric(p)
    q <- as.numeric(q)
    ncomp <- .fastpls_quiet(max(as.integer(ncomp), na.rm = TRUE))
    if (!is.finite(p) || !is.finite(q) || !is.finite(ncomp)) {
        return(FALSE)
    }
    s_mb <- p * q * 8 / 1024^2
    isTRUE(s_mb > 32)
}

.ablation_xprod_override <- function(value) {
    if (!identical(Sys.getenv("FASTPLS_ABLATION_MODE", "0"), "1")) {
        return(value)
    }
    override <- Sys.getenv("FASTPLS_ABLATION_XPROD", "auto")
    if (override %in% c("1", "true", "TRUE")) {
        return(TRUE)
    }
    if (override %in% c("0", "false", "FALSE")) {
        return(FALSE)
    }
    value
}

.should_store_coefficients <- function(
    p,
    q,
    nslices = 1L,
    compact_prediction_available = TRUE
) {
    mode <- tolower(Sys.getenv("FASTPLS_STORE_B", unset = "auto"))
    if (mode %in% c("always", "1", "true", "yes")) {
        return(TRUE)
    }
    if (mode %in% c("never", "0", "false", "no")) {
        return(FALSE)
    }
    if (!isTRUE(compact_prediction_available)) {
        return(TRUE)
    }
    max_mb <- .fastpls_quiet(
        as.numeric(Sys.getenv("FASTPLS_STORE_B_MAX_MB", unset = "256"))
    )
    if (!is.finite(max_mb) || max_mb < 0) {
        max_mb <- 256
    }
    b_mb <- as.numeric(p) *
        as.numeric(q) *
        max(1L, as.integer(nslices)) *
        8 /
        1024^2
    isTRUE(b_mb <= max_mb)
}

.annotate_coefficient_storage <- function(model, store_B) {
    model$B_stored <- isTRUE(store_B)
    model$compact_prediction <- !isTRUE(store_B)
    model
}

.with_fastpls_seed <- function(seed, expr) {
    old_exists <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    old_seed <- if (old_exists) {
        get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    } else {
        NULL
    }
    on.exit(
        {
            if (old_exists) {
                assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
                rm(".Random.seed", envir = .GlobalEnv)
            }
        },
        add = TRUE
    )
    .fastpls_set_seed(seed)
    force(expr)
}

.fastpls_set_seed <- function(seed) {
    do.call("set.seed", list(as.integer(seed)[1L]))
}

.cuda_matmul_available <- function() {
    is.loaded("_fastPLS_cuda_matrix_multiply", PACKAGE = "fastPLS") &&
        isTRUE(has_cuda())
}

.cuda_matmul <- function(A, B) {
    .Call(
        "_fastPLS_cuda_matrix_multiply",
        as.matrix(A),
        as.matrix(B),
        PACKAGE = "fastPLS"
    )
}

.prepare_response <- function(Ytrain, materialize_labels = TRUE) {
    classification <- is.factor(Ytrain) || is.character(Ytrain)
    if (classification) {
        Ytrain <- droplevels(factor(Ytrain))
    }
    lev <- if (classification) levels(Ytrain) else NULL
    list(
        Ytrain = if (classification && isTRUE(materialize_labels)) {
            transformy(Ytrain)
        } else if (classification) {
            NULL
        } else {
            as.matrix(Ytrain)
        },
        classification = classification,
        lev = lev,
        labels = if (classification) as.integer(Ytrain) else NULL,
        n_classes = if (classification) length(lev) else NULL
    )
}

.fastpls_predictor_input <- function(x, label = "predictor input") {
    if (methods::is(x, "ExpressionSet")) {
        x <- t(Biobase::exprs(x))
    }
    if (is.null(dim(x)) || length(dim(x)) != 2L) {
        stop(label, " must be a matrix-like object or a Biobase ExpressionSet")
    }
    x
}

.is_float32 <- function(x) {
    inherits(x, "float32") || methods::is(x, "float32")
}

.has_float32_input <- function(...) {
    any(vapply(list(...), .is_float32, logical(1)))
}

.float32_response_shape <- function(Ytrain) {
    classification <- is.factor(Ytrain) || is.character(Ytrain)
    q <- if (classification) {
        length(unique(as.character(Ytrain)))
    } else if (length(dim(Ytrain)) == 2L) {
        ncol(Ytrain)
    } else {
        1L
    }
    list(classification = classification, q = as.integer(q))
}

.float32_capability_state <- function(backend) {
    list(
        warnings = character(),
        errors = character(),
        status = "validated",
        execution = if (backend == "cpu") {
            "compiled_cpu"
        } else {
            "device_accelerated"
        }
    )
}

.float32_add_warning <- function(
    state,
    warning,
    status = "experimental",
    execution = NULL
) {
    if (!length(state$errors) && state$status == "validated") {
        state$status <- status
    }
    state$warnings <- c(state$warnings, warning)
    if (!is.null(execution)) {
        state$execution <- execution
    }
    state
}

.float32_platform_rule <- function(
    state,
    backend,
    solver,
    classifier,
    os_type
) {
    if (os_type == "windows") {
        if (backend != "cpu") {
            state$status <- "unavailable"
            state$errors <- "float32 on Windows supports backend = 'cpu' only."
        } else if (solver != "rsvd") {
            state$status <- "unavailable"
            state$errors <- "float32 on Windows supports rSVD only."
        } else {
            state <- .float32_add_warning(
                state,
                paste(
                    "Windows uses portable float-package CPU routes rather",
                    "than native Unix-like single-precision implementations.",
                "OPLS, nonlinear kernel PLS, and LDA use portable CPU stages."
                ),
                "experimental",
                "portable_cpu"
            )
        }
    }
    if (!length(state$errors) && !identical(solver, "rsvd")) {
        state$status <- "unavailable"
        state$errors <- "float32 supports rSVD only."
    }
    state
}

.float32_route_rule <- function(state, method, backend, kernel) {
    unsupported <- FALSE
    if (!length(state$errors) && unsupported) {
        state$status <- "unavailable"
        state$errors <- sprintf(
            paste(
                "float32 %s/%s is unavailable because it does not yet have",
                "a fully device-native implementation."
            ),
            method,
            backend
        )
    }
    state
}

.float32_extreme_rule <- function(
    state,
    method,
    backend,
    q,
    k,
    classification
) {
    if (length(state$errors) || classification || q < 10000L || k < 50L) {
        return(state)
    }
    level <- if (method == "plssvd") "performance-risk" else "numerical-risk"
    status <- "experimental"
    .float32_add_warning(
        state,
        sprintf(
        "float32 %s/%s with q=%d and ncomp=%d is an extreme-response %s route.",
            method,
            backend,
            q,
            k,
            level
        ),
        status
    )
}

.float32_precision_rules <- function(
    state,
    method,
    solver,
    classification,
    kernel
) {
    if (length(state$errors)) {
        return(state)
    }
    if (method == "kernelpls" && kernel != "linear") {
        state <- .float32_add_warning(
            state,
            paste(
                "float32 nonlinear kernel PLS materializes an n-by-n Gram",
                "matrix and can amplify rounding in later components;",
                "confirm important conclusions with float64 when available."
            )
        )
    }
    state
}

.float32_capability_assessment <- function(method, backend, svd_method, q,
    ncomp,
    classification = FALSE, kernel = "linear", classifier = "argmax",
    os_type = .Platform$OS.type) {
    method <- match.arg(method, c("plssvd", "simpls", "opls", "kernelpls"))
    backend <- match.arg(
        backend,
        c("cpu", "cuda", "metal")
    )
    capability_backend <- backend
    classifier <- .normalize_classifier(classifier)
    classifier <- if (.is_lda_classifier(classifier))
        "lda"
    else "argmax"
    solver <- .normalize_svd_method(svd_method)
    if (solver %in% c("cpu_rsvd", "cuda_rsvd", "metal_rsvd")) {
        solver <- "rsvd"
    }
    k <- .fastpls_quiet(max(as.integer(ncomp), na.rm = TRUE))
    if (!is.finite(k)) {
        k <- 0L
    }
    q <- .fastpls_quiet(as.integer(q)[1L])
    if (!is.finite(q) || is.na(q)) {
        q <- 1L
    }
    state <- .float32_capability_state(capability_backend)
    state <- .float32_platform_rule(
        state, capability_backend, solver, classifier, os_type
    )
    state <- .float32_route_rule(
        state, method, capability_backend, kernel
    )
    state <- .float32_extreme_rule(
        state, method, capability_backend, q, k, classification
    )
    state <- .float32_precision_rules(state, method, solver, classification,
        kernel)
    if (identical(backend, "metal")) {
        state$execution <- "fixed_cpu_metal_operation_split"
    }
    list(status = state$status, execution = state$execution,
        action = if (length(state$errors)) {
            "error"
        } else if (length(state$warnings)) {
            "warn"
        } else {
            "allow"
        }, warnings = unique(state$warnings), errors = unique(state$errors))
}

.float32_warning_state <- new.env(parent = emptyenv())

.warn_float32_capability <- function(
    method,
    backend,
    svd_method,
    Ytrain,
    ncomp,
    kernel = "linear",
    classifier = "argmax",
    os_type = .Platform$OS.type
) {
    response <- .float32_response_shape(Ytrain)
    assessment <- .float32_capability_assessment(
        method = method,
        backend = backend,
        svd_method = svd_method,
        q = response$q,
        ncomp = ncomp,
        classification = response$classification,
        kernel = kernel,
        classifier = classifier,
        os_type = os_type
    )
    if (length(assessment$errors)) {
        assessment_errors <- paste(assessment$errors, collapse = " ")
        stop(assessment_errors, call. = FALSE)
    }
    for (message in assessment$warnings) {
        key <- message
        if (!isTRUE(.float32_warning_state[[key]])) {
            .float32_warning_state[[key]] <- TRUE
            warning(message, call. = FALSE)
        }
    }
    invisible(assessment)
}

.require_float_package <- function() {
    if (!requireNamespace("float", quietly = TRUE)) {
        stop("float32 input requires the 'float' package.", call. = FALSE)
    }
}

.as_float32_matrix <- function(x, name = "x") {
    .require_float_package()
    if (.is_float32(x)) {
        if (is.null(dim(x))) {
            return(float::fl(matrix(as.numeric(x), ncol = 1L)))
        }
        return(x)
    }
    if (is.factor(x)) {
        stop(
            sprintf("%s must be numeric before conversion to float32.", name),
            call. = FALSE
        )
    }
    float::fl(as.matrix(x))
}

.float32_zeros <- function(n, p) {
    .float32_from_bits(matrix(0L, nrow = n, ncol = p))
}

.float32_sweep_cols <- function(X, row, op = c("-", "/", "+")) {
    op <- match.arg(op)
    if (!.is_float32(row)) row <- float::fl(as.numeric(row))
    .float32_from_bits(float32_sweep_cols_cpp(X, row,
        match(op, c("-", "/", "+")) - 1L))
}

.float32_standardize <- function(X, center, scale) {
    if (!.is_float32(center)) center <- float::fl(as.numeric(center))
    if (!.is_float32(scale)) scale <- float::fl(as.numeric(scale))
    .float32_from_bits(float32_standardize_cpp(X, center, scale))
}

.float32_to_numeric_matrix <- function(x) {
    if (.is_float32(x)) {
        out <- float::dbl(x)
        if (is.null(dim(out))) {
            out <- matrix(out, ncol = 1L)
        }
        return(out)
    }
    out <- as.matrix(x)
    if (is.null(dim(out))) {
        out <- matrix(out, ncol = 1L)
    }
    out
}

.float32_from_bits <- function(bits) {
    .require_float_package()
    if (is.null(bits)) {
        return(NULL)
    }
    methods::new("float32", Data = bits)
}

.float32_bits_list_to_float <- function(x) {
    if (is.null(x)) {
        return(NULL)
    }
    out <- lapply(x, .float32_from_bits)
    names(out) <- names(x)
    out
}

.wrap_float32_cpp_model <- function(raw) {
    raw$R <- .float32_from_bits(raw$R)
    raw$Q <- .float32_from_bits(raw$Q)
    raw$Ttrain <- .float32_from_bits(raw$Ttrain)
    raw$W_latent <- .float32_bits_list_to_float(raw$W_latent)
    raw$mX <- .float32_from_bits(raw$mX)
    raw$vX <- .float32_from_bits(raw$vX)
    raw$mY <- .float32_from_bits(raw$mY)
    raw$Yfit <- .float32_bits_list_to_float(raw$Yfit)
    raw$predict_latent_ok <- TRUE
    raw$predict_backend <- "float32_cpp"
    raw$precision <- "float32"
    raw$xprod_default <- FALSE
    class(raw) <- "fastPLS"
    raw
}

.float32_train_scores <- function(model, Xtrain) {
    max_k <- max(as.integer(model$ncomp))
    if (
        .is_float32(model$Ttrain) &&
            nrow(model$Ttrain) == nrow(Xtrain) &&
            ncol(model$Ttrain) >= max_k
    ) {
        return(model$Ttrain[, seq_len(max_k), drop = FALSE])
    }
    Xs <- .as_float32_matrix(Xtrain, "Xtrain")
    Xs <- .float32_standardize(Xs, model$mX, model$vX)
    Xs %*% model$R[, seq_len(max_k), drop = FALSE]
}

.float32_lda_train_route <- function(model) {
    cuda_model <- identical(model$predict_backend, "float32_cuda")
    use_cuda <- cuda_model && isTRUE(has_cuda()) && exists(
            "lda_train_prefix_float32_cuda",
            envir = asNamespace("fastPLS"),
            inherits = FALSE
        )
    if (cuda_model && !use_cuda) {
        stop(
            "float32 CUDA LDA fitting is unavailable. ",
            "No CPU fallback is performed.",
            call. = FALSE
        )
    }
    if (use_cuda) {
        return(list(
            fun = lda_train_prefix_float32_cuda,
            backend = "float32_cuda_lda"
        ))
    }
    if (identical(.Platform$OS.type, "windows")) {
        return(list(
            fun = .float32_portable_lda_train_prefix,
            backend = "float32_portable_lda"
        ))
    }
    list(fun = lda_train_prefix_float32_cpp, backend = "float32_cpp_lda")
}

.attach_float32_classifier <- function(
    model,
    Xtrain,
    Ytrain_original,
    classifier,
    lda_ridge = 1e-8
) {
    model$classification_rule <- classifier
    model$lda_backend <- classifier
    if (!isTRUE(model$classification) || identical(classifier, "argmax")) {
        return(model)
    }
    yfac <- factor(Ytrain_original, levels = model$lev)
    y_codes <- as.integer(yfac)
    if (anyNA(y_codes)) {
        stop(
            "float32 classifier received labels outside the training levels",
            call. = FALSE
        )
    }
    Ttrain32 <- .float32_train_scores(model, Xtrain)
    unique_ncomp <- sort(unique(as.integer(model$ncomp)))

    if (.is_lda_classifier(classifier)) {
        route <- .float32_lda_train_route(model)
        lda_models <- route$fun(
            Ttrain32,
            y_codes,
            length(model$lev),
            as.integer(unique_ncomp)
        )
        names(lda_models) <- as.character(unique_ncomp)
        model$lda <- list(
            ncomp = unique_ncomp,
            models = lda_models,
            ridge = vapply(lda_models, `[[`, numeric(1L), "ridge"),
            train_backend = route$backend
        )
        return(model)
    }

    model
}

.float32_col_sd <- function(X) {
    n <- nrow(X)
    if (n < 2L) {
        return(float::fl(matrix(1, nrow = 1L, ncol = ncol(X))))
    }
    mu <- colMeans(X)
    Xc <- .float32_sweep_cols(
        X,
        float::fl(matrix(as.numeric(mu), nrow = 1L)),
        "-"
    )
    out <- sqrt(colSums(Xc * Xc) / (n - 1L))
    out <- float::fl(matrix(as.numeric(out), nrow = 1L))
    out[out == 0] <- 1
    out
}

.float32_center_scale <- function(X, scaling) {
    mX <- float::fl(matrix(0, nrow = 1L, ncol = ncol(X)))
    if (scaling < 3L) {
        mX <- float::fl(matrix(as.numeric(colMeans(X)), nrow = 1L))
        X <- .float32_sweep_cols(X, mX, "-")
    }
    vX <- float::fl(matrix(1, nrow = 1L, ncol = ncol(X)))
    if (scaling == 2L) {
        vX <- .float32_col_sd(X)
        X <- .float32_sweep_cols(X, vX, "/")
    }
    list(X = X, mX = mX, vX = vX)
}

.float32_rq <- function(y, yhat) {
    yd <- .float32_to_numeric_matrix(y)
    pd <- .float32_to_numeric_matrix(yhat)
    denom <- sum((sweep(yd, 2L, colMeans(yd), "-"))^2)
    if (!is.finite(denom) || denom <= 0) {
        return(NA_real_)
    }
    1 - sum((yd - pd)^2) / denom
}

.float32_q2_from_reference <- function(y, yhat, reference_mean) {
    .fastpls_q2_from_reference(
        .float32_to_numeric_matrix(y),
        .float32_to_numeric_matrix(yhat),
        .float32_to_numeric_matrix(reference_mean)
    )
}

.float32_rsvd_raw <- function(A, k, oversample = 32L, power = 5L, seed = 1L) {
    .require_float_package()
    k <- min(max(1L, as.integer(k)[1L]), min(nrow(A), ncol(A)))
    l <- min(ncol(A), k + max(0L, as.integer(oversample)[1L]))
    old_seed <- if (
        exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    ) {
        get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    } else {
        NULL
    }
    on.exit(
        {
            if (is.null(old_seed)) {
            if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
                    rm(".Random.seed", envir = .GlobalEnv)
                }
            } else {
                assign(".Random.seed", old_seed, envir = .GlobalEnv)
            }
        },
        add = TRUE
    )
    .fastpls_set_seed(seed)
    omega <- float::fl(matrix(
        stats::rnorm(ncol(A) * l),
        nrow = ncol(A),
        ncol = l
    ))
    Y <- A %*% omega
    if (power > 0L) {
        for (i in seq_len(as.integer(power))) {
            Qy <- qr.Q(qr(Y))
            Z <- crossprod(A, Qy)
            Qz <- qr.Q(qr(Z))
            Y <- A %*% Qz
        }
    }
    Q <- qr.Q(qr(Y))
    B <- crossprod(Q, A)
    sv <- float::svd(B)
    U <- Q %*% sv$u[, seq_len(k), drop = FALSE]
    list(
        u = U,
        d = sv$d[seq_len(k), , drop = FALSE],
        v = sv$v[, seq_len(k), drop = FALSE]
    )
}

.float32_rsvd_audit <- function(A, candidate, k, audit_k) {
    scale <- max(abs(as.numeric(candidate$d[1L, 1L])), 1e-6)
    residual <- 0
    for (j in seq_len(k)) {
        left <- candidate$u[, j, drop = FALSE]
        right <- candidate$v[, j, drop = FALSE]
        value <- as.numeric(candidate$d[j, 1L])
        residual <- max(
            residual,
            as.numeric(sqrt(sum((A %*% right - left * value)^2))) / scale,
            as.numeric(sqrt(sum((crossprod(A, left) - right * value)^2))) /
                scale
        )
    }
    ratio <- if (audit_k > k && as.numeric(candidate$d[k, 1L]) > 0) {
        abs(as.numeric(candidate$d[k + 1L, 1L] / candidate$d[k, 1L]))
    } else {
        0
    }
    list(residual = residual, ratio = ratio)
}

.float32_rsvd_result <- function(candidate, k, audit, attempt) {
    list(
        u = candidate$u[, seq_len(k), drop = FALSE],
        d = candidate$d[seq_len(k), , drop = FALSE],
        v = candidate$v[, seq_len(k), drop = FALSE],
        case_audited = TRUE,
        case_certified = TRUE,
        deterministic_fallback = FALSE,
        audit_attempts = attempt,
        audit_triplet_residual = audit$residual,
        audit_omitted_direction_ratio = audit$ratio
    )
}

.float32_rsvd <- function(A, k, oversample = 32L, power = 5L, seed = 1L) {
    max_rank <- min(nrow(A), ncol(A))
    k <- min(max(1L, as.integer(k)[1L]), max_rank)
    audit_k <- min(max_rank, k + 1L)
    attempts <- list(
        c(max(32L, oversample), max(5L, power)),
        c(max(48L, oversample), max(6L, power)),
        c(max(64L, oversample), max(7L, power))
    )
    for (i in seq_along(attempts)) {
        ctl <- attempts[[i]]
        candidate <- .float32_rsvd_raw(
            A,
            audit_k,
            ctl[[1L]],
            ctl[[2L]],
            seed + 104729L * (i - 1L)
        )
        audit <- .float32_rsvd_audit(A, candidate, k, audit_k)
        if (
            is.finite(audit$residual) &&
                audit$residual <= 1e-2 &&
                is.finite(audit$ratio) &&
                audit$ratio <= 0.95
        ) {
            return(.float32_rsvd_result(candidate, k, audit, i))
        }
    }

    exact <- float::svd(A)
    list(
        u = exact$u[, seq_len(k), drop = FALSE],
        d = exact$d[seq_len(k), , drop = FALSE],
        v = exact$v[, seq_len(k), drop = FALSE],
        case_audited = TRUE,
        case_certified = TRUE,
        deterministic_fallback = TRUE,
        audit_attempts = length(attempts)
    )
}

.float32_prepare_response <- function(Ytrain, materialize_labels = TRUE) {
    classification <- is.factor(Ytrain) || is.character(Ytrain)
    if (classification) {
        Ytrain <- droplevels(factor(Ytrain))
    }
    lev <- if (classification) levels(Ytrain) else NULL
    labels <- if (classification) as.integer(Ytrain) else NULL
    if (classification && anyNA(labels)) {
        stop("Ytrain contains missing or invalid class labels.", call. = FALSE)
    }
    Y <- if (classification && isTRUE(materialize_labels)) {
        float::fl(transformy(Ytrain))
    } else if (!classification) {
        .as_float32_matrix(Ytrain, "Ytrain")
    } else {
        NULL
    }
    list(
        Ytrain = Y,
        labels = labels,
        n_classes = if (classification) length(lev) else ncol(Y),
        classification = classification,
        lev = lev
    )
}

.float32_cpp_fit_args <- function(
    Xtrain,
    response,
    ncomp,
    scaling,
    method,
    backend,
    svd.method,
    oversample,
    power,
    seed,
    fit
) {
    list(
        .as_float32_matrix(Xtrain, "Xtrain"),
        response,
        as.integer(ncomp),
        as.integer(scaling),
        isTRUE(fit),
        if (identical(method, "plssvd")) 1L else 3L,
        .float32_backend_id(backend),
        .float32_svd_id(svd.method),
        as.integer(oversample),
        as.integer(power),
        as.integer(seed)
    )
}

.float32_finalize_fit <- function(
    raw_model,
    response,
    backend
) {
    if (!backend %in% c("cpu", "metal")) {
        message <- paste(
            "Internal error: nonresident float32 accelerator fitting is",
            "disabled. No hybrid result is returned."
        )
        stop(message, call. = FALSE)
    }
    model <- .wrap_float32_cpp_model(raw_model)
    model$classification <- response$classification
    model$lev <- response$lev
    model$execution_route <- if (identical(backend, "metal")) {
        "CPU/Metal hybrid (operation split)"
    } else {
        "CPU"
    }
    model$predict_backend <- "float32_cpp"
    model
}

.fit_float32_pls <- function(Xtrain, Ytrain, ncomp, scaling, method, backend,
    svd.method,
    rsvd_oversample, rsvd_power, seed, fit, store_scores = fit) {
    use_label_products <- is.factor(Ytrain) || is.character(Ytrain)
    yprep <- .float32_prepare_response(Ytrain,
        materialize_labels = !use_label_products)
    if (identical(method, "plssvd") && isTRUE(yprep$classification)) {
        cap <- .cap_plssvd_ncomp(ncomp, nrow(Xtrain), ncol(Xtrain),
            yprep$n_classes,
            factor_response = TRUE, warn = TRUE)
        ncomp <- cap$ncomp
    }
    response <- if (use_label_products)
        yprep$labels
    else yprep$Ytrain
    fit_args <- .float32_cpp_fit_args(Xtrain, response, ncomp, scaling, method,
        backend, svd.method, rsvd_oversample, rsvd_power,
        seed, fit)
    raw_model <- if (use_label_products && identical(backend, "cpu")) {
        pls_float32_labels_core_cpp(
            fit_args[[1L]], fit_args[[2L]], yprep$n_classes,
            fit_args[[3L]], fit_args[[4L]], fit_args[[5L]], fit_args[[6L]],
            fit_args[[9L]], fit_args[[10L]], fit_args[[11L]],
            store_scores = store_scores
        )
    } else if (use_label_products) {
        do.call(pls_float32_labels_cpp, append(fit_args, yprep$n_classes,
            after = 2L))
    }
    else if (identical(backend, "cpu")) {
        pls_float32_matrix_core_cpp(
            fit_args[[1L]], fit_args[[2L]], fit_args[[3L]], fit_args[[4L]],
            fit_args[[5L]], fit_args[[6L]], fit_args[[9L]], fit_args[[10L]],
            fit_args[[11L]], store_scores = store_scores
        )
    } else {
        do.call(pls_float32_cpu_cpp, fit_args)
    }
    .float32_finalize_fit(raw_model, yprep, backend)
}

.float32_backend_id <- function(backend) {
    switch(backend, cpu = 0L, cuda = 1L, metal = 3L)
}

.float32_product_backend_id <- function(backend) {
    switch(backend, cpu = 0L, cuda = 1L, metal = 2L)
}

.float32_svd_id <- function(svd.method) {
    if (!.normalize_svd_method(svd.method) %in%
        c("cpu_rsvd", "cuda_rsvd", "metal_rsvd")) {
        stop("float32 supports rSVD only.", call. = FALSE)
    }
    3L
}

.float32_outer_model_fields <- function(inner) {
    list(
        ncomp = inner$ncomp,
        Yfit = inner$Yfit,
        R2Y = inner$R2Y,
        classification = inner$classification,
        lev = inner$lev,
        classification_rule = inner$classification_rule,
        precision = "float32",
        predict_backend = inner$predict_backend,
        pls_method = inner$pls_method,
        xprod_mode = inner$xprod_mode,
        gpu_resident = isTRUE(inner$gpu_resident),
        execution_route = inner$execution_route
    )
}

.float32_lda_moments <- function(Ttrain, y, n_classes, kmax) {
    counts <- tabulate(y, nbins = n_classes)
    if (any(counts == 0L)) {
        stop("float32 PLS-LDA received an empty class.", call. = FALSE)
    }
    Tk <- Ttrain[, seq_len(kmax), drop = FALSE]
    means <- .float32_zeros(n_classes, kmax)
    for (class_id in seq_len(n_classes)) {
        means[class_id, ] <- t(colMeans(Tk[y == class_id, , drop = FALSE]))
    }
    pooled_full <- crossprod(Tk)
    for (class_id in seq_len(n_classes)) {
        mu <- means[class_id, , drop = FALSE]
        pooled_full <- pooled_full -
            (t(mu) %*% mu) * float::fl(counts[[class_id]])
    }
    pooled_full <- pooled_full /
        float::fl(max(1L, nrow(Ttrain) - n_classes))
    list(means = means, pooled = pooled_full, counts = counts)
}

.float32_lda_solve <- function(pooled, means) {
    relative_ridges <- c(1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
    k <- ncol(means)
    scale <- sum(vapply(
        seq_len(k),
        function(j) as.numeric(pooled[j, j]),
        numeric(1L)
    )) /
        k
    if (!is.finite(scale) || scale <= 0) {
        scale <- 1
    }
    solved <- NULL
    last_error <- NULL
    ridge <- NA_real_
    relative_ridge <- NA_real_
    for (rho in relative_ridges) {
        ridge_try <- rho * scale
        regularized <- pooled + float::fl(diag(ridge_try, nrow = k))
        chol_factor <- try(float::chol(regularized), silent = TRUE)
        if (inherits(chol_factor, "try-error")) {
            last_error <- as.character(chol_factor)
            next
        }
        solved_try <- try(
            float::backsolve(
                chol_factor,
                float::forwardsolve(t(chol_factor), t(means))
            ),
            silent = TRUE
        )
        if (inherits(solved_try, "try-error")) {
            last_error <- as.character(solved_try)
            next
        }
        solved <- t(solved_try)
        ridge <- ridge_try
        relative_ridge <- rho
        break
    }
    if (is.null(solved)) {
        stop(
            "Portable float32 LDA Cholesky factorization failed. ",
            if (is.null(last_error)) "" else last_error,
            call. = FALSE
        )
    }
    list(linear = solved, ridge = ridge, relative_ridge = relative_ridge)
}

.float32_lda_model <- function(k, moments, sample_count) {
    means <- moments$means[, seq_len(k), drop = FALSE]
    pooled <- moments$pooled[seq_len(k), seq_len(k), drop = FALSE]
    solved <- .float32_lda_solve(pooled, means)
    priors <- moments$counts / sample_count
    constants <- .float32_zeros(1L, length(priors))
    for (class_id in seq_along(priors)) {
        quadratic <- means[class_id, , drop = FALSE] %*%
            t(solved$linear[class_id, , drop = FALSE])
        constants[1L, class_id] <- float::fl(-0.5) *
            quadratic +
            log(float::fl(priors[[class_id]]))
    }
    list(
        means = means,
        linear = solved$linear,
        constants = constants,
        priors = float::fl(matrix(priors, nrow = 1L)),
        ridge = solved$ridge,
        ridge_relative = solved$relative_ridge,
        precision = "float32",
        backend = "portable_float"
    )
}

.float32_portable_lda_train_prefix <- function(Ttrain, y, n_classes, ncomp) {
    Ttrain <- .as_float32_matrix(Ttrain, "Ttrain")
    y <- as.integer(y)
    ncomp <- as.integer(ncomp)
    kmax <- max(ncomp)
    if (length(y) != nrow(Ttrain) || kmax < 1L || kmax > ncol(Ttrain)) {
        stop("Invalid portable float32 LDA dimensions.", call. = FALSE)
    }
    moments <- .float32_lda_moments(Ttrain, y, n_classes, kmax)
    models <- lapply(
        ncomp,
        .float32_lda_model,
        moments = moments,
        sample_count = nrow(Ttrain)
    )
    names(models) <- as.character(ncomp)
    models
}

.float32_portable_lda_predict <- function(Ttest, lda, return_scores = TRUE) {
    Ttest <- .as_float32_matrix(Ttest, "Ttest")
    scores <- Ttest %*% t(lda$linear)
    scores <- .float32_sweep_cols(scores, lda$constants, "+")
    pred <- float32_argmax_cpp(scores)
    list(pred = as.integer(pred), scores = if (return_scores) scores else NULL)
}

.float32_opls_filter <- function(
    Xtrain,
    response,
    north,
    scaling,
    backend,
    svd.method,
    oversample,
    power,
    seed
) {
    filter_backend <- backend
    raw <- if (identical(filter_backend, "cpu")) {
        opls_filter_float32_core_cpp(
            .as_float32_matrix(Xtrain, "Xtrain"), response,
            as.integer(north), as.integer(scaling), as.integer(oversample),
            as.integer(power), as.integer(seed)
        )
    } else {
        opls_filter_float32_backend_core_cpp(
            .as_float32_matrix(Xtrain, "Xtrain"), response,
            as.integer(north), as.integer(scaling),
            .float32_product_backend_id(filter_backend),
            as.integer(oversample),
            as.integer(power), as.integer(seed)
        )
    }
    out <- lapply(
        raw[c("X", "mX", "vX", "W_orth", "P_orth")],
        .float32_from_bits
    )
    out$north <- as.integer(raw$north)
    out$backend <- filter_backend
    out
}

.float32_opls_filter_labels <- function(
    Xtrain,
    labels,
    n_classes,
    north,
    scaling,
    backend,
    svd.method,
    oversample,
    power,
    seed
) {
    filter_backend <- backend
    raw <- if (identical(filter_backend, "cpu")) {
        opls_filter_float32_labels_core_cpp(
            .as_float32_matrix(Xtrain, "Xtrain"), as.integer(labels),
            as.integer(n_classes), as.integer(north), as.integer(scaling)
        )
    } else {
        opls_filter_float32_labels_backend_core_cpp(
            .as_float32_matrix(Xtrain, "Xtrain"), as.integer(labels),
            as.integer(n_classes), as.integer(north), as.integer(scaling),
            .float32_product_backend_id(filter_backend),
            as.integer(oversample),
            as.integer(power), as.integer(seed)
        )
    }
    out <- lapply(
        raw[c("X", "mX", "vX", "W_orth", "P_orth")],
        .float32_from_bits
    )
    out$north <- as.integer(raw$north)
    out$backend <- filter_backend
    out
}

.float32_opls_model <- function(filter, inner, backend) {
    out <- c(
        list(
            inner_model = inner,
            mX = filter$mX,
            vX = filter$vX,
            W_orth = filter$W_orth,
            P_orth = filter$P_orth,
            north = filter$north,
            opls_engine = paste0("float32_", backend),
            opls_filter_engine = paste0(
                "float32_", filter$backend %||% backend
            )
        ),
        .float32_outer_model_fields(inner)
    )
    class(out) <- c("fastPLSOpls", "fastPLS")
    out
}

.fit_float32_resident_inner <- function(
    Xtrain,
    Ytrain,
    ncomp,
    backend,
    svd.method,
    rsvd_oversample,
    rsvd_power,
    seed,
    fit,
    classifier,
    scaling = 3L
) {
    resident_context <- list(
        Xtrain = Xtrain,
        Ytrain = Ytrain,
        backend = backend,
        method = "simpls",
        float32 = TRUE,
        classification = is.factor(Ytrain) || is.character(Ytrain),
        classifier = classifier,
        scaling = c("centering", "autoscaling", "none")[[scaling]],
        scal = as.integer(scaling),
        control = list(
            svd.method = svd.method,
            rsvd_oversample = rsvd_oversample,
            rsvd_power = rsvd_power,
            seed = seed
        )
    )
    resident_config <- list(
        ncomp = ncomp,
        perm.test = FALSE,
        return_loadings = FALSE,
        return_variance = FALSE,
        fit = fit,
        proj = FALSE
    )
    if (identical(backend, "cuda")) {
        .pls_fit_resident_cuda(resident_context, resident_config)
    } else if (identical(backend, "metal")) {
        .pls_fit_resident_metal(resident_context, resident_config)
    } else {
        stop("A resident inner fit requires backend = 'cuda' or 'metal'.")
    }
}

.float32_resident_inner_enabled <- function(Xtrain, Ytrain, backend) {
    identical(backend, "cuda")
}

.fit_float32_opls <- function(
    Xtrain,
    Ytrain,
    ncomp,
    scaling,
    north,
    backend,
    svd.method,
    rsvd_oversample,
    rsvd_power,
    seed,
    fit,
    classifier,
    lda_ridge
) {
    yprep <- .float32_prepare_response(
        Ytrain,
        materialize_labels = !(is.factor(Ytrain) || is.character(Ytrain))
    )
    filt <- if (isTRUE(yprep$classification)) {
        .float32_opls_filter_labels(
            Xtrain, yprep$labels, yprep$n_classes, north, scaling, backend,
            svd.method, rsvd_oversample, rsvd_power, seed
        )
    } else {
        .float32_opls_filter(
            Xtrain, yprep$Ytrain, north, scaling, backend, svd.method,
            rsvd_oversample, rsvd_power, seed
        )
    }
    .opls_require_predictive_rank(ncomp, filt$X, filt$north, scaling < 3L)
    if (.float32_resident_inner_enabled(filt$X, Ytrain, backend)) {
        inner <- .fit_float32_resident_inner(
            filt$X, Ytrain, ncomp, backend, svd.method,
            rsvd_oversample, rsvd_power, seed, fit, classifier
        )
    } else {
        inner <- .fit_float32_pls(
            Xtrain = filt$X,
            Ytrain = Ytrain,
            ncomp = ncomp,
            scaling = 3L,
            method = "simpls",
            backend = backend,
            svd.method = svd.method,
            rsvd_oversample = rsvd_oversample,
            rsvd_power = rsvd_power,
            seed = seed,
            fit = fit,
            store_scores = fit || .is_lda_classifier(classifier)
        )
        inner <- .attach_float32_classifier(
            inner,
            Xtrain = filt$X,
            Ytrain_original = Ytrain,
            classifier = classifier,
            lda_ridge = lda_ridge
        )
    }
    .float32_opls_model(filt, inner, backend)
}

.float32_kernel_matrix <- function(
    Xtrain,
    scaling,
    kernel,
    gamma,
    degree,
    coef0,
    backend
) {
    .kernel_pls_memory_guard(
        nrow(Xtrain), 4,
        sprintf("Float32 nonlinear %s kernel PLS", kernel)
    )
    prep <- .float32_center_scale(
        .as_float32_matrix(Xtrain, "Xtrain"),
        scaling
    )
    gamma <- .kernel_pls_gamma(gamma, prep$X)
    kernel_id <- .kernel_pls_kernel_id(kernel)
    matrix_backend <- backend
    raw <- kernel_matrix_float32_cpp(
        prep$X,
        prep$X,
        kernel_id,
        gamma,
        as.integer(degree),
        coef0,
        .float32_product_backend_id(matrix_backend)
    )
    centered <- center_kernel_train_float32_cpp(.float32_from_bits(raw$K))
    list(
        K = .float32_from_bits(centered$K),
        prep = prep,
        gamma = gamma,
        kernel_id = kernel_id,
        center = list(
            col_means = .float32_from_bits(centered$col_means),
            grand_mean = as.numeric(centered$grand_mean)
        )
    )
}

.float32_kernel_model <- function(
    kernel_data,
    inner,
    kernel,
    degree,
    coef0,
    backend
) {
    out <- c(
        list(
            inner_model = inner,
            Xref = kernel_data$prep$X,
            mX = kernel_data$prep$mX,
            vX = kernel_data$prep$vX,
            kernel = kernel,
            kernel_id = kernel_data$kernel_id,
            gamma = kernel_data$gamma,
            degree = as.integer(degree),
            coef0 = coef0,
            kernel_center = kernel_data$center,
            kernel_engine = paste0("float32_", backend)
        ),
        .float32_outer_model_fields(inner)
    )
    class(out) <- c("fastPLSKernel", "fastPLS")
    out
}

.float32_linear_kernel_fit <- function(
    Xtrain,
    Ytrain,
    ncomp,
    scaling,
    backend,
    svd.method,
    oversample,
    power,
    seed,
    fit,
    classifier,
    lda_ridge
) {
    inner <- if (.float32_resident_inner_enabled(Xtrain, Ytrain, backend)) {
        .fit_float32_resident_inner(
            Xtrain, Ytrain, ncomp, backend, svd.method,
            oversample, power, seed, fit, classifier, scaling
        )
    } else {
        value <- .fit_float32_pls(
            Xtrain, Ytrain, ncomp, scaling, "simpls", backend,
            svd.method, oversample, power, seed, fit,
            store_scores = fit || .is_lda_classifier(classifier)
        )
        .attach_float32_classifier(
            value, Xtrain, Ytrain, classifier, lda_ridge
        )
    }
    inner$kernel <- "linear"
    inner$kernel_engine <- paste0("float32_", backend, "_direct")
    inner$kernel_linear_direct <- TRUE
    inner
}

.fit_float32_kernelpls <- function(Xtrain, Ytrain, ncomp, scaling, kernel,
    gamma,
    degree, coef0, backend, svd.method, rsvd_oversample, rsvd_power, seed, fit,
    classifier, lda_ridge) {
    kernel <- match.arg(kernel, c("linear", "rbf", "poly"))
    if (identical(kernel, "linear")) {
        return(.float32_linear_kernel_fit(Xtrain, Ytrain, ncomp, scaling,
            backend,
            svd.method, rsvd_oversample, rsvd_power, seed, fit, classifier,
            lda_ridge))
    }
    kernel_data <- .float32_kernel_matrix(Xtrain, scaling, kernel, gamma,
        degree,
        coef0, backend)
    inner <- if (.float32_resident_inner_enabled(
        kernel_data$K, Ytrain, backend
    )) {
        .fit_float32_resident_inner(
            kernel_data$K, Ytrain, ncomp, backend, svd.method,
            rsvd_oversample, rsvd_power, seed, fit, classifier
        )
    } else {
        value <- .fit_float32_pls(
            Xtrain = kernel_data$K, Ytrain = Ytrain, ncomp = ncomp,
            scaling = 3L, method = "simpls", backend = backend,
            svd.method = svd.method, rsvd_oversample = rsvd_oversample,
            rsvd_power = rsvd_power, seed = seed, fit = fit,
            store_scores = fit || .is_lda_classifier(classifier)
        )
        .attach_float32_classifier(
            value, Xtrain = kernel_data$K, Ytrain_original = Ytrain,
            classifier = classifier, lda_ridge = lda_ridge
        )
    }
    .float32_kernel_model(kernel_data, inner, kernel, degree, coef0, backend)
}

.normalize_pls_method <- function(method) {
    method <- match.arg(method, c("simpls", "plssvd", "opls", "kernelpls"))
    switch(method, plssvd = 1L, simpls = 3L, opls = 4L, kernelpls = 5L)
}

pls.model1 <- function(Xtrain, Ytrain, ncomp, fit = FALSE, scaling = 1,
    svd.method = 1,
    rsvd_oversample = 32L, rsvd_power = 5L, svds_tol = 0, seed = 1L) {
    Xtrain <- as.matrix(Xtrain)
    Ytrain <- as.matrix(Ytrain)
    cap <- .cap_plssvd_ncomp(ncomp, nrow(Xtrain), ncol(Xtrain), ncol(Ytrain),
        warn = TRUE)
    model <- pls_model1(Xtrain, Ytrain, cap$ncomp, scaling,
        fit, svd.method, rsvd_oversample, rsvd_power, svds_tol, seed)
    model$pls_method <- "plssvd"
    model$predict_latent_ok <- TRUE
    class(model) <- "fastPLS"
    model
}

pls.model1.gpu <-
    function(
        Xtrain,
        Ytrain,
        ncomp,
        fit = FALSE,
        scaling = 1,
        svd.method = "cuda_rsvd",
        rsvd_oversample = 32L,
        rsvd_power = 5L,
        svds_tol = 0,
        seed = 1L
    ) {
        if (!has_cuda()) {
            stop(
                "pls.model1.gpu requires an available CUDA backend. ",
                "No CPU fallback is performed.",
                call. = FALSE
            )
        }
        Xtrain <- as.matrix(Xtrain)
        Ytrain <- as.matrix(Ytrain)
    svd.method <- match.arg(.normalize_svd_method(svd.method), c("cuda_rsvd"))
        cap <- .cap_plssvd_ncomp(
            ncomp,
            nrow(Xtrain),
            ncol(Xtrain),
            ncol(Ytrain),
            warn = TRUE
        )
        model <- pls_model1_gpu(
            Xtrain,
            Ytrain,
            cap$ncomp,
            scaling,
            fit,
            .svd_method_id(svd.method),
            rsvd_oversample,
            rsvd_power,
            svds_tol,
            seed
        )
        model$pls_method <- "plssvd"
        model$predict_latent_ok <- TRUE
        class(model) <- "fastPLS"
        model
    }

pls.model1.gpu.implicit.xprod <-
    function(
        Xtrain,
        Ytrain,
        ncomp,
        fit = FALSE,
        scaling = 1,
        svd.method = "cuda_rsvd",
        rsvd_oversample = 32L,
        rsvd_power = 5L,
        svds_tol = 0,
        seed = 1L
    ) {
        if (!has_cuda()) {
            stop(
                "pls.model1.gpu.implicit.xprod requires an available CUDA ",
                "backend. No CPU fallback is performed.",
                call. = FALSE
            )
        }
        Xtrain <- as.matrix(Xtrain)
        Ytrain <- as.matrix(Ytrain)
        cap <- .cap_plssvd_ncomp(
            ncomp,
            nrow(Xtrain),
            ncol(Xtrain),
            ncol(Ytrain),
            warn = TRUE
        )
        model <- pls_model1_gpu_implicit_xprod(
            Xtrain,
            Ytrain,
            cap$ncomp,
            scaling,
            fit,
            rsvd_oversample,
            rsvd_power,
            svds_tol,
            seed
        )
        model$pls_method <- "plssvd"
        model$predict_latent_ok <- TRUE
        class(model) <- "fastPLS"
        model
    }

pls.model2 <-
    function(
        Xtrain,
        Ytrain,
        ncomp,
        fit = FALSE,
        scaling = 1,
        svd.method = 1,
        rsvd_oversample = 32L,
        rsvd_power = 5L,
        svds_tol = 0,
        seed = 1L
    ) {
        model <- pls_model2(
                Xtrain,
                Ytrain,
                ncomp,
                scaling,
                fit,
                svd.method,
                rsvd_oversample,
                rsvd_power,
                svds_tol,
                seed
            )
        model$pls_method <- "simpls"
        model$predict_latent_ok <- TRUE
        class(model) <- "fastPLS"
        model
    }

pls.model2.fast <-
    function(
        Xtrain,
        Ytrain,
        ncomp,
        fit = FALSE,
        scaling = 1,
        svd.method = 1,
        rsvd_oversample = 32L,
        rsvd_power = 5L,
        svds_tol = 0,
        seed = 1L,
        return_ttrain = FALSE
    ) {
        model <- .with_fastpls_fast_options(
                pls_model2_fast(
                    Xtrain,
                    Ytrain,
                    ncomp,
                    scaling,
                    fit,
                    svd.method,
                    rsvd_oversample,
                    rsvd_power,
                    svds_tol,
                    seed
                ),
                return_ttrain = return_ttrain
            )
        model$pls_method <- "simpls"
        model$predict_latent_ok <- TRUE
        class(model) <- "fastPLS"
        model
    }

pls.model1.rsvd.xprod.precision <- function(Xtrain, Ytrain, ncomp, fit = FALSE,
    scaling = 1, rsvd_oversample = 32L, rsvd_power = 5L, svds_tol = 0,
    seed = 1L, xprod_precision = c("implicit64", "double")) {
    xprod_precision <- match.arg(xprod_precision)
    precision_id <- switch(xprod_precision, double = 0L, implicit64 = 3L)
    Xtrain <- as.matrix(Xtrain)
    Ytrain <- as.matrix(Ytrain)
    cap <- .cap_plssvd_ncomp(ncomp, nrow(Xtrain), ncol(Xtrain), ncol(Ytrain),
        warn = TRUE)
    model <- pls_model1_rsvd_xprod_precision(Xtrain,
        Ytrain,
        cap$ncomp, scaling, fit, as.integer(rsvd_oversample),
        as.integer(rsvd_power),
        svds_tol, as.integer(seed), as.integer(precision_id))
    model$pls_method <- "plssvd"
    model$predict_latent_ok <- TRUE
    class(model) <- "fastPLS"
    model
}

pls.model2.fast.rsvd.xprod.precision <- function(Xtrain, Ytrain, ncomp,
    fit = FALSE,
    scaling = 1, rsvd_oversample = 32L, rsvd_power = 5L, svds_tol = 0,
    seed = 1L, xprod_precision = c("implicit64", "double"),
    return_ttrain = FALSE) {
    xprod_precision <- match.arg(xprod_precision)
    precision_id <- switch(xprod_precision, double = 0L, implicit64 = 3L)
    model <- .with_fastpls_fast_options(pls_model2_fast_rsvd_xprod_precision(
            as.matrix(Xtrain), as.matrix(Ytrain), as.integer(ncomp),
            scaling, fit,
            as.integer(rsvd_oversample),
            as.integer(rsvd_power), svds_tol, as.integer(seed),
            as.integer(precision_id)
        ),
    return_ttrain = return_ttrain)
    model$pls_method <- "simpls"
    model$predict_latent_ok <- TRUE
    class(model) <- "fastPLS"
    model
}

pls.model2.fast.gpu <-
    function(
        Xtrain,
        Ytrain,
        ncomp,
        fit = FALSE,
        scaling = 1,
        svd.method = "cuda_rsvd",
        rsvd_oversample = 32L,
        rsvd_power = 5L,
        svds_tol = 0,
        seed = 1L
    ) {
        if (!has_cuda()) {
            stop(
                "pls.model2.fast.gpu requires an available CUDA backend. ",
                "No CPU fallback is performed.",
                call. = FALSE
            )
        }
        model <- .with_fastpls_fast_options({
    svd.method <- match.arg(.normalize_svd_method(svd.method), c("cuda_rsvd"))
            pls_model2_fast_gpu(
                Xtrain,
                Ytrain,
                ncomp,
                scaling,
                fit,
                .svd_method_id(svd.method),
                rsvd_oversample,
                rsvd_power,
                svds_tol,
                seed
            )
        })
        model$pls_method <- "simpls"
        model$predict_latent_ok <- TRUE
        class(model) <- "fastPLS"
        model
    }


.model_prediction_backend <- function(object) {
    stored <- object$predict_backend %||% "cpu"
    if (stored %in% c("cuda_flash", "cuda_fused_lda", "float32_cuda")) {
        return("cuda_flash")
    }
    if (stored %in% c("metal", "float32_metal")) {
        return("metal")
    }
    "cpu"
}

.model_public_backend <- function(object) {
    switch(
        .model_prediction_backend(object),
        cuda_flash = "cuda",
        metal = "metal",
        "cpu"
    )
}

.prediction_route <- function(object, Xtest, backend, block_size) {
    if (is.null(backend)) {
        backend <- .fastpls_resolve_backend(NULL)
        if (backend == "cuda") backend <- "cuda_flash"
    } else {
        backend <- match.arg(
            backend,
            c(
                "auto", "cpu", "cuda", "metal",
                "cpu_flash", "cuda_flash"
            )
        )
    }
    selected <- if (backend == "auto") {
        stored <- .model_prediction_backend(object)
        switch(stored, cuda_flash = "cuda", metal = "metal", "cpu")
    } else if (backend %in% c("cuda", "cuda_flash")) {
        "cuda"
    } else if (backend == "metal") {
        .fastpls_require_backend_available("metal", "Prediction")
        "cpu"
    } else {
        "cpu"
    }
    .fastpls_require_backend_available(selected, "Prediction")
    if (identical(selected, "cpu")) {
        .fastpls_apply_cpu_cores()
    }
    if (is.null(block_size)) {
        block_size <- object$flash_block_size
    }
    if (is.null(block_size) || !length(block_size) || is.na(block_size)) {
        block_size <- 4096L
    }
    list(
        backend = backend,
        selected = selected,
        block_size = as.integer(block_size),
        cuda = identical(selected, "cuda"),
        cpu_flash = backend == "cpu_flash" ||
            (backend == "auto" && identical(selected, "cpu") &&
                .should_use_cpu_flash_prediction(object, Xtest)),
        metal = identical(selected, "metal")
    )
}

.predict_lda_result <- function(object, Xtest, Ytest, proj, top, raw_scores) {
    value <- .fastpls_lda_predictions(
        object,
        Xtest,
        return_scores = raw_scores || top > 1L,
        keep_ttest = proj
    )
    result <- list(Ypred = value$Ypred, Q2Y = NULL)
    if (!is.null(value$lda_scores)) {
        if (raw_scores) {
            result$LDA_scores <- value$lda_scores
        }
        if (top > 1L) {
            ranked <- .class_topk_from_score_cube(
                value$lda_scores,
                object$lev,
                object$ncomp,
                top
            )
            result[names(ranked)] <- ranked
        }
    }
    if (proj) {
        result$Ttest <- value$Ttest
    }
    if (!is.null(Ytest)) {
        result$accuracy <- .fastpls_accuracy_from_class_labels(
            object$lev,
            Ytest,
            result$Ypred
        )
        result$Q2Y <- rep(NA_real_, length(object$ncomp))
    }
    result
}

.predict_argmax_shortcut <- function(
    object,
    Xtest,
    Ytest,
    proj,
    top,
    raw_scores,
    route
) {
    eligible <- object$classification &&
        is.null(Ytest) &&
        !raw_scores &&
        !is.list(object$W_latent) &&
        !route$metal &&
        (is.null(object$classification_rule) ||
            object$classification_rule == "argmax")
    if (!eligible) {
        return(NULL)
    }
    backend <- if (route$cuda) {
        "cuda"
    } else {
        "cpp"
    }
    result <- .class_topk_predict(object, Xtest, top, proj, backend)
    result$Q2Y <- NULL
    result
}

.predict_backend_result <- function(object, Xtest, proj, route) {
    if (
        !route$cuda && !route$metal &&
            object$pls_method %in% c("plssvd", "simpls") &&
            startsWith(
                object$xprod_mode %||% "",
                "float64_"
            )
    ) {
        return(pls_labels_core_predict_cpp(object, Xtest, proj))
    }
    if (route$metal) {
        return(.pls_predict_metal(object, Xtest, proj))
    }
    if (route$cuda) {
        return(pls_predict_flash_cuda(object, Xtest, proj))
    }
    if (route$cpu_flash) {
        return(tryCatch(
            pls_predict_flash_cpu(
                object,
                Xtest,
                proj,
                route$block_size
            ),
            error = function(error) {
                if (route$backend == "cpu_flash") {
                    stop(conditionMessage(error), call. = FALSE)
                }
                pls_predict(object, Xtest, proj)
            }
        ))
    }
    pls_predict(object, Xtest, proj)
}

.predict_attach_q2 <- function(result, object, Ytest) {
    result$Q2Y <- NULL
    if (is.null(Ytest)) {
        return(result)
    }
    observed <- if (object$classification) {
        .fastpls_one_hot_labels(Ytest, object$lev)
    } else {
        as.matrix(Ytest)
    }
    result$Q2Y <- vapply(
        seq_along(object$ncomp),
        function(index) {
            predicted <- matrix(
                result$Ypred[, , index],
                nrow = dim(result$Ypred)[1L],
                ncol = dim(result$Ypred)[2L]
            )
            .fastpls_q2_from_reference(observed, predicted, object$mY)
        },
        numeric(1L)
    )
    result
}

.predict_classification_result <- function(result, object, Xtest, Ytest, proj,
    top, raw_scores) {
    if (!object$classification) {
        return(result)
    }
    rule <- object$classification_rule %||% "argmax"
    if (.is_lda_classifier(rule)) {
        value <- .fastpls_lda_predictions(object, Xtest, Ttest = result$Ttest,
            return_scores = raw_scores || top > 1L)
        result$Ypred <- value$Ypred
        if (raw_scores) {
            result$LDA_scores <- value$lda_scores
        }
        if (top > 1L) {
            ranked <- .class_topk_from_score_cube(value$lda_scores, object$lev,
                object$ncomp, top)
            result[names(ranked)] <- ranked
        }
        if (proj || !is.null(result$Ttest))
            result$Ttest <- value$Ttest
    }
    else {
        score_cube <- result$Ypred
        ranked <- .class_topk_from_score_cube(score_cube, object$lev,
            object$ncomp,
            top)
        result[names(ranked)] <- ranked
        if (raw_scores)
            result$Yscore <- score_cube
    }
    if (!is.null(Ytest)) {
        result$accuracy <- .fastpls_accuracy_from_class_labels(object$lev,
            Ytest,
            result$Ypred)
    }
    result
}

#' Predict from fitted fastPLS models
#'
#' Generates predictions for new samples from fitted PLS-SVD, SIMPLS, OPLS, or
#' kernel PLS models. Stored centering, scaling, latent projections, and
#' model-specific filtering are applied before producing numeric response
#' predictions or classification labels.
#'
#' @param object A fitted `fastPLS`, `fastPLSKernel`, or `fastPLSOpls` object.
#' @param newdata Numeric predictor matrix.
#' @param Ytest Optional observed response used to compute independent-test
#'   `Q2Y` relative to the response mean stored from model training.
#' @param proj Logical; return projected `Ttest` when `TRUE`.
#' @param backend Prediction backend: `"cpu"`, `"cuda"`, or `"metal"`.
#'   Operation-split Metal models predict on CPU because their retained
#'   matrices are host-accessible; Metal is used for fitting sample-matrix
#'   products.
#'   When omitted, the session backend setting is used. `"auto"` retains the
#'   backend stored in the fitted model. An unavailable CUDA or Metal selection
#'   raises an error; prediction is never silently moved to CPU.
#' @param flash.block_size Row block size for \code{cpu_flash} prediction.
#' @param top Number of ranked classes to return for classification. Rank 1 is
#'   the ordinary predicted class stored in `Ypred`; ranks 2, 3, and so on are
#'   lower-scoring alternatives returned in `Ypred_top` when `top > 1`.
#' @param top5 Convenience flag equivalent to `top = max(top, 5)`, useful for
#'   reporting ImageNet-style top-5 candidate labels.
#' @param raw_scores If `TRUE`, keep raw classification score cubes as
#'   `Yscore` when available. This can require substantial memory. With
#'   float32 inputs and `raw_scores = FALSE`, ranked prediction is evaluated in
#'   bounded row blocks and only the requested ranks are retained.
#' @param ... Unused.
#' @return A list containing `Ypred`, optional independent-test `Q2Y`, optional
#'   `Ttest`, optional `Ypred_top` and `Ypred_top_score` ranked-class outputs,
#'   and optional raw classification scores.
#' @examples
#' X <- as.matrix(mtcars[, c("disp", "hp", "wt", "qsec")])
#' y <- mtcars$mpg
#' fit <- pls(X, y,
#'     ncomp = 2, method = "simpls", backend = "cpu",
#'     svd.method = "rsvd", return_variance = FALSE
#' )
#' pred <- predict(fit, X[seq_len(3), , drop = FALSE])
#' pred$Ypred
#' @export
predict.fastPLS <- function(object, newdata, Ytest = NULL, proj = FALSE,
    backend = NULL,
    flash.block_size = NULL, top = 1L, top5 = FALSE, raw_scores = FALSE, ...) {
    if (!is(object, "fastPLS")) {
        stop("object is not a fastPLS object")
    }
    object <- .fastpls_restore_internal_output_fields(object)
    top <- .resolve_top_k(top, top5)
    if (!is.null(object$resident_state)) {
        resident_backend <- object$resident_backend %||% "cuda"
        selected <- if (is.null(backend) || identical(backend, "auto")) {
            resident_backend
        } else {
            backend
        }
        compatible <- if (identical(resident_backend, "cuda")) {
            selected %in% c("cuda", "cuda_flash")
        } else {
            identical(selected, "metal")
        }
        if (!compatible) {
            stop(
                "This model retains ", resident_backend,
                " device state; prediction requires the same backend. No CPU fallback is performed.",
                call. = FALSE
            )
        }
        if (identical(resident_backend, "metal")) {
            return(.resident_metal_predict(
                object, newdata, Ytest, proj, top, raw_scores
            ))
        }
        return(.resident_cuda_predict(object, newdata, Ytest, proj, top, raw_scores))
    }
    if (is.null(backend) && identical(object$precision %||% "double",
        "float32")) {
        backend <- switch(
            object$predict_backend %||% "",
            float32_metal = "metal",
            float32_cuda = "cuda",
            NULL
        )
    }
    route <- .prediction_route(object, newdata, backend, flash.block_size)
    newdata <- .fastpls_predictor_input(newdata, "newdata")
    if (object$precision %||% "double" == "float32") {
        result <- .predict_fastpls_float32(object, newdata, Ytest, proj, top,
            raw_scores = raw_scores, backend = route$selected)
        return(.fastpls_public_predict_output(result, object$ncomp))
    }
    Xtest <- as.matrix(newdata)
    rule <- object$classification_rule %||% "argmax"
    if (object$classification && .is_lda_classifier(rule)) {
        result <- .predict_lda_result(object, Xtest, Ytest, proj, top,
            raw_scores)
        return(.fastpls_public_predict_output(result, object$ncomp))
    }
    result <- .predict_argmax_shortcut(object, Xtest, Ytest, proj, top,
        raw_scores,
        route)
    if (is.null(result)) {
        result <- .predict_backend_result(object, Xtest, proj, route)
        result <- .predict_attach_q2(result, object, Ytest)
        result <- .predict_classification_result(result, object, Xtest, Ytest,
            proj, top, raw_scores)
    }
    .fastpls_public_predict_output(result, object$ncomp)
}

.fastpls_preprocess_train <- function(X, scaling) {
    X <- as.matrix(X)
    scal <- if (is.character(scaling)) {
        pmatch(scaling, c("centering", "autoscaling", "none"))[1]
    } else {
        as.integer(scaling)
    }
    mX <- rep(0, ncol(X))
    if (scal < 3L) {
        mX <- colMeans(X)
        X <- sweep(X, 2, mX, "-")
    }
    vX <- rep(1, ncol(X))
    if (scal == 2L) {
        vX <- apply(X, 2, sd)
        vX[!is.finite(vX) | vX == 0] <- 1
        X <- sweep(X, 2, vX, "/")
    }
    list(
        X = X,
        mX = matrix(mX, nrow = 1),
        vX = matrix(vX, nrow = 1),
        scaling = scal
    )
}

.fastpls_preprocess_test <- function(X, mX, vX) {
    X <- as.matrix(X)
    X <- sweep(X, 2, as.numeric(mX[1, ]), "-")
    sweep(X, 2, as.numeric(vX[1, ]), "/")
}

.kernel_pls_kernel_id <- function(kernel) {
    kernel <- match.arg(kernel, c("linear", "rbf", "poly"))
    switch(kernel, linear = 1L, rbf = 2L, poly = 3L)
}

.kernel_pls_gamma <- function(gamma, Xtrain) {
    if (is.null(gamma)) {
        gamma <- 1 / max(1L, ncol(Xtrain))
    }
    gamma <- as.numeric(gamma)[1]
    if (!is.finite(gamma) || gamma <= 0) {
        stop("gamma must be a finite positive number", call. = FALSE)
    }
    gamma
}

.kernel_pls_memory_guard <- function(n, bytes_per_value, context) {
    # Kernel construction, centering, and fitting overlap at peak usage.
    estimated_bytes <- 3 * as.double(n) * as.double(n) * bytes_per_value
    limit_bytes <- 4 * 1024^3
    if (!is.finite(estimated_bytes) || estimated_bytes > limit_bytes) {
        template <- paste(
            "%s requires an n-by-n Gram matrix. Its estimated core workspace",
            "is %.2f GiB for n=%d, above the 4 GiB safety limit. Use",
            "kernel = 'linear' or reduce the training set."
        )
        message <- sprintf(
            template,
            context,
            estimated_bytes / 1024^3,
            as.integer(n)
        )
        stop(message, call. = FALSE)
    }
    invisible(estimated_bytes)
}

.center_kernel_train_base <- function(K) {
    col_means <- colMeans(K)
    row_means <- rowMeans(K)
    grand_mean <- mean(col_means)
    Kc <- sweep(K, 2, col_means, "-")
    Kc <- sweep(Kc, 1, row_means, "-")
    Kc <- Kc + grand_mean
list(K = Kc, col_means = matrix(col_means, nrow = 1), grand_mean = grand_mean)
}

.center_kernel_test_base <- function(Ktest, train_col_means, train_grand_mean) {
    Kc <- sweep(Ktest, 2, as.numeric(train_col_means[1, ]), "-")
    Kc <- sweep(Kc, 1, rowMeans(Ktest), "-")
    Kc + train_grand_mean
}

.supervised_response_matrix <- function(Y) {
    if (is.factor(Y)) {
        return(transformy(Y))
    }
    as.matrix(Y)
}

.kernel_pls_inner_fit <- function(
    fit_fun,
    Xtrain,
    Ytrain,
    ncomp,
    scaling,
    fit,
    inner_args
) {
    do.call(
        fit_fun,
        c(
            list(
                Xtrain = Xtrain,
                Ytrain = Ytrain,
                Xtest = NULL,
                Ytest = NULL,
                ncomp = ncomp,
                scaling = scaling,
                fit = fit,
                proj = FALSE
            ),
            inner_args
        )
    )
}

.kernel_pls_linear_fit <- function(
    Xtrain,
    Ytrain,
    Xtest,
    Ytest,
    ncomp,
    scaling,
    fit,
    proj,
    engine,
    fit_fun,
    inner_args
) {
    inner <- .kernel_pls_inner_fit(
        fit_fun,
        Xtrain,
        Ytrain,
        ncomp,
        scaling,
        fit,
        inner_args
    )
    inner <- .fastpls_restore_internal_output_fields(inner)
    inner$kernel <- "linear"
    inner$kernel_engine <- paste0(engine, "_direct")
    inner$kernel_linear_direct <- TRUE
    class(inner) <- "fastPLS"
    if (!is.null(Xtest)) {
        inner <- c(
            inner,
            predict.fastPLS(
                inner,
                as.matrix(Xtest),
                Ytest = Ytest,
                proj = proj,
                backend = .model_public_backend(inner)
            )
        )
        class(inner) <- "fastPLS"
    }
    .fastpls_public_pls_output(inner, inner$ncomp)
}

.kernel_pls_model <- function(
    inner,
    prep,
    kernel,
    kernel_id,
    gamma,
    degree,
    coef0,
    center,
    engine
) {
    out <- list(
        inner_model = inner,
        Xref = prep$X,
        mX = prep$mX,
        vX = prep$vX,
        kernel = kernel,
        kernel_id = kernel_id,
        gamma = gamma,
        degree = as.integer(degree),
        coef0 = coef0,
        kernel_center = center,
        kernel_engine = engine,
        ncomp = inner$ncomp,
        xprod_mode = inner$xprod_mode,
        gpu_resident = isTRUE(inner$gpu_resident)
    )
    out <- .inherit_inner_variance_explained(out, inner)
    out <- .inherit_inner_fit_outputs(out, inner)
    class(out) <- c("fastPLSKernel", "fastPLS")
    out
}

.kernel_pls_fit <- function(Xtrain, Ytrain, Xtest, Ytest, ncomp, scaling,
    kernel,
    gamma, degree, coef0, fit, proj, kernel_engine, fit_fun, inner_args) {
    kernel <- match.arg(kernel, c("linear", "rbf", "poly"))
    if (identical(kernel, "linear")) {
        return(.kernel_pls_linear_fit(Xtrain, Ytrain, Xtest, Ytest, ncomp,
            scaling,
            fit, proj, kernel_engine, fit_fun, inner_args))
    }
    .kernel_pls_memory_guard(
        nrow(Xtrain), 8,
        sprintf("Nonlinear %s kernel PLS", kernel)
    )
    prep <- .fastpls_preprocess_train(Xtrain, scaling)
    gamma <- .kernel_pls_gamma(gamma, prep$X)
    kernel_id <- .kernel_pls_kernel_id(kernel)
    K <- kernel_matrix_cpp(prep$X, prep$X, kernel_id, gamma,
        as.integer(degree),
        coef0)
    kc <- center_kernel_train_cpp(K)
    inner <- .kernel_pls_inner_fit(fit_fun, kc$K, Ytrain, ncomp, "none", fit,
        inner_args)
    inner <- .fastpls_restore_internal_output_fields(inner)
    out <- .kernel_pls_model(inner, prep, kernel, kernel_id, gamma, degree,
        coef0,
        kc, kernel_engine)
    if (!is.null(Xtest)) {
        res <- predict(
            out,
            Xtest,
            Ytest = Ytest,
            proj = proj,
            backend = .model_public_backend(inner)
        )
        out <- c(out, res)
        class(out) <- c("fastPLSKernel", "fastPLS")
    }
    out <- .attach_backend_control(out)
    out
}

#' Kernel PLS
#'
#'  Fits PLS on a centered training kernel. The CUDA variant uses the GPU PLS
#' core
#' after host-side kernel construction and centering.
#'
#' @inheritParams pls
#' @param kernel Kernel type: \code{linear}, \code{rbf}, or \code{poly}.
#' @param gamma Kernel scale. Defaults to `1 / ncol(Xtrain)`.
#' @param degree Polynomial kernel degree.
#' @param coef0 Polynomial kernel offset.
#' @param ... Additional arguments passed to the inner PLS fit.
#' @return A `fastPLSKernel` object.
#' @noRd
.kernel_pls_cpp <- function(Xtrain, Ytrain, Xtest = NULL, Ytest = NULL,
    ncomp = 2,
    scaling = c("centering", "autoscaling", "none"), kernel = c("linear",
        "rbf",
        "poly"), gamma = NULL, degree = 3L, coef0 = 1,
    svd.method = "cpu_rsvd", rsvd_oversample = 32L,
    rsvd_power = 5L, svds_tol = 0,
    seed = 1L, classifier = c("argmax", "lda"), lda_ridge = 1e-08, fit = FALSE,
    return_variance = TRUE, proj = FALSE) {
    classifier <- .resolve_classifier_for_backend(classifier, "cpu")
    svd.method <- match.arg(.normalize_svd_method(svd.method), c("cpu_rsvd"))
    .kernel_pls_fit(Xtrain, Ytrain, Xtest, Ytest, ncomp, match.arg(scaling),
        match.arg(kernel),
        gamma, degree, coef0, fit, proj, "cpp", pls, list(method = "simpls",
            svd.method = svd.method,
            rsvd_oversample = rsvd_oversample, rsvd_power = rsvd_power,
            seed = seed,
            classifier = classifier,
            return_variance = return_variance))
}

#' @noRd
.kernel_pls_cuda <- function(Xtrain, Ytrain, Xtest = NULL, Ytest = NULL,
    ncomp = 2,
    scaling = c("centering", "autoscaling", "none"), kernel = c("linear",
        "rbf",
        "poly"), gamma = NULL, degree = 3L, coef0 = 1, rsvd_oversample = 32L,
    rsvd_power = 5L,
    svds_tol = 0, svd.method = "cuda_rsvd", seed = 1L, classifier = c("argmax",
        "lda"), lda_ridge = 1e-08, fit = FALSE, return_variance = TRUE,
    proj = FALSE,
    ...) {
    classifier <- .resolve_classifier_for_backend(classifier, "cuda")
    fit_fun <- .simpls_gpu
    .kernel_pls_fit(Xtrain, Ytrain, Xtest, Ytest, ncomp, match.arg(scaling),
        match.arg(kernel),
        gamma, degree, coef0, fit, proj, "cuda", fit_fun,
        c(list(rsvd_oversample = rsvd_oversample,
            rsvd_power = rsvd_power, svds_tol = svds_tol,
            svd.method = svd.method,
            seed = seed, classifier = classifier, lda_ridge = lda_ridge,
            return_variance = return_variance),
        list(...)))
}

#' @rdname predict.fastPLS
#' @export
predict.fastPLSKernel <- function(object, newdata, Ytest = NULL, proj = FALSE,
    ...) {
    if (!is(object, "fastPLSKernel")) {
        stop("object is not a fastPLSKernel object", call. = FALSE)
    }
    .fastpls_require_prediction_backend(list(...), "Kernel-PLS prediction")
    object <- .fastpls_restore_internal_output_fields(object)
    if (identical(object$precision, "float32")) {
        Xnew <- .as_float32_matrix(newdata, "newdata")
        Xnew <- .float32_standardize(Xnew, object$mX, object$vX)
        kernel_backend <- sub("^float32_", "", object$kernel_engine)
        Kraw <- kernel_matrix_float32_cpp(
            Xnew, object$Xref, object$kernel_id,
            object$gamma, object$degree, object$coef0,
            .float32_product_backend_id(kernel_backend)
        )
        Ktest <- .float32_from_bits(Kraw$K)
        centered_raw <- center_kernel_test_float32_cpp(Ktest,
            object$kernel_center$col_means,
            object$kernel_center$grand_mean)
        centered <- .float32_from_bits(centered_raw$K)
        return(predict.fastPLS(object$inner_model, centered, Ytest = Ytest,
            proj = proj,
            ...))
    }
    Xnew <- .fastpls_preprocess_test(newdata, object$mX, object$vX)
    if (identical(object$kernel_engine, "metal")) {
        Ktest <- .kernel_matrix_metal(Xnew, object$Xref, object$kernel,
            object$gamma,
            object$degree, object$coef0)
        Ktest <- .center_kernel_test_base(Ktest,
            object$kernel_center$col_means,
            object$kernel_center$grand_mean)
    }
    else {
        Ktest <- kernel_matrix_cpp(Xnew, object$Xref, object$kernel_id,
            object$gamma,
            object$degree, object$coef0)
        Ktest <- center_kernel_test_cpp(Ktest, object$kernel_center$col_means,
            object$kernel_center$grand_mean)
    }
    predict.fastPLS(object$inner_model, Ktest, Ytest = Ytest, proj = proj, ...)
}

.opls_fit <- function(Xtrain, Ytrain, Xtest, Ytest, ncomp, scaling, north, fit,
    proj, filter_engine, fit_fun, inner_args) {
    scaling_id <- pmatch(scaling, c("centering", "autoscaling", "none"))[1]
    if (is.factor(Ytrain) || is.character(Ytrain)) {
        labels <- droplevels(factor(Ytrain))
        filt <- opls_filter_labels_core_cpp(
            as.matrix(Xtrain), as.integer(labels), nlevels(labels),
            as.integer(north), scaling_id
        )
    } else {
        Yfilter <- .supervised_response_matrix(Ytrain)
        filt <- opls_filter_core_cpp(
            as.matrix(Xtrain), Yfilter, as.integer(north), scaling_id
        )
    }
    .opls_require_predictive_rank(ncomp, filt$X, filt$north,
        !identical(scaling, "none"))
    inner <- do.call(fit_fun, c(list(Xtrain = filt$X, Ytrain = Ytrain,
        Xtest = NULL,
        Ytest = NULL, ncomp = ncomp, scaling = "none", fit = fit,
        proj = FALSE),
    inner_args))
    inner <- .fastpls_restore_internal_output_fields(inner)
    out <- list(inner_model = inner, mX = filt$mX, vX = filt$vX,
        W_orth = filt$W_orth,
        P_orth = filt$P_orth, north = filt$north, opls_engine = filter_engine,
        ncomp = inner$ncomp, xprod_mode = inner$xprod_mode,
        gpu_resident = isTRUE(inner$gpu_resident))
    out <- .inherit_inner_variance_explained(out, inner)
    out <- .inherit_inner_fit_outputs(out, inner)
    class(out) <- c("fastPLSOpls", "fastPLS")
    if (!is.null(Xtest)) {
        res <- predict(
            out,
            Xtest,
            Ytest = Ytest,
            proj = proj,
            backend = .model_public_backend(inner)
        )
        out <- c(out, res)
        class(out) <- c("fastPLSOpls", "fastPLS")
    }
    out <- .attach_backend_control(out)
    out
}

#' Orthogonal PLS
#'
#' Removes supervised orthogonal variation from `Xtrain`, then fits the SIMPLS
#' core. This legacy CPU helper is not used by the public CUDA or Metal routes;
#' accelerator OPLS filtering and fitting are fully device-resident.
#'
#' @inheritParams pls
#' @param north Number of orthogonal components to remove before PLS fitting.
#' @param ... Additional arguments passed to the inner PLS fit.
#' @return A `fastPLSOpls` object.
#' @noRd
.opls_cpp <- function(Xtrain, Ytrain, Xtest = NULL, Ytest = NULL, ncomp = 2,
    north = 1L,
    scaling = c("centering", "autoscaling", "none"), svd.method = "cpu_rsvd",
    rsvd_oversample = 32L, rsvd_power = 5L, svds_tol = 0,
    seed = 1L, classifier = c("argmax", "lda"), lda_ridge = 1e-08, fit = FALSE,
    return_variance = TRUE, proj = FALSE) {
    classifier <- .resolve_classifier_for_backend(classifier, "cpu")
    svd.method <- match.arg(.normalize_svd_method(svd.method), c("cpu_rsvd"))
    .opls_fit(Xtrain, Ytrain, Xtest, Ytest, ncomp, match.arg(scaling), north,
        fit,
        proj, "cpp", pls, list(method = "simpls", svd.method = svd.method,
            rsvd_oversample = rsvd_oversample,
            rsvd_power = rsvd_power, seed = seed, classifier = classifier,
            return_variance = return_variance))
}

#' @noRd
.opls_cuda <- function(
    Xtrain,
    Ytrain,
    Xtest = NULL,
    Ytest = NULL,
    ncomp = 2,
    north = 1L,
    scaling = c("centering", "autoscaling", "none"),
    rsvd_oversample = 32L,
    rsvd_power = 5L,
    svds_tol = 0,
    svd.method = "cuda_rsvd",
    seed = 1L,
    classifier = c("argmax", "lda"),
    lda_ridge = 1e-8,
    fit = FALSE,
    return_variance = TRUE,
    proj = FALSE,
    ...
) {
    classifier <- .resolve_classifier_for_backend(classifier, "cuda")
    fit_fun <- .simpls_gpu
    .opls_fit(
        Xtrain,
        Ytrain,
        Xtest,
        Ytest,
        ncomp,
        match.arg(scaling),
        north,
        fit,
        proj,
        "cpp",
        fit_fun,
        c(
            list(
                rsvd_oversample = rsvd_oversample,
                rsvd_power = rsvd_power,
                svds_tol = svds_tol,
                svd.method = svd.method,
                seed = seed,
                classifier = classifier,
                lda_ridge = lda_ridge,
                return_variance = return_variance
            ),
            list(...)
        )
    )
}

#' @rdname predict.fastPLS
#' @export
predict.fastPLSOpls <- function(object, newdata, Ytest = NULL, proj = FALSE,
    ...) {
    if (!is(object, "fastPLSOpls")) {
        stop("object is not a fastPLSOpls object", call. = FALSE)
    }
    .fastpls_require_prediction_backend(list(...), "OPLS prediction")
    object <- .fastpls_restore_internal_output_fields(object)
    if (identical(object$precision, "float32")) {
        engine <- sub(
            "^float32_", "",
            object$opls_filter_engine %||% object$opls_engine
        )
        filter_backend <- .float32_product_backend_id(engine)
        raw <- opls_apply_filter_float32_cpp(
            .as_float32_matrix(newdata, "newdata"),
            object$mX, object$vX, object$W_orth, object$P_orth,
            filter_backend
        )
        filtered <- .float32_from_bits(raw$X)
        return(predict.fastPLS(object$inner_model, filtered, Ytest = Ytest,
            proj = proj,
            ...))
    }
    Xnew <- if (identical(object$opls_engine, "metal")) {
        .opls_apply_filter_metal(newdata, object$mX, object$vX, object$W_orth,
            object$P_orth)
    }
    else {
        opls_apply_filter_cpp(as.matrix(newdata), object$mX, object$vX,
            object$W_orth,
            object$P_orth)
    }
    predict.fastPLS(object$inner_model, Xnew, Ytest = Ytest, proj = proj, ...)
}

.gpu_pls_response <- function(Xtrain, Ytrain, classifier, method) {
    X <- as.matrix(Xtrain)
    prepared <- .prepare_response(Ytrain)
    ncomp_cap <- function(ncomp) {
        if (!prepared$classification || method != "plssvd") {
            return(ncomp)
        }
        .cap_plssvd_ncomp(
            ncomp,
            nrow(X),
            ncol(X),
            ncol(prepared$Ytrain),
            factor_response = TRUE,
            warn = TRUE
        )$ncomp
    }
    list(
        X = X,
        Y = prepared$Ytrain,
        original = Ytrain,
        classification = prepared$classification,
        levels = prepared$lev,
        classifier = .resolve_classifier_for_backend(classifier, "cuda"),
        cap = ncomp_cap
    )
}

.gpu_pls_finalize <- function(model, response, Xtest, Ytest, ncomp, fit, proj,
    method, classifier, lda_ridge, variance) {
    model$classification <- response$classification
    model$lev <- response$levels
    model$pls_method <- method
    model$predict_latent_ok <- TRUE
    if (fit) {
        model <- .attach_train_scores(model, response$X)
    }
    model <- .enable_flash_prediction(model, "cuda")
    model <- .attach_lda_classifier(model, response$X, response$original,
        classifier,
        lda_ridge)
    model <- .maybe_attach_pls_variance_explained(model, response$X, variance)
    if (!is.null(Xtest)) {
        model <- c(model, predict.fastPLS(model, as.matrix(Xtest),
            Ytest = Ytest,
            proj = proj,
            backend = "cuda_flash"))
    }
    if (response$classification && fit && !is.null(model$Yfit)) {
        fitted <- lapply(seq_along(ncomp), function(i) {
            index <- apply(model$Yfit[, , i], 1L, which.max)
            factor(response$levels[index], levels = response$levels)
        })
        model$Yfit <- as.data.frame(fitted)
        names(model$Yfit) <- paste0("ncomp=", ncomp)
    }
    class(model) <- "fastPLS"
    .attach_backend_control(model)
}

.gpu_fused_lda <- function(
    method_id,
    method_name,
    response,
    Xtest,
    Ytest,
    ncomp,
    scaling,
    use_xprod,
    controls
) {
    if (!response$classification || response$classifier != "lda_cuda") {
        return(NULL)
    }
    do.call(
        .try_cuda_native_lda_fit_predict,
        c(
            list(
                method_id = method_id,
                method_name = method_name,
                Xtrain = response$X,
                Ytrain = response$Y,
                Ytrain_original = response$original,
                Xtest = Xtest,
                Ytest = Ytest,
                ncomp = ncomp,
                scaling_id = scaling,
                use_xprod_default = use_xprod,
                lev = response$levels
            ),
            controls
        )
    )
}

.gpu_pls_response <- function(Xtrain, Ytrain, classifier, method) {
    X <- as.matrix(Xtrain)
    prepared <- .prepare_response(Ytrain)
    ncomp_cap <- function(ncomp) {
        if (!prepared$classification || method != "plssvd") {
            return(ncomp)
        }
        .cap_plssvd_ncomp(
            ncomp,
            nrow(X),
            ncol(X),
            ncol(prepared$Ytrain),
            factor_response = TRUE,
            warn = TRUE
        )$ncomp
    }
    list(
        X = X,
        Y = prepared$Ytrain,
        original = Ytrain,
        classification = prepared$classification,
        levels = prepared$lev,
        classifier = .resolve_classifier_for_backend(classifier, "cuda"),
        cap = ncomp_cap
    )
}

.gpu_pls_finalize <- function(model, response, Xtest, Ytest, ncomp, fit, proj,
    method, classifier, lda_ridge, variance) {
    model$classification <- response$classification
    model$lev <- response$levels
    model$pls_method <- method
    model$predict_latent_ok <- TRUE
    if (fit) {
        model <- .attach_train_scores(model, response$X)
    }
    model <- .enable_flash_prediction(model, "cuda")
    model <- .attach_lda_classifier(model, response$X, response$original,
        classifier,
        lda_ridge)
    model <- .maybe_attach_pls_variance_explained(model, response$X, variance)
    if (!is.null(Xtest)) {
        model <- c(model, predict.fastPLS(model, as.matrix(Xtest),
            Ytest = Ytest,
            proj = proj,
            backend = "cuda_flash"))
    }
    if (response$classification && fit && !is.null(model$Yfit)) {
        fitted <- lapply(seq_along(ncomp), function(i) {
            index <- apply(model$Yfit[, , i], 1L, which.max)
            factor(response$levels[index], levels = response$levels)
        })
        model$Yfit <- as.data.frame(fitted)
        names(model$Yfit) <- paste0("ncomp=", ncomp)
    }
    class(model) <- "fastPLS"
    .attach_backend_control(model)
}

.gpu_fused_lda <- function(
    method_id,
    method_name,
    response,
    Xtest,
    Ytest,
    ncomp,
    scaling,
    use_xprod,
    controls
) {
    if (!response$classification || response$classifier != "lda_cuda") {
        return(NULL)
    }
    do.call(
        .try_cuda_native_lda_fit_predict,
        c(
            list(
                method_id = method_id,
                method_name = method_name,
                Xtrain = response$X,
                Ytrain = response$Y,
                Ytrain_original = response$original,
                Xtest = Xtest,
                Ytest = Ytest,
                ncomp = ncomp,
                scaling_id = scaling,
                use_xprod_default = use_xprod,
                lev = response$levels
            ),
            controls
        )
    )
}

.gpu_pls_response <- function(Xtrain, Ytrain, classifier, method) {
    X <- as.matrix(Xtrain)
    prepared <- .prepare_response(Ytrain)
    ncomp_cap <- function(ncomp) {
        if (!prepared$classification || method != "plssvd") {
            return(ncomp)
        }
        .cap_plssvd_ncomp(
            ncomp,
            nrow(X),
            ncol(X),
            ncol(prepared$Ytrain),
            factor_response = TRUE,
            warn = TRUE
        )$ncomp
    }
    list(
        X = X,
        Y = prepared$Ytrain,
        original = Ytrain,
        classification = prepared$classification,
        levels = prepared$lev,
        classifier = .resolve_classifier_for_backend(classifier, "cuda"),
        cap = ncomp_cap
    )
}

.gpu_pls_finalize <- function(model, response, Xtest, Ytest, ncomp, fit, proj,
    method, classifier, lda_ridge, variance) {
    model$classification <- response$classification
    model$lev <- response$levels
    model$pls_method <- method
    model$predict_latent_ok <- TRUE
    if (fit) {
        model <- .attach_train_scores(model, response$X)
    }
    model <- .enable_flash_prediction(model, "cuda")
    model <- .attach_lda_classifier(model, response$X, response$original,
        classifier,
        lda_ridge)
    model <- .maybe_attach_pls_variance_explained(model, response$X, variance)
    if (!is.null(Xtest)) {
        model <- c(model, predict.fastPLS(model, as.matrix(Xtest),
            Ytest = Ytest,
            proj = proj,
            backend = "cuda_flash"))
    }
    if (response$classification && fit && !is.null(model$Yfit)) {
        fitted <- lapply(seq_along(ncomp), function(i) {
            index <- apply(model$Yfit[, , i], 1L, which.max)
            factor(response$levels[index], levels = response$levels)
        })
        model$Yfit <- as.data.frame(fitted)
        names(model$Yfit) <- paste0("ncomp=", ncomp)
    }
    class(model) <- "fastPLS"
    .attach_backend_control(model)
}

.gpu_fused_lda <- function(
    method_id,
    method_name,
    response,
    Xtest,
    Ytest,
    ncomp,
    scaling,
    use_xprod,
    controls
) {
    if (!response$classification || response$classifier != "lda_cuda") {
        return(NULL)
    }
    do.call(
        .try_cuda_native_lda_fit_predict,
        c(
            list(
                method_id = method_id,
                method_name = method_name,
                Xtrain = response$X,
                Ytrain = response$Y,
                Ytrain_original = response$original,
                Xtest = Xtest,
                Ytest = Ytest,
                ncomp = ncomp,
                scaling_id = scaling,
                use_xprod_default = use_xprod,
                lev = response$levels
            ),
            controls
        )
    )
}

.gpu_pls_response <- function(Xtrain, Ytrain, classifier, method) {
    X <- as.matrix(Xtrain)
    prepared <- .prepare_response(Ytrain)
    ncomp_cap <- function(ncomp) {
        if (!prepared$classification || method != "plssvd") {
            return(ncomp)
        }
        .cap_plssvd_ncomp(
            ncomp,
            nrow(X),
            ncol(X),
            ncol(prepared$Ytrain),
            factor_response = TRUE,
            warn = TRUE
        )$ncomp
    }
    list(
        X = X,
        Y = prepared$Ytrain,
        original = Ytrain,
        classification = prepared$classification,
        levels = prepared$lev,
        classifier = .resolve_classifier_for_backend(classifier, "cuda"),
        cap = ncomp_cap
    )
}

.gpu_pls_finalize <- function(model, response, Xtest, Ytest, ncomp, fit, proj,
    method, classifier, lda_ridge, variance) {
    model$classification <- response$classification
    model$lev <- response$levels
    model$pls_method <- method
    model$predict_latent_ok <- TRUE
    if (fit) {
        model <- .attach_train_scores(model, response$X)
    }
    model <- .enable_flash_prediction(model, "cuda")
    model <- .attach_lda_classifier(model, response$X, response$original,
        classifier,
        lda_ridge)
    model <- .maybe_attach_pls_variance_explained(model, response$X, variance)
    if (!is.null(Xtest)) {
        model <- c(model, predict.fastPLS(model, as.matrix(Xtest),
            Ytest = Ytest,
            proj = proj,
            backend = "cuda_flash"))
    }
    if (response$classification && fit && !is.null(model$Yfit)) {
        fitted <- lapply(seq_along(ncomp), function(i) {
            index <- apply(model$Yfit[, , i], 1L, which.max)
            factor(response$levels[index], levels = response$levels)
        })
        model$Yfit <- as.data.frame(fitted)
        names(model$Yfit) <- paste0("ncomp=", ncomp)
    }
    class(model) <- "fastPLS"
    .attach_backend_control(model)
}

.gpu_fused_lda <- function(
    method_id,
    method_name,
    response,
    Xtest,
    Ytest,
    ncomp,
    scaling,
    use_xprod,
    controls
) {
    if (!response$classification || response$classifier != "lda_cuda") {
        return(NULL)
    }
    do.call(
        .try_cuda_native_lda_fit_predict,
        c(
            list(
                method_id = method_id,
                method_name = method_name,
                Xtrain = response$X,
                Ytrain = response$Y,
                Ytrain_original = response$original,
                Xtest = Xtest,
                Ytest = Ytest,
                ncomp = ncomp,
                scaling_id = scaling,
                use_xprod_default = use_xprod,
                lev = response$levels
            ),
            controls
        )
    )
}

.gpu_simpls_core <- function(
    response,
    ncomp,
    scaling,
    solver,
    controls,
    use_xprod
) {
    fit_expr <- function() {
        pls.model2.fast.gpu(
            Xtrain = response$X,
            Ytrain = response$Y,
            ncomp = as.integer(ncomp),
            fit = controls$fit,
            scaling = scaling,
            svd.method = solver,
            rsvd_oversample = controls$rsvd_oversample,
            rsvd_power = controls$rsvd_power,
            svds_tol = controls$svds_tol,
            seed = controls$seed
        )
    }
    model <- .with_gpu_native_options(
        if (use_xprod) .with_simpls_gpu_xprod(fit_expr()) else fit_expr(),
        gpu_device_state = controls$gpu_device_state,
        gpu_qr = controls$gpu_qr,
        gpu_eig = controls$gpu_eig,
        gpu_finalize_threshold = controls$gpu_finalize_threshold
    )
    model$xprod_default <- use_xprod
    model
}

.gpu_plssvd_core <- function(
    response,
    ncomp,
    scaling,
    solver,
    controls,
    use_xprod
) {
    fit_fun <- if (use_xprod) pls.model1.gpu.implicit.xprod else pls.model1.gpu
    model <- .with_gpu_native_options(
        fit_fun(
            Xtrain = response$X,
            Ytrain = response$Y,
            ncomp = as.integer(ncomp),
            fit = controls$fit,
            scaling = scaling,
            svd.method = solver,
            rsvd_oversample = controls$rsvd_oversample,
            rsvd_power = controls$rsvd_power,
            svds_tol = controls$svds_tol,
            seed = controls$seed
        ),
        gpu_device_state = FALSE,
        gpu_qr = controls$gpu_qr,
        gpu_eig = controls$gpu_eig,
        gpu_finalize_threshold = controls$gpu_finalize_threshold
    )
    model$xprod_default <- use_xprod
    model
}

.gpu_fit_controls <- function(
    fit,
    proj,
    oversample,
    power,
    tolerance,
    seed,
    ridge,
    device_state,
    qr,
    eig,
    threshold
) {
    list(
        fit = fit,
        proj = proj,
        rsvd_oversample = oversample,
        rsvd_power = power,
        svds_tol = tolerance,
        seed = seed,
        lda_ridge = ridge,
        gpu_device_state = device_state,
        gpu_qr = qr,
        gpu_eig = eig,
        gpu_finalize_threshold = threshold
    )
}

#' GPU-native SIMPLS fit
#'
#' Uses a CUDA-oriented `simpls` engine that keeps the training
#' matrices and deflated cross-covariance resident on device throughout the fit.
#'
#' @param Xtrain Numeric training predictor matrix.
#'   Alternatively, a result returned by [pls.single.cv()]. In that case
#'   `pls()` refits the selected model on the full cross-validation training
#'   set and predicts `Xtest` using `best_ncomp` and the selected tuning
#'   settings.
#' @param Ytrain Training response (numeric or factor).
#'   When `Xtrain` is a [pls.single.cv()] result, the second positional argument
#'   may be used as `Xtest`.
#' @param Xtest Optional test predictor matrix.
#' @param Ytest Optional observed response used to compute independent-test
#'   `Q2Y` relative to the training-response mean.
#' @param ncomp Number of components (scalar or vector).
#' @param scaling One of \code{centering}, \code{autoscaling}, or \code{none}.
#' @param rsvd_oversample RSVD oversampling.
#' @param rsvd_power RSVD power iterations.
#' @param svds_tol Tolerance placeholder passed through to the backend.
#' @param seed Random seed.
#' @param fit Return fitted values and `R2Y` when `TRUE`.
#' @param return_variance Compute predictor-space latent-variable variance
#'   explained. Set to `FALSE` for timing/memory benchmarks that do not need
#'   plotting variance metadata.
#' @param return_loadings Compute and store predictor loadings `P`. The default
#'   is `FALSE` because `P` is mainly used for interpretation/loading plots and
#'   is not needed for prediction, VIP, or the benchmark pipelines.
#' @param proj Return projected `Ttest` when `TRUE`.
#'  @param gpu_device_state Keep selected SIMPLS workspaces resident on the GPU
#' when `TRUE`.
#' @param gpu_qr Use GPU QR finalization when available.
#' @param gpu_eig Use GPU eigensolver finalization when available.
#'  @param gpu_finalize_threshold Component threshold controlling GPU-side
#' finalization.
#' @return A `fastPLS` object.
#' @noRd
.simpls_gpu <- function(Xtrain, Ytrain, Xtest = NULL, Ytest = NULL, ncomp = 2,
    scaling = c("centering", "autoscaling", "none"), rsvd_oversample = 32L,
    rsvd_power = 5L,
    svds_tol = 0, svd.method = "cuda_rsvd", seed = 1L, fit = FALSE,
    proj = FALSE,
    gpu_device_state = TRUE, gpu_qr = TRUE, gpu_eig = TRUE,
    gpu_finalize_threshold = 32L,
    classifier = c("argmax", "lda"), lda_ridge = 1e-08,
    return_variance = TRUE) {
    if (!has_cuda()) {
        stop(
            "simpls_gpu requires an available CUDA backend. ",
            "No CPU fallback is performed.",
            call. = FALSE
        )
    }
    svd.method <- match.arg(.normalize_svd_method(svd.method), c("cuda_rsvd"))
    scal <- pmatch(scaling, c("centering", "autoscaling", "none"))[1]
    if (is.factor(Ytrain) && .dense_indicator_exceeds_cuda_guard(nrow(Xtrain),
        nlevels(Ytrain))) {
        .stop_unsafe_cuda_simpls_response(nrow(Xtrain), nlevels(Ytrain))
    }
    response <- .gpu_pls_response(Xtrain, Ytrain, classifier, "simpls")
    use_xprod_default <- .should_use_xprod_default(ncol(response$X),
        ncol(response$Y),
        ncomp)
    fused_controls <- .gpu_fit_controls(fit, proj, rsvd_oversample, rsvd_power,
        svds_tol, seed, lda_ridge, gpu_device_state, gpu_qr, gpu_eig,
        gpu_finalize_threshold)
    fused_model <- .gpu_fused_lda(3L, "simpls", response, Xtest, Ytest, ncomp,
        scal, use_xprod_default, fused_controls)
    if (!is.null(fused_model)) {
        fused_model <- .maybe_attach_pls_variance_explained(fused_model,
            response$X,
            return_variance)
        fused_model <- .attach_backend_control(fused_model)
        return(fused_model)
    }
    model <- .gpu_simpls_core(response, ncomp, scal, svd.method,
        fused_controls,
        use_xprod_default)
    .gpu_pls_finalize(model, response, Xtest, Ytest, ncomp, fit, proj,
        "simpls",
        response$classifier, lda_ridge, return_variance)
}

#' GPU-native PLSSVD fit
#'
#' Uses a dedicated CUDA PLSSVD engine that keeps the cross-covariance SVD and
#' latent linear algebra on device, returning the standard `fastPLS` object
#' structure for prediction and plotting.
#'
#' @param Xtrain Numeric training predictor matrix.
#' @param Ytrain Training response (numeric or factor).
#' @param Xtest Optional test predictor matrix.
#' @param Ytest Optional observed response used to compute independent-test
#'   `Q2Y` relative to the training-response mean.
#' @param ncomp Number of components (scalar or vector).
#' @param scaling One of \code{centering}, \code{autoscaling}, or \code{none}.
#' @param rsvd_oversample RSVD oversampling.
#' @param rsvd_power RSVD power iterations.
#' @param svds_tol Tolerance placeholder passed through to the backend.
#' @param seed Random seed.
#' @param fit Return fitted values and `R2Y` when `TRUE`.
#' @param proj Return projected `Ttest` when `TRUE`.
#' @param gpu_qr Use GPU QR finalization when available.
#' @param gpu_eig Use GPU eigensolver finalization when available.
#'  @param gpu_finalize_threshold Component threshold controlling GPU-side
#' finalization.
#' @return A `fastPLS` object fitted with GPU PLSSVD.
#' @noRd
.plssvd_gpu <- function(Xtrain, Ytrain, Xtest = NULL, Ytest = NULL, ncomp = 2,
    scaling = c("centering", "autoscaling", "none"), rsvd_oversample = 32L,
    rsvd_power = 5L,
    svds_tol = 0, svd.method = "cuda_rsvd", seed = 1L, fit = FALSE,
    proj = FALSE,
    gpu_qr = TRUE, gpu_eig = TRUE, gpu_finalize_threshold = 32L,
    classifier = c("argmax",
        "lda"), lda_ridge = 1e-08, return_variance = TRUE) {
    if (!has_cuda()) {
        stop(
            "plssvd_gpu requires an available CUDA backend. ",
            "No CPU fallback is performed.",
            call. = FALSE
        )
    }
    svd.method <- match.arg(.normalize_svd_method(svd.method), c("cuda_rsvd"))
    scal <- pmatch(scaling, c("centering", "autoscaling", "none"))[1]
    response <- .gpu_pls_response(Xtrain, Ytrain, classifier, "plssvd")
    ncomp <- response$cap(ncomp)
    use_xprod_default <- .should_use_xprod_default(ncol(response$X),
        ncol(response$Y),
        ncomp)
    controls <- .gpu_fit_controls(fit, proj, rsvd_oversample, rsvd_power,
        svds_tol,
        seed, lda_ridge, FALSE, gpu_qr, gpu_eig, gpu_finalize_threshold)
    fused_model <- .gpu_fused_lda(1L, "plssvd", response, Xtest, Ytest, ncomp,
        scal, use_xprod_default, controls)
    if (!is.null(fused_model)) {
        fused_model <- .maybe_attach_pls_variance_explained(fused_model,
            response$X,
            return_variance)
        fused_model <- .attach_backend_control(fused_model)
        return(fused_model)
    }
    model <- .gpu_plssvd_core(response, ncomp, scal, svd.method, controls,
        use_xprod_default)
    .gpu_pls_finalize(model, response, Xtest, Ytest, ncomp, fit, proj,
        "plssvd",
        response$classifier, lda_ridge, return_variance)
}

.predict_flash_attach <- function(model, Xtest, Ytest, proj) {
    model$predict_backend <- "cuda_flash"
    model$flash_svd <- TRUE
    if (!is.null(Xtest)) {
        res <- predict.fastPLS(
            model,
            as.matrix(Xtest),
            Ytest = Ytest,
            proj = proj,
            backend = "cuda"
        )
        model <- c(model, res)
    }
    model
}

#' GPU PLS-SVD with compact low-rank CUDA prediction
#'
#' Fits with the standard GPU PLSSVD backend and marks the model so prediction
#' uses a CUDA low-rank path that applies `X %*% R %*% W` without materializing
#' the full coefficient matrix `B`.
#' @noRd
.plssvd_flash_gpu <- function(
    Xtrain,
    Ytrain,
    Xtest = NULL,
    Ytest = NULL,
    ncomp = 2,
    scaling = c("centering", "autoscaling", "none"),
    rsvd_oversample = 32L,
    rsvd_power = 5L,
    svds_tol = 0,
    seed = 1L,
    fit = FALSE,
    proj = FALSE,
    gpu_qr = TRUE,
    gpu_eig = TRUE,
    gpu_finalize_threshold = 32L
) {
    model <- .plssvd_gpu(
        Xtrain = Xtrain,
        Ytrain = Ytrain,
        Xtest = NULL,
        Ytest = NULL,
        ncomp = ncomp,
        scaling = scaling,
        rsvd_oversample = rsvd_oversample,
        rsvd_power = rsvd_power,
        svds_tol = svds_tol,
        seed = seed,
        fit = fit,
        proj = FALSE,
        gpu_qr = gpu_qr,
        gpu_eig = gpu_eig,
        gpu_finalize_threshold = gpu_finalize_threshold
    )
    .predict_flash_attach(model, Xtest, Ytest, proj)
}

#' GPU SIMPLS with compact low-rank CUDA prediction
#' @noRd
.simpls_flash_gpu <- function(
    Xtrain,
    Ytrain,
    Xtest = NULL,
    Ytest = NULL,
    ncomp = 2,
    scaling = c("centering", "autoscaling", "none"),
    rsvd_oversample = 32L,
    rsvd_power = 5L,
    svds_tol = 0,
    seed = 1L,
    fit = FALSE,
    proj = FALSE,
    gpu_device_state = TRUE,
    gpu_qr = TRUE,
    gpu_eig = TRUE,
    gpu_finalize_threshold = 32L
) {
    model <- .simpls_gpu(
        Xtrain = Xtrain,
        Ytrain = Ytrain,
        Xtest = NULL,
        Ytest = NULL,
        ncomp = ncomp,
        scaling = scaling,
        rsvd_oversample = rsvd_oversample,
        rsvd_power = rsvd_power,
        svds_tol = svds_tol,
        seed = seed,
        fit = fit,
        proj = FALSE,
        gpu_device_state = gpu_device_state,
        gpu_qr = gpu_qr,
        gpu_eig = gpu_eig,
        gpu_finalize_threshold = gpu_finalize_threshold
    )
    .predict_flash_attach(model, Xtest, Ytest, proj)
}

#' GPU OPLS with compact low-rank CUDA prediction
#' @noRd
.opls_flash_gpu <- function(
    Xtrain,
    Ytrain,
    Xtest = NULL,
    Ytest = NULL,
    ncomp = 2,
    north = 1L,
    scaling = c("centering", "autoscaling", "none"),
    rsvd_oversample = 32L,
    rsvd_power = 5L,
    svds_tol = 0,
    seed = 1L,
    fit = FALSE,
    proj = FALSE,
    ...
) {
    model <- .opls_cuda(
        Xtrain = Xtrain,
        Ytrain = Ytrain,
        Xtest = NULL,
        Ytest = NULL,
        ncomp = ncomp,
        north = north,
        scaling = scaling,
        rsvd_oversample = rsvd_oversample,
        rsvd_power = rsvd_power,
        svds_tol = svds_tol,
        seed = seed,
        fit = fit,
        proj = FALSE,
        ...
    )
    model$inner_model$predict_backend <- "cuda_flash"
    model$inner_model$flash_svd <- TRUE
    model$flash_svd <- TRUE
    if (!is.null(Xtest)) {
        res <- predict(
            model,
            as.matrix(Xtest),
            Ytest = Ytest,
            proj = proj,
            backend = "cuda"
        )
        model <- c(model, res)
        class(model) <- c("fastPLSOpls", "fastPLS")
    }
    model
}

#' GPU kernel PLS with compact low-rank CUDA prediction
#' @noRd
.kernel_pls_flash_gpu <- function(
    Xtrain,
    Ytrain,
    Xtest = NULL,
    Ytest = NULL,
    ncomp = 2,
    scaling = c("centering", "autoscaling", "none"),
    kernel = c("linear", "rbf", "poly"),
    gamma = NULL,
    degree = 3L,
    coef0 = 1,
    rsvd_oversample = 32L,
    rsvd_power = 5L,
    svds_tol = 0,
    seed = 1L,
    fit = FALSE,
    proj = FALSE,
    ...
) {
    model <- .kernel_pls_cuda(
        Xtrain = Xtrain,
        Ytrain = Ytrain,
        Xtest = NULL,
        Ytest = NULL,
        ncomp = ncomp,
        scaling = scaling,
        kernel = kernel,
        gamma = gamma,
        degree = degree,
        coef0 = coef0,
        rsvd_oversample = rsvd_oversample,
        rsvd_power = rsvd_power,
        svds_tol = svds_tol,
        seed = seed,
        fit = fit,
        proj = FALSE,
        ...
    )
    .kernel_pls_flash_attach(model, Xtest, Ytest, proj)
}

.kernel_pls_flash_attach <- function(model, Xtest, Ytest, proj) {
    model$inner_model$predict_backend <- "cuda_flash"
    model$inner_model$flash_svd <- TRUE
    model$flash_svd <- TRUE
    if (!is.null(Xtest)) {
        res <- predict(
            model,
            as.matrix(Xtest),
            Ytest = Ytest,
            proj = proj,
            backend = "cuda"
        )
        model <- c(model, res)
        class(model) <- c("fastPLSKernel", "fastPLS")
    }
    model
}

.cv_normalize_selection_metric <- function(selection_metric = NULL) {
    if (is.null(selection_metric) || !length(selection_metric)) {
        return("auto")
    }
    metric <- tolower(gsub(
        "[[:space:]-]+",
        "_",
        as.character(selection_metric[[1L]])
    ))
    aliases <- c(
        auto = "auto",
        acc = "accuracy",
        cv_accuracy = "accuracy",
        accuracy = "accuracy",
        balanced = "balanced_accuracy",
        balanced_acc = "balanced_accuracy",
        balancedaccuracy = "balanced_accuracy",
        bacc = "balanced_accuracy",
        balanced_accuracy = "balanced_accuracy",
        r2 = "r2",
        r_squared = "r2",
        rsquared = "r2",
        q2 = "q2",
        q_squared = "q2",
        rmsd = "rmsd",
        rmse = "rmsd"
    )
    if (metric %in% names(aliases)) {
        metric <- unname(aliases[[metric]])
    }
    if (
    !metric %in% c("auto", "accuracy", "balanced_accuracy", "r2", "q2", "rmsd")
    ) {
        stop(
            sprintf(
                "%s%s",
                "selection_metric must be one of 'auto', 'accuracy', ",
                "'balanced_accuracy', 'r2', 'q2', or 'rmsd'."
            ),
            call. = FALSE
        )
    }
    metric
}

.cv_balanced_accuracy <- function(observed, predicted, levels = NULL) {
    if (is.null(levels)) {
        levels <- unique(c(as.character(observed), as.character(predicted)))
        levels <- levels[!is.na(levels)]
    }
    observed <- factor(observed, levels = levels)
    predicted <- factor(predicted, levels = levels)
    tab <- table(observed, predicted)
    denominators <- rowSums(tab)
    recalls <- rep(NA_real_, length(levels))
    present <- denominators > 0
    recalls[present] <- diag(tab)[present] / denominators[present]
    if (any(is.finite(recalls))) mean(recalls, na.rm = TRUE) else NA_real_
}

.cv_selection_metric_from_dots <- function(dots) {
    if (!is.list(dots)) {
        dots <- list()
    }
    keys <- intersect(
        c("selection_metric", "metric", "opt_metric", "criterion"),
        names(dots)
    )
    if (!length(keys)) {
        return(list(metric = "auto", dots = dots))
    }
    metric <- dots[[keys[[1L]]]]
    dots[keys] <- NULL
    list(metric = .cv_normalize_selection_metric(metric), dots = dots)
}

.cv_metric_from_matrix <- function(
    Ytrue,
    Ypred,
    Ytrain = NULL,
    metric = "auto"
) {
    metric <- .cv_normalize_selection_metric(metric)
    Ytrue <- .float32_to_numeric_matrix(Ytrue)
    Ypred <- .float32_to_numeric_matrix(Ypred)
    if (!all(dim(Ytrue) == dim(Ypred))) {
        stop(
            "Ytrue and Ypred must have the same dimensions for CV metric ",
            "calculation.",
            call. = FALSE
        )
    }
    if (identical(metric, "auto")) {
        metric <- if (ncol(Ytrue) == 1L) "q2" else "rmsd"
    }
    if (metric %in% c("accuracy", "balanced_accuracy")) {
        stop(
            "Classification metrics are only available for factor responses.",
            call. = FALSE
        )
    }
    if (identical(metric, "rmsd")) {
        return(list(
            metric_name = "rmsd",
            metric_value = sqrt(mean((Ypred - Ytrue)^2, na.rm = TRUE))
        ))
    }
    if (identical(metric, "q2") && is.null(Ytrain)) {
        stop(
            "Q2 requires an explicit training-response reference; use the ",
            "fold-aware Q2 helper for cross-validation.",
            call. = FALSE
        )
    }
    Ytrain_mat <- if (!is.null(Ytrain)) {
        .float32_to_numeric_matrix(Ytrain)
    } else {
        Ytrue
    }
    center <- colMeans(Ytrain_mat, na.rm = TRUE)
    press <- sum((Ypred - Ytrue)^2, na.rm = TRUE)
    tss <- sum(sweep(Ytrue, 2L, center, "-")^2, na.rm = TRUE)
    list(
        metric_name = metric,
        metric_value = if (is.finite(tss) && tss > 0) {
            1 - press / tss
        } else {
            NA_real_
        }
    )
}

.cv_regression_q2_rmsd <- function(Ytrue, Ypred, Ytrain = NULL) {
    q2 <- .cv_metric_from_matrix(
        Ytrue,
        Ypred,
        Ytrain = Ytrain,
        metric = "q2"
    )$metric_value
    rmsd <- .cv_metric_from_matrix(
        Ytrue,
        Ypred,
        Ytrain = Ytrain,
        metric = "rmsd"
    )$metric_value
    list(Q2Y = q2, RMSD = rmsd)
}

.cv_classification_q2_path <- function(Ytrue, Ypred, lev, fold = NULL) {
    dims <- dim(Ypred)
    if (length(dims) != 3L) {
        return(NA_real_)
    }
    Ymat <- .fastpls_one_hot_labels(Ytrue, lev)
    if (!is.null(fold)) {
        return(.fastpls_fold_q2_path(Ymat, Ypred, fold))
    }
    vapply(
        seq_len(dims[[3L]]),
        function(i) {
            pred_i <- matrix(Ypred[, , i], nrow = dims[[1L]], ncol = dims[[2L]])
            if (!any(is.finite(pred_i))) {
                return(NA_real_)
            }
            .cv_metric_from_matrix(
                Ytrue = Ymat,
                Ypred = pred_i,
                Ytrain = Ymat,
                metric = "q2"
            )$metric_value
        },
        numeric(1)
    )
}

.cv_normalize_training_summary <- function(output, ncomp) {
    if (is.null(output$R2Y) || !length(output$R2Y)) {
        output$R2Y <- rep(NA_real_, length(ncomp))
    } else if (length(output$R2Y) != length(ncomp)) {
        output$R2Y <- rep_len(output$R2Y, length(ncomp))
    }
    output$R2Y <- .fastpls_name_metric_path(output$R2Y, ncomp)
    output
}

.cv_training_fit_summary <- function(Xdata, Ydata, ncomp, scaling, method,
    backend,
    svd.method, rsvd_oversample, rsvd_power, svds_tol, seed, north, kernel,
    gamma, degree,
    coef0) {
    out <- tryCatch({
        fit <- pls(Xtrain = Xdata, Ytrain = Ydata, ncomp = ncomp,
            scaling = scaling,
            method = method, svd.method = svd.method,
            rsvd_oversample = rsvd_oversample,
            rsvd_power = rsvd_power, seed = seed,
            fit = TRUE, return_variance = FALSE,
            proj = FALSE,
            backend = backend, north = north, kernel = kernel, gamma = gamma,
            degree = degree,
            coef0 = coef0)
        r2 <- fit$R2Y
        if (is.null(r2) || !length(r2)) {
            r2 <- rep(NA, length(ncomp))
        }
        list(R2Y = .fastpls_name_metric_path(r2, ncomp), Yfit = fit$Yfit)
    }, error = function(e) {
        list(R2Y = .fastpls_name_metric_path(rep(NA, length(ncomp)), ncomp),
            Yfit = NULL)
    })
    .cv_normalize_training_summary(out, ncomp)
}

.is_single_pls_cv_result <- function(x) {
    inherits(x, "fastPLSCV") ||
        (is.list(x) && !is.null(x$best_ncomp) && !is.null(x$tuning_config))
}

.cv_attach_fit_data <- function(res, Xdata, Ydata) {
    attr(res, "fit_data") <- list(Xdata = Xdata, Ydata = Ydata)
    class(res) <- unique(c("fastPLSCV", class(res)))
    res
}

.cv_drop_fit_data <- function(res) {
    attr(res, "fit_data") <- NULL
    class(res) <- setdiff(class(res), "fastPLSCV")
    res
}

.pls_cv_refit_args <- function(cv, fit_data, Xtest, Ytest, options) {
    params <- .cv_config_list(cv$tuning_config)
    args <- c(
        list(
            Xtrain = fit_data$Xdata,
            Ytrain = fit_data$Ydata,
            Xtest = Xtest,
            Ytest = Ytest,
            ncomp = as.integer(cv$best_ncomp[[1L]]),
            scaling = params$scaling %||% "centering",
            method = params$method %||% "simpls",
            svd.method = params$svd.method %||% "rsvd",
            classifier = params$classifier %||% "argmax",
            fit = options$fit,
            bycol = options$bycol,
            return_variance = options$return_variance,
            return_loadings = options$return_loadings,
            proj = options$proj,
            perm.test = options$perm.test,
            times = options$times,
            backend = params$backend %||% "cpu",
            north = params$north %||% 1L,
            kernel = params$kernel %||% "linear",
            gamma = params$gamma,
            degree = params$degree %||% 3L,
            coef0 = params$coef0 %||% 1
        ),
        cv$tuning_config$svd_dots %||% list()
    )
    args[!vapply(args, is.null, logical(1L))]
}

.pls_from_single_cv_result <- function(
    cv,
    Xtest = NULL,
    Ytest = NULL,
    fit = FALSE,
    bycol = FALSE,
    return_variance = TRUE,
    return_loadings = FALSE,
    proj = FALSE,
    perm.test = FALSE,
    times = 100
) {
    fit_data <- attr(cv, "fit_data", exact = TRUE)
if (is.null(fit_data) || is.null(fit_data$Xdata) || is.null(fit_data$Ydata)) {
        stop(
    "This pls.single.cv() result does not contain the training data needed ",
    "for automatic refitting. Please rerun pls.single.cv() with the current ",
            "fastPLS version, or call pls(Xtrain, Ytrain, ...) manually using ",
            "cv$best_parameters.",
            call. = FALSE
        )
    }
    cfg <- cv$tuning_config
    if (is.null(cfg)) {
        stop(
            "The pls.single.cv() result does not contain tuning_config.",
            call. = FALSE
        )
    }
    options <- list(
        fit = fit,
        bycol = bycol,
        return_variance = return_variance,
        return_loadings = return_loadings,
        proj = proj,
        perm.test = perm.test,
        times = times
    )
    args <- .pls_cv_refit_args(cv, fit_data, Xtest, Ytest, options)
    model <- do.call(pls, args)
    model$cv_best_parameters <- cv$best_parameters
    model$cv_best_metric_name <- cv$best_metric_name
    model$cv_best_metric_value <- cv$best_metric_value
    model
}

.cv_metric_frame <- function(values, name) {
    data.frame(
        ncomp_index = seq_along(values),
        metric_name = rep(name, length(values)),
        metric_value = values,
        stringsAsFactors = FALSE
    )
}

.cv_empty_metric_frame <- function(n) {
    data.frame(
        ncomp_index = seq_len(n), metric_name = character(n),
        metric_value = numeric(n), stringsAsFactors = FALSE
    )
}

.cv_classification_selection_metrics <- function(cv_res, Ydata,
    selection_metric) {
    if (identical(selection_metric, "auto")) selection_metric <- "accuracy"
    if (identical(selection_metric, "accuracy")) {
        return(cv_res$metrics)
    }
    if (identical(selection_metric, "balanced_accuracy")) {
        predictions <- cv_res$pred
        if (is.null(predictions)) {
            stop(
                "Stored class predictions are required to optimize ",
                "selection_metric = 'balanced_accuracy'.",
                call. = FALSE
            )
        }
        if (!is.list(predictions)) predictions <- list(predictions)
        values <- vapply(predictions, function(predicted) {
            .cv_balanced_accuracy(Ydata, predicted, levels = cv_res$levels)
        }, numeric(1L))
        return(.cv_metric_frame(values, "balanced_accuracy"))
    }
    if (identical(selection_metric, "q2")) {
        if (is.null(cv_res$Ypred)) {
            stop(
                "Stored classification scores are required to optimize ",
                "selection_metric = 'q2'.",
                call. = FALSE
            )
        }
        q2 <- .cv_classification_q2_path(Ydata, cv_res$Ypred, cv_res$levels,
            fold = cv_res$fold)
        return(.cv_metric_frame(q2, "q2"))
    }
    if (!identical(selection_metric, "r2")) {
        stop(
            "Classification CV can optimize selection_metric = ",
            "'accuracy', 'balanced_accuracy', or 'q2'.",
            call. = FALSE
        )
    }
    stop("Classification selection_metric = 'r2' is based on the full-data ",
        "training fit and cannot be optimized by held-out folds.",
        call. = FALSE)
}

.cv_regression_selection_metrics <- function(cv_res, Ydata, selection_metric) {
    if (identical(selection_metric, "auto")) selection_metric <- "rmsd"
    if (selection_metric %in% c("accuracy", "balanced_accuracy")) {
        stop(
            "Regression CV can only optimize selection_metric = 'r2', 'q2', ",
            "or 'rmsd'.",
            call. = FALSE
        )
    }
    if (!is.null(cv_res$metrics) && is.null(cv_res$Ypred)) {
        return(cv_res$metrics)
    }
    if (is.null(cv_res$Ypred)) {
        stop(
            "Stored CV predictions are required to optimize the requested ",
            "regression metric.",
            call. = FALSE
        )
    }
    dims <- dim(cv_res$Ypred)
    if (length(dims) != 3L) {
        stop("Internal CV prediction output must be a 3D array.", call. = FALSE)
    }
    metrics <- .cv_empty_metric_frame(dims[[3L]])
    for (i in seq_len(dims[[3L]])) {
        mat <- cv_res$Ypred[, , i, drop = TRUE]
    metric <- if (identical(selection_metric, "q2") && !is.null(cv_res$fold)) {
            list(
                metric_name = "q2",
            metric_value = .fastpls_fold_q2_path(Ydata, mat, cv_res$fold)[[1L]]
            )
        } else {
            .cv_metric_from_matrix(
                Ydata,
                mat,
                Ytrain = Ydata,
                metric = selection_metric
            )
        }
        metrics$metric_name[[i]] <- metric$metric_name
        metrics$metric_value[[i]] <- metric$metric_value
    }
    metrics
}

.cv_selection_metrics <- function(
    cv_res,
    Ydata,
    classification,
    selection_metric = "auto"
) {
    selection_metric <- .cv_normalize_selection_metric(selection_metric)
    if (classification) {
        return(.cv_classification_selection_metrics(
            cv_res,
            Ydata,
            selection_metric
        ))
    }
    .cv_regression_selection_metrics(cv_res, Ydata, selection_metric)
}

.decode_cv_predictions <- function(Ypred, Ydata, classification, lev) {
    if (classification && is.null(Ypred)) {
        stop(
            "Classification CV output is missing both class predictions and ",
            "score predictions",
            call. = FALSE
        )
    }
    dims <- dim(Ypred)
    if (length(dims) != 3L) {
        stop("Internal CV prediction output must be a 3D array")
    }
    out <- vector("list", dims[[3L]])
    metrics <- data.frame(
        ncomp_index = seq_len(dims[[3L]]),
        metric_name = character(dims[[3L]]),
        metric_value = numeric(dims[[3L]]),
        stringsAsFactors = FALSE
    )
    for (i in seq_len(dims[[3L]])) {
        mat <- Ypred[, , i, drop = TRUE]
        if (classification) {
        pred <- factor(lev[max.col(mat, ties.method = "first")], levels = lev)
            out[[i]] <- pred
            metrics$metric_name[[i]] <- "accuracy"
            metrics$metric_value[[i]] <- mean(
                as.character(pred) == as.character(Ydata),
                na.rm = TRUE
            )
        } else {
            out[[i]] <- as.matrix(mat)
            metric <- .cv_metric_from_matrix(Ydata, mat, Ytrain = Ydata)
            metrics$metric_name[[i]] <- metric$metric_name
            metrics$metric_value[[i]] <- metric$metric_value
        }
    }
    list(
        pred = if (length(out) == 1L) out[[1L]] else out,
        metrics = metrics
    )
}

.decode_cv_class_predictions <- function(class_pred, Ydata, lev) {
    pred_mat <- as.matrix(class_pred)
    out <- vector("list", ncol(pred_mat))
    metrics <- data.frame(
        ncomp_index = seq_len(ncol(pred_mat)),
        metric_name = rep("accuracy", ncol(pred_mat)),
        metric_value = numeric(ncol(pred_mat)),
        stringsAsFactors = FALSE
    )
    for (i in seq_len(ncol(pred_mat))) {
        idx <- as.integer(pred_mat[, i])
        ok <- is.finite(idx) & idx >= 1L & idx <= length(lev)
        pred <- rep(NA_character_, length(idx))
        pred[ok] <- lev[idx[ok]]
        pred <- factor(pred, levels = lev)
        out[[i]] <- pred
        metrics$metric_value[[i]] <- mean(
            as.character(pred) == as.character(Ydata),
            na.rm = TRUE
        )
    }
    list(pred = out, metrics = metrics)
}

.compiled_cv_response <- function(Ydata, kodama_class_codes, float32 = FALSE) {
    classification <- is.factor(Ydata)
    if (classification) {
        Ydata <- droplevels(Ydata)
        levels <- levels(Ydata)
        original <- Ydata
        matrix <- matrix(as.integer(Ydata), ncol = 1L)
        responses <- length(levels)
    } else {
        levels <- NULL
        matrix <- if (isTRUE(float32)) {
            .as_float32_matrix(Ydata, "Ydata")
        } else {
            as.matrix(Ydata)
        }
        original <- matrix
        responses <- ncol(matrix)
    }
    codes <- matrix(numeric(0), 0L, 0L)
    backend_responses <- responses
    if (!is.null(kodama_class_codes)) {
        if (!classification) {
            stop(
                "KODAMA Gaussian class-code CV requires factor responses.",
                call. = FALSE
            )
        }
        codes <- as.matrix(kodama_class_codes)
        if (nrow(codes) != responses || ncol(codes) < 1L) {
            stop("kodama_class_codes has invalid dimensions.", call. = FALSE)
        }
        backend_responses <- ncol(codes)
    }
    list(
        classification = classification,
        levels = levels,
        original = original,
        matrix = matrix,
        responses = responses,
        backend_responses = backend_responses,
        codes = codes
    )
}

.compiled_cv_solver <- function(backend, method) {
    method <- match.arg(
        .normalize_svd_method(method),
        c("cpu_rsvd", "cuda_rsvd", "metal_rsvd")
    )
    if (identical(backend, "cpp")) {
        if (!method %in% c("cpu_rsvd")) {
            stop("CPU CV supports rSVD only.", call. = FALSE)
        }
    } else if (identical(backend, "cuda")) {
        method <- .backend_svd_method(method, "cuda")
    } else {
        method <- "metal_rsvd"
    }
    list(name = method, id = .svd_method_id(method))
}

.compiled_cv_xprod <- function(value, backend, solver, X, q, ncomp) {
    if (!is.null(value)) {
        return(isTRUE(value))
    }
    if (identical(backend, "cuda")) {
        return(.should_use_xprod_default(ncol(X), q, ncomp))
    }
    if (identical(backend, "metal")) {
        return(FALSE)
    }
    if (identical(solver, "cpu_rsvd")) {
        return(.should_use_xprod_default(ncol(X), q, ncomp))
    }
    FALSE
}

.compiled_cv_context <- function(Xdata, Ydata, constrain, ncomp, scaling,
    method,
    backend, svd.method, xprod, kodama_class_codes, classifier) {
    method <- match.arg(method, c("plssvd", "simpls", "opls", "kernelpls"))
    backend <- match.arg(backend, c("cpp", "cuda", "metal"))
    classifier <- .normalize_classifier_public(classifier)
    float32 <- .has_float32_input(Xdata, Ydata)
    Xdata <- if (float32) {
        .as_float32_matrix(Xdata, "Xdata")
    } else {
        as.matrix(Xdata)
    }
    if (is.null(constrain)) {
        constrain <- seq_len(nrow(Xdata))
    }
    constrain <- as.integer(as.factor(constrain))
    response <- .compiled_cv_response(Ydata, kodama_class_codes, float32)
    ncomp <- as.integer(ncomp)
    if (identical(method, "plssvd")) {
        ncomp <- .cap_plssvd_ncomp(ncomp, nrow(Xdata), ncol(Xdata),
            response$responses,
            factor_response = response$classification, warn = TRUE)$ncomp
    }
    if (identical(backend, "cuda") && !has_cuda()) {
        .fastpls_require_backend_available("cuda", "Cross-validation")
    }
    if (identical(backend, "metal") && !isTRUE(has_metal())) {
        .fastpls_require_backend_available("metal", "Cross-validation")
    }
    solver <- .compiled_cv_solver(backend, svd.method)
    list(X = Xdata, constrain = constrain, ncomp = ncomp, method = method,
        method_id = .normalize_pls_method(method),
        backend = backend, backend_id = match(backend, c("cpp", "cuda",
            "metal")) -
            1L, scaling = pmatch(scaling, c("centering", "autoscaling",
            "none"))[1L],
        solver = solver, response = response, classifier = classifier,
        float32 = float32,
        classifier_id = switch(classifier,
            argmax = 0L, lda = 1L), xprod = .compiled_cv_xprod(xprod, backend,
            solver$name, Xdata, response$backend_responses, ncomp))
}

.compiled_cv_call <- function(context, controls) {
    if (!is.null(controls$seed)) {
        .fastpls_set_seed(controls$seed)
    }
    core_route <- identical(context$backend, "cpp") &&
        context$method %in% c("plssvd", "simpls", "opls") &&
        !isTRUE(context$xprod) &&
        length(context$response$codes) == 0L
    if (core_route) {
        labels <- if (context$response$classification) {
            as.integer(context$response$matrix[, 1L])
        } else {
            NULL
        }
        folds <- cv_folds_core_cpp(
            groups = context$constrain,
            labels = labels,
            class_count = if (is.null(labels)) 0L else context$response$responses,
            folds = .compiled_cv_kfold_arg(controls$kfold, context$constrain)
        )
        method_id <- if (identical(context$method, "plssvd")) 1L else 3L
        if (context$response$classification) {
            if (identical(context$method, "opls")) {
                runner <- if (context$float32) {
                    pls_cv_opls_classification_float32_core_cpp
                } else {
                    pls_cv_opls_classification_core_cpp
                }
                result <- runner(
                    predictors = context$X,
                    labels = labels,
                    class_count = context$response$responses,
                    folds = folds,
                    components = context$ncomp,
                    scaling = context$scaling,
                    classifier = context$classifier_id,
                    north = as.integer(controls$north),
                    oversample = as.integer(controls$oversample),
                    power = as.integer(controls$power),
                    seed = as.integer(controls$seed),
                    store_predictions = isTRUE(controls$store_predictions),
                    store_scores = isTRUE(controls$store_predictions) &&
                        isTRUE(controls$return_scores)
                )
            } else {
                runner <- if (context$float32) {
                    pls_cv_classification_float32_core_cpp
                } else {
                    pls_cv_classification_core_cpp
                }
                result <- runner(
                    predictors = context$X,
                    labels = labels,
                    class_count = context$response$responses,
                    folds = folds,
                    components = context$ncomp,
                    scaling = context$scaling,
                    method = method_id,
                    classifier = context$classifier_id,
                    oversample = as.integer(controls$oversample),
                    power = as.integer(controls$power),
                    seed = as.integer(controls$seed),
                    store_predictions = isTRUE(controls$store_predictions),
                    store_scores = isTRUE(controls$store_predictions) &&
                        isTRUE(controls$return_scores)
                )
            }
        } else {
            if (identical(context$method, "opls")) {
                runner <- if (context$float32) {
                    pls_cv_opls_regression_float32_core_cpp
                } else {
                    pls_cv_opls_regression_core_cpp
                }
                result <- runner(
                    predictors = context$X,
                    responses = context$response$matrix,
                    folds = folds,
                    components = context$ncomp,
                    scaling = context$scaling,
                    metric = .cv_metric_id(
                        controls$selection_metric,
                        context$response$classification
                    ),
                    north = as.integer(controls$north),
                    oversample = as.integer(controls$oversample),
                    power = as.integer(controls$power),
                    seed = as.integer(controls$seed),
                    store_predictions = isTRUE(controls$store_predictions)
                )
            } else {
                runner <- if (context$float32) {
                    pls_cv_regression_float32_core_cpp
                } else {
                    pls_cv_regression_core_cpp
                }
                result <- runner(
                    predictors = context$X,
                    responses = context$response$matrix,
                    folds = folds,
                    components = context$ncomp,
                    scaling = context$scaling,
                    method = method_id,
                    metric = .cv_metric_id(
                        controls$selection_metric,
                        context$response$classification
                    ),
                    oversample = as.integer(controls$oversample),
                    power = as.integer(controls$power),
                    seed = as.integer(controls$seed),
                    store_predictions = isTRUE(controls$store_predictions)
                )
            }
        }
        metric_name <- if (context$response$classification) {
            "accuracy"
        } else {
            switch(
                as.character(.cv_metric_id(
                    controls$selection_metric,
                    context$response$classification
                )),
                `2` = "r2", `3` = "q2", "rmsd"
            )
        }
        result$metrics <- data.frame(
            ncomp_index = seq_along(context$ncomp),
            metric_name = rep(metric_name, length(context$ncomp)),
            metric_value = as.numeric(result$metric_value),
            stringsAsFactors = FALSE
        )
        result$metric_value <- NULL
        result$method <- context$method
        result$backend <- "cpp"
        result$prediction_backend <- if (context$response$classification &&
            identical(context$classifier, "lda")) {
            "cpp_lda_cv"
        } else {
            "cpu"
        }
        result$classifier <- context$classifier
        result$xprod <- FALSE
        result$stratified_folds <- context$response$classification
        result$score_predictions_stored <- !is.null(result$Ypred)
        return(result)
    }
    pls_cv_predict_compiled(
        Xdata = context$X,
        Ydata = context$response$matrix,
        constrain = context$constrain,
        ncomp = context$ncomp,
        scaling = context$scaling,
        kfold = .compiled_cv_kfold_arg(
            controls$kfold,
            context$constrain
        ),
        method = context$method_id,
        backend = context$backend_id,
        svd_method = context$solver$id,
        rsvd_oversample = as.integer(controls$oversample),
        rsvd_power = as.integer(controls$power),
        svds_tol = controls$svds_tol,
        seed = as.integer(controls$seed),
        classification = context$response$classification,
        n_response = as.integer(context$response$responses),
        xprod = context$xprod,
        opls_north = as.integer(controls$north),
        return_scores = isTRUE(controls$return_scores),
        class_codes = context$response$codes,
        classifier = context$classifier_id,
        lda_ridge = controls$lda_ridge,
        store_predictions = isTRUE(controls$store_predictions),
        metric_id = .cv_metric_id(
            controls$selection_metric,
            context$response$classification
        )
    )
}

.compiled_cv_run_backend <- function(context, controls) {
    run <- function() .compiled_cv_call(context, controls)
    profiled <- if (context$method %in% c("simpls", "opls", "kernelpls")) {
        function() .with_fastpls_fast_options(run())
    } else {
        run
    }
    if (identical(context$backend, "cuda")) {
        family <- context$method %in% c("simpls", "opls", "kernelpls")
        value <- .with_gpu_native_options(
            profiled(),
            gpu_device_state = family,
            gpu_qr = controls$gpu_qr,
            gpu_eig = controls$gpu_eig,
            gpu_finalize_threshold = controls$gpu_finalize_threshold
        )
        if (family && context$xprod) {
            value <- .with_simpls_gpu_xprod(value)
        }
        return(value)
    }
    if (identical(context$backend, "cpp")) {
        return(profiled())
    }
    profiled()
}

.compiled_cv_decode <- function(result, context, return_scores) {
    response <- context$response
    decoded <- if (response$classification && !is.null(result$class_pred)) {
        .decode_cv_class_predictions(
            result$class_pred,
            response$original,
            response$levels
        )
    } else if (!is.null(result$Ypred)) {
        .decode_cv_predictions(
            result$Ypred,
            response$original,
            response$classification,
            response$levels
        )
    } else {
        list(pred = NULL, metrics = result$metrics)
    }
    if (response$classification && !is.null(result$Ypred)) {
        result$Yscore <- result$Ypred
        result$Q2Y <- .cv_classification_q2_path(
            response$original,
            result$Ypred,
            response$levels,
            fold = result$fold
        )
        if (!isTRUE(return_scores)) result$Ypred <- NULL
    }
    if (response$classification && !is.null(decoded$metrics)) {
        result$accuracy <- as.numeric(decoded$metrics$metric_value)
    }
    result$pred <- decoded$pred
    result$metrics <- decoded$metrics
    result$classification <- response$classification
    result$levels <- response$levels
    if (identical(result$backend, "cpp")) {
        result$backend <- "cpu"
    }
    result
}

.pls_cv_compiled <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L,
    kfold = 10L,
    scaling = c("centering", "autoscaling", "none"), method = c("plssvd",
        "simpls",
        "opls", "kernelpls"), backend = c("cpp", "cuda", "metal"),
    svd.method = "rsvd", rsvd_oversample = 32L,
    rsvd_power = 5L, svds_tol = 0,
    seed = 1L, xprod = NULL, north = 1L, return_scores = FALSE,
    kodama_class_codes = NULL,
    classifier = c("argmax", "lda"), lda_ridge = 1e-08, gpu_qr = TRUE,
    gpu_eig = TRUE,
    gpu_finalize_threshold = 32L, store_predictions = TRUE,
    selection_metric = "auto") {
    context <- .compiled_cv_context(Xdata, Ydata, constrain, ncomp, scaling,
        method,
        backend, svd.method, xprod, kodama_class_codes, classifier)
    controls <- list(kfold = kfold, oversample = rsvd_oversample,
        power = rsvd_power,
        svds_tol = svds_tol, seed = seed, north = north,
        return_scores = return_scores,
        lda_ridge = lda_ridge, store_predictions = store_predictions,
        selection_metric = selection_metric,
        gpu_qr = gpu_qr, gpu_eig = gpu_eig,
        gpu_finalize_threshold = gpu_finalize_threshold)
    result <- .compiled_cv_run_backend(context, controls)
    .compiled_cv_decode(result, context, return_scores)
}

.is_loocv_kfold <- function(kfold) {
    if (is.character(kfold)) {
        key <- tolower(trimws(kfold[[1L]]))
        return(
            key %in%
            c("loocv", "loo", "leave-one-out", "leave_one_out", "leave one out")
        )
    }
    FALSE
}

.cv_kfold_int <- function(kfold, n_groups, context = "cross-validation") {
    if (.is_loocv_kfold(kfold)) {
        return(as.integer(n_groups))
    }
    if (length(kfold) != 1L || is.na(kfold)) {
    stop(context, ": kfold must be a single integer or 'loocv'.", call. = FALSE)
    }
    kfold_int <- .fastpls_quiet(as.integer(kfold))
    if (is.na(kfold_int) || !is.finite(kfold_int)) {
    stop(context, ": kfold must be a finite integer or 'loocv'.", call. = FALSE)
    }
    if (kfold_int >= n_groups) {
        return(as.integer(n_groups))
    }
    max(2L, kfold_int)
}

.cv_is_leave_one_group_out <- function(kfold, n_groups) {
    .is_loocv_kfold(kfold) || .cv_kfold_int(kfold, n_groups) >= n_groups
}

.compiled_cv_kfold_arg <- function(kfold, constrain) {
    n_groups <- length(unique(as.integer(as.factor(constrain))))
    if (.cv_is_leave_one_group_out(kfold, n_groups)) {
        return(-1L)
    }
    .cv_kfold_int(kfold, n_groups, context = "compiled cross-validation")
}

.make_single_cv_folds <- function(Ydata, constrain, kfold, seed) {
    n <- if (is.matrix(Ydata) || is.data.frame(Ydata)) {
        nrow(Ydata)
    } else {
        length(Ydata)
    }
    if (is.null(constrain)) {
        constrain <- seq_len(n)
    }
    constrain <- as.integer(as.factor(constrain))
    groups <- sort(unique(constrain))
    n_groups <- length(groups)
    if (n_groups < 1L) {
        stop(
            "cross-validation requires at least one constraint group.",
            call. = FALSE
        )
    }
    group_fold <- integer(length(groups))
    names(group_fold) <- as.character(groups)
    if (.cv_is_leave_one_group_out(kfold, n_groups)) {
        group_fold[] <- seq_along(groups) - 1L
        return(as.integer(group_fold[as.character(constrain)]))
    }
    kfold <- .cv_kfold_int(kfold, n_groups)
    .fastpls_set_seed(seed)
    if (is.factor(Ydata)) {
        first_group_class <- vapply(
            groups,
            function(g) as.character(Ydata[which(constrain == g)[1L]]),
            character(1)
        )
        for (cls in unique(first_group_class)) {
            idx <- which(first_group_class == cls)
            idx <- sample(idx, length(idx))
            group_fold[idx] <- (seq_along(idx) - 1L) %% kfold
        }
    } else {
        idx <- sample(seq_along(groups), length(groups))
        group_fold[idx] <- (seq_along(idx) - 1L) %% kfold
    }
    as.integer(group_fold[as.character(constrain)])
}

.cv_class_predictions_from_fit <- function(fit, component_index, ntest) {
    pred <- fit$Ypred
    if (is.data.frame(pred) || is.list(pred)) {
        return(as.character(pred[[component_index]]))
    }
    if (is.matrix(pred)) {
        if (ncol(pred) >= component_index) {
            return(as.character(pred[, component_index]))
        }
        if (ncol(pred) == 1L) {
            return(as.character(pred[, 1L]))
        }
    }
    if (length(pred) == ntest) {
        return(as.character(pred))
    }
    stop(
        "Could not extract classification predictions from fold fit.",
        call. = FALSE
    )
}

.cv_regression_predictions_from_fit <- function(
    fit,
    component_index,
    ntest,
    q_response
) {
    pred <- fit$Ypred
    dims <- dim(pred)
    if (length(dims) == 3L) {
        return(matrix(
            pred[, , component_index, drop = TRUE],
            nrow = ntest,
            ncol = q_response
        ))
    }
    if (is.data.frame(pred)) {
        pred <- as.matrix(pred)
    }
    if (is.list(pred) && length(pred) >= component_index) {
        selected <- .float32_to_numeric_matrix(pred[[component_index]])
        if (nrow(selected) == ntest && ncol(selected) == q_response) {
            return(selected)
        }
    }
    if (is.matrix(pred)) {
        if (q_response == 1L && ncol(pred) >= component_index) {
            return(matrix(pred[, component_index], nrow = ntest, ncol = 1L))
        }
        if (ncol(pred) == q_response) {
            return(matrix(pred, nrow = ntest, ncol = q_response))
        }
    }
    if (length(pred) == ntest * q_response) {
        return(matrix(pred, nrow = ntest, ncol = q_response))
    }
stop("Could not extract regression predictions from fold fit.", call. = FALSE)
}

.via_pls_response <- function(Ydata) {
    classification <- is.factor(Ydata)
    original <- if (classification) Ydata else as.matrix(Ydata)
    levels <- if (classification) levels(Ydata) else NULL
    responses <- if (classification) length(levels) else ncol(original)
    list(
        classification = classification,
        original = original,
        levels = levels,
        responses = responses
    )
}

.via_pls_ncomp <- function(ncomp, method, X, response) {
    ncomp <- as.integer(ncomp)
    if (!identical(method, "plssvd")) {
        return(ncomp)
    }
    .cap_plssvd_ncomp(
        ncomp,
        nrow(X),
        ncol(X),
        response$responses,
        factor_response = response$classification,
        warn = TRUE
    )$ncomp
}

.via_pls_validate_route <- function(backend, xprod) {
    if (identical(backend, "cuda") && !isTRUE(has_cuda())) {
        .fastpls_require_backend_available("cuda", "Cross-validation")
    }
    if (identical(backend, "metal") && !isTRUE(has_metal())) {
        .fastpls_require_backend_available("metal", "Cross-validation")
    }
    if (!is.null(xprod)) {
        warning(
            "Explicit xprod is ignored; fold fits use backend defaults.",
            call. = FALSE
        )
    }
}

.via_pls_context <- function(Xdata, Ydata, constrain, ncomp, kfold, scaling,
    method,
    backend, svd.method, seed, xprod, kernel, classifier, dots) {
    method <- match.arg(method, c("plssvd", "simpls", "opls", "kernelpls"))
    backend <- match.arg(
        backend,
        c("cpu", "cuda", "metal")
    )
    scaling <- match.arg(scaling, c("centering", "autoscaling", "none"))
    classifier <- .resolve_classifier_for_backend(classifier, backend)
    control <- .resolve_svd_control(svd.method = svd.method,
        dots = c(.svd_control_from_dots(dots)$dots,
            list(seed = seed)), context = ".pls_cv_via_pls()")
    control <- .apply_pls_rsvd_controls(
        control,
        backend,
        ".pls_cv_via_pls()",
        method,
        is.factor(Ydata) || is.character(Ydata),
        Xdata,
        Ydata
    )
    control$svd.method <- match.arg(
        .normalize_svd_method(
            .backend_svd_method(control$svd.method, backend)
        ),
        c("cpu_rsvd", "cuda_rsvd", "metal_rsvd")
    )
    Xdata <- as.matrix(Xdata)
    if (is.null(constrain)) {
        constrain <- seq_len(nrow(Xdata))
    }
    constrain <- as.integer(as.factor(constrain))
    .via_pls_validate_route(backend, xprod)
    response <- .via_pls_response(Ydata)
    ncomp <- .via_pls_ncomp(ncomp, method, Xdata, response)
    fold_response <- if (response$classification) {
        Ydata
    } else if (inherits(response$original, "float32")) {
        as.numeric(float::dbl(response$original[, 1L, drop = FALSE]))
    } else {
        response$original[, 1L]
    }
    fold <- .make_single_cv_folds(
        fold_response, constrain, kfold, as.integer(control$seed)
    )
    list(X = Xdata, Y = Ydata, original = response$original,
        levels = response$levels,
        classification = response$classification,
        responses = response$responses,
        ncomp = ncomp, fold = fold, scaling = scaling, method = method,
        backend = backend,
        kernel = match.arg(kernel, c("linear", "rbf", "poly")),
        classifier = classifier, control = control)
}

.via_pls_state <- function(context, store_predictions, selection_metric) {
    n <- nrow(context$X)
    slices <- length(context$ncomp)
    class_pred <- if (context$classification && store_predictions) {
        matrix(NA_integer_, n, slices)
    } else {
        NULL
    }
    score_pred <- if (context$classification || store_predictions) {
        array(NA_real_, c(n, context$responses, slices))
    } else {
        NULL
    }
    metric_id <- .cv_metric_id(selection_metric, context$classification)
    tss <- if (!context$classification && metric_id %in% c(2L, 3L)) {
        center <- colMeans(context$original, na.rm = TRUE)
        sum(sweep(context$original, 2L, center, "-")^2, na.rm = TRUE)
    } else {
        NA_real_
    }
    list(
        class_pred = class_pred,
        score_pred = score_pred,
        correct = numeric(slices),
        total = numeric(slices),
        sse = numeric(slices),
        count = numeric(slices),
        metric_id = metric_id,
        tss = tss
    )
}

.via_pls_fit_fold <- function(context, config, train, test, fold_id) {
    Xtrain <- context$X[train, , drop = FALSE]
    Ytrain <- if (context$classification) {
        context$Y[train]
    } else {
        context$original[train, , drop = FALSE]
    }
    Ytest <- if (context$classification) {
        NULL
    } else {
        context$original[test, , drop = FALSE]
    }
    ctl <- context$control
    use_minimal_accelerator_fit <-
        !identical(Sys.getenv("FASTPLS_CV_MINIMAL_ACCELERATOR_FIT", "1"), "0")
    if (identical(context$backend, "cuda") &&
        use_minimal_accelerator_fit) {
        resident_method <- context$method
        if (identical(resident_method, "kernelpls") &&
            identical(context$kernel, "linear")) {
            resident_method <- "simpls"
        }
        resident_context <- list(
            Xtrain = Xtrain,
            Ytrain = Ytrain,
            Xtest = NULL,
            Ytest = NULL,
            float32 = .has_float32_input(
                Xtrain, Ytrain
            ),
            classification = context$classification,
            method = resident_method,
            scal = pmatch(
                context$scaling,
                c("centering", "autoscaling", "none")
            )[[1L]],
            classifier = context$classifier,
            control = ctl
        )
        resident_config <- list(
            ncomp = context$ncomp,
            north = config$north,
            kernel = context$kernel,
            gamma = config$gamma,
            degree = config$degree,
            coef0 = config$coef0,
            fit = FALSE,
            proj = FALSE,
            return_loadings = FALSE,
            return_variance = FALSE,
            perm.test = FALSE,
            cv_internal = TRUE
        )
        return(.pls_fit_resident_cuda(resident_context, resident_config))
    }
    # Classification scoring needs both decoded labels and the dummy-response
    # path. Defer prediction so the resident backend can project the test fold
    # once for all requested component prefixes.
    Xtest <- if (context$classification) {
        NULL
    } else {
        context$X[test, , drop = FALSE]
    }
    pls(
        Xtrain = Xtrain,
        Ytrain = Ytrain,
        Xtest = Xtest,
        Ytest = Ytest,
        ncomp = context$ncomp,
        scaling = context$scaling,
        method = context$method,
        svd.method = ctl$svd.method,
        rsvd_oversample = ctl$rsvd_oversample,
        rsvd_power = ctl$rsvd_power,
        seed = as.integer(ctl$seed) + as.integer(fold_id),
        fit = FALSE,
        proj = FALSE,
        return_variance = FALSE,
        backend = context$backend,
        north = config$north,
        kernel = context$kernel,
        gamma = config$gamma,
        degree = config$degree,
        coef0 = config$coef0,
        classifier = context$classifier
    )
}

.via_pls_fallback_fold <- function(state, context, train, test) {
    label <- names(which.max(table(context$Y[train])))
    index <- match(label, context$levels)
    if (!is.null(state$class_pred)) {
        state$class_pred[test, ] <- index
    }
    for (slice in seq_along(context$ncomp)) {
        predicted <- rep(label, length(test))
        state$correct[[slice]] <- state$correct[[slice]] +
            sum(predicted == as.character(context$Y[test]), na.rm = TRUE)
        state$total[[slice]] <- state$total[[slice]] + length(test)
    }
    state
}

.via_pls_class_fold <- function(state, context, fit, test) {
    minimal_resident_fit <- isTRUE(fit$cv_internal)
    prediction_backend <- if (minimal_resident_fit) context$backend else switch(
        context$backend,
        cuda = "cuda_flash",
        metal = "cpu",
        "cpu"
    )
    Xtest <- context$X[test, , drop = FALSE]
    internal_fit <- .fastpls_restore_internal_output_fields(fit)
    combined <- tryCatch({
        if (identical(context$backend, "cuda")) {
            .resident_cuda_cv_classification_path(internal_fit, Xtest)
        } else if (identical(context$backend, "metal") &&
            minimal_resident_fit) {
            .resident_metal_cv_classification_path(internal_fit, Xtest)
        } else {
            # Q2 uses dummy-response PLS predictions, not LDA scores.
            score_fit <- internal_fit
            score_fit$classification_rule <- "argmax"
            classified <- predict(fit, Xtest, backend = prediction_backend)
            raw <- predict(
                score_fit, Xtest, raw_scores = TRUE,
                backend = prediction_backend
            )
            list(classified = classified, raw = raw)
        }
    }, error = identity)
    if (inherits(combined, "error")) {
        stop(
            "Fold classification prediction failed: ",
            conditionMessage(combined),
            call. = FALSE
        )
    }
    if (identical(context$backend, "cuda") ||
        (identical(context$backend, "metal") && minimal_resident_fit)) {
        classified <- combined
        raw <- combined
    } else {
        classified <- combined$classified
        raw <- combined$raw
    }
    scores <- raw$Yscore %||% raw$Ypred_scores
    local_levels <- as.character(internal_fit$lev %||% context$levels)
    global_columns <- match(local_levels, context$levels)
    global_columns <- global_columns[!is.na(global_columns)]
    if (is.list(scores) && length(scores) >= length(context$ncomp)) {
        for (slice in seq_along(context$ncomp)) {
            local_scores <- .float32_to_numeric_matrix(scores[[slice]])
            state$score_pred[test, , slice] <- 0
            state$score_pred[test, global_columns, slice] <-
                local_scores[, seq_along(global_columns), drop = FALSE]
        }
    } else if (!is.null(scores) && length(dim(scores)) == 3L) {
        for (slice in seq_along(context$ncomp)) {
            local_scores <- matrix(
                scores[, , slice],
                nrow = length(test),
                ncol = dim(scores)[2L]
            )
            state$score_pred[test, , slice] <- 0
            state$score_pred[test, global_columns, slice] <-
                local_scores[, seq_along(global_columns), drop = FALSE]
        }
    }
    for (slice in seq_along(context$ncomp)) {
        predicted <- .cv_class_predictions_from_fit(
            classified,
            slice,
            length(test)
        )
        if (!is.null(state$class_pred)) {
            state$class_pred[test, slice] <- match(
                predicted,
                context$levels
            )
        }
        state$correct[[slice]] <- state$correct[[slice]] +
            sum(predicted == as.character(context$Y[test]), na.rm = TRUE)
        state$total[[slice]] <- state$total[[slice]] + length(test)
    }
    state
}

.via_pls_regression_fold <- function(state, context, fit, test) {
    observed <- context$original[test, , drop = FALSE]
    if (!is.null(fit$resident_state)) {
        predicted <- if (identical(context$backend, "cuda")) {
            .resident_cuda_predict(fit, context$X[test, , drop = FALSE], observed)
        } else {
            .resident_metal_predict(fit, context$X[test, , drop = FALSE], observed)
        }
        fit[names(predicted)] <- predicted
    }
    for (slice in seq_along(context$ncomp)) {
        predicted <- .cv_regression_predictions_from_fit(
            fit,
            slice,
            length(test),
            context$responses
        )
        if (!is.null(state$score_pred)) {
            state$score_pred[test, , slice] <- predicted
        }
        difference <- predicted - observed
        state$sse[[slice]] <- state$sse[[slice]] +
            sum(difference^2, na.rm = TRUE)
        state$count[[slice]] <- state$count[[slice]] +
            sum(is.finite(difference))
    }
    state
}

.via_pls_run_folds <- function(context, state, config) {
    for (fold_id in sort(unique(context$fold))) {
        test <- which(context$fold == fold_id)
        train <- which(context$fold != fold_id)
        if (!length(test) || !length(train)) {
            next
        }
        if (
            context$classification &&
                length(unique(context$Y[train])) < 2L
        ) {
            state <- .via_pls_fallback_fold(state, context, train, test)
            next
        }
        fit <- .via_pls_fit_fold(context, config, train, test, fold_id)
        state <- if (context$classification) {
            .via_pls_class_fold(state, context, fit, test)
        } else {
            .via_pls_regression_fold(state, context, fit, test)
        }
    }
    state
}

.via_pls_metrics <- function(context, state) {
    slices <- length(context$ncomp)
    metric_name <- if (context$classification) {
        rep("accuracy", slices)
    } else if (state$metric_id == 2L) {
        rep("r2", slices)
    } else if (state$metric_id == 3L) {
        rep("q2", slices)
    } else {
        rep("rmsd", slices)
    }
    value <- if (context$classification) {
        ifelse(state$total > 0, state$correct / state$total, NA_real_)
    } else if (state$metric_id %in% c(2L, 3L)) {
        if (is.finite(state$tss) && state$tss > 0) {
            1 - state$sse / state$tss
        } else {
            rep(NA_real_, slices)
        }
    } else {
        sqrt(state$sse / pmax(state$count, 1))
    }
    data.frame(
        ncomp_index = seq_len(slices),
        metric_name = metric_name,
        metric_value = value,
        stringsAsFactors = FALSE
    )
}

.via_pls_result <- function(context, state) {
    online <- .via_pls_metrics(context, state)
    decoded <- if (context$classification && !is.null(state$class_pred)) {
        .decode_cv_class_predictions(state$class_pred, context$Y,
            context$levels)
    }
    else if (!is.null(state$score_pred)) {
        .decode_cv_predictions(state$score_pred, context$original, FALSE, NULL)
    }
    else {
        list(pred = NULL, metrics = online)
    }
    q2 <- if (context$classification) {
        .cv_classification_q2_path(context$Y, state$score_pred, context$levels,
            fold = context$fold)
    }
    else if (!is.null(state$score_pred)) {
        .fastpls_fold_q2_path(context$original, state$score_pred, context$fold)
    }
    else {
        rep(NA, length(context$ncomp))
    }
    result <- list(Ypred = state$score_pred,
        Yscore = if (context$classification) state$score_pred else NULL,
        class_pred = state$class_pred, fold = context$fold,
        ncomp = context$ncomp,
        method = context$method, backend = context$backend,
        classification = context$classification,
        levels = context$levels, status = "ok", pred = decoded$pred,
        metrics = decoded$metrics %||%
            online, Q2Y = as.numeric(q2), RMSD = if (context$classification) {
            rep(NA, length(context$ncomp))
        } else {
            sqrt(state$sse / pmax(state$count, 1))
        })
    if (identical(context$backend, "cuda")) {
        result$residency <- list(
            fold_orchestration = "host",
            fold_model_fit = paste("resident", context$backend),
            fold_prediction = paste("resident", context$backend),
            fold_classifier = if (context$classification) {
                paste("resident", context$backend)
            } else {
                "not_requested"
            },
            metric_reduction = paste(
                "host aggregation of device-computed predictions and sums"
            ),
            fallback = "none"
        )
    } else if (identical(context$backend, "metal")) {
        result$residency <- list(
            fold_orchestration = "host",
            fold_model_fit = "fixed CPU/Metal operation split",
            fold_prediction = "cpu",
            fold_classifier = if (context$classification) {
                "cpu"
            } else {
                "not_requested"
            },
            metric_reduction = "host aggregation",
            fallback = "none"
        )
    }
    result
}

.pls_cv_via_pls <- function(Xdata, Ydata, constrain = NULL, ncomp = 2L,
    kfold = 10L,
    scaling = c("centering", "autoscaling", "none"), method = c("plssvd",
        "simpls",
        "opls", "kernelpls"),
    backend = c("cpu", "cuda", "metal"),
    svd.method = "rsvd", seed = 1L, xprod = NULL, north = 1L,
    kernel = c("linear",
        "rbf",
        "poly"), gamma = NULL, degree = 3L, coef0 = 1, classifier = c("argmax",
        "lda"), lda_ridge = 1e-08, return_scores = TRUE,
    store_predictions = TRUE,
    selection_metric = "auto", ...) {
    context <- .via_pls_context(Xdata, Ydata, constrain, ncomp, kfold, scaling,
        method, backend, if (missing(svd.method))
            NULL
        else svd.method, seed, xprod, kernel, classifier, list(...))
    config <- list(north = north, gamma = gamma, degree = degree,
        coef0 = coef0,
        lda_ridge = lda_ridge, return_scores = return_scores)
    state <- .via_pls_state(context, store_predictions, selection_metric)
    state <- .via_pls_run_folds(context, state, config)
    .via_pls_result(context, state)
}

#' Fast grouped PLS cross-validation for compiled backends
#'
#' These fixed-component helpers perform grouped k-fold cross-validation with
#' compiled fastPLS models only. They accept classification factors or numeric
#' regression responses and return fold predictions plus accuracy, Q2, or RMSD.
#'
#' @param Xdata Numeric predictor matrix.
#'  @param Ydata Factor response for classification, or numeric vector/matrix
#' for regression.
#'  @param constrain Optional grouping vector; samples with the same value stay
#' in the same fold.
#' @param ncomp Number of PLS components.
#' @param kfold Number of CV folds.
#' @param scaling Scaling mode.
#' @param svd.method CPU SVD backend for Cpp functions.
#' @param xprod Use the matrix-free xprod backend where available. The default
#'   `NULL` applies the same size thresholds used by [pls()]; `TRUE` forces the
#'   route and `FALSE` disables it.
#' @param ... Additional backend tuning arguments.
#' @return A list with `Ypred`, decoded `pred`, `metrics`, `fold`, and status.
#' @noRd
.plssvd_cv_cpp <- function(
    Xdata,
    Ydata,
    constrain = NULL,
    ncomp = 2L,
    kfold = 10L,
    scaling = c("centering", "autoscaling", "none"),
    svd.method = c("cpu_rsvd"),
    xprod = NULL,
    ...
) {
    .pls_cv_compiled(
        Xdata,
        Ydata,
        constrain,
        ncomp,
        kfold,
        scaling,
        "plssvd",
        "cpp",
        svd.method,
        xprod = xprod,
        ...
    )
}

#' @noRd
.simpls_cv_cpp <- function(
    Xdata,
    Ydata,
    constrain = NULL,
    ncomp = 2L,
    kfold = 10L,
    scaling = c("centering", "autoscaling", "none"),
    svd.method = c("cpu_rsvd"),
    xprod = NULL,
    ...
) {
    .pls_cv_compiled(
        Xdata,
        Ydata,
        constrain,
        ncomp,
        kfold,
        scaling,
        "simpls",
        "cpp",
        svd.method,
        xprod = xprod,
        ...
    )
}

.simpls_fast_cv_cpp <- function(
    Xdata,
    Ydata,
    constrain = NULL,
    ncomp = 2L,
    kfold = 10L,
    scaling = c("centering", "autoscaling", "none"),
    svd.method = c("cpu_rsvd"),
    xprod = NULL,
    ...
) {
    .simpls_cv_cpp(
        Xdata,
        Ydata,
        constrain,
        ncomp,
        kfold,
        scaling,
        svd.method,
        xprod = xprod,
        ...
    )
}

#' @noRd
.opls_cv_cpp <- function(
    Xdata,
    Ydata,
    constrain = NULL,
    ncomp = 2L,
    kfold = 10L,
    north = 1L,
    scaling = c("centering", "autoscaling", "none"),
    svd.method = c("cpu_rsvd"),
    xprod = NULL,
    ...
) {
    pred_ncomp <- pmax(1L, as.integer(ncomp) - as.integer(north))
    .pls_cv_compiled(
        Xdata,
        Ydata,
        constrain,
        pred_ncomp,
        kfold,
        scaling,
        "opls",
        "cpp",
        svd.method,
        xprod = xprod,
        north = north,
        ...
    )
}

#' @noRd
.kernelpls_cv_cpp <- function(
    Xdata,
    Ydata,
    constrain = NULL,
    ncomp = 2L,
    kfold = 10L,
    scaling = c("centering", "autoscaling", "none"),
    svd.method = c("cpu_rsvd"),
    xprod = NULL,
    ...
) {
    .pls_cv_compiled(
        Xdata,
        Ydata,
        constrain,
        ncomp,
        kfold,
        scaling,
        "kernelpls",
        "cpp",
        svd.method,
        xprod = xprod,
        ...
    )
}

#' @noRd
.plssvd_cv_cuda <- function(
    Xdata,
    Ydata,
    constrain = NULL,
    ncomp = 2L,
    kfold = 10L,
    scaling = c("centering", "autoscaling", "none"),
    xprod = NULL,
    ...
) {
    .pls_cv_compiled(
        Xdata,
        Ydata,
        constrain,
        ncomp,
        kfold,
        scaling,
        "plssvd",
        "cuda",
        xprod = xprod,
        ...
    )
}

#' @noRd
.simpls_cv_cuda <- function(
    Xdata,
    Ydata,
    constrain = NULL,
    ncomp = 2L,
    kfold = 10L,
    scaling = c("centering", "autoscaling", "none"),
    xprod = NULL,
    ...
) {
    .pls_cv_compiled(
        Xdata,
        Ydata,
        constrain,
        ncomp,
        kfold,
        scaling,
        "simpls",
        "cuda",
        xprod = xprod,
        ...
    )
}

.simpls_fast_cv_cuda <- function(
    Xdata,
    Ydata,
    constrain = NULL,
    ncomp = 2L,
    kfold = 10L,
    scaling = c("centering", "autoscaling", "none"),
    xprod = NULL,
    ...
) {
    .simpls_cv_cuda(
        Xdata,
        Ydata,
        constrain,
        ncomp,
        kfold,
        scaling,
        xprod = xprod,
        ...
    )
}

#' @noRd
.opls_cv_cuda <- function(
    Xdata,
    Ydata,
    constrain = NULL,
    ncomp = 2L,
    kfold = 10L,
    north = 1L,
    scaling = c("centering", "autoscaling", "none"),
    xprod = NULL,
    ...
) {
    pred_ncomp <- pmax(1L, as.integer(ncomp) - as.integer(north))
    .pls_cv_compiled(
        Xdata,
        Ydata,
        constrain,
        pred_ncomp,
        kfold,
        scaling,
        "opls",
        "cuda",
        xprod = xprod,
        north = north,
        ...
    )
}

#' @noRd
.kernelpls_cv_cuda <- function(
    Xdata,
    Ydata,
    constrain = NULL,
    ncomp = 2L,
    kfold = 10L,
    scaling = c("centering", "autoscaling", "none"),
    xprod = NULL,
    ...
) {
    .pls_cv_compiled(
        Xdata,
        Ydata,
        constrain,
        ncomp,
        kfold,
        scaling,
        "kernelpls",
        "cuda",
        xprod = xprod,
        ...
    )
}

#' @noRd
.plssvd_cv_metal <- function(
    Xdata,
    Ydata,
    constrain = NULL,
    ncomp = 2L,
    kfold = 10L,
    scaling = c("centering", "autoscaling", "none"),
    xprod = NULL,
    ...
) {
    .pls_cv_compiled(
        Xdata,
        Ydata,
        constrain,
        ncomp,
        kfold,
        scaling,
        "plssvd",
        "metal",
        xprod = xprod,
        ...
    )
}

#' @noRd
.simpls_cv_metal <- function(
    Xdata,
    Ydata,
    constrain = NULL,
    ncomp = 2L,
    kfold = 10L,
    scaling = c("centering", "autoscaling", "none"),
    xprod = NULL,
    ...
) {
    .pls_cv_compiled(
        Xdata,
        Ydata,
        constrain,
        ncomp,
        kfold,
        scaling,
        "simpls",
        "metal",
        xprod = xprod,
        ...
    )
}

.simpls_fast_cv_metal <- function(
    Xdata,
    Ydata,
    constrain = NULL,
    ncomp = 2L,
    kfold = 10L,
    scaling = c("centering", "autoscaling", "none"),
    xprod = NULL,
    ...
) {
    .simpls_cv_metal(
        Xdata,
        Ydata,
        constrain,
        ncomp,
        kfold,
        scaling,
        xprod = xprod,
        ...
    )
}

#' @noRd
.opls_cv_metal <- function(
    Xdata,
    Ydata,
    constrain = NULL,
    ncomp = 2L,
    kfold = 10L,
    north = 1L,
    scaling = c("centering", "autoscaling", "none"),
    xprod = NULL,
    ...
) {
    pred_ncomp <- pmax(1L, as.integer(ncomp) - as.integer(north))
    .pls_cv_compiled(
        Xdata,
        Ydata,
        constrain,
        pred_ncomp,
        kfold,
        scaling,
        "opls",
        "metal",
        xprod = xprod,
        north = north,
        ...
    )
}

#' @noRd
.kernelpls_cv_metal <- function(
    Xdata,
    Ydata,
    constrain = NULL,
    ncomp = 2L,
    kfold = 10L,
    scaling = c("centering", "autoscaling", "none"),
    xprod = NULL,
    ...
) {
    .pls_cv_compiled(
        Xdata,
        Ydata,
        constrain,
        ncomp,
        kfold,
        scaling,
        "kernelpls",
        "metal",
        xprod = xprod,
        ...
    )
}

.svd_methods_internal <- c(
    "exact",
    "cpu_rsvd",
    "cuda_rsvd",
    "metal_rsvd"
)
.svd_methods_public <- "rsvd"
.svd_methods_cpu <- "cpu_rsvd"

.svd_method_id <- function(method) {
    method <- .normalize_svd_method(method)
    method <- match.arg(method, .svd_methods_internal)
    switch(
        method,
        exact = 3L,
        cpu_rsvd = 4L,
        cuda_rsvd = 5L,
        metal_rsvd = 6L
    )
}

#' List available SVD backends
#'
#' Reports backend labels accepted by high-level APIs and whether each backend
#' is currently available.
#'
#' @return Data frame with columns `backend`, `method`, `svd.method`, and
#'   `enabled`.
#' @noRd
.svd_methods <- function() {
    combos <- data.frame(
        backend = c("cpu", "cuda", "metal"),
        method = rep("rsvd", 3L),
        svd.method = c("cpu_rsvd", "cuda_rsvd", "metal_rsvd"),
        enabled = c(TRUE, isTRUE(has_cuda()), isTRUE(has_metal())),
        stringsAsFactors = FALSE
    )
    combos
}


.truncated_rsvd_metal <- function(A, k, rsvd_oversample = 32L, rsvd_power = 5L,
    seed = 1L, left_only = FALSE) {
    if (!isTRUE(has_metal())) {
        .fastpls_require_backend_available("metal", "Metal rSVD")
    }
    A <- as.matrix(A); max_rank <- min(nrow(A), ncol(A))
    target <- min(max_rank, max(1L, as.integer(k)[1L]))
    sketch_rank <- min(max_rank, target + max(0L,
        as.integer(rsvd_oversample)[1L]))
    if (max_rank <= .metal_exact_max_rank() || sketch_rank >= max_rank) {
        exact <- svd(A, nu = target, nv = if (isTRUE(left_only))
            0L
        else target)
        return(list(U = exact$u[, seq_len(target), drop = FALSE],
            s = exact$d[seq_len(target)],
            Vt = if (isTRUE(left_only)) {
                NULL
            } else {
                t(exact$v[, seq_len(target), drop = FALSE])
            }))
    }
    .fastpls_set_seed(seed)
    omega <- matrix(rnorm(ncol(A) * sketch_rank), nrow = ncol(A),
        ncol = sketch_rank)
    Y <- metal_matrix_multiply_cpp(A, omega)
    n_power <- max(0L, as.integer(rsvd_power)[1L])
    if (n_power > 0L) {
        for (i in seq_len(n_power)) {
            Qy <- qr.Q(qr(Y))
            Z <- metal_crossprod_cpp(A, Qy)
            Qz <- qr.Q(qr(Z))
            Y <- metal_matrix_multiply_cpp(A, Qz)
        }
    }
    Q <- qr.Q(qr(Y)); B <- metal_crossprod_cpp(Q, A)
    small <- svd(B, nu = target, nv = if (isTRUE(left_only))
        0L
    else target)
    usable <- min(target, length(small$d), ncol(small$u))
    U <- Q %*% small$u[, seq_len(usable), drop = FALSE]
    Vt <- if (isTRUE(left_only)) {
        NULL
    }
    else {
        t(small$v[, seq_len(usable), drop = FALSE])
    }
    list(U = U, s = small$d[seq_len(usable)], Vt = Vt)
}

.metal_xprod_multiply <- function(X, Y, values) {
    .metal_crossprod(X, .metal_mm(Y, values))
}

.metal_xprod_transpose_multiply <- function(X, Y, values) {
    .metal_crossprod(Y, .metal_mm(X, values))
}

.truncated_rsvd_metal_xprod <- function(X, Y, k, rsvd_oversample = 32L,
    rsvd_power = 5L,
    seed = 1L, left_only = FALSE) {
    if (!isTRUE(has_metal())) {
        .fastpls_require_backend_available("metal", "Matrix-free Metal rSVD")
    }
    X <- as.matrix(X)
    Y <- as.matrix(Y)
    if (nrow(X) != nrow(Y)) {
        stop(
            "Metal matrix-free SVD requires X and Y with equal row counts.",
            call. = FALSE
        )
    }
    p <- ncol(X)
    q <- ncol(Y)
    max_rank <- min(p, q)
    target <- min(max_rank, max(1L, as.integer(k)[1L]))
    sketch_rank <- min(max_rank, target + max(0L,
        as.integer(rsvd_oversample)[1L]))
    .fastpls_set_seed(seed)
    omega <- matrix(rnorm(q * sketch_rank), nrow = q, ncol = sketch_rank)
    metal_xprod_rsvd_cpp(X, Y, omega, target,
        max(0L, as.integer(rsvd_power)[1L]), isTRUE(left_only))
}

.svd_dispatch_metal <- function(A, k, oversample, power, seed, left_only) {
    elapsed <- system.time({
        output <- .truncated_rsvd_metal(
            A = as.matrix(A),
            k = as.integer(k),
            rsvd_oversample = as.integer(oversample),
            rsvd_power = as.integer(power),
            seed = as.integer(seed),
            left_only = isTRUE(left_only)
        )
    })["elapsed"]
    list(
        U = output$U,
        s = as.vector(output$s),
        Vt = output$Vt,
        method = "metal_rsvd",
        elapsed = as.numeric(elapsed)
    )
}

.normalize_svd_debug_output <- function(output, method, elapsed) {
    list(
        U = output$u,
        s = as.vector(output$d),
        Vt = output$vt,
        method = method,
        elapsed = as.numeric(elapsed),
        case_audited = isTRUE(output$case_audited),
        case_certified = isTRUE(output$case_certified),
        deterministic_fallback = isTRUE(output$deterministic_fallback),
        audit_attempts = output$audit_attempts,
        effective_oversample = output$effective_oversample,
        effective_power = output$effective_power,
        effective_seed = output$effective_seed,
        audit_subspace_error = output$audit_subspace_error,
        audit_singular_value_error = output$audit_singular_value_error,
        audit_triplet_residual = output$audit_triplet_residual,
        audit_omitted_direction_ratio = output$audit_omitted_direction_ratio
    )
}

.svd_dispatch_compiled <- function(
    A,
    k,
    method,
    oversample,
    power,
    tolerance,
    seed,
    left_only
) {
    if (identical(method, "cpu_rsvd")) {
        elapsed <- system.time({
            output <- fastsvd_core_cpp(
                as.matrix(A), as.integer(k), as.integer(oversample),
                as.integer(power), as.integer(seed), isTRUE(left_only)
            )
        })["elapsed"]
        output$elapsed <- as.numeric(elapsed)
        output$method <- method
        output$precision <- "double"
        return(output)
    }
    method_id <- .svd_method_id(method)
    if (is.na(method_id)) {
        stop("Unknown method", call. = FALSE)
    }
    elapsed <- system.time({
        output <- truncated_svd_debug(
            A = as.matrix(A),
            k = as.integer(k),
            svd_method = as.integer(method_id),
            rsvd_oversample = as.integer(oversample),
            rsvd_power = as.integer(power),
            svds_tol = as.numeric(tolerance),
            seed = as.integer(seed),
            left_only = isTRUE(left_only)
        )
    })["elapsed"]
    .normalize_svd_debug_output(output, method, elapsed)
}

.svd_dispatch <- function(
    A,
    k,
    method = c("cpu_rsvd", "cuda_rsvd", "metal_rsvd"),
    rsvd_oversample = 32L,
    rsvd_power = 5L,
    svds_tol = 0,
    seed = 1L,
    left_only = FALSE
) {
    method <- .normalize_svd_method(method)
    method <- match.arg(method)
    if (identical(method, "cuda_rsvd") && !has_cuda()) {
        .fastpls_require_backend_available("cuda", "SVD dispatch")
    }
    if (identical(method, "metal_rsvd") && !has_metal()) {
        .fastpls_require_backend_available("metal", "SVD dispatch")
    }
    if (identical(method, "metal_rsvd")) {
        return(.svd_dispatch_metal(
            A,
            k,
            rsvd_oversample,
            rsvd_power,
            seed,
            left_only
        ))
    }
    .svd_dispatch_compiled(
        A,
        k,
        method,
        rsvd_oversample,
        rsvd_power,
        svds_tol,
        seed,
        left_only
    )
}

.fastsvd_float32_windows <- function(
    x,
    k,
    backend,
    svd.method,
    oversample,
    power,
    seed,
    left_only = FALSE
) {
    if (!identical(backend, "cpu") || !identical(svd.method, "cpu_rsvd")) {
        stop(
            "The portable Windows float32 implementation supports ",
            "backend = 'cpu' and method = 'rsvd' only. ",
            "No CPU fallback is performed.",
            call. = FALSE
        )
    }
    t_elapsed <- system.time({
        raw <- .float32_rsvd(
            .as_float32_matrix(x, "x"),
            k,
            oversample = oversample,
            power = power,
            seed = seed
        )
    })["elapsed"]
    list(
        U = raw$u,
        s = as.vector(raw$d),
        Vt = if (isTRUE(left_only)) NULL else t(raw$v),
        method = svd.method,
        elapsed = as.numeric(t_elapsed),
        precision = "float32",
        case_audited = isTRUE(raw$case_audited),
        case_certified = isTRUE(raw$case_certified),
        deterministic_fallback = isTRUE(raw$deterministic_fallback),
        audit_attempts = raw$audit_attempts,
        audit_triplet_residual = raw$audit_triplet_residual,
        audit_omitted_direction_ratio = raw$audit_omitted_direction_ratio
    )
}

.fastsvd_float32 <- function(x, k, backend, svd.method, oversample, power,
    seed,
    left_only = FALSE) {
    if (identical(.Platform$OS.type, "windows")) {
        return(.fastsvd_float32_windows(x, k, backend, svd.method, oversample,
            power, seed, left_only))
    }
    if (identical(backend, "cpu")) {
        t_elapsed <- system.time({
            raw <- fastsvd_float32_core_cpp(
                .as_float32_matrix(x, "x"), as.integer(k),
                as.integer(oversample), as.integer(power), as.integer(seed),
                isTRUE(left_only)
            )
        })["elapsed"]
        return(list(
            U = .float32_from_bits(raw$U),
            s = as.vector(raw$s),
            Vt = if (is.null(raw$Vt)) NULL else .float32_from_bits(raw$Vt),
            method = svd.method,
            elapsed = as.numeric(t_elapsed),
            precision = "float32",
            case_audited = isTRUE(raw$case_audited),
            case_certified = isTRUE(raw$case_certified),
            deterministic_fallback = isTRUE(raw$deterministic_fallback),
            audit_attempts = raw$audit_attempts,
            effective_oversample = raw$effective_oversample,
            effective_power = raw$effective_power,
            effective_seed = raw$effective_seed,
            audit_subspace_error = raw$audit_subspace_error,
            audit_singular_value_error = raw$audit_singular_value_error,
            audit_triplet_residual = raw$audit_triplet_residual,
            audit_omitted_direction_ratio = raw$audit_omitted_direction_ratio
        ))
    }
    backend_id <- switch(backend, cpu = 0L, cuda = 1L, metal = 2L)
    svd_id <- .float32_svd_id(svd.method)
    t_elapsed <- system.time({
        raw <- fastsvd_float32_cpp(.as_float32_matrix(x, "x"), as.integer(k),
            as.integer(backend_id),
            as.integer(svd_id), as.integer(oversample), as.integer(power),
            as.integer(seed),
            isTRUE(left_only))
    })["elapsed"]
    list(U = .float32_from_bits(raw$u), s = as.vector(raw$d),
        Vt = if (is.null(raw$v)) NULL else t(.float32_from_bits(raw$v)),
        method = svd.method, elapsed = as.numeric(t_elapsed),
        precision = "float32",
        case_audited = isTRUE(raw$case_audited),
        case_certified = isTRUE(raw$case_certified),
        deterministic_fallback = isTRUE(raw$deterministic_fallback),
        audit_attempts = raw$audit_attempts,
        effective_oversample = raw$effective_oversample,
        effective_power = raw$effective_power,
        effective_seed = raw$effective_seed,
        audit_triplet_residual = raw$audit_triplet_residual,
        audit_omitted_direction_ratio = raw$audit_omitted_direction_ratio)
}

.fastsvd_basic_diagnostics <- function(decomposition) {
    u <- decomposition$U
    vt <- decomposition$Vt
    values <- as.numeric(decomposition$s)
    returned_rank <- length(values)
    finite <- returned_rank > 0L &&
        all(is.finite(values)) &&
        !is.null(u) &&
        all(is.finite(u)) &&
        !is.null(vt) &&
        length(vt) > 0L &&
        all(is.finite(vt))

    list(
    status = if (finite) "basic_checks_passed" else "failed_nonfinite_or_empty",
        returned_rank = returned_rank,
        finite = finite,
        residual_failure_threshold = 0.1,
        residual_warning_threshold = 0.01,
        orthogonality_warning_threshold = 1e-5
    )
}

.fastsvd_triplet_diagnostics <- function(x, decomposition, indices) {
    u <- decomposition$U
    v <- t(decomposition$Vt)
    values <- as.numeric(decomposition$s)
    ud <- u[, indices, drop = FALSE]
    vd <- v[, indices, drop = FALSE]
    sd <- values[indices]
    av <- x %*% vd
    atu <- crossprod(x, ud)
    scaled_u <- sweep(ud, 2L, sd, "*")
    scaled_v <- sweep(vd, 2L, sd, "*")
    left_denom <- pmax(sqrt(colSums(av * av)), abs(sd), .Machine$double.eps)
    right_denom <- pmax(sqrt(colSums(atu * atu)), abs(sd), .Machine$double.eps)
    list(
        max_residual = max(
            sqrt(colSums((av - scaled_u)^2)) / left_denom,
            sqrt(colSums((atu - scaled_v)^2)) / right_denom
        ),
        u_orthogonality = norm(crossprod(ud) - diag(ncol(ud)), type = "F"),
        v_orthogonality = norm(crossprod(vd) - diag(ncol(vd)), type = "F")
    )
}

.fastsvd_diagnostic_status <- function(out, triplet, randomized) {
    if (triplet$max_residual > out$residual_failure_threshold) {
        return("failed_large_triplet_residual")
    }
    warning_limit <- out$orthogonality_warning_threshold
    if (
        triplet$max_residual > out$residual_warning_threshold ||
        max(triplet$u_orthogonality, triplet$v_orthogonality) > warning_limit
    ) {
        return("warning_approximation_quality")
    }
    if (randomized) {
        "rsvd_triplet_checks_passed"
    } else {
        "deterministic_triplet_checks_passed"
    }
}

.fastsvd_numerical_diagnostics <- function(x, decomposition, randomized) {
    out <- .fastsvd_basic_diagnostics(decomposition)
    if (!out$finite) {
        return(out)
    }

    if (.is_float32(x)) {
        out$status <- "basic_checks_passed_residual_not_computed_float32"
        out$note <- paste(
        "The float32 path checks rank and finiteness without converting the",
            "input to double solely for diagnostics."
        )
        return(out)
    }

    x <- as.matrix(x)
    returned_rank <- out$returned_rank
    diagnostic_index <- unique(c(1L, ceiling(returned_rank / 2), returned_rank))
    diagnostic_index <- diagnostic_index[
        diagnostic_index <= ncol(decomposition$U) &
            diagnostic_index <= nrow(decomposition$Vt)
    ]
    if (!length(diagnostic_index)) {
        out$status <- "failed_missing_singular_vectors"
        return(out)
    }

    triplet <- .fastsvd_triplet_diagnostics(x, decomposition, diagnostic_index)
    out$checked_components <- diagnostic_index
    out$max_relative_triplet_residual <- triplet$max_residual
    out$left_orthogonality_error <- triplet$u_orthogonality
    out$right_orthogonality_error <- triplet$v_orthogonality
    out$status <- .fastsvd_diagnostic_status(out, triplet, randomized)
    out
}

.fastsvd_resolve_solver <- function(float32, backend, method) {
    backend <- .normalize_public_backend(backend)
    if (!identical(method, "rsvd")) {
        stop("fastsvd() supports method = 'rsvd' only.", call. = FALSE)
    }
    solver <- switch(backend, cpu = "cpu_rsvd", cuda = "cuda_rsvd",
        metal = "metal_rsvd")
    list(backend = backend, method = method, solver = solver)
}

.fastsvd_validate_solver <- function(float32, backend, method, solver) {
    .fastpls_require_backend_available(backend, "fastsvd()")
    if (solver %in% c("cuda_rsvd", "metal_rsvd")) {
        stop(
            "Standalone accelerator fastsvd() is unavailable because every ",
            "matrix shape does not yet have a fully device-native reduced ",
            "QR/SVD stage. Use backend = 'cpu'. GPU rSVD remains available ",
            "inside the operation-split pls() Metal route; no silent ",
            "fallback is performed.",
            call. = FALSE
        )
    }
}

.fastsvd_randomized_control <- function(
    method,
    backend,
    oversample,
    power,
    supplied
) {
    if (method == "rsvd") {
        control <- .svd_control_defaults()
        control$rsvd_oversample <- as.integer(oversample)[1L]
        control$rsvd_power <- as.integer(power)[1L]
        control$supplied <- supplied
        control <- .apply_backend_rsvd_controls(control, backend, "fastsvd()")
        oversample <- control$rsvd_oversample
        power <- control$rsvd_power
    }
    list(oversample = oversample, power = power)
}

.fastsvd_rank_configuration <- function(x, nu, nv, ncomp, float32) {
    rank_limit <- min(dim(if (float32) x else as.matrix(x)))
    if (is.null(nu)) {
        nu <- rank_limit
    }
    if (is.null(nv)) {
        nv <- rank_limit
    }
    requested <- if (is.null(ncomp)) {
        max(as.integer(c(nu, nv)), 1L)
    } else {
        as.integer(ncomp)[1L]
    }
    list(
        nu = nu,
        nv = nv,
        k = max(1L, min(requested, rank_limit))
    )
}

.fastsvd_configuration <- function(
    x,
    nu,
    nv,
    ncomp,
    backend,
    method,
    oversample,
    power,
    supplied
) {
    backend <- .normalize_public_backend(backend)
    .fastpls_require_backend_available(backend, "fastsvd()")
    float32 <- .is_float32(x)
    rank <- .fastsvd_rank_configuration(x, nu, nv, ncomp, float32)
    resolved <- .fastsvd_resolve_solver(float32, backend, method)
    .fastsvd_validate_solver(
        float32,
        resolved$backend,
        resolved$method,
        resolved$solver
    )
    control <- .fastsvd_randomized_control(
        resolved$method,
        resolved$backend,
        oversample,
        power,
        supplied
    )
    list(
        float32 = float32,
        backend = resolved$backend,
        method = resolved$method,
        solver = resolved$solver,
        oversample = control$oversample,
        power = control$power,
        nu = rank$nu,
        nv = rank$nv,
        k = rank$k
    )
}

.fastsvd_compute <- function(x, config, seed) {
    if (config$method == "rsvd") {
        try(rsvd_audit_reset_debug(), silent = TRUE)
    }
    if (config$float32) {
        return(.fastsvd_float32(
            x,
            config$k,
            config$backend,
            config$solver,
            config$oversample,
            config$power,
            seed,
            FALSE
        ))
    }
    .svd_dispatch(
            as.matrix(x),
            config$k,
            config$solver,
            config$oversample,
            config$power,
            0,
            seed,
            FALSE
        )
}

.fastsvd_audit_diagnostics <- function(diagnostics, output, config, seed) {
    diagnostics$rsvd_case_audit <- list(
        performed = isTRUE(output$case_audited),
        certified = isTRUE(output$case_certified),
        deterministic_fallback = isTRUE(output$deterministic_fallback),
        attempts = output$audit_attempts %||% NA_integer_,
        effective_oversample = output$effective_oversample %||%
            config$oversample,
        effective_power = output$effective_power %||% config$power,
        effective_seed = output$effective_seed %||% seed,
        subspace_error = output$audit_subspace_error %||% NA_real_,
        singular_value_error = output$audit_singular_value_error %||% NA_real_,
        triplet_residual = output$audit_triplet_residual %||% NA_real_,
        omitted_direction_ratio = output$audit_omitted_direction_ratio %||%
            NA_real_
    )
    diagnostics$rsvd_qualification <- .rsvd_configuration_qualification(
        config$backend,
        config$oversample,
        config$power
    )
    if (isTRUE(output$case_certified)) {
        diagnostics$status <- if (isTRUE(output$deterministic_fallback)) {
            "rsvd_case_audit_passed_with_deterministic_recovery"
        } else {
            "rsvd_case_audit_passed"
        }
    }
    diagnostics
}

.fastsvd_result <- function(output, u, v, config, diagnostics) {
    result <- list(
        d = output$s,
        u = u,
        v = v,
        method = config$method,
        backend = config$backend,
        svd.method = config$solver,
        elapsed = output$elapsed,
        ncomp = config$k,
        precision = output$precision %||% "double",
        diagnostics = diagnostics
    )
    if (startsWith(diagnostics$status, "failed_")) {
        warning(
            "fastsvd numerical diagnostics: ",
            diagnostics$status,
            call. = FALSE
        )
    }
    result
}

#' Singular value decomposition through fastPLS backends
#'
#' Computes a truncated singular value decomposition of a dense numeric matrix
#' with the CPU backend. The result contains singular
#' values, requested singular vectors, decomposition rank, elapsed time, and
#' backend metadata.
#'
#' @param x Numeric matrix to decompose, with observations or rows in rows and
#'   variables or columns in columns. Sparse matrices should be converted by the
#'   caller; `fastsvd()` currently works on a dense numeric matrix.
#' @param nu Number of left singular vectors to return. If `NULL`, the function
#'   uses the largest feasible rank implied by the matrix dimensions. When
#'   `ncomp` is supplied, `ncomp` controls the decomposition rank and `nu`
#'   controls only how many left vectors are kept in the returned object.
#' @param nv Number of right singular vectors to return. If `NULL`, the
#'   function uses the largest feasible rank implied by the matrix dimensions.
#'   When `ncomp` is supplied, `ncomp` controls the decomposition rank and `nv`
#'   controls only how many right vectors are kept in the returned object.
#' @param ncomp Optional truncated rank. When supplied, it overrides the rank
#'   implied by `nu` and `nv`; the final rank is always capped at
#'   `min(nrow(x), ncol(x))`.
#' @param backend Compute backend. \code{cpu} runs on the host CPU. Standalone
#'   CUDA and Metal rSVD are unavailable because their reduced QR/SVD stages
#'   are not fully device-native for every matrix shape; both accelerators
#'   remain available for supported resident PLS fits. When omitted, the
#'   function uses `options(backend = ...)`, then `FASTPLS_BACKEND`, then CPU.
#'   For CPU execution, `options(cores = n)` requests `n` BLAS/OpenMP threads.
#'   An unavailable or unsupported accelerator selection raises an error; no
#'   CPU fallback is
#'   performed.
#' @param method SVD algorithm. Only \code{rsvd}, the native randomized SVD,
#'   is supported on the selected backend.
#' @param oversample Non-negative oversampling dimension used by
#'   randomized SVD. The sketch dimension is approximately
#'   `ncomp + oversample`, capped by the matrix rank. Larger values can improve
#'   approximation accuracy at the cost of extra time and memory. The default
#'   starting value is 32. Panel agreement is not a guarantee for a
#'   new matrix; CPU float64 fits additionally apply the case-specific audit
#'   described in Details.
#' @param power Number of randomized-SVD power iterations. The default of five
#'   is used on CPU. Together with backend-specific
#'   oversampling, these controls met the current numerical validation panel.
#'   Larger values can improve
#'   accuracy when singular values decay slowly, but each iteration adds matrix
#'   multiplications. Panel agreement alone is not general-use certification.
#' @param seed Random seed used by randomized backends to generate the Gaussian
#'   sketch. It affects \code{rsvd} results and is ignored by deterministic
#'   backends.
#' @return A list compatible with `base::svd()` containing `d`, `u`, and `v`,
#'   plus backend metadata and numerical `diagnostics`. Diagnostics include
#'   rank and finiteness checks and, for double-precision input, normalized
#'   singular-triplet residuals for the first, middle, and last returned
#'   components. A residual above 0.01 produces a warning status and a residual
#'   above 0.1 is classified as a numerical failure.
#' @details CPU rSVD retries unsuccessful sketches with more oversampling and
#'   power iterations. If the ordinary attempts fail, native operator rSVD
#'   makes up to four further attempts, subject to a conservative 256 MiB
#'   estimate of its major numerical buffers. This estimate excludes inputs,
#'   existing model storage, runtime overhead and allocator behavior. Each
#'   recovery result is rechecked; failed or over-budget recovery returns an
#'   error rather than an unchecked result.
#'   Small CPU float64 inputs use a separately recorded dense-SVD route.
#'   A full-width sketch can also use a dense SVD as part of the native
#'   algorithm. These decomposition checks do not
#'   establish agreement of a complete sequential PLS fit.
#' @examples
#' set.seed(1)
#' x <- matrix(rnorm(12 * 5), 12, 5)
#' s <- fastsvd(x, ncomp = 2, backend = "cpu", method = "rsvd", seed = 1)
#' s$d
#' @export
fastsvd <- function(x, nu = NULL, nv = NULL, ncomp = NULL, backend = NULL,
    method = "rsvd", oversample = 32L, power = 5L, seed = 1L) {
    supplied <- c(if (!missing(oversample)) "rsvd_oversample",
        if (!missing(power)) "rsvd_power")
    config <- .fastsvd_configuration(x, nu, nv, ncomp, backend, method,
        oversample,
        power, supplied)
    out <- .fastsvd_compute(x, config, seed)
    u <- out$U
    v <- if (is.null(out$Vt) || length(out$Vt) == 0L)
        NULL
    else t(out$Vt)
    if (!is.null(u) && ncol(u) > config$nu) {
        u <- u[, seq_len(config$nu), drop = FALSE]
    }
    if (!is.null(v) && ncol(v) > config$nv) {
        v <- v[, seq_len(config$nv), drop = FALSE]
    }
    diagnostics <- .fastsvd_numerical_diagnostics(x, out,
        randomized = config$method ==
            "rsvd")
    if (config$method == "rsvd") {
        diagnostics <- .fastsvd_audit_diagnostics(diagnostics, out, config,
            seed)
    }
    .fastsvd_result(out, u, v, config, diagnostics)
}

.fastpls_ellipse <- function(
    scores,
    conf = 0.95,
    type = c("confidence", "hotelling"),
    npoints = 100L
) {
    type <- match.arg(type)
    scores <- as.matrix(scores)
    scores <- scores[stats::complete.cases(scores), , drop = FALSE]
    if (nrow(scores) < 3L || ncol(scores) < 2L) {
        return(NULL)
    }
    center <- colMeans(scores)
    cov2 <- stats::cov(scores)
    if (any(!is.finite(cov2)) || qr(cov2)$rank < 2L) {
        return(NULL)
    }
    radius <- if (identical(type, "hotelling")) {
        sqrt(
            2 *
                (nrow(scores) - 1) /
                (nrow(scores) - 2) *
                stats::qf(conf, 2, nrow(scores) - 2)
        )
    } else {
        sqrt(stats::qchisq(conf, df = 2))
    }
    theta <- seq(0, 2 * pi, length.out = npoints)
    circle <- cbind(cos(theta), sin(theta))
    eig <- eigen(cov2, symmetric = TRUE)
    transform <- eig$vectors %*% diag(sqrt(pmax(eig$values, 0)), 2)
    sweep(radius * circle %*% t(transform), 2L, center, "+")
}

.fastpls_plot_palette <- function(n) {
    n <- as.integer(n)
    base <- c(
        "#0073C2FF",
        "#EFC000FF",
        "#CD534CFF",
        "#009E73FF",
        "#868686FF",
        "#56B4E9FF",
        "#D55E00FF",
        "#CC79A7FF",
        "#003C67FF",
        "#8F7700FF",
        "#A73030FF",
        "#005F45FF"
    )
    if (n <= length(base)) {
        return(base[seq_len(n)])
    }
    grDevices::hcl.colors(n, "Dark 3")
}

.fastpls_plot_call <- function(x, y, args) {
    do.call(graphics::plot, c(list(x = x, y = y), args))
}

.fastpls_plot_args <- function(xlab, ylab, main, dots) {
    if (is.null(dots$xlab)) {
        dots$xlab <- xlab
    }
    if (is.null(dots$ylab)) {
        dots$ylab <- ylab
    }
    if (is.null(dots$main)) {
        dots$main <- main
    }
    dots
}

.fastpls_score_axis_label <- function(scores, component, label) {
    if (!is.null(label) && !is.na(label)) {
        return(label)
    }
    candidate <- colnames(scores)[component]
    if (!is.null(candidate) && !is.na(candidate)) {
        return(candidate)
    }
    paste0("Component ", component)
}

.fastpls_add_score_ellipses <- function(xy, groups, palette, conf, type) {
    for (level in levels(groups)) {
        index <- which(groups == level)
        ellipse <- .fastpls_ellipse(
            xy[index, , drop = FALSE],
            conf = conf,
            type = type
        )
        if (!is.null(ellipse)) {
            graphics::lines(
                ellipse[, 1L],
                ellipse[, 2L],
                col = palette[match(level, levels(groups))],
                lwd = 2
            )
        }
    }
}

.fastpls_plot_scores <- function(scores, comps = c(1L, 2L), groups = NULL,
    ellipse = FALSE,
    ellipse.type = c("confidence", "hotelling"), conf = 0.95, main = NULL,
    xlab = NULL,
    ylab = NULL, ...) {
    scores <- as.matrix(scores)
    comps <- as.integer(comps)
    if (length(comps) != 2L || any(comps < 1L) || max(comps) > ncol(scores)) {
        stop("comps must contain two valid component indices.", call. = FALSE)
    }
    xy <- scores[, comps, drop = FALSE]
    xlab <- .fastpls_score_axis_label(scores, comps[1L], xlab)
    ylab <- .fastpls_score_axis_label(scores, comps[2L], ylab)
    dots <- list(...)
    if (is.null(groups)) {
        dots$pch <- dots$pch %||% 21
        dots$col <- dots$col %||% "black"
        dots$bg <- dots$bg %||% "#0073C2FF"
        .fastpls_plot_call(xy[, 1L], xy[, 2L], .fastpls_plot_args(xlab, ylab,
            main,
            dots))
        if (isTRUE(ellipse)) {
            el <- .fastpls_ellipse(xy, conf = conf, type = ellipse.type)
            if (!is.null(el)) {
                graphics::lines(el[, 1L], el[, 2L], col = "firebrick", lwd = 2)
            }
        }
        return(invisible(xy))
    }
    groups <- as.factor(groups); pal <- .fastpls_plot_palette(nlevels(groups))
    bg <- pal[as.integer(groups)]
    dots$pch <- dots$pch %||% 21
    dots$col <- dots$col %||% "black"
    dots$bg <- dots$bg %||% bg
    .fastpls_plot_call(xy[, 1L], xy[, 2L], .fastpls_plot_args(xlab, ylab, main,
        dots))
    graphics::legend("topright", legend = levels(groups), pt.bg = pal,
        col = "black",
        pch = dots$pch, bty = "n")
    if (isTRUE(ellipse)) {
        .fastpls_add_score_ellipses(xy, groups, pal, conf, ellipse.type)
    }
    invisible(xy)
}

.fastpls_score_matrix <- function(x, slot) {
    scores <- x[[slot]]
    if (!is.null(scores) && length(scores) > 0L && all(dim(scores) > 0L)) {
        scores <- as.matrix(scores)
        colnames(scores) <- paste0("LV", seq_len(ncol(scores)))
        return(scores)
    }
    NULL
}

.fastpls_model_variance_explained <- function(x) {
    vx <- x$variance_explained
    if (!is.null(vx) && length(vx) > 0L) {
        return(as.numeric(vx))
    }
    vx <- x$x_variance_explained
    if (!is.null(vx) && length(vx) > 0L) {
        return(as.numeric(vx))
    }
    if (!is.null(x$inner_model)) {
        return(.fastpls_model_variance_explained(x$inner_model))
    }
    NULL
}

.fastpls_model_scores <- function(x, score.set = c("auto", "train", "test")) {
    score.set <- match.arg(score.set)
    if (identical(score.set, "train")) {
        scores <- .fastpls_score_matrix(x, "Ttrain")
        if (!is.null(scores)) {
            return(scores)
        }
        if (!is.null(x$inner_model)) {
            return(.fastpls_model_scores(x$inner_model, score.set = "train"))
        }
        return(NULL)
    }
    if (identical(score.set, "test")) {
        scores <- .fastpls_score_matrix(x, "Ttest")
        if (!is.null(scores)) {
            return(scores)
        }
        if (!is.null(x$inner_model)) {
            return(.fastpls_model_scores(x$inner_model, score.set = "test"))
        }
        return(NULL)
    }
if (!is.null(x$Ttrain) && length(x$Ttrain) > 0L && all(dim(x$Ttrain) > 0L)) {
        scores <- as.matrix(x$Ttrain)
        colnames(scores) <- paste0("LV", seq_len(ncol(scores)))
        return(scores)
    }
    if (!is.null(x$inner_model)) {
        scores <- .fastpls_model_scores(x$inner_model, score.set = "auto")
        if (!is.null(scores)) {
            return(scores)
        }
    }
    if (!is.null(x$Ttest) && length(x$Ttest) > 0L && all(dim(x$Ttest) > 0L)) {
        scores <- as.matrix(x$Ttest)
        colnames(scores) <- paste0("LV", seq_len(ncol(scores)))
        return(scores)
    }
    NULL
}

#' Plot PLS latent scores
#'
#' Draws a two-component score plot for a fitted `fastPLS` object. Optional
#' ellipses are computed either as data confidence ellipses or Hotelling T2
#' score ellipses. Axis labels include predictor-space variance explained when
#' it was requested during model fitting.
#'
#' @param x A fitted `fastPLS` object.
#' @param comps Two component indices.
#' @param groups Optional grouping vector used for point fills and grouped
#'   ellipses.
#' @param score.set Plot `train` scores, `test` scores, or use `auto` to select
#'   stored training scores before test scores.
#' @param ellipse Draw confidence ellipses when `TRUE`.
#' @param ellipse.type Use `confidence` or `hotelling` ellipses.
#' @param conf Confidence level.
#' @param ... Additional arguments passed to `plot()`.
#' @return Invisibly returns the plotted score matrix.
#' @examples
#' X <- as.matrix(iris[, seq_len(4)])
#' fit <- pls(X, iris$Species,
#'     ncomp = 2, fit = TRUE,
#'     return_variance = TRUE, seed = 1
#' )
#' plot(fit, groups = iris$Species, ellipse = TRUE)
#' @export
plot.fastPLS <- function(x, comps = c(1L, 2L), groups = NULL,
    score.set = c("auto",
        "train", "test"), ellipse = FALSE, ellipse.type = c("confidence",
        "hotelling"),
    conf = 0.95, ...) {
    x <- .fastpls_restore_internal_output_fields(x)
    score.set <- match.arg(score.set)
    scores <- .fastpls_model_scores(x, score.set = score.set)
    if (is.null(scores)) {
        stop(
            "The requested PLS scores are not stored. Refit with fit=TRUE ",
            "for training scores or proj=TRUE for test scores.",
            call. = FALSE
        )
    }
    dots <- list(...)
    main <- if (is.null(dots$main))
        "fastPLS scores"
    else dots$main
    var_exp <- .fastpls_model_variance_explained(x)
    xlab <- dots$xlab
    ylab <- dots$ylab
    if (is.null(xlab) && !is.null(var_exp) && length(var_exp) >= comps[1L] &&
        is.finite(var_exp[comps[1L]])) {
        xlab <- sprintf("LV%d (%.1f%%)", comps[1L], 100 * var_exp[comps[1L]])
    }
    if (is.null(ylab) && !is.null(var_exp) && length(var_exp) >= comps[2L] &&
        is.finite(var_exp[comps[2L]])) {
        ylab <- sprintf("LV%d (%.1f%%)", comps[2L], 100 * var_exp[comps[2L]])
    }
    dots$main <- NULL
    dots$xlab <- NULL
    dots$ylab <- NULL
    do.call(.fastpls_plot_scores, c(list(scores = scores, comps = comps,
        groups = groups,
        ellipse = ellipse, ellipse.type = match.arg(ellipse.type), conf = conf,
        main = main, xlab = xlab, ylab = ylab), dots))
}

.permutation_plot_data <- function(x, ncomp) {
    perm <- if (is.data.frame(x)) x else x$permutation
    if (is.null(perm) || !is.data.frame(perm) || !nrow(perm)) {
        stop(
            "No permutation table found. Refit pls() with perm.test = TRUE.",
            call. = FALSE
        )
    }
    required <- c("type", "ncomp", "metric", "cor", "value")
    missing_cols <- setdiff(required, names(perm))
    if (length(missing_cols)) {
        stop(
            sprintf(
                "Permutation table is missing required columns: %s",
                paste(missing_cols, collapse = ", ")
            ),
            call. = FALSE
        )
    }
    if (is.null(ncomp)) {
        ncomp <- max(perm$ncomp, na.rm = TRUE)
    }
    ncomp <- as.integer(ncomp)[1L]
    keep <- perm$ncomp == ncomp & perm$metric %in% c("R2", "Q2")
    dat <- perm[
        keep & is.finite(perm$cor) & is.finite(perm$value),
        ,
        drop = FALSE
    ]
    if (!nrow(dat)) {
        stop(
            "No finite permutation values available for ncomp = ",
            ncomp,
            ".",
            call. = FALSE
        )
    }
    list(data = dat, ncomp = ncomp)
}

.permutation_plot_metric <- function(dat, metric, col, pch) {
    simulated <- dat$metric == metric & dat$type == "permutation"
    observed <- dat$metric == metric & dat$type == "observed"
    d <- dat[simulated, , drop = FALSE]
    obs <- dat[observed, , drop = FALSE]
    if (nrow(d)) {
        graphics::points(d$cor, d$value, col = col, pch = pch)
    }
    if (!nrow(obs)) {
        return(invisible(NULL))
    }
    graphics::points(obs$cor, obs$value, col = col, pch = pch, cex = 1.3)
    if (nrow(d)) {
        graphics::segments(
            mean(d$cor, na.rm = TRUE),
            mean(d$value, na.rm = TRUE),
            obs$cor[[1L]],
            obs$value[[1L]],
            col = col,
            lty = 2
        )
    }
    invisible(NULL)
}

#' Plot PLS permutation-test R2 and Q2 values
#'
#' Draws the permutation-test diagnostic plot produced by `pls(...,
#' perm.test = TRUE)`. The x-axis is the correlation between the original and
#' permuted response structure; the y-axis is the observed or permuted R2/Q2
#' value. R2 is shown in blue and Q2 in red.
#'
#' @param x A `fastPLS` model fitted with `perm.test = TRUE`, or a permutation
#'   data frame stored in `model$permutation`.
#' @param ncomp Component count to plot. Defaults to the largest component
#'   stored in the permutation table.
#' @param main,xlab,ylab Plot title and axis labels.
#' @param col,pch Colors and point symbols for R2 and Q2.
#' @param legend_position Legend position passed to [legend()].
#' @param ... Additional graphical parameters passed to [plot()].
#' @return Invisibly returns the plotted permutation data.
#' @examples
#' set.seed(1)
#' X <- as.matrix(iris[, seq_len(4)])
#' y <- iris$Sepal.Length
#' idx <- sample(seq_len(nrow(X)), 30)
#' fit <- pls(X[idx, ], y[idx], X[idx, ], y[idx],
#'     ncomp = 2, perm.test = TRUE, times = 5
#' )
#' plot.permutation(fit)
#' @export
plot.permutation <- function(
    x,
    ncomp = NULL,
    main = NULL,
    xlab = "Cor",
    ylab = "Value",
    col = c(R2 = "#3155B7", Q2 = "#E5332A"),
    pch = c(R2 = 16, Q2 = 15),
    legend_position = "bottomright",
    ...
) {
    plot_data <- .permutation_plot_data(x, ncomp)
    dat <- plot_data$data
    if (is.null(main)) {
        main <- paste("Permutation test, ncomp =", plot_data$ncomp)
    }
    xlim <- range(c(0, 1, dat$cor), finite = TRUE)
    ylim <- range(dat$value, finite = TRUE)
    pad <- diff(ylim) * 0.08
    if (!is.finite(pad) || pad == 0) {
        pad <- 0.1
    }
    ylim <- ylim + c(-pad, pad)
    graphics::plot(
        dat$cor,
        dat$value,
        type = "n",
        xlim = xlim,
        ylim = ylim,
        xlab = xlab,
        ylab = ylab,
        main = main,
        ...
    )
    for (metric in c("R2", "Q2")) {
        .permutation_plot_metric(dat, metric, col[[metric]], pch[[metric]])
    }
    graphics::legend(
        legend_position,
        legend = c("R2", "Q2"),
        col = col[c("R2", "Q2")],
        pch = pch[c("R2", "Q2")],
        bty = "o"
    )
    invisible(dat)
}

.metal_mm <- function(A, B) {
    if (!isTRUE(has_metal())) {
        .fastpls_require_backend_available(
            "metal",
            "Metal matrix multiplication"
        )
    }
    A <- as.matrix(A)
    B <- as.matrix(B)
    metal_matrix_multiply_cpp(A, B)
}

.metal_crossprod <- function(A, B) {
    if (!isTRUE(has_metal())) {
        .fastpls_require_backend_available("metal", "Metal cross-product")
    }
    A <- as.matrix(A)
    B <- as.matrix(B)
    metal_crossprod_cpp(A, B)
}

.metal_outer <- function(a, b) {
    tcrossprod(as.numeric(a), as.numeric(b))
}

.metal_min_flops <- function() {
    val <- .fastpls_quiet(
        as.numeric(Sys.getenv("FASTPLS_METAL_MIN_FLOPS", "200000000"))
    )
    if (!is.finite(val) || val < 0) 2e8 else val
}

.metal_exact_max_rank <- function() {
    val <- .fastpls_quiet(
        as.integer(Sys.getenv("FASTPLS_METAL_EXACT_MAX_RANK", "256"))
    )
    if (!is.finite(val) || is.na(val) || val < 0L) 256L else val
}

.metal_should_use_mm <- function(m, k, n) {
    m <- as.numeric(m)
    k <- as.numeric(k)
    n <- as.numeric(n)
    if (!is.finite(m) || !is.finite(k) || !is.finite(n)) {
        return(FALSE)
    }
    if (m <= 0 || k <= 0 || n <= 0) {
        return(FALSE)
    }
    # Matrix-vector and very thin products spend more time copying/dispatching
    # than computing unless the matrix is very large. BLAS is safer there.
    if (min(m, n) <= 1 && (m * k * n) < (.metal_min_flops() * 4)) {
        return(FALSE)
    }
    (2 * m * k * n) >= .metal_min_flops()
}

.metal_experimental_iterative_enabled <- function() {
    tolower(Sys.getenv("FASTPLS_METAL_EXPERIMENTAL_ITERATIVE", "false")) %in%
        c("1", "true", "yes", "y")
}

.metal_resident_simpls_enabled <- function() {
    !tolower(Sys.getenv("FASTPLS_METAL_RESIDENT_SIMPLS", "true")) %in%
        c("0", "false", "no", "n")
}

.metal_pls_preprocess <- function(Xtrain, Ytrain, scaling) {
    n <- nrow(Xtrain)
    p <- ncol(Xtrain)
    m <- ncol(Ytrain)
    mean_x <- matrix(0, 1L, p)
    if (scaling < 3L) {
        mean_x <- matrix(colMeans(Xtrain), nrow = 1L)
        Xtrain <- sweep(Xtrain, 2L, mean_x[1L, ], "-")
    }
    scale_x <- matrix(1, 1L, p)
    if (scaling == 2L) {
        scale_x <- matrix(apply(Xtrain, 2L, sd), nrow = 1L)
        scale_x[!is.finite(scale_x) | scale_x == 0] <- 1
        Xtrain <- sweep(Xtrain, 2L, scale_x[1L, ], "/")
    }
    mean_y <- matrix(colMeans(Ytrain), nrow = 1L)
    list(
        X = Xtrain,
        Y = sweep(Ytrain, 2L, mean_y[1L, ], "-"),
        mX = mean_x,
        vX = scale_x,
        mY = mean_y,
        n = n,
        p = p,
        m = m
    )
}

.metal_plssvd_decomposition <- function(prep, ncomp, oversample, power, seed) {
    rank <- min(max(ncomp), prep$n, prep$p, prep$m)
    if (rank < 1L) {
        stop("PLS-SVD effective rank is below one")
    }
    implicit <- .should_use_xprod_default(prep$p, prep$m, ncomp)
    value <- if (implicit) {
        .truncated_rsvd_metal_xprod(
            prep$X,
            prep$Y,
            rank,
            oversample,
            power,
            seed
        )
    } else {
        .truncated_rsvd_metal(
            .metal_crossprod(prep$X, prep$Y),
            rank,
            oversample,
            power,
            seed
        )
    }
    rank <- min(rank, ncol(value$U), nrow(value$Vt))
    list(
        R = value$U[, seq_len(rank), drop = FALSE],
        Q = t(value$Vt[seq_len(rank), , drop = FALSE]),
        singular = value$s,
        rank = rank,
        implicit = implicit
    )
}

.metal_plssvd_path <- function(prep, decomposition, ncomp, fit) {
    scores <- .metal_mm(prep$X, decomposition$R)
    gram <- .metal_crossprod(scores, scores)
    slices <- length(ncomp)
    rank <- decomposition$rank
    store <- .should_store_coefficients(prep$p, prep$m, slices, TRUE)
    B <- if (store) array(0, c(prep$p, prep$m, slices)) else NULL
    C <- array(0, c(rank, rank, slices))
    store_weights <- slices == 1L ||
        as.double(rank) * prep$m * slices * 8 <= 32 * 1024^2
    W <- if (store_weights) array(0, c(rank, prep$m, slices)) else NULL
    fitted <- if (fit) array(0, c(prep$n, prep$m, slices)) else NULL
    r2 <- rep(NA_real_, slices)
    for (index in seq_along(ncomp)) {
        k <- min(ncomp[[index]], rank)
        coefficient <- solve(
            gram[seq_len(k), seq_len(k), drop = FALSE],
            diag(decomposition$singular[seq_len(k)], nrow = k)
        )
        C[seq_len(k), seq_len(k), index] <- coefficient
        if (store_weights || store || fit) {
            weight <- coefficient %*%
                t(decomposition$Q[, seq_len(k), drop = FALSE])
        }
        if (store_weights) W[seq_len(k), , index] <- weight
        if (store) {
            B[, , index] <- .metal_mm(
                decomposition$R[, seq_len(k), drop = FALSE],
                weight
            )
        }
        if (fit) {
            value <- .metal_mm(scores[, seq_len(k), drop = FALSE], weight)
            r2[[index]] <- RQ(prep$Y, value)
            fitted[, , index] <- sweep(value, 2L, prep$mY[1L, ], "+")
        }
    }
    list(
        B = B,
        C = C,
        W = W,
        scores = scores,
        fitted = fitted,
        r2 = r2,
        store = store
    )
}

.metal_plssvd_model <- function(prep, decomposition, path, ncomp) {
    model <- list(
        C_latent = path$C,
        Q = decomposition$Q,
        Ttrain = path$scores,
        R = decomposition$R,
        mX = prep$mX,
        vX = prep$vX,
        mY = prep$mY,
        p = prep$p,
        m = prep$m,
        ncomp = ncomp,
        Yfit = path$fitted,
        R2Y = path$r2,
        backend = "metal",
        svd.method = "metal_rsvd",
        xprod_default = decomposition$implicit,
        xprod_mode = if (decomposition$implicit) {
            "metal_implicit"
        } else {
            "materialized"
        }
    )
    if (!is.null(path$W)) model$W_latent <- path$W
    if (path$store) {
        model$B <- path$B
    }
    model <- .annotate_coefficient_storage(model, path$store)
    class(model) <- "fastPLS"
    .attach_backend_control(model)
}

.pls_model1_metal <- function(
    Xtrain,
    Ytrain,
    ncomp,
    scaling,
    fit,
    rsvd_oversample,
    rsvd_power,
    seed
) {
    ncomp <- as.integer(ncomp)
    prep <- .metal_pls_preprocess(Xtrain, Ytrain, scaling)
    decomposition <- .metal_plssvd_decomposition(
        prep,
        ncomp,
        rsvd_oversample,
        rsvd_power,
        seed
    )
    path <- .metal_plssvd_path(prep, decomposition, ncomp, fit)
    .metal_plssvd_model(prep, decomposition, path, ncomp)
}

.metal_project_left <- function(values, basis, used) {
    if (used < 1L) return(values)
    previous <- basis[, seq_len(used), drop = FALSE]
    values - previous %*% crossprod(previous, values)
}

.metal_simpls_direction <- function(prep, S, V, used, oversample, power, seed) {
    if (!is.null(S)) {
        return(.truncated_rsvd_metal(
            S,
            k = 1L,
            rsvd_oversample = oversample,
            rsvd_power = power,
            seed = seed,
            left_only = TRUE
        )$U[, 1L])
    }
    sketch_rank <- min(prep$p, prep$m, 1L + max(0L, as.integer(oversample)))
    .fastpls_set_seed(seed)
    omega <- matrix(rnorm(prep$m * sketch_rank), prep$m, sketch_rank)
    sketch <- .metal_xprod_multiply(prep$X, prep$Y, omega)
    sketch <- .metal_project_left(sketch, V, used)
    for (iteration in seq_len(max(0L, as.integer(power)))) {
        q_left <- qr.Q(qr(sketch))
        right <- .metal_xprod_transpose_multiply(prep$X, prep$Y, q_left)
        q_right <- qr.Q(qr(right))
        sketch <- .metal_xprod_multiply(prep$X, prep$Y, q_right)
        sketch <- .metal_project_left(sketch, V, used)
    }
    q_left <- qr.Q(qr(sketch))
    reduced <- t(.metal_xprod_transpose_multiply(prep$X, prep$Y, q_left))
    small <- svd(reduced, nu = 1L, nv = 0L)
    drop(q_left %*% small$u[, 1L, drop = FALSE])
}

.is_usable_simpls_norm <- function(value) {
    is.finite(value) && value > .Machine$double.eps
}

.metal_simpls_resident_factors <- function(
    prep, ncomp, requested, power, seed
) {
    native <- metal_simpls_resident_cpp(
        prep$X, prep$Y, requested, max(1L, as.integer(power)), as.integer(seed)
    )
    rank <- min(
        as.integer(native$ncomp), ncol(native$R), ncol(native$Q), ncol(native$V)
    )
    if (!is.finite(rank) || rank < 1L) {
        stop("Resident Metal SIMPLS did not return a usable component.")
    }
    list(
        R = native$R[, seq_len(rank), drop = FALSE],
        Q = native$Q[, seq_len(rank), drop = FALSE],
        V = native$V[, seq_len(rank), drop = FALSE],
        rank = rank, ncomp = pmin(ncomp, rank), xprod = TRUE, resident = TRUE
    )
}

.metal_simpls_component <- function(
    prep, S, V, rank, oversample, power, component, seed
) {
    direction <- .metal_simpls_direction(
        prep, S, V, rank, oversample, power, seed + component - 1L
    )
    direction <- .metal_project_left(direction, V, rank)
    direction_norm <- sqrt(sum(direction * direction))
    if (!.is_usable_simpls_norm(direction_norm)) return(NULL)
    direction <- direction / direction_norm
    score <- drop(.metal_mm(prep$X, matrix(direction, ncol = 1L)))
    score_norm <- sqrt(sum(score * score))
    if (!.is_usable_simpls_norm(score_norm)) return(NULL)
    score <- score / score_norm
    direction <- direction / score_norm
    loading <- drop(.metal_crossprod(prep$X, matrix(score, ncol = 1L)))
    response_loading <- drop(.metal_crossprod(prep$Y, matrix(score, ncol = 1L)))
    orthogonal_loading <- .metal_project_left(loading, V, rank)
    orthogonal_norm <- sqrt(sum(orthogonal_loading * orthogonal_loading))
    if (!.is_usable_simpls_norm(orthogonal_norm)) return(NULL)
    list(
        direction = direction,
        response_loading = response_loading,
        orthogonal_loading = orthogonal_loading / orthogonal_norm
    )
}

.metal_simpls_oversampled_factors <- function(
    prep, ncomp, requested, oversample, power, seed, S
) {
    R <- matrix(0, prep$p, requested)
    Q <- matrix(0, prep$m, requested)
    V <- matrix(0, prep$p, requested)
    rank <- 0L
    for (component in seq_len(requested)) {
        next_component <- .metal_simpls_component(
            prep, S, V, rank, oversample, power, component, seed
        )
        if (is.null(next_component)) break
        R[, component] <- next_component$direction
        Q[, component] <- next_component$response_loading
        V[, component] <- next_component$orthogonal_loading
        rank <- component
        if (!is.null(S)) {
            orthogonal_loading <- next_component$orthogonal_loading
            deflation_row <- drop(crossprod(orthogonal_loading, S))
            S <- S - tcrossprod(orthogonal_loading, deflation_row)
        }
    }
    if (rank < 1L) stop("Metal SIMPLS did not return a usable component.")
    list(
        R = R[, seq_len(rank), drop = FALSE],
        Q = Q[, seq_len(rank), drop = FALSE],
        V = V[, seq_len(rank), drop = FALSE],
        rank = rank, ncomp = pmin(ncomp, rank), xprod = is.null(S),
        resident = FALSE
    )
}

.metal_simpls_factors <- function(prep, ncomp, oversample, power, seed) {
    requested <- min(max(ncomp), prep$n - 1L, prep$p)
    if (requested < 1L) stop("SIMPLS Metal effective rank is below one.")
    S_mb <- as.double(prep$p) * as.double(prep$m) * 8 / 1024^2
    if (S_mb > 512 && isTRUE(.metal_resident_simpls_enabled())) {
        return(.metal_simpls_resident_factors(
            prep, ncomp, requested, power, seed
        ))
    }
    S <- if (S_mb <= 512) .metal_crossprod(prep$X, prep$Y) else NULL
    .metal_simpls_oversampled_factors(
        prep, ncomp, requested, oversample, power, seed, S
    )
}

.metal_simpls_path <- function(prep, factors, ncomp, fit) {
    scores <- .metal_mm(prep$X, factors$R)
    slices <- length(ncomp); rank <- factors$rank
    store <- .should_store_coefficients(prep$p, prep$m, slices, TRUE)
    B <- if (store) array(0, c(prep$p, prep$m, slices)) else NULL
    fitted <- if (fit) array(0, c(prep$n, prep$m, slices)) else NULL
    r2 <- rep(NA_real_, slices)
    for (index in seq_along(ncomp)) {
        k <- min(ncomp[[index]], rank)
        if (store || fit) {
            weight <- t(factors$Q[, seq_len(k), drop = FALSE])
        }
        if (store) {
            B[, , index] <- .metal_mm(
                factors$R[, seq_len(k), drop = FALSE], weight
            )
        }
        if (fit) {
            value <- .metal_mm(scores[, seq_len(k), drop = FALSE], weight)
            r2[[index]] <- RQ(prep$Y, value)
            fitted[, , index] <- sweep(value, 2L, prep$mY[1L, ], "+")
        }
    }
    list(B = B, scores = scores, fitted = fitted, r2 = r2,
        store = store, ncomp = pmin(ncomp, rank))
}

.metal_simpls_model <- function(prep, factors, path) {
    model <- list(
        Q = factors$Q, Ttrain = path$scores,
        R = factors$R, V = factors$V, mX = prep$mX, vX = prep$vX,
        mY = prep$mY, p = prep$p, m = prep$m, ncomp = path$ncomp,
        Yfit = path$fitted, R2Y = path$r2, backend = "metal",
        svd.method = "metal_rsvd", xprod_default = isTRUE(factors$xprod),
        xprod_mode = if (isTRUE(factors$resident)) {
            "metal_resident_rank_one_simpls"
        } else {
            "metal_oversampled_simpls"
        }
    )
    if (path$store) model$B <- path$B
    model <- .annotate_coefficient_storage(model, path$store)
    class(model) <- "fastPLS"
    .attach_backend_control(model)
}

.pls_model2_fast_metal <- function(
    Xtrain,
    Ytrain,
    ncomp,
    scaling,
    fit,
    rsvd_oversample,
    rsvd_power,
    seed
) {
    ncomp <- sort(unique(as.integer(ncomp)))
    prep <- .metal_pls_preprocess(Xtrain, Ytrain, scaling)
    factors <- .metal_simpls_factors(
        prep, ncomp, rsvd_oversample, rsvd_power, seed
    )
    path <- .metal_simpls_path(prep, factors, ncomp, fit)
    .metal_simpls_model(prep, factors, path)
}

.metal_prediction_weights <- function(object, component, slice) {
    index <- seq_len(component)
    if (!is.null(object$W_latent)) {
        return(matrix(object$W_latent[index, , slice],
            nrow = component, ncol = object$m))
    }
    if (is.null(object$Q)) return(NULL)
    weights <- t(object$Q[, index, drop = FALSE])
    if (!is.null(object$C_latent)) {
        coefficient <- matrix(object$C_latent[index, index, slice],
            nrow = component, ncol = component)
        # Retain the stored weights' host arithmetic before Metal prediction.
        weights <- coefficient %*% weights
    }
    weights
}

.pls_predict_metal <- function(object, Xtest, proj = FALSE) {
    Xscaled <- .fastpls_preprocess_test(Xtest, object$mX, object$vX)
    ncomp <- as.integer(object$ncomp)
    ns <- length(ncomp)
    n <- nrow(Xscaled)
    m <- as.integer(object$m)
    Ypred <- array(0, dim = c(n, m, ns))
    B_obj <- object[["B", exact = TRUE]]

    Tfull <- NULL
    if (!is.null(object$R) && length(object$R) > 0L) {
        maxc <- min(max(ncomp), ncol(object$R))
        Tfull <- .metal_mm(Xscaled, object$R[, seq_len(maxc), drop = FALSE])
    }

    for (i in seq_len(ns)) {
        mc <- min(ncomp[i], if (!is.null(Tfull)) ncol(Tfull) else ncomp[i])
        W <- if (!is.null(Tfull)) {
            .metal_prediction_weights(object, mc, i)
        } else NULL
        if (!is.null(W)) {
            y <- .metal_mm(Tfull[, seq_len(mc), drop = FALSE], W)
        } else if (!is.null(B_obj)) {
            B_i <- matrix(B_obj[, , i], nrow = object$p, ncol = object$m)
            y <- .metal_mm(Xscaled, B_i)
        } else {
            stop(
                "Metal prediction requires compact factors or coefficients.",
                call. = FALSE
            )
        }
        Ypred[, , i] <- sweep(y, 2, as.numeric(object$mY[1, ]), "+")
    }
    out <- list(Ypred = Ypred)
    if (isTRUE(proj) && !is.null(Tfull)) {
        out$Ttest <- Tfull
    }
    out
}

.opls_filter_metal <- function(X, Y, north, scaling) {
    prep <- .fastpls_preprocess_train(X, scaling); Xf <- prep$X
    Yc <- sweep(as.matrix(Y), 2, colMeans(as.matrix(Y)), "-")
    north <- as.integer(north); used <- 0L
    W_orth <- matrix(0, nrow = ncol(Xf), ncol = max(0L, north))
    P_orth <- matrix(0, nrow = ncol(Xf), ncol = max(0L, north))
    if (north > 0L) {
        for (a in seq_len(north)) {
            s <- fastsvd(.metal_crossprod(Xf, Yc), ncomp = 1L,
                backend = "metal",
                method = "rsvd", oversample = 32L, power = 5L)
            w <- s$u[, 1L, drop = FALSE]
            w_norm <- sqrt(sum(w * w))
            if (!is.finite(w_norm) || w_norm <= 0) {
                break
            }
            w <- w / w_norm; tt <- .metal_mm(Xf, w)
            tt_ss <- drop(crossprod(tt))
            if (!is.finite(tt_ss) || tt_ss <= 0) {
                break
            }
            pp <- .metal_crossprod(Xf, tt) / tt_ss
            w_orth <- pp - w %*% crossprod(w, pp) / drop(crossprod(w))
            wo_norm <- sqrt(sum(w_orth * w_orth))
            if (!is.finite(wo_norm) || wo_norm <= 0) {
                break
            }
            w_orth <- w_orth / wo_norm; t_orth <- .metal_mm(Xf, w_orth)
            to_ss <- drop(crossprod(t_orth))
            if (!is.finite(to_ss) || to_ss <= 0) {
                break
            }
            p_orth <- .metal_crossprod(Xf, t_orth) / to_ss
            Xf <- Xf - .metal_outer(t_orth, p_orth)
            used <- used + 1L
            W_orth[, used] <- w_orth[, 1L]
            P_orth[, used] <- p_orth[, 1L]
        }
    }
    if (used == 0L) {
        W_orth <- matrix(0, nrow = ncol(Xf), ncol = 0L)
        P_orth <- matrix(0, nrow = ncol(Xf), ncol = 0L)
    }
    else {
        W_orth <- W_orth[, seq_len(used), drop = FALSE]
        P_orth <- P_orth[, seq_len(used), drop = FALSE]
    }
    list(X = Xf, mX = prep$mX, vX = prep$vX, W_orth = W_orth, P_orth = P_orth,
        north = used)
}

.opls_apply_filter_metal <- function(X, mX, vX, W_orth, P_orth) {
    Xf <- .fastpls_preprocess_test(X, mX, vX)
    if (ncol(W_orth) > 0L) {
        for (a in seq_len(ncol(W_orth))) {
            t_orth <- .metal_mm(Xf, W_orth[, a, drop = FALSE])
            Xf <- Xf - .metal_outer(t_orth, P_orth[, a, drop = FALSE])
        }
    }
    Xf
}

.kernel_matrix_metal <- function(X1, X2, kernel, gamma, degree, coef0) {
    dots <- .metal_mm(X1, t(X2))
    if (identical(kernel, "linear")) {
        return(dots)
    }
    if (identical(kernel, "poly")) {
        return((gamma * dots + coef0)^as.integer(degree))
    }
    n1 <- rowSums(X1 * X1)
    n2 <- rowSums(X2 * X2)
    dist2 <- outer(n1, n2, "+") - 2 * dots
    dist2[dist2 < 0 & dist2 > -1e-10] <- 0
    exp(-gamma * dist2)
}

.pls_metal_fit_core <- function(
    Xtrain, Ytrain, ncomp, scaling, method, fit,
    rsvd_oversample, rsvd_power, seed
) {
    if (identical(method, "plssvd")) {
        cap <- .cap_plssvd_ncomp(
            ncomp,
            nrow(Xtrain),
            ncol(Xtrain),
            ncol(Ytrain),
            warn = TRUE
        )
        return(.pls_model1_metal(
            Xtrain,
            Ytrain,
            cap$ncomp,
            scaling,
            fit,
            rsvd_oversample = rsvd_oversample,
            rsvd_power = rsvd_power,
            seed = seed
        ))
    }
    if (
        !isTRUE(.metal_resident_simpls_enabled()) &&
            !isTRUE(.metal_experimental_iterative_enabled())
    ) {
        stop(
            "backend='metal' requires the Metal SIMPLS-family path; enable ",
            "FASTPLS_METAL_RESIDENT_SIMPLS or use backend='cpu'.",
            call. = FALSE
        )
    }
    .pls_model2_fast_metal(
        Xtrain,
        Ytrain,
        ncomp,
        scaling,
        fit,
        rsvd_oversample = rsvd_oversample,
        rsvd_power = rsvd_power,
        seed = seed
    )
}

.pls_metal_finish <- function(
    model,
    Xtrain,
    Ytrain_original,
    yprep,
    classifier,
    lda_ridge,
    return_variance,
    Xtest,
    Ytest,
    proj
) {
    model$predict_backend <- "metal"
    model$backend <- "metal"
    model$svd.method <- "metal_rsvd"
    model$predict_latent_ok <- TRUE
    model <- .enable_flash_prediction(model, "cpu")
    model$predict_backend <- "metal"
    model$classification <- yprep$classification
    model$lev <- yprep$lev
    model <- .attach_lda_classifier(
        model,
        Xtrain,
        Ytrain_original,
        classifier,
        lda_ridge
    )
model <- .maybe_attach_pls_variance_explained(model, Xtrain, return_variance)
    class(model) <- "fastPLS"
    if (!is.null(Xtest)) {
        res <- predict(
            model,
            Xtest,
            Ytest = Ytest,
            proj = proj,
            backend = "metal"
        )
        model <- c(model, res)
        class(model) <- "fastPLS"
    }
    model <- .attach_backend_control(model)
    model
}

.pls_metal_context <- function(
    Xtrain,
    Ytrain,
    ncomp,
    scaling,
    method,
    kernel,
    classifier
) {
    if (!isTRUE(has_metal())) {
        .fastpls_require_backend_available("metal", "PLS fitting")
    }
    method <- match.arg(
        method,
        c("simpls", "plssvd", "opls", "kernelpls")
    )
    scaling <- match.arg(
        scaling,
        c("centering", "autoscaling", "none")
    )
    kernel <- match.arg(kernel, c("linear", "rbf", "poly"))
    classifier <- .resolve_classifier_for_backend(classifier, "metal")
    Xtrain <- as.matrix(Xtrain)
    yprep <- .prepare_response(Ytrain)
    if (identical(method, "plssvd") && isTRUE(yprep$classification)) {
        ncomp <- .cap_plssvd_ncomp(
            ncomp,
            nrow(Xtrain),
            ncol(Xtrain),
            ncol(yprep$Ytrain),
            factor_response = TRUE,
            warn = TRUE
        )$ncomp
    }
    list(
        X = Xtrain,
        Y = Ytrain,
        yprep = yprep,
        ncomp = ncomp,
        scaling = scaling,
        scal = pmatch(
            scaling,
            c("centering", "autoscaling", "none")
        )[1L],
        method = method,
        kernel = kernel,
        classifier = classifier
    )
}

.pls_metal_opls <- function(context, config) {
    filt <- .opls_filter_metal(context$X,
        .supervised_response_matrix(context$Y),
        config$north, context$scaling)
    .opls_require_predictive_rank(context$ncomp, filt$X, filt$north,
        !identical(context$scaling, "none"))
    inner <- .pls_metal_fit_core(filt$X, context$yprep$Ytrain, context$ncomp,
        3L,
        "simpls", config$fit, config$oversample, config$power, config$seed)
    inner <- .pls_metal_finish(inner, filt$X, context$Y, context$yprep,
        context$classifier,
        config$lda_ridge, config$return_variance, NULL, NULL, FALSE)
    model <- list(inner_model = inner, mX = filt$mX, vX = filt$vX,
        W_orth = filt$W_orth,
        P_orth = filt$P_orth, north = filt$north, opls_engine = "metal",
        ncomp = inner$ncomp,
        backend = "metal", predict_backend = "metal",
        svd.method = inner$svd.method)
    model <- .inherit_inner_variance_explained(model, inner)
    model <- .inherit_inner_fit_outputs(model, inner)
    class(model) <- c("fastPLSOpls", "fastPLS")
    if (!is.null(config$Xtest)) {
        result <- predict(model, config$Xtest, Ytest = config$Ytest,
            proj = config$proj, backend = "metal")
        model <- c(model, result)
        class(model) <- c("fastPLSOpls", "fastPLS")
    }
    .attach_backend_control(model)
}

.pls_metal_linear_kernel <- function(context, config) {
    model <- .pls_metal_fit_core(
        context$X,
        context$yprep$Ytrain,
        context$ncomp,
        context$scal,
        "simpls",
        config$fit,
        config$oversample,
        config$power,
        config$seed
    )
    model$kernel <- "linear"
    model$kernel_engine <- "metal_direct"
    model$kernel_linear_direct <- TRUE
    .pls_metal_finish(
        model,
        context$X,
        context$Y,
        context$yprep,
        context$classifier,
        config$lda_ridge,
        config$return_variance,
        config$Xtest,
        config$Ytest,
        config$proj
    )
}

.pls_metal_nonlinear_kernel <- function(context, config) {
    prep <- .fastpls_preprocess_train(context$X, context$scaling)
    gamma <- .kernel_pls_gamma(config$gamma, prep$X)
    K <- .kernel_matrix_metal(prep$X, prep$X, context$kernel, gamma,
        config$degree,
        config$coef0)
    centered <- .center_kernel_train_base(K)
    inner <- .pls_metal_fit_core(centered$K, context$yprep$Ytrain,
        context$ncomp,
        3L, "simpls", config$fit, config$oversample, config$power, config$seed)
    inner <- .pls_metal_finish(inner, centered$K, context$Y, context$yprep,
        context$classifier,
        config$lda_ridge, config$return_variance, NULL, NULL, FALSE)
    model <- list(inner_model = inner, Xref = prep$X, mX = prep$mX,
        vX = prep$vX,
        kernel = context$kernel,
        kernel_id = .kernel_pls_kernel_id(context$kernel),
        gamma = gamma, degree = as.integer(config$degree),
        coef0 = config$coef0,
        kernel_center = centered, kernel_engine = "metal", ncomp = inner$ncomp,
        backend = "metal", predict_backend = "metal",
        svd.method = inner$svd.method)
    model <- .inherit_inner_variance_explained(model, inner)
    model <- .inherit_inner_fit_outputs(model, inner)
    class(model) <- c("fastPLSKernel", "fastPLS")
    if (!is.null(config$Xtest)) {
        result <- predict(model, config$Xtest, Ytest = config$Ytest,
            proj = config$proj, backend = "metal")
        model <- c(model, result)
        class(model) <- c("fastPLSKernel", "fastPLS")
    }
    .attach_backend_control(model)
}

.pls_metal_kernel <- function(context, config) {
    if (identical(context$kernel, "linear")) {
        return(.pls_metal_linear_kernel(context, config))
    }
    .pls_metal_nonlinear_kernel(context, config)
}

.pls_metal_standard <- function(context, config) {
    model <- .pls_metal_fit_core(
        context$X,
        context$yprep$Ytrain,
        context$ncomp,
        context$scal,
        context$method,
        config$fit,
        config$oversample,
        config$power,
        config$seed
    )
    model$pls_method <- context$method
    .pls_metal_finish(
        model,
        context$X,
        context$Y,
        context$yprep,
        context$classifier,
        config$lda_ridge,
        config$return_variance,
        config$Xtest,
        config$Ytest,
        config$proj
    )
}

.pls_metal <- function(Xtrain, Ytrain, Xtest = NULL, Ytest = NULL, ncomp = 2,
    scaling = c("centering",
        "autoscaling", "none"), method = c("simpls", "plssvd", "opls",
        "kernelpls"),
    north = 1L, kernel = c("linear", "rbf", "poly"), gamma = NULL, degree = 3L,
    coef0 = 1, rsvd_oversample = 32L, rsvd_power = 5L, seed = 1L,
    classifier = c("argmax",
        "lda"), lda_ridge = 1e-08, fit = FALSE, return_variance = TRUE,
    proj = FALSE) {
    context <- .pls_metal_context(Xtrain, Ytrain, ncomp, scaling, method,
        kernel,
        classifier)
    config <- list(Xtest = Xtest, Ytest = Ytest, north = north, gamma = gamma,
        degree = degree, coef0 = coef0, oversample = rsvd_oversample,
        power = rsvd_power,
        seed = seed, lda_ridge = lda_ridge, fit = fit,
        return_variance = return_variance,
        proj = proj)
    if (identical(context$method, "opls")) {
        return(.pls_metal_opls(context, config))
    }
    if (identical(context$method, "kernelpls")) {
        return(.pls_metal_kernel(context, config))
    }
    .pls_metal_standard(context, config)
}

.pls_svd_context <- function(
    svd.method,
    dots,
    backend,
    method,
    classification,
    Xtrain,
    Ytrain
) {
    control <- .resolve_svd_control(
        svd.method = svd.method,
        dots = dots,
        context = "pls()"
    )
    control <- .apply_pls_rsvd_controls(
        control,
        backend,
        "pls()",
        method,
        classification,
        Xtrain,
        Ytrain
    )
    control
}

.pls_context <- function(Xtrain, Ytrain, Xtest, Ytest, method, svd.method,
    dots,
    backend, classifier, scaling) {
    backend <- .normalize_public_backend(backend)
    .fastpls_require_backend_available(backend, "pls()")
    Xtrain <- .fastpls_predictor_input(Xtrain, "Xtrain")
    if (!is.null(Xtest)) {
        Xtest <- .fastpls_predictor_input(Xtest, "Xtest")
    }
    method <- match.arg(method, c("simpls", "plssvd", "opls", "kernelpls"))
    classification <- is.factor(Ytrain) || is.character(Ytrain)
    control <- .pls_svd_context(
        svd.method,
        dots,
        backend,
        method,
        classification,
        Xtrain,
        Ytrain
    )
    float32 <- .has_float32_input(Xtrain, Ytrain, Xtest, Ytest)
    if (!float32) {
        control$svd.method <- .backend_svd_method(control$svd.method, backend)
    }
    classifier <- .resolve_classifier_for_backend(classifier, backend)
    if (.normalize_svd_method(control$svd.method) %in% c("cpu_rsvd",
        "cuda_rsvd",
        "metal_rsvd")) {
        try(rsvd_audit_reset_debug(), silent = TRUE)
    }
    list(Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest,
        method = method,
        backend = backend, backend_compiled = .compiled_backend(backend),
        classifier = classifier,
        classification = classification,
        scaling = scaling, scal = pmatch(scaling, c("centering", "autoscaling",
            "none"))[1L], control = control, float32 = float32)
}

.fastpls_attach_rsvd_profile <- function(model, control) {
    if (!is.null(model$diagnostics$rsvd)) {
        profile <- control$rsvd_profile %||% "explicit"
        model$diagnostics$rsvd$control_profile <- profile
        model$diagnostics$rsvd$requested_oversample <-
            control$rsvd_requested_oversample %||% control$rsvd_oversample
        model$diagnostics$rsvd$requested_power <-
            control$rsvd_requested_power %||% control$rsvd_power
        model$diagnostics$rsvd$setting_guidance <- switch(
            profile,
            ordinary_fast = paste(
                "The automatic ordinary-shape profile uses 32 oversampling",
                "directions and five power iterations."
            ),
            high_response_stable = paste(
                "The automatic high-response profile uses 48 oversampling",
                "directions and six power iterations."
            ),
            sparse_high_class_stable = paste(
                "The automatic sparse high-class-count profile uses 64",
                "oversampling directions and seven power iterations."
            ),
            massive_crosscovariance = paste(
                "The automatic massive-shape profile requests 12",
                "oversampling directions and one power iteration. Resident",
                "backends record the executed refresh width separately;",
                "massive float32 regression uses fresh bounded candidate",
                "blocks followed by sequential SIMPLS component updates."
            ),
            model$diagnostics$rsvd$setting_guidance
        )
    }
    model
}

.fastpls_algorithm_variant <- function(model, context, config) {
    if (identical(context$method, "plssvd")) {
        return("randomized_pls_svd")
    }
    direction <- model$diagnostics$simpls_direction
    block <- as.integer(
        direction$directions_per_solve %||%
            model$resident_controls$refresh_block %||% 1L
    )
    block_limit <- as.integer(
        direction$block_component_limit %||%
            model$resident_controls$refresh_block_limit %||% 0L
    )
    core <- if (is.finite(block_limit) && block_limit > 0L) {
        "bounded_block_randomized_simpls"
    } else if (is.finite(block) && block > 1L) {
        "block_randomized_simpls"
    } else {
        "componentwise_randomized_simpls"
    }
    if (identical(context$method, "opls")) {
        return(paste0("opls_with_", core, "_core"))
    }
    if (identical(context$method, "kernelpls")) {
        kernel <- as.character(config$kernel %||% "linear")[[1L]]
        return(paste0(kernel, "_kernel_pls_with_", core, "_core"))
    }
    core
}

.assert_fully_native_gpu_model <- function(model, context) {
    if (!identical(context$backend, "cuda")) {
        return(invisible(TRUE))
    }
    if (is.null(model$resident_state)) {
        message <- paste(
            "Internal error: the selected GPU route returned host-resident",
            "model state. No hybrid result is returned."
        )
        stop(message, call. = FALSE)
    }
    if (isTRUE(model$resident_controls$host_assisted_components)) {
        message <- paste(
            "Internal error: the selected GPU route used host-assisted",
            "component updates. No hybrid result is returned."
        )
        stop(message, call. = FALSE)
    }
    backend <- model$resident_backend %||% context$backend
    if (!identical(backend, context$backend)) {
        message <- paste(
            "Internal error: fitted model residency differs from the",
            "requested GPU backend."
        )
        stop(message, call. = FALSE)
    }
    invisible(TRUE)
}

.pls_finalize <- function(model, context, config) {
    # Restore family-specific bookkeeping while public diagnostics are built.
    model <- .fastpls_restore_internal_output_fields(model)
    .assert_fully_native_gpu_model(model, context)
    model <- .maybe_attach_x_loadings(
        model, context$Xtrain, config$return_loadings
    )
    model <- .attach_backend_control(model)
    control <- context$control
    model <- .fastpls_attach_solver_diagnostics(
        model, control$svd.method, control$rsvd_oversample,
        control$rsvd_power, control$seed, pls_family = context$method,
        classification = context$classification,
        training_samples = nrow(context$Xtrain),
        execution_backend = context$backend
    )
    model <- .fastpls_attach_rsvd_profile(model, control)
    if (!is.null(model$diagnostics$rsvd)) {
        model$diagnostics$rsvd$oversample <- as.integer(
            model$rsvd_effective_oversample %||%
                model$diagnostics$rsvd$oversample
        )
        model$diagnostics$rsvd$power <- as.integer(
            model$rsvd_effective_power %||%
                model$diagnostics$rsvd$power
        )
    }
    if (identical(context$backend, "metal")) {
        model$diagnostics$metal_operation_split <- list(
            policy = "fixed_operation_split",
            requested_oversample = control$rsvd_oversample,
            requested_power = control$rsvd_power,
            effective_oversample = as.integer(
                model$rsvd_effective_oversample %||%
                    control$rsvd_oversample
            ),
            effective_power = as.integer(
                model$rsvd_effective_power %||% control$rsvd_power
            ),
            preprocessing = "cpu",
            sequential_component_updates = "cpu",
            prediction = "cpu",
            metal_operations = paste(
                "fitting products involving the training sample matrix,",
                "including explicit",
                "cross-covariance formation, fused score/loading geometry",
                "and implicit randomized range-finder products"
            ),
            cpu_operations = paste(
                "centering, scaling, QR and reduced decomposition,",
                "orthogonalization, deflation state and coefficients"
            )
        )
        model$diagnostics$residency <- list(
            preprocessing = "cpu",
            cross_products = paste(
                "Metal for fitting products involving the training sample",
                "matrix; CPU for prediction and",
                "small cross-covariance and component-state products"
            ),
            cross_covariance = paste(
                "host-resident state with Metal sample products, or an",
                "implicit operator with persistent Metal products"
            ),
            decomposition = "CPU/Metal hybrid",
            component_updates = "cpu",
            prediction = "cpu",
            reporting = "R output assembly and metric summaries",
            route = model$execution_route
        )
    }
    if (!is.null(model$resident_state)) {
        resident_backend <- model$resident_backend %||% context$backend
        resident_simpls <- !identical(context$method, "plssvd")
        model$diagnostics$residency <- list(
            preprocessing = resident_backend,
            cross_products = resident_backend,
            cross_covariance = if (isTRUE(
                model$resident_controls$implicit_crosscovariance
            )) {
                "implicit composed X/Y products; cross-covariance not stored"
            } else {
                "explicit device-resident cross-covariance"
            },
            decomposition = resident_backend,
            component_updates = resident_backend,
            prediction = resident_backend,
            lda = if (context$classification &&
                .is_lda_classifier(context$classifier)) {
                resident_backend
            } else {
                "not_requested"
            },
            response_sums_of_squares = resident_backend,
            reporting = "R output assembly and metric summaries",
            route = paste0("resident ", resident_backend)
        )
        model$diagnostics$resident_controls <- model$resident_controls
        if (!is.null(model$diagnostics$rsvd)) {
            model$diagnostics$rsvd$oversample <- as.integer(
                model$resident_controls$effective_oversample %||%
                    model$diagnostics$rsvd$oversample
            )
            model$diagnostics$rsvd$power <- as.integer(
                model$resident_controls$effective_power %||%
                    model$diagnostics$rsvd$power
            )
            model$diagnostics$rsvd$requested_oversample <- as.integer(
                model$resident_controls$requested_oversample %||%
                    model$diagnostics$rsvd$requested_oversample
            )
            model$diagnostics$rsvd$requested_power <- as.integer(
                model$resident_controls$requested_power %||%
                    model$diagnostics$rsvd$requested_power
            )
        }
        resident_block <- as.integer(
            model$resident_controls$refresh_block %||% 1L
        )
        resident_block_limit <- as.integer(
            model$resident_controls$refresh_block_limit %||% 0L
        )
        resident_precision <- if (isTRUE(context$float32)) {
            "float32"
        } else {
            "float64"
        }
        model$diagnostics$direction_refresh <- if (resident_simpls) {
            if (resident_block > 1L) {
                if (resident_block_limit > 0L) {
                    paste(
                        "Fresh randomized candidate blocks through component",
                        resident_block_limit,
                        "followed by fresh component-wise calculations"
                    )
                } else {
                    paste(
                        "Fresh randomized candidate block followed by",
                        "sequential SIMPLS component updates"
                    )
                }
            } else {
                "Fresh randomized range calculation for each deflated component"
            }
        } else "Single randomized decomposition for the retained subspace"
        if (resident_simpls) {
            resident_power <- model$resident_controls$effective_power %||%
                control$rsvd_power
            model$diagnostics$simpls_direction <- list(
                rule = if (resident_block_limit > 0L) {
                    paste0("fresh_", resident_backend,
                        "_bounded_candidate_block")
                } else if (resident_block > 1L) {
                    paste0("fresh_", resident_backend, "_candidate_block")
                } else if (identical(resident_backend, "metal")) {
                    "fresh_randomized_direction_per_component"
                } else {
                    "fresh_oversampled_sketch_per_component"
                },
                directions_per_solve = resident_block,
                candidate_block_refresh = resident_block > 1L,
                fresh_start = TRUE,
                refresh_width = resident_block,
                block_component_limit = resident_block_limit,
                refresh_iterations = as.integer(resident_power),
                seed_rule = if (resident_block > 1L) {
                    "seed_plus_candidate_block_start"
                } else {
                    "seed_plus_component_index"
                },
                active_optimizations = c(
                    "cached_rank_one_deflation_product",
                    "persistent_device_workspace",
                    "compact_prediction",
                    if (isTRUE(model$resident_controls$implicit_crosscovariance)) {
                        "implicit_projected_crosscovariance"
                    },
                    if (isTRUE(model$resident_controls$predictor_crossprod_cache)) {
                        "cached_predictor_crossproduct"
                    },
                    paste0(
                        resident_backend, "_resident_", resident_precision,
                        "_buffers"
                    )
                ),
                approximate_execution = TRUE,
                initialization = "fresh",
                refresh = if (resident_block > 1L) {
                    "after each candidate block"
                } else {
                    "each component"
                },
                solver = if (identical(resident_backend, "metal")) {
                    "Metal power iteration on the deflated cross-covariance"
                } else {
                    "GPU randomized range and reduced SVD"
                },
                execution = paste("resident", resident_backend,
                    "component updates")
            )
        }
    }
    model$diagnostics$algorithm_variant <-
        .fastpls_algorithm_variant(model, context, config)
    model <- .fastpls_attach_pls_metrics(
        model, context$Ytrain, context$Ytest, config$bycol
    )
    .fastpls_public_pls_output(model, model$ncomp)
}

.pls_validate_float32 <- function(context, config) {
    backend <- context$backend
    solver <- context$control$svd.method
    if (!backend %in% c("cpu", "cuda", "metal")) {
        stop("float32 PLS requires a CPU, CUDA, or Metal backend.")
    }
    if (identical(backend, "metal") && !isTRUE(has_metal())) {
        .fastpls_require_backend_available("metal", "float32 PLS fitting")
    }
    if (identical(backend, "cuda") && !isTRUE(has_cuda())) {
        .fastpls_require_backend_available("cuda", "float32 PLS fitting")
    }
    if (!.normalize_svd_method(solver) %in%
        c("cpu_rsvd", "cuda_rsvd", "metal_rsvd")) {
        stop("float32 input supports rSVD only.")
    }
    if (config$return_loadings) {
        stop("return_loadings is unavailable for float32 input.")
    }
    if (config$perm.test) {
        stop("permutation tests are unavailable for float32 input.")
    }
    invisible(TRUE)
}

.pls_fit_float32_model <- function(context, config) {
    ctl <- context$control
    if (identical(context$method, "opls")) {
        .fit_float32_opls(context$Xtrain, context$Ytrain, config$ncomp,
            context$scal,
            config$north, context$backend, ctl$svd.method, ctl$rsvd_oversample,
            ctl$rsvd_power, ctl$seed, config$fit, context$classifier,
            config$lda_ridge)
    } else if (identical(context$method, "kernelpls")) {
        .fit_float32_kernelpls(context$Xtrain, context$Ytrain, config$ncomp,
            context$scal,
            config$kernel, config$gamma, config$degree, config$coef0,
            context$backend,
            ctl$svd.method, ctl$rsvd_oversample, ctl$rsvd_power, ctl$seed,
            config$fit,
            context$classifier, config$lda_ridge)
    }
    else {
        fitted <- .fit_float32_pls(context$Xtrain, context$Ytrain,
            config$ncomp,
            context$scal, context$method, context$backend, ctl$svd.method,
            ctl$rsvd_oversample,
            ctl$rsvd_power, ctl$seed, config$fit,
            store_scores = config$fit ||
                .is_lda_classifier(context$classifier))
        .attach_float32_classifier(fitted, context$Xtrain, context$Ytrain,
            context$classifier,
            config$lda_ridge)
    }
}

.pls_finish_float32 <- function(model, context, config) {
    prediction_backend <- if (identical(context$backend, "metal")) {
        "cpu"
    } else {
        context$backend
    }
    if (!config$fit && !is.null(model$R2Y)) {
        model$R2Y <- rep(NA, length(model$ncomp))
    }
    if (isTRUE(context$classification) && isTRUE(config$fit)) {
        model$Yfit <- predict(
            model,
            context$Xtrain,
            backend = prediction_backend
        )$Ypred
    }
    if (!is.null(context$Xtest)) {
        original_class <- class(model)
        predicted <- predict(
            model,
            context$Xtest,
            Ytest = context$Ytest,
            proj = config$proj,
            backend = prediction_backend
        )
        model <- c(model, predicted)
        class(model) <- original_class
    }
    model
}

.pls_fit_float32 <- function(context, config) {
    .pls_validate_float32(context, config)
    ctl <- context$control
    .warn_float32_capability(method = context$method,
        backend = context$backend,
        svd_method = ctl$svd.method, Ytrain = context$Ytrain,
        ncomp = config$ncomp,
        kernel = config$kernel, classifier = context$classifier)
    model <- .pls_fit_float32_model(context, config)
    .pls_finish_float32(model, context, config)
}

.pls_family_arguments <- function(context, config) {
    ctl <- context$control
    arguments <- list(
        Xtrain = context$Xtrain,
        Ytrain = context$Ytrain,
        Xtest = context$Xtest,
        Ytest = context$Ytest,
        ncomp = config$ncomp,
        scaling = context$scaling,
        rsvd_oversample = ctl$rsvd_oversample,
        rsvd_power = ctl$rsvd_power,
        svds_tol = ctl$svds_tol,
        svd.method = ctl$svd.method,
        seed = ctl$seed,
        fit = config$fit,
        proj = config$proj,
        classifier = context$classifier,
        lda_ridge = config$lda_ridge,
        return_variance = config$return_variance
    )
    arguments
}

.pls_fit_special_family <- function(context, config) {
    arguments <- .pls_family_arguments(context, config)
    if (identical(context$method, "opls")) {
        arguments$north <- config$north
        function_ <- switch(
            context$backend_compiled,
            cpp = .opls_cpp,
            cuda = .opls_cuda
        )
    } else {
        arguments$kernel <- config$kernel
        arguments$gamma <- config$gamma
        arguments$degree <- config$degree
        arguments$coef0 <- config$coef0
        function_ <- switch(
            context$backend_compiled,
            cpp = .kernel_pls_cpp,
            cuda = .kernel_pls_cuda
        )
    }
    do.call(function_, arguments)
}

.pls_cpu_context <- function(context, config) {
    ctl <- context$control
    solver <- match.arg(
        .normalize_svd_method(ctl$svd.method),
        c("cpu_rsvd")
    )
    X <- as.matrix(context$Xtrain)
    method_id <- .normalize_pls_method(context$method)
    compact_labels <- (is.factor(context$Ytrain) || is.character(context$Ytrain)) &&
        method_id %in% c(1L, 3L) && !isTRUE(config$perm.test)
    response <- .prepare_response(
        context$Ytrain,
        materialize_labels = !compact_labels
    )
    Y <- response$Ytrain
    response_columns <- if (compact_labels) response$n_classes else ncol(Y)
    xprod <- method_id %in%
        c(1L, 3L) &&
        !compact_labels &&
        .should_use_xprod_default(ncol(X), response_columns, config$ncomp)
    list(
        X = X,
        Y = Y,
        response = response,
        compact_labels = compact_labels,
        response_columns = response_columns,
        method_id = method_id,
        solver = solver,
        solver_id = .svd_method_id(solver),
        xprod = .ablation_xprod_override(xprod),
        precision = "implicit64"
    )
}

.pls_core_store_coefficients <- function(result, method_id) {
    store <- .should_store_coefficients(
        result$p, result$m, length(result$ncomp), TRUE
    )
    if (store) {
        coefficients <- array(
            0,
            dim = c(result$p, result$m, length(result$ncomp))
        )
        for (index in seq_along(result$ncomp)) {
            count <- result$ncomp[[index]]
            coefficients[, , index] <- if (method_id == 1L) {
                latent <- if (is.list(result$W_latent)) {
                    result$W_latent[[index]]
                } else {
                    matrix(
                        result$W_latent[seq_len(count), , index],
                        nrow = count,
                        ncol = result$m
                    )
                }
                result$R[, seq_len(count), drop = FALSE] %*%
                    latent
            } else {
                tcrossprod(
                    result$R[, seq_len(count), drop = FALSE],
                    result$Q[, seq_len(count), drop = FALSE]
                )
            }
        }
        result$B <- coefficients
    }
    .annotate_coefficient_storage(result, store)
}

.pls_cpu_fit <- function(context, config, cpu) {
    if (!cpu$method_id %in% c(1L, 3L) || cpu$solver_id != 4L) {
        stop(
            "Internal error: the CPU package route supports native rSVD ",
            "PLS-SVD and SIMPLS only.",
            call. = FALSE
        )
    }
    ctl <- context$control
    ncomp <- config$ncomp
    if (cpu$method_id == 1L) {
        ncomp <- .cap_plssvd_ncomp(
            ncomp,
            nrow(cpu$X),
            ncol(cpu$X),
            cpu$response_columns,
            factor_response = cpu$response$classification,
            warn = TRUE
        )$ncomp
    }
    store_scores <- config$fit || .is_lda_classifier(context$classifier)
    if (isTRUE(cpu$compact_labels)) {
        fit_core <- if (cpu$method_id == 1L) {
            pls_labels_core_cpp
        } else {
            pls_simpls_labels_core_cpp
        }
        result <- fit_core(
            predictors = cpu$X,
            labels = cpu$response$labels,
            class_count = cpu$response$n_classes,
            components = as.integer(ncomp),
            scaling = context$scal,
            fit = config$fit,
            store_scores = store_scores,
            oversample = ctl$rsvd_oversample,
            power = ctl$rsvd_power,
            seed = ctl$seed
        )
    } else {
        fit_core <- if (isTRUE(cpu$xprod)) {
            pls_matrix_core_xprod_cpp
        } else {
            pls_matrix_core_cpp
        }
        result <- fit_core(
            predictors = cpu$X,
            responses = cpu$Y,
            components = as.integer(ncomp),
            scaling = context$scal,
            fit = config$fit,
            store_scores = store_scores,
            method = cpu$method_id,
            oversample = ctl$rsvd_oversample,
            power = ctl$rsvd_power,
            seed = ctl$seed
        )
    }
    model <- .pls_core_store_coefficients(result, cpu$method_id)
    class(model) <- "fastPLS"
    model
}

.pls_permutation_fit <- function(context, config, cpu, X) {
    permutation_cpu <- cpu
    permutation_cpu$X <- X
    permutation_config <- config
    permutation_config$fit <- TRUE
    permutation_config$perm.test <- FALSE
    .pls_cpu_fit(context, permutation_config, permutation_cpu)
}

.pls_permutation_tables <- function(model, values, r2, correlations, ncomp) {
    times <- nrow(values)
    permutation <- data.frame(
        type = rep("permutation", times * length(ncomp) * 2L),
        permutation = rep(seq_len(times), times = length(ncomp) * 2L),
        ncomp = rep(rep(as.integer(ncomp), each = times), times = 2L),
        metric = rep(c("R2", "Q2"), each = times * length(ncomp)),
        cor = rep(rep(correlations, times = length(ncomp)), times = 2L),
        value = c(as.numeric(r2), as.numeric(values)),
        stringsAsFactors = FALSE
    )
    observed <- data.frame(
        type = "observed",
        permutation = NA_integer_,
        ncomp = rep(as.integer(ncomp), times = 2L),
        metric = rep(c("R2", "Q2"), each = length(ncomp)),
        cor = 1,
        value = c(as.numeric(model$R2Y), as.numeric(model$Q2Y)),
        stringsAsFactors = FALSE
    )
    rbind(permutation, observed)
}

.pls_attach_permutation_summary <- function(
    model,
    values,
    r2,
    correlations,
    errors,
    ncomp
) {
    times <- nrow(values)
    model$pval <- vapply(
        seq_along(ncomp),
        function(index) {
            .fastpls_permutation_pvalue(values[, index], model$Q2Y[[index]])
        },
        numeric(1L)
    )
    names(model$pval) <- names(model$Q2Y)
    model$permutation_unit <- "rows"
    model$permutation_group_sizes_preserved <- TRUE
    model$permutation_class_frequencies_preserved <- TRUE
    model$permutation_folds <- "not applicable (fixed train/test split)"
    model$permutation_solver_seed <- "fixed across observed and null fits"
    model$permutation_requested <- times
    model$permutation_completed <- sum(rowSums(is.finite(values)) > 0L)
    model$permutation_failed <- times - model$permutation_completed
    model$permutation_completed_by_component <- colSums(is.finite(values))
    model$permutation_failed_by_component <- times -
        model$permutation_completed_by_component
    names(model$permutation_completed_by_component) <- names(model$Q2Y)
    names(model$permutation_failed_by_component) <- names(model$Q2Y)
    model$permutation_errors <- errors
    model$permutation <- .pls_permutation_tables(
        model,
        values,
        r2,
        correlations,
        ncomp
    )
    model
}

.pls_permutation_attempt <- function(context, config, cpu, permutation) {
    tryCatch(
        {
            fit <- .pls_permutation_fit(
                context,
                config,
                cpu,
                cpu$X[permutation, , drop = FALSE]
            )
            fit$classification <- cpu$response$classification
            fit$lev <- cpu$response$lev
            predicted <- predict(
                fit,
                context$Xtest,
                context$Ytest,
                backend = "cpu"
            )
            list(r2 = fit$R2Y, q2 = predicted$Q2Y, error = NA_character_)
        },
        error = function(error) {
            list(r2 = NULL, q2 = NULL, error = conditionMessage(error))
        }
    )
}

.pls_run_permutations <- function(model, context, config, cpu) {
    times <- as.integer(config$times)[1L]
    indices <- .fastpls_permutation_indices(
        seq_len(nrow(cpu$X)),
        times,
        as.integer(context$control$seed) + 100000L
    )
    values <- r2 <- matrix(NA_real_, times, length(model$ncomp))
    correlations <- rep(NA_real_, times)
    errors <- rep(NA_character_, times)
    for (index in seq_len(times)) {
        permutation <- indices[[index]]
        correlations[[index]] <- .fastpls_permutation_cor(
            cpu$Y,
            permutation
        )
        attempt <- .pls_permutation_attempt(context, config, cpu, permutation)
        errors[[index]] <- attempt$error
        if (!is.null(attempt$r2)) {
            r2[index, ] <- as.numeric(attempt$r2)
        }
        if (!is.null(attempt$q2)) values[index, ] <- as.numeric(attempt$q2)
    }
    .pls_attach_permutation_summary(
        model,
        values,
        r2,
        correlations,
        errors,
        model$ncomp
    )
}

.pls_finish_cpu <- function(model, context, config, cpu) {
    model$xprod_default <- cpu$xprod
    model$pls_method <- if (cpu$method_id == 1L) "plssvd" else "simpls"
    model$predict_latent_ok <- TRUE
    if (config$fit) {
        model <- .attach_train_scores(model, cpu$X)
    }
    model <- .enable_flash_prediction(model, "cpu")
    model$classification <- cpu$response$classification
    model$lev <- cpu$response$lev
    model <- .attach_lda_classifier(
        model,
        cpu$X,
        context$Ytrain,
        context$classifier,
        config$lda_ridge
    )
    model <- .maybe_attach_pls_variance_explained(
        model,
        cpu$X,
        config$return_variance
    )
    if (!config$fit && !is.null(model$R2Y)) {
        model$R2Y <- rep(NA_real_, length(model$ncomp))
    }
    if (!is.null(context$Xtest)) {
        model <- c(
            model,
            predict(
                model,
                as.matrix(context$Xtest),
                context$Ytest,
                proj = config$proj,
                backend = "cpu"
            )
        )
        if (config$perm.test) {
            model <- .pls_run_permutations(model, context, config, cpu)
        }
    }
    if (cpu$response$classification && config$fit) {
        class(model) <- "fastPLS"
        model$Yfit <- predict.fastPLS(model, cpu$X, backend = "cpu")$Ypred
    }
    class(model) <- "fastPLS"
    model
}

.pls_dispatch <- function(context, config) {
    if (identical(context$method, "kernelpls") &&
        identical(config$kernel, "linear")) {
        # A linear kernel is the original predictor space. Reuse the same
        # accelerated SIMPLS route on every backend.
        direct_context <- context
        direct_context$method <- "simpls"
        model <- .pls_dispatch(direct_context, config)
        model$kernel <- "linear"
        model$kernel_engine <- paste0(context$backend, "_direct")
        model$kernel_linear_direct <- TRUE
        return(model)
    }
    if (identical(context$backend, "cuda")) {
        return(.pls_fit_resident_cuda(context, config))
    }
    if (identical(context$backend, "metal")) {
        if (!isTRUE(context$float32)) {
            stop(
                "Apple Metal does not provide native float64 arithmetic. Use float32 input or another backend; no CPU fallback is performed.",
                call. = FALSE
            )
        }
        return(.pls_fit_float32(context, config))
    }
    if (context$float32) {
        return(.pls_fit_float32(context, config))
    }
    if (context$method %in% c("opls", "kernelpls")) {
        return(.pls_fit_special_family(context, config))
    }
    cpu <- .pls_cpu_context(context, config)
    model <- .pls_cpu_fit(context, config, cpu)
    .pls_finish_cpu(model, context, config, cpu)
}

#' Partial Least Squares with selectable model family and backend
#'
#' Fits PLS-SVD, SIMPLS, OPLS, or kernel PLS models for regression or
#' classification using a selected CPU, CUDA, or operation-split Apple Metal
#' backend. The fitted
#' model can include predictions for held-out samples, latent scores, fitted
#' values, variance summaries, and optional classification heads.
#'
#' The compiled CPU backend uses the BLAS/LAPACK implementation selected at
#' package build time. A multithreaded BLAS can execute eligible matrix products
#' on several CPU cores, but the SIMPLS deflation sequence remains serial and
#' additional threads are not guaranteed to reduce runtime.
#'
#' Supplying `float::float32` predictors or responses requests single-precision
#' execution without silent promotion to double. Float32 is route-specific,
#' however, rather than a package-wide speed or memory guarantee. CPU, CUDA,
#' and Metal rSVD PLS-SVD and SIMPLS have current route-level validation.
#' CUDA provides fully device-resident float32 and float64 routes for all four
#' PLS families. `backend = "metal"` is a float32 Apple-silicon route. It
#' uses one operation-split implementation rather than selecting a complete
#' CPU or Metal estimator from dataset dimensions. CPU code handles
#' preprocessing, QR and reduced decompositions, sequential orthogonalization,
#' deflation bookkeeping, LDA, and coefficient assembly. Persistent Metal
#' workspaces handle fitting products involving the training sample matrix,
#' including
#' explicit cross-covariance formation, fused score/loading products, and
#' randomized range-finder products for an implicit cross-covariance operator.
#' CPU code retains the smaller cross-covariance and component-state products.
#' Prediction uses the compact host
#' factors. Its CPU/Metal assignment is independent of dataset shape. Apple
#' GPUs do not provide the native float64 arithmetic required here.
#'
#' Float64 remains the baseline precision. Raw float32 matrices require about
#' half the input storage, but paired tests show route-dependent runtime,
#' incremental host RSS, and device-memory behavior. `pls()` therefore warns
#' for experimental or measured-risk regimes, including nonlinear kernels
#' and extreme multivariate responses. For
#' numeric responses with at least 10,000 columns and at least 50 components,
#' PLS-SVD and SIMPLS-derived routes warn about observed performance, memory,
#' or precision risk in regimes for which current validation remains limited.
#' Unsupported combinations stop before allocation.
#'
#' For a factor response with \eqn{C} levels, centred dummy coding has rank at
#' most \eqn{C-1}; consequently, \code{method = "plssvd"} caps the effective
#' component count at \eqn{C-1}. The requested PLS estimator is always
#' enforced: `method = "simpls"` fits SIMPLS and `method = "plssvd"` fits
#' PLS-SVD. If a large CUDA classification response would exceed the guarded
#' dense-indicator path, SIMPLS stops with a clear error rather than silently
#' substituting PLS-SVD. CUDA SIMPLS-LDA uses SIMPLS latent scores.
#'
#' The rSVD SIMPLS route is an approximate execution of the de Jong component
#' sequence. Component-wise CPU, CUDA, and Metal routes generate a fresh seeded
#' sketch for each component from the current deflated cross-covariance
#' operator. When the explicit
#' predictor-response cross-covariance would exceed 512 MiB, CPU, Metal, and
#' CUDA float64 use a fresh seeded rank-one randomized direction for
#' every component; CUDA keeps the matrices and component state resident on the
#' device. For massive float32 regression, CUDA can refresh up to eight fresh
#' candidate directions at a time through component 64 and then returns to
#' component-wise direction calculations. Eligible
#' large CPU, CUDA, and Metal classification workloads can instead refresh a small block
#' of candidates, which are accepted sequentially by the standard SIMPLS
#' orthogonalization and deflation updates. This block route preserves the
#' supervised update geometry but is not claimed to reproduce a fresh dominant-
#' direction solve after every deflation. PLS-SVD uses 32 oversampling
#' directions and five power iterations by default. For ordinary matrix shapes,
#' accelerated SIMPLS, OPLS, and kernel-PLS use 32 directions and five power
#' iterations. When the explicit cross-covariance exceeds 512 MiB, the
#' SIMPLS-family profile requests `oversample = 12` and `power = 1`; the
#' executed refresh width and effective controls are recorded separately in
#' `diagnostics`.
#' Explicit controls supplied through `...` override this
#' policy. `fastsvd()` also uses 32 oversampling directions and five power
#' iterations. Only native rSVD is available through the public interface.
#'
#' Like reference SIMPLS software, one fit supplies the sequential component
#' path through the largest requested component count; fastPLS does not claim
#' this path construction as a novelty. Additional optimizations include cached
#' rank-one deflation products, incremental coefficient and fitted-value
#' updates, conditional cross-product caching, compact latent prediction, and
#' matrix-free cross-covariance products. When the response dimension is no
#' larger than the predictor dimension and is at most 512, the randomized
#' direction calculation can operate through the response-side Gram matrix
#' `S' S`. After each accepted SIMPLS component this smaller matrix receives a
#' rank-one update, avoiding repeated tall cross-covariance sketches while
#' retaining the same subsequent orthogonalization and deflation sequence.
#'
#' Randomized SVD is an approximate solver. The returned `diagnostics`
#' performs inexpensive structural checks for finite latent factors and the
#' requested effective component count. These checks do not establish agreement
#' with an independent reference fit. Use an independent reference for
#' confirmatory
#' inference,
#' coefficient or subspace interpretation, ill-conditioned or rank-deficient
#' matrices, slowly decaying singular spectra, or whenever repeated rSVD seeds
#' give materially different predictions. The package validation treats an
#' rSVD approximation as outside the numerical screen when relative prediction
#' or score error exceeds 0.01, the corresponding correlation is below 0.995,
#' a latent-subspace angle exceeds 0.1 degrees, classification-label agreement
#' is below 0.995, or the absolute predictive-metric difference exceeds 0.005
#' relative to a matched independent high-accuracy fit.
#'
#' @details For latent-space LDA, the pooled covariance is computed as
#'   \eqn{(T^T T - \sum_c n_c \mu_c \mu_c^T) / \max(1, n-C)} without creating
#'   centered class blocks. Class coefficients are obtained by Cholesky
#'   factorization and triangular solves, never by explicit covariance
#'   inversion. Let \eqn{s = \mathrm{trace}(\Sigma)/q}, with \eqn{s=1} when the
#'   scale is non-finite or non-positive. The implementation tries
#'   \eqn{\lambda=\rho s} for \eqn{\rho} equal to \code{1e-8}, \code{1e-6},
#'   \code{1e-5}, \code{1e-4}, \code{1e-3}, and \code{1e-2}, in that order, and
#'   advances only after Cholesky failure. This is a deterministic numerical
#'   fallback, not a fitted hyperparameter. Prediction uses
#'   \eqn{t^T w_c - 0.5\mu_c^T w_c + \log(n_c/n)}.
#'
#' @param Xtrain Numeric training predictor matrix, a `float::float32`
#'   predictor matrix for the supported float32 route, or a
#'   `Biobase::ExpressionSet`. ExpressionSet assay rows are treated as
#'   variables and columns as samples.
#' @param Ytrain Training response (numeric or factor).
#' @param Xtest Optional test predictor matrix or `Biobase::ExpressionSet`.
#' @param Ytest Optional test response for independent-test `Q2Y`, whose
#'   denominator uses the training-response mean.
#' @param ncomp Number of components (scalar or vector).
#' @param scaling One of \code{centering}, \code{autoscaling}, or \code{none}.
#' @param method One of \code{simpls}, \code{plssvd}, \code{opls}, or
#' \code{kernelpls}.
#'   `simpls` uses the fastPLS accelerated SIMPLS core.
#' @param svd.method SVD algorithm. Only \code{rsvd}, the native randomized
#'   SVD for the selected backend, is supported.
#' @param classifier Classification decision rule. \code{argmax} keeps the
#'   standard PLS-DA response-score argmax. \code{lda} fits a regularized LDA
#'   classifier on the PLS latent scores.
#' @param lda_ridge Deprecated compatibility argument. It is ignored and emits
#'   a warning when supplied. PLS-LDA uses a fixed, scale-normalized Cholesky
#'   fallback sequence rather than a user-tuned ridge.
#' @param fit Return fitted values and `R2Y` when `TRUE`.
#' @param bycol For matrix-valued regression responses, calculate response-wise
#'   metrics in `metrics`. The default `FALSE` returns only aggregate metrics.
#' @param return_variance Compute predictor-space latent-variable variance
#'   explained. Set to `FALSE` for timing/memory benchmarks that do not need
#'   plotting variance metadata.
#' @param return_loadings Compute and store predictor loadings `P`. The default
#'   is `FALSE` because most prediction workflows only need the projection
#'   weights and response-side coefficients.
#' @param proj Return projected `Ttest` when `TRUE`.
#' @param perm.test Run a single-split permutation test when `Xtest` and
#'   `Ytest` are supplied. Because `pls()` has no grouping argument, the
#'   exchangeability unit is one training row. Training rows are permuted, the
#'   model is refitted with the same randomized-solver seed, and the permuted
#'   test-set `Q2Y` path is compared with the observed path.
#' @param times Number of requested permutations. For each component, `pls()`
#'   returns `(b + 1) / (B + 1)`, where `b` is the number of successful null
#'   fits with `Q2Y` at least as large as observed and `B` is the number of
#'   successful null fits. Failed fits are excluded from `B` and reported.
#' @param backend Implementation backend: \code{cpu} for compiled CPU,
#'   \code{cuda} for CUDA-native fitting, or \code{metal} for operation-split
#'   float32 CPU/Metal fitting on Apple silicon. When omitted,
#'   `options(backend = ...)`
#'   defines the session default. For CPU execution, `options(cores = n)`
#'   requests `n` BLAS/OpenMP threads. Selecting unavailable CUDA or Metal
#'   support raises an error without a CPU fallback.
#' @param north Number of orthogonal components removed by OPLS.
#'   The predictive count `ncomp` must fit within the predictor rank remaining
#'   after filtering. Requests above this bound raise an error, including
#'   within cross-validation folds, rather than fitting a numerical null
#'   direction. Orthogonal components are not counted as predictive components.
#' @param kernel Kernel type for kernel PLS: \code{linear}, \code{rbf}, or
#' \code{poly}.
#' @param gamma Kernel scale. Defaults internally to `1 / ncol(Xtrain)`.
#' @param degree Polynomial kernel degree.
#' @param coef0 Polynomial kernel offset.
#' @param ... Optional SVD tuning controls forwarded to the selected backend.
#'   Use the same compact names documented in [fastsvd()], such as
#'   `oversample`, `power`, and `seed`.
#' @return A `fastPLS` object. The object is a list whose fields depend on the
#'   selected method, backend, classifier, and whether test data or optional
#'   summaries were requested. `metrics` is a list of complete `evaluate()`
#'   results, organized as `metrics$fitted` and `metrics$test`, with one element
#'   per requested component count. Common fields are:
#'
#'   * `P`: predictor loadings, with one column per latent component.
#'   * `Q`: response loadings or response-side latent coefficients.
#'   * `R`: predictor weights/rotations used to project new samples into the PLS
#'     latent space.
#'   * `Ttrain`: training latent scores. Resident CUDA fits materialize this
#'     host-side matrix only when `fit = TRUE`; with `fit = FALSE`, CUDA scores
#'     remain on the device only as long as required to prepare compact
#'     prediction or LDA state. Operation-split Metal fits use host-accessible
#'     scores for the CPU component of the calculation. Compiled
#'     cross-validation continues to return its documented score outputs.
#'   * `C_latent`, `W_latent`: low-rank latent prediction factors used by
#'     PLS-SVD-style compact prediction when a full coefficient array is
#'     avoided.
#'   * `B`: regression coefficient matrix or coefficient array, when stored.
#'     For vector-valued `ncomp`, a three-dimensional array may contain the
#'     coefficient path for all requested component counts.
#'   * `mX`, `vX`: training predictor centering and scaling values. `vX` is one
#'     when no scaling is applied.
#'   * `mY`: response centering values for regression or dummy-coded PLS-DA.
#'   * `lev`: factor levels used for classification.
#'   * `Yfit`: fitted training responses or fitted class labels, returned when
#'     `fit = TRUE`.
#'   * `R2Y`: training-set coefficient of determination path when `fit = TRUE`;
#'     otherwise `NA` placeholders may be returned for compatibility. Elements
#'     are named by component count, for example `"ncomp=2"`. For PLS-DA this
#'     is a dummy-response quantity, not classification accuracy.
#'   * `Ypred`: predictions for `Xtest`, returned only when `Xtest` is supplied
#'     to `pls()`. For classification this contains predicted factor labels; for
#'     regression it contains numeric predictions.
#'   * `Ypred_index`: integer class indices for classification predictions, when
#'     available.
#'   * `Ttest`: test-set latent scores, returned when `proj = TRUE`.
#'   * `Q2Y`: independent-test Q2 whose denominator centers each response on
#'     its corresponding training-response mean before aggregating sums of
#'     squares. For factor `Ytest`, this is dummy-response PLS-DA Q2 relative
#'     to training class proportions, not classification accuracy. It is
#'     returned when response scores are available. Elements are named by
#'     component count.
#'   * `accuracy`: decoded-label accuracy for factor `Ytest`, returned when
#'     classification predictions are available. Elements are named by component
#'     count.
#'   * `metrics`: complete `evaluate()` outputs. `definitions` records the exact
#'     R2Y and Q2Y denominator conventions. `fitted` evaluates `Yfit`
#'     against `Ytrain`; `test` evaluates `Ypred` against `Ytest`. For
#'     multivariate regression, response-wise metrics are included only when
#'     `bycol = TRUE`. `metrics$permutation` stores permutation metrics and
#'     p-values when `perm.test = TRUE`.
#'   * `pval`: corrected Monte Carlo permutation-test p-values by component,
#'     returned when `perm.test = TRUE`.
#'   * `permutation`: long-format permutation table, returned when
#'     `perm.test = TRUE`, with observed and permuted `R2`/`Q2` values and the
#'     permutation correlation used by `plot.permutation()`.
#'   * `permutation_unit`, `permutation_group_sizes_preserved`,
#'     `permutation_class_frequencies_preserved`, `permutation_folds`,
#'     `permutation_solver_seed`, `permutation_requested`,
#'     `permutation_completed`, `permutation_failed`, and `permutation_errors`:
#'     the permutation contract and null-fit audit.
#'   * `variance`, `variance_explained`, `cumulative_variance_explained`,
#'     `variance_total`, `variance_basis`: predictor-space variance summaries
#'     returned when `return_variance = TRUE`.
#'   * `x_variance`, `x_variance_explained`,
#'     `x_cumulative_variance_explained`, `x_variance_total`: aliases of the
#'     predictor-space variance summaries.
#'   * `inner_model`: fitted inner PLS model used by OPLS.
#'   * `W_orth`, `P_orth`, `north`, `opls_engine`, `xprod_mode`,
#'     `gpu_resident`: OPLS-specific orthogonal-component and backend metadata.
#'   * `kernel`, `kernel_engine`, `kernel_linear_direct`: kernelPLS-specific
#'     kernel settings and execution metadata.
#'   * `diagnostics`: numerical solver diagnostics. For rSVD this records the
#'     structural-check status, finiteness, requested and effective component
#'     counts and randomized controls. CPU float64 rSVD fits also record each
#'     case-specific residual audit, strengthened retry, and deterministic
#'     recovery. Panel evidence is reported separately and is not interpreted
#'     as general-use certification. SIMPLS-family fits additionally record
#'     whether the active approximate route uses a component-wise oversampled
#'     sketch or an eligible CPU/CUDA/Metal candidate block, together with the active execution
#'     optimizations.
#'
#'   Function settings and backend bookkeeping, such as the component grid and
#'   resolved classifier backend, are retained internally for prediction and
#'   plotting but are not shown as public output fields.
#' @examples
#' X <- as.matrix(mtcars[, c("disp", "hp", "wt", "qsec")])
#' y <- mtcars$mpg
#' fit <- pls(X, y,
#'     ncomp = 2, method = "simpls", backend = "cpu",
#'     svd.method = "rsvd", return_variance = FALSE
#' )
#' head(predict(fit, X)$Ypred)
#'
#' cv <- pls.single.cv(X, y,
#'     ncomp = seq_len(2), kfold = 3, method = "simpls",
#'     backend = "cpu", svd.method = "rsvd", seed = 1
#' )
#' fit_cv <- pls(cv, Xtest = X, return_variance = FALSE)
#' cv$best_ncomp
#' head(fit_cv$Ypred)
#' @export
pls <- function(Xtrain, Ytrain, Xtest = NULL, Ytest = NULL, ncomp = 2,
    scaling = c("centering",
        "autoscaling", "none"), method = c("simpls", "plssvd", "opls",
        "kernelpls"),
    svd.method = "rsvd", classifier = c("argmax", "lda"),
    lda_ridge = NULL,
    fit = FALSE, bycol = FALSE, return_variance = TRUE,
    return_loadings = FALSE,
    proj = FALSE, perm.test = FALSE, times = 100, backend = NULL, north = 1L,
    kernel = c("linear",
        "rbf", "poly"), gamma = NULL, degree = 3L, coef0 = 1, ...) {
    lda_ridge <- .resolve_deprecated_lda_ridge(lda_ridge, !missing(lda_ridge),
        "pls()")
    if (.is_single_pls_cv_result(Xtrain)) {
        cv_Xtest <- if (!missing(Ytrain) && missing(Xtest)) {
            Ytrain
        }
        else if (missing(Xtest)) {
            NULL
        }
        else {
            Xtest
        }
        return(.pls_from_single_cv_result(cv = Xtrain, Xtest = cv_Xtest,
            Ytest = if (missing(Ytest)) NULL else Ytest,
            fit = fit, bycol = bycol, return_variance = return_variance,
            return_loadings = return_loadings,
            proj = proj, perm.test = perm.test, times = times))
    }
    context <- .pls_context(Xtrain, Ytrain, Xtest, Ytest, method,
        if (missing(svd.method))
            NULL
        else svd.method, .svd_control_from_dots(list(...))$dots, backend,
        classifier,
        scaling)
    config <- list(ncomp = ncomp, lda_ridge = lda_ridge, fit = fit,
        bycol = bycol,
        return_variance = return_variance, return_loadings = return_loadings,
        proj = proj,
        perm.test = perm.test, times = times, north = north,
        kernel = match.arg(kernel),
        gamma = gamma, degree = degree, coef0 = coef0)
    model <- .pls_dispatch(context, config)
    .pls_finalize(model, context, config)
}

.cv_best_index <- function(metrics, selection_metric = "auto") {
    selection_metric <- .cv_normalize_selection_metric(selection_metric)
    values <- as.numeric(metrics$metric_value)
    metric_names <- tolower(as.character(metrics$metric_name))
    finite <- is.finite(values)
    if (!any(finite)) {
        return(1L)
    }
    if (!identical(selection_metric, "auto")) {
        target_names <- switch(
            selection_metric,
            accuracy = "accuracy",
            balanced_accuracy = "balanced_accuracy",
            r2 = c("r2", "q2"),
            q2 = c("q2", "r2"),
            rmsd = c("rmsd", "rmse"),
            character(0)
        )
        finite <- finite & metric_names %in% target_names
        if (!any(finite)) {
            available_metrics <- paste(unique(metric_names), collapse = ", ")
            message_format <- paste0(
                "selection_metric = '%s' is unavailable in these CV results. ",
                "Available metrics: %s."
            )
            stop(
                sprintf(
                    message_format,
                    selection_metric,
                    available_metrics
                ),
                call. = FALSE
            )
        }
    }
    loss_metric <- any(
        metric_names[finite] %in% c("rmsd", "rmse", "mae", "mse")
    )
    idx <- if (loss_metric) {
        which.min(ifelse(finite, values, Inf))
    } else {
        which.max(ifelse(finite, values, -Inf))
    }
    as.integer(idx[1L])
}

.cv_numeric_metric_values <- function(metrics) {
    values <- as.numeric(metrics$metric_value)
    names(values) <- as.character(metrics$metric_name)
    values
}

.cv_extract_prediction_at <- function(cv_res, idx) {
    class_pred <- cv_res[["class_pred", exact = TRUE]]
    score_pred <- cv_res[["Ypred", exact = TRUE]]
    pred <- cv_res[["pred", exact = TRUE]]
    if (!is.null(class_pred)) {
        return(pred[[idx]])
    }
    if (!is.null(score_pred)) {
        return(score_pred[, , idx, drop = FALSE])
    }
    if (is.null(pred) || length(pred) < idx) {
        return(NULL)
    }
    pred[[idx]]
}

.cv_metric_name_at <- function(metrics, idx) {
    as.character(metrics$metric_name[[idx]])
}

.cv_metric_id <- function(metric, classification) {
    metric <- .cv_normalize_selection_metric(metric)
    if (isTRUE(classification)) {
        return(1L)
    }
    switch(metric, r2 = 2L, q2 = 3L, rmsd = 4L, auto = 4L, 4L)
}

.cv_grid_choice_values <- function(
    value,
    missing_arg,
    choices,
    default = choices[[1L]],
    name = "argument",
    normalizer = NULL
) {
    if (isTRUE(missing_arg)) {
        value <- default
    }
    value <- as.character(value)
    if (!length(value)) {
        stop(name, " must contain at least one value.", call. = FALSE)
    }
    if (!is.null(normalizer)) {
        value <- vapply(value, normalizer, character(1L), USE.NAMES = FALSE)
    }
    bad <- setdiff(value, choices)
    if (length(bad)) {
        choices_text <- paste(choices, collapse = ", ")
        bad_text <- paste(bad, collapse = ", ")
        stop(
            sprintf(
                "%s must use values from: %s. Invalid: %s.",
                name,
                choices_text,
                bad_text
            ),
            call. = FALSE
        )
    }
    as.list(unique(value))
}

.cv_grid_scalar_values <- function(
    value,
    missing_arg = FALSE,
    default = NULL,
    name = "argument",
    cast = identity,
    allow_null = TRUE
) {
    if (isTRUE(missing_arg)) {
        value <- default
    }
    if (is.null(value)) {
        if (!allow_null) {
            stop(name, " cannot be NULL.", call. = FALSE)
        }
        return(list(NULL))
    }
    if (!length(value)) {
        stop(name, " must contain at least one value.", call. = FALSE)
    }
    value <- cast(value)
    as.list(unique(value))
}

.cv_expand_prediction_grid <- function(params) {
    lens <- vapply(params, length, integer(1L))
    if (!length(lens) || any(lens < 1L)) {
        return(list(list()))
    }
    idx <- expand.grid(
        lapply(lens, seq_len),
        KEEP.OUT.ATTRS = FALSE,
        stringsAsFactors = FALSE
    )
    names(idx) <- names(params)
    lapply(seq_len(nrow(idx)), function(i) {
        cfg <- vector("list", length(params))
        names(cfg) <- names(params)
        for (nm in names(params)) {
            cfg[[nm]] <- params[[nm]][[idx[[nm]][[i]]]]
        }
        cfg
    })
}

.cv_normalize_svd_grid_dots <- function(dots, context) {
    if (!is.list(dots)) {
        dots <- list()
    }
    .normalize_svd_parameter_list(
        dots,
        accepted = names(.svd_control_defaults()),
        aliases = .svd_direct_aliases(),
        label = sprintf("... in %s", context)
    )
}

.cv_canonicalize_prediction_config <- function(cfg) {
    if (!identical(cfg$method, "kernelpls")) {
        cfg$kernel <- "linear"
        cfg["gamma"] <- list(NULL)
        cfg$degree <- 3L
        cfg$coef0 <- 1
    } else if (identical(cfg$kernel, "linear")) {
        cfg["gamma"] <- list(NULL)
        cfg$degree <- 3L
        cfg$coef0 <- 1
    } else if (identical(cfg$kernel, "rbf")) {
        cfg$degree <- 3L
        cfg$coef0 <- 1
    }
    if (!identical(cfg$method, "opls")) {
        cfg$north <- 1L
    }
    cfg
}

.cv_config_key <- function(cfg) {
    rec <- .cv_config_record(cfg)
    paste(
        names(rec),
        vapply(rec, function(x) as.character(x[[1L]]), character(1L)),
        sep = "=",
        collapse = "|"
    )
}

.cv_choice_grid <- function(scaling, scaling_missing, method, method_missing,
    backend,
    backend_missing, svd.method, svd_missing, kernel, kernel_missing,
    classifier,
    classifier_missing) {
    backend_default <- .fastpls_resolve_backend(NULL)
    svd_normalizer <- function(x) {
        x <- as.character(x)
        if (identical(x, "rsvd")) {
            x <- "cpu_rsvd"
        }
        x
    }
    list(scaling = .cv_grid_choice_values(scaling, scaling_missing,
        c("centering",
            "autoscaling", "none"), "centering", "scaling"),
    method = .cv_grid_choice_values(method,
        method_missing, c("simpls", "plssvd", "opls", "kernelpls"), "simpls",
        "method"),
    backend = .cv_grid_choice_values(backend, backend_missing, c("cpu",
        "cuda",
        "metal"), backend_default, "backend",
        .normalize_public_backend),
    svd.method = .cv_grid_choice_values(svd.method,
        svd_missing, "cpu_rsvd", "cpu_rsvd", "svd.method",
        svd_normalizer),
    kernel = .cv_grid_choice_values(kernel, kernel_missing, c("linear",
        "rbf",
        "poly"), "linear", "kernel"),
    classifier = .cv_grid_choice_values(classifier,
        classifier_missing, .classifier_public_choices, "argmax",
        "classifier",
        as.character))
}

.cv_scalar_grid <- function(north, gamma, degree, coef0, xprod) {
    list(
        north = .cv_grid_scalar_values(
            north,
            name = "north",
            cast = as.integer,
            allow_null = FALSE
        ),
        gamma = .cv_grid_scalar_values(gamma, name = "gamma"),
        degree = .cv_grid_scalar_values(
            degree,
            name = "degree",
            cast = as.integer,
            allow_null = FALSE
        ),
        coef0 = .cv_grid_scalar_values(
            coef0,
            name = "coef0",
            cast = as.numeric,
            allow_null = FALSE
        ),
        xprod = .cv_grid_scalar_values(xprod, name = "xprod")
    )
}

.cv_make_prediction_grid <- function(scaling, scaling_missing, method,
    method_missing,
    backend, backend_missing, svd.method, svd_missing, north, kernel,
    kernel_missing,
    gamma, degree, coef0, classifier, classifier_missing, xprod, dots = list(),
    context = "cross-validation") {
    dots <- .cv_normalize_svd_grid_dots(dots, context = context)
    dots <- dots[setdiff(names(dots), "seed")]
    dot_params <- lapply(names(dots), function(nm) {
        .cv_grid_scalar_values(dots[[nm]], name = nm, allow_null = FALSE)
    })
    names(dot_params) <- names(dots)
    params <- c(.cv_choice_grid(scaling, scaling_missing, method,
        method_missing,
        backend, backend_missing, svd.method, svd_missing, kernel,
        kernel_missing,
        classifier, classifier_missing), .cv_scalar_grid(north, gamma, degree,
        coef0, xprod), dot_params)
    configs <- .cv_expand_prediction_grid(params)
    dot_names <- names(dot_params)
    configs <- lapply(configs, function(cfg) {
        cfg$svd_dots <- cfg[dot_names]
        cfg[dot_names] <- NULL
        .cv_canonicalize_prediction_config(cfg)
    })
    configs[!duplicated(vapply(configs, .cv_config_key, character(1L)))]
}

.cv_require_backends_available <- function(configs, context) {
    backends <- unique(vapply(
        configs,
        function(config) config$backend,
        character(1L)
    ))
    for (backend in backends) {
        .fastpls_require_backend_available(backend, context)
    }
    invisible(configs)
}

.cv_config_record <- function(cfg) {
    svd_dots <- cfg$svd_dots
    cfg <- cfg[setdiff(names(cfg), "svd_dots")]
    if (length(svd_dots)) {
        cfg <- c(cfg, svd_dots)
    }
    as.data.frame(
        lapply(cfg, function(x) {
            if (is.null(x) || !length(x)) {
                return(NA)
            }
            x[[1L]]
        }),
        stringsAsFactors = FALSE,
        check.names = FALSE
    )
}

.cv_config_list <- function(cfg) {
    rec <- .cv_config_record(cfg)
    as.list(rec[1L, , drop = FALSE])
}

.cv_prune_config_for_output <- function(cfg) {
keep <- c("scaling", "method", "backend", "svd.method", "classifier", "xprod")
    if (identical(cfg$method, "opls")) {
        keep <- c(keep, "north")
    }
    if (identical(cfg$method, "kernelpls")) {
        keep <- c(keep, "kernel")
        if (identical(cfg$kernel, "rbf")) {
            keep <- c(keep, "gamma")
        } else if (identical(cfg$kernel, "poly")) {
            keep <- c(keep, "gamma", "degree", "coef0")
        }
    }
    keep <- intersect(unique(keep), names(cfg))
    out <- cfg[keep]
    if (!is.null(cfg$svd_dots) && length(cfg$svd_dots)) {
        out$svd_dots <- cfg$svd_dots
    }
    out
}

.cv_varied_parameter_names <- function(configs) {
    if (length(configs) <= 1L) {
        return(character(0))
    }
    recs <- do.call(rbind, lapply(configs, .cv_config_record))
    keep <- vapply(
        recs,
        function(x) {
            x <- x[!is.na(x)]
            length(unique(as.character(x))) > 1L
        },
        logical(1L)
    )
    names(recs)[keep]
}

.cv_selected_parameters <- function(cfg, configs, best_ncomp) {
    full <- .cv_config_list(cfg)
    varied <- .cv_varied_parameter_names(configs)
    selected <- full[intersect(varied, names(full))]
    c(list(ncomp = as.integer(best_ncomp[[1L]])), selected)
}

.cv_select_best_result_from_grid <- function(results, summaries, metrics,
    selection_metric = "auto") {
    ok <- vapply(results, function(x) is.list(x) && identical(x$status, "ok"),
        logical(1L))
    if (!any(ok)) {
        first_errors <- paste(head(summaries$error[!is.na(summaries$error)],
            5L),
        collapse = " | ")
        stop(
            sprintf(
                "All CV tuning configurations failed. Diagnostics: %s",
                first_errors
            ),
            call. = FALSE
        )
    }
    pick_df <- summaries[ok, , drop = FALSE]
    pick_data <- data.frame(
        metric_name = pick_df$best_metric_name,
        metric_value = pick_df$best_metric_value,
        stringsAsFactors = FALSE
    )
    pick_idx <- .cv_best_index(
        pick_data, selection_metric = selection_metric
    )
    best_grid_id <- pick_df$grid_id[[pick_idx]]
    best <- results[[best_grid_id]]
    best$tuning_results <- results
    best$tuning_summary <- summaries
    best$tuning_metrics <- metrics
    best$best_grid_id <- best_grid_id
    full_configs <- lapply(results, function(x) {
        x$tuning_config_full %||% x$tuning_config
    })
    selected_result <- results[[best_grid_id]]
    best_full_config <- selected_result$tuning_config_full %||%
        selected_result$tuning_config
    best$best_parameters <- .cv_selected_parameters(best_full_config,
        full_configs,
        best$best_ncomp)
    best$tuning_config_full <- NULL
    if (length(best$tuning_results)) {
        best$tuning_results <- lapply(best$tuning_results, function(x) {
            x$tuning_config_full <- NULL
            x
        })
    }
    best
}

.cv_value_or_default <- function(params, name, default) {
    if (!is.null(params) && name %in% names(params)) {
        value <- params[[name]]
        if (!is.null(value) && length(value) && !is.na(value[[1L]])) {
            return(value[[1L]])
        }
    }
    default
}

.cv_grid_arg_values <- function(configs, name) {
    vals <- lapply(configs, function(cfg) cfg[[name]])
    nonnull <- !vapply(vals, is.null, logical(1L))
    if (!any(nonnull)) {
        return(NULL)
    }
    unique(unlist(vals[nonnull], recursive = FALSE, use.names = FALSE))
}

.cv_grid_dot_values <- function(configs, name) {
    vals <- lapply(configs, function(cfg) cfg$svd_dots[[name]])
    nonnull <- !vapply(vals, is.null, logical(1L))
    if (!any(nonnull)) {
        return(NULL)
    }
    unique(unlist(vals[nonnull], recursive = FALSE, use.names = FALSE))
}

.single_cv_selection <- function(selection_metric, missing_metric, dots) {
    from_dots <- .cv_selection_metric_from_dots(dots)
    if (missing_metric) {
        return(from_dots)
    }
    list(
        metric = .cv_normalize_selection_metric(selection_metric),
        dots = from_dots$dots
    )
}

.single_cv_grid_call <- function(parameters, config, selection_metric) {
    arguments <- c(
        list(
            Xdata = parameters$Xdata,
            Ydata = parameters$Ydata,
            ncomp = parameters$ncomp,
            constrain = parameters$constrain,
            scaling = config$scaling,
            method = config$method,
            backend = config$backend,
            svd.method = config$svd.method,
            seed = parameters$seed,
            kfold = parameters$kfold,
            north = config$north,
            kernel = config$kernel,
            gamma = config$gamma,
            degree = config$degree,
            coef0 = config$coef0,
            classifier = config$classifier,
            fit = parameters$fit,
            bycol = parameters$bycol,
            xprod = config$xprod,
            selection_metric = selection_metric
        ),
        config$svd_dots
    )
    tryCatch(
        do.call(pls.single.cv, arguments),
        error = function(error) {
            list(
                status = "error",
                error = conditionMessage(error),
                tuning_config = config
            )
        }
    )
}

.single_cv_grid_record <- function(result, config, grid_id) {
    if (!identical(result$status, "error")) {
        result$cv_status <- result$status
        result$status <- "ok"
    }
    result <- .cv_drop_fit_data(result)
    result$tuning_config_full <- config
    result$tuning_config <- .cv_prune_config_for_output(config)
    ok <- identical(result$status, "ok")
    summary <- cbind(
        data.frame(
            grid_id = grid_id,
            status = if (ok) "ok" else "error",
            best_ncomp = if (ok && length(result$best_ncomp)) {
                result$best_ncomp[[1L]]
            } else {
                NA_integer_
            },
            best_metric_name = if (ok && length(result$best_metric_name)) {
                result$best_metric_name[[1L]]
            } else {
                NA_character_
            },
            best_metric_value = if (ok && length(result$best_metric_value)) {
                result$best_metric_value[[1L]]
            } else {
                NA_real_
            },
            error = if (ok) {
                NA_character_
            } else {
                result$error %||% "configuration failed"
            },
            stringsAsFactors = FALSE
        ),
        .cv_config_record(config)
    )
    list(result = result, summary = summary)
}

.single_cv_grid_metrics <- function(record, config, grid_id) {
    if (!identical(record$result$status, "ok")) {
        return(NULL)
    }
    metric <- record$result$selection_metrics
    metric$ncomp <- record$result$ncomp
    config_record <- .cv_config_record(config)
    cbind(
        data.frame(grid_id = grid_id, stringsAsFactors = FALSE),
        config_record[rep(1L, nrow(metric)), , drop = FALSE],
        metric
    )
}

.single_cv_run_grid <- function(grid, parameters, selection_metric) {
    records <- lapply(seq_along(grid), function(index) {
        config <- grid[[index]]
        record <- .single_cv_grid_record(
            .single_cv_grid_call(parameters, config, selection_metric),
            config,
            index
        )
        record$metrics <- .single_cv_grid_metrics(record, config, index)
        record
    })
    results <- lapply(records, function(x) x[["result"]])
    summaries <- do.call(rbind, lapply(records, function(x) x[["summary"]]))
    available <- Filter(
        Negate(is.null),
        lapply(records, function(x) x[["metrics"]])
    )
    metrics <- if (length(available)) {
        do.call(rbind, available)
    } else {
        data.frame()
    }
    best <- .cv_select_best_result_from_grid(
        results,
        summaries,
        metrics,
        selection_metric
    )
    .cv_attach_fit_data(best, parameters$Xdata, parameters$Ydata)
}

.single_cv_context <- function(Xdata, Ydata, constrain, config, seed,
    selection_metric) {
    backend <- .normalize_public_backend(config$backend)
    control <- .resolve_svd_control(
        svd.method = config$svd.method,
        dots = c(
            .svd_control_from_dots(config$svd_dots)$dots,
            list(seed = seed)
        ),
        context = "pls.single.cv()"
    )
    control <- .apply_pls_rsvd_controls(
        control,
        backend,
        "pls.single.cv()",
        config$method,
        is.factor(Ydata) || is.character(Ydata),
        Xdata,
        Ydata
    )
    control$svd.method <- match.arg(
        .normalize_svd_method(control$svd.method),
        c("cpu_rsvd")
    )
    float32 <- .has_float32_input(Xdata, Ydata)
    Xdata <- if (float32) {
        .as_float32_matrix(Xdata, "Xdata")
    } else {
        as.matrix(Xdata)
    }
    if (is.null(constrain)) {
        constrain <- seq_len(nrow(Xdata))
    }
    list(
        X = Xdata,
        Y = Ydata,
        constrain = constrain,
        config = config,
        backend = backend,
        backend_compiled = .compiled_backend(backend),
        control = control,
        float32 = float32,
        classification = is.factor(Ydata),
        selection_metric = selection_metric
    )
}

.single_cv_engine_arguments <- function(context, ncomp, kfold) {
    config <- context$config
    control <- context$control
    list(
        Xdata = context$X,
        Ydata = context$Y,
        constrain = context$constrain,
        ncomp = as.integer(ncomp),
        kfold = kfold,
        scaling = config$scaling,
        method = config$method,
        backend = context$backend,
        svd.method = control$svd.method,
        rsvd_oversample = control$rsvd_oversample,
        rsvd_power = control$rsvd_power,
        seed = control$seed,
        xprod = config$xprod,
        north = config$north,
        return_scores = TRUE,
        classifier = config$classifier,
        lda_ridge = .fixed_lda_relative_ridge,
        store_predictions = TRUE,
        selection_metric = context$selection_metric
    )
}

.single_cv_run_engine <- function(context, ncomp, kfold) {
    arguments <- .single_cv_engine_arguments(context, ncomp, kfold)
    if (context$backend %in% c("cuda", "metal") ||
        !identical(context$config$kernel, "linear")) {
        arguments$backend <- context$backend
        arguments$kernel <- context$config$kernel
        arguments$gamma <- context$config$gamma
        arguments$degree <- context$config$degree
        arguments$coef0 <- context$config$coef0
        return(do.call(.pls_cv_via_pls, arguments))
    }
    arguments$backend <- context$backend_compiled
    do.call(.pls_cv_compiled, arguments)
}

.single_cv_metric_paths <- function(result, context) {
    values <- as.numeric(result$metrics$metric_value)
    q2 <- result$Q2Y
    rmsd <- result$RMSD
    if (context$classification) {
        accuracy <- result$accuracy %||% values
    if (is.null(q2) || length(q2) != length(values) || all(!is.finite(q2))) {
            scores <- result$Yscore %||% result$Ypred
            q2 <- if (is.null(scores)) {
                rep(NA_real_, length(values))
            } else {
                .cv_classification_q2_path(
                    context$Y,
                    scores,
                    result$levels,
                    fold = result$fold
                )
            }
        }
        return(list(
            values = values,
            q2 = q2,
            rmsd = rmsd,
            accuracy = as.numeric(accuracy)
        ))
    }
    if (!is.null(result$Ypred) && !is.null(result$fold)) {
        slices <- dim(result$Ypred)[[3L]]
        q2 <- rmsd <- rep(NA_real_, slices)
        for (index in seq_len(slices)) {
            predicted <- result$Ypred[, , index, drop = TRUE]
            q2[[index]] <- .fastpls_fold_q2_path(
                context$Y,
                predicted,
                result$fold
            )[[1L]]
            rmsd[[index]] <- .cv_regression_q2_rmsd(
                context$Y,
                predicted,
                context$Y
            )$RMSD
        }
    }
    list(values = values, q2 = q2, rmsd = rmsd, accuracy = NULL)
}

.single_cv_attach_selection <- function(result, context, paths) {
    selection <- .cv_selection_metrics(
        result,
        context$Y,
        context$classification,
        context$selection_metric
    )
    index <- .cv_best_index(selection, context$selection_metric)
    selected <- as.numeric(selection$metric_value)
    result$best_ncomp <- as.integer(result$ncomp[[index]])
    result$best_index <- index
    result$selection_metric <- context$selection_metric
    result$selection_metrics <- selection
    result$selection_values <- selected
    result$best_metric_name <- .cv_metric_name_at(selection, index)
    result$best_metric_value <- selected[[index]]
    if (context$classification) {
        result$accuracy <- paths$accuracy
        if (identical(context$selection_metric, "balanced_accuracy")) {
            result$balanced_accuracy <- selected
        } else if (!is.null(result$pred)) {
            predictions <- if (is.list(result$pred)) {
                result$pred
            } else {
                list(result$pred)
            }
            result$balanced_accuracy <- vapply(
                predictions,
                .cv_balanced_accuracy,
                numeric(1L),
                observed = context$Y,
                levels = result$levels
            )
        }
    }
    result$Q2Y <- as.numeric(paths$q2)
    result$RMSD <- if (context$classification) {
        rep(NA_real_, length(paths$values))
    } else {
        as.numeric(paths$rmsd)
    }
    result
}

.single_cv_training_fit <- function(context, result, fit) {
    if (!isTRUE(fit)) {
        return(list(
            R2Y = rep(NA_real_, length(result$ncomp)),
            Yfit = NULL
        ))
    }
    config <- context$config
    control <- context$control
    .cv_training_fit_summary(
        Xdata = context$X,
        Ydata = context$Y,
        ncomp = as.integer(result$ncomp),
        scaling = config$scaling,
        method = config$method,
        backend = context$backend,
        svd.method = control$svd.method,
        rsvd_oversample = control$rsvd_oversample,
        rsvd_power = control$rsvd_power,
        svds_tol = control$svds_tol,
        seed = control$seed,
        north = config$north,
        kernel = config$kernel,
        gamma = config$gamma,
        degree = config$degree,
        coef0 = config$coef0
    )
}

.single_cv_finish <- function(result, context, grid, fit, bycol) {
    training <- .single_cv_training_fit(context, result, fit)
    result$R2Y <- training$R2Y
    result$Yfit <- training$Yfit
    result$Ypred_optim <- .cv_extract_prediction_at(
        result,
        result$best_index
    )
    result$tuning_config <- .cv_prune_config_for_output(context$config)
    result$best_parameters <- .cv_selected_parameters(
        context$config,
        grid,
        result$best_ncomp
    )
    result <- .fastpls_attach_single_cv_metrics(
        result,
        context$Y,
        fit,
        bycol
    )
    output <- .cv_attach_fit_data(result, context$X, context$Y)
    if (context$float32) {
        attr(output, "fastPLS_internal") <- list(
            precision = "float32",
            cv_engine = "float32_fold_pls",
            pls_method = context$config$method,
            backend = context$backend
        )
    }
    output
}

#' Single cross-validation for PLS component optimization
#'
#' Performs grouped k-fold or leave-one-out cross-validation over candidate
#' component counts and, when vector-valued predictive arguments are supplied,
#' over a compact hyperparameter grid. The selection can be based on
#' cross-validated accuracy, balanced accuracy, R2, Q2, or RMSD.
#'
#' @inheritParams pls
#' @param Xdata Predictor matrix.
#' @param Ydata Response (numeric or factor).
#' @param constrain Optional grouping vector for grouped cross-validation. It
#'   must have one value per sample. Samples with the same value are assigned to
#'   the same fold, so all rows from the same patient, subject, batch, or
#'   technical replicate stay together in training or test data. When `NULL`,
#'   each sample is treated as its own group.
#' @param kfold Number of folds, or `"loocv"` for leave-one-out
#'   cross-validation. When `constrain` is supplied, LOOCV means
#'   leave-one-constraint-group-out: samples sharing the same constraint value
#'   are always held out together and are never split across training and test.
#' @param method One or more of \code{simpls}, \code{plssvd}, \code{opls}, or
#'   \code{kernelpls}. Multiple values are treated as a tuning grid.
#' @param backend Implementation backend: \code{cpu}, \code{cuda}, or
#'   \code{metal}. Multiple values are treated as a tuning grid. Metal requires
#'   float32 input and uses fixed CPU/Metal operation splitting with CPU fold
#'   orchestration and prediction. When omitted,
#'   `options(backend = ...)` defines the session default; `options(cores = n)`
#'   controls the CPU thread request. Every requested backend must be available;
#'   unavailable accelerator entries raise an error instead of using CPU.
#' @param seed Random seed used for fold assignment and randomized SVD steps.
#' @param gamma Kernel scale. Defaults internally to `1 / ncol(Xdata)`. For
#'   \code{method = "kernelpls"}, multiple values are treated as a tuning grid.
#' @param classifier Classification rule for factor responses: `"argmax"` or
#'   latent-space `"lda"`. Multiple values are treated as a tuning grid.
#' @param lda_ridge Deprecated compatibility argument. It is ignored and emits
#'   a warning when supplied. LDA regularization follows the fixed numerical
#'   fallback sequence documented in [pls()].
#' @param fit Fit one additional model on the full dataset and return its
#'   fitted values (`Yfit`) and training `R2Y` path. The default is `TRUE` for
#'   backward compatibility. Set to `FALSE` to skip this extra full-data fit;
#'   held-out cross-validated `Q2Y` and `RMSD` are still calculated.
#' @param bycol For matrix-valued regression responses, calculate response-wise
#'   metrics in the returned `metrics` list. The default `FALSE` returns only
#'   aggregate metrics.
#' @param selection_metric Metric used to select predictive settings from
#'   held-out folds. Use `"auto"` (accuracy for classification and RMSD for
#'   regression), `"accuracy"`, `"balanced_accuracy"`, `"r2"`, `"q2"`, or
#'   `"rmsd"`. For numeric responses, `"r2"` is calculated from held-out
#'   predictions relative to the mean of all observed responses, whereas
#'   `"q2"` uses the corresponding fold-training response mean. The
#'   full-data training `R2Y` returned when `fit = TRUE` is descriptive and is
#'   not used for component selection. Classification can use accuracy,
#'   balanced accuracy, or dummy-response Q2, but not training R2. Balanced
#'   accuracy is the unweighted mean of class-specific recalls and is
#'   appropriate when class frequencies are unequal.
#' @param xprod Use the matrix-free cross-product route where available.
#'   `NULL` applies fastPLS defaults.
#' @param ... Optional SVD tuning controls forwarded to the selected backend.
#'   Use the same compact names documented in [fastsvd()], such as
#'   `oversample` and `power`. Vector values are included in the tuning grid.
#' @details For LDA classification, each training fold uses only the classes
#'   represented in that fold and maps predictions back to the original factor
#'   levels. A class absent from a fold's training data cannot be predicted in
#'   that fold; its held-out observations remain in the reported metrics.
#' @return A list describing the cross-validation run and selected model.
#'   `metrics$cross_validated` contains complete `evaluate()` results for each
#'   requested component count and `metrics$fitted` contains the corresponding
#'   full-data fit results when `fit = TRUE`. `metrics$definitions` records the
#'   exact R2Y and Q2Y denominator conventions. Other fields are:
#'   \itemize{
#'   \item `best_ncomp`: number of components selected by the chosen metric.
#'   \item `best_index`: position of `best_ncomp` in the tested component grid.
#'   \item `selection_metric`: metric used for optimization. With `"auto"`,
#'   classification uses accuracy and regression uses the default prediction
#'   error rule.
#'   \item `best_metric_name` and `best_metric_value`: name and value of the
#'   metric at the selected component count.
#'   \item `Q2Y`: held-out cross-validated Q2; every held-out fold is centered
#'   on its corresponding fold-training response mean. For factor responses,
#'   this is dummy-response PLS-DA Q2 using fold-training class proportions and
#'   is not classification accuracy.
#'   \item `accuracy`: held-out decoded-label accuracy for factor responses.
#'   \item `balanced_accuracy`: held-out mean class recall for factor responses.
#'   \item `RMSD`: held-out root mean squared deviation for regression. It is
#'   `NA` for classification.
#'   \item `Yfit`: fitted values from the full-data model when `fit = TRUE`.
#'   \item `R2Y`: training-set explained-variance path from a model fitted on
#'   the full dataset when `fit = TRUE`; otherwise `NA`. For factor
#'   responses, this is calculated on the dummy-coded PLS-DA response scores,
#'   not on the decoded class labels.
#'   \item `fold`: fold assignment used for each sample.
#'   \item `pred`: decoded cross-validated predictions when predictions are
#'   stored.
#'   \item `Ypred`: raw prediction array when score predictions are stored.
#'   \item `metrics`: complete `evaluate()` outputs. `cross_validated` contains
#'   one result per requested component count from held-out predictions and
#'   `fitted` contains full-data fit results when `fit = TRUE`. For multivariate
#'   regression, response-wise metrics are included only when `bycol = TRUE`.
#'   \item `selection_metrics`: compact per-component metric table used for
#'   component selection by the CV backend.
#'   \item `best_parameters`: compact list containing only `ncomp` plus the
#'   arguments that were actually optimized, for example `classifier` when
#'   `classifier = c("argmax", "lda")`.
#'   \item `tuning_config`: relevant selected configuration used for the run.
#'   Irrelevant classifier- or method-specific defaults are omitted; for
#'   example, controls belonging to an unselected classifier are omitted.
#'   \item `tuning_summary` and `tuning_metrics`: tables for all tested
#'   configurations when more than one predictive configuration is supplied.
#'   \item The returned object can be passed as the first argument to [pls()] to
#'   refit the selected model on the full training data and predict new samples.
#'   }
#' @examples
#' idx <- c(seq_len(12), 51:62, 101:112)
#' X <- as.matrix(iris[idx, seq_len(4)])
#' y <- factor(iris[idx, 5])
#' opt <- pls.single.cv(X, y,
#'     ncomp = seq_len(2), kfold = 3, method = "simpls",
#'     backend = "cpu", svd.method = "rsvd", seed = 1
#' )
#' opt$best_ncomp
#' opt_kernel <- pls.single.cv(X, y,
#'     ncomp = seq_len(2), kfold = 3,
#'     method = "kernelpls", backend = "cpu",
#'     svd.method = "rsvd",
#'     kernel = c("linear", "rbf"),
#'     gamma = c(0.1, 1), seed = 1
#' )
#' opt_kernel$best_parameters
#' @export
pls.single.cv <- function(Xdata, Ydata, ncomp = 2, constrain = NULL,
    scaling = c("centering",
        "autoscaling", "none"), method = c("simpls", "plssvd", "opls",
        "kernelpls"),
    backend = NULL, svd.method = "rsvd", seed = 1L, kfold = 10,
    north = 1L,
    kernel = c("linear", "rbf", "poly"), gamma = NULL, degree = 3L, coef0 = 1,
    classifier = c("argmax", "lda"), lda_ridge = NULL, fit = TRUE,
    bycol = FALSE,
    xprod = NULL, selection_metric = "auto", ...) {
    .resolve_deprecated_lda_ridge(lda_ridge, !missing(lda_ridge),
        "pls.single.cv()")
    if (sum(is.na(Xdata)) > 0) {
        stop("Missing values are present")
    }
    selection <- .single_cv_selection(selection_metric,
        missing(selection_metric),
        list(...))
    grid <- .cv_make_prediction_grid(scaling, missing(scaling), method,
        missing(method),
        backend, missing(backend), svd.method, missing(svd.method), north,
        kernel,
        missing(kernel), gamma, degree, coef0, classifier, missing(classifier),
        xprod, selection$dots, "pls.single.cv()")
    .cv_require_backends_available(grid, "pls.single.cv()")
    parameters <- list(Xdata = Xdata, Ydata = Ydata, ncomp = ncomp,
        constrain = constrain,
        seed = seed, kfold = kfold, fit = fit, bycol = bycol)
    if (length(grid) > 1L) {
        return(.single_cv_run_grid(grid, parameters, selection$metric))
    }
    context <- .single_cv_context(Xdata, Ydata, constrain, grid[[1L]], seed,
        selection$metric)
    result <- .single_cv_run_engine(context, ncomp, kfold)
    paths <- .single_cv_metric_paths(result, context)
    result <- .single_cv_attach_selection(result, context, paths)
    .single_cv_finish(result, context, grid, fit, bycol)
}


.double_cv_grid_values <- function(grid) {
    names <- c(
        "scaling",
        "method",
        "backend",
        "svd.method",
        "north",
        "kernel",
        "gamma",
        "degree",
        "coef0",
        "classifier",
        "xprod"
    )
    values <- lapply(names, function(name) .cv_grid_arg_values(grid, name))
    names(values) <- names
    dot_names <- unique(unlist(
        lapply(grid, function(config) names(config$svd_dots)),
        use.names = FALSE
    ))
    values$svd_dots <- lapply(
        dot_names,
        function(name) .cv_grid_dot_values(grid, name)
    )
    names(values$svd_dots) <- dot_names
    values
}

.double_cv_control <- function(config, seed, classification, Xdata, Ydata) {
    control <- .resolve_svd_control(
        svd.method = config$svd.method,
        dots = c(
            .svd_control_from_dots(config$svd_dots)$dots,
            list(seed = seed)
        ),
        context = "pls.double.cv()"
    )
    control <- .apply_pls_rsvd_controls(
        control,
        config$backend,
        "pls.double.cv()",
        config$method,
        classification,
        Xdata,
        Ydata
    )
    control$svd.method <- match.arg(
        .normalize_svd_method(control$svd.method),
        c("cpu_rsvd")
    )
    control
}

.double_cv_response <- function(Ydata) {
    classification <- is.factor(Ydata)
    original <- Ydata
    if (classification) {
        list(
            classification = TRUE,
            original = original,
            data = Ydata,
            levels = levels(Ydata)
        )
    } else {
        list(
            classification = FALSE,
            original = original,
            data = as.matrix(Ydata),
            levels = NULL
        )
    }
}

.double_cv_context <- function(
    Xdata,
    Ydata,
    constrain,
    ncomp,
    grid,
    selection_metric,
    seed
) {
    base <- grid[[1L]]
    response <- .double_cv_response(Ydata)
    Xdata <- as.matrix(Xdata)
    list(
        X = Xdata,
        response = response,
        constrain = as.integer(as.factor(constrain)),
        ncomp = as.integer(ncomp),
        grid = grid,
        grid_values = .double_cv_grid_values(grid),
        base = base,
        selection_metric = selection_metric,
        control = .double_cv_control(
            base,
            seed,
            response$classification,
            Xdata,
            Ydata
        ),
        seed = seed,
        defaults = list(
            scaling = base$scaling,
            method = base$method,
            backend = base$backend,
            svd.method = base$svd.method,
            north = base$north,
            kernel = base$kernel,
            gamma = base$gamma,
            degree = base$degree,
            coef0 = base$coef0,
            classifier = base$classifier,
            xprod = base$xprod
        )
    )
}

.double_cv_run_state <- function(context, fold) {
    count <- length(unique(fold))
    response <- context$response
    list(
        fold = fold,
        best_comp = integer(count),
        inner = vector("list", count),
        parameters = vector("list", count),
        train_r2 = rep(NA_real_, count),
        q2 = rep(NA_real_, count),
        prediction = if (response$classification) {
            rep(NA_character_, nrow(context$X))
        } else {
            matrix(
                NA_real_,
                nrow(context$X),
                ncol(response$data)
            )
        }
    )
}

.double_cv_inner_arguments <- function(
    context,
    train,
    run_index,
    fold_index,
    config
) {
    grid <- context$grid_values
    c(
        list(
            Xdata = context$X[train, , drop = FALSE],
            Ydata = config$Ytrain,
            ncomp = context$ncomp,
            constrain = context$constrain[train],
            scaling = grid$scaling,
            method = grid$method,
            backend = grid$backend,
            svd.method = grid$svd.method,
            seed = as.integer(context$seed) + 1000L * run_index + fold_index,
            kfold = config$kfold_inner,
            north = grid$north,
            kernel = grid$kernel,
            gamma = grid$gamma,
            degree = grid$degree,
            coef0 = grid$coef0,
            classifier = grid$classifier,
            bycol = config$bycol,
            xprod = grid$xprod,
            selection_metric = context$selection_metric
        ),
        grid$svd_dots
    )
}

.double_cv_selected_config <- function(context, selected) {
    output <- lapply(names(context$defaults), function(name) {
        .cv_value_or_default(selected, name, context$defaults[[name]])
    })
    names(output) <- names(context$defaults)
    control <- context$control
    control_names <- c(
        "rsvd_oversample",
        "rsvd_power",
        "svds_tol"
    )
    for (name in control_names) {
        output[[name]] <- .cv_value_or_default(
            selected,
            name,
            control[[name]]
        )
    }
    output
}

.double_cv_outer_fit <- function(
    context,
    selected,
    train,
    test,
    ncomp,
    run_index,
    fold_index
) {
    config <- .double_cv_selected_config(context, selected)
    response <- context$response
    Ytrain <- if (response$classification) {
        response$original[train]
    } else {
        response$data[train, , drop = FALSE]
    }
    Ytest <- if (response$classification) {
        response$original[test]
    } else {
        response$data[test, , drop = FALSE]
    }
    pls(
        Xtrain = context$X[train, , drop = FALSE],
        Ytrain = Ytrain,
        Xtest = context$X[test, , drop = FALSE],
        Ytest = Ytest,
        ncomp = ncomp,
        scaling = config$scaling,
        method = config$method,
        svd.method = config$svd.method,
        rsvd_oversample = config$rsvd_oversample,
        rsvd_power = config$rsvd_power,
        seed = as.integer(context$seed) + 2000L * run_index + fold_index,
        fit = TRUE,
        proj = FALSE,
        backend = config$backend,
        north = config$north,
        kernel = config$kernel,
        gamma = config$gamma,
        degree = config$degree,
        coef0 = config$coef0,
        classifier = config$classifier
    )
}

.double_cv_update_fold <- function(state, context, fit, test, index) {
    if (!is.null(fit$R2Y) && length(fit$R2Y)) {
        state$train_r2[[index]] <- as.numeric(tail(fit$R2Y, 1L))
    }
    if (context$response$classification) {
        if (!is.null(fit$Q2Y) && length(fit$Q2Y)) {
            state$q2[[index]] <- as.numeric(tail(fit$Q2Y, 1L))
        }
        prediction <- if (is.data.frame(fit$Ypred)) {
            fit$Ypred[[1L]]
        } else {
            fit$Ypred
        }
        state$prediction[test] <- as.character(prediction)
    } else {
        prediction <- fit$Ypred
        if (length(dim(prediction)) == 3L) {
            prediction <- prediction[, , 1L, drop = TRUE]
        }
        state$prediction[test, ] <- as.matrix(prediction)
    }
    state
}

.double_cv_process_fold <- function(state, context, fold_value, index,
    run_index,
    config) {
    test <- which(state$fold == fold_value)
    train <- which(state$fold != fold_value)
    if (!length(test) || !length(train)) {
        return(state)
    }
    response <- context$response
    Ytrain <- if (response$classification) {
        response$original[train]
    }
    else {
        response$data[train, , drop = FALSE]
    }
    if (response$classification && length(unique(Ytrain)) < 2L) {
        state$prediction[test] <- names(which.max(table(Ytrain)))
        state$best_comp[[index]] <- min(context$ncomp)
        return(state)
    }
    inner <- do.call(pls.single.cv, .double_cv_inner_arguments(context, train,
        run_index, index, list(Ytrain = Ytrain,
            kfold_inner = config$kfold_inner,
            bycol = config$bycol)))
    state$best_comp[[index]] <- as.integer(inner$best_ncomp[[1L]])
    state$inner[[index]] <- inner
    state$parameters[[index]] <- inner$best_parameters
    fit <- .double_cv_outer_fit(context, inner$best_parameters, train, test,
        state$best_comp[[index]],
        run_index, index)
    .double_cv_update_fold(state, context, fit, test, index)
}

.double_cv_class_run <- function(state, context) {
    response <- context$response
    prediction <- factor(state$prediction, levels = response$levels)
    accuracy <- mean(
        as.character(prediction) == as.character(response$original),
        na.rm = TRUE
    )
    balanced <- .cv_balanced_accuracy(
        response$original,
        prediction,
        levels = response$levels
    )
metric_name <- if (identical(context$selection_metric, "balanced_accuracy")) {
        "balanced_accuracy"
    } else {
        "accuracy"
    }
    list(
        Ypred = prediction,
        pred = prediction,
        fold = state$fold + 1L,
        best_ncomp = state$best_comp,
        best_parameters = state$parameters,
        inner = state$inner,
        metric_name = metric_name,
        metric_value = if (metric_name == "balanced_accuracy") {
            balanced
        } else {
            accuracy
        },
        accuracy = accuracy,
        balanced_accuracy = balanced,
        Q2Y = if (any(is.finite(state$q2))) {
            mean(state$q2, na.rm = TRUE)
        } else {
            NA_real_
        },
        R2Y = if (any(is.finite(state$train_r2))) {
            mean(state$train_r2, na.rm = TRUE)
        } else {
            NA_real_
        },
        RMSD = NA_real_
    )
}

.double_cv_regression_run <- function(state, context) {
    response <- context$response$data
    selection <- if (context$selection_metric %in% c("r2", "q2", "rmsd")) {
        context$selection_metric
    } else {
        "auto"
    }
    q2 <- .fastpls_fold_q2_path(
        response,
        state$prediction,
        state$fold
    )[[1L]]
    metric <- if (identical(selection, "q2")) {
        list(metric_name = "q2", metric_value = q2)
    } else {
        .cv_metric_from_matrix(
            response,
            state$prediction,
            Ytrain = response,
            metric = selection
        )
    }
    list(
        Ypred = state$prediction,
        pred = state$prediction,
        fold = state$fold + 1L,
        best_ncomp = state$best_comp,
        best_parameters = state$parameters,
        inner = state$inner,
        metric_name = metric$metric_name,
        metric_value = metric$metric_value,
        Q2Y = q2,
        R2Y = if (any(is.finite(state$train_r2))) {
            mean(state$train_r2, na.rm = TRUE)
        } else {
            NA_real_
        },
        RMSD = .cv_regression_q2_rmsd(
            response,
            state$prediction,
            response
        )$RMSD
    )
}

.double_cv_run_once <- function(context, run_index, config) {
    fold <- .make_single_cv_folds(
        if (context$response$classification) {
            context$response$original
        } else {
            context$response$data
        },
        context$constrain,
        config$kfold_outer,
        as.integer(context$seed) + run_index - 1L
    )
    state <- .double_cv_run_state(context, fold)
    values <- sort(unique(fold))
    for (index in seq_along(values)) {
        state <- .double_cv_process_fold(
            state,
            context,
            values[[index]],
            index,
            run_index,
            config
        )
    }
    result <- if (context$response$classification) {
        .double_cv_class_run(state, context)
    } else {
        .double_cv_regression_run(state, context)
    }
    result$backend <- context$base$backend
    result$method <- context$base$method
    result
}

.double_cv_class_aggregate <- function(results, context) {
    levels <- context$response$levels
    votes <- matrix(0, nrow(context$X), length(levels))
    colnames(votes) <- levels
    for (result in results) {
        index <- match(as.character(result$Ypred), levels)
        valid <- is.finite(index)
        votes[cbind(which(valid), index[valid])] <- votes[cbind(which(valid),
            index[valid])] +
            1
    }
    index <- max.col(votes, ties.method = "first")
    index[rowSums(votes) <= 0] <- NA_integer_
    prediction <- factor(ifelse(is.na(index), NA, levels[index]),
        levels = levels)
    confusion <- table(prediction, factor(context$response$original,
        levels = levels))
    percent <- .fastpls_quiet(t(t(confusion) / colSums(confusion)) * 100)
    percent[!is.finite(percent)] <- 0
    count <- sum(diag(confusion))
    list(Ypred = prediction, vote_counts = votes, acc_tot = paste0(round(count,
        1), " (", 100 * round(count, 1) / nrow(context$X), "%)"),
    conf = matrix(paste0(round(confusion,
        1), " (", round(percent, 1), "%)"), ncol = length(levels),
    dimnames = list(levels,
        levels)), accuracy = vapply(results, function(x) x$accuracy,
        numeric(1L)),
    balanced_accuracy = vapply(results, function(x) x$balanced_accuracy,
        numeric(1L)))
}

.double_cv_result <- function(results, context, runn) {
    output <- list(results = results)
    if (context$response$classification) {
        output <- c(output, .double_cv_class_aggregate(results, context))
    } else {
        output$Ypred <- Reduce(
            "+",
            lapply(results, function(x) x$Ypred)
        ) /
            as.integer(runn)
    }
    output$Q2Y <- vapply(results, function(x) x$Q2Y, numeric(1L))
    output$R2Y <- vapply(results, function(x) x$R2Y, numeric(1L))
    output$RMSD <- vapply(results, function(x) x$RMSD, numeric(1L))
    output$metric_name <- vapply(
        results,
        function(x) x$metric_name,
        character(1L)
    )
    if (as.integer(runn) > 1L) {
        output$medianR2Y <- median(output$R2Y, na.rm = TRUE)
        output$CI95R2Y <- as.numeric(quantile(
            output$R2Y,
            c(0.025, 0.975),
            na.rm = TRUE
        ))
        output$medianQ2Y <- median(output$Q2Y, na.rm = TRUE)
        output$CI95Q2Y <- as.numeric(quantile(
            output$Q2Y,
            c(0.025, 0.975),
            na.rm = TRUE
        ))
        output$medianRMSD <- median(output$RMSD, na.rm = TRUE)
        output$CI95RMSD <- as.numeric(quantile(
            output$RMSD,
            c(0.025, 0.975),
            na.rm = TRUE
        ))
    }
    components <- unlist(lapply(results, function(x) x$best_ncomp))
    output$bcomp <- names(which.max(table(components)))
    output$backend <- context$base$backend
    output$method <- context$base$method
    output$selection_metric <- context$selection_metric
    output
}

.double_cv_metric_values <- function(object, metric) {
    values <- switch(
        metric,
        accuracy = object$accuracy,
        balanced_accuracy = object$balanced_accuracy,
        r2 = object$R2Y,
        q2 = object$Q2Y,
        rmsd = object$RMSD,
        NULL
    )
    if (is.null(values) || !length(values)) {
        stop(
            sprintf(
                "Permutation metric '%s' is unavailable.",
                metric
            ),
            call. = FALSE
        )
    }
    as.numeric(values)
}

.double_cv_permutation_call <- function(context, config, index, runn) {
    base <- context$base
    control <- context$control
    pls.double.cv(
        Xdata = context$X[index, , drop = FALSE],
        Ydata = context$response$original,
        ncomp = context$ncomp,
        constrain = context$constrain,
        scaling = base$scaling,
        method = base$method,
        backend = base$backend,
        svd.method = control$svd.method,
        rsvd_oversample = control$rsvd_oversample,
        rsvd_power = control$rsvd_power,
        seed = control$seed,
        perm.test = FALSE,
        runn = runn,
        kfold_inner = config$kfold_inner,
        kfold_outer = config$kfold_outer,
        north = base$north,
        kernel = base$kernel,
        gamma = base$gamma,
        degree = base$degree,
        coef0 = base$coef0,
        classifier = base$classifier,
        bycol = config$bycol,
        xprod = base$xprod,
        selection_metric = context$selection_metric
    )
}

.double_cv_permutation_contract <- function(
    result,
    context,
    metric,
    sampled,
    errors,
    times
) {
    observed <- median(
        .double_cv_metric_values(result, metric),
        na.rm = TRUE
    )
    result$permutation_metric <- metric
    result$permutation_observed <- observed
    result$permutation_sampled <- sampled
    result$permutation_unit <- if (
        length(unique(context$constrain)) == nrow(context$X)
    ) {
        "rows"
    } else {
        paste0(
            "constraint groups within equal-size exchangeability ",
            "strata"
        )
    }
    result$permutation_group_sizes_preserved <- TRUE
    result$permutation_class_frequencies_preserved <- TRUE
    result$permutation_folds <- "fixed across observed and null fits"
    result$permutation_solver_seed <- "fixed across observed and null fits"
    result$permutation_requested <- times
    result$permutation_completed <- sum(is.finite(sampled))
    result$permutation_failed <- times - result$permutation_completed
    result$permutation_errors <- errors
    if (identical(metric, "q2")) {
        result$Q2Ysampled <- sampled
    }
    result$p.value <- .fastpls_permutation_pvalue(
        sampled,
        observed,
        lower_tail = metric %in% c("rmsd", "rmse", "mae", "mse")
    )
    result
}

.double_cv_attach_permutation <- function(result, context, config, times,
    runn) {
    metric <- context$selection_metric
    if (identical(metric, "auto")) {
        metric <- if (context$response$classification)
            "accuracy"
        else "rmsd"
    }
    times <- as.integer(times)[1L]
    indices <- .fastpls_permutation_indices(context$constrain, times,
        as.integer(context$seed) +
            100000L)
    sampled <- rep(NA, times)
    errors <- rep(NA, times)
    for (index in seq_len(times)) {
        attempt <- tryCatch({
            permuted <- .double_cv_permutation_call(context, config,
                indices[[index]],
                runn)
            value <- median(.double_cv_metric_values(permuted, metric),
                na.rm = TRUE)
            if (!is.finite(value)) {
                stop("Permutation fit returned no finite metric.")
            }
            list(value = value, error = NA)
        }, error = function(error) {
            list(value = NA, error = conditionMessage(error))
        })
        sampled[[index]] <- attempt$value
        errors[[index]] <- attempt$error
    }
    .double_cv_permutation_contract(result, context, metric, sampled, errors,
        times)
}

#' Nested cross-validation for PLS
#'
#' Performs nested grouped cross-validation with an outer loop for unbiased
#' performance estimation and an inner loop for component and hyperparameter
#' selection. Constraint groups are respected in both loops so related samples
#' remain in the same fold.
#'
#' @inheritParams pls
#' @param Xdata Predictor matrix.
#' @param Ydata Response (numeric or factor).
#' @param constrain Grouping vector for grouped cross-validation. It must have
#'   one value per sample. Samples with the same value are assigned to the same
#'   fold, so all rows from the same patient, subject, batch, or technical
#'   replicate stay together in training or test data. The default
#'   `seq_len(nrow(Xdata))` treats every sample as an independent group.
#' @param runn Number of repeated runs.
#' @param kfold_inner Inner-fold count, or `"loocv"` to leave out one
#'   constraint group at a time inside each outer training set.
#' @param kfold_outer Outer-fold count, or `"loocv"` to leave out one
#'   constraint group at a time in the outer loop. In both loops, samples
#'   sharing the same constraint value are never split across training and test.
#' @param method One or more of \code{simpls}, \code{plssvd}, \code{opls}, or
#'   \code{kernelpls}. Multiple values are tuned in the inner loop.
#' @param backend Implementation backend: \code{cpu}, \code{cuda}, or
#'   \code{metal}. Multiple values are tuned in the inner loop. Metal requires
#'   float32 input and Apple Metal. Fold orchestration remains on the host;
#'   every fold uses the same fixed operation split as `pls()`, with
#'   training-sample products on Metal and reduced factorizations, sequential
#'   component updates, LDA, and prediction on the CPU. When omitted,
#'   `options(backend = ...)` defines the session default; `options(cores = n)`
#'   controls the CPU thread request. Every requested backend must be available;
#'   unavailable accelerator entries raise an error instead of using CPU.
#' @param seed Random seed used for outer/inner fold assignment and randomized
#'   SVD steps.
#' @param gamma Kernel scale. Defaults internally to `1 / ncol(Xdata)`. For
#'   \code{method = "kernelpls"}, multiple values are tuned in the inner loop.
#' @param xprod Use the matrix-free cross-product route where available for
#'   inner component optimization. `NULL` applies fastPLS defaults.
#' @param bycol For matrix-valued regression responses, calculate response-wise
#'   metrics in the returned `metrics` list. The default `FALSE` returns only
#'   aggregate metrics.
#' @param selection_metric Metric used by inner CV and by the permutation test.
#'   Use `"auto"` (accuracy for classification and RMSD for regression),
#'   `"accuracy"`, `"balanced_accuracy"`, `"r2"`, `"q2"`, or `"rmsd"`.
#'   For numeric responses, `"r2"` is an observed-mean held-out endpoint and
#'   `"q2"` uses fold-training response means. Training `R2Y` is descriptive
#'   and is never optimized. Classification can use accuracy, balanced
#'   accuracy, or dummy-response Q2, but not training R2. Balanced accuracy
#'   gives each observed class equal weight.
#' @param perm.test Run a nested-CV permutation test. Independent rows are
#'   permuted individually. With repeated `constrain` values, complete blocks
#'   are exchanged only among groups with the same number of rows, preserving
#'   group sizes, within-group response structure, and class frequencies.
#'   Outer/inner folds and randomized-solver seeds are fixed across observed
#'   and null fits. The statistic is the median outer-CV value of the metric
#'   used for inner model selection.
#' @param times Number of requested permutations. The corrected Monte Carlo
#'   p-value is `(b + 1) / (B + 1)`, using the upper tail for metrics where
#'   larger is better and the lower tail for losses such as RMSD. `B` counts
#'   successful null fits only; failed fits are reported. A grouped test
#'   requires at least two constraint groups of equal size.
#' @param ... Optional SVD tuning controls forwarded to the selected backend.
#'   Use the same compact names documented in [fastsvd()], such as
#'   `oversample` and `power`. Vector values are tuned in the inner loop.
#' @details For LDA classification, each inner and outer training fold uses
#'   only the classes represented in that fold and maps predictions back to the
#'   original factor levels. A class absent from a fold's training data cannot
#'   be predicted in that fold; its held-out observations remain in the
#'   reported metrics.
#' @return A list with the following elements. `metrics$cross_validated`
#'   contains one complete `evaluate()` result per repeated outer-CV run, and
#'   `metrics$aggregate` evaluates the final vote-aggregated or averaged
#'   prediction. `metrics$definitions` records the exact R2Y and Q2Y
#'   denominator conventions.
#'
#'   * `results`: list with one element per repeated run. Each run stores
#'     `Ypred`/`pred`, the outer `fold` assignment, `best_ncomp` selected in
#'     each outer fold, fold-level `best_parameters`, the complete inner-CV
#'     objects in `inner`, run-level `metric_name` and `metric_value`, and the
#'     default `backend` and `method`.
#'   * `Ypred`: final cross-validated predictions. For classification, repeated
#'     runs are combined by voting; for regression, numeric predictions are
#'     averaged across runs.
#'   * `Q2Y`: one outer cross-validated Q2 value per repeated run. Numeric
#'     responses use each outer fold's training-response mean. For factor
#'     responses this is the mean outer-fold dummy-response PLS-DA Q2 using
#'     outer-training class proportions and is not classification accuracy.
#'   * `R2Y`: one training-fit R2 value per repeated run, averaged across the
#'     selected outer-fold models.
#'   * `RMSD`: one held-out RMSD value per repeated run for numeric responses;
#'     `NA` for classification.
#'   * `metric_name`: held-out metric used for each repeated run.
#'   * `bcomp`: most frequently selected component count across outer folds and
#'     repeated runs.
#'   * `backend`, `method`: default backend and PLS method supplied to the call.
#'      If vector-valued methods or backends are tuned, selected fold-level
#' values
#'     are stored in `results[[run]]$best_parameters`.
#'   * `selection_metric`: criterion used by the inner CV loop.
#'   * `acc_tot`: classification-only text summary of correctly classified
#'     samples and percentage accuracy.
#'   * `conf`: classification-only confusion matrix printed as counts and
#'     column percentages.
#'   * `vote_counts`: classification-only vote-count matrix with one row per
#'     sample and one column per class.
#'   * `accuracy`: classification-only decoded-label accuracy, one value per
#'     repeated run.
#'    * `balanced_accuracy`: classification-only unweighted mean of class
#' recalls,
#'     one value per repeated run.
#'   * `medianR2Y`, `CI95R2Y`, `medianQ2Y`, `CI95Q2Y`, `medianRMSD`,
#'     `CI95RMSD`: repeated-run summaries returned only when `runn > 1`.
#'   * `permutation_metric`, `permutation_observed`, and `permutation_sampled`:
#'     metric name, observed median, and permuted medians returned when
#'     `perm.test = TRUE`. `Q2Ysampled` is retained only when Q2 is the selected
#'     permutation metric.
#'   * `p.value`: permutation-test p-value returned when `perm.test = TRUE`.
#'   * `permutation_unit`, `permutation_group_sizes_preserved`,
#'     `permutation_class_frequencies_preserved`, `permutation_folds`,
#'     `permutation_solver_seed`, `permutation_requested`,
#'     `permutation_completed`, `permutation_failed`, and `permutation_errors`:
#'     the exchangeability contract and null-fit audit.
#' @examples
#' idx <- c(seq_len(10), 51:60, 101:110)
#' X <- as.matrix(iris[idx, seq_len(4)])
#' y <- factor(iris[idx, 5])
#' dcv <- pls.double.cv(X, y,
#'     ncomp = seq_len(2), runn = 1, kfold_inner = 2,
#'     kfold_outer = 2, method = "simpls", backend = "cpu",
#'     svd.method = "rsvd", seed = 1
#' )
#' names(dcv)
#' @export
pls.double.cv <- function(Xdata, Ydata, ncomp = 2,
    constrain = seq_len(nrow(Xdata)),
    scaling = c("centering", "autoscaling", "none"), method = c("simpls",
        "plssvd",
        "opls", "kernelpls"), backend = NULL, svd.method = "rsvd",
    seed = 1L, perm.test = FALSE, times = 100, runn = 1, kfold_inner = 10,
    kfold_outer = 10,
    north = 1L, kernel = c("linear", "rbf", "poly"), gamma = NULL, degree = 3L,
    coef0 = 1, classifier = c("argmax", "lda"), lda_ridge = NULL,
    bycol = FALSE,
    xprod = NULL, selection_metric = "auto", ...) {
    .resolve_deprecated_lda_ridge(lda_ridge, !missing(lda_ridge),
        "pls.double.cv()")
    if (sum(is.na(Xdata)) > 0) {
        stop("Missing values are present")
    }
    selection <- .single_cv_selection(selection_metric,
        missing(selection_metric),
        list(...))
    grid <- .cv_make_prediction_grid(scaling, missing(scaling), method,
        missing(method),
        backend, missing(backend), svd.method, missing(svd.method), north,
        kernel,
        missing(kernel), gamma, degree, coef0, classifier, missing(classifier),
        xprod, selection$dots, "pls.double.cv()")
    .cv_require_backends_available(grid, "pls.double.cv()")
    context <- .double_cv_context(Xdata, Ydata, constrain, ncomp, grid,
        selection$metric,
        seed)
    config <- list(kfold_inner = kfold_inner, kfold_outer = kfold_outer,
        bycol = bycol)
    results <- lapply(seq_len(as.integer(runn)), function(index) {
        .double_cv_run_once(context, index, config)
    })
    result <- .double_cv_result(results, context, runn)
    if (perm.test) {
        result <- .double_cv_attach_permutation(result, context, config, times,
            runn)
    }
    .fastpls_attach_double_cv_metrics(result, context$response$original, bycol)
}


.evaluate_is_onehot <- function(x) {
    is.matrix(x) &&
        is.numeric(x) &&
        ncol(x) > 1L &&
        all(is.finite(x) | is.na(x)) &&
        all(abs(rowSums(x, na.rm = TRUE) - 1) < 1e-8, na.rm = TRUE) &&
        all(x[!is.na(x)] %in% c(0, 1))
}

.evaluate_class_labels <- function(x, levels_ref = NULL) {
    if (is.factor(x) || is.character(x)) {
        return(as.character(x))
    }
    if (!is.matrix(x) && !is.data.frame(x)) {
        return(as.character(x))
    }
    x <- as.matrix(x)
    if (!is.numeric(x)) {
        return(as.character(x[, 1L]))
    }
    labels <- colnames(x)
    if (is.null(labels)) {
        labels <- levels_ref %||% as.character(seq_len(ncol(x)))
    }
    labels[max.col(x, ties.method = "first")]
}

.evaluate_class_inputs <- function(observed, predicted, na.rm) {
    levels_ref <- if (is.factor(observed)) levels(observed) else NULL
    if (is.matrix(observed) && !is.null(colnames(observed))) {
        levels_ref <- colnames(observed)
    }
    observed_labels <- .evaluate_class_labels(observed, levels_ref)
    predicted_labels <- .evaluate_class_labels(predicted, levels_ref)
    levels_all <- unique(c(levels_ref, observed_labels, predicted_labels))
    levels_all <- levels_all[!is.na(levels_all)]
    if (!length(levels_all)) {
        stop("No valid class labels were found.", call. = FALSE)
    }
    observed_factor <- factor(observed_labels, levels = levels_all)
    predicted_factor <- factor(predicted_labels, levels = levels_all)
    keep <- rep(TRUE, length(observed_factor))
    if (na.rm) {
        keep <- !is.na(observed_factor) & !is.na(predicted_factor)
        observed_factor <- observed_factor[keep]
        predicted_factor <- predicted_factor[keep]
    }
    if (length(observed_factor) != length(predicted_factor)) {
        stop(
            "observed and predicted must have the same number of samples.",
            call. = FALSE
        )
    }
    list(
        observed = observed_factor,
        predicted = predicted_factor,
        levels = levels_ref,
        keep = keep
    )
}

.evaluate_class_summary <- function(confusion) {
    true_positive <- diag(confusion)
    support <- colSums(confusion)
    predicted_support <- rowSums(confusion)
    recall <- true_positive / support
    precision <- true_positive / predicted_support
    recall[!is.finite(recall)] <- NA_real_
    precision[!is.finite(precision)] <- NA_real_
    f1 <- 2 * precision * recall / (precision + recall)
    f1[!is.finite(f1)] <- NA_real_
    n <- sum(confusion)
    accuracy <- if (n > 0) sum(true_positive) / n else NA_real_
    null_rate <- if (n > 0) max(support) / n else NA_real_
    lift <- if (is.finite(null_rate) && null_rate > 0) {
        accuracy / null_rate
    } else {
        NA_real_
    }
    expected <- if (n > 0) {
        sum(rowSums(confusion) * support) / n^2
    } else {
        NA_real_
    }
    kappa <- if (is.finite(expected) && expected < 1) {
        (accuracy - expected) / (1 - expected)
    } else {
        NA_real_
    }
    list(
        n = n,
        true_positive = true_positive,
        support = support,
        recall = recall,
        precision = precision,
        f1 = f1,
        accuracy = accuracy,
        null_rate = null_rate,
        lift = lift,
        kappa = kappa
    )
}

.evaluate_topk_one <- function(score, observed, labels, k) {
    k <- min(max(1L, k), ncol(score))
    correct <- vapply(
        seq_len(nrow(score)),
        function(i) {
            ranked <- order(score[i, ], decreasing = TRUE)[seq_len(k)]
            observed[[i]] %in% labels[ranked]
        },
        logical(1L)
    )
    mean(correct, na.rm = TRUE)
}

.evaluate_topk <- function(observed, predicted, inputs, top_k, na.rm) {
    is_score <- is.matrix(predicted) || is.data.frame(predicted)
    if (!is_score || !is.numeric(as.matrix(predicted))) {
        return(NULL)
    }
    score <- as.matrix(predicted)
    labels <- colnames(score)
    if (is.null(labels)) {
        labels <- inputs$levels %||% as.character(seq_len(ncol(score)))
    }
    observed_labels <- .evaluate_class_labels(observed, labels)
    if (na.rm) {
        observed_labels <- observed_labels[inputs$keep]
        score <- score[inputs$keep, , drop = FALSE]
    }
    top_k <- as.integer(top_k)
    data.frame(
        k = top_k,
        accuracy = vapply(
            top_k,
            .evaluate_topk_one,
            numeric(1L),
            score = score,
            observed = observed_labels,
            labels = labels
        )
    )
}

.evaluate_classification <- function(observed, predicted, top_k, na.rm) {
    inputs <- .evaluate_class_inputs(observed, predicted, na.rm)
    confusion <- table(
        predicted = inputs$predicted,
        observed = inputs$observed
    )
    summary <- .evaluate_class_summary(confusion)
    metrics <- data.frame(
        n = as.integer(summary$n),
        accuracy = summary$accuracy,
        no_information_rate = summary$null_rate,
        lift_accuracy = summary$lift,
        balanced_accuracy = mean(summary$recall, na.rm = TRUE),
        macro_precision = mean(summary$precision, na.rm = TRUE),
        macro_recall = mean(summary$recall, na.rm = TRUE),
        macro_f1 = mean(summary$f1, na.rm = TRUE),
        kappa = summary$kappa,
        stringsAsFactors = FALSE
    )
    per_class <- data.frame(
        class = names(summary$true_positive),
        support = as.integer(summary$support),
        precision = as.numeric(summary$precision),
        recall = as.numeric(summary$recall),
        f1 = as.numeric(summary$f1),
        stringsAsFactors = FALSE
    )
    list(
        task = "classification",
        metrics = metrics,
        metric_definitions = list(
            accuracy = "Proportion of observed labels predicted correctly.",
            balanced_accuracy = "Unweighted mean of class-specific recalls."
        ),
        per_class = per_class,
        confusion = confusion,
        topk = .evaluate_topk(observed, predicted, inputs, top_k, na.rm)
    )
}

.evaluate_regression_inputs <- function(observed, predicted, ytrain) {
    observed <- as.matrix(observed)
    predicted <- as.matrix(predicted)
    if (!is.numeric(observed) || !is.numeric(predicted)) {
        stop(
            "Regression evaluation requires numeric observed and predicted ",
            "values.",
            call. = FALSE
        )
    }
    if (!all(dim(observed) == dim(predicted))) {
        stop(
            "observed and predicted must have the same dimensions.",
            call. = FALSE
        )
    }
    training <- if (is.null(ytrain)) NULL else as.matrix(ytrain)
    if (!is.null(training) && ncol(training) != ncol(observed)) {
        stop(
            "ytrain must have the same number of response columns as ",
            "observed.",
            call. = FALSE
        )
    }
    list(observed = observed, predicted = predicted, training = training)
}

.evaluate_reference_tss <- function(observed, center, keep) {
    centered <- sweep(observed, 2L, center, "-")
    finite_center <- matrix(
        is.finite(center), nrow(observed), ncol(observed), byrow = TRUE
    )
    sum(centered[keep & finite_center]^2)
}

.evaluate_regression_reference_metrics <- function(inputs) {
    observed <- inputs$observed
    predicted <- inputs$predicted
    keep <- is.finite(observed) & is.finite(predicted)
    sse <- sum((predicted - observed)[keep]^2)
    observed_center <- vapply(seq_len(ncol(observed)), function(index) {
        values <- observed[keep[, index], index]
        if (length(values)) mean(values) else NA_real_
    }, numeric(1L))
    observed_tss <- .evaluate_reference_tss(observed, observed_center, keep)
    training_tss <- if (is.null(inputs$training)) {
        NA_real_
    } else {
        .evaluate_reference_tss(
            observed, colMeans(inputs$training, na.rm = TRUE), keep
        )
    }

    c(
        R2 = if (is.finite(observed_tss) && observed_tss > 0) {
            1 - sse / observed_tss
        } else {
            NA_real_
        },
        Q2 = if (is.finite(training_tss) && training_tss > 0) {
            1 - sse / training_tss
        } else {
            NA_real_
        }
    )
}

.evaluate_regression_metric_definitions <- function(has_training) {
    list(
        R2 = paste(
            "Observed-set R2; each response is centered on its observed",
            "mean before sums of squares are aggregated."
        ),
        Q2 = if (!has_training) {
            "Not computed: independent-test Q2 requires ytrain."
        } else {
            paste(
                "Independent-test Q2; each response is centered on its",
                "training-response mean before sums of squares are aggregated."
            )
        }
    )
}

.evaluate_relative_stat <- function(error, observed, epsilon, fun) {
    keep <- is.finite(observed) & abs(observed) > epsilon
    values <- abs(error[keep] / observed[keep]) * 100
    if (length(values)) fun(values, na.rm = TRUE) else NA_real_
}

.evaluate_correlation <- function(observed, predicted, method) {
    if (identical(method, "spearman") && length(observed) >= 4096L &&
        is.null(dim(observed)) && is.null(dim(predicted))) {
        return(spearman_correlation_cpp(observed, predicted))
    }
    .fastpls_quiet(stats::cor(
        observed,
        predicted,
        method = method,
        use = "complete.obs"
    ))
}

.evaluate_rpd <- function(observed, rmsd) {
    if (is.finite(rmsd) && rmsd > 0) {
        return(stats::sd(observed, na.rm = TRUE) / rmsd)
    }
    NA_real_
}

.evaluate_regression_one <- function(observed, predicted, training,
    relative_epsilon,
    na.rm) {
    keep <- is.finite(observed) & is.finite(predicted)
    if (na.rm) {
        observed <- observed[keep]
        predicted <- predicted[keep]
    }
    if (!length(observed)) {
        return(rep(NA, 12L))
    }
    error <- predicted - observed
    sse <- sum(error^2, na.rm = TRUE)
    rmsd <- sqrt(mean(error^2, na.rm = TRUE))
    tss_observed <- sum((observed - mean(observed, na.rm = TRUE))^2,
        na.rm = TRUE)
    tss_training <- if (is.null(training)) {
        NA
    }
    else {
        sum((observed - mean(training, na.rm = TRUE))^2, na.rm = TRUE)
    }
    c(n = length(observed), R2 = if (is.finite(tss_observed) && tss_observed >
        0) {
        1 - sse / tss_observed
    } else {
        NA
    }, Q2 = if (is.finite(tss_training) && tss_training > 0) {
        1 - sse / tss_training
    } else {
        NA
    }, RMSD = rmsd, RMSE = rmsd, MAE = mean(abs(error), na.rm = TRUE),
    bias = mean(error,
        na.rm = TRUE), MRE_percent = .evaluate_relative_stat(error, observed,
        relative_epsilon,
        stats::median), MAPE_percent = .evaluate_relative_stat(error, observed,
        relative_epsilon, mean), RPD = .evaluate_rpd(observed, rmsd),
    Pearson_r = .evaluate_correlation(observed,
        predicted, "pearson"), Spearman_r = .evaluate_correlation(observed,
        predicted,
        "spearman"))
}

.evaluate_regression_by_column <- function(inputs, relative_epsilon, na.rm) {
    observed <- inputs$observed
    values <- t(vapply(
        seq_len(ncol(observed)),
        function(j) {
            training <- if (is.null(inputs$training)) {
                NULL
            } else {
                inputs$training[, j]
            }
            .evaluate_regression_one(
                observed[, j],
                inputs$predicted[, j],
                training,
                relative_epsilon,
                na.rm
            )
        },
        numeric(12L)
    ))
    values <- as.data.frame(values)
    values$response <- colnames(observed) %||%
        paste0(
            "Y",
            seq_len(ncol(observed))
        )
    values[, c("response", setdiff(names(values), "response")), drop = FALSE]
}

.evaluate_regression <- function(observed, predicted, ytrain, bycol,
    relative_epsilon, na.rm) {
    inputs <- .evaluate_regression_inputs(observed, predicted, ytrain)
    training <- if (is.null(inputs$training)) {
        NULL
    } else as.vector(inputs$training)
    overall <- .evaluate_regression_one(
        as.vector(inputs$observed),
        as.vector(inputs$predicted),
        training,
        relative_epsilon,
        na.rm
    )
    reference_metrics <- .evaluate_regression_reference_metrics(inputs)
    overall[["R2"]] <- reference_metrics[["R2"]]
    overall[["Q2"]] <- reference_metrics[["Q2"]]
    output <- list(
        task = "regression",
        metrics = as.data.frame(as.list(overall)),
        metric_definitions = .evaluate_regression_metric_definitions(
            !is.null(training)
        ),
        per_response = if (bycol) {
            .evaluate_regression_by_column(inputs, relative_epsilon, na.rm)
        } else {
            NULL
        }
    )
    if (is.null(training)) {
        output$notes <- paste(
            "Q2 was not computed because ytrain was not supplied;",
            "R2 remains referenced to the observed responses."
        )
    }
    output
}

#' Evaluate prediction performance
#'
#' Computes common classification or regression performance metrics from
#'  observed and predicted values. The function accepts two vectors, two
#' matrices,
#' or classification score matrices. For NMR-style multivariate regression, it
#' reports RMSE/RMSD, R2, Q2, MAE, median relative error percentage, RPD, and
#' correlations. For classification, it reports accuracy, balanced accuracy,
#' macro precision, macro recall, macro F1, Cohen's kappa, and the confusion
#' matrix. Classification output also includes the no-information rate and
#' lift accuracy, defined as accuracy divided by the no-information rate (the
#' accuracy obtained by always predicting the most frequent observed class).
#'
#' @param observed Observed response values. Use a factor/character vector for
#'   classification, a numeric vector/matrix for regression, or a one-hot
#'   matrix for classification.
#' @param predicted Predicted values. Use a factor/character vector for
#'   predicted classes, a numeric vector/matrix for regression, or a class-score
#'   matrix for classification.
#' @param task One of `"auto"`, `"classification"`, or `"regression"`.
#'   `"auto"` treats factor/character observed values as classification and
#'   numeric values as regression, unless a one-hot observed matrix is detected.
#' @param ytrain Optional training response for independent-test regression Q2.
#'   When supplied, each response is centered on its corresponding training
#'   mean before the denominator sums are aggregated across responses.
#'   When omitted, Q2 is returned as `NA` rather than being silently equated
#'   with R2.
#' @param top_k Integer vector of top-k classification accuracies to compute
#'   when `predicted` is a class-score matrix.
#' @param bycol For multivariate regression, calculate and return metrics for
#'   each response column. The default is `TRUE` for direct `evaluate()` calls.
#' @param relative_epsilon Values with absolute observed response below this
#'   threshold are ignored for relative-error metrics.
#' @param na.rm Remove incomplete observations before computing metrics.
#'  @return A list with `task`, `metrics`, `metric_definitions`, and optionally
#' `per_response`,
#'   `per_class`, `confusion`, and `topk`. A `notes` element is included only
#'   when the evaluation has an explanatory note to report.
#' @examples
#' evaluate(iris$Species, iris$Species)
#'
#' set.seed(1)
#' y <- mtcars$mpg
#' pred <- y + rnorm(length(y), sd = 2)
#' evaluate(y, pred)$metrics
#' @export
evaluate <- function(
    observed,
    predicted,
    task = c("auto", "classification", "regression"),
    ytrain = NULL,
    top_k = c(1L, 5L),
    bycol = TRUE,
    relative_epsilon = .Machine$double.eps,
    na.rm = TRUE
) {
    task <- match.arg(task)
    if (identical(task, "auto")) {
        classification <- is.factor(observed) ||
            is.character(observed) ||
            .evaluate_is_onehot(observed)
        task <- if (classification) "classification" else "regression"
    }
    if (identical(task, "classification")) {
        return(.evaluate_classification(observed, predicted, top_k, na.rm))
    }
    .evaluate_regression(
        observed,
        predicted,
        ytrain,
        bycol,
        relative_epsilon,
        na.rm
    )
}


Vip <- function(object) {
    SS <- c(object$Q)^2 * colSums(object$Ttrain^2)
    Wnorm2 <- colSums(object$R^2)
    SSW <- sweep(object$R^2, 2, SS / Wnorm2, "*")
    sqrt(nrow(SSW) * apply(SSW, 1, cumsum) / cumsum(SS))
}


#' Variable importance in projection (VIP)
#'
#' Computes VIP trajectories from fitted model components.
#'
#' @param model Fitted `fastPLS` model.
#' @return Numeric matrix (single response) or list of matrices
#' (multi-response).
#' @examples
#' X <- as.matrix(mtcars[, c("disp", "hp", "wt", "qsec")])
#' y <- mtcars$mpg
#' fit <- pls(X, y,
#'     ncomp = 1, method = "plssvd", backend = "cpu",
#'     svd.method = "rsvd", return_variance = FALSE
#' )
#' ViP(fit)
#' @export
ViP <- function(model) {
    u <- nrow(model$Q)
    if (u == 1) {
        return(as.matrix(Vip(model)))
    }
    V <- list()
    for (i in seq_len(u)) {
        V[[i]] <- Vip(list(
            Q = model$Q[i, ],
            Ttrain = model$Ttrain,
            R = model$R
        ))
    }
    return(V)
}


fastcor <- function(a, b = NULL, byrow = TRUE, diag = TRUE) {
    ## if byrow == T rows are correlated (much faster) else columns
## if diag == T only the diagonal of the cor matrix is returned (much faster)
    ## b can be NULL

    if (!byrow) {
        a <- t(a)
    }
    a <- a - rowMeans(a)
    a <- a / sqrt(rowSums(a * a))
    if (!is.null(b)) {
        if (!byrow) {
            b <- t(b)
        }
        b <- b - rowMeans(b)
        b <- b / sqrt(rowSums(b * b))
        if (diag) {
            return(rowSums(a * b))
        } else {
            return(tcrossprod(a, b))
        }
    } else {
        return(tcrossprod(a))
    }
}
