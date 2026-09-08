#' Configure the default fastPLS execution backend
#'
#' An explicit function argument takes precedence over
#' `options(backend = ...)`, then `FASTPLS_BACKEND`; CPU is the final
#' default.
#'
#' For CPU execution, `options(cores = n)` requests `n` threads from the
#' linked BLAS and OpenMP runtimes. Matrix operations can use those threads
#' when the installed numerical library supports runtime thread control;
#' sequential PLS deflation steps remain serial. macOS builds use Apple
#' Accelerate by default. On other Unix-like systems, configuration
#' automatically selects OpenBLAS when it is found through `OPENBLAS_ROOT` or
#' `pkg-config`; otherwise fastPLS uses the BLAS selected by R.
#'
#' @param backend Optional backend: `"cpu"`, `"cuda"`, or `"metal"`. The
#'   Metal route is a float32 Apple-silicon PLS route
#'   with a fixed mathematical split. CPU code performs preprocessing, reduced
#'   decompositions and sequential component updates. Persistent Metal
#'   workspaces perform fitting products involving the training sample matrix,
#'   including
#'   explicit cross-covariance formation, fused score/loading products, and
#'   randomized range-finder products for implicit cross-covariance operators.
#'   Smaller cross-covariance and component-state products, LDA, and prediction
#'   remain on the CPU. This assignment does not depend on dataset shape.
#'   Setting or retrieving the session backend rejects an
#'   unavailable accelerator immediately; fastPLS does not silently substitute
#'   the CPU backend.
#' @return The configured, available backend. Setting returns the previous
#'   option invisibly.
#' @examples
#' old_options <- options(backend = NULL)
#' current <- fastPLS_backend()
#' current
#' fastPLS_backend("cpu")
#' options(old_options)
#' @export
fastPLS_backend <- function(backend = NULL) {
    if (is.null(backend)) {
        backend <- .fastpls_resolve_backend(NULL)
        .fastpls_require_backend_available(backend, "fastPLS_backend()")
        return(backend)
    }
    backend <- .fastpls_validate_backend(backend, "backend")
    .fastpls_require_backend_available(backend, "fastPLS_backend()")
    old <- getOption("backend", NULL)
    options(backend = backend)
    invisible(old)
}

.fastpls_validate_backend <- function(backend, label = "backend") {
    backend <- tolower(as.character(backend))
    if (
        length(backend) != 1L ||
            is.na(backend) ||
            !nzchar(backend) ||
            !backend %in% c("cpu", "cuda", "metal")
    ) {
        stop(
            sprintf(
                "`%s` must be one of \"cpu\", \"cuda\", or \"metal\".",
                label
            ),
            call. = FALSE
        )
    }
    backend
}

.fastpls_resolve_backend <- function(backend = NULL, allow_auto = FALSE) {
    if (!is.null(backend)) {
        value <- tolower(as.character(backend))
        if (length(value) == 1L && allow_auto && identical(value, "auto")) {
            return("auto")
        }
        return(.fastpls_validate_backend(value))
    }
    option <- getOption("backend", NULL)
    if (!is.null(option)) {
        return(.fastpls_validate_backend(option, "option backend"))
    }
    environment <- Sys.getenv("FASTPLS_BACKEND", unset = "")
    if (nzchar(environment)) {
        return(.fastpls_validate_backend(environment, "FASTPLS_BACKEND"))
    }
    "cpu"
}

.fastpls_require_prediction_backend <- function(dots, context) {
    requested <- dots$backend %||% NULL
    selected <- .fastpls_resolve_backend(requested, allow_auto = TRUE)
    if (!identical(selected, "auto")) {
        .fastpls_require_backend_available(selected, context)
    }
    invisible(selected)
}

.fastpls_backend_available <- function(backend) {
    switch(
        .fastpls_validate_backend(backend),
        cpu = TRUE,
        cuda = isTRUE(has_cuda()),
        metal = isTRUE(has_metal())
    )
}

.fastpls_require_backend_available <- function(
    backend,
    context = "The requested operation",
    available = NULL
) {
    backend <- .fastpls_validate_backend(backend)
    if (is.null(available)) {
        available <- .fastpls_backend_available(backend)
    }
    if (isTRUE(available)) {
        return(backend)
    }
    requirement <- switch(
        backend,
        cuda = "a CUDA-enabled fastPLS build and an available NVIDIA GPU",
        metal = "a macOS fastPLS build with Apple Metal support"
    )
    stop(
        context,
        " requested backend='",
        backend,
        "', which requires ",
        requirement,
        ". No CPU fallback is performed.",
        call. = FALSE
    )
}

.fastpls_validate_cores <- function(cores) {
    if (
        length(cores) != 1L ||
            !is.numeric(cores) ||
            is.na(cores) ||
            !is.finite(cores) ||
            cores < 1 ||
            cores != floor(cores)
    ) {
        stop("`options(cores = ...)` must contain one positive integer.",
            call. = FALSE)
    }
    as.integer(cores)
}

.fastpls_cpu_cores <- function() {
    cores <- getOption("cores", NULL)
    if (is.null(cores)) {
        return(NULL)
    }
    .fastpls_validate_cores(cores)
}

.fastpls_apply_cpu_cores <- function() {
    cores <- .fastpls_cpu_cores()
    if (is.null(cores)) {
        return(invisible(NULL))
    }
    value <- as.character(cores)
    do.call(
        Sys.setenv,
        as.list(stats::setNames(
            rep(value, 6L),
            c(
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "GOTO_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"
            )
        ))
    )
    if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
        try(RhpcBLASctl::blas_set_num_threads(cores), silent = TRUE)
        try(RhpcBLASctl::omp_set_num_threads(cores), silent = TRUE)
    }
    invisible(cores)
}
