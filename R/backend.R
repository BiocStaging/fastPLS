#' Report the numerical library selected when fastPLS was compiled
#'
#' This reports the CPU matrix library selected by the package configuration,
#' rather than attempting to infer a library from the current R session.
#' macOS builds normally report `"Accelerate"`. Linux and Windows builds report
#' `"OpenBLAS"` when OpenBLAS was found or explicitly requested, and
#' `"R BLAS/LAPACK"` when the package used R's portable fallback.
#'
#' For reproducible performance benchmarks on Linux or Windows, install with
#' `FASTPLS_USE_OPENBLAS=1` and verify that this function returns
#' `"OpenBLAS"` before running the analysis.
#'
#' @return A single character string: `"Accelerate"`, `"OpenBLAS"`, or
#'   `"R BLAS/LAPACK"`.
#' @examples
#' fastPLS_blas()
#' @export
fastPLS_blas <- function() {
    blas_backend_cpp()
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

.fastpls_validate_cores <- function(n.cores, source = "n.cores") {
    if (
        length(n.cores) != 1L ||
            !is.numeric(n.cores) ||
            is.na(n.cores) ||
            !is.finite(n.cores) ||
            n.cores < 1 ||
            n.cores != floor(n.cores)
    ) {
        stop("`", source, "` must contain one positive integer.", call. = FALSE)
    }
    as.integer(n.cores)
}

.fastpls_cpu_cores <- function(n.cores = NULL) {
    if (!is.null(n.cores)) {
        return(.fastpls_validate_cores(n.cores))
    }
    n.cores <- getOption("n.cores", NULL)
    if (is.null(n.cores)) {
        return(NULL)
    }
    .fastpls_validate_cores(n.cores, "options(n.cores = ...)")
}

.fastpls_apply_cpu_cores <- function(n.cores = NULL) {
    cores <- .fastpls_cpu_cores(n.cores)
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
    set_cpu_threads_cpp(cores)
    invisible(cores)
}
