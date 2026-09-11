\name{fastsvd}
\alias{fastsvd}
\title{Singular value decomposition through fastPLS backends}
\usage{
fastsvd(
    x,
    nu = NULL,
    nv = NULL,
    ncomp = NULL,
    backend = NULL,
    method = "rsvd",
    oversample = 32L,
    power = 5L,
    seed = 1L
)
}
\arguments{
\item{x}{Numeric matrix to decompose, with observations or rows in rows and
variables or columns in columns. Sparse matrices should be converted by the
caller; \code{fastsvd()} currently works on a dense numeric matrix.}

\item{nu}{Number of left singular vectors to return. If \code{NULL}, the
    function
uses the largest feasible rank implied by the matrix dimensions. When
\code{ncomp} is supplied, \code{ncomp} controls the decomposition rank and
\code{nu} controls only how many left vectors are kept in the returned object.}

\item{nv}{Number of right singular vectors to return. If \code{NULL}, the
function uses the largest feasible rank implied by the matrix dimensions.
When \code{ncomp} is supplied, \code{ncomp} controls the decomposition rank and
\code{nv} controls only how many right vectors are kept in the returned object.}

\item{ncomp}{Optional truncated rank. When supplied, it overrides the rank
implied by \code{nu} and \code{nv}; the final rank is always capped at
\code{min(nrow(x), ncol(x))}.}

\item{backend}{Compute backend. \code{cpu} runs on the host CPU. Standalone
CUDA and Metal rSVD are unavailable because their reduced QR/SVD stages are
not fully device-native for every matrix shape; both accelerators remain
available for supported resident PLS fits. When omitted, the function uses
\code{options(backend = ...)}, then \code{FASTPLS_BACKEND}, then CPU. For CPU
execution, \code{options(cores = n)} requests \code{n} BLAS/OpenMP threads.
An unavailable or unsupported accelerator selection raises an error; no CPU
fallback is performed.}

\item{method}{SVD algorithm. Only \code{rsvd}, the native randomized SVD,
is supported on the selected backend.}

\item{oversample}{Non-negative oversampling dimension used by randomized SVD.
The sketch dimension is approximately \code{ncomp + oversample}, capped by the
matrix rank. Larger values can improve approximation accuracy at the cost of
extra time and memory. The default starting value is 32. Panel
agreement is not a guarantee for a new matrix; CPU float32 and float64 fits
apply the native case-specific audit described in Details.}

\item{power}{Number of randomized-SVD power iterations. The default is five on
CPU. CPU float64 and float32 execution applies case-specific auditing and
recovery. Each additional iteration adds matrix multiplications.}

\item{seed}{Random seed used to generate the Gaussian sketch. It affects
\code{rsvd} results.}
}
\value{
A list compatible with \code{base::svd()} containing \code{d}, \code{u}, and
\code{v}, plus backend metadata \code{backend}, \code{method},
\code{svd.method}, \code{elapsed}, and \code{ncomp}. The
\code{diagnostics} element reports rank and finiteness checks. Both CPU
precisions receive the native case-specific rSVD audit, strengthened retries,
and effective controls. For float32, the audit remains in single precision and
the R layer does not convert the input to double solely for diagnostics.
Double-precision input additionally reports selected singular-triplet residuals
and left/right orthogonality errors. Recovery after failed CPU sketches uses
strengthened native-rSVD attempts. These fields describe the numerical checks
applied to the current decomposition.
}
\description{
Computes a truncated singular value decomposition of a dense numeric matrix
with the CPU backend. The result contains singular
values, requested singular vectors, decomposition rank, elapsed time, and
backend metadata.
}
\details{
Randomized SVD is stochastic. CPU float32 and float64 execution audits each decomposition
using normalized left/right singular-triplet residuals and an omitted-direction
search. An estimated omitted-to-retained boundary ratio above 0.95 is treated
as weak separation, not by itself as a failed approximation. A residual above
0.01 or an omitted-direction ratio above 1.01 triggers stronger sketches;
independent-sketch agreement supplies an additional acceptance check.
If the ordinary CPU attempts fail, native operator rSVD makes up to four
further attempts with wider sketches and additional power iterations. Each
recovery result is rechecked. If recovery exceeds its estimated 256 MiB
major-workspace budget or still fails its checks, an error is returned.
This budget excludes input matrices, model storage, and runtime overhead;
it is not a bound on process RSS. Small CPU inputs can use a separately
recorded dense-SVD route. A full-width sketch can also use a dense SVD as
part of the native algorithm. Accelerator routes report their structural
diagnostics and exact controls. Confirm coefficient or subspace
interpretation against an independent dense decomposition when feasible,
and compare seeds when conclusions are near a decision boundary.
}
\examples{
set.seed(1)
x <- matrix(rnorm(12 * 5), 12, 5)
s <- fastsvd(x, ncomp = 2, backend = "cpu", method = "rsvd", seed = 1)
s$d

}
