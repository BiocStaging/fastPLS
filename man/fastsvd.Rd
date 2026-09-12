\name{fastsvd}
\alias{fastsvd}
\title{Native Randomized Singular Value Decomposition}
\usage{
fastsvd(
    x,
    nu = NULL,
    nv = NULL,
    ncomp = NULL,
    backend = NULL,
    n.cores = NULL,
    oversample = 32L,
    power = 5L,
    seed = 1L
)
}
\arguments{
\item{x}{Dense matrix to decompose. A base R numeric matrix is processed in
float64. A \code{float::float32} matrix is processed end to end in float32 and
returns float32 left and right singular vectors. Sparse matrices should be
converted by the caller.}

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
\code{options(backend = ...)}, then \code{FASTPLS_BACKEND}, then CPU.
An unavailable or unsupported accelerator selection raises an error; no CPU
fallback is performed.}

\item{n.cores}{Number of CPU cores requested for supported BLAS/OpenMP host
operations. An explicit value takes precedence over
\code{options(n.cores = ...)}. The setting does not control CUDA or Metal
device parallelism.}

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
\code{svd.method}, \code{elapsed}, \code{ncomp}, and \code{precision}. For base
R numeric input, \code{u} and \code{v} are float64 matrices. For
\code{float::float32} input, \code{u} and \code{v} remain float32 matrices.
Both CPU precisions receive the native case-specific rSVD audit, strengthened
retries, and effective controls. The float32 audit remains in single precision,
and the R layer does not convert the input to float64 solely for diagnostics.
Float64 input additionally reports selected singular-triplet residuals and
left/right orthogonality errors. Recovery after failed CPU sketches uses
strengthened native-rSVD attempts. These fields describe the numerical checks
applied to the current decomposition.
}
\description{
Computes a truncated randomized singular value decomposition (rSVD) with the
CPU backend. Ordinary R numeric matrices use float64 calculations;
\code{float::float32} matrices automatically use the native float32 path
without conversion to float64.
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
s64 <- fastsvd(x, ncomp = 2, backend = "cpu", seed = 1)
s64$d

x32 <- float::fl(x)
s32 <- fastsvd(x32, ncomp = 2, backend = "cpu", seed = 1)
s32$d

}
