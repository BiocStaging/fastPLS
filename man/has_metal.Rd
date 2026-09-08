\name{has_metal}
\alias{has_metal}
\title{Check Apple Metal Backend Availability for fastPLS}

\description{
Report whether the fixed CPU/Metal operation-split fastPLS backend can be used
in the current session.
}

\usage{
has_metal()
}

\details{
\code{has_metal()} returns \code{TRUE} only when both conditions hold:
\itemize{
\item the package was built with Apple Metal support;
\item the Metal backend is available in the current session.
}

If either condition is not met, the function returns \code{FALSE}. This includes
non-macOS builds and portable builds that use the Metal stub backend. The result
describes backend availability for the current R session; it does not select a
device, allocate GPU memory, or run a test computation. When the Xcode Metal
compiler is available during installation, fastPLS compiles and embeds its
custom shader library. Device and pipeline initialization is then incurred by
the first Metal operation, while subsequent operations in the same R session
reuse the initialized state. A true result identifies availability of the
operation-split backend; it does not imply that an entire PLS fit is GPU
resident.
}

\value{A length-1 logical.}

\seealso{\code{\link{has_cuda}}, \code{\link{fastsvd}}, \code{\link{pls}}}

\examples{
has_metal()
}
