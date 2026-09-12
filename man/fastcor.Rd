\name{fastcor}
\alias{fastcor}
\title{Fast Pearson Correlation}
\usage{
fastcor(a, b = NULL, byrow = TRUE, diag = TRUE, n.cores = NULL)
}
\arguments{
\item{a}{Numeric matrix.}
\item{b}{Optional numeric matrix with the same row or column orientation as
\code{a}.}
\item{byrow}{Logical; when \code{TRUE}, rows are correlated. When
\code{FALSE}, columns are correlated.}
\item{diag}{Logical; when \code{b} is supplied and \code{diag = TRUE}, return
only the pairwise diagonal correlations between matching rows (or columns).}
\item{n.cores}{Number of CPU cores requested for the compiled matrix
operations. An explicit value takes precedence over
\code{options(n.cores = ...)}.}
}
\value{
A correlation matrix, or a numeric vector of diagonal correlations when
\code{b} is supplied with \code{diag = TRUE}.
}
\description{
Centers and normalizes rows (or columns when \code{byrow = FALSE}) and then
computes Pearson correlations using fast matrix cross-products. This function
does not rank-transform the inputs and therefore does not compute Spearman
correlation.
}
\author{Stefano Cacciatore, Leonardo Tenori, Dupe Ojo, Alessia Vignoli}
\seealso{\code{\link{pls.single.cv}}, \code{\link{pls.double.cv}}}
\examples{
data(iris)
x <- as.matrix(iris[1:10, -5])
fastcor(x)
}
