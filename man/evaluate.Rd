\name{evaluate}
\alias{evaluate}
\title{Evaluate prediction performance}
\usage{
evaluate(
    observed,
    predicted,
    ytrain = NULL,
    bycol = TRUE,
    relative_epsilon = .Machine$double.eps,
    na.rm = TRUE
)
}
\arguments{
\item{observed}{Observed response values. Use a factor or character vector for
classification, a numeric vector or matrix for regression, or a one-hot matrix
for classification.}

\item{predicted}{Predicted values. Use a factor or character vector for
predicted classes, a numeric vector or matrix for regression, a class-score or
ranked-label matrix for classification, or the complete object returned by
\code{predict()} for a fitted fastPLS model.}

\item{ytrain}{Optional training response for independent-test regression Q2.
When supplied, each response is centered on its corresponding training mean
before denominator sums are aggregated across responses. When omitted, Q2 is
returned as \code{NA} rather than being silently equated with R2.}

\item{bycol}{For multivariate regression, calculate and return metrics for each
response column. The default is \code{TRUE} for direct \code{evaluate()} calls.}

\item{relative_epsilon}{Values with absolute observed response below this
threshold are ignored for relative-error metrics.}

\item{na.rm}{Remove incomplete observations before computing metrics.}
}
\value{
A list with \code{task}, \code{metrics}, \code{metric_definitions}, and
optionally \code{per_response}, \code{per_class}, \code{confusion}, and
\code{topk}. Top-k ranks are inferred from score or ranked-prediction columns.
For a prediction object with several component counts, \code{metrics} has one
row per count and \code{by_component} contains each complete evaluation. A
\code{notes} element is included only when the evaluation has an explanatory
note to report.
}
\description{
Computes common classification or regression performance metrics from observed
and predicted values. The function accepts vectors, matrices, classification
score or rank matrices, or complete fastPLS prediction results.
}
\details{
For classification, the returned metrics include accuracy, the no-information
rate, lift accuracy, balanced accuracy, macro precision, macro recall, macro
F1, Cohen's kappa, and a confusion matrix. The no-information rate is the
accuracy obtained by always predicting the most prevalent observed class. Lift
accuracy is accuracy divided by this baseline; values above one indicate an
improvement over the majority-class baseline. When class scores or ranked
labels are supplied, top-k accuracy is inferred and reported automatically.

For regression, the returned metrics include R2, Q2, RMSD/RMSE, MAE, bias,
median relative error percentage, mean absolute percentage error, ratio of
performance to deviation, Pearson correlation, and Spearman correlation. These
include the main spectral prediction metrics used by Vignoli et al. (2025):
median relative error percentage, RMSE, R2, and RPD.
Set \code{bycol = FALSE} to return only the aggregate metrics and avoid
calculating a separate metric vector for every response column.
For multivariate responses, aggregate R2 centers each response column on its
own observed mean before summing the residual and reference sums of squares.
Independent-test Q2 analogously centers each response column on the
corresponding training mean supplied through \code{ytrain}. If that reference
is unavailable, Q2 is \code{NA} and \code{metric_definitions} records why it
was not calculated.
}
\examples{
evaluate(iris$Species, iris$Species)

set.seed(1)
y <- mtcars$mpg
pred <- y + rnorm(length(y), sd = 2)
evaluate(y, pred)$metrics
}
