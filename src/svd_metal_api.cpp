#include "svd_metal_backend.h"

#include <RcppArmadillo.h>

namespace {
using Crossproduct = fastpls_svd::MetalFloatCrossproduct;

Rcpp::XPtr<Crossproduct> checked_crossproduct(SEXP workspace) {
  if (TYPEOF(workspace) != EXTPTRSXP ||
      R_ExternalPtrTag(workspace) != Rf_install("fastPLS_metal_crossproduct")) {
    Rcpp::stop("Invalid Metal cross-product workspace");
  }
  return Rcpp::XPtr<Crossproduct>(workspace);
}
}

// [[Rcpp::export(rng = false)]]
SEXP metal_xprod_workspace_cpp(const arma::mat& X, const arma::mat& Y) {
  if (!fastpls_svd::has_metal_backend()) {
    Rcpp::stop("Metal cross-product unavailable; no CPU fallback is performed");
  }
  return Rcpp::XPtr<Crossproduct>(new Crossproduct(X, Y), true,
      Rf_install("fastPLS_metal_crossproduct"));
}

// [[Rcpp::export(rng = false)]]
arma::mat metal_xprod_workspace_multiply_cpp(SEXP workspace, const arma::mat& B,
                                            bool transpose) {
  auto pointer = checked_crossproduct(workspace);
  if (!pointer.get()) Rcpp::stop("Metal cross-product workspace has been released");
  return pointer->multiply(B, transpose);
}

// [[Rcpp::export(rng = false)]]
void metal_xprod_workspace_release_cpp(SEXP workspace) {
  auto pointer = checked_crossproduct(workspace);
  delete pointer.get();
  R_ClearExternalPtr(workspace);
}

// [[Rcpp::export(rng = false)]]
Rcpp::List metal_xprod_rsvd_cpp(const arma::mat& X, const arma::mat& Y,
                              const arma::mat& omega, int target,
                              int power, bool left_only) {
  if (!fastpls_svd::has_metal_backend()) {
    Rcpp::stop("Metal cross-product unavailable; no CPU fallback is performed");
  }
  if (target < 1 || power < 0 || omega.n_rows != Y.n_cols ||
      omega.n_cols < static_cast<arma::uword>(target)) {
    Rcpp::stop("Invalid Metal matrix-free rSVD sketch or controls");
  }
  Crossproduct op(X, Y);
  arma::mat sketch = op.multiply(omega, false);
  arma::mat basis, triangular, reverse, right_basis;
  auto orthonormalize = [&](arma::mat& Q, const arma::mat& values) {
    if (!values.is_finite() || !arma::qr_econ(Q, triangular, values)) {
      Rcpp::stop("Metal matrix-free rSVD: host QR factorization failed");
    }
  };
  // GPU products retain the same float arithmetic. Host QR and the reduced
  // SVD use the package's linked double BLAS/LAPACK, not R's LINPACK QR.
  for (int iteration = 0; iteration < power; ++iteration) {
    orthonormalize(basis, sketch);
    reverse = op.multiply(basis, true);
    orthonormalize(right_basis, reverse);
    sketch = op.multiply(right_basis, false);
  }
  orthonormalize(basis, sketch);
  arma::mat reduced = op.multiply(basis, true).t();
  arma::mat U, V;
  arma::vec singular;
  if (!reduced.is_finite() ||
      !arma::svd_econ(U, singular, V, reduced, left_only ? "left" : "both")) {
    Rcpp::stop("Metal matrix-free rSVD: reduced SVD failed");
  }
  const arma::uword used = std::min(static_cast<arma::uword>(target), singular.n_elem);
  if (used == 0) Rcpp::stop("Metal matrix-free rSVD produced no singular directions");
  arma::mat left = basis * U.cols(0, used - 1);
  return Rcpp::List::create(
      Rcpp::Named("U") = left,
      Rcpp::Named("s") = Rcpp::NumericVector(singular.begin(), singular.begin() + used),
      Rcpp::Named("Vt") = left_only ? R_NilValue :
          Rcpp::wrap(arma::mat(V.cols(0, used - 1).t())));
}

// [[Rcpp::export]]
bool has_metal() {
  return fastpls_svd::has_metal_backend();
}

// [[Rcpp::export]]
arma::mat metal_matrix_multiply_cpp(const arma::mat& A, const arma::mat& B) {
  return fastpls_svd::metal_matrix_multiply(A, B);
}

// [[Rcpp::export]]
arma::mat metal_crossprod_cpp(const arma::mat& A, const arma::mat& B) {
  return fastpls_svd::metal_crossprod(A, B);
}

// [[Rcpp::export]]
Rcpp::List metal_simpls_resident_cpp(const arma::mat& X,
                                     const arma::mat& Y,
                                     int ncomp,
                                     int power_iters = 5,
                                     int seed = 1) {
  return fastpls_svd::metal_simpls_resident(X, Y, ncomp, power_iters, seed);
}
