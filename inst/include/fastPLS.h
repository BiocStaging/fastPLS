#ifndef fastPLS_fastPLS_H
#define fastPLS_fastPLS_H

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;
using namespace arma;

double RQ(arma::mat yData, arma::mat yPred);
arma::mat variance(const arma::mat& x);
arma::mat transformy(arma::ivec y);
bool has_cuda();

Rcpp::List truncated_svd_debug(
  const arma::mat& A,
  int k,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed,
  bool left_only
);

List pls_model1(
  arma::mat Xtrain,
  arma::mat Ytrain,
  arma::ivec ncomp,
  int scaling,
  bool fit,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed
);

List pls_model2(
  arma::mat Xtrain,
  arma::mat Ytrain,
  arma::ivec ncomp,
  int scaling,
  bool fit,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed
);

List pls_model2_fast(
  SEXP XtrainSEXP,
  SEXP YtrainSEXP,
  arma::ivec ncomp,
  int scaling,
  bool fit,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed
);

List pls_predict(List& model, arma::mat Xtest, bool proj);
IntegerVector samplewithoutreplace(IntegerVector yy, int size);

List optim_pls_cv(
  arma::mat Xdata,
  arma::mat Ydata,
  arma::ivec constrain,
  arma::ivec ncomp,
  int scaling,
  int kfold,
  int method,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed
);

#endif
