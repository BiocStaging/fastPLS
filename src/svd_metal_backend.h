#ifndef FASTPLS_SVD_METAL_BACKEND_H
#define FASTPLS_SVD_METAL_BACKEND_H

#include <RcppArmadillo.h>
#include <memory>

namespace fastpls_svd {

bool has_metal_backend();

// One explicit operator per fit/solve. Values are refreshed by the owner after
// deflation; no matrix contents persist beyond this object's lifetime.
class MetalFloatOperator {
 public:
  explicit MetalFloatOperator(const arma::fmat& A);
  ~MetalFloatOperator();
  MetalFloatOperator(const MetalFloatOperator&) = delete;
  MetalFloatOperator& operator=(const MetalFloatOperator&) = delete;
  void update(const arma::fmat& A);
  arma::fmat multiply(const arma::fmat& B, bool transpose_left);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// X and Y remain resident; the n-by-width intermediate never returns to R.
class MetalFloatCrossproduct {
 public:
  MetalFloatCrossproduct(const arma::fmat& X, const arma::fmat& Y);
  MetalFloatCrossproduct(const arma::mat& X, const arma::mat& Y);
  ~MetalFloatCrossproduct();
  MetalFloatCrossproduct(const MetalFloatCrossproduct&) = delete;
  MetalFloatCrossproduct& operator=(const MetalFloatCrossproduct&) = delete;
  arma::fmat multiply(const arma::fmat& B, bool transpose);
  arma::mat multiply(const arma::mat& B, bool transpose);
  void deflate(const arma::fvec& v);
  arma::fmat left_factor();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

arma::mat metal_matrix_multiply(const arma::mat& A, const arma::mat& B);
arma::mat metal_matrix_multiply(const arma::mat& A,
                                const arma::mat& B,
                                bool transpose_left,
                                bool transpose_right);
arma::fmat metal_matrix_multiply_float(const arma::fmat& A,
                                       const arma::fmat& B,
                                       bool transpose_left,
                                       bool transpose_right);
arma::mat metal_crossprod(const arma::mat& A, const arma::mat& B);
Rcpp::List metal_simpls_resident(const arma::mat& X,
                                 const arma::mat& Y,
                                 int ncomp,
                                 int power_iters,
                                 int seed);

} // namespace fastpls_svd

#endif
