#ifndef FASTPLS_FLOAT_CROSSCOV_OPERATOR_H
#define FASTPLS_FLOAT_CROSSCOV_OPERATOR_H

#include "svd_cuda_rsvd.h"
#include "svd_metal_backend.h"
#include <fastpls/native/operators.hpp>
#include <stdexcept>

namespace fastpls_svd {

using ExplicitFloatSVDOperator = fastpls::native::MatrixOperator<float>;

// S_a = F_a'Y, F_0 = X, F_a = F_{a-1}(I-v_a v_a'). Updating the
// smaller left factor avoids subtracting nearly equal full cross-products.
class FloatCrosscovOperator {
 public:
  FloatCrosscovOperator(const arma::fmat& X, const arma::fmat& Y,
                        arma::uword max_components, int backend)
      : n_rows(X.n_cols), n_cols(Y.n_cols), Y_(Y), capacity_(max_components) {
    if (X.n_rows != Y.n_rows || X.n_elem == 0 || Y.n_elem == 0) {
      throw std::runtime_error("float32 cross-product requires nonempty X/Y with matching rows");
    }
    if (backend == 1) cuda_.reset(new CudaFloatCrossproduct(X, Y));
    else if (backend == 2) metal_.reset(new MetalFloatCrossproduct(X, Y));
    else if (backend == 0) cpu_.reset(new fastpls::native::CrosscovOperator<float>(
      X, Y, max_components));
    else throw std::runtime_error("Invalid float32 cross-product backend");
  }
  arma::uword workspace_rows() const { return Y_.n_rows; }

  arma::fmat multiply(const arma::fmat& B, bool transpose = false) {
    if (B.n_rows != (transpose ? n_rows : n_cols)) {
      throw std::runtime_error("float32 cross-product: non-conformable matrices");
    }
    arma::fmat result;
    if (cuda_) result = cuda_->multiply(B, transpose);
    else if (metal_) result = metal_->multiply(B, transpose);
    else result = cpu_->multiply(B, transpose);
    return result;
  }

  void deflate(const arma::fvec& v) {
    if (v.n_elem != n_rows || active_ >= capacity_) {
      throw std::runtime_error("Invalid float32 cross-product deflation dimensions");
    }
    if (cuda_) cuda_->deflate(v);
    else if (metal_) metal_->deflate(v);
    else cpu_->deflate(v);
    ++active_;
  }

  void full_svd(arma::fmat& U, arma::fvec& d, arma::fmat& V, bool left_only) {
    // Exact factor-product reduction for breakdown/full-rank cases, without
    // allocating the p-by-q operator. This reduced decomposition is host-side.
    if (cpu_) {
      cpu_->full_svd(U, d, V, left_only);
      return;
    }
    arma::fmat factor = cuda_ ? cuda_->left_factor() : metal_->left_factor();
    fastpls::native::factor_product_svd(factor, Y_, U, d, V, left_only);
  }
  arma::fmat reduced(const arma::fmat& Q) { return multiply(Q, true).t(); }
  void qr(arma::fmat& Q, arma::fmat& R, const arma::fmat& B) {
    if (cuda_) {
      Q = B;
      cuda_->orthonormalize(Q);
    } else if (!arma::qr_econ(Q, R, B)) {
      throw std::runtime_error("float32 cross-product QR failed");
    }
  }
  const arma::uword n_rows, n_cols;

 private:
  const arma::fmat& Y_;
  std::unique_ptr<fastpls::native::CrosscovOperator<float>> cpu_;
  const arma::uword capacity_;
  arma::uword active_ = 0;
  std::unique_ptr<CudaFloatCrossproduct> cuda_;
  std::unique_ptr<MetalFloatCrossproduct> metal_;
};

} // namespace fastpls_svd
#endif
