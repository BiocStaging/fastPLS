// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_NATIVE_OPERATORS_HPP
#define FASTPLS_NATIVE_OPERATORS_HPP
#include <fastpls/native/rsvd.hpp>

namespace fastpls { namespace native {

template<typename T>
void factor_product_svd(const arma::Mat<T>& left, const arma::Mat<T>& right,
                        arma::Mat<T>& U, arma::Col<T>& d, arma::Mat<T>& V,
                        bool left_only) {
  arma::Mat<T> L = left.t(), H = right.t();
  arma::Mat<T> Qx, Rx, Qy, Ry, smallU, smallV;
  if (!arma::qr_econ(Qx, Rx, L) || !arma::qr_econ(Qy, Ry, H)) {
    throw std::runtime_error("factor-product QR failed");
  }
  arma::Mat<T> reduced = Rx * Ry.t();
  if (!arma::svd_econ(smallU, d, smallV, reduced, left_only ? "left" : "both")) {
    throw std::runtime_error("factor-product decomposition failed");
  }
  U = Qx * smallU;
  if (!left_only) V = Qy * smallV;
}

template<typename T>
class MatrixOperator {
 public:
  explicit MatrixOperator(const arma::Mat<T>& matrix)
      : n_rows(matrix.n_rows), n_cols(matrix.n_cols), matrix_(matrix) {}
  arma::uword workspace_rows() const { return 0; }
  arma::Mat<T> multiply(const arma::Mat<T>& B, bool transpose = false) {
    if (transpose) return matrix_.t() * B;
    return matrix_ * B;
  }
  void full_svd(arma::Mat<T>& U, arma::Col<T>& d, arma::Mat<T>& V, bool left_only) {
    if (!arma::svd_econ(U, d, V, matrix_, left_only ? "left" : "both")) {
      throw std::runtime_error("full decomposition failed");
    }
  }
  arma::Mat<T> reduced(const arma::Mat<T>& Q) { return Q.t() * matrix_; }
  void qr(arma::Mat<T>& Q, arma::Mat<T>& R, const arma::Mat<T>& B) {
    if (!arma::qr_econ(Q, R, B)) throw std::runtime_error("QR failed");
  }
  const arma::uword n_rows, n_cols;
 private:
  const arma::Mat<T>& matrix_;
};

// S_a = F_a'Y with F_a = F_{a-1}(I-v_a v_a'). Keep the smaller
// factor instead of allocating/subtracting full predictor-response products.
template<typename T>
class CrosscovOperator {
 public:
  CrosscovOperator(const arma::Mat<T>& X, const arma::Mat<T>& Y,
                    arma::uword max_components)
      : n_rows(X.n_cols), n_cols(Y.n_cols), right_(Y), capacity_(max_components) {
    if (X.n_rows != Y.n_rows || X.is_empty() || Y.is_empty()) {
      throw std::invalid_argument("cross-product requires nonempty X/Y with matching rows");
    }
    left_ = X;
  }
  arma::uword workspace_rows() const { return right_.n_rows; }
  arma::Mat<T> multiply(const arma::Mat<T>& B, bool transpose = false) {
    if (B.n_rows != (transpose ? n_rows : n_cols)) {
      throw std::invalid_argument("cross-product: non-conformable matrices");
    }
    if (transpose) {
      intermediate_ = left_ * B;
      return right_.t() * intermediate_;
    }
    intermediate_ = right_ * B;
    return left_.t() * intermediate_;
  }
  void deflate(const arma::Col<T>& v) {
    if (v.n_elem != n_rows || active_ >= capacity_) {
      throw std::invalid_argument("Invalid cross-product deflation dimensions");
    }
    score_ = left_ * v;
    for (arma::uword column = 0; column < left_.n_cols; ++column) {
      column_ = score_ * v(column);
      left_.col(column) -= column_;
    }
    ++active_;
  }
  void full_svd(arma::Mat<T>& U, arma::Col<T>& d, arma::Mat<T>& V, bool left_only) {
    factor_product_svd(left_, right_, U, d, V, left_only);
  }
  arma::Mat<T> reduced(const arma::Mat<T>& Q) { return multiply(Q, true).t(); }
  void qr(arma::Mat<T>& Q, arma::Mat<T>& R, const arma::Mat<T>& B) {
    if (!arma::qr_econ(Q, R, B)) throw std::runtime_error("cross-product QR failed");
  }
  const arma::uword n_rows, n_cols;
 private:
  const arma::Mat<T>& right_;
  arma::Mat<T> left_, intermediate_;
  arma::Col<T> score_, column_;
  const arma::uword capacity_;
  arma::uword active_ = 0;
};

} } // namespace fastpls::native
#endif
