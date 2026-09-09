#ifndef FASTPLS_FLOAT_CROSSCOV_OPERATOR_H
#define FASTPLS_FLOAT_CROSSCOV_OPERATOR_H

#include "svd_cuda_rsvd.h"
#include "svd_metal_backend.h"
#include "core_cpu_backend.h"
#include <fastpls/core/operator_rsvd.hpp>
#include <fastpls/core/operators.hpp>
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
    else if (backend == 2 || backend == 3) {
      metal_.reset(new MetalFloatCrossproduct(X, Y));
    }
    else if (backend == 0) {
      response_mean_.assign(Y.n_cols, 0.0f);
      cpu_base_.reset(new CpuBase(
        fastpls::core::make_const_view(
          X.memptr(), X.n_rows, X.n_cols, X.n_rows
        ),
        fastpls::core::make_const_view(
          Y.memptr(), Y.n_rows, Y.n_cols, Y.n_rows
        ),
        response_mean_.data(), response_mean_.size(), cpu_backend_
      ));
      cpu_.reset(new CpuCrosscov(
        *cpu_base_, std::max<arma::uword>(max_components, 1), cpu_backend_
      ));
    }
    else throw std::runtime_error("Invalid float32 cross-product backend");
  }
  arma::uword workspace_rows() const { return Y_.n_rows; }
  std::size_t rows() const noexcept { return n_rows; }
  std::size_t columns() const noexcept { return n_cols; }

  void multiply(fastpls::core::ConstMatrixView<float> right,
                bool transpose,
                fastpls::core::Matrix<float>& output) {
    if (!cpu_) {
      throw std::runtime_error(
        "Direct core products are available only for the CPU operator"
      );
    }
    cpu_->multiply(right, transpose, output);
  }

  fastpls::runtime::CpuLinearAlgebraF32& cpu_backend() {
    if (!cpu_) {
      throw std::runtime_error(
        "The float32 CPU linear-algebra backend is not active"
      );
    }
    return cpu_backend_;
  }

  arma::fmat multiply(const arma::fmat& B, bool transpose = false) {
    if (B.n_rows != (transpose ? n_rows : n_cols)) {
      throw std::runtime_error("float32 cross-product: non-conformable matrices");
    }
    arma::fmat result;
    if (cuda_) result = cuda_->multiply(B, transpose);
    else if (metal_) result = metal_->multiply(B, transpose);
    else {
      const arma::uword output_rows = transpose ? n_cols : n_rows;
      result.set_size(output_rows, B.n_cols);
      cpu_->multiply(
        fastpls::core::make_const_view(
          B.memptr(), B.n_rows, B.n_cols, B.n_rows
        ),
        transpose,
        fastpls::core::make_view(
          result.memptr(), result.n_rows, result.n_cols, result.n_rows
        )
      );
    }
    return result;
  }

  void deflate(const arma::fvec& v) {
    if (v.n_elem != n_rows || active_ >= capacity_) {
      throw std::runtime_error("Invalid float32 cross-product deflation dimensions");
    }
    if (cuda_) cuda_->deflate(v);
    else if (metal_) metal_->deflate(v);
    else {
      cpu_->deflate(fastpls::core::make_const_view(
        v.memptr(), v.n_elem, std::size_t(1), v.n_elem
      ));
    }
    ++active_;
  }

  void full_svd(arma::fmat& U, arma::fvec& d, arma::fmat& V, bool left_only) {
    // Exact factor-product reduction for breakdown/full-rank cases, without
    // allocating the p-by-q operator. This reduced decomposition is host-side.
    if (cpu_) {
      fastpls::core::Matrix<float> matrix;
      fastpls::core::Matrix<float> core_u;
      fastpls::core::Matrix<float> core_vt;
      std::vector<float> singular;
      cpu_->materialize(matrix);
      if (!cpu_backend_.svd_economy(
            matrix.view(), left_only, core_u, singular, core_vt)) {
        throw std::runtime_error(
          "float32 cross-covariance decomposition failed"
        );
      }
      U = arma::fmat(
        core_u.data(), core_u.rows(), core_u.columns()
      );
      d = arma::fvec(singular.data(), singular.size());
      if (left_only) {
        V.reset();
      } else {
        V.set_size(core_vt.columns(), core_vt.rows());
        for (arma::uword column = 0; column < V.n_cols; ++column) {
          for (arma::uword row = 0; row < V.n_rows; ++row) {
            V(row, column) = core_vt(column, row);
          }
        }
      }
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
  using CpuBase = fastpls::core::CenteredCrosscovOperator<
    float, fastpls::runtime::CpuLinearAlgebraF32
  >;
  using CpuCrosscov = fastpls::core::ProjectedOperator<
    float, CpuBase, fastpls::runtime::CpuLinearAlgebraF32
  >;
  const arma::fmat& Y_;
  fastpls::runtime::CpuLinearAlgebraF32 cpu_backend_;
  std::vector<float> response_mean_;
  std::unique_ptr<CpuBase> cpu_base_;
  std::unique_ptr<CpuCrosscov> cpu_;
  const arma::uword capacity_;
  arma::uword active_ = 0;
  std::unique_ptr<CudaFloatCrossproduct> cuda_;
  std::unique_ptr<MetalFloatCrossproduct> metal_;
};

} // namespace fastpls_svd
#endif
