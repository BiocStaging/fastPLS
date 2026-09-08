#include "svd_metal_backend.h"
#include "accelerator_core_backend.h"

#include <stdexcept>

namespace fastpls_svd {

namespace {

[[noreturn]] void throw_metal_unavailable() {
  throw std::runtime_error(
    "Metal backend is only available on macOS builds with Apple Metal support; "
    "no CPU fallback is performed"
  );
}

} // namespace

bool has_metal_backend() {
  return false;
}

struct MetalFloatOperator::Impl {};

MetalFloatOperator::MetalFloatOperator(const arma::fmat&) {
  throw_metal_unavailable();
}

MetalFloatOperator::~MetalFloatOperator() = default;

void MetalFloatOperator::update(const arma::fmat&) {
  throw_metal_unavailable();
}

arma::fmat MetalFloatOperator::multiply(const arma::fmat&, bool) {
  throw_metal_unavailable();
}

void MetalFloatOperator::geometry(const arma::fmat&, arma::fmat&, arma::fmat&) {
  throw_metal_unavailable();
}

struct MetalFloatCrossproduct::Impl {};
MetalFloatCrossproduct::MetalFloatCrossproduct(const arma::fmat&, const arma::fmat&) {
  throw_metal_unavailable();
}
MetalFloatCrossproduct::MetalFloatCrossproduct(const arma::mat&, const arma::mat&) {
  throw_metal_unavailable();
}
MetalFloatCrossproduct::~MetalFloatCrossproduct() = default;
arma::fmat MetalFloatCrossproduct::multiply(const arma::fmat&, bool) {
  throw_metal_unavailable();
}
arma::mat MetalFloatCrossproduct::multiply(const arma::mat&, bool) {
  throw_metal_unavailable();
}
void MetalFloatCrossproduct::deflate(const arma::fvec&) {
  throw_metal_unavailable();
}
arma::fmat MetalFloatCrossproduct::left_factor() {
  throw_metal_unavailable();
}

arma::mat metal_matrix_multiply(const arma::mat& A, const arma::mat& B) {
  (void) A;
  (void) B;
  throw_metal_unavailable();
}

arma::mat metal_matrix_multiply(const arma::mat& A,
                                const arma::mat& B,
                                bool transpose_left,
                                bool transpose_right) {
  (void) A;
  (void) B;
  (void) transpose_left;
  (void) transpose_right;
  throw_metal_unavailable();
}

arma::fmat metal_matrix_multiply_float(const arma::fmat& A,
                                       const arma::fmat& B,
                                       bool transpose_left,
                                       bool transpose_right) {
  (void) A;
  (void) B;
  (void) transpose_left;
  (void) transpose_right;
  throw_metal_unavailable();
}

fastpls::core::Matrix<float> metal_core_gemm_f32(
    fastpls::core::ConstMatrixView<float>,
    fastpls::core::ConstMatrixView<float>,
    bool,
    bool) {
  throw_metal_unavailable();
}

arma::mat metal_crossprod(const arma::mat& A, const arma::mat& B) {
  (void) A;
  (void) B;
  throw_metal_unavailable();
}

Rcpp::List metal_simpls_resident(const arma::mat& X,
                                 const arma::mat& Y,
                                 int ncomp,
                                 int power_iters,
                                 int seed) {
  (void) X;
  (void) Y;
  (void) ncomp;
  (void) power_iters;
  (void) seed;
  throw_metal_unavailable();
}

} // namespace fastpls_svd
