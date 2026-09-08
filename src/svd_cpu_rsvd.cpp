#include "svd_iface.h"
#include "core_cpu_backend.h"

#include <fastpls/core/rsvd.hpp>

#include <utility>

namespace fastpls_svd {
namespace {

fastpls::core::ConstMatrixView<double> core_view(const Mat& matrix) {
  return fastpls::core::make_const_view(
    matrix.memptr(), matrix.n_rows, matrix.n_cols, matrix.n_rows
  );
}

Mat arma_matrix(const fastpls::core::Matrix<double>& matrix) {
  if (matrix.size() == 0) return Mat(matrix.rows(), matrix.columns());
  return Mat(matrix.data(), matrix.rows(), matrix.columns());
}

Vec arma_vector(const std::vector<double>& values) {
  if (values.empty()) return Vec();
  return Vec(values.data(), values.size());
}

SVDResult from_core(fastpls::core::SingularTriplets<double>&& result) {
  SVDResult out;
  out.U = arma_matrix(result.U);
  out.s = arma_vector(result.singular_values);
  out.Vt = arma_matrix(result.Vt);
  return out;
}

} // namespace

SVDResult finalize_rsvd_from_sample(const Mat& A, const Mat& Y, int k, bool left_only) {
  fastpls::runtime::CpuLinearAlgebraF64 backend;
  return from_core(fastpls::core::finalize_rsvd_sample(
    core_view(A), core_view(Y), k, left_only, backend
  ));
}

SVDResult truncated_svd_cpu_rsvd(const Mat& A, int k, const SVDOptions& opt) {
  fastpls::core::RsvdControls controls;
  controls.oversample = opt.oversample;
  controls.power = opt.power_iters;
  controls.seed = opt.seed;
  controls.left_only = opt.left_only;
  fastpls::runtime::CpuLinearAlgebraF64 backend;
  return from_core(fastpls::core::randomized_svd(
    core_view(A), k, controls, backend
  ));
}

} // namespace fastpls_svd
