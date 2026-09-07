// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_NATIVE_OPERATOR_RSVD_HPP
#define FASTPLS_NATIVE_OPERATOR_RSVD_HPP
#include <fastpls/native/operators.hpp>

namespace fastpls { namespace native {

template<typename T>
struct OperatorRsvdWorkspace {
  arma::Mat<T> omega, sample, reverse, Qy, Ry, Qz, Rz, Q, R, projected, U, V;
  arma::Col<T> singular;
};

template<typename T, class Operator>
SingularTriplets<T> operator_rsvd(Operator& A, int k, const RsvdControls& controls,
                                 OperatorRsvdWorkspace<T>& work,
                                 bool project_full_rank = false) {
  const arma::uword maximum = std::min(A.n_rows, A.n_cols);
  const arma::uword target = std::min(maximum, arma::uword(std::max(k, 1)));
  if (target == 0) return {};
  const arma::uword width = std::min(maximum,
    target + arma::uword(std::max(controls.oversample, 0)));
  if (width >= maximum && !project_full_rank) {
    A.full_svd(work.U, work.singular, work.V, controls.left_only);
  } else {
    std::mt19937 generator(controls.seed);
    std::normal_distribution<T> normal(T(0), T(1));
    work.omega.set_size(A.n_cols, width);
    for (auto& value : work.omega) value = normal(generator);
    work.sample = A.multiply(work.omega);
    for (int iteration = 0; iteration < std::max(controls.power, 0); ++iteration) {
      A.qr(work.Qy, work.Ry, work.sample);
      work.reverse = A.multiply(work.Qy, true);
      A.qr(work.Qz, work.Rz, work.reverse);
      work.sample = A.multiply(work.Qz);
    }
    A.qr(work.Q, work.R, work.sample);
    work.projected = A.reduced(work.Q);
    if (!arma::svd_econ(work.U, work.singular, work.V, work.projected,
                        controls.left_only ? "left" : "both")) {
      throw std::runtime_error("rSVD reduced decomposition failed");
    }
    work.U = work.Q * work.U;
  }
  if (work.U.n_cols < target || work.singular.n_elem < target ||
      (!controls.left_only && work.V.n_cols < target)) {
    throw std::runtime_error("rSVD operator returned fewer directions than requested");
  }
  SingularTriplets<T> out;
  out.U = work.U.head_cols(target);
  out.s = work.singular.head(target);
  if (!controls.left_only) out.Vt = work.V.head_cols(target).t();
  return out;
}

template<class Operator>
auto intermediate_rows(const Operator& op, int) -> decltype(op.workspace_rows()) {
  return op.workspace_rows();
}
template<class Operator>
arma::uword intermediate_rows(const Operator&, long) { return 0; }

template<typename T, class Operator>
long double recovery_workspace_bytes(const Operator& op, arma::uword width) {
  // Bound the major simultaneously live sketches, operator intermediate and
  // reduced SVD buffers. This is not complete-process RSS or allocator accounting.
  const long double rows = static_cast<long double>(op.n_rows) + op.n_cols +
    intermediate_rows(op, 0);
  const long double w = width;
  return sizeof(T) * (8 * rows * w + 16 * w * w);
}

template<typename T>
struct RsvdRecovery {
  SingularTriplets<T> result;
  RsvdControls effective;
  int attempts = 0;
  long double estimated_workspace_bytes = 0;
};

template<typename T, class Operator, class Check>
RsvdRecovery<T> recover_operator_rsvd(
    Operator& op, int k, RsvdControls previous, OperatorRsvdWorkspace<T>& work,
    Check&& check, long double workspace_limit = 256.L * 1024 * 1024) {
  const arma::uword maximum = std::min(op.n_rows, op.n_cols);
  const arma::uword target = std::min(maximum, arma::uword(std::max(k, 1)));
  if (target == 0) throw std::invalid_argument("rSVD recovery requires a nonempty operator");
  arma::uword lower = 0, upper = maximum;
  while (lower < upper) {
    const arma::uword middle = lower + (upper - lower + 1) / 2;
    if (recovery_workspace_bytes<T>(op, middle) <= workspace_limit) lower = middle;
    else upper = middle - 1;
  }
  if (lower < target) {
    throw std::runtime_error("rSVD recovery cannot fit the requested rank within its workspace budget");
  }
  arma::uword width = std::min(maximum, target + arma::uword(std::max(previous.oversample, 0)));
  RsvdRecovery<T> recovery;
  for (int attempt = 0; attempt < 4; ++attempt) {
    width = std::min(lower, std::max(width + std::min(arma::uword(32), maximum - width),
                                    width + std::min(width, maximum - width)));
    RsvdControls controls = previous;
    controls.oversample = static_cast<int>(width - target);
    controls.power = std::max(previous.power, 6) + 2 * attempt;
    controls.seed = previous.seed + 104729U * static_cast<unsigned int>(attempt + 1);
    controls.left_only = false;
    recovery.result = operator_rsvd<T>(op, k, controls, work, true);
    recovery.effective = controls;
    recovery.attempts = attempt + 1;
    recovery.estimated_workspace_bytes = recovery_workspace_bytes<T>(op, width);
    if (check(recovery.result)) return recovery;
  }
  throw std::runtime_error(
    "rSVD recovery did not meet numerical tolerances within its workspace/iteration budget; "
    "no unchecked result was returned");
}

} } // namespace fastpls::native
#endif
