#include "svd_iface.h"
#include <fastpls/native/operator_rsvd.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

#ifdef FASTPLS_HAS_CUDA
#include "svd_cuda_rsvd.h"
#endif

namespace fastpls_svd {

namespace {
thread_local RSVDAuditSummary rsvd_audit_summary;
}

void reset_rsvd_audit_summary() {
  rsvd_audit_summary = RSVDAuditSummary();
}

void record_rsvd_audit_result(const SVDResult& result, bool failure) {
  ++rsvd_audit_summary.solves;
  if (failure) {
    ++rsvd_audit_summary.failures;
    return;
  }
  if (result.case_certified) ++rsvd_audit_summary.certified;
  if (result.deterministic_fallback) ++rsvd_audit_summary.deterministic_fallbacks;
  rsvd_audit_summary.max_attempts = std::max(
    rsvd_audit_summary.max_attempts, result.audit_attempts
  );
  rsvd_audit_summary.max_effective_oversample = std::max(
    rsvd_audit_summary.max_effective_oversample, result.effective_oversample
  );
  rsvd_audit_summary.max_effective_power_iters = std::max(
    rsvd_audit_summary.max_effective_power_iters, result.effective_power_iters
  );
  rsvd_audit_summary.max_triplet_residual = std::max(
    rsvd_audit_summary.max_triplet_residual, result.audit_triplet_residual
  );
  rsvd_audit_summary.max_omitted_direction_ratio = std::max(
    rsvd_audit_summary.max_omitted_direction_ratio,
    result.audit_omitted_direction_ratio
  );
}

RSVDAuditSummary current_rsvd_audit_summary() {
  return rsvd_audit_summary;
}

namespace {

SVDResult raw_rsvd(const Mat& A, int k, const SVDOptions& opt, Backend backend) {
  if (backend == Backend::CUDA) {
#ifdef FASTPLS_HAS_CUDA
    return truncated_svd_cuda_rsvd(A, k, opt);
#else
    throw std::runtime_error(
      "CUDA backend requested but fastPLS was built without CUDA support; "
      "no CPU fallback is performed"
    );
#endif
  }
  return truncated_svd_cpu_rsvd(A, k, opt);
}

bool finite_result(const SVDResult& x) {
  return x.U.is_finite() && x.s.is_finite() &&
    (x.Vt.n_elem == 0 || x.Vt.is_finite()) && x.U.n_cols > 0 && x.s.n_elem > 0;
}

double relative_triplet_residual(const Mat& A, const SVDResult& x) {
  if (x.Vt.n_rows < 1 || x.U.n_cols < 1 || x.s.n_elem < 1) {
    return std::numeric_limits<double>::infinity();
  }
  const arma::uword r = std::min(x.U.n_cols, std::min(x.Vt.n_rows, x.s.n_elem));
  const double scale = std::max(x.s(0), std::numeric_limits<double>::epsilon());
  double worst = 0.0;
  for (arma::uword j = 0; j < r; ++j) {
    const arma::vec u = x.U.col(j);
    const arma::vec v = x.Vt.row(j).t();
    const double sj = x.s(j);
    const double residual = std::max(
      arma::norm(A * v - sj * u, 2),
      arma::norm(A.t() * u - sj * v, 2)
    ) / scale;
    worst = std::max(worst, residual);
  }
  return worst;
}

double omitted_direction_ratio(
  const Mat& A,
  const SVDResult& x,
  arma::uword retained,
  unsigned int seed
) {
  retained = std::min(retained, x.U.n_cols);
  if (retained < 1 || x.s.n_elem < retained) {
    return std::numeric_limits<double>::infinity();
  }
  const arma::mat U = x.U.cols(0, retained - 1);
  const double boundary = std::max(
    x.s(retained - 1), std::numeric_limits<double>::epsilon()
  );
  std::mt19937 rng(seed);
  std::normal_distribution<double> normal(0.0, 1.0);
  double largest = 0.0;
  for (int probe = 0; probe < 3; ++probe) {
    arma::vec g(A.n_cols);
    for (arma::uword i = 0; i < g.n_elem; ++i) g(i) = normal(rng);
    g /= std::max(arma::norm(g, 2), std::numeric_limits<double>::epsilon());
    arma::vec y;
    for (int iteration = 0; iteration < 2; ++iteration) {
      y = A * g;
      y -= U * (U.t() * y);
      const double ynorm = arma::norm(y, 2);
      if (!std::isfinite(ynorm) || ynorm <= std::numeric_limits<double>::epsilon()) {
        break;
      }
      y /= ynorm;
      g = A.t() * y;
      g /= std::max(arma::norm(g, 2), std::numeric_limits<double>::epsilon());
    }
    y = A * g;
    y -= U * (U.t() * y);
    largest = std::max(largest, arma::norm(y, 2));
  }
  return largest / boundary;
}

void trim_result(SVDResult& x, arma::uword retained, bool left_only) {
  retained = std::min(retained, std::min(x.U.n_cols, x.s.n_elem));
  if (retained < 1) return;
  x.U = x.U.cols(0, retained - 1);
  x.s = x.s.head(retained);
  if (left_only) {
    x.Vt.reset();
  } else if (x.Vt.n_rows >= retained) {
    x.Vt = x.Vt.rows(0, retained - 1);
  }
}

bool rsvd_a_posteriori_check(
  const Mat& A,
  const SVDResult& x,
  arma::uword retained,
  unsigned int seed,
  double& triplet_residual,
  double& omitted_ratio,
  bool& weak_boundary
) {
  if (!finite_result(x) || x.U.n_cols < retained || x.s.n_elem < retained) return false;
  triplet_residual = relative_triplet_residual(A, x);
  omitted_ratio = omitted_direction_ratio(A, x, retained, seed + 32452843U);
  weak_boundary = false;
  if (x.s.n_elem > retained) {
    const double boundary = std::max(
      x.s(retained - 1), std::numeric_limits<double>::epsilon()
    );
    const double boundary_ratio = x.s(retained) / boundary;
    omitted_ratio = std::max(omitted_ratio, boundary_ratio);
    weak_boundary = boundary_ratio > 0.95;
  }
  // A ratio near one indicates a weak spectral boundary, not an inaccurate
  // retained triplet. Reject only when the probe finds an omitted direction
  // materially stronger than the retained boundary.
  return triplet_residual <= 1e-2 && omitted_ratio <= 1.01;
}

bool rsvd_consensus(
  const Mat& A,
  const SVDResult& lhs,
  const SVDResult& rhs,
  double& subspace_error,
  double& singular_value_error,
  double& triplet_residual
) {
  if (!finite_result(lhs) || !finite_result(rhs)) return false;
  const arma::uword r = std::min(lhs.U.n_cols, rhs.U.n_cols);
  if (r < 1 || lhs.U.n_cols != rhs.U.n_cols || lhs.s.n_elem != rhs.s.n_elem) return false;

  const arma::mat cross = lhs.U.cols(0, r - 1).t() * rhs.U.cols(0, r - 1);
  const double overlap_sq = std::min<double>(
    static_cast<double>(r), arma::accu(arma::square(cross))
  );
  subspace_error = std::sqrt(std::max(0.0, static_cast<double>(r) - overlap_sq) /
    static_cast<double>(r));

  const arma::uword ns = std::min(lhs.s.n_elem, rhs.s.n_elem);
  const double scale = std::max(rhs.s(0), std::numeric_limits<double>::epsilon());
  singular_value_error = arma::abs(lhs.s.head(ns) - rhs.s.head(ns)).max() / scale;
  triplet_residual = relative_triplet_residual(A, rhs);

  return subspace_error <= 1e-3 && singular_value_error <= 1e-5 &&
    triplet_residual <= 1e-6;
}

SVDResult audited_rsvd(const Mat& A, int k, const SVDOptions& requested, Backend backend) {
  const arma::uword max_rank = std::min(A.n_rows, A.n_cols);
  const arma::uword retained = std::min<arma::uword>(
    max_rank, static_cast<arma::uword>(std::max(k, 1))
  );
  const int audit_rank = static_cast<int>(std::min(max_rank, retained + 1));
  SVDOptions first_opt = requested;
  first_opt.left_only = false;
  SVDResult first = raw_rsvd(A, audit_rank, first_opt, backend);

  double triplet_residual = std::numeric_limits<double>::infinity();
  double omitted_ratio = std::numeric_limits<double>::infinity();
  bool weak_boundary = false;
  const bool first_pass = rsvd_a_posteriori_check(
    A, first, retained, first_opt.seed, triplet_residual, omitted_ratio, weak_boundary
  );
  if (first_pass) {
    first.randomized = true;
    first.case_audited = true;
    first.case_certified = true;
    first.audit_attempts = 1;
    first.effective_oversample = first_opt.oversample;
    first.effective_power_iters = first_opt.power_iters;
    first.effective_seed = first_opt.seed;
    first.audit_subspace_error = 0.0;
    first.audit_singular_value_error = 0.0;
    first.audit_triplet_residual = triplet_residual;
    first.audit_omitted_direction_ratio = omitted_ratio;
    trim_result(first, retained, requested.left_only);
    record_rsvd_audit_result(first);
    return first;
  }

  SVDOptions second_opt = first_opt;
  second_opt.oversample = std::max(first_opt.oversample, 32);
  second_opt.power_iters = std::max(first_opt.power_iters, 3);
  second_opt.seed = first_opt.seed + 104729U;
  SVDResult second = raw_rsvd(A, audit_rank, second_opt, backend);

  double subspace_error = std::numeric_limits<double>::infinity();
  double singular_value_error = std::numeric_limits<double>::infinity();
  double second_omitted_ratio = std::numeric_limits<double>::infinity();
  bool second_weak_boundary = false;
  const bool second_pass = rsvd_a_posteriori_check(
    A, second, retained, second_opt.seed, triplet_residual,
    second_omitted_ratio, second_weak_boundary
  );
  const bool second_consensus = rsvd_consensus(
    A, first, second, subspace_error, singular_value_error, triplet_residual
  );
  // A weak retained/omitted boundary is not itself a numerical failure when
  // two independent sketches recover the same retained subspace and spectrum.
  bool certified = second_pass || second_consensus;

  SVDResult accepted = second;
  int attempts = 2;
  SVDOptions accepted_opt = second_opt;
  if (!certified) {
    SVDOptions third_opt = second_opt;
    third_opt.oversample = std::max(second_opt.oversample, 48);
    third_opt.power_iters = std::max(second_opt.power_iters, 4);
    third_opt.seed = first_opt.seed + 209759U;
    SVDResult third = raw_rsvd(A, audit_rank, third_opt, backend);
    double third_omitted_ratio = std::numeric_limits<double>::infinity();
    bool third_weak_boundary = false;
    const bool third_pass = rsvd_a_posteriori_check(
      A, third, retained, third_opt.seed, triplet_residual,
      third_omitted_ratio, third_weak_boundary
    );
    const bool third_consensus = rsvd_consensus(
      A, second, third, subspace_error, singular_value_error, triplet_residual
    );
    certified = third_pass || third_consensus;
    accepted = third;
    accepted_opt = third_opt;
    second_omitted_ratio = third_omitted_ratio;
    attempts = 3;
  }

  if (!certified) {
    if (backend != Backend::CPU) {
      record_rsvd_audit_result(accepted, true);
      throw std::runtime_error(
        "rSVD case audit did not converge across independent strengthened sketches; "
        "no accelerator result was returned and no CPU fallback was performed."
      );
    }
    fastpls::native::MatrixOperator<double> op(A);
    fastpls::native::OperatorRsvdWorkspace<double> workspace;
    fastpls::native::RsvdControls controls;
    controls.oversample = accepted_opt.oversample;
    controls.power = accepted_opt.power_iters;
    controls.seed = accepted_opt.seed;
    auto check = [&](const fastpls::native::SingularTriplets<double>& candidate) {
      accepted.U = candidate.U;
      accepted.s = candidate.s;
      accepted.Vt = candidate.Vt;
      bool weak = false;
      return rsvd_a_posteriori_check(
        A, accepted, retained, controls.seed, triplet_residual,
        second_omitted_ratio, weak
      );
    };
    try {
      const auto recovery = fastpls::native::recover_operator_rsvd<double>(
        op, audit_rank, controls, workspace, check
      );
      attempts += recovery.attempts;
      accepted_opt.oversample = recovery.effective.oversample;
      accepted_opt.power_iters = recovery.effective.power;
      accepted_opt.seed = recovery.effective.seed;
      subspace_error = singular_value_error = 0.0;
    } catch (...) {
      record_rsvd_audit_result(accepted, true);
      throw;
    }
  }

  accepted.randomized = true;
  accepted.case_audited = true;
  accepted.case_certified = true;
  accepted.audit_attempts = attempts;
  accepted.effective_oversample = accepted_opt.oversample;
  accepted.effective_power_iters = accepted_opt.power_iters;
  accepted.effective_seed = accepted_opt.seed;
  accepted.audit_subspace_error = subspace_error;
  accepted.audit_singular_value_error = singular_value_error;
  accepted.audit_triplet_residual = triplet_residual;
  accepted.audit_omitted_direction_ratio = second_omitted_ratio;
  trim_result(accepted, retained, requested.left_only);
  record_rsvd_audit_result(accepted);
  return accepted;
}

} // namespace

SVDOptions options_from_method_id(
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  unsigned int seed,
  bool left_only,
  bool use_full_svd
) {
  SVDOptions opt;
  opt.oversample = std::max(rsvd_oversample, 0);
  opt.power_iters = std::max(rsvd_power, 0);
  opt.svds_tol = std::max(svds_tol, 0.0);
  opt.seed = seed;
  opt.left_only = left_only;
  opt.use_full_svd = use_full_svd;

  switch (svd_method) {
    case 1:
      throw std::runtime_error("IRLBA is not part of fastPLS");
    case SVD_METHOD_CPU_RSVD:
    case SVD_METHOD_CUDA_RSVD:
      opt.method = Method::RSVD;
      break;
    case SVD_METHOD_CPU_EXACT:
      opt.method = Method::EXACT;
      break;
    default:
      throw std::runtime_error("Unsupported internal SVD method");
  }

  return opt;
}

Backend backend_from_method_id(int svd_method) {
  switch (svd_method) {
    case SVD_METHOD_CUDA_RSVD:
      return Backend::CUDA;
    default:
      return Backend::CPU;
  }
}

SVDResult truncated_svd(const Mat& A, int k, const SVDOptions& opt, Backend backend) {
  if (k < 1) {
    throw std::runtime_error("truncated_svd: k must be >= 1");
  }
  if (backend != Backend::CPU && backend != Backend::CUDA) {
    throw std::runtime_error("Unsupported SVD backend; no CPU fallback is performed");
  }
  if (backend == Backend::CUDA && !has_cuda_backend()) {
    throw std::runtime_error(
      "CUDA backend requested but no CUDA device is available; "
      "no CPU fallback is performed"
    );
  }

  const arma::uword min_dim = std::min(A.n_rows, A.n_cols);
  const bool force_exact = opt.use_full_svd;
  if (force_exact || min_dim < 6) {
    SVDOptions full_opt = opt;
    full_opt.method = Method::EXACT;
    full_opt.use_full_svd = true;
    SVDResult exact = truncated_svd_cpu_exact(A, k, full_opt);
    if (opt.method == Method::RSVD) {
      exact.randomized = true;
      exact.case_audited = true;
      exact.case_certified = true;
      exact.deterministic_fallback = true;
      exact.audit_attempts = 0;
      exact.effective_oversample = opt.oversample;
      exact.effective_power_iters = opt.power_iters;
      exact.effective_seed = opt.seed;
      exact.audit_subspace_error = 0.0;
      exact.audit_singular_value_error = 0.0;
      exact.audit_triplet_residual = 0.0;
      exact.audit_omitted_direction_ratio = 0.0;
      record_rsvd_audit_result(exact);
    }
    return exact;
  }

  if (backend == Backend::CUDA) {
#ifdef FASTPLS_HAS_CUDA
    if (opt.method != Method::RSVD) {
      throw std::runtime_error(
        "the requested SVD method is not available on the CUDA backend; "
        "no CPU fallback is performed"
      );
    }
    return audited_rsvd(A, k, opt, backend);
#else
    throw std::runtime_error(
      "CUDA backend requested but fastPLS was built without CUDA support; "
      "no CPU fallback is performed"
    );
#endif
  }

  if (opt.method == Method::RSVD) {
    return audited_rsvd(A, k, opt, backend);
  }

  return truncated_svd_cpu_exact(A, k, opt);
}

bool has_cuda_backend() {
#ifdef FASTPLS_HAS_CUDA
  return cuda_runtime_available();
#else
  return false;
#endif
}

} // namespace fastpls_svd
