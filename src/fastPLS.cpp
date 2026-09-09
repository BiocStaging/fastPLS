

#include <RcppArmadillo.h>
#include <R_ext/Rdynload.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <limits>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fastPLS.h"
#include "core_cpu_backend.h"
#include "svd_iface.h"
#include "svd_cuda_rsvd.h"
#include "svd_metal_backend.h"
#include "float_crosscov_operator.h"
#include <fastpls/core/lda.hpp>
#include <fastpls/core/opls.hpp>
#include <fastpls/core/rsvd.hpp>
#include <fastpls/core/simpls.hpp>
#include <fastpls/core/statistics.hpp>
#include <fastpls/native/simpls.hpp>
#include <fastpls/native/opls.hpp>
#include <fastpls/native/kernels.hpp>
#include <fastpls/native/plssvd.hpp>
#include <fastpls/native/operator_rsvd.hpp>

// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

namespace {

constexpr int kCpuSimplsBlockSize = 64;
constexpr int kAcceleratorSimplsBlockSize = 64;
#ifdef FASTPLS_USE_ACCELERATE
constexpr int kCpuCrossprodMaxPredictors = 2048;
#else
constexpr int kCpuCrossprodMaxPredictors = 512;
#endif

int accelerated_simpls_block_size(
  const int remaining,
  const int p,
  const int m,
  const bool classification_response = false,
  const int n = 0,
  const int maximum_block_size = kCpuSimplsBlockSize
) {
  // Batched candidate refresh amortizes decomposition and device-launch costs
  // when repeated response-wide products dominate component extraction.
  const double response_work =
    static_cast<double>(std::max(n, 0)) *
    static_cast<double>(std::max(p, 0)) *
    static_cast<double>(std::max(m, 0));
  if (m <= 1 || remaining < 4 || response_work < 5.0e8) {
    return 1;
  }
  if (!classification_response) {
    const double crosscov_elements =
      static_cast<double>(std::max(p, 0)) *
      static_cast<double>(std::max(m, 0));
    if (crosscov_elements <= 64.0 * 1024.0 * 1024.0) return 1;
    return std::max(
      1,
      std::min({8, maximum_block_size, remaining, p, m})
    );
  }
  if (m > 2048) return 1;
  return std::max(
    1,
    std::min({maximum_block_size, remaining, p, m})
  );
}

using fastpls::native::is_one_hot_response;

void dense_product_into(
  const arma::mat& left,
  const arma::mat& right,
  arma::mat& output
) {
  if (left.n_cols != right.n_rows ||
      output.n_rows != left.n_rows || output.n_cols != right.n_cols) {
    stop("Internal dense-product dimensions are inconsistent");
  }
  const char no_transpose = 'N';
  const arma::blas_int rows = static_cast<arma::blas_int>(left.n_rows);
  const arma::blas_int columns = static_cast<arma::blas_int>(right.n_cols);
  const arma::blas_int inner = static_cast<arma::blas_int>(left.n_cols);
  const double alpha = 1.0;
  const double beta = 0.0;
  arma::blas::gemm<double>(
    &no_transpose,
    &no_transpose,
    &rows,
    &columns,
    &inner,
    &alpha,
    left.memptr(),
    &rows,
    right.memptr(),
    &inner,
    &beta,
    output.memptr(),
    &rows
  );
}

fastpls_svd::SVDResult compute_truncated_svd_dispatch(
  const arma::mat& S,
  int k,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  unsigned int seed,
  bool left_only,
  bool use_full_svd
);

int env_int_or(const char* key, int fallback, int lo, int hi) {
  const char* raw = std::getenv(key);
  if (raw == nullptr) return fallback;
  char* endptr = nullptr;
  long v = std::strtol(raw, &endptr, 10);
  if (endptr == raw) return fallback;
  if (v < lo) v = lo;
  if (v > hi) v = hi;
  return static_cast<int>(v);
}

arma::fmat float_matrix_multiply_cpu(
  const arma::fmat& left,
  const arma::fmat& right,
  const bool transpose_left = false,
  const bool transpose_right = false
) {
  const arma::uword rows = transpose_left ? left.n_cols : left.n_rows;
  const arma::uword inner_left = transpose_left ? left.n_rows : left.n_cols;
  const arma::uword inner_right = transpose_right ? right.n_cols : right.n_rows;
  const arma::uword columns = transpose_right ? right.n_rows : right.n_cols;
  if (inner_left != inner_right) {
    Rcpp::stop("Internal float32 matrix-product dimensions are inconsistent");
  }
  arma::fmat output(rows, columns, arma::fill::none);
  fastpls::runtime::cpu_gemm_f32(
    fastpls::core::make_const_view(
      left.memptr(), static_cast<std::size_t>(left.n_rows),
      static_cast<std::size_t>(left.n_cols),
      static_cast<std::size_t>(left.n_rows)
    ),
    fastpls::core::make_const_view(
      right.memptr(), static_cast<std::size_t>(right.n_rows),
      static_cast<std::size_t>(right.n_cols),
      static_cast<std::size_t>(right.n_rows)
    ),
    transpose_left, transpose_right,
    fastpls::core::make_view(
      output.memptr(), static_cast<std::size_t>(output.n_rows),
      static_cast<std::size_t>(output.n_cols),
      static_cast<std::size_t>(output.n_rows)
    )
  );
  return output;
}

void float_sample_geometry_cpu(
  const arma::fmat& predictors,
  const arma::fmat& directions,
  arma::fmat& scores,
  arma::fmat& loadings
) {
  scores = float_matrix_multiply_cpu(predictors, directions);
  loadings = float_matrix_multiply_cpu(predictors, scores, true, false);
}

bool should_store_coefficients(
  const int p,
  const int m,
  const int n_slices,
  const bool compact_prediction_available
) {
  const char* mode = std::getenv("FASTPLS_STORE_B");
  if (mode != nullptr) {
    std::string value(mode);
    for (char& c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (value == "always" || value == "1" || value == "true" || value == "yes") return true;
    if (value == "never" || value == "0" || value == "false" || value == "no") return false;
  }
  if (!compact_prediction_available) return true;
  const int max_mb = env_int_or("FASTPLS_STORE_B_MAX_MB", 256, 0, 1048576);
  const double b_mb =
    static_cast<double>(p) *
    static_cast<double>(m) *
    static_cast<double>(std::max(n_slices, 1)) *
    static_cast<double>(sizeof(double)) /
    (1024.0 * 1024.0);
  return b_mb <= static_cast<double>(max_mb);
}

void annotate_coefficient_storage(Rcpp::List& out, const bool store_B) {
  out["B_stored"] = store_B;
  out["compact_prediction"] = !store_B;
}

arma::mat gaussian_matrix_local(
  const arma::uword n_rows,
  const arma::uword n_cols,
  const unsigned int seed
) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> norm(0.0, 1.0);

  arma::mat out(n_rows, n_cols);
  double* ptr = out.memptr();
  const arma::uword n_elem = out.n_elem;
  for (arma::uword i = 0; i < n_elem; ++i) {
    ptr[i] = norm(rng);
  }

  return out;
}

arma::vec leading_left_vec_dispatch(
  const arma::mat& S,
  const int svd_method,
  const int rsvd_oversample,
  const int rsvd_power,
  const double svds_tol,
  const unsigned int seed
) {
  if (S.n_rows < 1 || S.n_cols < 1) {
    return arma::vec();
  }

  fastpls_svd::SVDResult svd_res = compute_truncated_svd_dispatch(
    S,
    1,
    svd_method,
    rsvd_oversample,
    rsvd_power,
    svds_tol,
    seed,
    true,
    false
  );
  if (svd_res.U.n_cols < 1) {
    return arma::vec();
  }
  return svd_res.U.col(0);
}

bool finalize_left_block_from_bsmall(
  const arma::mat& Bsmall,
  arma::mat& Uhat,
  arma::vec& shat,
  arma::mat& Vhat
) {
  if (Bsmall.n_rows < 1 || Bsmall.n_cols < 1) {
    return false;
  }

  const int eig_threshold = env_int_or("FASTPLS_GPU_FINALIZE_THRESHOLD", 4, 1, 256);
  if (static_cast<int>(Bsmall.n_rows) >= eig_threshold) {
    arma::mat gram = Bsmall * Bsmall.t();
    arma::vec evals;
    arma::mat evecs;
    const bool ok = arma::eig_sym(evals, evecs, gram);
    if (ok && evals.n_elem > 0) {
      arma::uvec ord = arma::sort_index(evals, "descend");
      Uhat = evecs.cols(ord);
      shat = arma::sqrt(arma::clamp(evals(ord), 0.0, std::numeric_limits<double>::infinity()));
      Vhat.reset();
      return true;
    }
  }

  arma::svd_econ(Uhat, shat, Vhat, Bsmall, "left");
  return Uhat.n_cols > 0;
}

fastpls_svd::SVDResult compute_truncated_svd_dispatch(
  const arma::mat& S,
  int k,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  unsigned int seed,
  bool left_only,
  bool use_full_svd
) {
  fastpls_svd::SVDOptions opt = fastpls_svd::options_from_method_id(
    svd_method,
    rsvd_oversample,
    rsvd_power,
    svds_tol,
    seed,
    left_only,
    use_full_svd
  );

  const fastpls_svd::Backend backend = fastpls_svd::backend_from_method_id(svd_method);
  return fastpls_svd::truncated_svd(S, k, opt, backend);
}

bool plssvd_use_small_exact_svd(const int max_rank, const int svd_method) {
  const int threshold = env_int_or("FASTPLS_PLSSVD_SMALL_EXACT_MAX_RANK", 32, 5, 512);
  return max_rank <= threshold;
}

arma::mat numeric_matrix_view(SEXP x, const char* name) {
  if (!Rf_isReal(x) || !Rf_isMatrix(x)) {
    Rcpp::stop("%s must be a numeric matrix", name);
  }
  Rcpp::NumericMatrix rx(x);
  return arma::mat(
    REAL(rx),
    static_cast<arma::uword>(rx.nrow()),
    static_cast<arma::uword>(rx.ncol()),
    false,
    true
  );
}

fastpls_svd::SVDResult finalize_rsvd_from_q_b_double(
  const arma::mat& Q,
  const arma::mat& B,
  const int k,
  const bool left_only
) {
  fastpls_svd::SVDResult out;
  const arma::uword max_rank = std::min(B.n_rows, B.n_cols);
  const arma::uword rank = std::min<arma::uword>(
    max_rank,
    static_cast<arma::uword>(std::max(k, 1))
  );
  if (rank == 0 || Q.n_cols < 1 || B.n_rows < 1 || B.n_cols < 1) {
    return out;
  }

  arma::mat Uhat;
  arma::vec s;
  arma::mat V;
  if (left_only) {
    arma::svd_econ(Uhat, s, V, B, "left");
  } else {
    arma::svd_econ(Uhat, s, V, B, "both");
  }

  arma::uword actual = std::min<arma::uword>(rank, std::min<arma::uword>(Uhat.n_cols, s.n_elem));
  if (!left_only) {
    actual = std::min<arma::uword>(actual, V.n_cols);
  }
  if (actual == 0) {
    return out;
  }

  out.U = Q * Uhat.cols(0, actual - 1);
  out.s = s.subvec(0, actual - 1);
  if (!left_only) {
    out.Vt = V.cols(0, actual - 1).t();
  }
  return out;
}

template <typename ATimes, typename ATTimes>
fastpls_svd::SVDResult raw_rsvd_operator_double(
  const arma::uword p,
  const arma::uword m,
  const int k,
  const int oversample,
  const int power,
  const unsigned int seed,
  const bool left_only,
  const ATimes& a_times,
  const ATTimes& at_times
) {
  const arma::uword max_rank = std::min(p, m);
  const arma::uword target = std::min<arma::uword>(
    max_rank, static_cast<arma::uword>(std::max(k, 1))
  );
  const arma::uword l = std::min<arma::uword>(
    max_rank, target + static_cast<arma::uword>(std::max(oversample, 0))
  );
  arma::mat sample = a_times(gaussian_matrix_local(m, l, seed));
  for (int iteration = 0; iteration < std::max(power, 0); ++iteration) {
    arma::mat Qy;
    arma::mat Ry;
    arma::qr_econ(Qy, Ry, sample);
    arma::mat Z = at_times(Qy);
    arma::mat Qz;
    arma::mat Rz;
    arma::qr_econ(Qz, Rz, Z);
    sample = a_times(Qz);
  }
  arma::mat Q;
  arma::mat R;
  arma::qr_econ(Q, R, sample);
  if (Q.n_cols < 1) return fastpls_svd::SVDResult();
  return finalize_rsvd_from_q_b_double(
    Q, at_times(Q).t(), static_cast<int>(target), left_only
  );
}

template <typename ATimes, typename ATTimes>
bool audit_rsvd_operator_double(
  const fastpls_svd::SVDResult& result,
  const arma::uword retained,
  const unsigned int seed,
  const ATimes& a_times,
  const ATTimes& at_times,
  double& triplet_residual,
  double& omitted_ratio
) {
  (void)seed;
  if (result.U.n_cols < retained || result.Vt.n_rows < retained ||
      result.s.n_elem < retained || !result.U.is_finite() ||
      !result.Vt.is_finite() || !result.s.is_finite()) return false;
  const double spectral_scale = std::max(
    result.s(0), std::numeric_limits<double>::epsilon()
  );
  const arma::uword audit_rank = std::min(
    result.s.n_elem,
    std::min(result.U.n_cols, result.Vt.n_rows)
  );
  const arma::mat audit_u = result.U.cols(0, audit_rank - 1);
  const arma::mat audit_v = result.Vt.rows(0, audit_rank - 1).t();
  const arma::mat scaled_u = audit_u.each_row() %
    result.s.head(audit_rank).t();
  const arma::mat scaled_v = audit_v.each_row() %
    result.s.head(audit_rank).t();
  const arma::mat left_residuals = a_times(audit_v) - scaled_u;
  const arma::mat right_residuals = at_times(audit_u) - scaled_v;
  triplet_residual = 0.0;
  for (arma::uword j = 0; j < audit_rank; ++j) {
    triplet_residual = std::max(triplet_residual, std::max(
      arma::norm(left_residuals.col(j), 2),
      arma::norm(right_residuals.col(j), 2)
    ) / spectral_scale);
  }

  const double boundary = std::max(
    result.s(retained - 1), std::numeric_limits<double>::epsilon()
  );
  omitted_ratio = result.s.n_elem > retained ?
    result.s(retained) / boundary : 0.0;
  // Near-tied boundary values are valid singular directions. The triplet
  // residual audits accuracy; this ratio only rejects a materially stronger
  // omitted direction.
  return triplet_residual <= 1e-2 && omitted_ratio <= 1.01;
}

template <typename ATimes, typename ATTimes>
bool rsvd_operator_consensus_double(
  const fastpls_svd::SVDResult& lhs,
  const fastpls_svd::SVDResult& rhs,
  const arma::uword retained,
  const double rhs_residual,
  double& subspace_error,
  double& singular_value_error
) {
  if (lhs.U.n_cols < retained || rhs.U.n_cols < retained ||
      lhs.s.n_elem < retained || rhs.s.n_elem < retained) return false;
  const arma::mat cross = lhs.U.cols(0, retained - 1).t() *
    rhs.U.cols(0, retained - 1);
  const double overlap_sq = std::min<double>(
    static_cast<double>(retained), arma::accu(arma::square(cross))
  );
  subspace_error = std::sqrt(
    std::max(0.0, static_cast<double>(retained) - overlap_sq) /
    static_cast<double>(retained)
  );
  const double scale = std::max(
    rhs.s(0), std::numeric_limits<double>::epsilon()
  );
  singular_value_error = arma::abs(
    lhs.s.head(retained) - rhs.s.head(retained)
  ).max() / scale;
  return subspace_error <= 1e-3 && singular_value_error <= 1e-5 &&
    rhs_residual <= 1e-6;
}

template <typename ATimes, typename ATTimes>
fastpls_svd::SVDResult audited_rsvd_operator_double(
  const arma::uword p,
  const arma::uword m,
  const int k,
  const int requested_oversample,
  const int requested_power,
  const unsigned int requested_seed,
  const bool left_only,
  const ATimes& a_times,
  const ATTimes& at_times
) {
  const arma::uword retained = std::min<arma::uword>(
    std::min(p, m), static_cast<arma::uword>(std::max(k, 1))
  );
  const int audit_rank = static_cast<int>(std::min(std::min(p, m), retained + 1));
  const int oversamples[] = {
    requested_oversample,
    std::max(requested_oversample, 32),
    std::max(requested_oversample, 48)
  };
  const int powers[] = {
    requested_power,
    std::max(requested_power, 3),
    std::max(requested_power, 4)
  };
  const unsigned int seeds[] = {
    requested_seed,
    requested_seed + 104729U,
    requested_seed + 209759U
  };
  fastpls_svd::SVDResult previous;
  bool have_previous = false;
  for (int attempt = 0; attempt < 3; ++attempt) {
    fastpls_svd::SVDResult result = raw_rsvd_operator_double(
      p, m, audit_rank, oversamples[attempt], powers[attempt], seeds[attempt],
      false, a_times, at_times
    );
    double residual = std::numeric_limits<double>::infinity();
    double omitted = std::numeric_limits<double>::infinity();
    const bool residual_pass = audit_rsvd_operator_double(
      result, retained, seeds[attempt], a_times, at_times, residual, omitted
    );
    double subspace_error = std::numeric_limits<double>::infinity();
    double singular_value_error = std::numeric_limits<double>::infinity();
    const bool consensus_pass = have_previous && rsvd_operator_consensus_double<ATimes, ATTimes>(
      previous, result, retained, residual, subspace_error, singular_value_error
    );
    if (!residual_pass && !consensus_pass) {
      previous = result;
      have_previous = true;
      continue;
    }

    result.randomized = true;
    result.case_audited = true;
    result.case_certified = true;
    result.audit_attempts = attempt + 1;
    result.effective_oversample = oversamples[attempt];
    result.effective_power_iters = powers[attempt];
    result.effective_seed = seeds[attempt];
    result.audit_triplet_residual = residual;
    result.audit_omitted_direction_ratio = omitted;
    result.audit_subspace_error = consensus_pass ? subspace_error : 0.0;
    result.audit_singular_value_error = consensus_pass ? singular_value_error : 0.0;
    result.U = result.U.cols(0, retained - 1);
    result.s = result.s.head(retained);
    if (left_only) result.Vt.reset();
    else result.Vt = result.Vt.rows(0, retained - 1);
    fastpls_svd::record_rsvd_audit_result(result);
    return result;
  }
  throw std::runtime_error(
    "Matrix-free rSVD failed its case-specific residual audit after three "
    "strengthened attempts; no uncertified fit was returned."
  );
}

fastpls_svd::SVDResult truncated_rsvd_crossprod_double(
  const arma::mat& X,
  const arma::mat& Ymat,
  const int k,
  const int rsvd_oversample,
  const int rsvd_power,
  const unsigned int seed,
  const bool left_only,
  const bool use_full_svd
) {
  const arma::uword p = X.n_cols;
  const arma::uword m = Ymat.n_cols;
  const arma::uword max_rank = std::min(p, m);
  const arma::uword target = std::min<arma::uword>(
    max_rank,
    static_cast<arma::uword>(std::max(k, 1))
  );
  const arma::uword l = std::min<arma::uword>(
    max_rank,
    target + static_cast<arma::uword>(std::max(rsvd_oversample, 0))
  );

  if (target == 0) {
    return fastpls_svd::SVDResult();
  }

  if (use_full_svd || l >= max_rank) {
    arma::mat S = X.t() * Ymat;
    return compute_truncated_svd_dispatch(
      S,
      static_cast<int>(target),
      fastpls_svd::SVD_METHOD_CPU_RSVD,
      rsvd_oversample,
      rsvd_power,
      0.0,
      seed,
      left_only,
      true
    );
  }

  auto a_times = [&](const arma::mat& M) -> arma::mat {
    return X.t() * (Ymat * M);
  };
  auto at_times = [&](const arma::mat& M) -> arma::mat {
    return Ymat.t() * (X * M);
  };

  return audited_rsvd_operator_double(
    p, m, static_cast<int>(target), rsvd_oversample, rsvd_power, seed,
    left_only, a_times, at_times
  );
}

arma::mat project_deflated_left_double(
  arma::mat M,
  const arma::mat& V,
  const int n_prev
) {
  if (n_prev > 0) {
    const arma::uword cols = std::min<arma::uword>(
      static_cast<arma::uword>(n_prev),
      V.n_cols
    );
    if (cols > 0) {
      const auto Vprev = V.cols(0, cols - 1);
      M -= Vprev * (Vprev.t() * M);
    }
  }
  return M;
}

bool refresh_deflated_crossprod_left_double(
  const arma::mat& X,
  const arma::mat& Ymat,
  const arma::mat& V,
  const int n_prev,
  const int k_block,
  const int oversample,
  const int power_iters,
  const unsigned int seed,
  arma::mat& Ublock,
  arma::vec& shat
) {
  const arma::uword p = X.n_cols;
  const arma::uword m = Ymat.n_cols;
  if (p < 1 || m < 1 || k_block < 1) {
    return false;
  }

  auto a_times = [&](const arma::mat& M) -> arma::mat {
    return project_deflated_left_double(X.t() * (Ymat * M), V, n_prev);
  };
  auto at_times = [&](const arma::mat& M) -> arma::mat {
    arma::mat Mp = project_deflated_left_double(M, V, n_prev);
    return Ymat.t() * (X * Mp);
  };

  const double crosscov_bytes =
    static_cast<double>(p) * static_cast<double>(m) * sizeof(double);
  const bool use_rank_one_refresh =
    k_block == 1 && crosscov_bytes > 512.0 * 1024.0 * 1024.0;
  if (use_rank_one_refresh) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    arma::vec u(p);
    for (arma::uword i = 0; i < p; ++i) u(i) = normal(rng);
    u = project_deflated_left_double(u, V, n_prev);
    double unorm = arma::norm(u, 2);
    if (!std::isfinite(unorm) || unorm <= std::numeric_limits<double>::epsilon()) {
      return false;
    }
    u /= unorm;
    double sigma = 0.0;
    for (int iteration = 0; iteration < std::max(power_iters, 1); ++iteration) {
      arma::vec right = at_times(arma::mat(u));
      const double right_norm = arma::norm(right, 2);
      if (!std::isfinite(right_norm) ||
          right_norm <= std::numeric_limits<double>::epsilon()) {
        return false;
      }
      right /= right_norm;
      u = a_times(arma::mat(right));
      u = project_deflated_left_double(u, V, n_prev);
      unorm = arma::norm(u, 2);
      if (!std::isfinite(unorm) ||
          unorm <= std::numeric_limits<double>::epsilon()) {
        return false;
      }
      u /= unorm;
      sigma = right_norm;
    }
    Ublock = arma::mat(u);
    shat = arma::vec(1, arma::fill::value(sigma));
    return true;
  }

  // Propagate decomposition/allocation failures; do not substitute a solver.
  fastpls_svd::SVDResult result = raw_rsvd_operator_double(
    p, m, k_block, std::max(oversample, 0), power_iters, seed, true,
    a_times, at_times
  );
  Ublock = result.U;
  shat = result.s;
  return Ublock.n_cols > 0;
}

struct SimplsFastRefreshWorkspace {
  arma::mat Omega;
  arma::mat Y;
  arma::mat Z;
  arma::mat Q;
  arma::mat R;
  arma::mat Bsmall;
  arma::mat Uhat;
  arma::vec shat;
  arma::mat Vhat;
  arma::mat Qy;
  arma::mat Ry;
  arma::mat Qz;
  arma::mat Rz;
  bool gpu_refresh_enabled = false;

  void prepare_gpu_refresh(
    const int s_rows,
    const int s_cols,
    const int k_block,
    const int oversample,
    const int power_iters,
    const unsigned int seed
  ) {
    Omega.reset();
    Uhat.reset();
    const int sketch_width = std::min(
      std::min(s_rows, s_cols),
      k_block + std::max(oversample, 0)
    );
    Y.set_size(static_cast<arma::uword>(s_rows), static_cast<arma::uword>(sketch_width));
    shat.set_size(static_cast<arma::uword>(k_block));
    fastpls_svd::cuda_rsvd_refresh_left_block_u_resident(
      s_rows,
      s_cols,
      k_block,
      sketch_width,
      seed,
      std::max(power_iters, 0),
      Y.memptr(),
      shat.memptr()
    );
  }

  void prepare_cpu_refresh(
    const arma::mat& S,
    const int k_block,
    const int oversample,
    const int power_iters,
    const unsigned int seed
  ) {
    const arma::uword sketch_width = std::min(
      std::min(S.n_rows, S.n_cols),
      static_cast<arma::uword>(k_block + std::max(oversample, 0))
    );
    Omega = gaussian_matrix_local(
      S.n_cols,
      sketch_width,
      seed
    );

    Y = S * Omega;
    for (int it = 0; it < power_iters; ++it) {
      arma::qr_econ(Qy, Ry, Y);
      Z = S.t() * Qy;
      arma::qr_econ(Qz, Rz, Z);
      Y = S * Qz;
    }
  }

  void prepare_cpu_refresh_from_right_gram(
    const arma::mat& S,
    const arma::mat& right_gram,
    const int k_block,
    const int oversample,
    const int power_iters,
    const unsigned int seed
  ) {
    const arma::uword sketch_width = std::min(
      std::min(S.n_rows, S.n_cols),
      static_cast<arma::uword>(k_block + std::max(oversample, 0))
    );
    Omega = gaussian_matrix_local(S.n_cols, sketch_width, seed);
    Z = Omega;
    for (int it = 0; it < power_iters; ++it) {
      Y = right_gram * Z;
      arma::qr_econ(Qz, Rz, Y);
      Z = Qz;
    }
    if (power_iters == 0) {
      arma::qr_econ(Qz, Rz, Z);
      Z = Qz;
    }
  }

  bool refresh(
    const arma::mat& S,
    const int k_block,
    const int oversample,
    const int power_iters,
    const unsigned int seed,
    arma::mat& Ublock
  ) {
    if (S.n_rows < 1 || S.n_cols < 1 || k_block < 1) {
      return false;
    }

    if (gpu_refresh_enabled) {
      prepare_gpu_refresh(
        static_cast<int>(S.n_rows),
        static_cast<int>(S.n_cols),
        k_block,
        oversample,
        power_iters,
        seed
      );
      Ublock = Y;
      if (Ublock.n_cols > static_cast<arma::uword>(k_block)) {
        Ublock = Ublock.cols(0, static_cast<arma::uword>(k_block - 1));
      }
      return (Ublock.n_cols > 0);
    } else {
      prepare_cpu_refresh(S, k_block, oversample, power_iters, seed);
    }

    arma::qr_econ(Q, R, Y);
    if (Q.n_cols < 1) {
      return false;
    }

    Bsmall = Q.t() * S;
    if (Bsmall.n_rows < 1 || Bsmall.n_cols < 1) {
      return false;
    }

    if (!finalize_left_block_from_bsmall(Bsmall, Uhat, shat, Vhat) || Uhat.n_cols < 1) {
      return false;
    }

    Ublock = Q * Uhat;
    if (Ublock.n_cols > static_cast<arma::uword>(k_block)) {
      Ublock = Ublock.cols(0, static_cast<arma::uword>(k_block - 1));
    }
    return (Ublock.n_cols > 0);
  }

  bool refresh_from_right_gram(
    const arma::mat& S,
    const arma::mat& right_gram,
    const int k_block,
    const int oversample,
    const int power_iters,
    const unsigned int seed,
    arma::mat& Ublock
  ) {
    if (
      S.n_rows < 1 || S.n_cols < 1 || k_block < 1 ||
      right_gram.n_rows != S.n_cols || right_gram.n_cols != S.n_cols
    ) {
      return false;
    }

    prepare_cpu_refresh_from_right_gram(
      S, right_gram, k_block, oversample, power_iters, seed
    );
    // Reuse the smaller Gram operator for power iterations, but solve the
    // same final left-range projection as the direct randomized refresh.
    // An SVD restricted to S * Z alone drops its coupling to the rest of S.
    Y = S * Z;
    if (!arma::qr_econ(Q, R, Y) || Q.n_cols < 1) {
      return false;
    }
    Bsmall = Q.t() * S;
    if (!finalize_left_block_from_bsmall(Bsmall, Uhat, shat, Vhat) || Uhat.n_cols < 1) {
      return false;
    }
    Ublock = Q * Uhat;
    if (Ublock.n_cols > static_cast<arma::uword>(k_block)) {
      Ublock = Ublock.cols(0, static_cast<arma::uword>(k_block - 1));
    }
    return Ublock.n_cols > 0;
  }
};

} // namespace

double RQ(const arma::mat& yData, const arma::mat& yPred) {
  return fastpls::core::observed_mean_r2(
    fastpls::core::make_const_view(
      yData.memptr(), static_cast<std::size_t>(yData.n_rows),
      static_cast<std::size_t>(yData.n_cols),
      static_cast<std::size_t>(yData.n_rows)
    ),
    fastpls::core::make_const_view(
      yPred.memptr(), static_cast<std::size_t>(yPred.n_rows),
      static_cast<std::size_t>(yPred.n_cols),
      static_cast<std::size_t>(yPred.n_rows)
    )
  );
}



arma::mat variance(const arma::mat& x) {
  int nrow = x.n_rows, ncol = x.n_cols;
  arma::mat out(1,ncol);
  
  for (int j = 0; j < ncol; j++) {
    double mean = 0;
    double M2 = 0;
    int n=0;
    double delta, xx;
    for (int i = 0; i < nrow; i++) {
      n = i+1;
      xx = x(i,j);
      delta = xx - mean;
      mean += delta/n;
      M2 = M2 + delta*(xx-mean);
    }
    out(0,j) = sqrt(M2/(n-1));
  }
  return out;
}


namespace {

#ifndef _WIN32

arma::fmat float32_bits_to_fmat(SEXP xSEXP, const char* name) {
  Rcpp::S4 x(xSEXP);
  Rcpp::IntegerMatrix bits = x.slot("Data");
  if (bits.nrow() < 1 || bits.ncol() < 1) {
    Rcpp::stop("%s must be a non-empty float32 matrix", name);
  }
  arma::fmat out(bits.nrow(), bits.ncol());
  const int* src = INTEGER(bits);
  float* dst = out.memptr();
  const arma::uword n = out.n_elem;
  for (arma::uword i = 0; i < n; ++i) {
    static_assert(sizeof(float) == sizeof(int), "float32 bridge requires 32-bit float and int");
    std::memcpy(dst + i, src + i, sizeof(float));
  }
  return out;
}

Rcpp::IntegerMatrix fmat_to_float32_bits(const arma::fmat& x) {
  Rcpp::IntegerMatrix bits(x.n_rows, x.n_cols);
  int* dst = INTEGER(bits);
  const float* src = x.memptr();
  const arma::uword n = x.n_elem;
  for (arma::uword i = 0; i < n; ++i) {
    std::memcpy(dst + i, src + i, sizeof(float));
  }
  return bits;
}

// Rcpp may simplify a one-column Armadillo matrix returned through an
// intermediate List to a numeric vector. Keep the numerical value in
// float32 and restore its unambiguous one-column matrix form for the
// rank-one response case used by univariate PLS regression.
arma::fmat r_object_to_fmat(SEXP xSEXP, const char* name) {
  if (!Rf_isReal(xSEXP)) {
    Rcpp::stop("%s must be a numeric matrix or vector", name);
  }
  SEXP dims = Rf_getAttrib(xSEXP, R_DimSymbol);
  const R_xlen_t rows = (dims != R_NilValue && XLENGTH(dims) == 2)
    ? INTEGER(dims)[0]
    : XLENGTH(xSEXP);
  const R_xlen_t cols = (dims != R_NilValue && XLENGTH(dims) == 2)
    ? INTEGER(dims)[1]
    : 1;
  if (rows < 1 || cols < 1 || rows * cols != XLENGTH(xSEXP)) {
    Rcpp::stop("%s must be a non-empty numeric matrix or vector", name);
  }
  arma::fmat out(static_cast<arma::uword>(rows), static_cast<arma::uword>(cols));
  const double* src = REAL(xSEXP);
  float* dst = out.memptr();
  for (R_xlen_t i = 0; i < XLENGTH(xSEXP); ++i) {
    dst[i] = static_cast<float>(src[i]);
  }
  return out;
}

arma::fvec r_object_to_fvec(SEXP xSEXP, const char* name) {
  if (Rf_isReal(xSEXP)) {
    Rcpp::NumericVector x(xSEXP);
    arma::fvec out(x.size());
    for (R_xlen_t i = 0; i < x.size(); ++i) {
      out(static_cast<arma::uword>(i)) = static_cast<float>(x[i]);
    }
    return out;
  }
  Rcpp::stop("%s must be a numeric vector", name);
}

arma::fmat integer_bits_to_fmat(SEXP xSEXP, const char* name) {
  Rcpp::IntegerMatrix bits(xSEXP);
  if (bits.nrow() < 1 || bits.ncol() < 1) {
    Rcpp::stop("%s must be a non-empty float32 bit matrix", name);
  }
  arma::fmat out(bits.nrow(), bits.ncol());
  const int* src = INTEGER(bits);
  float* dst = out.memptr();
  for (arma::uword i = 0; i < out.n_elem; ++i) {
    std::memcpy(dst + i, src + i, sizeof(float));
  }
  return out;
}

arma::frowvec float_col_sd(const arma::fmat& X) {
  arma::frowvec out(X.n_cols, arma::fill::ones);
  if (X.n_rows < 2) {
    return out;
  }
  for (arma::uword j = 0; j < X.n_cols; ++j) {
    const float mu = arma::mean(X.col(j));
    double ss = 0.0;
    for (arma::uword i = 0; i < X.n_rows; ++i) {
      const double d = static_cast<double>(X(i, j) - mu);
      ss += d * d;
    }
    const double sd = std::sqrt(ss / static_cast<double>(X.n_rows - 1));
    out(j) = (std::isfinite(sd) && sd > 0.0) ? static_cast<float>(sd) : 1.0f;
  }
  return out;
}

float rq_float32(const arma::fmat& yData, const arma::fmat& yPred) {
  double tss = 0.0;
  double press = 0.0;
  for (arma::uword j = 0; j < yData.n_cols; ++j) {
    const double mu = arma::mean(arma::conv_to<arma::vec>::from(yData.col(j)));
    for (arma::uword i = 0; i < yData.n_rows; ++i) {
      const double obs = static_cast<double>(yData(i, j));
      const double pred = static_cast<double>(yPred(i, j));
      const double d = obs - mu;
      const double e = obs - pred;
      tss += d * d;
      press += e * e;
    }
  }
  if (!std::isfinite(tss) || tss <= 0.0) {
    return NA_REAL;
  }
  return static_cast<float>(1.0 - press / tss);
}

arma::fmat gaussian_matrix_float(arma::uword n_rows, arma::uword n_cols, unsigned int seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> norm(0.0f, 1.0f);
  arma::fmat out(n_rows, n_cols);
  float* ptr = out.memptr();
  for (arma::uword i = 0; i < out.n_elem; ++i) {
    ptr[i] = norm(rng);
  }
  return out;
}

template<class Operator>
Rcpp::List rsvd_float32_operator_raw(Operator& A,
                            int k,
                            int oversample,
                            int power_iters,
                            unsigned int seed,
                            bool left_only) {
  fastpls::native::OperatorRsvdWorkspace<float> workspace;
  fastpls::native::RsvdControls controls;
  controls.oversample = oversample;
  controls.power = power_iters;
  controls.seed = seed;
  controls.left_only = left_only;
  const auto candidate = fastpls::native::operator_rsvd<float>(A, k, controls, workspace);
  return Rcpp::List::create(
    Rcpp::Named("u") = candidate.U,
    Rcpp::Named("d") = candidate.s,
    Rcpp::Named("v") = arma::fmat(candidate.Vt.t()),
    Rcpp::Named("effective_oversample") = controls.oversample,
    Rcpp::Named("effective_power") = controls.power,
    Rcpp::Named("effective_seed") = static_cast<double>(controls.seed)
  );
}

template<class Operator>
Rcpp::List rsvd_float32_core_operator(
    Operator& A,
    fastpls::runtime::CpuLinearAlgebraF32& backend,
    int k,
    int oversample,
    int power_iters,
    unsigned int seed,
    bool left_only) {
  const std::size_t max_rank = std::min(A.rows(), A.columns());
  const std::size_t target = std::min<std::size_t>(
    max_rank, static_cast<std::size_t>(std::max(k, 1))
  );
  if (target == 0) Rcpp::stop("rSVD requires a nonempty matrix");
  const int audit_rank = std::min<int>(
    static_cast<int>(max_rank), static_cast<int>(target) + 1
  );
  const int oversamples[] = {
    std::max(oversample, 20),
    std::max(oversample, 32),
    std::max(oversample, 48)
  };
  const int powers[] = {
    std::max(power_iters, 2),
    std::max(power_iters, 3),
    std::max(power_iters, 4)
  };

  fastpls::core::RsvdControls controls;
  controls.left_only = false;
  fastpls::core::SingularTriplets<float> candidate;
  fastpls::core::OperatorRsvdWorkspace<float> workspace;
  float max_residual = std::numeric_limits<float>::infinity();
  float omitted_ratio = std::numeric_limits<float>::infinity();
  auto check = [&](const fastpls::core::SingularTriplets<float>& value) {
    if (value.U.columns() < target ||
        value.singular_values.size() < target ||
        value.Vt.rows() < target) {
      return false;
    }
    max_residual = 0.0f;
    const float scale = std::max(
      value.singular_values.empty() ? 0.0f :
        std::abs(value.singular_values.front()),
      1e-6f
    );
    fastpls::core::Matrix<float> left(A.rows(), target);
    fastpls::core::Matrix<float> right(A.columns(), target);
    fastpls::core::Matrix<float> v(A.columns(), target);
    for (std::size_t column = 0; column < target; ++column) {
      for (std::size_t row = 0; row < A.columns(); ++row) {
        v(row, column) = value.Vt(column, row);
      }
    }
    A.multiply(v.view(), false, left);
    A.multiply(value.U.view(), true, right);
    for (std::size_t column = 0; column < target; ++column) {
      float left_ss = 0.0f;
      float right_ss = 0.0f;
      const float singular = value.singular_values[column];
      for (std::size_t row = 0; row < A.rows(); ++row) {
        const float residual = left(row, column) -
          singular * value.U(row, column);
        left_ss += residual * residual;
      }
      for (std::size_t row = 0; row < A.columns(); ++row) {
        const float residual = right(row, column) -
          singular * v(row, column);
        right_ss += residual * residual;
      }
      max_residual = std::max(
        max_residual,
        std::max(std::sqrt(left_ss), std::sqrt(right_ss)) / scale
      );
    }
    omitted_ratio = target < value.singular_values.size() &&
      value.singular_values[target - 1] > 0.0f ?
      std::abs(value.singular_values[target] /
               value.singular_values[target - 1]) : 0.0f;
    return std::isfinite(max_residual) && max_residual <= 1e-2f &&
      std::isfinite(omitted_ratio) && omitted_ratio <= 1.01f;
  };

  bool accepted = false;
  int attempts = 0;
  for (int attempt = 0; attempt < 3; ++attempt) {
    controls.oversample = oversamples[attempt];
    controls.power = powers[attempt];
    controls.seed = seed + static_cast<unsigned int>(104729 * attempt);
    candidate = fastpls::core::randomized_operator_svd<float>(
      A, audit_rank, controls, backend, workspace
    );
    attempts = attempt + 1;
    if (check(candidate)) {
      accepted = true;
      break;
    }
  }
  for (int recovery = 0; !accepted && recovery < 4; ++recovery) {
    const int current_width = std::min<int>(
      static_cast<int>(max_rank),
      audit_rank + std::max(controls.oversample, 0)
    );
    const int enlarged_width = std::min<int>(
      static_cast<int>(max_rank),
      std::max(current_width + std::min(32, static_cast<int>(max_rank) - current_width),
               current_width + std::min(current_width,
                                        static_cast<int>(max_rank) - current_width))
    );
    controls.oversample = enlarged_width - audit_rank;
    controls.power = std::max(controls.power, 6) + 2 * recovery;
    controls.seed += 104729U * static_cast<unsigned int>(recovery + 1);
    candidate = fastpls::core::randomized_operator_svd<float>(
      A, audit_rank, controls, backend, workspace
    );
    ++attempts;
    accepted = check(candidate);
  }
  if (!accepted) {
    fastpls_svd::record_rsvd_audit_result(fastpls_svd::SVDResult(), true);
    Rcpp::stop(
      "rSVD recovery did not meet numerical tolerances within its "
      "workspace/iteration budget; no unchecked result was returned"
    );
  }

  arma::fmat u(candidate.U.data(), candidate.U.rows(), target);
  arma::fvec d(candidate.singular_values.data(), target);
  arma::fmat v;
  if (!left_only) {
    v.set_size(A.columns(), target);
    for (std::size_t column = 0; column < target; ++column) {
      for (std::size_t row = 0; row < A.columns(); ++row) {
        v(row, column) = candidate.Vt(column, row);
      }
    }
  }

  fastpls_svd::SVDResult audit_record;
  audit_record.randomized = true;
  audit_record.case_audited = true;
  audit_record.case_certified = true;
  audit_record.audit_attempts = attempts;
  audit_record.effective_oversample = controls.oversample;
  audit_record.effective_power_iters = controls.power;
  audit_record.effective_seed = controls.seed;
  audit_record.audit_triplet_residual = max_residual;
  audit_record.audit_omitted_direction_ratio = omitted_ratio;
  fastpls_svd::record_rsvd_audit_result(audit_record);
  return Rcpp::List::create(
    Rcpp::Named("u") = u,
    Rcpp::Named("d") = d,
    Rcpp::Named("v") = v,
    Rcpp::Named("case_audited") = true,
    Rcpp::Named("case_certified") = true,
    Rcpp::Named("deterministic_fallback") = false,
    Rcpp::Named("audit_attempts") = attempts,
    Rcpp::Named("effective_oversample") = controls.oversample,
    Rcpp::Named("effective_power") = controls.power,
    Rcpp::Named("effective_seed") = static_cast<double>(controls.seed),
    Rcpp::Named("audit_triplet_residual") = max_residual,
    Rcpp::Named("audit_omitted_direction_ratio") = omitted_ratio
  );
}

Rcpp::List rsvd_float32(const arma::fmat& A, int k, int oversample,
                        int power_iters, unsigned int seed, bool left_only) {
  fastpls::runtime::CpuLinearAlgebraF32 backend;
  fastpls::core::ExplicitOperator<
    float, fastpls::runtime::CpuLinearAlgebraF32
  > input(
    fastpls::core::make_const_view(
      A.memptr(), A.n_rows, A.n_cols, A.n_rows
    ),
    backend
  );
  return rsvd_float32_core_operator(
    input, backend, k, oversample, power_iters, seed, left_only
  );
}

Rcpp::List truncated_svd_float32(const arma::fmat& A,
                                 int k,
                                 int svd_method,
                                 int rsvd_oversample,
                                 int rsvd_power,
                                 unsigned int seed,
                                 bool left_only) {
  if (svd_method == 1) {
    Rcpp::stop("float32 supports rSVD only; IRLBA is not part of fastPLS");
  }
  return rsvd_float32(A, k, rsvd_oversample, rsvd_power, seed, left_only);
}

arma::fmat rsvd_sample_float32_metal(const arma::fmat& A,
                                     int l,
                                     int power_iters,
                                     unsigned int seed,
                                     arma::fmat* omega_out = nullptr,
                                     fastpls_svd::MetalFloatOperator* metal_operator = nullptr);

Rcpp::List rsvd_float32_metal_right_gram(
    const arma::fmat& A,
    const arma::fmat& right_gram,
    int k,
    int oversample,
    int power_iters,
    unsigned int seed,
    bool left_only,
    fastpls_svd::MetalFloatOperator* matrix_operator = nullptr,
    fastpls_svd::MetalFloatOperator* gram_operator = nullptr) {
  const arma::uword max_rank = std::min(A.n_rows, A.n_cols);
  const arma::uword target = std::min<arma::uword>(
    max_rank, static_cast<arma::uword>(std::max(k, 1))
  );
  const arma::uword width = std::min<arma::uword>(
    max_rank,
    target + static_cast<arma::uword>(std::max(oversample, 0))
  );
  if (right_gram.n_rows != A.n_cols || right_gram.n_cols != A.n_cols) {
    Rcpp::stop("float32 Metal right-Gram dimensions do not match the matrix");
  }

  std::unique_ptr<fastpls_svd::MetalFloatOperator> owned_matrix;
  std::unique_ptr<fastpls_svd::MetalFloatOperator> owned_gram;
  if (matrix_operator == nullptr) {
    owned_matrix.reset(new fastpls_svd::MetalFloatOperator(A));
    matrix_operator = owned_matrix.get();
  }
  if (gram_operator == nullptr) {
    owned_gram.reset(new fastpls_svd::MetalFloatOperator(right_gram));
    gram_operator = owned_gram.get();
  }

  arma::fmat Z = gaussian_matrix_float(A.n_cols, width, seed);
  arma::fmat Qz;
  arma::fmat Rz;
  const int iterations = std::max(power_iters, 0);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    arma::fmat sample = gram_operator->multiply(Z, false);
    if (!arma::qr_econ(Qz, Rz, sample) || Qz.n_cols < 1) {
      Rcpp::stop("float32 Metal right-Gram QR factorization failed");
    }
    Z = Qz;
  }
  if (iterations == 0) {
    if (!arma::qr_econ(Qz, Rz, Z) || Qz.n_cols < 1) {
      Rcpp::stop("float32 Metal right-Gram sketch QR failed");
    }
    Z = Qz;
  }

  arma::fmat sample = matrix_operator->multiply(Z, false);
  arma::fmat Q;
  arma::fmat R;
  if (!arma::qr_econ(Q, R, sample) || Q.n_cols < 1) {
    Rcpp::stop("float32 Metal range QR factorization failed");
  }
  arma::fmat B = matrix_operator->multiply(Q, true).t();
  arma::fmat Uhat;
  arma::fvec singular;
  arma::fmat V;
  if (!arma::svd_econ(Uhat, singular, V, B, "both")) {
    Rcpp::stop("float32 Metal right-Gram reduced SVD failed");
  }
  const arma::uword usable = std::min<arma::uword>(
    target, std::min(Uhat.n_cols, V.n_cols)
  );
  if (usable < 1) {
    Rcpp::stop("float32 Metal right-Gram rSVD returned no directions");
  }
  arma::fmat U = Q * Uhat.cols(0, usable - 1);
  return Rcpp::List::create(
    Rcpp::Named("u") = U,
    Rcpp::Named("d") = singular.head(usable),
    Rcpp::Named("v") = left_only ? arma::fmat() : V.cols(0, usable - 1)
  );
}

Rcpp::List rsvd_float32_metal(const arma::fmat& A,
                              int k,
                              int oversample,
                              int power_iters,
                              unsigned int seed,
                              bool left_only,
                              fastpls_svd::MetalFloatOperator* metal_operator = nullptr) {
  const arma::uword max_rank = std::min(A.n_rows, A.n_cols);
  const arma::uword target = std::min<arma::uword>(
    max_rank,
    static_cast<arma::uword>(std::max(k, 1))
  );
  const arma::uword l = std::min<arma::uword>(
    max_rank,
    target + static_cast<arma::uword>(std::max(oversample, 0))
  );
  if (target < 1) {
    return Rcpp::List::create(
      Rcpp::Named("u") = arma::fmat(),
      Rcpp::Named("d") = arma::fvec(),
      Rcpp::Named("v") = arma::fmat()
    );
  }
  if (l >= max_rank || max_rank < 6) {
    arma::fmat U;
    arma::fvec s;
    arma::fmat V;
    arma::svd_econ(U, s, V, A, left_only ? "left" : "both");
    return Rcpp::List::create(
      Rcpp::Named("u") = U.cols(0, target - 1),
      Rcpp::Named("d") = s.subvec(0, target - 1),
      Rcpp::Named("v") = left_only ? arma::fmat() : V.cols(0, target - 1)
    );
  }

  std::unique_ptr<fastpls_svd::MetalFloatOperator> owned_operator;
  if (metal_operator == nullptr) {
    owned_operator.reset(new fastpls_svd::MetalFloatOperator(A));
    metal_operator = owned_operator.get();
  }
  if (A.n_cols <= A.n_rows && A.n_cols <= 512) {
    const arma::fmat right_gram = metal_operator->multiply(A, true);
    return rsvd_float32_metal_right_gram(
      A, right_gram, k, oversample, power_iters, seed, left_only,
      metal_operator, nullptr
    );
  }
  arma::fmat Omega;
  arma::fmat Y = rsvd_sample_float32_metal(
    A,
    static_cast<int>(l),
    power_iters,
    seed,
    &Omega,
    metal_operator
  );
  (void) Omega;

  arma::fmat Q;
  arma::fmat R;
  arma::qr_econ(Q, R, Y);
  arma::fmat Bt = metal_operator->multiply(Q, true);
  arma::fmat B = Bt.t();
  arma::fmat Uhat;
  arma::fvec s;
  arma::fmat V;
  if (!arma::svd_econ(Uhat, s, V, B, "both")) {
    Rcpp::stop("float32 Metal rSVD failed on the reduced matrix");
  }
  arma::fmat U = Q * Uhat;

  return Rcpp::List::create(
    Rcpp::Named("u") = U.cols(0, target - 1),
    Rcpp::Named("d") = s.subvec(0, target - 1),
    Rcpp::Named("v") = left_only ? arma::fmat() : V.cols(0, target - 1)
  );
}

arma::fmat rsvd_sample_float32_cuda(const arma::fmat& A,
                                    int l,
                                    int power_iters,
                                    unsigned int seed,
                                    arma::fmat* omega_out);

Rcpp::List rsvd_float32_cuda(const arma::fmat& A,
                             int k,
                             int oversample,
                             int power_iters,
                             unsigned int seed,
                             bool left_only) {
  const arma::uword max_rank = std::min(A.n_rows, A.n_cols);
  const arma::uword target = std::min<arma::uword>(
    max_rank,
    static_cast<arma::uword>(std::max(k, 1))
  );
  const arma::uword l = std::min<arma::uword>(
    max_rank,
    target + static_cast<arma::uword>(std::max(oversample, 0))
  );
  if (target < 1) {
    return Rcpp::List::create(
      Rcpp::Named("u") = arma::fmat(),
      Rcpp::Named("d") = arma::fvec(),
      Rcpp::Named("v") = arma::fmat()
    );
  }
  if (l >= max_rank || max_rank < 6) {
    arma::fmat U;
    arma::fvec s;
    arma::fmat V;
    arma::svd_econ(U, s, V, A, left_only ? "left" : "both");
    return Rcpp::List::create(
      Rcpp::Named("u") = U.cols(0, target - 1),
      Rcpp::Named("d") = s.subvec(0, target - 1),
      Rcpp::Named("v") = left_only ? arma::fmat() : V.cols(0, target - 1)
    );
  }

  arma::fmat Omega;
  arma::fmat Y = rsvd_sample_float32_cuda(
    A,
    static_cast<int>(l),
    power_iters,
    seed,
    &Omega
  );
  (void) Omega;

  arma::fmat Q;
  arma::fmat R;
  arma::qr_econ(Q, R, Y);

  // The large range-finder products above stay in float32 CUDA. The projected
  // matrix is only l x ncol(A), so forming the small SVD on host keeps the
  // implementation portable while preserving the float32 operator path.
  arma::fmat B = Q.t() * A;
  arma::fmat Uhat;
  arma::fvec s;
  arma::fmat V;
  if (!arma::svd_econ(Uhat, s, V, B, "both")) {
    Rcpp::stop("float32 CUDA rSVD failed on the reduced matrix");
  }
  arma::fmat U = Q * Uhat;

  return Rcpp::List::create(
    Rcpp::Named("u") = U.cols(0, target - 1),
    Rcpp::Named("d") = s.subvec(0, target - 1),
    Rcpp::Named("v") = left_only ? arma::fmat() : V.cols(0, target - 1)
  );
}

Rcpp::List truncated_svd_float32_backend(const arma::fmat& A,
                                         int k,
                                         int backend,
                                         int svd_method,
                                         int rsvd_oversample,
                                         int rsvd_power,
                                         unsigned int seed,
                                         bool left_only,
                                         fastpls_svd::MetalFloatOperator* metal_operator = nullptr) {
  if (backend == 2) {
    if (!fastpls_svd::has_metal_backend()) {
      Rcpp::stop(
        "The requested Metal route requires Apple Metal support. "
        "No CPU fallback is performed."
      );
    }
    if (svd_method == 1) {
      Rcpp::stop("float32 supports rSVD only; IRLBA is not part of fastPLS");
    }
    return rsvd_float32_metal(
      A, k, rsvd_oversample, rsvd_power, seed, left_only, metal_operator
    );
  }
  if (backend == 1) {
    if (!fastpls_svd::has_cuda_backend()) {
      Rcpp::stop(
        "backend = 'cuda' requires a CUDA-enabled fastPLS build and an "
        "available NVIDIA GPU. No CPU fallback is performed."
      );
    }
    if (svd_method == 1) {
      Rcpp::stop(
        "float32 CUDA currently supports method = 'rsvd' only. "
        "No CPU fallback is performed."
      );
    }
    return rsvd_float32_cuda(A, k, rsvd_oversample, rsvd_power, seed, left_only);
  }
  if (backend == 0) {
    return truncated_svd_float32(A, k, svd_method, rsvd_oversample, rsvd_power, seed, left_only);
  }
  if (backend == 3) {
    // The hybrid algorithm keeps explicit cross-covariance decomposition on
    // the CPU. Metal is reserved for persistent implicit X'Y/Y'X products,
    // avoiding command and synchronization overhead on host-resident S.
    return truncated_svd_float32(
      A, k, svd_method, rsvd_oversample, rsvd_power, seed, left_only
    );
  }
  Rcpp::stop(
    "float32 SVD supports backend = 'cpu', 'cuda', or operation-split 'metal'"
  );
}

arma::fmat rsvd_sample_float32_cuda(const arma::fmat& A,
                                    int l,
                                    int power_iters,
                                    unsigned int seed,
                                    arma::fmat* omega_out = nullptr) {
  if (l < 1) {
    Rcpp::stop("l must be positive");
  }
  arma::fmat Omega = gaussian_matrix_float(A.n_cols, static_cast<arma::uword>(l), seed);
  arma::fmat Y(A.n_rows, static_cast<arma::uword>(l), arma::fill::zeros);
  fastpls_svd::cuda_rsvd_sample_y_float(
    A.memptr(),
    static_cast<int>(A.n_rows),
    static_cast<int>(A.n_cols),
    Omega.memptr(),
    l,
    std::max(power_iters, 0),
    Y.memptr()
  );
  if (omega_out != nullptr) {
    *omega_out = Omega;
  }
  return Y;
}

arma::fmat rsvd_sample_float32_metal(const arma::fmat& A,
                                     int l,
                                     int power_iters,
                                     unsigned int seed,
                                     arma::fmat* omega_out,
                                     fastpls_svd::MetalFloatOperator* metal_operator) {
  if (l < 1) {
    Rcpp::stop("l must be positive");
  }
  std::unique_ptr<fastpls_svd::MetalFloatOperator> owned_operator;
  if (metal_operator == nullptr) {
    owned_operator.reset(new fastpls_svd::MetalFloatOperator(A));
    metal_operator = owned_operator.get();
  }
  arma::fmat Omega = gaussian_matrix_float(A.n_cols, static_cast<arma::uword>(l), seed);
  arma::fmat Y = metal_operator->multiply(Omega, false);
  const int q = std::max(power_iters, 0);
  for (int i = 0; i < q; ++i) {
    arma::fmat Qy;
    arma::fmat Ry;
    if (!arma::qr_econ(Qy, Ry, Y)) {
      Rcpp::stop("float32 Metal rSVD failed to orthonormalize the left sketch");
    }
    arma::fmat Z = metal_operator->multiply(Qy, true);
    arma::fmat Qz;
    arma::fmat Rz;
    if (!arma::qr_econ(Qz, Rz, Z)) {
      Rcpp::stop("float32 Metal rSVD failed to orthonormalize the right sketch");
    }
    Y = metal_operator->multiply(Qz, false);
  }
  if (omega_out != nullptr) {
    *omega_out = Omega;
  }
  return Y;
}

Rcpp::List fmat_list_to_bits(const std::vector<arma::fmat>& xs, const arma::ivec& ncomp) {
  Rcpp::List out(xs.size());
  Rcpp::CharacterVector names(xs.size());
  for (std::size_t i = 0; i < xs.size(); ++i) {
    out[i] = fmat_to_float32_bits(xs[i]);
    names[i] = std::string("ncomp=") + std::to_string(ncomp(static_cast<arma::uword>(i)));
  }
  out.attr("names") = names;
  return out;
}

arma::fmat float32_backend_matmul(const arma::fmat& A,
                                  const arma::fmat& B,
                                  const int backend,
                                  const bool transpose_left = false,
                                  const bool transpose_right = false) {
  if (backend == 1) {
    return fastpls_svd::cuda_matrix_multiply_float(
      A, B, transpose_left, transpose_right
    );
  }
  if (backend == 2 || backend == 3) {
    return fastpls_svd::metal_matrix_multiply_float(
      A, B, transpose_left, transpose_right
    );
  }
  if (backend != 0) {
    Rcpp::stop("float32 matrix multiplication requires backend 0, 1, or 2");
  }
  if (transpose_left && transpose_right) return A.t() * B.t();
  if (transpose_left) return A.t() * B;
  if (transpose_right) return A * B.t();
  return A * B;
}

#endif

} // namespace

#ifndef _WIN32

// [[Rcpp::export]]
Rcpp::List opls_filter_float32_cpp(SEXP XSEXP,
                                   SEXP YSEXP,
                                   int north,
                                   int scaling,
                                   int backend,
                                   int svd_method,
                                   int rsvd_oversample,
                                   int rsvd_power,
                                   int seed) {
  arma::fmat X = float32_bits_to_fmat(XSEXP, "X");
  arma::fmat Y = float32_bits_to_fmat(YSEXP, "Y");
  if (X.n_rows != Y.n_rows) {
    Rcpp::stop("X and Y must have the same number of rows");
  }
  if (north < 0) {
    Rcpp::stop("north must be >= 0");
  }

  arma::frowvec mX(X.n_cols, arma::fill::zeros);
  if (scaling < 3) {
    mX = arma::mean(X, 0);
    X.each_row() -= mX;
  }
  arma::frowvec vX(X.n_cols, arma::fill::ones);
  if (scaling == 2) {
    vX = float_col_sd(X);
    X.each_row() /= vX;
  }
  const arma::frowvec mY = arma::mean(Y, 0);
  Y.each_row() -= mY;

  arma::fmat W_orth(X.n_cols, static_cast<arma::uword>(north), arma::fill::zeros);
  arma::fmat P_orth(X.n_cols, static_cast<arma::uword>(north), arma::fill::zeros);
  int used = 0;
  for (int component = 0; component < north; ++component) {
    const arma::fmat S = float32_backend_matmul(X, Y, backend, true, false);
    Rcpp::List sv = truncated_svd_float32_backend(
      S,
      1,
      backend,
      svd_method,
      rsvd_oversample,
      rsvd_power,
      static_cast<unsigned int>(seed + component),
      true
    );
    const arma::fmat U = Rcpp::as<arma::fmat>(sv["u"]);
    if (U.n_cols < 1) break;
    arma::fvec w = U.col(0);
    const float w_norm = arma::norm(w, 2);
    if (!std::isfinite(w_norm) || w_norm <= 0.0f) break;
    w /= w_norm;

    arma::fmat w_matrix(w.n_elem, 1);
    w_matrix.col(0) = w;
    const arma::fvec t = float32_backend_matmul(
      X, w_matrix, backend, false, false
    ).col(0);
    const float t_ss = arma::dot(t, t);
    if (!std::isfinite(t_ss) || t_ss <= 0.0f) break;
    arma::fmat t_matrix(t.n_elem, 1);
    t_matrix.col(0) = t;
    const arma::fvec p = float32_backend_matmul(
      X, t_matrix, backend, true, false
    ).col(0) / t_ss;

    const float ww = arma::dot(w, w);
    arma::fvec w_orth = p - w * (arma::dot(w, p) / ww);
    const float wo_norm = arma::norm(w_orth, 2);
    if (!std::isfinite(wo_norm) || wo_norm <= 0.0f) break;
    w_orth /= wo_norm;
    arma::fmat wo_matrix(w_orth.n_elem, 1);
    wo_matrix.col(0) = w_orth;
    const arma::fvec t_orth = float32_backend_matmul(
      X, wo_matrix, backend, false, false
    ).col(0);
    const float to_ss = arma::dot(t_orth, t_orth);
    if (!std::isfinite(to_ss) || to_ss <= 0.0f) break;
    arma::fmat to_matrix(t_orth.n_elem, 1);
    to_matrix.col(0) = t_orth;
    const arma::fvec p_orth = float32_backend_matmul(
      X, to_matrix, backend, true, false
    ).col(0) / to_ss;
    arma::fmat po_matrix(p_orth.n_elem, 1);
    po_matrix.col(0) = p_orth;
    X -= float32_backend_matmul(to_matrix, po_matrix, backend, false, true);
    W_orth.col(static_cast<arma::uword>(used)) = w_orth;
    P_orth.col(static_cast<arma::uword>(used)) = p_orth;
    ++used;
  }

  if (used == 0) {
    W_orth.set_size(X.n_cols, 0);
    P_orth.set_size(X.n_cols, 0);
  } else if (used < north) {
    W_orth = W_orth.cols(0, static_cast<arma::uword>(used - 1));
    P_orth = P_orth.cols(0, static_cast<arma::uword>(used - 1));
  }
  return Rcpp::List::create(
    Rcpp::Named("X") = fmat_to_float32_bits(X),
    Rcpp::Named("mX") = fmat_to_float32_bits(arma::fmat(mX)),
    Rcpp::Named("vX") = fmat_to_float32_bits(arma::fmat(vX)),
    Rcpp::Named("W_orth") = fmat_to_float32_bits(W_orth),
    Rcpp::Named("P_orth") = fmat_to_float32_bits(P_orth),
    Rcpp::Named("north") = used
  );
}

// [[Rcpp::export]]
Rcpp::List opls_filter_float32_labels_cpp(
    SEXP XSEXP,
    const Rcpp::IntegerVector& labels,
    int n_classes,
    int north,
    int scaling,
    int backend,
    int svd_method,
    int rsvd_oversample,
    int rsvd_power,
    int seed) {
  arma::fmat X = float32_bits_to_fmat(XSEXP, "X");
  if (labels.size() != static_cast<R_xlen_t>(X.n_rows) || n_classes < 2) {
    stop("label-aware OPLS requires one valid label per row and at least two classes");
  }
  arma::uvec compact_labels(X.n_rows);
  for (arma::uword row = 0; row < X.n_rows; ++row) {
    const int label = labels[static_cast<R_xlen_t>(row)] - 1;
    if (label < 0 || label >= n_classes) {
      stop("label-aware OPLS labels must be encoded as 1..n_classes");
    }
    compact_labels(row) = static_cast<arma::uword>(label);
  }
  auto solve = [&](const arma::fmat& S, int component, arma::fvec& w) {
    if (backend == 0) {
      return fastpls::native::leading_left_from_smaller_gram(S, w);
    }
    Rcpp::List decomposition = truncated_svd_float32_backend(
      S, 1, backend, svd_method, rsvd_oversample, rsvd_power,
      static_cast<unsigned int>(seed + component), true
    );
    arma::fmat directions = r_object_to_fmat(
      decomposition["u"], "float32 OPLS candidate direction"
    );
    if (directions.n_cols < 1) return false;
    w = directions.col(0);
    return true;
  };
  const arma::fmat no_dense_response;
  auto filter = fastpls::native::fit_opls_filter_with_solver(
    std::move(X), no_dense_response, north, scaling, solve,
    &compact_labels, n_classes
  );
  return Rcpp::List::create(
    Rcpp::Named("X") = fmat_to_float32_bits(filter.X),
    Rcpp::Named("mX") = fmat_to_float32_bits(arma::fmat(filter.x_mean)),
    Rcpp::Named("vX") = fmat_to_float32_bits(arma::fmat(filter.x_scale)),
    Rcpp::Named("W_orth") = fmat_to_float32_bits(filter.W),
    Rcpp::Named("P_orth") = fmat_to_float32_bits(filter.P),
    Rcpp::Named("north") = filter.completed
  );
}

// [[Rcpp::export]]
Rcpp::List lda_train_prefix_float32_cuda(SEXP TtrainSEXP,
                                         const Rcpp::IntegerVector& y,
                                         int n_classes,
                                         const Rcpp::IntegerVector& ncomp) {
  if (!fastpls_svd::cuda_lda_native_available()) {
    Rcpp::stop(
      "float32 CUDA LDA fitting requires native CUDA LDA support. "
      "No CPU fallback is performed."
    );
  }
  const arma::fmat Ttrain = float32_bits_to_fmat(TtrainSEXP, "Ttrain");
  arma::ivec labels(y.size());
  for (R_xlen_t i = 0; i < y.size(); ++i) {
    labels(static_cast<arma::uword>(i)) = y[i];
  }
  arma::ivec components(ncomp.size());
  for (R_xlen_t i = 0; i < ncomp.size(); ++i) {
    components(static_cast<arma::uword>(i)) = ncomp[i];
  }
  const std::vector<fastpls_svd::LDAFloatGPUModel> fitted =
    fastpls_svd::cuda_lda_train_prefix_float(
      Ttrain, labels, n_classes, components
    );
  Rcpp::List models(fitted.size());
  Rcpp::CharacterVector model_names(fitted.size());
  for (std::size_t i = 0; i < fitted.size(); ++i) {
    models[static_cast<R_xlen_t>(i)] = Rcpp::List::create(
      Rcpp::Named("means") = fmat_to_float32_bits(fitted[i].means),
      Rcpp::Named("linear") = fmat_to_float32_bits(fitted[i].linear),
      Rcpp::Named("constants") = fmat_to_float32_bits(fitted[i].constants),
      Rcpp::Named("priors") = fmat_to_float32_bits(fitted[i].priors.t()),
      Rcpp::Named("ridge") = fitted[i].ridge,
      Rcpp::Named("ridge_relative") = fitted[i].relative_ridge,
      Rcpp::Named("precision") = "float32",
      Rcpp::Named("backend") = "cuda_native"
    );
    model_names[static_cast<R_xlen_t>(i)] =
      std::to_string(ncomp[static_cast<R_xlen_t>(i)]);
  }
  models.attr("names") = model_names;
  return models;
}

// [[Rcpp::export]]
Rcpp::List lda_predict_float32_cuda(SEXP TtestSEXP,
                                    const Rcpp::List& lda,
                                    bool return_scores = true) {
  if (!fastpls_svd::cuda_lda_native_available()) {
    Rcpp::stop(
      "float32 CUDA LDA prediction requires native CUDA LDA support. "
      "No CPU fallback is performed."
    );
  }
  const arma::fmat Ttest = float32_bits_to_fmat(TtestSEXP, "Ttest");
  const arma::fmat linear = integer_bits_to_fmat(lda["linear"], "lda$linear");
  const arma::fmat constants_matrix = integer_bits_to_fmat(
    lda["constants"], "lda$constants"
  );
  const arma::frowvec constants = arma::vectorise(constants_matrix, 1);
  const fastpls_svd::LDAFloatPrediction out =
    fastpls_svd::cuda_lda_predict_float(
      Ttest, linear, constants, return_scores
    );
  return Rcpp::List::create(
    Rcpp::Named("pred") = Rcpp::wrap(out.pred),
    Rcpp::Named("scores") = return_scores ?
      Rcpp::RObject(fmat_to_float32_bits(out.scores)) :
      Rcpp::RObject(R_NilValue)
  );
}

// [[Rcpp::export]]
Rcpp::List fastsvd_float32_cpp(SEXP ASEXP,
                               int k,
                               int backend,
                               int svd_method,
                               int rsvd_oversample,
                               int rsvd_power,
                               int seed,
                               bool left_only = false) {
  arma::fmat A = float32_bits_to_fmat(ASEXP, "x");
  Rcpp::List sv = truncated_svd_float32_backend(
    A,
    k,
    backend,
    svd_method,
    rsvd_oversample,
    rsvd_power,
    static_cast<unsigned int>(seed),
    left_only
  );

  arma::fmat U = Rcpp::as<arma::fmat>(sv["u"]);
  arma::fvec d = Rcpp::as<arma::fvec>(sv["d"]);
  arma::fmat V = Rcpp::as<arma::fmat>(sv["v"]);
  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("u") = fmat_to_float32_bits(U),
    Rcpp::Named("d") = d,
    Rcpp::Named("v") = left_only ?
      Rcpp::RObject(R_NilValue) :
      Rcpp::RObject(fmat_to_float32_bits(V))
  );
  const char* audit_fields[] = {
    "case_audited",
    "case_certified",
    "deterministic_fallback",
    "audit_attempts",
    "effective_oversample",
    "effective_power",
    "effective_seed",
    "audit_triplet_residual",
    "audit_omitted_direction_ratio"
  };
  for (const char* field : audit_fields) {
    if (sv.containsElementNamed(field)) out[field] = sv[field];
  }
  return out;
}

void deflate_float32_in_place(arma::fmat& S, const arma::fvec& v,
                              arma::frowvec& row, arma::fvec& column) {
  // Evaluate the row before modifying S. Keeping multiplication and
  // subtraction in separate vector operations preserves float32 rounding,
  // without the full p-by-q alias-protection temporary of S -= v*(v.t()*S).
  row = v.t() * S;
  for (arma::uword j = 0; j < S.n_cols; ++j) {
    column = v * row(j);
    S.col(j) -= column;
  }
}

bool use_float32_implicit_crosscov(int n, int p, int q, int components) {
  const double explicit_size = static_cast<double>(p) * q;
  const double factor_size = static_cast<double>(n + components) * (p + q);
  return explicit_size * sizeof(float) > 32.0 * 1024.0 * 1024.0 &&
    factor_size < explicit_size && components < n &&
    components + 64 < std::min(p, q);
}

void stabilize_float32_direction(arma::fvec& r, const arma::fmat& V, int used) {
  if (used == 0) return;
  arma::fvec projection = V.head_cols(used).t() * r;
  const float roundoff = 32.0f * std::numeric_limits<float>::epsilon() * arma::norm(r, 2);
  if (arma::norm(projection, 2) <= roundoff) return;
  // A deflated SIMPLS direction lies in the complement of prior loadings.
  // Restore that invariant when float32 roundoff reintroduces old directions.
  r -= V.head_cols(used) * projection;
  projection = V.head_cols(used).t() * r;
  r -= V.head_cols(used) * projection;
}

void stabilize_float32_scores(arma::fvec& r, arma::fvec& t,
                               const arma::fmat& R, const arma::fmat& T,
                               int used) {
  if (used == 0) return;
  arma::fvec projection = T.head_cols(used).t() * t;
  const float roundoff = 32.0f * std::numeric_limits<float>::epsilon() * arma::norm(t, 2);
  if (arma::norm(projection, 2) <= roundoff) return;
  // Apply the same correction to scores and prediction weights. Merely
  // reorthogonalizing the stored scores would change training/test semantics.
  for (int pass = 0; pass < 2; ++pass) {
    t -= T.head_cols(used) * projection;
    r -= R.head_cols(used) * projection;
    projection = T.head_cols(used).t() * t;
  }
}

bool refresh_rank_one_float32(
  fastpls_svd::FloatCrosscovOperator& op,
  const arma::fmat& V,
  int used,
  int power_iters,
  unsigned int seed,
  arma::fvec& direction
) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  direction.set_size(op.n_rows);
  for (arma::uword i = 0; i < direction.n_elem; ++i) {
    direction(i) = normal(rng);
  }
  stabilize_float32_direction(direction, V, used);
  float direction_norm = arma::norm(direction, 2);
  if (!std::isfinite(direction_norm) ||
      direction_norm <= std::numeric_limits<float>::epsilon()) {
    return false;
  }
  direction /= direction_norm;

  for (int iteration = 0; iteration < std::max(power_iters, 1); ++iteration) {
    arma::fmat right_matrix = op.multiply(arma::fmat(direction), true);
    arma::fvec right = right_matrix.col(0);
    const float right_norm = arma::norm(right, 2);
    if (!std::isfinite(right_norm) ||
        right_norm <= std::numeric_limits<float>::epsilon()) {
      return false;
    }
    right /= right_norm;
    arma::fmat left_matrix = op.multiply(arma::fmat(right), false);
    direction = left_matrix.col(0);
    stabilize_float32_direction(direction, V, used);
    direction_norm = arma::norm(direction, 2);
    if (!std::isfinite(direction_norm) ||
        direction_norm <= std::numeric_limits<float>::epsilon()) {
      return false;
    }
    direction /= direction_norm;
  }
  return true;
}

bool refresh_block_float32(
  fastpls_svd::FloatCrosscovOperator& op,
  const arma::fmat& V,
  int used,
  int width,
  int power_iters,
  unsigned int seed,
  arma::fmat& directions
) {
  if (width < 1) return false;
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  directions.set_size(op.n_rows, static_cast<arma::uword>(width));
  for (arma::uword i = 0; i < directions.n_elem; ++i) {
    directions(i) = normal(rng);
  }
  auto orthogonalize = [&] (arma::fmat& value) {
    if (used > 0) {
      const arma::fmat previous = V.head_cols(static_cast<arma::uword>(used));
      value -= previous * (previous.t() * value);
      value -= previous * (previous.t() * value);
    }
    arma::fmat triangular;
    return arma::qr_econ(value, triangular, value) &&
      value.n_cols == static_cast<arma::uword>(width) && value.is_finite();
  };
  if (!orthogonalize(directions)) return false;
  for (int iteration = 0; iteration < std::max(power_iters, 1); ++iteration) {
    arma::fmat right = op.multiply(directions, true);
    arma::fmat triangular;
    if (!arma::qr_econ(right, triangular, right) || !right.is_finite()) {
      return false;
    }
    directions = op.multiply(right, false);
    if (!orthogonalize(directions)) return false;
  }
  return true;
}

Rcpp::List crosscov_svd_float32(fastpls_svd::FloatCrosscovOperator& op,
                                int k, int backend, int method,
                                int oversample, int power, unsigned int seed,
                                bool left_only) {
  if (method == 1) {
    if (backend == 1) {
      Rcpp::stop("float32 CUDA supports rSVD only; no CPU fallback is performed");
    }
    Rcpp::stop("float32 supports rSVD only; IRLBA is not part of fastPLS");
  }
  if (backend == 0) {
    return rsvd_float32_core_operator(
      op, op.cpu_backend(), k, oversample, power, seed, left_only
    );
  }
  if (backend == 3 && k > 1) {
    oversample = std::max(oversample, 32);
    power = std::max(power, 7);
  }
  return rsvd_float32_operator_raw(op, k, oversample, power, seed, left_only);
}

Rcpp::List fit_simpls_float32_blocked(
    const arma::fmat& Xtrain,
    const arma::fmat& Ytrain,
    arma::ivec ncomp,
    int scaling,
    bool fit,
    int backend,
    int svd_method,
    int rsvd_oversample,
    int rsvd_power,
    unsigned int seed) {
  fastpls::native::SimplsOptions controls;
  controls.scaling = scaling;
  controls.fitted = fit;
  controls.store_scores = true;
  controls.store_coefficients = false;
  controls.reorthogonalize = true;
  controls.randomized_directions = true;
  controls.maximum_block = backend == 0 ? kCpuSimplsBlockSize :
    kAcceleratorSimplsBlockSize;
  controls.svd.oversample = rsvd_oversample;
  controls.svd.power = rsvd_power;
  controls.svd.seed = seed;

  fastpls::native::DirectionWorkspace<float> cpu_workspace;
  std::unique_ptr<fastpls_svd::MetalFloatOperator> metal_matrix_workspace;
  std::unique_ptr<fastpls_svd::MetalFloatOperator> metal_gram_workspace;
  auto solver = [&](const arma::fmat& current_crosscov,
                    const arma::fmat& right_gram,
                    bool use_right_gram,
                    int width,
                    unsigned int direction_seed,
                    arma::fmat& directions) {
    if (backend == 0 || backend == 3) {
      return use_right_gram ? cpu_workspace.refresh_from_right_gram(
        current_crosscov, right_gram, width, rsvd_oversample,
        std::max(rsvd_power, 0), direction_seed, directions
      ) : cpu_workspace.refresh(
        current_crosscov, width, rsvd_oversample,
        std::max(rsvd_power, 0), direction_seed, directions
      );
    }
    if (backend == 2 && use_right_gram) {
      if (!metal_matrix_workspace) {
        metal_matrix_workspace.reset(
          new fastpls_svd::MetalFloatOperator(current_crosscov)
        );
        metal_gram_workspace.reset(
          new fastpls_svd::MetalFloatOperator(right_gram)
        );
      } else {
        metal_matrix_workspace->update(current_crosscov);
        metal_gram_workspace->update(right_gram);
      }
      Rcpp::List decomposition = rsvd_float32_metal_right_gram(
        current_crosscov, right_gram, width, rsvd_oversample,
        rsvd_power, direction_seed, true,
        metal_matrix_workspace.get(), metal_gram_workspace.get()
      );
      directions = r_object_to_fmat(
        decomposition["u"], "float32 Metal SIMPLS candidate directions"
      );
      return directions.n_cols > 0;
    }
    Rcpp::List decomposition = truncated_svd_float32_backend(
      current_crosscov, width, backend, svd_method, rsvd_oversample,
      rsvd_power, direction_seed, true
    );
    directions = r_object_to_fmat(
      decomposition["u"], "float32 SIMPLS candidate directions"
    );
    return directions.n_cols > 0;
  };

  std::unique_ptr<fastpls_svd::MetalFloatOperator> hybrid_sample_workspace;
  auto hybrid_sample_product = [&] (
      const arma::fmat& left, const arma::fmat& right,
      const bool transpose_left) {
    if (!hybrid_sample_workspace) {
      hybrid_sample_workspace.reset(
        new fastpls_svd::MetalFloatOperator(left)
      );
    }
    return hybrid_sample_workspace->multiply(right, transpose_left);
  };
  const std::function<arma::fmat(
    const arma::fmat&, const arma::fmat&, bool
  )> hybrid_sample_product_fn = hybrid_sample_product;
  const std::function<void(
    const arma::fmat&, const arma::fmat&, arma::fmat&, arma::fmat&
  )> hybrid_sample_geometry_fn = [&] (
      const arma::fmat& left,
      const arma::fmat& directions,
      arma::fmat& scores,
      arma::fmat& loadings) {
    if (!hybrid_sample_workspace) {
      hybrid_sample_workspace.reset(
        new fastpls_svd::MetalFloatOperator(left)
      );
    }
    hybrid_sample_workspace->geometry(directions, scores, loadings);
  };
  const std::function<arma::fmat(
    const arma::fmat&, const arma::fmat&, bool
  )> cpu_sample_product_fn = [] (
      const arma::fmat& left, const arma::fmat& right,
      const bool transpose_left) {
    return float_matrix_multiply_cpu(
      left, right, transpose_left, false
    );
  };
  const std::function<void(
    const arma::fmat&, const arma::fmat&, arma::fmat&, arma::fmat&
  )> cpu_sample_geometry_fn = [] (
      const arma::fmat& left,
      const arma::fmat& directions,
      arma::fmat& scores,
      arma::fmat& loadings) {
    float_sample_geometry_cpu(left, directions, scores, loadings);
  };
  const std::function<arma::fmat(const arma::fmat&)>
    cpu_sample_crossprod_fn = [] (const arma::fmat& value) {
      return float_matrix_multiply_cpu(value, value, true, false);
    };

  auto model = backend == 3 ? fastpls::native::fit_simpls_with_solver(
    Xtrain, Ytrain, std::move(ncomp), controls, solver,
    static_cast<arma::fmat*>(nullptr), nullptr, 0,
    hybrid_sample_product_fn, hybrid_sample_geometry_fn
  ) : backend == 0 ? fastpls::native::fit_simpls_with_solver(
    Xtrain, Ytrain, std::move(ncomp), controls, solver,
    static_cast<arma::fmat*>(nullptr), nullptr, 0,
    cpu_sample_product_fn, cpu_sample_geometry_fn, cpu_sample_crossprod_fn
  ) : fastpls::native::fit_simpls_with_solver(
    Xtrain, Ytrain, std::move(ncomp), controls, solver
  );
  std::vector<arma::fmat> fitted_values(
    static_cast<std::size_t>(model.components.n_elem)
  );
  if (fit) {
    for (arma::uword index = 0; index < model.components.n_elem; ++index) {
      fitted_values[static_cast<std::size_t>(index)] = model.fitted.slice(index);
    }
  }
  return Rcpp::List::create(
    Rcpp::Named("P") = R_NilValue,
    Rcpp::Named("R") = fmat_to_float32_bits(model.R),
    Rcpp::Named("Q") = fmat_to_float32_bits(model.Q),
    Rcpp::Named("Ttrain") = fmat_to_float32_bits(model.scores),
    Rcpp::Named("mX") = fmat_to_float32_bits(model.x_mean),
    Rcpp::Named("vX") = fmat_to_float32_bits(model.x_scale),
    Rcpp::Named("mY") = fmat_to_float32_bits(model.y_mean),
    Rcpp::Named("p") = static_cast<int>(Xtrain.n_cols),
    Rcpp::Named("m") = static_cast<int>(Ytrain.n_cols),
    Rcpp::Named("ncomp") = model.components,
    Rcpp::Named("Yfit") = fit ?
      Rcpp::RObject(fmat_list_to_bits(fitted_values, model.components)) :
      Rcpp::RObject(R_NilValue),
    Rcpp::Named("R2Y") = model.r2,
    Rcpp::Named("pls_method") = "simpls",
    Rcpp::Named("xprod_mode") = "float32_explicit_blocked"
  );
}

// [[Rcpp::export]]
Rcpp::List pls_float32_cpu_cpp(
  SEXP XtrainSEXP,
  SEXP YtrainSEXP,
  arma::ivec ncomp,
  int scaling,
  bool fit,
  int method,
  int backend,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  int seed
) {
  arma::fmat Xtrain = float32_bits_to_fmat(XtrainSEXP, "Xtrain");
  arma::fmat Ytrain = float32_bits_to_fmat(YtrainSEXP, "Ytrain");
  if (Xtrain.n_rows != Ytrain.n_rows) {
    Rcpp::stop("Xtrain and Ytrain must have the same number of rows");
  }
  if (ncomp.n_elem < 1) {
    Rcpp::stop("ncomp must contain at least one value");
  }
  for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
    if (ncomp(i) < 1) ncomp(i) = 1;
  }
  if (method == 1) {
    const int rank_cap = static_cast<int>(std::min(Xtrain.n_cols, Ytrain.n_cols));
    for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
      if (ncomp(i) > rank_cap) ncomp(i) = rank_cap;
    }
  }

  const int max_ncomp = arma::max(ncomp);
  const int length_ncomp = static_cast<int>(ncomp.n_elem);
  const int n = static_cast<int>(Xtrain.n_rows);
  const int p = static_cast<int>(Xtrain.n_cols);
  const int m = static_cast<int>(Ytrain.n_cols);
  const bool implicit_requested = use_float32_implicit_crosscov(
    n, p, m, max_ncomp
  );
  if (method != 1 && !implicit_requested) {
    return fit_simpls_float32_blocked(
      Xtrain, Ytrain, std::move(ncomp), scaling, fit, backend, svd_method,
      rsvd_oversample, rsvd_power, static_cast<unsigned int>(seed)
    );
  }

  arma::frowvec mX(p, arma::fill::zeros);
  if (scaling < 3) {
    mX = arma::mean(Xtrain, 0);
    Xtrain.each_row() -= mX;
  }
  arma::frowvec vX(p, arma::fill::ones);
  if (scaling == 2) {
    vX = float_col_sd(Xtrain);
    Xtrain.each_row() /= vX;
  }
  arma::frowvec mY = arma::mean(Ytrain, 0);
  Ytrain.each_row() -= mY;

  const bool implicit = implicit_requested;
  std::unique_ptr<fastpls_svd::FloatCrosscovOperator> crosscov_operator;
  std::unique_ptr<fastpls_svd::MetalFloatOperator> hybrid_sample_operator;
  arma::fmat S;
  if (implicit) {
    if (backend == 1 && svd_method == 1) {
      Rcpp::stop("float32 CUDA supports rSVD only; no CPU fallback is performed");
    }
    crosscov_operator.reset(new fastpls_svd::FloatCrosscovOperator(
      Xtrain, Ytrain, method == 1 ? 0 : max_ncomp, backend));
  } else {
    if (backend == 3) {
      hybrid_sample_operator.reset(
        new fastpls_svd::MetalFloatOperator(Xtrain)
      );
      S = hybrid_sample_operator->multiply(Ytrain, true);
    } else {
      S = Xtrain.t() * Ytrain;
    }
  }
  arma::fmat Rmat(p, max_ncomp, arma::fill::zeros);
  arma::fmat Qmat(m, max_ncomp, arma::fill::zeros);
  arma::fvec R2Y(length_ncomp, arma::fill::value(NA_REAL));
  std::vector<arma::fmat> Yfit_vec(length_ncomp);
  std::vector<arma::fmat> Wlat_vec(length_ncomp);

  if (method == 1) {
    Rcpp::List sv = implicit ? crosscov_svd_float32(*crosscov_operator,
      max_ncomp, backend, svd_method, rsvd_oversample, rsvd_power,
      static_cast<unsigned int>(seed), false) : truncated_svd_float32_backend(
      S,
      max_ncomp,
      backend,
      svd_method,
      rsvd_oversample,
      rsvd_power,
      static_cast<unsigned int>(seed),
      false
    );
    arma::fmat U = r_object_to_fmat(sv["u"], "float32 left singular vectors");
    arma::fmat V = r_object_to_fmat(sv["v"], "float32 right singular vectors");
    arma::fvec d = r_object_to_fvec(sv["d"], "float32 singular values");
    const int effective_oversample = sv.containsElementNamed(
      "effective_oversample"
    ) ? Rcpp::as<int>(sv["effective_oversample"]) : rsvd_oversample;
    const int effective_power = sv.containsElementNamed(
      "effective_power"
    ) ? Rcpp::as<int>(sv["effective_power"]) : rsvd_power;
    Rmat.cols(0, U.n_cols - 1) = U;
    Qmat.cols(0, V.n_cols - 1) = V;

    if (backend == 3 && !hybrid_sample_operator) {
      hybrid_sample_operator.reset(
        new fastpls_svd::MetalFloatOperator(Xtrain)
      );
    }
    arma::fmat Ttrain = backend == 3 ?
      hybrid_sample_operator->multiply(U, false) : Xtrain * U;
    arma::fmat G = Ttrain.t() * Ttrain;
    for (int i = 0; i < length_ncomp; ++i) {
      const int k = ncomp(i);
      arma::fmat D(k, k, arma::fill::zeros);
      for (int j = 0; j < k; ++j) D(j, j) = d(j);
      arma::fmat Ck = arma::solve(G.submat(0, 0, k - 1, k - 1), D);
      arma::fmat Wk = Ck * V.cols(0, k - 1).t();
      Wlat_vec[static_cast<std::size_t>(i)] = Wk;
      if (fit) {
        arma::fmat yf = Ttrain.cols(0, k - 1) * Wk;
        R2Y(i) = rq_float32(Ytrain, yf);
        yf.each_row() += mY;
        Yfit_vec[static_cast<std::size_t>(i)] = yf;
      }
    }
    Rcpp::RObject Yfit_obj = fit ?
      Rcpp::RObject(fmat_list_to_bits(Yfit_vec, ncomp)) :
      Rcpp::RObject(R_NilValue);

    return Rcpp::List::create(
      Rcpp::Named("R") = fmat_to_float32_bits(Rmat),
      Rcpp::Named("Q") = fmat_to_float32_bits(Qmat),
      Rcpp::Named("Ttrain") = fmat_to_float32_bits(Ttrain),
      Rcpp::Named("W_latent") = fmat_list_to_bits(Wlat_vec, ncomp),
      Rcpp::Named("mX") = fmat_to_float32_bits(arma::fmat(mX)),
      Rcpp::Named("vX") = fmat_to_float32_bits(arma::fmat(vX)),
      Rcpp::Named("mY") = fmat_to_float32_bits(arma::fmat(mY)),
      Rcpp::Named("p") = p,
      Rcpp::Named("m") = m,
      Rcpp::Named("ncomp") = ncomp,
      Rcpp::Named("Yfit") = Yfit_obj,
      Rcpp::Named("R2Y") = R2Y,
      Rcpp::Named("pls_method") = "plssvd",
      Rcpp::Named("xprod_mode") = implicit ? "implicit_float32" : "explicit_float32",
      Rcpp::Named("rsvd_effective_oversample") = effective_oversample,
      Rcpp::Named("rsvd_effective_power") = effective_power
    );
  }

  arma::fmat Vmat(p, max_ncomp, arma::fill::zeros);
  arma::fmat Tmat(n, max_ncomp, arma::fill::zeros);
  arma::fmat Yfit_cur;
  if (fit) {
    Yfit_cur.zeros(n, m);
  }
  std::unique_ptr<fastpls_svd::MetalFloatOperator> metal_operator;
  if (!implicit && backend == 2 && std::min(p, m) >= 6 &&
      (svd_method == 1 || 1 + std::max(rsvd_oversample, 0) < std::min(p, m))) {
    metal_operator.reset(new fastpls_svd::MetalFloatOperator(S));
  }
  arma::frowvec deflation_row;
  arma::fvec deflation_column;
  const bool use_massive_block_refresh = implicit &&
    static_cast<double>(p) * static_cast<double>(m) * sizeof(float) >
      512.0 * 1024.0 * 1024.0;
  int out_idx = 0;
  int a = 0;
  while (a < max_ncomp) {
    if (a > 0 && metal_operator) metal_operator->update(S);
    const int block = use_massive_block_refresh ?
      std::min(8, max_ncomp - a) : 1;
    arma::fmat candidates;
    if (block > 1) {
      if (!refresh_block_float32(
            *crosscov_operator, Vmat, a, block, rsvd_power,
            static_cast<unsigned int>(seed + a), candidates)) {
        break;
      }
    } else {
      arma::fvec rr;
      if (use_massive_block_refresh) {
        if (!refresh_rank_one_float32(
              *crosscov_operator, Vmat, a, rsvd_power,
              static_cast<unsigned int>(seed + a), rr)) {
          break;
        }
      } else {
        Rcpp::List sv = implicit ? crosscov_svd_float32(*crosscov_operator,
          1, backend, svd_method, rsvd_oversample, rsvd_power,
          static_cast<unsigned int>(seed + a), true) :
          truncated_svd_float32_backend(
            S,
            1,
            backend,
            svd_method,
            rsvd_oversample,
            rsvd_power,
            static_cast<unsigned int>(seed + a),
            true,
            metal_operator.get()
          );
        arma::fmat U = r_object_to_fmat(
          sv["u"], "float32 left singular vectors"
        );
        if (U.n_cols < 1) break;
        rr = U.col(0);
      }
      candidates = arma::fmat(rr);
    }

    bool failed = false;
    for (int j = 0; j < block; ++j) {
      const int component = a + j;
      arma::fvec rr = candidates.col(static_cast<arma::uword>(j));
      stabilize_float32_direction(rr, Vmat, component);
      arma::fvec tt = Xtrain * rr;
      stabilize_float32_scores(rr, tt, Rmat, Tmat, component);
      const float tnorm = arma::norm(tt, 2);
      if (!std::isfinite(tnorm) || tnorm <= 0.0f) {
        failed = true;
        break;
      }
      tt /= tnorm;
      rr /= tnorm;
      arma::fvec pp = Xtrain.t() * tt;
      arma::fvec qq = Ytrain.t() * tt;
      arma::fvec vv = pp;
      if (component > 0) {
        arma::fmat Vprev = Vmat.cols(0, component - 1);
        vv -= Vprev * (Vprev.t() * pp);
        vv -= Vprev * (Vprev.t() * vv);
      }
      const float vnorm = arma::norm(vv, 2);
      if (!std::isfinite(vnorm) || vnorm <= 0.0f) {
        failed = true;
        break;
      }
      vv /= vnorm;
      if (implicit) crosscov_operator->deflate(vv);
      else deflate_float32_in_place(S, vv, deflation_row, deflation_column);
      Rmat.col(component) = rr;
      Qmat.col(component) = qq;
      Vmat.col(component) = vv;
      Tmat.col(component) = tt;
      if (fit) {
        Yfit_cur += tt * qq.t();
      }
      while (out_idx < length_ncomp &&
             ncomp(out_idx) == component + 1) {
        if (fit) {
          R2Y(out_idx) = rq_float32(Ytrain, Yfit_cur);
          arma::fmat yf = Yfit_cur;
          yf.each_row() += mY;
          Yfit_vec[static_cast<std::size_t>(out_idx)] = yf;
        }
        ++out_idx;
      }
    }
    if (failed) break;
    a += block;
  }
  Rcpp::RObject Yfit_obj = fit ?
    Rcpp::RObject(fmat_list_to_bits(Yfit_vec, ncomp)) :
    Rcpp::RObject(R_NilValue);

  return Rcpp::List::create(
    Rcpp::Named("P") = R_NilValue,
    Rcpp::Named("R") = fmat_to_float32_bits(Rmat),
    Rcpp::Named("Q") = fmat_to_float32_bits(Qmat),
    Rcpp::Named("Ttrain") = fmat_to_float32_bits(Tmat),
    Rcpp::Named("mX") = fmat_to_float32_bits(arma::fmat(mX)),
    Rcpp::Named("vX") = fmat_to_float32_bits(arma::fmat(vX)),
    Rcpp::Named("mY") = fmat_to_float32_bits(arma::fmat(mY)),
    Rcpp::Named("p") = p,
    Rcpp::Named("m") = m,
    Rcpp::Named("ncomp") = ncomp,
    Rcpp::Named("Yfit") = Yfit_obj,
    Rcpp::Named("R2Y") = R2Y,
    Rcpp::Named("pls_method") = "simpls",
    Rcpp::Named("xprod_mode") = implicit ? "implicit_float32" : "explicit_float32"
  );
}

namespace {

struct Float32LabelResponse {
  arma::frowvec mean;
  arma::fmat crosscov;
  arma::uvec labels;
};

Float32LabelResponse label_response_float32(
  const arma::fmat& X,
  const Rcpp::IntegerVector& labels,
  int n_classes
) {
  if (labels.size() != static_cast<R_xlen_t>(X.n_rows)) {
    Rcpp::stop("float32 label-aware PLS requires one label per training row");
  }
  if (n_classes < 2) {
    Rcpp::stop("float32 label-aware PLS requires at least two classes");
  }

  Float32LabelResponse out;
  out.labels.set_size(X.n_rows);
  arma::fvec counts(static_cast<arma::uword>(n_classes), arma::fill::zeros);
  for (arma::uword i = 0; i < X.n_rows; ++i) {
    const int cls = labels[static_cast<R_xlen_t>(i)] - 1;
    if (cls < 0 || cls >= n_classes) {
      Rcpp::stop("float32 label-aware PLS labels must be encoded as 1..n_classes");
    }
    out.labels(i) = static_cast<arma::uword>(cls);
    counts(static_cast<arma::uword>(cls)) += 1.0f;
  }
  if (arma::any(counts <= 0.0f)) {
    Rcpp::stop("float32 label-aware PLS received an empty class");
  }
  out.mean = counts.t() / static_cast<float>(X.n_rows);
  out.crosscov = fastpls::native::dummy_crossprod<float>(
    X, out.labels, -out.mean
  );
  return out;
}

arma::fvec label_score_product_float32(
  const arma::fvec& score,
  const Float32LabelResponse& response
) {
  arma::fvec out(response.mean.n_elem, arma::fill::zeros);
  for (arma::uword i = 0; i < score.n_elem; ++i) {
    out(response.labels(i)) += score(i);
  }
  out -= response.mean.t() * arma::accu(score);
  return out;
}

float label_rq_float32(
  const Float32LabelResponse& response,
  const arma::fmat& fitted_centered
) {
  const double tss = static_cast<double>(fitted_centered.n_rows) *
    (1.0 - arma::dot(
      arma::conv_to<arma::rowvec>::from(response.mean),
      arma::conv_to<arma::rowvec>::from(response.mean)
    ));
  if (!std::isfinite(tss) || tss <= 0.0) {
    return NA_REAL;
  }
  double cross_term = 0.0;
  for (arma::uword i = 0; i < fitted_centered.n_rows; ++i) {
    double mean_projection = 0.0;
    for (arma::uword cls = 0; cls < fitted_centered.n_cols; ++cls) {
      mean_projection += static_cast<double>(fitted_centered(i, cls)) *
        static_cast<double>(response.mean(cls));
    }
    cross_term += static_cast<double>(fitted_centered(i, response.labels(i))) -
      mean_projection;
  }
  const double press = tss +
    static_cast<double>(arma::accu(arma::square(fitted_centered))) -
    2.0 * cross_term;
  return static_cast<float>(1.0 - press / tss);
}

} // namespace

// Fit classification PLS from compact labels without materializing one-hot Y.
// [[Rcpp::export]]
Rcpp::List pls_float32_labels_cpp(
  SEXP XtrainSEXP,
  const Rcpp::IntegerVector& labels,
  int n_classes,
  arma::ivec ncomp,
  int scaling,
  bool fit,
  int method,
  int backend,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  int seed
) {
  arma::fmat Xtrain = float32_bits_to_fmat(XtrainSEXP, "Xtrain");
  if (ncomp.n_elem < 1) {
    Rcpp::stop("ncomp must contain at least one value");
  }
  const int rank_cap = method == 1 ?
    std::max(
      1,
      std::min(
        static_cast<int>(Xtrain.n_cols),
        n_classes - 1
      )
    ) :
    std::max(
      1,
      std::min(
        static_cast<int>(Xtrain.n_cols),
        static_cast<int>(Xtrain.n_rows) - 1
      )
    );
  for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
    const int requested = static_cast<int>(ncomp(i));
    ncomp(i) = std::max(1, std::min(requested, rank_cap));
  }

  const int max_ncomp = arma::max(ncomp);
  const int length_ncomp = static_cast<int>(ncomp.n_elem);
  const int n = static_cast<int>(Xtrain.n_rows);
  const int p = static_cast<int>(Xtrain.n_cols);
  const int m = n_classes;

  if (method != 1) {
    arma::uvec compact_labels(Xtrain.n_rows);
    for (arma::uword row = 0; row < Xtrain.n_rows; ++row) {
      const int cls = labels[static_cast<R_xlen_t>(row)] - 1;
      if (cls < 0 || cls >= n_classes) {
        Rcpp::stop("float32 label-aware PLS labels must be encoded as 1..n_classes");
      }
      compact_labels(row) = static_cast<arma::uword>(cls);
    }
    const bool phase_timing = env_int_or(
      "FASTPLS_BENCH_PHASE_TIMING", 0, 0, 1
    ) == 1;
    fastpls::native::SimplsOptions controls;
    controls.scaling = scaling;
    controls.fitted = fit;
    controls.phase_timing = phase_timing;
    controls.store_scores = true;
    controls.store_coefficients = false;
    controls.randomized_directions = true;
    controls.crossprod_min_components = env_int_or(
      "FASTPLS_FAST_CROSSPROD_MIN_NCOMP", 20, 1, 1024
    );
    controls.crossprod_max_predictors = env_int_or(
      "FASTPLS_FAST_CROSSPROD_MAX_P", kCpuCrossprodMaxPredictors, 16, 65536
    );
    controls.crossprod_min_ratio = env_int_or(
      "FASTPLS_FAST_CROSSPROD_MIN_N_TO_P_RATIO", 8, 1, 1024
    );
    controls.maximum_block = backend == 0 ? kCpuSimplsBlockSize :
      kAcceleratorSimplsBlockSize;
    controls.svd.oversample = rsvd_oversample;
    controls.svd.power = rsvd_power;
    controls.svd.seed = static_cast<unsigned int>(seed);
    fastpls::native::DirectionWorkspace<float> cpu_workspace;
    std::unique_ptr<fastpls_svd::MetalFloatOperator> metal_matrix_workspace;
    std::unique_ptr<fastpls_svd::MetalFloatOperator> metal_gram_workspace;
    auto solver = [&](const arma::fmat& current_crosscov,
                      const arma::fmat& right_gram,
                      bool use_right_gram,
                      int width,
                      unsigned int direction_seed,
                      arma::fmat& directions) {
      if (backend == 0 || backend == 3) {
        return use_right_gram ? cpu_workspace.refresh_from_right_gram(
          current_crosscov, right_gram, width, rsvd_oversample,
          std::max(rsvd_power, 0), direction_seed, directions
        ) : cpu_workspace.refresh(
          current_crosscov, width, rsvd_oversample,
          std::max(rsvd_power, 0), direction_seed, directions
        );
      }
      if (backend == 2 && use_right_gram) {
        if (!metal_matrix_workspace) {
          metal_matrix_workspace.reset(
            new fastpls_svd::MetalFloatOperator(current_crosscov)
          );
          metal_gram_workspace.reset(
            new fastpls_svd::MetalFloatOperator(right_gram)
          );
        } else {
          metal_matrix_workspace->update(current_crosscov);
          metal_gram_workspace->update(right_gram);
        }
        Rcpp::List decomposition = rsvd_float32_metal_right_gram(
          current_crosscov, right_gram, width, rsvd_oversample,
          rsvd_power, direction_seed, true,
          metal_matrix_workspace.get(), metal_gram_workspace.get()
        );
        directions = r_object_to_fmat(
          decomposition["u"], "float32 Metal SIMPLS candidate directions"
        );
        return directions.n_cols > 0;
      }
      Rcpp::List decomposition = truncated_svd_float32_backend(
        current_crosscov, width, backend, svd_method, rsvd_oversample,
        rsvd_power, direction_seed, true
      );
      directions = r_object_to_fmat(
        decomposition["u"], "float32 SIMPLS candidate directions"
      );
      return directions.n_cols > 0;
    };
    std::unique_ptr<fastpls_svd::MetalFloatOperator>
      hybrid_sample_workspace;
    auto hybrid_sample_product = [&] (
        const arma::fmat& left, const arma::fmat& right,
        const bool transpose_left) {
      if (!hybrid_sample_workspace) {
        hybrid_sample_workspace.reset(
          new fastpls_svd::MetalFloatOperator(left)
        );
      }
      return hybrid_sample_workspace->multiply(right, transpose_left);
    };
    const std::function<arma::fmat(
      const arma::fmat&, const arma::fmat&, bool
    )> hybrid_sample_product_fn = hybrid_sample_product;
    const std::function<void(
      const arma::fmat&, const arma::fmat&, arma::fmat&, arma::fmat&
    )> hybrid_sample_geometry_fn = [&] (
        const arma::fmat& left,
        const arma::fmat& directions,
        arma::fmat& scores,
        arma::fmat& loadings) {
      if (!hybrid_sample_workspace) {
        hybrid_sample_workspace.reset(
          new fastpls_svd::MetalFloatOperator(left)
        );
      }
      hybrid_sample_workspace->geometry(directions, scores, loadings);
    };
    const std::function<arma::fmat(
      const arma::fmat&, const arma::fmat&, bool
    )> cpu_sample_product_fn = [] (
        const arma::fmat& left, const arma::fmat& right,
        const bool transpose_left) {
      return float_matrix_multiply_cpu(
        left, right, transpose_left, false
      );
    };
    const std::function<void(
      const arma::fmat&, const arma::fmat&, arma::fmat&, arma::fmat&
    )> cpu_sample_geometry_fn = [] (
        const arma::fmat& left,
        const arma::fmat& directions,
        arma::fmat& scores,
        arma::fmat& loadings) {
      float_sample_geometry_cpu(left, directions, scores, loadings);
    };
    const std::function<arma::fmat(const arma::fmat&)>
      cpu_sample_crossprod_fn = [] (const arma::fmat& value) {
        return float_matrix_multiply_cpu(value, value, true, false);
      };
    const arma::fmat no_dense_response;
    auto model = backend == 3 ? fastpls::native::fit_simpls_with_solver(
      Xtrain, no_dense_response, std::move(ncomp), controls, solver,
      static_cast<arma::fmat*>(nullptr), &compact_labels, n_classes,
      hybrid_sample_product_fn, hybrid_sample_geometry_fn
    ) : backend == 0 ? fastpls::native::fit_simpls_with_solver(
      Xtrain, no_dense_response, std::move(ncomp), controls, solver,
      static_cast<arma::fmat*>(nullptr), &compact_labels, n_classes,
      cpu_sample_product_fn, cpu_sample_geometry_fn, cpu_sample_crossprod_fn
    ) : fastpls::native::fit_simpls_with_solver(
      Xtrain, no_dense_response, std::move(ncomp), controls, solver,
      static_cast<arma::fmat*>(nullptr), &compact_labels, n_classes
    );
    std::vector<arma::fmat> fitted_values(
      static_cast<std::size_t>(model.components.n_elem)
    );
    if (fit) {
      for (arma::uword index = 0; index < model.components.n_elem; ++index) {
        fitted_values[static_cast<std::size_t>(index)] = model.fitted.slice(index);
      }
    }
    Rcpp::List out = Rcpp::List::create(
      Rcpp::Named("P") = R_NilValue,
      Rcpp::Named("R") = fmat_to_float32_bits(model.R),
      Rcpp::Named("Q") = fmat_to_float32_bits(model.Q),
      Rcpp::Named("Ttrain") = fmat_to_float32_bits(model.scores),
      Rcpp::Named("mX") = fmat_to_float32_bits(model.x_mean),
      Rcpp::Named("vX") = fmat_to_float32_bits(model.x_scale),
      Rcpp::Named("mY") = fmat_to_float32_bits(model.y_mean),
      Rcpp::Named("p") = p,
      Rcpp::Named("m") = m,
      Rcpp::Named("ncomp") = model.components,
      Rcpp::Named("Yfit") = fit ?
        Rcpp::RObject(fmat_list_to_bits(fitted_values, model.components)) :
        Rcpp::RObject(R_NilValue),
      Rcpp::Named("R2Y") = model.r2,
      Rcpp::Named("pls_method") = "simpls",
      Rcpp::Named("xprod_mode") = "float32_label_class_sums_blocked"
    );
    if (controls.phase_timing) {
      const auto& timing = model.timing;
      out["benchmark_phase_timing"] = Rcpp::List::create(
        Rcpp::Named("preprocess_crosscov_sec") = timing.preprocess,
        Rcpp::Named("response_crosscov_sec") = timing.response_crosscov,
        Rcpp::Named("crossprod_cache_sec") = timing.crossprod_cache,
        Rcpp::Named("right_gram_sec") = timing.right_gram,
        Rcpp::Named("estimator_sec") = timing.estimator,
        Rcpp::Named("direction_sec") = timing.direction,
        Rcpp::Named("component_update_sec") = timing.component_update,
        Rcpp::Named("coefficient_path_sec") = timing.coefficients,
        Rcpp::Named("fitted_values_sec") = timing.fitted,
        Rcpp::Named("cpp_total_sec") = timing.total
      );
    }
    return out;
  }

  arma::frowvec mX(p, arma::fill::zeros);
  if (scaling < 3) {
    mX = arma::mean(Xtrain, 0);
    Xtrain.each_row() -= mX;
  }
  arma::frowvec vX(p, arma::fill::ones);
  if (scaling == 2) {
    vX = float_col_sd(Xtrain);
    Xtrain.each_row() /= vX;
  }
  const Float32LabelResponse response =
    label_response_float32(Xtrain, labels, n_classes);
  arma::fmat S = response.crosscov;

  arma::fmat Rmat(p, max_ncomp, arma::fill::zeros);
  arma::fmat Qmat(m, max_ncomp, arma::fill::zeros);
  arma::fvec R2Y(length_ncomp, arma::fill::value(NA_REAL));
  std::vector<arma::fmat> Yfit_vec(static_cast<std::size_t>(length_ncomp));
  std::vector<arma::fmat> Wlat_vec(static_cast<std::size_t>(length_ncomp));

  if (method == 1) {
    Rcpp::List sv = truncated_svd_float32_backend(
      S,
      max_ncomp,
      backend,
      svd_method,
      rsvd_oversample,
      rsvd_power,
      static_cast<unsigned int>(seed),
      false
    );
    arma::fmat U = r_object_to_fmat(sv["u"], "float32 left singular vectors");
    arma::fmat V = r_object_to_fmat(sv["v"], "float32 right singular vectors");
    arma::fvec d = r_object_to_fvec(sv["d"], "float32 singular values");
    const int effective_oversample = sv.containsElementNamed(
      "effective_oversample"
    ) ? Rcpp::as<int>(sv["effective_oversample"]) : rsvd_oversample;
    const int effective_power = sv.containsElementNamed(
      "effective_power"
    ) ? Rcpp::as<int>(sv["effective_power"]) : rsvd_power;
    const int effective = std::min(
      max_ncomp,
      static_cast<int>(std::min(U.n_cols, V.n_cols))
    );
    if (effective < max_ncomp) {
      Rcpp::stop("float32 label-aware PLS-SVD returned fewer components than requested");
    }
    Rmat.cols(0, max_ncomp - 1) = U.cols(0, max_ncomp - 1);
    Qmat.cols(0, max_ncomp - 1) = V.cols(0, max_ncomp - 1);

    std::unique_ptr<fastpls_svd::MetalFloatOperator>
      hybrid_sample_operator;
    if (backend == 3) {
      hybrid_sample_operator.reset(
        new fastpls_svd::MetalFloatOperator(Xtrain)
      );
    }
    arma::fmat Ttrain = backend == 3 ? hybrid_sample_operator->multiply(
      U.cols(0, max_ncomp - 1), false
    ) : Xtrain * U.cols(0, max_ncomp - 1);
    arma::fmat G = Ttrain.t() * Ttrain;
    for (int i = 0; i < length_ncomp; ++i) {
      const int k = ncomp(i);
      arma::fmat D(k, k, arma::fill::zeros);
      for (int j = 0; j < k; ++j) D(j, j) = d(j);
      arma::fmat Ck = arma::solve(
        G.submat(0, 0, k - 1, k - 1),
        D
      );
      arma::fmat Wk = Ck * V.cols(0, k - 1).t();
      Wlat_vec[static_cast<std::size_t>(i)] = Wk;
      if (fit) {
        arma::fmat yf = Ttrain.cols(0, k - 1) * Wk;
        R2Y(i) = label_rq_float32(response, yf);
        yf.each_row() += response.mean;
        Yfit_vec[static_cast<std::size_t>(i)] = std::move(yf);
      }
    }
    return Rcpp::List::create(
      Rcpp::Named("R") = fmat_to_float32_bits(Rmat),
      Rcpp::Named("Q") = fmat_to_float32_bits(Qmat),
      Rcpp::Named("Ttrain") = fmat_to_float32_bits(Ttrain),
      Rcpp::Named("W_latent") = fmat_list_to_bits(Wlat_vec, ncomp),
      Rcpp::Named("mX") = fmat_to_float32_bits(arma::fmat(mX)),
      Rcpp::Named("vX") = fmat_to_float32_bits(arma::fmat(vX)),
      Rcpp::Named("mY") = fmat_to_float32_bits(arma::fmat(response.mean)),
      Rcpp::Named("p") = p,
      Rcpp::Named("m") = m,
      Rcpp::Named("ncomp") = ncomp,
      Rcpp::Named("Yfit") = fit ?
        Rcpp::RObject(fmat_list_to_bits(Yfit_vec, ncomp)) :
        Rcpp::RObject(R_NilValue),
      Rcpp::Named("R2Y") = R2Y,
      Rcpp::Named("pls_method") = "plssvd",
      Rcpp::Named("xprod_mode") = "float32_label_class_sums",
      Rcpp::Named("rsvd_effective_oversample") = effective_oversample,
      Rcpp::Named("rsvd_effective_power") = effective_power
    );
  }

  arma::fmat Vmat(p, max_ncomp, arma::fill::zeros);
  arma::fmat Tmat(n, max_ncomp, arma::fill::zeros);
  arma::fmat Yfit_cur;
  if (fit) {
    Yfit_cur.zeros(n, m);
  }
  std::unique_ptr<fastpls_svd::MetalFloatOperator> metal_operator;
  if (backend == 2 && std::min(p, m) >= 6 &&
      (svd_method == 1 || 1 + std::max(rsvd_oversample, 0) < std::min(p, m))) {
    metal_operator.reset(new fastpls_svd::MetalFloatOperator(S));
  }
  arma::frowvec deflation_row;
  arma::fvec deflation_column;
  int out_idx = 0;
  for (int a = 0; a < max_ncomp; ++a) {
    if (a > 0 && metal_operator) metal_operator->update(S);
    Rcpp::List sv = truncated_svd_float32_backend(
      S,
      1,
      backend,
      svd_method,
      rsvd_oversample,
      rsvd_power,
      static_cast<unsigned int>(seed + a),
      true,
      metal_operator.get()
    );
    arma::fmat U = r_object_to_fmat(sv["u"], "float32 left singular vectors");
    if (U.n_cols < 1) break;
    arma::fvec rr = U.col(0);
    stabilize_float32_direction(rr, Vmat, a);
    arma::fvec tt = Xtrain * rr;
    stabilize_float32_scores(rr, tt, Rmat, Tmat, a);
    const float tnorm = arma::norm(tt, 2);
    if (!std::isfinite(tnorm) || tnorm <= 0.0f) break;
    tt /= tnorm;
    rr /= tnorm;
    arma::fvec pp = Xtrain.t() * tt;
    arma::fvec qq = label_score_product_float32(tt, response);
    arma::fvec vv = pp;
    if (a > 0) {
      const arma::fmat Vprev = Vmat.cols(0, a - 1);
      vv -= Vprev * (Vprev.t() * pp);
      vv -= Vprev * (Vprev.t() * vv);
    }
    const float vnorm = arma::norm(vv, 2);
    if (!std::isfinite(vnorm) || vnorm <= 0.0f) break;
    vv /= vnorm;
    deflate_float32_in_place(S, vv, deflation_row, deflation_column);
    Rmat.col(a) = rr;
    Qmat.col(a) = qq;
    Vmat.col(a) = vv;
    Tmat.col(a) = tt;
    if (fit) {
      Yfit_cur += tt * qq.t();
    }
    while (out_idx < length_ncomp && ncomp(out_idx) == a + 1) {
      if (fit) {
        R2Y(out_idx) = label_rq_float32(response, Yfit_cur);
        arma::fmat yf = Yfit_cur;
        yf.each_row() += response.mean;
        Yfit_vec[static_cast<std::size_t>(out_idx)] = std::move(yf);
      }
      ++out_idx;
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("P") = R_NilValue,
    Rcpp::Named("R") = fmat_to_float32_bits(Rmat),
    Rcpp::Named("Q") = fmat_to_float32_bits(Qmat),
    Rcpp::Named("Ttrain") = fmat_to_float32_bits(Tmat),
    Rcpp::Named("mX") = fmat_to_float32_bits(arma::fmat(mX)),
    Rcpp::Named("vX") = fmat_to_float32_bits(arma::fmat(vX)),
    Rcpp::Named("mY") = fmat_to_float32_bits(arma::fmat(response.mean)),
    Rcpp::Named("p") = p,
    Rcpp::Named("m") = m,
    Rcpp::Named("ncomp") = ncomp,
    Rcpp::Named("Yfit") = fit ?
      Rcpp::RObject(fmat_list_to_bits(Yfit_vec, ncomp)) :
      Rcpp::RObject(R_NilValue),
    Rcpp::Named("R2Y") = R2Y,
    Rcpp::Named("pls_method") = "simpls",
    Rcpp::Named("xprod_mode") = "float32_label_class_sums"
  );
}

#else

namespace {
Rcpp::List windows_float32_unavailable() {
  Rcpp::stop("Native float32 fastPLS kernels are not available on Windows because the R Windows BLAS/LAPACK toolchain does not provide the required single-precision Fortran symbols; use standard numeric input on Windows or a Linux/macOS/CUDA build for native float32 execution.");
}

float windows_bits_to_float(const int bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(float));
  return value;
}

int windows_float_to_bits(const float value) {
  int bits;
  std::memcpy(&bits, &value, sizeof(float));
  return bits;
}

Rcpp::IntegerMatrix windows_float32_bits(SEXP xSEXP, const char* name) {
  if (!Rf_isS4(xSEXP)) {
    Rcpp::stop("%s must be a float32 matrix", name);
  }
  Rcpp::S4 x(xSEXP);
  Rcpp::IntegerMatrix bits = x.slot("Data");
  if (bits.nrow() < 1 || bits.ncol() < 1) {
    Rcpp::stop("%s must be a non-empty float32 matrix", name);
  }
  return bits;
}
}

Rcpp::List fastsvd_float32_cpp(SEXP ASEXP, int k, int backend, int svd_method, int rsvd_oversample, int rsvd_power, int seed, bool left_only) {
  return windows_float32_unavailable();
}

Rcpp::List pls_float32_cpu_cpp(SEXP XtrainSEXP, SEXP YtrainSEXP, arma::ivec ncomp, int scaling, bool fit, int method, int backend, int svd_method, int rsvd_oversample, int rsvd_power, int seed) {
  return windows_float32_unavailable();
}

Rcpp::List pls_float32_labels_cpp(SEXP XtrainSEXP, const Rcpp::IntegerVector& labels, int n_classes, arma::ivec ncomp, int scaling, bool fit, int method, int backend, int svd_method, int rsvd_oversample, int rsvd_power, int seed) {
  return windows_float32_unavailable();
}

Rcpp::List opls_filter_float32_cpp(SEXP XSEXP, SEXP YSEXP, int north, int scaling, int backend, int svd_method, int rsvd_oversample, int rsvd_power, int seed) {
  return windows_float32_unavailable();
}

Rcpp::List opls_filter_float32_labels_cpp(SEXP XSEXP,
                                           const Rcpp::IntegerVector& labels,
                                           int n_classes, int north,
                                           int scaling, int backend,
                                           int svd_method,
                                           int rsvd_oversample,
                                           int rsvd_power, int seed) {
  return windows_float32_unavailable();
}

Rcpp::List lda_train_prefix_float32_cuda(SEXP TtrainSEXP,
                                         const Rcpp::IntegerVector& y,
                                         int n_classes,
                                         const Rcpp::IntegerVector& ncomp) {
  return windows_float32_unavailable();
}

Rcpp::List lda_predict_float32_cuda(SEXP TtestSEXP, const Rcpp::List& lda, bool return_scores) {
  return windows_float32_unavailable();
}

#endif

static Rcpp::IntegerVector lda_labels_from_scores(const arma::mat& scores,
                                                  const arma::rowvec& constants) {
  Rcpp::IntegerVector pred(scores.n_rows);
  for (arma::uword i = 0; i < scores.n_rows; ++i) {
    arma::uword best = 0;
    double best_val = scores(i, 0) + constants(0);
    for (arma::uword c = 1; c < scores.n_cols; ++c) {
      const double val = scores(i, c) + constants(c);
      if (val > best_val) {
        best_val = val;
        best = c;
      }
    }
    pred[i] = static_cast<int>(best) + 1;
  }
  return pred;
}

namespace {

fastpls::core::ConstMatrixView<double> lda_core_view(
    const arma::mat& values) {
  return fastpls::core::make_const_view(
    values.memptr(), static_cast<std::size_t>(values.n_rows),
    static_cast<std::size_t>(values.n_cols),
    static_cast<std::size_t>(values.n_rows)
  );
}

arma::mat lda_arma_matrix(const fastpls::core::Matrix<double>& values) {
  arma::mat output(values.rows(), values.columns());
  std::copy(values.data(), values.data() + values.size(), output.memptr());
  return output;
}

arma::rowvec lda_arma_row(const std::vector<double>& values) {
  arma::rowvec output(values.size());
  std::copy(values.begin(), values.end(), output.memptr());
  return output;
}

arma::vec lda_arma_column(const std::vector<double>& values) {
  arma::vec output(values.size());
  std::copy(values.begin(), values.end(), output.memptr());
  return output;
}

Rcpp::List lda_core_model_to_list(
    const fastpls::core::LdaModel<double>& model) {
  return Rcpp::List::create(
    Rcpp::Named("means") = lda_arma_matrix(model.means),
    Rcpp::Named("inv_cov") = arma::mat(),
    Rcpp::Named("linear") = lda_arma_matrix(model.linear),
    Rcpp::Named("constants") = lda_arma_row(model.constants),
    Rcpp::Named("priors") = lda_arma_column(model.priors),
    Rcpp::Named("ridge") = model.ridge,
    Rcpp::Named("ridge_relative") = model.relative_ridge
  );
}

Rcpp::List lda_core_models_to_list(
    const std::vector<fastpls::core::LdaModel<double>>& models,
    const Rcpp::IntegerVector& components) {
  Rcpp::List output(models.size());
  Rcpp::CharacterVector names(models.size());
  for (std::size_t index = 0; index < models.size(); ++index) {
    output[static_cast<R_xlen_t>(index)] =
      lda_core_model_to_list(models[index]);
    names[static_cast<R_xlen_t>(index)] =
      std::to_string(components[static_cast<R_xlen_t>(index)]);
  }
  output.attr("names") = names;
  return output;
}

Rcpp::List lda_train_moments_core(
    const arma::mat& gram, const arma::mat& class_sums,
    const arma::vec& counts, int sample_count,
    const Rcpp::IntegerVector& components) {
  const auto models = fastpls::core::train_lda_prefixes_from_moments<double>(
    lda_core_view(gram), lda_core_view(class_sums), counts.memptr(),
    static_cast<std::size_t>(counts.n_elem),
    static_cast<std::size_t>(sample_count), components.begin(),
    static_cast<std::size_t>(components.size())
  );
  return lda_core_models_to_list(models, components);
}

} // namespace

Rcpp::List lda_train_prefix_cpp(const arma::mat& Ttrain,
                                const Rcpp::IntegerVector& y,
                                int n_classes,
                                const Rcpp::IntegerVector& ncomp,
                                double ridge) {
  (void)ridge;  // Retained in the internal ABI; regularization is deterministic.
  if (Ttrain.n_rows == 0 || Ttrain.n_cols == 0) {
    stop("lda_train_prefix_cpp requires a non-empty score matrix");
  }
  if (static_cast<R_xlen_t>(Ttrain.n_rows) != y.size()) {
    stop("lda_train_prefix_cpp requires one class label per training row");
  }
  if (n_classes < 2) {
    stop("lda_train_prefix_cpp requires at least two classes");
  }
  if (ncomp.size() < 1) {
    stop("lda_train_prefix_cpp requires at least one component count");
  }

  int kmax_i = 0;
  for (R_xlen_t i = 0; i < ncomp.size(); ++i) {
    if (ncomp[i] > kmax_i) kmax_i = ncomp[i];
  }
  if (kmax_i < 1 || kmax_i > static_cast<int>(Ttrain.n_cols)) {
    stop("lda_train_prefix_cpp component counts must be in 1..ncol(Ttrain)");
  }

  const arma::uword kmax = static_cast<arma::uword>(kmax_i);
  arma::mat Tk_storage;
  const arma::mat* Tk_ptr = &Ttrain;
  if (kmax < Ttrain.n_cols) {
    Tk_storage = Ttrain.cols(0, kmax - 1);
    Tk_ptr = &Tk_storage;
  }
  const arma::mat& Tk = *Tk_ptr;
  arma::vec counts(n_classes, arma::fill::zeros);
  arma::mat class_sums(n_classes, kmax, arma::fill::zeros);

  for (arma::uword i = 0; i < Ttrain.n_rows; ++i) {
    const int cls = y[i] - 1;
    if (cls < 0 || cls >= n_classes) {
      stop("lda_train_prefix_cpp labels must be encoded as 1..n_classes");
    }
    counts(cls) += 1.0;
    class_sums.row(cls) += Tk.row(i);
  }

  const arma::mat gram = arma::symmatu(Tk.t() * Tk);
  return lda_train_moments_core(
    gram, class_sums, counts, static_cast<int>(Ttrain.n_rows), ncomp
  );
}

Rcpp::List lda_train_moments_prefix_cpp(const arma::mat& gram,
                                        const arma::mat& class_sums,
                                        const arma::vec& counts,
                                        int n,
                                        const Rcpp::IntegerVector& ncomp) {
  return lda_train_moments_core(gram, class_sums, counts, n, ncomp);
}

// [[Rcpp::export]]
Rcpp::List lda_train_prefix_cuda(const arma::mat& Ttrain,
                                 const Rcpp::IntegerVector& y,
                                 int n_classes,
                                 const Rcpp::IntegerVector& ncomp,
                                 double ridge) {
  if (!fastpls_svd::cuda_lda_native_available()) {
    Rcpp::stop(
      "CUDA LDA fitting requires native CUDA LDA support. "
      "No CPU fallback is performed."
    );
  }
  arma::ivec y_arma(y.size());
  for (R_xlen_t i = 0; i < y.size(); ++i) {
    y_arma(static_cast<arma::uword>(i)) = y[i];
  }
  arma::ivec ncomp_arma(ncomp.size());
  for (R_xlen_t i = 0; i < ncomp.size(); ++i) {
    ncomp_arma(static_cast<arma::uword>(i)) = ncomp[i];
  }
  std::vector<fastpls_svd::LDAGPUModel> gpu_models =
    fastpls_svd::cuda_lda_train_prefix(Ttrain, y_arma, n_classes, ncomp_arma, ridge);

  Rcpp::List models(ncomp.size());
  Rcpp::CharacterVector model_names(ncomp.size());
  for (R_xlen_t idx = 0; idx < ncomp.size(); ++idx) {
    models[idx] = Rcpp::List::create(
      Rcpp::Named("means") = gpu_models[static_cast<size_t>(idx)].means,
      Rcpp::Named("inv_cov") = arma::mat(),
      Rcpp::Named("linear") = gpu_models[static_cast<size_t>(idx)].linear,
      Rcpp::Named("constants") = gpu_models[static_cast<size_t>(idx)].constants,
      Rcpp::Named("priors") = gpu_models[static_cast<size_t>(idx)].priors,
      Rcpp::Named("ridge") = gpu_models[static_cast<size_t>(idx)].ridge,
      Rcpp::Named("ridge_relative") = gpu_models[static_cast<size_t>(idx)].relative_ridge,
      Rcpp::Named("backend") = "cuda_native"
    );
    model_names[idx] = std::to_string(ncomp[idx]);
  }
  models.attr("names") = model_names;
  return models;
}

// [[Rcpp::export]]
Rcpp::List lda_project_train_prefix_cuda(const arma::mat& Xtrain,
                                         const arma::mat& R,
                                         const arma::rowvec& offset,
                                         const Rcpp::IntegerVector& y,
                                         int n_classes,
                                         const Rcpp::IntegerVector& ncomp,
                                         double ridge) {
  if (!fastpls_svd::cuda_lda_native_available()) {
    Rcpp::stop(
      "Projected CUDA LDA fitting requires native CUDA LDA support. "
      "No CPU fallback is performed."
    );
  }
  arma::ivec y_arma(y.size());
  for (R_xlen_t i = 0; i < y.size(); ++i) {
    y_arma(static_cast<arma::uword>(i)) = y[i];
  }
  arma::ivec ncomp_arma(ncomp.size());
  for (R_xlen_t i = 0; i < ncomp.size(); ++i) {
    ncomp_arma(static_cast<arma::uword>(i)) = ncomp[i];
  }
  std::vector<fastpls_svd::LDAGPUModel> gpu_models =
    fastpls_svd::cuda_lda_project_train_prefix(Xtrain, R, offset, y_arma, n_classes, ncomp_arma, ridge);

  Rcpp::List models(ncomp.size());
  Rcpp::CharacterVector model_names(ncomp.size());
  for (R_xlen_t idx = 0; idx < ncomp.size(); ++idx) {
    models[idx] = Rcpp::List::create(
      Rcpp::Named("means") = gpu_models[static_cast<size_t>(idx)].means,
      Rcpp::Named("inv_cov") = arma::mat(),
      Rcpp::Named("linear") = gpu_models[static_cast<size_t>(idx)].linear,
      Rcpp::Named("constants") = gpu_models[static_cast<size_t>(idx)].constants,
      Rcpp::Named("priors") = gpu_models[static_cast<size_t>(idx)].priors,
      Rcpp::Named("ridge") = gpu_models[static_cast<size_t>(idx)].ridge,
      Rcpp::Named("ridge_relative") = gpu_models[static_cast<size_t>(idx)].relative_ridge,
      Rcpp::Named("backend") = "cuda_native_project"
    );
    model_names[idx] = std::to_string(ncomp[idx]);
  }
  models.attr("names") = model_names;
  return models;
}

// [[Rcpp::export]]
Rcpp::List lda_predict_cuda(const arma::mat& Ttest,
                            const Rcpp::List& lda) {
  if (!fastpls_svd::cuda_lda_native_available()) {
    Rcpp::stop(
      "CUDA LDA prediction requires native CUDA LDA support. "
      "No CPU fallback is performed."
    );
  }
  if (Ttest.n_rows == 0 || Ttest.n_cols == 0) {
    stop("lda_predict_cuda requires a non-empty score matrix");
  }
  arma::mat linear = Rcpp::as<arma::mat>(lda["linear"]);
  arma::rowvec constants = Rcpp::as<arma::rowvec>(lda["constants"]);
  if (Ttest.n_cols != linear.n_cols) {
    stop("lda_predict_cuda score dimension does not match the LDA model");
  }
  if (constants.n_elem != linear.n_rows) {
    stop("lda_predict_cuda has inconsistent LDA constants");
  }
  return fastpls_svd::cuda_lda_predict(Ttest, linear, constants, true);
}

// [[Rcpp::export]]
Rcpp::IntegerVector lda_predict_labels_cuda(const arma::mat& Ttest,
                                            const Rcpp::List& lda) {
  if (!fastpls_svd::cuda_lda_native_available()) {
    Rcpp::stop(
      "CUDA LDA label prediction requires native CUDA LDA support. "
      "No CPU fallback is performed."
    );
  }
  if (Ttest.n_rows == 0 || Ttest.n_cols == 0) {
    stop("lda_predict_labels_cuda requires a non-empty score matrix");
  }
  arma::mat linear = Rcpp::as<arma::mat>(lda["linear"]);
  arma::rowvec constants = Rcpp::as<arma::rowvec>(lda["constants"]);
  if (Ttest.n_cols != linear.n_cols) {
    stop("lda_predict_labels_cuda score dimension does not match the LDA model");
  }
  if (constants.n_elem != linear.n_rows) {
    stop("lda_predict_labels_cuda has inconsistent LDA constants");
  }
  Rcpp::List pred = fastpls_svd::cuda_lda_predict(Ttest, linear, constants, false);
  return pred["pred"];
}

// [[Rcpp::export]]
Rcpp::List lda_project_predict_cuda(const arma::mat& Xtest,
                                    const arma::mat& R,
                                    const arma::rowvec& offset,
                                    const Rcpp::List& lda,
                                    bool return_scores = false) {
  if (!fastpls_svd::cuda_lda_native_available()) {
    Rcpp::stop(
      "Projected CUDA LDA prediction requires native CUDA LDA support. "
      "No CPU fallback is performed."
    );
  }
  if (Xtest.n_rows == 0 || Xtest.n_cols == 0 || R.n_rows == 0 || R.n_cols == 0) {
    stop("lda_project_predict_cuda requires non-empty X and projection matrices");
  }
  arma::mat linear = Rcpp::as<arma::mat>(lda["linear"]);
  arma::rowvec constants = Rcpp::as<arma::rowvec>(lda["constants"]);
  return fastpls_svd::cuda_lda_project_predict(Xtest, R, offset, linear, constants, return_scores);
}

// [[Rcpp::export]]
Rcpp::List truncated_svd_debug(
  const arma::mat& A,
  int k,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed,
  bool left_only
) {
  fastpls_svd::SVDResult res = compute_truncated_svd_dispatch(
    A,
    k,
    svd_method,
    rsvd_oversample,
    rsvd_power,
    svds_tol,
    static_cast<unsigned int>(seed),
    left_only,
    false
  );

  return Rcpp::List::create(
    Rcpp::Named("u") = res.U,
    Rcpp::Named("d") = res.s,
    Rcpp::Named("vt") = res.Vt,
    Rcpp::Named("randomized") = res.randomized,
    Rcpp::Named("case_audited") = res.case_audited,
    Rcpp::Named("case_certified") = res.case_certified,
    Rcpp::Named("deterministic_fallback") = res.deterministic_fallback,
    Rcpp::Named("audit_attempts") = res.audit_attempts,
    Rcpp::Named("effective_oversample") = res.effective_oversample,
    Rcpp::Named("effective_power") = res.effective_power_iters,
    Rcpp::Named("effective_seed") = res.effective_seed,
    Rcpp::Named("audit_subspace_error") = res.audit_subspace_error,
    Rcpp::Named("audit_singular_value_error") = res.audit_singular_value_error,
    Rcpp::Named("audit_triplet_residual") = res.audit_triplet_residual,
    Rcpp::Named("audit_omitted_direction_ratio") = res.audit_omitted_direction_ratio
  );
}

// [[Rcpp::export]]
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
) {

  // n <-dim(Xtrain)[1]
  int n = Xtrain.n_rows;
  
  // p <-dim(Xtrain)[2]
  int p = Xtrain.n_cols;
  
  // m <- dim(Y)[2]
  int m = Ytrain.n_cols;
  
  int max_ncomp=max(ncomp);
  int length_ncomp=ncomp.n_elem;

  // X <- scale(Xtrain,center=TRUE,scale=FALSE)
  // Xtest <-scale(Xtest,center=mX)
  arma::mat mX(1,p); 
  mX.zeros();
  if(scaling<3){
    mX=mean(Xtrain,0);
    Xtrain.each_row()-=mX;
  } 
  
  arma::mat vX(1,p); 
  vX.ones();
  if(scaling==2){
    vX=variance(Xtrain); 
    Xtrain.each_row()/=vX;
  }
  
  //X=Xtrain
  arma::mat X=Xtrain;
  
  //Y=Ytrain
  arma::mat Y=Ytrain;
  
  // Y <- scale(Ytrain,center=TRUE,scale=FALSE)
  arma::mat mY=mean(Ytrain,0);
  Y.each_row()-=mY;
  
  // S <- crossprod(X,Y)
  arma::mat S=trans(X)*Y;
  
  //  RR<-matrix(0,ncol=ncomp,nrow=p)
  arma::mat RR(p,max_ncomp);
  RR.zeros();
  
  //  PP<-matrix(0,ncol=ncomp,nrow=p)
  arma::mat PP(p,max_ncomp);
  PP.zeros();
  
  //  QQ<-matrix(0,ncol=ncomp,nrow=m)
  arma::mat QQ(m,max_ncomp);
  QQ.zeros();
  
  //  TT<-matrix(0,ncol=ncomp,nrow=n)
  arma::mat TT(n,max_ncomp);
  TT.zeros();
  
  //  VV<-matrix(0,ncol=ncomp,nrow=p)
  arma::mat VV(p,max_ncomp);
  VV.zeros();
  
  const bool store_B = should_store_coefficients(p, m, length_ncomp, true);
  arma::cube B;
  if (store_B) {
    B.set_size(p, m, length_ncomp);
    B.zeros();
  }
  
  // Yfit <- matrix(0,ncol=m,nrow=n)
  arma::cube Yfit;
  arma::vec R2Y(length_ncomp);
  if(fit){
    Yfit.resize(n,m,length_ncomp);
//    Yfit.zeros();  
  }
  
  arma::mat qq;
  arma::mat pp;
  arma::mat rr;
  arma::mat tt;
  arma::mat vv;
  
  int i_out=0; //position of the saving output
  
  // for(a in 1:ncomp){
  for (int a=0; a<max_ncomp; a++) {
    //qq<-svd(S)$v[,1]
    //rr <- S%*%qq
//    if(S.n_rows<=16 || S.n_cols<=16){
    rr = leading_left_vec_dispatch(
      S,
      svd_method,
      rsvd_oversample,
      rsvd_power,
      svds_tol,
      static_cast<unsigned int>(seed + a)
    );
    if (rr.n_elem != static_cast<arma::uword>(S.n_rows)) {
      break;
    }
  
    // tt<-scale(X%*%rr,scale=FALSE)
    tt=X*rr; 
    arma::mat mtt=mean(tt,0);
    tt.each_row()-=mtt;
    
    //tnorm<-sqrt(sum(tt*tt))
    double tnorm=sqrt(sum(sum(tt%tt)));
    
    //tt<-tt/tnorm
    tt/=tnorm;
    
    //rr<-rr/tnorm
    rr/=tnorm;
    
    // pp <- crossprod(X,tt)
    pp=trans(X)*tt;
    
    // qq <- crossprod(Y,tt)
    qq=trans(Y)*tt;
    
    //vv<-pp
    vv=pp;
    
    if(a>0){
      //vv<-vv-VV%*%crossprod(VV,pp)
      vv-=VV*(trans(VV)*pp);
    }
    
    //vv <- vv/sqrt(sum(vv*vv))
    vv/=sqrt(sum(sum(vv%vv)));
    
    //S <- S-vv%*%crossprod(vv,S)
    S-=vv*(trans(vv)*S);
    
    //RR[,a]=rr
    RR.col(a)=rr;
    TT.col(a)=tt;
    PP.col(a)=pp;
    QQ.col(a)=qq;
    VV.col(a)=vv;
    
    if(a==(ncomp(i_out)-1)){
      arma::mat R_a = RR.cols(0, a);
      arma::mat Q_a = QQ.cols(0, a);
      if (store_B) {
        B.slice(i_out) = R_a * trans(Q_a);
      }
      if(fit){
        arma::mat temp1 = TT.cols(0, a) * trans(Q_a);
        temp1.each_row()+=mY;
        Yfit.slice(i_out)=temp1;
        R2Y(i_out)=RQ(Ytrain,temp1);
        
      }
      i_out++;
    }
  } 
  List out = List::create(
    Named("P")       = PP,
    Named("Q")       = QQ,
    Named("Ttrain")  = TT,
    Named("R")       = RR,
    Named("mX")      = mX,
    Named("vX")      = vX,
    Named("mY")      = mY,
    Named("p")       = p,
    Named("m")       = m,
    Named("ncomp")   = ncomp,
    Named("Yfit")    = Yfit,
    Named("R2Y")     = R2Y
  );
  if (store_B) {
    out["B"] = B;
  }
  annotate_coefficient_storage(out, store_B);
  return out;
}

List pls_model2_fast_impl(
  const arma::mat& Xinput,
  const arma::mat& Yinput,
  arma::ivec ncomp,
  int scaling,
  bool fit,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed,
  arma::mat* owned_X = nullptr,
  const arma::uvec* compact_labels = nullptr,
  int compact_classes = 0,
  const arma::mat* precomputed_crossprod = nullptr,
  const arma::mat* precomputed_crosscov = nullptr,
  const arma::mat* precomputed_x_mean = nullptr,
  const arma::mat* precomputed_x_scale = nullptr,
  const arma::mat* precomputed_y_mean = nullptr,
  int precomputed_training_rows = 0
) {
  using Clock = std::chrono::steady_clock;
  const auto started = Clock::now();
  fastpls::native::SimplsOptions controls;
  controls.scaling = scaling;
  controls.fitted = fit;
  controls.svd.oversample = rsvd_oversample;
  controls.svd.power = rsvd_power;
  controls.svd.seed = static_cast<unsigned int>(seed);
  controls.phase_timing = env_int_or("FASTPLS_BENCH_PHASE_TIMING", 0, 0, 1) == 1;
  controls.center_scores = env_int_or("FASTPLS_FAST_CENTER_T", 0, 0, 1);
  controls.reorthogonalize = env_int_or("FASTPLS_FAST_REORTH_V", 0, 0, 1);
  controls.cache_deflation = env_int_or("FASTPLS_FAST_DEFLCACHE", 1, 0, 1);
  controls.cache_crossprod = env_int_or("FASTPLS_FAST_OPTIMIZED", 1, 0, 1);
  controls.incremental_coefficients = env_int_or("FASTPLS_INCREMENTAL_COEFFICIENTS", 1, 0, 1);
  controls.crossprod_min_components = env_int_or("FASTPLS_FAST_CROSSPROD_MIN_NCOMP", 20, 1, 1024);
#ifdef FASTPLS_USE_ACCELERATE
  constexpr int max_predictors = 2048;
#else
  constexpr int max_predictors = 512;
#endif
  controls.crossprod_max_predictors = env_int_or("FASTPLS_FAST_CROSSPROD_MAX_P", max_predictors, 16, 65536);
  controls.crossprod_min_ratio = env_int_or("FASTPLS_FAST_CROSSPROD_MIN_N_TO_P_RATIO", 8, 1, 1024);
  controls.store_scores = env_int_or("FASTPLS_RETURN_TTRAIN", 0, 0, 1) == 1;
  const int response_columns = compact_labels == nullptr ?
    static_cast<int>(Yinput.n_cols) : compact_classes;
  controls.store_coefficients = should_store_coefficients(
    Xinput.n_cols, response_columns, ncomp.n_elem, true
  );
  controls.randomized_directions =
    svd_method == fastpls_svd::SVD_METHOD_CPU_RSVD ||
    svd_method == fastpls_svd::SVD_METHOD_CUDA_RSVD;

  fastpls::native::DirectionWorkspace<double> refresh;
  refresh.eigen_threshold = env_int_or("FASTPLS_GPU_FINALIZE_THRESHOLD", 4, 1, 256);
  auto solver = [&](const arma::mat& S, const arma::mat& gram, bool use_gram,
                    int width, unsigned int direction_seed, arma::mat& directions) {
    if (controls.randomized_directions) {
      return use_gram ? refresh.refresh_from_right_gram(
        S, gram, width, rsvd_oversample, std::max(rsvd_power, 0), direction_seed, directions
      ) : refresh.refresh(
        S, width, rsvd_oversample, std::max(rsvd_power, 0), direction_seed, directions
      );
    }
    auto decomposition = compute_truncated_svd_dispatch(S, 1, svd_method,
      rsvd_oversample, rsvd_power, svds_tol, direction_seed, true, false);
    directions = std::move(decomposition.U);
    return directions.n_cols > 0;
  };
  auto model = fastpls::native::fit_simpls_with_solver(
    Xinput, Yinput, std::move(ncomp), controls, solver, owned_X,
    compact_labels, compact_classes, {}, {}, {}, precomputed_crossprod,
    precomputed_crosscov, precomputed_x_mean, precomputed_x_scale,
    precomputed_y_mean, precomputed_training_rows
  );
  const auto assembly_started = controls.phase_timing ? Clock::now() : Clock::time_point();
  List out = List::create(
    Named("P") = arma::mat(), Named("Q") = model.Q,
    Named("Ttrain") = model.scores, Named("R") = model.R,
    Named("mX") = model.x_mean, Named("vX") = model.x_scale,
    Named("mY") = model.y_mean, Named("p") = static_cast<int>(Xinput.n_cols),
    Named("m") = response_columns, Named("ncomp") = model.components,
    Named("Yfit") = model.fitted, Named("R2Y") = model.r2
  );
  if (controls.store_coefficients) out["B"] = model.coefficients;
  annotate_coefficient_storage(out, controls.store_coefficients);
  if (controls.phase_timing) {
    const double assembly = std::chrono::duration<double>(Clock::now() - assembly_started).count();
    const double total = std::chrono::duration<double>(Clock::now() - started).count();
    const auto& t = model.timing;
    out["benchmark_phase_timing"] = List::create(
      Named("preprocess_crosscov_sec") = t.preprocess,
      Named("response_crosscov_sec") = t.response_crosscov,
      Named("crossprod_cache_sec") = t.crossprod_cache,
      Named("right_gram_sec") = t.right_gram,
      Named("estimator_sec") = t.estimator,
      Named("direction_sec") = t.direction,
      Named("component_update_sec") = t.component_update,
      Named("coefficient_path_sec") = t.coefficients,
      Named("fitted_values_sec") = t.fitted,
      Named("model_assembly_sec") = assembly,
      Named("cpp_total_sec") = total
    );
  }
  return out;
}

// [[Rcpp::export]]
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
) {
  const arma::mat Xtrain = numeric_matrix_view(XtrainSEXP, "Xtrain");
  const arma::mat Ytrain = numeric_matrix_view(YtrainSEXP, "Ytrain");
  return pls_model2_fast_impl(
    Xtrain, Ytrain, std::move(ncomp), scaling, fit, svd_method,
    rsvd_oversample, rsvd_power, svds_tol, seed
  );
}

// Fit double-precision PLS-DA directly from compact class labels.
// [[Rcpp::export]]
List pls_labels_cpp(
  SEXP XtrainSEXP,
  const Rcpp::IntegerVector& labels,
  int n_classes,
  arma::ivec ncomp,
  int scaling,
  bool fit,
  int method,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed
) {
  const arma::mat Xtrain = numeric_matrix_view(XtrainSEXP, "Xtrain");
  if (labels.size() != static_cast<R_xlen_t>(Xtrain.n_rows) || n_classes < 2) {
    stop("label-aware PLS requires one valid label per row and at least two classes");
  }
  arma::uvec compact_labels(Xtrain.n_rows);
  arma::uvec counts(static_cast<arma::uword>(n_classes), arma::fill::zeros);
  for (arma::uword row = 0; row < Xtrain.n_rows; ++row) {
    const int label = labels[static_cast<R_xlen_t>(row)] - 1;
    if (label < 0 || label >= n_classes) {
      stop("label-aware PLS labels must be encoded as 1..n_classes");
    }
    compact_labels(row) = static_cast<arma::uword>(label);
    counts(static_cast<arma::uword>(label)) += 1;
  }
  if (arma::accu(counts == 0) > 0) {
    stop("label-aware PLS received an empty class");
  }
  const arma::mat no_dense_response;
  if (method == 3) {
    return pls_model2_fast_impl(
      Xtrain, no_dense_response, std::move(ncomp), scaling, fit, svd_method,
      rsvd_oversample, rsvd_power, svds_tol, seed, nullptr,
      &compact_labels, n_classes
    );
  }
  if (method != 1) {
    stop("label-aware PLS supports PLS-SVD or accelerated SIMPLS");
  }

  const int p = static_cast<int>(Xtrain.n_cols);
  fastpls::native::PlssvdOptions controls;
  controls.scaling = scaling;
  controls.fitted = fit;
  controls.store_coefficients = should_store_coefficients(
    p, n_classes, ncomp.n_elem, true
  );
  controls.cache_score_gram = env_int_or("FASTPLS_PLSSVD_OPTIMIZED", 1, 0, 1);
  auto solver = [&](const arma::mat& S, int retained, int rank_bound) {
    auto result = compute_truncated_svd_dispatch(
      S, retained, svd_method, rsvd_oversample, rsvd_power, svds_tol,
      static_cast<unsigned int>(seed), false,
      plssvd_use_small_exact_svd(rank_bound, svd_method)
    );
    return fastpls::native::SingularTriplets<double>{
      std::move(result.U), std::move(result.s), std::move(result.Vt)
    };
  };
  auto model = fastpls::native::fit_plssvd_with_solver(
    Xtrain, no_dense_response, std::move(ncomp), controls, solver,
    static_cast<arma::mat*>(nullptr),
    &compact_labels, n_classes
  );
  List out = List::create(
    Named("C_latent") = model.latent_coefficients,
    Named("W_latent") = model.prediction_weights,
    Named("Q") = model.Q, Named("Ttrain") = model.scores,
    Named("R") = model.R, Named("mX") = model.x_mean,
    Named("vX") = model.x_scale, Named("mY") = model.y_mean,
    Named("p") = p, Named("m") = n_classes,
    Named("ncomp") = model.components, Named("Yfit") = model.fitted,
    Named("R2Y") = model.r2
  );
  if (controls.store_coefficients) out["B"] = model.coefficients;
  annotate_coefficient_storage(out, controls.store_coefficients);
  return out;
}

List pls_model2_fast_gpu_impl(
  const arma::mat& Xinput,
  arma::mat Ytrain,
  arma::ivec ncomp,
  int scaling,
  bool fit,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed,
  arma::mat* owned_X = nullptr
) {
  if (!fastpls_svd::has_cuda_backend()) {
    stop(
      "pls_model2_fast_gpu requires an available CUDA backend. "
      "No CPU fallback is performed."
    );
  }
  if (svd_method != fastpls_svd::SVD_METHOD_CUDA_RSVD) {
    stop("pls_model2_fast_gpu requires svd.method='cuda_rsvd'");
  }

  const int n = Xinput.n_rows;
  const int p = Xinput.n_cols;
  const int m = Ytrain.n_cols;

  if (ncomp.n_elem < 1) {
    stop("ncomp must contain at least one value");
  }
  for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
    if (ncomp(i) < 1) {
      ncomp(i) = 1;
    }
  }

  const int max_ncomp = max(ncomp);
  const int length_ncomp = ncomp.n_elem;
  const bool classification_response =
    n >= 5000 && max_ncomp >= 50 && is_one_hot_response(Ytrain);

  arma::mat Xwork;
  const arma::mat* Xptr = &Xinput;
  arma::mat mX(1, p, fill::zeros);
  arma::mat& scaled_X = owned_X == nullptr ? Xwork : *owned_X;
  if (scaling < 3) {
    if (owned_X == nullptr) scaled_X = Xinput;
    mX = mean(scaled_X, 0);
    scaled_X.each_row() -= mX;
    Xptr = &scaled_X;
  }

  arma::mat vX(1, p, fill::ones);
  if (scaling == 2) {
    vX = variance(scaled_X);
    scaled_X.each_row() /= vX;
  }
  const arma::mat& Xtrain = *Xptr;

  arma::mat mY = mean(Ytrain, 0);
  Ytrain.each_row() -= mY;

  const bool use_implicit_xprod =
    (env_int_or("FASTPLS_GPU_SIMPLS_XPROD", 0, 0, 1) == 1);
  const bool use_device_state =
    (env_int_or("FASTPLS_GPU_DEVICE_STATE", 0, 0, 1) == 1);
  arma::mat Xt;
  arma::mat Yt;
  if (!use_implicit_xprod && !use_device_state) {
    Xt = Xtrain.t();
    Yt = Ytrain.t();
  }

  arma::mat RR(p, max_ncomp, fill::zeros);
  arma::mat QQ(m, max_ncomp, fill::zeros);
  arma::mat VV(p, max_ncomp, fill::zeros);
  const bool store_B = should_store_coefficients(p, m, length_ncomp, true);
  arma::cube B;
  if (store_B) {
    B.zeros(p, m, length_ncomp);
  }

  arma::cube Yfit;
  arma::vec R2Y(length_ncomp, fill::zeros);
  arma::mat Yfit_cur;
  if (fit) {
    Yfit.set_size(n, m, length_ncomp);
    Yfit_cur.zeros(n, m);
  }

  arma::mat Bcur;
  if (store_B) {
    Bcur.zeros(p, m);
  }
  int i_out = 0;

  const int center_t = env_int_or("FASTPLS_FAST_CENTER_T", 0, 0, 1);
  const int reorth_v = env_int_or("FASTPLS_FAST_REORTH_V", 0, 0, 1);
  const int defl_cache = env_int_or("FASTPLS_FAST_DEFLCACHE", 1, 0, 1);
  (void)defl_cache;
  const int sketch_dim = std::min(
    std::min(p, m),
    1 + std::max(rsvd_oversample, 0)
  );
  const int requested_power_iters = std::max(rsvd_power, 0);
  if (center_t == 1) {
    stop("pls_model2_fast_gpu does not support FASTPLS_FAST_CENTER_T=1");
  }

  fastpls_svd::cuda_simpls_fast_set_training_matrices(
    Xtrain.memptr(),
    n,
    p,
    Ytrain.memptr(),
    m,
    fit,
    !use_implicit_xprod
  );
  if (use_device_state) {
    fastpls_svd::cuda_simpls_fast_begin_device_loop(
      n, p, m, max_ncomp, fit, store_B
    );
    int a = 0;
    while (a < max_ncomp) {
      const int k_block = accelerated_simpls_block_size(
        max_ncomp - a, p, m, classification_response, n,
        std::min(kAcceleratorSimplsBlockSize, sketch_dim)
      );
      const int refresh_width = sketch_dim;
      const int refresh_power = requested_power_iters;
      arma::vec shat_block(k_block, arma::fill::zeros);
      if (use_implicit_xprod) {
        fastpls_svd::cuda_simpls_fast_refresh_block_implicit_resident(
          n,
          p,
          m,
          refresh_width,
          k_block,
          a,
          static_cast<unsigned int>(seed + a),
          refresh_power,
          shat_block.memptr()
        );
      } else {
        fastpls_svd::cuda_simpls_fast_refresh_block_resident(
          p,
          m,
          refresh_width,
          k_block,
          static_cast<unsigned int>(seed + a),
          refresh_power,
          shat_block.memptr()
        );
      }

      bool stop_now = false;
      for (int j = 0; j < k_block && a < max_ncomp;) {
        bool used_retry_refresh = false;
        bool appended = fastpls_svd::cuda_simpls_fast_append_component_from_block(
              n,
              p,
              m,
              a,
              j,
              a,
              (reorth_v == 1),
              fit,
              !use_implicit_xprod
            );
        if (!appended) {
          // A randomized direction can occasionally land in a direction
          // removed by SIMPLS deflation. Retry with a fresh independent sketch
          // instead of terminating the coefficient path.
          const int max_gpu_refresh_retries = 8;
          for (int retry = 0; retry < max_gpu_refresh_retries && !appended; ++retry) {
            arma::vec retry_shat(1, arma::fill::zeros);
            const unsigned int retry_seed =
              static_cast<unsigned int>(seed + a + 7919 * (retry + 1));
            if (use_implicit_xprod) {
              fastpls_svd::cuda_simpls_fast_refresh_block_implicit_resident(
                n,
                p,
                m,
                sketch_dim,
                1,
                a,
                retry_seed,
                requested_power_iters,
                retry_shat.memptr()
              );
            } else {
              fastpls_svd::cuda_simpls_fast_refresh_block_resident(
                p,
                m,
                sketch_dim,
                1,
                retry_seed,
                requested_power_iters,
                retry_shat.memptr()
              );
            }
            appended = fastpls_svd::cuda_simpls_fast_append_component_from_block(
              n,
              p,
              m,
              a,
              0,
              a,
              (reorth_v == 1),
              fit,
              !use_implicit_xprod
            );
            used_retry_refresh = appended;
          }
        }
        if (!appended) {
          stop_now = true;
          break;
        }
        while (i_out < length_ncomp && a == (ncomp(i_out) - 1)) {
          if (store_B) {
            fastpls_svd::cuda_simpls_fast_copy_bcur(B.slice(i_out).memptr(), p, m);
          }
          if (fit) {
            fastpls_svd::cuda_simpls_fast_copy_yfit(Yfit_cur.memptr(), n, m);
            R2Y(i_out) = RQ(Ytrain, Yfit_cur);
            arma::mat yf = Yfit_cur;
            yf.each_row() += mY;
            Yfit.slice(i_out) = yf;
          }
          ++i_out;
        }
        ++a;
        if (used_retry_refresh) {
          break;
        }
        ++j;
      }
      if (stop_now) {
        break;
      }
    }

    while (i_out < length_ncomp) {
      if (store_B) {
        fastpls_svd::cuda_simpls_fast_copy_bcur(B.slice(i_out).memptr(), p, m);
      }
      if (fit) {
        fastpls_svd::cuda_simpls_fast_copy_yfit(Yfit_cur.memptr(), n, m);
        R2Y(i_out) = RQ(Ytrain, Yfit_cur);
        arma::mat yf = Yfit_cur;
        yf.each_row() += mY;
        Yfit.slice(i_out) = yf;
      }
      ++i_out;
    }

    fastpls_svd::cuda_simpls_fast_copy_rr(RR.memptr(), p, max_ncomp);
    fastpls_svd::cuda_simpls_fast_copy_qq(QQ.memptr(), m, max_ncomp);
  } else {
    arma::mat S_shape;
    if (!use_implicit_xprod) {
      S_shape = Xt * Ytrain;
    }
    SimplsFastRefreshWorkspace refresh_ws;
    refresh_ws.gpu_refresh_enabled = false;
    auto append_component = [&](arma::vec rr, const int a_idx) -> bool {
      arma::vec tt(n, arma::fill::zeros);
      arma::vec pp(p, arma::fill::zeros);
      arma::vec qq(m, arma::fill::zeros);
      double tnorm = 0.0;
      bool gpu_stats_ok = true;
      try {
        fastpls_svd::cuda_simpls_fast_component_stats(
          rr.memptr(),
          n,
          p,
          m,
          tt.memptr(),
          pp.memptr(),
          qq.memptr(),
          &tnorm
        );
      } catch (const std::exception&) {
        gpu_stats_ok = false;
      }

      if (!gpu_stats_ok || !std::isfinite(tnorm) || tnorm <= 0.0) {
        tt = Xtrain * rr;
        const double host_tnorm = arma::norm(tt, 2);
        if (!std::isfinite(host_tnorm) || host_tnorm <= 0.0) {
          return false;
        }
        tt /= host_tnorm;
        rr /= host_tnorm;
        pp = Xtrain.t() * tt;
        qq = Ytrain.t() * tt;
      } else {
        rr /= tnorm;
      }

      arma::vec vv = pp;
      if (a_idx > 0) {
        auto Vprev = VV.cols(0, a_idx - 1);
        vv -= Vprev * (Vprev.t() * pp);
        if (reorth_v == 1) {
          vv -= Vprev * (Vprev.t() * vv);
        }
      }
      const double vnorm = arma::norm(vv, 2);
      if (!std::isfinite(vnorm) || vnorm <= 0.0) {
        return false;
      }
      vv /= vnorm;

      if (!use_implicit_xprod) {
        arma::rowvec vS = vv.t() * S_shape;
        S_shape -= vv * vS;
      }

      RR.col(a_idx) = rr;
      QQ.col(a_idx) = qq;
      VV.col(a_idx) = vv;
      if (store_B) {
        Bcur += rr * qq.t();
      }

      while (i_out < length_ncomp && a_idx == (ncomp(i_out) - 1)) {
        if (store_B) {
          B.slice(i_out) = Bcur;
        }
        if (fit) {
          fastpls_svd::cuda_simpls_fast_rank1_fit_update(
            tt.memptr(),
            n,
            qq.memptr(),
            m,
            Yfit_cur.memptr()
          );
          R2Y(i_out) = RQ(Ytrain, Yfit_cur);
          arma::mat yf = Yfit_cur;
          yf.each_row() += mY;
          Yfit.slice(i_out) = yf;
        }
        ++i_out;
      }
      return true;
    };

    int a = 0;
    while (a < max_ncomp) {
      const int k_block = accelerated_simpls_block_size(
        max_ncomp - a, p, m, classification_response, n,
        std::min(kAcceleratorSimplsBlockSize, sketch_dim)
      );
      arma::mat Ublock;
      if (use_implicit_xprod) {
        arma::vec shat_block;
        if (!refresh_deflated_crossprod_left_double(
              Xtrain,
              Ytrain,
              VV,
              a,
              k_block,
              rsvd_oversample,
              requested_power_iters,
              static_cast<unsigned int>(seed + a),
              Ublock,
              shat_block
            )) {
          break;
        }
      } else {
        if (!refresh_ws.refresh(
              S_shape,
              k_block,
              rsvd_oversample,
              requested_power_iters,
              static_cast<unsigned int>(seed + a),
              Ublock
            )) {
          break;
        }
      }
      if (Ublock.n_cols < 1) {
        break;
      }

      const int use_cols = std::min<int>(Ublock.n_cols, k_block);
      bool stop_now = false;
      for (int j = 0; j < use_cols && a < max_ncomp; ++j, ++a) {
        if (!append_component(Ublock.col(j), a)) {
          stop_now = true;
          break;
        }
      }
      if (stop_now) {
        break;
      }
    }
  }

  List out = List::create(
    Named("P")       = arma::mat(),
    Named("Q")       = QQ,
    Named("Ttrain")  = arma::mat(),
    Named("R")       = RR,
    Named("mX")      = mX,
    Named("vX")      = vX,
    Named("mY")      = mY,
    Named("p")       = p,
    Named("m")       = m,
    Named("ncomp")   = ncomp,
    Named("Yfit")    = Yfit,
    Named("R2Y")     = R2Y,
    Named("xprod_mode") = use_implicit_xprod ?
      (use_device_state ? "implicit_resident" : "implicit") :
      (use_device_state ? "materialized_resident" : "materialized"),
    Named("gpu_resident") = use_device_state
  );
  if (store_B) {
    out["B"] = B;
  }
  annotate_coefficient_storage(out, store_B);
  return out;
}

// [[Rcpp::export]]
List pls_model2_fast_gpu(
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
) {
  const arma::mat Xview = numeric_matrix_view(XtrainSEXP, "Xtrain");
  const arma::mat Yview = numeric_matrix_view(YtrainSEXP, "Ytrain");
  return pls_model2_fast_gpu_impl(
    Xview,
    Yview,
    std::move(ncomp),
    scaling,
    fit,
    svd_method,
    rsvd_oversample,
    rsvd_power,
    svds_tol,
    seed
  );
}

List pls_predict_impl(List& model, const arma::mat& Xinput, bool proj) {

  // columns of Ytrain
  const int m = Rcpp::as<int>(model["m"]);
  
  // w <-dim(Xtest)[1]
  const int w = Xinput.n_rows;
  
  arma::ivec ncomp = Rcpp::as<arma::ivec>(model["ncomp"]);
  const arma::uword length_ncomp = static_cast<arma::uword>(ncomp.n_elem);
  
  //scaling factors
  Rcpp::NumericVector mX_vec = model["mX"];
  arma::rowvec mX(mX_vec.begin(), mX_vec.size(), false, true);
  Rcpp::NumericVector vX_vec = model["vX"];
  arma::rowvec vX(vX_vec.begin(), vX_vec.size(), false, true);
  bool transform_x = false;
  for (arma::uword col = 0; col < mX.n_elem; ++col) {
    if (mX(col) != 0.0 || vX(col) != 1.0) {
      transform_x = true;
      break;
    }
  }
  arma::mat Xwork;
  const arma::mat* Xptr = &Xinput;
  if (transform_x) {
    Xwork = Xinput;
    Xwork.each_row() -= mX;
    Xwork.each_row() /= vX;
    Xptr = &Xwork;
  }
  const arma::mat& Xtest = *Xptr;
  Rcpp::NumericVector mY_vec = model["mY"];
  arma::rowvec mY(mY_vec.begin(), mY_vec.size(), false, true);
  const bool add_response_mean = arma::any(mY != 0.0);

  arma::cube Ypred(w, m, length_ncomp, arma::fill::none);
  bool used_latent_predict = false;
  arma::mat latent_scores;

  std::string pls_method;
  if (model.containsElementNamed("pls_method")) {
    pls_method = Rcpp::as<std::string>(model["pls_method"]);
  }
  bool latent_predict_enabled = false;
  if (model.containsElementNamed("predict_latent_ok")) {
    latent_predict_enabled = Rcpp::as<bool>(model["predict_latent_ok"]);
  }

  const int latent_min_b_mb = env_int_or("FASTPLS_PREDICT_LATENT_MIN_B_MB", 256, 0, 1048576);
  const double coefficient_matrix_mb =
    static_cast<double>(Xtest.n_cols) * static_cast<double>(m) * sizeof(double) /
    (1024.0 * 1024.0);
  const bool prefer_latent_predict =
    (latent_min_b_mb == 0) || (coefficient_matrix_mb >= static_cast<double>(latent_min_b_mb));
  const bool has_B = model.containsElementNamed("B");
  const bool use_latent_predict = prefer_latent_predict || !has_B;

  if (latent_predict_enabled &&
      use_latent_predict &&
      (pls_method == "simpls" || pls_method == "simpls_fast")) {
    Rcpp::NumericVector R_vec = model["R"];
    Rcpp::NumericVector Q_vec = model["Q"];
    Rcpp::IntegerVector R_dim = R_vec.attr("dim");
    Rcpp::IntegerVector Q_dim = Q_vec.attr("dim");
    if (R_dim.size() == 2L && Q_dim.size() == 2L &&
        R_dim[0] == Xtest.n_cols && Q_dim[0] == m &&
        R_dim[1] > 0 && Q_dim[1] > 0) {
      const arma::mat RR(
        R_vec.begin(),
        static_cast<arma::uword>(R_dim[0]),
        static_cast<arma::uword>(R_dim[1]),
        false,
        true
      );
      const arma::mat QQ(
        Q_vec.begin(),
        static_cast<arma::uword>(Q_dim[0]),
        static_cast<arma::uword>(Q_dim[1]),
        false,
        true
      );
      bool latent_ok = length_ncomp > 0 && ncomp.min() >= 1 &&
        ncomp.max() <= static_cast<int>(std::min(RR.n_cols, QQ.n_cols));
      if (latent_ok) {
        const arma::uword columns = proj ? RR.n_cols :
          static_cast<arma::uword>(ncomp.max());
        latent_scores = Xtest * RR.cols(0, columns - 1);
      }
      for (arma::uword a = 0; a < length_ncomp; ++a) {
        if (!latent_ok) break;
        const int mc = ncomp(a);
        if (mc < 1 ||
            mc > static_cast<int>(RR.n_cols) ||
            mc > static_cast<int>(QQ.n_cols)) {
          latent_ok = false;
          break;
        }
        const arma::mat scores(latent_scores.memptr(), latent_scores.n_rows,
                               static_cast<arma::uword>(mc), false, true);
        Ypred.slice(a) = scores * QQ.cols(0, static_cast<arma::uword>(mc - 1)).t();
        Ypred.slice(a).each_row() += mY;
      }
      used_latent_predict = latent_ok;
    }
  }

  if (!used_latent_predict &&
      use_latent_predict &&
      pls_method == "plssvd" &&
      !model.containsElementNamed("W_latent") &&
      model.containsElementNamed("C_latent")) {
    Rcpp::NumericVector R_vec = model["R"];
    Rcpp::NumericVector Q_vec = model["Q"];
    Rcpp::NumericVector C_vec = model["C_latent"];
    Rcpp::IntegerVector R_dim = R_vec.attr("dim");
    Rcpp::IntegerVector Q_dim = Q_vec.attr("dim");
    Rcpp::IntegerVector C_dim = C_vec.attr("dim");
    if (R_dim.size() == 2L && Q_dim.size() == 2L && C_dim.size() == 3L &&
        R_dim[0] == Xtest.n_cols && Q_dim[0] == m &&
        C_dim[0] == R_dim[1] && C_dim[1] == R_dim[1] &&
        C_dim[2] >= static_cast<int>(length_ncomp) &&
        R_dim[1] > 0 && Q_dim[1] == R_dim[1]) {
      const arma::mat RR(
        R_vec.begin(),
        static_cast<arma::uword>(R_dim[0]),
        static_cast<arma::uword>(R_dim[1]),
        false,
        true
      );
      const arma::mat QQ(
        Q_vec.begin(),
        static_cast<arma::uword>(Q_dim[0]),
        static_cast<arma::uword>(Q_dim[1]),
        false,
        true
      );
      const arma::cube CC(
        C_vec.begin(),
        static_cast<arma::uword>(C_dim[0]),
        static_cast<arma::uword>(C_dim[1]),
        static_cast<arma::uword>(C_dim[2]),
        false,
        true
      );
      bool latent_ok = length_ncomp > 0 && ncomp.min() >= 1 &&
        ncomp.max() <= static_cast<int>(std::min(RR.n_cols, QQ.n_cols));
      if (latent_ok) {
        const arma::uword columns = proj ? RR.n_cols :
          static_cast<arma::uword>(ncomp.max());
        latent_scores = Xtest * RR.cols(0, columns - 1);
      }
      for (arma::uword a = 0; a < length_ncomp; ++a) {
        if (!latent_ok) break;
        const int mc = ncomp(a);
        if (mc < 1 ||
            mc > static_cast<int>(RR.n_cols) ||
            mc > static_cast<int>(QQ.n_cols) ||
            a >= CC.n_slices) {
          latent_ok = false;
          break;
        }
        arma::mat coeff = CC.slice(a).submat(0, 0, mc - 1, mc - 1);
        const arma::mat scores(latent_scores.memptr(), latent_scores.n_rows,
                               static_cast<arma::uword>(mc), false, true);
        Ypred.slice(a) = scores * coeff * QQ.cols(0, static_cast<arma::uword>(mc - 1)).t();
        Ypred.slice(a).each_row() += mY;
      }
      used_latent_predict = latent_ok;
    }
  }

  if (!used_latent_predict &&
      use_latent_predict &&
      pls_method == "plssvd" &&
      model.containsElementNamed("W_latent")) {
    Rcpp::NumericVector R_vec = model["R"];
    Rcpp::NumericVector W_vec = model["W_latent"];
    Rcpp::IntegerVector R_dim = R_vec.attr("dim");
    Rcpp::IntegerVector W_dim = W_vec.attr("dim");
    if (R_dim.size() == 2L && W_dim.size() == 3L &&
        R_dim[0] == Xtest.n_cols &&
        W_dim[0] == R_dim[1] && W_dim[1] == m &&
        W_dim[2] >= static_cast<int>(length_ncomp) &&
        R_dim[1] > 0) {
      const arma::mat RR(
        R_vec.begin(),
        static_cast<arma::uword>(R_dim[0]),
        static_cast<arma::uword>(R_dim[1]),
        false,
        true
      );
      const arma::cube WW(
        W_vec.begin(),
        static_cast<arma::uword>(W_dim[0]),
        static_cast<arma::uword>(W_dim[1]),
        static_cast<arma::uword>(W_dim[2]),
        false,
        true
      );
      bool latent_ok = length_ncomp > 0 && ncomp.min() >= 1 &&
        ncomp.max() <= static_cast<int>(std::min(RR.n_cols, WW.n_rows));
      if (latent_ok) {
        const arma::uword columns = proj ? RR.n_cols :
          static_cast<arma::uword>(ncomp.max());
        latent_scores = Xtest * RR.cols(0, columns - 1);
      }
      for (arma::uword a = 0; a < length_ncomp; ++a) {
        if (!latent_ok) break;
        const int mc = ncomp(a);
        if (mc < 1 ||
            mc > static_cast<int>(RR.n_cols) ||
            mc > static_cast<int>(WW.n_rows) ||
            a >= WW.n_slices) {
          latent_ok = false;
          break;
        }
        const arma::mat scores(latent_scores.memptr(), latent_scores.n_rows,
                               static_cast<arma::uword>(mc), false, true);
        Ypred.slice(a) = scores * WW.slice(a).rows(0, static_cast<arma::uword>(mc - 1));
        Ypred.slice(a).each_row() += mY;
      }
      used_latent_predict = latent_ok;
    }
  }

  if (!used_latent_predict) {
    if (!has_B) {
      Rcpp::stop("Model does not store `B`, and compact latent prediction was not available");
    }
    Rcpp::NumericVector B_vec = model["B"];
    Rcpp::IntegerVector B_dim = B_vec.attr("dim");
    if (B_dim.size() != 3L) {
      Rcpp::stop("Model coefficient array `B` must have 3 dimensions");
    }
    const arma::cube B(
      B_vec.begin(),
      static_cast<arma::uword>(B_dim[0]),
      static_cast<arma::uword>(B_dim[1]),
      static_cast<arma::uword>(B_dim[2]),
      false,
      true
    );
    if (B.n_slices < length_ncomp) {
      Rcpp::stop("Model coefficient array `B` has fewer slices than `ncomp`");
    }
    for (arma::uword a = 0; a < length_ncomp; ++a) {
      dense_product_into(Xtest, B.slice(a), Ypred.slice(a));
      if (add_response_mean) {
        Ypred.slice(a).each_row() += mY;
      }
    }
  }

  arma::mat T_Xtest;
  if (proj && used_latent_predict) {
    T_Xtest = std::move(latent_scores);
  } else if (proj) {
    Rcpp::NumericVector RR_vec = model["R"];
    Rcpp::IntegerVector RR_dim = RR_vec.attr("dim");
    if (RR_dim.size() == 2L && RR_dim[0] > 0 && RR_dim[1] > 0) {
      const arma::mat RR(
        RR_vec.begin(),
        static_cast<arma::uword>(RR_dim[0]),
        static_cast<arma::uword>(RR_dim[1]),
        false,
        true
      );
      T_Xtest = Xtest*RR;
    } else {
      T_Xtest.set_size(w, 0);
    }
  }

  return List::create(
    Named("Ypred")  = Ypred,
    Named("Ttest")   = T_Xtest
  );
}

// [[Rcpp::export]]
List pls_predict(List& model, SEXP XtestSEXP, bool proj) {
  const arma::mat Xtest = numeric_matrix_view(XtestSEXP, "Xtest");
  return pls_predict_impl(model, Xtest, proj);
}

// [[Rcpp::export]]
List pls_predict_flash_cuda(List& model, arma::mat Xtest, bool proj) {
  if (!fastpls_svd::has_cuda_backend()) {
    Rcpp::stop(
      "pls_predict_flash_cuda requires an available CUDA backend. "
      "No CPU fallback is performed."
    );
  }

  const int m = Rcpp::as<int>(model["m"]);
  arma::ivec ncomp = Rcpp::as<arma::ivec>(model["ncomp"]);
  const arma::uword length_ncomp = static_cast<arma::uword>(ncomp.n_elem);

  Rcpp::NumericVector mX_vec = model["mX"];
  arma::rowvec mX(mX_vec.begin(), mX_vec.size(), false, true);
  Xtest.each_row() -= mX;
  Rcpp::NumericVector vX_vec = model["vX"];
  arma::rowvec vX(vX_vec.begin(), vX_vec.size(), false, true);
  Xtest.each_row() /= vX;
  Rcpp::NumericVector mY_vec = model["mY"];
  arma::rowvec mY(mY_vec.begin(), mY_vec.size(), false, true);

  std::string pls_method;
  if (model.containsElementNamed("pls_method")) {
    pls_method = Rcpp::as<std::string>(model["pls_method"]);
  }

  Rcpp::NumericVector R_vec = model["R"];
  Rcpp::IntegerVector R_dim = R_vec.attr("dim");
  if (R_dim.size() != 2L || R_dim[0] != Xtest.n_cols || R_dim[1] < 1) {
    Rcpp::stop("Model `R` is not compatible with CUDA flash prediction");
  }
  const arma::mat RR(
    R_vec.begin(),
    static_cast<arma::uword>(R_dim[0]),
    static_cast<arma::uword>(R_dim[1]),
    false,
    true
  );
  const int kmax = static_cast<int>(RR.n_cols);

  arma::cube Wflash;
  if ((pls_method == "simpls" || pls_method == "simpls_fast") &&
      model.containsElementNamed("Q")) {
    Rcpp::NumericVector Q_vec = model["Q"];
    Rcpp::IntegerVector Q_dim = Q_vec.attr("dim");
    if (Q_dim.size() != 2L || Q_dim[0] != m || Q_dim[1] < 1) {
      Rcpp::stop("Model `Q` is not compatible with CUDA flash prediction");
    }
    const arma::mat QQ(
      Q_vec.begin(),
      static_cast<arma::uword>(Q_dim[0]),
      static_cast<arma::uword>(Q_dim[1]),
      false,
      true
    );
    Wflash.zeros(kmax, m, length_ncomp);
    for (arma::uword a = 0; a < length_ncomp; ++a) {
      int mc = ncomp(a);
      if (mc < 1 || mc > kmax || mc > static_cast<int>(QQ.n_cols)) {
        Rcpp::stop("ncomp exceeds latent rank for CUDA flash prediction");
      }
      Wflash.slice(a).rows(0, static_cast<arma::uword>(mc - 1)) =
        QQ.cols(0, static_cast<arma::uword>(mc - 1)).t();
    }
  } else if (pls_method == "plssvd" && model.containsElementNamed("W_latent")) {
    Rcpp::NumericVector W_vec = model["W_latent"];
    Rcpp::IntegerVector W_dim = W_vec.attr("dim");
    if (W_dim.size() != 3L || W_dim[0] != kmax || W_dim[1] != m ||
        W_dim[2] < static_cast<int>(length_ncomp)) {
      Rcpp::stop("Model `W_latent` is not compatible with CUDA flash prediction");
    }
    const arma::cube WW(
      W_vec.begin(),
      static_cast<arma::uword>(W_dim[0]),
      static_cast<arma::uword>(W_dim[1]),
      static_cast<arma::uword>(W_dim[2]),
      false,
      true
    );
    Wflash = WW.slices(0, length_ncomp - 1);
  } else if (pls_method == "plssvd" &&
             model.containsElementNamed("C_latent") &&
             model.containsElementNamed("Q")) {
    Rcpp::NumericVector Q_vec = model["Q"];
    Rcpp::NumericVector C_vec = model["C_latent"];
    Rcpp::IntegerVector Q_dim = Q_vec.attr("dim");
    Rcpp::IntegerVector C_dim = C_vec.attr("dim");
    if (Q_dim.size() != 2L || C_dim.size() != 3L ||
        Q_dim[0] != m || Q_dim[1] != kmax ||
        C_dim[0] != kmax || C_dim[1] != kmax ||
        C_dim[2] < static_cast<int>(length_ncomp)) {
      Rcpp::stop("Model latent PLSSVD factors are not compatible with CUDA flash prediction");
    }
    const arma::mat QQ(
      Q_vec.begin(),
      static_cast<arma::uword>(Q_dim[0]),
      static_cast<arma::uword>(Q_dim[1]),
      false,
      true
    );
    const arma::cube CC(
      C_vec.begin(),
      static_cast<arma::uword>(C_dim[0]),
      static_cast<arma::uword>(C_dim[1]),
      static_cast<arma::uword>(C_dim[2]),
      false,
      true
    );
    Wflash.zeros(kmax, m, length_ncomp);
    for (arma::uword a = 0; a < length_ncomp; ++a) {
      int mc = ncomp(a);
      if (mc < 1 || mc > kmax) {
        Rcpp::stop("ncomp exceeds latent rank for CUDA flash prediction");
      }
      arma::mat Cmc = CC.slice(a).submat(0, 0, mc - 1, mc - 1);
      Wflash.slice(a).rows(0, static_cast<arma::uword>(mc - 1)) =
        Cmc * QQ.cols(0, static_cast<arma::uword>(mc - 1)).t();
    }
  } else {
    Rcpp::stop("CUDA flash prediction requires compact low-rank factors");
  }

  arma::cube Ypred = fastpls_svd::cuda_flash_lowrank_predict(
    Xtest,
    RR,
    Wflash,
    mY,
    ncomp
  );

  arma::mat T_Xtest;
  if (proj) {
    T_Xtest = Xtest * RR;
  }

  return List::create(
    Named("Ypred") = Ypred,
    Named("Ttest") = T_Xtest,
    Named("predict_backend") = "cuda_flash"
  );
}

// [[Rcpp::export]]
List pls_predict_flash_cpu(List& model, SEXP XtestSEXP, bool proj, int block_size) {
  const arma::mat Xtest = numeric_matrix_view(XtestSEXP, "Xtest");
  const int m = Rcpp::as<int>(model["m"]);
  arma::ivec ncomp = Rcpp::as<arma::ivec>(model["ncomp"]);
  const arma::uword length_ncomp = static_cast<arma::uword>(ncomp.n_elem);

  Rcpp::NumericVector mX_vec = model["mX"];
  arma::rowvec mX(mX_vec.begin(), mX_vec.size(), false, true);
  Rcpp::NumericVector vX_vec = model["vX"];
  arma::rowvec vX(vX_vec.begin(), vX_vec.size(), false, true);
  Rcpp::NumericVector mY_vec = model["mY"];
  arma::rowvec mY(mY_vec.begin(), mY_vec.size(), false, true);

  std::string pls_method;
  if (model.containsElementNamed("pls_method")) {
    pls_method = Rcpp::as<std::string>(model["pls_method"]);
  }

  Rcpp::NumericVector R_vec = model["R"];
  Rcpp::IntegerVector R_dim = R_vec.attr("dim");
  if (R_dim.size() != 2L || R_dim[0] != Xtest.n_cols || R_dim[1] < 1) {
    Rcpp::stop("Model `R` is not compatible with CPU flash prediction");
  }
  const arma::mat RR(
    R_vec.begin(),
    static_cast<arma::uword>(R_dim[0]),
    static_cast<arma::uword>(R_dim[1]),
    false,
    true
  );
  const int kmax = static_cast<int>(RR.n_cols);
  arma::mat RR_scaled = RR;
  RR_scaled.each_col() /= vX.t();
  const arma::rowvec score_offset = (mX / vX) * RR;

  if (length_ncomp == 0 || m < 1) {
    Rcpp::stop("CPU flash prediction requires responses and component counts");
  }
  for (arma::uword a = 0; a < length_ncomp; ++a) {
    if (ncomp(a) < 1 || ncomp(a) > kmax) {
      Rcpp::stop("ncomp exceeds latent rank for CPU flash prediction");
    }
  }
  // SIMPLS shares Q across prefixes; existing PLS-SVD weights are read-only.
  arma::cube owned_weights;
  Rcpp::NumericVector stored_weights;
  double* weight_data = nullptr;
  arma::uword weight_slices = length_ncomp;
  if ((pls_method == "simpls" || pls_method == "simpls_fast") &&
      model.containsElementNamed("Q")) {
    Rcpp::NumericVector Q_vec = model["Q"];
    Rcpp::IntegerVector Q_dim = Q_vec.attr("dim");
    if (Q_dim.size() != 2L || Q_dim[0] != m || Q_dim[1] < 1) {
      Rcpp::stop("Model `Q` is not compatible with CPU flash prediction");
    }
    const arma::mat QQ(
      Q_vec.begin(),
      static_cast<arma::uword>(Q_dim[0]),
      static_cast<arma::uword>(Q_dim[1]),
      false,
      true
    );
    const arma::uword max_requested = static_cast<arma::uword>(ncomp.max());
    if (max_requested > QQ.n_cols) {
      Rcpp::stop("ncomp exceeds latent rank for CPU flash prediction");
    }
    weight_slices = 1;
    owned_weights.zeros(kmax, static_cast<arma::uword>(m), weight_slices);
    owned_weights.slice(0).rows(0, max_requested - 1) =
      QQ.cols(0, max_requested - 1).t();
    weight_data = owned_weights.memptr();
  } else if (pls_method == "plssvd" && model.containsElementNamed("W_latent")) {
    stored_weights = model["W_latent"];
    Rcpp::IntegerVector W_dim = stored_weights.attr("dim");
    if (W_dim.size() != 3L || W_dim[0] != kmax || W_dim[1] != m ||
        W_dim[2] < static_cast<int>(length_ncomp)) {
      Rcpp::stop("Model `W_latent` is not compatible with CPU flash prediction");
    }
    weight_data = stored_weights.begin();
  } else if (pls_method == "plssvd" &&
             model.containsElementNamed("C_latent") &&
             model.containsElementNamed("Q")) {
    Rcpp::NumericVector Q_vec = model["Q"];
    Rcpp::NumericVector C_vec = model["C_latent"];
    Rcpp::IntegerVector Q_dim = Q_vec.attr("dim");
    Rcpp::IntegerVector C_dim = C_vec.attr("dim");
    if (Q_dim.size() != 2L || C_dim.size() != 3L ||
        Q_dim[0] != m || Q_dim[1] != kmax ||
        C_dim[0] != kmax || C_dim[1] != kmax ||
        C_dim[2] < static_cast<int>(length_ncomp)) {
      Rcpp::stop("Model latent PLSSVD factors are not compatible with CPU flash prediction");
    }
    const arma::mat QQ(
      Q_vec.begin(),
      static_cast<arma::uword>(Q_dim[0]),
      static_cast<arma::uword>(Q_dim[1]),
      false,
      true
    );
    const arma::cube CC(
      C_vec.begin(),
      static_cast<arma::uword>(C_dim[0]),
      static_cast<arma::uword>(C_dim[1]),
      static_cast<arma::uword>(C_dim[2]),
      false,
      true
    );
    owned_weights.zeros(kmax, static_cast<arma::uword>(m), length_ncomp);
    for (arma::uword a = 0; a < length_ncomp; ++a) {
      const int mc = ncomp(a);
      if (mc < 1 || mc > kmax) {
        Rcpp::stop("ncomp exceeds latent rank for CPU flash prediction");
      }
      arma::mat Cmc = CC.slice(a).submat(0, 0, mc - 1, mc - 1);
      owned_weights.slice(a).rows(0, static_cast<arma::uword>(mc - 1)) =
        Cmc * QQ.cols(0, static_cast<arma::uword>(mc - 1)).t();
    }
    weight_data = owned_weights.memptr();
  } else {
    Rcpp::stop("CPU flash prediction requires compact low-rank factors");
  }
  const arma::cube Wflash(weight_data, kmax, static_cast<arma::uword>(m),
                         weight_slices, false, true);

  const arma::uword ntest = Xtest.n_rows;
  arma::cube Ypred(ntest, static_cast<arma::uword>(m), length_ncomp, arma::fill::none);
  arma::mat T_Xtest;
  if (proj) {
    T_Xtest.set_size(ntest, static_cast<arma::uword>(kmax));
  }

  arma::uword bs = static_cast<arma::uword>(block_size > 0 ? block_size : 4096);
  if (bs == 0 || bs > ntest) {
    bs = ntest;
  }

  for (arma::uword start = 0; start < ntest; start += bs) {
    const arma::uword stop = std::min(start + bs - 1, ntest - 1);
    arma::mat scores = Xtest.rows(start, stop) * RR_scaled;
    scores.each_row() -= score_offset;
    if (proj) {
      T_Xtest.rows(start, stop) = scores;
    }
    for (arma::uword a = 0; a < length_ncomp; ++a) {
      const int mc = ncomp(a);
      arma::mat Yblock =
        scores.cols(0, static_cast<arma::uword>(mc - 1)) *
        Wflash.slice(weight_slices == 1 ? 0 : a).rows(
          0, static_cast<arma::uword>(mc - 1));
      Yblock.each_row() += mY;
      Ypred.slice(a).rows(start, stop) = Yblock;
    }
  }

  return List::create(
    Named("Ypred") = Ypred,
    Named("Ttest") = T_Xtest,
    Named("predict_backend") = "cpu_flash"
  );
}

arma::cube compact_prediction_weights(List& model, const int m, const arma::ivec& ncomp) {
  std::string pls_method;
  if (model.containsElementNamed("pls_method")) {
    pls_method = Rcpp::as<std::string>(model["pls_method"]);
  }

  Rcpp::NumericVector R_vec = model["R"];
  Rcpp::IntegerVector R_dim = R_vec.attr("dim");
  if (R_dim.size() != 2L || R_dim[1] < 1) {
    Rcpp::stop("Model `R` is not compatible with compact class prediction");
  }
  const int kmax = R_dim[1];
  const arma::uword length_ncomp = static_cast<arma::uword>(ncomp.n_elem);
  arma::cube Wflash(static_cast<arma::uword>(kmax), static_cast<arma::uword>(m), length_ncomp, arma::fill::zeros);

  if ((pls_method == "simpls" || pls_method == "simpls_fast") &&
      model.containsElementNamed("Q")) {
    Rcpp::NumericVector Q_vec = model["Q"];
    Rcpp::IntegerVector Q_dim = Q_vec.attr("dim");
    if (Q_dim.size() != 2L || Q_dim[0] != m || Q_dim[1] < 1) {
      Rcpp::stop("Model `Q` is not compatible with compact class prediction");
    }
    const arma::mat QQ(
      Q_vec.begin(),
      static_cast<arma::uword>(Q_dim[0]),
      static_cast<arma::uword>(Q_dim[1]),
      false,
      true
    );
    for (arma::uword a = 0; a < length_ncomp; ++a) {
      const int mc = ncomp(a);
      if (mc < 1 || mc > kmax || mc > static_cast<int>(QQ.n_cols)) {
        Rcpp::stop("ncomp exceeds latent rank for compact class prediction");
      }
      Wflash.slice(a).rows(0, static_cast<arma::uword>(mc - 1)) =
        QQ.cols(0, static_cast<arma::uword>(mc - 1)).t();
    }
    return Wflash;
  }

  if (pls_method == "plssvd" && model.containsElementNamed("W_latent")) {
    Rcpp::NumericVector W_vec = model["W_latent"];
    Rcpp::IntegerVector W_dim = W_vec.attr("dim");
    if (W_dim.size() != 3L || W_dim[0] != kmax || W_dim[1] != m ||
        W_dim[2] < static_cast<int>(length_ncomp)) {
      Rcpp::stop("Model `W_latent` is not compatible with compact class prediction");
    }
    const arma::cube WW(
      W_vec.begin(),
      static_cast<arma::uword>(W_dim[0]),
      static_cast<arma::uword>(W_dim[1]),
      static_cast<arma::uword>(W_dim[2]),
      false,
      true
    );
    return WW.slices(0, length_ncomp - 1);
  }

  if (pls_method == "plssvd" &&
      model.containsElementNamed("C_latent") &&
      model.containsElementNamed("Q")) {
    Rcpp::NumericVector Q_vec = model["Q"];
    Rcpp::NumericVector C_vec = model["C_latent"];
    Rcpp::IntegerVector Q_dim = Q_vec.attr("dim");
    Rcpp::IntegerVector C_dim = C_vec.attr("dim");
    if (Q_dim.size() != 2L || C_dim.size() != 3L ||
        Q_dim[0] != m || Q_dim[1] != kmax ||
        C_dim[0] != kmax || C_dim[1] != kmax ||
        C_dim[2] < static_cast<int>(length_ncomp)) {
      Rcpp::stop("Model latent PLSSVD factors are not compatible with compact class prediction");
    }
    const arma::mat QQ(
      Q_vec.begin(),
      static_cast<arma::uword>(Q_dim[0]),
      static_cast<arma::uword>(Q_dim[1]),
      false,
      true
    );
    const arma::cube CC(
      C_vec.begin(),
      static_cast<arma::uword>(C_dim[0]),
      static_cast<arma::uword>(C_dim[1]),
      static_cast<arma::uword>(C_dim[2]),
      false,
      true
    );
    for (arma::uword a = 0; a < length_ncomp; ++a) {
      const int mc = ncomp(a);
      if (mc < 1 || mc > kmax) {
        Rcpp::stop("ncomp exceeds latent rank for compact class prediction");
      }
      arma::mat Cmc = CC.slice(a).submat(0, 0, mc - 1, mc - 1);
      Wflash.slice(a).rows(0, static_cast<arma::uword>(mc - 1)) =
        Cmc * QQ.cols(0, static_cast<arma::uword>(mc - 1)).t();
    }
    return Wflash;
  }

  Rcpp::stop("Compact class prediction requires compact low-rank factors");
}

arma::mat class_prediction_offsets(List& model, const int m, const arma::uword length_ncomp) {
  Rcpp::NumericVector mY_vec = model["mY"];
  arma::rowvec mY(mY_vec.begin(), mY_vec.size(), false, true);
  arma::mat offsets(static_cast<arma::uword>(m), length_ncomp, arma::fill::zeros);
  for (arma::uword a = 0; a < length_ncomp; ++a) {
    offsets.col(a) = mY.t();
  }
  return offsets;
}

void fill_topk_from_yblock(
  const arma::mat& yblock,
  const arma::vec& offset,
  const arma::uword total_n,
  const arma::uword row_offset,
  const arma::uword slice,
  const int top_k,
  Rcpp::IntegerVector& top_index,
  Rcpp::NumericVector& top_score
) {
  const int m = static_cast<int>(yblock.n_cols);
  const int use_top_k = std::max(1, std::min(top_k, m));
  const size_t slice_offset =
    static_cast<size_t>(slice) *
    static_cast<size_t>(total_n) *
    static_cast<size_t>(use_top_k);

  for (arma::uword i = 0; i < yblock.n_rows; ++i) {
    std::vector<double> best_score(static_cast<size_t>(use_top_k), -std::numeric_limits<double>::infinity());
    std::vector<int> best_index(static_cast<size_t>(use_top_k), 0);
    for (int j = 0; j < m; ++j) {
      const double value = yblock(i, static_cast<arma::uword>(j)) + offset(static_cast<arma::uword>(j));
      for (int r = 0; r < use_top_k; ++r) {
        if (value > best_score[static_cast<size_t>(r)]) {
          for (int rr = use_top_k - 1; rr > r; --rr) {
            best_score[static_cast<size_t>(rr)] = best_score[static_cast<size_t>(rr - 1)];
            best_index[static_cast<size_t>(rr)] = best_index[static_cast<size_t>(rr - 1)];
          }
          best_score[static_cast<size_t>(r)] = value;
          best_index[static_cast<size_t>(r)] = j + 1;
          break;
        }
      }
    }
    for (int r = 0; r < use_top_k; ++r) {
      const size_t out_pos =
        slice_offset +
        static_cast<size_t>(row_offset + i) +
        static_cast<size_t>(total_n) * static_cast<size_t>(r);
      top_index[out_pos] = best_index[static_cast<size_t>(r)];
      top_score[out_pos] = best_score[static_cast<size_t>(r)];
    }
  }
}

List class_topk_from_cube(
  const arma::cube& Ypred,
  List& model,
  const int top_k
) {
  const int m = Rcpp::as<int>(model["m"]);
  arma::ivec ncomp = Rcpp::as<arma::ivec>(model["ncomp"]);
  const arma::uword length_ncomp = static_cast<arma::uword>(ncomp.n_elem);
  const arma::uword ntest = Ypred.n_rows;
  const int use_top_k = std::max(1, std::min(top_k, m));
  arma::vec zero_offset(static_cast<arma::uword>(m), arma::fill::zeros);

  Rcpp::IntegerVector top_index(ntest * static_cast<arma::uword>(use_top_k) * length_ncomp);
  top_index.attr("dim") = Rcpp::IntegerVector::create(
    static_cast<int>(ntest),
    use_top_k,
    static_cast<int>(length_ncomp)
  );
  Rcpp::NumericVector top_score(ntest * static_cast<arma::uword>(use_top_k) * length_ncomp);
  top_score.attr("dim") = top_index.attr("dim");

  for (arma::uword a = 0; a < length_ncomp; ++a) {
    fill_topk_from_yblock(Ypred.slice(a), zero_offset, ntest, 0, a, use_top_k, top_index, top_score);
  }

  return List::create(
    Named("top_index") = top_index,
    Named("top_score") = top_score
  );
}

// [[Rcpp::export]]
List pls_class_predict_topk_cpp(List& model, arma::mat Xtest, int top_k, bool proj, int block_size) {
  const int m = Rcpp::as<int>(model["m"]);
  arma::ivec ncomp = Rcpp::as<arma::ivec>(model["ncomp"]);
  const arma::uword length_ncomp = static_cast<arma::uword>(ncomp.n_elem);
  const int use_top_k = std::max(1, std::min(top_k, m));

  Rcpp::NumericVector mX_vec = model["mX"];
  arma::rowvec mX(mX_vec.begin(), mX_vec.size(), false, true);
  Xtest.each_row() -= mX;
  Rcpp::NumericVector vX_vec = model["vX"];
  arma::rowvec vX(vX_vec.begin(), vX_vec.size(), false, true);
  Xtest.each_row() /= vX;

  const arma::uword ntest = Xtest.n_rows;
  arma::mat offsets = class_prediction_offsets(model, m, length_ncomp);

  Rcpp::IntegerVector top_index(ntest * static_cast<arma::uword>(use_top_k) * length_ncomp);
  top_index.attr("dim") = Rcpp::IntegerVector::create(
    static_cast<int>(ntest),
    use_top_k,
    static_cast<int>(length_ncomp)
  );
  Rcpp::NumericVector top_score(ntest * static_cast<arma::uword>(use_top_k) * length_ncomp);
  top_score.attr("dim") = top_index.attr("dim");

  arma::uword bs = static_cast<arma::uword>(std::max(1, block_size));
  if (bs > ntest) bs = ntest;

  bool used_compact = false;
  try {
    Rcpp::NumericVector R_vec = model["R"];
    Rcpp::IntegerVector R_dim = R_vec.attr("dim");
    if (R_dim.size() == 2L && R_dim[0] == Xtest.n_cols && R_dim[1] > 0) {
      const arma::mat RR(
        R_vec.begin(),
        static_cast<arma::uword>(R_dim[0]),
        static_cast<arma::uword>(R_dim[1]),
        false,
        true
      );
      const int kmax = static_cast<int>(RR.n_cols);
      arma::cube Wflash = compact_prediction_weights(model, m, ncomp);
      for (arma::uword start = 0; start < ntest; start += bs) {
        const arma::uword stop = std::min(start + bs - 1, ntest - 1);
        const arma::mat scores = Xtest.rows(start, stop) * RR;
        for (arma::uword a = 0; a < length_ncomp; ++a) {
          const int mc = ncomp(a);
          if (mc < 1 || mc > kmax) {
            Rcpp::stop("ncomp exceeds latent rank for compact class prediction");
          }
          arma::mat yblock =
            scores.cols(0, static_cast<arma::uword>(mc - 1)) *
            Wflash.slice(a).rows(0, static_cast<arma::uword>(mc - 1));
          fill_topk_from_yblock(yblock, offsets.col(a), ntest, start, a, use_top_k, top_index, top_score);
        }
      }
      used_compact = true;
    }
  } catch (...) {
    used_compact = false;
  }

  if (!used_compact) {
    if (!model.containsElementNamed("B")) {
      Rcpp::stop("Compact class prediction requires compact factors or stored B");
    }
    Rcpp::NumericVector B_vec = model["B"];
    Rcpp::IntegerVector B_dim = B_vec.attr("dim");
    if (B_dim.size() != 3L || B_dim[0] != Xtest.n_cols || B_dim[1] != m ||
        B_dim[2] < static_cast<int>(length_ncomp)) {
      Rcpp::stop("Model coefficient array `B` is not compatible with compact class prediction");
    }
    const arma::cube B(
      B_vec.begin(),
      static_cast<arma::uword>(B_dim[0]),
      static_cast<arma::uword>(B_dim[1]),
      static_cast<arma::uword>(B_dim[2]),
      false,
      true
    );
    for (arma::uword start = 0; start < ntest; start += bs) {
      const arma::uword stop = std::min(start + bs - 1, ntest - 1);
      const arma::mat Xblock = Xtest.rows(start, stop);
      for (arma::uword a = 0; a < length_ncomp; ++a) {
        arma::mat yblock = Xblock * B.slice(a);
        fill_topk_from_yblock(yblock, offsets.col(a), ntest, start, a, use_top_k, top_index, top_score);
      }
    }
  }

  arma::mat T_Xtest;
  if (proj) {
    Rcpp::NumericVector RR_vec = model["R"];
    Rcpp::IntegerVector RR_dim = RR_vec.attr("dim");
    if (RR_dim.size() == 2L && RR_dim[0] > 0 && RR_dim[1] > 0) {
      const arma::mat RR(
        RR_vec.begin(),
        static_cast<arma::uword>(RR_dim[0]),
        static_cast<arma::uword>(RR_dim[1]),
        false,
        true
      );
      T_Xtest = Xtest * RR;
    } else {
      T_Xtest.set_size(ntest, 0);
    }
  }

  return List::create(
    Named("top_index") = top_index,
    Named("top_score") = top_score,
    Named("Ttest") = T_Xtest,
    Named("predict_backend") = "cpp_topk"
  );
}

// [[Rcpp::export]]
List pls_class_predict_topk_cuda(List& model, arma::mat Xtest, int top_k, bool proj) {
  if (!fastpls_svd::has_cuda_backend()) {
    Rcpp::stop(
      "CUDA top-ranked prediction requires an available CUDA backend. "
      "No CPU fallback is performed."
    );
  }

  try {
    const int m = Rcpp::as<int>(model["m"]);
    arma::ivec ncomp = Rcpp::as<arma::ivec>(model["ncomp"]);
    Rcpp::NumericVector mX_vec = model["mX"];
    arma::rowvec mX(mX_vec.begin(), mX_vec.size(), false, true);
    Xtest.each_row() -= mX;
    Rcpp::NumericVector vX_vec = model["vX"];
    arma::rowvec vX(vX_vec.begin(), vX_vec.size(), false, true);
    Xtest.each_row() /= vX;

    Rcpp::NumericVector mY_vec = model["mY"];
    arma::rowvec mY(mY_vec.begin(), mY_vec.size(), false, true);
    Rcpp::NumericVector R_vec = model["R"];
    Rcpp::IntegerVector R_dim = R_vec.attr("dim");
    if (R_dim.size() != 2L || R_dim[0] != Xtest.n_cols || R_dim[1] < 1) {
      Rcpp::stop("Model `R` is not compatible with CUDA top-k prediction");
    }
    const arma::mat RR(
      R_vec.begin(),
      static_cast<arma::uword>(R_dim[0]),
      static_cast<arma::uword>(R_dim[1]),
      false,
      true
    );
    arma::cube Wflash = compact_prediction_weights(model, m, ncomp);
    arma::cube Ypred = fastpls_svd::cuda_flash_lowrank_predict(Xtest, RR, Wflash, mY, ncomp);
    Rcpp::List out = class_topk_from_cube(Ypred, model, top_k);
    arma::mat T_Xtest;
    if (proj) {
      T_Xtest = Xtest * RR;
    }
    out["Ttest"] = T_Xtest;
    out["predict_backend"] = "cuda_topk";
    return out;
  } catch (const std::exception& error) {
    Rcpp::stop(
      "CUDA top-ranked prediction failed: %s. No CPU fallback is performed.",
      error.what()
    );
  } catch (...) {
    Rcpp::stop(
      "CUDA top-ranked prediction failed with an unknown error. "
      "No CPU fallback is performed."
    );
  }
}

arma::imat pls_predict_classes_compact_cpu(List& model, arma::mat Xtest) {
  const int m = Rcpp::as<int>(model["m"]);
  arma::ivec ncomp = Rcpp::as<arma::ivec>(model["ncomp"]);
  const arma::uword length_ncomp = static_cast<arma::uword>(ncomp.n_elem);

  Rcpp::NumericVector mX_vec = model["mX"];
  arma::rowvec mX(mX_vec.begin(), mX_vec.size(), false, true);
  Xtest.each_row() -= mX;
  Rcpp::NumericVector vX_vec = model["vX"];
  arma::rowvec vX(vX_vec.begin(), vX_vec.size(), false, true);
  Xtest.each_row() /= vX;
  Rcpp::NumericVector mY_vec = model["mY"];
  arma::rowvec mY(mY_vec.begin(), mY_vec.size(), false, true);

  Rcpp::NumericVector R_vec = model["R"];
  Rcpp::IntegerVector R_dim = R_vec.attr("dim");
  if (R_dim.size() != 2L || R_dim[0] != Xtest.n_cols || R_dim[1] < 1) {
    Rcpp::stop("Model `R` is not compatible with compact class prediction");
  }
  const arma::mat RR(
    R_vec.begin(),
    static_cast<arma::uword>(R_dim[0]),
    static_cast<arma::uword>(R_dim[1]),
    false,
    true
  );
  const int kmax = static_cast<int>(RR.n_cols);
  arma::cube Wflash = compact_prediction_weights(model, m, ncomp);

  const arma::uword ntest = Xtest.n_rows;
  arma::imat class_pred(ntest, length_ncomp, arma::fill::zeros);
  arma::uword bs = static_cast<arma::uword>(env_int_or("FASTPLS_COMPACT_CLASS_BLOCK_SIZE", 4096, 128, 1048576));
  if (bs == 0 || bs > ntest) bs = ntest;

  for (arma::uword start = 0; start < ntest; start += bs) {
    const arma::uword stop = std::min(start + bs - 1, ntest - 1);
    const arma::mat Xblock = Xtest.rows(start, stop);
    const arma::mat scores = Xblock * RR;
    for (arma::uword a = 0; a < length_ncomp; ++a) {
      const int mc = ncomp(a);
      if (mc < 1 || mc > kmax) {
        Rcpp::stop("ncomp exceeds latent rank for compact class prediction");
      }
      arma::mat yblock =
        scores.cols(0, static_cast<arma::uword>(mc - 1)) *
        Wflash.slice(a).rows(0, static_cast<arma::uword>(mc - 1));
      yblock.each_row() += mY;
      for (arma::uword i = 0; i < yblock.n_rows; ++i) {
        class_pred(start + i, a) = static_cast<int>(yblock.row(i).index_max()) + 1;
      }
    }
  }
  return class_pred;
}

arma::imat pls_predict_classes_compact_cuda(List& model, arma::mat Xtest) {
  if (!fastpls_svd::has_cuda_backend()) {
    Rcpp::stop(
      "CUDA compact class prediction requires an available CUDA backend. "
      "No CPU fallback is performed."
    );
  }

  const int m = Rcpp::as<int>(model["m"]);
  arma::ivec ncomp = Rcpp::as<arma::ivec>(model["ncomp"]);

  Rcpp::NumericVector mX_vec = model["mX"];
  arma::rowvec mX(mX_vec.begin(), mX_vec.size(), false, true);
  Xtest.each_row() -= mX;
  Rcpp::NumericVector vX_vec = model["vX"];
  arma::rowvec vX(vX_vec.begin(), vX_vec.size(), false, true);
  Xtest.each_row() /= vX;
  Rcpp::NumericVector mY_vec = model["mY"];
  arma::rowvec mY(mY_vec.begin(), mY_vec.size(), false, true);

  Rcpp::NumericVector R_vec = model["R"];
  Rcpp::IntegerVector R_dim = R_vec.attr("dim");
  if (R_dim.size() != 2L || R_dim[0] != Xtest.n_cols || R_dim[1] < 1) {
    Rcpp::stop("Model `R` is not compatible with CUDA compact class prediction");
  }
  const arma::mat RR(
    R_vec.begin(),
    static_cast<arma::uword>(R_dim[0]),
    static_cast<arma::uword>(R_dim[1]),
    false,
    true
  );
  arma::cube Wflash = compact_prediction_weights(model, m, ncomp);
  return fastpls_svd::cuda_flash_lowrank_predict_classes(Xtest, RR, Wflash, mY, ncomp);
}

arma::ivec nearest_code_classes(const arma::mat& Z, const arma::mat& class_codes) {
  if (class_codes.n_rows < 1 || class_codes.n_cols != Z.n_cols) {
    Rcpp::stop("Class codebook is not compatible with predicted code dimensions");
  }
  arma::mat score = 2.0 * (Z * class_codes.t());
  arma::rowvec code_norm = arma::sum(class_codes % class_codes, 1).t();
  score.each_row() -= code_norm;
  arma::ivec out(Z.n_rows);
  for (arma::uword i = 0; i < Z.n_rows; ++i) {
    out(i) = static_cast<int>(score.row(i).index_max()) + 1;
  }
  return out;
}

arma::imat pls_predict_code_classes_compact_cpu(List& model, arma::mat Xtest, const arma::mat& class_codes) {
  const int m = Rcpp::as<int>(model["m"]);
  if (m != static_cast<int>(class_codes.n_cols)) {
    Rcpp::stop("Model response dimension does not match class codebook");
  }
  arma::ivec ncomp = Rcpp::as<arma::ivec>(model["ncomp"]);
  const arma::uword length_ncomp = static_cast<arma::uword>(ncomp.n_elem);

  Rcpp::NumericVector mX_vec = model["mX"];
  arma::rowvec mX(mX_vec.begin(), mX_vec.size(), false, true);
  Xtest.each_row() -= mX;
  Rcpp::NumericVector vX_vec = model["vX"];
  arma::rowvec vX(vX_vec.begin(), vX_vec.size(), false, true);
  Xtest.each_row() /= vX;
  Rcpp::NumericVector mY_vec = model["mY"];
  arma::rowvec mY(mY_vec.begin(), mY_vec.size(), false, true);

  Rcpp::NumericVector R_vec = model["R"];
  Rcpp::IntegerVector R_dim = R_vec.attr("dim");
  if (R_dim.size() != 2L || R_dim[0] != Xtest.n_cols || R_dim[1] < 1) {
    Rcpp::stop("Model `R` is not compatible with compact code prediction");
  }
  const arma::mat RR(
    R_vec.begin(),
    static_cast<arma::uword>(R_dim[0]),
    static_cast<arma::uword>(R_dim[1]),
    false,
    true
  );
  const int kmax = static_cast<int>(RR.n_cols);
  arma::cube Wflash = compact_prediction_weights(model, m, ncomp);

  const arma::uword ntest = Xtest.n_rows;
  arma::imat class_pred(ntest, length_ncomp, arma::fill::zeros);
  arma::uword bs = static_cast<arma::uword>(env_int_or("FASTPLS_COMPACT_CLASS_BLOCK_SIZE", 4096, 128, 1048576));
  if (bs == 0 || bs > ntest) bs = ntest;

  for (arma::uword start = 0; start < ntest; start += bs) {
    const arma::uword stop = std::min(start + bs - 1, ntest - 1);
    const arma::mat Xblock = Xtest.rows(start, stop);
    const arma::mat scores = Xblock * RR;
    for (arma::uword a = 0; a < length_ncomp; ++a) {
      const int mc = ncomp(a);
      if (mc < 1 || mc > kmax) {
        Rcpp::stop("ncomp exceeds latent rank for compact code prediction");
      }
      arma::mat zblock =
        scores.cols(0, static_cast<arma::uword>(mc - 1)) *
        Wflash.slice(a).rows(0, static_cast<arma::uword>(mc - 1));
      zblock.each_row() += mY;
      arma::ivec fold_class = nearest_code_classes(zblock, class_codes);
      class_pred.submat(start, a, stop, a) = fold_class;
    }
  }
  return class_pred;
}

arma::imat pls_predict_code_classes_compact_cuda(List& model, arma::mat Xtest, const arma::mat& class_codes) {
  if (!fastpls_svd::has_cuda_backend()) {
    Rcpp::stop(
      "CUDA compact code prediction requires an available CUDA backend. "
      "No CPU fallback is performed."
    );
  }
  const int m = Rcpp::as<int>(model["m"]);
  if (m != static_cast<int>(class_codes.n_cols)) {
    Rcpp::stop("Model response dimension does not match class codebook");
  }
  Rcpp::NumericVector mX_vec = model["mX"];
  arma::rowvec mX(mX_vec.begin(), mX_vec.size(), false, true);
  Xtest.each_row() -= mX;
  Rcpp::NumericVector vX_vec = model["vX"];
  arma::rowvec vX(vX_vec.begin(), vX_vec.size(), false, true);
  Xtest.each_row() /= vX;
  Rcpp::NumericVector mY_vec = model["mY"];
  arma::rowvec mY(mY_vec.begin(), mY_vec.size(), false, true);

  Rcpp::NumericVector R_vec = model["R"];
  Rcpp::IntegerVector R_dim = R_vec.attr("dim");
  if (R_dim.size() != 2L || R_dim[0] != Xtest.n_cols || R_dim[1] < 1) {
    Rcpp::stop("Model `R` is not compatible with CUDA compact code prediction");
  }
  const arma::mat RR(
    R_vec.begin(),
    static_cast<arma::uword>(R_dim[0]),
    static_cast<arma::uword>(R_dim[1]),
    false,
    true
  );
  arma::ivec ncomp = Rcpp::as<arma::ivec>(model["ncomp"]);
  arma::cube Wflash = compact_prediction_weights(model, m, ncomp);
  arma::cube Zpred = fastpls_svd::cuda_flash_lowrank_predict(Xtest, RR, Wflash, mY, ncomp);
  arma::imat class_pred(Zpred.n_rows, Zpred.n_slices, arma::fill::zeros);
  for (arma::uword s = 0; s < Zpred.n_slices; ++s) {
    class_pred.col(s) = nearest_code_classes(Zpred.slice(s), class_codes);
  }
  return class_pred;
}

// This function performs a random selection of the elements of a vector "yy".
// The number of elements to select is defined by the variable "size".

IntegerVector samplewithoutreplace(IntegerVector yy,int size){
  if (size < 0 || size > yy.size()) {
    stop("Sample size must be between zero and the population size");
  }
  IntegerVector xx(size);
  const int population = yy.size();
  std::vector<int> remaining(static_cast<std::size_t>(population) + 1);
  for (int i = 1; i <= population; ++i) remaining[i] = i & -i;
  int top_bit = 1;
  while (top_bit <= population / 2) top_bit *= 2;
  // Select the same ordered remaining element with the same RNG draw as
  // vector erasure, using an order-statistic tree instead of shifting rows.
  for (int i = 0; i < size; ++i) {
    int rank = static_cast<int>(unif_rand() * (population - i));
    int index = 0;
    for (int bit = top_bit; bit > 0; bit /= 2) {
      const int next = index + bit;
      if (next <= population && remaining[next] <= rank) {
        rank -= remaining[next];
        index = next;
      }
    }
    xx[i] = yy[index];
    for (int j = index + 1; j <= population; j += j & -j) --remaining[j];
  }
  return xx;
}



List pls_model1_impl(
  const arma::mat& Xtrain,
  const arma::mat& Ytrain,
  arma::ivec ncomp,
  int scaling,
  bool fit,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed,
  const arma::mat* precomputed_crosscov = nullptr,
  const arma::mat* precomputed_crossprod = nullptr,
  const arma::mat* precomputed_x_mean = nullptr,
  const arma::mat* precomputed_x_scale = nullptr,
  const arma::mat* precomputed_y_mean = nullptr,
  int precomputed_training_rows = 0
) {
  const int p = Xtrain.n_cols, m = Ytrain.n_cols;
  fastpls::native::PlssvdOptions controls;
  controls.scaling = scaling;
  controls.fitted = fit;
  controls.store_coefficients = should_store_coefficients(p, m, ncomp.n_elem, true);
  controls.cache_score_gram = env_int_or("FASTPLS_PLSSVD_OPTIMIZED", 1, 0, 1);
  auto solver = [&](const arma::mat& S, int retained, int rank_bound) {
    auto result = compute_truncated_svd_dispatch(S, retained, svd_method,
      rsvd_oversample, rsvd_power, svds_tol, static_cast<unsigned int>(seed),
      false, plssvd_use_small_exact_svd(rank_bound, svd_method));
    return fastpls::native::SingularTriplets<double>{
      std::move(result.U), std::move(result.s), std::move(result.Vt)
    };
  };
  auto model = fastpls::native::fit_plssvd_with_solver(
    Xtrain, Ytrain, std::move(ncomp), controls, solver,
    static_cast<arma::mat*>(nullptr),
    static_cast<const arma::uvec*>(nullptr), 0, precomputed_crosscov,
    precomputed_crossprod, precomputed_x_mean, precomputed_x_scale,
    precomputed_y_mean, precomputed_training_rows
  );
  List out = List::create(
    Named("C_latent") = model.latent_coefficients,
    Named("W_latent") = model.prediction_weights,
    Named("Q") = model.Q, Named("Ttrain") = model.scores,
    Named("R") = model.R, Named("mX") = model.x_mean,
    Named("vX") = model.x_scale, Named("mY") = model.y_mean,
    Named("p") = p, Named("m") = m, Named("ncomp") = model.components,
    Named("Yfit") = model.fitted, Named("R2Y") = model.r2
  );
  if (controls.store_coefficients) out["B"] = model.coefficients;
  annotate_coefficient_storage(out, controls.store_coefficients);
  return out;
}

// [[Rcpp::export]]
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
) {
  return pls_model1_impl(
    std::move(Xtrain), std::move(Ytrain), std::move(ncomp), scaling, fit,
    svd_method, rsvd_oversample, rsvd_power, svds_tol, seed
  );
}

List pls_model1_metal_cv(
  arma::mat Xtrain,
  arma::mat Ytrain,
  arma::ivec ncomp,
  int scaling,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed
) {
  if (!fastpls_svd::has_metal_backend()) {
    stop(
      "Metal CV requires a macOS build with Apple Metal support. "
      "No CPU fallback is performed."
    );
  }

  const int n = Xtrain.n_rows;
  const int p = Xtrain.n_cols;
  const int m = Ytrain.n_cols;
  if (ncomp.n_elem < 1) stop("ncomp must contain at least one value");

  const int max_plssvd_rank = std::min(n, std::min(p, m));
  for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
    if (ncomp(i) > max_plssvd_rank) ncomp(i) = max_plssvd_rank;
    if (ncomp(i) < 1) ncomp(i) = 1;
  }
  const int max_ncomp = max(ncomp);
  int max_ncomp_eff = std::min(max_ncomp, max_plssvd_rank);
  if (max_ncomp_eff < 1) stop("plssvd Metal CV effective rank is < 1");

  arma::mat mX(1, p, arma::fill::zeros);
  if (scaling < 3) {
    mX = mean(Xtrain, 0);
    Xtrain.each_row() -= mX;
  }
  arma::mat vX(1, p, arma::fill::ones);
  if (scaling == 2) {
    vX = variance(Xtrain);
    Xtrain.each_row() /= vX;
  }

  arma::mat mY = mean(Ytrain, 0);
  Ytrain.each_row() -= mY;

  arma::mat S = fastpls_svd::metal_crossprod(Xtrain, Ytrain);
  fastpls_svd::SVDResult svd_res = compute_truncated_svd_dispatch(
    S,
    max_ncomp_eff,
    fastpls_svd::SVD_METHOD_CPU_RSVD,
    rsvd_oversample,
    rsvd_power,
    svds_tol,
    static_cast<unsigned int>(seed),
    false,
    plssvd_use_small_exact_svd(max_plssvd_rank, fastpls_svd::SVD_METHOD_CPU_RSVD)
  );

  arma::mat R = svd_res.U;
  arma::vec s = svd_res.s;
  arma::mat Q = svd_res.Vt.t();
  max_ncomp_eff = std::min(max_ncomp_eff, static_cast<int>(R.n_cols));
  if (Q.n_cols > 0) {
    max_ncomp_eff = std::min(max_ncomp_eff, static_cast<int>(Q.n_cols));
  }
  if (max_ncomp_eff < 1) stop("plssvd Metal CV effective rank is < 1 after SVD");
  R = R.cols(0, max_ncomp_eff - 1);
  Q = Q.cols(0, max_ncomp_eff - 1);

  arma::mat T = fastpls_svd::metal_matrix_multiply(Xtrain, R);
  arma::mat G = fastpls_svd::metal_crossprod(T, T);
  const int length_ncomp = ncomp.n_elem;
  arma::cube B(p, m, length_ncomp, arma::fill::zeros);
  arma::cube C_latent(max_ncomp_eff, max_ncomp_eff, length_ncomp, arma::fill::zeros);
  arma::cube W_latent(max_ncomp_eff, m, length_ncomp, arma::fill::zeros);
  arma::vec R2Y(length_ncomp, arma::fill::zeros);

  for (int a = 0; a < length_ncomp; ++a) {
    const int mc = std::min(static_cast<int>(ncomp(a)), max_ncomp_eff);
    arma::mat G_a = G.submat(0, 0, mc - 1, mc - 1);
    arma::mat D_a(mc, mc, arma::fill::zeros);
    D_a.diag() = s.subvec(0, mc - 1);
    arma::mat coeff_latent;
    bool solved = arma::solve(coeff_latent, G_a, D_a, arma::solve_opts::likely_sympd);
    if (!solved) solved = arma::solve(coeff_latent, G_a, D_a);
    if (!solved) stop("plssvd Metal CV latent solve failed");
    C_latent.slice(a).submat(0, 0, mc - 1, mc - 1) = coeff_latent;
    arma::mat W_a = coeff_latent * Q.cols(0, mc - 1).t();
    W_latent.slice(a).submat(0, 0, mc - 1, m - 1) = W_a;
    B.slice(a) = fastpls_svd::metal_matrix_multiply(R.cols(0, mc - 1), W_a);
  }

  List out = List::create(
    Named("C_latent") = C_latent,
    Named("W_latent") = W_latent,
    Named("Q") = Q,
    Named("Ttrain") = arma::mat(),
    Named("R") = R,
    Named("mX") = mX,
    Named("vX") = vX,
    Named("mY") = mY,
    Named("p") = p,
    Named("m") = m,
    Named("ncomp") = ncomp,
    Named("B") = B,
    Named("Yfit") = arma::cube(),
    Named("R2Y") = R2Y,
    Named("backend") = "metal",
    Named("svd.method") = "metal_rsvd",
    Named("pls_method") = "plssvd",
    Named("predict_latent_ok") = true
  );
  annotate_coefficient_storage(out, true);
  return out;
}

List pls_model2_fast_metal_cv(
  arma::mat Xtrain,
  arma::mat Ytrain,
  arma::ivec ncomp,
  int scaling,
  int rsvd_power,
  int seed
) {
  if (!fastpls_svd::has_metal_backend()) {
    stop(
      "Metal CV requires a macOS build with Apple Metal support. "
      "No CPU fallback is performed."
    );
  }

  const int n = Xtrain.n_rows;
  const int p = Xtrain.n_cols;
  const int m = Ytrain.n_cols;
  if (ncomp.n_elem < 1) stop("ncomp must contain at least one value");
  for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
    if (ncomp(i) < 1) ncomp(i) = 1;
  }

  arma::mat mX(1, p, arma::fill::zeros);
  if (scaling < 3) {
    mX = mean(Xtrain, 0);
    Xtrain.each_row() -= mX;
  }
  arma::mat vX(1, p, arma::fill::ones);
  if (scaling == 2) {
    vX = variance(Xtrain);
    Xtrain.each_row() /= vX;
  }

  arma::mat mY = mean(Ytrain, 0);
  Ytrain.each_row() -= mY;

  const int max_ncomp_req = std::max(1, static_cast<int>(max(ncomp)));
  const int max_ncomp_eff = std::max(1, std::min(max_ncomp_req, std::min(p, n - 1)));
  List native = fastpls_svd::metal_simpls_resident(
    Xtrain,
    Ytrain,
    max_ncomp_eff,
    std::max(1, rsvd_power),
    seed
  );

  arma::mat R = Rcpp::as<arma::mat>(native["R"]);
  arma::mat Q = Rcpp::as<arma::mat>(native["Q"]);
  if (R.n_cols == 0 || Q.n_cols == 0) {
    stop("Metal SIMPLS CV returned no latent components");
  }
  const int available = std::max(
    1,
    std::min(
      max_ncomp_eff,
      std::min(static_cast<int>(R.n_cols), static_cast<int>(Q.n_cols))
    )
  );
  const int length_ncomp = ncomp.n_elem;
  for (int a = 0; a < length_ncomp; ++a) {
    const int mc = std::max(1, std::min(static_cast<int>(ncomp(a)), available));
    ncomp(a) = mc;
  }
  if (R.n_cols > static_cast<arma::uword>(available)) R = R.cols(0, available - 1);
  if (Q.n_cols > static_cast<arma::uword>(available)) Q = Q.cols(0, available - 1);

  List out = List::create(
    Named("P") = arma::mat(),
    Named("Q") = Q,
    Named("Ttrain") = arma::mat(),
    Named("R") = R,
    Named("mX") = mX,
    Named("vX") = vX,
    Named("mY") = mY,
    Named("p") = p,
    Named("m") = m,
    Named("ncomp") = ncomp,
    Named("Yfit") = arma::cube(),
    Named("R2Y") = arma::vec(length_ncomp, arma::fill::zeros),
    Named("backend") = "metal",
    Named("svd.method") = "metal_resident_simpls",
    Named("pls_method") = "simpls_fast",
    Named("predict_latent_ok") = true
  );
  annotate_coefficient_storage(out, false);
  return out;
}

// [[Rcpp::export]]
List pls_model1_rsvd_xprod_precision(
  arma::mat Xtrain,
  arma::mat Ytrain,
  arma::ivec ncomp,
  int scaling,
  bool fit,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed,
  int xprod_precision
) {
  if (xprod_precision == 5) {
    Rcpp::stop("IRLBA is not part of fastPLS");
  }

  const int n = Xtrain.n_rows;
  const int p = Xtrain.n_cols;
  const int m = Ytrain.n_cols;
  const int max_plssvd_rank = std::min(n, std::min(p, m));
  const int length_ncomp = ncomp.n_elem;

  for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
    if (ncomp(i) > max_plssvd_rank) ncomp(i) = max_plssvd_rank;
    if (ncomp(i) < 1) ncomp(i) = 1;
  }

  const int max_ncomp = max(ncomp);
  int max_ncomp_eff = std::min(max_ncomp, max_plssvd_rank);
  if (max_ncomp_eff < 1) {
    stop("plssvd effective rank is < 1");
  }

  arma::mat mX(1, p, fill::zeros);
  if (scaling < 3) {
    mX = mean(Xtrain, 0);
    Xtrain.each_row() -= mX;
  }

  arma::mat vX(1, p, fill::ones);
  if (scaling == 2) {
    vX = variance(Xtrain);
    Xtrain.each_row() /= vX;
  }

  arma::mat mY = mean(Ytrain, 0);
  Ytrain.each_row() -= mY;

  if (xprod_precision == 1 || xprod_precision == 2 || xprod_precision == 4) {
    Rcpp::stop("xprod_precision values 1, 2, and 4 have been removed from fastPLS.");
  }

  fastpls_svd::SVDResult svd_res;
  if (xprod_precision == 3) {
    // Matrix-free 64-bit RSVD for A = X'Y: avoid materializing the huge
    // p-by-q crossproduct while preserving double-precision arithmetic.
    svd_res = truncated_rsvd_crossprod_double(
      Xtrain,
      Ytrain,
      max_ncomp_eff,
      rsvd_oversample,
      rsvd_power,
      static_cast<unsigned int>(seed),
      false,
      plssvd_use_small_exact_svd(max_plssvd_rank, fastpls_svd::SVD_METHOD_CPU_RSVD)
    );
  } else {
    arma::mat S = Xtrain.t() * Ytrain;
    svd_res = compute_truncated_svd_dispatch(
      S,
      max_ncomp_eff,
      fastpls_svd::SVD_METHOD_CPU_RSVD,
      rsvd_oversample,
      rsvd_power,
      svds_tol,
      static_cast<unsigned int>(seed),
      false,
      plssvd_use_small_exact_svd(max_plssvd_rank, fastpls_svd::SVD_METHOD_CPU_RSVD)
    );
  }

  arma::mat svd_u = svd_res.U;
  arma::vec svd_s = svd_res.s;
  arma::mat svd_v = svd_res.Vt.t();

  const bool store_B = should_store_coefficients(p, m, length_ncomp, true);
  arma::cube B;
  if (store_B) {
    B.zeros(p, m, length_ncomp);
  }
  arma::cube Yfit;
  if (fit) {
    Yfit.set_size(n, m, length_ncomp);
  }

  max_ncomp_eff = std::min(max_ncomp_eff, static_cast<int>(svd_u.n_cols));
  if (svd_v.n_cols > 0) {
    max_ncomp_eff = std::min(max_ncomp_eff, static_cast<int>(svd_v.n_cols));
  }
  if (max_ncomp_eff < 1) {
    stop("plssvd effective rank is < 1 after SVD");
  }

  svd_u = svd_u.cols(0, max_ncomp_eff - 1);
  if (svd_v.n_cols > static_cast<arma::uword>(max_ncomp_eff)) {
    svd_v = svd_v.cols(0, max_ncomp_eff - 1);
  }

  arma::mat T_eff = Xtrain * svd_u;
  arma::mat G_full = T_eff.t() * T_eff;
  arma::cube C_latent(max_ncomp_eff, max_ncomp_eff, length_ncomp, arma::fill::zeros);
  arma::cube W_latent(max_ncomp_eff, m, length_ncomp, arma::fill::zeros);
  arma::vec R2Y(length_ncomp, fill::zeros);

  for (int a = 0; a < length_ncomp; ++a) {
    const int mc_eff = std::min(static_cast<int>(ncomp(a)), max_ncomp_eff);
    arma::mat svd_u_mc = svd_u.cols(0, mc_eff - 1);
    arma::mat svd_v_mc = svd_v.cols(0, mc_eff - 1);
    arma::mat T_a = T_eff.cols(0, mc_eff - 1);
    arma::mat G_a = G_full.submat(0, 0, mc_eff - 1, mc_eff - 1);
    arma::mat D_a(mc_eff, mc_eff, fill::zeros);
    D_a.diag() = svd_s.subvec(0, mc_eff - 1);

    arma::mat coeff_latent;
    bool solved = arma::solve(coeff_latent, G_a, D_a, arma::solve_opts::likely_sympd);
    if (!solved) solved = arma::solve(coeff_latent, G_a, D_a);
    if (!solved) stop("plssvd latent solve failed");

    C_latent.slice(a).submat(0, 0, mc_eff - 1, mc_eff - 1) = coeff_latent;
    arma::mat W_a = coeff_latent * svd_v_mc.t();
    W_latent.slice(a).submat(0, 0, mc_eff - 1, m - 1) = W_a;
    if (store_B) {
      B.slice(a) = svd_u_mc * W_a;
    }
    if (fit) {
      arma::mat temp1 = T_a * W_a;
      R2Y(a) = RQ(Ytrain, temp1);
      temp1.each_row() += mY;
      Yfit.slice(a) = temp1;
    }
  }

  List out = List::create(
    Named("C_latent") = C_latent,
    Named("W_latent") = W_latent,
    Named("Q")       = svd_v,
    Named("Ttrain")  = T_eff,
    Named("R")       = svd_u,
    Named("mX")      = mX,
    Named("vX")      = vX,
    Named("mY")      = mY,
    Named("p")       = p,
    Named("m")       = m,
    Named("ncomp")   = ncomp,
    Named("Yfit")    = Yfit,
    Named("R2Y")     = R2Y,
    Named("xprod_precision") = xprod_precision,
    Named("xprod_mode") = (xprod_precision == 3 ? "implicit" : "materialized")
  );
  if (store_B) {
    out["B"] = B;
  }
  annotate_coefficient_storage(out, store_B);
  return out;
}

// [[Rcpp::export]]
List pls_model2_fast_rsvd_xprod_precision(
  arma::mat Xtrain,
  arma::mat Ytrain,
  arma::ivec ncomp,
  int scaling,
  bool fit,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed,
  int xprod_precision
) {
  if (xprod_precision == 5) {
    Rcpp::stop("IRLBA is not part of fastPLS");
  }
  const int n = Xtrain.n_rows;
  const int p = Xtrain.n_cols;
  const int m = Ytrain.n_cols;

  if (ncomp.n_elem < 1) {
    stop("ncomp must contain at least one value");
  }
  for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
    if (ncomp(i) < 1) ncomp(i) = 1;
  }

  const int max_ncomp = max(ncomp);
  const int length_ncomp = ncomp.n_elem;

  arma::mat mX(1, p, fill::zeros);
  if (scaling < 3) {
    mX = mean(Xtrain, 0);
    Xtrain.each_row() -= mX;
  }

  arma::mat vX(1, p, fill::ones);
  if (scaling == 2) {
    vX = variance(Xtrain);
    Xtrain.each_row() /= vX;
  }

  arma::mat mY = mean(Ytrain, 0);
  Ytrain.each_row() -= mY;

  if (xprod_precision == 1 || xprod_precision == 2 || xprod_precision == 4) {
    Rcpp::stop("xprod_precision values 1, 2, and 4 have been removed from fastPLS.");
  }

  const bool use_implicit_double_xprod = (xprod_precision == 3);
  const bool use_implicit_xprod = use_implicit_double_xprod;

  arma::mat Xt;
  arma::mat Yt;
  if (!use_implicit_xprod) {
    Xt = Xtrain.t();
    Yt = Ytrain.t();
  }
  arma::mat S;
  if (!use_implicit_xprod) {
    S = Xt * Ytrain;
  }

  arma::mat XtX_cache;
  arma::mat Sxy_cache;
  arma::mat RR(p, max_ncomp, fill::zeros);
  arma::mat QQ(m, max_ncomp, fill::zeros);
  arma::mat VV(p, max_ncomp, fill::zeros);
  const bool store_B = should_store_coefficients(p, m, length_ncomp, true);
  arma::cube B;
  if (store_B) {
    B.zeros(p, m, length_ncomp);
  }

  arma::cube Yfit;
  arma::vec R2Y(length_ncomp, fill::zeros);
  arma::mat Yfit_cur;
  if (fit) {
    Yfit.set_size(n, m, length_ncomp);
    Yfit_cur.zeros(n, m);
  }

  arma::mat Bcur;
  if (store_B) {
    Bcur.zeros(p, m);
  }
  int i_out = 0;

  const int center_t = env_int_or("FASTPLS_FAST_CENTER_T", 0, 0, 1);
  const int reorth_v = env_int_or("FASTPLS_FAST_REORTH_V", 0, 0, 1);
  const int defl_cache = env_int_or("FASTPLS_FAST_DEFLCACHE", 1, 0, 1);
  const int fast_optimized = env_int_or("FASTPLS_FAST_OPTIMIZED", 1, 0, 1);
  const int incremental_coefficients = env_int_or("FASTPLS_INCREMENTAL_COEFFICIENTS", 1, 0, 1);
  const int fast_crossprod_min_ncomp = env_int_or("FASTPLS_FAST_CROSSPROD_MIN_NCOMP", 20, 1, 1024);
  const int fast_crossprod_max_p = env_int_or("FASTPLS_FAST_CROSSPROD_MAX_P", 512, 16, 65536);
  const int fast_crossprod_min_n_to_p_ratio = env_int_or("FASTPLS_FAST_CROSSPROD_MIN_N_TO_P_RATIO", 8, 1, 1024);
  const bool return_ttrain = env_int_or("FASTPLS_RETURN_TTRAIN", 0, 0, 1) == 1;
  const bool use_crossprod_cache =
    (!use_implicit_xprod) &&
    (fast_optimized == 1) &&
    (center_t == 0) &&
    (max_ncomp >= fast_crossprod_min_ncomp) &&
    (p <= n) &&
    (n >= p * fast_crossprod_min_n_to_p_ratio) &&
    (p <= fast_crossprod_max_p);

  if (use_crossprod_cache) {
    XtX_cache = Xt * Xtrain;
    Sxy_cache = S;
  }

  arma::mat TT;
  if (return_ttrain) {
    TT.zeros(n, max_ncomp);
  }
  auto append_component = [&](arma::vec rr, const int a_idx) -> bool {
    arma::vec pp;
    arma::vec qq;
    arma::vec tt;

    if (use_crossprod_cache) {
      pp = XtX_cache * rr;
      const double tnorm_sq = arma::dot(rr, pp);
      if (!std::isfinite(tnorm_sq) || tnorm_sq <= 0.0) return false;
      const double tnorm = std::sqrt(tnorm_sq);
      rr /= tnorm;
      pp /= tnorm;
      qq = Sxy_cache.t() * rr;
      if (fit || return_ttrain) tt = Xtrain * rr;
    } else if (use_implicit_xprod) {
      tt = Xtrain * rr;
      if (center_t == 1) {
        tt -= arma::mean(tt);
      }
      const double tnorm = arma::norm(tt, 2);
      if (!std::isfinite(tnorm) || tnorm <= 0.0) return false;
      tt /= tnorm;
      rr /= tnorm;
      pp = Xtrain.t() * tt;
      qq = Ytrain.t() * tt;
    } else {
      tt = Xtrain * rr;
      if (center_t == 1) {
        tt -= arma::mean(tt);
      }
      const double tnorm = arma::norm(tt, 2);
      if (!std::isfinite(tnorm) || tnorm <= 0.0) return false;
      tt /= tnorm;
      rr /= tnorm;
      pp = Xt * tt;
      qq = Yt * tt;
    }

    arma::vec vv = pp;
    if (a_idx > 0) {
      auto Vprev = VV.cols(0, a_idx - 1);
      vv -= Vprev * (Vprev.t() * pp);
      if (reorth_v == 1) {
        vv -= Vprev * (Vprev.t() * vv);
      }
    }
    const double vnorm = arma::norm(vv, 2);
    if (!std::isfinite(vnorm) || vnorm <= 0.0) return false;
    vv /= vnorm;

    if (use_implicit_xprod) {
      // No persistent S exists in the implicit paths. Future refreshes apply
      // the VV projector directly to X'Y.
    } else if (defl_cache == 1) {
      arma::rowvec vS = vv.t() * S;
      S -= vv * vS;
    } else {
      S -= vv * (vv.t() * S);
    }

    RR.col(a_idx) = rr;
    QQ.col(a_idx) = qq;
    VV.col(a_idx) = vv;
    if (return_ttrain && tt.n_elem == static_cast<arma::uword>(n)) {
      TT.col(a_idx) = tt;
    }
    if (store_B && incremental_coefficients == 1) {
      Bcur += rr * qq.t();
    }
    if (fit) {
      Yfit_cur += tt * qq.t();
    }

    while (i_out < length_ncomp && a_idx == (ncomp(i_out) - 1)) {
      if (store_B) {
        B.slice(i_out) = incremental_coefficients == 1 ?
          Bcur :
          RR.cols(0, a_idx) * QQ.cols(0, a_idx).t();
      }
      if (fit) {
        R2Y(i_out) = RQ(Ytrain, Yfit_cur);
        arma::mat yf = Yfit_cur;
        yf.each_row() += mY;
        Yfit.slice(i_out) = yf;
      }
      ++i_out;
    }
    return true;
  };

  SimplsFastRefreshWorkspace refresh_ws;
  const int rsvd_sketch_dim = std::min(
    std::min(p, m),
    1 + std::max(rsvd_oversample, 0)
  );
  const int requested_power_iters = std::max(rsvd_power, 0);
  int a = 0;
  while (a < max_ncomp) {
    const int k_block = accelerated_simpls_block_size(max_ncomp - a, p, m);
    arma::mat Ublock;
    if (use_implicit_double_xprod) {
      arma::vec shat_block;
      if (!refresh_deflated_crossprod_left_double(
            Xtrain,
            Ytrain,
            VV,
            a,
            k_block,
            rsvd_oversample,
            requested_power_iters,
            static_cast<unsigned int>(seed + a),
            Ublock,
            shat_block
          )) {
        break;
      }
    } else {
      fastpls_svd::SVDResult direction = compute_truncated_svd_dispatch(
        S,
        k_block,
        fastpls_svd::SVD_METHOD_CPU_RSVD,
        std::max(rsvd_sketch_dim - 1, 0),
        requested_power_iters,
        0.0,
        static_cast<unsigned int>(seed + a),
        true,
        false
      );
      Ublock = direction.U;
      refresh_ws.shat = direction.s;
      if (Ublock.n_cols < 1) {
        break;
      }
    }
    if (Ublock.n_cols < 1) break;

    const int use_cols = std::min<int>(Ublock.n_cols, k_block);
    bool stop_now = false;
    for (int j = 0; j < use_cols && a < max_ncomp; ++j, ++a) {
      if (!append_component(Ublock.col(j), a)) {
        stop_now = true;
        break;
      }
    }
    if (stop_now) break;
  }

  List out = List::create(
    Named("P")       = arma::mat(),
    Named("Q")       = QQ,
    Named("Ttrain")  = return_ttrain ? TT : arma::mat(),
    Named("R")       = RR,
    Named("mX")      = mX,
    Named("vX")      = vX,
    Named("mY")      = mY,
    Named("p")       = p,
    Named("m")       = m,
    Named("ncomp")   = ncomp,
    Named("Yfit")    = Yfit,
    Named("R2Y")     = R2Y,
    Named("xprod_precision") = xprod_precision,
    Named("xprod_mode") = (use_implicit_double_xprod ? "implicit" : "materialized")
  );
  if (store_B) {
    out["B"] = B;
  }
  annotate_coefficient_storage(out, store_B);
  return out;
}

// [[Rcpp::export]]
List pls_model1_gpu(
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
) {
  if (!fastpls_svd::has_cuda_backend()) {
    stop(
      "pls_model1_gpu requires an available CUDA backend. "
      "No CPU fallback is performed."
    );
  }
  if (svd_method != fastpls_svd::SVD_METHOD_CUDA_RSVD) {
    stop("pls_model1_gpu requires svd.method='cuda_rsvd'");
  }

  const int n = Xtrain.n_rows;
  const int p = Xtrain.n_cols;
  const int m = Ytrain.n_cols;
  const int max_plssvd_rank = std::min(n, std::min(p, m));
  for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
    if (ncomp(i) > max_plssvd_rank) {
      ncomp(i) = max_plssvd_rank;
    }
    if (ncomp(i) < 1) {
      ncomp(i) = 1;
    }
  }
  const int max_ncomp = max(ncomp);
  const int max_ncomp_eff = std::min(max_ncomp, max_plssvd_rank);
  if (max_ncomp_eff < 1) {
    stop("plssvd effective rank is < 1");
  }

  arma::mat mX(1, p, fill::zeros);
  if (scaling < 3) {
    mX = mean(Xtrain, 0);
    Xtrain.each_row() -= mX;
  }

  arma::mat vX(1, p, fill::ones);
  if (scaling == 2) {
    vX = variance(Xtrain);
    Xtrain.each_row() /= vX;
  }

  arma::mat mY = mean(Ytrain, 0);
  Ytrain.each_row() -= mY;

  fastpls_svd::SVDOptions opt = fastpls_svd::options_from_method_id(
    svd_method,
    rsvd_oversample,
    rsvd_power,
    svds_tol,
    static_cast<unsigned int>(seed),
    false,
    false
  );

  const bool store_B = should_store_coefficients(p, m, ncomp.n_elem, true);
  fastpls_svd::PLSSVDGPUResult gpu = fastpls_svd::cuda_plssvd_fit(
    Xtrain,
    Ytrain,
    ncomp,
    fit,
    opt,
    store_B
  );

  arma::cube Yfit = gpu.Yfit;
  if (fit && Yfit.n_elem > 0) {
    for (arma::uword i = 0; i < Yfit.n_slices; ++i) {
      Yfit.slice(i).each_row() += mY;
    }
  }

  List out = List::create(
    Named("C_latent") = gpu.C_latent,
    Named("W_latent") = gpu.W_latent,
    Named("Q")       = gpu.Q,
    Named("Ttrain")  = gpu.Ttrain,
    Named("R")       = gpu.R,
    Named("mX")      = mX,
    Named("vX")      = vX,
    Named("mY")      = mY,
    Named("p")       = p,
    Named("m")       = m,
    Named("ncomp")   = ncomp,
    Named("Yfit")    = Yfit,
    Named("R2Y")     = gpu.R2Y
  );
  if (store_B) {
    out["B"] = gpu.B;
  }
  annotate_coefficient_storage(out, store_B);
  return out;
}

// [[Rcpp::export]]
List pls_model1_gpu_implicit_xprod(
  arma::mat Xtrain,
  arma::mat Ytrain,
  arma::ivec ncomp,
  int scaling,
  bool fit,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed
) {
  if (!fastpls_svd::has_cuda_backend()) {
    stop(
      "pls_model1_gpu_implicit_xprod requires an available CUDA backend. "
      "No CPU fallback is performed."
    );
  }

  const int n = Xtrain.n_rows;
  const int p = Xtrain.n_cols;
  const int m = Ytrain.n_cols;
  const int max_plssvd_rank = std::min(n, std::min(p, m));
  for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
    if (ncomp(i) > max_plssvd_rank) {
      ncomp(i) = max_plssvd_rank;
    }
    if (ncomp(i) < 1) {
      ncomp(i) = 1;
    }
  }
  const int max_ncomp = max(ncomp);
  const int max_ncomp_eff = std::min(max_ncomp, max_plssvd_rank);
  if (max_ncomp_eff < 1) {
    stop("plssvd effective rank is < 1");
  }

  arma::mat mX(1, p, fill::zeros);
  if (scaling < 3) {
    mX = mean(Xtrain, 0);
    Xtrain.each_row() -= mX;
  }

  arma::mat vX(1, p, fill::ones);
  if (scaling == 2) {
    vX = variance(Xtrain);
    Xtrain.each_row() /= vX;
  }

  arma::mat mY = mean(Ytrain, 0);
  Ytrain.each_row() -= mY;

  fastpls_svd::SVDOptions opt;
  opt.method = fastpls_svd::Method::RSVD;
  opt.oversample = std::max(rsvd_oversample, 0);
  opt.power_iters = std::max(rsvd_power, 0);
  opt.svds_tol = std::max(svds_tol, 0.0);
  opt.seed = static_cast<unsigned int>(seed);
  opt.left_only = false;
  opt.use_full_svd = false;

  const bool store_B = should_store_coefficients(p, m, ncomp.n_elem, true);
  fastpls_svd::PLSSVDGPUResult gpu = fastpls_svd::cuda_plssvd_fit_implicit_xprod(
    Xtrain,
    Ytrain,
    ncomp,
    fit,
    opt,
    store_B
  );

  arma::cube Yfit = gpu.Yfit;
  if (fit && Yfit.n_elem > 0) {
    for (arma::uword i = 0; i < Yfit.n_slices; ++i) {
      Yfit.slice(i).each_row() += mY;
    }
  }

  List out = List::create(
    Named("C_latent") = gpu.C_latent,
    Named("W_latent") = gpu.W_latent,
    Named("Q")       = gpu.Q,
    Named("Ttrain")  = gpu.Ttrain,
    Named("R")       = gpu.R,
    Named("mX")      = mX,
    Named("vX")      = vX,
    Named("mY")      = mY,
    Named("p")       = p,
    Named("m")       = m,
    Named("ncomp")   = ncomp,
    Named("Yfit")    = Yfit,
    Named("R2Y")     = gpu.R2Y,
    Named("xprod_mode") = "implicit"
  );
  if (store_B) {
    out["B"] = gpu.B;
  }
  annotate_coefficient_storage(out, store_B);
  return out;
}

// [[Rcpp::export]]
List pls_lda_gpu_native(
  arma::mat Xtrain,
  arma::mat Ytrain,
  arma::ivec y,
  arma::mat Xtest,
  arma::ivec ncomp,
  int n_classes,
  int method,
  int scaling,
  bool xprod,
  bool fit,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed,
  double lda_ridge
) {
  if (!fastpls_svd::has_cuda_backend() || !fastpls_svd::cuda_lda_native_available()) {
    stop(
      "pls_lda_gpu_native requires CUDA PLS and native CUDA LDA support. "
      "No CPU fallback is performed."
    );
  }
  if (Xtrain.n_rows == 0 || Xtrain.n_cols == 0 || Ytrain.n_rows != Xtrain.n_rows) {
    stop("pls_lda_gpu_native requires compatible non-empty Xtrain and Ytrain");
  }
  if (static_cast<arma::uword>(y.n_elem) != Xtrain.n_rows) {
    stop("pls_lda_gpu_native requires one class label per training row");
  }
  if (Xtest.n_cols != Xtrain.n_cols) {
    stop("pls_lda_gpu_native Xtest columns must match Xtrain columns");
  }
  if (n_classes < 2) {
    stop("pls_lda_gpu_native requires at least two classes");
  }
  if (ncomp.n_elem < 1) {
    stop("pls_lda_gpu_native requires at least one component count");
  }
  const arma::uword n_train_rows = Xtrain.n_rows;

  List model;
  bool direct_plssvd_gpu = false;
  fastpls_svd::PLSSVDGPUResult direct_gpu;
  if (method == 1) {
    const int n = Xtrain.n_rows;
    const int p = Xtrain.n_cols;
    const int m = Ytrain.n_cols;
    const int max_plssvd_rank = std::min(n, std::min(p, m));
    for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
      if (ncomp(i) > max_plssvd_rank) {
        ncomp(i) = max_plssvd_rank;
      }
      if (ncomp(i) < 1) {
        ncomp(i) = 1;
      }
    }
    const int max_ncomp_eff = std::min(static_cast<int>(max(ncomp)), max_plssvd_rank);
    if (max_ncomp_eff < 1) {
      stop("plssvd effective rank is < 1");
    }

    arma::mat mX(1, p, fill::zeros);
    if (scaling < 3) {
      mX = mean(Xtrain, 0);
      Xtrain.each_row() -= mX;
    }

    arma::mat vX(1, p, fill::ones);
    if (scaling == 2) {
      vX = variance(Xtrain);
      Xtrain.each_row() /= vX;
    }

    arma::mat mY = mean(Ytrain, 0);
    Ytrain.each_row() -= mY;

    fastpls_svd::SVDOptions opt;
    opt.method = fastpls_svd::Method::RSVD;
    opt.oversample = std::max(rsvd_oversample, 0);
    opt.power_iters = std::max(rsvd_power, 0);
    opt.svds_tol = std::max(svds_tol, 0.0);
    opt.seed = static_cast<unsigned int>(seed);
    opt.left_only = false;
    opt.use_full_svd = false;

    const bool store_B = should_store_coefficients(p, m, ncomp.n_elem, true);
    direct_gpu = xprod ?
      fastpls_svd::cuda_plssvd_fit_implicit_xprod(Xtrain, Ytrain, ncomp, fit, opt, store_B) :
      fastpls_svd::cuda_plssvd_fit(Xtrain, Ytrain, ncomp, fit, opt, store_B);
    direct_plssvd_gpu = true;

    arma::cube Yfit = direct_gpu.Yfit;
    if (fit && Yfit.n_elem > 0) {
      for (arma::uword i = 0; i < Yfit.n_slices; ++i) {
        Yfit.slice(i).each_row() += mY;
      }
    }

    model = List::create(
      Named("C_latent") = direct_gpu.C_latent,
      Named("W_latent") = direct_gpu.W_latent,
      Named("Q")       = direct_gpu.Q,
      Named("Ttrain")  = arma::mat(),
      Named("R")       = direct_gpu.R,
      Named("mX")      = mX,
      Named("vX")      = vX,
      Named("mY")      = mY,
      Named("p")       = p,
      Named("m")       = m,
      Named("ncomp")   = ncomp,
      Named("Yfit")    = Yfit,
      Named("R2Y")     = direct_gpu.R2Y,
      Named("xprod_mode") = xprod ? "implicit" : "materialized"
    );
    if (store_B) {
      model["B"] = direct_gpu.B;
    }
    annotate_coefficient_storage(model, store_B);
  } else if (method == 3) {
    model = pls_model2_fast_gpu_impl(
      Xtrain,
      Ytrain,
      ncomp,
      scaling,
      fit,
      fastpls_svd::SVD_METHOD_CUDA_RSVD,
      rsvd_oversample,
      rsvd_power,
      svds_tol,
      seed
    );
  } else {
    stop("pls_lda_gpu_native currently supports method=1 (plssvd) or method=3 (simpls)");
  }

  arma::mat R = Rcpp::as<arma::mat>(model["R"]);
  arma::mat mX = Rcpp::as<arma::mat>(model["mX"]);
  arma::mat vX = Rcpp::as<arma::mat>(model["vX"]);
  arma::ivec ncomp_eff = Rcpp::as<arma::ivec>(model["ncomp"]);
  int kmax = 0;
  for (arma::uword i = 0; i < ncomp_eff.n_elem; ++i) {
    if (ncomp_eff(i) > kmax) kmax = ncomp_eff(i);
  }
  if (kmax < 1 || kmax > static_cast<int>(R.n_cols)) {
    stop("pls_lda_gpu_native has invalid effective component count");
  }

  arma::mat R_predict = R.cols(0, static_cast<arma::uword>(kmax) - 1);
  if (vX.n_elem == R_predict.n_rows) {
    arma::vec scale = arma::vectorise(vX);
    for (arma::uword j = 0; j < R_predict.n_rows; ++j) {
      double s = scale(j);
      if (!std::isfinite(s) || s == 0.0) s = 1.0;
      R_predict.row(j) /= s;
    }
  }
  arma::rowvec offset(kmax, arma::fill::zeros);
  if (mX.n_elem == R_predict.n_rows) {
    offset = arma::vectorise(mX).t() * R_predict;
  }

  arma::ivec unique_ncomp = arma::unique(ncomp_eff);
  std::string lda_train_backend = "cuda_fused_project";
  std::vector<fastpls_svd::LDAGPUModel> gpu_models;
  if (direct_plssvd_gpu &&
      direct_gpu.Ttrain.n_rows == n_train_rows &&
      direct_gpu.Ttrain.n_cols >= static_cast<arma::uword>(kmax)) {
    gpu_models = fastpls_svd::cuda_lda_train_prefix(
      direct_gpu.Ttrain.cols(0, static_cast<arma::uword>(kmax) - 1),
      y,
      n_classes,
      unique_ncomp,
      lda_ridge
    );
    lda_train_backend = "cuda_fused_ttrain";
  } else if (model.containsElementNamed("Ttrain")) {
    arma::mat Ttrain = Rcpp::as<arma::mat>(model["Ttrain"]);
    if (Ttrain.n_rows == n_train_rows &&
        Ttrain.n_cols >= static_cast<arma::uword>(kmax)) {
      gpu_models = fastpls_svd::cuda_lda_train_prefix(
        Ttrain.cols(0, static_cast<arma::uword>(kmax) - 1),
        y,
        n_classes,
        unique_ncomp,
        lda_ridge
      );
      lda_train_backend = "cuda_fused_ttrain";
    }
  }
  if (gpu_models.empty()) {
    gpu_models = fastpls_svd::cuda_lda_project_train_prefix(
      Xtrain,
      R_predict,
      offset,
      y,
      n_classes,
      unique_ncomp,
      lda_ridge
    );
  }

  Rcpp::List lda_models(unique_ncomp.n_elem);
  Rcpp::CharacterVector lda_names(unique_ncomp.n_elem);
  for (arma::uword i = 0; i < unique_ncomp.n_elem; ++i) {
    const fastpls_svd::LDAGPUModel& gm = gpu_models[static_cast<size_t>(i)];
    lda_models[i] = Rcpp::List::create(
      Rcpp::Named("means") = gm.means,
      Rcpp::Named("inv_cov") = arma::mat(),
      Rcpp::Named("linear") = gm.linear,
      Rcpp::Named("constants") = gm.constants,
      Rcpp::Named("priors") = gm.priors,
      Rcpp::Named("ridge") = gm.ridge,
      Rcpp::Named("backend") = "cuda_native_fused"
    );
    lda_names[i] = std::to_string(unique_ncomp(i));
  }
  lda_models.attr("names") = lda_names;
  model["lda"] = Rcpp::List::create(
    Rcpp::Named("ncomp") = unique_ncomp,
    Rcpp::Named("models") = lda_models,
    Rcpp::Named("ridge") = lda_ridge,
    Rcpp::Named("train_backend") = lda_train_backend
  );
  model["R_predict"] = R_predict;
  model["R_offset"] = offset;
  model["classification_rule"] = "lda_cuda";
  model["lda_backend"] = "cuda_fused";

  if (Xtest.n_rows > 0) {
    Rcpp::IntegerMatrix pred_codes(Xtest.n_rows, ncomp_eff.n_elem);
    for (arma::uword i = 0; i < ncomp_eff.n_elem; ++i) {
      const int kk = ncomp_eff(i);
      arma::uword model_idx = 0;
      while (model_idx < unique_ncomp.n_elem && unique_ncomp(model_idx) != kk) {
        ++model_idx;
      }
      if (model_idx >= unique_ncomp.n_elem) {
        stop("pls_lda_gpu_native could not match LDA model to ncomp");
      }
      const fastpls_svd::LDAGPUModel& gm = gpu_models[static_cast<size_t>(model_idx)];
      Rcpp::List pred = fastpls_svd::cuda_lda_project_predict(
        Xtest,
        R_predict.cols(0, static_cast<arma::uword>(kk) - 1),
        offset.subvec(0, static_cast<arma::uword>(kk) - 1),
        gm.linear,
        gm.constants,
        false
      );
      Rcpp::IntegerVector col = pred["pred"];
      for (R_xlen_t r = 0; r < col.size(); ++r) {
        pred_codes(r, i) = col[r];
      }
    }
    model["pred_codes"] = pred_codes;
  }

  model["predict_backend"] = "cuda_fused_lda";
  model["flash_svd"] = true;
  model["flash_svd_backend"] = "cuda";
  model["flash_svd_mode"] = "fused_pls_lda";
  return model;
}

arma::cube pls_predict_scores_b_metal_cv(List& model, arma::mat Xtest) {
  if (!fastpls_svd::has_metal_backend()) {
    stop(
      "Metal CV prediction requires a macOS build with Apple Metal support. "
      "No CPU fallback is performed."
    );
  }

  const int m = Rcpp::as<int>(model["m"]);
  arma::ivec ncomp = Rcpp::as<arma::ivec>(model["ncomp"]);
  const arma::uword length_ncomp = static_cast<arma::uword>(ncomp.n_elem);

  Rcpp::NumericVector mX_vec = model["mX"];
  arma::rowvec mX(mX_vec.begin(), mX_vec.size(), false, true);
  Xtest.each_row() -= mX;
  Rcpp::NumericVector vX_vec = model["vX"];
  arma::rowvec vX(vX_vec.begin(), vX_vec.size(), false, true);
  Xtest.each_row() /= vX;
  Rcpp::NumericVector mY_vec = model["mY"];
  arma::rowvec mY(mY_vec.begin(), mY_vec.size(), false, true);

  arma::cube Ypred(Xtest.n_rows, static_cast<arma::uword>(m), length_ncomp, arma::fill::none);

  if (model.containsElementNamed("W_latent") && model.containsElementNamed("R")) {
    arma::mat R = Rcpp::as<arma::mat>(model["R"]);
    arma::cube W_latent = Rcpp::as<arma::cube>(model["W_latent"]);
    const int kmax = std::max(
      1,
      std::min(
        static_cast<int>(max(ncomp)),
        std::min(static_cast<int>(R.n_cols), static_cast<int>(W_latent.n_rows))
      )
    );
    arma::mat T = fastpls_svd::metal_matrix_multiply(Xtest, R.cols(0, kmax - 1));
    for (arma::uword a = 0; a < length_ncomp; ++a) {
      const int mc = std::max(1, std::min(static_cast<int>(ncomp(a)), kmax));
      arma::mat y = fastpls_svd::metal_matrix_multiply(
        T.cols(0, mc - 1),
        W_latent.slice(a).rows(0, mc - 1)
      );
      y.each_row() += mY;
      Ypred.slice(a) = y;
    }
    return Ypred;
  }

  if (model.containsElementNamed("B")) {
    Rcpp::NumericVector B_vec = model["B"];
    Rcpp::IntegerVector B_dim = B_vec.attr("dim");
    if (B_dim.size() != 3L ||
        B_dim[0] != Xtest.n_cols ||
        B_dim[1] != m ||
        B_dim[2] < static_cast<int>(length_ncomp)) {
      stop("Metal CV model coefficients are not compatible with prediction");
    }
    const arma::cube B(
      B_vec.begin(),
      static_cast<arma::uword>(B_dim[0]),
      static_cast<arma::uword>(B_dim[1]),
      static_cast<arma::uword>(B_dim[2]),
      false,
      true
    );
    for (arma::uword a = 0; a < length_ncomp; ++a) {
      arma::mat y = fastpls_svd::metal_matrix_multiply(Xtest, B.slice(a));
      y.each_row() += mY;
      Ypred.slice(a) = y;
    }
    return Ypred;
  }

  if (model.containsElementNamed("R") && model.containsElementNamed("Q")) {
    arma::mat R = Rcpp::as<arma::mat>(model["R"]);
    arma::mat Q = Rcpp::as<arma::mat>(model["Q"]);
    if (R.n_cols == 0 || Q.n_cols == 0) {
      stop("Metal CV compact model contains no latent components");
    }
    const int kmax = std::max(
      1,
      std::min(
        static_cast<int>(max(ncomp)),
        std::min(static_cast<int>(R.n_cols), static_cast<int>(Q.n_cols))
      )
    );
    arma::mat T = fastpls_svd::metal_matrix_multiply(Xtest, R.cols(0, kmax - 1));
    for (arma::uword a = 0; a < length_ncomp; ++a) {
      const int mc = std::max(1, std::min(static_cast<int>(ncomp(a)), kmax));
      arma::mat y = fastpls_svd::metal_matrix_multiply(
        T.cols(0, mc - 1),
        Q.cols(0, mc - 1).t()
      );
      y.each_row() += mY;
      Ypred.slice(a) = y;
    }
    return Ypred;
  }

  stop("Metal CV model does not contain usable prediction factors");
}

static arma::mat cv_projection_matrix(List& model, const int kmax, const arma::uword p) {
  Rcpp::NumericVector R_vec = model["R"];
  Rcpp::IntegerVector R_dim = R_vec.attr("dim");
  if (R_dim.size() != 2L || R_dim[0] != static_cast<int>(p) || R_dim[1] < kmax) {
    Rcpp::stop("CV classifier requires a compatible latent projection matrix R");
  }
  const arma::mat R(
    R_vec.begin(),
    static_cast<arma::uword>(R_dim[0]),
    static_cast<arma::uword>(R_dim[1]),
    false,
    true
  );
  arma::mat R_predict = R.cols(0, static_cast<arma::uword>(kmax) - 1);
  Rcpp::NumericVector vX_vec = model["vX"];
  arma::rowvec vX(vX_vec.begin(), vX_vec.size(), false, true);
  if (vX.n_elem == R_predict.n_rows) {
    for (arma::uword j = 0; j < R_predict.n_rows; ++j) {
      double s = vX(j);
      if (!std::isfinite(s) || s == 0.0) s = 1.0;
      R_predict.row(j) /= s;
    }
  }
  return R_predict;
}

static arma::rowvec cv_projection_offset(List& model, const arma::mat& R_predict) {
  arma::rowvec offset(R_predict.n_cols, arma::fill::zeros);
  Rcpp::NumericVector mX_vec = model["mX"];
  arma::rowvec mX(mX_vec.begin(), mX_vec.size(), false, true);
  if (mX.n_elem == R_predict.n_rows) {
    offset = mX * R_predict;
  }
  return offset;
}

static arma::mat cv_latent_scores(List& model,
                                  const arma::mat& X,
                                  const int kmax,
                                  const bool prefer_stored_ttrain) {
  if (prefer_stored_ttrain && model.containsElementNamed("Ttrain")) {
    arma::mat Ttrain = Rcpp::as<arma::mat>(model["Ttrain"]);
    if (Ttrain.n_rows == X.n_rows && Ttrain.n_cols >= static_cast<arma::uword>(kmax)) {
      return Ttrain.cols(0, static_cast<arma::uword>(kmax) - 1);
    }
  }
  arma::mat R_predict = cv_projection_matrix(model, kmax, X.n_cols);
  arma::rowvec offset = cv_projection_offset(model, R_predict);
  arma::mat T = X * R_predict;
  if (offset.n_elem >= static_cast<arma::uword>(kmax)) {
    T.each_row() -= offset;
  }
  return T;
}

static arma::imat cv_lda_predict_prefix_labels_cpp(const arma::mat& Ttest,
                                                   const Rcpp::List& lda_models,
                                                   const arma::ivec& ncomp) {
  const int ncopy = std::min(static_cast<int>(ncomp.n_elem), static_cast<int>(lda_models.size()));
  arma::imat out(Ttest.n_rows, static_cast<arma::uword>(ncopy), arma::fill::zeros);
  for (int s = 0; s < ncopy; ++s) {
    Rcpp::List lda_model = lda_models[s];
    arma::mat linear = Rcpp::as<arma::mat>(lda_model["linear"]);
    arma::rowvec constants = Rcpp::as<arma::rowvec>(lda_model["constants"]);
    const int kk = ncomp(static_cast<arma::uword>(s));
    if (kk < 1 ||
        kk > static_cast<int>(Ttest.n_cols) ||
        linear.n_cols != static_cast<arma::uword>(kk) ||
        constants.n_elem != linear.n_rows) {
      Rcpp::stop("CV LDA prefix model is not compatible with the requested component count");
    }
    arma::mat scores = Ttest.cols(0, static_cast<arma::uword>(kk) - 1) * linear.t();
    Rcpp::IntegerVector pred = lda_labels_from_scores(scores, constants);
    for (R_xlen_t ii = 0; ii < pred.size(); ++ii) {
      out(static_cast<arma::uword>(ii), static_cast<arma::uword>(s)) = pred[ii];
    }
  }
  return out;
}

List pls_cv_predict_compiled_impl(
  const arma::mat& Xdata,
  const arma::mat& Ydata,
  arma::ivec constrain,
  arma::ivec ncomp,
  int scaling,
  int kfold,
  int method,
  int backend,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed,
  bool classification,
  int n_response,
  bool xprod,
  int opls_north,
  bool return_scores,
  arma::mat class_codes,
  int classifier,
  double lda_ridge,
  bool store_predictions,
  int metric_id
) {
  if (svd_method == 1) stop("IRLBA is not part of fastPLS");
  const int nsamples = Xdata.n_rows;
  const bool label_classification = classification && Ydata.n_cols == 1;
  const bool use_class_codes =
    classification && label_classification && class_codes.n_rows > 0 && class_codes.n_cols > 0;
  const int n_classes = label_classification ?
    std::max(n_response, 1) :
    static_cast<int>(Ydata.n_cols);
  int ncolY = static_cast<int>(Ydata.n_cols);
  if (use_class_codes) {
    if (class_codes.n_rows != static_cast<arma::uword>(n_classes)) {
      stop("class_codes must have one row for each response class");
    }
    ncolY = static_cast<int>(class_codes.n_cols);
  }
  if (label_classification) {
    ncolY = use_class_codes ? ncolY : n_classes;
  }
  if (nsamples < 2) stop("Xdata must contain at least two samples");
  if (Ydata.n_rows != static_cast<arma::uword>(nsamples)) {
    stop("Ydata must have the same number of rows as Xdata");
  }
  if (constrain.n_elem != static_cast<arma::uword>(nsamples)) {
    stop("constrain must have one value for each sample");
  }
  if (ncomp.n_elem < 1) stop("ncomp must contain at least one value");
  const bool requested_leave_one_group_out = kfold < 0;
  if (!requested_leave_one_group_out && kfold < 2) kfold = 2;
  if (method < 1 || method > 5) {
    stop("method must be 1=plssvd, 2=simpls, 3=simpls_fast, 4=opls, or 5=kernelpls");
  }
  if (backend < 0 || backend > 2) stop("backend must be 0=cpp, 1=cuda, or 2=metal");
  if (classifier < 0 || classifier > 1) classifier = 0;
  if (!classification) classifier = 0;
  if (use_class_codes && classifier != 0) {
    stop("LDA CV is not available with Gaussian/code response compression");
  }
  if (backend == 1 && method == 2) {
    stop("CUDA classic SIMPLS is not implemented; use simpls_fast CUDA instead");
  }
  if (backend == 1 && !fastpls_svd::has_cuda_backend()) {
    stop(
      "CUDA CV requires a CUDA-enabled fastPLS build and an available GPU. "
      "No CPU fallback is performed."
    );
  }
  if (backend == 2 && !fastpls_svd::has_metal_backend()) {
    stop(
      "Metal CV requires a macOS build with Apple Metal support. "
      "No CPU fallback is performed."
    );
  }
  if (method == 1) {
    const int max_plssvd_rank = std::min(
      nsamples,
      std::min(static_cast<int>(Xdata.n_cols), ncolY)
    );
    for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
      if (ncomp(i) > max_plssvd_rank) ncomp(i) = max_plssvd_rank;
      if (ncomp(i) < 1) ncomp(i) = 1;
    }
  } else {
    for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
      if (ncomp(i) < 1) ncomp(i) = 1;
    }
  }

  auto class_label_at_sample = [&](const arma::uword i) -> int {
    if (label_classification) {
      return static_cast<int>(std::round(Ydata(i, 0)));
    }
    if (classification) {
      arma::uword best = 0;
      double best_val = Ydata(i, 0);
      for (arma::uword c = 1; c < Ydata.n_cols; ++c) {
        if (Ydata(i, c) > best_val) {
          best_val = Ydata(i, c);
          best = c;
        }
      }
      return static_cast<int>(best) + 1;
    }
    return 1;
  };

  arma::ivec unique_groups = arma::unique(constrain);
  arma::ivec constrain2(constrain.n_elem);
  arma::uvec first_sample(unique_groups.n_elem);
  first_sample.fill(static_cast<arma::uword>(nsamples));
  std::unordered_map<arma::sword, arma::uword> group_index;
  group_index.reserve(unique_groups.n_elem);
  for (arma::uword j = 0; j < unique_groups.n_elem; ++j) {
    group_index.emplace(unique_groups(j), j);
  }
  for (arma::uword i = 0; i < constrain.n_elem; ++i) {
    const arma::uword j = group_index.at(constrain(i));
    constrain2(i) = static_cast<int>(j) + 1;
    if (first_sample(j) == static_cast<arma::uword>(nsamples)) {
      first_sample(j) = i;
    }
  }

  const int ngroups = unique_groups.n_elem;
  const bool leave_one_group_out = requested_leave_one_group_out || kfold >= ngroups;
  if (leave_one_group_out) {
    kfold = std::max(ngroups, 1);
  }
  arma::ivec group_fold(ngroups, arma::fill::zeros);
  if (leave_one_group_out) {
    for (int j = 0; j < ngroups; ++j) {
      group_fold(j) = j;
    }
  } else if (classification && n_classes > 1) {
    std::vector<std::vector<int> > groups_by_class(static_cast<std::size_t>(n_classes));
    for (int j = 0; j < ngroups; ++j) {
      int cls = class_label_at_sample(first_sample(j));
      if (cls < 1) cls = 1;
      if (cls > n_classes) cls = n_classes;
      groups_by_class[static_cast<std::size_t>(cls - 1)].push_back(j);
    }
    for (int cls = 0; cls < n_classes; ++cls) {
      const int n_class_groups = static_cast<int>(groups_by_class[static_cast<std::size_t>(cls)].size());
      if (n_class_groups < 1) continue;
      IntegerVector frame = seq_len(n_class_groups);
      IntegerVector perm = samplewithoutreplace(frame, n_class_groups);
      for (int pos = 0; pos < n_class_groups; ++pos) {
        const int perm_pos = perm[pos] - 1;
        const int group_idx = groups_by_class[static_cast<std::size_t>(cls)][static_cast<std::size_t>(perm_pos)];
        group_fold(group_idx) = pos % kfold;
      }
    }
  } else {
    IntegerVector frame = seq_len(ngroups);
    IntegerVector perm = samplewithoutreplace(frame, ngroups);
    for (int j = 0; j < ngroups; ++j) {
      group_fold(j) = (perm[j] - 1) % kfold;
    }
  }
  arma::ivec fold(nsamples);
  for (int i = 0; i < nsamples; ++i) {
    const int group_idx = constrain2(i) - 1;
    fold(i) = group_fold(group_idx);
  }

  const int length_ncomp = ncomp.n_elem;
  const bool store_score_predictions =
    store_predictions && ((!classification) || (return_scores && classifier == 0));
  const bool store_class_predictions = store_predictions && classification;
  const bool latent_classifier_cv = classification && !store_score_predictions && classifier != 0;
  arma::cube Ypred;
  if (store_score_predictions) {
    Ypred.zeros(nsamples, ncolY, length_ncomp);
  }
  arma::imat class_pred;
  if (store_class_predictions) {
    class_pred.zeros(nsamples, length_ncomp);
  }
  arma::vec metric_sse(length_ncomp, arma::fill::zeros);
  arma::vec metric_count(length_ncomp, arma::fill::zeros);
  arma::vec metric_correct(length_ncomp, arma::fill::zeros);
  arma::vec metric_total(length_ncomp, arma::fill::zeros);
  if (classification) {
    metric_id = 1;
  } else if (metric_id < 2 || metric_id > 4) {
    metric_id = 4;
  }
  double metric_tss = NA_REAL;
  if (!classification && (metric_id == 2 || metric_id == 3)) {
    arma::rowvec y_center = arma::mean(Ydata, 0);
    metric_tss = 0.0;
    for (arma::uword column = 0; column < Ydata.n_cols; ++column) {
      const double* values = Ydata.colptr(column);
      for (arma::uword row = 0; row < Ydata.n_rows; ++row) {
        const double centered = values[row] - y_center(column);
        metric_tss += centered * centered;
      }
    }
  }
  arma::ivec status(kfold, arma::fill::zeros);

  std::vector<arma::uvec> fold_test_index(static_cast<std::size_t>(kfold));
  std::vector<arma::uvec> fold_train_index(static_cast<std::size_t>(kfold));
  for (int f = 0; f < kfold; ++f) {
    fold_test_index[static_cast<std::size_t>(f)] = arma::find(fold == f);
    fold_train_index[static_cast<std::size_t>(f)] = arma::find(fold != f);
  }

  // For the compiled linear SIMPLS path, obtain each training-fold Gram
  // matrix from the full-data and held-out raw cross-products. This performs
  // two data-wide cross-products in total instead of one per training fold.
  // Fold means and scales are still calculated from training rows only.
  std::vector<arma::mat> fold_scaled_crossprod(static_cast<std::size_t>(kfold));
  std::vector<arma::mat> fold_x_mean(static_cast<std::size_t>(kfold));
  std::vector<arma::mat> fold_x_scale(static_cast<std::size_t>(kfold));
  std::vector<unsigned char> fold_has_crossprod(static_cast<std::size_t>(kfold), 0);
  const int max_ncomp = static_cast<int>(ncomp.max());
  const int cv_crossprod_max_p = env_int_or(
    "FASTPLS_CV_FOLD_GRAM_MAX_P", 2048, 16, 4096
  );
  const bool cv_crossprod_enabled =
    env_int_or("FASTPLS_CV_FOLD_GRAM_CACHE", 1, 0, 1) == 1 &&
    backend == 0 && !xprod && (method == 1 || method == 3 || method == 5) &&
    (method == 1 ||
     max_ncomp >= env_int_or("FASTPLS_FAST_CROSSPROD_MIN_NCOMP", 20, 1, 1024)) &&
    static_cast<int>(Xdata.n_cols) <= cv_crossprod_max_p;
  if (cv_crossprod_enabled) {
    const arma::mat full_raw_crossprod = fastpls::native::symmetric_crossprod(Xdata);
    const arma::rowvec full_sum = arma::sum(Xdata, 0);
    const int min_ratio = env_int_or(
      "FASTPLS_FAST_CROSSPROD_MIN_N_TO_P_RATIO", 8, 1, 1024
    );
    for (int f = 0; f < kfold; ++f) {
      const arma::uvec& test_idx = fold_test_index[static_cast<std::size_t>(f)];
      const arma::uword ntrain = Xdata.n_rows - test_idx.n_elem;
      if (test_idx.n_elem == 0 || ntrain < 2 ||
          ntrain < Xdata.n_cols * static_cast<arma::uword>(min_ratio)) {
        continue;
      }
      arma::mat heldout = Xdata.rows(test_idx);
      arma::mat train_crossprod = full_raw_crossprod -
        fastpls::native::symmetric_crossprod(heldout);
      arma::rowvec train_mean(Xdata.n_cols, arma::fill::zeros);
      arma::rowvec train_scale(Xdata.n_cols, arma::fill::ones);
      if (scaling < 3) {
        const arma::rowvec train_sum = full_sum - arma::sum(heldout, 0);
        train_mean = train_sum / static_cast<double>(ntrain);
        train_crossprod -= static_cast<double>(ntrain) *
          train_mean.t() * train_mean;
        if (scaling == 2) {
          train_scale = arma::sqrt(
            train_crossprod.diag().t() / static_cast<double>(ntrain - 1)
          );
          train_crossprod.each_col() /= train_scale.t();
          train_crossprod.each_row() /= train_scale;
        }
      }
      train_crossprod = arma::symmatu(train_crossprod);
      fold_scaled_crossprod[static_cast<std::size_t>(f)] =
        std::move(train_crossprod);
      fold_x_mean[static_cast<std::size_t>(f)] = std::move(train_mean);
      fold_x_scale[static_cast<std::size_t>(f)] = std::move(train_scale);
      fold_has_crossprod[static_cast<std::size_t>(f)] = 1;
    }
  }

  arma::mat full_raw_crosscov;
  arma::rowvec full_x_sum;
  arma::rowvec full_y_sum;
  const bool cv_crosscov_enabled =
    env_int_or("FASTPLS_CV_FOLD_CROSSCOV_CACHE", 1, 0, 1) == 1 &&
    backend == 0 && !classification && !xprod && scaling == 1 &&
    (method == 1 || method == 3 || method == 5);
  if (cv_crosscov_enabled) {
    full_raw_crosscov = Xdata.t() * Ydata;
    full_x_sum = arma::sum(Xdata, 0);
    full_y_sum = arma::sum(Ydata, 0);
  }
  arma::mat full_class_sums;
  arma::uvec all_compact_labels;
  const bool cv_class_crosscov_enabled =
    env_int_or("FASTPLS_CV_CLASS_SUM_CACHE", 1, 0, 1) == 1 &&
    backend == 0 && label_classification && !use_class_codes && !xprod &&
    scaling == 1 && (method == 3 || method == 5);
  if (cv_class_crosscov_enabled) {
    all_compact_labels.set_size(Xdata.n_rows);
    for (arma::uword row = 0; row < Xdata.n_rows; ++row) {
      all_compact_labels(row) = static_cast<arma::uword>(
        std::max(1, class_label_at_sample(row)) - 1
      );
    }
    full_class_sums = fastpls::native::dummy_crossprod(
      Xdata, all_compact_labels,
      arma::rowvec(static_cast<arma::uword>(n_classes), arma::fill::zeros)
    );
    if (full_x_sum.n_elem == 0) full_x_sum = arma::sum(Xdata, 0);
  }

  const std::string method_name =
    (method == 1) ? "plssvd" :
    ((method == 2) ? "simpls" :
    ((method == 4) ? "opls" :
    ((method == 5) ? "kernelpls" : "simpls_fast")));

  auto response_rows_into = [&](const arma::uvec& idx, arma::mat& out) {
    out.zeros(idx.n_elem, static_cast<arma::uword>(ncolY));
    for (arma::uword ii = 0; ii < idx.n_elem; ++ii) {
      const int cls = static_cast<int>(std::round(Ydata(idx(ii), 0)));
      if (use_class_codes) {
        if (cls >= 1 && cls <= n_classes) {
          out.row(ii) = class_codes.row(static_cast<arma::uword>(cls - 1));
        }
      } else if (cls >= 1 && cls <= ncolY) {
        out(ii, static_cast<arma::uword>(cls - 1)) = 1.0;
      }
    }
  };
  auto response_rows = [&](const arma::uvec& idx) -> arma::mat {
    arma::mat out;
    response_rows_into(idx, out);
    return out;
  };

  // Retain storage between folds when a fitting route does not take ownership.
  // Every fold still overwrites all values from the original input matrices.
  arma::mat Xtrain;
  arma::mat Xtest;
  arma::mat Ytrain;
  for (int f = 0; f < kfold; ++f) {
    Rcpp::checkUserInterrupt();
    const arma::uvec& test_idx = fold_test_index[static_cast<std::size_t>(f)];
    const arma::uvec& train_idx = fold_train_index[static_cast<std::size_t>(f)];
    if (test_idx.n_elem == 0) {
      status(f) = 2; // empty fold
      continue;
    }
    if (train_idx.n_elem == 0) {
      for (int s = 0; s < length_ncomp; ++s) {
        if (classification) {
          for (arma::uword ii = 0; ii < test_idx.n_elem; ++ii) {
            const int actual_class = class_label_at_sample(test_idx(ii));
            if (store_class_predictions) {
              class_pred(test_idx(ii), s) = actual_class;
            }
            metric_correct(s) += 1.0;
            metric_total(s) += 1.0;
          }
          if (store_score_predictions) {
            if (label_classification) {
              Ypred.slice(s).rows(test_idx) = response_rows(test_idx);
            } else {
              Ypred.slice(s).rows(test_idx) = Ydata.rows(test_idx);
            }
          }
        } else {
          if (store_score_predictions) {
            Ypred.slice(s).rows(test_idx) = Ydata.rows(test_idx);
          }
          metric_count(s) += static_cast<double>(test_idx.n_elem * Ydata.n_cols);
        }
      }
      status(f) = 3; // no training data
      continue;
    }

    Xtest = Xdata.rows(test_idx);
    arma::uvec fold_compact_labels;
    const bool use_compact_fold_labels = cv_class_crosscov_enabled;
    if (use_compact_fold_labels) {
      fold_compact_labels = all_compact_labels.elem(train_idx);
      Ytrain.reset();
    } else if (label_classification) {
      response_rows_into(train_idx, Ytrain);
    } else {
      Ytrain = Ydata.rows(train_idx);
    }
    arma::mat fold_crosscov;
    arma::mat fold_y_mean;
    if (cv_crosscov_enabled) {
      const arma::mat Ytest_fold = Ydata.rows(test_idx);
      const arma::rowvec train_x_sum = full_x_sum - arma::sum(Xtest, 0);
      const arma::rowvec train_y_sum = full_y_sum - arma::sum(Ytest_fold, 0);
      const arma::rowvec train_y_mean =
        train_y_sum / static_cast<double>(train_idx.n_elem);
      fold_y_mean = train_y_mean;
      fold_crosscov = full_raw_crosscov - Xtest.t() * Ytest_fold -
        train_x_sum.t() * train_y_mean;
    } else if (cv_class_crosscov_enabled) {
      const arma::uvec test_labels = all_compact_labels.elem(test_idx);
      arma::mat heldout_class_sums = fastpls::native::dummy_crossprod(
        Xtest, test_labels,
        arma::rowvec(static_cast<arma::uword>(n_classes), arma::fill::zeros)
      );
      arma::rowvec train_counts(static_cast<arma::uword>(n_classes), arma::fill::zeros);
      for (arma::uword row = 0; row < fold_compact_labels.n_elem; ++row) {
        train_counts(fold_compact_labels(row)) += 1.0;
      }
      fold_y_mean = train_counts / static_cast<double>(train_idx.n_elem);
      const arma::rowvec train_x_sum = full_x_sum - arma::sum(Xtest, 0);
      fold_crosscov = full_class_sums - heldout_class_sums -
        train_x_sum.t() *
          (train_counts / static_cast<double>(train_idx.n_elem));
    }

    arma::rowvec fold_class_counts;
    if (classification) {
      fold_class_counts.zeros(n_classes);
      for (arma::uword ii = 0; ii < train_idx.n_elem; ++ii) {
        int cls = class_label_at_sample(train_idx(ii));
        if (cls >= 1 && cls <= n_classes) {
          fold_class_counts(static_cast<arma::uword>(cls - 1)) += 1.0;
        }
      }
      arma::uvec active = arma::find(fold_class_counts > 0.5);
      if (active.n_elem <= 1) {
        arma::rowvec fallback(ncolY, arma::fill::zeros);
        if (active.n_elem == 1) {
          if (use_class_codes) {
            fallback = class_codes.row(active(0));
          } else {
            fallback(active(0)) = 1.0;
          }
        } else {
          fallback = label_classification ? arma::mean(response_rows(train_idx), 0) : arma::mean(Ydata, 0);
        }
        const int fallback_class = (active.n_elem == 1) ?
          static_cast<int>(active(0)) + 1 :
          static_cast<int>(nearest_code_classes(fallback, use_class_codes ? class_codes : arma::eye(ncolY, ncolY))(0));
        for (int s = 0; s < length_ncomp; ++s) {
          for (arma::uword ii = 0; ii < test_idx.n_elem; ++ii) {
            if (store_class_predictions) {
              class_pred(test_idx(ii), s) = fallback_class;
            }
            metric_correct(s) += (fallback_class == class_label_at_sample(test_idx(ii))) ? 1.0 : 0.0;
            metric_total(s) += 1.0;
            if (store_score_predictions) {
              Ypred.slice(s).row(test_idx(ii)) = fallback;
            }
          }
        }
        status(f) = 4; // degenerate classification fold
        continue;
      }
    }

    const bool use_fold_sufficient_stats =
      backend == 0 &&
      fold_has_crossprod[static_cast<std::size_t>(f)] &&
      (cv_crosscov_enabled || cv_class_crosscov_enabled) &&
      method != 4;
    if (!use_fold_sufficient_stats) {
      Xtrain = Xdata.rows(train_idx);
    } else {
      Xtrain.reset();
    }

    int fit_method = method;
    int fit_scaling = scaling;
    if (method == 4) {
      const int north_eff = std::max(opls_north, 0);
      fastpls::runtime::CpuLinearAlgebraF64 core_backend;
      auto filter = fastpls::core::fit_opls_filter_inplace(
        fastpls::core::make_view(
          Xtrain.memptr(), Xtrain.n_rows, Xtrain.n_cols, Xtrain.n_rows
        ),
        fastpls::core::make_const_view(
          Ytrain.memptr(), Ytrain.n_rows, Ytrain.n_cols, Ytrain.n_rows
        ),
        static_cast<std::size_t>(north_eff),
        static_cast<fastpls::core::PredictorScaling>(scaling), core_backend
      );
      const int removed = static_cast<int>(filter.completed_components);
      const int available = std::min(
        static_cast<int>(Xtrain.n_rows) - (scaling < 3 ? 1 : 0),
        static_cast<int>(Xtrain.n_cols)
      ) - removed;
      if (ncomp.max() > available) {
        Rcpp::stop(
          "OPLS requested %d predictive components, but at most %d remain "
          "after removing %d orthogonal components; reduce ncomp or north.",
          static_cast<int>(ncomp.max()), available, removed
        );
      }
      fastpls::core::apply_opls_filter_inplace<double>(
        fastpls::core::make_view(
          Xtest.memptr(), Xtest.n_rows, Xtest.n_cols, Xtest.n_rows
        ),
        filter.predictor_center.data(), filter.predictor_scale.data(),
        filter.predictor_center.size(),
        fastpls::core::ConstMatrixView<double>(filter.weights.view()),
        fastpls::core::ConstMatrixView<double>(filter.loadings.view()),
        core_backend
      );
      fit_method = 3;
      fit_scaling = 3;
    } else if (method == 5) {
      // Linear kernelPLS is algebraically the direct SIMPLS core. Nonlinear
      // kernels still use the ordinary kernel_pls_* fit wrappers.
      fit_method = 3;
    }

    auto fit_input_X = [&](arma::mat& X) -> arma::mat {
      if (latent_classifier_cv) {
        return X;
      }
      return std::move(X);
    };

    List model;
    if (backend == 1) {
      if (fit_method == 1) {
        if (xprod) {
          model = pls_model1_gpu_implicit_xprod(
            fit_input_X(Xtrain), std::move(Ytrain), ncomp, fit_scaling, false,
            rsvd_oversample, rsvd_power, svds_tol, seed + f
          );
        } else {
          model = pls_model1_gpu(
            fit_input_X(Xtrain), std::move(Ytrain), ncomp, fit_scaling, false,
            fastpls_svd::SVD_METHOD_CUDA_RSVD,
            rsvd_oversample, rsvd_power, svds_tol, seed + f
          );
        }
      } else {
        model = pls_model2_fast_gpu_impl(
          Xtrain, std::move(Ytrain), ncomp, fit_scaling, false,
          fastpls_svd::SVD_METHOD_CUDA_RSVD,
          rsvd_oversample, rsvd_power, svds_tol, seed + f,
          latent_classifier_cv ? nullptr : &Xtrain
        );
      }
    } else if (backend == 2) {
      if (fit_method == 1) {
        model = pls_model1_metal_cv(
          fit_input_X(Xtrain), std::move(Ytrain), ncomp, fit_scaling,
          rsvd_oversample, rsvd_power, svds_tol, seed + f
        );
      } else {
        model = pls_model2_fast_metal_cv(
          fit_input_X(Xtrain), std::move(Ytrain), ncomp, fit_scaling,
          rsvd_power, seed + f
        );
      }
    } else {
      if (fit_method == 1) {
        if (xprod) {
          const int xprod_precision = 3;
          model = pls_model1_rsvd_xprod_precision(
            fit_input_X(Xtrain), std::move(Ytrain), ncomp, fit_scaling, false,
            rsvd_oversample, rsvd_power, svds_tol, seed + f, xprod_precision
          );
        } else {
          model = pls_model1_impl(
            use_fold_sufficient_stats ? Xdata : Xtrain,
            use_fold_sufficient_stats ? Ydata : Ytrain,
            ncomp, fit_scaling, false, svd_method,
            rsvd_oversample, rsvd_power, svds_tol, seed + f,
            cv_crosscov_enabled ? &fold_crosscov : nullptr,
            use_fold_sufficient_stats ?
              &fold_scaled_crossprod[static_cast<std::size_t>(f)] : nullptr,
            use_fold_sufficient_stats ?
              &fold_x_mean[static_cast<std::size_t>(f)] : nullptr,
            use_fold_sufficient_stats ?
              &fold_x_scale[static_cast<std::size_t>(f)] : nullptr,
            use_fold_sufficient_stats ? &fold_y_mean : nullptr,
            use_fold_sufficient_stats ? static_cast<int>(train_idx.n_elem) : 0
          );
        }
      } else if (fit_method == 2) {
        model = pls_model2(
          fit_input_X(Xtrain), std::move(Ytrain), ncomp, fit_scaling, false, svd_method,
          rsvd_oversample, rsvd_power, svds_tol, seed + f
        );
      } else {
        if (xprod) {
          const int xprod_precision = 3;
          model = pls_model2_fast_rsvd_xprod_precision(
            fit_input_X(Xtrain), std::move(Ytrain), ncomp, fit_scaling, false,
            rsvd_oversample, rsvd_power, svds_tol, seed + f, xprod_precision
          );
        } else {
          const arma::uvec* fit_labels = use_compact_fold_labels ?
            (use_fold_sufficient_stats ? &all_compact_labels : &fold_compact_labels) :
            nullptr;
          model = pls_model2_fast_impl(
            use_fold_sufficient_stats ? Xdata : Xtrain,
            use_fold_sufficient_stats ? Ydata : Ytrain,
            ncomp, fit_scaling, false, svd_method,
            rsvd_oversample, rsvd_power, svds_tol, seed + f,
            (latent_classifier_cv || use_fold_sufficient_stats) ? nullptr : &Xtrain,
            fit_labels,
            use_compact_fold_labels ? n_classes : 0,
            fold_has_crossprod[static_cast<std::size_t>(f)] ?
              &fold_scaled_crossprod[static_cast<std::size_t>(f)] : nullptr,
            (cv_crosscov_enabled || cv_class_crosscov_enabled) ?
              &fold_crosscov : nullptr,
            use_fold_sufficient_stats ?
              &fold_x_mean[static_cast<std::size_t>(f)] : nullptr,
            use_fold_sufficient_stats ?
              &fold_x_scale[static_cast<std::size_t>(f)] : nullptr,
            use_fold_sufficient_stats ? &fold_y_mean : nullptr,
            use_fold_sufficient_stats ? static_cast<int>(train_idx.n_elem) : 0
          );
        }
      }
    }

    const std::string fit_method_name =
      (fit_method == 1) ? "plssvd" : ((fit_method == 2) ? "simpls" : "simpls_fast");
    model["pls_method"] = fit_method_name;
    model["predict_latent_ok"] = true;

    if (classification && !store_score_predictions) {
      arma::imat fold_class_pred;
      if (classifier == 1) {
        int kmax = 0;
        for (arma::uword a = 0; a < ncomp.n_elem; ++a) {
          if (ncomp(a) > kmax) kmax = ncomp(a);
        }
        arma::mat Ttrain;
        if (!use_fold_sufficient_stats) {
          Ttrain = cv_latent_scores(model, Xtrain, kmax, true);
        }
        arma::mat Ttest = cv_latent_scores(model, Xtest, kmax, false);
        Rcpp::IntegerVector y_train_vec(train_idx.n_elem);
        for (arma::uword ii = 0; ii < train_idx.n_elem; ++ii) {
          y_train_vec[static_cast<R_xlen_t>(ii)] = class_label_at_sample(train_idx(ii));
        }
        // LDA uses a dense 1..C encoding even when a rare class is absent
        // from this fold; predictions are mapped back to global labels below.
        arma::uvec lda_class_counts(
          static_cast<arma::uword>(n_classes), arma::fill::zeros
        );
        for (R_xlen_t ii = 0; ii < y_train_vec.size(); ++ii) {
          const int cls = y_train_vec[ii];
          if (cls >= 1 && cls <= n_classes) {
            lda_class_counts(static_cast<arma::uword>(cls - 1)) += 1;
          }
        }
        const arma::uvec lda_active = arma::find(lda_class_counts > 0);
        Rcpp::IntegerVector lda_class_map(n_classes + 1, 0);
        for (arma::uword cls = 0; cls < lda_active.n_elem; ++cls) {
          lda_class_map[static_cast<R_xlen_t>(lda_active(cls) + 1)] =
            static_cast<int>(cls) + 1;
        }
        Rcpp::IntegerVector y_train_lda(y_train_vec.size());
        for (R_xlen_t ii = 0; ii < y_train_vec.size(); ++ii) {
          y_train_lda[ii] = lda_class_map[y_train_vec[ii]];
        }
        auto restore_lda_class = [&](const int compact_class) -> int {
          if (
            compact_class < 1 ||
            compact_class > static_cast<int>(lda_active.n_elem)
          ) {
            Rcpp::stop("LDA cross-validation returned an invalid compact class");
          }
          return static_cast<int>(
            lda_active(static_cast<arma::uword>(compact_class - 1))
          ) + 1;
        };
        Rcpp::IntegerVector ncomp_vec(ncomp.n_elem);
        for (arma::uword s = 0; s < ncomp.n_elem; ++s) {
          ncomp_vec[static_cast<R_xlen_t>(s)] = ncomp(s);
        }
        Rcpp::List lda_models;
        if (use_fold_sufficient_stats) {
          arma::mat Rfull = Rcpp::as<arma::mat>(model["R"]);
          arma::mat Rmax = Rfull.cols(0, static_cast<arma::uword>(kmax - 1));
          arma::mat score_gram = Rmax.t() *
            fold_scaled_crossprod[static_cast<std::size_t>(f)] * Rmax;
          arma::mat class_score_sums =
            fold_crosscov.cols(lda_active).t() * Rmax;
          arma::vec active_counts(lda_active.n_elem);
          for (arma::uword cls = 0; cls < lda_active.n_elem; ++cls) {
            active_counts(cls) = lda_class_counts(lda_active(cls));
          }
          lda_models = lda_train_moments_prefix_cpp(
            score_gram, class_score_sums, active_counts,
            static_cast<int>(train_idx.n_elem), ncomp_vec
          );
        } else if (backend == 1) {
          lda_models = lda_train_prefix_cuda(
            Ttrain, y_train_lda, static_cast<int>(lda_active.n_elem),
            ncomp_vec, lda_ridge
          );
        } else {
          lda_models = lda_train_prefix_cpp(
            Ttrain, y_train_lda, static_cast<int>(lda_active.n_elem),
            ncomp_vec, lda_ridge
          );
        }
        if (backend == 1) {
          fold_class_pred.set_size(test_idx.n_elem, length_ncomp);
          for (int s = 0; s < length_ncomp; ++s) {
            const int kk = ncomp(static_cast<arma::uword>(s));
            Rcpp::List lda_model = lda_models[s];
            arma::mat Ttest_k = Ttest.cols(0, static_cast<arma::uword>(kk) - 1);
            Rcpp::IntegerVector pred = lda_predict_labels_cuda(Ttest_k, lda_model);
            for (R_xlen_t ii = 0; ii < pred.size(); ++ii) {
              const int compact_class = pred[ii];
              fold_class_pred(
                static_cast<arma::uword>(ii), static_cast<arma::uword>(s)
              ) = restore_lda_class(compact_class);
            }
          }
        } else {
          fold_class_pred = cv_lda_predict_prefix_labels_cpp(Ttest, lda_models, ncomp);
          for (arma::uword s = 0; s < fold_class_pred.n_cols; ++s) {
            for (arma::uword ii = 0; ii < fold_class_pred.n_rows; ++ii) {
              const int compact_class = fold_class_pred(ii, s);
              fold_class_pred(ii, s) = restore_lda_class(compact_class);
            }
          }
        }
      } else if (backend == 2) {
        arma::cube fold_scores = pls_predict_scores_b_metal_cv(model, std::move(Xtest));
        fold_class_pred.set_size(fold_scores.n_rows, fold_scores.n_slices);
        for (arma::uword s = 0; s < fold_scores.n_slices; ++s) {
          if (use_class_codes) {
            arma::ivec fold_class = nearest_code_classes(fold_scores.slice(s), class_codes);
            fold_class_pred.col(s) = fold_class;
          } else {
            for (arma::uword ii = 0; ii < fold_scores.n_rows; ++ii) {
              fold_class_pred(ii, s) =
                static_cast<int>(fold_scores.slice(s).row(ii).index_max()) + 1;
            }
          }
        }
      } else if (use_class_codes) {
        fold_class_pred = (backend == 1) ?
          pls_predict_code_classes_compact_cuda(model, std::move(Xtest), class_codes) :
          pls_predict_code_classes_compact_cpu(model, std::move(Xtest), class_codes);
      } else {
        fold_class_pred = (backend == 1) ?
          pls_predict_classes_compact_cuda(model, std::move(Xtest)) :
          pls_predict_classes_compact_cpu(model, std::move(Xtest));
      }
      const int ncopy = std::min(length_ncomp, static_cast<int>(fold_class_pred.n_cols));
      for (int s = 0; s < ncopy; ++s) {
        for (arma::uword ii = 0; ii < test_idx.n_elem; ++ii) {
          const int pred_class = fold_class_pred(ii, s);
          if (store_class_predictions) {
            class_pred(test_idx(ii), s) = pred_class;
          }
          metric_correct(s) += (pred_class == class_label_at_sample(test_idx(ii))) ? 1.0 : 0.0;
          metric_total(s) += 1.0;
        }
      }
    } else {
      auto consume_fold_scores = [&](const int s, const arma::mat& scores) {
        if (classification) {
          if (use_class_codes) {
            arma::ivec fold_class = nearest_code_classes(scores, class_codes);
            for (arma::uword ii = 0; ii < test_idx.n_elem; ++ii) {
              const int pred_class = fold_class(ii);
              if (store_class_predictions) {
                class_pred(test_idx(ii), s) = pred_class;
              }
              metric_correct(s) += (pred_class == class_label_at_sample(test_idx(ii))) ? 1.0 : 0.0;
              metric_total(s) += 1.0;
            }
          } else {
            for (arma::uword ii = 0; ii < test_idx.n_elem; ++ii) {
              arma::uword best = 0;
              double best_val = scores(ii, 0);
              for (arma::uword c = 1; c < scores.n_cols; ++c) {
                if (scores(ii, c) > best_val) {
                  best_val = scores(ii, c);
                  best = c;
                }
              }
              const int pred_class = static_cast<int>(best) + 1;
              if (store_class_predictions) {
                class_pred(test_idx(ii), s) = pred_class;
              }
              metric_correct(s) += (pred_class == class_label_at_sample(test_idx(ii))) ? 1.0 : 0.0;
              metric_total(s) += 1.0;
            }
          }
          if (store_score_predictions) {
            Ypred.slice(s).rows(test_idx) = scores;
          }
        } else {
          double sse = 0.0;
          for (arma::uword column = 0; column < scores.n_cols; ++column) {
            const double* predicted = scores.colptr(column);
            const double* observed = Ydata.colptr(column);
            for (arma::uword row = 0; row < test_idx.n_elem; ++row) {
              const double error = predicted[row] - observed[test_idx(row)];
              sse += error * error;
            }
          }
          metric_sse(s) += sse;
          metric_count(s) += static_cast<double>(scores.n_elem);
          if (store_score_predictions) {
            Ypred.slice(s).rows(test_idx) = scores;
          }
        }
      };

      bool incremental_simpls_done = false;
      if (fit_method != 1 &&
          model.containsElementNamed("R") &&
          model.containsElementNamed("Q") &&
          model.containsElementNamed("mX") &&
          model.containsElementNamed("vX") &&
          model.containsElementNamed("mY")) {
        arma::mat RR = Rcpp::as<arma::mat>(model["R"]);
        arma::mat QQ = Rcpp::as<arma::mat>(model["Q"]);
        arma::rowvec mX = Rcpp::as<arma::rowvec>(model["mX"]);
        arma::rowvec vX = Rcpp::as<arma::rowvec>(model["vX"]);
        arma::rowvec mY = Rcpp::as<arma::rowvec>(model["mY"]);
        int max_requested = 0;
        bool ncomp_ok = true;
        for (arma::uword a = 0; a < ncomp.n_elem; ++a) {
          if (ncomp(a) < 1) ncomp_ok = false;
          if (ncomp(a) > max_requested) max_requested = ncomp(a);
        }
        const int kcap = std::min(static_cast<int>(RR.n_cols), static_cast<int>(QQ.n_cols));
        if (ncomp_ok &&
            max_requested >= 1 &&
            max_requested <= kcap &&
            RR.n_rows == Xtest.n_cols &&
            QQ.n_rows == static_cast<arma::uword>(ncolY) &&
            mX.n_elem == Xtest.n_cols &&
            vX.n_elem == Xtest.n_cols &&
            mY.n_elem == QQ.n_rows) {
          arma::mat Xscaled = Xtest;
          Xscaled.each_row() -= mX;
          Xscaled.each_row() /= vX;
          arma::mat T = Xscaled * RR.cols(0, static_cast<arma::uword>(max_requested) - 1);
          arma::mat accumulated(Xtest.n_rows, QQ.n_rows, arma::fill::zeros);
          arma::vec ncomp_order_key = arma::conv_to<arma::vec>::from(ncomp);
          arma::uvec order = arma::sort_index(ncomp_order_key);
          int previous_components = 0;
          for (arma::uword ord_i = 0; ord_i < order.n_elem; ++ord_i) {
            const int s = static_cast<int>(order(ord_i));
            const int mc = ncomp(static_cast<arma::uword>(s));
            for (int comp = previous_components; comp < mc; ++comp) {
              accumulated +=
                T.col(static_cast<arma::uword>(comp)) *
                QQ.col(static_cast<arma::uword>(comp)).t();
            }
            previous_components = mc;
            arma::mat scores = accumulated;
            scores.each_row() += mY;
            consume_fold_scores(s, scores);
          }
          incremental_simpls_done = true;
        }
      }

      if (!incremental_simpls_done) {
        arma::cube fold_pred;
        if (backend == 2) {
          fold_pred = pls_predict_scores_b_metal_cv(model, std::move(Xtest));
        } else {
          List pred = (backend == 1) ?
            pls_predict_flash_cuda(model, std::move(Xtest), false) :
            pls_predict_impl(model, Xtest, false);
          fold_pred = Rcpp::as<arma::cube>(pred["Ypred"]);
        }
        const int ncopy = std::min(length_ncomp, static_cast<int>(fold_pred.n_slices));
        for (int s = 0; s < ncopy; ++s) {
          consume_fold_scores(s, fold_pred.slice(static_cast<arma::uword>(s)));
        }
      }
    }
    status(f) = 1; // ok
  }

  const char* classifier_name = (classifier == 1) ? "lda" : "argmax";
  const char* prediction_backend =
    (classifier == 1 && backend == 1) ? "cuda_lda_cv" :
    (classifier == 1 ? "cpp_lda_cv" :
    (backend == 1 ? "cuda_flash" : (backend == 2 ? "metal" : "cpu")));

  List out = List::create(
    Named("fold") = fold + 1,
    Named("status") = status,
    Named("ncomp") = ncomp,
    Named("method") = method_name,
    Named("backend") = (backend == 1 ? "cuda" : (backend == 2 ? "metal" : "cpp")),
    Named("prediction_backend") = prediction_backend,
    Named("classifier") = classifier_name,
    Named("xprod") = xprod,
    Named("stratified_folds") = classification,
    Named("score_predictions_stored") = store_score_predictions
  );
  CharacterVector metric_name(length_ncomp);
  NumericVector metric_value(length_ncomp);
  IntegerVector metric_index(length_ncomp);
  for (int s = 0; s < length_ncomp; ++s) {
    metric_index[s] = s + 1;
    if (classification) {
      metric_name[s] = "accuracy";
      metric_value[s] = (metric_total(s) > 0.0) ?
        (metric_correct(s) / metric_total(s)) :
        NA_REAL;
    } else if (metric_id == 2 || metric_id == 3) {
      metric_name[s] = (metric_id == 2) ? "r2" : "q2";
      metric_value[s] = (std::isfinite(metric_tss) && metric_tss > 0.0) ?
        (1.0 - metric_sse(s) / metric_tss) :
        NA_REAL;
    } else {
      metric_name[s] = "rmsd";
      metric_value[s] = (metric_count(s) > 0.0) ?
        std::sqrt(metric_sse(s) / metric_count(s)) :
        NA_REAL;
    }
  }
  out["metrics"] = DataFrame::create(
    Named("ncomp_index") = metric_index,
    Named("metric_name") = metric_name,
    Named("metric_value") = metric_value,
    Named("stringsAsFactors") = false
  );
  if (store_score_predictions) {
    out["Ypred"] = Ypred;
  }
  if (store_class_predictions) {
    out["class_pred"] = class_pred;
  }
  return out;
}

// [[Rcpp::export]]
List pls_cv_predict_compiled(
  SEXP Xdata,
  SEXP Ydata,
  arma::ivec constrain,
  arma::ivec ncomp,
  int scaling,
  int kfold,
  int method,
  int backend,
  int svd_method,
  int rsvd_oversample,
  int rsvd_power,
  double svds_tol,
  int seed,
  bool classification,
  int n_response,
  bool xprod,
  int opls_north,
  bool return_scores,
  arma::mat class_codes,
  int classifier,
  double lda_ridge,
  bool store_predictions,
  int metric_id
) {
  if (!Rf_isMatrix(Xdata) || !Rf_isNumeric(Xdata) ||
      !Rf_isMatrix(Ydata) || !Rf_isNumeric(Ydata)) {
    stop("Xdata and Ydata must be numeric matrices");
  }
  Rcpp::Shield<SEXP> Xnumeric(Rf_coerceVector(Xdata, REALSXP));
  Rcpp::Shield<SEXP> Ynumeric(Rf_coerceVector(Ydata, REALSXP));
  const arma::mat Xview = numeric_matrix_view(Xnumeric, "Xdata");
  const arma::mat Yview = numeric_matrix_view(Ynumeric, "Ydata");
  return pls_cv_predict_compiled_impl(
    Xview, Yview, std::move(constrain), std::move(ncomp), scaling,
    kfold, method, backend, svd_method, rsvd_oversample, rsvd_power,
    svds_tol, seed, classification, n_response, xprod, opls_north,
    return_scores, std::move(class_codes), classifier, lda_ridge,
    store_predictions, metric_id
  );
}
