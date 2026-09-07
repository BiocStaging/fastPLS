#include "svd_cuda_rsvd.h"

#ifdef FASTPLS_HAS_CUDA

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <algorithm>

#include <cstdlib>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fastpls_svd {
namespace {

Mat gaussian_matrix_local(arma::uword n_rows, arma::uword n_cols, unsigned int seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> norm(0.0, 1.0);

  Mat out(n_rows, n_cols);
  double* ptr = out.memptr();
  const arma::uword n_elem = out.n_elem;
  for (arma::uword i = 0; i < n_elem; ++i) {
    ptr[i] = norm(rng);
  }

  return out;
}

SVDResult finalize_from_q_bsmall(const arma::mat& Q, const arma::mat& B, int k, bool left_only) {
  SVDResult out;
  const arma::uword max_rank = std::min(B.n_rows, B.n_cols);
  const arma::uword rank = std::min<arma::uword>(max_rank, static_cast<arma::uword>(std::max(k, 1)));

  if (B.n_rows <= B.n_cols) {
    arma::vec evals;
    arma::mat eigvec;
    if (arma::eig_sym(evals, eigvec, B * B.t())) {
      arma::uvec order = arma::sort_index(evals, "descend");
      arma::vec evals_desc = evals.elem(order);
      arma::mat Uhat_desc = eigvec.cols(order);

      const double tol = std::numeric_limits<double>::epsilon() *
        static_cast<double>(std::max(B.n_rows, B.n_cols)) *
        (evals_desc.n_elem > 0 ? std::max(evals_desc(0), 1.0) : 1.0);

      arma::uword usable = 0;
      while (usable < evals_desc.n_elem && evals_desc(usable) > tol) {
        ++usable;
      }
      usable = std::min<arma::uword>(usable, rank);

      if (usable > 0) {
        arma::vec s = arma::sqrt(arma::clamp(evals_desc.subvec(0, usable - 1), 0.0, arma::datum::inf));
        arma::mat Uhat = Uhat_desc.cols(0, usable - 1);
        out.U = Q * Uhat;
        out.s = s;

        if (!left_only) {
          arma::mat Vt = Uhat.t() * B;
          Vt.each_col() /= s;
          out.Vt = Vt;
        }
        return out;
      }
    }
  }

  arma::mat Uhat;
  arma::vec s;
  arma::mat V;
  if (left_only) {
    arma::svd_econ(Uhat, s, V, B, "left");
    arma::mat U = Q * Uhat;
    out.U = U.cols(0, rank - 1);
    out.s = s.subvec(0, rank - 1);
    out.Vt.reset();
    return out;
  }

  arma::svd_econ(Uhat, s, V, B, "both");
  arma::mat U = Q * Uhat;
  out.U = U.cols(0, rank - 1);
  out.s = s.subvec(0, rank - 1);
  out.Vt = V.cols(0, rank - 1).t();
  return out;
}

arma::mat plssvd_latent_coeff_from_gram(const arma::mat& G_full,
                                        const arma::vec& singular_values,
                                        int mc) {
  arma::mat G = G_full.submat(0, 0, mc - 1, mc - 1);
  arma::mat D(mc, mc, arma::fill::zeros);
  D.diag() = singular_values.subvec(0, mc - 1);

  arma::mat C;
  bool solved = arma::solve(C, G, D, arma::solve_opts::likely_sympd);
  if (!solved) {
    solved = arma::solve(C, G, D);
  }
  if (!solved) {
    C = arma::pinv(G) * D;
  }
  return C;
}

void throw_cuda_error(cudaError_t code, const char* where) {
  throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(code));
}

void check_cuda(cudaError_t code, const char* where) {
  if (code != cudaSuccess) {
    throw_cuda_error(code, where);
  }
}

void check_cublas(cublasStatus_t code, const char* where) {
  if (code != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(where) + ": cublas call failed");
  }
}

void check_curand(curandStatus_t code, const char* where) {
  if (code != CURAND_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(where) + ": curand call failed");
  }
}

void reset_curand_seed(curandGenerator_t generator, unsigned int seed) {
  check_curand(
    curandSetPseudoRandomGeneratorSeed(
      generator,
      static_cast<unsigned long long>(seed)
    ),
    "curandSetPseudoRandomGeneratorSeed"
  );
  check_curand(
    curandSetGeneratorOffset(generator, 0ULL),
    "curandSetGeneratorOffset"
  );
}

void check_cusolver(cusolverStatus_t code, const char* where) {
  if (code != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(where) + ": cusolver call failed");
  }
}

#ifdef FASTPLS_HAS_CUDA_KERNELS
extern "C" {
void fastpls_cuda_lda_means(double* means,
                            const double* counts,
                            int kmax,
                            int n_classes,
                            cudaStream_t stream);
void fastpls_cuda_lda_label_sums(const double* T,
                                 const int* y,
                                 int n,
                                 int kmax,
                                 int n_classes,
                                 double* sums,
                                 cudaStream_t stream);
void fastpls_cuda_lda_subtract_offsets(double* T,
                                       const double* offsets,
                                       int n,
                                       int kmax,
                                       cudaStream_t stream);
void fastpls_cuda_lda_pooled(double* pooled,
                             const double* means,
                             const double* counts,
                             int n,
                             int kmax,
                             int n_classes,
                             cudaStream_t stream);
void fastpls_cuda_lda_copy_cov(const double* pooled,
                               double* cov,
                               int kmax,
                               int kk,
                               cudaStream_t stream);
void fastpls_cuda_lda_add_ridge(double* cov,
                                int kk,
                                double ridge,
                                double* lambda_out,
                                cudaStream_t stream);
void fastpls_cuda_lda_means_to_rhs(const double* means,
                                   double* rhs,
                                   int kmax,
                                   int kk,
                                   int n_classes,
                                   cudaStream_t stream);
void fastpls_cuda_lda_finalize_linear(const double* rhs,
                                      const double* means,
                                      const double* counts,
                                      double* linear,
                                      double* constants,
                                      int n,
                                      int kmax,
                                      int kk,
                                      int n_classes,
                                      cudaStream_t stream);
void fastpls_cuda_lda_score_argmax(double* scores,
                                   const double* constants,
                                   int* pred,
                                   int n,
                                   int n_classes,
                                   cudaStream_t stream);
void fastpls_cuda_lda_means_float(float* means,
                                  const float* counts,
                                  int kmax,
                                  int n_classes,
                                  cudaStream_t stream);
void fastpls_cuda_lda_label_sums_float(const float* T,
                                       const int* y,
                                       int n,
                                       int kmax,
                                       int n_classes,
                                       float* sums,
                                       cudaStream_t stream);
void fastpls_cuda_lda_pooled_float(float* pooled,
                                   const float* means,
                                   const float* counts,
                                   int n,
                                   int kmax,
                                   int n_classes,
                                   cudaStream_t stream);
void fastpls_cuda_lda_copy_cov_float(const float* pooled,
                                     float* cov,
                                     int kmax,
                                     int kk,
                                     cudaStream_t stream);
void fastpls_cuda_lda_add_ridge_float(float* cov,
                                      int kk,
                                      float rho,
                                      float* lambda_out,
                                      cudaStream_t stream);
void fastpls_cuda_lda_means_to_rhs_float(const float* means,
                                         float* rhs,
                                         int kmax,
                                         int kk,
                                         int n_classes,
                                         cudaStream_t stream);
void fastpls_cuda_lda_finalize_linear_float(const float* rhs,
                                            const float* means,
                                            const float* counts,
                                            float* linear,
                                            float* constants,
                                            int n,
                                            int kmax,
                                            int kk,
                                            int n_classes,
                                            cudaStream_t stream);
void fastpls_cuda_lda_score_argmax_float(float* scores,
                                         const float* constants,
                                         int* pred,
                                         int n,
                                         int n_classes,
                                         cudaStream_t stream);
}

void check_kernel_launch(const char* where) {
  check_cuda(cudaPeekAtLastError(), where);
}
#endif

arma::mat inv_sqrt_psd(const arma::mat& G) {
  arma::vec evals;
  arma::mat evecs;
  if (!arma::eig_sym(evals, evecs, G)) {
    throw std::runtime_error("eig_sym failed in inv_sqrt_psd");
  }
  if (evals.n_elem < 1) {
    throw std::runtime_error("empty eigenspectrum in inv_sqrt_psd");
  }
  const double max_eval = std::max(evals.max(), 0.0);
  const double tol = std::max(max_eval, 1.0) * 1e-10;
  arma::vec scale = arma::zeros<arma::vec>(evals.n_elem);
  for (arma::uword i = 0; i < evals.n_elem; ++i) {
    if (evals(i) > tol) {
      scale(i) = 1.0 / std::sqrt(evals(i));
    }
  }
  return evecs * arma::diagmat(scale) * evecs.t();
}

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

double rq_value_local(const arma::mat& yData, const arma::mat& yPred) {
  double tss = 0.0;
  double press = 0.0;
  for (arma::uword i = 0; i < yData.n_cols; ++i) {
    const double my = arma::mean(yData.col(i));
    for (arma::uword j = 0; j < yData.n_rows; ++j) {
      const double diff = yData(j, i) - yPred(j, i);
      const double centered = yData(j, i) - my;
      press += diff * diff;
      tss += centered * centered;
    }
  }
  if (!std::isfinite(tss) || tss <= 0.0) {
    return NA_REAL;
  }
  return 1.0 - (press / tss);
}

class CudaRSVDWorkspace {
 public:
  CudaRSVDWorkspace() = default;

  ~CudaRSVDWorkspace() {
    release();
  }

  void reset() {
    release();
  }

  void ensure_capacity(int m, int n, int l) {
    ensure_stream();
    if (!handle_ready_) {
      check_cublas(cublasCreate(&handle_), "cublasCreate");
      if (stream_ready_) {
        check_cublas(cublasSetStream(handle_, stream_), "cublasSetStream");
      }
      handle_ready_ = true;
    }
    if (!solver_ready_) {
      check_cusolver(cusolverDnCreate(&solver_), "cusolverDnCreate");
      if (stream_ready_) {
        check_cusolver(cusolverDnSetStream(solver_, stream_), "cusolverDnSetStream");
      }
      solver_ready_ = true;
    }

    ensure_buffer(dA_, bytes_for(m, n), bytes_A_, "cudaMalloc(dA)");
    ensure_buffer(dOmega_, bytes_for(n, l), bytes_Omega_, "cudaMalloc(dOmega)");
    ensure_buffer(dY_, bytes_for_padded_random(m, l), bytes_Y_, "cudaMalloc(dY)");
    ensure_buffer(dZ_, bytes_for(n, l), bytes_Z_, "cudaMalloc(dZ)");
    ensure_buffer(dQ_, bytes_for(m, l), bytes_Q_, "cudaMalloc(dQ)");
    ensure_buffer(dBsmall_, bytes_for(l, n), bytes_Bsmall_, "cudaMalloc(dBsmall)");
    ensure_buffer(dGram_, bytes_for(l, l), bytes_Gram_, "cudaMalloc(dGram)");
    ensure_buffer(dTau_, bytes_for(l, 1), bytes_Tau_, "cudaMalloc(dTau)");
    ensure_buffer(dEvals_, bytes_for(l, 1), bytes_Evals_, "cudaMalloc(dEvals)");
    ensure_int_buffer(dInfo_, sizeof(int), bytes_Info_, "cudaMalloc(dInfo)");

    if (!rng_ready_) {
      check_curand(curandCreateGenerator(&rng_, CURAND_RNG_PSEUDO_DEFAULT), "curandCreateGenerator");
      if (stream_ready_) {
        check_curand(curandSetStream(rng_, stream_), "curandSetStream");
      }
      rng_ready_ = true;
    }
  }

  void ensure_matrix_free_capacity(int m, int n, int l) {
    ensure_stream();
    if (!handle_ready_) {
      check_cublas(cublasCreate(&handle_), "cublasCreate");
      if (stream_ready_) {
        check_cublas(cublasSetStream(handle_, stream_), "cublasSetStream");
      }
      handle_ready_ = true;
    }
    if (!solver_ready_) {
      check_cusolver(cusolverDnCreate(&solver_), "cusolverDnCreate");
      if (stream_ready_) {
        check_cusolver(cusolverDnSetStream(solver_, stream_), "cusolverDnSetStream");
      }
      solver_ready_ = true;
    }

    ensure_buffer(dOmega_, bytes_for_padded_random(n, l), bytes_Omega_, "cudaMalloc(dOmega)");
    ensure_buffer(dY_, bytes_for_padded_random(m, l), bytes_Y_, "cudaMalloc(dY)");
    ensure_buffer(dZ_, bytes_for(n, l), bytes_Z_, "cudaMalloc(dZ)");
    ensure_buffer(dQ_, bytes_for(m, l), bytes_Q_, "cudaMalloc(dQ)");
    ensure_buffer(dBsmall_, bytes_for(l, n), bytes_Bsmall_, "cudaMalloc(dBsmall)");
    ensure_buffer(dGram_, bytes_for(l, l), bytes_Gram_, "cudaMalloc(dGram)");
    ensure_buffer(dTau_, bytes_for(l, 1), bytes_Tau_, "cudaMalloc(dTau)");
    ensure_buffer(dEvals_, bytes_for(l, 1), bytes_Evals_, "cudaMalloc(dEvals)");
    ensure_int_buffer(dInfo_, sizeof(int), bytes_Info_, "cudaMalloc(dInfo)");

    if (!rng_ready_) {
      check_curand(curandCreateGenerator(&rng_, CURAND_RNG_PSEUDO_DEFAULT), "curandCreateGenerator");
      if (stream_ready_) {
        check_curand(curandSetStream(rng_, stream_), "curandSetStream");
      }
      rng_ready_ = true;
    }
  }

  void set_pls_training_matrices(
    const double* hX,
    int n,
    int p,
    const double* hY,
    int m,
    bool fit,
    bool form_crossprod = true
  ) {
    if (form_crossprod) {
      ensure_capacity(p, m, 1);
    } else {
      ensure_matrix_free_capacity(p, m, 1);
    }
    ensure_buffer(dX_, bytes_for(n, p), bytes_X_, "cudaMalloc(dX)");
    ensure_buffer(dYtrain_, bytes_for(n, m), bytes_Ytrain_, "cudaMalloc(dYtrain)");
    check_cuda(cudaMemcpy(dX_, hX, bytes_for(n, p), cudaMemcpyHostToDevice), "cudaMemcpy(Xtrain)");
    check_cuda(cudaMemcpy(dYtrain_, hY, bytes_for(n, m), cudaMemcpyHostToDevice), "cudaMemcpy(Ytrain)");
    ensure_buffer(dTvec_, bytes_for(n, 1), bytes_Tvec_, "cudaMalloc(dTvec)");
    ensure_buffer(dPvec_, bytes_for(p, 1), bytes_Pvec_, "cudaMalloc(dPvec)");
    ensure_buffer(dQvec_, bytes_for(m, 1), bytes_Qvec_, "cudaMalloc(dQvec)");
    ensure_buffer(dRvec_, bytes_for(p, 1), bytes_Rvec_, "cudaMalloc(dRvec)");
    if (fit) {
      ensure_buffer(dYfit_, bytes_for(n, m), bytes_Yfit_, "cudaMalloc(dYfit)");
      check_cuda(cudaMemset(dYfit_, 0, bytes_for(n, m)), "cudaMemset(dYfit)");
    } else {
      release_pls_fitted_response_buffer();
    }
    if (form_crossprod) {
      crossprod_training_xy(n, p, m);
    }
  }

  void release_pls_ytrain_buffer() {
    if (dYtrain_ != nullptr) {
      cudaFree(dYtrain_);
      dYtrain_ = nullptr;
      bytes_Ytrain_ = 0;
    }
  }

  void release_pls_coefficient_buffer() {
    if (dBcur_ != nullptr) {
      cudaFree(dBcur_);
      dBcur_ = nullptr;
      bytes_Bcur_ = 0;
    }
  }

  void release_pls_fitted_response_buffer() {
    if (dYfit_ != nullptr) {
      cudaFree(dYfit_);
      dYfit_ = nullptr;
      bytes_Yfit_ = 0;
    }
  }

  void simpls_fast_begin_device_loop(
    int n,
    int p,
    int m,
    int max_ncomp,
    bool fit,
    bool store_B
  ) {
    ensure_buffer(dRRmat_, bytes_for(p, max_ncomp), bytes_RRmat_, "cudaMalloc(dRRmat)");
    ensure_buffer(dQQmat_, bytes_for(m, max_ncomp), bytes_QQmat_, "cudaMalloc(dQQmat)");
    ensure_buffer(dVVmat_, bytes_for(p, max_ncomp), bytes_VVmat_, "cudaMalloc(dVVmat)");
    ensure_buffer(dCoeff_, bytes_for(std::max(max_ncomp, 1), 1), bytes_Coeff_, "cudaMalloc(dCoeff)");
    check_cuda(cudaMemset(dRRmat_, 0, bytes_for(p, max_ncomp)), "cudaMemset(dRRmat)");
    check_cuda(cudaMemset(dQQmat_, 0, bytes_for(m, max_ncomp)), "cudaMemset(dQQmat)");
    check_cuda(cudaMemset(dVVmat_, 0, bytes_for(p, max_ncomp)), "cudaMemset(dVVmat)");
    if (store_B) {
      ensure_buffer(dBcur_, bytes_for(p, m), bytes_Bcur_, "cudaMalloc(dBcur)");
      check_cuda(cudaMemset(dBcur_, 0, bytes_for(p, m)), "cudaMemset(dBcur)");
    } else {
      // Compact prediction uses RR and QQ; no dense coefficient accumulator.
      release_pls_coefficient_buffer();
    }
    if (fit) {
      ensure_buffer(dYfit_, bytes_for(n, m), bytes_Yfit_, "cudaMalloc(dYfit)");
      check_cuda(cudaMemset(dYfit_, 0, bytes_for(n, m)), "cudaMemset(dYfit)");
    } else {
      release_pls_fitted_response_buffer();
    }
  }

  void simpls_fast_refresh_block_resident(
    int p,
    int m,
    int l,
    int k,
    unsigned int seed,
    int power_iters,
    double* hSvals
  ) {
    ensure_capacity(p, m, l);
    reset_curand_seed(rng_, seed);
    check_curand(
      curandGenerateNormalDouble(rng_, dY_, padded_random_elems(p, l), 0.0, 1.0),
      "curandGenerateNormalDouble(Y0)"
    );
    const double alpha = 1.0;
    const double beta = 0.0;

    for (int i = 0; i < power_iters; ++i) {
      orthonormalize_refresh_inplace(p, l);
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, m, l, p, &alpha, dA_, p, dY_, p, &beta, dZ_, m),
        "cublasDgemm(S^T*Y)"
      );
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, p, l, m, &alpha, dA_, p, dZ_, m, &beta, dY_, p),
        "cublasDgemm(S*Z)"
      );
    }

    orthonormalize_refresh_inplace(p, l);
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, l, m, p, &alpha, dY_, p, dA_, p, &beta, dBsmall_, l),
      "cublasDgemm(Bsmall=Q^T*S)"
    );
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T, l, l, m, &alpha, dBsmall_, l, dBsmall_, l, &beta, dGram_, l),
      "cublasDgemm(Bsmall*Bsmall^T)"
    );

    finalize_left_block_from_gram_inplace(l, k);
    copy_top_singular_values(l, k, hSvals);

    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, p, k, l, &alpha, dY_, p, dOmega_, l, &beta, dQ_, p),
      "cublasDgemm(Ublock=Q*Uhat)"
    );
  }

  void simpls_fast_refresh_block_implicit_resident(
    int n,
    int p,
    int m,
    int l,
    int k,
    int prev_v_cols,
    unsigned int seed,
    int power_iters,
    double* hSvals
  ) {
    ensure_matrix_free_capacity(p, m, l);
    reset_curand_seed(rng_, seed);
    check_curand(
      curandGenerateNormalDouble(rng_, dY_, padded_random_elems(p, l), 0.0, 1.0),
      "curandGenerateNormalDouble(Y0_implicit)"
    );
    for (int i = 0; i < power_iters; ++i) {
      subtract_left_projection_inplace(
        p, l, prev_v_cols, dY_, "implicit_SIMPLS_project_before_power_qr"
      );
      orthonormalize_refresh_inplace(p, l);
      implicit_at_times_mat_deflated(
        n,
        p,
        m,
        l,
        dY_,
        dZ_,
        prev_v_cols,
        "implicit_SIMPLS_SAT_times_Y"
      );
      implicit_a_times_mat_deflated(
        n,
        p,
        m,
        l,
        dZ_,
        dY_,
        prev_v_cols,
        "implicit_SIMPLS_S_times_Z"
      );
    }

    subtract_left_projection_inplace(
      p,
      l,
      prev_v_cols,
      dY_,
      "implicit_SIMPLS_project_before_qr"
    );
    orthonormalize_refresh_inplace(p, l);
    implicit_bsmall_from_deflated_basis(
      n,
      p,
      m,
      l,
      dY_,
      dBsmall_,
      prev_v_cols,
      "implicit_SIMPLS_Bsmall=QTA"
    );
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T, l, l, m, &one_, dBsmall_, l, dBsmall_, l, &zero_, dGram_, l),
      "cublasDgemm(implicit_Bsmall*Bsmall^T)"
    );

    finalize_left_block_from_gram_inplace(l, k);
    copy_top_singular_values(l, k, hSvals);

    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, p, k, l, &one_, dY_, p, dOmega_, l, &zero_, dQ_, p),
      "cublasDgemm(implicit_Ublock=Q*Uhat)"
    );
  }

  bool simpls_fast_append_component_from_block(
    int n,
    int p,
    int m,
    int a_idx,
    int col_idx,
    int prev_v_cols,
    bool reorth_v,
    bool fit,
    bool materialized_crossprod
  ) {
    ensure_buffer(dTvec_, bytes_for(n, 1), bytes_Tvec_, "cudaMalloc(dTvec)");
    ensure_buffer(dPvec_, bytes_for(p, 1), bytes_Pvec_, "cudaMalloc(dPvec)");
    ensure_buffer(dQvec_, bytes_for(m, 1), bytes_Qvec_, "cudaMalloc(dQvec)");
    ensure_buffer(dRvec_, bytes_for(p, 1), bytes_Rvec_, "cudaMalloc(dRvec)");
    ensure_buffer(dCoeff_, bytes_for(std::max(prev_v_cols, 1), 1), bytes_Coeff_, "cudaMalloc(dCoeff)");

    const double alpha = 1.0;
    const double beta = 0.0;
    const double minus_one = -1.0;

    check_cublas(
      cublasDcopy(handle_, p, dQ_ + static_cast<size_t>(col_idx) * static_cast<size_t>(p), 1, dRvec_, 1),
      "cublasDcopy(Ublock_col->r)"
    );
    x_times_vec(n, p, dRvec_, dTvec_, "cublasGemm(t=X*r)");

    double tnorm = 0.0;
    check_cublas(cublasDnrm2(handle_, n, dTvec_, 1, &tnorm), "cublasDnrm2(t)");
    if (!std::isfinite(tnorm) || tnorm <= 0.0) {
      return false;
    }

    const double inv_tnorm = 1.0 / tnorm;
    check_cublas(cublasDscal(handle_, n, &inv_tnorm, dTvec_, 1), "cublasDscal(t)");
    check_cublas(cublasDscal(handle_, p, &inv_tnorm, dRvec_, 1), "cublasDscal(r)");

    xt_times_vec(n, p, dTvec_, dPvec_, "cublasGemm(p=X^T*t)");
    yt_times_vec(n, m, dTvec_, dQvec_, "cublasGemm(q=Y^T*t)");

    if (prev_v_cols > 0) {
      check_cublas(
        cublasDgemm(
          handle_,
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          prev_v_cols,
          1,
          p,
          &alpha,
          dVVmat_,
          p,
          dPvec_,
          p,
          &beta,
          dCoeff_,
          prev_v_cols
        ),
        "cublasDgemm(V^T*p)"
      );
      check_cublas(
        cublasDgemm(
          handle_,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          p,
          1,
          prev_v_cols,
          &minus_one,
          dVVmat_,
          p,
          dCoeff_,
          prev_v_cols,
          &alpha,
          dPvec_,
          p
        ),
        "cublasDgemm(p-=V*coeff)"
      );
      if (reorth_v) {
        check_cublas(
          cublasDgemm(
            handle_,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            prev_v_cols,
            1,
            p,
            &alpha,
            dVVmat_,
            p,
            dPvec_,
            p,
            &beta,
            dCoeff_,
            prev_v_cols
          ),
          "cublasDgemm(V^T*v)"
        );
        check_cublas(
          cublasDgemm(
            handle_,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            p,
            1,
            prev_v_cols,
            &minus_one,
            dVVmat_,
            p,
            dCoeff_,
            prev_v_cols,
            &alpha,
            dPvec_,
            p
          ),
          "cublasDgemm(v-=V*coeff)"
        );
      }
    }

    double vnorm = 0.0;
    check_cublas(cublasDnrm2(handle_, p, dPvec_, 1, &vnorm), "cublasDnrm2(v)");
    if (!std::isfinite(vnorm) || vnorm <= 0.0) {
      return false;
    }
    const double inv_vnorm = 1.0 / vnorm;
    check_cublas(cublasDscal(handle_, p, &inv_vnorm, dPvec_, 1), "cublasDscal(v)");

    if (materialized_crossprod) {
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, m, 1, p, &alpha, dA_, p, dPvec_, p, &beta, dZ_, m),
        "cublasDgemm(v^T*S)"
      );
      check_cublas(
        cublasDger(handle_, p, m, &minus_one, dPvec_, 1, dZ_, 1, dA_, p),
        "cublasDger(deflate)"
      );
    }

    check_cublas(
      cublasDcopy(handle_, p, dRvec_, 1, dRRmat_ + static_cast<size_t>(a_idx) * static_cast<size_t>(p), 1),
      "cublasDcopy(r->RR)"
    );
    check_cublas(
      cublasDcopy(handle_, m, dQvec_, 1, dQQmat_ + static_cast<size_t>(a_idx) * static_cast<size_t>(m), 1),
      "cublasDcopy(q->QQ)"
    );
    check_cublas(
      cublasDcopy(handle_, p, dPvec_, 1, dVVmat_ + static_cast<size_t>(a_idx) * static_cast<size_t>(p), 1),
      "cublasDcopy(v->VV)"
    );
    if (dBcur_ != nullptr) {
      check_cublas(
        cublasDger(handle_, p, m, &alpha, dRvec_, 1, dQvec_, 1, dBcur_, p),
        "cublasDger(Bcur+=rq^T)"
      );
    }

    if (fit) {
      check_cublas(
        cublasDger(handle_, n, m, &alpha, dTvec_, 1, dQvec_, 1, dYfit_, n),
        "cublasDger(Yfit+=tq^T)"
      );
    }

    return true;
  }

  void simpls_fast_copy_rr(double* hRR, int p, int max_ncomp) {
    check_cuda(cudaMemcpy(hRR, dRRmat_, bytes_for(p, max_ncomp), cudaMemcpyDeviceToHost), "cudaMemcpy(RR)");
  }

  void simpls_fast_copy_qq(double* hQQ, int m, int max_ncomp) {
    check_cuda(cudaMemcpy(hQQ, dQQmat_, bytes_for(m, max_ncomp), cudaMemcpyDeviceToHost), "cudaMemcpy(QQ)");
  }

  void simpls_fast_copy_bcur(double* hB, int p, int m) {
    if (dBcur_ == nullptr) {
      throw std::runtime_error("Dense SIMPLS coefficients were not requested");
    }
    check_cuda(cudaMemcpy(hB, dBcur_, bytes_for(p, m), cudaMemcpyDeviceToHost), "cudaMemcpy(Bcur)");
  }

  void simpls_fast_copy_yfit(double* hYfit, int n, int m) {
    check_cuda(cudaMemcpy(hYfit, dYfit_, bytes_for(n, m), cudaMemcpyDeviceToHost), "cudaMemcpy(Yfit)");
  }

  void simpls_fast_component_stats(
    const double* hR,
    int n,
    int p,
    int m,
    double* hT,
    double* hP,
    double* hQ,
    double* hTnorm
  ) {
    if (dX_ == nullptr || dYtrain_ == nullptr) {
      throw std::runtime_error("training matrices are not initialized for GPU workspace");
    }
    ensure_buffer(dTvec_, bytes_for(n, 1), bytes_Tvec_, "cudaMalloc(dTvec)");
    ensure_buffer(dPvec_, bytes_for(p, 1), bytes_Pvec_, "cudaMalloc(dPvec)");
    ensure_buffer(dQvec_, bytes_for(m, 1), bytes_Qvec_, "cudaMalloc(dQvec)");
    ensure_buffer(dRvec_, bytes_for(p, 1), bytes_Rvec_, "cudaMalloc(dRvec)");

    check_cuda(cudaMemcpy(dRvec_, hR, bytes_for(p, 1), cudaMemcpyHostToDevice), "cudaMemcpy(rr)");

    const double alpha = 1.0;
    const double beta = 0.0;
    x_times_vec(n, p, dRvec_, dTvec_, "cublasGemm(t=X*r)");

    double tnorm = 0.0;
    check_cublas(cublasDnrm2(handle_, n, dTvec_, 1, &tnorm), "cublasDnrm2(t)");
    if (!std::isfinite(tnorm) || tnorm <= 0.0) {
      throw std::runtime_error("invalid tnorm in cuda_simpls_fast_component_stats");
    }

    const double inv_tnorm = 1.0 / tnorm;
    check_cublas(cublasDscal(handle_, n, &inv_tnorm, dTvec_, 1), "cublasDscal(t)");
    check_cublas(cublasDscal(handle_, p, &inv_tnorm, dRvec_, 1), "cublasDscal(r)");

    xt_times_vec(n, p, dTvec_, dPvec_, "cublasGemm(p=X^T*t)");
    yt_times_vec(n, m, dTvec_, dQvec_, "cublasGemm(q=Y^T*t)");

    check_cuda(cudaMemcpy(hT, dTvec_, bytes_for(n, 1), cudaMemcpyDeviceToHost), "cudaMemcpy(t)");
    check_cuda(cudaMemcpy(hP, dPvec_, bytes_for(p, 1), cudaMemcpyDeviceToHost), "cudaMemcpy(p)");
    check_cuda(cudaMemcpy(hQ, dQvec_, bytes_for(m, 1), cudaMemcpyDeviceToHost), "cudaMemcpy(q)");
    *hTnorm = tnorm;
  }

  void simpls_fast_rank1_fit_update(
    const double* hT,
    int n,
    const double* hQ,
    int m,
    double* hDelta
  ) {
    ensure_buffer(dYfit_, bytes_for(n, m), bytes_Yfit_, "cudaMalloc(dYfit)");
    ensure_buffer(dTvec_, bytes_for(n, 1), bytes_Tvec_, "cudaMalloc(dTvec)");
    ensure_buffer(dQvec_, bytes_for(m, 1), bytes_Qvec_, "cudaMalloc(dQvec)");
    check_cuda(cudaMemcpy(dTvec_, hT, bytes_for(n, 1), cudaMemcpyHostToDevice), "cudaMemcpy(fit_t)");
    check_cuda(cudaMemcpy(dQvec_, hQ, bytes_for(m, 1), cudaMemcpyHostToDevice), "cudaMemcpy(fit_q)");
    const double alpha = 1.0;
    check_cublas(
      cublasDger(handle_, n, m, &alpha, dTvec_, 1, dQvec_, 1, dYfit_, n),
      "cublasDger(Yfit+=tq^T)"
    );
    check_cuda(cudaMemcpy(hDelta, dYfit_, bytes_for(n, m), cudaMemcpyDeviceToHost), "cudaMemcpy(Yfit_cur)");
  }

  void solve_spd_system_inplace(int k, int nrhs) {
    int lwork_potrf = 0;
    check_cusolver(
      cusolverDnDpotrf_bufferSize(
        solver_,
        CUBLAS_FILL_MODE_LOWER,
        k,
        dGram_,
        k,
        &lwork_potrf
      ),
      "cusolverDnDpotrf_bufferSize"
    );
    ensure_buffer(
      dWorkEig_,
      sizeof(double) * static_cast<size_t>(lwork_potrf),
      bytes_WorkEig_,
      "cudaMalloc(dWorkEig)"
    );
    check_cusolver(
      cusolverDnDpotrf(
        solver_,
        CUBLAS_FILL_MODE_LOWER,
        k,
        dGram_,
        k,
        dWorkEig_,
        lwork_potrf,
        dInfo_
      ),
      "cusolverDnDpotrf"
    );
    check_cuda(cudaMemcpy(&hInfo_, dInfo_, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(info_potrf)");
    if (hInfo_ != 0) {
      throw std::runtime_error("cusolverDnDpotrf returned non-zero info");
    }

    check_cublas(
      cublasDtrsm(
        handle_,
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT,
        k,
        nrhs,
        &one_,
        dGram_,
        k,
        dBsmall_,
        k
      ),
      "cublasDtrsm(lower)"
    );
    check_cublas(
      cublasDtrsm(
        handle_,
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_T,
        CUBLAS_DIAG_NON_UNIT,
        k,
        nrhs,
        &one_,
        dGram_,
        k,
        dBsmall_,
        k
      ),
      "cublasDtrsm(lower^T)"
    );
  }

  PLSSVDGPUResult plssvd_fit(
    const arma::mat& Xtrain,
    const arma::mat& Ytrain,
    const arma::ivec& ncomp,
    bool fit,
    const SVDOptions& opt,
    bool return_B
  ) {
    const int n = static_cast<int>(Xtrain.n_rows);
    const int p = static_cast<int>(Xtrain.n_cols);
    const int m = static_cast<int>(Ytrain.n_cols);
    const int max_rank = std::min(p, m);
    const int target = std::min(max_rank, std::max(static_cast<int>(ncomp.max()), 1));
    if (target < 1) {
      throw std::runtime_error("cuda_plssvd_fit target rank is < 1");
    }
    const int l = std::min(max_rank, target + std::max(opt.oversample, 0));

    ensure_capacity(p, m, l);
    set_pls_training_matrices(Xtrain.memptr(), n, p, Ytrain.memptr(), m, fit);
    release_pls_ytrain_buffer();
    ensure_buffer(dRRmat_, bytes_for(n, target), bytes_RRmat_, "cudaMalloc(dRRmat)");
    ensure_buffer(dQQmat_, bytes_for(target, target), bytes_QQmat_, "cudaMalloc(dQQmat)");
    if (return_B) {
      ensure_buffer(dBcur_, bytes_for(p, m), bytes_Bcur_, "cudaMalloc(dBcur)");
    } else {
      release_pls_coefficient_buffer();
    }
    if (fit) {
      ensure_buffer(dYfit_, bytes_for(n, m), bytes_Yfit_, "cudaMalloc(dYfit)");
    }

    reset_curand_seed(rng_, opt.seed);
    check_curand(
      curandGenerateNormalDouble(rng_, dOmega_, padded_random_elems(m, l), 0.0, 1.0),
      "curandGenerateNormalDouble(Omega)"
    );

    const double alpha = 1.0;
    const double beta = 0.0;
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, p, l, m, &alpha, dA_, p, dOmega_, m, &beta, dY_, p),
      "cublasDgemm(S*Omega)"
    );
    for (int i = 0; i < std::max(opt.power_iters, 0); ++i) {
      orthonormalize_qr_inplace(p, l);
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, m, l, p, &alpha, dA_, p, dY_, p, &beta, dZ_, m),
        "cublasDgemm(S^T*Y)"
      );
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, p, l, m, &alpha, dA_, p, dZ_, m, &beta, dY_, p),
        "cublasDgemm(S*Z)"
      );
    }

    orthonormalize_qr_inplace(p, l);
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, l, m, p, &alpha, dY_, p, dA_, p, &beta, dBsmall_, l),
      "cublasDgemm(Bsmall=Q^T*S)"
    );
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T, l, l, m, &alpha, dBsmall_, l, dBsmall_, l, &beta, dGram_, l),
      "cublasDgemm(Bsmall*Bsmall^T)"
    );
    finalize_left_block_from_gram_inplace(l, target);
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, p, target, l, &alpha, dY_, p, dOmega_, l, &beta, dQ_, p),
      "cublasDgemm(U=Q*Uhat)"
    );
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, target, m, l, &alpha, dOmega_, l, dBsmall_, l, &beta, dZ_, target),
      "cublasDgemm(RHS=Uhat^T*Bsmall)"
    );

    ensure_buffer(dRRmat_, bytes_for(n, target), bytes_RRmat_, "cudaMalloc(dRRmat)");
    x_times_mat(n, p, target, dQ_, dRRmat_, "cublasGemm(T=X*U)");
    ensure_buffer(dQQmat_, bytes_for(target, target), bytes_QQmat_, "cudaMalloc(dQQmat)");
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, target, target, n, &alpha, dRRmat_, n, dRRmat_, n, &beta, dQQmat_, target),
      "cublasDgemm(G=T^T*T)"
    );
    arma::mat G_full_host(static_cast<arma::uword>(target), static_cast<arma::uword>(target), arma::fill::zeros);
    check_cuda(
      cudaMemcpy(G_full_host.memptr(), dQQmat_, bytes_for(target, target), cudaMemcpyDeviceToHost),
      "cudaMemcpy(G_full_plssvd)"
    );

    PLSSVDGPUResult out;
    out.R.set_size(static_cast<arma::uword>(p), static_cast<arma::uword>(target));
    out.Ttrain.set_size(static_cast<arma::uword>(n), static_cast<arma::uword>(target));
    check_cuda(cudaMemcpy(out.R.memptr(), dQ_, bytes_for(p, target), cudaMemcpyDeviceToHost), "cudaMemcpy(R_plssvd)");
    check_cuda(cudaMemcpy(out.Ttrain.memptr(), dRRmat_, bytes_for(n, target), cudaMemcpyDeviceToHost), "cudaMemcpy(T_plssvd)");

    arma::vec s_host(target, arma::fill::zeros);
    copy_top_singular_values(l, target, s_host.memptr());
    arma::mat rhs_host(static_cast<arma::uword>(target), static_cast<arma::uword>(m), arma::fill::zeros);
    check_cuda(cudaMemcpy(rhs_host.memptr(), dZ_, bytes_for(target, m), cudaMemcpyDeviceToHost), "cudaMemcpy(RHS_plssvd)");
    out.Q.set_size(static_cast<arma::uword>(m), static_cast<arma::uword>(target));
    out.Q.zeros();
    for (int j = 0; j < target; ++j) {
      const double sj = s_host(static_cast<arma::uword>(j));
      if (std::isfinite(sj) && sj > 0.0) {
        out.Q.col(static_cast<arma::uword>(j)) = (rhs_host.row(static_cast<arma::uword>(j)) / sj).t();
      }
    }

    if (return_B) {
      out.B.set_size(static_cast<arma::uword>(p), static_cast<arma::uword>(m), ncomp.n_elem);
      out.B.zeros();
    }
    out.C_latent.set_size(
      static_cast<arma::uword>(target),
      static_cast<arma::uword>(target),
      ncomp.n_elem
    );
    out.C_latent.zeros();
    out.W_latent.set_size(
      static_cast<arma::uword>(target),
      static_cast<arma::uword>(m),
      ncomp.n_elem
    );
    out.W_latent.zeros();
    out.R2Y.set_size(ncomp.n_elem);
    out.R2Y.zeros();
    if (fit) {
      out.Yfit.set_size(static_cast<arma::uword>(n), static_cast<arma::uword>(m), ncomp.n_elem);
    }

    for (arma::uword a = 0; a < ncomp.n_elem; ++a) {
      const int requested_mc = static_cast<int>(ncomp(a));
      const int mc = std::min(std::max(requested_mc, 1), target);
      arma::mat c_host = plssvd_latent_coeff_from_gram(G_full_host, s_host, mc);
      out.C_latent.slice(a).submat(0, 0, mc - 1, mc - 1) = c_host;
      check_cuda(
        cudaMemcpy2D(
          dGram_,
          bytes_for(mc, 1),
          dQQmat_,
          bytes_for(target, 1),
          bytes_for(mc, 1),
          static_cast<size_t>(mc),
          cudaMemcpyDeviceToDevice
        ),
        "cudaMemcpy2D(G_a)"
      );
      check_cuda(
        cudaMemcpy2D(
          dBsmall_,
          bytes_for(mc, 1),
          dZ_,
          bytes_for(target, 1),
          bytes_for(mc, 1),
          static_cast<size_t>(m),
          cudaMemcpyDeviceToDevice
        ),
        "cudaMemcpy2D(RHS_a)"
      );
      solve_spd_system_inplace(mc, m);

      arma::mat w_host(static_cast<arma::uword>(mc), static_cast<arma::uword>(m), arma::fill::zeros);
      check_cuda(cudaMemcpy(w_host.memptr(), dBsmall_, bytes_for(mc, m), cudaMemcpyDeviceToHost), "cudaMemcpy(W_latent)");
      out.W_latent.slice(a).submat(0, 0, mc - 1, m - 1) = w_host;

      if (return_B) {
        check_cublas(
          cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, p, m, mc, &alpha, dQ_, p, dBsmall_, mc, &beta, dBcur_, p),
          "cublasDgemm(B=U*W)"
        );
        check_cuda(
          cudaMemcpy(out.B.slice(a).memptr(), dBcur_, bytes_for(p, m), cudaMemcpyDeviceToHost),
          "cudaMemcpy(B_slice)"
        );
      }

      if (fit) {
        check_cublas(
          cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, m, mc, &alpha, dRRmat_, n, dBsmall_, mc, &beta, dYfit_, n),
          "cublasDgemm(Yfit=T*W)"
        );
        check_cuda(
          cudaMemcpy(out.Yfit.slice(a).memptr(), dYfit_, bytes_for(n, m), cudaMemcpyDeviceToHost),
          "cudaMemcpy(Yfit_slice)"
        );
        out.R2Y(a) = rq_value_local(Ytrain, out.Yfit.slice(a));
      }
    }

    return out;
  }

  PLSSVDGPUResult plssvd_fit_implicit_xprod(
    const arma::mat& Xtrain,
    const arma::mat& Ytrain,
    const arma::ivec& ncomp,
    bool fit,
    const SVDOptions& opt,
    bool return_B
  ) {
    const int n = static_cast<int>(Xtrain.n_rows);
    const int p = static_cast<int>(Xtrain.n_cols);
    const int m = static_cast<int>(Ytrain.n_cols);
    const int max_rank = std::min(p, m);
    const int target = std::min(max_rank, std::max(static_cast<int>(ncomp.max()), 1));
    if (target < 1) {
      throw std::runtime_error("cuda_plssvd_fit_implicit_xprod target rank is < 1");
    }
    const int l = std::min(max_rank, target + std::max(opt.oversample, 0));

    ensure_matrix_free_capacity(p, m, l);
    set_pls_training_matrices(Xtrain.memptr(), n, p, Ytrain.memptr(), m, fit, false);
    ensure_matrix_free_capacity(p, m, l);

    reset_curand_seed(rng_, opt.seed);
    check_curand(
      curandGenerateNormalDouble(rng_, dOmega_, padded_random_elems(m, l), 0.0, 1.0),
      "curandGenerateNormalDouble(Omega)"
    );

    const double alpha = 1.0;
    const double beta = 0.0;

    implicit_a_times_mat(n, p, m, l, dOmega_, dY_, "implicit_A_times_Omega");
    for (int i = 0; i < std::max(opt.power_iters, 0); ++i) {
      orthonormalize_qr_inplace(p, l);
      implicit_at_times_mat(n, p, m, l, dY_, dZ_, "implicit_AT_times_Y");
      implicit_a_times_mat(n, p, m, l, dZ_, dY_, "implicit_A_times_Z");
    }

    orthonormalize_qr_inplace(p, l);
    implicit_bsmall_from_basis(n, p, m, l, dY_, dBsmall_, "implicit_Bsmall=QTA");
    release_pls_ytrain_buffer();

    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T, l, l, m, &alpha, dBsmall_, l, dBsmall_, l, &beta, dGram_, l),
      "cublasDgemm(Bsmall*Bsmall^T)"
    );
    finalize_left_block_from_gram_inplace(l, target);
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, p, target, l, &alpha, dY_, p, dOmega_, l, &beta, dQ_, p),
      "cublasDgemm(U=Q*Uhat)"
    );
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, target, m, l, &alpha, dOmega_, l, dBsmall_, l, &beta, dZ_, target),
      "cublasDgemm(RHS=Uhat^T*Bsmall)"
    );

    ensure_buffer(dRRmat_, bytes_for(n, target), bytes_RRmat_, "cudaMalloc(dRRmat)");
    x_times_mat(n, p, target, dQ_, dRRmat_, "cublasGemm(T=X*U)");
    ensure_buffer(dQQmat_, bytes_for(target, target), bytes_QQmat_, "cudaMalloc(dQQmat)");
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, target, target, n, &alpha, dRRmat_, n, dRRmat_, n, &beta, dQQmat_, target),
      "cublasDgemm(G=T^T*T)"
    );
    arma::mat G_full_host(static_cast<arma::uword>(target), static_cast<arma::uword>(target), arma::fill::zeros);
    check_cuda(
      cudaMemcpy(G_full_host.memptr(), dQQmat_, bytes_for(target, target), cudaMemcpyDeviceToHost),
      "cudaMemcpy(G_full_plssvd)"
    );

    if (return_B) {
      ensure_buffer(dBcur_, bytes_for(p, m), bytes_Bcur_, "cudaMalloc(dBcur)");
    } else {
      release_pls_coefficient_buffer();
    }
    if (fit) {
      ensure_buffer(dYfit_, bytes_for(n, m), bytes_Yfit_, "cudaMalloc(dYfit)");
    }

    PLSSVDGPUResult out;
    out.R.set_size(static_cast<arma::uword>(p), static_cast<arma::uword>(target));
    out.Ttrain.set_size(static_cast<arma::uword>(n), static_cast<arma::uword>(target));
    check_cuda(cudaMemcpy(out.R.memptr(), dQ_, bytes_for(p, target), cudaMemcpyDeviceToHost), "cudaMemcpy(R_plssvd)");
    check_cuda(cudaMemcpy(out.Ttrain.memptr(), dRRmat_, bytes_for(n, target), cudaMemcpyDeviceToHost), "cudaMemcpy(T_plssvd)");

    arma::vec s_host(target, arma::fill::zeros);
    copy_top_singular_values(l, target, s_host.memptr());
    arma::mat rhs_host(static_cast<arma::uword>(target), static_cast<arma::uword>(m), arma::fill::zeros);
    check_cuda(cudaMemcpy(rhs_host.memptr(), dZ_, bytes_for(target, m), cudaMemcpyDeviceToHost), "cudaMemcpy(RHS_plssvd)");
    out.Q.set_size(static_cast<arma::uword>(m), static_cast<arma::uword>(target));
    out.Q.zeros();
    for (int j = 0; j < target; ++j) {
      const double sj = s_host(static_cast<arma::uword>(j));
      if (std::isfinite(sj) && sj > 0.0) {
        out.Q.col(static_cast<arma::uword>(j)) = (rhs_host.row(static_cast<arma::uword>(j)) / sj).t();
      }
    }

    if (return_B) {
      out.B.set_size(static_cast<arma::uword>(p), static_cast<arma::uword>(m), ncomp.n_elem);
      out.B.zeros();
    }
    out.C_latent.set_size(
      static_cast<arma::uword>(target),
      static_cast<arma::uword>(target),
      ncomp.n_elem
    );
    out.C_latent.zeros();
    out.W_latent.set_size(
      static_cast<arma::uword>(target),
      static_cast<arma::uword>(m),
      ncomp.n_elem
    );
    out.W_latent.zeros();
    out.R2Y.set_size(ncomp.n_elem);
    out.R2Y.zeros();
    if (fit) {
      out.Yfit.set_size(static_cast<arma::uword>(n), static_cast<arma::uword>(m), ncomp.n_elem);
    }

    for (arma::uword a = 0; a < ncomp.n_elem; ++a) {
      const int requested_mc = static_cast<int>(ncomp(a));
      const int mc = std::min(std::max(requested_mc, 1), target);
      arma::mat c_host = plssvd_latent_coeff_from_gram(G_full_host, s_host, mc);
      out.C_latent.slice(a).submat(0, 0, mc - 1, mc - 1) = c_host;
      check_cuda(
        cudaMemcpy2D(
          dGram_,
          bytes_for(mc, 1),
          dQQmat_,
          bytes_for(target, 1),
          bytes_for(mc, 1),
          static_cast<size_t>(mc),
          cudaMemcpyDeviceToDevice
        ),
        "cudaMemcpy2D(G_a)"
      );
      check_cuda(
        cudaMemcpy2D(
          dBsmall_,
          bytes_for(mc, 1),
          dZ_,
          bytes_for(target, 1),
          bytes_for(mc, 1),
          static_cast<size_t>(m),
          cudaMemcpyDeviceToDevice
        ),
        "cudaMemcpy2D(RHS_a)"
      );
      solve_spd_system_inplace(mc, m);

      arma::mat w_host(static_cast<arma::uword>(mc), static_cast<arma::uword>(m), arma::fill::zeros);
      check_cuda(cudaMemcpy(w_host.memptr(), dBsmall_, bytes_for(mc, m), cudaMemcpyDeviceToHost), "cudaMemcpy(W_latent)");
      out.W_latent.slice(a).submat(0, 0, mc - 1, m - 1) = w_host;

      if (return_B) {
        check_cublas(
          cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, p, m, mc, &alpha, dQ_, p, dBsmall_, mc, &beta, dBcur_, p),
          "cublasDgemm(B=U*W)"
        );
        check_cuda(
          cudaMemcpy(out.B.slice(a).memptr(), dBcur_, bytes_for(p, m), cudaMemcpyDeviceToHost),
          "cudaMemcpy(B_slice)"
        );
      }

      if (fit) {
        check_cublas(
          cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, m, mc, &alpha, dRRmat_, n, dBsmall_, mc, &beta, dYfit_, n),
          "cublasDgemm(Yfit=T*W)"
        );
        check_cuda(
          cudaMemcpy(out.Yfit.slice(a).memptr(), dYfit_, bytes_for(n, m), cudaMemcpyDeviceToHost),
          "cudaMemcpy(Yfit_slice)"
        );
        out.R2Y(a) = rq_value_local(Ytrain, out.Yfit.slice(a));
      }
    }

    return out;
  }

  void orthonormalize_refresh_inplace(int m, int l) {
    if (l == 1) {
      double norm = 0.0;
      check_cublas(
        cublasDnrm2(handle_, m, dY_, 1, &norm),
        "cublasDnrm2(rank_one_refresh)"
      );
      if (!std::isfinite(norm) || norm <= 0.0) {
        throw std::runtime_error("rank-one randomized refresh has zero norm");
      }
      const double inverse = 1.0 / norm;
      check_cublas(
        cublasDscal(handle_, m, &inverse, dY_, 1),
        "cublasDscal(rank_one_refresh)"
      );
      return;
    }
    orthonormalize_qr_inplace(m, l);
  }

  void orthonormalize_qr_inplace(int m, int l) {
    if (env_int_or("FASTPLS_GPU_QR", 1, 0, 1) == 0) {
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, l, l, m, &one_, dY_, m, dY_, m, &zero_, dGram_, l),
        "cublasDgemm(Y^T*Y)"
      );
      hGram_host_.set_size(static_cast<arma::uword>(l), static_cast<arma::uword>(l));
      check_cuda(cudaMemcpy(hGram_host_.memptr(), dGram_, bytes_for(l, l), cudaMemcpyDeviceToHost), "cudaMemcpy(Gram)");
      arma::mat T = inv_sqrt_psd(hGram_host_);
      check_cuda(cudaMemcpy(dOmega_, T.memptr(), bytes_for(l, l), cudaMemcpyHostToDevice), "cudaMemcpy(T)");
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, l, l, &one_, dY_, m, dOmega_, l, &zero_, dQ_, m),
        "cublasDgemm(Q=Y*T)"
      );
      check_cuda(cudaMemcpy(dY_, dQ_, bytes_for(m, l), cudaMemcpyDeviceToDevice), "cudaMemcpy(Q->Y)");
      return;
    }

    int lwork_geqrf = 0;
    int lwork_orgqr = 0;
    check_cusolver(cusolverDnDgeqrf_bufferSize(solver_, m, l, dY_, m, &lwork_geqrf), "cusolverDnDgeqrf_bufferSize");
    check_cusolver(cusolverDnDorgqr_bufferSize(solver_, m, l, l, dY_, m, dTau_, &lwork_orgqr), "cusolverDnDorgqr_bufferSize");
    const int lwork = std::max(lwork_geqrf, lwork_orgqr);
    ensure_buffer(dWorkQR_, sizeof(double) * static_cast<size_t>(lwork), bytes_WorkQR_, "cudaMalloc(dWorkQR)");

    check_cusolver(cusolverDnDgeqrf(solver_, m, l, dY_, m, dTau_, dWorkQR_, lwork_geqrf, dInfo_), "cusolverDnDgeqrf");
    check_cuda(cudaMemcpy(&hInfo_, dInfo_, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(info_geqrf)");
    if (hInfo_ != 0) {
      throw std::runtime_error("cusolverDnDgeqrf returned non-zero info");
    }

    check_cusolver(cusolverDnDorgqr(solver_, m, l, l, dY_, m, dTau_, dWorkQR_, lwork_orgqr, dInfo_), "cusolverDnDorgqr");
    check_cuda(cudaMemcpy(&hInfo_, dInfo_, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(info_orgqr)");
    if (hInfo_ != 0) {
      throw std::runtime_error("cusolverDnDorgqr returned non-zero info");
    }
  }

  bool use_qless_qr_mode(int l) const {
    return l > 1;
  }

  arma::mat prepare_qless_transform(int m, int l) {
    ensure_buffer(dTransform_, bytes_for(l, l), bytes_Transform_, "cudaMalloc(dTransform)");
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, l, l, m, &one_, dY_, m, dY_, m, &zero_, dGram_, l),
      "cublasDgemm(Y^T*Y)"
    );
    hGram_host_.set_size(static_cast<arma::uword>(l), static_cast<arma::uword>(l));
    check_cuda(cudaMemcpy(hGram_host_.memptr(), dGram_, bytes_for(l, l), cudaMemcpyDeviceToHost), "cudaMemcpy(Gram)");
    arma::mat T = inv_sqrt_psd(hGram_host_);
    check_cuda(cudaMemcpy(dTransform_, T.memptr(), bytes_for(l, l), cudaMemcpyHostToDevice), "cudaMemcpy(T)");
    return T;
  }

  void form_bsmall_qless(int m, int n, int l) {
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, l, n, m, &one_, dY_, m, dA_, m, &zero_, dBsmall_, l),
      "cublasDgemm(Y^T*A)"
    );
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, l, n, l, &one_, dTransform_, l, dBsmall_, l, &zero_, dZ_, l),
      "cublasDgemm(T*Y^T*A)"
    );
    check_cuda(cudaMemcpy(dBsmall_, dZ_, bytes_for(l, n), cudaMemcpyDeviceToDevice), "cudaMemcpy(Bsmall_qless)");
  }

  void form_u_from_qless(int m, int l, int k) {
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, l, k, l, &one_, dTransform_, l, dOmega_, l, &zero_, dZ_, l),
      "cublasDgemm(T*Uhat)"
    );
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, k, l, &one_, dY_, m, dZ_, l, &zero_, dQ_, m),
      "cublasDgemm(Y*(T*Uhat))"
    );
  }

  void finalize_left_block_from_gram_inplace(int l, int k) {
    if (env_int_or("FASTPLS_GPU_EIG", 1, 0, 1) == 0) {
      hGram_host_.set_size(static_cast<arma::uword>(l), static_cast<arma::uword>(l));
      check_cuda(cudaMemcpy(hGram_host_.memptr(), dGram_, bytes_for(l, l), cudaMemcpyDeviceToHost), "cudaMemcpy(BsmallGram)");
      arma::vec evals;
      arma::mat evecs;
      if (!arma::eig_sym(evals, evecs, hGram_host_)) {
        throw std::runtime_error("eig_sym failed in finalize_left_block_from_gram_inplace");
      }
      arma::uvec ord = arma::sort_index(evals, "descend");
      arma::mat Uhat = evecs.cols(ord.head(static_cast<arma::uword>(k)));
      check_cuda(cudaMemcpy(dOmega_, Uhat.memptr(), bytes_for(l, k), cudaMemcpyHostToDevice), "cudaMemcpy(Uhat)");
      return;
    }

    int lwork_syevd = 0;
    check_cusolver(
      cusolverDnDsyevd_bufferSize(
        solver_,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        l,
        dGram_,
        l,
        dEvals_,
        &lwork_syevd
      ),
      "cusolverDnDsyevd_bufferSize"
    );
    ensure_buffer(dWorkEig_, sizeof(double) * static_cast<size_t>(lwork_syevd), bytes_WorkEig_, "cudaMalloc(dWorkEig)");
    check_cusolver(
      cusolverDnDsyevd(
        solver_,
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_UPPER,
        l,
        dGram_,
        l,
        dEvals_,
        dWorkEig_,
        lwork_syevd,
        dInfo_
      ),
      "cusolverDnDsyevd"
    );
    check_cuda(cudaMemcpy(&hInfo_, dInfo_, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(info_syevd)");
    if (hInfo_ != 0) {
      throw std::runtime_error("cusolverDnDsyevd returned non-zero info");
    }

    for (int j = 0; j < k; ++j) {
      const int src_col = l - 1 - j;
      check_cublas(
        cublasDcopy(handle_, l, dGram_ + static_cast<size_t>(src_col) * static_cast<size_t>(l), 1,
                    dOmega_ + static_cast<size_t>(j) * static_cast<size_t>(l), 1),
        "cublasDcopy(Uhat)"
      );
    }
  }

  void copy_top_singular_values(int l, int k, double* hSvals) {
    if (hSvals == nullptr) {
      return;
    }
    arma::vec evals_host(l);
    check_cuda(cudaMemcpy(evals_host.memptr(), dEvals_, bytes_for(l, 1), cudaMemcpyDeviceToHost), "cudaMemcpy(evals)");
    for (int j = 0; j < k; ++j) {
      const int idx = l - 1 - j;
      hSvals[j] = std::sqrt(std::max(evals_host(static_cast<arma::uword>(idx)), 0.0));
    }
  }

  void sample_y(
    const double* hA,
    int m,
    int n,
    const double* hOmega,
    int l,
    int power_iters,
    double* hY
  ) {
    ensure_capacity(m, n, l);

    check_cuda(cudaMemcpy(dA_, hA, bytes_for(m, n), cudaMemcpyHostToDevice), "cudaMemcpy(A)");
    check_cuda(cudaMemcpy(dOmega_, hOmega, bytes_for(n, l), cudaMemcpyHostToDevice), "cudaMemcpy(Omega)");

    const double alpha = 1.0;
    const double beta = 0.0;

    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, l, n, &alpha, dA_, m, dOmega_, n, &beta, dY_, m),
      "cublasDgemm(A*Omega)"
    );

    for (int i = 0; i < power_iters; ++i) {
      orthonormalize_qr_inplace(m, l);
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, n, l, m, &alpha, dA_, m, dY_, m, &beta, dZ_, n),
        "cublasDgemm(A^T*Y)"
      );
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, l, n, &alpha, dA_, m, dZ_, n, &beta, dY_, m),
        "cublasDgemm(A*Z)"
      );
    }

    check_cuda(cudaMemcpy(hY, dY_, bytes_for(m, l), cudaMemcpyDeviceToHost), "cudaMemcpy(Y)");
  }

  void set_matrix(const double* hA, int m, int n) {
    ensure_capacity(m, n, 1);
    check_cuda(cudaMemcpy(dA_, hA, bytes_for(m, n), cudaMemcpyHostToDevice), "cudaMemcpy(A)");
  }

  void refresh_left_block(
    const double* hA,
    int m,
    int n,
    unsigned int seed,
    int l,
    int power_iters,
    double* hY
  ) {
    ensure_capacity(m, n, l);
    hOmega_host_.set_size(static_cast<arma::uword>(n), static_cast<arma::uword>(l));

    check_cuda(cudaMemcpy(dA_, hA, bytes_for(m, n), cudaMemcpyHostToDevice), "cudaMemcpy(A)");
    reset_curand_seed(rng_, seed);
    check_curand(curandGenerateNormalDouble(rng_, dY_, padded_random_elems(m, l), 0.0, 1.0), "curandGenerateNormalDouble(Y0)");

    const double alpha = 1.0;
    const double beta = 0.0;

    for (int i = 0; i < power_iters; ++i) {
      orthonormalize_qr_inplace(m, l);
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, n, l, m, &alpha, dA_, m, dY_, m, &beta, dZ_, n),
        "cublasDgemm(A^T*Y)"
      );
      check_cuda(cudaMemcpy(hOmega_host_.memptr(), dZ_, bytes_for(n, l), cudaMemcpyDeviceToHost), "cudaMemcpy(Z_host)");

      arma::mat Qz;
      arma::mat Rz;
      arma::qr_econ(Qz, Rz, hOmega_host_);
      if (Qz.n_cols > static_cast<arma::uword>(l)) {
        Qz = Qz.cols(0, static_cast<arma::uword>(l - 1));
      }
      check_cuda(cudaMemcpy(dOmega_, Qz.memptr(), bytes_for(n, static_cast<int>(Qz.n_cols)), cudaMemcpyHostToDevice), "cudaMemcpy(Qz)");
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, static_cast<int>(Qz.n_cols), n, &alpha, dA_, m, dOmega_, n, &beta, dY_, m),
        "cublasDgemm(A*Qz)"
      );
    }

    check_cuda(cudaMemcpy(hY, dY_, bytes_for(m, l), cudaMemcpyDeviceToHost), "cudaMemcpy(Y)");
  }

  void refresh_left_block_u(
    const double* hA,
    int m,
    int n,
    unsigned int seed,
    int l,
    int k,
    int power_iters,
    double* hUblock,
    double* hSvals
  ) {
    ensure_capacity(m, n, l);
    check_cuda(cudaMemcpy(dA_, hA, bytes_for(m, n), cudaMemcpyHostToDevice), "cudaMemcpy(A)");
    reset_curand_seed(rng_, seed);
    check_curand(curandGenerateNormalDouble(rng_, dY_, padded_random_elems(m, l), 0.0, 1.0), "curandGenerateNormalDouble(Y0)");

    const double alpha = 1.0;
    const double beta = 0.0;

    for (int i = 0; i < power_iters; ++i) {
      orthonormalize_qr_inplace(m, l);
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, n, l, m, &alpha, dA_, m, dY_, m, &beta, dZ_, n),
        "cublasDgemm(A^T*Y)"
      );
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, l, n, &alpha, dA_, m, dZ_, n, &beta, dY_, m),
        "cublasDgemm(A*Z)"
      );
    }
    const bool use_qless = use_qless_qr_mode(l);
    if (use_qless) {
      prepare_qless_transform(m, l);
      form_bsmall_qless(m, n, l);
    } else {
      orthonormalize_qr_inplace(m, l);
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, l, n, m, &alpha, dY_, m, dA_, m, &beta, dBsmall_, l),
        "cublasDgemm(Bsmall=Q^T*A)"
      );
    }
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T, l, l, n, &alpha, dBsmall_, l, dBsmall_, l, &beta, dGram_, l),
      "cublasDgemm(Bsmall*Bsmall^T)"
    );

    finalize_left_block_from_gram_inplace(l, k);
    copy_top_singular_values(l, k, hSvals);
    if (use_qless) {
      form_u_from_qless(m, l, k);
    } else {
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, k, l, &alpha, dY_, m, dOmega_, l, &beta, dQ_, m),
        "cublasDgemm(Ublock=Q*Uhat)"
      );
    }
    check_cuda(cudaMemcpy(hUblock, dQ_, bytes_for(m, k), cudaMemcpyDeviceToHost), "cudaMemcpy(Ublock)");
  }

  void refresh_left_block_u_resident(
    int m,
    int n,
    int l,
    int k,
    unsigned int seed,
    int power_iters,
    double* hUblock,
    double* hSvals
  ) {
    ensure_capacity(m, n, l);
    reset_curand_seed(rng_, seed);
    check_curand(curandGenerateNormalDouble(rng_, dY_, padded_random_elems(m, l), 0.0, 1.0), "curandGenerateNormalDouble(Y0)");

    const double alpha = 1.0;
    const double beta = 0.0;

    for (int i = 0; i < power_iters; ++i) {
      orthonormalize_qr_inplace(m, l);
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, n, l, m, &alpha, dA_, m, dY_, m, &beta, dZ_, n),
        "cublasDgemm(A^T*Y)"
      );
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, l, n, &alpha, dA_, m, dZ_, n, &beta, dY_, m),
        "cublasDgemm(A*Z)"
      );
    }
    const bool use_qless = use_qless_qr_mode(l);
    if (use_qless) {
      prepare_qless_transform(m, l);
      form_bsmall_qless(m, n, l);
    } else {
      orthonormalize_qr_inplace(m, l);
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, l, n, m, &alpha, dY_, m, dA_, m, &beta, dBsmall_, l),
        "cublasDgemm(Bsmall=Q^T*A)"
      );
    }
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T, l, l, n, &alpha, dBsmall_, l, dBsmall_, l, &beta, dGram_, l),
      "cublasDgemm(Bsmall*Bsmall^T)"
    );

    finalize_left_block_from_gram_inplace(l, k);
    copy_top_singular_values(l, k, hSvals);
    if (use_qless) {
      form_u_from_qless(m, l, k);
    } else {
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, k, l, &alpha, dY_, m, dOmega_, l, &beta, dQ_, m),
        "cublasDgemm(Ublock=Q*Uhat)"
      );
    }
    check_cuda(cudaMemcpy(hUblock, dQ_, bytes_for(m, k), cudaMemcpyDeviceToHost), "cudaMemcpy(Ublock)");
  }

  SVDResult resident_svd(
    const arma::mat& A,
    int k,
    const SVDOptions& opt
  ) {
    const int m = static_cast<int>(A.n_rows);
    const int n = static_cast<int>(A.n_cols);
    const int max_rank = std::min(m, n);
    const int target = std::min(max_rank, std::max(k, 1));
    const int l = std::min(max_rank, target + std::max(opt.oversample, 0));

    ensure_capacity(m, n, l);
    set_matrix(A.memptr(), m, n);

    reset_curand_seed(rng_, opt.seed);
    check_curand(curandGenerateNormalDouble(rng_, dOmega_, padded_random_elems(n, l), 0.0, 1.0), "curandGenerateNormalDouble(Omega)");

    const double alpha = 1.0;
    const double beta = 0.0;
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, l, n, &alpha, dA_, m, dOmega_, n, &beta, dY_, m),
      "cublasDgemm(A*Omega)"
    );

    const int power_iters = std::max(opt.power_iters, 0);
    for (int i = 0; i < power_iters; ++i) {
      orthonormalize_qr_inplace(m, l);
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, n, l, m, &alpha, dA_, m, dY_, m, &beta, dZ_, n),
        "cublasDgemm(A^T*Y)"
      );
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, l, n, &alpha, dA_, m, dZ_, n, &beta, dY_, m),
        "cublasDgemm(A*Z)"
      );
    }
    const bool use_qless = use_qless_qr_mode(l);
    arma::mat T_host;
    if (use_qless) {
      T_host = prepare_qless_transform(m, l);
      form_bsmall_qless(m, n, l);
    } else {
      orthonormalize_qr_inplace(m, l);
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, l, n, m, &alpha, dY_, m, dA_, m, &beta, dBsmall_, l),
        "cublasDgemm(Bsmall=Q^T*A)"
      );
    }

    const int gpu_finalize_threshold = env_int_or("FASTPLS_GPU_FINALIZE_THRESHOLD", 32, 1, 4096);
    if (l <= gpu_finalize_threshold) {
      arma::mat hQ(m, l);
      arma::mat hB(l, n);
      check_cuda(cudaMemcpy(hQ.memptr(), dY_, bytes_for(m, l), cudaMemcpyDeviceToHost), "cudaMemcpy(Q_or_Y_host)");
      if (use_qless) {
        hQ = hQ * T_host;
      }
      check_cuda(cudaMemcpy(hB.memptr(), dBsmall_, bytes_for(l, n), cudaMemcpyDeviceToHost), "cudaMemcpy(Bsmall_host)");
      return finalize_from_q_bsmall(hQ, hB, target, opt.left_only);
    }

    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_T, l, l, n, &alpha, dBsmall_, l, dBsmall_, l, &beta, dGram_, l),
      "cublasDgemm(Bsmall*Bsmall^T)"
    );

    finalize_left_block_from_gram_inplace(l, target);
    if (use_qless) {
      form_u_from_qless(m, l, target);
    } else {
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, target, l, &alpha, dY_, m, dOmega_, l, &beta, dQ_, m),
        "cublasDgemm(U=Q*Uhat)"
      );
    }

    SVDResult out;
    out.U.set_size(static_cast<arma::uword>(m), static_cast<arma::uword>(target));
    check_cuda(cudaMemcpy(out.U.memptr(), dQ_, bytes_for(m, target), cudaMemcpyDeviceToHost), "cudaMemcpy(U)");

    arma::vec evals_host(l);
    check_cuda(cudaMemcpy(evals_host.memptr(), dEvals_, bytes_for(l, 1), cudaMemcpyDeviceToHost), "cudaMemcpy(evals)");
    out.s.set_size(static_cast<arma::uword>(target));
    for (int j = 0; j < target; ++j) {
      const int idx = l - 1 - j;
      out.s(static_cast<arma::uword>(j)) = std::sqrt(std::max(evals_host(static_cast<arma::uword>(idx)), 0.0));
    }

    if (!opt.left_only) {
      check_cublas(
        cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, target, n, l, &alpha, dOmega_, l, dBsmall_, l, &beta, dZ_, target),
        "cublasDgemm(Vt=Uhat^T*Bsmall)"
      );
      out.Vt.set_size(static_cast<arma::uword>(target), static_cast<arma::uword>(n));
      check_cuda(cudaMemcpy(out.Vt.memptr(), dZ_, bytes_for(target, n), cudaMemcpyDeviceToHost), "cudaMemcpy(Vt)");
      for (int j = 0; j < target; ++j) {
        const double sj = out.s(static_cast<arma::uword>(j));
        if (std::isfinite(sj) && sj > 0.0) {
          out.Vt.row(static_cast<arma::uword>(j)) /= sj;
        }
      }
    }

    return out;
  }

  void project_left_row(const double* hV, int m, int n, double* hVS) {
    ensure_capacity(m, n, 1);
    check_cuda(cudaMemcpy(dY_, hV, bytes_for(m, 1), cudaMemcpyHostToDevice), "cudaMemcpy(v)");
    const double alpha = 1.0;
    const double beta = 0.0;
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, n, 1, m, &alpha, dA_, m, dY_, m, &beta, dZ_, n),
      "cublasDgemm(A^T*v)"
    );
    check_cuda(cudaMemcpy(hVS, dZ_, bytes_for(n, 1), cudaMemcpyDeviceToHost), "cudaMemcpy(vS)");
  }

  void deflate_left_rank1(const double* hV, const double* hVS, int m, int n) {
    ensure_capacity(m, n, 1);
    check_cuda(cudaMemcpy(dY_, hV, bytes_for(m, 1), cudaMemcpyHostToDevice), "cudaMemcpy(v)");
    check_cuda(cudaMemcpy(dZ_, hVS, bytes_for(n, 1), cudaMemcpyHostToDevice), "cudaMemcpy(vS)");
    const double alpha = -1.0;
    check_cublas(
      cublasDger(handle_, m, n, &alpha, dY_, 1, dZ_, 1, dA_, m),
      "cublasDger(deflate)"
    );
  }

 private:
  void ensure_stream() {
    if (env_int_or("FASTPLS_CUDA_WORKSPACE_STREAMS", 0, 0, 1) == 0) {
      return;
    }
    if (!stream_ready_) {
      check_cuda(
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags"
      );
      stream_ready_ = true;
    }
  }

  static size_t bytes_for(int m, int n) {
    return sizeof(double) * static_cast<size_t>(m) * static_cast<size_t>(n);
  }

  static size_t bytes_for_padded_random(int m, int n) {
    size_t elems = static_cast<size_t>(m) * static_cast<size_t>(n);
    if ((elems % 2U) != 0U) {
      ++elems;
    }
    return sizeof(double) * elems;
  }

  static size_t padded_random_elems(int m, int n) {
    size_t elems = static_cast<size_t>(m) * static_cast<size_t>(n);
    if ((elems % 2U) != 0U) {
      ++elems;
    }
    return elems;
  }

  void ensure_buffer(double*& ptr, size_t required, size_t& current, const char* where) {
    if (required <= current && ptr != nullptr) {
      return;
    }
    if (ptr != nullptr) {
      cudaFree(ptr);
      ptr = nullptr;
      current = 0;
    }
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr), required), where);
    current = required;
  }

  void crossprod_training_xy(int n, int p, int m) {
    const double alpha = 1.0;
    const double beta = 0.0;
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, p, m, n, &alpha, dX_, n, dYtrain_, n, &beta, dA_, p),
      "cublasDgemm(S=X^T*Y)"
    );
  }

  void x_times_vec(int n, int p, const double* dVec, double* dOut, const char* where) {
    const double alpha = 1.0;
    const double beta = 0.0;
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, 1, p, &alpha, dX_, n, dVec, p, &beta, dOut, n),
      where
    );
  }

  void xt_times_vec(int n, int p, const double* dVec, double* dOut, const char* where) {
    const double alpha = 1.0;
    const double beta = 0.0;
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, p, 1, n, &alpha, dX_, n, dVec, n, &beta, dOut, p),
      where
    );
  }

  void yt_times_vec(int n, int m, const double* dVec, double* dOut, const char* where) {
    const double alpha = 1.0;
    const double beta = 0.0;
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, m, 1, n, &alpha, dYtrain_, n, dVec, n, &beta, dOut, m),
      where
    );
  }

  void x_times_mat(int n, int p, int k, const double* dRight, double* dOut, const char* where) {
    const double alpha = 1.0;
    const double beta = 0.0;
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, k, p, &alpha, dX_, n, dRight, p, &beta, dOut, n),
      where
    );
  }

  void implicit_a_times_mat(int n, int p, int m, int k, const double* dRight, double* dOut, const char* where) {
    const double alpha = 1.0;
    const double beta = 0.0;
    ensure_buffer(dTransform_, bytes_for(n, k), bytes_Transform_, "cudaMalloc(dTransform)");
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, k, m, &alpha, dYtrain_, n,
                  dRight, m, &beta, dTransform_, n),
      where
    );
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, p, k, n, &alpha, dX_, n,
                  dTransform_, n, &beta, dOut, p),
      where
    );
  }

  void implicit_at_times_mat(int n, int p, int m, int k, const double* dRight, double* dOut, const char* where) {
    const double alpha = 1.0;
    const double beta = 0.0;
    ensure_buffer(dTransform_, bytes_for(n, k), bytes_Transform_, "cudaMalloc(dTransform)");
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, k, p, &alpha, dX_, n,
                  dRight, p, &beta, dTransform_, n),
      where
    );
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, m, k, n, &alpha, dYtrain_, n,
                  dTransform_, n, &beta, dOut, m),
      where
    );
  }

  void implicit_bsmall_from_basis(int n, int p, int m, int l, const double* dBasis, double* dOut, const char* where) {
    const double alpha = 1.0;
    const double beta = 0.0;
    ensure_buffer(dTransform_, bytes_for(n, l), bytes_Transform_, "cudaMalloc(dTransform)");
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, n, l, p, &alpha, dX_, n,
                  dBasis, p, &beta, dTransform_, n),
      where
    );
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, l, m, n, &alpha, dTransform_, n,
                  dYtrain_, n, &beta, dOut, l),
      where
    );
  }

  void subtract_left_projection_inplace(
    int p,
    int k,
    int prev_v_cols,
    double* dMat,
    const char* where
  ) {
    if (prev_v_cols <= 0) {
      return;
    }
    const double alpha = 1.0;
    const double beta = 0.0;
    const double minus_one = -1.0;
    ensure_buffer(dCoeff_, bytes_for(prev_v_cols, k), bytes_Coeff_, "cudaMalloc(dCoeff)");
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, prev_v_cols, k, p,
                  &alpha, dVVmat_, p, dMat, p, &beta, dCoeff_, prev_v_cols),
      where
    );
    check_cublas(
      cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, p, k, prev_v_cols,
                  &minus_one, dVVmat_, p, dCoeff_, prev_v_cols, &alpha, dMat, p),
      where
    );
  }

  void implicit_a_times_mat_deflated(
    int n,
    int p,
    int m,
    int k,
    const double* dRight,
    double* dOut,
    int prev_v_cols,
    const char* where
  ) {
    implicit_a_times_mat(n, p, m, k, dRight, dOut, where);
    subtract_left_projection_inplace(p, k, prev_v_cols, dOut, where);
  }

  void implicit_at_times_mat_deflated(
    int n,
    int p,
    int m,
    int k,
    const double* dRight,
    double* dOut,
    int prev_v_cols,
    const char* where
  ) {
    if (prev_v_cols <= 0) {
      implicit_at_times_mat(n, p, m, k, dRight, dOut, where);
      return;
    }
    ensure_buffer(dQ_, bytes_for(p, k), bytes_Q_, "cudaMalloc(dQ)");
    check_cuda(cudaMemcpy(dQ_, dRight, bytes_for(p, k), cudaMemcpyDeviceToDevice), "cudaMemcpy(deflated_basis)");
    subtract_left_projection_inplace(p, k, prev_v_cols, dQ_, where);
    implicit_at_times_mat(n, p, m, k, dQ_, dOut, where);
  }

  void implicit_bsmall_from_deflated_basis(
    int n,
    int p,
    int m,
    int l,
    const double* dBasis,
    double* dOut,
    int prev_v_cols,
    const char* where
  ) {
    if (prev_v_cols <= 0) {
      implicit_bsmall_from_basis(n, p, m, l, dBasis, dOut, where);
      return;
    }
    ensure_buffer(dQ_, bytes_for(p, l), bytes_Q_, "cudaMalloc(dQ)");
    check_cuda(cudaMemcpy(dQ_, dBasis, bytes_for(p, l), cudaMemcpyDeviceToDevice), "cudaMemcpy(deflated_bsmall_basis)");
    subtract_left_projection_inplace(p, l, prev_v_cols, dQ_, where);
    implicit_bsmall_from_basis(n, p, m, l, dQ_, dOut, where);
  }

  void ensure_int_buffer(int*& ptr, size_t required, size_t& current, const char* where) {
    if (required <= current && ptr != nullptr) {
      return;
    }
    if (ptr != nullptr) {
      cudaFree(ptr);
      ptr = nullptr;
      current = 0;
    }
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&ptr), required), where);
    current = required;
  }

  void release() {
    if (stream_ready_) {
      cudaStreamSynchronize(stream_);
    }
    if (dX_ != nullptr) cudaFree(dX_);
    if (dYtrain_ != nullptr) cudaFree(dYtrain_);
    if (dA_ != nullptr) cudaFree(dA_);
    if (dOmega_ != nullptr) cudaFree(dOmega_);
    if (dY_ != nullptr) cudaFree(dY_);
    if (dZ_ != nullptr) cudaFree(dZ_);
    if (dQ_ != nullptr) cudaFree(dQ_);
    if (dTransform_ != nullptr) cudaFree(dTransform_);
    if (dRRmat_ != nullptr) cudaFree(dRRmat_);
    if (dQQmat_ != nullptr) cudaFree(dQQmat_);
    if (dVVmat_ != nullptr) cudaFree(dVVmat_);
    if (dBcur_ != nullptr) cudaFree(dBcur_);
    if (dCoeff_ != nullptr) cudaFree(dCoeff_);
    if (dTvec_ != nullptr) cudaFree(dTvec_);
    if (dPvec_ != nullptr) cudaFree(dPvec_);
    if (dQvec_ != nullptr) cudaFree(dQvec_);
    if (dRvec_ != nullptr) cudaFree(dRvec_);
    if (dYfit_ != nullptr) cudaFree(dYfit_);
    if (dBsmall_ != nullptr) cudaFree(dBsmall_);
    if (dGram_ != nullptr) cudaFree(dGram_);
    if (dTau_ != nullptr) cudaFree(dTau_);
    if (dEvals_ != nullptr) cudaFree(dEvals_);
    if (dWorkQR_ != nullptr) cudaFree(dWorkQR_);
    if (dWorkEig_ != nullptr) cudaFree(dWorkEig_);
    if (dInfo_ != nullptr) cudaFree(dInfo_);
    dX_ = nullptr;
    dYtrain_ = nullptr;
    dA_ = nullptr;
    dOmega_ = nullptr;
    dY_ = nullptr;
    dZ_ = nullptr;
    dQ_ = nullptr;
    dTransform_ = nullptr;
    dRRmat_ = nullptr;
    dQQmat_ = nullptr;
    dVVmat_ = nullptr;
    dBcur_ = nullptr;
    dCoeff_ = nullptr;
    dTvec_ = nullptr;
    dPvec_ = nullptr;
    dQvec_ = nullptr;
    dRvec_ = nullptr;
    dYfit_ = nullptr;
    dBsmall_ = nullptr;
    dGram_ = nullptr;
    dTau_ = nullptr;
    dEvals_ = nullptr;
    dWorkQR_ = nullptr;
    dWorkEig_ = nullptr;
    dInfo_ = nullptr;
    bytes_X_ = bytes_Ytrain_ = 0;
    bytes_A_ = bytes_Omega_ = bytes_Y_ = bytes_Z_ = 0;
    bytes_Q_ = bytes_Bsmall_ = bytes_Gram_ = 0;
    bytes_Transform_ = 0;
    bytes_RRmat_ = bytes_QQmat_ = bytes_VVmat_ = bytes_Bcur_ = bytes_Coeff_ = 0;
    bytes_Tvec_ = bytes_Pvec_ = bytes_Qvec_ = bytes_Rvec_ = bytes_Yfit_ = 0;
    bytes_Tau_ = bytes_Evals_ = bytes_WorkQR_ = bytes_WorkEig_ = bytes_Info_ = 0;
    if (handle_ready_) {
      cublasDestroy(handle_);
      handle_ready_ = false;
    }
    if (solver_ready_) {
      cusolverDnDestroy(solver_);
      solver_ready_ = false;
    }
    if (rng_ready_) {
      curandDestroyGenerator(rng_);
      rng_ready_ = false;
    }
    if (stream_ready_) {
      cudaStreamDestroy(stream_);
      stream_ready_ = false;
      stream_ = nullptr;
    }
  }

  double* dX_ = nullptr;
  double* dYtrain_ = nullptr;
  double* dA_ = nullptr;
  double* dOmega_ = nullptr;
  double* dY_ = nullptr;
  double* dZ_ = nullptr;
  double* dQ_ = nullptr;
  double* dTransform_ = nullptr;
  double* dRRmat_ = nullptr;
  double* dQQmat_ = nullptr;
  double* dVVmat_ = nullptr;
  double* dBcur_ = nullptr;
  double* dCoeff_ = nullptr;
  double* dTvec_ = nullptr;
  double* dPvec_ = nullptr;
  double* dQvec_ = nullptr;
  double* dRvec_ = nullptr;
  double* dYfit_ = nullptr;
  double* dBsmall_ = nullptr;
  double* dGram_ = nullptr;
  double* dTau_ = nullptr;
  double* dEvals_ = nullptr;
  double* dWorkQR_ = nullptr;
  double* dWorkEig_ = nullptr;
  int* dInfo_ = nullptr;
  size_t bytes_X_ = 0;
  size_t bytes_Ytrain_ = 0;
  size_t bytes_A_ = 0;
  size_t bytes_Omega_ = 0;
  size_t bytes_Y_ = 0;
  size_t bytes_Z_ = 0;
  size_t bytes_Q_ = 0;
  size_t bytes_Transform_ = 0;
  size_t bytes_RRmat_ = 0;
  size_t bytes_QQmat_ = 0;
  size_t bytes_VVmat_ = 0;
  size_t bytes_Bcur_ = 0;
  size_t bytes_Coeff_ = 0;
  size_t bytes_Tvec_ = 0;
  size_t bytes_Pvec_ = 0;
  size_t bytes_Qvec_ = 0;
  size_t bytes_Rvec_ = 0;
  size_t bytes_Yfit_ = 0;
  size_t bytes_Bsmall_ = 0;
  size_t bytes_Gram_ = 0;
  size_t bytes_Tau_ = 0;
  size_t bytes_Evals_ = 0;
  size_t bytes_WorkQR_ = 0;
  size_t bytes_WorkEig_ = 0;
  size_t bytes_Info_ = 0;
  cublasHandle_t handle_ = nullptr;
  bool handle_ready_ = false;
  cusolverDnHandle_t solver_ = nullptr;
  bool solver_ready_ = false;
  curandGenerator_t rng_ = nullptr;
  bool rng_ready_ = false;
  cudaStream_t stream_ = nullptr;
  bool stream_ready_ = false;
  int hInfo_ = 0;
  const double one_ = 1.0;
  const double zero_ = 0.0;
  arma::mat hOmega_host_;
  arma::mat hGram_host_;
};

template <typename T>
class CudaLDADeviceBuffer {
 public:
  CudaLDADeviceBuffer() = default;
  ~CudaLDADeviceBuffer() { reset(); }
  CudaLDADeviceBuffer(const CudaLDADeviceBuffer&) = delete;
  CudaLDADeviceBuffer& operator=(const CudaLDADeviceBuffer&) = delete;

  void ensure(std::size_t n) {
    if (n <= capacity_) return;
    reset();
    if (n > 0) check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), n * sizeof(T)),
                          "cudaMalloc(CudaLDADeviceBuffer)");
    capacity_ = n;
  }

  void reset() {
    if (data_ != nullptr) cudaFree(data_);
    data_ = nullptr;
    capacity_ = 0;
  }

  T* data() { return data_; }
  const T* data() const { return data_; }

 private:
  T* data_ = nullptr;
  std::size_t capacity_ = 0;
};

class CudaFloatWorkspace {
 public:
  CudaFloatWorkspace() = default;
  ~CudaFloatWorkspace() { reset(); }
  CudaFloatWorkspace(const CudaFloatWorkspace&) = delete;
  CudaFloatWorkspace& operator=(const CudaFloatWorkspace&) = delete;

  void initialize() {
    if (ready_) return;
    check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate(float32 workspace)");
    check_cublas(cublasCreate(&blas_), "cublasCreate(float32 workspace)");
    check_cublas(
      cublasSetStream(blas_, stream_),
      "cublasSetStream(float32 workspace)"
    );
    check_cusolver(
      cusolverDnCreate(&solver_),
      "cusolverDnCreate(float32 workspace)"
    );
    check_cusolver(
      cusolverDnSetStream(solver_, stream_),
      "cusolverDnSetStream(float32 workspace)"
    );
    ready_ = true;
  }

  void reset() {
    a.reset();
    b.reset();
    c.reset();
    z.reset();
    tau.reset();
    qr_work.reset();
    info.reset();
    if (solver_ != nullptr) cusolverDnDestroy(solver_);
    if (blas_ != nullptr) cublasDestroy(blas_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
    solver_ = nullptr;
    blas_ = nullptr;
    stream_ = nullptr;
    ready_ = false;
  }

  void orthonormalize(float* matrix, int rows, int cols, const char* where) {
    if (rows < cols || cols < 1) {
      throw std::runtime_error(std::string(where) + ": invalid QR dimensions");
    }
    tau.ensure(static_cast<std::size_t>(cols));
    info.ensure(1U);
    int lwork_geqrf = 0;
    int lwork_orgqr = 0;
    check_cusolver(
      cusolverDnSgeqrf_bufferSize(
        solver_, rows, cols, matrix, rows, &lwork_geqrf
      ),
      "cusolverDnSgeqrf_bufferSize(float32 rSVD)"
    );
    check_cusolver(
      cusolverDnSorgqr_bufferSize(
        solver_, rows, cols, cols, matrix, rows, tau.data(), &lwork_orgqr
      ),
      "cusolverDnSorgqr_bufferSize(float32 rSVD)"
    );
    qr_work.ensure(static_cast<std::size_t>(std::max(lwork_geqrf, lwork_orgqr)));
    check_cusolver(
      cusolverDnSgeqrf(
        solver_, rows, cols, matrix, rows, tau.data(), qr_work.data(),
        lwork_geqrf, info.data()
      ),
      "cusolverDnSgeqrf(float32 rSVD)"
    );
    int host_info = 0;
    check_cuda(
      cudaMemcpy(&host_info, info.data(), sizeof(int), cudaMemcpyDeviceToHost),
      "cudaMemcpy(float32 rSVD geqrf info)"
    );
    if (host_info != 0) {
      throw std::runtime_error(std::string(where) + ": float32 geqrf failed");
    }
    check_cusolver(
      cusolverDnSorgqr(
        solver_, rows, cols, cols, matrix, rows, tau.data(), qr_work.data(),
        lwork_orgqr, info.data()
      ),
      "cusolverDnSorgqr(float32 rSVD)"
    );
    check_cuda(
      cudaMemcpy(&host_info, info.data(), sizeof(int), cudaMemcpyDeviceToHost),
      "cudaMemcpy(float32 rSVD orgqr info)"
    );
    if (host_info != 0) {
      throw std::runtime_error(std::string(where) + ": float32 orgqr failed");
    }
  }

  cudaStream_t stream() const { return stream_; }
  cublasHandle_t blas() const { return blas_; }

  CudaLDADeviceBuffer<float> a;
  CudaLDADeviceBuffer<float> b;
  CudaLDADeviceBuffer<float> c;
  CudaLDADeviceBuffer<float> z;
  CudaLDADeviceBuffer<float> tau;
  CudaLDADeviceBuffer<float> qr_work;
  CudaLDADeviceBuffer<int> info;

 private:
  cudaStream_t stream_ = nullptr;
  cublasHandle_t blas_ = nullptr;
  cusolverDnHandle_t solver_ = nullptr;
  bool ready_ = false;
};

class CudaLDAFloatWorkspace {
 public:
  CudaLDAFloatWorkspace() = default;
  ~CudaLDAFloatWorkspace() { reset(); }
  CudaLDAFloatWorkspace(const CudaLDAFloatWorkspace&) = delete;
  CudaLDAFloatWorkspace& operator=(const CudaLDAFloatWorkspace&) = delete;

  void initialize() {
    if (ready_) return;
    check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate(float32 LDA workspace)");
    check_cublas(cublasCreate(&blas_), "cublasCreate(float32 LDA workspace)");
    check_cublas(cublasSetStream(blas_, stream_), "cublasSetStream(float32 LDA workspace)");
    check_cusolver(cusolverDnCreate(&solver_), "cusolverDnCreate(float32 LDA workspace)");
    check_cusolver(cusolverDnSetStream(solver_, stream_), "cusolverDnSetStream(float32 LDA workspace)");
    ready_ = true;
  }

  void reset() {
    train_scores.reset();
    labels.reset();
    counts.reset();
    means.reset();
    pooled.reset();
    covariance.reset();
    rhs.reset();
    linear.reset();
    constants.reset();
    lambda.reset();
    solver_work.reset();
    info.reset();
    test_latent.reset();
    test_scores.reset();
    predictions.reset();
    if (solver_ != nullptr) cusolverDnDestroy(solver_);
    if (blas_ != nullptr) cublasDestroy(blas_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
    solver_ = nullptr;
    blas_ = nullptr;
    stream_ = nullptr;
    ready_ = false;
  }

  cudaStream_t stream() const { return stream_; }
  cublasHandle_t blas() const { return blas_; }
  cusolverDnHandle_t solver() const { return solver_; }

  CudaLDADeviceBuffer<float> train_scores;
  CudaLDADeviceBuffer<int> labels;
  CudaLDADeviceBuffer<float> counts;
  CudaLDADeviceBuffer<float> means;
  CudaLDADeviceBuffer<float> pooled;
  CudaLDADeviceBuffer<float> covariance;
  CudaLDADeviceBuffer<float> rhs;
  CudaLDADeviceBuffer<float> linear;
  CudaLDADeviceBuffer<float> constants;
  CudaLDADeviceBuffer<float> lambda;
  CudaLDADeviceBuffer<float> solver_work;
  CudaLDADeviceBuffer<int> info;
  CudaLDADeviceBuffer<float> test_latent;
  CudaLDADeviceBuffer<float> test_scores;
  CudaLDADeviceBuffer<int> predictions;

 private:
  cudaStream_t stream_ = nullptr;
  cublasHandle_t blas_ = nullptr;
  cusolverDnHandle_t solver_ = nullptr;
  bool ready_ = false;
};

thread_local CudaRSVDWorkspace g_workspace;
thread_local CudaFloatWorkspace g_float_workspace;
thread_local CudaLDAFloatWorkspace g_lda_float_workspace;

} // namespace

bool cuda_runtime_available() {
  int n_devices = 0;
  const cudaError_t status = cudaGetDeviceCount(&n_devices);
  return (status == cudaSuccess && n_devices > 0);
}

struct CudaFloatCrossproduct::Impl {
  Impl(const arma::fmat& X, const arma::fmat& Y) {
    if (!cuda_runtime_available()) {
      throw std::runtime_error("CUDA cross-product unavailable; no CPU fallback is performed");
    }
    if (X.n_rows != Y.n_rows || X.n_elem == 0 || Y.n_elem == 0) {
      throw std::runtime_error("CUDA cross-product requires nonempty X and Y with matching rows");
    }
    const auto limit = static_cast<arma::uword>(std::numeric_limits<int>::max());
    if (X.n_rows > limit || X.n_cols > limit || Y.n_cols > limit) {
      throw std::runtime_error("CUDA cross-product dimensions exceed CUDA int limits");
    }
    n = static_cast<int>(X.n_rows);
    p = static_cast<int>(X.n_cols);
    q = static_cast<int>(Y.n_cols);
    workspace.initialize();
    x.ensure(X.n_elem);
    y.ensure(Y.n_elem);
    check_cuda(cudaMemcpyAsync(x.data(), X.memptr(), sizeof(float) * X.n_elem,
      cudaMemcpyHostToDevice, workspace.stream()), "upload float32 cross-product X");
    check_cuda(cudaMemcpyAsync(y.data(), Y.memptr(), sizeof(float) * Y.n_elem,
      cudaMemcpyHostToDevice, workspace.stream()), "upload float32 cross-product Y");
    check_cuda(cudaStreamSynchronize(workspace.stream()), "synchronize cross-product upload");
  }
  int n, p, q;
  CudaFloatWorkspace workspace;
  CudaLDADeviceBuffer<float> x, y;
  std::mutex mutex;
};

CudaFloatCrossproduct::CudaFloatCrossproduct(const arma::fmat& X, const arma::fmat& Y)
    : impl_(new Impl(X, Y)) {}
CudaFloatCrossproduct::~CudaFloatCrossproduct() = default;
arma::fmat CudaFloatCrossproduct::multiply(const arma::fmat& B, bool transpose) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const int inner = transpose ? impl_->p : impl_->q;
  const int rows = transpose ? impl_->q : impl_->p;
  if (B.n_rows != static_cast<arma::uword>(inner) ||
      B.n_cols > static_cast<arma::uword>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("CUDA float32 cross-product: invalid matrix dimensions");
  }
  arma::fmat result(rows, B.n_cols);
  if (B.n_cols == 0) return result;
  const int width = static_cast<int>(B.n_cols);
  auto& ws = impl_->workspace;
  ws.b.ensure(B.n_elem);
  ws.z.ensure(static_cast<std::size_t>(impl_->n) * width);
  ws.c.ensure(result.n_elem);
  check_cuda(cudaMemcpyAsync(ws.b.data(), B.memptr(), sizeof(float) * B.n_elem,
    cudaMemcpyHostToDevice, ws.stream()), "upload float32 cross-product sketch");
  const float one = 1.0f, zero = 0.0f;
  check_cublas(cublasSgemm(ws.blas(), CUBLAS_OP_N, CUBLAS_OP_N,
    impl_->n, width, inner, &one, transpose ? impl_->x.data() : impl_->y.data(),
    impl_->n, ws.b.data(), inner, &zero, ws.z.data(), impl_->n),
    "float32 cross-product inner GEMM");
  check_cublas(cublasSgemm(ws.blas(), CUBLAS_OP_T, CUBLAS_OP_N,
    rows, width, impl_->n, &one, transpose ? impl_->y.data() : impl_->x.data(),
    impl_->n, ws.z.data(), impl_->n, &zero, ws.c.data(), rows),
    "float32 cross-product outer GEMM");
  check_cuda(cudaMemcpyAsync(result.memptr(), ws.c.data(), sizeof(float) * result.n_elem,
    cudaMemcpyDeviceToHost, ws.stream()), "download float32 cross-product result");
  check_cuda(cudaStreamSynchronize(ws.stream()), "synchronize float32 cross-product");
  return result;
}

void CudaFloatCrossproduct::deflate(const arma::fvec& v) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (v.n_elem != static_cast<arma::uword>(impl_->p)) {
    throw std::runtime_error("CUDA float32 factor deflation: invalid direction size");
  }
  auto& ws = impl_->workspace;
  ws.b.ensure(v.n_elem);
  ws.z.ensure(impl_->n);
  check_cuda(cudaMemcpyAsync(ws.b.data(), v.memptr(), sizeof(float) * v.n_elem,
    cudaMemcpyHostToDevice, ws.stream()), "upload float32 factor deflation direction");
  const float one = 1.0f, zero = 0.0f, negative = -1.0f;
  check_cublas(cublasSgemv(ws.blas(), CUBLAS_OP_N, impl_->n, impl_->p,
    &one, impl_->x.data(), impl_->n, ws.b.data(), 1, &zero, ws.z.data(), 1),
    "float32 factor deflation score");
  check_cublas(cublasSger(ws.blas(), impl_->n, impl_->p, &negative,
    ws.z.data(), 1, ws.b.data(), 1, impl_->x.data(), impl_->n),
    "float32 factor deflation update");
  check_cuda(cudaStreamSynchronize(ws.stream()), "synchronize float32 factor deflation");
}

arma::fmat CudaFloatCrossproduct::left_factor() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  arma::fmat X(impl_->n, impl_->p);
  auto& ws = impl_->workspace;
  check_cuda(cudaMemcpyAsync(X.memptr(), impl_->x.data(), sizeof(float) * X.n_elem,
    cudaMemcpyDeviceToHost, ws.stream()), "download float32 left factor");
  check_cuda(cudaStreamSynchronize(ws.stream()), "synchronize float32 left factor");
  return X;
}

void CudaFloatCrossproduct::orthonormalize(arma::fmat& B) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (B.n_rows > static_cast<arma::uword>(std::numeric_limits<int>::max()) ||
      B.n_cols > B.n_rows || B.n_cols == 0) {
    throw std::runtime_error("Invalid float32 cross-product QR dimensions");
  }
  auto& ws = impl_->workspace;
  ws.a.ensure(B.n_elem);
  check_cuda(cudaMemcpyAsync(ws.a.data(), B.memptr(), sizeof(float) * B.n_elem,
    cudaMemcpyHostToDevice, ws.stream()), "upload float32 cross-product QR");
  ws.orthonormalize(ws.a.data(), static_cast<int>(B.n_rows),
                    static_cast<int>(B.n_cols), "float32 cross-product QR");
  check_cuda(cudaMemcpyAsync(B.memptr(), ws.a.data(), sizeof(float) * B.n_elem,
    cudaMemcpyDeviceToHost, ws.stream()), "download float32 cross-product QR");
  check_cuda(cudaStreamSynchronize(ws.stream()), "synchronize float32 cross-product QR");
}

void cuda_reset_workspace() {
  g_workspace.reset();
  g_float_workspace.reset();
  g_lda_float_workspace.reset();
}

Mat cuda_matrix_multiply(const Mat& A, const Mat& B) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  if (A.n_cols != B.n_rows) {
    throw std::runtime_error("cuda_matrix_multiply: non-conformable matrices");
  }
  if (A.n_rows > static_cast<arma::uword>(std::numeric_limits<int>::max()) ||
      A.n_cols > static_cast<arma::uword>(std::numeric_limits<int>::max()) ||
      B.n_cols > static_cast<arma::uword>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("cuda_matrix_multiply: matrix dimension exceeds CUDA int limits");
  }

  const int m = static_cast<int>(A.n_rows);
  const int k = static_cast<int>(A.n_cols);
  const int n = static_cast<int>(B.n_cols);
  if (m < 1 || k < 1 || n < 1) {
    return Mat(A.n_rows, B.n_cols, arma::fill::zeros);
  }

  double* dA = nullptr;
  double* dB = nullptr;
  double* dC = nullptr;
  cublasHandle_t handle = nullptr;

  auto cleanup = [&]() {
    if (dA) cudaFree(dA);
    if (dB) cudaFree(dB);
    if (dC) cudaFree(dC);
    if (handle) cublasDestroy(handle);
  };

  try {
    const size_t bytes_A = sizeof(double) * static_cast<size_t>(m) * static_cast<size_t>(k);
    const size_t bytes_B = sizeof(double) * static_cast<size_t>(k) * static_cast<size_t>(n);
    const size_t bytes_C = sizeof(double) * static_cast<size_t>(m) * static_cast<size_t>(n);
    Mat C(A.n_rows, B.n_cols, arma::fill::none);

    check_cublas(cublasCreate(&handle), "cublasCreate(cuda_matrix_multiply)");
    check_cuda(cudaMalloc(&dA, bytes_A), "cudaMalloc(cuda_matrix_multiply A)");
    check_cuda(cudaMalloc(&dB, bytes_B), "cudaMalloc(cuda_matrix_multiply B)");
    check_cuda(cudaMalloc(&dC, bytes_C), "cudaMalloc(cuda_matrix_multiply C)");
    check_cuda(cudaMemcpy(dA, A.memptr(), bytes_A, cudaMemcpyHostToDevice), "cudaMemcpy(cuda_matrix_multiply A)");
    check_cuda(cudaMemcpy(dB, B.memptr(), bytes_B, cudaMemcpyHostToDevice), "cudaMemcpy(cuda_matrix_multiply B)");

    const double one = 1.0;
    const double zero = 0.0;
    check_cublas(
      cublasDgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m,
        n,
        k,
        &one,
        dA,
        m,
        dB,
        k,
        &zero,
        dC,
        m
      ),
      "cublasDgemm(cuda_matrix_multiply)"
    );
    check_cuda(cudaMemcpy(C.memptr(), dC, bytes_C, cudaMemcpyDeviceToHost), "cudaMemcpy(cuda_matrix_multiply C)");

    cleanup();
    return C;
  } catch (...) {
    cleanup();
    throw;
  }
}

arma::fmat cuda_matrix_multiply_float(const arma::fmat& A,
                                      const arma::fmat& B,
                                      const bool transpose_left,
                                      const bool transpose_right) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }

  const arma::uword a_rows = transpose_left ? A.n_cols : A.n_rows;
  const arma::uword a_cols = transpose_left ? A.n_rows : A.n_cols;
  const arma::uword b_rows = transpose_right ? B.n_cols : B.n_rows;
  const arma::uword b_cols = transpose_right ? B.n_rows : B.n_cols;
  if (a_cols != b_rows) {
    throw std::runtime_error("cuda_matrix_multiply_float: non-conformable matrices");
  }
  if (A.n_rows > static_cast<arma::uword>(std::numeric_limits<int>::max()) ||
      A.n_cols > static_cast<arma::uword>(std::numeric_limits<int>::max()) ||
      B.n_rows > static_cast<arma::uword>(std::numeric_limits<int>::max()) ||
      B.n_cols > static_cast<arma::uword>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("cuda_matrix_multiply_float: matrix dimension exceeds CUDA int limits");
  }

  const int m = static_cast<int>(a_rows);
  const int k = static_cast<int>(a_cols);
  const int n = static_cast<int>(b_cols);
  if (m < 1 || k < 1 || n < 1) {
    return arma::fmat(a_rows, b_cols, arma::fill::zeros);
  }

  const size_t elems_A = static_cast<size_t>(A.n_elem);
  const size_t elems_B = static_cast<size_t>(B.n_elem);
  const size_t elems_C = static_cast<size_t>(m) * static_cast<size_t>(n);
  arma::fmat C(a_rows, b_cols, arma::fill::none);
  CudaFloatWorkspace& ws = g_float_workspace;
  ws.initialize();
  ws.a.ensure(elems_A);
  ws.b.ensure(elems_B);
  ws.c.ensure(elems_C);
  check_cuda(cudaMemcpyAsync(
    ws.a.data(), A.memptr(), sizeof(float) * elems_A,
    cudaMemcpyHostToDevice, ws.stream()
  ), "cudaMemcpyAsync(cuda_matrix_multiply_float A)");
  check_cuda(cudaMemcpyAsync(
    ws.b.data(), B.memptr(), sizeof(float) * elems_B,
    cudaMemcpyHostToDevice, ws.stream()
  ), "cudaMemcpyAsync(cuda_matrix_multiply_float B)");

  const float one = 1.0f;
  const float zero = 0.0f;
  check_cublas(
    cublasSgemm(
      ws.blas(),
      transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N,
      transpose_right ? CUBLAS_OP_T : CUBLAS_OP_N,
      m,
      n,
      k,
      &one,
      ws.a.data(),
      static_cast<int>(A.n_rows),
      ws.b.data(),
      static_cast<int>(B.n_rows),
      &zero,
      ws.c.data(),
      m
    ),
    "cublasSgemm(cuda_matrix_multiply_float)"
  );
  check_cuda(cudaMemcpyAsync(
    C.memptr(), ws.c.data(), sizeof(float) * elems_C,
    cudaMemcpyDeviceToHost, ws.stream()
  ), "cudaMemcpyAsync(cuda_matrix_multiply_float C)");
  check_cuda(
    cudaStreamSynchronize(ws.stream()),
    "cudaStreamSynchronize(cuda_matrix_multiply_float)"
  );
  return C;
}

Mat cuda_thin_qr(const Mat& A) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  if (A.n_rows < A.n_cols) {
    throw std::runtime_error("cuda_thin_qr: expected nrow(A) >= ncol(A)");
  }
  if (A.n_rows > static_cast<arma::uword>(std::numeric_limits<int>::max()) ||
      A.n_cols > static_cast<arma::uword>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("cuda_thin_qr: matrix dimension exceeds CUDA int limits");
  }

  const int m = static_cast<int>(A.n_rows);
  const int n = static_cast<int>(A.n_cols);
  if (m < 1 || n < 1) {
    return Mat(A.n_rows, A.n_cols, arma::fill::zeros);
  }

  double* dA = nullptr;
  double* dTau = nullptr;
  double* dWork = nullptr;
  int* dInfo = nullptr;
  cusolverDnHandle_t solver = nullptr;

  auto cleanup = [&]() {
    if (dA) cudaFree(dA);
    if (dTau) cudaFree(dTau);
    if (dWork) cudaFree(dWork);
    if (dInfo) cudaFree(dInfo);
    if (solver) cusolverDnDestroy(solver);
  };

  try {
    const size_t bytes_A = sizeof(double) * static_cast<size_t>(m) * static_cast<size_t>(n);
    Mat Q(A.n_rows, A.n_cols, arma::fill::none);
    check_cusolver(cusolverDnCreate(&solver), "cusolverDnCreate(cuda_thin_qr)");
    check_cuda(cudaMalloc(&dA, bytes_A), "cudaMalloc(cuda_thin_qr A)");
    check_cuda(cudaMalloc(&dTau, sizeof(double) * static_cast<size_t>(n)), "cudaMalloc(cuda_thin_qr tau)");
    check_cuda(cudaMalloc(&dInfo, sizeof(int)), "cudaMalloc(cuda_thin_qr info)");
    check_cuda(cudaMemcpy(dA, A.memptr(), bytes_A, cudaMemcpyHostToDevice), "cudaMemcpy(cuda_thin_qr A)");

    int lwork_geqrf = 0;
    int lwork_orgqr = 0;
    check_cusolver(
      cusolverDnDgeqrf_bufferSize(solver, m, n, dA, m, &lwork_geqrf),
      "cusolverDnDgeqrf_bufferSize(cuda_thin_qr)"
    );
    check_cusolver(
      cusolverDnDorgqr_bufferSize(solver, m, n, n, dA, m, dTau, &lwork_orgqr),
      "cusolverDnDorgqr_bufferSize(cuda_thin_qr)"
    );
    const int lwork = std::max(lwork_geqrf, lwork_orgqr);
    check_cuda(cudaMalloc(&dWork, sizeof(double) * static_cast<size_t>(std::max(lwork, 1))), "cudaMalloc(cuda_thin_qr work)");

    check_cusolver(
      cusolverDnDgeqrf(solver, m, n, dA, m, dTau, dWork, lwork, dInfo),
      "cusolverDnDgeqrf(cuda_thin_qr)"
    );
    int info = 0;
    check_cuda(cudaMemcpy(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(cuda_thin_qr geqrf info)");
    if (info != 0) {
      throw std::runtime_error("cuda_thin_qr: geqrf failed");
    }

    check_cusolver(
      cusolverDnDorgqr(solver, m, n, n, dA, m, dTau, dWork, lwork, dInfo),
      "cusolverDnDorgqr(cuda_thin_qr)"
    );
    check_cuda(cudaMemcpy(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(cuda_thin_qr orgqr info)");
    if (info != 0) {
      throw std::runtime_error("cuda_thin_qr: orgqr failed");
    }
    check_cuda(cudaMemcpy(Q.memptr(), dA, bytes_A, cudaMemcpyDeviceToHost), "cudaMemcpy(cuda_thin_qr Q)");

    cleanup();
    return Q;
  } catch (...) {
    cleanup();
    throw;
  }
}

bool cuda_rsvd_prefer_block_gpu(int m, int n, int l, int power_iters) {
  if (!cuda_runtime_available()) {
    return false;
  }
  const int min_m = env_int_or("FASTPLS_FAST_GPU_MIN_M", 512, 1, 1 << 30);
  const int min_n = env_int_or("FASTPLS_FAST_GPU_MIN_N", 16, 1, 1 << 30);
  const int min_work = env_int_or("FASTPLS_FAST_GPU_MIN_WORK", 200000, 1, 1 << 30);
  const long long work = static_cast<long long>(m) * static_cast<long long>(n) * static_cast<long long>(std::max(l, 1));
  return (m >= min_m) && (n >= min_n) && (work >= min_work) && (power_iters >= 1);
}

bool cuda_rsvd_use_resident_public(int m, int n, int l, int power_iters) {
  if (!cuda_runtime_available()) {
    return false;
  }
  const int enabled = env_int_or("FASTPLS_CUDA_RSVD_RESIDENT_PUBLIC", 0, 0, 1);
  if (enabled == 0) {
    return false;
  }
  const int min_l = env_int_or("FASTPLS_CUDA_RSVD_RESIDENT_MIN_L", 48, 1, 4096);
  const int min_work = env_int_or("FASTPLS_CUDA_RSVD_RESIDENT_MIN_WORK", 1000000, 1, 1 << 30);
  const long long work = static_cast<long long>(m) * static_cast<long long>(n) * static_cast<long long>(std::max(l, 1));
  return cuda_rsvd_prefer_block_gpu(m, n, l, power_iters) &&
    (l >= min_l) &&
    (work >= min_work);
}

void cuda_rsvd_sample_y(
  const double* hA,
  int m,
  int n,
  const double* hOmega,
  int l,
  int power_iters,
  double* hY
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  g_workspace.sample_y(hA, m, n, hOmega, l, power_iters, hY);
}

void cuda_rsvd_sample_y_float(
  const float* hA,
  int m,
  int n,
  const float* hOmega,
  int l,
  int power_iters,
  float* hY
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  if (m <= 0 || n <= 0 || l <= 0) {
    throw std::runtime_error("cuda_rsvd_sample_y_float received non-positive dimensions");
  }

  const size_t elems_A = static_cast<size_t>(m) * static_cast<size_t>(n);
  const size_t elems_Omega = static_cast<size_t>(n) * static_cast<size_t>(l);
  const size_t elems_Y = static_cast<size_t>(m) * static_cast<size_t>(l);
  const size_t elems_Z = static_cast<size_t>(n) * static_cast<size_t>(l);
  CudaFloatWorkspace& ws = g_float_workspace;
  ws.initialize();
  ws.a.ensure(elems_A);
  ws.b.ensure(elems_Omega);
  ws.c.ensure(elems_Y);
  ws.z.ensure(elems_Z);
  check_cuda(cudaMemcpyAsync(
    ws.a.data(), hA, sizeof(float) * elems_A,
    cudaMemcpyHostToDevice, ws.stream()
  ), "cudaMemcpyAsync(float A)");
  check_cuda(cudaMemcpyAsync(
    ws.b.data(), hOmega, sizeof(float) * elems_Omega,
    cudaMemcpyHostToDevice, ws.stream()
  ), "cudaMemcpyAsync(float Omega)");

  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublas(
    cublasSgemm(
      ws.blas(), CUBLAS_OP_N, CUBLAS_OP_N, m, l, n,
      &alpha, ws.a.data(), m, ws.b.data(), n, &beta, ws.c.data(), m
    ),
    "cublasSgemm(float A*Omega)"
  );
  const int q = std::max(power_iters, 0);
  for (int i = 0; i < q; ++i) {
    ws.orthonormalize(
      ws.c.data(), m, l, "float32 CUDA rSVD left sketch"
    );
    check_cublas(
      cublasSgemm(
        ws.blas(), CUBLAS_OP_T, CUBLAS_OP_N, n, l, m,
        &alpha, ws.a.data(), m, ws.c.data(), m, &beta, ws.z.data(), n
      ),
      "cublasSgemm(float A^T*Y)"
    );
    ws.orthonormalize(
      ws.z.data(), n, l, "float32 CUDA rSVD right sketch"
    );
    check_cublas(
      cublasSgemm(
        ws.blas(), CUBLAS_OP_N, CUBLAS_OP_N, m, l, n,
        &alpha, ws.a.data(), m, ws.z.data(), n, &beta, ws.c.data(), m
      ),
      "cublasSgemm(float A*Z)"
    );
  }
  check_cuda(cudaMemcpyAsync(
    hY, ws.c.data(), sizeof(float) * elems_Y,
    cudaMemcpyDeviceToHost, ws.stream()
  ), "cudaMemcpyAsync(float Y)");
  check_cuda(
    cudaStreamSynchronize(ws.stream()),
    "cudaStreamSynchronize(float rSVD)"
  );
}

void cuda_rsvd_set_resident_matrix(
  const double* hA,
  int m,
  int n
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  g_workspace.set_matrix(hA, m, n);
}

void cuda_rsvd_refresh_left_block(
  const double* hA,
  int m,
  int n,
  unsigned int seed,
  int l,
  int power_iters,
  double* hY
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  g_workspace.refresh_left_block(hA, m, n, seed, l, power_iters, hY);
}

void cuda_rsvd_refresh_left_block_u(
  const double* hA,
  int m,
  int n,
  unsigned int seed,
  int l,
  int k,
  int power_iters,
  double* hUblock,
  double* hSvals
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  g_workspace.refresh_left_block_u(hA, m, n, seed, l, k, power_iters, hUblock, hSvals);
}

void cuda_rsvd_refresh_left_block_u_resident(
  int m,
  int n,
  int l,
  int k,
  unsigned int seed,
  int power_iters,
  double* hUblock,
  double* hSvals
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  g_workspace.refresh_left_block_u_resident(m, n, l, k, seed, power_iters, hUblock, hSvals);
}

void cuda_rsvd_project_left_row(
  const double* hV,
  int m,
  int n,
  double* hVS
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  g_workspace.project_left_row(hV, m, n, hVS);
}

void cuda_rsvd_deflate_left_rank1(
  const double* hV,
  const double* hVS,
  int m,
  int n
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  g_workspace.deflate_left_rank1(hV, hVS, m, n);
}

void cuda_simpls_fast_set_training_matrices(
  const double* hX,
  int n,
  int p,
  const double* hY,
  int m,
  bool fit,
  bool form_crossprod
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  g_workspace.set_pls_training_matrices(hX, n, p, hY, m, fit, form_crossprod);
}

void cuda_simpls_fast_begin_device_loop(
  int n,
  int p,
  int m,
  int max_ncomp,
  bool fit,
  bool store_B
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  g_workspace.simpls_fast_begin_device_loop(n, p, m, max_ncomp, fit, store_B);
}

void cuda_simpls_fast_refresh_block_resident(
  int p,
  int m,
  int l,
  int k,
  unsigned int seed,
  int power_iters,
  double* hSvals
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  g_workspace.simpls_fast_refresh_block_resident(
    p, m, l, k, seed, power_iters, hSvals
  );
}

void cuda_simpls_fast_refresh_block_implicit_resident(
  int n,
  int p,
  int m,
  int l,
  int k,
  int prev_v_cols,
  unsigned int seed,
  int power_iters,
  double* hSvals
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  g_workspace.simpls_fast_refresh_block_implicit_resident(
    n,
    p,
    m,
    l,
    k,
    prev_v_cols,
    seed,
    power_iters,
    hSvals
  );
}

bool cuda_simpls_fast_append_component_from_block(
  int n,
  int p,
  int m,
  int a_idx,
  int col_idx,
  int prev_v_cols,
  bool reorth_v,
  bool fit,
  bool materialized_crossprod
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  return g_workspace.simpls_fast_append_component_from_block(
    n,
    p,
    m,
    a_idx,
    col_idx,
    prev_v_cols,
    reorth_v,
    fit,
    materialized_crossprod
  );
}

void cuda_simpls_fast_copy_rr(
  double* hRR,
  int p,
  int max_ncomp
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  g_workspace.simpls_fast_copy_rr(hRR, p, max_ncomp);
}

void cuda_simpls_fast_copy_qq(
  double* hQQ,
  int m,
  int max_ncomp
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  g_workspace.simpls_fast_copy_qq(hQQ, m, max_ncomp);
}

void cuda_simpls_fast_copy_bcur(
  double* hB,
  int p,
  int m
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  g_workspace.simpls_fast_copy_bcur(hB, p, m);
}

void cuda_simpls_fast_copy_yfit(
  double* hYfit,
  int n,
  int m
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  g_workspace.simpls_fast_copy_yfit(hYfit, n, m);
}

void cuda_simpls_fast_component_stats(
  const double* hR,
  int n,
  int p,
  int m,
  double* hT,
  double* hP,
  double* hQ,
  double* hTnorm
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  g_workspace.simpls_fast_component_stats(hR, n, p, m, hT, hP, hQ, hTnorm);
}

void cuda_simpls_fast_rank1_fit_update(
  const double* hT,
  int n,
  const double* hQ,
  int m,
  double* hDelta
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  g_workspace.simpls_fast_rank1_fit_update(hT, n, hQ, m, hDelta);
}

SVDResult cuda_rsvd_resident_svd(const Mat& A, int k, const SVDOptions& opt) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  return g_workspace.resident_svd(A, k, opt);
}

PLSSVDGPUResult cuda_plssvd_fit(
  const Mat& Xtrain,
  const Mat& Ytrain,
  const arma::ivec& ncomp,
  bool fit,
  const SVDOptions& opt,
  bool return_B
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  return g_workspace.plssvd_fit(Xtrain, Ytrain, ncomp, fit, opt, return_B);
}

PLSSVDGPUResult cuda_plssvd_fit_implicit_xprod(
  const Mat& Xtrain,
  const Mat& Ytrain,
  const arma::ivec& ncomp,
  bool fit,
  const SVDOptions& opt,
  bool return_B
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  return g_workspace.plssvd_fit_implicit_xprod(Xtrain, Ytrain, ncomp, fit, opt, return_B);
}

arma::cube cuda_flash_lowrank_predict(
  const Mat& Xtest,
  const Mat& R,
  const arma::cube& W_latent,
  const arma::rowvec& mY,
  const arma::ivec& ncomp
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  if (Xtest.n_cols != R.n_rows) {
    throw std::runtime_error("cuda_flash_lowrank_predict: Xtest and R dimensions do not match");
  }
  if (W_latent.n_slices < static_cast<arma::uword>(ncomp.n_elem)) {
    throw std::runtime_error("cuda_flash_lowrank_predict: W_latent has too few slices");
  }
  if (mY.n_elem != W_latent.n_cols) {
    throw std::runtime_error("cuda_flash_lowrank_predict: mY length does not match response dimension");
  }

  const int n = static_cast<int>(Xtest.n_rows);
  const int p = static_cast<int>(Xtest.n_cols);
  const int kmax = static_cast<int>(R.n_cols);
  const int m = static_cast<int>(W_latent.n_cols);
  const int nslices = static_cast<int>(ncomp.n_elem);
  if (n < 1 || p < 1 || kmax < 1 || m < 1 || nslices < 1) {
    throw std::runtime_error("cuda_flash_lowrank_predict: empty input");
  }

  cublasHandle_t handle = nullptr;
  check_cublas(cublasCreate(&handle), "cublasCreate(cuda_flash_lowrank_predict)");

  double* dX = nullptr;
  double* dR = nullptr;
  double* dScores = nullptr;
  double* dW = nullptr;
  double* dY = nullptr;
  double* dMY = nullptr;
  double* dOnes = nullptr;

  auto cleanup = [&]() {
    if (dX) cudaFree(dX);
    if (dR) cudaFree(dR);
    if (dScores) cudaFree(dScores);
    if (dW) cudaFree(dW);
    if (dY) cudaFree(dY);
    if (dMY) cudaFree(dMY);
    if (dOnes) cudaFree(dOnes);
    if (handle) cublasDestroy(handle);
  };

  try {
    const size_t bytes_X = static_cast<size_t>(n) * static_cast<size_t>(p) * sizeof(double);
    const size_t bytes_R = static_cast<size_t>(p) * static_cast<size_t>(kmax) * sizeof(double);
    const size_t bytes_scores = static_cast<size_t>(n) * static_cast<size_t>(kmax) * sizeof(double);
    const size_t bytes_W = static_cast<size_t>(kmax) * static_cast<size_t>(m) * sizeof(double);
    const size_t bytes_Y = static_cast<size_t>(n) * static_cast<size_t>(m) * sizeof(double);
    const size_t bytes_mY = static_cast<size_t>(m) * sizeof(double);
    const size_t bytes_ones = static_cast<size_t>(n) * sizeof(double);

    check_cuda(cudaMalloc(&dX, bytes_X), "cudaMalloc(flash dX)");
    check_cuda(cudaMalloc(&dR, bytes_R), "cudaMalloc(flash dR)");
    check_cuda(cudaMalloc(&dScores, bytes_scores), "cudaMalloc(flash dScores)");
    check_cuda(cudaMalloc(&dW, bytes_W), "cudaMalloc(flash dW)");
    check_cuda(cudaMalloc(&dY, bytes_Y), "cudaMalloc(flash dY)");
    check_cuda(cudaMalloc(&dMY, bytes_mY), "cudaMalloc(flash dMY)");
    check_cuda(cudaMalloc(&dOnes, bytes_ones), "cudaMalloc(flash dOnes)");

    check_cuda(cudaMemcpy(dX, Xtest.memptr(), bytes_X, cudaMemcpyHostToDevice), "cudaMemcpy(flash Xtest)");
    check_cuda(cudaMemcpy(dR, R.memptr(), bytes_R, cudaMemcpyHostToDevice), "cudaMemcpy(flash R)");
    check_cuda(cudaMemcpy(dMY, mY.memptr(), bytes_mY, cudaMemcpyHostToDevice), "cudaMemcpy(flash mY)");
    std::vector<double> hOnes(static_cast<size_t>(n), 1.0);
    check_cuda(cudaMemcpy(dOnes, hOnes.data(), bytes_ones, cudaMemcpyHostToDevice), "cudaMemcpy(flash ones)");

    arma::cube Ypred(n, m, nslices, arma::fill::none);
    const double one = 1.0;
    const double zero = 0.0;

    for (int s = 0; s < nslices; ++s) {
      int mc = ncomp(s);
      if (mc < 1) mc = 1;
      if (mc > kmax) mc = kmax;
      if (mc > static_cast<int>(W_latent.n_rows)) {
        throw std::runtime_error("cuda_flash_lowrank_predict: ncomp exceeds W_latent rank");
      }

      check_cublas(
        cublasDgemm(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          n,
          mc,
          p,
          &one,
          dX,
          n,
          dR,
          p,
          &zero,
          dScores,
          n
        ),
        "cublasDgemm(flash scores)"
      );

      arma::mat Wslice = W_latent.slice(static_cast<arma::uword>(s));
      check_cuda(
        cudaMemcpy(dW, Wslice.memptr(), bytes_W, cudaMemcpyHostToDevice),
        "cudaMemcpy(flash W)"
      );
      check_cublas(
        cublasDgemm(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          n,
          m,
          mc,
          &one,
          dScores,
          n,
          dW,
          kmax,
          &zero,
          dY,
          n
        ),
        "cublasDgemm(flash Y)"
      );

      check_cublas(
        cublasDger(
          handle,
          n,
          m,
          &one,
          dOnes,
          1,
          dMY,
          1,
          dY,
          n
        ),
        "cublasDger(flash add mY)"
      );
      check_cuda(
        cudaMemcpy(Ypred.slice(static_cast<arma::uword>(s)).memptr(), dY, bytes_Y, cudaMemcpyDeviceToHost),
        "cudaMemcpy(flash Ypred)"
      );
    }

    cleanup();
    return Ypred;
  } catch (...) {
    cleanup();
    throw;
  }
}

arma::imat cuda_flash_lowrank_predict_classes(
  const Mat& Xtest,
  const Mat& R,
  const arma::cube& W_latent,
  const arma::rowvec& mY,
  const arma::ivec& ncomp
) {
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  if (Xtest.n_cols != R.n_rows) {
    throw std::runtime_error("cuda_flash_lowrank_predict_classes: Xtest and R dimensions do not match");
  }
  if (W_latent.n_slices < static_cast<arma::uword>(ncomp.n_elem)) {
    throw std::runtime_error("cuda_flash_lowrank_predict_classes: W_latent has too few slices");
  }
  if (mY.n_elem != W_latent.n_cols) {
    throw std::runtime_error("cuda_flash_lowrank_predict_classes: mY length does not match response dimension");
  }

  const int n = static_cast<int>(Xtest.n_rows);
  const int p = static_cast<int>(Xtest.n_cols);
  const int kmax = static_cast<int>(R.n_cols);
  const int m = static_cast<int>(W_latent.n_cols);
  const int nslices = static_cast<int>(ncomp.n_elem);
  if (n < 1 || p < 1 || kmax < 1 || m < 1 || nslices < 1) {
    throw std::runtime_error("cuda_flash_lowrank_predict_classes: empty input");
  }

  cublasHandle_t handle = nullptr;
  check_cublas(cublasCreate(&handle), "cublasCreate(cuda_flash_lowrank_predict_classes)");

  double* dX = nullptr;
  double* dR = nullptr;
  double* dScores = nullptr;
  double* dW = nullptr;
  double* dY = nullptr;

  auto cleanup = [&]() {
    if (dX) cudaFree(dX);
    if (dR) cudaFree(dR);
    if (dScores) cudaFree(dScores);
    if (dW) cudaFree(dW);
    if (dY) cudaFree(dY);
    if (handle) cublasDestroy(handle);
  };

  try {
    const size_t bytes_X = static_cast<size_t>(n) * static_cast<size_t>(p) * sizeof(double);
    const size_t bytes_R = static_cast<size_t>(p) * static_cast<size_t>(kmax) * sizeof(double);
    const size_t bytes_scores = static_cast<size_t>(n) * static_cast<size_t>(kmax) * sizeof(double);
    const size_t bytes_W = static_cast<size_t>(kmax) * static_cast<size_t>(m) * sizeof(double);
    const size_t bytes_Y = static_cast<size_t>(n) * static_cast<size_t>(m) * sizeof(double);

    check_cuda(cudaMalloc(&dX, bytes_X), "cudaMalloc(class dX)");
    check_cuda(cudaMalloc(&dR, bytes_R), "cudaMalloc(class dR)");
    check_cuda(cudaMalloc(&dScores, bytes_scores), "cudaMalloc(class dScores)");
    check_cuda(cudaMalloc(&dW, bytes_W), "cudaMalloc(class dW)");
    check_cuda(cudaMalloc(&dY, bytes_Y), "cudaMalloc(class dY)");

    check_cuda(cudaMemcpy(dX, Xtest.memptr(), bytes_X, cudaMemcpyHostToDevice), "cudaMemcpy(class Xtest)");
    check_cuda(cudaMemcpy(dR, R.memptr(), bytes_R, cudaMemcpyHostToDevice), "cudaMemcpy(class R)");

    const double one = 1.0;
    const double zero = 0.0;
    arma::imat out(n, nslices, arma::fill::none);

    for (int s = 0; s < nslices; ++s) {
      int mc = ncomp(s);
      if (mc < 1) mc = 1;
      if (mc > kmax) mc = kmax;
      if (mc > static_cast<int>(W_latent.n_rows)) {
        throw std::runtime_error("cuda_flash_lowrank_predict_classes: ncomp exceeds W_latent rank");
      }

      check_cublas(
        cublasDgemm(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          n,
          mc,
          p,
          &one,
          dX,
          n,
          dR,
          p,
          &zero,
          dScores,
          n
        ),
        "cublasDgemm(class scores)"
      );

      arma::mat Wslice = W_latent.slice(static_cast<arma::uword>(s));
      check_cuda(cudaMemcpy(dW, Wslice.memptr(), bytes_W, cudaMemcpyHostToDevice), "cudaMemcpy(class W)");
      check_cublas(
        cublasDgemm(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          n,
          m,
          mc,
          &one,
          dScores,
          n,
          dW,
          kmax,
          &zero,
          dY,
          n
        ),
        "cublasDgemm(class Y)"
      );
      arma::mat yblock(n, m, arma::fill::none);
      check_cuda(cudaMemcpy(yblock.memptr(), dY, bytes_Y, cudaMemcpyDeviceToHost), "cudaMemcpy(class Y)");
      yblock.each_row() += mY;
      for (int i = 0; i < n; ++i) {
        out(i, s) = static_cast<int>(yblock.row(static_cast<arma::uword>(i)).index_max()) + 1;
      }
    }

    cleanup();
    return out;
  } catch (...) {
    cleanup();
    throw;
  }
}

bool cuda_lda_native_available() {
#ifdef FASTPLS_HAS_CUDA_KERNELS
  return cuda_runtime_available();
#else
  return false;
#endif
}

std::vector<LDAGPUModel> cuda_lda_train_prefix(
  const Mat& Ttrain,
  const arma::ivec& y,
  int n_classes,
  const arma::ivec& ncomp,
  double ridge
) {
  (void)ridge; // Retained in the internal ABI; regularization is deterministic.
#ifndef FASTPLS_HAS_CUDA_KERNELS
  (void)Ttrain;
  (void)y;
  (void)n_classes;
  (void)ncomp;
  (void)ridge;
  throw std::runtime_error("CUDA LDA kernels not compiled");
#else
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  if (Ttrain.n_rows == 0 || Ttrain.n_cols == 0) {
    throw std::runtime_error("cuda_lda_train_prefix requires a non-empty score matrix");
  }
  if (static_cast<arma::uword>(y.n_elem) != Ttrain.n_rows) {
    throw std::runtime_error("cuda_lda_train_prefix requires one label per training row");
  }
  if (n_classes < 2) {
    throw std::runtime_error("cuda_lda_train_prefix requires at least two classes");
  }
  if (ncomp.n_elem < 1) {
    throw std::runtime_error("cuda_lda_train_prefix requires at least one component count");
  }
  if (Ttrain.n_rows > static_cast<arma::uword>(std::numeric_limits<int>::max()) ||
      Ttrain.n_cols > static_cast<arma::uword>(std::numeric_limits<int>::max()) ||
      n_classes > std::numeric_limits<int>::max()) {
    throw std::runtime_error("cuda_lda_train_prefix matrix dimension exceeds CUDA int limits");
  }

  int kmax = 0;
  for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
    if (ncomp(i) > kmax) kmax = ncomp(i);
  }
  if (kmax < 1 || kmax > static_cast<int>(Ttrain.n_cols)) {
    throw std::runtime_error("cuda_lda_train_prefix component counts must be in 1..ncol(Ttrain)");
  }

  const int n = static_cast<int>(Ttrain.n_rows);
  const int C = n_classes;
  const size_t bytes_T = sizeof(double) * static_cast<size_t>(n) * static_cast<size_t>(kmax);
  const size_t bytes_y = sizeof(int) * static_cast<size_t>(n);
  const size_t bytes_counts = sizeof(double) * static_cast<size_t>(C);
  const size_t bytes_means = sizeof(double) * static_cast<size_t>(C) * static_cast<size_t>(kmax);
  const size_t bytes_pooled = sizeof(double) * static_cast<size_t>(kmax) * static_cast<size_t>(kmax);
  const size_t bytes_rhs_max = sizeof(double) * static_cast<size_t>(kmax) * static_cast<size_t>(C);
  const size_t bytes_linear_max = sizeof(double) * static_cast<size_t>(C) * static_cast<size_t>(kmax);

  double* dT = nullptr;
  int* dLabels = nullptr;
  double* dCounts = nullptr;
  double* dMeans = nullptr;
  double* dPooled = nullptr;
  double* dCov = nullptr;
  double* dRhs = nullptr;
  double* dLinear = nullptr;
  double* dConstants = nullptr;
  double* dLambda = nullptr;
  double* dWork = nullptr;
  int* dInfo = nullptr;
  cublasHandle_t handle = nullptr;
  cusolverDnHandle_t solver = nullptr;
  cudaStream_t stream = nullptr;

  auto cleanup = [&]() {
    if (dT) cudaFree(dT);
    if (dLabels) cudaFree(dLabels);
    if (dCounts) cudaFree(dCounts);
    if (dMeans) cudaFree(dMeans);
    if (dPooled) cudaFree(dPooled);
    if (dCov) cudaFree(dCov);
    if (dRhs) cudaFree(dRhs);
    if (dLinear) cudaFree(dLinear);
    if (dConstants) cudaFree(dConstants);
    if (dLambda) cudaFree(dLambda);
    if (dWork) cudaFree(dWork);
    if (dInfo) cudaFree(dInfo);
    if (solver) cusolverDnDestroy(solver);
    if (handle) cublasDestroy(handle);
    if (stream) cudaStreamDestroy(stream);
  };

  try {
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate(cuda_lda_train_prefix)");
    check_cublas(cublasCreate(&handle), "cublasCreate(cuda_lda_train_prefix)");
    check_cublas(cublasSetStream(handle, stream), "cublasSetStream(cuda_lda_train_prefix)");
    check_cusolver(cusolverDnCreate(&solver), "cusolverDnCreate(cuda_lda_train_prefix)");
    check_cusolver(cusolverDnSetStream(solver, stream), "cusolverDnSetStream(cuda_lda_train_prefix)");

    check_cuda(cudaMalloc(&dT, bytes_T), "cudaMalloc(cuda_lda T)");
    check_cuda(cudaMalloc(&dLabels, bytes_y), "cudaMalloc(cuda_lda labels)");
    check_cuda(cudaMalloc(&dCounts, bytes_counts), "cudaMalloc(cuda_lda counts)");
    check_cuda(cudaMalloc(&dMeans, bytes_means), "cudaMalloc(cuda_lda means)");
    check_cuda(cudaMalloc(&dPooled, bytes_pooled), "cudaMalloc(cuda_lda pooled)");
    check_cuda(cudaMalloc(&dCov, bytes_pooled), "cudaMalloc(cuda_lda cov)");
    check_cuda(cudaMalloc(&dRhs, bytes_rhs_max), "cudaMalloc(cuda_lda rhs)");
    check_cuda(cudaMalloc(&dLinear, bytes_linear_max), "cudaMalloc(cuda_lda linear)");
    check_cuda(cudaMalloc(&dConstants, bytes_counts), "cudaMalloc(cuda_lda constants)");
    check_cuda(cudaMalloc(&dLambda, sizeof(double)), "cudaMalloc(cuda_lda lambda)");
    check_cuda(cudaMalloc(&dInfo, sizeof(int)), "cudaMalloc(cuda_lda info)");

    check_cuda(cudaMemcpyAsync(dT, Ttrain.memptr(), bytes_T, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(cuda_lda T)");
    std::vector<int> labels_host(static_cast<size_t>(n));
    arma::vec counts_host(C, arma::fill::zeros);
    for (int i = 0; i < n; ++i) {
      const int cls = y(static_cast<arma::uword>(i)) - 1;
      if (cls < 0 || cls >= C) {
        throw std::runtime_error("cuda_lda_train_prefix labels must be encoded as 1..n_classes");
      }
      labels_host[static_cast<size_t>(i)] = cls + 1;
      counts_host(cls) += 1.0;
    }
    for (int c = 0; c < C; ++c) {
      if (counts_host(c) <= 0.0) {
        throw std::runtime_error("cuda_lda_train_prefix received an empty class");
      }
    }
    check_cuda(cudaMemcpyAsync(dLabels, labels_host.data(), bytes_y, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(cuda_lda labels)");
    check_cuda(cudaMemcpyAsync(dCounts, counts_host.memptr(), bytes_counts, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(cuda_lda counts)");

    const double one = 1.0;
    const double zero = 0.0;
    fastpls_cuda_lda_label_sums(dT, dLabels, n, kmax, C, dMeans, stream);
    check_kernel_launch("fastpls_cuda_lda_label_sums");
    fastpls_cuda_lda_means(dMeans, dCounts, kmax, C, stream);
    check_kernel_launch("fastpls_cuda_lda_means");

    check_cublas(
      cublasDgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        kmax,
        kmax,
        n,
        &one,
        dT,
        n,
        dT,
        n,
        &zero,
        dPooled,
        kmax
      ),
      "cublasDgemm(cuda_lda TtT)"
    );
    fastpls_cuda_lda_pooled(dPooled, dMeans, dCounts, n, kmax, C, stream);
    check_kernel_launch("fastpls_cuda_lda_pooled");

    arma::mat means_host(C, kmax, arma::fill::zeros);
    check_cuda(cudaMemcpyAsync(means_host.memptr(), dMeans, bytes_means, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(cuda_lda means_host)");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(cuda_lda means)");

    std::vector<LDAGPUModel> out;
    out.reserve(static_cast<size_t>(ncomp.n_elem));
    for (arma::uword idx = 0; idx < ncomp.n_elem; ++idx) {
      const int kk = ncomp(idx);
      if (kk < 1 || kk > kmax) {
        throw std::runtime_error("cuda_lda_train_prefix component counts must be in 1..max(ncomp)");
      }
      fastpls_cuda_lda_means_to_rhs(dMeans, dRhs, kmax, kk, C, stream);
      check_kernel_launch("fastpls_cuda_lda_means_to_rhs");

      int lwork = 0;
      check_cusolver(
        cusolverDnDpotrf_bufferSize(
          solver,
          CUBLAS_FILL_MODE_LOWER,
          kk,
          dCov,
          kk,
          &lwork
        ),
        "cusolverDnDpotrf_bufferSize(cuda_lda)"
      );
      if (dWork) {
        check_cuda(cudaFree(dWork), "cudaFree(cuda_lda old work)");
        dWork = nullptr;
      }
      check_cuda(cudaMalloc(&dWork, sizeof(double) * static_cast<size_t>(std::max(lwork, 1))), "cudaMalloc(cuda_lda work)");
      int info = 0;
      double selected_rho = 0.0;
      bool factorized = false;
      constexpr double ridge_grid[] = {1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2};
      for (double rho : ridge_grid) {
        fastpls_cuda_lda_copy_cov(dPooled, dCov, kmax, kk, stream);
        fastpls_cuda_lda_add_ridge(dCov, kk, rho, dLambda, stream);
        check_kernel_launch("fastpls_cuda_lda_regularize");
        check_cusolver(cusolverDnDpotrf(
          solver, CUBLAS_FILL_MODE_LOWER, kk, dCov, kk, dWork, lwork, dInfo
        ), "cusolverDnDpotrf(cuda_lda)");
        check_cuda(cudaMemcpyAsync(
          &info, dInfo, sizeof(int), cudaMemcpyDeviceToHost, stream
        ), "cudaMemcpyAsync(cuda_lda potrf info)");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(cuda_lda potrf)");
        if (info == 0) {
          factorized = true;
          selected_rho = rho;
          break;
        }
      }
      if (!factorized) {
        throw std::runtime_error(
          "cusolverDnDpotrf(cuda_lda) failed at every regularization level"
        );
      }
      check_cusolver(
        cusolverDnDpotrs(
          solver,
          CUBLAS_FILL_MODE_LOWER,
          kk,
          C,
          dCov,
          kk,
          dRhs,
          kk,
          dInfo
        ),
        "cusolverDnDpotrs(cuda_lda)"
      );
      check_cuda(cudaMemcpyAsync(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(cuda_lda potrs info)");
      fastpls_cuda_lda_finalize_linear(dRhs, dMeans, dCounts, dLinear, dConstants, n, kmax, kk, C, stream);
      check_kernel_launch("fastpls_cuda_lda_finalize_linear");
      check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(cuda_lda solve/finalize)");
      if (info != 0) {
        throw std::runtime_error("cusolverDnDpotrs(cuda_lda) returned non-zero info");
      }

      LDAGPUModel model;
      model.means = means_host.cols(0, kk - 1);
      model.linear.set_size(C, kk);
      model.constants.set_size(C);
      model.priors = counts_host / static_cast<double>(n);
      check_cuda(cudaMemcpy(model.linear.memptr(), dLinear, sizeof(double) * static_cast<size_t>(C) * static_cast<size_t>(kk), cudaMemcpyDeviceToHost), "cudaMemcpy(cuda_lda linear)");
      check_cuda(cudaMemcpy(model.constants.memptr(), dConstants, bytes_counts, cudaMemcpyDeviceToHost), "cudaMemcpy(cuda_lda constants)");
      check_cuda(cudaMemcpy(&model.ridge, dLambda, sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(cuda_lda lambda)");
      model.relative_ridge = selected_rho;
      out.push_back(model);
    }

    cleanup();
    return out;
  } catch (...) {
    cleanup();
    throw;
  }
#endif
}

std::vector<LDAFloatGPUModel> cuda_lda_train_prefix_float(
  const arma::fmat& Ttrain,
  const arma::ivec& y,
  int n_classes,
  const arma::ivec& ncomp
) {
#ifndef FASTPLS_HAS_CUDA_KERNELS
  (void)Ttrain;
  (void)y;
  (void)n_classes;
  (void)ncomp;
  throw std::runtime_error("CUDA LDA kernels not compiled");
#else
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  if (Ttrain.n_rows == 0 || Ttrain.n_cols == 0 || ncomp.n_elem < 1) {
    throw std::runtime_error("cuda_lda_train_prefix_float requires non-empty scores and components");
  }
  if (y.n_elem != Ttrain.n_rows || n_classes < 2) {
    throw std::runtime_error("cuda_lda_train_prefix_float received inconsistent labels");
  }
  int kmax = 0;
  for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
    kmax = std::max(kmax, static_cast<int>(ncomp(i)));
  }
  if (kmax < 1 || kmax > static_cast<int>(Ttrain.n_cols)) {
    throw std::runtime_error("cuda_lda_train_prefix_float component count exceeds score rank");
  }

  const int n = static_cast<int>(Ttrain.n_rows);
  const int C = n_classes;
  std::vector<int> labels(static_cast<std::size_t>(n));
  arma::fvec counts(C, arma::fill::zeros);
  for (int i = 0; i < n; ++i) {
    const int cls = y(static_cast<arma::uword>(i));
    if (cls < 1 || cls > C) {
      throw std::runtime_error("cuda_lda_train_prefix_float labels must be encoded as 1..n_classes");
    }
    labels[static_cast<std::size_t>(i)] = cls;
    counts(static_cast<arma::uword>(cls - 1)) += 1.0f;
  }
  if (arma::any(counts <= 0.0f)) {
    throw std::runtime_error("cuda_lda_train_prefix_float received an empty class");
  }

  CudaLDAFloatWorkspace& ws = g_lda_float_workspace;
  ws.initialize();
  cudaStream_t stream = ws.stream();
  cublasHandle_t blas = ws.blas();
  cusolverDnHandle_t solver = ws.solver();
  ws.train_scores.ensure(static_cast<std::size_t>(n) * static_cast<std::size_t>(kmax));
  ws.labels.ensure(static_cast<std::size_t>(n));
  ws.counts.ensure(static_cast<std::size_t>(C));
  ws.means.ensure(static_cast<std::size_t>(C) * static_cast<std::size_t>(kmax));
  ws.pooled.ensure(static_cast<std::size_t>(kmax) * static_cast<std::size_t>(kmax));
  ws.covariance.ensure(static_cast<std::size_t>(kmax) * static_cast<std::size_t>(kmax));
  ws.rhs.ensure(static_cast<std::size_t>(kmax) * static_cast<std::size_t>(C));
  ws.linear.ensure(static_cast<std::size_t>(C) * static_cast<std::size_t>(kmax));
  ws.constants.ensure(static_cast<std::size_t>(C));
  ws.lambda.ensure(1);
  ws.info.ensure(1);

  check_cuda(cudaMemcpyAsync(
    ws.train_scores.data(), Ttrain.memptr(),
    sizeof(float) * static_cast<std::size_t>(n) * static_cast<std::size_t>(kmax),
    cudaMemcpyHostToDevice, stream
  ), "cudaMemcpyAsync(float32 LDA scores)");
  check_cuda(cudaMemcpyAsync(
    ws.labels.data(), labels.data(), sizeof(int) * labels.size(),
    cudaMemcpyHostToDevice, stream
  ), "cudaMemcpyAsync(float32 LDA labels)");
  check_cuda(cudaMemcpyAsync(
    ws.counts.data(), counts.memptr(), sizeof(float) * static_cast<std::size_t>(C),
    cudaMemcpyHostToDevice, stream
  ), "cudaMemcpyAsync(float32 LDA counts)");

  fastpls_cuda_lda_label_sums_float(
    ws.train_scores.data(), ws.labels.data(), n, kmax, C, ws.means.data(), stream
  );
  check_kernel_launch("fastpls_cuda_lda_label_sums_float");
  fastpls_cuda_lda_means_float(ws.means.data(), ws.counts.data(), kmax, C, stream);
  check_kernel_launch("fastpls_cuda_lda_means_float");
  const float one = 1.0f;
  const float zero = 0.0f;
  check_cublas(cublasSgemm(
    blas, CUBLAS_OP_T, CUBLAS_OP_N, kmax, kmax, n,
    &one, ws.train_scores.data(), n, ws.train_scores.data(), n,
    &zero, ws.pooled.data(), kmax
  ), "cublasSgemm(float32 LDA TtT)");
  fastpls_cuda_lda_pooled_float(
    ws.pooled.data(), ws.means.data(), ws.counts.data(), n, kmax, C, stream
  );
  check_kernel_launch("fastpls_cuda_lda_pooled_float");

  arma::fmat means_host(C, kmax, arma::fill::zeros);
  check_cuda(cudaMemcpyAsync(
    means_host.memptr(), ws.means.data(), sizeof(float) * means_host.n_elem,
    cudaMemcpyDeviceToHost, stream
  ), "cudaMemcpyAsync(float32 LDA means)");
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(float32 LDA moments)");

  constexpr float ridge_grid[] = {
    1e-8f, 1e-6f, 1e-5f, 1e-4f, 1e-3f, 1e-2f
  };
  std::vector<LDAFloatGPUModel> out;
  out.reserve(static_cast<std::size_t>(ncomp.n_elem));
  for (arma::uword index = 0; index < ncomp.n_elem; ++index) {
    const int kk = ncomp(index);
    if (kk < 1 || kk > kmax) {
      throw std::runtime_error("cuda_lda_train_prefix_float component count is out of range");
    }
    int lwork = 0;
    check_cusolver(cusolverDnSpotrf_bufferSize(
      solver, CUBLAS_FILL_MODE_LOWER, kk, ws.covariance.data(), kk, &lwork
    ), "cusolverDnSpotrf_bufferSize(float32 LDA)");
    ws.solver_work.ensure(static_cast<std::size_t>(std::max(1, lwork)));

    bool factorized = false;
    float selected_rho = 0.0f;
    int info = 0;
    for (float rho : ridge_grid) {
      fastpls_cuda_lda_copy_cov_float(
        ws.pooled.data(), ws.covariance.data(), kmax, kk, stream
      );
      fastpls_cuda_lda_add_ridge_float(
        ws.covariance.data(), kk, rho, ws.lambda.data(), stream
      );
      check_kernel_launch("float32 LDA covariance regularization");
      check_cusolver(cusolverDnSpotrf(
        solver, CUBLAS_FILL_MODE_LOWER, kk, ws.covariance.data(), kk,
        ws.solver_work.data(), lwork, ws.info.data()
      ), "cusolverDnSpotrf(float32 LDA)");
      check_cuda(cudaMemcpyAsync(
        &info, ws.info.data(), sizeof(int), cudaMemcpyDeviceToHost, stream
      ), "cudaMemcpyAsync(float32 LDA potrf info)");
      check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(float32 LDA potrf)");
      if (info == 0) {
        factorized = true;
        selected_rho = rho;
        break;
      }
    }
    if (!factorized) {
      throw std::runtime_error("float32 CUDA PLS-LDA Cholesky failed at every regularization level");
    }

    fastpls_cuda_lda_means_to_rhs_float(
      ws.means.data(), ws.rhs.data(), kmax, kk, C, stream
    );
    check_cusolver(cusolverDnSpotrs(
      solver, CUBLAS_FILL_MODE_LOWER, kk, C, ws.covariance.data(), kk,
      ws.rhs.data(), kk, ws.info.data()
    ), "cusolverDnSpotrs(float32 LDA)");
    fastpls_cuda_lda_finalize_linear_float(
      ws.rhs.data(), ws.means.data(), ws.counts.data(), ws.linear.data(),
      ws.constants.data(), n, kmax, kk, C, stream
    );
    check_kernel_launch("fastpls_cuda_lda_finalize_linear_float");
    check_cuda(cudaMemcpyAsync(
      &info, ws.info.data(), sizeof(int), cudaMemcpyDeviceToHost, stream
    ), "cudaMemcpyAsync(float32 LDA potrs info)");

    LDAFloatGPUModel model;
    model.means = means_host.cols(0, static_cast<arma::uword>(kk - 1));
    model.linear.set_size(C, kk);
    model.constants.set_size(C);
    model.priors = counts / static_cast<float>(n);
    model.relative_ridge = selected_rho;
    check_cuda(cudaMemcpyAsync(
      model.linear.memptr(), ws.linear.data(), sizeof(float) * model.linear.n_elem,
      cudaMemcpyDeviceToHost, stream
    ), "cudaMemcpyAsync(float32 LDA linear)");
    check_cuda(cudaMemcpyAsync(
      model.constants.memptr(), ws.constants.data(), sizeof(float) * model.constants.n_elem,
      cudaMemcpyDeviceToHost, stream
    ), "cudaMemcpyAsync(float32 LDA constants)");
    check_cuda(cudaMemcpyAsync(
      &model.ridge, ws.lambda.data(), sizeof(float), cudaMemcpyDeviceToHost, stream
    ), "cudaMemcpyAsync(float32 LDA lambda)");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(float32 LDA model)");
    if (info != 0) {
      throw std::runtime_error("cusolverDnSpotrs(float32 LDA) returned non-zero info");
    }
    out.push_back(std::move(model));
  }
  return out;
#endif
}

LDAFloatPrediction cuda_lda_predict_float(
  const arma::fmat& Ttest,
  const arma::fmat& linear,
  const arma::frowvec& constants,
  bool return_scores
) {
#ifndef FASTPLS_HAS_CUDA_KERNELS
  (void)Ttest;
  (void)linear;
  (void)constants;
  throw std::runtime_error("CUDA LDA kernels not compiled");
#else
  if (!cuda_runtime_available() || Ttest.n_rows == 0 || Ttest.n_cols == 0) {
    throw std::runtime_error("cuda_lda_predict_float requires CUDA and non-empty scores");
  }
  if (Ttest.n_cols != linear.n_cols || constants.n_elem != linear.n_rows) {
    throw std::runtime_error("cuda_lda_predict_float dimensions do not match the model");
  }
  const int n = static_cast<int>(Ttest.n_rows);
  const int k = static_cast<int>(Ttest.n_cols);
  const int C = static_cast<int>(linear.n_rows);
  CudaLDAFloatWorkspace& ws = g_lda_float_workspace;
  ws.initialize();
  cudaStream_t stream = ws.stream();
  ws.test_latent.ensure(Ttest.n_elem);
  ws.linear.ensure(linear.n_elem);
  ws.constants.ensure(constants.n_elem);
  ws.test_scores.ensure(static_cast<std::size_t>(n) * static_cast<std::size_t>(C));
  ws.predictions.ensure(static_cast<std::size_t>(n));
  check_cuda(cudaMemcpyAsync(
    ws.test_latent.data(), Ttest.memptr(), sizeof(float) * Ttest.n_elem,
    cudaMemcpyHostToDevice, stream
  ), "cudaMemcpyAsync(float32 LDA Ttest)");
  check_cuda(cudaMemcpyAsync(
    ws.linear.data(), linear.memptr(), sizeof(float) * linear.n_elem,
    cudaMemcpyHostToDevice, stream
  ), "cudaMemcpyAsync(float32 LDA prediction linear)");
  check_cuda(cudaMemcpyAsync(
    ws.constants.data(), constants.memptr(), sizeof(float) * constants.n_elem,
    cudaMemcpyHostToDevice, stream
  ), "cudaMemcpyAsync(float32 LDA prediction constants)");
  const float one = 1.0f;
  const float zero = 0.0f;
  check_cublas(cublasSgemm(
    ws.blas(), CUBLAS_OP_N, CUBLAS_OP_T, n, C, k,
    &one, ws.test_latent.data(), n, ws.linear.data(), C,
    &zero, ws.test_scores.data(), n
  ), "cublasSgemm(float32 LDA prediction)");
  fastpls_cuda_lda_score_argmax_float(
    ws.test_scores.data(), ws.constants.data(), ws.predictions.data(), n, C, stream
  );
  check_kernel_launch("fastpls_cuda_lda_score_argmax_float");
  LDAFloatPrediction out;
  out.pred.resize(static_cast<std::size_t>(n));
  if (return_scores) {
    out.scores.set_size(Ttest.n_rows, linear.n_rows);
  }
  check_cuda(cudaMemcpyAsync(
    out.pred.data(), ws.predictions.data(), sizeof(int) * static_cast<std::size_t>(n),
    cudaMemcpyDeviceToHost, stream
  ), "cudaMemcpyAsync(float32 LDA predictions)");
  if (return_scores) {
    check_cuda(cudaMemcpyAsync(
      out.scores.memptr(), ws.test_scores.data(), sizeof(float) * out.scores.n_elem,
      cudaMemcpyDeviceToHost, stream
    ), "cudaMemcpyAsync(float32 LDA prediction scores)");
  }
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(float32 LDA prediction)");
  return out;
#endif
}

std::vector<LDAGPUModel> cuda_lda_project_train_prefix(
  const Mat& Xtrain,
  const Mat& R,
  const arma::rowvec& offset,
  const arma::ivec& y,
  int n_classes,
  const arma::ivec& ncomp,
  double ridge
) {
  (void)ridge; // Retained in the internal ABI; regularization is deterministic.
#ifndef FASTPLS_HAS_CUDA_KERNELS
  (void)Xtrain;
  (void)R;
  (void)offset;
  (void)y;
  (void)n_classes;
  (void)ncomp;
  (void)ridge;
  throw std::runtime_error("CUDA LDA kernels not compiled");
#else
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  if (Xtrain.n_rows == 0 || Xtrain.n_cols == 0 || R.n_rows == 0 || R.n_cols == 0) {
    throw std::runtime_error("cuda_lda_project_train_prefix requires non-empty X and projection matrices");
  }
  if (Xtrain.n_cols != R.n_rows) {
    throw std::runtime_error("cuda_lda_project_train_prefix X columns do not match projection rows");
  }
  if (static_cast<arma::uword>(y.n_elem) != Xtrain.n_rows) {
    throw std::runtime_error("cuda_lda_project_train_prefix requires one label per training row");
  }
  if (n_classes < 2) {
    throw std::runtime_error("cuda_lda_project_train_prefix requires at least two classes");
  }
  if (ncomp.n_elem < 1) {
    throw std::runtime_error("cuda_lda_project_train_prefix requires at least one component count");
  }
  if (offset.n_elem > 0 && offset.n_elem < R.n_cols) {
    throw std::runtime_error("cuda_lda_project_train_prefix offset length is smaller than projection dimension");
  }
  if (Xtrain.n_rows > static_cast<arma::uword>(std::numeric_limits<int>::max()) ||
      Xtrain.n_cols > static_cast<arma::uword>(std::numeric_limits<int>::max()) ||
      R.n_cols > static_cast<arma::uword>(std::numeric_limits<int>::max()) ||
      n_classes > std::numeric_limits<int>::max()) {
    throw std::runtime_error("cuda_lda_project_train_prefix matrix dimension exceeds CUDA int limits");
  }

  int kmax = 0;
  for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
    if (ncomp(i) > kmax) kmax = ncomp(i);
  }
  if (kmax < 1 || kmax > static_cast<int>(R.n_cols)) {
    throw std::runtime_error("cuda_lda_project_train_prefix component counts must be in 1..ncol(R)");
  }

  const int n = static_cast<int>(Xtrain.n_rows);
  const int p = static_cast<int>(Xtrain.n_cols);
  const int C = n_classes;
  const size_t bytes_X = sizeof(double) * static_cast<size_t>(n) * static_cast<size_t>(p);
  const size_t bytes_R = sizeof(double) * static_cast<size_t>(p) * static_cast<size_t>(kmax);
  const size_t bytes_T = sizeof(double) * static_cast<size_t>(n) * static_cast<size_t>(kmax);
  const size_t bytes_y = sizeof(int) * static_cast<size_t>(n);
  const size_t bytes_counts = sizeof(double) * static_cast<size_t>(C);
  const size_t bytes_means = sizeof(double) * static_cast<size_t>(C) * static_cast<size_t>(kmax);
  const size_t bytes_pooled = sizeof(double) * static_cast<size_t>(kmax) * static_cast<size_t>(kmax);
  const size_t bytes_rhs_max = sizeof(double) * static_cast<size_t>(kmax) * static_cast<size_t>(C);
  const size_t bytes_linear_max = sizeof(double) * static_cast<size_t>(C) * static_cast<size_t>(kmax);
  const size_t bytes_offsets = sizeof(double) * static_cast<size_t>(kmax);

  double* dX = nullptr;
  double* dR = nullptr;
  double* dT = nullptr;
  double* dOffsets = nullptr;
  int* dLabels = nullptr;
  double* dCounts = nullptr;
  double* dMeans = nullptr;
  double* dPooled = nullptr;
  double* dCov = nullptr;
  double* dRhs = nullptr;
  double* dLinear = nullptr;
  double* dConstants = nullptr;
  double* dLambda = nullptr;
  double* dWork = nullptr;
  int* dInfo = nullptr;
  cublasHandle_t handle = nullptr;
  cusolverDnHandle_t solver = nullptr;
  cudaStream_t stream = nullptr;

  auto cleanup = [&]() {
    if (dX) cudaFree(dX);
    if (dR) cudaFree(dR);
    if (dT) cudaFree(dT);
    if (dOffsets) cudaFree(dOffsets);
    if (dLabels) cudaFree(dLabels);
    if (dCounts) cudaFree(dCounts);
    if (dMeans) cudaFree(dMeans);
    if (dPooled) cudaFree(dPooled);
    if (dCov) cudaFree(dCov);
    if (dRhs) cudaFree(dRhs);
    if (dLinear) cudaFree(dLinear);
    if (dConstants) cudaFree(dConstants);
    if (dLambda) cudaFree(dLambda);
    if (dWork) cudaFree(dWork);
    if (dInfo) cudaFree(dInfo);
    if (solver) cusolverDnDestroy(solver);
    if (handle) cublasDestroy(handle);
    if (stream) cudaStreamDestroy(stream);
  };

  try {
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate(cuda_lda_project_train_prefix)");
    check_cublas(cublasCreate(&handle), "cublasCreate(cuda_lda_project_train_prefix)");
    check_cublas(cublasSetStream(handle, stream), "cublasSetStream(cuda_lda_project_train_prefix)");
    check_cusolver(cusolverDnCreate(&solver), "cusolverDnCreate(cuda_lda_project_train_prefix)");
    check_cusolver(cusolverDnSetStream(solver, stream), "cusolverDnSetStream(cuda_lda_project_train_prefix)");

    check_cuda(cudaMalloc(&dX, bytes_X), "cudaMalloc(cuda_lda_project X)");
    check_cuda(cudaMalloc(&dR, bytes_R), "cudaMalloc(cuda_lda_project R)");
    check_cuda(cudaMalloc(&dT, bytes_T), "cudaMalloc(cuda_lda_project T)");
    check_cuda(cudaMalloc(&dOffsets, bytes_offsets), "cudaMalloc(cuda_lda_project offsets)");
    check_cuda(cudaMalloc(&dLabels, bytes_y), "cudaMalloc(cuda_lda_project labels)");
    check_cuda(cudaMalloc(&dCounts, bytes_counts), "cudaMalloc(cuda_lda_project counts)");
    check_cuda(cudaMalloc(&dMeans, bytes_means), "cudaMalloc(cuda_lda_project means)");
    check_cuda(cudaMalloc(&dPooled, bytes_pooled), "cudaMalloc(cuda_lda_project pooled)");
    check_cuda(cudaMalloc(&dCov, bytes_pooled), "cudaMalloc(cuda_lda_project cov)");
    check_cuda(cudaMalloc(&dRhs, bytes_rhs_max), "cudaMalloc(cuda_lda_project rhs)");
    check_cuda(cudaMalloc(&dLinear, bytes_linear_max), "cudaMalloc(cuda_lda_project linear)");
    check_cuda(cudaMalloc(&dConstants, bytes_counts), "cudaMalloc(cuda_lda_project constants)");
    check_cuda(cudaMalloc(&dLambda, sizeof(double)), "cudaMalloc(cuda_lda_project lambda)");
    check_cuda(cudaMalloc(&dInfo, sizeof(int)), "cudaMalloc(cuda_lda_project info)");

    check_cuda(cudaMemcpyAsync(dX, Xtrain.memptr(), bytes_X, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(cuda_lda_project X)");
    check_cuda(cudaMemcpyAsync(dR, R.memptr(), bytes_R, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(cuda_lda_project R)");
    arma::rowvec offset_host(kmax, arma::fill::zeros);
    if (offset.n_elem >= static_cast<arma::uword>(kmax)) {
      offset_host = offset.subvec(0, static_cast<arma::uword>(kmax) - 1);
    }
    check_cuda(cudaMemcpyAsync(dOffsets, offset_host.memptr(), bytes_offsets, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(cuda_lda_project offsets)");

    const double one = 1.0;
    const double zero = 0.0;
    check_cublas(
      cublasDgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        n,
        kmax,
        p,
        &one,
        dX,
        n,
        dR,
        p,
        &zero,
        dT,
        n
      ),
      "cublasDgemm(cuda_lda_project latent)"
    );
    fastpls_cuda_lda_subtract_offsets(dT, dOffsets, n, kmax, stream);
    check_kernel_launch("fastpls_cuda_lda_subtract_offsets");

    std::vector<int> labels_host(static_cast<size_t>(n));
    arma::vec counts_host(C, arma::fill::zeros);
    for (int i = 0; i < n; ++i) {
      const int cls = y(static_cast<arma::uword>(i)) - 1;
      if (cls < 0 || cls >= C) {
        throw std::runtime_error("cuda_lda_project_train_prefix labels must be encoded as 1..n_classes");
      }
      labels_host[static_cast<size_t>(i)] = cls + 1;
      counts_host(cls) += 1.0;
    }
    for (int c = 0; c < C; ++c) {
      if (counts_host(c) <= 0.0) {
        throw std::runtime_error("cuda_lda_project_train_prefix received an empty class");
      }
    }
    check_cuda(cudaMemcpyAsync(dLabels, labels_host.data(), bytes_y, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(cuda_lda_project labels)");
    check_cuda(cudaMemcpyAsync(dCounts, counts_host.memptr(), bytes_counts, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(cuda_lda_project counts)");

    fastpls_cuda_lda_label_sums(dT, dLabels, n, kmax, C, dMeans, stream);
    check_kernel_launch("fastpls_cuda_lda_label_sums(project)");
    fastpls_cuda_lda_means(dMeans, dCounts, kmax, C, stream);
    check_kernel_launch("fastpls_cuda_lda_means(project)");

    check_cublas(
      cublasDgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        kmax,
        kmax,
        n,
        &one,
        dT,
        n,
        dT,
        n,
        &zero,
        dPooled,
        kmax
      ),
      "cublasDgemm(cuda_lda_project TtT)"
    );
    fastpls_cuda_lda_pooled(dPooled, dMeans, dCounts, n, kmax, C, stream);
    check_kernel_launch("fastpls_cuda_lda_pooled(project)");

    arma::mat means_host(C, kmax, arma::fill::zeros);
    check_cuda(cudaMemcpyAsync(means_host.memptr(), dMeans, bytes_means, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(cuda_lda_project means_host)");
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(cuda_lda_project means)");

    std::vector<LDAGPUModel> out;
    out.reserve(static_cast<size_t>(ncomp.n_elem));
    for (arma::uword idx = 0; idx < ncomp.n_elem; ++idx) {
      const int kk = ncomp(idx);
      if (kk < 1 || kk > kmax) {
        throw std::runtime_error("cuda_lda_project_train_prefix component counts must be in 1..max(ncomp)");
      }
      fastpls_cuda_lda_means_to_rhs(dMeans, dRhs, kmax, kk, C, stream);
      check_kernel_launch("fastpls_cuda_lda_means_to_rhs(project)");

      int lwork = 0;
      check_cusolver(
        cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_LOWER, kk, dCov, kk, &lwork),
        "cusolverDnDpotrf_bufferSize(cuda_lda_project)"
      );
      if (dWork) {
        check_cuda(cudaFree(dWork), "cudaFree(cuda_lda_project old work)");
        dWork = nullptr;
      }
      check_cuda(cudaMalloc(&dWork, sizeof(double) * static_cast<size_t>(std::max(lwork, 1))), "cudaMalloc(cuda_lda_project work)");
      int info = 0;
      double selected_rho = 0.0;
      bool factorized = false;
      constexpr double ridge_grid[] = {1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2};
      for (double rho : ridge_grid) {
        fastpls_cuda_lda_copy_cov(dPooled, dCov, kmax, kk, stream);
        fastpls_cuda_lda_add_ridge(dCov, kk, rho, dLambda, stream);
        check_kernel_launch("fastpls_cuda_lda_regularize(project)");
        check_cusolver(cusolverDnDpotrf(
          solver, CUBLAS_FILL_MODE_LOWER, kk, dCov, kk, dWork, lwork, dInfo
        ), "cusolverDnDpotrf(cuda_lda_project)");
        check_cuda(cudaMemcpyAsync(
          &info, dInfo, sizeof(int), cudaMemcpyDeviceToHost, stream
        ), "cudaMemcpyAsync(cuda_lda_project potrf info)");
        check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(cuda_lda_project potrf)");
        if (info == 0) {
          factorized = true;
          selected_rho = rho;
          break;
        }
      }
      if (!factorized) {
        throw std::runtime_error(
          "cusolverDnDpotrf(cuda_lda_project) failed at every regularization level"
        );
      }
      check_cusolver(
        cusolverDnDpotrs(solver, CUBLAS_FILL_MODE_LOWER, kk, C, dCov, kk, dRhs, kk, dInfo),
        "cusolverDnDpotrs(cuda_lda_project)"
      );
      check_cuda(cudaMemcpyAsync(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(cuda_lda_project potrs info)");
      fastpls_cuda_lda_finalize_linear(dRhs, dMeans, dCounts, dLinear, dConstants, n, kmax, kk, C, stream);
      check_kernel_launch("fastpls_cuda_lda_finalize_linear(project)");
      check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(cuda_lda_project solve/finalize)");
      if (info != 0) {
        throw std::runtime_error("cusolverDnDpotrs(cuda_lda_project) returned non-zero info");
      }

      LDAGPUModel model;
      model.means = means_host.cols(0, kk - 1);
      model.linear.set_size(C, kk);
      model.constants.set_size(C);
      model.priors = counts_host / static_cast<double>(n);
      check_cuda(cudaMemcpy(model.linear.memptr(), dLinear, sizeof(double) * static_cast<size_t>(C) * static_cast<size_t>(kk), cudaMemcpyDeviceToHost), "cudaMemcpy(cuda_lda_project linear)");
      check_cuda(cudaMemcpy(model.constants.memptr(), dConstants, bytes_counts, cudaMemcpyDeviceToHost), "cudaMemcpy(cuda_lda_project constants)");
      check_cuda(cudaMemcpy(&model.ridge, dLambda, sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy(cuda_lda_project lambda)");
      model.relative_ridge = selected_rho;
      out.push_back(model);
    }

    cleanup();
    return out;
  } catch (...) {
    cleanup();
    throw;
  }
#endif
}

Rcpp::List cuda_lda_predict(
  const Mat& Ttest,
  const Mat& linear,
  const arma::rowvec& constants,
  bool return_scores
) {
#ifndef FASTPLS_HAS_CUDA_KERNELS
  (void)Ttest;
  (void)linear;
  (void)constants;
  (void)return_scores;
  throw std::runtime_error("CUDA LDA kernels not compiled");
#else
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  if (Ttest.n_rows == 0 || Ttest.n_cols == 0) {
    throw std::runtime_error("cuda_lda_predict requires a non-empty score matrix");
  }
  if (Ttest.n_cols != linear.n_cols) {
    throw std::runtime_error("cuda_lda_predict score dimension does not match LDA model");
  }
  if (constants.n_elem != linear.n_rows) {
    throw std::runtime_error("cuda_lda_predict has inconsistent LDA constants");
  }
  if (Ttest.n_rows > static_cast<arma::uword>(std::numeric_limits<int>::max()) ||
      Ttest.n_cols > static_cast<arma::uword>(std::numeric_limits<int>::max()) ||
      linear.n_rows > static_cast<arma::uword>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("cuda_lda_predict matrix dimension exceeds CUDA int limits");
  }

  const int n = static_cast<int>(Ttest.n_rows);
  const int k = static_cast<int>(Ttest.n_cols);
  const int C = static_cast<int>(linear.n_rows);
  const size_t bytes_T = sizeof(double) * static_cast<size_t>(n) * static_cast<size_t>(k);
  const size_t bytes_linear = sizeof(double) * static_cast<size_t>(C) * static_cast<size_t>(k);
  const size_t bytes_constants = sizeof(double) * static_cast<size_t>(C);
  const size_t bytes_scores = sizeof(double) * static_cast<size_t>(n) * static_cast<size_t>(C);
  const size_t bytes_pred = sizeof(int) * static_cast<size_t>(n);

  double* dT = nullptr;
  double* dLinear = nullptr;
  double* dConstants = nullptr;
  double* dScores = nullptr;
  int* dPred = nullptr;
  cublasHandle_t handle = nullptr;
  cudaStream_t stream = nullptr;

  auto cleanup = [&]() {
    if (dT) cudaFree(dT);
    if (dLinear) cudaFree(dLinear);
    if (dConstants) cudaFree(dConstants);
    if (dScores) cudaFree(dScores);
    if (dPred) cudaFree(dPred);
    if (handle) cublasDestroy(handle);
    if (stream) cudaStreamDestroy(stream);
  };

  try {
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate(cuda_lda_predict)");
    check_cublas(cublasCreate(&handle), "cublasCreate(cuda_lda_predict)");
    check_cublas(cublasSetStream(handle, stream), "cublasSetStream(cuda_lda_predict)");
    check_cuda(cudaMalloc(&dT, bytes_T), "cudaMalloc(cuda_lda_predict T)");
    check_cuda(cudaMalloc(&dLinear, bytes_linear), "cudaMalloc(cuda_lda_predict linear)");
    check_cuda(cudaMalloc(&dConstants, bytes_constants), "cudaMalloc(cuda_lda_predict constants)");
    check_cuda(cudaMalloc(&dScores, bytes_scores), "cudaMalloc(cuda_lda_predict scores)");
    check_cuda(cudaMalloc(&dPred, bytes_pred), "cudaMalloc(cuda_lda_predict pred)");
    check_cuda(cudaMemcpyAsync(dT, Ttest.memptr(), bytes_T, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(cuda_lda_predict T)");
    check_cuda(cudaMemcpyAsync(dLinear, linear.memptr(), bytes_linear, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(cuda_lda_predict linear)");
    check_cuda(cudaMemcpyAsync(dConstants, constants.memptr(), bytes_constants, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(cuda_lda_predict constants)");

    const double one = 1.0;
    const double zero = 0.0;
    check_cublas(
      cublasDgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        n,
        C,
        k,
        &one,
        dT,
        n,
        dLinear,
        C,
        &zero,
        dScores,
        n
      ),
      "cublasDgemm(cuda_lda_predict scores)"
    );
    fastpls_cuda_lda_score_argmax(dScores, dConstants, dPred, n, C, stream);
    check_kernel_launch("fastpls_cuda_lda_score_argmax");

    Rcpp::IntegerVector pred(n);
    check_cuda(cudaMemcpyAsync(INTEGER(pred), dPred, bytes_pred, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(cuda_lda_predict pred)");
    if (return_scores) {
      arma::mat scores(Ttest.n_rows, linear.n_rows, arma::fill::none);
      check_cuda(cudaMemcpyAsync(scores.memptr(), dScores, bytes_scores, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(cuda_lda_predict scores)");
      check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(cuda_lda_predict)");
      cleanup();
      return Rcpp::List::create(
        Rcpp::Named("pred") = pred,
        Rcpp::Named("scores") = scores
      );
    }
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(cuda_lda_predict labels)");
    cleanup();
    return Rcpp::List::create(
      Rcpp::Named("pred") = pred
    );
  } catch (...) {
    cleanup();
    throw;
  }
#endif
}

Rcpp::List cuda_lda_project_predict(
  const Mat& Xtest,
  const Mat& R,
  const arma::rowvec& offset,
  const Mat& linear,
  const arma::rowvec& constants,
  bool return_scores
) {
#ifndef FASTPLS_HAS_CUDA_KERNELS
  (void)Xtest;
  (void)R;
  (void)offset;
  (void)linear;
  (void)constants;
  (void)return_scores;
  throw std::runtime_error("CUDA LDA kernels not compiled");
#else
  if (!cuda_runtime_available()) {
    throw std::runtime_error("CUDA runtime not available");
  }
  if (Xtest.n_rows == 0 || Xtest.n_cols == 0 || R.n_rows == 0 || R.n_cols == 0) {
    throw std::runtime_error("cuda_lda_project_predict requires non-empty X and projection matrices");
  }
  if (Xtest.n_cols != R.n_rows) {
    throw std::runtime_error("cuda_lda_project_predict X columns do not match projection rows");
  }
  if (R.n_cols != linear.n_cols) {
    throw std::runtime_error("cuda_lda_project_predict projection dimension does not match LDA model");
  }
  if (constants.n_elem != linear.n_rows) {
    throw std::runtime_error("cuda_lda_project_predict has inconsistent LDA constants");
  }
  if (offset.n_elem > 0 && offset.n_elem < R.n_cols) {
    throw std::runtime_error("cuda_lda_project_predict offset length is smaller than projection dimension");
  }
  if (Xtest.n_rows > static_cast<arma::uword>(std::numeric_limits<int>::max()) ||
      Xtest.n_cols > static_cast<arma::uword>(std::numeric_limits<int>::max()) ||
      R.n_cols > static_cast<arma::uword>(std::numeric_limits<int>::max()) ||
      linear.n_rows > static_cast<arma::uword>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("cuda_lda_project_predict matrix dimension exceeds CUDA int limits");
  }

  const int n = static_cast<int>(Xtest.n_rows);
  const int p = static_cast<int>(Xtest.n_cols);
  const int k = static_cast<int>(R.n_cols);
  const int C = static_cast<int>(linear.n_rows);
  const size_t bytes_X = sizeof(double) * static_cast<size_t>(n) * static_cast<size_t>(p);
  const size_t bytes_R = sizeof(double) * static_cast<size_t>(p) * static_cast<size_t>(k);
  const size_t bytes_T = sizeof(double) * static_cast<size_t>(n) * static_cast<size_t>(k);
  const size_t bytes_linear = sizeof(double) * static_cast<size_t>(C) * static_cast<size_t>(k);
  const size_t bytes_direct = sizeof(double) * static_cast<size_t>(p) * static_cast<size_t>(C);
  const size_t bytes_constants = sizeof(double) * static_cast<size_t>(C);
  const size_t bytes_scores = sizeof(double) * static_cast<size_t>(n) * static_cast<size_t>(C);
  const size_t bytes_pred = sizeof(int) * static_cast<size_t>(n);
  const double latent_ops = static_cast<double>(n) * static_cast<double>(k) *
    (static_cast<double>(p) + static_cast<double>(C));
  const double direct_ops = static_cast<double>(n) * static_cast<double>(p) *
    static_cast<double>(C);
  const bool use_direct = std::isfinite(latent_ops) &&
    std::isfinite(direct_ops) &&
    direct_ops < latent_ops;

  arma::rowvec adjusted_constants = constants;
  if (offset.n_elem >= R.n_cols) {
    const arma::rowvec offset_k = offset.subvec(0, R.n_cols - 1);
    adjusted_constants -= offset_k * linear.t();
  }

  double* dX = nullptr;
  double* dR = nullptr;
  double* dT = nullptr;
  double* dLinear = nullptr;
  double* dDirect = nullptr;
  double* dConstants = nullptr;
  double* dScores = nullptr;
  int* dPred = nullptr;
  cublasHandle_t handle = nullptr;
  cudaStream_t stream = nullptr;

  auto cleanup = [&]() {
    if (dX) cudaFree(dX);
    if (dR) cudaFree(dR);
    if (dT) cudaFree(dT);
    if (dLinear) cudaFree(dLinear);
    if (dDirect) cudaFree(dDirect);
    if (dConstants) cudaFree(dConstants);
    if (dScores) cudaFree(dScores);
    if (dPred) cudaFree(dPred);
    if (handle) cublasDestroy(handle);
    if (stream) cudaStreamDestroy(stream);
  };

  try {
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate(cuda_lda_project_predict)");
    check_cublas(cublasCreate(&handle), "cublasCreate(cuda_lda_project_predict)");
    check_cublas(cublasSetStream(handle, stream), "cublasSetStream(cuda_lda_project_predict)");
    check_cuda(cudaMalloc(&dX, bytes_X), "cudaMalloc(cuda_lda_project_predict X)");
    if (use_direct) {
      check_cuda(cudaMalloc(&dR, bytes_R), "cudaMalloc(cuda_lda_project_predict R)");
      check_cuda(cudaMalloc(&dLinear, bytes_linear), "cudaMalloc(cuda_lda_project_predict linear)");
      check_cuda(cudaMalloc(&dDirect, bytes_direct), "cudaMalloc(cuda_lda_project_predict direct)");
    } else {
      check_cuda(cudaMalloc(&dR, bytes_R), "cudaMalloc(cuda_lda_project_predict R)");
      check_cuda(cudaMalloc(&dT, bytes_T), "cudaMalloc(cuda_lda_project_predict T)");
      check_cuda(cudaMalloc(&dLinear, bytes_linear), "cudaMalloc(cuda_lda_project_predict linear)");
    }
    check_cuda(cudaMalloc(&dConstants, bytes_constants), "cudaMalloc(cuda_lda_project_predict constants)");
    check_cuda(cudaMalloc(&dScores, bytes_scores), "cudaMalloc(cuda_lda_project_predict scores)");
    check_cuda(cudaMalloc(&dPred, bytes_pred), "cudaMalloc(cuda_lda_project_predict pred)");

    check_cuda(cudaMemcpyAsync(dX, Xtest.memptr(), bytes_X, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(cuda_lda_project_predict X)");
    check_cuda(cudaMemcpyAsync(dConstants, adjusted_constants.memptr(), bytes_constants, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(cuda_lda_project_predict constants)");

    const double one = 1.0;
    const double zero = 0.0;
    if (use_direct) {
      check_cuda(cudaMemcpyAsync(dR, R.memptr(), bytes_R, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(cuda_lda_project_predict R)");
      check_cuda(cudaMemcpyAsync(dLinear, linear.memptr(), bytes_linear, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(cuda_lda_project_predict linear)");
      check_cublas(
        cublasDgemm(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_T,
          p,
          C,
          k,
          &one,
          dR,
          p,
          dLinear,
          C,
          &zero,
          dDirect,
          p
        ),
        "cublasDgemm(cuda_lda_project_predict direct weights)"
      );
      check_cublas(
        cublasDgemm(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          n,
          C,
          p,
          &one,
          dX,
          n,
          dDirect,
          p,
          &zero,
          dScores,
          n
        ),
        "cublasDgemm(cuda_lda_project_predict direct scores)"
      );
    } else {
      check_cuda(cudaMemcpyAsync(dR, R.memptr(), bytes_R, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(cuda_lda_project_predict R)");
      check_cuda(cudaMemcpyAsync(dLinear, linear.memptr(), bytes_linear, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(cuda_lda_project_predict linear)");
      check_cublas(
        cublasDgemm(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          n,
          k,
          p,
          &one,
          dX,
          n,
          dR,
          p,
          &zero,
          dT,
          n
        ),
        "cublasDgemm(cuda_lda_project_predict latent)"
      );
      check_cublas(
        cublasDgemm(
          handle,
          CUBLAS_OP_N,
          CUBLAS_OP_T,
          n,
          C,
          k,
          &one,
          dT,
          n,
          dLinear,
          C,
          &zero,
          dScores,
          n
        ),
        "cublasDgemm(cuda_lda_project_predict scores)"
      );
    }
    fastpls_cuda_lda_score_argmax(dScores, dConstants, dPred, n, C, stream);
    check_kernel_launch("fastpls_cuda_lda_score_argmax(project)");

    Rcpp::IntegerVector pred(n);
    check_cuda(cudaMemcpyAsync(INTEGER(pred), dPred, bytes_pred, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(cuda_lda_project_predict pred)");
    if (return_scores) {
      arma::mat scores(Xtest.n_rows, linear.n_rows, arma::fill::none);
      check_cuda(cudaMemcpyAsync(scores.memptr(), dScores, bytes_scores, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(cuda_lda_project_predict scores)");
      check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(cuda_lda_project_predict)");
      cleanup();
      return Rcpp::List::create(
        Rcpp::Named("pred") = pred,
        Rcpp::Named("scores") = scores
      );
    }
    check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(cuda_lda_project_predict labels)");
    cleanup();
    return Rcpp::List::create(Rcpp::Named("pred") = pred);
  } catch (...) {
    cleanup();
    throw;
  }
#endif
}

SVDResult truncated_svd_cuda_rsvd(const Mat& A, int k, const SVDOptions& opt) {
  const arma::uword max_rank = std::min(A.n_rows, A.n_cols);
  const arma::uword target = std::min<arma::uword>(
    max_rank,
    static_cast<arma::uword>(std::max(k, 1))
  );
  const arma::uword l = std::min<arma::uword>(
    max_rank,
    target + static_cast<arma::uword>(std::max(opt.oversample, 0))
  );

  if (l >= max_rank) {
    SVDOptions exact_opt = opt;
    exact_opt.method = Method::EXACT;
    return truncated_svd_cpu_exact(A, static_cast<int>(target), exact_opt);
  }

  if (cuda_rsvd_use_resident_public(
      static_cast<int>(A.n_rows),
      static_cast<int>(A.n_cols),
      static_cast<int>(l),
      std::max(opt.power_iters, 0))) {
    return cuda_rsvd_resident_svd(A, static_cast<int>(target), opt);
  }

  Mat Omega = gaussian_matrix_local(A.n_cols, l, opt.seed);
  Mat Y(A.n_rows, l);

  cuda_rsvd_sample_y(
    A.memptr(),
    static_cast<int>(A.n_rows),
    static_cast<int>(A.n_cols),
    Omega.memptr(),
    static_cast<int>(l),
    std::max(opt.power_iters, 0),
    Y.memptr()
  );

  return finalize_rsvd_from_sample(A, Y, static_cast<int>(target), opt.left_only);
}

} // namespace fastpls_svd

#else

#include <stdexcept>

namespace fastpls_svd {

struct CudaFloatCrossproduct::Impl {};
CudaFloatCrossproduct::CudaFloatCrossproduct(const arma::fmat&, const arma::fmat&) {
  throw std::runtime_error("CUDA backend not compiled; no CPU fallback is performed");
}
CudaFloatCrossproduct::~CudaFloatCrossproduct() = default;
arma::fmat CudaFloatCrossproduct::multiply(const arma::fmat&, bool) {
  throw std::runtime_error("CUDA backend not compiled; no CPU fallback is performed");
}
void CudaFloatCrossproduct::orthonormalize(arma::fmat&) {
  throw std::runtime_error("CUDA backend not compiled; no CPU fallback is performed");
}
void CudaFloatCrossproduct::deflate(const arma::fvec&) {
  throw std::runtime_error("CUDA backend not compiled; no CPU fallback is performed");
}
arma::fmat CudaFloatCrossproduct::left_factor() {
  throw std::runtime_error("CUDA backend not compiled; no CPU fallback is performed");
}

SVDResult truncated_svd_cuda_rsvd(const Mat&, int, const SVDOptions&) {
  throw std::runtime_error("CUDA backend not compiled");
}

SVDResult cuda_rsvd_resident_svd(const Mat&, int, const SVDOptions&) {
  throw std::runtime_error("CUDA backend not compiled");
}

PLSSVDGPUResult cuda_plssvd_fit(
  const Mat&,
  const Mat&,
  const arma::ivec&,
  bool,
  const SVDOptions&,
  bool
) {
  throw std::runtime_error("CUDA backend not compiled");
}

PLSSVDGPUResult cuda_plssvd_fit_implicit_xprod(
  const Mat&,
  const Mat&,
  const arma::ivec&,
  bool,
  const SVDOptions&,
  bool
) {
  throw std::runtime_error("CUDA backend not compiled");
}

arma::cube cuda_flash_lowrank_predict(
  const Mat&,
  const Mat&,
  const arma::cube&,
  const arma::rowvec&,
  const arma::ivec&
) {
  throw std::runtime_error("CUDA backend not compiled");
}

arma::imat cuda_flash_lowrank_predict_classes(
  const Mat&,
  const Mat&,
  const arma::cube&,
  const arma::rowvec&,
  const arma::ivec&
) {
  throw std::runtime_error("CUDA backend not compiled");
}

Mat cuda_matrix_multiply(const Mat&, const Mat&) {
  throw std::runtime_error("CUDA backend not compiled");
}

arma::fmat cuda_matrix_multiply_float(const arma::fmat&,
                                      const arma::fmat&,
                                      bool,
                                      bool) {
  throw std::runtime_error("CUDA backend not compiled");
}

Mat cuda_thin_qr(const Mat&) {
  throw std::runtime_error("CUDA backend not compiled");
}

std::vector<LDAGPUModel> cuda_lda_train_prefix(
  const Mat&,
  const arma::ivec&,
  int,
  const arma::ivec&,
  double
) {
  throw std::runtime_error("CUDA backend not compiled");
}

std::vector<LDAGPUModel> cuda_lda_project_train_prefix(
  const Mat&,
  const Mat&,
  const arma::rowvec&,
  const arma::ivec&,
  int,
  const arma::ivec&,
  double
) {
  throw std::runtime_error("CUDA backend not compiled");
}

Rcpp::List cuda_lda_predict(
  const Mat&,
  const Mat&,
  const arma::rowvec&,
  bool
) {
  throw std::runtime_error("CUDA backend not compiled");
}

Rcpp::List cuda_lda_project_predict(
  const Mat&,
  const Mat&,
  const arma::rowvec&,
  const Mat&,
  const arma::rowvec&,
  bool
) {
  throw std::runtime_error("CUDA backend not compiled");
}

std::vector<LDAFloatGPUModel> cuda_lda_train_prefix_float(
  const arma::fmat&,
  const arma::ivec&,
  int,
  const arma::ivec&
) {
  throw std::runtime_error("CUDA backend not compiled");
}

LDAFloatPrediction cuda_lda_predict_float(
  const arma::fmat&,
  const arma::fmat&,
  const arma::frowvec&,
  bool
) {
  throw std::runtime_error("CUDA backend not compiled");
}

bool cuda_lda_native_available() {
  return false;
}

bool cuda_runtime_available() {
  return false;
}

void cuda_reset_workspace() {
}

void cuda_rsvd_sample_y(
  const double*,
  int,
  int,
  const double*,
  int,
  int,
  double*
) {
  throw std::runtime_error("CUDA backend not compiled");
}

void cuda_rsvd_sample_y_float(
  const float*,
  int,
  int,
  const float*,
  int,
  int,
  float*
) {
  throw std::runtime_error("CUDA backend not compiled");
}

bool cuda_rsvd_prefer_block_gpu(int, int, int, int) {
  return false;
}

void cuda_rsvd_set_resident_matrix(
  const double*,
  int,
  int
) {
  throw std::runtime_error("CUDA backend not compiled");
}

void cuda_rsvd_refresh_left_block(
  const double*,
  int,
  int,
  unsigned int,
  int,
  int,
  double*
) {
  throw std::runtime_error("CUDA backend not compiled");
}

void cuda_rsvd_refresh_left_block_u(
  const double*,
  int,
  int,
  unsigned int,
  int,
  int,
  int,
  double*,
  double*
) {
  throw std::runtime_error("CUDA backend not compiled");
}

void cuda_rsvd_refresh_left_block_u_resident(
  int,
  int,
  int,
  int,
  unsigned int,
  int,
  double*,
  double*
) {
  throw std::runtime_error("CUDA backend not compiled");
}

void cuda_rsvd_project_left_row(
  const double*,
  int,
  int,
  double*
) {
  throw std::runtime_error("CUDA backend not compiled");
}

void cuda_rsvd_deflate_left_rank1(
  const double*,
  const double*,
  int,
  int
) {
  throw std::runtime_error("CUDA backend not compiled");
}

void cuda_simpls_fast_set_training_matrices(
  const double*,
  int,
  int,
  const double*,
  int,
  bool,
  bool
) {
  throw std::runtime_error("CUDA backend not compiled");
}

void cuda_simpls_fast_begin_device_loop(
  int,
  int,
  int,
  int,
  bool,
  bool
) {
  throw std::runtime_error("CUDA backend not compiled");
}

void cuda_simpls_fast_refresh_block_resident(
  int,
  int,
  int,
  int,
  unsigned int,
  int,
  double*
) {
  throw std::runtime_error("CUDA backend not compiled");
}

void cuda_simpls_fast_refresh_block_implicit_resident(
  int,
  int,
  int,
  int,
  int,
  int,
  unsigned int,
  int,
  double*
) {
  throw std::runtime_error("CUDA backend not compiled");
}

bool cuda_simpls_fast_append_component_from_block(
  int,
  int,
  int,
  int,
  int,
  int,
  bool,
  bool,
  bool
) {
  throw std::runtime_error("CUDA backend not compiled");
}

void cuda_simpls_fast_copy_rr(
  double*,
  int,
  int
) {
  throw std::runtime_error("CUDA backend not compiled");
}

void cuda_simpls_fast_copy_qq(
  double*,
  int,
  int
) {
  throw std::runtime_error("CUDA backend not compiled");
}

void cuda_simpls_fast_copy_bcur(
  double*,
  int,
  int
) {
  throw std::runtime_error("CUDA backend not compiled");
}

void cuda_simpls_fast_copy_yfit(
  double*,
  int,
  int
) {
  throw std::runtime_error("CUDA backend not compiled");
}

void cuda_simpls_fast_component_stats(
  const double*,
  int,
  int,
  int,
  double*,
  double*,
  double*,
  double*
) {
  throw std::runtime_error("CUDA backend not compiled");
}

void cuda_simpls_fast_rank1_fit_update(
  const double*,
  int,
  const double*,
  int,
  double*
) {
  throw std::runtime_error("CUDA backend not compiled");
}

} // namespace fastpls_svd

#endif
