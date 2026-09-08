#ifndef FASTPLS_SVD_CUDA_RSVD_H
#define FASTPLS_SVD_CUDA_RSVD_H

#include "svd_iface.h"
#include <memory>

namespace fastpls_svd {

class CudaFloatCrossproduct {
 public:
  CudaFloatCrossproduct(const arma::fmat& X, const arma::fmat& Y);
  ~CudaFloatCrossproduct();
  CudaFloatCrossproduct(const CudaFloatCrossproduct&) = delete;
  CudaFloatCrossproduct& operator=(const CudaFloatCrossproduct&) = delete;
  arma::fmat multiply(const arma::fmat& B, bool transpose);
  void orthonormalize(arma::fmat& B);
  void deflate(const arma::fvec& v);
  arma::fmat left_factor();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct PLSSVDGPUResult {
  Mat R;
  Mat Q;
  Mat Ttrain;
  arma::cube C_latent;
  arma::cube W_latent;
  arma::cube B;
  arma::cube Yfit;
  Vec R2Y;
};

struct LDAGPUModel {
  Mat means;
  Mat linear;
  arma::rowvec constants;
  Vec priors;
  double ridge = 0.0;
  double relative_ridge = 0.0;
};

struct LDAFloatGPUModel {
  arma::fmat means;
  arma::fmat linear;
  arma::frowvec constants;
  arma::fvec priors;
  float ridge = 0.0f;
  float relative_ridge = 0.0f;
};

struct LDAFloatPrediction {
  std::vector<int> pred;
  arma::fmat scores;
};

SVDResult truncated_svd_cuda_rsvd(const Mat& A, int k, const SVDOptions& opt);
SVDResult cuda_rsvd_resident_svd(const Mat& A, int k, const SVDOptions& opt);
PLSSVDGPUResult cuda_plssvd_fit(
  const Mat& Xtrain,
  const Mat& Ytrain,
  const arma::ivec& ncomp,
  bool fit,
  const SVDOptions& opt,
  bool return_B = false
);
PLSSVDGPUResult cuda_plssvd_fit_implicit_xprod(
  const Mat& Xtrain,
  const Mat& Ytrain,
  const arma::ivec& ncomp,
  bool fit,
  const SVDOptions& opt,
  bool return_B = false
);
arma::cube cuda_flash_lowrank_predict(
  const Mat& Xtest,
  const Mat& R,
  const arma::cube& W_latent,
  const arma::rowvec& mY,
  const arma::ivec& ncomp
);
arma::imat cuda_flash_lowrank_predict_classes(
  const Mat& Xtest,
  const Mat& R,
  const arma::cube& W_latent,
  const arma::rowvec& mY,
  const arma::ivec& ncomp
);
arma::fmat cuda_matrix_multiply_float(
  const arma::fmat& A,
  const arma::fmat& B,
  bool transpose_left = false,
  bool transpose_right = false
);
std::vector<LDAGPUModel> cuda_lda_train_prefix(
  const Mat& Ttrain,
  const arma::ivec& y,
  int n_classes,
  const arma::ivec& ncomp,
  double ridge
);
std::vector<LDAGPUModel> cuda_lda_project_train_prefix(
  const Mat& Xtrain,
  const Mat& R,
  const arma::rowvec& offset,
  const arma::ivec& y,
  int n_classes,
  const arma::ivec& ncomp,
  double ridge
);
Rcpp::List cuda_lda_predict(
  const Mat& Ttest,
  const Mat& linear,
  const arma::rowvec& constants,
  bool return_scores
);
Rcpp::List cuda_lda_project_predict(
  const Mat& Xtest,
  const Mat& R,
  const arma::rowvec& offset,
  const Mat& linear,
  const arma::rowvec& constants,
  bool return_scores
);
std::vector<LDAFloatGPUModel> cuda_lda_train_prefix_float(
  const arma::fmat& Ttrain,
  const arma::ivec& y,
  int n_classes,
  const arma::ivec& ncomp
);
LDAFloatPrediction cuda_lda_predict_float(
  const arma::fmat& Ttest,
  const arma::fmat& linear,
  const arma::frowvec& constants,
  bool return_scores = true
);
bool cuda_lda_native_available();
bool cuda_runtime_available();
void cuda_rsvd_sample_y(
  const double* hA,
  int m,
  int n,
  const double* hOmega,
  int l,
  int power_iters,
  double* hY
);
void cuda_rsvd_sample_y_float(
  const float* hA,
  int m,
  int n,
  const float* hOmega,
  int l,
  int power_iters,
  float* hY
);
bool cuda_rsvd_prefer_block_gpu(int m, int n, int l, int power_iters);
void cuda_rsvd_set_resident_matrix(
  const double* hA,
  int m,
  int n
);
void cuda_rsvd_refresh_left_block(
  const double* hA,
  int m,
  int n,
  unsigned int seed,
  int l,
  int power_iters,
  double* hY
);
void cuda_rsvd_refresh_left_block_u(
  const double* hA,
  int m,
  int n,
  unsigned int seed,
  int l,
  int k,
  int power_iters,
  double* hUblock,
  double* hSvals = nullptr
);
void cuda_rsvd_refresh_left_block_u_resident(
  int m,
  int n,
  int l,
  int k,
  unsigned int seed,
  int power_iters,
  double* hUblock,
  double* hSvals = nullptr
);
void cuda_rsvd_project_left_row(
  const double* hV,
  int m,
  int n,
  double* hVS
);
void cuda_rsvd_deflate_left_rank1(
  const double* hV,
  const double* hVS,
  int m,
  int n
);
void cuda_simpls_fast_set_training_matrices(
  const double* hX,
  int n,
  int p,
  const double* hY,
  int m,
  bool fit,
  bool form_crossprod = true
);
void cuda_simpls_fast_begin_device_loop(
  int n,
  int p,
  int m,
  int max_ncomp,
  bool fit,
  bool store_B
);
void cuda_simpls_fast_refresh_block_resident(
  int p,
  int m,
  int l,
  int k,
  unsigned int seed,
  int power_iters,
  double* hSvals = nullptr
);
void cuda_simpls_fast_refresh_block_implicit_resident(
  int n,
  int p,
  int m,
  int l,
  int k,
  int prev_v_cols,
  unsigned int seed,
  int power_iters,
  double* hSvals = nullptr
);
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
);
void cuda_simpls_fast_copy_rr(
  double* hRR,
  int p,
  int max_ncomp
);
void cuda_simpls_fast_copy_qq(
  double* hQQ,
  int m,
  int max_ncomp
);
void cuda_simpls_fast_copy_bcur(
  double* hB,
  int p,
  int m
);
void cuda_simpls_fast_copy_yfit(
  double* hYfit,
  int n,
  int m
);
void cuda_simpls_fast_component_stats(
  const double* hR,
  int n,
  int p,
  int m,
  double* hT,
  double* hP,
  double* hQ,
  double* hTnorm
);
void cuda_simpls_fast_rank1_fit_update(
  const double* hT,
  int n,
  const double* hQ,
  int m,
  double* hDelta
);

} // namespace fastpls_svd

#endif
