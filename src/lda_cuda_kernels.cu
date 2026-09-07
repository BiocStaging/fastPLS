#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include "cuda_resident_api.cuh"

extern "C" {

__global__ void fastpls_lda_means_kernel(double* means,
                                         const double* counts,
                                         int kmax,
                                         int n_classes) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kmax * n_classes;
  if (idx >= total) return;
  const int cls = idx % n_classes;
  const double cnt = counts[cls];
  means[idx] = (cnt > 0.0) ? means[idx] / cnt : 0.0;
}

__global__ void fastpls_lda_label_sums_kernel(const double* T,
                                              const int* y,
                                              int n,
                                              int kmax,
                                              int n_classes,
                                              double* sums) {
  const int cls = blockIdx.x;
  const int j = blockIdx.y;
  if (cls >= n_classes || j >= kmax) return;

  extern __shared__ double partial[];
  double value = 0.0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    if (y[i] == cls + 1) {
      value += T[i + j * n];
    }
  }
  partial[threadIdx.x] = value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    sums[cls + j * n_classes] = partial[0];
  }
}

__global__ void fastpls_lda_subtract_offsets_kernel(double* T,
                                                    const double* offsets,
                                                    int n,
                                                    int kmax) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n * kmax;
  if (idx >= total) return;
  const int j = idx / n;
  T[idx] -= offsets[j];
}

__global__ void fastpls_lda_pooled_kernel(double* pooled,
                                          const double* means,
                                          const double* counts,
                                          int n,
                                          int kmax,
                                          int n_classes) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kmax * kmax;
  if (idx >= total) return;
  const int r = idx % kmax;
  const int c = idx / kmax;
  double between = 0.0;
  for (int cls = 0; cls < n_classes; ++cls) {
    between += counts[cls] * means[cls + r * n_classes] * means[cls + c * n_classes];
  }
  const double df = fmax(1.0, static_cast<double>(n - n_classes));
  pooled[idx] = (pooled[idx] - between) / df;
}

__global__ void fastpls_lda_copy_cov_kernel(const double* pooled,
                                            double* cov,
                                            int kmax,
                                            int kk) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kk * kk;
  if (idx >= total) return;
  const int r = idx % kk;
  const int c = idx / kk;
  cov[idx] = pooled[r + c * kmax];
}

__global__ void fastpls_lda_add_ridge_kernel(double* cov,
                                             int kk,
                                             double ridge,
                                             double* lambda_out) {
  __shared__ double trace;
  if (threadIdx.x == 0) trace = 0.0;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 0; i < kk; ++i) {
      trace += cov[i + i * kk];
    }
  }
  __syncthreads();
  const double scale = isfinite(trace) && trace > 0.0 ? trace / static_cast<double>(kk) : 1.0;
  const double lambda = (isfinite(ridge) && ridge >= 0.0 ? ridge : 1e-8) * scale;
  if (threadIdx.x == 0) *lambda_out = lambda;
  for (int i = threadIdx.x; i < kk; i += blockDim.x) {
    cov[i + i * kk] += lambda;
  }
}

__global__ void fastpls_lda_means_to_rhs_kernel(const double* means,
                                                double* rhs,
                                                int kmax,
                                                int kk,
                                                int n_classes) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kk * n_classes;
  if (idx >= total) return;
  const int j = idx % kk;
  const int cls = idx / kk;
  rhs[j + cls * kk] = means[cls + j * n_classes];
}

__global__ void fastpls_lda_finalize_linear_kernel(const double* rhs,
                                                  const double* means,
                                                  const double* counts,
                                                  double* linear,
                                                  double* constants,
                                                  int n,
                                                  int kmax,
                                                  int kk,
                                                  int n_classes) {
  const int cls = blockIdx.x * blockDim.x + threadIdx.x;
  if (cls >= n_classes) return;
  double dot = 0.0;
  for (int j = 0; j < kk; ++j) {
    const double value = rhs[j + cls * kk];
    linear[cls + j * n_classes] = value;
    dot += means[cls + j * n_classes] * value;
  }
  const double prior = fmax(counts[cls] / static_cast<double>(n), 2.2250738585072014e-308);
  constants[cls] = -0.5 * dot + log(prior);
}

__global__ void fastpls_lda_score_argmax_kernel(double* scores,
                                                const double* constants,
                                                int* pred,
                                                int n,
                                                int n_classes) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  int best = 0;
  double best_value = scores[row] + constants[0];
  scores[row] = best_value;
  for (int cls = 1; cls < n_classes; ++cls) {
    const int offset = row + cls * n;
    const double value = scores[offset] + constants[cls];
    scores[offset] = value;
    if (value > best_value) {
      best_value = value;
      best = cls;
    }
  }
  pred[row] = best + 1;
}

__global__ void fastpls_lda_means_float_kernel(float* means,
                                               const float* counts,
                                               int kmax,
                                               int n_classes) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kmax * n_classes;
  if (idx >= total) return;
  const int cls = idx % n_classes;
  const float cnt = counts[cls];
  means[idx] = (cnt > 0.0f) ? means[idx] / cnt : 0.0f;
}

__global__ void fastpls_lda_label_sums_float_kernel(const float* T,
                                                    const int* y,
                                                    int n,
                                                    int kmax,
                                                    int n_classes,
                                                    float* sums) {
  const int cls = blockIdx.x;
  const int j = blockIdx.y;
  if (cls >= n_classes || j >= kmax) return;
  extern __shared__ float partial_float[];
  float value = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    if (y[i] == cls + 1) value += T[i + j * n];
  }
  partial_float[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial_float[threadIdx.x] += partial_float[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) sums[cls + j * n_classes] = partial_float[0];
}

__global__ void fastpls_lda_pooled_float_kernel(float* pooled,
                                                const float* means,
                                                const float* counts,
                                                int n,
                                                int kmax,
                                                int n_classes) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kmax * kmax;
  if (idx >= total) return;
  const int r = idx % kmax;
  const int c = idx / kmax;
  float between = 0.0f;
  for (int cls = 0; cls < n_classes; ++cls) {
    between += counts[cls] * means[cls + r * n_classes] *
      means[cls + c * n_classes];
  }
  const float df = fmaxf(1.0f, static_cast<float>(n - n_classes));
  pooled[idx] = (pooled[idx] - between) / df;
}

__global__ void fastpls_lda_copy_cov_float_kernel(const float* pooled,
                                                  float* cov,
                                                  int kmax,
                                                  int kk) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kk * kk;
  if (idx >= total) return;
  const int r = idx % kk;
  const int c = idx / kk;
  cov[idx] = pooled[r + c * kmax];
}

__global__ void fastpls_lda_add_ridge_float_kernel(float* cov,
                                                   int kk,
                                                   float rho,
                                                   float* lambda_out) {
  __shared__ float trace;
  if (threadIdx.x == 0) {
    trace = 0.0f;
    for (int i = 0; i < kk; ++i) trace += cov[i + i * kk];
  }
  __syncthreads();
  const float scale = isfinite(trace) && trace > 0.0f ?
    trace / static_cast<float>(kk) : 1.0f;
  const float lambda = rho * scale;
  if (threadIdx.x == 0) *lambda_out = lambda;
  for (int i = threadIdx.x; i < kk; i += blockDim.x) {
    cov[i + i * kk] += lambda;
  }
}

__global__ void fastpls_lda_means_to_rhs_float_kernel(const float* means,
                                                       float* rhs,
                                                       int kmax,
                                                       int kk,
                                                       int n_classes) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = kk * n_classes;
  if (idx >= total) return;
  const int j = idx % kk;
  const int cls = idx / kk;
  rhs[j + cls * kk] = means[cls + j * n_classes];
}

__global__ void fastpls_lda_finalize_linear_float_kernel(const float* rhs,
                                                         const float* means,
                                                         const float* counts,
                                                         float* linear,
                                                         float* constants,
                                                         int n,
                                                         int kmax,
                                                         int kk,
                                                         int n_classes) {
  const int cls = blockIdx.x * blockDim.x + threadIdx.x;
  if (cls >= n_classes) return;
  float dot = 0.0f;
  for (int j = 0; j < kk; ++j) {
    const float value = rhs[j + cls * kk];
    linear[cls + j * n_classes] = value;
    dot += means[cls + j * n_classes] * value;
  }
  const float prior = fmaxf(counts[cls] / static_cast<float>(n), FLT_MIN);
  constants[cls] = -0.5f * dot + logf(prior);
}

__global__ void fastpls_lda_score_argmax_float_kernel(float* scores,
                                                      const float* constants,
                                                      int* pred,
                                                      int n,
                                                      int n_classes) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  int best = 0;
  float best_value = scores[row] + constants[0];
  scores[row] = best_value;
  for (int cls = 1; cls < n_classes; ++cls) {
    const int offset = row + cls * n;
    const float value = scores[offset] + constants[cls];
    scores[offset] = value;
    if (value > best_value) {
      best_value = value;
      best = cls;
    }
  }
  pred[row] = best + 1;
}

void fastpls_cuda_lda_means(double* means,
                            const double* counts,
                            int kmax,
                            int n_classes,
                            cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (kmax * n_classes + threads - 1) / threads;
  fastpls_lda_means_kernel<<<blocks, threads, 0, stream>>>(means, counts, kmax, n_classes);
}

void fastpls_cuda_lda_label_sums(const double* T,
                                 const int* y,
                                 int n,
                                 int kmax,
                                 int n_classes,
                                 double* sums,
                                 cudaStream_t stream) {
  const int threads = 256;
  const dim3 blocks(n_classes, kmax);
  const size_t shared = sizeof(double) * static_cast<size_t>(threads);
  fastpls_lda_label_sums_kernel<<<blocks, threads, shared, stream>>>(T, y, n, kmax, n_classes, sums);
}

void fastpls_cuda_lda_subtract_offsets(double* T,
                                       const double* offsets,
                                       int n,
                                       int kmax,
                                       cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (n * kmax + threads - 1) / threads;
  fastpls_lda_subtract_offsets_kernel<<<blocks, threads, 0, stream>>>(T, offsets, n, kmax);
}

void fastpls_cuda_lda_pooled(double* pooled,
                             const double* means,
                             const double* counts,
                             int n,
                             int kmax,
                             int n_classes,
                             cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (kmax * kmax + threads - 1) / threads;
  fastpls_lda_pooled_kernel<<<blocks, threads, 0, stream>>>(pooled, means, counts, n, kmax, n_classes);
}

void fastpls_cuda_lda_copy_cov(const double* pooled,
                               double* cov,
                               int kmax,
                               int kk,
                               cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (kk * kk + threads - 1) / threads;
  fastpls_lda_copy_cov_kernel<<<blocks, threads, 0, stream>>>(pooled, cov, kmax, kk);
}

void fastpls_cuda_lda_add_ridge(double* cov,
                                int kk,
                                double ridge,
                                double* lambda_out,
                                cudaStream_t stream) {
  fastpls_lda_add_ridge_kernel<<<1, 256, 0, stream>>>(cov, kk, ridge, lambda_out);
}

void fastpls_cuda_lda_means_to_rhs(const double* means,
                                   double* rhs,
                                   int kmax,
                                   int kk,
                                   int n_classes,
                                   cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (kk * n_classes + threads - 1) / threads;
  fastpls_lda_means_to_rhs_kernel<<<blocks, threads, 0, stream>>>(means, rhs, kmax, kk, n_classes);
}

void fastpls_cuda_lda_finalize_linear(const double* rhs,
                                      const double* means,
                                      const double* counts,
                                      double* linear,
                                      double* constants,
                                      int n,
                                      int kmax,
                                      int kk,
                                      int n_classes,
                                      cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (n_classes + threads - 1) / threads;
  fastpls_lda_finalize_linear_kernel<<<blocks, threads, 0, stream>>>(
    rhs, means, counts, linear, constants, n, kmax, kk, n_classes);
}

void fastpls_cuda_lda_score_argmax(double* scores,
                                   const double* constants,
                                   int* pred,
                                   int n,
                                   int n_classes,
                                   cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  fastpls_lda_score_argmax_kernel<<<blocks, threads, 0, stream>>>(scores, constants, pred, n, n_classes);
}

void fastpls_cuda_lda_means_float(float* means,
                                  const float* counts,
                                  int kmax,
                                  int n_classes,
                                  cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (kmax * n_classes + threads - 1) / threads;
  fastpls_lda_means_float_kernel<<<blocks, threads, 0, stream>>>(
    means, counts, kmax, n_classes
  );
}

void fastpls_cuda_lda_label_sums_float(const float* T,
                                       const int* y,
                                       int n,
                                       int kmax,
                                       int n_classes,
                                       float* sums,
                                       cudaStream_t stream) {
  const int threads = 256;
  const dim3 blocks(n_classes, kmax);
  const size_t shared = sizeof(float) * static_cast<size_t>(threads);
  fastpls_lda_label_sums_float_kernel<<<blocks, threads, shared, stream>>>(
    T, y, n, kmax, n_classes, sums
  );
}

void fastpls_cuda_lda_pooled_float(float* pooled,
                                   const float* means,
                                   const float* counts,
                                   int n,
                                   int kmax,
                                   int n_classes,
                                   cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (kmax * kmax + threads - 1) / threads;
  fastpls_lda_pooled_float_kernel<<<blocks, threads, 0, stream>>>(
    pooled, means, counts, n, kmax, n_classes
  );
}

void fastpls_cuda_lda_copy_cov_float(const float* pooled,
                                     float* cov,
                                     int kmax,
                                     int kk,
                                     cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (kk * kk + threads - 1) / threads;
  fastpls_lda_copy_cov_float_kernel<<<blocks, threads, 0, stream>>>(
    pooled, cov, kmax, kk
  );
}

void fastpls_cuda_lda_add_ridge_float(float* cov,
                                      int kk,
                                      float rho,
                                      float* lambda_out,
                                      cudaStream_t stream) {
  fastpls_lda_add_ridge_float_kernel<<<1, 256, 0, stream>>>(
    cov, kk, rho, lambda_out
  );
}

void fastpls_cuda_lda_means_to_rhs_float(const float* means,
                                         float* rhs,
                                         int kmax,
                                         int kk,
                                         int n_classes,
                                         cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (kk * n_classes + threads - 1) / threads;
  fastpls_lda_means_to_rhs_float_kernel<<<blocks, threads, 0, stream>>>(
    means, rhs, kmax, kk, n_classes
  );
}

void fastpls_cuda_lda_finalize_linear_float(const float* rhs,
                                            const float* means,
                                            const float* counts,
                                            float* linear,
                                            float* constants,
                                            int n,
                                            int kmax,
                                            int kk,
                                            int n_classes,
                                            cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (n_classes + threads - 1) / threads;
  fastpls_lda_finalize_linear_float_kernel<<<blocks, threads, 0, stream>>>(
    rhs, means, counts, linear, constants, n, kmax, kk, n_classes
  );
}

void fastpls_cuda_lda_score_argmax_float(float* scores,
                                         const float* constants,
                                         int* pred,
                                         int n,
                                         int n_classes,
                                         cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  fastpls_lda_score_argmax_float_kernel<<<blocks, threads, 0, stream>>>(
    scores, constants, pred, n, n_classes
  );
}

}
