// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore

#ifndef FASTPLS_CUDA_RESIDENT_PREPROCESS_CUH
#define FASTPLS_CUDA_RESIDENT_PREPROCESS_CUH

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

namespace fastpls_device {

// One block per column; merge Welford accumulators without widening float32.
template<class T>
__global__ void column_statistics(const T* x, int n, int p, int scaling,
                                  T* mean, T* scale) {
    const int j = blockIdx.x;
    if (j >= p) return;
    __shared__ T means[256], m2s[256];
    __shared__ int counts[256];
    const int tid = threadIdx.x;
    T m = 0, m2 = 0;
    int count = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        const T v = x[static_cast<size_t>(j) * n + i];
        const T delta = v - m;
        ++count;
        m += delta / T(count);
        m2 += delta * (v - m);
    }
    means[tid] = m; m2s[tid] = m2; counts[tid] = count;
    __syncthreads();
    for (int step = 128; step; step /= 2) {
        if (tid < step && counts[tid + step]) {
            const int a = counts[tid], b = counts[tid + step];
            const T delta = means[tid + step] - means[tid];
            m2s[tid] += m2s[tid + step] + delta * delta *
                (T(a) * T(b) / T(a + b));
            means[tid] += delta * (T(b) / T(a + b));
            counts[tid] += b;
        }
        __syncthreads();
    }
    if (tid == 0) {
        mean[j] = scaling < 3 ? means[0] : T(0);
        T sd = scaling == 2 && n > 1 ? sqrt(m2s[0] / T(n - 1)) : T(1);
        scale[j] = sd > T(0) && isfinite(sd) ? sd : T(1);
    }
}

template<class T>
__global__ void standardize(T* x, size_t size, int n,
                            const T* mean, const T* scale) {
    for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
         i < size; i += size_t(blockDim.x) * gridDim.x) {
        const size_t j = i / n;
        x[i] = (x[i] - mean[j]) / scale[j];
    }
}

template<class T>
__global__ void identity_statistics(T* mean, T* scale, int columns) {
    for (int column = blockIdx.x * blockDim.x + threadIdx.x;
         column < columns; column += blockDim.x * gridDim.x) {
        mean[column] = T(0);
        scale[column] = T(1);
    }
}

template<class T>
cudaError_t initialize_identity_statistics(T* mean, T* scale, int columns,
                                           cudaStream_t stream) {
    if (!mean || !scale || columns < 1) return cudaErrorInvalidValue;
    identity_statistics<<<std::min(256, (columns + 255) / 256), 256, 0,
                          stream>>>(mean, scale, columns);
    return cudaGetLastError();
}

// Device pointers only. Test data reuse training statistics with standardize.
template<class T>
cudaError_t preprocess(T* x, int n, int p, int scaling, T* mean, T* scale,
                       cudaStream_t stream) {
    if (n < 1 || p < 1 || scaling < 1 || scaling > 3)
        return cudaErrorInvalidValue;
    column_statistics<<<p, 256, 0, stream>>>(x, n, p, scaling, mean, scale);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    standardize<<<256, 256, 0, stream>>>(x, size_t(n) * p, n, mean, scale);
    return cudaGetLastError();
}

} // namespace fastpls_device
#endif
