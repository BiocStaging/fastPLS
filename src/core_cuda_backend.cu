// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore

#include "accelerator_core_backend.h"
#include "cuda_resident_api.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace fastpls_svd {
namespace {

void require_cuda_status(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
      std::string(operation) + ": " + cudaGetErrorString(status)
    );
  }
}

void require_blas_status(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed");
  }
}

int cuda_dimension(std::size_t value, const char* name) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(std::string(name) + " exceeds CUDA limits");
  }
  return static_cast<int>(value);
}

}  // namespace

bool has_cuda_backend() {
  int devices = 0;
  return cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
}

bool cuda_lda_native_available() {
  return has_cuda_backend();
}

fastpls::core::Matrix<float> cuda_core_gemm_f32(
    fastpls::core::ConstMatrixView<float> left,
    fastpls::core::ConstMatrixView<float> right,
    bool transpose_left, bool transpose_right) {
  const std::size_t rows = transpose_left ? left.columns() : left.rows();
  const std::size_t inner = transpose_left ? left.rows() : left.columns();
  const std::size_t right_inner =
    transpose_right ? right.columns() : right.rows();
  const std::size_t columns =
    transpose_right ? right.rows() : right.columns();
  if (inner != right_inner) {
    throw std::invalid_argument("CUDA matrix dimensions are not conformable");
  }
  if (!has_cuda_backend()) {
    throw std::runtime_error(
      "CUDA is unavailable; no CPU fallback is performed"
    );
  }

  fastpls::core::Matrix<float> result(rows, columns);
  if (rows == 0 || columns == 0 || inner == 0) return result;

  const std::size_t left_size = left.rows() * left.columns();
  const std::size_t right_size = right.rows() * right.columns();

  float* device_left = nullptr;
  float* device_right = nullptr;
  float* device_result = nullptr;
  cudaStream_t stream = nullptr;
  cublasHandle_t handle = nullptr;
  auto release = [&] {
    cudaFree(device_left);
    cudaFree(device_right);
    cudaFree(device_result);
    if (handle != nullptr) cublasDestroy(handle);
    if (stream != nullptr) cudaStreamDestroy(stream);
  };

  try {
    require_cuda_status(cudaStreamCreate(&stream), "cudaStreamCreate");
    require_blas_status(cublasCreate(&handle), "cublasCreate");
    require_blas_status(cublasSetStream(handle, stream), "cublasSetStream");
    require_cuda_status(
      cudaMalloc(&device_left, left_size * sizeof(float)),
      "cudaMalloc(left)"
    );
    require_cuda_status(
      cudaMalloc(&device_right, right_size * sizeof(float)),
      "cudaMalloc(right)"
    );
    require_cuda_status(
      cudaMalloc(&device_result, result.size() * sizeof(float)),
      "cudaMalloc(result)"
    );
    require_cuda_status(
      cudaMemcpyAsync(device_left, left.data(), left_size * sizeof(float),
                      cudaMemcpyHostToDevice, stream),
      "cudaMemcpyAsync(left)"
    );
    require_cuda_status(
      cudaMemcpyAsync(device_right, right.data(), right_size * sizeof(float),
                      cudaMemcpyHostToDevice, stream),
      "cudaMemcpyAsync(right)"
    );
    const float one = 1.0f;
    const float zero = 0.0f;
    require_blas_status(
      cublasSgemm(
        handle,
        transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N,
        transpose_right ? CUBLAS_OP_T : CUBLAS_OP_N,
        cuda_dimension(rows, "rows"),
        cuda_dimension(columns, "columns"),
        cuda_dimension(inner, "inner dimension"),
        &one,
        device_left,
        cuda_dimension(left.leading_dimension(), "left leading dimension"),
        device_right,
        cuda_dimension(right.leading_dimension(), "right leading dimension"),
        &zero,
        device_result,
        cuda_dimension(rows, "result leading dimension")
      ),
      "cublasSgemm"
    );
    require_cuda_status(
      cudaMemcpyAsync(result.data(), device_result,
                      result.size() * sizeof(float),
                      cudaMemcpyDeviceToHost, stream),
      "cudaMemcpyAsync(result)"
    );
    require_cuda_status(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    release();
    return result;
  } catch (...) {
    release();
    throw;
  }
}

}  // namespace fastpls_svd
