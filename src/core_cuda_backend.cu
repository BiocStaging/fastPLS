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

struct CudaCoreWorkspaceF32 {
  cudaStream_t stream = nullptr;
  cublasHandle_t handle = nullptr;
  float* left = nullptr;
  float* right = nullptr;
  float* result = nullptr;
  std::size_t left_capacity = 0;
  std::size_t right_capacity = 0;
  std::size_t result_capacity = 0;

  CudaCoreWorkspaceF32() {
    require_cuda_status(cudaStreamCreate(&stream), "cudaStreamCreate");
    try {
      require_blas_status(cublasCreate(&handle), "cublasCreate");
      require_blas_status(cublasSetStream(handle, stream), "cublasSetStream");
    } catch (...) {
      if (handle != nullptr) cublasDestroy(handle);
      cudaStreamDestroy(stream);
      throw;
    }
  }

  ~CudaCoreWorkspaceF32() {
    cudaFree(left);
    cudaFree(right);
    cudaFree(result);
    if (handle != nullptr) cublasDestroy(handle);
    if (stream != nullptr) cudaStreamDestroy(stream);
  }
};

void ensure_device_capacity(float*& pointer, std::size_t& capacity,
                            std::size_t required, const char* operation) {
  if (required <= capacity) return;
  cudaFree(pointer);
  pointer = nullptr;
  capacity = 0;
  require_cuda_status(
    cudaMalloc(&pointer, required * sizeof(float)), operation
  );
  capacity = required;
}

void upload_matrix(fastpls::core::ConstMatrixView<float> source,
                   float* destination, cudaStream_t stream,
                   const char* operation) {
  require_cuda_status(
    cudaMemcpy2DAsync(
      destination, source.leading_dimension() * sizeof(float), source.data(),
      source.leading_dimension() * sizeof(float), source.rows() * sizeof(float),
      source.columns(), cudaMemcpyHostToDevice, stream
    ),
    operation
  );
}

void download_matrix(const float* source, std::size_t source_leading_dimension,
                     fastpls::core::MatrixView<float> destination,
                     cudaStream_t stream, const char* operation) {
  require_cuda_status(
    cudaMemcpy2DAsync(
      destination.data(), destination.leading_dimension() * sizeof(float),
      source, source_leading_dimension * sizeof(float),
      destination.rows() * sizeof(float), destination.columns(),
      cudaMemcpyDeviceToHost, stream
    ),
    operation
  );
}

void mirror_lower_to_upper(fastpls::core::MatrixView<float> matrix) {
  for (std::size_t column = 0; column < matrix.columns(); ++column) {
    for (std::size_t row = 0; row < column; ++row) {
      matrix(row, column) = matrix(column, row);
    }
  }
}

}  // namespace

bool has_cuda_backend() {
  int devices = 0;
  return cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
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

  CudaCoreWorkspaceF32 workspace;
  cuda_core_gemm_into_f32(
    &workspace, left, right, transpose_left, transpose_right, result.view()
  );
  return result;
}

void* cuda_core_workspace_create_f32() {
  if (!has_cuda_backend()) {
    throw std::runtime_error(
      "CUDA is unavailable; no CPU fallback is performed"
    );
  }
  return new CudaCoreWorkspaceF32();
}

void cuda_core_workspace_destroy_f32(void* workspace) noexcept {
  delete static_cast<CudaCoreWorkspaceF32*>(workspace);
}

bool cuda_core_gemm_into_f32(
    void* opaque_workspace,
    fastpls::core::ConstMatrixView<float> left,
    fastpls::core::ConstMatrixView<float> right,
    bool transpose_left, bool transpose_right,
    fastpls::core::MatrixView<float> output) {
  auto* workspace = static_cast<CudaCoreWorkspaceF32*>(opaque_workspace);
  if (workspace == nullptr) {
    throw std::invalid_argument("CUDA core workspace is null");
  }
  const std::size_t rows = transpose_left ? left.columns() : left.rows();
  const std::size_t inner = transpose_left ? left.rows() : left.columns();
  const std::size_t right_inner =
    transpose_right ? right.columns() : right.rows();
  const std::size_t columns =
    transpose_right ? right.rows() : right.columns();
  if (inner != right_inner || output.rows() != rows ||
      output.columns() != columns) {
    throw std::invalid_argument("CUDA matrix dimensions are not conformable");
  }
  if (rows == 0 || columns == 0 || inner == 0) return true;

  const std::size_t left_storage =
    left.leading_dimension() * left.columns();
  const std::size_t right_storage =
    right.leading_dimension() * right.columns();
  const std::size_t result_storage = rows * columns;
  ensure_device_capacity(
    workspace->left, workspace->left_capacity, left_storage,
    "cudaMalloc(left workspace)"
  );
  ensure_device_capacity(
    workspace->right, workspace->right_capacity, right_storage,
    "cudaMalloc(right workspace)"
  );
  ensure_device_capacity(
    workspace->result, workspace->result_capacity, result_storage,
    "cudaMalloc(result workspace)"
  );
  upload_matrix(left, workspace->left, workspace->stream, "upload(left)");
  upload_matrix(right, workspace->right, workspace->stream, "upload(right)");
  const float one = 1.0f;
  const float zero = 0.0f;
  require_blas_status(
    cublasSgemm(
      workspace->handle,
      transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N,
      transpose_right ? CUBLAS_OP_T : CUBLAS_OP_N,
      cuda_dimension(rows, "rows"), cuda_dimension(columns, "columns"),
      cuda_dimension(inner, "inner dimension"), &one, workspace->left,
      cuda_dimension(left.leading_dimension(), "left leading dimension"),
      workspace->right,
      cuda_dimension(right.leading_dimension(), "right leading dimension"),
      &zero, workspace->result, cuda_dimension(rows, "result leading dimension")
    ),
    "cublasSgemm"
  );
  download_matrix(
    workspace->result, rows, output, workspace->stream, "download(result)"
  );
  require_cuda_status(
    cudaStreamSynchronize(workspace->stream), "cudaStreamSynchronize"
  );
  return true;
}

bool cuda_core_self_gram_into_f32(
    void* opaque_workspace,
    fastpls::core::ConstMatrixView<float> input,
    bool transpose_input, fastpls::core::MatrixView<float> output,
    bool full_output) {
  auto* workspace = static_cast<CudaCoreWorkspaceF32*>(opaque_workspace);
  if (workspace == nullptr) {
    throw std::invalid_argument("CUDA core workspace is null");
  }
  const std::size_t dimension =
    transpose_input ? input.columns() : input.rows();
  const std::size_t rank = transpose_input ? input.rows() : input.columns();
  if (output.rows() != dimension || output.columns() != dimension) {
    throw std::invalid_argument(
      "CUDA float32 self-Gram dimensions are inconsistent"
    );
  }
  if (dimension == 0 || rank == 0) return true;

  const std::size_t input_storage =
    input.leading_dimension() * input.columns();
  const std::size_t result_storage = dimension * dimension;
  ensure_device_capacity(
    workspace->left, workspace->left_capacity, input_storage,
    "cudaMalloc(self-Gram input workspace)"
  );
  ensure_device_capacity(
    workspace->result, workspace->result_capacity, result_storage,
    "cudaMalloc(self-Gram result workspace)"
  );
  upload_matrix(
    input, workspace->left, workspace->stream, "upload(self-Gram input)"
  );
  require_cuda_status(
    cudaMemsetAsync(
      workspace->result, 0, result_storage * sizeof(float), workspace->stream
    ),
    "cudaMemsetAsync(self-Gram result)"
  );
  const float one = 1.0f;
  const float zero = 0.0f;
  require_blas_status(
    cublasSsyrk(
      workspace->handle, CUBLAS_FILL_MODE_LOWER,
      transpose_input ? CUBLAS_OP_T : CUBLAS_OP_N,
      cuda_dimension(dimension, "self-Gram dimension"),
      cuda_dimension(rank, "self-Gram rank"), &one, workspace->left,
      cuda_dimension(input.leading_dimension(), "input leading dimension"),
      &zero, workspace->result,
      cuda_dimension(dimension, "self-Gram leading dimension")
    ),
    "cublasSsyrk"
  );
  download_matrix(
    workspace->result, dimension, output, workspace->stream,
    "download(self-Gram result)"
  );
  require_cuda_status(
    cudaStreamSynchronize(workspace->stream), "cudaStreamSynchronize"
  );
  if (full_output) mirror_lower_to_upper(output);
  return true;
}

}  // namespace fastpls_svd
