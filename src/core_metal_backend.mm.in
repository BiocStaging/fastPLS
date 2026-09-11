// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore

#include "accelerator_core_backend.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <list>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace fastpls_svd {
namespace {

id<MTLDevice> metal_device() {
  static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  return device;
}

id<MTLCommandQueue> metal_queue() {
  static id<MTLCommandQueue> queue = [metal_device() newCommandQueue];
  return queue;
}

struct MetalMatrix {
  id<MTLBuffer> buffer = nil;
  NSUInteger rows = 0;
  NSUInteger columns = 0;
  NSUInteger row_bytes = 0;
};

constexpr std::size_t shared_page_size = 16384;

std::size_t shared_allocation_size(std::size_t bytes) {
  if (bytes > std::numeric_limits<std::size_t>::max() -
      (shared_page_size - 1)) {
    throw std::overflow_error("Metal shared-buffer size overflow");
  }
  return ((bytes + shared_page_size - 1) / shared_page_size) *
    shared_page_size;
}

bool can_wrap_shared(fastpls::core::ConstMatrixView<float> source) {
  if (!source.contiguous() || source.data() == nullptr || source.empty()) {
    return false;
  }
  const auto address = reinterpret_cast<std::uintptr_t>(source.data());
  const NSUInteger row_bytes = [MPSMatrixDescriptor
    rowBytesFromColumns:source.rows() dataType:MPSDataTypeFloat32];
  return address % shared_page_size == 0 &&
    row_bytes == source.rows() * sizeof(float);
}

MetalMatrix wrap_shared(fastpls::core::ConstMatrixView<float> source) {
  if (!can_wrap_shared(source)) {
    throw std::invalid_argument("matrix cannot use zero-copy Metal storage");
  }
  MetalMatrix result;
  result.rows = source.columns();
  result.columns = source.rows();
  result.row_bytes = source.rows() * sizeof(float);
  const std::size_t logical_bytes = source.rows() * source.columns() *
    sizeof(float);
  result.buffer = [metal_device()
    newBufferWithBytesNoCopy:const_cast<float*>(source.data())
    length:shared_allocation_size(logical_bytes)
    options:MTLResourceStorageModeShared
    deallocator:nil];
  if (result.buffer == nil) {
    throw std::runtime_error("Metal failed to wrap shared matrix storage");
  }
  return result;
}

MetalMatrix allocate_matrix(NSUInteger rows, NSUInteger columns) {
  MetalMatrix result;
  result.rows = rows;
  result.columns = columns;
  result.row_bytes = [MPSMatrixDescriptor
    rowBytesFromColumns:columns dataType:MPSDataTypeFloat32];
  const std::size_t bytes = static_cast<std::size_t>(result.row_bytes) * rows;
  result.buffer = [metal_device() newBufferWithLength:bytes
    options:MTLResourceStorageModeShared];
  if (result.buffer == nil) {
    throw std::runtime_error("Metal failed to allocate a matrix buffer");
  }
  std::memset([result.buffer contents], 0, bytes);
  return result;
}

MPSMatrix* mps_matrix(const MetalMatrix& value) {
  MPSMatrixDescriptor* descriptor = [MPSMatrixDescriptor
    matrixDescriptorWithRows:value.rows columns:value.columns
    rowBytes:value.row_bytes dataType:MPSDataTypeFloat32];
  return [[MPSMatrix alloc] initWithBuffer:value.buffer descriptor:descriptor];
}

void upload(fastpls::core::ConstMatrixView<float> source,
            const MetalMatrix& destination) {
  if (destination.rows != source.columns() ||
      destination.columns != source.rows()) {
    throw std::invalid_argument("Metal transpose view dimensions differ");
  }
  char* base = static_cast<char*>([destination.buffer contents]);
  for (std::size_t column = 0; column < source.columns(); ++column) {
    const float* input = source.data() +
      column * source.leading_dimension();
    void* output = base + column * destination.row_bytes;
    std::memcpy(output, input, source.rows() * sizeof(float));
  }
}

fastpls::core::Matrix<float> download(const MetalMatrix& source) {
  fastpls::core::Matrix<float> result(source.columns, source.rows);
  const char* base = static_cast<const char*>([source.buffer contents]);
  for (std::size_t column = 0; column < result.columns(); ++column) {
    const void* input = base + column * source.row_bytes;
    float* output = result.data() + column * result.rows();
    std::memcpy(output, input, result.rows() * sizeof(float));
  }
  return result;
}

MetalMatrix stage_or_wrap(fastpls::core::ConstMatrixView<float> source) {
  if (can_wrap_shared(source)) return wrap_shared(source);
  MetalMatrix result = allocate_matrix(source.columns(), source.rows());
  upload(source, result);
  return result;
}

class ProductWorkspace {
 public:
  ProductWorkspace(NSUInteger left_rows, NSUInteger left_columns,
                   NSUInteger right_rows, NSUInteger right_columns,
                   bool transpose_left, bool transpose_right)
      : left(allocate_matrix(left_columns, left_rows)),
        right(allocate_matrix(right_columns, right_rows)),
        result(allocate_matrix(
          transpose_right ? right_rows : right_columns,
          transpose_left ? left_columns : left_rows)),
        transpose_left(transpose_left),
        transpose_right(transpose_right) {
    left_matrix = mps_matrix(left);
    right_matrix = mps_matrix(right);
    result_matrix = mps_matrix(result);
    product = [[MPSMatrixMultiplication alloc]
      initWithDevice:metal_device()
      transposeLeft:transpose_right
      transposeRight:transpose_left
      resultRows:result.rows
      resultColumns:result.columns
      interiorColumns:transpose_right ? right_columns : right_rows
      alpha:1.0
      beta:0.0];
  }

  bool matches(NSUInteger left_rows, NSUInteger left_columns,
               NSUInteger right_rows, NSUInteger right_columns,
               bool requested_transpose_left,
               bool requested_transpose_right) const {
    return left.rows == left_columns && left.columns == left_rows &&
      right.rows == right_columns && right.columns == right_rows &&
      transpose_left == requested_transpose_left &&
      transpose_right == requested_transpose_right;
  }

  std::size_t bytes() const {
    return static_cast<std::size_t>(left.row_bytes) * left.rows +
      static_cast<std::size_t>(right.row_bytes) * right.rows +
      static_cast<std::size_t>(result.row_bytes) * result.rows;
  }

  MetalMatrix left;
  MetalMatrix right;
  MetalMatrix result;
  MPSMatrix* left_matrix = nil;
  MPSMatrix* right_matrix = nil;
  MPSMatrix* result_matrix = nil;
  MPSMatrixMultiplication* product = nil;
  bool transpose_left;
  bool transpose_right;
};

class WorkspaceCache {
 public:
  ProductWorkspace& acquire(
      NSUInteger left_rows, NSUInteger left_columns,
      NSUInteger right_rows, NSUInteger right_columns,
      bool transpose_left, bool transpose_right,
      std::unique_ptr<ProductWorkspace>& transient) {
    for (auto entry = workspaces.begin(); entry != workspaces.end(); ++entry) {
      if ((*entry)->matches(
            left_rows, left_columns, right_rows, right_columns,
            transpose_left, transpose_right)) {
        workspaces.splice(workspaces.begin(), workspaces, entry);
        return *workspaces.front();
      }
    }
    auto candidate = std::make_unique<ProductWorkspace>(
      left_rows, left_columns, right_rows, right_columns,
      transpose_left, transpose_right
    );
    if (candidate->bytes() > maximum_bytes) {
      transient = std::move(candidate);
      return *transient;
    }
    while (!workspaces.empty() &&
           (workspaces.size() >= maximum_entries ||
            retained_bytes + candidate->bytes() > maximum_bytes)) {
      retained_bytes -= workspaces.back()->bytes();
      workspaces.pop_back();
    }
    retained_bytes += candidate->bytes();
    workspaces.push_front(std::move(candidate));
    return *workspaces.front();
  }

  std::mutex mutex;

 private:
  static constexpr std::size_t maximum_entries = 8;
  static constexpr std::size_t maximum_bytes = 64u * 1024u * 1024u;
  std::size_t retained_bytes = 0;
  std::list<std::unique_ptr<ProductWorkspace>> workspaces;
};

WorkspaceCache& workspace_cache() {
  static WorkspaceCache cache;
  return cache;
}

class SampleGramWorkspace {
 public:
  SampleGramWorkspace(
      fastpls::core::ConstMatrixView<float> predictors,
      fastpls::core::ConstMatrixView<float> sample_gram)
      : n(predictors.rows()), p(predictors.columns()),
        predictors(stage_or_wrap(predictors)),
        sample_gram(stage_or_wrap(sample_gram)),
        score(allocate_matrix(1, n)),
        reverse_sample(allocate_matrix(1, n)) {
    if (sample_gram.rows() != n || sample_gram.columns() != n) {
      throw std::invalid_argument(
        "Metal sample-Gram workspace dimensions differ"
      );
    }
    predictor_matrix = mps_matrix(this->predictors);
    sample_gram_matrix = mps_matrix(this->sample_gram);
    score_matrix = mps_matrix(score);
    reverse_sample_matrix = mps_matrix(reverse_sample);
    forward = [[MPSMatrixMultiplication alloc]
      initWithDevice:metal_device()
      transposeLeft:NO
      transposeRight:NO
      resultRows:1
      resultColumns:n
      interiorColumns:p
      alpha:1.0
      beta:0.0];
    sample_product = [[MPSMatrixMultiplication alloc]
      initWithDevice:metal_device()
      transposeLeft:NO
      transposeRight:NO
      resultRows:1
      resultColumns:n
      interiorColumns:n
      alpha:1.0
      beta:0.0];
    reverse = [[MPSMatrixMultiplication alloc]
      initWithDevice:metal_device()
      transposeLeft:NO
      transposeRight:YES
      resultRows:1
      resultColumns:p
      interiorColumns:n
      alpha:1.0
      beta:0.0];
  }

  std::size_t n;
  std::size_t p;
  MetalMatrix predictors;
  MetalMatrix sample_gram;
  MetalMatrix score;
  MetalMatrix reverse_sample;
  MPSMatrix* predictor_matrix = nil;
  MPSMatrix* sample_gram_matrix = nil;
  MPSMatrix* score_matrix = nil;
  MPSMatrix* reverse_sample_matrix = nil;
  MPSMatrixMultiplication* forward = nil;
  MPSMatrixMultiplication* sample_product = nil;
  MPSMatrixMultiplication* reverse = nil;
};

}  // namespace

bool has_metal_backend() {
  return metal_device() != nil && metal_queue() != nil;
}

fastpls::core::Matrix<float> metal_core_gemm_f32(
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
    throw std::invalid_argument("Metal matrix dimensions are not conformable");
  }
  if (!has_metal_backend()) {
    throw std::runtime_error(
      "Metal is unavailable; no CPU fallback is performed"
    );
  }
  if (rows == 0 || columns == 0 || inner == 0) {
    return fastpls::core::Matrix<float>(rows, columns);
  }

  @autoreleasepool {
    fastpls::core::Matrix<float> direct_result(rows, columns);
    const bool direct_left_available = can_wrap_shared(left);
    const bool direct_right_available = can_wrap_shared(right);
    const bool direct_output_available = can_wrap_shared(direct_result.view());
    const std::size_t left_bytes = left.rows() * left.columns() * sizeof(float);
    const std::size_t right_bytes =
      right.rows() * right.columns() * sizeof(float);
    const bool avoids_large_copy =
      (direct_left_available && left_bytes > 64u * 1024u * 1024u) ||
      (direct_right_available && right_bytes > 64u * 1024u * 1024u);
    if ((direct_left_available && direct_right_available &&
         direct_output_available) || avoids_large_copy) {
      const MetalMatrix left_storage = stage_or_wrap(left);
      const MetalMatrix right_storage = stage_or_wrap(right);
      const MetalMatrix output_storage = direct_output_available ?
        wrap_shared(direct_result.view()) : allocate_matrix(columns, rows);
      MPSMatrix* left_matrix = mps_matrix(left_storage);
      MPSMatrix* right_matrix = mps_matrix(right_storage);
      MPSMatrix* output_matrix = mps_matrix(output_storage);
      MPSMatrixMultiplication* product = [[MPSMatrixMultiplication alloc]
        initWithDevice:metal_device()
        transposeLeft:transpose_right
        transposeRight:transpose_left
        resultRows:output_storage.rows
        resultColumns:output_storage.columns
        interiorColumns:transpose_right ? right.columns() : right.rows()
        alpha:1.0
        beta:0.0];
      id<MTLCommandBuffer> command = [metal_queue() commandBuffer];
      if (command == nil) {
        throw std::runtime_error("Metal failed to create a command buffer");
      }
      [product encodeToCommandBuffer:command
        leftMatrix:right_matrix
        rightMatrix:left_matrix
        resultMatrix:output_matrix];
      [command commit];
      [command waitUntilCompleted];
      if ([command error] != nil) {
        throw std::runtime_error(
          std::string("Metal matrix multiplication failed: ") +
          [[[command error] localizedDescription] UTF8String]
        );
      }
      return direct_output_available ?
        std::move(direct_result) : download(output_storage);
    }

    auto& cache = workspace_cache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    std::unique_ptr<ProductWorkspace> transient;
    ProductWorkspace& workspace = cache.acquire(
      left.rows(), left.columns(), right.rows(), right.columns(),
      transpose_left, transpose_right, transient
    );
    upload(left, workspace.left);
    upload(right, workspace.right);
    id<MTLCommandBuffer> command = [metal_queue() commandBuffer];
    if (command == nil) {
      throw std::runtime_error("Metal failed to create a command buffer");
    }
    [workspace.product encodeToCommandBuffer:command
      leftMatrix:workspace.right_matrix
      rightMatrix:workspace.left_matrix
      resultMatrix:workspace.result_matrix];
    [command commit];
    [command waitUntilCompleted];
    if ([command error] != nil) {
      throw std::runtime_error(
        std::string("Metal matrix multiplication failed: ") +
        [[[command error] localizedDescription] UTF8String]
      );
    }
    return download(workspace.result);
  }
}

bool metal_core_gemm_into_f32_impl(
    fastpls::core::ConstMatrixView<float> left,
    fastpls::core::ConstMatrixView<float> right,
    bool transpose_left, bool transpose_right,
    fastpls::core::MatrixView<float> output, float beta) {
  const std::size_t rows = transpose_left ? left.columns() : left.rows();
  const std::size_t inner = transpose_left ? left.rows() : left.columns();
  const std::size_t right_inner =
    transpose_right ? right.columns() : right.rows();
  const std::size_t columns =
    transpose_right ? right.rows() : right.columns();
  if (inner != right_inner || output.rows() != rows ||
      output.columns() != columns) {
    throw std::invalid_argument("Metal matrix dimensions are not conformable");
  }
  if (!has_metal_backend()) {
    throw std::runtime_error(
      "Metal is unavailable; no CPU fallback is performed"
    );
  }
  if (rows == 0 || columns == 0 || inner == 0) return true;
  if (!can_wrap_shared(output)) return false;

  @autoreleasepool {
    const MetalMatrix left_storage = stage_or_wrap(left);
    const MetalMatrix right_storage = stage_or_wrap(right);
    const MetalMatrix output_storage = wrap_shared(output);
    MPSMatrix* left_matrix = mps_matrix(left_storage);
    MPSMatrix* right_matrix = mps_matrix(right_storage);
    MPSMatrix* output_matrix = mps_matrix(output_storage);
    MPSMatrixMultiplication* product = [[MPSMatrixMultiplication alloc]
      initWithDevice:metal_device()
      transposeLeft:transpose_right
      transposeRight:transpose_left
      resultRows:output_storage.rows
      resultColumns:output_storage.columns
      interiorColumns:transpose_right ? right.columns() : right.rows()
      alpha:1.0
      beta:beta];
    id<MTLCommandBuffer> command = [metal_queue() commandBuffer];
    if (command == nil) {
      throw std::runtime_error("Metal failed to create a command buffer");
    }
    [product encodeToCommandBuffer:command
      leftMatrix:right_matrix
      rightMatrix:left_matrix
      resultMatrix:output_matrix];
    [command commit];
    [command waitUntilCompleted];
    if ([command error] != nil) {
      throw std::runtime_error(
        std::string("Metal matrix multiplication failed: ") +
        [[[command error] localizedDescription] UTF8String]
      );
    }
  }
  return true;
}

bool metal_core_gemm_into_f32(
    fastpls::core::ConstMatrixView<float> left,
    fastpls::core::ConstMatrixView<float> right,
    bool transpose_left, bool transpose_right,
    fastpls::core::MatrixView<float> output) {
  return metal_core_gemm_into_f32_impl(
    left, right, transpose_left, transpose_right, output, 0.0f
  );
}

bool metal_core_gemm_accumulate_into_f32(
    fastpls::core::ConstMatrixView<float> left,
    fastpls::core::ConstMatrixView<float> right,
    bool transpose_left, bool transpose_right,
    fastpls::core::MatrixView<float> output) {
  return metal_core_gemm_into_f32_impl(
    left, right, transpose_left, transpose_right, output, 1.0f
  );
}

bool metal_core_rank1_subtract_f32(
    fastpls::core::MatrixView<float> target,
    fastpls::core::ConstMatrixView<float> column,
    fastpls::core::ConstMatrixView<float> row) {
  if (target.empty() || column.empty() || row.empty() ||
      column.columns() != 1 || row.rows() != 1 ||
      column.rows() != target.rows() || row.columns() != target.columns()) {
    throw std::invalid_argument("Metal rank-one update dimensions differ");
  }
  const fastpls::core::ConstMatrixView<float> target_view(target);
  if (!can_wrap_shared(target_view)) return false;
  if (!has_metal_backend()) {
    throw std::runtime_error(
      "Metal is unavailable; no CPU fallback is performed"
    );
  }

  @autoreleasepool {
    const MetalMatrix target_storage = wrap_shared(target_view);
    const MetalMatrix column_storage = stage_or_wrap(column);
    const MetalMatrix row_storage = stage_or_wrap(row);
    MPSMatrix* target_matrix = mps_matrix(target_storage);
    MPSMatrix* column_matrix = mps_matrix(column_storage);
    MPSMatrix* row_matrix = mps_matrix(row_storage);
    MPSMatrixMultiplication* update = [[MPSMatrixMultiplication alloc]
      initWithDevice:metal_device()
      transposeLeft:NO
      transposeRight:NO
      resultRows:target_storage.rows
      resultColumns:target_storage.columns
      interiorColumns:1
      alpha:-1.0
      beta:1.0];
    id<MTLCommandBuffer> command = [metal_queue() commandBuffer];
    if (command == nil) {
      throw std::runtime_error("Metal failed to create a command buffer");
    }
    [update encodeToCommandBuffer:command
      leftMatrix:row_matrix
      rightMatrix:column_matrix
      resultMatrix:target_matrix];
    [command commit];
    [command waitUntilCompleted];
    if ([command error] != nil) {
      throw std::runtime_error(
        std::string("Metal rank-one update failed: ") +
        [[[command error] localizedDescription] UTF8String]
      );
    }
  }
  return true;
}

void* metal_sample_gram_workspace_create_f32(
    fastpls::core::ConstMatrixView<float> predictors,
    fastpls::core::ConstMatrixView<float> sample_gram) {
  if (!has_metal_backend()) {
    throw std::runtime_error(
      "Metal is unavailable; no CPU fallback is performed"
    );
  }
  if (predictors.empty() || sample_gram.rows() != predictors.rows() ||
      sample_gram.columns() != predictors.rows()) {
    throw std::invalid_argument(
      "Metal sample-Gram workspace dimensions differ"
    );
  }
  @autoreleasepool {
    return new SampleGramWorkspace(predictors, sample_gram);
  }
}

void metal_sample_gram_workspace_destroy_f32(void* workspace) noexcept {
  if (workspace == nullptr) return;
  @autoreleasepool {
    delete static_cast<SampleGramWorkspace*>(workspace);
  }
}

bool metal_sample_gram_apply_f32(
    void* opaque_workspace,
    fastpls::core::ConstMatrixView<float> direction,
    fastpls::core::MatrixView<float> output) {
  auto* workspace = static_cast<SampleGramWorkspace*>(opaque_workspace);
  if (workspace == nullptr || direction.rows() != workspace->p ||
      direction.columns() != 1 || output.rows() != workspace->p ||
      output.columns() != 1) {
    throw std::invalid_argument(
      "Metal sample-Gram direction dimensions differ"
    );
  }
  @autoreleasepool {
    const MetalMatrix direction_storage = stage_or_wrap(direction);
    const MetalMatrix output_storage = stage_or_wrap(output);
    MPSMatrix* direction_matrix = mps_matrix(direction_storage);
    MPSMatrix* output_matrix = mps_matrix(output_storage);
    id<MTLCommandBuffer> command = [metal_queue() commandBuffer];
    if (command == nil) {
      throw std::runtime_error("Metal failed to create a command buffer");
    }
    [workspace->forward encodeToCommandBuffer:command
      leftMatrix:direction_matrix
      rightMatrix:workspace->predictor_matrix
      resultMatrix:workspace->score_matrix];
    [workspace->sample_product encodeToCommandBuffer:command
      leftMatrix:workspace->score_matrix
      rightMatrix:workspace->sample_gram_matrix
      resultMatrix:workspace->reverse_sample_matrix];
    [workspace->reverse encodeToCommandBuffer:command
      leftMatrix:workspace->reverse_sample_matrix
      rightMatrix:workspace->predictor_matrix
      resultMatrix:output_matrix];
    [command commit];
    [command waitUntilCompleted];
    if ([command error] != nil) {
      throw std::runtime_error(
        std::string("Metal sample-Gram product failed: ") +
        [[[command error] localizedDescription] UTF8String]
      );
    }
    if (output_storage.buffer != nil &&
        !can_wrap_shared(output)) {
      const auto downloaded = download(output_storage);
      std::copy_n(downloaded.data(), output.rows(), output.data());
    }
    return true;
  }
}

bool metal_sample_geometry_f32(
    void* opaque_workspace,
    fastpls::core::ConstMatrixView<float> direction,
    fastpls::core::MatrixView<float> score,
    fastpls::core::MatrixView<float> loading) {
  auto* workspace = static_cast<SampleGramWorkspace*>(opaque_workspace);
  if (workspace == nullptr || direction.rows() != workspace->p ||
      direction.columns() != 1 || score.rows() != workspace->n ||
      score.columns() != 1 || loading.rows() != workspace->p ||
      loading.columns() != 1) {
    throw std::invalid_argument(
      "Metal sample-geometry dimensions differ"
    );
  }
  @autoreleasepool {
    const MetalMatrix direction_storage = stage_or_wrap(direction);
    const MetalMatrix score_storage = stage_or_wrap(score);
    const MetalMatrix loading_storage = stage_or_wrap(loading);
    MPSMatrix* direction_matrix = mps_matrix(direction_storage);
    MPSMatrix* score_matrix = mps_matrix(score_storage);
    MPSMatrix* loading_matrix = mps_matrix(loading_storage);
    id<MTLCommandBuffer> command = [metal_queue() commandBuffer];
    if (command == nil) {
      throw std::runtime_error("Metal failed to create a command buffer");
    }
    [workspace->forward encodeToCommandBuffer:command
      leftMatrix:direction_matrix
      rightMatrix:workspace->predictor_matrix
      resultMatrix:score_matrix];
    [workspace->reverse encodeToCommandBuffer:command
      leftMatrix:score_matrix
      rightMatrix:workspace->predictor_matrix
      resultMatrix:loading_matrix];
    [command commit];
    [command waitUntilCompleted];
    if ([command error] != nil) {
      throw std::runtime_error(
        std::string("Metal sample geometry failed: ") +
        [[[command error] localizedDescription] UTF8String]
      );
    }
    if (!can_wrap_shared(score)) {
      const auto downloaded = download(score_storage);
      std::copy_n(downloaded.data(), score.rows(), score.data());
    }
    if (!can_wrap_shared(loading)) {
      const auto downloaded = download(loading_storage);
      std::copy_n(downloaded.data(), loading.rows(), loading.data());
    }
    return true;
  }
}

}  // namespace fastpls_svd
