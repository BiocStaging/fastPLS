// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore

#include "accelerator_core_backend.h"

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <cstring>
#include <list>
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

}  // namespace fastpls_svd
