#include "metal_resident_backend.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#if __has_include("metal_resident_kernels_data.h")
#include "metal_resident_kernels_data.h"
#define FASTPLS_HAS_PRECOMPILED_RESIDENT_METAL 1
#endif

namespace fastpls_svd {
namespace {

struct MetalMatrix {
  id<MTLBuffer> buffer = nil;
  NSUInteger rows = 0;
  NSUInteger cols = 0;
  NSUInteger row_bytes = 0;
};

MetalMatrix make_matrix(id<MTLDevice> device, NSUInteger rows, NSUInteger cols) {
  if (rows == 0 || cols == 0) {
    throw std::invalid_argument("resident Metal matrices must be nonempty");
  }
  MetalMatrix result;
  result.rows = rows;
  result.cols = cols;
  result.row_bytes = [MPSMatrixDescriptor rowBytesFromColumns:cols
                                                     dataType:MPSDataTypeFloat32];
  result.buffer = [device newBufferWithLength:result.row_bytes * rows
                                      options:MTLResourceStorageModeShared];
  if (result.buffer == nil) {
    throw std::runtime_error("resident Metal buffer allocation failed");
  }
  return result;
}

MetalMatrix leading_columns(const MetalMatrix& matrix, NSUInteger columns) {
  if (columns < 1 || columns > matrix.cols) {
    throw std::invalid_argument("resident Metal column view exceeds storage");
  }
  MetalMatrix result = matrix;
  result.cols = columns;
  return result;
}

MetalMatrix matrix_shape(const MetalMatrix& matrix, NSUInteger rows,
                         NSUInteger columns) {
  if (rows < 1 || columns < 1 || rows > matrix.rows ||
      columns > matrix.cols) {
    throw std::invalid_argument("resident Metal matrix shape exceeds storage");
  }
  MetalMatrix result = matrix;
  result.rows = rows;
  result.cols = columns;
  return result;
}

arma::fmat copy_from_matrix(const MetalMatrix& source,
                            NSUInteger columns = 0) {
  const NSUInteger used_columns = columns == 0 ? source.cols : columns;
  if (used_columns > source.cols) {
    throw std::invalid_argument("resident Metal export exceeds matrix columns");
  }
  arma::fmat result(source.rows, used_columns);
  const char* base = static_cast<const char*>([source.buffer contents]);
  for (NSUInteger row = 0; row < source.rows; ++row) {
    const float* values = reinterpret_cast<const float*>(
        base + row * source.row_bytes);
    for (NSUInteger column = 0; column < used_columns; ++column) {
      result(row, column) = values[column];
    }
  }
  return result;
}

MPSMatrix* matrix_view(const MetalMatrix& matrix) {
  MPSMatrixDescriptor* descriptor = [MPSMatrixDescriptor
      matrixDescriptorWithRows:matrix.rows
                       columns:matrix.cols
                      rowBytes:matrix.row_bytes
                      dataType:MPSDataTypeFloat32];
  return [[MPSMatrix alloc] initWithBuffer:matrix.buffer descriptor:descriptor];
}

MPSMatrix* matrix_view(const MetalMatrix& matrix, NSUInteger rows,
                       NSUInteger columns) {
  if (rows > matrix.rows || columns > matrix.cols) {
    throw std::invalid_argument("resident Metal matrix view exceeds storage");
  }
  MPSMatrixDescriptor* descriptor = [MPSMatrixDescriptor
      matrixDescriptorWithRows:rows
                       columns:columns
                      rowBytes:matrix.row_bytes
                      dataType:MPSDataTypeFloat32];
  return [[MPSMatrix alloc] initWithBuffer:matrix.buffer descriptor:descriptor];
}

class MetalProduct {
 public:
  MetalProduct(id<MTLDevice> device,
               const MetalMatrix& left,
               const MetalMatrix& right,
               const MetalMatrix& result,
               bool transpose_left,
               bool transpose_right,
               double alpha = 1.0,
               double beta = 0.0)
      : left_(matrix_view(left)),
        right_(matrix_view(right)),
        result_(matrix_view(result)) {
    const NSUInteger rows = transpose_left ? left.cols : left.rows;
    const NSUInteger inner = transpose_left ? left.rows : left.cols;
    const NSUInteger right_inner = transpose_right ? right.cols : right.rows;
    const NSUInteger columns = transpose_right ? right.rows : right.cols;
    if (inner != right_inner || rows != result.rows ||
        columns != result.cols) {
      throw std::invalid_argument("resident Metal product dimensions do not conform");
    }
    operation_ = [[MPSMatrixMultiplication alloc]
        initWithDevice:device
         transposeLeft:transpose_left
        transposeRight:transpose_right
            resultRows:rows
         resultColumns:columns
       interiorColumns:inner
                  alpha:alpha
                   beta:beta];
  }

  void encode(id<MTLCommandBuffer> command) const {
    [operation_ encodeToCommandBuffer:command
                           leftMatrix:left_
                          rightMatrix:right_
                          resultMatrix:result_];
  }

 private:
  MPSMatrix* left_;
  MPSMatrix* right_;
  MPSMatrix* result_;
  MPSMatrixMultiplication* operation_;
};

void finish_command(id<MTLCommandBuffer> command, const char* context) {
  [command commit];
  [command waitUntilCompleted];
  NSError* error = [command error];
  if (error != nil) {
    throw std::runtime_error(std::string(context) + ": " +
                             std::string([[error localizedDescription] UTF8String]));
  }
}

void run_product(id<MTLCommandQueue> queue, const MetalProduct& product) {
  id<MTLCommandBuffer> command = [queue commandBuffer];
  if (command == nil) {
    throw std::runtime_error("resident Metal failed to create a command buffer");
  }
  product.encode(command);
  finish_command(command, "resident Metal matrix product failed");
}

class ResidentKernels {
 public:
  explicit ResidentKernels(id<MTLDevice> device) : device_(device) {
    NSString* source = [NSString stringWithUTF8String:R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void column_stats(device float* x [[buffer(0)]],
                         device float* means [[buffer(1)]],
                         device float* scales [[buffer(2)]],
                         constant uint& rows [[buffer(3)]],
                         constant uint& cols [[buffer(4)]],
                         constant uint& stride [[buffer(5)]],
                         constant int& scaling [[buffer(6)]],
                         uint column [[thread_position_in_grid]]) {
  if (column >= cols) return;
  float mean = 0.0f;
  float m2 = 0.0f;
  for (uint row = 0; row < rows; ++row) {
    float value = x[row * stride + column];
    float delta = value - mean;
    mean += delta / float(row + 1);
    m2 += delta * (value - mean);
  }
  means[column] = scaling < 3 ? mean : 0.0f;
  float scale = 1.0f;
  if (scaling == 2 && rows > 1) {
    scale = sqrt(max(m2 / float(rows - 1), 0.0f));
    if (!isfinite(scale) || scale <= 1.0e-20f) scale = 1.0f;
  }
  scales[column] = scale;
}

kernel void column_major_stats(
    device const float* source [[buffer(0)]],
    device float* means [[buffer(1)]],
    device float* scales [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant int& scaling [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],
    uint column [[threadgroup_position_in_grid]],
    uint lanes [[threads_per_threadgroup]]) {
  if (column >= cols) return;
  threadgroup float sums[256];
  threadgroup float squares[256];
  float sum = 0.0f;
  float square_sum = 0.0f;
  for (uint row = lane; row < rows; row += lanes) {
    const float value = source[column * rows + row];
    sum += value;
    square_sum += value * value;
  }
  sums[lane] = sum;
  squares[lane] = square_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint width = lanes >> 1; width > 0; width >>= 1) {
    if (lane < width) {
      sums[lane] += sums[lane + width];
      squares[lane] += squares[lane + width];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0) {
    const float mean = scaling < 3 ? sums[0] / float(rows) : 0.0f;
    float scale = 1.0f;
    if (scaling == 2 && rows > 1) {
      const float centered = max(
          squares[0] - sums[0] * sums[0] / float(rows), 0.0f);
      scale = sqrt(centered / float(rows - 1));
      if (!isfinite(scale) || scale <= 1.0e-20f) scale = 1.0f;
    }
    means[column] = mean;
    scales[column] = scale;
  }
}

kernel void standardize_column_major_to_row_major(
    device const float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    device const float* means [[buffer(2)]],
    device const float* scales [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& destination_stride [[buffer(6)]],
    uint2 index [[thread_position_in_grid]]) {
  if (index.y >= rows || index.x >= cols) return;
  destination[index.y * destination_stride + index.x] =
      (source[index.x * rows + index.y] - means[index.x]) / scales[index.x];
}

kernel void standardize(device float* x [[buffer(0)]],
                        device const float* means [[buffer(1)]],
                        device const float* scales [[buffer(2)]],
                        constant uint& rows [[buffer(3)]],
                        constant uint& cols [[buffer(4)]],
                        constant uint& stride [[buffer(5)]],
                        uint2 index [[thread_position_in_grid]]) {
  if (index.y >= rows || index.x >= cols) return;
  x[index.y * stride + index.x] =
      (x[index.y * stride + index.x] - means[index.x]) / scales[index.x];
}

kernel void identity_statistics(device float* means [[buffer(0)]],
                                device float* scales [[buffer(1)]],
                                constant uint& columns [[buffer(2)]],
                                uint column [[thread_position_in_grid]]) {
  if (column >= columns) return;
  means[column] = 0.0f;
  scales[column] = 1.0f;
}

kernel void vector_difference_ratio(
    device float* output [[buffer(0)]],
    device const float* loading [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device const float* ratio [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& output_stride [[buffer(5)]],
    constant uint& loading_stride [[buffer(6)]],
    constant uint& weight_stride [[buffer(7)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= rows) return;
  output[row * output_stride] =
      loading[row * loading_stride] -
      weight[row * weight_stride] * ratio[0];
}

kernel void rank1_subtract_columns(
    device float* matrix [[buffer(0)]],
    device const float* column [[buffer(1)]],
    device const float* loading [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    constant uint& matrix_stride [[buffer(5)]],
    constant uint& column_stride [[buffer(6)]],
    constant uint& loading_stride [[buffer(7)]],
    uint2 index [[thread_position_in_grid]]) {
  if (index.y >= rows || index.x >= columns) return;
  matrix[index.y * matrix_stride + index.x] -=
      column[index.y * column_stride] *
      loading[index.x * loading_stride];
}

kernel void subtract_matrix(device float* destination [[buffer(0)]],
                            device const float* correction [[buffer(1)]],
                            constant uint& rows [[buffer(2)]],
                            constant uint& columns [[buffer(3)]],
                            constant uint& destination_stride [[buffer(4)]],
                            constant uint& correction_stride [[buffer(5)]],
                            uint2 index [[thread_position_in_grid]]) {
  if (index.y >= rows || index.x >= columns) return;
  destination[index.y * destination_stride + index.x] -=
      correction[index.y * correction_stride + index.x];
}

kernel void copy_block_columns(device const float* source [[buffer(0)]],
                               device float* destination [[buffer(1)]],
                               constant uint& rows [[buffer(2)]],
                               constant uint& columns [[buffer(3)]],
                               constant uint& destination_offset [[buffer(4)]],
                               constant uint& source_stride [[buffer(5)]],
                               constant uint& destination_stride [[buffer(6)]],
                               uint2 index [[thread_position_in_grid]]) {
  if (index.y >= rows || index.x >= columns) return;
  destination[index.y * destination_stride + destination_offset + index.x] =
      source[index.y * source_stride + index.x];
}

kernel void scalar_ratio(device const float* numerator [[buffer(0)]],
                         device const float* denominator [[buffer(1)]],
                         device float* output [[buffer(2)]],
                         device atomic_int* invalid [[buffer(3)]],
                         uint index [[thread_position_in_grid]]) {
  if (index != 0) return;
  const float value = denominator[0];
  if (!isfinite(value) || value <= 1.0e-40f ||
      !isfinite(numerator[0])) {
    atomic_store_explicit(invalid, 1, memory_order_relaxed);
    output[0] = 0.0f;
    return;
  }
  output[0] = numerator[0] / value;
}

kernel void reciprocal_scalar(device float* value [[buffer(0)]],
                              device atomic_int* invalid [[buffer(1)]],
                              uint index [[thread_position_in_grid]]) {
  if (index != 0) return;
  if (!isfinite(value[0]) || value[0] <= 1.0e-40f) {
    atomic_store_explicit(invalid, 1, memory_order_relaxed);
    value[0] = 0.0f;
    return;
  }
  value[0] = 1.0f / value[0];
}

kernel void row_squared_norms(device const float* matrix [[buffer(0)]],
                              device float* norms [[buffer(1)]],
                              constant uint& rows [[buffer(2)]],
                              constant uint& columns [[buffer(3)]],
                              constant uint& matrix_stride [[buffer(4)]],
                              uint row [[thread_position_in_grid]]) {
  if (row >= rows) return;
  float total = 0.0f;
  for (uint column = 0; column < columns; ++column) {
    const float value = matrix[row * matrix_stride + column];
    total += value * value;
  }
  norms[row] = total;
}

kernel void transform_kernel(device float* matrix [[buffer(0)]],
                             device const float* left_norms [[buffer(1)]],
                             device const float* right_norms [[buffer(2)]],
                             constant uint& rows [[buffer(3)]],
                             constant uint& columns [[buffer(4)]],
                             constant uint& matrix_stride [[buffer(5)]],
                             constant int& kind [[buffer(6)]],
                             constant float& gamma [[buffer(7)]],
                             constant int& degree [[buffer(8)]],
                             constant float& coefficient [[buffer(9)]],
                             uint2 index [[thread_position_in_grid]]) {
  if (index.y >= rows || index.x >= columns) return;
  const uint offset = index.y * matrix_stride + index.x;
  const float dot = matrix[offset];
  if (kind == 2) {
    const float distance = max(
        0.0f, left_norms[index.y] + right_norms[index.x] - 2.0f * dot);
    matrix[offset] = exp(-gamma * distance);
  } else {
    float value = 1.0f;
    const float base = gamma * dot + coefficient;
    for (int power = 0; power < degree; ++power) value *= base;
    matrix[offset] = value;
  }
}

kernel void column_means(device const float* matrix [[buffer(0)]],
                         device float* means [[buffer(1)]],
                         constant uint& rows [[buffer(2)]],
                         constant uint& columns [[buffer(3)]],
                         constant uint& matrix_stride [[buffer(4)]],
                         uint column [[thread_position_in_grid]]) {
  if (column >= columns) return;
  float total = 0.0f;
  for (uint row = 0; row < rows; ++row) {
    total += matrix[row * matrix_stride + column];
  }
  means[column] = total / float(rows);
}

kernel void vector_mean(device const float* values [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        constant uint& size [[buffer(2)]],
                        uint index [[thread_position_in_grid]]) {
  if (index != 0) return;
  float total = 0.0f;
  for (uint i = 0; i < size; ++i) total += values[i];
  output[0] = total / float(size);
}

kernel void center_training_kernel(
    device float* matrix [[buffer(0)]],
    device const float* means [[buffer(1)]],
    device const float* grand [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    constant uint& matrix_stride [[buffer(5)]],
    uint2 index [[thread_position_in_grid]]) {
  if (index.y >= rows || index.x >= columns) return;
  matrix[index.y * matrix_stride + index.x] -=
      means[index.y] + means[index.x] - grand[0];
}

kernel void row_means(device const float* matrix [[buffer(0)]],
                      device float* means [[buffer(1)]],
                      constant uint& rows [[buffer(2)]],
                      constant uint& columns [[buffer(3)]],
                      constant uint& matrix_stride [[buffer(4)]],
                      uint row [[thread_position_in_grid]]) {
  if (row >= rows) return;
  float total = 0.0f;
  for (uint column = 0; column < columns; ++column) {
    total += matrix[row * matrix_stride + column];
  }
  means[row] = total / float(columns);
}

kernel void center_test_kernel(
    device float* matrix [[buffer(0)]],
    device const float* row_average [[buffer(1)]],
    device const float* training_average [[buffer(2)]],
    device const float* grand [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& columns [[buffer(5)]],
    constant uint& matrix_stride [[buffer(6)]],
    uint2 index [[thread_position_in_grid]]) {
  if (index.y >= rows || index.x >= columns) return;
  matrix[index.y * matrix_stride + index.x] -=
      row_average[index.y] + training_average[index.x] - grand[0];
}

kernel void column_major_to_row_major(
    device const float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& columns [[buffer(3)]],
    constant uint& destination_stride [[buffer(4)]],
    uint2 index [[thread_position_in_grid]]) {
  if (index.y >= rows || index.x >= columns) return;
  destination[index.y * destination_stride + index.x] =
      source[index.y + index.x * rows];
}

kernel void class_priors(device const int* labels [[buffer(0)]],
                         device float* priors [[buffer(1)]],
                         device atomic_int* invalid [[buffer(2)]],
                         constant uint& rows [[buffer(3)]],
                         constant uint& classes [[buffer(4)]],
                         uint cls [[thread_position_in_grid]]) {
  if (cls >= classes) return;
  uint count = 0;
  for (uint row = 0; row < rows; ++row) {
    int value = labels[row];
    if (value < 1 || value > int(classes)) {
      atomic_store_explicit(invalid, 1, memory_order_relaxed);
    } else if (value == int(cls + 1)) {
      ++count;
    }
  }
  if (count == 0) atomic_store_explicit(invalid, 1, memory_order_relaxed);
  priors[cls] = float(count) / float(rows);
}

kernel void class_product(device const float* x [[buffer(0)]],
                          device const int* labels [[buffer(1)]],
                          device float* result [[buffer(2)]],
                          constant uint& rows [[buffer(3)]],
                          constant uint& predictors [[buffer(4)]],
                          constant uint& classes [[buffer(5)]],
                          constant uint& x_stride [[buffer(6)]],
                          constant uint& result_stride [[buffer(7)]],
                          uint predictor [[thread_position_in_grid]]) {
  if (predictor >= predictors) return;
  for (uint cls = 0; cls < classes; ++cls) {
    result[predictor * result_stride + cls] = 0.0f;
  }
  for (uint row = 0; row < rows; ++row) {
    int cls = labels[row] - 1;
    if (cls >= 0 && cls < int(classes)) {
      result[predictor * result_stride + uint(cls)] +=
          x[row * x_stride + predictor];
    }
  }
}

kernel void dummy_response(device const int* labels [[buffer(0)]],
                           device const float* priors [[buffer(1)]],
                           device float* response [[buffer(2)]],
                           constant uint& rows [[buffer(3)]],
                           constant uint& classes [[buffer(4)]],
                           constant uint& response_stride [[buffer(5)]],
                           uint2 index [[thread_position_in_grid]]) {
  if (index.y >= rows || index.x >= classes) return;
  response[index.y * response_stride + index.x] =
      (labels[index.y] == int(index.x + 1) ? 1.0f : 0.0f) - priors[index.x];
}

kernel void random_vector(device float* values [[buffer(0)]],
                          constant uint& size [[buffer(1)]],
                          constant uint& seed [[buffer(2)]],
                          constant uint& stride [[buffer(3)]],
                          uint index [[thread_position_in_grid]]) {
  if (index >= size) return;
  uint x = seed ^ ((index + 1u) * 747796405u + 2891336453u);
  x ^= x >> 16;
  x *= 2246822519u;
  x ^= x >> 13;
  x *= 3266489917u;
  x ^= x >> 16;
  float unit = (float(x) + 1.0f) / 4294967297.0f;
  values[index * stride] = 2.0f * unit - 1.0f;
}

kernel void random_matrix(device float* values [[buffer(0)]],
                          constant uint& rows [[buffer(1)]],
                          constant uint& columns [[buffer(2)]],
                          constant uint& seed [[buffer(3)]],
                          constant uint& stride [[buffer(4)]],
                          uint2 index [[thread_position_in_grid]]) {
  if (index.y >= rows || index.x >= columns) return;
  uint linear = index.y * columns + index.x;
  uint x = seed ^ ((linear + 1u) * 747796405u + 2891336453u);
  x ^= x >> 16;
  x *= 2246822519u;
  x ^= x >> 13;
  x *= 3266489917u;
  x ^= x >> 16;
  float unit = (float(x) + 1.0f) / 4294967297.0f;
  values[index.y * stride + index.x] = 2.0f * unit - 1.0f;
}

kernel void matrix_vector(device const float* matrix [[buffer(0)]],
                          device const float* input [[buffer(1)]],
                          device float* output [[buffer(2)]],
                          constant uint& rows [[buffer(3)]],
                          constant uint& columns [[buffer(4)]],
                          constant uint& matrix_stride [[buffer(5)]],
                          constant uint& input_stride [[buffer(6)]],
                          constant uint& output_stride [[buffer(7)]],
                          uint row [[thread_position_in_grid]]) {
  if (row >= rows) return;
  float value = 0.0f;
  for (uint column = 0; column < columns; ++column) {
    value += matrix[row * matrix_stride + column] *
             input[column * input_stride];
  }
  output[row * output_stride] = value;
}

kernel void matrix_transpose_vector(
    device const float* matrix [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    constant uint& matrix_stride [[buffer(5)]],
    constant uint& input_stride [[buffer(6)]],
    constant uint& output_stride [[buffer(7)]],
    uint column [[thread_position_in_grid]]) {
  if (column >= columns) return;
  float value = 0.0f;
  for (uint row = 0; row < rows; ++row) {
    value += matrix[row * matrix_stride + column] *
             input[row * input_stride];
  }
  output[column * output_stride] = value;
}

kernel void orthonormalize(device float* vector [[buffer(0)]],
                           device const float* basis [[buffer(1)]],
                           device atomic_int* invalid [[buffer(2)]],
                           constant uint& size [[buffer(3)]],
                           constant uint& used [[buffer(4)]],
                           constant uint& vector_stride [[buffer(5)]],
                           constant uint& basis_stride [[buffer(6)]],
                           uint index [[thread_position_in_grid]]) {
  if (index != 0) return;
  for (uint pass = 0; pass < 2; ++pass) {
    for (uint column = 0; column < used; ++column) {
      float dot = 0.0f;
      for (uint row = 0; row < size; ++row) {
        dot += basis[row * basis_stride + column] * vector[row * vector_stride];
      }
      for (uint row = 0; row < size; ++row) {
        vector[row * vector_stride] -= dot * basis[row * basis_stride + column];
      }
    }
  }
  float norm2 = 0.0f;
  for (uint row = 0; row < size; ++row) {
    float value = vector[row * vector_stride];
    norm2 += value * value;
  }
  float norm = sqrt(norm2);
  if (!isfinite(norm) || norm <= 1.0e-20f) {
    atomic_store_explicit(invalid, 1, memory_order_relaxed);
    return;
  }
  for (uint row = 0; row < size; ++row) vector[row * vector_stride] /= norm;
}

kernel void orthonormalize_block(
    device float* block [[buffer(0)]],
    device const float* external_basis [[buffer(1)]],
    device atomic_int* invalid [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    constant uint& external_used [[buffer(5)]],
    constant uint& block_stride [[buffer(6)]],
    constant uint& basis_stride [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
  if (index != 0) return;
  for (uint column = 0; column < columns; ++column) {
    for (uint pass = 0; pass < 2; ++pass) {
      for (uint basis_column = 0; basis_column < external_used; ++basis_column) {
        float dot = 0.0f;
        for (uint row = 0; row < rows; ++row) {
          dot += external_basis[row * basis_stride + basis_column] *
                 block[row * block_stride + column];
        }
        for (uint row = 0; row < rows; ++row) {
          block[row * block_stride + column] -=
              dot * external_basis[row * basis_stride + basis_column];
        }
      }
      for (uint prior = 0; prior < column; ++prior) {
        float dot = 0.0f;
        for (uint row = 0; row < rows; ++row) {
          dot += block[row * block_stride + prior] *
                 block[row * block_stride + column];
        }
        for (uint row = 0; row < rows; ++row) {
          block[row * block_stride + column] -=
              dot * block[row * block_stride + prior];
        }
      }
    }
    float norm2 = 0.0f;
    for (uint row = 0; row < rows; ++row) {
      float value = block[row * block_stride + column];
      norm2 += value * value;
    }
    float norm = sqrt(norm2);
    if (!isfinite(norm) || norm <= 1.0e-20f) {
      atomic_store_explicit(invalid, 1, memory_order_relaxed);
      return;
    }
    for (uint row = 0; row < rows; ++row) {
      block[row * block_stride + column] /= norm;
    }
  }
}

kernel void orthonormalize_block_parallel(
    device float* block [[buffer(0)]],
    device const float* external_basis [[buffer(1)]],
    device atomic_int* invalid [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    constant uint& external_used [[buffer(5)]],
    constant uint& block_stride [[buffer(6)]],
    constant uint& basis_stride [[buffer(7)]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
  threadgroup float partial[256];
  for (uint column = 0; column < columns; ++column) {
    for (uint pass = 0; pass < 2; ++pass) {
      for (uint base = 0; base < external_used; base += lanes) {
        const uint basis_column = base + lane;
        float dot = 0.0f;
        if (basis_column < external_used) {
          for (uint row = 0; row < rows; ++row) {
            dot += external_basis[row * basis_stride + basis_column] *
                   block[row * block_stride + column];
          }
        }
        partial[lane] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint row = lane; row < rows; row += lanes) {
          float correction = 0.0f;
          const uint count = min(lanes, external_used - base);
          for (uint offset = 0; offset < count; ++offset) {
            correction += partial[offset] *
                external_basis[row * basis_stride + base + offset];
          }
          block[row * block_stride + column] -= correction;
        }
        threadgroup_barrier(mem_flags::mem_device);
      }
      float dot = 0.0f;
      if (lane < column) {
        for (uint row = 0; row < rows; ++row) {
          dot += block[row * block_stride + lane] *
                 block[row * block_stride + column];
        }
      }
      partial[lane] = dot;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint row = lane; row < rows; row += lanes) {
        float correction = 0.0f;
        for (uint prior = 0; prior < column; ++prior) {
          correction += partial[prior] *
              block[row * block_stride + prior];
        }
        block[row * block_stride + column] -= correction;
      }
      threadgroup_barrier(mem_flags::mem_device);
    }
    float norm2 = 0.0f;
    for (uint row = lane; row < rows; row += lanes) {
      float value = block[row * block_stride + column];
      norm2 += value * value;
    }
    partial[lane] = norm2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = lanes >> 1; width > 0; width >>= 1) {
      if (lane < width) partial[lane] += partial[lane + width];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float norm = sqrt(partial[0]);
    if (!isfinite(norm) || norm <= 1.0e-20f) {
      if (lane == 0) atomic_store_explicit(invalid, 1, memory_order_relaxed);
      return;
    }
    for (uint row = lane; row < rows; row += lanes) {
      block[row * block_stride + column] /= norm;
    }
    threadgroup_barrier(mem_flags::mem_device);
  }
}

kernel void append_cached_simpls_block(
    device const float* candidates [[buffer(0)]],
    device const float* predictor_gram [[buffer(1)]],
    device const float* original_crosscov [[buffer(2)]],
    device float* crosscov [[buffer(3)]],
    device float* predictor_loading [[buffer(4)]],
    device float* response_loading [[buffer(5)]],
    device float* orthogonal_loading [[buffer(6)]],
    device float* deflation_row [[buffer(7)]],
    device float* directions [[buffer(8)]],
    device float* response_loadings [[buffer(9)]],
    device float* orthogonal_loadings [[buffer(10)]],
    device float* predictor_loadings [[buffer(11)]],
    device atomic_int* invalid [[buffer(12)]],
    constant uint& predictors [[buffer(13)]],
    constant uint& responses [[buffer(14)]],
    constant uint& block_columns [[buffer(15)]],
    constant uint& component_offset [[buffer(16)]],
    constant uint& candidate_stride [[buffer(17)]],
    constant uint& gram_stride [[buffer(18)]],
    constant uint& original_crosscov_stride [[buffer(19)]],
    constant uint& crosscov_stride [[buffer(20)]],
    constant uint& vector_stride [[buffer(21)]],
    constant uint& direction_stride [[buffer(22)]],
    constant uint& response_loading_stride [[buffer(23)]],
    constant uint& orthogonal_loading_stride [[buffer(24)]],
    constant uint& predictor_loading_stride [[buffer(25)]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
  threadgroup float partial[256];
  for (uint offset = 0; offset < block_columns; ++offset) {
    const uint component = component_offset + offset;

    for (uint row = lane; row < predictors; row += lanes) {
      float value = 0.0f;
      for (uint column = 0; column < predictors; ++column) {
        value += predictor_gram[row * gram_stride + column] *
                 candidates[column * candidate_stride + offset];
      }
      predictor_loading[row * vector_stride] = value;
    }
    threadgroup_barrier(mem_flags::mem_device);

    float norm2 = 0.0f;
    for (uint row = lane; row < predictors; row += lanes) {
      norm2 += candidates[row * candidate_stride + offset] *
               predictor_loading[row * vector_stride];
    }
    partial[lane] = norm2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = lanes >> 1; width > 0; width >>= 1) {
      if (lane < width) partial[lane] += partial[lane + width];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float norm = sqrt(partial[0]);
    if (!isfinite(norm) || norm <= 1.0e-20f) {
      if (lane == 0) atomic_store_explicit(invalid, 1, memory_order_relaxed);
      return;
    }
    const float reciprocal = 1.0f / norm;
    for (uint row = lane; row < predictors; row += lanes) {
      const float direction =
          candidates[row * candidate_stride + offset] * reciprocal;
      const float loading = predictor_loading[row * vector_stride] * reciprocal;
      predictor_loading[row * vector_stride] = loading;
      orthogonal_loading[row * vector_stride] = loading;
      directions[row * direction_stride + component] = direction;
      predictor_loadings[row * predictor_loading_stride + component] = loading;
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (uint response = lane; response < responses; response += lanes) {
      float value = 0.0f;
      for (uint row = 0; row < predictors; ++row) {
        value += original_crosscov[row * original_crosscov_stride + response] *
                 directions[row * direction_stride + component];
      }
      response_loading[response * vector_stride] = value;
      response_loadings[response * response_loading_stride + component] = value;
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (uint pass = 0; pass < 2; ++pass) {
      for (uint prior = 0; prior < component; ++prior) {
        float dot = 0.0f;
        for (uint row = lane; row < predictors; row += lanes) {
          dot += orthogonal_loadings[row * orthogonal_loading_stride + prior] *
                 orthogonal_loading[row * vector_stride];
        }
        partial[lane] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint width = lanes >> 1; width > 0; width >>= 1) {
          if (lane < width) partial[lane] += partial[lane + width];
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float projection = partial[0];
        for (uint row = lane; row < predictors; row += lanes) {
          orthogonal_loading[row * vector_stride] -= projection *
              orthogonal_loadings[row * orthogonal_loading_stride + prior];
        }
        threadgroup_barrier(mem_flags::mem_device);
      }
    }

    float loading_norm2 = 0.0f;
    for (uint row = lane; row < predictors; row += lanes) {
      const float value = orthogonal_loading[row * vector_stride];
      loading_norm2 += value * value;
    }
    partial[lane] = loading_norm2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = lanes >> 1; width > 0; width >>= 1) {
      if (lane < width) partial[lane] += partial[lane + width];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float loading_norm = sqrt(partial[0]);
    if (!isfinite(loading_norm) || loading_norm <= 1.0e-20f) {
      if (lane == 0) atomic_store_explicit(invalid, 1, memory_order_relaxed);
      return;
    }
    const float loading_reciprocal = 1.0f / loading_norm;
    for (uint row = lane; row < predictors; row += lanes) {
      const float value = orthogonal_loading[row * vector_stride] *
          loading_reciprocal;
      orthogonal_loading[row * vector_stride] = value;
      orthogonal_loadings[row * orthogonal_loading_stride + component] = value;
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (uint response = lane; response < responses; response += lanes) {
      float value = 0.0f;
      for (uint row = 0; row < predictors; ++row) {
        value += orthogonal_loading[row * vector_stride] *
                 crosscov[row * crosscov_stride + response];
      }
      deflation_row[response] = value;
    }
    threadgroup_barrier(mem_flags::mem_device);
    const uint crosscov_size = predictors * responses;
    for (uint index = lane; index < crosscov_size; index += lanes) {
      const uint row = index / responses;
      const uint response = index - row * responses;
      crosscov[row * crosscov_stride + response] -=
          orthogonal_loading[row * vector_stride] * deflation_row[response];
    }
    threadgroup_barrier(mem_flags::mem_device);
  }
}

kernel void append_precomputed_simpls_block(
    device const float* candidate_directions [[buffer(0)]],
    device const float* candidate_loadings [[buffer(1)]],
    device const float* candidate_responses [[buffer(2)]],
    device float* crosscov [[buffer(3)]],
    device float* orthogonal_loading [[buffer(4)]],
    device float* deflation_row [[buffer(5)]],
    device float* directions [[buffer(6)]],
    device float* response_loadings [[buffer(7)]],
    device float* orthogonal_loadings [[buffer(8)]],
    device float* predictor_loadings [[buffer(9)]],
    device float* projections [[buffer(10)]],
    device atomic_int* invalid [[buffer(11)]],
    constant uint& predictors [[buffer(12)]],
    constant uint& responses [[buffer(13)]],
    constant uint& block_columns [[buffer(14)]],
    constant uint& component_offset [[buffer(15)]],
    constant uint& candidate_direction_stride [[buffer(16)]],
    constant uint& candidate_loading_stride [[buffer(17)]],
    constant uint& candidate_response_stride [[buffer(18)]],
    constant uint& crosscov_stride [[buffer(19)]],
    constant uint& vector_stride [[buffer(20)]],
    constant uint& direction_stride [[buffer(21)]],
    constant uint& response_loading_stride [[buffer(22)]],
    constant uint& orthogonal_loading_stride [[buffer(23)]],
    constant uint& predictor_loading_stride [[buffer(24)]],
    constant uint& projection_stride [[buffer(25)]],
    uint lane [[thread_index_in_threadgroup]],
    uint lanes [[threads_per_threadgroup]]) {
  threadgroup float partial[256];
  for (uint offset = 0; offset < block_columns; ++offset) {
    const uint component = component_offset + offset;
    for (uint row = lane; row < predictors; row += lanes) {
      const float direction =
          candidate_directions[row * candidate_direction_stride + offset];
      const float loading =
          candidate_loadings[row * candidate_loading_stride + offset];
      directions[row * direction_stride + component] = direction;
      predictor_loadings[row * predictor_loading_stride + component] = loading;
      orthogonal_loading[row * vector_stride] = loading;
    }
    for (uint response = lane; response < responses; response += lanes) {
      response_loadings[response * response_loading_stride + component] =
          candidate_responses[response * candidate_response_stride + offset];
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (uint pass = 0; pass < 2; ++pass) {
      for (uint prior = lane; prior < component; prior += lanes) {
        float dot = 0.0f;
        for (uint row = 0; row < predictors; ++row) {
          dot += orthogonal_loadings[row * orthogonal_loading_stride + prior] *
                 orthogonal_loading[row * vector_stride];
        }
        projections[prior * projection_stride] = dot;
      }
      threadgroup_barrier(mem_flags::mem_device);
      for (uint row = lane; row < predictors; row += lanes) {
        float correction = 0.0f;
        for (uint prior = 0; prior < component; ++prior) {
          correction += orthogonal_loadings[
              row * orthogonal_loading_stride + prior] *
              projections[prior * projection_stride];
        }
        orthogonal_loading[row * vector_stride] -= correction;
      }
      threadgroup_barrier(mem_flags::mem_device);
    }

    float loading_norm2 = 0.0f;
    for (uint row = lane; row < predictors; row += lanes) {
      const float value = orthogonal_loading[row * vector_stride];
      loading_norm2 += value * value;
    }
    partial[lane] = loading_norm2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = lanes >> 1; width > 0; width >>= 1) {
      if (lane < width) partial[lane] += partial[lane + width];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float loading_norm = sqrt(partial[0]);
    if (!isfinite(loading_norm) || loading_norm <= 1.0e-20f) {
      if (lane == 0) atomic_store_explicit(invalid, 1, memory_order_relaxed);
      return;
    }
    const float reciprocal = 1.0f / loading_norm;
    for (uint row = lane; row < predictors; row += lanes) {
      const float value = orthogonal_loading[row * vector_stride] * reciprocal;
      orthogonal_loading[row * vector_stride] = value;
      orthogonal_loadings[row * orthogonal_loading_stride + component] = value;
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (uint response = lane; response < responses; response += lanes) {
      float value = 0.0f;
      for (uint row = 0; row < predictors; ++row) {
        value += orthogonal_loading[row * vector_stride] *
                 crosscov[row * crosscov_stride + response];
      }
      deflation_row[response] = value;
    }
    threadgroup_barrier(mem_flags::mem_device);
    const uint crosscov_size = predictors * responses;
    for (uint index = lane; index < crosscov_size; index += lanes) {
      const uint row = index / responses;
      const uint response = index - row * responses;
      crosscov[row * crosscov_stride + response] -=
          orthogonal_loading[row * vector_stride] * deflation_row[response];
    }
    threadgroup_barrier(mem_flags::mem_device);
  }
}

kernel void extract_column(device const float* matrix [[buffer(0)]],
                           device float* vector [[buffer(1)]],
                           constant uint& rows [[buffer(2)]],
                           constant uint& matrix_stride [[buffer(3)]],
                           constant uint& vector_stride [[buffer(4)]],
                           constant uint& column [[buffer(5)]],
                           uint row [[thread_position_in_grid]]) {
  if (row >= rows) return;
  vector[row * vector_stride] = matrix[row * matrix_stride + column];
}

kernel void symmetric_eigenvectors(device float* matrix [[buffer(0)]],
                                   device float* vectors [[buffer(1)]],
                                   device atomic_int* invalid [[buffer(2)]],
                                   constant uint& size [[buffer(3)]],
                                   constant uint& matrix_stride [[buffer(4)]],
                                   constant uint& vector_stride [[buffer(5)]],
                                   uint index [[thread_position_in_grid]]) {
  if (index != 0) return;
  for (uint row = 0; row < size; ++row) {
    for (uint column = 0; column < size; ++column) {
      vectors[row * vector_stride + column] = row == column ? 1.0f : 0.0f;
    }
  }
  for (uint sweep = 0; sweep < 12; ++sweep) {
    float largest = 0.0f;
    float diagonal_scale = 0.0f;
    for (uint row = 0; row < size; ++row) {
      diagonal_scale = max(diagonal_scale,
          abs(matrix[row * matrix_stride + row]));
    }
    for (uint pivot_row = 0; pivot_row < size; ++pivot_row) {
      for (uint pivot_column = pivot_row + 1;
           pivot_column < size; ++pivot_column) {
        float app = matrix[pivot_row * matrix_stride + pivot_row];
        float aqq = matrix[pivot_column * matrix_stride + pivot_column];
        float apq = matrix[pivot_row * matrix_stride + pivot_column];
        largest = max(largest, abs(apq));
        if (abs(apq) <= 1.0e-7f * max(1.0f, diagonal_scale)) continue;
        if (!isfinite(app) || !isfinite(aqq) || !isfinite(apq)) {
          atomic_store_explicit(invalid, 1, memory_order_relaxed);
          return;
        }
        float angle = 0.5f * atan2(2.0f * apq, aqq - app);
        float cosine = cos(angle);
        float sine = sin(angle);
        for (uint row = 0; row < size; ++row) {
          if (row == pivot_row || row == pivot_column) continue;
          float arp = matrix[row * matrix_stride + pivot_row];
          float arq = matrix[row * matrix_stride + pivot_column];
          float next_p = cosine * arp - sine * arq;
          float next_q = sine * arp + cosine * arq;
          matrix[row * matrix_stride + pivot_row] = next_p;
          matrix[pivot_row * matrix_stride + row] = next_p;
          matrix[row * matrix_stride + pivot_column] = next_q;
          matrix[pivot_column * matrix_stride + row] = next_q;
        }
        matrix[pivot_row * matrix_stride + pivot_row] =
            cosine * cosine * app - 2.0f * sine * cosine * apq +
            sine * sine * aqq;
        matrix[pivot_column * matrix_stride + pivot_column] =
            sine * sine * app + 2.0f * sine * cosine * apq +
            cosine * cosine * aqq;
        matrix[pivot_row * matrix_stride + pivot_column] = 0.0f;
        matrix[pivot_column * matrix_stride + pivot_row] = 0.0f;
        for (uint row = 0; row < size; ++row) {
          float vrp = vectors[row * vector_stride + pivot_row];
          float vrq = vectors[row * vector_stride + pivot_column];
          vectors[row * vector_stride + pivot_row] = cosine * vrp - sine * vrq;
          vectors[row * vector_stride + pivot_column] = sine * vrp + cosine * vrq;
        }
      }
    }
    if (largest <= 1.0e-6f * max(1.0f, diagonal_scale)) break;
  }
  for (uint target = 0; target < size; ++target) {
    uint best = target;
    float best_value = matrix[target * matrix_stride + target];
    for (uint candidate = target + 1; candidate < size; ++candidate) {
      float value = matrix[candidate * matrix_stride + candidate];
      if (value > best_value) {
        best = candidate;
        best_value = value;
      }
    }
    if (best != target) {
      float diagonal = matrix[target * matrix_stride + target];
      matrix[target * matrix_stride + target] =
          matrix[best * matrix_stride + best];
      matrix[best * matrix_stride + best] = diagonal;
      for (uint row = 0; row < size; ++row) {
        float value = vectors[row * vector_stride + target];
        vectors[row * vector_stride + target] =
            vectors[row * vector_stride + best];
        vectors[row * vector_stride + best] = value;
      }
    }
  }
}

kernel void cholesky_inverse_upper(
    device float* gram [[buffer(0)]],
    device float* inverse [[buffer(1)]],
    device atomic_int* invalid [[buffer(2)]],
    constant uint& size [[buffer(3)]],
    constant uint& gram_stride [[buffer(4)]],
    constant uint& inverse_stride [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
  if (index != 0) return;
  for (uint row = 0; row < size; ++row) {
    for (uint column = 0; column < size; ++column) {
      inverse[row * inverse_stride + column] = 0.0f;
    }
  }
  for (uint row = 0; row < size; ++row) {
    for (uint column = 0; column <= row; ++column) {
      float value = gram[row * gram_stride + column];
      for (uint prior = 0; prior < column; ++prior) {
        value -= gram[row * gram_stride + prior] *
                 gram[column * gram_stride + prior];
      }
      if (row == column) {
        if (!isfinite(value) || value <= 1.0e-12f) {
          atomic_store_explicit(invalid, 1, memory_order_relaxed);
          return;
        }
        gram[row * gram_stride + column] = sqrt(value);
      } else {
        gram[row * gram_stride + column] =
            value / gram[column * gram_stride + column];
      }
    }
  }
  for (uint rhs = 0; rhs < size; ++rhs) {
    for (int row = int(size) - 1; row >= 0; --row) {
      float value = uint(row) == rhs ? 1.0f : 0.0f;
      for (uint column = uint(row) + 1; column < size; ++column) {
        value -= gram[column * gram_stride + uint(row)] *
                 inverse[column * inverse_stride + rhs];
      }
      inverse[uint(row) * inverse_stride + rhs] =
          value / gram[uint(row) * gram_stride + uint(row)];
    }
  }
}

kernel void regularize_gram(device float* matrix [[buffer(0)]],
                            constant uint& size [[buffer(1)]],
                            constant uint& stride [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
  if (index != 0) return;
  float trace = 0.0f;
  for (uint diagonal = 0; diagonal < size; ++diagonal) {
    trace += matrix[diagonal * stride + diagonal];
  }
  float ridge = 1.0e-6f * max(1.0f, trace / float(size));
  for (uint diagonal = 0; diagonal < size; ++diagonal) {
    matrix[diagonal * stride + diagonal] += ridge;
  }
}

kernel void normalize_score_pair(device float* score [[buffer(0)]],
                                 device float* direction [[buffer(1)]],
                                 device atomic_int* invalid [[buffer(2)]],
                                 constant uint& rows [[buffer(3)]],
                                 constant uint& score_stride [[buffer(4)]],
                                 constant uint& direction_size [[buffer(5)]],
                                 constant uint& direction_stride [[buffer(6)]],
                                 uint index [[thread_position_in_grid]]) {
  if (index != 0) return;
  float norm2 = 0.0f;
  for (uint row = 0; row < rows; ++row) {
    float value = score[row * score_stride];
    norm2 += value * value;
  }
  float norm = sqrt(norm2);
  if (!isfinite(norm) || norm <= 1.0e-20f) {
    atomic_store_explicit(invalid, 1, memory_order_relaxed);
    return;
  }
  for (uint row = 0; row < rows; ++row) score[row * score_stride] /= norm;
  for (uint row = 0; row < direction_size; ++row) {
    direction[row * direction_stride] /= norm;
  }
}

kernel void reciprocal_sqrt_scalar(device float* value [[buffer(0)]],
                                   device atomic_int* invalid [[buffer(1)]],
                                   uint index [[thread_position_in_grid]]) {
  if (index != 0) return;
  float norm2 = value[0];
  if (!isfinite(norm2) || norm2 <= 1.0e-40f) {
    atomic_store_explicit(invalid, 1, memory_order_relaxed);
    value[0] = 0.0f;
    return;
  }
  value[0] = rsqrt(norm2);
}

kernel void scale_score_pair(device float* score [[buffer(0)]],
                             device float* direction [[buffer(1)]],
                             device const float* reciprocal [[buffer(2)]],
                             constant uint& score_rows [[buffer(3)]],
                             constant uint& direction_rows [[buffer(4)]],
                             constant uint& score_stride [[buffer(5)]],
                             constant uint& direction_stride [[buffer(6)]],
                             uint index [[thread_position_in_grid]]) {
  float scale = reciprocal[0];
  if (index < score_rows) score[index * score_stride] *= scale;
  if (index < direction_rows) direction[index * direction_stride] *= scale;
}

kernel void scale_vector(device float* vector [[buffer(0)]],
                         device const float* reciprocal [[buffer(1)]],
                         constant uint& rows [[buffer(2)]],
                         constant uint& stride [[buffer(3)]],
                         uint index [[thread_position_in_grid]]) {
  if (index < rows) vector[index * stride] *= reciprocal[0];
}

kernel void normalize_candidate_geometry(
    device float* directions [[buffer(0)]],
    device float* loadings [[buffer(1)]],
    device atomic_int* invalid [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    constant uint& direction_stride [[buffer(5)]],
    constant uint& loading_stride [[buffer(6)]],
    uint column [[thread_position_in_grid]]) {
  if (column >= columns) return;
  float norm2 = 0.0f;
  for (uint row = 0; row < rows; ++row) {
    norm2 += directions[row * direction_stride + column] *
             loadings[row * loading_stride + column];
  }
  if (!isfinite(norm2) || norm2 <= 1.0e-40f) {
    atomic_store_explicit(invalid, 1, memory_order_relaxed);
    return;
  }
  const float reciprocal = rsqrt(norm2);
  for (uint row = 0; row < rows; ++row) {
    directions[row * direction_stride + column] *= reciprocal;
    loadings[row * loading_stride + column] *= reciprocal;
  }
}

kernel void reorthogonalize_score_pair(
    device float* score [[buffer(0)]],
    device float* direction [[buffer(1)]],
    device const float* score_basis [[buffer(2)]],
    device const float* direction_basis [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& direction_size [[buffer(5)]],
    constant uint& used [[buffer(6)]],
    constant uint& score_stride [[buffer(7)]],
    constant uint& direction_stride [[buffer(8)]],
    constant uint& score_basis_stride [[buffer(9)]],
    constant uint& direction_basis_stride [[buffer(10)]],
    uint index [[thread_position_in_grid]]) {
  if (index != 0) return;
  for (uint pass = 0; pass < 2; ++pass) {
    for (uint column = 0; column < used; ++column) {
      float dot = 0.0f;
      for (uint row = 0; row < rows; ++row) {
        dot += score_basis[row * score_basis_stride + column] *
               score[row * score_stride];
      }
      for (uint row = 0; row < rows; ++row) {
        score[row * score_stride] -=
            dot * score_basis[row * score_basis_stride + column];
      }
      for (uint row = 0; row < direction_size; ++row) {
        direction[row * direction_stride] -=
            dot * direction_basis[row * direction_basis_stride + column];
      }
    }
  }
}

kernel void copy_vector(device const float* source [[buffer(0)]],
                        device float* destination [[buffer(1)]],
                        constant uint& size [[buffer(2)]],
                        constant uint& source_stride [[buffer(3)]],
                        constant uint& destination_stride [[buffer(4)]],
                        constant uint& column [[buffer(5)]],
                        uint row [[thread_position_in_grid]]) {
  if (row >= size) return;
  destination[row * destination_stride + column] = source[row * source_stride];
}

kernel void copy_plain(device const float* source [[buffer(0)]],
                       device float* destination [[buffer(1)]],
                       constant uint& size [[buffer(2)]],
                       constant uint& source_stride [[buffer(3)]],
                       constant uint& destination_stride [[buffer(4)]],
                       uint row [[thread_position_in_grid]]) {
  if (row >= size) return;
  destination[row * destination_stride] = source[row * source_stride];
}

kernel void label_response_loading(device const float* score [[buffer(0)]],
                                   device const int* labels [[buffer(1)]],
                                   device const float* priors [[buffer(2)]],
                                   device float* result [[buffer(3)]],
                                   constant uint& rows [[buffer(4)]],
                                   constant uint& classes [[buffer(5)]],
                                   constant uint& score_stride [[buffer(6)]],
                                   constant uint& result_stride [[buffer(7)]],
                                   uint cls [[thread_position_in_grid]]) {
  if (cls >= classes) return;
  float total = 0.0f;
  float selected = 0.0f;
  for (uint row = 0; row < rows; ++row) {
    float value = score[row * score_stride];
    total += value;
    if (labels[row] == int(cls + 1)) selected += value;
  }
  result[cls * result_stride] = selected - priors[cls] * total;
}

kernel void rank1_subtract(device float* matrix [[buffer(0)]],
                           device const float* column [[buffer(1)]],
                           device const float* row_values [[buffer(2)]],
                           constant uint& rows [[buffer(3)]],
                           constant uint& cols [[buffer(4)]],
                           constant uint& matrix_stride [[buffer(5)]],
                           constant uint& column_stride [[buffer(6)]],
                           uint2 index [[thread_position_in_grid]]) {
  if (index.y >= rows || index.x >= cols) return;
  matrix[index.y * matrix_stride + index.x] -=
      column[index.y * column_stride] * row_values[index.x];
}

kernel void zero_score_suffix(device float* scores [[buffer(0)]],
                              constant uint& rows [[buffer(1)]],
                              constant uint& cols [[buffer(2)]],
                              constant uint& stride [[buffer(3)]],
                              constant uint& prefix [[buffer(4)]],
                              uint2 index [[thread_position_in_grid]]) {
  if (index.y >= rows || index.x >= cols || index.x < prefix) return;
  scores[index.y * stride + index.x] = 0.0f;
}

kernel void add_response_mean(device float* prediction [[buffer(0)]],
                              device const float* mean [[buffer(1)]],
                              constant uint& rows [[buffer(2)]],
                              constant uint& cols [[buffer(3)]],
                              constant uint& stride [[buffer(4)]],
                              uint2 index [[thread_position_in_grid]]) {
  if (index.y >= rows || index.x >= cols) return;
  prediction[index.y * stride + index.x] += mean[index.x];
}

kernel void top_classes(device const float* values [[buffer(0)]],
                        device int* output [[buffer(1)]],
                        constant uint& rows [[buffer(2)]],
                        constant uint& classes [[buffer(3)]],
                        constant uint& stride [[buffer(4)]],
                        constant uint& top [[buffer(5)]],
                        uint row [[thread_position_in_grid]]) {
  if (row >= rows) return;
  for (uint rank = 0; rank < top; ++rank) {
    int best = -1;
    float best_value = 0.0f;
    for (uint cls = 0; cls < classes; ++cls) {
      bool used = false;
      for (uint previous = 0; previous < rank; ++previous) {
        if (output[previous * rows + row] == int(cls + 1)) used = true;
      }
      float value = values[row * stride + cls];
      if (!used && (best < 0 || value > best_value)) {
        best = int(cls);
        best_value = value;
      }
    }
    output[rank * rows + row] = best + 1;
  }
}

kernel void lda_class_means(device const float* scores [[buffer(0)]],
                            device const int* labels [[buffer(1)]],
                            device float* means [[buffer(2)]],
                            device float* weighted [[buffer(3)]],
                            constant uint& rows [[buffer(4)]],
                            constant uint& components [[buffer(5)]],
                            constant uint& classes [[buffer(6)]],
                            constant uint& score_stride [[buffer(7)]],
                            constant uint& mean_stride [[buffer(8)]],
                            uint2 index [[thread_position_in_grid]]) {
  if (index.x >= components || index.y >= classes) return;
  float total = 0.0f;
  uint count = 0;
  for (uint row = 0; row < rows; ++row) {
    if (labels[row] == int(index.y + 1)) {
      total += scores[row * score_stride + index.x];
      ++count;
    }
  }
  float value = count > 0 ? total / float(count) : 0.0f;
  means[index.y * mean_stride + index.x] = value;
  weighted[index.y * mean_stride + index.x] = value * sqrt(float(count));
}

kernel void lda_ridge_scale(device const float* gram [[buffer(0)]],
                            device const float* mean_cross [[buffer(1)]],
                            device float* ridge [[buffer(2)]],
                            constant uint& full_components [[buffer(3)]],
                            constant uint& prefix [[buffer(4)]],
                            constant uint& gram_stride [[buffer(5)]],
                            constant uint& mean_cross_stride [[buffer(6)]],
                            constant uint& denominator [[buffer(7)]],
                            constant float& rho [[buffer(8)]],
                            uint index [[thread_position_in_grid]]) {
  if (index != 0) return;
  float trace = 0.0f;
  for (uint component = 0; component < prefix; ++component) {
    trace += (gram[component * gram_stride + component] -
              mean_cross[component * mean_cross_stride + component]) /
             float(denominator);
  }
  float scale = trace / float(prefix);
  if (!isfinite(scale) || scale <= 0.0f) scale = 1.0f;
  ridge[0] = rho * scale;
}

kernel void lda_prepare_factor(device const float* gram [[buffer(0)]],
                               device const float* mean_cross [[buffer(1)]],
                               device const float* ridge [[buffer(2)]],
                               device float* factor [[buffer(3)]],
                               constant uint& prefix [[buffer(4)]],
                               constant uint& gram_stride [[buffer(5)]],
                               constant uint& mean_cross_stride [[buffer(6)]],
                               constant uint& factor_stride [[buffer(7)]],
                               constant uint& denominator [[buffer(8)]],
                               uint2 index [[thread_position_in_grid]]) {
  if (index.x >= prefix || index.y >= prefix) return;
  float value = (gram[index.y * gram_stride + index.x] -
                 mean_cross[index.y * mean_cross_stride + index.x]) /
                float(denominator);
  if (index.x == index.y) value += ridge[0];
  factor[index.y * factor_stride + index.x] = value;
}

kernel void lda_transpose_means(device const float* means [[buffer(0)]],
                                device float* right_hand_side [[buffer(1)]],
                                constant uint& prefix [[buffer(2)]],
                                constant uint& classes [[buffer(3)]],
                                constant uint& mean_stride [[buffer(4)]],
                                constant uint& rhs_stride [[buffer(5)]],
                                uint2 index [[thread_position_in_grid]]) {
  if (index.x >= classes || index.y >= prefix) return;
  right_hand_side[index.y * rhs_stride + index.x] =
      means[index.x * mean_stride + index.y];
}

kernel void copy_prefix_matrix(device const float* source [[buffer(0)]],
                               device float* destination [[buffer(1)]],
                               constant uint& rows [[buffer(2)]],
                               constant uint& cols [[buffer(3)]],
                               constant uint& source_stride [[buffer(4)]],
                               constant uint& destination_stride [[buffer(5)]],
                               uint2 index [[thread_position_in_grid]]) {
  if (index.x >= cols || index.y >= rows) return;
  destination[index.y * destination_stride + index.x] =
      source[index.y * source_stride + index.x];
}

kernel void lda_constants(device const float* means [[buffer(0)]],
                          device const float* linear [[buffer(1)]],
                          device const float* priors [[buffer(2)]],
                          device float* constants [[buffer(3)]],
                          constant uint& prefix [[buffer(4)]],
                          constant uint& classes [[buffer(5)]],
                          constant uint& mean_stride [[buffer(6)]],
                          constant uint& linear_stride [[buffer(7)]],
                          uint cls [[thread_position_in_grid]]) {
  if (cls >= classes) return;
  float value = 0.0f;
  for (uint component = 0; component < prefix; ++component) {
    value += means[cls * mean_stride + component] *
             linear[component * linear_stride + cls];
  }
  constants[cls] = -0.5f * value + log(priors[cls]);
}

kernel void lda_add_constants(device float* scores [[buffer(0)]],
                              device const float* constants [[buffer(1)]],
                              constant uint& rows [[buffer(2)]],
                              constant uint& classes [[buffer(3)]],
                              constant uint& stride [[buffer(4)]],
                              uint2 index [[thread_position_in_grid]]) {
  if (index.x >= classes || index.y >= rows) return;
  scores[index.y * stride + index.x] += constants[index.x];
}

kernel void response_sums(device const float* prediction [[buffer(0)]],
                          device const float* observed [[buffer(1)]],
                          device const int* labels [[buffer(2)]],
                          device const float* training_mean [[buffer(3)]],
                          device float* output [[buffer(4)]],
                          device atomic_int* invalid [[buffer(5)]],
                          constant uint& rows [[buffer(6)]],
                          constant uint& responses [[buffer(7)]],
                          constant uint& prediction_stride [[buffer(8)]],
                          constant uint& observed_stride [[buffer(9)]],
                          constant uint& output_stride [[buffer(10)]],
                          constant int& classification [[buffer(11)]],
                          uint response [[thread_position_in_grid]]) {
  if (response >= responses) return;
  float mean = 0.0f;
  float sse = 0.0f;
  for (uint row = 0; row < rows; ++row) {
    float truth = classification != 0 ?
        (labels[row] == int(response + 1) ? 1.0f : 0.0f) :
        observed[row * observed_stride + response];
    float estimate = prediction[row * prediction_stride + response];
    if (!isfinite(truth) || !isfinite(estimate)) {
      atomic_store_explicit(invalid, 1, memory_order_relaxed);
    }
    mean += truth;
    float residual = truth - estimate;
    sse += residual * residual;
  }
  mean /= float(rows);
  float training_sst = 0.0f;
  float observed_sst = 0.0f;
  for (uint row = 0; row < rows; ++row) {
    float truth = classification != 0 ?
        (labels[row] == int(response + 1) ? 1.0f : 0.0f) :
        observed[row * observed_stride + response];
    float first = truth - training_mean[response];
    float second = truth - mean;
    training_sst += first * first;
    observed_sst += second * second;
  }
  output[0 * output_stride + response] = sse;
  output[1 * output_stride + response] = training_sst;
  output[2 * output_stride + response] = observed_sst;
}

kernel void matrix_sum_squares(device const float* matrix [[buffer(0)]],
                               device float* output [[buffer(1)]],
                               constant uint& rows [[buffer(2)]],
                               constant uint& cols [[buffer(3)]],
                               constant uint& stride [[buffer(4)]],
                               constant uint& output_index [[buffer(5)]],
                               uint index [[thread_position_in_grid]]) {
  if (index != 0) return;
  float total = 0.0f;
  for (uint row = 0; row < rows; ++row) {
    for (uint column = 0; column < cols; ++column) {
      float value = matrix[row * stride + column];
      total += value * value;
    }
  }
  output[output_index] = total;
}

kernel void matrix_column_sums_squares(
    device const float* matrix [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant uint& stride [[buffer(4)]],
    uint column [[thread_position_in_grid]]) {
  if (column >= cols) return;
  float total = 0.0f;
  for (uint row = 0; row < rows; ++row) {
    float value = matrix[row * stride + column];
    total += value * value;
  }
  output[column] = total;
}
)METAL"];
    NSError* error = nil;
#ifdef FASTPLS_HAS_PRECOMPILED_RESIDENT_METAL
    dispatch_data_t compiled = dispatch_data_create(
        fastpls_metal_library, fastpls_metal_library_len,
        dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    library_ = [device newLibraryWithData:compiled error:&error];
#endif
    if (library_ == nil) {
      error = nil;
      library_ = [device newLibraryWithSource:source options:nil error:&error];
    }
    if (library_ == nil || error != nil) {
      NSString* message = error == nil ? @"unknown Metal compiler error" :
                                         [error localizedDescription];
      throw std::runtime_error(
          "resident Metal kernel compilation failed: " +
          std::string([message UTF8String]));
    }
  }

  id<MTLComputePipelineState> pipeline(const char* name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = pipelines_.find(name);
    if (found != pipelines_.end()) return found->second;
    NSString* function_name = [NSString stringWithUTF8String:name];
    id<MTLFunction> function = [library_ newFunctionWithName:function_name];
    if (function == nil) {
      throw std::runtime_error(std::string("resident Metal function not found: ") + name);
    }
    NSError* error = nil;
    id<MTLComputePipelineState> result =
        [device_ newComputePipelineStateWithFunction:function error:&error];
    if (result == nil || error != nil) {
      NSString* message = error == nil ? @"unknown pipeline error" :
                                         [error localizedDescription];
      throw std::runtime_error(std::string("resident Metal pipeline failed: ") +
                               [message UTF8String]);
    }
    pipelines_[name] = result;
    return result;
  }

 private:
  id<MTLDevice> device_;
  id<MTLLibrary> library_;
  std::mutex mutex_;
  std::map<std::string, id<MTLComputePipelineState>> pipelines_;
};

std::shared_ptr<ResidentKernels> resident_kernels(id<MTLDevice> device) {
  static std::mutex mutex;
  static std::map<uint64_t, std::shared_ptr<ResidentKernels>> cache;
  const uint64_t registry_id = [device registryID];
  std::lock_guard<std::mutex> lock(mutex);
  auto found = cache.find(registry_id);
  if (found != cache.end()) return found->second;
  auto compiled = std::make_shared<ResidentKernels>(device);
  cache[registry_id] = compiled;
  return compiled;
}

void encode_1d(id<MTLCommandBuffer> command,
               id<MTLComputePipelineState> pipeline,
               NSUInteger size,
               const std::function<void(id<MTLComputeCommandEncoder>)>& bind) {
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  bind(encoder);
  const NSUInteger width = std::min<NSUInteger>(
      std::max<NSUInteger>(1, pipeline.threadExecutionWidth), 256);
  [encoder dispatchThreads:MTLSizeMake(size, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
  [encoder endEncoding];
}

void dispatch_1d(id<MTLCommandQueue> queue,
                 id<MTLComputePipelineState> pipeline,
                 NSUInteger size,
                 const std::function<void(id<MTLComputeCommandEncoder>)>& bind) {
  id<MTLCommandBuffer> command = [queue commandBuffer];
  encode_1d(command, pipeline, size, bind);
  finish_command(command, "resident Metal kernel failed");
}

void encode_2d(id<MTLCommandBuffer> command,
               id<MTLComputePipelineState> pipeline,
               NSUInteger columns,
               NSUInteger rows,
               const std::function<void(id<MTLComputeCommandEncoder>)>& bind) {
  id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
  [encoder setComputePipelineState:pipeline];
  bind(encoder);
  [encoder dispatchThreads:MTLSizeMake(columns, rows, 1)
      threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
  [encoder endEncoding];
}

void dispatch_2d(id<MTLCommandQueue> queue,
                 id<MTLComputePipelineState> pipeline,
                 NSUInteger columns,
                 NSUInteger rows,
                 const std::function<void(id<MTLComputeCommandEncoder>)>& bind) {
  id<MTLCommandBuffer> command = [queue commandBuffer];
  encode_2d(command, pipeline, columns, rows, bind);
  finish_command(command, "resident Metal kernel failed");
}

uint32_t stride(const MetalMatrix& matrix) {
  return static_cast<uint32_t>(matrix.row_bytes / sizeof(float));
}

void set_u32(id<MTLComputeCommandEncoder> encoder, NSUInteger index,
             uint32_t value) {
  [encoder setBytes:&value length:sizeof(value) atIndex:index];
}

void set_i32(id<MTLComputeCommandEncoder> encoder, NSUInteger index,
             int32_t value) {
  [encoder setBytes:&value length:sizeof(value) atIndex:index];
}

class ResidentMetalSimpls final : public MetalResidentModel {
 public:
  ResidentMetalSimpls(const arma::fmat& host_x,
                      const arma::fmat* host_y,
                      const Rcpp::IntegerVector* host_labels,
                      int classes,
                      int requested_components,
                      int scaling,
                      int oversample,
                      int power,
                      unsigned int seed,
                      int method,
                      int north,
                      int kernel,
                      float gamma,
                      int degree,
                      float coefficient)
      : n_(static_cast<int>(host_x.n_rows)),
        p_(method == 5 ? static_cast<int>(host_x.n_rows) :
                         static_cast<int>(host_x.n_cols)),
        input_p_(static_cast<int>(host_x.n_cols)),
        q_(host_labels == nullptr ? static_cast<int>(host_y->n_cols) : classes),
        components_(std::min(requested_components,
                             std::min(p_, std::max(1, n_ - 1)))),
        classification_(host_labels != nullptr),
        plssvd_(method == 1),
        opls_(method == 4),
        kernelpls_(method == 5),
        north_(std::max(0, north)),
        kernel_kind_(kernel),
        kernel_degree_(degree),
        kernel_gamma_(gamma),
        kernel_coefficient_(coefficient),
        implicit_crosscov_(
            method != 4 && method != 5 && host_labels == nullptr &&
            static_cast<long long>(n_) * (p_ + q_) <
                static_cast<long long>(p_) * q_),
        right_gram_refresh_(
            method != 1 && !implicit_crosscov_ && q_ <= p_ && q_ <= 512),
        refresh_block_(1),
        solver_width_(std::min(
            {p_, q_, std::max(1, oversample + 1)})),
        effective_power_(std::max(1, power)),
        seed_(seed) {
    if (n_ < 2 || p_ < 1 || q_ < 1 || components_ < 1) {
      throw std::invalid_argument("invalid resident Metal SIMPLS dimensions");
    }
    if ((host_y == nullptr) == (host_labels == nullptr)) {
      throw std::invalid_argument("provide responses or labels, not both");
    }
    if (scaling < 1 || scaling > 3) {
      throw std::invalid_argument("invalid resident Metal scaling mode");
    }
    if (method != 1 && method != 3 && method != 4 && method != 5) {
      throw std::invalid_argument("invalid resident Metal PLS method");
    }
    if (kernelpls_ && (kernel_kind_ < 2 || kernel_kind_ > 3 ||
                       kernel_degree_ < 1 || !(kernel_gamma_ > 0.0f))) {
      throw std::invalid_argument(
          "invalid resident Metal nonlinear-kernel controls");
    }
    if (plssvd_) components_ = std::min(components_, q_);
    if (plssvd_ && classification_ && components_ >= q_) {
      components_ = q_ - 1;
    }
    if (components_ < 1) {
      throw std::invalid_argument("PLS-SVD has no eligible component");
    }
    @autoreleasepool {
      device_ = MTLCreateSystemDefaultDevice();
      if (device_ == nil) {
        throw std::runtime_error(
            "resident Metal fitting is unavailable; no CPU fallback is performed");
      }
      queue_ = [device_ newCommandQueue];
      if (queue_ == nil) {
        throw std::runtime_error("resident Metal command queue creation failed");
      }
      kernels_ = resident_kernels(device_);
      if (kernelpls_) {
        const double gram_bytes = static_cast<double>(n_) * n_ * sizeof(float);
        const double reference_bytes =
            static_cast<double>(n_) * input_p_ * sizeof(float);
        const double recommended =
            static_cast<double>([device_ recommendedMaxWorkingSetSize]);
        if (recommended > 0.0 && gram_bytes + reference_bytes >
                                     0.60 * recommended) {
          throw std::runtime_error(
              "nonlinear kernel PLS requires an n-by-n Gram matrix that "
              "exceeds the guarded Metal memory budget");
        }
      }
      invalid_ = [device_ newBufferWithLength:sizeof(int32_t)
                                      options:MTLResourceStorageModeShared];
      X_ = make_matrix(device_, n_, p_);
      if (!implicit_crosscov_) S_ = make_matrix(device_, p_, q_);
      R_ = make_matrix(device_, p_, components_);
      Q_ = make_matrix(device_, q_, components_);
      V_ = make_matrix(device_, p_, components_);
      P_ = make_matrix(device_, p_, components_);
      T_ = make_matrix(device_, n_, components_);
      predictor_ss_ = make_matrix(device_, 1, components_ + 1);
      mean_x_ = make_matrix(device_, 1, input_p_);
      scale_x_ = make_matrix(device_, 1, input_p_);
      mean_y_ = make_matrix(device_, 1, q_);
      scale_y_ = make_matrix(device_, 1, q_);
      r_ = make_matrix(device_, p_, 1);
      z_ = make_matrix(device_, q_, 1);
      t_ = make_matrix(device_, n_, 1);
      p_vector_ = make_matrix(device_, p_, 1);
      q_vector_ = make_matrix(device_, q_, 1);
      v_ = make_matrix(device_, p_, 1);
      row_ = make_matrix(device_, 1, q_);
      const double work = static_cast<double>(n_) * p_ * q_;
      const double crosscov_bytes =
          static_cast<double>(p_) * q_ * sizeof(float);
      if (!plssvd_ && implicit_crosscov_ &&
          crosscov_bytes > 512.0 * 1024.0 * 1024.0) {
        // Match the massive CPU/CUDA route: refresh one fresh direction
        // instead of propagating an oversampled response-wide block.
        solver_width_ = 1;
      }
      if (components_ >= 4 && work >= 5.0e8) {
        const int maximum_block = classification_ ? 64 :
            (plssvd_ ? (implicit_crosscov_ ? 256 : 8) : 1);
        refresh_block_ = std::min({maximum_block, components_, p_, q_});
      }
      // Large candidate blocks already amortize S products and avoid the
      // small-matrix eigensolver required by the response-Gram formulation.
      if (work >= 5.0e8 && refresh_block_ > 1) right_gram_refresh_ = false;
      const int workspace_width = std::max(refresh_block_, solver_width_);
      candidate_basis_ = make_matrix(device_, p_, workspace_width);
      candidate_rotated_ = make_matrix(
          device_, std::max(p_, q_), workspace_width);
      right_block_ = make_matrix(device_, q_, workspace_width);
      sample_block_ = make_matrix(device_, n_, workspace_width);
      use_sample_response_gram_ =
          !classification_ && !plssvd_ && implicit_crosscov_ &&
          q_ >= 4 * n_ && workspace_width == 1;
      if (use_sample_response_gram_) {
        response_gram_ = make_matrix(device_, n_, n_);
        sample_gram_vector_ = make_matrix(device_, n_, 1);
      }
      projected_gram_ = make_matrix(
          device_, workspace_width, workspace_width);
      projected_vectors_ = make_matrix(
          device_, workspace_width, workspace_width);
      block_projection_ = make_matrix(
          device_, components_, workspace_width);
      reduced_block_ = make_matrix(device_, workspace_width, q_);
      reduced_vector_ = make_matrix(device_, workspace_width, 1);
      if (right_gram_refresh_) {
        right_gram_ = make_matrix(device_, q_, q_);
        right_gram_delta_ = make_matrix(device_, q_, q_);
      }
      block_status_ = [device_ newBufferWithLength:sizeof(int32_t)
                                            options:MTLResourceStorageModeShared];
      score_norm_ = make_matrix(device_, 1, 1);
      vector_norm_ = make_matrix(device_, 1, 1);
      if (opls_ && north_ > 0) {
        opls_weights_ = make_matrix(device_, input_p_, north_);
        opls_loadings_ = make_matrix(device_, input_p_, north_);
        opls_score_ = make_matrix(device_, n_, 1);
        opls_loading_ = make_matrix(device_, input_p_, 1);
        opls_weight_ = make_matrix(device_, input_p_, 1);
        opls_ratio_ = make_matrix(device_, 1, 1);
        opls_denominator_ = make_matrix(device_, 1, 1);
        opls_effective_r_ = make_matrix(device_, input_p_, components_);
        opls_component_row_ = make_matrix(device_, 1, components_);
      }
      if (kernelpls_) {
        reference_ = make_matrix(device_, n_, input_p_);
        reference_norms_ = make_matrix(device_, 1, n_);
        kernel_training_means_ = make_matrix(device_, 1, n_);
        kernel_grand_ = make_matrix(device_, 1, 1);
      }
      if (plssvd_) {
        plssvd_gram_ = make_matrix(device_, components_, components_);
        plssvd_cross_ = make_matrix(device_, components_, q_);
        plssvd_factor_ = make_matrix(device_, components_, components_);
        plssvd_rhs_ = make_matrix(device_, components_, q_);
        plssvd_weights_ = make_matrix(device_, components_, q_);
        plssvd_ridge_ = make_matrix(device_, 1, 1);
        plssvd_zero_ = make_matrix(device_, components_, components_);
        std::memset([plssvd_zero_.buffer contents], 0,
                    plssvd_zero_.row_bytes * plssvd_zero_.rows);
        plssvd_status_ = [device_ newBufferWithLength:sizeof(int32_t)
                                              options:MTLResourceStorageModeShared];
      }
      prepare_training_predictors(host_x, scaling);
      if (classification_) {
        labels_ = [device_ newBufferWithLength:sizeof(int32_t) * n_
                                       options:MTLResourceStorageModeShared];
        std::memcpy([labels_ contents], INTEGER(*host_labels), sizeof(int32_t) * n_);
        if (static_cast<double>(n_) * q_ * sizeof(float) <=
            256.0 * 1024.0 * 1024.0) {
          Y_ = make_matrix(device_, n_, q_);
        }
        lda_status_ = [device_ newBufferWithLength:sizeof(int32_t)
                                           options:MTLResourceStorageModeShared];
        lda_means_ = make_matrix(device_, q_, components_);
        lda_weighted_ = make_matrix(device_, q_, components_);
        lda_gram_ = make_matrix(device_, components_, components_);
        lda_mean_cross_ = make_matrix(device_, components_, components_);
        lda_factor_ = make_matrix(device_, components_, components_);
        lda_rhs_ = make_matrix(device_, components_, q_);
        lda_linear_ = make_matrix(device_, components_, q_);
        lda_constants_ = make_matrix(device_, 1, q_);
        lda_ridge_ = make_matrix(device_, 1, 1);
      } else {
        Y_ = make_matrix(device_, n_, q_);
        upload_matrix(*host_y, Y_);
      }
      if (classification_) {
        prepare_class_response();
        if (opls_) {
          fit_opls_filters();
          prepare_class_response();
        }
        cached_predictor_crossprod_ = !plssvd_ && components_ >= 8 &&
            n_ >= 8 * p_;
        if (cached_predictor_crossprod_) {
          predictor_gram_ = make_matrix(device_, p_, p_);
          original_crosscov_ = make_matrix(device_, p_, q_);
          deflation_block_ = make_matrix(device_, p_, q_);
          MetalProduct predictor_gram(
              device_, X_, X_, predictor_gram_, true, false);
          run_product(queue_, predictor_gram);
          dispatch_2d(queue_, kernels_->pipeline("copy_prefix_matrix"), q_, p_,
              [&](id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:S_.buffer offset:0 atIndex:0];
                [encoder setBuffer:original_crosscov_.buffer offset:0 atIndex:1];
                set_u32(encoder, 2, static_cast<uint32_t>(p_));
                set_u32(encoder, 3, static_cast<uint32_t>(q_));
                set_u32(encoder, 4, stride(S_));
                set_u32(encoder, 5, stride(original_crosscov_));
              });
        }
      } else {
        preprocess(Y_, mean_y_, scale_y_, 1);
        if (!implicit_crosscov_) {
          MetalProduct cross(device_, X_, Y_, S_, true, false);
          run_product(queue_, cross);
        }
        if (opls_) {
          fit_opls_filters();
          MetalProduct cross(device_, X_, Y_, S_, true, false);
          run_product(queue_, cross);
        }
      }
      if (right_gram_refresh_) {
        MetalProduct right_gram(
            device_, S_, S_, right_gram_, true, false);
        run_product(queue_, right_gram);
      }
      if (use_sample_response_gram_) {
        MetalProduct response_gram_product(
            device_, Y_, Y_, response_gram_, false, true);
        run_product(queue_, response_gram_product);
        rank1_x_forward_.reset(new MetalProduct(
            device_, X_, candidate_basis_, sample_block_, false, false));
        rank1_response_gram_.reset(new MetalProduct(
            device_, response_gram_, sample_block_, sample_gram_vector_,
            false, false));
        rank1_x_transpose_.reset(new MetalProduct(
            device_, X_, sample_gram_vector_, candidate_basis_, true, false));
      } else if (!plssvd_ && implicit_crosscov_ && workspace_width == 1) {
        rank1_x_forward_.reset(new MetalProduct(
            device_, X_, candidate_basis_, sample_block_, false, false));
        rank1_y_transpose_.reset(new MetalProduct(
            device_, Y_, sample_block_, right_block_, true, false));
        rank1_y_forward_.reset(new MetalProduct(
            device_, Y_, right_block_, sample_block_, false, false));
        rank1_x_transpose_.reset(new MetalProduct(
            device_, X_, sample_block_, candidate_basis_, true, false));
      }
      fit();
      if (opls_) prepare_opls_projection();
      if (plssvd_) prepare_plssvd_moments();
    }
  }

  arma::fmat export_field(int field) override {
    std::lock_guard<std::mutex> lock(mutex_);
    switch (field) {
      case 0: return copy_from_matrix(opls_ ? opls_effective_r_ : R_);
      case 1: return copy_from_matrix(Q_);
      case 2: return copy_from_matrix(T_);
      case 3: return copy_from_matrix(mean_x_);
      case 4: return copy_from_matrix(scale_x_);
      case 5: return copy_from_matrix(mean_y_);
      case 6: return copy_from_matrix(P_);
      case 7:
        prepare_variance();
        return copy_from_matrix(predictor_ss_);
      default: throw std::invalid_argument("invalid resident Metal export field");
    }
  }

  void compact(bool prepare_lda) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (prepare_lda) {
      if (!classification_) {
        throw std::invalid_argument("LDA compaction requires classification labels");
      }
      prepare_lda_moments();
    }
    X_ = MetalMatrix{};
    Y_ = MetalMatrix{};
    S_ = MetalMatrix{};
    predictor_gram_ = MetalMatrix{};
    original_crosscov_ = MetalMatrix{};
    deflation_block_ = MetalMatrix{};
    V_ = MetalMatrix{};
    P_ = MetalMatrix{};
    T_ = MetalMatrix{};
    predictor_ss_ = MetalMatrix{};
    r_ = MetalMatrix{};
    z_ = MetalMatrix{};
    t_ = MetalMatrix{};
    p_vector_ = MetalMatrix{};
    q_vector_ = MetalMatrix{};
    v_ = MetalMatrix{};
    row_ = MetalMatrix{};
    candidate_basis_ = MetalMatrix{};
    candidate_rotated_ = MetalMatrix{};
    right_block_ = MetalMatrix{};
    sample_block_ = MetalMatrix{};
    projected_gram_ = MetalMatrix{};
    projected_vectors_ = MetalMatrix{};
    reduced_block_ = MetalMatrix{};
    reduced_vector_ = MetalMatrix{};
    block_projection_ = MetalMatrix{};
    score_norm_ = MetalMatrix{};
    vector_norm_ = MetalMatrix{};
    labels_ = nil;
    block_status_ = nil;
    if (!prepare_lda) {
      lda_means_ = MetalMatrix{};
      lda_weighted_ = MetalMatrix{};
      lda_gram_ = MetalMatrix{};
      lda_mean_cross_ = MetalMatrix{};
      lda_factor_ = MetalMatrix{};
      lda_rhs_ = MetalMatrix{};
      lda_linear_ = MetalMatrix{};
      lda_constants_ = MetalMatrix{};
      lda_ridge_ = MetalMatrix{};
      lda_status_ = nil;
    }
  }

  arma::fmat predict(const arma::fmat& x, int prefix,
                     bool use_lda) override {
    std::lock_guard<std::mutex> lock(mutex_);
    PredictionWorkspace& workspace = prediction_scores(x);
    if (use_lda) {
      lda_predict(workspace, prefix);
      return copy_from_matrix(workspace.lda_prediction);
    }
    response_predict(workspace, prefix);
    return copy_from_matrix(workspace.prediction);
  }

  arma::fcube predict_path(
      const arma::fmat& x, const Rcpp::IntegerVector& prefixes,
      bool use_lda) override {
    if (prefixes.size() < 1) {
      throw std::invalid_argument("empty resident Metal prediction path");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    PredictionWorkspace& workspace = prediction_scores(x);
    arma::fcube result(x.n_rows, q_, prefixes.size());
    int previous = 0;
    for (R_xlen_t index = 0; index < prefixes.size(); ++index) {
      const int prefix = prefixes[index];
      if (prefix <= previous) {
        throw std::invalid_argument("component path must be strictly increasing");
      }
      validate_prefix(prefix);
      if (use_lda) {
        lda_predict(workspace, prefix);
        result.slice(index) = copy_from_matrix(workspace.lda_prediction);
      } else {
        response_predict(workspace, prefix);
        result.slice(index) = copy_from_matrix(workspace.prediction);
      }
      previous = prefix;
    }
    return result;
  }

  arma::fmat project(const arma::fmat& x, int prefix) override {
    std::lock_guard<std::mutex> lock(mutex_);
    validate_prefix(prefix);
    PredictionWorkspace& workspace = prediction_scores(x);
    return copy_from_matrix(workspace.scores, prefix);
  }

  arma::imat classify(const arma::fmat& x, int prefix, int top,
                      bool use_lda) override {
    if (!classification_) {
      throw std::invalid_argument("resident Metal classification requires labels");
    }
    if (top < 1 || top > q_) {
      throw std::invalid_argument("invalid resident Metal top-k request");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    validate_prefix(prefix);
    PredictionWorkspace& workspace = prediction_scores(x);
    MetalMatrix& values = use_lda ? workspace.lda_prediction :
                                    workspace.prediction;
    if (use_lda) {
      lda_predict(workspace, prefix);
    } else {
      response_predict(workspace, prefix);
    }
    arma::imat result(x.n_rows, top);
    classify_values(values, x.n_rows, top, result.memptr());
    return result;
  }

  arma::icube classify_path(
      const arma::fmat& x, const Rcpp::IntegerVector& prefixes, int top,
      bool use_lda) override {
    if (!classification_) {
      throw std::invalid_argument("resident Metal classification requires labels");
    }
    if (top < 1 || top > q_ || prefixes.size() < 1) {
      throw std::invalid_argument("invalid resident Metal component-path request");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    PredictionWorkspace& workspace = prediction_scores(x);
    arma::icube result(x.n_rows, top, prefixes.size());
    for (R_xlen_t index = 0; index < prefixes.size(); ++index) {
      const int prefix = prefixes[index];
      validate_prefix(prefix);
      MetalMatrix& values = use_lda ? workspace.lda_prediction :
                                      workspace.prediction;
      if (use_lda) {
        lda_predict(workspace, prefix);
      } else {
        response_predict(workspace, prefix);
      }
      classify_values(values, x.n_rows, top, result.slice(index).memptr());
    }
    return result;
  }

  void classify_response_path(
      const arma::fmat& x, const Rcpp::IntegerVector& prefixes, int top,
      bool use_lda, arma::icube& labels,
      arma::fcube& predictions) override {
    if (!classification_) {
      throw std::invalid_argument("resident Metal classification requires labels");
    }
    if (top < 1 || top > q_ || prefixes.size() < 1) {
      throw std::invalid_argument(
          "invalid resident Metal classification-response path request");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    PredictionWorkspace& workspace = prediction_scores(x);
    labels.set_size(x.n_rows, top, prefixes.size());
    predictions.set_size(x.n_rows, q_, prefixes.size());
    int previous = 0;
    for (R_xlen_t index = 0; index < prefixes.size(); ++index) {
      const int prefix = prefixes[index];
      if (prefix <= previous) {
        throw std::invalid_argument(
            "component path must be strictly increasing");
      }
      validate_prefix(prefix);
      response_predict(workspace, prefix);
      predictions.slice(index) = copy_from_matrix(workspace.prediction);
      MetalMatrix& values = use_lda ? workspace.lda_prediction :
                                      workspace.prediction;
      if (use_lda) {
        lda_predict(workspace, prefix);
      }
      classify_values(values, x.n_rows, top, labels.slice(index).memptr());
      previous = prefix;
    }
  }

  void classify_values(MetalMatrix& values, size_t rows, int top,
                       arma::sword* output) {
    const size_t required = rows * top;
    if (required > top_capacity_) {
      top_indices_ = [device_ newBufferWithLength:required * sizeof(int32_t)
                                           options:MTLResourceStorageModeShared];
      if (top_indices_ == nil) {
        throw std::runtime_error("resident Metal top-k allocation failed");
      }
      top_capacity_ = required;
    }
    dispatch_1d(queue_, kernels_->pipeline("top_classes"), rows,
        [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:values.buffer offset:0 atIndex:0];
          [encoder setBuffer:top_indices_ offset:0 atIndex:1];
          set_u32(encoder, 2, static_cast<uint32_t>(rows));
          set_u32(encoder, 3, static_cast<uint32_t>(q_));
          set_u32(encoder, 4, stride(values));
          set_u32(encoder, 5, static_cast<uint32_t>(top));
        });
    const int32_t* encoded = static_cast<const int32_t*>([top_indices_ contents]);
    for (size_t index = 0; index < required; ++index) {
      output[index] = static_cast<arma::sword>(encoded[index]);
    }
  }

  arma::fmat response_sums(const arma::fmat& x,
                           const arma::fmat* y,
                           const Rcpp::IntegerVector* labels,
                           int prefix) override {
    if ((y == nullptr) == (labels == nullptr) ||
        classification_ != (labels != nullptr)) {
      throw std::invalid_argument("invalid resident Metal response representation");
    }
    if (y != nullptr &&
        (y->n_rows != x.n_rows || static_cast<int>(y->n_cols) != q_)) {
      throw std::invalid_argument("observed response dimensions differ from predictions");
    }
    if (labels != nullptr && labels->size() != static_cast<R_xlen_t>(x.n_rows)) {
      throw std::invalid_argument("observed class-label count differs from predictions");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    PredictionWorkspace& workspace = prediction_workspace(x, prefix);
    const size_t rows = x.n_rows;
    if (metric_rows_ < rows) {
      metric_observed_ = make_matrix(device_, rows, q_);
      metric_labels_ = [device_ newBufferWithLength:rows * sizeof(int32_t)
                                            options:MTLResourceStorageModeShared];
      if (metric_labels_ == nil) {
        throw std::runtime_error("resident Metal metric-label allocation failed");
      }
      metric_rows_ = rows;
    }
    if (metric_output_.rows == 0) metric_output_ = make_matrix(device_, 3, q_);
    if (y != nullptr) {
      if (metric_observed_.rows != rows) {
        metric_observed_ = make_matrix(device_, rows, q_);
      }
      upload_matrix(*y, metric_observed_);
    } else {
      std::memcpy([metric_labels_ contents], INTEGER(*labels),
                  rows * sizeof(int32_t));
    }
    clear_invalid();
    dispatch_1d(queue_, kernels_->pipeline("response_sums"), q_,
        [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:workspace.prediction.buffer offset:0 atIndex:0];
          [encoder setBuffer:(y != nullptr ? metric_observed_.buffer :
                                              workspace.prediction.buffer)
                     offset:0 atIndex:1];
          [encoder setBuffer:metric_labels_ offset:0 atIndex:2];
          [encoder setBuffer:mean_y_.buffer offset:0 atIndex:3];
          [encoder setBuffer:metric_output_.buffer offset:0 atIndex:4];
          [encoder setBuffer:invalid_ offset:0 atIndex:5];
          set_u32(encoder, 6, static_cast<uint32_t>(rows));
          set_u32(encoder, 7, static_cast<uint32_t>(q_));
          set_u32(encoder, 8, stride(workspace.prediction));
          set_u32(encoder, 9, y != nullptr ? stride(metric_observed_) : 0);
          set_u32(encoder, 10, stride(metric_output_));
          set_i32(encoder, 11, classification_ ? 1 : 0);
        });
    require_valid("resident Metal response metrics");
    return copy_from_matrix(metric_output_);
  }

  void controls(int& oversample, int& power, int& block) const override {
    oversample = 0;
    power = effective_power_;
    block = refresh_block_;
  }

  int observations() const override { return n_; }
  int predictors() const override { return input_p_; }
  int responses() const override { return q_; }
  int components() const override { return components_; }
  bool classification() const override { return classification_; }
  bool implicit_crosscovariance() const override {
    return implicit_crosscov_;
  }
  bool predictor_crossprod_cache() const override {
    return cached_predictor_crossprod_;
  }
  bool host_assisted_components() const override { return false; }

 private:
  struct PredictionWorkspace {
    PredictionWorkspace(id<MTLDevice> device,
                        const MetalMatrix& R,
                        int rows,
                        int input_predictors,
                        int feature_predictors,
                        int components,
                        int responses)
        : input(make_matrix(device, rows, input_predictors)),
          scores(make_matrix(device, rows, components)),
          prediction(make_matrix(device, rows, responses)),
          lda_prediction(make_matrix(device, rows, responses)),
          filter_score(make_matrix(device, rows, 1)),
          norms(make_matrix(device, 1, rows)),
          row_means(make_matrix(device, 1, rows)) {
      X = input_predictors == feature_predictors ? input :
          make_matrix(device, rows, feature_predictors);
      project_product.reset(
          new MetalProduct(device, X, R, scores, false, false));
    }
    MetalMatrix input, X, scores, prediction, lda_prediction, filter_score;
    MetalMatrix norms, row_means;
    std::unique_ptr<MetalProduct> project_product;
  };

  void clear_invalid() {
    *static_cast<int32_t*>([invalid_ contents]) = 0;
  }

  void upload_matrix(const arma::fmat& source, MetalMatrix& destination) {
    if (source.n_rows != destination.rows || source.n_cols != destination.cols) {
      throw std::invalid_argument("resident Metal input dimensions changed");
    }
    const size_t bytes = source.n_elem * sizeof(float);
    if (bytes > upload_capacity_) {
      upload_buffer_ = [device_ newBufferWithLength:bytes
                                            options:MTLResourceStorageModeShared];
      if (upload_buffer_ == nil) {
        throw std::runtime_error("resident Metal upload allocation failed");
      }
      upload_capacity_ = bytes;
    }
    std::memcpy([upload_buffer_ contents], source.memptr(), bytes);
    dispatch_2d(queue_, kernels_->pipeline("column_major_to_row_major"),
        destination.cols, destination.rows,
        [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:upload_buffer_ offset:0 atIndex:0];
          [encoder setBuffer:destination.buffer offset:0 atIndex:1];
          set_u32(encoder, 2, static_cast<uint32_t>(destination.rows));
          set_u32(encoder, 3, static_cast<uint32_t>(destination.cols));
          set_u32(encoder, 4, stride(destination));
        });
  }

  void upload_preprocessed_matrix(const arma::fmat& source,
                                  MetalMatrix& destination,
                                  MetalMatrix& mean,
                                  MetalMatrix& scale,
                                  int scaling,
                                  bool calculate_statistics) {
    if (source.n_rows != destination.rows || source.n_cols != destination.cols) {
      throw std::invalid_argument("resident Metal input dimensions changed");
    }
    const size_t bytes = source.n_elem * sizeof(float);
    if (bytes > upload_capacity_) {
      upload_buffer_ = [device_ newBufferWithLength:bytes
                                            options:MTLResourceStorageModeShared];
      if (upload_buffer_ == nil) {
        throw std::runtime_error("resident Metal upload allocation failed");
      }
      upload_capacity_ = bytes;
    }
    std::memcpy([upload_buffer_ contents], source.memptr(), bytes);
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    if (calculate_statistics) {
      id<MTLComputePipelineState> pipeline =
          kernels_->pipeline("column_major_stats");
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:upload_buffer_ offset:0 atIndex:0];
      [encoder setBuffer:mean.buffer offset:0 atIndex:1];
      [encoder setBuffer:scale.buffer offset:0 atIndex:2];
      set_u32(encoder, 3, static_cast<uint32_t>(destination.rows));
      set_u32(encoder, 4, static_cast<uint32_t>(destination.cols));
      set_i32(encoder, 5, scaling);
      const NSUInteger threads = std::min<NSUInteger>(
          256, pipeline.maxTotalThreadsPerThreadgroup);
      [encoder dispatchThreadgroups:MTLSizeMake(destination.cols, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
      [encoder endEncoding];
    }
    encode_2d(command,
        kernels_->pipeline("standardize_column_major_to_row_major"),
        destination.cols, destination.rows,
        [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:upload_buffer_ offset:0 atIndex:0];
          [encoder setBuffer:destination.buffer offset:0 atIndex:1];
          [encoder setBuffer:mean.buffer offset:0 atIndex:2];
          [encoder setBuffer:scale.buffer offset:0 atIndex:3];
          set_u32(encoder, 4, static_cast<uint32_t>(destination.rows));
          set_u32(encoder, 5, static_cast<uint32_t>(destination.cols));
          set_u32(encoder, 6, stride(destination));
        });
    finish_command(command, "resident Metal input preprocessing failed");
  }

  void require_valid(const char* context) {
    if (*static_cast<int32_t*>([invalid_ contents]) != 0) {
      throw std::runtime_error(std::string(context) +
                               ": nonfinite value or rank breakdown");
    }
  }

  void preprocess(MetalMatrix& matrix, MetalMatrix& mean,
                  MetalMatrix& scale, int scaling) {
    dispatch_1d(queue_, kernels_->pipeline("column_stats"), matrix.cols,
        [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:matrix.buffer offset:0 atIndex:0];
          [encoder setBuffer:mean.buffer offset:0 atIndex:1];
          [encoder setBuffer:scale.buffer offset:0 atIndex:2];
          set_u32(encoder, 3, static_cast<uint32_t>(matrix.rows));
          set_u32(encoder, 4, static_cast<uint32_t>(matrix.cols));
          set_u32(encoder, 5, stride(matrix));
          set_i32(encoder, 6, scaling);
        });
    dispatch_2d(queue_, kernels_->pipeline("standardize"), matrix.cols,
        matrix.rows, [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:matrix.buffer offset:0 atIndex:0];
          [encoder setBuffer:mean.buffer offset:0 atIndex:1];
          [encoder setBuffer:scale.buffer offset:0 atIndex:2];
          set_u32(encoder, 3, static_cast<uint32_t>(matrix.rows));
          set_u32(encoder, 4, static_cast<uint32_t>(matrix.cols));
          set_u32(encoder, 5, stride(matrix));
        });
  }

  void initialize_identity(MetalMatrix& mean, MetalMatrix& scale,
                           int columns) {
    dispatch_1d(queue_, kernels_->pipeline("identity_statistics"), columns,
        [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:mean.buffer offset:0 atIndex:0];
          [encoder setBuffer:scale.buffer offset:0 atIndex:1];
          set_u32(encoder, 2, static_cast<uint32_t>(columns));
        });
  }

  void prepare_training_predictors(const arma::fmat& host_x, int scaling) {
    if (!kernelpls_) {
      upload_preprocessed_matrix(
          host_x, X_, mean_x_, scale_x_, scaling, true);
      return;
    }
    upload_preprocessed_matrix(
        host_x, reference_, mean_x_, scale_x_, scaling, true);
    dispatch_1d(queue_, kernels_->pipeline("row_squared_norms"), n_,
        [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:reference_.buffer offset:0 atIndex:0];
          [encoder setBuffer:reference_norms_.buffer offset:0 atIndex:1];
          set_u32(encoder, 2, static_cast<uint32_t>(n_));
          set_u32(encoder, 3, static_cast<uint32_t>(input_p_));
          set_u32(encoder, 4, stride(reference_));
        });
    MetalProduct gram(
        device_, reference_, reference_, X_, false, true);
    run_product(queue_, gram);
    dispatch_2d(queue_, kernels_->pipeline("transform_kernel"), n_, n_,
        [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:X_.buffer offset:0 atIndex:0];
          [encoder setBuffer:reference_norms_.buffer offset:0 atIndex:1];
          [encoder setBuffer:reference_norms_.buffer offset:0 atIndex:2];
          set_u32(encoder, 3, static_cast<uint32_t>(n_));
          set_u32(encoder, 4, static_cast<uint32_t>(n_));
          set_u32(encoder, 5, stride(X_));
          set_i32(encoder, 6, kernel_kind_);
          [encoder setBytes:&kernel_gamma_ length:sizeof(kernel_gamma_)
                     atIndex:7];
          set_i32(encoder, 8, kernel_degree_);
          [encoder setBytes:&kernel_coefficient_
                     length:sizeof(kernel_coefficient_) atIndex:9];
        });
    dispatch_1d(queue_, kernels_->pipeline("column_means"), n_,
        [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:X_.buffer offset:0 atIndex:0];
          [encoder setBuffer:kernel_training_means_.buffer offset:0 atIndex:1];
          set_u32(encoder, 2, static_cast<uint32_t>(n_));
          set_u32(encoder, 3, static_cast<uint32_t>(n_));
          set_u32(encoder, 4, stride(X_));
        });
    dispatch_1d(queue_, kernels_->pipeline("vector_mean"), 1,
        [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:kernel_training_means_.buffer offset:0 atIndex:0];
          [encoder setBuffer:kernel_grand_.buffer offset:0 atIndex:1];
          set_u32(encoder, 2, static_cast<uint32_t>(n_));
        });
    dispatch_2d(queue_, kernels_->pipeline("center_training_kernel"), n_, n_,
        [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:X_.buffer offset:0 atIndex:0];
          [encoder setBuffer:kernel_training_means_.buffer offset:0 atIndex:1];
          [encoder setBuffer:kernel_grand_.buffer offset:0 atIndex:2];
          set_u32(encoder, 3, static_cast<uint32_t>(n_));
          set_u32(encoder, 4, static_cast<uint32_t>(n_));
          set_u32(encoder, 5, stride(X_));
        });
  }

  void fit_opls_filters() {
    if (!opls_ || north_ < 1) return;
    MetalProduct score_product(device_, X_, r_, opls_score_, false, false);
    MetalProduct score_norm(
        device_, opls_score_, opls_score_, score_norm_, true, false);
    MetalProduct loading_product(
        device_, X_, opls_score_, opls_loading_, true, false);
    MetalProduct numerator(
        device_, r_, opls_loading_, opls_ratio_, true, false);
    MetalProduct denominator(
        device_, r_, r_, opls_denominator_, true, false);
    MetalProduct orthogonal_score(
        device_, X_, opls_weight_, opls_score_, false, false);
    for (int component = 0; component < north_; ++component) {
      clear_invalid();
      solve_rank_one_direction(
          R_, 0, seed_ + 7919u + static_cast<unsigned int>(component));
      run_product(queue_, score_product);
      run_product(queue_, score_norm);
      dispatch_1d(queue_, kernels_->pipeline("reciprocal_scalar"), 1,
          [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:score_norm_.buffer offset:0 atIndex:0];
            [encoder setBuffer:invalid_ offset:0 atIndex:1];
          });
      run_product(queue_, loading_product);
      dispatch_1d(queue_, kernels_->pipeline("scale_vector"), p_,
          [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:opls_loading_.buffer offset:0 atIndex:0];
            [encoder setBuffer:score_norm_.buffer offset:0 atIndex:1];
            set_u32(encoder, 2, static_cast<uint32_t>(p_));
            set_u32(encoder, 3, stride(opls_loading_));
          });
      run_product(queue_, numerator);
      run_product(queue_, denominator);
      dispatch_1d(queue_, kernels_->pipeline("scalar_ratio"), 1,
          [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:opls_ratio_.buffer offset:0 atIndex:0];
            [encoder setBuffer:opls_denominator_.buffer offset:0 atIndex:1];
            [encoder setBuffer:opls_ratio_.buffer offset:0 atIndex:2];
            [encoder setBuffer:invalid_ offset:0 atIndex:3];
          });
      dispatch_1d(queue_, kernels_->pipeline("vector_difference_ratio"), p_,
          [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:opls_weight_.buffer offset:0 atIndex:0];
            [encoder setBuffer:opls_loading_.buffer offset:0 atIndex:1];
            [encoder setBuffer:r_.buffer offset:0 atIndex:2];
            [encoder setBuffer:opls_ratio_.buffer offset:0 atIndex:3];
            set_u32(encoder, 4, static_cast<uint32_t>(p_));
            set_u32(encoder, 5, stride(opls_weight_));
            set_u32(encoder, 6, stride(opls_loading_));
            set_u32(encoder, 7, stride(r_));
          });
      MetalProduct orthogonal_norm(
          device_, opls_weight_, opls_weight_, vector_norm_, true, false);
      run_product(queue_, orthogonal_norm);
      dispatch_1d(queue_, kernels_->pipeline("reciprocal_sqrt_scalar"), 1,
          [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:vector_norm_.buffer offset:0 atIndex:0];
            [encoder setBuffer:invalid_ offset:0 atIndex:1];
          });
      dispatch_1d(queue_, kernels_->pipeline("scale_vector"), p_,
          [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:opls_weight_.buffer offset:0 atIndex:0];
            [encoder setBuffer:vector_norm_.buffer offset:0 atIndex:1];
            set_u32(encoder, 2, static_cast<uint32_t>(p_));
            set_u32(encoder, 3, stride(opls_weight_));
          });
      run_product(queue_, orthogonal_score);
      run_product(queue_, score_norm);
      dispatch_1d(queue_, kernels_->pipeline("reciprocal_scalar"), 1,
          [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:score_norm_.buffer offset:0 atIndex:0];
            [encoder setBuffer:invalid_ offset:0 atIndex:1];
          });
      run_product(queue_, loading_product);
      dispatch_1d(queue_, kernels_->pipeline("scale_vector"), p_,
          [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:opls_loading_.buffer offset:0 atIndex:0];
            [encoder setBuffer:score_norm_.buffer offset:0 atIndex:1];
            set_u32(encoder, 2, static_cast<uint32_t>(p_));
            set_u32(encoder, 3, stride(opls_loading_));
          });
      copy_vector(opls_weight_, opls_weights_, component);
      copy_vector(opls_loading_, opls_loadings_, component);
      dispatch_2d(queue_, kernels_->pipeline("rank1_subtract_columns"), p_, n_,
          [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:X_.buffer offset:0 atIndex:0];
            [encoder setBuffer:opls_score_.buffer offset:0 atIndex:1];
            [encoder setBuffer:opls_loading_.buffer offset:0 atIndex:2];
            set_u32(encoder, 3, static_cast<uint32_t>(n_));
            set_u32(encoder, 4, static_cast<uint32_t>(p_));
            set_u32(encoder, 5, stride(X_));
            set_u32(encoder, 6, stride(opls_score_));
            set_u32(encoder, 7, stride(opls_loading_));
          });
      if (classification_) {
        prepare_class_response();
      } else {
        MetalProduct cross(device_, X_, Y_, S_, true, false);
        run_product(queue_, cross);
      }
      require_valid("resident Metal OPLS filtering");
    }
  }

  void prepare_opls_projection() {
    if (!opls_) return;
    dispatch_2d(queue_, kernels_->pipeline("copy_prefix_matrix"),
        components_, p_, [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:R_.buffer offset:0 atIndex:0];
          [encoder setBuffer:opls_effective_r_.buffer offset:0 atIndex:1];
          set_u32(encoder, 2, static_cast<uint32_t>(p_));
          set_u32(encoder, 3, static_cast<uint32_t>(components_));
          set_u32(encoder, 4, stride(R_));
          set_u32(encoder, 5, stride(opls_effective_r_));
        });
    for (int component = north_ - 1; component >= 0; --component) {
      id<MTLCommandBuffer> load = [queue_ commandBuffer];
      encode_extract_column(load, opls_weights_, opls_weight_, component);
      encode_extract_column(load, opls_loadings_, opls_loading_, component);
      finish_command(load, "resident Metal OPLS projection load failed");
      MetalProduct projection(
          device_, opls_loading_, opls_effective_r_, opls_component_row_,
          true, false);
      run_product(queue_, projection);
      dispatch_2d(queue_, kernels_->pipeline("rank1_subtract"),
          components_, p_, [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:opls_effective_r_.buffer offset:0 atIndex:0];
            [encoder setBuffer:opls_weight_.buffer offset:0 atIndex:1];
            [encoder setBuffer:opls_component_row_.buffer offset:0 atIndex:2];
            set_u32(encoder, 3, static_cast<uint32_t>(p_));
            set_u32(encoder, 4, static_cast<uint32_t>(components_));
            set_u32(encoder, 5, stride(opls_effective_r_));
            set_u32(encoder, 6, stride(opls_weight_));
          });
    }
  }

  void prepare_class_response() {
    clear_invalid();
    dispatch_1d(queue_, kernels_->pipeline("class_priors"), q_,
        [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:labels_ offset:0 atIndex:0];
          [encoder setBuffer:mean_y_.buffer offset:0 atIndex:1];
          [encoder setBuffer:invalid_ offset:0 atIndex:2];
          set_u32(encoder, 3, static_cast<uint32_t>(n_));
          set_u32(encoder, 4, static_cast<uint32_t>(q_));
        });
    if (Y_.rows > 0) {
      dispatch_2d(queue_, kernels_->pipeline("dummy_response"), q_, n_,
          [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:labels_ offset:0 atIndex:0];
            [encoder setBuffer:mean_y_.buffer offset:0 atIndex:1];
            [encoder setBuffer:Y_.buffer offset:0 atIndex:2];
            set_u32(encoder, 3, static_cast<uint32_t>(n_));
            set_u32(encoder, 4, static_cast<uint32_t>(q_));
            set_u32(encoder, 5, stride(Y_));
          });
      MetalProduct cross(device_, X_, Y_, S_, true, false);
      run_product(queue_, cross);
    } else {
      dispatch_1d(queue_, kernels_->pipeline("class_product"), p_,
          [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:X_.buffer offset:0 atIndex:0];
            [encoder setBuffer:labels_ offset:0 atIndex:1];
            [encoder setBuffer:S_.buffer offset:0 atIndex:2];
            set_u32(encoder, 3, static_cast<uint32_t>(n_));
            set_u32(encoder, 4, static_cast<uint32_t>(p_));
            set_u32(encoder, 5, static_cast<uint32_t>(q_));
            set_u32(encoder, 6, stride(X_));
            set_u32(encoder, 7, stride(S_));
          });
    }
    require_valid("resident Metal class response");
  }

  void encode_orthonormalize(id<MTLCommandBuffer> command,
                             MetalMatrix& vector,
                             const MetalMatrix& basis, int used) {
    encode_block_qr(command, vector, basis, 1, used);
  }

  void orthonormalize(MetalMatrix& vector, const MetalMatrix& basis,
                      int used) {
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    encode_orthonormalize(command, vector, basis, used);
    finish_command(command, "resident Metal orthonormalization failed");
  }

  void encode_orthonormalize_block(id<MTLCommandBuffer> command,
                                   MetalMatrix& block,
                                   const MetalMatrix& basis,
                                   int columns, int used) {
    encode_1d(command, kernels_->pipeline("orthonormalize_block"), 1,
        [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:block.buffer offset:0 atIndex:0];
          [encoder setBuffer:basis.buffer offset:0 atIndex:1];
          [encoder setBuffer:invalid_ offset:0 atIndex:2];
          set_u32(encoder, 3, static_cast<uint32_t>(block.rows));
          set_u32(encoder, 4, static_cast<uint32_t>(columns));
          set_u32(encoder, 5, static_cast<uint32_t>(used));
          set_u32(encoder, 6, stride(block));
          set_u32(encoder, 7, stride(basis));
        });
  }

  void encode_extract_column(id<MTLCommandBuffer> command,
                             const MetalMatrix& matrix,
                             MetalMatrix& vector, int column) {
    encode_1d(command, kernels_->pipeline("extract_column"), matrix.rows,
        [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:matrix.buffer offset:0 atIndex:0];
          [encoder setBuffer:vector.buffer offset:0 atIndex:1];
          set_u32(encoder, 2, static_cast<uint32_t>(matrix.rows));
          set_u32(encoder, 3, stride(matrix));
          set_u32(encoder, 4, stride(vector));
          set_u32(encoder, 5, static_cast<uint32_t>(column));
        });
  }

  void encode_block_qr(id<MTLCommandBuffer> command,
                       MetalMatrix& block,
                       const MetalMatrix& external_basis,
                       int columns, int used) {
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    id<MTLComputePipelineState> pipeline =
        kernels_->pipeline("orthonormalize_block_parallel");
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:block.buffer offset:0 atIndex:0];
    [encoder setBuffer:external_basis.buffer offset:0 atIndex:1];
    [encoder setBuffer:invalid_ offset:0 atIndex:2];
    set_u32(encoder, 3, static_cast<uint32_t>(block.rows));
    set_u32(encoder, 4, static_cast<uint32_t>(columns));
    set_u32(encoder, 5, static_cast<uint32_t>(used));
    set_u32(encoder, 6, stride(block));
    set_u32(encoder, 7, stride(external_basis));
    const NSUInteger threads = std::min<NSUInteger>(
        256, pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
  }

  void encode_cholesky_qr(id<MTLCommandBuffer> command,
                          MetalMatrix& block, int columns,
                          int stabilization_passes = 2) {
    if (columns < 16) {
      encode_block_qr(command, block, block, columns, 0);
      return;
    }
    MetalMatrix gram = matrix_shape(projected_gram_, columns, columns);
    MetalMatrix factor = matrix_shape(projected_vectors_, columns, columns);
    MetalMatrix rotated = matrix_shape(
        candidate_rotated_, block.rows, columns);
    for (int pass = 0; pass < stabilization_passes; ++pass) {
      MetalProduct form_gram(
          device_, block, block, gram, true, false);
      form_gram.encode(command);
      encode_1d(command, kernels_->pipeline("regularize_gram"), 1,
          [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:gram.buffer offset:0 atIndex:0];
            set_u32(encoder, 1, static_cast<uint32_t>(columns));
            set_u32(encoder, 2, stride(gram));
          });
      MPSMatrixDecompositionCholesky* decomposition =
          [[MPSMatrixDecompositionCholesky alloc]
              initWithDevice:device_ lower:NO order:columns];
      [decomposition encodeToCommandBuffer:command
                               sourceMatrix:matrix_view(gram)
                               resultMatrix:matrix_view(factor)
                                     status:block_status_];
      MPSMatrixSolveTriangular* solve = [[MPSMatrixSolveTriangular alloc]
          initWithDevice:device_
                   right:YES
                   upper:YES
               transpose:NO
                    unit:NO
                   order:columns
      numberOfRightHandSides:block.rows
                   alpha:1.0];
      [solve encodeToCommandBuffer:command
                      sourceMatrix:matrix_view(factor)
               rightHandSideMatrix:matrix_view(block)
                    solutionMatrix:matrix_view(rotated)];
      encode_2d(command, kernels_->pipeline("copy_prefix_matrix"),
          columns, block.rows,
          [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:rotated.buffer offset:0 atIndex:0];
            [encoder setBuffer:block.buffer offset:0 atIndex:1];
            set_u32(encoder, 2, static_cast<uint32_t>(block.rows));
            set_u32(encoder, 3, static_cast<uint32_t>(columns));
            set_u32(encoder, 4, stride(rotated));
            set_u32(encoder, 5, stride(block));
          });
    }
  }

  void encode_copy_vector(id<MTLCommandBuffer> command,
                          const MetalMatrix& source,
                          MetalMatrix& destination, int column) {
    encode_1d(command, kernels_->pipeline("copy_vector"), source.rows,
        [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:source.buffer offset:0 atIndex:0];
          [encoder setBuffer:destination.buffer offset:0 atIndex:1];
          set_u32(encoder, 2, static_cast<uint32_t>(source.rows));
          set_u32(encoder, 3, stride(source));
          set_u32(encoder, 4, stride(destination));
          set_u32(encoder, 5, static_cast<uint32_t>(column));
        });
  }

  void copy_vector(const MetalMatrix& source, MetalMatrix& destination,
                   int column) {
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    encode_copy_vector(command, source, destination, column);
    finish_command(command, "resident Metal vector copy failed");
  }

  void encode_copy_plain(id<MTLCommandBuffer> command,
                         const MetalMatrix& source,
                         MetalMatrix& destination) {
    encode_1d(command, kernels_->pipeline("copy_plain"), source.rows,
        [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:source.buffer offset:0 atIndex:0];
          [encoder setBuffer:destination.buffer offset:0 atIndex:1];
          set_u32(encoder, 2, static_cast<uint32_t>(source.rows));
          set_u32(encoder, 3, stride(source));
          set_u32(encoder, 4, stride(destination));
        });
  }

  void copy_plain(const MetalMatrix& source, MetalMatrix& destination) {
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    encode_copy_plain(command, source, destination);
    finish_command(command, "resident Metal plain-vector copy failed");
  }

  bool solve_rank_one_from_right_gram(const MetalMatrix& basis, int used,
                                      unsigned int solve_seed, int width) {
    clear_invalid();
    MetalMatrix right = leading_columns(right_block_, width);
    MetalMatrix right_scratch = matrix_shape(candidate_rotated_, q_, width);
    MetalMatrix candidate = leading_columns(candidate_basis_, width);
    MetalMatrix gram = matrix_shape(projected_gram_, width, width);
    MetalMatrix vectors = matrix_shape(projected_vectors_, width, width);
    MetalMatrix reduced = matrix_shape(reduced_block_, width, q_);
    MetalMatrix eigenvector = matrix_shape(reduced_vector_, width, 1);

    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    encode_2d(command, kernels_->pipeline("random_matrix"), width, q_,
        [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:right.buffer offset:0 atIndex:0];
          set_u32(encoder, 1, static_cast<uint32_t>(q_));
          set_u32(encoder, 2, static_cast<uint32_t>(width));
          set_u32(encoder, 3, solve_seed);
          set_u32(encoder, 4, stride(right));
        });

    MetalMatrix active = right;
    MetalMatrix scratch = right_scratch;
    for (int iteration = 0; iteration < effective_power_; ++iteration) {
      MetalProduct power_step(
          device_, right_gram_, active, scratch, false, false);
      power_step.encode(command);
      encode_cholesky_qr(command, scratch, width);
      std::swap(active, scratch);
    }
    MetalProduct form_candidate(
        device_, S_, active, candidate, false, false);
    form_candidate.encode(command);
    if (used > 0) {
      MetalMatrix previous = matrix_shape(V_, p_, used);
      MetalMatrix projection = matrix_shape(block_projection_, used, width);
      MetalMatrix projected = matrix_shape(candidate_rotated_, p_, width);
      for (int pass = 0; pass < 2; ++pass) {
        MetalProduct project(
            device_, previous, candidate, projection, true, false);
        MetalProduct reconstruct(
            device_, previous, projection, projected, false, false);
        project.encode(command);
        reconstruct.encode(command);
        encode_2d(command, kernels_->pipeline("subtract_matrix"), width, p_,
            [&](id<MTLComputeCommandEncoder> encoder) {
              [encoder setBuffer:candidate.buffer offset:0 atIndex:0];
              [encoder setBuffer:projected.buffer offset:0 atIndex:1];
              set_u32(encoder, 2, static_cast<uint32_t>(p_));
              set_u32(encoder, 3, static_cast<uint32_t>(width));
              set_u32(encoder, 4, stride(candidate));
              set_u32(encoder, 5, stride(projected));
            });
      }
    }
    encode_cholesky_qr(command, candidate, width);
    MetalProduct form_reduced(
        device_, candidate, S_, reduced, true, false);
    MetalProduct form_gram(
        device_, reduced, reduced, gram, false, true);
    form_reduced.encode(command);
    form_gram.encode(command);
    encode_1d(command, kernels_->pipeline("symmetric_eigenvectors"), 1,
        [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:gram.buffer offset:0 atIndex:0];
          [encoder setBuffer:vectors.buffer offset:0 atIndex:1];
          [encoder setBuffer:invalid_ offset:0 atIndex:2];
          set_u32(encoder, 3, static_cast<uint32_t>(width));
          set_u32(encoder, 4, stride(gram));
          set_u32(encoder, 5, stride(vectors));
        });
    encode_extract_column(command, vectors, eigenvector, 0);
    MetalProduct rotate(
        device_, candidate, eigenvector, r_, false, false);
    rotate.encode(command);
    encode_orthonormalize(command, r_, basis, used);
    finish_command(command, "resident Metal response-Gram rSVD failed");
    return *static_cast<int32_t*>([invalid_ contents]) == 0;
  }

  bool solve_rank_one_direction_attempt(const MetalMatrix& basis, int used,
                                        unsigned int solve_seed, int width) {
    if (right_gram_refresh_) {
      return solve_rank_one_from_right_gram(
          basis, used, solve_seed, width);
    }
    clear_invalid();
    MetalMatrix right = leading_columns(right_block_, width);
    MetalMatrix candidate = leading_columns(candidate_basis_, width);
    MetalMatrix samples = leading_columns(sample_block_, width);
    MetalMatrix gram = matrix_shape(projected_gram_, width, width);
    MetalMatrix vectors = matrix_shape(projected_vectors_, width, width);
    MetalMatrix reduced = matrix_shape(reduced_block_, width, q_);
    MetalMatrix eigenvector = matrix_shape(reduced_vector_, width, 1);

    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    if (width == 1 && implicit_crosscov_) {
      // For a rank-one solve, start directly in predictor space. A response-
      // side sketch followed by a one-dimensional reduced SVD produces the
      // same left direction up to sign but doubles the large operator work.
      encode_2d(command, kernels_->pipeline("random_matrix"), 1, p_,
          [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:candidate.buffer offset:0 atIndex:0];
            set_u32(encoder, 1, static_cast<uint32_t>(p_));
            set_u32(encoder, 2, 1);
            set_u32(encoder, 3, solve_seed);
            set_u32(encoder, 4, stride(candidate));
          });
      encode_block_qr(command, candidate, basis, 1, used);
      for (int iteration = 0; iteration < effective_power_; ++iteration) {
        rank1_x_forward_->encode(command);
        if (use_sample_response_gram_) {
          rank1_response_gram_->encode(command);
        } else {
          rank1_y_transpose_->encode(command);
          encode_block_qr(command, right, right, 1, 0);
          rank1_y_forward_->encode(command);
        }
        rank1_x_transpose_->encode(command);
        encode_block_qr(command, candidate, basis, 1, used);
      }
      encode_copy_plain(command, candidate, r_);
      finish_command(command, "resident Metal rank-one rSVD failed");
      return *static_cast<int32_t*>([invalid_ contents]) == 0;
    }
    encode_2d(command, kernels_->pipeline("random_matrix"), width, q_,
        [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:right.buffer offset:0 atIndex:0];
          set_u32(encoder, 1, static_cast<uint32_t>(q_));
          set_u32(encoder, 2, static_cast<uint32_t>(width));
          set_u32(encoder, 3, solve_seed);
          set_u32(encoder, 4, stride(right));
        });
    encode_block_qr(command, right, right, width, 0);
    if (implicit_crosscov_) {
      MetalProduct y_forward(device_, Y_, right, samples, false, false);
      MetalProduct x_transpose(
          device_, X_, samples, candidate, true, false);
      y_forward.encode(command);
      x_transpose.encode(command);
    } else {
      MetalProduct forward(device_, S_, right, candidate, false, false);
      forward.encode(command);
    }
    encode_block_qr(command, candidate, basis, width, used);
    for (int iteration = 0; iteration < effective_power_; ++iteration) {
      if (implicit_crosscov_) {
        MetalProduct x_forward(
            device_, X_, candidate, samples, false, false);
        MetalProduct y_transpose(
            device_, Y_, samples, right, true, false);
        x_forward.encode(command);
        y_transpose.encode(command);
      } else {
        MetalProduct transpose(
            device_, S_, candidate, right, true, false);
        transpose.encode(command);
      }
      encode_block_qr(command, right, right, width, 0);
      if (implicit_crosscov_) {
        MetalProduct y_forward(
            device_, Y_, right, samples, false, false);
        MetalProduct x_transpose(
            device_, X_, samples, candidate, true, false);
        y_forward.encode(command);
        x_transpose.encode(command);
      } else {
        MetalProduct forward(
            device_, S_, right, candidate, false, false);
        forward.encode(command);
      }
      encode_block_qr(command, candidate, basis, width, used);
    }
    if (implicit_crosscov_) {
      MetalProduct x_forward(
          device_, X_, candidate, samples, false, false);
      MetalProduct reduced_product(
          device_, samples, Y_, reduced, true, false);
      x_forward.encode(command);
      reduced_product.encode(command);
    } else {
      MetalProduct reduced_product(
          device_, candidate, S_, reduced, true, false);
      reduced_product.encode(command);
    }
    MetalProduct gram_product(
        device_, reduced, reduced, gram, false, true);
    gram_product.encode(command);
    encode_1d(command, kernels_->pipeline("symmetric_eigenvectors"), 1,
        [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:gram.buffer offset:0 atIndex:0];
          [encoder setBuffer:vectors.buffer offset:0 atIndex:1];
          [encoder setBuffer:invalid_ offset:0 atIndex:2];
          set_u32(encoder, 3, static_cast<uint32_t>(width));
          set_u32(encoder, 4, stride(gram));
          set_u32(encoder, 5, stride(vectors));
        });
    encode_extract_column(command, vectors, eigenvector, 0);
    MetalProduct rotate(
        device_, candidate, eigenvector, r_, false, false);
    rotate.encode(command);
    encode_orthonormalize(command, r_, basis, used);
    finish_command(command, "resident Metal oversampled rSVD failed");
    return *static_cast<int32_t*>([invalid_ contents]) == 0;
  }

  void encode_sample_gram_rank_one_direction(
      id<MTLCommandBuffer> command, const MetalMatrix& basis, int used,
      unsigned int solve_seed) {
    MetalMatrix candidate = leading_columns(candidate_basis_, 1);
    encode_2d(command, kernels_->pipeline("random_matrix"), 1, p_,
        [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:candidate.buffer offset:0 atIndex:0];
          set_u32(encoder, 1, static_cast<uint32_t>(p_));
          set_u32(encoder, 2, 1);
          set_u32(encoder, 3, solve_seed);
          set_u32(encoder, 4, stride(candidate));
        });
    encode_block_qr(command, candidate, basis, 1, used);
    for (int iteration = 0; iteration < effective_power_; ++iteration) {
      rank1_x_forward_->encode(command);
      rank1_response_gram_->encode(command);
      rank1_x_transpose_->encode(command);
      encode_block_qr(command, candidate, basis, 1, used);
    }
    encode_copy_plain(command, candidate, r_);
  }

  void solve_rank_one_direction(const MetalMatrix& basis, int used,
                                unsigned int solve_seed) {
    const int initial_width =
        std::max(1, std::min(solver_width_, p_ - used));
    int width = initial_width;
    while (true) {
      if (solve_rank_one_direction_attempt(basis, used, solve_seed, width)) {
        return;
      }
      if (width == 1) break;
      width = std::max(1, width / 2);
    }
    if (right_gram_refresh_) {
      // A heavily deflated response Gram matrix can lose positive rank before
      // the explicit cross-covariance does in float32. Continue with the
      // direct operator rather than returning an unchecked direction.
      right_gram_refresh_ = false;
      width = initial_width;
      while (true) {
        if (solve_rank_one_direction_attempt(
                basis, used, solve_seed, width)) {
          return;
        }
        if (width == 1) break;
        width = std::max(1, width / 2);
      }
    }
    throw std::runtime_error(
        "resident Metal oversampled rSVD: nonfinite value or rank breakdown");
  }

  void encode_matrix_vector(id<MTLCommandBuffer> command,
                            const MetalMatrix& matrix,
                            const MetalMatrix& input,
                            MetalMatrix& output,
                            bool transpose) {
    const NSUInteger result_size = transpose ? matrix.cols : matrix.rows;
    encode_1d(command, kernels_->pipeline(
        transpose ? "matrix_transpose_vector" : "matrix_vector"),
        result_size, [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:matrix.buffer offset:0 atIndex:0];
          [encoder setBuffer:input.buffer offset:0 atIndex:1];
          [encoder setBuffer:output.buffer offset:0 atIndex:2];
          set_u32(encoder, 3, static_cast<uint32_t>(matrix.rows));
          set_u32(encoder, 4, static_cast<uint32_t>(matrix.cols));
          set_u32(encoder, 5, stride(matrix));
          set_u32(encoder, 6, stride(input));
          set_u32(encoder, 7, stride(output));
        });
  }

  void prepare_plssvd_moments() {
    MetalProduct gram_product(device_, T_, T_, plssvd_gram_, true, false);
    run_product(queue_, gram_product);
    if (implicit_crosscov_) {
      dispatch_2d(queue_, kernels_->pipeline("lda_transpose_means"), q_,
          components_, [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:Q_.buffer offset:0 atIndex:0];
            [encoder setBuffer:plssvd_cross_.buffer offset:0 atIndex:1];
            set_u32(encoder, 2, static_cast<uint32_t>(components_));
            set_u32(encoder, 3, static_cast<uint32_t>(q_));
            set_u32(encoder, 4, stride(Q_));
            set_u32(encoder, 5, stride(plssvd_cross_));
          });
    } else {
      MetalProduct cross_product(
          device_, R_, S_, plssvd_cross_, true, false);
      run_product(queue_, cross_product);
    }
  }

  void prepare_plssvd(int prefix) {
    if (!plssvd_ || prefix < 1 || prefix > components_) {
      throw std::invalid_argument("invalid resident Metal PLS-SVD prefix");
    }
    if (prefix == plssvd_prefix_) return;

    bool success = false;
    for (float rho : {0.0f, 1.0e-8f, 1.0e-6f, 1.0e-5f, 1.0e-4f,
                      1.0e-3f, 1.0e-2f}) {
      dispatch_1d(queue_, kernels_->pipeline("lda_ridge_scale"), 1,
          [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:plssvd_gram_.buffer offset:0 atIndex:0];
            [encoder setBuffer:plssvd_zero_.buffer offset:0 atIndex:1];
            [encoder setBuffer:plssvd_ridge_.buffer offset:0 atIndex:2];
            set_u32(encoder, 3, static_cast<uint32_t>(components_));
            set_u32(encoder, 4, static_cast<uint32_t>(prefix));
            set_u32(encoder, 5, stride(plssvd_gram_));
            set_u32(encoder, 6, stride(plssvd_zero_));
            set_u32(encoder, 7, 1);
            [encoder setBytes:&rho length:sizeof(rho) atIndex:8];
          });
      dispatch_2d(queue_, kernels_->pipeline("lda_prepare_factor"), prefix,
          prefix, [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:plssvd_gram_.buffer offset:0 atIndex:0];
            [encoder setBuffer:plssvd_zero_.buffer offset:0 atIndex:1];
            [encoder setBuffer:plssvd_ridge_.buffer offset:0 atIndex:2];
            [encoder setBuffer:plssvd_factor_.buffer offset:0 atIndex:3];
            set_u32(encoder, 4, static_cast<uint32_t>(prefix));
            set_u32(encoder, 5, stride(plssvd_gram_));
            set_u32(encoder, 6, stride(plssvd_zero_));
            set_u32(encoder, 7, stride(plssvd_factor_));
            set_u32(encoder, 8, 1);
          });
      *static_cast<int32_t*>([plssvd_status_ contents]) =
          MPSMatrixDecompositionStatusFailure;
      MPSMatrixDecompositionCholesky* decomposition =
          [[MPSMatrixDecompositionCholesky alloc]
              initWithDevice:device_ lower:YES order:prefix];
      MPSMatrix* factor = matrix_view(plssvd_factor_, prefix, prefix);
      id<MTLCommandBuffer> command = [queue_ commandBuffer];
      [decomposition encodeToCommandBuffer:command
                               sourceMatrix:factor
                               resultMatrix:factor
                                     status:plssvd_status_];
      finish_command(command, "resident Metal PLS-SVD Cholesky failed");
      success = *static_cast<int32_t*>([plssvd_status_ contents]) ==
          MPSMatrixDecompositionStatusSuccess;
      if (success) break;
    }
    if (!success) {
      throw std::runtime_error(
          "resident Metal PLS-SVD score-Gram solve failed");
    }
    dispatch_2d(queue_, kernels_->pipeline("copy_prefix_matrix"), q_, prefix,
        [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:plssvd_cross_.buffer offset:0 atIndex:0];
          [encoder setBuffer:plssvd_rhs_.buffer offset:0 atIndex:1];
          set_u32(encoder, 2, static_cast<uint32_t>(prefix));
          set_u32(encoder, 3, static_cast<uint32_t>(q_));
          set_u32(encoder, 4, stride(plssvd_cross_));
          set_u32(encoder, 5, stride(plssvd_rhs_));
        });
    MPSMatrixSolveCholesky* solve = [[MPSMatrixSolveCholesky alloc]
        initWithDevice:device_
                  upper:NO
                  order:prefix
     numberOfRightHandSides:q_];
    id<MTLCommandBuffer> solve_command = [queue_ commandBuffer];
    [solve encodeToCommandBuffer:solve_command
                    sourceMatrix:matrix_view(plssvd_factor_, prefix, prefix)
             rightHandSideMatrix:matrix_view(plssvd_rhs_, prefix, q_)
                  solutionMatrix:matrix_view(plssvd_weights_, prefix, q_)];
    finish_command(
        solve_command, "resident Metal PLS-SVD triangular solve failed");
    plssvd_prefix_ = prefix;
  }

  void plssvd_predict(PredictionWorkspace& workspace, int prefix) {
    prepare_plssvd(prefix);
    MPSMatrixMultiplication* product = [[MPSMatrixMultiplication alloc]
        initWithDevice:device_
         transposeLeft:NO
        transposeRight:NO
            resultRows:workspace.scores.rows
         resultColumns:q_
       interiorColumns:prefix
                  alpha:1.0
                   beta:0.0];
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    [product encodeToCommandBuffer:command
                        leftMatrix:matrix_view(
                            workspace.scores, workspace.scores.rows, prefix)
                       rightMatrix:matrix_view(plssvd_weights_, prefix, q_)
                       resultMatrix:matrix_view(workspace.prediction)];
    finish_command(command, "resident Metal PLS-SVD prediction failed");
  }

  void prepare_lda_moments() {
    if (lda_moments_ready_) return;
    dispatch_2d(queue_, kernels_->pipeline("lda_class_means"), components_, q_,
        [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:T_.buffer offset:0 atIndex:0];
          [encoder setBuffer:labels_ offset:0 atIndex:1];
          [encoder setBuffer:lda_means_.buffer offset:0 atIndex:2];
          [encoder setBuffer:lda_weighted_.buffer offset:0 atIndex:3];
          set_u32(encoder, 4, static_cast<uint32_t>(n_));
          set_u32(encoder, 5, static_cast<uint32_t>(components_));
          set_u32(encoder, 6, static_cast<uint32_t>(q_));
          set_u32(encoder, 7, stride(T_));
          set_u32(encoder, 8, stride(lda_means_));
        });
    MetalProduct gram_product(device_, T_, T_, lda_gram_, true, false);
    MetalProduct mean_product(
        device_, lda_weighted_, lda_weighted_, lda_mean_cross_, true, false);
    run_product(queue_, gram_product);
    run_product(queue_, mean_product);
    lda_moments_ready_ = true;
  }

  void prepare_variance() {
    if (variance_ready_) return;
    dispatch_1d(queue_, kernels_->pipeline("matrix_sum_squares"), 1,
        [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:X_.buffer offset:0 atIndex:0];
          [encoder setBuffer:predictor_ss_.buffer offset:0 atIndex:1];
          set_u32(encoder, 2, static_cast<uint32_t>(n_));
          set_u32(encoder, 3, static_cast<uint32_t>(p_));
          set_u32(encoder, 4, stride(X_));
          set_u32(encoder, 5, static_cast<uint32_t>(components_));
        });
    dispatch_1d(queue_, kernels_->pipeline("matrix_column_sums_squares"),
        components_, [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:P_.buffer offset:0 atIndex:0];
          [encoder setBuffer:predictor_ss_.buffer offset:0 atIndex:1];
          set_u32(encoder, 2, static_cast<uint32_t>(p_));
          set_u32(encoder, 3, static_cast<uint32_t>(components_));
          set_u32(encoder, 4, stride(P_));
        });
    variance_ready_ = true;
  }

  void prepare_lda(int prefix) {
    if (!classification_ || prefix < 1 || prefix > components_) {
      throw std::invalid_argument("invalid resident Metal LDA prefix");
    }
    prepare_lda_moments();
    if (prefix == lda_prefix_) return;

    const uint32_t denominator = static_cast<uint32_t>(std::max(1, n_ - q_));
    bool success = false;
    for (float rho : {1.0e-8f, 1.0e-6f, 1.0e-5f, 1.0e-4f,
                      1.0e-3f, 1.0e-2f}) {
      dispatch_1d(queue_, kernels_->pipeline("lda_ridge_scale"), 1,
          [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:lda_gram_.buffer offset:0 atIndex:0];
            [encoder setBuffer:lda_mean_cross_.buffer offset:0 atIndex:1];
            [encoder setBuffer:lda_ridge_.buffer offset:0 atIndex:2];
            set_u32(encoder, 3, static_cast<uint32_t>(components_));
            set_u32(encoder, 4, static_cast<uint32_t>(prefix));
            set_u32(encoder, 5, stride(lda_gram_));
            set_u32(encoder, 6, stride(lda_mean_cross_));
            set_u32(encoder, 7, denominator);
            [encoder setBytes:&rho length:sizeof(rho) atIndex:8];
          });
      dispatch_2d(queue_, kernels_->pipeline("lda_prepare_factor"), prefix,
          prefix, [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:lda_gram_.buffer offset:0 atIndex:0];
            [encoder setBuffer:lda_mean_cross_.buffer offset:0 atIndex:1];
            [encoder setBuffer:lda_ridge_.buffer offset:0 atIndex:2];
            [encoder setBuffer:lda_factor_.buffer offset:0 atIndex:3];
            set_u32(encoder, 4, static_cast<uint32_t>(prefix));
            set_u32(encoder, 5, stride(lda_gram_));
            set_u32(encoder, 6, stride(lda_mean_cross_));
            set_u32(encoder, 7, stride(lda_factor_));
            set_u32(encoder, 8, denominator);
          });

      *static_cast<int32_t*>([lda_status_ contents]) =
          MPSMatrixDecompositionStatusFailure;
      MPSMatrixDecompositionCholesky* decomposition =
          [[MPSMatrixDecompositionCholesky alloc]
              initWithDevice:device_ lower:YES order:prefix];
      MPSMatrix* factor = matrix_view(lda_factor_, prefix, prefix);
      id<MTLCommandBuffer> command = [queue_ commandBuffer];
      [decomposition encodeToCommandBuffer:command
                               sourceMatrix:factor
                               resultMatrix:factor
                                     status:lda_status_];
      finish_command(command, "resident Metal LDA Cholesky failed");
      success = *static_cast<int32_t*>([lda_status_ contents]) ==
          MPSMatrixDecompositionStatusSuccess;
      if (success) break;
    }
    if (!success) {
      throw std::runtime_error(
          "resident Metal LDA Cholesky failed for every regularization value");
    }

    dispatch_2d(queue_, kernels_->pipeline("lda_transpose_means"), q_, prefix,
        [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:lda_means_.buffer offset:0 atIndex:0];
          [encoder setBuffer:lda_rhs_.buffer offset:0 atIndex:1];
          set_u32(encoder, 2, static_cast<uint32_t>(prefix));
          set_u32(encoder, 3, static_cast<uint32_t>(q_));
          set_u32(encoder, 4, stride(lda_means_));
          set_u32(encoder, 5, stride(lda_rhs_));
        });
    MPSMatrixSolveCholesky* solve = [[MPSMatrixSolveCholesky alloc]
        initWithDevice:device_
                  upper:NO
                  order:prefix
     numberOfRightHandSides:q_];
    id<MTLCommandBuffer> solve_command = [queue_ commandBuffer];
    [solve encodeToCommandBuffer:solve_command
                    sourceMatrix:matrix_view(lda_factor_, prefix, prefix)
             rightHandSideMatrix:matrix_view(lda_rhs_, prefix, q_)
                  solutionMatrix:matrix_view(lda_linear_, prefix, q_)];
    finish_command(solve_command, "resident Metal LDA triangular solve failed");
    dispatch_1d(queue_, kernels_->pipeline("lda_constants"), q_,
        [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:lda_means_.buffer offset:0 atIndex:0];
          [encoder setBuffer:lda_linear_.buffer offset:0 atIndex:1];
          [encoder setBuffer:mean_y_.buffer offset:0 atIndex:2];
          [encoder setBuffer:lda_constants_.buffer offset:0 atIndex:3];
          set_u32(encoder, 4, static_cast<uint32_t>(prefix));
          set_u32(encoder, 5, static_cast<uint32_t>(q_));
          set_u32(encoder, 6, stride(lda_means_));
          set_u32(encoder, 7, stride(lda_linear_));
        });
    lda_prefix_ = prefix;
  }

  void lda_predict(PredictionWorkspace& workspace, int prefix) {
    prepare_lda(prefix);
    MPSMatrixMultiplication* product = [[MPSMatrixMultiplication alloc]
        initWithDevice:device_
         transposeLeft:NO
        transposeRight:NO
            resultRows:workspace.scores.rows
         resultColumns:q_
       interiorColumns:prefix
                  alpha:1.0
                   beta:0.0];
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    [product encodeToCommandBuffer:command
                        leftMatrix:matrix_view(
                            workspace.scores, workspace.scores.rows, prefix)
                       rightMatrix:matrix_view(lda_linear_, prefix, q_)
                       resultMatrix:matrix_view(workspace.lda_prediction)];
    finish_command(command, "resident Metal LDA prediction failed");
    dispatch_2d(queue_, kernels_->pipeline("lda_add_constants"), q_,
        workspace.scores.rows, [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:workspace.lda_prediction.buffer offset:0 atIndex:0];
          [encoder setBuffer:lda_constants_.buffer offset:0 atIndex:1];
          set_u32(encoder, 2,
              static_cast<uint32_t>(workspace.scores.rows));
          set_u32(encoder, 3, static_cast<uint32_t>(q_));
          set_u32(encoder, 4, stride(workspace.lda_prediction));
        });
  }

  void fit() {
    MetalProduct score_product(device_, X_, r_, t_, false, false);
    MetalProduct score_norm_product(device_, t_, t_, score_norm_, true, false);
    MetalProduct predictor_loading(device_, X_, t_, p_vector_, true, false);
    std::unique_ptr<MetalProduct> cached_predictor_loading;
    std::unique_ptr<MetalProduct> cached_score_norm;
    std::unique_ptr<MetalProduct> cached_response_loading;
    if (cached_predictor_crossprod_) {
      cached_predictor_loading.reset(new MetalProduct(
          device_, predictor_gram_, r_, p_vector_, false, false));
      cached_score_norm.reset(new MetalProduct(
          device_, r_, p_vector_, score_norm_, true, false));
      cached_response_loading.reset(new MetalProduct(
          device_, original_crosscov_, r_, q_vector_, true, false));
    }
    std::unique_ptr<MetalProduct> response_loading;
    if (Y_.rows > 0 && !use_sample_response_gram_) {
      response_loading.reset(
          new MetalProduct(device_, Y_, t_, q_vector_, true, false));
    }
    std::unique_ptr<MetalProduct> deflation_row;
    if (!implicit_crosscov_) {
      deflation_row.reset(
          new MetalProduct(device_, v_, S_, row_, true, false));
    }

    auto append_component = [&](id<MTLCommandBuffer> command,
                                int component) {
      if (cached_predictor_crossprod_) {
        cached_predictor_loading->encode(command);
      } else {
        score_product.encode(command);
      }
      if (!plssvd_) {
        if (cached_predictor_crossprod_) {
          cached_score_norm->encode(command);
        } else {
          score_norm_product.encode(command);
        }
        encode_1d(command, kernels_->pipeline("reciprocal_sqrt_scalar"), 1,
            [&](id<MTLComputeCommandEncoder> encoder) {
              [encoder setBuffer:score_norm_.buffer offset:0 atIndex:0];
              [encoder setBuffer:invalid_ offset:0 atIndex:1];
            });
        encode_1d(command, kernels_->pipeline("scale_score_pair"),
            std::max(n_, p_), [&](id<MTLComputeCommandEncoder> encoder) {
              [encoder setBuffer:(cached_predictor_crossprod_ ?
                  p_vector_.buffer : t_.buffer) offset:0 atIndex:0];
              [encoder setBuffer:r_.buffer offset:0 atIndex:1];
              [encoder setBuffer:score_norm_.buffer offset:0 atIndex:2];
              set_u32(encoder, 3, static_cast<uint32_t>(
                  cached_predictor_crossprod_ ? p_ : n_));
              set_u32(encoder, 4, static_cast<uint32_t>(p_));
              set_u32(encoder, 5, stride(t_));
              set_u32(encoder, 6, stride(r_));
            });
      }
      if (!cached_predictor_crossprod_) predictor_loading.encode(command);
      if (cached_predictor_crossprod_) {
        cached_response_loading->encode(command);
      } else if (classification_ && Y_.rows == 0) {
        encode_1d(command, kernels_->pipeline("label_response_loading"), q_,
            [&](id<MTLComputeCommandEncoder> encoder) {
              [encoder setBuffer:t_.buffer offset:0 atIndex:0];
              [encoder setBuffer:labels_ offset:0 atIndex:1];
              [encoder setBuffer:mean_y_.buffer offset:0 atIndex:2];
              [encoder setBuffer:q_vector_.buffer offset:0 atIndex:3];
              set_u32(encoder, 4, static_cast<uint32_t>(n_));
              set_u32(encoder, 5, static_cast<uint32_t>(q_));
              set_u32(encoder, 6, stride(t_));
              set_u32(encoder, 7, stride(q_vector_));
            });
      } else if (!use_sample_response_gram_) {
        response_loading->encode(command);
      }
      if (!plssvd_) {
        encode_copy_plain(command, p_vector_, v_);
        encode_orthonormalize(command, v_, V_, component);
        if (!implicit_crosscov_) {
          deflation_row->encode(command);
          encode_2d(command, kernels_->pipeline("rank1_subtract"), q_, p_,
              [&](id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:S_.buffer offset:0 atIndex:0];
                [encoder setBuffer:v_.buffer offset:0 atIndex:1];
                [encoder setBuffer:row_.buffer offset:0 atIndex:2];
                set_u32(encoder, 3, static_cast<uint32_t>(p_));
                set_u32(encoder, 4, static_cast<uint32_t>(q_));
                set_u32(encoder, 5, stride(S_));
                set_u32(encoder, 6, stride(v_));
              });
          if (right_gram_refresh_) {
            MetalProduct gram_delta(
                device_, row_, row_, right_gram_delta_, true, false);
            gram_delta.encode(command);
            encode_2d(command, kernels_->pipeline("subtract_matrix"), q_, q_,
                [&](id<MTLComputeCommandEncoder> encoder) {
                  [encoder setBuffer:right_gram_.buffer offset:0 atIndex:0];
                  [encoder setBuffer:right_gram_delta_.buffer offset:0 atIndex:1];
                  set_u32(encoder, 2, static_cast<uint32_t>(q_));
                  set_u32(encoder, 3, static_cast<uint32_t>(q_));
                  set_u32(encoder, 4, stride(right_gram_));
                  set_u32(encoder, 5, stride(right_gram_delta_));
                });
          }
        }
      }
      encode_copy_vector(command, r_, R_, component);
      if (!use_sample_response_gram_)
        encode_copy_vector(command, q_vector_, Q_, component);
      encode_copy_vector(command, plssvd_ ? r_ : v_, V_, component);
      encode_copy_vector(command, p_vector_, P_, component);
      if (!cached_predictor_crossprod_)
        encode_copy_vector(command, t_, T_, component);
    };

    int component = 0;
    if (use_sample_response_gram_) {
      clear_invalid();
      id<MTLCommandBuffer> fit_command = [queue_ commandBuffer];
      for (; component < components_; ++component) {
        encode_sample_gram_rank_one_direction(
            fit_command, V_, component,
            seed_ + static_cast<unsigned int>(component));
        append_component(fit_command, component);
      }
      finish_command(
          fit_command, "resident Metal response-wide SIMPLS fit failed");
      require_valid("resident Metal randomized direction");
      require_valid("resident Metal SIMPLS fitting");
    }
    for (; component < components_;) {
      const int block = std::min(refresh_block_, components_ - component);
      clear_invalid();
      id<MTLCommandBuffer> refresh = [queue_ commandBuffer];
      if (block > 1) {
        MetalMatrix right = leading_columns(right_block_, block);
        MetalMatrix candidate = leading_columns(candidate_basis_, block);
        MetalMatrix samples = leading_columns(sample_block_, block);
        MetalMatrix right_scratch = matrix_shape(
            candidate_rotated_, q_, block);
        encode_2d(refresh, kernels_->pipeline("random_matrix"), block, q_,
            [&](id<MTLComputeCommandEncoder> encoder) {
              [encoder setBuffer:right.buffer offset:0 atIndex:0];
              set_u32(encoder, 1, static_cast<uint32_t>(q_));
              set_u32(encoder, 2, static_cast<uint32_t>(block));
              set_u32(encoder, 3,
                  seed_ + static_cast<unsigned int>(component));
              set_u32(encoder, 4, stride(right));
            });
        MetalMatrix active_right = right;
        if (right_gram_refresh_) {
          MetalMatrix scratch_right = right_scratch;
          for (int iteration = 0; iteration < effective_power_; ++iteration) {
            MetalProduct power_step(
                device_, right_gram_, active_right, scratch_right,
                false, false);
            power_step.encode(refresh);
            encode_cholesky_qr(refresh, scratch_right, block, 1);
            std::swap(active_right, scratch_right);
          }
        } else {
          encode_cholesky_qr(refresh, right, block, 1);
        }

        if (implicit_crosscov_) {
          MetalProduct implicit_y_forward(
              device_, Y_, active_right, samples, false, false);
          MetalProduct implicit_x_transpose(
              device_, X_, samples, candidate, true, false);
          implicit_y_forward.encode(refresh);
          implicit_x_transpose.encode(refresh);
        } else {
          MetalProduct forward_first(
              device_, S_, active_right, candidate, false, false);
          forward_first.encode(refresh);
        }
        auto encode_candidate_qr = [&]() {
          if (plssvd_) {
            encode_block_qr(refresh, candidate, R_, block, component);
            return;
          }
          if (component > 0) {
            MetalMatrix basis = matrix_shape(V_, p_, component);
            MetalMatrix projection = matrix_shape(
                block_projection_, component, block);
            MetalMatrix projected = matrix_shape(
                candidate_rotated_, p_, block);
            for (int pass = 0; pass < 2; ++pass) {
              MetalProduct project(
                  device_, basis, candidate, projection, true, false);
              MetalProduct reconstruct(
                  device_, basis, projection, projected, false, false);
              project.encode(refresh);
              reconstruct.encode(refresh);
              encode_2d(refresh, kernels_->pipeline("subtract_matrix"),
                  block, p_, [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:candidate.buffer offset:0 atIndex:0];
                    [encoder setBuffer:projected.buffer offset:0 atIndex:1];
                    set_u32(encoder, 2, static_cast<uint32_t>(p_));
                    set_u32(encoder, 3, static_cast<uint32_t>(block));
                    set_u32(encoder, 4, stride(candidate));
                    set_u32(encoder, 5, stride(projected));
                  });
            }
          }
          encode_cholesky_qr(refresh, candidate, block, 1);
        };
        encode_candidate_qr();

        for (int iteration = 0;
             !right_gram_refresh_ && iteration < effective_power_;
             ++iteration) {
          if (implicit_crosscov_) {
            MetalProduct x_forward(device_, X_, candidate, samples, false, false);
            MetalProduct y_transpose(device_, Y_, samples, right, true, false);
            x_forward.encode(refresh);
            y_transpose.encode(refresh);
          } else {
            MetalProduct transpose(
                device_, S_, candidate, right, true, false);
            transpose.encode(refresh);
          }
          encode_cholesky_qr(refresh, right, block, 1);
          if (implicit_crosscov_) {
            MetalProduct y_forward(device_, Y_, right, samples, false, false);
            MetalProduct x_transpose(
                device_, X_, samples, candidate, true, false);
            y_forward.encode(refresh);
            x_transpose.encode(refresh);
          } else {
            MetalProduct forward(
                device_, S_, right, candidate, false, false);
            forward.encode(refresh);
          }
          encode_candidate_qr();
        }

        if (right_gram_refresh_) {
          MetalMatrix reduced = matrix_shape(reduced_block_, block, q_);
          MetalMatrix gram = matrix_shape(projected_gram_, block, block);
          MetalMatrix vectors = matrix_shape(projected_vectors_, block, block);
          MetalMatrix rotated = matrix_shape(candidate_rotated_, p_, block);
          MetalProduct form_reduced(
              device_, candidate, S_, reduced, true, false);
          MetalProduct form_gram(
              device_, reduced, reduced, gram, false, true);
          MetalProduct rotate(
              device_, candidate, vectors, rotated, false, false);
          form_reduced.encode(refresh);
          form_gram.encode(refresh);
          encode_1d(refresh, kernels_->pipeline("symmetric_eigenvectors"), 1,
              [&](id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:gram.buffer offset:0 atIndex:0];
                [encoder setBuffer:vectors.buffer offset:0 atIndex:1];
                [encoder setBuffer:invalid_ offset:0 atIndex:2];
                set_u32(encoder, 3, static_cast<uint32_t>(block));
                set_u32(encoder, 4, stride(gram));
                set_u32(encoder, 5, stride(vectors));
              });
          rotate.encode(refresh);
          encode_2d(refresh, kernels_->pipeline("copy_prefix_matrix"),
              block, p_, [&](id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:rotated.buffer offset:0 atIndex:0];
                [encoder setBuffer:candidate.buffer offset:0 atIndex:1];
                set_u32(encoder, 2, static_cast<uint32_t>(p_));
                set_u32(encoder, 3, static_cast<uint32_t>(block));
                set_u32(encoder, 4, stride(rotated));
                set_u32(encoder, 5, stride(candidate));
              });
        }

        if (plssvd_) {
          MetalMatrix predictor_loadings = matrix_shape(
              candidate_rotated_, p_, block);
          MetalProduct form_scores(
              device_, X_, candidate, samples, false, false);
          MetalProduct form_predictor_loadings(
              device_, X_, samples, predictor_loadings, true, false);
          form_scores.encode(refresh);
          form_predictor_loadings.encode(refresh);
          if (implicit_crosscov_) {
            MetalProduct form_response_loadings(
                device_, Y_, samples, right, true, false);
            form_response_loadings.encode(refresh);
          } else {
            MetalProduct form_response_loadings(
                device_, S_, candidate, right, true, false);
            form_response_loadings.encode(refresh);
          }
          auto store_plssvd_block = [&](const MetalMatrix& source,
                                        MetalMatrix& destination, int rows) {
            encode_2d(refresh, kernels_->pipeline("copy_block_columns"),
                block, rows, [&](id<MTLComputeCommandEncoder> encoder) {
                  [encoder setBuffer:source.buffer offset:0 atIndex:0];
                  [encoder setBuffer:destination.buffer offset:0 atIndex:1];
                  set_u32(encoder, 2, static_cast<uint32_t>(rows));
                  set_u32(encoder, 3, static_cast<uint32_t>(block));
                  set_u32(encoder, 4, static_cast<uint32_t>(component));
                  set_u32(encoder, 5, stride(source));
                  set_u32(encoder, 6, stride(destination));
                });
          };
          store_plssvd_block(candidate, R_, p_);
          store_plssvd_block(candidate, V_, p_);
          store_plssvd_block(right, Q_, q_);
          store_plssvd_block(predictor_loadings, P_, p_);
          store_plssvd_block(samples, T_, n_);
          finish_command(refresh, "resident Metal blocked PLS-SVD fit failed");
          require_valid("resident Metal blocked PLS-SVD fitting");
          component += block;
          continue;
        }

        if (cached_predictor_crossprod_) {
          MetalMatrix candidate_loadings = matrix_shape(
              candidate_rotated_, p_, block);
          MetalProduct predictor_geometry(
              device_, predictor_gram_, candidate, candidate_loadings,
              false, false);
          predictor_geometry.encode(refresh);
          encode_1d(refresh, kernels_->pipeline("normalize_candidate_geometry"),
              block, [&](id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:candidate.buffer offset:0 atIndex:0];
                [encoder setBuffer:candidate_loadings.buffer offset:0 atIndex:1];
                [encoder setBuffer:invalid_ offset:0 atIndex:2];
                set_u32(encoder, 3, static_cast<uint32_t>(p_));
                set_u32(encoder, 4, static_cast<uint32_t>(block));
                set_u32(encoder, 5, stride(candidate));
                set_u32(encoder, 6, stride(candidate_loadings));
              });
          MetalProduct response_geometry(
              device_, original_crosscov_, candidate, right, true, false);
          response_geometry.encode(refresh);
          MetalMatrix orthogonal_block = matrix_shape(
              sample_block_, p_, block);
          encode_2d(refresh, kernels_->pipeline("copy_prefix_matrix"),
              block, p_, [&](id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:candidate_loadings.buffer offset:0 atIndex:0];
                [encoder setBuffer:orthogonal_block.buffer offset:0 atIndex:1];
                set_u32(encoder, 2, static_cast<uint32_t>(p_));
                set_u32(encoder, 3, static_cast<uint32_t>(block));
                set_u32(encoder, 4, stride(candidate_loadings));
                set_u32(encoder, 5, stride(orthogonal_block));
              });
          if (component > 0) {
            MetalMatrix basis = matrix_shape(V_, p_, component);
            MetalMatrix projection = matrix_shape(
                block_projection_, component, block);
            MetalMatrix projected = matrix_shape(
                deflation_block_, p_, block);
            for (int pass = 0; pass < 2; ++pass) {
              MetalProduct project(
                  device_, basis, orthogonal_block, projection, true, false);
              MetalProduct reconstruct(
                  device_, basis, projection, projected, false, false);
              project.encode(refresh);
              reconstruct.encode(refresh);
              encode_2d(refresh, kernels_->pipeline("subtract_matrix"),
                  block, p_, [&](id<MTLComputeCommandEncoder> encoder) {
                    [encoder setBuffer:orthogonal_block.buffer offset:0 atIndex:0];
                    [encoder setBuffer:projected.buffer offset:0 atIndex:1];
                    set_u32(encoder, 2, static_cast<uint32_t>(p_));
                    set_u32(encoder, 3, static_cast<uint32_t>(block));
                    set_u32(encoder, 4, stride(orthogonal_block));
                    set_u32(encoder, 5, stride(projected));
                  });
            }
          }
          encode_cholesky_qr(refresh, orthogonal_block, block, 1);

          MetalMatrix deflation_rows = matrix_shape(
              reduced_block_, block, q_);
          MetalProduct form_deflation_rows(
              device_, orthogonal_block, S_, deflation_rows, true, false);
          MetalProduct form_deflation(
              device_, orthogonal_block, deflation_rows, deflation_block_,
              false, false);
          form_deflation_rows.encode(refresh);
          form_deflation.encode(refresh);
          encode_2d(refresh, kernels_->pipeline("subtract_matrix"), q_, p_,
              [&](id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:S_.buffer offset:0 atIndex:0];
                [encoder setBuffer:deflation_block_.buffer offset:0 atIndex:1];
                set_u32(encoder, 2, static_cast<uint32_t>(p_));
                set_u32(encoder, 3, static_cast<uint32_t>(q_));
                set_u32(encoder, 4, stride(S_));
                set_u32(encoder, 5, stride(deflation_block_));
              });
          if (right_gram_refresh_) {
            MetalProduct gram_delta(
                device_, deflation_rows, deflation_rows,
                right_gram_delta_, true, false);
            gram_delta.encode(refresh);
            encode_2d(refresh, kernels_->pipeline("subtract_matrix"), q_, q_,
                [&](id<MTLComputeCommandEncoder> encoder) {
                  [encoder setBuffer:right_gram_.buffer offset:0 atIndex:0];
                  [encoder setBuffer:right_gram_delta_.buffer offset:0 atIndex:1];
                  set_u32(encoder, 2, static_cast<uint32_t>(q_));
                  set_u32(encoder, 3, static_cast<uint32_t>(q_));
                  set_u32(encoder, 4, stride(right_gram_));
                  set_u32(encoder, 5, stride(right_gram_delta_));
                });
          }

          auto store_block = [&](const MetalMatrix& source,
                                 MetalMatrix& destination, int rows) {
            encode_2d(refresh, kernels_->pipeline("copy_block_columns"),
                block, rows, [&](id<MTLComputeCommandEncoder> encoder) {
                  [encoder setBuffer:source.buffer offset:0 atIndex:0];
                  [encoder setBuffer:destination.buffer offset:0 atIndex:1];
                  set_u32(encoder, 2, static_cast<uint32_t>(rows));
                  set_u32(encoder, 3, static_cast<uint32_t>(block));
                  set_u32(encoder, 4, static_cast<uint32_t>(component));
                  set_u32(encoder, 5, stride(source));
                  set_u32(encoder, 6, stride(destination));
                });
          };
          store_block(candidate, R_, p_);
          store_block(right, Q_, q_);
          store_block(orthogonal_block, V_, p_);
          store_block(candidate_loadings, P_, p_);
          finish_command(refresh, "resident Metal fused SIMPLS fit failed");
          require_valid("resident Metal fused SIMPLS fitting");
          component += block;
          continue;
        }

        for (int offset = 0; offset < block; ++offset) {
          encode_extract_column(refresh, candidate, r_, offset);
          append_component(refresh, component + offset);
        }
        finish_command(refresh, "resident Metal blocked SIMPLS fit failed");
        require_valid("resident Metal blocked randomized direction");
        require_valid("resident Metal blocked SIMPLS fitting");
        component += block;
        continue;
      }
      solve_rank_one_direction(
          plssvd_ ? R_ : V_, component,
          seed_ + static_cast<unsigned int>(component));
      id<MTLCommandBuffer> append = [queue_ commandBuffer];
      append_component(append, component);
      finish_command(append, "resident Metal SIMPLS component failed");
      require_valid("resident Metal randomized direction");
      require_valid("resident Metal SIMPLS fitting");
      ++component;
    }
    if (cached_predictor_crossprod_) {
      MetalProduct training_scores(device_, X_, R_, T_, false, false);
      run_product(queue_, training_scores);
    }
    if (use_sample_response_gram_) {
      MetalProduct response_loadings(device_, Y_, T_, Q_, true, false);
      run_product(queue_, response_loadings);
    }
  }

  void validate_prefix(int prefix) const {
    if (prefix < 1 || prefix > components_) {
      throw std::invalid_argument("invalid resident Metal prediction request");
    }
  }

  PredictionWorkspace& prediction_scores(const arma::fmat& x) {
    if (x.n_cols != static_cast<arma::uword>(input_p_) || x.n_rows < 1) {
      throw std::invalid_argument("invalid resident Metal prediction request");
    }
    if (!prediction_ || prediction_->input.rows != x.n_rows) {
      prediction_.reset(new PredictionWorkspace(
          device_, R_, static_cast<int>(x.n_rows), input_p_, p_,
          components_, q_));
    }
    upload_preprocessed_matrix(
        x, prediction_->input, mean_x_, scale_x_, 1, false);
    if (opls_) {
      for (int component = 0; component < north_; ++component) {
        id<MTLCommandBuffer> command = [queue_ commandBuffer];
        encode_extract_column(command, opls_weights_, opls_weight_, component);
        encode_extract_column(command, opls_loadings_, opls_loading_, component);
        finish_command(command, "resident Metal OPLS filter load failed");
        MetalProduct score_product(
            device_, prediction_->input, opls_weight_,
            prediction_->filter_score,
            false, false);
        run_product(queue_, score_product);
        dispatch_2d(queue_, kernels_->pipeline("rank1_subtract_columns"),
            input_p_, x.n_rows,
            [&](id<MTLComputeCommandEncoder> encoder) {
              [encoder setBuffer:prediction_->input.buffer offset:0 atIndex:0];
              [encoder setBuffer:prediction_->filter_score.buffer
                          offset:0 atIndex:1];
              [encoder setBuffer:opls_loading_.buffer offset:0 atIndex:2];
              set_u32(encoder, 3, static_cast<uint32_t>(x.n_rows));
              set_u32(encoder, 4, static_cast<uint32_t>(input_p_));
              set_u32(encoder, 5, stride(prediction_->input));
              set_u32(encoder, 6, stride(prediction_->filter_score));
              set_u32(encoder, 7, stride(opls_loading_));
            });
      }
    } else if (kernelpls_) {
      dispatch_1d(queue_, kernels_->pipeline("row_squared_norms"), x.n_rows,
          [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:prediction_->input.buffer offset:0 atIndex:0];
            [encoder setBuffer:prediction_->norms.buffer offset:0 atIndex:1];
            set_u32(encoder, 2, static_cast<uint32_t>(x.n_rows));
            set_u32(encoder, 3, static_cast<uint32_t>(input_p_));
            set_u32(encoder, 4, stride(prediction_->input));
          });
      MetalProduct kernel_product(
          device_, prediction_->input, reference_, prediction_->X,
          false, true);
      run_product(queue_, kernel_product);
      dispatch_2d(queue_, kernels_->pipeline("transform_kernel"), n_, x.n_rows,
          [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:prediction_->X.buffer offset:0 atIndex:0];
            [encoder setBuffer:prediction_->norms.buffer offset:0 atIndex:1];
            [encoder setBuffer:reference_norms_.buffer offset:0 atIndex:2];
            set_u32(encoder, 3, static_cast<uint32_t>(x.n_rows));
            set_u32(encoder, 4, static_cast<uint32_t>(n_));
            set_u32(encoder, 5, stride(prediction_->X));
            set_i32(encoder, 6, kernel_kind_);
            [encoder setBytes:&kernel_gamma_ length:sizeof(kernel_gamma_)
                       atIndex:7];
            set_i32(encoder, 8, kernel_degree_);
            [encoder setBytes:&kernel_coefficient_
                       length:sizeof(kernel_coefficient_) atIndex:9];
          });
      dispatch_1d(queue_, kernels_->pipeline("row_means"), x.n_rows,
          [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:prediction_->X.buffer offset:0 atIndex:0];
            [encoder setBuffer:prediction_->row_means.buffer offset:0 atIndex:1];
            set_u32(encoder, 2, static_cast<uint32_t>(x.n_rows));
            set_u32(encoder, 3, static_cast<uint32_t>(n_));
            set_u32(encoder, 4, stride(prediction_->X));
          });
      dispatch_2d(queue_, kernels_->pipeline("center_test_kernel"), n_,
          x.n_rows, [&](id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:prediction_->X.buffer offset:0 atIndex:0];
            [encoder setBuffer:prediction_->row_means.buffer offset:0 atIndex:1];
            [encoder setBuffer:kernel_training_means_.buffer offset:0 atIndex:2];
            [encoder setBuffer:kernel_grand_.buffer offset:0 atIndex:3];
            set_u32(encoder, 4, static_cast<uint32_t>(x.n_rows));
            set_u32(encoder, 5, static_cast<uint32_t>(n_));
            set_u32(encoder, 6, stride(prediction_->X));
          });
    }
    run_product(queue_, *prediction_->project_product);
    return *prediction_;
  }

  void response_predict(PredictionWorkspace& workspace, int prefix) {
    validate_prefix(prefix);
    if (plssvd_) {
      plssvd_predict(workspace, prefix);
    } else {
      MPSMatrixMultiplication* product = [[MPSMatrixMultiplication alloc]
          initWithDevice:device_
           transposeLeft:NO
          transposeRight:YES
              resultRows:workspace.scores.rows
           resultColumns:q_
         interiorColumns:prefix
                    alpha:1.0
                     beta:0.0];
      id<MTLCommandBuffer> command = [queue_ commandBuffer];
      [product encodeToCommandBuffer:command
                          leftMatrix:matrix_view(
                              workspace.scores, workspace.scores.rows, prefix)
                         rightMatrix:matrix_view(Q_, q_, prefix)
                         resultMatrix:matrix_view(workspace.prediction)];
      finish_command(command, "resident Metal SIMPLS prediction failed");
    }
    dispatch_2d(queue_, kernels_->pipeline("add_response_mean"), q_,
        workspace.scores.rows,
        [&](id<MTLComputeCommandEncoder> encoder) {
          [encoder setBuffer:workspace.prediction.buffer offset:0 atIndex:0];
          [encoder setBuffer:mean_y_.buffer offset:0 atIndex:1];
          set_u32(encoder, 2,
              static_cast<uint32_t>(workspace.scores.rows));
          set_u32(encoder, 3, static_cast<uint32_t>(q_));
          set_u32(encoder, 4, stride(workspace.prediction));
        });
  }

  PredictionWorkspace& prediction_workspace(const arma::fmat& x, int prefix) {
    PredictionWorkspace& workspace = prediction_scores(x);
    response_predict(workspace, prefix);
    return workspace;
  }

  int n_, p_, input_p_, q_, components_;
  bool classification_;
  bool plssvd_;
  bool opls_;
  bool kernelpls_;
  int north_;
  int kernel_kind_;
  int kernel_degree_;
  float kernel_gamma_;
  float kernel_coefficient_;
  bool implicit_crosscov_;
  bool right_gram_refresh_;
  bool cached_predictor_crossprod_ = false;
  bool use_sample_response_gram_ = false;
  int refresh_block_;
  int solver_width_;
  int effective_power_;
  unsigned int seed_;
  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  id<MTLBuffer> labels_ = nil;
  id<MTLBuffer> invalid_ = nil;
  id<MTLBuffer> lda_status_ = nil;
  id<MTLBuffer> plssvd_status_ = nil;
  id<MTLBuffer> top_indices_ = nil;
  id<MTLBuffer> metric_labels_ = nil;
  id<MTLBuffer> block_status_ = nil;
  id<MTLBuffer> upload_buffer_ = nil;
  size_t top_capacity_ = 0;
  size_t metric_rows_ = 0;
  size_t upload_capacity_ = 0;
  std::shared_ptr<ResidentKernels> kernels_;
  MetalMatrix X_, Y_, S_, R_, Q_, V_, P_, T_, predictor_ss_;
  MetalMatrix predictor_gram_, original_crosscov_, deflation_block_;
  MetalMatrix mean_x_, scale_x_, mean_y_, scale_y_;
  MetalMatrix r_, z_, t_, p_vector_, q_vector_, v_, row_;
  MetalMatrix candidate_basis_, candidate_rotated_, right_block_, sample_block_;
  MetalMatrix response_gram_, sample_gram_vector_;
  std::unique_ptr<MetalProduct> rank1_x_forward_, rank1_y_transpose_;
  std::unique_ptr<MetalProduct> rank1_y_forward_, rank1_x_transpose_;
  std::unique_ptr<MetalProduct> rank1_response_gram_;
  MetalMatrix projected_gram_, projected_vectors_;
  MetalMatrix reduced_block_, reduced_vector_;
  MetalMatrix right_gram_, right_gram_delta_;
  MetalMatrix block_projection_;
  MetalMatrix score_norm_, vector_norm_;
  MetalMatrix lda_means_, lda_weighted_, lda_gram_, lda_mean_cross_;
  MetalMatrix lda_factor_, lda_rhs_, lda_linear_, lda_constants_, lda_ridge_;
  MetalMatrix plssvd_gram_, plssvd_cross_, plssvd_factor_, plssvd_rhs_;
  MetalMatrix plssvd_weights_, plssvd_ridge_, plssvd_zero_;
  MetalMatrix metric_observed_, metric_output_;
  MetalMatrix opls_weights_, opls_loadings_, opls_score_, opls_loading_;
  MetalMatrix opls_weight_, opls_ratio_, opls_denominator_;
  MetalMatrix opls_effective_r_, opls_component_row_;
  MetalMatrix reference_, reference_norms_, kernel_training_means_;
  MetalMatrix kernel_grand_;
  int lda_prefix_ = 0;
  bool lda_moments_ready_ = false;
  bool variance_ready_ = false;
  int plssvd_prefix_ = 0;
  std::unique_ptr<PredictionWorkspace> prediction_;
  std::mutex mutex_;
};

}  // namespace

std::unique_ptr<MetalResidentModel> metal_resident_simpls_create(
    const arma::fmat& x,
    const arma::fmat* y,
    const Rcpp::IntegerVector* labels,
    int classes,
    int components,
    int scaling,
    int oversample,
    int power,
    unsigned int seed,
    int method,
    int north,
    int kernel,
    float gamma,
    int degree,
    float coefficient) {
  return std::unique_ptr<MetalResidentModel>(new ResidentMetalSimpls(
      x, y, labels, classes, components, scaling, oversample, power, seed,
      method, north, kernel, gamma, degree, coefficient));
}

}  // namespace fastpls_svd
