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
