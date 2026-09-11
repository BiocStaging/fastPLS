// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore

#include "r_api.h"
#include "accelerator_core_backend.h"
#include "core_cpu_backend.h"
#include "rsvd_audit.h"

#include <R_ext/Error.h>
#include <R_ext/Random.h>
#include <fastpls/core/audited_rsvd.hpp>
#include <fastpls/core/classification.hpp>
#include <fastpls/core/cross_validation.hpp>
#include <fastpls/core/diagnostics.hpp>
#include <fastpls/core/folds.hpp>
#include <fastpls/core/kernels.hpp>
#include <fastpls/core/lda.hpp>
#include <fastpls/core/matrix.hpp>
#include <fastpls/core/operators.hpp>
#include <fastpls/core/opls.hpp>
#include <fastpls/core/plssvd.hpp>
#include <fastpls/core/simpls.hpp>
#include <fastpls/core/statistics.hpp>
#include <fastpls/core/supervised.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace fastpls_svd {
bool has_cuda_backend();
bool has_metal_backend();
}

namespace {

int environment_integer(const char* name, int fallback, int minimum,
                        int maximum) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  char* end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0') return fallback;
  return static_cast<int>(std::max<long>(
    minimum, std::min<long>(maximum, parsed)
  ));
}

float decode_float32(const int bits) {
  static_assert(sizeof(float) == sizeof(std::int32_t),
                "fastPLS requires 32-bit IEEE float storage");
  const std::int32_t encoded = static_cast<std::int32_t>(bits);
  float value = 0.0f;
  std::memcpy(&value, &encoded, sizeof(float));
  return value;
}

int encode_float32(const float value) {
  std::int32_t encoded = 0;
  std::memcpy(&encoded, &value, sizeof(float));
  return static_cast<int>(encoded);
}

fastpls::core::Matrix<float> float_matrix_from_s4_impl(
    SEXP object, const char* name, const bool allow_empty_columns) {
  if (!Rf_isS4(object)) {
    throw std::invalid_argument(std::string(name) + " must be a float32 matrix");
  }
  const SEXP data_symbol = Rf_install("Data");
  if (!R_has_slot(object, data_symbol)) {
    throw std::invalid_argument(
      std::string(name) + " does not contain a float32 Data slot"
    );
  }
  SEXP bits = PROTECT(R_do_slot(object, data_symbol));
  const SEXP dimensions = Rf_getAttrib(bits, R_DimSymbol);
  const int minimum_columns = allow_empty_columns ? 0 : 1;
  if (TYPEOF(bits) != INTSXP || TYPEOF(dimensions) != INTSXP ||
      XLENGTH(dimensions) != 2 || INTEGER(dimensions)[0] < 1 ||
      INTEGER(dimensions)[1] < minimum_columns) {
    UNPROTECT(1);
    throw std::invalid_argument(
      std::string(name) + (allow_empty_columns ?
        " must be a float32 matrix" : " must be a non-empty float32 matrix")
    );
  }
  const std::size_t rows = static_cast<std::size_t>(INTEGER(dimensions)[0]);
  const std::size_t columns = static_cast<std::size_t>(INTEGER(dimensions)[1]);
  fastpls::core::Matrix<float> values(rows, columns);
  const int* source = INTEGER(bits);
  static_assert(sizeof(float) == sizeof(int),
                "float32 bridge requires 32-bit float and int storage");
  std::memcpy(values.data(), source, values.size() * sizeof(float));
  UNPROTECT(1);
  return values;
}

fastpls::core::Matrix<float> float_matrix_from_s4(SEXP object,
                                                  const char* name) {
  return float_matrix_from_s4_impl(object, name, false);
}

fastpls::core::Matrix<float> float_matrix_from_s4_allow_empty(
    SEXP object, const char* name) {
  return float_matrix_from_s4_impl(object, name, true);
}

fastpls::core::Matrix<float> float_matrix_from_bits(SEXP object,
                                                    const char* name) {
  if (!Rf_isMatrix(object) || TYPEOF(object) != INTSXP) {
    throw std::invalid_argument(
      std::string(name) + " must be a float32 bit matrix"
    );
  }
  const SEXP dimensions = Rf_getAttrib(object, R_DimSymbol);
  const std::size_t rows = static_cast<std::size_t>(INTEGER(dimensions)[0]);
  const std::size_t columns = static_cast<std::size_t>(INTEGER(dimensions)[1]);
  if (rows < 1 || columns < 1) {
    throw std::invalid_argument(std::string(name) + " must be non-empty");
  }
  fastpls::core::Matrix<float> values(rows, columns);
  static_assert(sizeof(float) == sizeof(int),
                "float32 bridge requires 32-bit float and int storage");
  std::memcpy(
    values.data(), INTEGER(object), values.size() * sizeof(float)
  );
  return values;
}

SEXP list_element(SEXP object, const char* name) {
  if (TYPEOF(object) != VECSXP) {
    throw std::invalid_argument("LDA model must be a list");
  }
  const SEXP names = Rf_getAttrib(object, R_NamesSymbol);
  if (TYPEOF(names) != STRSXP) return R_NilValue;
  for (R_xlen_t index = 0; index < XLENGTH(object); ++index) {
    if (STRING_ELT(names, index) != NA_STRING &&
        !std::strcmp(CHAR(STRING_ELT(names, index)), name)) {
      return VECTOR_ELT(object, index);
    }
  }
  return R_NilValue;
}

fastpls::core::Matrix<double> numeric_matrix_from_sexp_impl(
    SEXP object, const char* name, const bool allow_empty_columns) {
  if (!Rf_isMatrix(object) ||
      (TYPEOF(object) != REALSXP && TYPEOF(object) != INTSXP)) {
    throw std::invalid_argument(std::string(name) + " must be a numeric matrix");
  }
  const SEXP dimensions = Rf_getAttrib(object, R_DimSymbol);
  const std::size_t rows = static_cast<std::size_t>(INTEGER(dimensions)[0]);
  const std::size_t columns = static_cast<std::size_t>(INTEGER(dimensions)[1]);
  if (rows < 1 || (!allow_empty_columns && columns < 1)) {
    throw std::invalid_argument(
      std::string(name) + (allow_empty_columns ?
        " has invalid dimensions" : " must be non-empty")
    );
  }
  fastpls::core::Matrix<double> values(rows, columns);
  if (TYPEOF(object) == REALSXP) {
    std::copy(REAL(object), REAL(object) + values.size(), values.data());
  } else {
    for (std::size_t index = 0; index < values.size(); ++index) {
      const int value = INTEGER(object)[index];
      if (value == NA_INTEGER) {
        values.data()[index] = NA_REAL;
      } else {
        values.data()[index] = static_cast<double>(value);
      }
    }
  }
  return values;
}

fastpls::core::Matrix<double> numeric_matrix_from_sexp(SEXP object,
                                                       const char* name) {
  return numeric_matrix_from_sexp_impl(object, name, false);
}

fastpls::core::Matrix<double> numeric_matrix_from_sexp_allow_empty(
    SEXP object, const char* name) {
  return numeric_matrix_from_sexp_impl(object, name, true);
}

fastpls::core::ConstMatrixView<double> numeric_matrix_view(
    SEXP object, const char* name) {
  if (!Rf_isMatrix(object) || TYPEOF(object) != REALSXP) {
    throw std::invalid_argument(std::string(name) + " must be a double matrix");
  }
  const SEXP dimensions = Rf_getAttrib(object, R_DimSymbol);
  if (TYPEOF(dimensions) != INTSXP || XLENGTH(dimensions) != 2 ||
      INTEGER(dimensions)[0] < 1 || INTEGER(dimensions)[1] < 1) {
    throw std::invalid_argument(std::string(name) + " must be non-empty");
  }
  const std::size_t rows = static_cast<std::size_t>(INTEGER(dimensions)[0]);
  return fastpls::core::make_const_view(
    REAL(object), rows, static_cast<std::size_t>(INTEGER(dimensions)[1]), rows
  );
}

std::vector<double> numeric_values(SEXP object, const char* name) {
  if (TYPEOF(object) != REALSXP && TYPEOF(object) != INTSXP) {
    throw std::invalid_argument(std::string(name) + " must be numeric");
  }
  std::vector<double> values(static_cast<std::size_t>(XLENGTH(object)));
  if (TYPEOF(object) == REALSXP) {
    std::copy(REAL(object), REAL(object) + values.size(), values.begin());
  } else {
    for (std::size_t index = 0; index < values.size(); ++index) {
      const int value = INTEGER(object)[index];
      values[index] = value == NA_INTEGER ? NA_REAL :
        static_cast<double>(value);
    }
  }
  return values;
}

SEXP float_bits_matrix(const fastpls::core::Matrix<float>& values) {
  SEXP result = Rf_allocMatrix(
    INTSXP, static_cast<int>(values.rows()), static_cast<int>(values.columns())
  );
  static_assert(sizeof(float) == sizeof(int),
                "float32 bridge requires 32-bit float and int storage");
  std::memcpy(INTEGER(result), values.data(), values.size() * sizeof(float));
  return result;
}

SEXP numeric_matrix(const fastpls::core::Matrix<double>& values) {
  SEXP result = Rf_allocMatrix(
    REALSXP, static_cast<int>(values.rows()), static_cast<int>(values.columns())
  );
  std::copy(values.data(), values.data() + values.size(), REAL(result));
  return result;
}

template<class T>
SEXP numeric_vector(const std::vector<T>& values) {
  SEXP result = Rf_allocVector(REALSXP, values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    REAL(result)[index] = static_cast<double>(values[index]);
  }
  return result;
}

SEXP core_matrix(const fastpls::core::Matrix<float>& values) {
  return float_bits_matrix(values);
}

SEXP core_matrix(const fastpls::core::Matrix<double>& values) {
  return numeric_matrix(values);
}

template<class T>
SEXP core_matrix_list(
    const std::vector<fastpls::core::Matrix<T>>& values,
    const int* components) {
  SEXP output = PROTECT(Rf_allocVector(VECSXP, values.size()));
  SEXP names = PROTECT(Rf_allocVector(STRSXP, values.size()));
  for (std::size_t index = 0; index < values.size(); ++index) {
    SET_VECTOR_ELT(output, index, core_matrix(values[index]));
    const std::string name =
      "ncomp=" + std::to_string(components[index]);
    SET_STRING_ELT(names, index, Rf_mkChar(name.c_str()));
  }
  Rf_setAttrib(output, R_NamesSymbol, names);
  UNPROTECT(2);
  return output;
}

template<class T>
SEXP core_matrix_cube(
    const std::vector<fastpls::core::Matrix<T>>& values,
    std::size_t rows, std::size_t columns, bool variable_columns = false,
    bool preserve_float = true) {
  const SEXPTYPE type = std::is_same<T, float>::value && preserve_float ?
    INTSXP : REALSXP;
  SEXP output = PROTECT(Rf_allocVector(
    type, static_cast<R_xlen_t>(rows * columns * values.size())
  ));
  SEXP dimensions = PROTECT(Rf_allocVector(INTSXP, 3));
  INTEGER(dimensions)[0] = static_cast<int>(rows);
  INTEGER(dimensions)[1] = static_cast<int>(columns);
  INTEGER(dimensions)[2] = static_cast<int>(values.size());
  Rf_setAttrib(output, R_DimSymbol, dimensions);
  const std::size_t slice_size = rows * columns;
  const bool complete_slices = !variable_columns &&
    std::all_of(values.begin(), values.end(), [=](const auto& value) {
      return value.rows() == rows && value.columns() == columns;
    });
  if (!complete_slices) {
    if (type == INTSXP) {
      std::fill(INTEGER(output), INTEGER(output) + XLENGTH(output), 0);
    } else {
      std::fill(REAL(output), REAL(output) + XLENGTH(output), 0.0);
    }
  }
  for (std::size_t slice = 0; slice < values.size(); ++slice) {
    if (values[slice].rows() > rows ||
        (variable_columns ? values[slice].columns() > columns :
                            values[slice].columns() != columns)) {
      UNPROTECT(2);
      throw std::invalid_argument(
        "fastPLS core matrix path has inconsistent dimensions"
      );
    }
    if (complete_slices) {
      const std::size_t destination = slice * slice_size;
      if (type == INTSXP) {
        std::memcpy(
          INTEGER(output) + destination, values[slice].data(),
          slice_size * sizeof(float)
        );
      } else if constexpr (std::is_same<T, double>::value) {
        std::copy_n(
          values[slice].data(), slice_size, REAL(output) + destination
        );
      } else {
        std::transform(
          values[slice].data(), values[slice].data() + slice_size,
          REAL(output) + destination,
          [](const T value) { return static_cast<double>(value); }
        );
      }
      continue;
    }
    for (std::size_t column = 0;
         column < values[slice].columns(); ++column) {
      for (std::size_t row = 0; row < values[slice].rows(); ++row) {
        const std::size_t destination = slice * slice_size +
          column * rows + row;
        if (type == INTSXP) {
          INTEGER(output)[destination] = encode_float32(
            static_cast<float>(values[slice](row, column))
          );
        } else {
          REAL(output)[destination] =
            static_cast<double>(values[slice](row, column));
        }
      }
    }
  }
  UNPROTECT(2);
  return output;
}

SEXP simpls_timing(const fastpls::core::SimplsTiming& timing) {
  constexpr int count = 11;
  SEXP output = PROTECT(Rf_allocVector(VECSXP, count));
  const double values[count] = {
    timing.setup, 0.0, 0.0, 0.0,
    timing.direction + timing.component_updates, timing.direction,
    timing.component_updates, 0.0, 0.0, timing.candidate_geometry,
    timing.total
  };
  const char* labels[count] = {
    "preprocess_crosscov_sec", "response_crosscov_sec",
    "crossprod_cache_sec", "right_gram_sec", "estimator_sec",
    "direction_sec", "component_update_sec", "coefficient_path_sec",
    "fitted_values_sec", "model_assembly_sec", "cpp_total_sec"
  };
  for (int index = 0; index < count; ++index) {
    SET_VECTOR_ELT(output, index, Rf_ScalarReal(values[index]));
  }
  SEXP names = PROTECT(Rf_allocVector(STRSXP, count));
  for (int index = 0; index < count; ++index) {
    SET_STRING_ELT(names, index, Rf_mkChar(labels[index]));
  }
  Rf_setAttrib(output, R_NamesSymbol, names);
  UNPROTECT(2);
  return output;
}

fastpls::core::Matrix<float> backend_gemm_f32(
    fastpls::core::ConstMatrixView<float> left,
    fastpls::core::ConstMatrixView<float> right,
    const bool transpose_left,
    const bool transpose_right,
    const int backend) {
  if (backend == 0) {
    const std::size_t rows = transpose_left ? left.columns() : left.rows();
    const std::size_t columns = transpose_right ? right.rows() : right.columns();
    fastpls::core::Matrix<float> result(rows, columns);
    fastpls::runtime::cpu_gemm_f32(
      left, right, transpose_left, transpose_right, result.view()
    );
    return result;
  }
  if (backend == 1) {
    return fastpls_svd::cuda_core_gemm_f32(
      left, right, transpose_left, transpose_right
    );
  }
  if (backend == 2) {
    return fastpls_svd::metal_core_gemm_f32(
      left, right, transpose_left, transpose_right
    );
  }
  throw std::invalid_argument("float32 backend must be 0, 1, or 2");
}

class RoutedLinearAlgebraF32 {
 public:
  explicit RoutedLinearAlgebraF32(
      const int backend, const std::size_t training_rows = 0,
      const std::size_t training_columns = 0,
      const std::size_t response_columns = 0)
      : backend_(backend), training_rows_(training_rows),
        training_columns_(training_columns),
        response_columns_(response_columns) {
    if (backend_ < 0 || backend_ > 2) {
      throw std::invalid_argument("float32 backend must be 0, 1, or 2");
    }
  }

  ~RoutedLinearAlgebraF32() {
    fastpls_svd::metal_sample_gram_workspace_destroy_f32(
      metal_sample_gram_workspace_
    );
  }

  RoutedLinearAlgebraF32(const RoutedLinearAlgebraF32&) = delete;
  RoutedLinearAlgebraF32& operator=(const RoutedLinearAlgebraF32&) = delete;

  void configure_problem(const std::size_t training_rows,
                         const std::size_t training_columns,
                         const std::size_t response_columns) {
    if (training_rows_ == training_rows &&
        training_columns_ == training_columns &&
        response_columns_ == response_columns) {
      return;
    }
    fastpls_svd::metal_sample_gram_workspace_destroy_f32(
      metal_sample_gram_workspace_
    );
    metal_sample_gram_workspace_ = nullptr;
    training_rows_ = training_rows;
    training_columns_ = training_columns;
    response_columns_ = response_columns;
  }

  void gemm(fastpls::core::ConstMatrixView<float> left,
            fastpls::core::ConstMatrixView<float> right,
            bool transpose_left, bool transpose_right,
            fastpls::core::MatrixView<float> output) const {
    const bool training_product = training_rows_ > 0 &&
      left.rows() == training_rows_ &&
      left.columns() == training_columns_;
    const auto is_crosscovariance = [&](const auto& value) {
      return response_columns_ > 0 &&
        value.rows() == training_columns_ &&
        value.columns() == response_columns_;
    };
    const bool crosscovariance_product =
      is_crosscovariance(left) || is_crosscovariance(right);
    const bool response_product = response_columns_ > 0 &&
      left.rows() == training_rows_ &&
      left.columns() == response_columns_;
    const bool vector_product = output.rows() == 1 || output.columns() == 1;
    const std::size_t inner = transpose_left ? left.rows() : left.columns();
    const long double operation_size =
      static_cast<long double>(output.rows()) * output.columns() * inner;
    const bool small_crosscovariance_product =
      crosscovariance_product && operation_size < 1.0e8L;
    const int operation_backend = backend_ == 2 &&
      (vector_product || small_crosscovariance_product ||
       (!training_product && !response_product && !crosscovariance_product)) ?
      0 : backend_;
    if (operation_backend == 2 && fastpls_svd::metal_core_gemm_into_f32(
          left, right, transpose_left, transpose_right, output
        )) {
      return;
    }
    const auto product = backend_gemm_f32(
      left, right, transpose_left, transpose_right, operation_backend
    );
    if (product.rows() != output.rows() ||
        product.columns() != output.columns()) {
      throw std::runtime_error("float32 backend returned an invalid product");
    }
    for (std::size_t column = 0; column < output.columns(); ++column) {
      std::copy_n(
        product.data() + column * product.rows(), product.rows(),
        output.data() + column * output.leading_dimension()
      );
    }
  }

  void gemm_accumulate(
      fastpls::core::ConstMatrixView<float> left,
      fastpls::core::ConstMatrixView<float> right,
      bool transpose_left, bool transpose_right,
      fastpls::core::MatrixView<float> output) const {
    const bool training_product = training_rows_ > 0 &&
      left.rows() == training_rows_ &&
      left.columns() == training_columns_;
    const auto is_crosscovariance = [&](const auto& value) {
      return response_columns_ > 0 &&
        value.rows() == training_columns_ &&
        value.columns() == response_columns_;
    };
    const bool crosscovariance_product =
      is_crosscovariance(left) || is_crosscovariance(right);
    const bool response_product = response_columns_ > 0 &&
      left.rows() == training_rows_ &&
      left.columns() == response_columns_;
    const bool vector_product = output.rows() == 1 || output.columns() == 1;
    const std::size_t inner = transpose_left ? left.rows() : left.columns();
    const long double operation_size =
      static_cast<long double>(output.rows()) * output.columns() * inner;
    const bool small_crosscovariance_product =
      crosscovariance_product && operation_size < 1.0e8L;
    const int operation_backend = backend_ == 2 &&
      (vector_product || small_crosscovariance_product ||
       (!training_product && !response_product && !crosscovariance_product)) ?
      0 : backend_;
    if (operation_backend == 2 &&
        fastpls_svd::metal_core_gemm_accumulate_into_f32(
          left, right, transpose_left, transpose_right, output
        )) {
      return;
    }
    if (operation_backend == 0) {
      fastpls::runtime::cpu_gemm_f32(
        left, right, transpose_left, transpose_right, output, true
      );
      return;
    }
    const auto product = backend_gemm_f32(
      left, right, transpose_left, transpose_right, operation_backend
    );
    if (product.rows() != output.rows() ||
        product.columns() != output.columns()) {
      throw std::runtime_error("float32 backend returned an invalid product");
    }
    for (std::size_t column = 0; column < output.columns(); ++column) {
      for (std::size_t row = 0; row < output.rows(); ++row) {
        output(row, column) += product(row, column);
      }
    }
  }

  bool qr_economy(fastpls::core::ConstMatrixView<float> input,
                  fastpls::core::Matrix<float>& q) const {
    return host_.qr_economy(input, q);
  }

  bool symmetric_eigen(fastpls::core::Matrix<float>& matrix,
                       std::vector<float>& eigenvalues) const {
    return host_.symmetric_eigen(matrix, eigenvalues);
  }

  bool svd_economy(fastpls::core::ConstMatrixView<float> input,
                   bool left_only, fastpls::core::Matrix<float>& u,
                   std::vector<float>& singular_values,
                   fastpls::core::Matrix<float>& vt) const {
    return host_.svd_economy(input, left_only, u, singular_values, vt);
  }

  bool rank1_subtract(
      fastpls::core::MatrixView<float> target,
      fastpls::core::ConstMatrixView<float> column,
      fastpls::core::ConstMatrixView<float> row) const {
    // Sequential rank-one updates are latency-bound on Metal. The core can
    // update the shared host matrix directly without command submission.
    return false;
  }

  bool sample_gram_apply(
      fastpls::core::ConstMatrixView<float> predictors,
      fastpls::core::ConstMatrixView<float> sample_gram,
      fastpls::core::ConstMatrixView<float> direction,
      fastpls::core::MatrixView<float> output) const {
    if (backend_ != 2) return false;
    if (metal_sample_gram_workspace_ == nullptr) {
      metal_sample_gram_workspace_ =
        fastpls_svd::metal_sample_gram_workspace_create_f32(
          predictors, sample_gram
        );
    }
    return fastpls_svd::metal_sample_gram_apply_f32(
      metal_sample_gram_workspace_, direction, output
    );
  }

  bool sample_geometry(
      fastpls::core::ConstMatrixView<float>,
      fastpls::core::ConstMatrixView<float>,
      fastpls::core::MatrixView<float>,
      fastpls::core::MatrixView<float>) const {
    // These vector products are faster on the CPU and avoid a second Metal
    // synchronization for every sequential SIMPLS component.
    return false;
  }

 private:
  int backend_;
  std::size_t training_rows_;
  std::size_t training_columns_;
  std::size_t response_columns_;
  mutable void* metal_sample_gram_workspace_ = nullptr;
  fastpls::runtime::CpuLinearAlgebraF32 host_;
};

template<class T>
fastpls::core::Matrix<T> row_matrix(const std::vector<T>& values) {
  fastpls::core::Matrix<T> result(1, values.size());
  std::copy(values.begin(), values.end(), result.data());
  return result;
}

SEXP float_lda_model(const fastpls::core::LdaModel<float>& model) {
  SEXP output = PROTECT(Rf_allocVector(VECSXP, 8));
  SET_VECTOR_ELT(output, 0, float_bits_matrix(model.means));
  SET_VECTOR_ELT(output, 1, float_bits_matrix(model.linear));
  SET_VECTOR_ELT(output, 2, float_bits_matrix(row_matrix(model.constants)));
  SET_VECTOR_ELT(output, 3, float_bits_matrix(row_matrix(model.priors)));
  SET_VECTOR_ELT(output, 4, Rf_ScalarReal(model.ridge));
  SET_VECTOR_ELT(output, 5, Rf_ScalarReal(model.relative_ridge));
  SET_VECTOR_ELT(output, 6, Rf_mkString("float32"));
  SET_VECTOR_ELT(output, 7, Rf_mkString("cpp_native"));
  SEXP names = PROTECT(Rf_allocVector(STRSXP, 8));
  const char* labels[] = {
    "means", "linear", "constants", "priors", "ridge",
    "ridge_relative", "precision", "backend"
  };
  for (int index = 0; index < 8; ++index) {
    SET_STRING_ELT(names, index, Rf_mkChar(labels[index]));
  }
  Rf_setAttrib(output, R_NamesSymbol, names);
  UNPROTECT(2);
  return output;
}

SEXP float_lda_models(
    const std::vector<fastpls::core::LdaModel<float>>& models,
    const int* components) {
  SEXP output = PROTECT(Rf_allocVector(VECSXP, models.size()));
  SEXP names = PROTECT(Rf_allocVector(STRSXP, models.size()));
  for (std::size_t index = 0; index < models.size(); ++index) {
    SET_VECTOR_ELT(output, index, float_lda_model(models[index]));
    SET_STRING_ELT(
      names, index, Rf_mkChar(std::to_string(components[index]).c_str())
    );
  }
  Rf_setAttrib(output, R_NamesSymbol, names);
  UNPROTECT(2);
  return output;
}

SEXP double_row_matrix(const std::vector<double>& values) {
  SEXP output = Rf_allocMatrix(REALSXP, 1, static_cast<int>(values.size()));
  std::copy(values.begin(), values.end(), REAL(output));
  return output;
}

SEXP double_column_matrix(const std::vector<double>& values) {
  SEXP output = Rf_allocMatrix(REALSXP, static_cast<int>(values.size()), 1);
  std::copy(values.begin(), values.end(), REAL(output));
  return output;
}

SEXP double_lda_model(const fastpls::core::LdaModel<double>& model,
                      const char* backend = nullptr) {
  const int field_count = backend == nullptr ? 7 : 8;
  SEXP output = PROTECT(Rf_allocVector(VECSXP, field_count));
  SET_VECTOR_ELT(output, 0, numeric_matrix(model.means));
  SET_VECTOR_ELT(output, 1, Rf_allocMatrix(REALSXP, 0, 0));
  SET_VECTOR_ELT(output, 2, numeric_matrix(model.linear));
  SET_VECTOR_ELT(output, 3, double_row_matrix(model.constants));
  SET_VECTOR_ELT(output, 4, double_column_matrix(model.priors));
  SET_VECTOR_ELT(output, 5, Rf_ScalarReal(model.ridge));
  SET_VECTOR_ELT(output, 6, Rf_ScalarReal(model.relative_ridge));
  if (backend != nullptr) SET_VECTOR_ELT(output, 7, Rf_mkString(backend));
  SEXP names = PROTECT(Rf_allocVector(STRSXP, field_count));
  const char* labels[] = {
    "means", "inv_cov", "linear", "constants", "priors", "ridge",
    "ridge_relative", "backend"
  };
  for (int index = 0; index < field_count; ++index) {
    SET_STRING_ELT(names, index, Rf_mkChar(labels[index]));
  }
  Rf_setAttrib(output, R_NamesSymbol, names);
  UNPROTECT(2);
  return output;
}

SEXP double_lda_models(
    const std::vector<fastpls::core::LdaModel<double>>& models,
    const int* components, const char* backend = nullptr) {
  SEXP output = PROTECT(Rf_allocVector(VECSXP, models.size()));
  SEXP names = PROTECT(Rf_allocVector(STRSXP, models.size()));
  for (std::size_t index = 0; index < models.size(); ++index) {
    SET_VECTOR_ELT(output, index, double_lda_model(models[index], backend));
    SET_STRING_ELT(
      names, index, Rf_mkChar(std::to_string(components[index]).c_str())
    );
  }
  Rf_setAttrib(output, R_NamesSymbol, names);
  UNPROTECT(2);
  return output;
}

std::vector<fastpls::core::LdaModel<double>> train_double_lda(
    fastpls::core::ConstMatrixView<double> scores, const int* labels,
    std::size_t label_count, std::size_t class_count,
    const int* components, std::size_t component_count) {
  if (scores.empty() || labels == nullptr || scores.rows() != label_count ||
      class_count < 2 || components == nullptr || component_count < 1) {
    throw std::invalid_argument("fastPLS LDA training dimensions are invalid");
  }
  std::size_t maximum = 0;
  for (std::size_t index = 0; index < component_count; ++index) {
    if (components[index] < 1 ||
        static_cast<std::size_t>(components[index]) > scores.columns()) {
      throw std::invalid_argument(
        "fastPLS LDA component counts must be within the score dimension"
      );
    }
    maximum = std::max(maximum, static_cast<std::size_t>(components[index]));
  }

  std::vector<double> counts(class_count, 0.0);
  fastpls::core::Matrix<double> class_sums(class_count, maximum);
  for (std::size_t sample = 0; sample < scores.rows(); ++sample) {
    const int encoded = labels[sample] - 1;
    if (encoded < 0 || static_cast<std::size_t>(encoded) >= class_count) {
      throw std::invalid_argument(
        "fastPLS LDA labels must be encoded as 1..n_classes"
      );
    }
    const std::size_t class_index = static_cast<std::size_t>(encoded);
    counts[class_index] += 1.0;
    for (std::size_t component = 0; component < maximum; ++component) {
      class_sums(class_index, component) += scores(sample, component);
    }
  }
  const auto retained_scores = fastpls::core::make_const_view(
    scores.data(), scores.rows(), maximum, scores.leading_dimension()
  );
  fastpls::core::Matrix<double> gram(maximum, maximum);
  fastpls::runtime::cpu_gemm_f64(
    retained_scores, retained_scores, true, false, gram.view()
  );
  return fastpls::core::train_lda_prefixes_from_moments<double>(
    gram.view(), class_sums.view(), counts.data(), counts.size(),
    scores.rows(), components, component_count
  );
}

fastpls::core::LdaModel<double> double_lda_model_from_sexp(SEXP object) {
  fastpls::core::LdaModel<double> model;
  model.linear = numeric_matrix_from_sexp(
    list_element(object, "linear"), "lda$linear"
  );
  model.constants = numeric_values(
    list_element(object, "constants"), "lda$constants"
  );
  return model;
}

SEXP integer_predictions(const std::vector<int>& predictions) {
  SEXP output = Rf_allocVector(INTSXP, predictions.size());
  std::copy(predictions.begin(), predictions.end(), INTEGER(output));
  return output;
}

class ProtectStack {
 public:
  SEXP add(SEXP object) {
    PROTECT(object);
    ++count_;
    return object;
  }

  ~ProtectStack() { UNPROTECT(count_); }

 private:
  int count_ = 0;
};

template<class T, class MatrixSerializer>
SEXP serialize_audited_rsvd(
    const fastpls::core::AuditedSingularTriplets<T>& result,
    MatrixSerializer&& serialize_matrix) {
  ProtectStack protect;
  SEXP output = protect.add(Rf_allocVector(VECSXP, 15));
  SEXP names = protect.add(Rf_allocVector(STRSXP, 15));
  const char* labels[] = {
    "U", "s", "Vt", "randomized", "case_audited", "case_certified",
    "deterministic_fallback", "audit_attempts", "effective_oversample",
    "effective_power", "effective_seed", "audit_subspace_error",
    "audit_singular_value_error", "audit_triplet_residual",
    "audit_omitted_direction_ratio"
  };
  for (int index = 0; index < 15; ++index) {
    SET_STRING_ELT(names, index, Rf_mkChar(labels[index]));
  }
  SET_VECTOR_ELT(output, 0, serialize_matrix(result.decomposition.U));
  SET_VECTOR_ELT(
    output, 1, numeric_vector(result.decomposition.singular_values)
  );
  SET_VECTOR_ELT(
    output, 2, result.decomposition.Vt.rows() == 0 ? R_NilValue :
      serialize_matrix(result.decomposition.Vt)
  );
  SET_VECTOR_ELT(output, 3, Rf_ScalarLogical(TRUE));
  SET_VECTOR_ELT(output, 4, Rf_ScalarLogical(TRUE));
  SET_VECTOR_ELT(
    output, 5, Rf_ScalarLogical(result.audit.certified ? TRUE : FALSE)
  );
  SET_VECTOR_ELT(
    output, 6,
    Rf_ScalarLogical(result.audit.deterministic_fallback ? TRUE : FALSE)
  );
  SET_VECTOR_ELT(output, 7, Rf_ScalarInteger(result.audit.attempts));
  SET_VECTOR_ELT(
    output, 8, Rf_ScalarInteger(result.audit.effective_oversample)
  );
  SET_VECTOR_ELT(
    output, 9, Rf_ScalarInteger(result.audit.effective_power)
  );
  SET_VECTOR_ELT(output, 10, Rf_ScalarReal(result.audit.effective_seed));
  SET_VECTOR_ELT(
    output, 11, Rf_ScalarReal(result.audit.subspace_error)
  );
  SET_VECTOR_ELT(
    output, 12, Rf_ScalarReal(result.audit.singular_value_error)
  );
  SET_VECTOR_ELT(
    output, 13, Rf_ScalarReal(result.audit.triplet_residual)
  );
  SET_VECTOR_ELT(
    output, 14, Rf_ScalarReal(result.audit.omitted_direction_ratio)
  );
  Rf_setAttrib(output, R_NamesSymbol, names);
  return output;
}

template<class T, class MatrixSerializer>
SEXP serialize_opls_filter(const fastpls::core::OplsFilter<T>& filter,
                           MatrixSerializer&& serialize_matrix) {
  ProtectStack protect;
  SEXP output = protect.add(Rf_allocVector(VECSXP, 6));
  SEXP names = protect.add(Rf_allocVector(STRSXP, 6));
  const char* labels[] = {
    "X", "mX", "vX", "W_orth", "P_orth", "north"
  };
  for (int index = 0; index < 6; ++index) {
    SET_STRING_ELT(names, index, Rf_mkChar(labels[index]));
  }
  const fastpls::core::Matrix<T> center = row_matrix(filter.predictor_center);
  const fastpls::core::Matrix<T> scale = row_matrix(filter.predictor_scale);
  SET_VECTOR_ELT(output, 0, serialize_matrix(filter.predictors));
  SET_VECTOR_ELT(output, 1, serialize_matrix(center));
  SET_VECTOR_ELT(output, 2, serialize_matrix(scale));
  SET_VECTOR_ELT(output, 3, serialize_matrix(filter.weights));
  SET_VECTOR_ELT(output, 4, serialize_matrix(filter.loadings));
  SET_VECTOR_ELT(
    output, 5, Rf_ScalarInteger(static_cast<int>(filter.completed_components))
  );
  Rf_setAttrib(output, R_NamesSymbol, names);
  return output;
}

template<class Callable>
SEXP translate_exceptions(const char* context, Callable&& callable) {
  try {
    return callable();
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  } catch (...) {
    Rf_error("Unknown error in %s", context);
  }
  return R_NilValue;
}

std::vector<std::size_t> encoded_class_labels(
    SEXP labels, std::size_t sample_count, int class_count,
    const char* context) {
  if (TYPEOF(labels) != INTSXP ||
      XLENGTH(labels) != static_cast<R_xlen_t>(sample_count) ||
      class_count < 2) {
    throw std::invalid_argument(
      std::string(context) + " requires one integer label per row"
    );
  }
  std::vector<std::size_t> encoded(sample_count);
  for (std::size_t row = 0; row < sample_count; ++row) {
    const int value = INTEGER(labels)[row] - 1;
    if (value < 0 || value >= class_count) {
      throw std::invalid_argument(
        std::string(context) + " labels must be encoded as 1..n_classes"
      );
    }
    encoded[row] = static_cast<std::size_t>(value);
  }
  return encoded;
}

template<class T, class Prepared, class Backend, class Metric>
SEXP serialize_plssvd_core_model(
    const fastpls::core::PlssvdModel<T>& model,
    fastpls::core::ConstMatrixView<T> predictors,
    const Prepared& prepared, SEXP effective_components, bool fitted,
    bool store_scores,
    const char* xprod_mode, Backend& backend,
    Metric metric, bool array_paths = false) {
  ProtectStack protect;
  std::vector<fastpls::core::Matrix<T>> fitted_values;
  std::vector<double> r2_values(
    static_cast<std::size_t>(XLENGTH(effective_components)), NA_REAL
  );
  if (fitted) {
    fitted_values.reserve(r2_values.size());
    for (std::size_t index = 0; index < r2_values.size(); ++index) {
      const std::size_t count = static_cast<std::size_t>(
        INTEGER(effective_components)[index]
      );
      const auto scores = fastpls::core::make_const_view(
        model.scores.data(), model.scores.rows(), count, model.scores.rows()
      );
      fastpls::core::Matrix<T> values(
        predictors.rows(), prepared.response_mean.size()
      );
      backend.gemm(
        scores, model.prediction_weights[index].view(), false, false,
        values.view()
      );
      r2_values[index] = metric(
        fastpls::core::ConstMatrixView<T>(values.view())
      );
      for (std::size_t response = 0;
           response < prepared.response_mean.size(); ++response) {
        for (std::size_t row = 0; row < values.rows(); ++row) {
          values(row, response) += prepared.response_mean[response];
        }
      }
      fitted_values.push_back(std::move(values));
    }
  }

  constexpr int field_count = 16;
  SEXP output = protect.add(Rf_allocVector(VECSXP, field_count));
  SEXP names = protect.add(Rf_allocVector(STRSXP, field_count));
  const char* field_names[field_count] = {
    "P", "R", "Q", "Ttrain", "W_latent", "mX", "vX", "mY",
    "p", "m", "ncomp", "Yfit", "R2Y", "pls_method", "xprod_mode",
    "C_latent"
  };
  for (int index = 0; index < field_count; ++index) {
    SET_STRING_ELT(names, index, Rf_mkChar(field_names[index]));
  }
  SET_VECTOR_ELT(output, 0, R_NilValue);
  SET_VECTOR_ELT(output, 1, core_matrix(model.weights));
  SET_VECTOR_ELT(output, 2, core_matrix(model.response_loadings));
  SET_VECTOR_ELT(
    output, 3, store_scores ? core_matrix(model.scores) : R_NilValue
  );
  SET_VECTOR_ELT(
    output, 4, array_paths ? core_matrix_cube(
      model.prediction_weights, model.weights.columns(),
      prepared.response_mean.size()
    ) : core_matrix_list(
      model.prediction_weights, INTEGER(effective_components)
    )
  );
  SET_VECTOR_ELT(output, 5, core_matrix(row_matrix(
    prepared.predictor_center
  )));
  SET_VECTOR_ELT(output, 6, core_matrix(row_matrix(
    prepared.predictor_scale
  )));
  SET_VECTOR_ELT(output, 7, core_matrix(row_matrix(
    prepared.response_mean
  )));
  SET_VECTOR_ELT(output, 8, Rf_ScalarInteger(
    static_cast<int>(predictors.columns())
  ));
  SET_VECTOR_ELT(output, 9, Rf_ScalarInteger(
    static_cast<int>(prepared.response_mean.size())
  ));
  SET_VECTOR_ELT(output, 10, effective_components);
  SET_VECTOR_ELT(
    output, 11, fitted ? (array_paths ? core_matrix_cube(
      fitted_values, predictors.rows(), prepared.response_mean.size()
    ) : core_matrix_list(
      fitted_values, INTEGER(effective_components)
    )) : R_NilValue
  );
  SEXP r2 = protect.add(Rf_allocVector(
    REALSXP, XLENGTH(effective_components)
  ));
  std::copy(r2_values.begin(), r2_values.end(), REAL(r2));
  SET_VECTOR_ELT(output, 12, r2);
  SET_VECTOR_ELT(output, 13, Rf_mkString("plssvd"));
  SET_VECTOR_ELT(output, 14, Rf_mkString(xprod_mode));
  SET_VECTOR_ELT(
    output, 15, array_paths ? core_matrix_cube(
      model.latent_coefficients, model.weights.columns(),
      model.weights.columns(), true
    ) : core_matrix_list(
      model.latent_coefficients, INTEGER(effective_components)
    )
  );
  Rf_setAttrib(output, R_NamesSymbol, names);
  return output;
}

SEXP capped_plssvd_components(
    SEXP components, std::size_t samples, std::size_t predictors,
    std::size_t response_rank_cap, ProtectStack& protect) {
  SEXP effective = protect.add(Rf_duplicate(components));
  const std::size_t sample_rank = std::max<std::size_t>(samples - 1, 1);
  const int rank_cap = static_cast<int>(std::min({
    predictors, sample_rank, response_rank_cap
  }));
  for (R_xlen_t index = 0; index < XLENGTH(effective); ++index) {
    const int value = INTEGER(effective)[index];
    if (value == NA_INTEGER) {
      throw std::invalid_argument("ncomp cannot contain missing values");
    }
    INTEGER(effective)[index] = std::max(1, std::min(value, rank_cap));
  }
  return effective;
}

template<class T, class Prepared, class Backend, class Metric>
SEXP fit_plssvd_core_prepared(
    fastpls::core::ConstMatrixView<T> predictors,
    const Prepared& prepared, std::size_t response_rank_cap,
    SEXP components, bool fitted, bool store_scores, int oversample, int power,
    unsigned int seed, const char* xprod_mode, Backend& backend,
    Metric metric, bool array_paths = false) {
  ProtectStack protect;
  SEXP effective = capped_plssvd_components(
    components, predictors.rows(), predictors.columns(), response_rank_cap,
    protect
  );
  fastpls::core::PlssvdControls controls;
  controls.rsvd.oversample = oversample;
  controls.rsvd.power = power;
  controls.rsvd.seed = seed;
  const auto model = fastpls::core::fit_plssvd_preprocessed<T>(
    predictors, prepared.crossprod.view(), INTEGER(effective),
    static_cast<std::size_t>(XLENGTH(effective)), controls, backend
  );
  return serialize_plssvd_core_model(
    model, predictors, prepared, effective, fitted, store_scores, xprod_mode,
    backend,
    metric, array_paths
  );
}

template<class T, class Backend>
SEXP fit_plssvd_label_core_prepared(
    fastpls::core::ConstMatrixView<T> predictors,
    const fastpls::core::LabelCrossprodResult<T>& prepared,
    const std::vector<std::size_t>& labels, int class_count,
    SEXP components, bool fitted, bool store_scores, int oversample, int power,
    unsigned int seed, const char* xprod_mode, Backend& backend) {
  return fit_plssvd_core_prepared(
    predictors, prepared, static_cast<std::size_t>(class_count - 1),
    components, fitted, store_scores, oversample, power, seed, xprod_mode,
    backend,
    [&](fastpls::core::ConstMatrixView<T> values) {
      return fastpls::core::dummy_response_r2(
        labels.data(), labels.size(), prepared.response_mean.data(),
        static_cast<std::size_t>(class_count), values
      );
    }
  );
}

template<class T, class Backend>
SEXP fit_plssvd_label_core(
    fastpls::core::Matrix<T>& predictors,
    const std::vector<std::size_t>& labels, int class_count,
    SEXP components, int scaling, bool fitted, bool store_scores,
    int oversample, int power,
    unsigned int seed, const char* xprod_mode, Backend& backend) {
  const auto prepared = fastpls::core::prepare_scaled_label_crossprod(
    predictors.view(), labels.data(), labels.size(),
    static_cast<std::size_t>(class_count),
    static_cast<fastpls::core::PredictorScaling>(scaling), backend
  );
  return fit_plssvd_label_core_prepared(
    fastpls::core::ConstMatrixView<T>(predictors.view()), prepared, labels,
    class_count, components, fitted, store_scores, oversample, power, seed,
    xprod_mode, backend
  );
}

fastpls::core::SimplsControls simpls_controls(
    std::size_t samples, std::size_t predictors, std::size_t responses,
    std::size_t components, bool classification,
    int oversample, int power, unsigned int seed) {
  fastpls::core::SimplsControls controls;
  controls.components = components;
  controls.maximum_block = fastpls::core::simpls_candidate_block_size(
    components, predictors, responses, classification, samples, 64
  );
  const int maximum_predictors = environment_integer(
    "FASTPLS_FAST_CROSSPROD_MAX_P",
#if defined(FASTPLS_USE_ACCELERATE)
    2048,
#else
    512,
#endif
    16, 65536
  );
  const int minimum_components = environment_integer(
    "FASTPLS_FAST_CROSSPROD_MIN_NCOMP", 20, 1, 1024
  );
  const int minimum_ratio = environment_integer(
    "FASTPLS_FAST_CROSSPROD_MIN_N_TO_P_RATIO", 8, 1, 1024
  );
  controls.cache_predictor_crossprod =
    components >= static_cast<std::size_t>(minimum_components) &&
    predictors <= samples &&
    samples >= predictors * static_cast<std::size_t>(minimum_ratio) &&
    predictors <= static_cast<std::size_t>(maximum_predictors);
  controls.batch_candidate_geometry =
    controls.maximum_block > 1 && !controls.cache_predictor_crossprod;
  controls.reorthogonalize = false;
  controls.store_scores = true;
  controls.use_right_gram = true;
  controls.phase_timing = environment_integer(
    "FASTPLS_BENCH_PHASE_TIMING", 0, 0, 1
  ) == 1;
  controls.rsvd.oversample = oversample;
  controls.rsvd.power = power;
  controls.rsvd.seed = seed;
  return controls;
}

SEXP capped_simpls_components(
    SEXP components, std::size_t samples, std::size_t predictors,
    int& maximum_components, ProtectStack& protect) {
  SEXP effective = protect.add(Rf_duplicate(components));
  const std::size_t sample_rank = std::max<std::size_t>(samples - 1, 1);
  const int rank_cap = static_cast<int>(std::min(predictors, sample_rank));
  maximum_components = 1;
  for (R_xlen_t index = 0; index < XLENGTH(effective); ++index) {
    const int value = INTEGER(effective)[index];
    if (value == NA_INTEGER) {
      throw std::invalid_argument("ncomp cannot contain missing values");
    }
    INTEGER(effective)[index] = std::max(1, std::min(value, rank_cap));
    maximum_components = std::max(
      maximum_components, INTEGER(effective)[index]
    );
  }
  return effective;
}

template<class T, class Prepared, class Backend, class Metric>
SEXP serialize_simpls_core_model(
    const fastpls::core::SimplsModel<T>& model,
    fastpls::core::ConstMatrixView<T> predictors,
    const Prepared& prepared, SEXP effective_components, bool fitted,
    bool store_scores,
    const char* xprod_mode, Backend& backend, Metric metric,
    const fastpls::core::SimplsControls& controls,
    bool array_paths = false) {
  ProtectStack protect;
  std::vector<fastpls::core::Matrix<T>> fitted_values;
  std::vector<double> r2_values(
    static_cast<std::size_t>(XLENGTH(effective_components)), NA_REAL
  );
  if (fitted) {
    fitted_values.reserve(r2_values.size());
    for (std::size_t index = 0; index < r2_values.size(); ++index) {
      const std::size_t count = static_cast<std::size_t>(
        INTEGER(effective_components)[index]
      );
      const auto scores = fastpls::core::make_const_view(
        model.scores.data(), model.scores.rows(), count, model.scores.rows()
      );
      const auto loadings = fastpls::core::make_const_view(
        model.response_loadings.data(), model.response_loadings.rows(), count,
        model.response_loadings.rows()
      );
      fastpls::core::Matrix<T> values(
        predictors.rows(), prepared.response_mean.size()
      );
      backend.gemm(scores, loadings, false, true, values.view());
      r2_values[index] = metric(
        fastpls::core::ConstMatrixView<T>(values.view())
      );
      for (std::size_t response = 0;
           response < prepared.response_mean.size(); ++response) {
        for (std::size_t row = 0; row < values.rows(); ++row) {
          values(row, response) += prepared.response_mean[response];
        }
      }
      fitted_values.push_back(std::move(values));
    }
  }

  const int field_count = controls.phase_timing ? 15 : 14;
  SEXP output = protect.add(Rf_allocVector(VECSXP, field_count));
  SEXP names = protect.add(Rf_allocVector(STRSXP, field_count));
  const char* field_names[15] = {
    "P", "R", "Q", "Ttrain", "mX", "vX", "mY", "p", "m",
    "ncomp", "Yfit", "R2Y", "pls_method", "xprod_mode",
    "benchmark_phase_timing"
  };
  for (int index = 0; index < field_count; ++index) {
    SET_STRING_ELT(names, index, Rf_mkChar(field_names[index]));
  }
  SET_VECTOR_ELT(output, 0, R_NilValue);
  SET_VECTOR_ELT(output, 1, core_matrix(model.weights));
  SET_VECTOR_ELT(output, 2, core_matrix(model.response_loadings));
  SET_VECTOR_ELT(
    output, 3, store_scores ? core_matrix(model.scores) : R_NilValue
  );
  SET_VECTOR_ELT(output, 4, core_matrix(row_matrix(
    prepared.predictor_center
  )));
  SET_VECTOR_ELT(output, 5, core_matrix(row_matrix(
    prepared.predictor_scale
  )));
  SET_VECTOR_ELT(output, 6, core_matrix(row_matrix(
    prepared.response_mean
  )));
  SET_VECTOR_ELT(output, 7, Rf_ScalarInteger(
    static_cast<int>(predictors.columns())
  ));
  SET_VECTOR_ELT(output, 8, Rf_ScalarInteger(
    static_cast<int>(prepared.response_mean.size())
  ));
  SET_VECTOR_ELT(output, 9, effective_components);
  SET_VECTOR_ELT(
    output, 10, fitted ? (array_paths ? core_matrix_cube(
      fitted_values, predictors.rows(), prepared.response_mean.size()
    ) : core_matrix_list(
      fitted_values, INTEGER(effective_components)
    )) : R_NilValue
  );
  SEXP r2 = protect.add(Rf_allocVector(
    REALSXP, XLENGTH(effective_components)
  ));
  std::copy(r2_values.begin(), r2_values.end(), REAL(r2));
  SET_VECTOR_ELT(output, 11, r2);
  SET_VECTOR_ELT(output, 12, Rf_mkString("simpls"));
  SET_VECTOR_ELT(output, 13, Rf_mkString(xprod_mode));
  if (controls.phase_timing) {
    SET_VECTOR_ELT(output, 14, simpls_timing(model.timing));
  }
  Rf_setAttrib(output, R_NamesSymbol, names);
  return output;
}

template<class T, class Prepared, class Backend, class Metric>
SEXP fit_simpls_core_prepared(
    fastpls::core::ConstMatrixView<T> predictors,
    const Prepared& prepared,
    SEXP components, bool fitted, bool store_scores,
    int oversample, int power,
    unsigned int seed, const char* xprod_mode, Backend& backend,
    Metric metric, bool array_paths = false,
    bool reorthogonalize_scores = false,
    bool classification = false) {
  ProtectStack protect;
  int maximum_components = 1;
  SEXP effective_components = capped_simpls_components(
    components, predictors.rows(), predictors.columns(), maximum_components,
    protect
  );

  auto controls = simpls_controls(
    predictors.rows(), predictors.columns(),
    prepared.response_mean.size(),
    static_cast<std::size_t>(maximum_components), classification,
    oversample, power, seed
  );
  controls.reorthogonalize = reorthogonalize_scores;
  controls.store_scores = fitted || store_scores;
  fastpls::core::SimplsWorkspace<T> workspace;
  const auto model = fastpls::core::fit_simpls_preprocessed<T>(
    predictors, prepared.crossprod.view(), controls, backend, workspace
  );
  if (model.completed_components < controls.components) {
    throw std::runtime_error(
      "fastPLS core SIMPLS returned fewer components than requested"
    );
  }
  return serialize_simpls_core_model(
    model, predictors, prepared, effective_components, fitted,
    controls.store_scores, xprod_mode, backend, metric, controls, array_paths
  );
}

template<class T, class Backend>
SEXP fit_simpls_label_core_prepared(
    fastpls::core::ConstMatrixView<T> predictors,
    const fastpls::core::LabelCrossprodResult<T>& prepared,
    const std::vector<std::size_t>& labels, int class_count,
    SEXP components, bool fitted, bool store_scores,
    int oversample, int power,
    unsigned int seed, const char* xprod_mode, Backend& backend) {
  return fit_simpls_core_prepared(
    predictors, prepared, components, fitted, store_scores,
    oversample, power, seed,
    xprod_mode, backend,
    [&](fastpls::core::ConstMatrixView<T> values) {
      return fastpls::core::dummy_response_r2(
        labels.data(), labels.size(), prepared.response_mean.data(),
        static_cast<std::size_t>(class_count), values
      );
    }, false, false, true
  );
}

template<class T, class Backend>
SEXP fit_simpls_label_core(
    fastpls::core::Matrix<T>& predictors,
    const std::vector<std::size_t>& labels, int class_count,
    SEXP components, int scaling, bool fitted, bool store_scores,
    int oversample, int power,
    unsigned int seed, const char* xprod_mode, Backend& backend) {
  const auto prepared = fastpls::core::prepare_scaled_label_crossprod(
    predictors.view(), labels.data(), labels.size(),
    static_cast<std::size_t>(class_count),
    static_cast<fastpls::core::PredictorScaling>(scaling), backend
  );
  return fit_simpls_label_core_prepared(
    fastpls::core::ConstMatrixView<T>(predictors.view()), prepared, labels,
    class_count, components, fitted, store_scores, oversample, power, seed,
    xprod_mode, backend
  );
}

template<class T, class Backend>
SEXP fit_dense_core_prepared(
    fastpls::core::ConstMatrixView<T> predictors,
    fastpls::core::ConstMatrixView<T> responses,
    const fastpls::core::DensePreprocessingResult<T>& prepared,
    SEXP components, bool fitted, bool store_scores,
    int method, int oversample, int power,
    unsigned int seed, const char* xprod_mode, Backend& backend,
    bool array_paths) {
  const auto metric = [&](fastpls::core::ConstMatrixView<T> values) {
    return fastpls::core::dense_response_r2(
      responses, prepared.response_mean.data(),
      prepared.response_mean.size(), values
    );
  };
  if (method == 1) {
    return fit_plssvd_core_prepared(
      predictors, prepared, prepared.response_mean.size(), components,
      fitted, store_scores, oversample, power, seed, xprod_mode, backend,
      metric, array_paths
    );
  }
  if (method == 3) {
    return fit_simpls_core_prepared(
      predictors, prepared, components, fitted, store_scores,
      oversample, power, seed, xprod_mode, backend, metric, array_paths,
      std::is_same<T, float>::value
    );
  }
  throw std::invalid_argument(
    "fastPLS dense core method must be PLS-SVD or SIMPLS"
  );
}

template<class T, class Backend>
SEXP fit_dense_plssvd_operator(
    fastpls::core::ConstMatrixView<T> predictors,
    fastpls::core::ConstMatrixView<T> responses,
    const fastpls::core::DensePreprocessingResult<T>& prepared,
    SEXP components, bool fitted, bool store_scores, int oversample, int power,
    unsigned int seed, const char* xprod_mode, Backend& backend,
    bool array_paths) {
  ProtectStack protect;
  SEXP effective = capped_plssvd_components(
    components, predictors.rows(), predictors.columns(), responses.columns(),
    protect
  );
  fastpls::core::CenteredCrosscovOperator<T, Backend> crosscov(
    predictors, responses, prepared.response_mean.data(),
    prepared.response_mean.size(), backend
  );
  fastpls::core::PlssvdControls controls;
  controls.rsvd.oversample = oversample;
  controls.rsvd.power = power;
  controls.rsvd.seed = seed;
  fastpls::core::OperatorRsvdWorkspace<T> workspace;
  const auto model = fastpls::core::fit_plssvd_operator<T>(
    predictors, crosscov, INTEGER(effective),
    static_cast<std::size_t>(XLENGTH(effective)), controls, backend, workspace
  );
  const auto metric = [&](fastpls::core::ConstMatrixView<T> values) {
    return fastpls::core::dense_response_r2(
      responses, prepared.response_mean.data(),
      prepared.response_mean.size(), values
    );
  };
  return serialize_plssvd_core_model(
    model, predictors, prepared, effective, fitted, store_scores, xprod_mode,
    backend, metric, array_paths
  );
}

template<class T, class Backend>
SEXP fit_dense_simpls_operator(
    fastpls::core::ConstMatrixView<T> predictors,
    fastpls::core::ConstMatrixView<T> responses,
    const fastpls::core::DensePreprocessingResult<T>& prepared,
    SEXP components, bool fitted, bool store_scores,
    int oversample, int power,
    unsigned int seed, const char* xprod_mode, Backend& backend,
    bool array_paths, bool rank_one_massive_operator = true,
    bool use_sample_response_gram = false) {
  ProtectStack protect;
  int maximum_components = 1;
  SEXP effective = capped_simpls_components(
    components, predictors.rows(), predictors.columns(), maximum_components,
    protect
  );
  auto controls = simpls_controls(
    predictors.rows(), predictors.columns(), responses.columns(),
    static_cast<std::size_t>(maximum_components), false,
    oversample, power, seed
  );
  controls.store_scores = fitted || store_scores;
  const long double crosscov_bytes =
    static_cast<long double>(predictors.columns()) *
    static_cast<long double>(responses.columns()) * sizeof(T);
  if (rank_one_massive_operator &&
      crosscov_bytes > 512.0L * 1024.0L * 1024.0L) {
    controls.maximum_block = 1;
    controls.batch_candidate_geometry = false;
    controls.rank_one_operator_direction = true;
    controls.reorthogonalize = true;
  }
  fastpls::core::CenteredCrosscovOperator<T, Backend> initial(
    predictors, responses, prepared.response_mean.data(),
    prepared.response_mean.size(), backend
  );
  fastpls::core::ProjectedOperator<
    T, fastpls::core::CenteredCrosscovOperator<T, Backend>, Backend
  > projected(initial, controls.components, backend);
  fastpls::core::SimplsWorkspace<T> workspace;
  fastpls::core::OperatorRsvdWorkspace<T> rsvd_workspace;
  fastpls::core::Matrix<T> sample_gram;
  if (use_sample_response_gram) {
    sample_gram.resize(responses.rows(), responses.rows());
    backend.gemm(
      responses, responses, false, true, sample_gram.view()
    );
    fastpls::core::Matrix<T> response_mean(responses.columns(), 1);
    T response_mean_square = T(0);
    for (std::size_t response = 0;
         response < responses.columns(); ++response) {
      response_mean(response, 0) = prepared.response_mean[response];
      response_mean_square += prepared.response_mean[response] *
        prepared.response_mean[response];
    }
    fastpls::core::Matrix<T> response_mean_product(responses.rows(), 1);
    backend.gemm(
      responses, response_mean.view(), false, false,
      response_mean_product.view()
    );
    for (std::size_t column = 0; column < sample_gram.columns(); ++column) {
      for (std::size_t row = 0; row < sample_gram.rows(); ++row) {
        sample_gram(row, column) -= response_mean_product(row, 0) +
          response_mean_product(column, 0) - response_mean_square;
      }
    }
  }
  const auto model = fastpls::core::fit_simpls_operator<T>(
    predictors, initial, projected, controls, backend, workspace,
    rsvd_workspace, sample_gram.view()
  );
  if (model.completed_components < controls.components) {
    throw std::runtime_error(
      "fastPLS implicit core SIMPLS returned fewer components than requested"
    );
  }
  const auto metric = [&](fastpls::core::ConstMatrixView<T> values) {
    return fastpls::core::dense_response_r2(
      responses, prepared.response_mean.data(),
      prepared.response_mean.size(), values
    );
  };
  return serialize_simpls_core_model(
    model, predictors, prepared, effective, fitted, controls.store_scores,
    xprod_mode, backend, metric, controls, array_paths
  );
}

fastpls::core::Matrix<double> project_double_scores(
    fastpls::core::ConstMatrixView<double> values,
    fastpls::core::ConstMatrixView<double> projection,
    const std::vector<double>& offset) {
  if (values.columns() != projection.rows() || projection.columns() < 1 ||
      (!offset.empty() && offset.size() < projection.columns())) {
    throw std::invalid_argument(
      "fastPLS projected LDA dimensions are inconsistent"
    );
  }
  fastpls::core::Matrix<double> scores(values.rows(), projection.columns());
  fastpls::runtime::cpu_gemm_f64(
    values, projection, false, false, scores.view()
  );
  if (!offset.empty()) {
    for (std::size_t column = 0; column < scores.columns(); ++column) {
      for (std::size_t row = 0; row < scores.rows(); ++row) {
        scores(row, column) -= offset[column];
      }
    }
  }
  return scores;
}

std::vector<int> labels_from_discriminants(
    fastpls::core::ConstMatrixView<double> discriminants) {
  std::vector<int> predictions(discriminants.rows());
  for (std::size_t row = 0; row < discriminants.rows(); ++row) {
    predictions[row] = static_cast<int>(
      fastpls::core::row_argmax(discriminants, row) + 1
    );
  }
  return predictions;
}

fastpls::core::KernelCvControls kernel_cv_controls(
    SEXP kernel, SEXP gamma, SEXP degree, SEXP offset) {
  const int kernel_code = Rf_asInteger(kernel);
  fastpls::core::KernelCvControls controls;
  if (kernel_code != static_cast<int>(fastpls::core::KernelType::radial_basis) &&
      kernel_code != static_cast<int>(fastpls::core::KernelType::polynomial)) {
    throw std::invalid_argument(
      "nonlinear kernel PLS CV requires an RBF or polynomial kernel"
    );
  }
  controls.kernel = static_cast<fastpls::core::KernelType>(kernel_code);
  controls.gamma = Rf_asReal(gamma);
  controls.degree = Rf_asInteger(degree);
  controls.offset = Rf_asReal(offset);
  if (!std::isfinite(controls.gamma) || controls.gamma <= 0.0 ||
      !std::isfinite(controls.offset) || controls.degree < 1 ||
      controls.degree == NA_INTEGER) {
    throw std::invalid_argument("nonlinear kernel PLS CV controls are invalid");
  }
  return controls;
}

fastpls::core::Matrix<double> double_lda_discriminants(
    fastpls::core::ConstMatrixView<double> scores,
    const fastpls::core::LdaModel<double>& model) {
  if (scores.empty() || scores.columns() != model.linear.columns() ||
      model.linear.rows() != model.constants.size()) {
    throw std::invalid_argument("fastPLS LDA prediction dimensions are invalid");
  }
  fastpls::core::Matrix<double> discriminants(
    scores.rows(), model.linear.rows()
  );
  fastpls::runtime::cpu_gemm_f64(
    scores, model.linear.view(), false, true, discriminants.view()
  );
  for (std::size_t class_index = 0;
       class_index < discriminants.columns(); ++class_index) {
    for (std::size_t row = 0; row < discriminants.rows(); ++row) {
      discriminants(row, class_index) += model.constants[class_index];
    }
  }
  return discriminants;
}

template<class T, class Backend>
SEXP classification_cv_result(
    fastpls::core::ConstMatrixView<T> predictors, SEXP labels,
    SEXP class_count, SEXP folds, SEXP components, SEXP scaling,
    SEXP method, SEXP classifier, SEXP oversample, SEXP power, SEXP seed,
    SEXP store_predictions, SEXP store_scores, Backend& backend,
    std::size_t orthogonal_components = 0,
    const fastpls::core::KernelCvControls& kernel_controls =
      fastpls::core::KernelCvControls()) {
  ProtectStack protect;
  SEXP label_values = protect.add(Rf_coerceVector(labels, INTSXP));
  SEXP fold_values = protect.add(Rf_coerceVector(folds, INTSXP));
  SEXP component_values = protect.add(Rf_coerceVector(components, INTSXP));
  if (XLENGTH(label_values) != static_cast<R_xlen_t>(predictors.rows()) ||
      XLENGTH(fold_values) != static_cast<R_xlen_t>(predictors.rows()) ||
      XLENGTH(component_values) < 1) {
    throw std::invalid_argument(
      "core classification CV dimensions are invalid"
    );
  }
  const int classes = Rf_asInteger(class_count);
  const int scaling_code = Rf_asInteger(scaling);
  const int method_code = Rf_asInteger(method);
  const int classifier_code = Rf_asInteger(classifier);
  const int retain = Rf_asLogical(store_predictions);
  const int retain_scores = Rf_asLogical(store_scores);
  if (classes < 2 || scaling_code < 1 || scaling_code > 3 ||
      (method_code != 1 && method_code != 3 && method_code != 4 &&
       method_code != 5) ||
      (classifier_code != 0 && classifier_code != 1) ||
      retain == NA_LOGICAL || retain_scores == NA_LOGICAL) {
    throw std::invalid_argument(
      "core classification CV controls are invalid"
    );
  }
  const int maximum = *std::max_element(
    INTEGER(component_values),
    INTEGER(component_values) + XLENGTH(component_values)
  );
  fastpls::core::PlssvdControls plssvd;
  plssvd.rsvd.oversample = Rf_asInteger(oversample);
  plssvd.rsvd.power = Rf_asInteger(power);
  plssvd.rsvd.seed = static_cast<unsigned int>(Rf_asInteger(seed));
  const auto simpls = simpls_controls(
    predictors.rows(), predictors.columns(), static_cast<std::size_t>(classes),
    static_cast<std::size_t>(maximum), true,
    plssvd.rsvd.oversample, plssvd.rsvd.power, plssvd.rsvd.seed
  );
  const auto result = fastpls::core::cross_validate_classification<T>(
    predictors, INTEGER(label_values), static_cast<std::size_t>(classes),
    INTEGER(fold_values), INTEGER(component_values),
    static_cast<std::size_t>(XLENGTH(component_values)),
    static_cast<fastpls::core::PredictorScaling>(scaling_code),
    static_cast<fastpls::core::LinearPlsFamily>(method_code),
    static_cast<fastpls::core::ClassificationHead>(classifier_code),
    plssvd, simpls, backend, retain == TRUE, retain_scores == TRUE,
    orthogonal_components, kernel_controls
  );

  SEXP output = protect.add(Rf_allocVector(VECSXP, 6));
  SEXP names = protect.add(Rf_allocVector(STRSXP, 6));
  const char* field_names[6] = {
    "fold", "status", "ncomp", "metric_value", "class_pred", "Ypred"
  };
  for (int index = 0; index < 6; ++index) {
    SET_STRING_ELT(names, index, Rf_mkChar(field_names[index]));
  }
  SET_VECTOR_ELT(output, 0, integer_predictions(result.folds));
  SET_VECTOR_ELT(output, 1, integer_predictions(result.status));
  SET_VECTOR_ELT(output, 2, component_values);
  SET_VECTOR_ELT(output, 3, numeric_vector(result.metrics));
  SET_VECTOR_ELT(output, 4, R_NilValue);
  if (retain == TRUE) {
    SEXP predictions = protect.add(Rf_allocMatrix(
      INTSXP, static_cast<int>(result.predictions.rows()),
      static_cast<int>(result.predictions.columns())
    ));
    std::copy(
      result.predictions.data(),
      result.predictions.data() + result.predictions.size(),
      INTEGER(predictions)
    );
    SET_VECTOR_ELT(output, 4, predictions);
  }
  SET_VECTOR_ELT(
    output, 5, retain_scores == TRUE ? core_matrix_cube(
      result.scores, predictors.rows(), static_cast<std::size_t>(classes),
      false, false
    ) : R_NilValue
  );
  Rf_setAttrib(output, R_NamesSymbol, names);
  return output;
}

template<class T, class Backend>
SEXP regression_cv_result(
    fastpls::core::ConstMatrixView<T> predictors,
    fastpls::core::ConstMatrixView<T> responses, SEXP folds,
    SEXP components, SEXP scaling, SEXP method, SEXP metric,
    SEXP oversample, SEXP power, SEXP seed, SEXP store_predictions,
    Backend& backend, std::size_t orthogonal_components = 0,
    const fastpls::core::KernelCvControls& kernel_controls =
      fastpls::core::KernelCvControls()) {
  ProtectStack protect;
  SEXP fold_values = protect.add(Rf_coerceVector(folds, INTSXP));
  SEXP component_values = protect.add(Rf_coerceVector(components, INTSXP));
  if (predictors.rows() != responses.rows() ||
      XLENGTH(fold_values) != static_cast<R_xlen_t>(predictors.rows()) ||
      XLENGTH(component_values) < 1) {
    throw std::invalid_argument("core regression CV dimensions are invalid");
  }
  const int scaling_code = Rf_asInteger(scaling);
  const int method_code = Rf_asInteger(method);
  const int metric_code = Rf_asInteger(metric);
  const int retain = Rf_asLogical(store_predictions);
  if (scaling_code < 1 || scaling_code > 3 ||
      (method_code != 1 && method_code != 3 && method_code != 4 &&
       method_code != 5) ||
      metric_code < 2 || metric_code > 4 || retain == NA_LOGICAL) {
    throw std::invalid_argument("core regression CV controls are invalid");
  }
  const int maximum = *std::max_element(
    INTEGER(component_values),
    INTEGER(component_values) + XLENGTH(component_values)
  );
  fastpls::core::PlssvdControls plssvd;
  plssvd.rsvd.oversample = Rf_asInteger(oversample);
  plssvd.rsvd.power = Rf_asInteger(power);
  plssvd.rsvd.seed = static_cast<unsigned int>(Rf_asInteger(seed));
  const auto simpls = simpls_controls(
    predictors.rows(), predictors.columns(), responses.columns(),
    static_cast<std::size_t>(maximum), false,
    plssvd.rsvd.oversample, plssvd.rsvd.power, plssvd.rsvd.seed
  );
  const auto result = fastpls::core::cross_validate_regression<T>(
    predictors, responses, INTEGER(fold_values), INTEGER(component_values),
    static_cast<std::size_t>(XLENGTH(component_values)),
    static_cast<fastpls::core::PredictorScaling>(scaling_code),
    static_cast<fastpls::core::LinearPlsFamily>(method_code),
    static_cast<fastpls::core::RegressionMetric>(metric_code),
    plssvd, simpls, backend, retain == TRUE, orthogonal_components,
    kernel_controls
  );

  SEXP output = protect.add(Rf_allocVector(VECSXP, 9));
  SEXP names = protect.add(Rf_allocVector(STRSXP, 9));
  const char* field_names[9] = {
    "fold", "status", "ncomp", "metric_value", "Ypred", "Q2Y", "RMSD",
    "CV_R2", "native_evaluation"
  };
  for (int index = 0; index < 9; ++index) {
    SET_STRING_ELT(names, index, Rf_mkChar(field_names[index]));
  }
  SET_VECTOR_ELT(output, 0, integer_predictions(result.folds));
  SET_VECTOR_ELT(output, 1, integer_predictions(result.status));
  SET_VECTOR_ELT(output, 2, component_values);
  SET_VECTOR_ELT(output, 3, numeric_vector(result.metrics));
  SET_VECTOR_ELT(
    output, 4, retain == TRUE ? core_matrix_cube(
      result.predictions, predictors.rows(), responses.columns(), false, false
    ) : R_NilValue
  );
  SET_VECTOR_ELT(output, 5, numeric_vector(result.q2));
  SET_VECTOR_ELT(output, 6, numeric_vector(result.rmsd));
  SET_VECTOR_ELT(output, 7, numeric_vector(result.observed_r2));
  if (result.evaluation.empty()) {
    SET_VECTOR_ELT(output, 8, R_NilValue);
  } else {
    SEXP evaluation = protect.add(Rf_allocMatrix(
      REALSXP, static_cast<int>(result.evaluation.size()), 12
    ));
    for (std::size_t row = 0; row < result.evaluation.size(); ++row) {
      for (std::size_t column = 0; column < 12; ++column) {
        REAL(evaluation)[row + column * result.evaluation.size()] =
          result.evaluation[row].values[column];
      }
    }
    SET_VECTOR_ELT(output, 8, evaluation);
  }
  Rf_setAttrib(output, R_NamesSymbol, names);
  return output;
}

}  // namespace

extern "C" SEXP _fastPLS_has_cuda() {
  return Rf_ScalarLogical(fastpls_svd::has_cuda_backend());
}

extern "C" SEXP _fastPLS_has_metal() {
  return Rf_ScalarLogical(fastpls_svd::has_metal_backend());
}

extern "C" SEXP _fastPLS_simpls_cache_predictor_crossprod(
    SEXP samples, SEXP predictors, SEXP components) {
  const int n = Rf_asInteger(samples);
  const int p = Rf_asInteger(predictors);
  const int a = Rf_asInteger(components);
  if (n < 1 || p < 1 || a < 1) {
    Rf_error("SIMPLS cache dimensions must be positive integers");
  }
  const fastpls::core::SimplsControls controls = simpls_controls(
    static_cast<std::size_t>(n), static_cast<std::size_t>(p), 1,
    static_cast<std::size_t>(a), false, 1, 0, 1
  );
  return Rf_ScalarLogical(controls.cache_predictor_crossprod);
}

extern "C" SEXP _fastPLS_set_cpu_threads(SEXP threads) {
  const int requested = Rf_asInteger(threads);
  if (requested == NA_INTEGER || requested < 1) {
    Rf_error("fastPLS CPU thread count must be a positive integer");
  }
  const std::vector<std::string> configured =
    fastpls::runtime::set_cpu_threads(requested);
  SEXP output = PROTECT(Rf_allocVector(STRSXP, configured.size()));
  for (R_xlen_t index = 0;
       index < static_cast<R_xlen_t>(configured.size()); ++index) {
    SET_STRING_ELT(
      output, index,
      Rf_mkChar(configured[static_cast<std::size_t>(index)].c_str())
    );
  }
  UNPROTECT(1);
  return output;
}

extern "C" SEXP _fastPLS_rsvd_audit_reset_debug() {
  fastpls_svd::reset_rsvd_audit_summary();
  return R_NilValue;
}

extern "C" SEXP _fastPLS_rsvd_audit_summary_debug() {
  const fastpls::core::RSVDAuditSummary summary =
    fastpls_svd::current_rsvd_audit_summary();
  SEXP output = PROTECT(Rf_allocVector(VECSXP, 9));
  SET_VECTOR_ELT(output, 0, Rf_ScalarInteger(summary.solves));
  SET_VECTOR_ELT(output, 1, Rf_ScalarInteger(summary.certified));
  SET_VECTOR_ELT(
    output, 2, Rf_ScalarInteger(summary.deterministic_fallbacks)
  );
  SET_VECTOR_ELT(output, 3, Rf_ScalarInteger(summary.failures));
  SET_VECTOR_ELT(output, 4, Rf_ScalarInteger(summary.max_attempts));
  SET_VECTOR_ELT(
    output, 5, Rf_ScalarInteger(summary.max_effective_oversample)
  );
  SET_VECTOR_ELT(
    output, 6, Rf_ScalarInteger(summary.max_effective_power_iters)
  );
  SET_VECTOR_ELT(
    output, 7, Rf_ScalarReal(summary.max_triplet_residual)
  );
  SET_VECTOR_ELT(
    output, 8, Rf_ScalarReal(summary.max_omitted_direction_ratio)
  );
  SEXP names = PROTECT(Rf_allocVector(STRSXP, 9));
  const char* labels[] = {
    "solves", "certified", "deterministic_fallbacks", "failures",
    "max_attempts", "max_effective_oversample", "max_effective_power",
    "max_triplet_residual", "max_omitted_direction_ratio"
  };
  for (int index = 0; index < 9; ++index) {
    SET_STRING_ELT(names, index, Rf_mkChar(labels[index]));
  }
  Rf_setAttrib(output, R_NamesSymbol, names);
  UNPROTECT(2);
  return output;
}

extern "C" SEXP _fastPLS_fastsvd_core_cpp(
    SEXP matrix, SEXP components, SEXP oversample, SEXP power, SEXP seed,
    SEXP left_only) {
  return translate_exceptions("double core rSVD", [&] {
    const int retained = Rf_asInteger(components);
    const int oversample_count = Rf_asInteger(oversample);
    const int power_count = Rf_asInteger(power);
    const int seed_value = Rf_asInteger(seed);
    const int left_only_value = Rf_asLogical(left_only);
    if (retained < 1 || oversample_count < 0 || power_count < 0 ||
        seed_value == NA_INTEGER || left_only_value == NA_LOGICAL) {
      throw std::invalid_argument("double core rSVD controls are invalid");
    }
    const auto values = numeric_matrix_view(matrix, "x");
    fastpls::runtime::CpuLinearAlgebraF64 backend;
    fastpls::core::ExplicitOperator<
      double, fastpls::runtime::CpuLinearAlgebraF64
    > input(values, backend);
    fastpls::core::RsvdControls controls;
    controls.oversample = oversample_count;
    controls.power = power_count;
    controls.seed = static_cast<unsigned int>(seed_value);
    controls.left_only = left_only_value == TRUE;
    try {
      const auto result = fastpls::core::audited_operator_rsvd<double>(
        input, retained, controls, backend
      );
      fastpls_svd::record_rsvd_audit_case(
        result.audit.certified, result.audit.deterministic_fallback,
        result.audit.attempts, result.audit.effective_oversample,
        result.audit.effective_power, result.audit.triplet_residual,
        result.audit.omitted_direction_ratio
      );
      return serialize_audited_rsvd<double>(
        result, [](const fastpls::core::Matrix<double>& value) {
          return numeric_matrix(value);
        }
      );
    } catch (...) {
      fastpls_svd::record_rsvd_audit_case(
        false, false, 0, 0, 0, 0.0, 0.0, true
      );
      throw;
    }
  });
}

extern "C" SEXP _fastPLS_fastsvd_float32_core_cpp(
    SEXP matrix, SEXP components, SEXP oversample, SEXP power, SEXP seed,
    SEXP left_only) {
  return translate_exceptions("float32 core rSVD", [&] {
    const int retained = Rf_asInteger(components);
    const int oversample_count = Rf_asInteger(oversample);
    const int power_count = Rf_asInteger(power);
    const int seed_value = Rf_asInteger(seed);
    const int left_only_value = Rf_asLogical(left_only);
    if (retained < 1 || oversample_count < 0 || power_count < 0 ||
        seed_value == NA_INTEGER || left_only_value == NA_LOGICAL) {
      throw std::invalid_argument("float32 core rSVD controls are invalid");
    }
    auto values = float_matrix_from_s4(matrix, "x");
    fastpls::runtime::CpuLinearAlgebraF32 backend;
    fastpls::core::ExplicitOperator<
      float, fastpls::runtime::CpuLinearAlgebraF32
    > input(values.view(), backend);
    fastpls::core::RsvdControls controls;
    controls.oversample = oversample_count;
    controls.power = power_count;
    controls.seed = static_cast<unsigned int>(seed_value);
    controls.left_only = left_only_value == TRUE;
    try {
      const auto result = fastpls::core::audited_operator_rsvd<float>(
        input, retained, controls, backend
      );
      fastpls_svd::record_rsvd_audit_case(
        result.audit.certified, result.audit.deterministic_fallback,
        result.audit.attempts, result.audit.effective_oversample,
        result.audit.effective_power, result.audit.triplet_residual,
        result.audit.omitted_direction_ratio
      );
      return serialize_audited_rsvd<float>(
        result, [](const fastpls::core::Matrix<float>& value) {
          return float_bits_matrix(value);
        }
      );
    } catch (...) {
      fastpls_svd::record_rsvd_audit_case(
        false, false, 0, 0, 0, 0.0, 0.0, true
      );
      throw;
    }
  });
}

extern "C" SEXP _fastPLS_cv_folds_core_cpp(
    SEXP groups, SEXP labels, SEXP class_count, SEXP folds) {
  return translate_exceptions("grouped cross-validation folds", [&] {
    ProtectStack protect;
    SEXP group_values = protect.add(Rf_coerceVector(groups, INTSXP));
    const std::size_t samples = static_cast<std::size_t>(
      XLENGTH(group_values)
    );
    if (samples < 2) {
      throw std::invalid_argument(
        "cross-validation requires at least two group assignments"
      );
    }
    SEXP label_values = R_NilValue;
    const int classes = Rf_asInteger(class_count);
    const int* label_data = nullptr;
    if (labels != R_NilValue) {
      label_values = protect.add(Rf_coerceVector(labels, INTSXP));
      if (XLENGTH(label_values) != static_cast<R_xlen_t>(samples)) {
        throw std::invalid_argument(
          "cross-validation labels must match group assignments"
        );
      }
      label_data = INTEGER(label_values);
    }
    GetRNGstate();
    std::vector<int> result;
    try {
      result = fastpls::core::grouped_folds(
        INTEGER(group_values), samples, label_data,
        static_cast<std::size_t>(std::max(classes, 0)), Rf_asInteger(folds),
        [](std::size_t remaining) {
          return static_cast<std::size_t>(R_unif_index(remaining));
        }
      );
    } catch (...) {
      PutRNGstate();
      throw;
    }
    PutRNGstate();
    return integer_predictions(result);
  });
}

extern "C" SEXP _fastPLS_pls_cv_classification_core_cpp(
    SEXP predictors, SEXP labels, SEXP class_count, SEXP folds,
    SEXP components, SEXP scaling, SEXP method, SEXP classifier,
    SEXP oversample, SEXP power, SEXP seed, SEXP store_predictions,
    SEXP store_scores) {
  return translate_exceptions("core classification cross-validation", [&] {
    const auto x = numeric_matrix_view(predictors, "Xdata");
    fastpls::runtime::CpuLinearAlgebraF64 backend;
    return classification_cv_result<double>(
      x, labels, class_count, folds, components, scaling, method, classifier,
      oversample, power, seed, store_predictions, store_scores, backend
    );
  });
}

extern "C" SEXP _fastPLS_pls_cv_classification_float32_core_cpp(
    SEXP predictors, SEXP labels, SEXP class_count, SEXP folds,
    SEXP components, SEXP scaling, SEXP method, SEXP classifier,
    SEXP oversample, SEXP power, SEXP seed, SEXP store_predictions,
    SEXP store_scores) {
  return translate_exceptions(
    "float32 core classification cross-validation", [&] {
      const auto x = float_matrix_from_s4(predictors, "Xdata");
      fastpls::runtime::CpuLinearAlgebraF32 backend;
      return classification_cv_result<float>(
        x.view(), labels, class_count, folds, components, scaling, method,
        classifier, oversample, power, seed, store_predictions, store_scores,
        backend
      );
    }
  );
}

extern "C" SEXP _fastPLS_pls_cv_classification_float32_metal_core_cpp(
    SEXP predictors, SEXP labels, SEXP class_count, SEXP folds,
    SEXP components, SEXP scaling, SEXP method, SEXP classifier,
    SEXP north, SEXP kernel, SEXP gamma, SEXP degree, SEXP offset,
    SEXP oversample, SEXP power, SEXP seed, SEXP store_predictions,
    SEXP store_scores) {
  return translate_exceptions(
    "Metal float32 classification cross-validation", [&] {
      if (!fastpls_svd::has_metal_backend()) {
        throw std::runtime_error(
          "Metal is unavailable; no CPU fallback is performed"
        );
      }
      const auto x = float_matrix_from_s4(predictors, "Xdata");
      const int classes = Rf_asInteger(class_count);
      const int method_code = Rf_asInteger(method);
      const int orthogonal = method_code == 4 ? Rf_asInteger(north) : 0;
      if (classes < 2 || (method_code == 4 && orthogonal < 1)) {
        throw std::invalid_argument(
          "Metal classification CV controls are invalid"
        );
      }
      fastpls::core::KernelCvControls kernel_controls;
      if (method_code == 5) {
        kernel_controls = kernel_cv_controls(
          kernel, gamma, degree, offset
        );
      }
      RoutedLinearAlgebraF32 backend(
        2, x.rows(), x.columns(), static_cast<std::size_t>(classes)
      );
      return classification_cv_result<float>(
        x.view(), labels, class_count, folds, components, scaling, method,
        classifier, oversample, power, seed, store_predictions,
        store_scores, backend, static_cast<std::size_t>(orthogonal),
        kernel_controls
      );
    }
  );
}

extern "C" SEXP _fastPLS_pls_cv_opls_classification_core_cpp(
    SEXP predictors, SEXP labels, SEXP class_count, SEXP folds,
    SEXP components, SEXP scaling, SEXP classifier, SEXP north,
    SEXP oversample, SEXP power, SEXP seed, SEXP store_predictions,
    SEXP store_scores) {
  return translate_exceptions("core OPLS classification CV", [&] {
    ProtectStack protect;
    SEXP method = protect.add(Rf_ScalarInteger(4));
    const int orthogonal = Rf_asInteger(north);
    if (orthogonal < 1) {
      throw std::invalid_argument("OPLS north must be positive");
    }
    const auto x = numeric_matrix_view(predictors, "Xdata");
    fastpls::runtime::CpuLinearAlgebraF64 backend;
    return classification_cv_result<double>(
      x, labels, class_count, folds, components, scaling, method, classifier,
      oversample, power, seed, store_predictions, store_scores, backend,
      static_cast<std::size_t>(orthogonal)
    );
  });
}

extern "C" SEXP _fastPLS_pls_cv_opls_classification_float32_core_cpp(
    SEXP predictors, SEXP labels, SEXP class_count, SEXP folds,
    SEXP components, SEXP scaling, SEXP classifier, SEXP north,
    SEXP oversample, SEXP power, SEXP seed, SEXP store_predictions,
    SEXP store_scores) {
  return translate_exceptions("float32 core OPLS classification CV", [&] {
    ProtectStack protect;
    SEXP method = protect.add(Rf_ScalarInteger(4));
    const int orthogonal = Rf_asInteger(north);
    if (orthogonal < 1) {
      throw std::invalid_argument("OPLS north must be positive");
    }
    const auto x = float_matrix_from_s4(predictors, "Xdata");
    fastpls::runtime::CpuLinearAlgebraF32 backend;
    return classification_cv_result<float>(
      x.view(), labels, class_count, folds, components, scaling, method,
      classifier, oversample, power, seed, store_predictions, store_scores,
      backend, static_cast<std::size_t>(orthogonal)
    );
  });
}

extern "C" SEXP _fastPLS_pls_cv_kernel_classification_core_cpp(
    SEXP predictors, SEXP labels, SEXP class_count, SEXP folds,
    SEXP components, SEXP scaling, SEXP classifier, SEXP kernel,
    SEXP gamma, SEXP degree, SEXP offset, SEXP oversample, SEXP power,
    SEXP seed, SEXP store_predictions, SEXP store_scores) {
  return translate_exceptions("core nonlinear kernel classification CV", [&] {
    ProtectStack protect;
    SEXP method = protect.add(Rf_ScalarInteger(5));
    const auto controls = kernel_cv_controls(kernel, gamma, degree, offset);
    const auto x = numeric_matrix_view(predictors, "Xdata");
    fastpls::runtime::CpuLinearAlgebraF64 backend;
    return classification_cv_result<double>(
      x, labels, class_count, folds, components, scaling, method, classifier,
      oversample, power, seed, store_predictions, store_scores, backend, 0,
      controls
    );
  });
}

extern "C" SEXP _fastPLS_pls_cv_kernel_classification_float32_core_cpp(
    SEXP predictors, SEXP labels, SEXP class_count, SEXP folds,
    SEXP components, SEXP scaling, SEXP classifier, SEXP kernel,
    SEXP gamma, SEXP degree, SEXP offset, SEXP oversample, SEXP power,
    SEXP seed, SEXP store_predictions, SEXP store_scores) {
  return translate_exceptions(
    "float32 core nonlinear kernel classification CV", [&] {
      ProtectStack protect;
      SEXP method = protect.add(Rf_ScalarInteger(5));
      const auto controls = kernel_cv_controls(kernel, gamma, degree, offset);
      const auto x = float_matrix_from_s4(predictors, "Xdata");
      fastpls::runtime::CpuLinearAlgebraF32 backend;
      return classification_cv_result<float>(
        x.view(), labels, class_count, folds, components, scaling, method,
        classifier, oversample, power, seed, store_predictions, store_scores,
        backend, 0, controls
      );
    }
  );
}

extern "C" SEXP _fastPLS_pls_cv_regression_core_cpp(
    SEXP predictors, SEXP responses, SEXP folds, SEXP components,
    SEXP scaling, SEXP method, SEXP metric, SEXP oversample, SEXP power,
    SEXP seed, SEXP store_predictions) {
  return translate_exceptions("core regression cross-validation", [&] {
    const auto x = numeric_matrix_view(predictors, "Xdata");
    const auto y = numeric_matrix_view(responses, "Ydata");
    fastpls::runtime::CpuLinearAlgebraF64 backend;
    return regression_cv_result<double>(
      x, y, folds, components, scaling, method, metric, oversample, power,
      seed, store_predictions, backend
    );
  });
}

extern "C" SEXP _fastPLS_pls_cv_regression_float32_core_cpp(
    SEXP predictors, SEXP responses, SEXP folds, SEXP components,
    SEXP scaling, SEXP method, SEXP metric, SEXP oversample, SEXP power,
    SEXP seed, SEXP store_predictions) {
  return translate_exceptions("float32 core regression cross-validation", [&] {
    const auto x = float_matrix_from_s4(predictors, "Xdata");
    const auto y = float_matrix_from_s4(responses, "Ydata");
    fastpls::runtime::CpuLinearAlgebraF32 backend;
    return regression_cv_result<float>(
      x.view(), y.view(), folds, components, scaling, method, metric,
      oversample, power, seed, store_predictions, backend
    );
  });
}

extern "C" SEXP _fastPLS_pls_cv_regression_float32_metal_core_cpp(
    SEXP predictors, SEXP responses, SEXP folds, SEXP components,
    SEXP scaling, SEXP method, SEXP metric, SEXP north, SEXP kernel,
    SEXP gamma, SEXP degree, SEXP offset, SEXP oversample, SEXP power,
    SEXP seed, SEXP store_predictions) {
  return translate_exceptions(
    "Metal float32 regression cross-validation", [&] {
      if (!fastpls_svd::has_metal_backend()) {
        throw std::runtime_error(
          "Metal is unavailable; no CPU fallback is performed"
        );
      }
      const auto x = float_matrix_from_s4(predictors, "Xdata");
      const auto y = float_matrix_from_s4(responses, "Ydata");
      const int method_code = Rf_asInteger(method);
      const int orthogonal = method_code == 4 ? Rf_asInteger(north) : 0;
      if (x.rows() != y.rows() || (method_code == 4 && orthogonal < 1)) {
        throw std::invalid_argument(
          "Metal regression CV controls are invalid"
        );
      }
      fastpls::core::KernelCvControls kernel_controls;
      if (method_code == 5) {
        kernel_controls = kernel_cv_controls(
          kernel, gamma, degree, offset
        );
      }
      RoutedLinearAlgebraF32 backend(
        2, x.rows(), x.columns(), y.columns()
      );
      return regression_cv_result<float>(
        x.view(), y.view(), folds, components, scaling, method, metric,
        oversample, power, seed, store_predictions, backend,
        static_cast<std::size_t>(orthogonal), kernel_controls
      );
    }
  );
}

extern "C" SEXP _fastPLS_pls_cv_opls_regression_core_cpp(
    SEXP predictors, SEXP responses, SEXP folds, SEXP components,
    SEXP scaling, SEXP metric, SEXP north, SEXP oversample, SEXP power,
    SEXP seed, SEXP store_predictions) {
  return translate_exceptions("core OPLS regression CV", [&] {
    ProtectStack protect;
    SEXP method = protect.add(Rf_ScalarInteger(4));
    const int orthogonal = Rf_asInteger(north);
    if (orthogonal < 1) {
      throw std::invalid_argument("OPLS north must be positive");
    }
    const auto x = numeric_matrix_view(predictors, "Xdata");
    const auto y = numeric_matrix_view(responses, "Ydata");
    fastpls::runtime::CpuLinearAlgebraF64 backend;
    return regression_cv_result<double>(
      x, y, folds, components, scaling, method, metric, oversample, power,
      seed, store_predictions, backend,
      static_cast<std::size_t>(orthogonal)
    );
  });
}

extern "C" SEXP _fastPLS_pls_cv_opls_regression_float32_core_cpp(
    SEXP predictors, SEXP responses, SEXP folds, SEXP components,
    SEXP scaling, SEXP metric, SEXP north, SEXP oversample, SEXP power,
    SEXP seed, SEXP store_predictions) {
  return translate_exceptions("float32 core OPLS regression CV", [&] {
    ProtectStack protect;
    SEXP method = protect.add(Rf_ScalarInteger(4));
    const int orthogonal = Rf_asInteger(north);
    if (orthogonal < 1) {
      throw std::invalid_argument("OPLS north must be positive");
    }
    const auto x = float_matrix_from_s4(predictors, "Xdata");
    const auto y = float_matrix_from_s4(responses, "Ydata");
    fastpls::runtime::CpuLinearAlgebraF32 backend;
    return regression_cv_result<float>(
      x.view(), y.view(), folds, components, scaling, method, metric,
      oversample, power, seed, store_predictions, backend,
      static_cast<std::size_t>(orthogonal)
    );
  });
}

extern "C" SEXP _fastPLS_pls_cv_kernel_regression_core_cpp(
    SEXP predictors, SEXP responses, SEXP folds, SEXP components,
    SEXP scaling, SEXP metric, SEXP kernel, SEXP gamma, SEXP degree,
    SEXP offset, SEXP oversample, SEXP power, SEXP seed,
    SEXP store_predictions) {
  return translate_exceptions("core nonlinear kernel regression CV", [&] {
    ProtectStack protect;
    SEXP method = protect.add(Rf_ScalarInteger(5));
    const auto controls = kernel_cv_controls(kernel, gamma, degree, offset);
    const auto x = numeric_matrix_view(predictors, "Xdata");
    const auto y = numeric_matrix_view(responses, "Ydata");
    fastpls::runtime::CpuLinearAlgebraF64 backend;
    return regression_cv_result<double>(
      x, y, folds, components, scaling, method, metric, oversample, power,
      seed, store_predictions, backend, 0, controls
    );
  });
}

extern "C" SEXP _fastPLS_pls_cv_kernel_regression_float32_core_cpp(
    SEXP predictors, SEXP responses, SEXP folds, SEXP components,
    SEXP scaling, SEXP metric, SEXP kernel, SEXP gamma, SEXP degree,
    SEXP offset, SEXP oversample, SEXP power, SEXP seed,
    SEXP store_predictions) {
  return translate_exceptions(
    "float32 core nonlinear kernel regression CV", [&] {
      ProtectStack protect;
      SEXP method = protect.add(Rf_ScalarInteger(5));
      const auto controls = kernel_cv_controls(kernel, gamma, degree, offset);
      const auto x = float_matrix_from_s4(predictors, "Xdata");
      const auto y = float_matrix_from_s4(responses, "Ydata");
      fastpls::runtime::CpuLinearAlgebraF32 backend;
      return regression_cv_result<float>(
        x.view(), y.view(), folds, components, scaling, method, metric,
        oversample, power, seed, store_predictions, backend, 0, controls
      );
    }
  );
}

extern "C" SEXP _fastPLS_lda_train_prefix_cpp(
    SEXP scores, SEXP labels, SEXP class_count, SEXP components, SEXP ridge) {
  return translate_exceptions("double PLS-LDA fitting", [&] {
    ProtectStack protect;
    SEXP scores_real = protect.add(Rf_coerceVector(scores, REALSXP));
    SEXP labels_integer = protect.add(Rf_coerceVector(labels, INTSXP));
    SEXP components_integer = protect.add(
      Rf_coerceVector(components, INTSXP)
    );
    (void)ridge;  // The regularization sequence is deterministic.
    const int classes = Rf_asInteger(class_count);
    if (classes < 2) {
      throw std::invalid_argument("fastPLS LDA requires at least two classes");
    }
    const auto models = train_double_lda(
      numeric_matrix_view(scores_real, "Ttrain"), INTEGER(labels_integer),
      static_cast<std::size_t>(XLENGTH(labels_integer)),
      static_cast<std::size_t>(classes), INTEGER(components_integer),
      static_cast<std::size_t>(XLENGTH(components_integer))
    );
    return protect.add(double_lda_models(
      models, INTEGER(components_integer)
    ));
  });
}

extern "C" SEXP _fastPLS_lda_train_moments_prefix_cpp(
    SEXP gram, SEXP class_sums, SEXP counts, SEXP sample_count,
    SEXP components) {
  return translate_exceptions("moment-based double PLS-LDA fitting", [&] {
    ProtectStack protect;
    SEXP gram_real = protect.add(Rf_coerceVector(gram, REALSXP));
    SEXP sums_real = protect.add(Rf_coerceVector(class_sums, REALSXP));
    SEXP counts_real = protect.add(Rf_coerceVector(counts, REALSXP));
    SEXP components_integer = protect.add(
      Rf_coerceVector(components, INTSXP)
    );
    const int samples = Rf_asInteger(sample_count);
    if (samples < 1) {
      throw std::invalid_argument("fastPLS LDA sample count must be positive");
    }
    const auto count_values = numeric_values(counts_real, "counts");
    const auto models =
      fastpls::core::train_lda_prefixes_from_moments<double>(
        numeric_matrix_view(gram_real, "gram"),
        numeric_matrix_view(sums_real, "class_sums"), count_values.data(),
        count_values.size(), static_cast<std::size_t>(samples),
        INTEGER(components_integer),
        static_cast<std::size_t>(XLENGTH(components_integer))
      );
    return protect.add(double_lda_models(
      models, INTEGER(components_integer)
    ));
  });
}

extern "C" SEXP _fastPLS_lda_project_train_prefix_cpp(
    SEXP predictors, SEXP projection, SEXP offset, SEXP labels,
    SEXP class_count, SEXP components, SEXP ridge) {
  return translate_exceptions("projected double PLS-LDA fitting", [&] {
    ProtectStack protect;
    SEXP predictors_real = protect.add(Rf_coerceVector(predictors, REALSXP));
    SEXP projection_real = protect.add(Rf_coerceVector(projection, REALSXP));
    SEXP offset_real = protect.add(Rf_coerceVector(offset, REALSXP));
    SEXP labels_integer = protect.add(Rf_coerceVector(labels, INTSXP));
    SEXP components_integer = protect.add(
      Rf_coerceVector(components, INTSXP)
    );
    (void)ridge;
    const auto x = numeric_matrix_view(predictors_real, "Xtrain");
    const auto weights = numeric_matrix_view(projection_real, "R");
    const auto offsets = numeric_values(offset_real, "offset");
    const auto scores = project_double_scores(x, weights, offsets);
    const int classes = Rf_asInteger(class_count);
    if (classes < 2) {
      throw std::invalid_argument("fastPLS LDA requires at least two classes");
    }
    const auto models = train_double_lda(
      scores.view(), INTEGER(labels_integer),
      static_cast<std::size_t>(XLENGTH(labels_integer)),
      static_cast<std::size_t>(classes), INTEGER(components_integer),
      static_cast<std::size_t>(XLENGTH(components_integer))
    );
    return protect.add(double_lda_models(
      models, INTEGER(components_integer), "cpp_project"
    ));
  });
}

extern "C" SEXP _fastPLS_lda_predict_cpp(SEXP scores, SEXP model) {
  return translate_exceptions("double PLS-LDA prediction", [&] {
    ProtectStack protect;
    SEXP scores_real = protect.add(Rf_coerceVector(scores, REALSXP));
    const auto values = numeric_matrix_view(scores_real, "Ttest");
    const auto fitted = double_lda_model_from_sexp(model);
    const auto discriminants = double_lda_discriminants(values, fitted);
    const auto predictions = labels_from_discriminants(discriminants.view());
    SEXP output = protect.add(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(output, 0, integer_predictions(predictions));
    SET_VECTOR_ELT(output, 1, numeric_matrix(discriminants));
    SEXP names = protect.add(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, Rf_mkChar("pred"));
    SET_STRING_ELT(names, 1, Rf_mkChar("scores"));
    Rf_setAttrib(output, R_NamesSymbol, names);
    return output;
  });
}

extern "C" SEXP _fastPLS_lda_predict_labels_cpp(SEXP scores, SEXP model) {
  return translate_exceptions("double PLS-LDA label prediction", [&] {
    ProtectStack protect;
    SEXP scores_real = protect.add(Rf_coerceVector(scores, REALSXP));
    const auto discriminants = double_lda_discriminants(
      numeric_matrix_view(scores_real, "Ttest"),
      double_lda_model_from_sexp(model)
    );
    const auto predictions = labels_from_discriminants(discriminants.view());
    return protect.add(integer_predictions(predictions));
  });
}

extern "C" SEXP _fastPLS_lda_project_predict_labels_cpp(
    SEXP predictors, SEXP projection, SEXP offset, SEXP model) {
  return translate_exceptions("projected double PLS-LDA prediction", [&] {
    ProtectStack protect;
    SEXP predictors_real = protect.add(Rf_coerceVector(predictors, REALSXP));
    SEXP projection_real = protect.add(Rf_coerceVector(projection, REALSXP));
    SEXP offset_real = protect.add(Rf_coerceVector(offset, REALSXP));
    const auto x = numeric_matrix_view(predictors_real, "Xtest");
    const auto weights = numeric_matrix_view(projection_real, "R");
    const auto offsets = numeric_values(offset_real, "offset");
    const auto fitted = double_lda_model_from_sexp(model);
    if (x.columns() != weights.rows() ||
        weights.columns() != fitted.linear.columns() ||
        fitted.linear.rows() != fitted.constants.size() ||
        (!offsets.empty() && offsets.size() < weights.columns())) {
      throw std::invalid_argument(
        "fastPLS projected LDA prediction dimensions are inconsistent"
      );
    }

    const double latent_work = static_cast<double>(x.rows()) *
      static_cast<double>(weights.columns()) *
      static_cast<double>(x.columns() + fitted.linear.rows());
    const double direct_work = static_cast<double>(x.rows()) *
      static_cast<double>(x.columns()) *
      static_cast<double>(fitted.linear.rows());
    std::vector<int> predictions;
    if (std::isfinite(latent_work) && std::isfinite(direct_work) &&
        direct_work < 0.5 * latent_work) {
      fastpls::core::Matrix<double> direct_weights(
        weights.rows(), fitted.linear.rows()
      );
      fastpls::runtime::cpu_gemm_f64(
        weights, fitted.linear.view(), false, true, direct_weights.view()
      );
      fastpls::core::Matrix<double> discriminants(
        x.rows(), fitted.linear.rows()
      );
      fastpls::runtime::cpu_gemm_f64(
        x, direct_weights.view(), false, false, discriminants.view()
      );
      for (std::size_t class_index = 0;
           class_index < discriminants.columns(); ++class_index) {
        double constant = fitted.constants[class_index];
        for (std::size_t component = 0;
             component < weights.columns() && !offsets.empty(); ++component) {
          constant -= offsets[component] *
            fitted.linear(class_index, component);
        }
        for (std::size_t row = 0; row < discriminants.rows(); ++row) {
          discriminants(row, class_index) += constant;
        }
      }
      predictions = labels_from_discriminants(discriminants.view());
    } else {
      const auto projected = project_double_scores(x, weights, offsets);
      const auto discriminants = double_lda_discriminants(
        projected.view(), fitted
      );
      predictions = labels_from_discriminants(discriminants.view());
    }
    return protect.add(integer_predictions(predictions));
  });
}

extern "C" SEXP _fastPLS_spearman_correlation_cpp(SEXP observed,
                                                   SEXP predicted) {
  if (!Rf_isVectorAtomic(observed) || !Rf_isVectorAtomic(predicted)) {
    Rf_error("Spearman correlation requires numeric vectors");
  }
  const R_xlen_t observed_size = XLENGTH(observed);
  if (observed_size != XLENGTH(predicted)) {
    Rf_error("Spearman correlation requires vectors of equal length");
  }
  SEXP observed_real = PROTECT(Rf_coerceVector(observed, REALSXP));
  SEXP predicted_real = PROTECT(Rf_coerceVector(predicted, REALSXP));
  try {
    const fastpls::core::CorrelationResult result =
      fastpls::core::spearman_correlation(
        REAL(observed_real), REAL(predicted_real),
        static_cast<std::size_t>(observed_size)
      );
    UNPROTECT(2);
    if (result.status == fastpls::core::CorrelationStatus::no_complete_pairs) {
      Rf_error("no complete element pairs");
    }
    if (result.status != fastpls::core::CorrelationStatus::success) {
      return Rf_ScalarReal(NA_REAL);
    }
    return Rf_ScalarReal(result.value);
  } catch (const std::exception& exception) {
    UNPROTECT(2);
    Rf_error("%s", exception.what());
  } catch (...) {
    UNPROTECT(2);
    Rf_error("Unknown error in Spearman correlation");
  }
  return R_NilValue;
}

extern "C" SEXP _fastPLS_evaluate_regression_core_cpp(
    SEXP observed, SEXP predicted, SEXP training, SEXP relative_epsilon,
    SEXP na_rm) {
  return translate_exceptions("compiled regression evaluation", [&] {
    if (!Rf_isMatrix(observed) || !Rf_isMatrix(predicted)) {
      throw std::invalid_argument(
        "compiled regression evaluation requires two matrices"
      );
    }
    ProtectStack protect;
    SEXP observed_real = protect.add(Rf_coerceVector(observed, REALSXP));
    SEXP predicted_real = protect.add(Rf_coerceVector(predicted, REALSXP));
    const SEXP dimensions = Rf_getAttrib(observed, R_DimSymbol);
    const SEXP predicted_dimensions = Rf_getAttrib(predicted, R_DimSymbol);
    if (TYPEOF(dimensions) != INTSXP || XLENGTH(dimensions) != 2 ||
        TYPEOF(predicted_dimensions) != INTSXP ||
        XLENGTH(predicted_dimensions) != 2 ||
        INTEGER(dimensions)[0] != INTEGER(predicted_dimensions)[0] ||
        INTEGER(dimensions)[1] != INTEGER(predicted_dimensions)[1]) {
      throw std::invalid_argument(
        "observed and predicted must have the same dimensions"
      );
    }
    const std::size_t rows = static_cast<std::size_t>(INTEGER(dimensions)[0]);
    const std::size_t columns = static_cast<std::size_t>(INTEGER(dimensions)[1]);
    const std::size_t size = rows * columns;
    const double* observed_data = REAL(observed_real);
    const double* predicted_data = REAL(predicted_real);
    const double epsilon = Rf_asReal(relative_epsilon);
    if (!std::isfinite(epsilon) || epsilon < 0.0) {
      throw std::invalid_argument("relative_epsilon must be finite and non-negative");
    }

    SEXP training_real = R_NilValue;
    const double* training_data = nullptr;
    std::size_t training_rows = 0;
    if (training != R_NilValue) {
      if (!Rf_isMatrix(training)) {
        throw std::invalid_argument("training responses must be a matrix");
      }
      const SEXP training_dimensions = Rf_getAttrib(training, R_DimSymbol);
      if (TYPEOF(training_dimensions) != INTSXP ||
          XLENGTH(training_dimensions) != 2 ||
          INTEGER(training_dimensions)[1] != static_cast<int>(columns)) {
        throw std::invalid_argument(
          "training responses must have the same number of columns"
        );
      }
      training_rows = static_cast<std::size_t>(INTEGER(training_dimensions)[0]);
      training_real = protect.add(Rf_coerceVector(training, REALSXP));
      training_data = REAL(training_real);
    }

    std::vector<long double> observed_sums(columns, 0.0L);
    std::vector<std::size_t> observed_counts(columns, 0);
    std::vector<long double> training_sums(columns, 0.0L);
    std::vector<std::size_t> training_counts(columns, 0);
    bool all_complete = true;
    std::size_t relative_pairs = 0;
    for (std::size_t column = 0; column < columns; ++column) {
      for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t index = column * rows + row;
        if (std::isfinite(observed_data[index]) &&
            std::isfinite(predicted_data[index])) {
          observed_sums[column] += observed_data[index];
          ++observed_counts[column];
          if (std::abs(observed_data[index]) > epsilon) ++relative_pairs;
        } else {
          all_complete = false;
        }
      }
      if (training_data) {
        for (std::size_t row = 0; row < training_rows; ++row) {
          const double value = training_data[column * training_rows + row];
          if (std::isfinite(value)) {
            training_sums[column] += value;
            ++training_counts[column];
          }
        }
      }
    }

    long double sse = 0.0L, absolute_error = 0.0L, error_sum = 0.0L;
    long double observed_tss = 0.0L, training_tss = 0.0L;
    long double observed_sum = 0.0L, observed_square = 0.0L;
    long double predicted_sum = 0.0L, predicted_square = 0.0L;
    long double cross_sum = 0.0L, relative_sum = 0.0L;
    std::size_t complete = 0, relative_count = 0;
    std::vector<double> relative_values;
    relative_values.reserve(relative_pairs);
    std::vector<double> complete_observed, complete_predicted;
    if (!all_complete) {
      complete_observed.reserve(size);
      complete_predicted.reserve(size);
    }
    for (std::size_t column = 0; column < columns; ++column) {
      const long double observed_mean = observed_counts[column] ?
        observed_sums[column] / observed_counts[column] : NAN;
      const long double training_mean = training_data && training_counts[column] ?
        training_sums[column] / training_counts[column] : NAN;
      for (std::size_t row = 0; row < rows; ++row) {
        const std::size_t index = column * rows + row;
        const double y = observed_data[index];
        const double estimate = predicted_data[index];
        if (!std::isfinite(y) || !std::isfinite(estimate)) continue;
        const long double error = estimate - y;
        sse += error * error;
        absolute_error += std::abs(error);
        error_sum += error;
        observed_tss += (y - observed_mean) * (y - observed_mean);
        if (training_data && std::isfinite(static_cast<double>(training_mean))) {
          training_tss += (y - training_mean) * (y - training_mean);
        }
        observed_sum += y;
        observed_square += static_cast<long double>(y) * y;
        predicted_sum += estimate;
        predicted_square += static_cast<long double>(estimate) * estimate;
        cross_sum += static_cast<long double>(y) * estimate;
        if (std::abs(y) > epsilon) {
          const double relative = std::abs(static_cast<double>(error) / y) * 100.0;
          relative_sum += relative;
          relative_values.push_back(relative);
          ++relative_count;
        }
        if (!all_complete) {
          complete_observed.push_back(y);
          complete_predicted.push_back(estimate);
        }
        ++complete;
      }
    }
    if (!complete) {
      throw std::invalid_argument("no complete regression pairs");
    }
    const long double count = static_cast<long double>(complete);
    const double rmsd = std::sqrt(static_cast<double>(sse / count));
    double median_relative = NA_REAL;
    if (!relative_values.empty()) {
      const std::size_t middle = relative_values.size() / 2;
      std::nth_element(
        relative_values.begin(), relative_values.begin() + middle,
        relative_values.end()
      );
      median_relative = relative_values[middle];
      if (relative_values.size() % 2 == 0) {
        const double lower = *std::max_element(
          relative_values.begin(), relative_values.begin() + middle
        );
        median_relative = (lower + median_relative) / 2.0;
      }
    }
    const long double pearson_left = observed_square -
      observed_sum * observed_sum / count;
    const long double pearson_right = predicted_square -
      predicted_sum * predicted_sum / count;
    const long double pearson_cross = cross_sum -
      observed_sum * predicted_sum / count;
    const long double pearson_denominator = std::sqrt(
      pearson_left * pearson_right
    );
    const double pearson = pearson_denominator > 0.0L ?
      static_cast<double>(pearson_cross / pearson_denominator) : NA_REAL;
    const double* spearman_observed = all_complete ? observed_data :
      complete_observed.data();
    const double* spearman_predicted = all_complete ? predicted_data :
      complete_predicted.data();
    const auto spearman_result = fastpls::core::spearman_correlation(
      spearman_observed, spearman_predicted, complete
    );
    const double spearman =
      spearman_result.status == fastpls::core::CorrelationStatus::success ?
      spearman_result.value : NA_REAL;
    const long double flattened_tss = observed_square -
      observed_sum * observed_sum / count;
    const double observed_sd = complete > 1 && flattened_tss >= 0.0L ?
      std::sqrt(static_cast<double>(flattened_tss / (count - 1.0L))) : NA_REAL;

    SEXP output = protect.add(Rf_allocVector(REALSXP, 12));
    const bool remove_missing = Rf_asLogical(na_rm) != FALSE;
    const double values[12] = {
      remove_missing ? static_cast<double>(complete) : static_cast<double>(size),
      observed_tss > 0.0L ? 1.0 - static_cast<double>(sse / observed_tss) : NA_REAL,
      training_data && training_tss > 0.0L ?
        1.0 - static_cast<double>(sse / training_tss) : NA_REAL,
      rmsd, rmsd, static_cast<double>(absolute_error / count),
      static_cast<double>(error_sum / count), median_relative,
      relative_count ? static_cast<double>(relative_sum / relative_count) : NA_REAL,
      std::isfinite(observed_sd) && rmsd > 0.0 ? observed_sd / rmsd : NA_REAL,
      pearson, spearman
    };
    std::copy(values, values + 12, REAL(output));
    SEXP names = protect.add(Rf_allocVector(STRSXP, 12));
    const char* labels[12] = {
      "n", "R2", "Q2", "RMSD", "RMSE", "MAE", "bias",
      "MRE_percent", "MAPE_percent", "RPD", "Pearson_r", "Spearman_r"
    };
    for (int index = 0; index < 12; ++index) {
      SET_STRING_ELT(names, index, Rf_mkChar(labels[index]));
    }
    Rf_setAttrib(output, R_NamesSymbol, names);
    return output;
  });
}

extern "C" SEXP _fastPLS_evaluate_regression_by_column_cpp(
    SEXP observed, SEXP predicted, SEXP training, SEXP relative_epsilon,
    SEXP na_rm) {
  return translate_exceptions("response-wise regression evaluation", [&] {
    if (!Rf_isMatrix(observed) || !Rf_isMatrix(predicted)) {
      throw std::invalid_argument(
        "regression evaluation requires two numeric matrices"
      );
    }
    ProtectStack protect;
    SEXP observed_real = protect.add(Rf_coerceVector(observed, REALSXP));
    SEXP predicted_real = protect.add(Rf_coerceVector(predicted, REALSXP));
    const SEXP dimensions = Rf_getAttrib(observed, R_DimSymbol);
    const SEXP predicted_dimensions = Rf_getAttrib(predicted, R_DimSymbol);
    if (TYPEOF(dimensions) != INTSXP || XLENGTH(dimensions) != 2 ||
        TYPEOF(predicted_dimensions) != INTSXP ||
        XLENGTH(predicted_dimensions) != 2 ||
        INTEGER(dimensions)[0] != INTEGER(predicted_dimensions)[0] ||
        INTEGER(dimensions)[1] != INTEGER(predicted_dimensions)[1]) {
      throw std::invalid_argument(
        "observed and predicted must have the same dimensions"
      );
    }
    const int rows = INTEGER(dimensions)[0];
    const int columns = INTEGER(dimensions)[1];
    SEXP training_real = R_NilValue;
    int training_rows = 0;
    if (training != R_NilValue) {
      if (!Rf_isMatrix(training)) {
        throw std::invalid_argument("training responses must be a matrix");
      }
      const SEXP training_dimensions = Rf_getAttrib(training, R_DimSymbol);
      if (TYPEOF(training_dimensions) != INTSXP ||
          XLENGTH(training_dimensions) != 2 ||
          INTEGER(training_dimensions)[1] != columns) {
        throw std::invalid_argument(
          "training responses must have the same number of columns"
        );
      }
      training_rows = INTEGER(training_dimensions)[0];
      training_real = protect.add(Rf_coerceVector(training, REALSXP));
    }

    SEXP output = protect.add(Rf_allocMatrix(REALSXP, columns, 12));
    std::fill(REAL(output), REAL(output) + static_cast<std::size_t>(columns) * 12,
              NA_REAL);
    const bool remove_missing = Rf_asLogical(na_rm) != FALSE;
    for (int column = 0; column < columns; ++column) {
      std::size_t complete = 0;
      for (int row = 0; row < rows; ++row) {
        const double y = REAL(observed_real)[column * rows + row];
        const double estimate = REAL(predicted_real)[column * rows + row];
        if (std::isfinite(y) && std::isfinite(estimate)) ++complete;
      }
      if (!complete) {
        REAL(output)[column] = remove_missing ? 0.0 : static_cast<double>(rows);
        continue;
      }
      int temporary_protects = 0;
      SEXP observed_column = PROTECT(Rf_allocMatrix(REALSXP, rows, 1));
      ++temporary_protects;
      SEXP predicted_column = PROTECT(Rf_allocMatrix(REALSXP, rows, 1));
      ++temporary_protects;
      std::copy_n(
        REAL(observed_real) + static_cast<std::size_t>(column) * rows, rows,
        REAL(observed_column)
      );
      std::copy_n(
        REAL(predicted_real) + static_cast<std::size_t>(column) * rows, rows,
        REAL(predicted_column)
      );
      SEXP training_column = R_NilValue;
      if (training_real != R_NilValue) {
        training_column = PROTECT(Rf_allocMatrix(REALSXP, training_rows, 1));
        ++temporary_protects;
        std::copy_n(
          REAL(training_real) + static_cast<std::size_t>(column) * training_rows,
          training_rows, REAL(training_column)
        );
      }
      SEXP values = PROTECT(_fastPLS_evaluate_regression_core_cpp(
        observed_column, predicted_column, training_column, relative_epsilon,
        na_rm
      ));
      ++temporary_protects;
      for (int metric = 0; metric < 12; ++metric) {
        REAL(output)[column + metric * columns] = REAL(values)[metric];
      }
      if (!remove_missing) {
        REAL(output)[column] = static_cast<double>(rows);
      }
      UNPROTECT(temporary_protects);
    }
    SEXP column_names = protect.add(Rf_allocVector(STRSXP, 12));
    const char* labels[12] = {
      "n", "R2", "Q2", "RMSD", "RMSE", "MAE", "bias",
      "MRE_percent", "MAPE_percent", "RPD", "Pearson_r", "Spearman_r"
    };
    for (int index = 0; index < 12; ++index) {
      SET_STRING_ELT(column_names, index, Rf_mkChar(labels[index]));
    }
    SEXP dimnames = protect.add(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(dimnames, 1, column_names);
    Rf_setAttrib(output, R_DimNamesSymbol, dimnames);
    return output;
  });
}

extern "C" SEXP _fastPLS_evaluate_is_onehot_cpp(SEXP values) {
  if (!Rf_isMatrix(values) ||
      (TYPEOF(values) != REALSXP && TYPEOF(values) != INTSXP)) {
    return Rf_ScalarLogical(FALSE);
  }
  const SEXP dimensions = Rf_getAttrib(values, R_DimSymbol);
  const int rows = INTEGER(dimensions)[0];
  const int columns = INTEGER(dimensions)[1];
  if (columns <= 1) return Rf_ScalarLogical(FALSE);
  for (int row = 0; row < rows; ++row) {
    double sum = 0.0;
    for (int column = 0; column < columns; ++column) {
      const std::size_t index = row + static_cast<std::size_t>(column) * rows;
      const double value = TYPEOF(values) == REALSXP ? REAL(values)[index] :
        (INTEGER(values)[index] == NA_INTEGER ? NA_REAL :
          static_cast<double>(INTEGER(values)[index]));
      if (std::isnan(value)) continue;
      if (!std::isfinite(value) || (value != 0.0 && value != 1.0)) {
        return Rf_ScalarLogical(FALSE);
      }
      sum += value;
    }
    if (std::abs(sum - 1.0) >= 1e-8) return Rf_ScalarLogical(FALSE);
  }
  return Rf_ScalarLogical(TRUE);
}

extern "C" SEXP _fastPLS_evaluate_class_labels_cpp(
    SEXP values, SEXP reference_levels) {
  return translate_exceptions("classification label decoding", [&] {
    ProtectStack protect;
    if (Rf_isFactor(values)) {
      const SEXP levels = Rf_getAttrib(values, R_LevelsSymbol);
      SEXP output = protect.add(Rf_allocVector(STRSXP, XLENGTH(values)));
      for (R_xlen_t index = 0; index < XLENGTH(values); ++index) {
        const int code = INTEGER(values)[index];
        SET_STRING_ELT(
          output, index,
          code == NA_INTEGER || code < 1 || code > XLENGTH(levels) ?
            NA_STRING : STRING_ELT(levels, code - 1)
        );
      }
      return output;
    }
    if (TYPEOF(values) == STRSXP && !Rf_isMatrix(values)) return values;
    if (Rf_inherits(values, "data.frame")) {
      if (XLENGTH(values) < 1) {
        throw std::invalid_argument("classification data frame is empty");
      }
      return _fastPLS_evaluate_class_labels_cpp(
        VECTOR_ELT(values, 0), reference_levels
      );
    }
    if (!Rf_isMatrix(values)) {
      return protect.add(Rf_coerceVector(values, STRSXP));
    }
    const SEXP dimensions = Rf_getAttrib(values, R_DimSymbol);
    const int rows = INTEGER(dimensions)[0];
    const int columns = INTEGER(dimensions)[1];
    if (TYPEOF(values) != REALSXP && TYPEOF(values) != INTSXP) {
      SEXP strings = protect.add(Rf_coerceVector(values, STRSXP));
      SEXP output = protect.add(Rf_allocVector(STRSXP, rows));
      for (int row = 0; row < rows; ++row) {
        SET_STRING_ELT(output, row, STRING_ELT(strings, row));
      }
      return output;
    }
    SEXP labels = R_NilValue;
    const SEXP dimnames = Rf_getAttrib(values, R_DimNamesSymbol);
    if (TYPEOF(dimnames) == VECSXP && XLENGTH(dimnames) == 2 &&
        TYPEOF(VECTOR_ELT(dimnames, 1)) == STRSXP) {
      labels = VECTOR_ELT(dimnames, 1);
    } else if (TYPEOF(reference_levels) == STRSXP &&
               XLENGTH(reference_levels) == columns) {
      labels = reference_levels;
    } else {
      labels = protect.add(Rf_allocVector(STRSXP, columns));
      for (int column = 0; column < columns; ++column) {
        SET_STRING_ELT(
          labels, column, Rf_mkChar(std::to_string(column + 1).c_str())
        );
      }
    }
    SEXP output = protect.add(Rf_allocVector(STRSXP, rows));
    for (int row = 0; row < rows; ++row) {
      int best = 0;
      double best_value = -std::numeric_limits<double>::infinity();
      for (int column = 0; column < columns; ++column) {
        const std::size_t index = row + static_cast<std::size_t>(column) * rows;
        const double value = TYPEOF(values) == REALSXP ? REAL(values)[index] :
          (INTEGER(values)[index] == NA_INTEGER ? NA_REAL :
            static_cast<double>(INTEGER(values)[index]));
        if (column == 0 || value > best_value) {
          best = column;
          best_value = value;
        }
      }
      SET_STRING_ELT(output, row, STRING_ELT(labels, best));
    }
    return output;
  });
}

extern "C" SEXP _fastPLS_evaluate_classification_core_cpp(
    SEXP observed, SEXP predicted, SEXP class_count, SEXP scores,
    SEXP score_observed, SEXP top_k) {
  return translate_exceptions("classification evaluation", [&] {
    ProtectStack protect;
    SEXP observed_integer = protect.add(Rf_coerceVector(observed, INTSXP));
    SEXP predicted_integer = protect.add(Rf_coerceVector(predicted, INTSXP));
    const int classes = Rf_asInteger(class_count);
    const R_xlen_t samples = XLENGTH(observed_integer);
    if (classes < 1 || XLENGTH(predicted_integer) != samples) {
      throw std::invalid_argument(
        "classification labels must have matching lengths and classes"
      );
    }
    std::vector<double> confusion(
      static_cast<std::size_t>(classes) * classes, 0.0
    );
    std::size_t complete = 0;
    for (R_xlen_t sample = 0; sample < samples; ++sample) {
      const int truth = INTEGER(observed_integer)[sample];
      const int estimate = INTEGER(predicted_integer)[sample];
      if (truth == NA_INTEGER || estimate == NA_INTEGER) continue;
      if (truth < 1 || truth > classes || estimate < 1 || estimate > classes) {
        throw std::invalid_argument("classification labels are out of range");
      }
      confusion[static_cast<std::size_t>(estimate - 1) +
        static_cast<std::size_t>(truth - 1) * classes] += 1.0;
      ++complete;
    }
    std::vector<double> support(classes, 0.0), predicted_support(classes, 0.0);
    std::vector<double> precision(classes, NA_REAL), recall(classes, NA_REAL);
    std::vector<double> f1(classes, NA_REAL);
    double correct = 0.0;
    for (int truth = 0; truth < classes; ++truth) {
      for (int estimate = 0; estimate < classes; ++estimate) {
        const double count = confusion[estimate + truth * classes];
        support[truth] += count;
        predicted_support[estimate] += count;
      }
      correct += confusion[truth + truth * classes];
    }
    double recall_sum = 0.0, precision_sum = 0.0, f1_sum = 0.0;
    std::size_t recall_count = 0, precision_count = 0, f1_count = 0;
    for (int cls = 0; cls < classes; ++cls) {
      const double tp = confusion[cls + cls * classes];
      if (support[cls] > 0.0) {
        recall[cls] = tp / support[cls];
        recall_sum += recall[cls];
        ++recall_count;
      }
      if (predicted_support[cls] > 0.0) {
        precision[cls] = tp / predicted_support[cls];
        precision_sum += precision[cls];
        ++precision_count;
      }
      if (std::isfinite(recall[cls]) && std::isfinite(precision[cls]) &&
          recall[cls] + precision[cls] > 0.0) {
        f1[cls] = 2.0 * recall[cls] * precision[cls] /
          (recall[cls] + precision[cls]);
        f1_sum += f1[cls];
        ++f1_count;
      }
    }
    const double n = static_cast<double>(complete);
    const double accuracy = complete ? correct / n : NA_REAL;
    const double null_rate = complete ?
      *std::max_element(support.begin(), support.end()) / n : NA_REAL;
    const double lift = std::isfinite(null_rate) && null_rate > 0.0 ?
      accuracy / null_rate : NA_REAL;
    double expected = 0.0;
    if (complete) {
      for (int cls = 0; cls < classes; ++cls) {
        expected += predicted_support[cls] * support[cls];
      }
      expected /= n * n;
    } else {
      expected = NA_REAL;
    }
    const double kappa = std::isfinite(expected) && expected < 1.0 ?
      (accuracy - expected) / (1.0 - expected) : NA_REAL;

    SEXP metrics = protect.add(Rf_allocVector(REALSXP, 9));
    const double metric_values[9] = {
      n, accuracy, null_rate, lift,
      recall_count ? recall_sum / recall_count : NA_REAL,
      precision_count ? precision_sum / precision_count : NA_REAL,
      recall_count ? recall_sum / recall_count : NA_REAL,
      f1_count ? f1_sum / f1_count : NA_REAL, kappa
    };
    std::copy(metric_values, metric_values + 9, REAL(metrics));
    SEXP metric_names = protect.add(Rf_allocVector(STRSXP, 9));
    const char* metric_labels[9] = {
      "n", "accuracy", "no_information_rate", "lift_accuracy",
      "balanced_accuracy", "macro_precision", "macro_recall",
      "macro_f1", "kappa"
    };
    for (int index = 0; index < 9; ++index) {
      SET_STRING_ELT(metric_names, index, Rf_mkChar(metric_labels[index]));
    }
    Rf_setAttrib(metrics, R_NamesSymbol, metric_names);

    SEXP per_class = protect.add(Rf_allocMatrix(REALSXP, classes, 4));
    for (int cls = 0; cls < classes; ++cls) {
      REAL(per_class)[cls] = support[cls];
      REAL(per_class)[cls + classes] = precision[cls];
      REAL(per_class)[cls + 2 * classes] = recall[cls];
      REAL(per_class)[cls + 3 * classes] = f1[cls];
    }
    SEXP per_class_names = protect.add(Rf_allocVector(STRSXP, 4));
    const char* class_labels[4] = {"support", "precision", "recall", "f1"};
    for (int index = 0; index < 4; ++index) {
      SET_STRING_ELT(per_class_names, index, Rf_mkChar(class_labels[index]));
    }
    SEXP per_class_dimnames = protect.add(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(per_class_dimnames, 1, per_class_names);
    Rf_setAttrib(per_class, R_DimNamesSymbol, per_class_dimnames);

    SEXP confusion_matrix = protect.add(Rf_allocMatrix(INTSXP, classes, classes));
    for (std::size_t index = 0; index < confusion.size(); ++index) {
      INTEGER(confusion_matrix)[index] = static_cast<int>(confusion[index]);
    }

    SEXP top_accuracy = R_NilValue;
    if (scores != R_NilValue && XLENGTH(top_k) > 0) {
      if (!Rf_isMatrix(scores) ||
          (TYPEOF(scores) != REALSXP && TYPEOF(scores) != INTSXP)) {
        throw std::invalid_argument("classification scores must be numeric");
      }
      SEXP score_real = protect.add(Rf_coerceVector(scores, REALSXP));
      SEXP score_truth = protect.add(Rf_coerceVector(score_observed, INTSXP));
      const SEXP score_dimensions = Rf_getAttrib(scores, R_DimSymbol);
      const int score_rows = INTEGER(score_dimensions)[0];
      const int score_columns = INTEGER(score_dimensions)[1];
      if (XLENGTH(score_truth) != score_rows) {
        throw std::invalid_argument("top-k labels must match score rows");
      }
      top_accuracy = protect.add(Rf_allocVector(REALSXP, XLENGTH(top_k)));
      std::vector<int> order(static_cast<std::size_t>(score_columns));
      for (R_xlen_t request = 0; request < XLENGTH(top_k); ++request) {
        const int requested = INTEGER(top_k)[request];
        if (requested == NA_INTEGER) {
          REAL(top_accuracy)[request] = NA_REAL;
          continue;
        }
        const int keep = std::min(std::max(requested, 1), score_columns);
        std::size_t valid = 0, hits = 0;
        for (int row = 0; row < score_rows; ++row) {
          const int truth = INTEGER(score_truth)[row];
          if (truth == NA_INTEGER || truth < 1 || truth > score_columns) continue;
          for (int column = 0; column < score_columns; ++column) {
            order[column] = column;
          }
          std::partial_sort(
            order.begin(), order.begin() + keep, order.end(),
            [&](int left, int right) {
              const double l = REAL(score_real)[row + left * score_rows];
              const double r = REAL(score_real)[row + right * score_rows];
              if (l == r) return left < right;
              return l > r;
            }
          );
          ++valid;
          if (std::find(order.begin(), order.begin() + keep, truth - 1) !=
              order.begin() + keep) ++hits;
        }
        REAL(top_accuracy)[request] = valid ?
          static_cast<double>(hits) / valid : NA_REAL;
      }
    }

    SEXP output = protect.add(Rf_allocVector(VECSXP, 4));
    SET_VECTOR_ELT(output, 0, metrics);
    SET_VECTOR_ELT(output, 1, per_class);
    SET_VECTOR_ELT(output, 2, confusion_matrix);
    SET_VECTOR_ELT(output, 3, top_accuracy);
    SEXP names = protect.add(Rf_allocVector(STRSXP, 4));
    const char* output_labels[4] = {
      "metrics", "per_class", "confusion", "top_accuracy"
    };
    for (int index = 0; index < 4; ++index) {
      SET_STRING_ELT(names, index, Rf_mkChar(output_labels[index]));
    }
    Rf_setAttrib(output, R_NamesSymbol, names);
    return output;
  });
}

template<class Scalar>
SEXP vip_values(const fastpls::core::Matrix<Scalar>& response_loadings,
                const fastpls::core::Matrix<Scalar>& scores,
                const fastpls::core::Matrix<Scalar>& weights) {
  if (scores.columns() != weights.columns() ||
      response_loadings.columns() != weights.columns()) {
    throw std::invalid_argument("VIP model component dimensions do not match");
  }
  const std::size_t components = weights.columns();
  const std::size_t predictors = weights.rows();
  std::vector<long double> score_squares(components, 0.0L);
  std::vector<long double> weight_squares(components, 0.0L);
  for (std::size_t component = 0; component < components; ++component) {
    for (std::size_t row = 0; row < scores.rows(); ++row) {
      const long double value = scores(row, component);
      score_squares[component] += value * value;
    }
    for (std::size_t row = 0; row < predictors; ++row) {
      const long double value = weights(row, component);
      weight_squares[component] += value * value;
    }
  }
  ProtectStack protect;
  SEXP output = protect.add(Rf_allocVector(
    VECSXP, response_loadings.rows()
  ));
  for (std::size_t response = 0; response < response_loadings.rows();
       ++response) {
    SEXP value = protect.add(Rf_allocMatrix(
      REALSXP, static_cast<int>(components), static_cast<int>(predictors)
    ));
    std::vector<long double> numerator(predictors, 0.0L);
    long double denominator = 0.0L;
    for (std::size_t component = 0; component < components; ++component) {
      const long double loading = response_loadings(response, component);
      const long double explained = loading * loading * score_squares[component];
      denominator += explained;
      for (std::size_t predictor = 0; predictor < predictors; ++predictor) {
        const long double weight = weights(predictor, component);
        if (weight_squares[component] > 0.0L) {
          numerator[predictor] += weight * weight * explained /
            weight_squares[component];
        }
        REAL(value)[component + predictor * components] = denominator > 0.0L ?
          std::sqrt(static_cast<double>(predictors * numerator[predictor] /
                                       denominator)) : NA_REAL;
      }
    }
    SET_VECTOR_ELT(output, response, value);
  }
  if (response_loadings.rows() == 1) return VECTOR_ELT(output, 0);
  return output;
}

extern "C" SEXP _fastPLS_vip_core_cpp(SEXP model) {
  return translate_exceptions("VIP calculation", [&] {
    if (TYPEOF(model) != VECSXP) {
      throw std::invalid_argument("VIP requires a fitted fastPLS model");
    }
    SEXP q = list_element(model, "Q");
    SEXP scores = list_element(model, "Ttrain");
    SEXP weights = list_element(model, "R");
    if (scores == R_NilValue || XLENGTH(scores) == 0) {
      throw std::invalid_argument("VIP requires a model fitted with fit = TRUE");
    }
    if (Rf_isS4(q)) {
      return vip_values(
        float_matrix_from_s4(q, "model$Q"),
        float_matrix_from_s4(scores, "model$Ttrain"),
        float_matrix_from_s4(weights, "model$R")
      );
    }
    return vip_values(
      numeric_matrix_from_sexp(q, "model$Q"),
      numeric_matrix_from_sexp(scores, "model$Ttrain"),
      numeric_matrix_from_sexp(weights, "model$R")
    );
  });
}

template<class Scalar>
fastpls::core::Matrix<Scalar> normalized_units(
    const fastpls::core::Matrix<Scalar>& input, bool by_row) {
  const std::size_t units = by_row ? input.rows() : input.columns();
  const std::size_t features = by_row ? input.columns() : input.rows();
  fastpls::core::Matrix<Scalar> output(units, features);
  for (std::size_t unit = 0; unit < units; ++unit) {
    long double mean = 0.0L;
    bool finite = true;
    for (std::size_t feature = 0; feature < features; ++feature) {
      const double value = by_row ? input(unit, feature) : input(feature, unit);
      if (!std::isfinite(value)) finite = false;
      mean += value;
    }
    mean /= static_cast<long double>(features);
    long double sum_squares = 0.0L;
    for (std::size_t feature = 0; feature < features; ++feature) {
      const long double value =
        (by_row ? input(unit, feature) : input(feature, unit)) - mean;
      sum_squares += value * value;
    }
    const long double norm = std::sqrt(sum_squares);
    for (std::size_t feature = 0; feature < features; ++feature) {
      const long double value =
        (by_row ? input(unit, feature) : input(feature, unit)) - mean;
      output(unit, feature) = finite && norm > 0.0L ?
        static_cast<Scalar>(value / norm) :
        std::numeric_limits<Scalar>::quiet_NaN();
    }
  }
  return output;
}

template<class Scalar>
SEXP fastcor_values(const fastpls::core::Matrix<Scalar>& left_input,
                    const fastpls::core::Matrix<Scalar>* right_input,
                    bool by_row, bool diagonal) {
  const auto left = normalized_units(left_input, by_row);
  fastpls::core::Matrix<Scalar> right;
  const fastpls::core::Matrix<Scalar>* right_values = nullptr;
  if (right_input != nullptr) {
    right = normalized_units(*right_input, by_row);
    right_values = &right;
    if (left.columns() != right.columns()) {
      throw std::invalid_argument(
        "a and b must contain the same number of correlated features"
      );
    }
  } else {
    right_values = &left;
    diagonal = false;
  }
  if (diagonal) {
    if (left.rows() != right_values->rows()) {
      throw std::invalid_argument(
        "diag = TRUE requires matching numbers of rows or columns"
      );
    }
    SEXP output = PROTECT(Rf_allocVector(REALSXP, left.rows()));
    for (std::size_t row = 0; row < left.rows(); ++row) {
      long double sum = 0.0L;
      for (std::size_t column = 0; column < left.columns(); ++column) {
        sum += static_cast<long double>(left(row, column)) *
          (*right_values)(row, column);
      }
      REAL(output)[row] = static_cast<double>(sum);
    }
    UNPROTECT(1);
    return output;
  }
  fastpls::core::Matrix<Scalar> product(left.rows(), right_values->rows());
  if constexpr (std::is_same_v<Scalar, float>) {
    fastpls::runtime::cpu_gemm_f32(
      left.view(), right_values->view(), false, true, product.view()
    );
  } else {
    fastpls::runtime::cpu_gemm_f64(
      left.view(), right_values->view(), false, true, product.view()
    );
  }
  SEXP output = PROTECT(Rf_allocMatrix(
    REALSXP, static_cast<int>(product.rows()),
    static_cast<int>(product.columns())
  ));
  for (std::size_t index = 0; index < product.size(); ++index) {
    REAL(output)[index] = static_cast<double>(product.data()[index]);
  }
  UNPROTECT(1);
  return output;
}

extern "C" SEXP _fastPLS_fastcor_core_cpp(SEXP left, SEXP right,
                                           SEXP by_row, SEXP diagonal) {
  return translate_exceptions("fast Pearson correlation", [&] {
    const bool rows = Rf_asLogical(by_row) != FALSE;
    const bool diag = Rf_asLogical(diagonal) != FALSE;
    if (Rf_isS4(left)) {
      const auto left_values = float_matrix_from_s4(left, "a");
      if (right == R_NilValue) {
        return fastcor_values(left_values, static_cast<const
          fastpls::core::Matrix<float>*>(nullptr), rows, diag);
      }
      const auto right_values = float_matrix_from_s4(right, "b");
      return fastcor_values(left_values, &right_values, rows, diag);
    }
    const auto left_values = numeric_matrix_from_sexp(left, "a");
    if (right == R_NilValue) {
      return fastcor_values(left_values, static_cast<const
        fastpls::core::Matrix<double>*>(nullptr), rows, diag);
    }
    const auto right_values = numeric_matrix_from_sexp(right, "b");
    return fastcor_values(left_values, &right_values, rows, diag);
  });
}

extern "C" SEXP _fastPLS_float32_argmax_cpp(SEXP scores) {
  try {
    const fastpls::core::Matrix<float> values =
      float_matrix_from_s4(scores, "scores");
    SEXP result = PROTECT(Rf_allocVector(INTSXP, values.rows()));
    for (std::size_t row = 0; row < values.rows(); ++row) {
      INTEGER(result)[row] = static_cast<int>(
        fastpls::core::row_argmax(values.view(), row) + 1
      );
    }
    UNPROTECT(1);
    return result;
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  } catch (...) {
    Rf_error("Unknown error in float32 argmax");
  }
  return R_NilValue;
}

extern "C" SEXP _fastPLS_float32_topk_cpp(SEXP scores, SEXP top) {
  try {
    const int requested = Rf_asInteger(top);
    if (requested == NA_INTEGER || requested < 1) {
      throw std::invalid_argument("top must be a positive integer");
    }
    const fastpls::core::Matrix<float> values =
      float_matrix_from_s4(scores, "scores");
    const std::size_t keep = std::min<std::size_t>(
      static_cast<std::size_t>(requested), values.columns()
    );
    SEXP index = PROTECT(Rf_allocMatrix(
      INTSXP, static_cast<int>(values.rows()), static_cast<int>(keep)
    ));
    SEXP value = PROTECT(Rf_allocMatrix(
      REALSXP, static_cast<int>(values.rows()), static_cast<int>(keep)
    ));
    std::vector<std::size_t> workspace;
    std::vector<std::size_t> row_indices(keep);
    std::vector<float> row_scores(keep);
    for (std::size_t row = 0; row < values.rows(); ++row) {
      fastpls::core::row_top_k(
        values.view(), row, keep, workspace, row_indices.data(),
        row_scores.data()
      );
      for (std::size_t rank = 0; rank < keep; ++rank) {
        const std::size_t offset = row + rank * values.rows();
        INTEGER(index)[offset] = static_cast<int>(row_indices[rank] + 1);
        REAL(value)[offset] = static_cast<double>(row_scores[rank]);
      }
    }
    SEXP result = PROTECT(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(result, 0, index);
    SET_VECTOR_ELT(result, 1, value);
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, Rf_mkChar("top_index"));
    SET_STRING_ELT(names, 1, Rf_mkChar("top_score"));
    Rf_setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(4);
    return result;
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  } catch (...) {
    Rf_error("Unknown error in float32 top-rank selection");
  }
  return R_NilValue;
}

extern "C" SEXP _fastPLS_lda_train_prefix_float32_cpp(
    SEXP scores, SEXP labels, SEXP class_count, SEXP components) {
  try {
    if (TYPEOF(labels) != INTSXP || TYPEOF(components) != INTSXP) {
      throw std::invalid_argument(
        "float32 PLS-LDA labels and component counts must be integer"
      );
    }
    const int classes = Rf_asInteger(class_count);
    if (classes < 2 || XLENGTH(components) < 1) {
      throw std::invalid_argument(
        "float32 PLS-LDA requires at least two classes and one component count"
      );
    }
    const fastpls::core::Matrix<float> values =
      float_matrix_from_s4(scores, "Ttrain");
    if (XLENGTH(labels) != static_cast<R_xlen_t>(values.rows())) {
      throw std::invalid_argument(
        "float32 PLS-LDA requires one label per score row"
      );
    }
    fastpls::core::Matrix<float> gram(values.columns(), values.columns());
    fastpls::runtime::cpu_crossprod_f32(values.view(), gram.view());
    fastpls::core::Matrix<float> class_sums(
      static_cast<std::size_t>(classes), values.columns()
    );
    std::vector<float> counts(static_cast<std::size_t>(classes), 0.0f);
    for (std::size_t row = 0; row < values.rows(); ++row) {
      const int encoded = INTEGER(labels)[row] - 1;
      if (encoded < 0 || encoded >= classes) {
        throw std::invalid_argument(
          "float32 PLS-LDA labels must be encoded as 1..n_classes"
        );
      }
      const std::size_t class_index = static_cast<std::size_t>(encoded);
      counts[class_index] += 1.0f;
      for (std::size_t component = 0;
           component < values.columns(); ++component) {
        class_sums(class_index, component) += values(row, component);
      }
    }
    const auto models = fastpls::core::train_lda_prefixes_from_moments<float>(
      gram.view(), class_sums.view(), counts.data(), counts.size(),
      values.rows(), INTEGER(components),
      static_cast<std::size_t>(XLENGTH(components))
    );
    return float_lda_models(models, INTEGER(components));
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  } catch (...) {
    Rf_error("Unknown error in float32 PLS-LDA fitting");
  }
  return R_NilValue;
}

extern "C" SEXP _fastPLS_lda_project_train_prefix_float32_cpp(
    SEXP model, SEXP predictors, SEXP labels, SEXP class_count,
    SEXP components) {
  return translate_exceptions("projected float32 PLS-LDA fitting", [&] {
    if (TYPEOF(labels) != INTSXP || TYPEOF(components) != INTSXP ||
        XLENGTH(components) < 1) {
      throw std::invalid_argument(
        "projected float32 PLS-LDA requires integer labels and components"
      );
    }
    const int classes = Rf_asInteger(class_count);
    if (classes < 2) {
      throw std::invalid_argument(
        "projected float32 PLS-LDA requires at least two classes"
      );
    }
    const auto x = float_matrix_from_s4(predictors, "Xtrain");
    const auto projection = float_matrix_from_s4(
      list_element(model, "R"), "model$R"
    );
    const auto center = float_matrix_from_s4(
      list_element(model, "mX"), "model$mX"
    );
    const auto scale = float_matrix_from_s4(
      list_element(model, "vX"), "model$vX"
    );
    if (x.rows() != static_cast<std::size_t>(XLENGTH(labels)) ||
        x.columns() != projection.rows() || center.size() != x.columns() ||
        scale.size() != x.columns()) {
      throw std::invalid_argument(
        "projected float32 PLS-LDA dimensions are inconsistent"
      );
    }
    std::size_t maximum = 0;
    for (R_xlen_t index = 0; index < XLENGTH(components); ++index) {
      const int count = INTEGER(components)[index];
      if (count < 1 ||
          static_cast<std::size_t>(count) > projection.columns()) {
        throw std::invalid_argument(
          "projected float32 PLS-LDA components exceed model rank"
        );
      }
      maximum = std::max(maximum, static_cast<std::size_t>(count));
    }

    fastpls::core::Matrix<float> scaled_projection(x.columns(), maximum);
    std::vector<float> offset(maximum, 0.0f);
    for (std::size_t component = 0; component < maximum; ++component) {
      for (std::size_t predictor = 0; predictor < x.columns(); ++predictor) {
        const float divisor = scale.data()[predictor];
        if (!std::isfinite(divisor) || divisor == 0.0f) {
          throw std::invalid_argument(
            "projected float32 PLS-LDA contains an invalid predictor scale"
          );
        }
        const float weight = projection(predictor, component) / divisor;
        scaled_projection(predictor, component) = weight;
        offset[component] += center.data()[predictor] * weight;
      }
    }
    fastpls::core::Matrix<float> scores(x.rows(), maximum);
    fastpls::runtime::cpu_gemm_f32(
      x.view(), scaled_projection.view(), false, false, scores.view()
    );
    for (std::size_t component = 0; component < maximum; ++component) {
      for (std::size_t row = 0; row < scores.rows(); ++row) {
        scores(row, component) -= offset[component];
      }
    }

    fastpls::core::Matrix<float> gram(maximum, maximum);
    fastpls::runtime::cpu_crossprod_f32(scores.view(), gram.view());
    fastpls::core::Matrix<float> class_sums(
      static_cast<std::size_t>(classes), maximum
    );
    std::vector<float> counts(static_cast<std::size_t>(classes), 0.0f);
    for (std::size_t row = 0; row < scores.rows(); ++row) {
      const int encoded = INTEGER(labels)[row] - 1;
      if (encoded < 0 || encoded >= classes) {
        throw std::invalid_argument(
          "projected float32 PLS-LDA labels must be encoded as 1..n_classes"
        );
      }
      const std::size_t class_index = static_cast<std::size_t>(encoded);
      counts[class_index] += 1.0f;
      for (std::size_t component = 0; component < maximum; ++component) {
        class_sums(class_index, component) += scores(row, component);
      }
    }
    const auto models = fastpls::core::train_lda_prefixes_from_moments<float>(
      gram.view(), class_sums.view(), counts.data(), counts.size(),
      scores.rows(), INTEGER(components),
      static_cast<std::size_t>(XLENGTH(components))
    );
    ProtectStack protect;
    SEXP output = protect.add(Rf_allocVector(VECSXP, 2));
    SEXP fitted = protect.add(float_lda_models(
      models, INTEGER(components)
    ));
    SEXP retained_scores = protect.add(float_bits_matrix(scores));
    SEXP names = protect.add(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, Rf_mkChar("models"));
    SET_STRING_ELT(names, 1, Rf_mkChar("Ttrain"));
    SET_VECTOR_ELT(output, 0, fitted);
    SET_VECTOR_ELT(output, 1, retained_scores);
    Rf_setAttrib(output, R_NamesSymbol, names);
    return output;
  });
}

extern "C" SEXP _fastPLS_lda_predict_float32_cpp(
    SEXP scores, SEXP model, SEXP return_scores) {
  try {
    const int retain = Rf_asLogical(return_scores);
    if (retain == NA_LOGICAL) {
      throw std::invalid_argument("return_scores must be TRUE or FALSE");
    }
    const fastpls::core::Matrix<float> values =
      float_matrix_from_s4(scores, "Ttest");
    fastpls::core::LdaModel<float> fitted;
    fitted.linear = float_matrix_from_bits(
      list_element(model, "linear"), "lda$linear"
    );
    const fastpls::core::Matrix<float> constants = float_matrix_from_bits(
      list_element(model, "constants"), "lda$constants"
    );
    fitted.constants.assign(
      constants.data(), constants.data() + constants.size()
    );
    const fastpls::core::Matrix<float> discriminants =
      fastpls::core::lda_scores(values.view(), fitted);
    SEXP prediction = PROTECT(Rf_allocVector(INTSXP, values.rows()));
    for (std::size_t row = 0; row < values.rows(); ++row) {
      INTEGER(prediction)[row] = static_cast<int>(
        fastpls::core::row_argmax(discriminants.view(), row) + 1
      );
    }
    SEXP output = PROTECT(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(output, 0, prediction);
    SET_VECTOR_ELT(
      output, 1, retain == TRUE ? float_bits_matrix(discriminants) : R_NilValue
    );
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, Rf_mkChar("pred"));
    SET_STRING_ELT(names, 1, Rf_mkChar("scores"));
    Rf_setAttrib(output, R_NamesSymbol, names);
    UNPROTECT(3);
    return output;
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  } catch (...) {
    Rf_error("Unknown error in float32 PLS-LDA prediction");
  }
  return R_NilValue;
}

extern "C" SEXP _fastPLS_float32_sweep_cols_cpp(SEXP matrix,
                                                  SEXP statistics,
                                                  SEXP operation) {
  if (!Rf_isS4(matrix) || !Rf_inherits(matrix, "float32") ||
      !Rf_isS4(statistics) || !Rf_inherits(statistics, "float32")) {
    Rf_error("float32 column operations require float32 inputs");
  }
  const int operation_code = Rf_asInteger(operation);
  if (operation_code < 0 || operation_code > 2 ||
      operation_code == NA_INTEGER) {
    Rf_error("Unknown float32 column operation");
  }
  SEXP matrix_bits = PROTECT(R_do_slot(matrix, Rf_install("Data")));
  SEXP statistic_bits = PROTECT(R_do_slot(statistics, Rf_install("Data")));
  const SEXP dimensions = Rf_getAttrib(matrix_bits, R_DimSymbol);
  if (TYPEOF(matrix_bits) != INTSXP || TYPEOF(statistic_bits) != INTSXP ||
      TYPEOF(dimensions) != INTSXP || XLENGTH(dimensions) != 2) {
    UNPROTECT(2);
    Rf_error("float32 inputs contain invalid Data slots");
  }
  const int rows = INTEGER(dimensions)[0];
  const int columns = INTEGER(dimensions)[1];
  if (XLENGTH(statistic_bits) != columns) {
    UNPROTECT(2);
    Rf_error("float32 column statistics must have length ncol(X)");
  }
  SEXP result = PROTECT(Rf_allocMatrix(INTSXP, rows, columns));
  const int* input = INTEGER(matrix_bits);
  int* output = INTEGER(result);
  for (int column = 0; column < columns; ++column) {
    const float statistic = decode_float32(INTEGER(statistic_bits)[column]);
    const R_xlen_t offset = static_cast<R_xlen_t>(rows) * column;
    for (int row = 0; row < rows; ++row) {
      float value = decode_float32(input[offset + row]);
      if (operation_code == 0) value -= statistic;
      else if (operation_code == 1) value /= statistic;
      else value += statistic;
      output[offset + row] = encode_float32(value);
    }
  }
  UNPROTECT(3);
  return result;
}

extern "C" SEXP _fastPLS_float32_standardize_cpp(SEXP matrix,
                                                   SEXP center,
                                                   SEXP scale) {
  for (SEXP input : {matrix, center, scale}) {
    if (!Rf_isS4(input) || !Rf_inherits(input, "float32")) {
      Rf_error("float32 standardization requires float32 inputs");
    }
  }
  SEXP matrix_bits = PROTECT(R_do_slot(matrix, Rf_install("Data")));
  SEXP center_bits = PROTECT(R_do_slot(center, Rf_install("Data")));
  SEXP scale_bits = PROTECT(R_do_slot(scale, Rf_install("Data")));
  const SEXP dimensions = Rf_getAttrib(matrix_bits, R_DimSymbol);
  if (TYPEOF(matrix_bits) != INTSXP || TYPEOF(center_bits) != INTSXP ||
      TYPEOF(scale_bits) != INTSXP || TYPEOF(dimensions) != INTSXP ||
      XLENGTH(dimensions) != 2) {
    UNPROTECT(3);
    Rf_error("float32 inputs contain invalid Data slots");
  }
  const int rows = INTEGER(dimensions)[0];
  const int columns = INTEGER(dimensions)[1];
  if (XLENGTH(center_bits) != columns || XLENGTH(scale_bits) != columns) {
    UNPROTECT(3);
    Rf_error("float32 column statistics must have length ncol(X)");
  }
  SEXP result = PROTECT(Rf_allocMatrix(INTSXP, rows, columns));
  const int* input = INTEGER(matrix_bits);
  int* output = INTEGER(result);
  for (int column = 0; column < columns; ++column) {
    const float mean = decode_float32(INTEGER(center_bits)[column]);
    const float divisor = decode_float32(INTEGER(scale_bits)[column]);
    const R_xlen_t offset = static_cast<R_xlen_t>(rows) * column;
    for (int row = 0; row < rows; ++row) {
      float value = decode_float32(input[offset + row]);
      value -= mean;
      value /= divisor;
      output[offset + row] = encode_float32(value);
    }
  }
  UNPROTECT(4);
  return result;
}

extern "C" SEXP _fastPLS_center_kernel_train_float32_cpp(SEXP kernel) {
  try {
    fastpls::core::Matrix<float> values = float_matrix_from_s4(kernel, "K");
    const auto centered = fastpls::core::center_kernel_train(values.view());
    SEXP result = PROTECT(Rf_allocVector(VECSXP, 3));
    SEXP centered_matrix = PROTECT(float_bits_matrix(values));
    fastpls::core::Matrix<float> means(1, centered.column_means.size());
    std::copy(centered.column_means.begin(), centered.column_means.end(),
              means.data());
    SEXP mean_matrix = PROTECT(float_bits_matrix(means));
    SET_VECTOR_ELT(result, 0, centered_matrix);
    SET_VECTOR_ELT(result, 1, mean_matrix);
    SET_VECTOR_ELT(result, 2, Rf_ScalarReal(centered.grand_mean));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 3));
    SET_STRING_ELT(names, 0, Rf_mkChar("K"));
    SET_STRING_ELT(names, 1, Rf_mkChar("col_means"));
    SET_STRING_ELT(names, 2, Rf_mkChar("grand_mean"));
    Rf_setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(4);
    return result;
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP _fastPLS_center_kernel_test_float32_cpp(
    SEXP kernel, SEXP training_means, SEXP training_grand_mean) {
  try {
    fastpls::core::Matrix<float> values =
      float_matrix_from_s4(kernel, "Ktest");
    const fastpls::core::Matrix<float> means =
      float_matrix_from_s4(training_means, "train_col_means");
    fastpls::core::center_kernel_test(
      values.view(), means.data(), means.size(),
      static_cast<float>(Rf_asReal(training_grand_mean))
    );
    SEXP output = PROTECT(Rf_allocVector(VECSXP, 1));
    SEXP centered = PROTECT(float_bits_matrix(values));
    SET_VECTOR_ELT(output, 0, centered);
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 1));
    SET_STRING_ELT(names, 0, Rf_mkChar("K"));
    Rf_setAttrib(output, R_NamesSymbol, names);
    UNPROTECT(3);
    return output;
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP _fastPLS_center_kernel_train_cpp(SEXP kernel) {
  try {
    fastpls::core::Matrix<double> values = numeric_matrix_from_sexp(kernel, "K");
    const auto centered = fastpls::core::center_kernel_train(values.view());
    SEXP result = PROTECT(Rf_allocVector(VECSXP, 3));
    SEXP centered_matrix = PROTECT(numeric_matrix(values));
    SEXP means = PROTECT(Rf_allocMatrix(
      REALSXP, 1, static_cast<int>(centered.column_means.size())
    ));
    std::copy(centered.column_means.begin(), centered.column_means.end(),
              REAL(means));
    SET_VECTOR_ELT(result, 0, centered_matrix);
    SET_VECTOR_ELT(result, 1, means);
    SET_VECTOR_ELT(result, 2, Rf_ScalarReal(centered.grand_mean));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 3));
    SET_STRING_ELT(names, 0, Rf_mkChar("K"));
    SET_STRING_ELT(names, 1, Rf_mkChar("col_means"));
    SET_STRING_ELT(names, 2, Rf_mkChar("grand_mean"));
    Rf_setAttrib(result, R_NamesSymbol, names);
    UNPROTECT(4);
    return result;
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP _fastPLS_cpu_float32_matrix_multiply_cpp(
    SEXP left, SEXP right, SEXP transpose_left, SEXP transpose_right) {
  try {
    const int transpose_left_value = Rf_asLogical(transpose_left);
    const int transpose_right_value = Rf_asLogical(transpose_right);
    if (transpose_left_value == NA_LOGICAL ||
        transpose_right_value == NA_LOGICAL) {
      throw std::invalid_argument("transpose controls must be TRUE or FALSE");
    }
    const fastpls::core::Matrix<float> left_values =
      float_matrix_from_s4(left, "A");
    const fastpls::core::Matrix<float> right_values =
      float_matrix_from_s4(right, "B");
    const fastpls::core::Matrix<float> product = backend_gemm_f32(
      left_values.view(), right_values.view(), transpose_left_value,
      transpose_right_value, 0
    );

    SEXP output = PROTECT(Rf_allocVector(VECSXP, 1));
    SEXP value = PROTECT(float_bits_matrix(product));
    SET_VECTOR_ELT(output, 0, value);
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 1));
    SET_STRING_ELT(names, 0, Rf_mkChar("C"));
    Rf_setAttrib(output, R_NamesSymbol, names);
    UNPROTECT(3);
    return output;
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP _fastPLS_metal_float32_matrix_multiply_cpp(
    SEXP left, SEXP right, SEXP transpose_left, SEXP transpose_right) {
  try {
    const int transpose_left_value = Rf_asLogical(transpose_left);
    const int transpose_right_value = Rf_asLogical(transpose_right);
    if (transpose_left_value == NA_LOGICAL ||
        transpose_right_value == NA_LOGICAL) {
      throw std::invalid_argument("transpose controls must be TRUE or FALSE");
    }
    const fastpls::core::Matrix<float> left_values =
      float_matrix_from_s4(left, "A");
    const fastpls::core::Matrix<float> right_values =
      float_matrix_from_s4(right, "B");
    const fastpls::core::Matrix<float> product = backend_gemm_f32(
      left_values.view(), right_values.view(), transpose_left_value,
      transpose_right_value, 2
    );

    SEXP output = PROTECT(Rf_allocVector(VECSXP, 1));
    SEXP value = PROTECT(float_bits_matrix(product));
    SET_VECTOR_ELT(output, 0, value);
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 1));
    SET_STRING_ELT(names, 0, Rf_mkChar("C"));
    Rf_setAttrib(output, R_NamesSymbol, names);
    UNPROTECT(3);
    return output;
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP _fastPLS_kernel_matrix_float32_cpp(
    SEXP left, SEXP right, SEXP kernel, SEXP gamma, SEXP degree,
    SEXP offset, SEXP backend) {
  try {
    const fastpls::core::Matrix<float> left_values =
      float_matrix_from_s4(left, "X1");
    const fastpls::core::Matrix<float> right_values =
      float_matrix_from_s4(right, "X2");
    if (left_values.columns() != right_values.columns()) {
      throw std::invalid_argument(
        "X1 and X2 must have the same number of columns"
      );
    }
    const int kernel_code = Rf_asInteger(kernel);
    if (kernel_code < 1 || kernel_code > 3 || kernel_code == NA_INTEGER) {
      throw std::invalid_argument("Unknown kernel type");
    }
    const int polynomial_degree = Rf_asInteger(degree);
    if (polynomial_degree == NA_INTEGER) {
      throw std::invalid_argument("Polynomial degree must be an integer");
    }
    const int backend_code = Rf_asInteger(backend);
    if (backend_code < 0 || backend_code > 2 || backend_code == NA_INTEGER) {
      throw std::invalid_argument("float32 kernel backend must be 0, 1, or 2");
    }

    fastpls::core::Matrix<float> result = backend_gemm_f32(
      left_values.view(), right_values.view(), false, true, backend_code
    );
    fastpls::core::kernel_from_dots(
      left_values.view(), right_values.view(), result.view(),
      static_cast<fastpls::core::KernelType>(kernel_code),
      static_cast<float>(Rf_asReal(gamma)), polynomial_degree,
      static_cast<float>(Rf_asReal(offset))
    );

    SEXP output = PROTECT(Rf_allocVector(VECSXP, 1));
    SEXP value = PROTECT(float_bits_matrix(result));
    SET_VECTOR_ELT(output, 0, value);
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 1));
    SET_STRING_ELT(names, 0, Rf_mkChar("K"));
    Rf_setAttrib(output, R_NamesSymbol, names);
    UNPROTECT(3);
    return output;
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP _fastPLS_opls_apply_filter_float32_cpp(
    SEXP matrix, SEXP center, SEXP scale, SEXP weights, SEXP loadings,
    SEXP backend) {
  try {
    fastpls::core::Matrix<float> values =
      float_matrix_from_s4(matrix, "X");
    const fastpls::core::Matrix<float> center_values =
      float_matrix_from_s4(center, "mX");
    const fastpls::core::Matrix<float> scale_values =
      float_matrix_from_s4(scale, "vX");
    if (center_values.size() != values.columns() ||
        scale_values.size() != values.columns()) {
      throw std::invalid_argument(
        "X columns must match stored OPLS preprocessing"
      );
    }
    for (std::size_t column = 0; column < values.columns(); ++column) {
      const float denominator = scale_values.data()[column];
      for (std::size_t row = 0; row < values.rows(); ++row) {
        values(row, column) =
          (values(row, column) - center_values.data()[column]) / denominator;
      }
    }

    const fastpls::core::Matrix<float> weight_values =
      float_matrix_from_s4_allow_empty(weights, "W_orth");
    const fastpls::core::Matrix<float> loading_values =
      float_matrix_from_s4_allow_empty(loadings, "P_orth");
    if (weight_values.columns() != loading_values.columns() ||
        weight_values.rows() != values.columns() ||
        loading_values.rows() != values.columns()) {
      throw std::invalid_argument("Invalid OPLS orthogonal filter dimensions");
    }

    const int backend_code = Rf_asInteger(backend);
    if (backend_code < 0 || backend_code > 2 || backend_code == NA_INTEGER) {
      throw std::invalid_argument("float32 OPLS backend must be 0, 1, or 2");
    }
    for (std::size_t component = 0;
         component < weight_values.columns(); ++component) {
      const auto weight = fastpls::core::make_const_view(
        weight_values.data() + component * weight_values.rows(),
        weight_values.rows(), 1, weight_values.rows()
      );
      const auto loading = fastpls::core::make_const_view(
        loading_values.data() + component * loading_values.rows(),
        loading_values.rows(), 1, loading_values.rows()
      );
      const fastpls::core::Matrix<float> score = backend_gemm_f32(
        values.view(), weight, false, false, backend_code
      );
      const fastpls::core::Matrix<float> correction = backend_gemm_f32(
        score.view(), loading, false, true, backend_code
      );
      for (std::size_t index = 0; index < values.size(); ++index) {
        values.data()[index] -= correction.data()[index];
      }
    }

    SEXP output = PROTECT(Rf_allocVector(VECSXP, 1));
    SEXP filtered = PROTECT(float_bits_matrix(values));
    SET_VECTOR_ELT(output, 0, filtered);
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 1));
    SET_STRING_ELT(names, 0, Rf_mkChar("X"));
    Rf_setAttrib(output, R_NamesSymbol, names);
    UNPROTECT(3);
    return output;
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP _fastPLS_opls_apply_filter_cpp(
    SEXP matrix, SEXP center, SEXP scale, SEXP weights, SEXP loadings) {
  try {
    fastpls::core::Matrix<double> values =
      numeric_matrix_from_sexp(matrix, "X");
    const std::vector<double> center_values = numeric_values(center, "mX");
    const std::vector<double> scale_values = numeric_values(scale, "vX");
    if (center_values.size() != values.columns() ||
        scale_values.size() != values.columns()) {
      throw std::invalid_argument(
        "X columns must match stored OPLS preprocessing"
      );
    }
    for (std::size_t column = 0; column < values.columns(); ++column) {
      for (std::size_t row = 0; row < values.rows(); ++row) {
        values(row, column) =
          (values(row, column) - center_values[column]) /
          scale_values[column];
      }
    }

    const fastpls::core::Matrix<double> weight_values =
      numeric_matrix_from_sexp_allow_empty(weights, "W_orth");
    const fastpls::core::Matrix<double> loading_values =
      numeric_matrix_from_sexp_allow_empty(loadings, "P_orth");
    if (weight_values.columns() != loading_values.columns() ||
        weight_values.rows() != values.columns() ||
        loading_values.rows() != values.columns()) {
      throw std::invalid_argument("Invalid OPLS orthogonal filter dimensions");
    }
    for (std::size_t component = 0;
         component < weight_values.columns(); ++component) {
      const auto weight = fastpls::core::make_const_view(
        weight_values.data() + component * weight_values.rows(),
        weight_values.rows(), 1, weight_values.rows()
      );
      const auto loading = fastpls::core::make_const_view(
        loading_values.data() + component * loading_values.rows(),
        loading_values.rows(), 1, loading_values.rows()
      );
      fastpls::core::Matrix<double> score(values.rows(), 1);
      fastpls::runtime::cpu_gemm_f64(
        values.view(), weight, false, false, score.view()
      );
      fastpls::core::Matrix<double> correction(
        values.rows(), values.columns()
      );
      fastpls::runtime::cpu_gemm_f64(
        score.view(), loading, false, true, correction.view()
      );
      for (std::size_t index = 0; index < values.size(); ++index) {
        values.data()[index] -= correction.data()[index];
      }
    }
    return numeric_matrix(values);
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP _fastPLS_kernel_matrix_cpp(SEXP left, SEXP right,
                                             SEXP kernel, SEXP gamma,
                                             SEXP degree, SEXP offset) {
  try {
    if (!Rf_isReal(left) || !Rf_isMatrix(left) ||
        !Rf_isReal(right) || !Rf_isMatrix(right)) {
      throw std::invalid_argument("Kernel inputs must be numeric matrices");
    }
    const SEXP left_dimensions = Rf_getAttrib(left, R_DimSymbol);
    const SEXP right_dimensions = Rf_getAttrib(right, R_DimSymbol);
    const std::size_t left_rows = INTEGER(left_dimensions)[0];
    const std::size_t left_columns = INTEGER(left_dimensions)[1];
    const std::size_t right_rows = INTEGER(right_dimensions)[0];
    const std::size_t right_columns = INTEGER(right_dimensions)[1];
    if (left_rows < 1 || right_rows < 1 || left_columns < 1 ||
        left_columns != right_columns) {
      throw std::invalid_argument(
        "Kernel inputs must be non-empty and have matching columns"
      );
    }
    const int kernel_code = Rf_asInteger(kernel);
    if (kernel_code < 1 || kernel_code > 3 || kernel_code == NA_INTEGER) {
      throw std::invalid_argument("Unknown kernel type");
    }
    const int polynomial_degree = Rf_asInteger(degree);
    if (polynomial_degree == NA_INTEGER) {
      throw std::invalid_argument("Polynomial degree must be an integer");
    }
    const auto left_view = fastpls::core::make_const_view(
      REAL(left), left_rows, left_columns, left_rows
    );
    const auto right_view = fastpls::core::make_const_view(
      REAL(right), right_rows, right_columns, right_rows
    );
    fastpls::core::Matrix<double> result(left_rows, right_rows);
    fastpls::runtime::cpu_gemm_f64(
      left_view, right_view, false, true, result.view()
    );
    fastpls::core::kernel_from_dots(
      left_view, right_view, result.view(),
      static_cast<fastpls::core::KernelType>(kernel_code), Rf_asReal(gamma),
      polynomial_degree, Rf_asReal(offset)
    );
    return numeric_matrix(result);
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}

extern "C" SEXP _fastPLS_center_kernel_test_cpp(
    SEXP kernel, SEXP training_means, SEXP training_grand_mean) {
  try {
    fastpls::core::Matrix<double> values =
      numeric_matrix_from_sexp(kernel, "Ktest");
    const fastpls::core::Matrix<double> means =
      numeric_matrix_from_sexp(training_means, "train_col_means");
    fastpls::core::center_kernel_test(
      values.view(), means.data(), means.size(), Rf_asReal(training_grand_mean)
    );
    return numeric_matrix(values);
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  }
  return R_NilValue;
}


extern "C" SEXP _fastPLS_opls_filter_core_cpp(
    SEXP predictors, SEXP responses, SEXP north, SEXP scaling) {
  return translate_exceptions("standalone-core OPLS filtering", [&] {
    fastpls::core::Matrix<double> x = numeric_matrix_from_sexp(
      predictors, "Xtrain"
    );
    const fastpls::core::Matrix<double> y = numeric_matrix_from_sexp(
      responses, "Ytrain"
    );
    const int component_count = Rf_asInteger(north);
    const int scaling_code = Rf_asInteger(scaling);
    if (x.rows() != y.rows() || component_count < 0 ||
        scaling_code < 1 || scaling_code > 3) {
      throw std::invalid_argument("standalone-core OPLS controls are invalid");
    }
    fastpls::runtime::CpuLinearAlgebraF64 backend;
    const auto filter = fastpls::core::fit_opls_filter(
      std::move(x), y.view(), static_cast<std::size_t>(component_count),
      static_cast<fastpls::core::PredictorScaling>(scaling_code), backend
    );
    return serialize_opls_filter<double>(
      filter, [](const fastpls::core::Matrix<double>& value) {
        return numeric_matrix(value);
      }
    );
  });
}

extern "C" SEXP _fastPLS_opls_filter_labels_core_cpp(
    SEXP predictors, SEXP labels, SEXP class_count, SEXP north,
    SEXP scaling) {
  return translate_exceptions("standalone-core label-aware OPLS filtering", [&] {
    fastpls::core::Matrix<double> x = numeric_matrix_from_sexp(
      predictors, "Xtrain"
    );
    const int classes = Rf_asInteger(class_count);
    const int component_count = Rf_asInteger(north);
    const int scaling_code = Rf_asInteger(scaling);
    if (component_count < 0 || scaling_code < 1 || scaling_code > 3) {
      throw std::invalid_argument(
        "standalone-core label-aware OPLS controls are invalid"
      );
    }
    const auto encoded = encoded_class_labels(
      labels, x.rows(), classes, "standalone-core label-aware OPLS"
    );
    fastpls::runtime::CpuLinearAlgebraF64 backend;
    const auto filter = fastpls::core::fit_opls_filter_labels(
      std::move(x), encoded.data(), encoded.size(),
      static_cast<std::size_t>(classes),
      static_cast<std::size_t>(component_count),
      static_cast<fastpls::core::PredictorScaling>(scaling_code), backend
    );
    return serialize_opls_filter<double>(
      filter, [](const fastpls::core::Matrix<double>& value) {
        return numeric_matrix(value);
      }
    );
  });
}

extern "C" SEXP _fastPLS_opls_filter_float32_core_cpp(
    SEXP predictors, SEXP responses, SEXP north, SEXP scaling,
    SEXP oversample, SEXP power, SEXP seed) {
  return translate_exceptions("standalone-core float32 OPLS filtering", [&] {
    fastpls::core::Matrix<float> x = float_matrix_from_s4(
      predictors, "Xtrain"
    );
    const fastpls::core::Matrix<float> y = float_matrix_from_s4(
      responses, "Ytrain"
    );
    const int component_count = Rf_asInteger(north);
    const int scaling_code = Rf_asInteger(scaling);
    const int oversample_count = Rf_asInteger(oversample);
    const int power_count = Rf_asInteger(power);
    const int seed_value = Rf_asInteger(seed);
    if (x.rows() != y.rows() || component_count < 0 ||
        scaling_code < 1 || scaling_code > 3 || oversample_count < 0 ||
        power_count < 0 || seed_value == NA_INTEGER) {
      throw std::invalid_argument(
        "standalone-core float32 OPLS controls are invalid"
      );
    }
    fastpls::core::RsvdControls controls;
    controls.oversample = oversample_count;
    controls.power = power_count;
    controls.seed = static_cast<unsigned int>(seed_value);
    controls.left_only = true;
    fastpls::runtime::CpuLinearAlgebraF32 backend;
    const auto filter = fastpls::core::fit_opls_filter_rsvd(
      std::move(x), y.view(), static_cast<std::size_t>(component_count),
      static_cast<fastpls::core::PredictorScaling>(scaling_code), controls,
      backend
    );
    return serialize_opls_filter<float>(
      filter, [](const fastpls::core::Matrix<float>& value) {
        return float_bits_matrix(value);
      }
    );
  });
}

extern "C" SEXP _fastPLS_opls_filter_float32_labels_core_cpp(
    SEXP predictors, SEXP labels, SEXP class_count, SEXP north,
    SEXP scaling) {
  return translate_exceptions(
    "standalone-core label-aware float32 OPLS filtering", [&] {
      fastpls::core::Matrix<float> x = float_matrix_from_s4(
        predictors, "Xtrain"
      );
      const int classes = Rf_asInteger(class_count);
      const int component_count = Rf_asInteger(north);
      const int scaling_code = Rf_asInteger(scaling);
      if (component_count < 0 || scaling_code < 1 || scaling_code > 3) {
        throw std::invalid_argument(
          "standalone-core label-aware float32 OPLS controls are invalid"
        );
      }
      const auto encoded = encoded_class_labels(
        labels, x.rows(), classes,
        "standalone-core label-aware float32 OPLS"
      );
      fastpls::runtime::CpuLinearAlgebraF32 backend;
      const auto filter = fastpls::core::fit_opls_filter_labels(
        std::move(x), encoded.data(), encoded.size(),
        static_cast<std::size_t>(classes),
        static_cast<std::size_t>(component_count),
        static_cast<fastpls::core::PredictorScaling>(scaling_code), backend
      );
      return serialize_opls_filter<float>(
        filter, [](const fastpls::core::Matrix<float>& value) {
          return float_bits_matrix(value);
        }
      );
    }
  );
}

extern "C" SEXP _fastPLS_opls_filter_float32_backend_core_cpp(
    SEXP predictors, SEXP responses, SEXP north, SEXP scaling,
    SEXP backend, SEXP oversample, SEXP power, SEXP seed) {
  return translate_exceptions("routed-core float32 OPLS filtering", [&] {
    fastpls::core::Matrix<float> x = float_matrix_from_s4(
      predictors, "Xtrain"
    );
    const fastpls::core::Matrix<float> y = float_matrix_from_s4(
      responses, "Ytrain"
    );
    const int component_count = Rf_asInteger(north);
    const int scaling_code = Rf_asInteger(scaling);
    const int backend_code = Rf_asInteger(backend);
    const int oversample_count = Rf_asInteger(oversample);
    const int power_count = Rf_asInteger(power);
    const int seed_value = Rf_asInteger(seed);
    if (x.rows() != y.rows() || component_count < 0 ||
        scaling_code < 1 || scaling_code > 3 || backend_code < 0 ||
        backend_code > 2 || oversample_count < 0 || power_count < 0 ||
        seed_value == NA_INTEGER) {
      throw std::invalid_argument(
        "routed-core float32 OPLS controls are invalid"
      );
    }
    fastpls::core::RsvdControls controls;
    controls.oversample = oversample_count;
    controls.power = power_count;
    controls.seed = static_cast<unsigned int>(seed_value);
    controls.left_only = true;
    RoutedLinearAlgebraF32 routed_backend(
      backend_code, x.rows(), x.columns(), y.columns()
    );
    const auto filter = fastpls::core::fit_opls_filter_rsvd(
      std::move(x), y.view(), static_cast<std::size_t>(component_count),
      static_cast<fastpls::core::PredictorScaling>(scaling_code), controls,
      routed_backend
    );
    return serialize_opls_filter<float>(
      filter, [](const fastpls::core::Matrix<float>& value) {
        return float_bits_matrix(value);
      }
    );
  });
}

extern "C" SEXP _fastPLS_opls_filter_float32_labels_backend_core_cpp(
    SEXP predictors, SEXP labels, SEXP class_count, SEXP north,
    SEXP scaling, SEXP backend, SEXP oversample, SEXP power, SEXP seed) {
  return translate_exceptions(
    "routed-core label-aware float32 OPLS filtering", [&] {
      fastpls::core::Matrix<float> x = float_matrix_from_s4(
        predictors, "Xtrain"
      );
      const int classes = Rf_asInteger(class_count);
      const int component_count = Rf_asInteger(north);
      const int scaling_code = Rf_asInteger(scaling);
      const int backend_code = Rf_asInteger(backend);
      const int oversample_count = Rf_asInteger(oversample);
      const int power_count = Rf_asInteger(power);
      const int seed_value = Rf_asInteger(seed);
      if (component_count < 0 || scaling_code < 1 || scaling_code > 3 ||
          backend_code < 0 || backend_code > 2 || oversample_count < 0 ||
          power_count < 0 || seed_value == NA_INTEGER) {
        throw std::invalid_argument(
          "routed-core label-aware float32 OPLS controls are invalid"
        );
      }
      const auto encoded = encoded_class_labels(
        labels, x.rows(), classes,
        "routed-core label-aware float32 OPLS"
      );
      fastpls::core::RsvdControls controls;
      controls.oversample = oversample_count;
      controls.power = power_count;
      controls.seed = static_cast<unsigned int>(seed_value);
      controls.left_only = true;
      RoutedLinearAlgebraF32 routed_backend(
        backend_code, x.rows(), x.columns(),
        static_cast<std::size_t>(classes)
      );
      const auto filter = fastpls::core::fit_opls_filter_labels_rsvd(
        std::move(x), encoded.data(), encoded.size(),
        static_cast<std::size_t>(classes),
        static_cast<std::size_t>(component_count),
        static_cast<fastpls::core::PredictorScaling>(scaling_code), controls,
        routed_backend
      );
      return serialize_opls_filter<float>(
        filter, [](const fastpls::core::Matrix<float>& value) {
          return float_bits_matrix(value);
        }
      );
    }
  );
}

extern "C" SEXP _fastPLS_pls_labels_core_cpp(
    SEXP predictors, SEXP labels, SEXP class_count, SEXP components,
    SEXP scaling, SEXP fit, SEXP store_scores, SEXP oversample, SEXP power,
    SEXP seed) {
  return translate_exceptions("double core PLS-SVD fitting", [&] {
    if (TYPEOF(components) != INTSXP || XLENGTH(components) < 1) {
      throw std::invalid_argument(
        "double core PLS-SVD requires integer component counts"
      );
    }
    const int classes = Rf_asInteger(class_count);
    const int scaling_code = Rf_asInteger(scaling);
    const int fit_code = Rf_asLogical(fit);
    const int store_scores_code = Rf_asLogical(store_scores);
    if (scaling_code < 1 || scaling_code > 3 || fit_code == NA_LOGICAL ||
        store_scores_code == NA_LOGICAL) {
      throw std::invalid_argument(
        "double core PLS-SVD training controls are invalid"
      );
    }
    fastpls::runtime::CpuLinearAlgebraF64 backend;
    if (scaling_code == static_cast<int>(
        fastpls::core::PredictorScaling::none) &&
        Rf_isMatrix(predictors) && TYPEOF(predictors) == REALSXP) {
      const auto x = numeric_matrix_view(predictors, "Xtrain");
      const auto encoded = encoded_class_labels(
        labels, x.rows(), classes, "double core PLS-SVD"
      );
      const auto prepared = fastpls::core::scaled_label_crossprod(
        x, encoded.data(), encoded.size(),
        static_cast<std::size_t>(classes),
        fastpls::core::PredictorScaling::none, backend
      );
      return fit_plssvd_label_core_prepared(
        x, prepared, encoded, classes, components, fit_code,
        store_scores_code,
        Rf_asInteger(oversample), Rf_asInteger(power),
        static_cast<unsigned int>(Rf_asInteger(seed)),
        "float64_label_class_sums", backend
      );
    }
    fastpls::core::Matrix<double> x = numeric_matrix_from_sexp(
      predictors, "Xtrain"
    );
    const auto encoded = encoded_class_labels(
      labels, x.rows(), classes, "double core PLS-SVD"
    );
    return fit_plssvd_label_core(
      x, encoded, classes, components, scaling_code, fit_code,
      store_scores_code,
      Rf_asInteger(oversample), Rf_asInteger(power),
      static_cast<unsigned int>(Rf_asInteger(seed)),
      "float64_label_class_sums", backend
    );
  });
}

extern "C" SEXP _fastPLS_pls_simpls_labels_core_cpp(
    SEXP predictors, SEXP labels, SEXP class_count, SEXP components,
    SEXP scaling, SEXP fit, SEXP store_scores, SEXP oversample, SEXP power,
    SEXP seed) {
  return translate_exceptions("double core SIMPLS fitting", [&] {
    if (TYPEOF(components) != INTSXP || XLENGTH(components) < 1) {
      throw std::invalid_argument(
        "double core SIMPLS requires integer component counts"
      );
    }
    const int classes = Rf_asInteger(class_count);
    const int scaling_code = Rf_asInteger(scaling);
    const int fit_code = Rf_asLogical(fit);
    const int store_scores_code = Rf_asLogical(store_scores);
    if (scaling_code < 1 || scaling_code > 3 || fit_code == NA_LOGICAL ||
        store_scores_code == NA_LOGICAL) {
      throw std::invalid_argument(
        "double core SIMPLS training controls are invalid"
      );
    }
    fastpls::runtime::CpuLinearAlgebraF64 backend;
    if (scaling_code == static_cast<int>(
        fastpls::core::PredictorScaling::none) &&
        Rf_isMatrix(predictors) && TYPEOF(predictors) == REALSXP) {
      const auto x = numeric_matrix_view(predictors, "Xtrain");
      const auto encoded = encoded_class_labels(
        labels, x.rows(), classes, "double core SIMPLS"
      );
      const auto prepared = fastpls::core::scaled_label_crossprod(
        x, encoded.data(), encoded.size(),
        static_cast<std::size_t>(classes),
        fastpls::core::PredictorScaling::none, backend
      );
      return fit_simpls_label_core_prepared(
        x, prepared, encoded, classes, components, fit_code,
        store_scores_code,
        Rf_asInteger(oversample), Rf_asInteger(power),
        static_cast<unsigned int>(Rf_asInteger(seed)),
        "float64_label_class_sums_blocked", backend
      );
    }
    fastpls::core::Matrix<double> x = numeric_matrix_from_sexp(
      predictors, "Xtrain"
    );
    const auto encoded = encoded_class_labels(
      labels, x.rows(), classes, "double core SIMPLS"
    );
    return fit_simpls_label_core(
      x, encoded, classes, components, scaling_code, fit_code,
      store_scores_code,
      Rf_asInteger(oversample), Rf_asInteger(power),
      static_cast<unsigned int>(Rf_asInteger(seed)),
      "float64_label_class_sums_blocked", backend
    );
  });
}

extern "C" SEXP _fastPLS_pls_matrix_core_cpp(
    SEXP predictors, SEXP responses, SEXP components, SEXP scaling,
    SEXP fit, SEXP store_scores, SEXP method, SEXP oversample, SEXP power,
    SEXP seed) {
  return translate_exceptions("double dense core PLS fitting", [&] {
    if (TYPEOF(components) != INTSXP || XLENGTH(components) < 1) {
      throw std::invalid_argument(
        "double dense core PLS requires integer component counts"
      );
    }
    fastpls::core::Matrix<double> owned_y;
    fastpls::core::ConstMatrixView<double> y;
    if (Rf_isMatrix(responses) && TYPEOF(responses) == REALSXP) {
      y = numeric_matrix_view(responses, "Ytrain");
    } else {
      owned_y = numeric_matrix_from_sexp(responses, "Ytrain");
      y = fastpls::core::ConstMatrixView<double>(owned_y.view());
    }
    const int scaling_code = Rf_asInteger(scaling);
    const int fit_code = Rf_asLogical(fit);
    const int store_scores_code = Rf_asLogical(store_scores);
    const int method_code = Rf_asInteger(method);
    if (scaling_code < 1 || scaling_code > 3 || fit_code == NA_LOGICAL ||
        store_scores_code == NA_LOGICAL ||
        (method_code != 1 && method_code != 3)) {
      throw std::invalid_argument(
        "double dense core PLS controls are invalid"
      );
    }
    fastpls::runtime::CpuLinearAlgebraF64 backend;
    if (scaling_code == static_cast<int>(
        fastpls::core::PredictorScaling::none) &&
        Rf_isMatrix(predictors) && TYPEOF(predictors) == REALSXP) {
      const auto x = numeric_matrix_view(predictors, "Xtrain");
      const auto prepared = fastpls::core::unscaled_dense_crossprod(
        x, y, backend
      );
      return fit_dense_core_prepared(
        x, y, prepared, components, fit_code, store_scores_code, method_code,
        Rf_asInteger(oversample), Rf_asInteger(power),
        static_cast<unsigned int>(Rf_asInteger(seed)),
        "float64_dense_crosscov", backend, true
      );
    }
    fastpls::core::Matrix<double> x = numeric_matrix_from_sexp(
      predictors, "Xtrain"
    );
    const auto prepared = fastpls::core::prepare_scaled_dense_crossprod(
      x.view(), y,
      static_cast<fastpls::core::PredictorScaling>(scaling_code), backend
    );
    return fit_dense_core_prepared(
      fastpls::core::ConstMatrixView<double>(x.view()), y, prepared,
      components, fit_code, store_scores_code, method_code,
      Rf_asInteger(oversample),
      Rf_asInteger(power), static_cast<unsigned int>(Rf_asInteger(seed)),
      "float64_dense_crosscov", backend, true
    );
  });
}

extern "C" SEXP _fastPLS_pls_matrix_core_xprod_cpp(
    SEXP predictors, SEXP responses, SEXP components, SEXP scaling,
    SEXP fit, SEXP store_scores, SEXP method, SEXP oversample, SEXP power,
    SEXP seed) {
  return translate_exceptions("double implicit core PLS fitting", [&] {
    const int method_code = Rf_asInteger(method);
    if (TYPEOF(components) != INTSXP || XLENGTH(components) < 1 ||
        (method_code != 1 && method_code != 3)) {
      throw std::invalid_argument(
        "double implicit core PLS requires PLS-SVD or SIMPLS"
      );
    }
    fastpls::core::Matrix<double> x = numeric_matrix_from_sexp(
      predictors, "Xtrain"
    );
    fastpls::core::Matrix<double> owned_y;
    fastpls::core::ConstMatrixView<double> y;
    if (Rf_isMatrix(responses) && TYPEOF(responses) == REALSXP) {
      y = numeric_matrix_view(responses, "Ytrain");
    } else {
      owned_y = numeric_matrix_from_sexp(responses, "Ytrain");
      y = fastpls::core::ConstMatrixView<double>(owned_y.view());
    }
    const int scaling_code = Rf_asInteger(scaling);
    const int fit_code = Rf_asLogical(fit);
    const int store_scores_code = Rf_asLogical(store_scores);
    if (x.rows() != y.rows() || scaling_code < 1 || scaling_code > 3 ||
        fit_code == NA_LOGICAL || store_scores_code == NA_LOGICAL) {
      throw std::invalid_argument(
        "double implicit core PLS dimensions or controls are invalid"
      );
    }
    fastpls::runtime::CpuLinearAlgebraF64 backend;
    const auto prepared = fastpls::core::prepare_scaled_dense_operator(
      x.view(), y,
      static_cast<fastpls::core::PredictorScaling>(scaling_code), backend
    );
    const auto x_view = fastpls::core::ConstMatrixView<double>(x.view());
    if (method_code == 1) {
      return fit_dense_plssvd_operator(
        x_view, y, prepared, components, fit_code, store_scores_code,
        Rf_asInteger(oversample), Rf_asInteger(power),
        static_cast<unsigned int>(Rf_asInteger(seed)),
        "float64_implicit_crosscov", backend, true
      );
    }
    return fit_dense_simpls_operator(
      x_view, y, prepared, components, fit_code, store_scores_code,
      Rf_asInteger(oversample), Rf_asInteger(power),
      static_cast<unsigned int>(Rf_asInteger(seed)),
      "float64_implicit_crosscov", backend, true
    );
  });
}

namespace {

SEXP fit_float32_matrix_core(
    SEXP predictors, SEXP responses, SEXP components, SEXP scaling,
    SEXP fit, SEXP store_scores, SEXP method, SEXP oversample, SEXP power,
    SEXP seed, int backend_code, const char* route) {
  return translate_exceptions("float32 dense core PLS fitting", [&] {
    if (TYPEOF(components) != INTSXP || XLENGTH(components) < 1) {
      throw std::invalid_argument(
        "float32 dense core PLS requires integer component counts"
      );
    }
    fastpls::core::Matrix<float> x =
      float_matrix_from_s4(predictors, "Xtrain");
    const auto y = float_matrix_from_s4(responses, "Ytrain");
    RoutedLinearAlgebraF32 backend(
      backend_code, x.rows(), x.columns(), y.columns()
    );
    const int scaling_code = Rf_asInteger(scaling);
    const int fit_code = Rf_asLogical(fit);
    const int store_scores_code = Rf_asLogical(store_scores);
    const int method_code = Rf_asInteger(method);
    if (x.rows() != y.rows() || scaling_code < 1 || scaling_code > 3 ||
        fit_code == NA_LOGICAL || store_scores_code == NA_LOGICAL ||
        (method_code != 1 && method_code != 3)) {
      throw std::invalid_argument(
        "float32 dense core PLS dimensions or controls are invalid"
      );
    }
    const long double crosscovariance_bytes =
      static_cast<long double>(x.columns()) *
      static_cast<long double>(y.columns()) * sizeof(float);
    if ((backend_code == 0 || backend_code == 2) &&
        crosscovariance_bytes > 512.0L * 1024.0L * 1024.0L) {
      const auto prepared = fastpls::core::prepare_scaled_dense_operator(
        x.view(), y.view(),
        static_cast<fastpls::core::PredictorScaling>(scaling_code), backend
      );
      const auto x_view = fastpls::core::ConstMatrixView<float>(x.view());
      const char* implicit_route = backend_code == 0 ?
        "float32_implicit_crosscov" :
        "float32_metal_implicit_crosscov";
      if (method_code == 1) {
        return fit_dense_plssvd_operator(
          x_view, y.view(), prepared, components, fit_code,
          store_scores_code, Rf_asInteger(oversample), Rf_asInteger(power),
          static_cast<unsigned int>(Rf_asInteger(seed)),
          implicit_route, backend, false
        );
      }
      return fit_dense_simpls_operator(
        x_view, y.view(), prepared, components, fit_code,
        store_scores_code, Rf_asInteger(oversample), Rf_asInteger(power),
        static_cast<unsigned int>(Rf_asInteger(seed)),
        implicit_route, backend, false
      );
    }
    const auto prepared = fastpls::core::prepare_scaled_dense_crossprod(
      x.view(), y.view(),
      static_cast<fastpls::core::PredictorScaling>(scaling_code), backend
    );
    return fit_dense_core_prepared(
      fastpls::core::ConstMatrixView<float>(x.view()),
      fastpls::core::ConstMatrixView<float>(y.view()), prepared,
      components, fit_code, store_scores_code, method_code,
      Rf_asInteger(oversample),
      Rf_asInteger(power), static_cast<unsigned int>(Rf_asInteger(seed)),
      route, backend, false
    );
  });
}

}  // namespace

extern "C" SEXP _fastPLS_pls_float32_matrix_backend_core_cpp(
    SEXP predictors, SEXP responses, SEXP components, SEXP scaling,
    SEXP fit, SEXP store_scores, SEXP method, SEXP oversample, SEXP power,
    SEXP seed, SEXP backend) {
  const int backend_code = Rf_asInteger(backend);
  if (backend_code < 0 || backend_code > 2 || backend_code == NA_INTEGER) {
    Rf_error("float32 PLS backend must be CPU, CUDA, or Metal");
  }
  const char* route = backend_code == 0 ? "float32_dense_crosscov" :
    backend_code == 1 ? "float32_cuda_hybrid_dense_crosscov" :
    "float32_metal_hybrid_dense_crosscov";
  return fit_float32_matrix_core(
    predictors, responses, components, scaling, fit, store_scores, method,
    oversample, power, seed, backend_code, route
  );
}

extern "C" SEXP _fastPLS_pls_labels_core_predict_cpp(
    SEXP model, SEXP predictors, SEXP project) {
  return translate_exceptions("double core PLS-SVD prediction", [&] {
    fastpls::core::Matrix<double> owned_x;
    fastpls::core::ConstMatrixView<double> x;
    if (Rf_isMatrix(predictors) && TYPEOF(predictors) == REALSXP) {
      x = numeric_matrix_view(predictors, "newdata");
    } else {
      owned_x = numeric_matrix_from_sexp(predictors, "newdata");
      x = fastpls::core::ConstMatrixView<double>(owned_x.view());
    }
    const auto projection = numeric_matrix_view(
      list_element(model, "R"), "model$R"
    );
    const auto center = numeric_values(
      list_element(model, "mX"), "model$mX"
    );
    const auto scale = numeric_values(
      list_element(model, "vX"), "model$vX"
    );
    const auto response_mean = numeric_values(
      list_element(model, "mY"), "model$mY"
    );
    SEXP components = list_element(model, "ncomp");
    SEXP stored_weights = list_element(model, "W_latent");
    SEXP response_loadings = list_element(model, "Q");
    const bool plssvd_list = TYPEOF(stored_weights) == VECSXP;
    const SEXP stored_dimensions = Rf_getAttrib(
      stored_weights, R_DimSymbol
    );
    const bool plssvd_cube = TYPEOF(stored_weights) == REALSXP &&
      TYPEOF(stored_dimensions) == INTSXP &&
      XLENGTH(stored_dimensions) == 3;
    const bool plssvd_weights = plssvd_list || plssvd_cube;
    fastpls::core::ConstMatrixView<double> simpls_loadings;
    if (!plssvd_weights) {
      simpls_loadings = numeric_matrix_view(response_loadings, "model$Q");
    }
    const int return_projection = Rf_asLogical(project);
    if (x.columns() != projection.rows() ||
        center.size() != x.columns() || scale.size() != x.columns() ||
        response_mean.empty() || TYPEOF(components) != INTSXP ||
        (plssvd_list &&
          XLENGTH(components) != XLENGTH(stored_weights)) ||
        (plssvd_cube &&
          (INTEGER(stored_dimensions)[1] !=
             static_cast<int>(response_mean.size()) ||
           INTEGER(stored_dimensions)[2] != XLENGTH(components))) ||
        (!plssvd_weights &&
          (simpls_loadings.rows() != response_mean.size() ||
           simpls_loadings.columns() < projection.columns())) ||
        return_projection == NA_LOGICAL) {
      throw std::invalid_argument(
        "double core PLS-SVD model is incompatible with newdata"
      );
    }
    fastpls::core::Matrix<double> scaled_projection(
      projection.rows(), projection.columns()
    );
    std::vector<double> score_offset(projection.columns(), 0.0);
    for (std::size_t column = 0; column < x.columns(); ++column) {
      if (!std::isfinite(scale[column]) || scale[column] == 0.0) {
        throw std::invalid_argument(
          "double core PLS-SVD model contains an invalid predictor scale"
        );
      }
      for (std::size_t component = 0;
           component < projection.columns(); ++component) {
        scaled_projection(column, component) =
          projection(column, component) / scale[column];
        score_offset[component] +=
          center[column] * scaled_projection(column, component);
      }
    }

    fastpls::runtime::CpuLinearAlgebraF64 backend;
    fastpls::core::Matrix<double> scores(x.rows(), projection.columns());
    backend.gemm(
      x, scaled_projection.view(), false, false, scores.view()
    );
    for (std::size_t component = 0;
         component < scores.columns(); ++component) {
      for (std::size_t row = 0; row < scores.rows(); ++row) {
        scores(row, component) -= score_offset[component];
      }
    }

    ProtectStack protect;
    const R_xlen_t prefix_count = XLENGTH(components);
    SEXP response = protect.add(Rf_allocVector(
      REALSXP,
      static_cast<R_xlen_t>(x.rows() * response_mean.size()) * prefix_count
    ));
    SEXP dimensions = protect.add(Rf_allocVector(INTSXP, 3));
    INTEGER(dimensions)[0] = static_cast<int>(x.rows());
    INTEGER(dimensions)[1] = static_cast<int>(response_mean.size());
    INTEGER(dimensions)[2] = static_cast<int>(prefix_count);
    Rf_setAttrib(response, R_DimSymbol, dimensions);
    const std::size_t slice_size = x.rows() * response_mean.size();
    for (R_xlen_t index = 0; index < prefix_count; ++index) {
      const int count = INTEGER(components)[index];
      if (count < 1 || static_cast<std::size_t>(count) > scores.columns()) {
        throw std::invalid_argument(
          "double core PLS component counts are inconsistent"
        );
      }
      const auto prefix = fastpls::core::make_const_view(
        scores.data(), scores.rows(), static_cast<std::size_t>(count),
        scores.rows()
      );
      fastpls::core::Matrix<double> values(
        x.rows(), response_mean.size()
      );
      if (plssvd_list) {
        const auto weights = numeric_matrix_view(
          VECTOR_ELT(stored_weights, index), "model$W_latent"
        );
        if (weights.rows() != static_cast<std::size_t>(count) ||
            weights.columns() != response_mean.size()) {
          throw std::invalid_argument(
            "double core PLS-SVD latent weights are inconsistent"
          );
        }
        backend.gemm(prefix, weights, false, false, values.view());
      } else if (plssvd_cube) {
        const std::size_t stored_rows = static_cast<std::size_t>(
          INTEGER(stored_dimensions)[0]
        );
        if (stored_rows < static_cast<std::size_t>(count)) {
          throw std::invalid_argument(
            "double core PLS-SVD latent cube is inconsistent"
          );
        }
        const auto weights = fastpls::core::make_const_view(
          REAL(stored_weights) + static_cast<std::size_t>(index) *
            stored_rows * response_mean.size(),
          static_cast<std::size_t>(count), response_mean.size(), stored_rows
        );
        backend.gemm(prefix, weights, false, false, values.view());
      } else {
        const auto loadings = fastpls::core::make_const_view(
          simpls_loadings.data(), simpls_loadings.rows(),
          static_cast<std::size_t>(count),
          simpls_loadings.leading_dimension()
        );
        backend.gemm(prefix, loadings, false, true, values.view());
      }
      double* destination = REAL(response) +
        static_cast<std::size_t>(index) * slice_size;
      for (std::size_t column = 0; column < values.columns(); ++column) {
        for (std::size_t row = 0; row < values.rows(); ++row) {
          destination[row + column * values.rows()] =
            values(row, column) + response_mean[column];
        }
      }
    }

    SEXP output = protect.add(Rf_allocVector(VECSXP, 2));
    SET_VECTOR_ELT(output, 0, response);
    SET_VECTOR_ELT(
      output, 1,
      return_projection ? numeric_matrix(scores) : Rf_allocMatrix(
        REALSXP, static_cast<int>(x.rows()), 0
      )
    );
    SEXP names = protect.add(Rf_allocVector(STRSXP, 2));
    SET_STRING_ELT(names, 0, Rf_mkChar("Ypred"));
    SET_STRING_ELT(names, 1, Rf_mkChar("Ttest"));
    Rf_setAttrib(output, R_NamesSymbol, names);
    return output;
  });
}

extern "C" SEXP _fastPLS_pls_class_predict_topk_core_cpp(
    SEXP model, SEXP predictors, SEXP top, SEXP project, SEXP block_size) {
  return translate_exceptions("double core top-k PLS prediction", [&] {
    const auto x = numeric_matrix_view(predictors, "newdata");
    const auto projection = numeric_matrix_view(
      list_element(model, "R"), "model$R"
    );
    const auto loadings = numeric_matrix_view(
      list_element(model, "Q"), "model$Q"
    );
    const auto center = numeric_values(
      list_element(model, "mX"), "model$mX"
    );
    const auto scale = numeric_values(
      list_element(model, "vX"), "model$vX"
    );
    const auto response_mean = numeric_values(
      list_element(model, "mY"), "model$mY"
    );
    SEXP components = list_element(model, "ncomp");
    const int requested_top = Rf_asInteger(top);
    const int return_projection = Rf_asLogical(project);
    const int requested_block = Rf_asInteger(block_size);
    if (x.columns() != projection.rows() ||
        loadings.rows() != response_mean.size() ||
        loadings.columns() < projection.columns() ||
        center.size() != x.columns() || scale.size() != x.columns() ||
        TYPEOF(components) != INTSXP || XLENGTH(components) < 1 ||
        requested_top < 1 || requested_block < 1 ||
        return_projection == NA_LOGICAL) {
      throw std::invalid_argument(
        "double core top-k model is incompatible with newdata"
      );
    }
    const std::size_t class_count = response_mean.size();
    const std::size_t keep = std::min<std::size_t>(
      static_cast<std::size_t>(requested_top), class_count
    );
    const std::size_t prefix_count = static_cast<std::size_t>(
      XLENGTH(components)
    );
    const std::size_t maximum_components = projection.columns();
    for (std::size_t column = 0; column < x.columns(); ++column) {
      if (!std::isfinite(scale[column]) || scale[column] == 0.0) {
        throw std::invalid_argument(
          "double core top-k model contains an invalid predictor scale"
        );
      }
    }
    for (std::size_t index = 0; index < prefix_count; ++index) {
      const int count = INTEGER(components)[index];
      if (count < 1 || static_cast<std::size_t>(count) > maximum_components) {
        throw std::invalid_argument(
          "double core top-k component counts are inconsistent"
        );
      }
    }

    ProtectStack protect;
    const R_xlen_t output_length = static_cast<R_xlen_t>(
      x.rows() * keep * prefix_count
    );
    SEXP top_index = protect.add(Rf_allocVector(INTSXP, output_length));
    SEXP top_score = protect.add(Rf_allocVector(REALSXP, output_length));
    SEXP dimensions = protect.add(Rf_allocVector(INTSXP, 3));
    INTEGER(dimensions)[0] = static_cast<int>(x.rows());
    INTEGER(dimensions)[1] = static_cast<int>(keep);
    INTEGER(dimensions)[2] = static_cast<int>(prefix_count);
    Rf_setAttrib(top_index, R_DimSymbol, dimensions);
    Rf_setAttrib(top_score, R_DimSymbol, dimensions);
    SEXP projected = protect.add(Rf_allocMatrix(
      REALSXP, static_cast<int>(x.rows()),
      return_projection ? static_cast<int>(maximum_components) : 0
    ));

    fastpls::runtime::CpuLinearAlgebraF64 backend;
    const std::size_t rows_per_block = std::min<std::size_t>(
      static_cast<std::size_t>(requested_block), x.rows()
    );
    for (std::size_t start = 0; start < x.rows(); start += rows_per_block) {
      const std::size_t rows = std::min(rows_per_block, x.rows() - start);
      fastpls::core::Matrix<double> standardized(rows, x.columns());
      for (std::size_t column = 0; column < x.columns(); ++column) {
        for (std::size_t row = 0; row < rows; ++row) {
          standardized(row, column) =
            (x(start + row, column) - center[column]) / scale[column];
        }
      }
      fastpls::core::Matrix<double> scores(rows, maximum_components);
      backend.gemm(
        standardized.view(), projection, false, false, scores.view()
      );
      if (return_projection) {
        for (std::size_t component = 0;
             component < maximum_components; ++component) {
          for (std::size_t row = 0; row < rows; ++row) {
            REAL(projected)[start + row + component * x.rows()] =
              scores(row, component);
          }
        }
      }
      for (std::size_t prefix = 0; prefix < prefix_count; ++prefix) {
        const std::size_t count = static_cast<std::size_t>(
          INTEGER(components)[prefix]
        );
        const auto score_prefix = fastpls::core::make_const_view(
          scores.data(), rows, count, scores.rows()
        );
        const auto loading_prefix = fastpls::core::make_const_view(
          loadings.data(), loadings.rows(), count,
          loadings.leading_dimension()
        );
        fastpls::core::Matrix<double> values(rows, class_count);
        backend.gemm(
          score_prefix, loading_prefix, false, true, values.view()
        );
        for (std::size_t row = 0; row < rows; ++row) {
          std::vector<double> best_scores(
            keep, -std::numeric_limits<double>::infinity()
          );
          std::vector<int> best_indices(keep, 0);
          for (std::size_t category = 0;
               category < class_count; ++category) {
            const double value =
              values(row, category) + response_mean[category];
            for (std::size_t rank = 0; rank < keep; ++rank) {
              if (value > best_scores[rank]) {
                for (std::size_t lower = keep - 1; lower > rank; --lower) {
                  best_scores[lower] = best_scores[lower - 1];
                  best_indices[lower] = best_indices[lower - 1];
                }
                best_scores[rank] = value;
                best_indices[rank] = static_cast<int>(category + 1);
                break;
              }
            }
          }
          for (std::size_t rank = 0; rank < keep; ++rank) {
            const std::size_t destination = start + row +
              x.rows() * rank + x.rows() * keep * prefix;
            INTEGER(top_index)[destination] = best_indices[rank];
            REAL(top_score)[destination] = best_scores[rank];
          }
        }
      }
    }

    SEXP output = protect.add(Rf_allocVector(VECSXP, 4));
    SEXP names = protect.add(Rf_allocVector(STRSXP, 4));
    const char* field_names[4] = {
      "top_index", "top_score", "Ttest", "predict_backend"
    };
    for (int index = 0; index < 4; ++index) {
      SET_STRING_ELT(names, index, Rf_mkChar(field_names[index]));
    }
    SET_VECTOR_ELT(output, 0, top_index);
    SET_VECTOR_ELT(output, 1, top_score);
    SET_VECTOR_ELT(output, 2, projected);
    SET_VECTOR_ELT(output, 3, Rf_mkString("core_topk"));
    Rf_setAttrib(output, R_NamesSymbol, names);
    return output;
  });
}

extern "C" SEXP _fastPLS_pls_float32_class_predict_compact_cpp(
    SEXP model, SEXP predictors, SEXP use_lda, SEXP block_size) {
  return translate_exceptions("compact float32 class prediction", [&] {
    if (!Rf_isS4(predictors) || !Rf_inherits(predictors, "float32")) {
      throw std::invalid_argument(
        "compact float32 prediction requires float32 newdata"
      );
    }
    const int lda = Rf_asLogical(use_lda);
    const int requested_block = Rf_asInteger(block_size);
    if (lda == NA_LOGICAL || requested_block < 1) {
      throw std::invalid_argument(
        "compact float32 prediction controls are invalid"
      );
    }

    ProtectStack protect;
    SEXP bits = protect.add(R_do_slot(predictors, Rf_install("Data")));
    const SEXP input_dimensions = Rf_getAttrib(bits, R_DimSymbol);
    if (TYPEOF(bits) != INTSXP || TYPEOF(input_dimensions) != INTSXP ||
        XLENGTH(input_dimensions) != 2 || INTEGER(input_dimensions)[0] < 1 ||
        INTEGER(input_dimensions)[1] < 1) {
      throw std::invalid_argument("newdata contains invalid float32 storage");
    }
    const std::size_t sample_count = static_cast<std::size_t>(
      INTEGER(input_dimensions)[0]
    );
    const std::size_t predictor_count = static_cast<std::size_t>(
      INTEGER(input_dimensions)[1]
    );
    const int* input_bits = INTEGER(bits);

    const auto projection = float_matrix_from_s4(
      list_element(model, "R"), "model$R"
    );
    const auto center_matrix = float_matrix_from_s4(
      list_element(model, "mX"), "model$mX"
    );
    const auto scale_matrix = float_matrix_from_s4(
      list_element(model, "vX"), "model$vX"
    );
    const auto response_mean_matrix = float_matrix_from_s4(
      list_element(model, "mY"), "model$mY"
    );
    SEXP components = list_element(model, "ncomp");
    if (projection.rows() != predictor_count ||
        center_matrix.size() != predictor_count ||
        scale_matrix.size() != predictor_count ||
        response_mean_matrix.size() < 2 || TYPEOF(components) != INTSXP ||
        XLENGTH(components) < 1) {
      throw std::invalid_argument(
        "compact float32 model is incompatible with newdata"
      );
    }
    const std::size_t class_count = response_mean_matrix.size();
    const std::size_t prefix_count = static_cast<std::size_t>(
      XLENGTH(components)
    );
    for (std::size_t column = 0; column < predictor_count; ++column) {
      if (!std::isfinite(scale_matrix.data()[column]) ||
          scale_matrix.data()[column] == 0.0f) {
        throw std::invalid_argument(
          "compact float32 model contains an invalid predictor scale"
        );
      }
    }

    std::vector<fastpls::core::Matrix<float>> weights(prefix_count);
    std::vector<std::vector<float>> constants(prefix_count);
    if (lda == TRUE) {
      SEXP lda_data = list_element(model, "lda");
      SEXP models = list_element(lda_data, "models");
      const SEXP model_names = Rf_getAttrib(models, R_NamesSymbol);
      if (TYPEOF(models) != VECSXP || TYPEOF(model_names) != STRSXP) {
        throw std::invalid_argument(
          "compact float32 LDA models do not match the component path"
        );
      }
      for (std::size_t index = 0; index < prefix_count; ++index) {
        const std::string key = std::to_string(INTEGER(components)[index]);
        SEXP fitted = R_NilValue;
        for (R_xlen_t candidate = 0; candidate < XLENGTH(models); ++candidate) {
          if (STRING_ELT(model_names, candidate) != NA_STRING &&
              key == CHAR(STRING_ELT(model_names, candidate))) {
            fitted = VECTOR_ELT(models, candidate);
            break;
          }
        }
        if (fitted == R_NilValue) {
          throw std::invalid_argument(
            "compact float32 LDA model is missing a component prefix"
          );
        }
        const auto linear = float_matrix_from_bits(
          list_element(fitted, "linear"), "LDA linear coefficients"
        );
        weights[index].resize(linear.columns(), linear.rows());
        for (std::size_t class_index = 0;
             class_index < linear.rows(); ++class_index) {
          for (std::size_t component = 0;
               component < linear.columns(); ++component) {
            weights[index](component, class_index) =
              linear(class_index, component);
          }
        }
        const auto intercept = float_matrix_from_bits(
          list_element(fitted, "constants"), "LDA constants"
        );
        constants[index].assign(
          intercept.data(), intercept.data() + intercept.size()
        );
      }
    } else {
      SEXP stored = list_element(model, "W_latent");
      if (TYPEOF(stored) == VECSXP && XLENGTH(stored) == XLENGTH(components)) {
        for (std::size_t index = 0; index < prefix_count; ++index) {
          weights[index] = float_matrix_from_s4(
            VECTOR_ELT(stored, static_cast<R_xlen_t>(index)),
            "PLS-SVD latent prediction weights"
          );
        }
      } else {
        const auto loadings = float_matrix_from_s4(
          list_element(model, "Q"), "SIMPLS response loadings"
        );
        for (std::size_t index = 0; index < prefix_count; ++index) {
          const int retained = INTEGER(components)[index];
          if (retained < 1 ||
              static_cast<std::size_t>(retained) > loadings.columns()) {
            throw std::invalid_argument(
              "SIMPLS component path exceeds the response loadings"
            );
          }
          weights[index].resize(
            static_cast<std::size_t>(retained), class_count
          );
          for (std::size_t response = 0; response < class_count; ++response) {
            for (int component = 0; component < retained; ++component) {
              weights[index](static_cast<std::size_t>(component), response) =
                loadings(response, static_cast<std::size_t>(component));
            }
          }
        }
      }
    }

    for (std::size_t index = 0; index < prefix_count; ++index) {
      const int retained = INTEGER(components)[index];
      if (retained < 1 ||
          static_cast<std::size_t>(retained) > projection.columns() ||
          weights[index].rows() != static_cast<std::size_t>(retained) ||
          weights[index].columns() != class_count ||
          (lda == TRUE && constants[index].size() != class_count)) {
        throw std::invalid_argument(
          "compact float32 classifier dimensions are inconsistent"
        );
      }
    }

    SEXP output = protect.add(Rf_allocMatrix(
      INTSXP, static_cast<int>(sample_count), static_cast<int>(prefix_count)
    ));
    fastpls::runtime::CpuLinearAlgebraF32 backend;
    const std::size_t rows_per_block = std::min<std::size_t>(
      static_cast<std::size_t>(requested_block), sample_count
    );
    for (std::size_t start = 0; start < sample_count;
         start += rows_per_block) {
      const std::size_t count = std::min(rows_per_block, sample_count - start);
      fastpls::core::Matrix<float> block(count, predictor_count);
      for (std::size_t column = 0; column < predictor_count; ++column) {
        std::memcpy(
          block.data() + column * count,
          input_bits + column * sample_count + start,
          count * sizeof(float)
        );
        const float location = center_matrix.data()[column];
        const float divisor = scale_matrix.data()[column];
        for (std::size_t row = 0; row < count; ++row) {
          block(row, column) = (block(row, column) - location) / divisor;
        }
      }
      fastpls::core::Matrix<float> scores(count, projection.columns());
      backend.gemm(block.view(), projection.view(), false, false, scores.view());
      for (std::size_t index = 0; index < prefix_count; ++index) {
        const std::size_t retained = static_cast<std::size_t>(
          INTEGER(components)[index]
        );
        const auto score_prefix = fastpls::core::make_const_view(
          scores.data(), count, retained, scores.rows()
        );
        fastpls::core::Matrix<float> discriminants(count, class_count);
        backend.gemm(
          score_prefix, weights[index].view(), false, false,
          discriminants.view()
        );
        const float* offset = lda == TRUE ? constants[index].data() :
          response_mean_matrix.data();
        for (std::size_t response = 0; response < class_count; ++response) {
          for (std::size_t row = 0; row < count; ++row) {
            discriminants(row, response) += offset[response];
          }
        }
        for (std::size_t row = 0; row < count; ++row) {
          INTEGER(output)[start + row + index * sample_count] =
            static_cast<int>(
              fastpls::core::row_argmax(discriminants.view(), row) + 1
            );
        }
      }
    }
    return output;
  });
}

namespace {

SEXP fit_float32_labels_core(
    SEXP predictors, SEXP labels, SEXP class_count, SEXP components,
    SEXP scaling, SEXP fit, SEXP store_scores, SEXP method, SEXP oversample,
    SEXP power, SEXP seed, int backend_code, const char* plssvd_route,
    const char* simpls_route) {
  return translate_exceptions("float32 core PLS fitting", [&] {
    if (TYPEOF(labels) != INTSXP || TYPEOF(components) != INTSXP ||
        XLENGTH(components) < 1) {
      throw std::invalid_argument(
        "float32 core PLS requires integer labels and component counts"
      );
    }
    fastpls::core::Matrix<float> x =
      float_matrix_from_s4(predictors, "Xtrain");
    const int classes = Rf_asInteger(class_count);
    RoutedLinearAlgebraF32 backend(
      backend_code, x.rows(), x.columns(),
      classes > 0 ? static_cast<std::size_t>(classes) : 0
    );
    const int scaling_code = Rf_asInteger(scaling);
    const int fit_code = Rf_asLogical(fit);
    const int store_scores_code = Rf_asLogical(store_scores);
    const int method_code = Rf_asInteger(method);
    if (classes < 2 || scaling_code < 1 || scaling_code > 3 ||
        fit_code == NA_LOGICAL || store_scores_code == NA_LOGICAL ||
        (method_code != 1 && method_code != 3) ||
        XLENGTH(labels) != static_cast<R_xlen_t>(x.rows())) {
      throw std::invalid_argument(
        "float32 core PLS training dimensions or controls are invalid"
      );
    }
    const auto encoded = encoded_class_labels(
      labels, x.rows(), classes, "float32 core PLS"
    );
    if (method_code == 1) {
      return fit_plssvd_label_core(
        x, encoded, classes, components, scaling_code, fit_code,
        store_scores_code,
        Rf_asInteger(oversample), Rf_asInteger(power),
        static_cast<unsigned int>(Rf_asInteger(seed)),
        plssvd_route, backend
      );
    }
    const auto prepared = fastpls::core::prepare_scaled_label_crossprod(
      x.view(), encoded.data(), encoded.size(),
      static_cast<std::size_t>(classes),
      static_cast<fastpls::core::PredictorScaling>(scaling_code), backend
    );
    return fit_simpls_label_core_prepared(
      fastpls::core::ConstMatrixView<float>(x.view()), prepared, encoded,
      classes, components, fit_code, store_scores_code,
      Rf_asInteger(oversample),
      Rf_asInteger(power), static_cast<unsigned int>(Rf_asInteger(seed)),
      simpls_route, backend
    );
  });
}

}  // namespace

extern "C" SEXP _fastPLS_pls_float32_labels_backend_core_cpp(
    SEXP predictors, SEXP labels, SEXP class_count, SEXP components,
    SEXP scaling, SEXP fit, SEXP store_scores, SEXP method, SEXP oversample,
    SEXP power, SEXP seed, SEXP backend) {
  const int backend_code = Rf_asInteger(backend);
  if (backend_code < 0 || backend_code > 2 || backend_code == NA_INTEGER) {
    Rf_error("float32 PLS backend must be CPU, CUDA, or Metal");
  }
  const char* plssvd_route = backend_code == 0 ?
    "float32_label_class_sums" : backend_code == 1 ?
    "float32_cuda_hybrid_label_class_sums" :
    "float32_metal_hybrid_label_class_sums";
  const char* simpls_route = backend_code == 0 ?
    "float32_label_class_sums_blocked" : backend_code == 1 ?
    "float32_cuda_hybrid_label_class_sums_blocked" :
    "float32_metal_hybrid_label_class_sums_blocked";
  return fit_float32_labels_core(
    predictors, labels, class_count, components, scaling, fit, store_scores,
    method, oversample, power, seed, backend_code, plssvd_route,
    simpls_route
  );
}
