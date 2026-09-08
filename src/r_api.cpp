#include "r_api.h"
#include "core_cpu_backend.h"

#include <R_ext/Error.h>
#include <fastpls/core/classification.hpp>
#include <fastpls/core/diagnostics.hpp>
#include <fastpls/core/kernels.hpp>
#include <fastpls/core/matrix.hpp>
#include <fastpls/core/statistics.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace fastpls_svd {
bool has_cuda_backend();
bool has_metal_backend();
bool cuda_lda_native_available();
void reset_rsvd_audit_summary();
fastpls::core::RSVDAuditSummary current_rsvd_audit_summary();
}

namespace {

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

fastpls::core::Matrix<float> float_matrix_from_s4(SEXP object,
                                                  const char* name) {
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
  if (TYPEOF(bits) != INTSXP || TYPEOF(dimensions) != INTSXP ||
      XLENGTH(dimensions) != 2 || INTEGER(dimensions)[0] < 1 ||
      INTEGER(dimensions)[1] < 1) {
    UNPROTECT(1);
    throw std::invalid_argument(std::string(name) + " must be a non-empty float32 matrix");
  }
  const std::size_t rows = static_cast<std::size_t>(INTEGER(dimensions)[0]);
  const std::size_t columns = static_cast<std::size_t>(INTEGER(dimensions)[1]);
  fastpls::core::Matrix<float> values(rows, columns);
  const int* source = INTEGER(bits);
  for (std::size_t index = 0; index < values.size(); ++index) {
    values.data()[index] = decode_float32(source[index]);
  }
  UNPROTECT(1);
  return values;
}

fastpls::core::Matrix<double> numeric_matrix_from_sexp(SEXP object,
                                                       const char* name) {
  if (!Rf_isMatrix(object) ||
      (TYPEOF(object) != REALSXP && TYPEOF(object) != INTSXP)) {
    throw std::invalid_argument(std::string(name) + " must be a numeric matrix");
  }
  const SEXP dimensions = Rf_getAttrib(object, R_DimSymbol);
  const std::size_t rows = static_cast<std::size_t>(INTEGER(dimensions)[0]);
  const std::size_t columns = static_cast<std::size_t>(INTEGER(dimensions)[1]);
  if (rows < 1 || columns < 1) {
    throw std::invalid_argument(std::string(name) + " must be non-empty");
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

SEXP float_bits_matrix(const fastpls::core::Matrix<float>& values) {
  SEXP result = Rf_allocMatrix(
    INTSXP, static_cast<int>(values.rows()), static_cast<int>(values.columns())
  );
  for (std::size_t index = 0; index < values.size(); ++index) {
    INTEGER(result)[index] = encode_float32(values.data()[index]);
  }
  return result;
}

SEXP numeric_matrix(const fastpls::core::Matrix<double>& values) {
  SEXP result = Rf_allocMatrix(
    REALSXP, static_cast<int>(values.rows()), static_cast<int>(values.columns())
  );
  std::copy(values.data(), values.data() + values.size(), REAL(result));
  return result;
}

}  // namespace

extern "C" SEXP _fastPLS_has_cuda() {
  return Rf_ScalarLogical(fastpls_svd::has_cuda_backend());
}

extern "C" SEXP _fastPLS_has_metal() {
  return Rf_ScalarLogical(fastpls_svd::has_metal_backend());
}

extern "C" SEXP _fastPLS_lda_cuda_native_available() {
  return Rf_ScalarLogical(fastpls_svd::cuda_lda_native_available());
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

extern "C" SEXP _fastPLS_label_crossprod_scaled_cpp(SEXP predictors,
                                                       SEXP labels,
                                                       SEXP class_count,
                                                       SEXP scaling) {
  try {
    if (!Rf_isReal(predictors) || !Rf_isMatrix(predictors)) {
      throw std::invalid_argument("Xtrain must be a numeric matrix");
    }
    if (TYPEOF(labels) != INTSXP) {
      throw std::invalid_argument("classification labels must be integers");
    }
    const SEXP dimensions = Rf_getAttrib(predictors, R_DimSymbol);
    const int rows = INTEGER(dimensions)[0];
    const int columns = INTEGER(dimensions)[1];
    if (rows < 1 || columns < 1 || XLENGTH(labels) != rows) {
      throw std::invalid_argument(
        "label_crossprod_scaled_cpp requires one label per training row"
      );
    }
    const int classes = Rf_asInteger(class_count);
    if (classes < 2 || classes == NA_INTEGER) {
      throw std::invalid_argument(
        "label_crossprod_scaled_cpp requires at least two classes"
      );
    }
    const int scaling_code = Rf_asInteger(scaling);
    if (scaling_code < 1 || scaling_code > 3 || scaling_code == NA_INTEGER) {
      throw std::invalid_argument("scaling must be 1, 2, or 3");
    }
    std::vector<std::size_t> zero_based_labels(static_cast<std::size_t>(rows));
    for (int row = 0; row < rows; ++row) {
      const int label = INTEGER(labels)[row];
      if (label == NA_INTEGER || label < 1 || label > classes) {
        throw std::invalid_argument(
          "label_crossprod_scaled_cpp requires labels encoded as 1..n_classes"
        );
      }
      zero_based_labels[static_cast<std::size_t>(row)] =
        static_cast<std::size_t>(label - 1);
    }
    const auto result = fastpls::core::scaled_label_crossprod(
      fastpls::core::make_const_view(
        REAL(predictors), static_cast<std::size_t>(rows),
        static_cast<std::size_t>(columns), static_cast<std::size_t>(rows)
      ),
      zero_based_labels.data(), zero_based_labels.size(),
      static_cast<std::size_t>(classes),
      static_cast<fastpls::core::PredictorScaling>(scaling_code)
    );

    SEXP crossprod = PROTECT(Rf_allocMatrix(REALSXP, columns, classes));
    std::copy(
      result.crossprod.data(),
      result.crossprod.data() + result.crossprod.size(), REAL(crossprod)
    );
    SEXP center = PROTECT(Rf_allocMatrix(REALSXP, 1, columns));
    std::copy(
      result.predictor_center.begin(), result.predictor_center.end(),
      REAL(center)
    );
    SEXP scale = PROTECT(Rf_allocMatrix(REALSXP, 1, columns));
    std::copy(
      result.predictor_scale.begin(), result.predictor_scale.end(), REAL(scale)
    );
    SEXP response_mean = PROTECT(Rf_allocMatrix(REALSXP, 1, classes));
    std::copy(
      result.response_mean.begin(), result.response_mean.end(),
      REAL(response_mean)
    );
    SEXP counts = PROTECT(Rf_allocMatrix(REALSXP, classes, 1));
    std::copy(
      result.class_counts.begin(), result.class_counts.end(), REAL(counts)
    );
    SEXP output = PROTECT(Rf_allocVector(VECSXP, 5));
    SET_VECTOR_ELT(output, 0, crossprod);
    SET_VECTOR_ELT(output, 1, center);
    SET_VECTOR_ELT(output, 2, scale);
    SET_VECTOR_ELT(output, 3, response_mean);
    SET_VECTOR_ELT(output, 4, counts);
    SEXP names = PROTECT(Rf_allocVector(STRSXP, 5));
    SET_STRING_ELT(names, 0, Rf_mkChar("S"));
    SET_STRING_ELT(names, 1, Rf_mkChar("mX"));
    SET_STRING_ELT(names, 2, Rf_mkChar("vX"));
    SET_STRING_ELT(names, 3, Rf_mkChar("mY"));
    SET_STRING_ELT(names, 4, Rf_mkChar("counts"));
    Rf_setAttrib(output, R_NamesSymbol, names);
    UNPROTECT(7);
    return output;
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  } catch (...) {
    Rf_error("Unknown error in label-aware cross-product");
  }
  return R_NilValue;
}
