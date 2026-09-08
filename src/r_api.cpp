#include "r_api.h"

#include <R_ext/Error.h>
#include <fastpls/core/classification.hpp>
#include <fastpls/core/diagnostics.hpp>
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
