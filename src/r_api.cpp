#include "r_api.h"
#include "accelerator_core_backend.h"
#include "core_cpu_backend.h"

#include <R_ext/Error.h>
#include <fastpls/core/classification.hpp>
#include <fastpls/core/diagnostics.hpp>
#include <fastpls/core/kernels.hpp>
#include <fastpls/core/lda.hpp>
#include <fastpls/core/matrix.hpp>
#include <fastpls/core/plssvd.hpp>
#include <fastpls/core/simpls.hpp>
#include <fastpls/core/statistics.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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
  for (std::size_t index = 0; index < values.size(); ++index) {
    values.data()[index] = decode_float32(source[index]);
  }
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
  for (std::size_t index = 0; index < values.size(); ++index) {
    values.data()[index] = decode_float32(INTEGER(object)[index]);
  }
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

template<class T, class Backend>
SEXP fit_plssvd_label_core_prepared(
    fastpls::core::ConstMatrixView<T> predictors,
    const fastpls::core::LabelCrossprodResult<T>& prepared,
    const std::vector<std::size_t>& labels, int class_count,
    SEXP components, bool fitted, int oversample, int power,
    unsigned int seed, const char* xprod_mode, Backend& backend) {
  ProtectStack protect;
  SEXP effective_components = protect.add(Rf_duplicate(components));
  const std::size_t sample_rank = std::max<std::size_t>(
    predictors.rows() - 1, 1
  );
  const int rank_cap = static_cast<int>(std::min({
    predictors.columns(), sample_rank,
    static_cast<std::size_t>(class_count - 1)
  }));
  for (R_xlen_t index = 0; index < XLENGTH(effective_components); ++index) {
    const int value = INTEGER(effective_components)[index];
    if (value == NA_INTEGER) {
      throw std::invalid_argument("ncomp cannot contain missing values");
    }
    INTEGER(effective_components)[index] = std::max(
      1, std::min(value, rank_cap)
    );
  }

  fastpls::core::PlssvdControls controls;
  controls.rsvd.oversample = oversample;
  controls.rsvd.power = power;
  controls.rsvd.seed = seed;
  const auto model = fastpls::core::fit_plssvd_preprocessed<T>(
    predictors, prepared.crossprod.view(),
    INTEGER(effective_components),
    static_cast<std::size_t>(XLENGTH(effective_components)), controls,
    backend
  );

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
      fastpls::core::Matrix<T> values(predictors.rows(), class_count);
      backend.gemm(
        scores, model.prediction_weights[index].view(), false, false,
        values.view()
      );
      r2_values[index] = fastpls::core::dummy_response_r2(
        labels.data(), labels.size(), prepared.response_mean.data(),
        static_cast<std::size_t>(class_count), values.view()
      );
      for (std::size_t response = 0;
           response < static_cast<std::size_t>(class_count); ++response) {
        for (std::size_t row = 0; row < values.rows(); ++row) {
          values(row, response) += prepared.response_mean[response];
        }
      }
      fitted_values.push_back(std::move(values));
    }
  }

  constexpr int field_count = 15;
  SEXP output = protect.add(Rf_allocVector(VECSXP, field_count));
  SEXP names = protect.add(Rf_allocVector(STRSXP, field_count));
  const char* field_names[field_count] = {
    "P", "R", "Q", "Ttrain", "W_latent", "mX", "vX", "mY",
    "p", "m", "ncomp", "Yfit", "R2Y", "pls_method", "xprod_mode"
  };
  for (int index = 0; index < field_count; ++index) {
    SET_STRING_ELT(names, index, Rf_mkChar(field_names[index]));
  }
  SET_VECTOR_ELT(output, 0, R_NilValue);
  SET_VECTOR_ELT(output, 1, core_matrix(model.weights));
  SET_VECTOR_ELT(output, 2, core_matrix(model.response_loadings));
  SET_VECTOR_ELT(output, 3, core_matrix(model.scores));
  SET_VECTOR_ELT(
    output, 4, core_matrix_list(
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
  SET_VECTOR_ELT(output, 9, Rf_ScalarInteger(class_count));
  SET_VECTOR_ELT(output, 10, effective_components);
  SET_VECTOR_ELT(
    output, 11, fitted ? core_matrix_list(
      fitted_values, INTEGER(effective_components)
    ) : R_NilValue
  );
  SEXP r2 = protect.add(Rf_allocVector(REALSXP, XLENGTH(components)));
  std::copy(r2_values.begin(), r2_values.end(), REAL(r2));
  SET_VECTOR_ELT(output, 12, r2);
  SET_VECTOR_ELT(output, 13, Rf_mkString("plssvd"));
  SET_VECTOR_ELT(output, 14, Rf_mkString(xprod_mode));
  Rf_setAttrib(output, R_NamesSymbol, names);
  return output;
}

template<class T, class Backend>
SEXP fit_plssvd_label_core(
    fastpls::core::Matrix<T>& predictors,
    const std::vector<std::size_t>& labels, int class_count,
    SEXP components, int scaling, bool fitted, int oversample, int power,
    unsigned int seed, const char* xprod_mode, Backend& backend) {
  const auto prepared = fastpls::core::prepare_scaled_label_crossprod(
    predictors.view(), labels.data(), labels.size(),
    static_cast<std::size_t>(class_count),
    static_cast<fastpls::core::PredictorScaling>(scaling), backend
  );
  return fit_plssvd_label_core_prepared(
    fastpls::core::ConstMatrixView<T>(predictors.view()), prepared, labels,
    class_count, components, fitted, oversample, power, seed, xprod_mode,
    backend
  );
}

fastpls::core::SimplsControls simpls_controls(
    std::size_t samples, std::size_t predictors, std::size_t responses,
    std::size_t components, int oversample, int power, unsigned int seed) {
  fastpls::core::SimplsControls controls;
  controls.components = components;
  controls.maximum_block = fastpls::core::simpls_candidate_block_size(
    components, predictors, responses, true, samples, 64
  );
  const int maximum_predictors = environment_integer(
    "FASTPLS_FAST_CROSSPROD_MAX_P",
#ifdef FASTPLS_USE_ACCELERATE
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

template<class T, class Backend>
SEXP fit_simpls_label_core_prepared(
    fastpls::core::ConstMatrixView<T> predictors,
    const fastpls::core::LabelCrossprodResult<T>& prepared,
    const std::vector<std::size_t>& labels, int class_count,
    SEXP components, bool fitted, int oversample, int power,
    unsigned int seed, const char* xprod_mode, Backend& backend) {
  ProtectStack protect;
  SEXP effective_components = protect.add(Rf_duplicate(components));
  const std::size_t sample_rank = std::max<std::size_t>(
    predictors.rows() - 1, 1
  );
  const int rank_cap = static_cast<int>(std::min(
    predictors.columns(), sample_rank
  ));
  int maximum_components = 1;
  for (R_xlen_t index = 0; index < XLENGTH(effective_components); ++index) {
    const int value = INTEGER(effective_components)[index];
    if (value == NA_INTEGER) {
      throw std::invalid_argument("ncomp cannot contain missing values");
    }
    INTEGER(effective_components)[index] = std::max(
      1, std::min(value, rank_cap)
    );
    maximum_components = std::max(
      maximum_components, INTEGER(effective_components)[index]
    );
  }

  const auto controls = simpls_controls(
    predictors.rows(), predictors.columns(),
    static_cast<std::size_t>(class_count),
    static_cast<std::size_t>(maximum_components), oversample, power, seed
  );
  fastpls::core::SimplsWorkspace<T> workspace;
  const auto model = fastpls::core::fit_simpls_preprocessed<T>(
    predictors, prepared.crossprod.view(), controls, backend, workspace
  );
  if (model.completed_components < controls.components) {
    throw std::runtime_error(
      "fastPLS core SIMPLS returned fewer components than requested"
    );
  }

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
      fastpls::core::Matrix<T> values(predictors.rows(), class_count);
      backend.gemm(scores, loadings, false, true, values.view());
      r2_values[index] = fastpls::core::dummy_response_r2(
        labels.data(), labels.size(), prepared.response_mean.data(),
        static_cast<std::size_t>(class_count), values.view()
      );
      for (std::size_t response = 0;
           response < static_cast<std::size_t>(class_count); ++response) {
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
  SET_VECTOR_ELT(output, 3, core_matrix(model.scores));
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
  SET_VECTOR_ELT(output, 8, Rf_ScalarInteger(class_count));
  SET_VECTOR_ELT(output, 9, effective_components);
  SET_VECTOR_ELT(
    output, 10, fitted ? core_matrix_list(
      fitted_values, INTEGER(effective_components)
    ) : R_NilValue
  );
  SEXP r2 = protect.add(Rf_allocVector(REALSXP, XLENGTH(components)));
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

template<class T, class Backend>
SEXP fit_simpls_label_core(
    fastpls::core::Matrix<T>& predictors,
    const std::vector<std::size_t>& labels, int class_count,
    SEXP components, int scaling, bool fitted, int oversample, int power,
    unsigned int seed, const char* xprod_mode, Backend& backend) {
  const auto prepared = fastpls::core::prepare_scaled_label_crossprod(
    predictors.view(), labels.data(), labels.size(),
    static_cast<std::size_t>(class_count),
    static_cast<fastpls::core::PredictorScaling>(scaling), backend
  );
  return fit_simpls_label_core_prepared(
    fastpls::core::ConstMatrixView<T>(predictors.view()), prepared, labels,
    class_count, components, fitted, oversample, power, seed, xprod_mode,
    backend
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
    const auto models = fastpls::core::train_lda_prefixes(
      values.view(), INTEGER(labels), static_cast<std::size_t>(XLENGTH(labels)),
      static_cast<std::size_t>(classes), INTEGER(components),
      static_cast<std::size_t>(XLENGTH(components))
    );
    SEXP output = PROTECT(Rf_allocVector(VECSXP, models.size()));
    SEXP names = PROTECT(Rf_allocVector(STRSXP, models.size()));
    for (std::size_t index = 0; index < models.size(); ++index) {
      SET_VECTOR_ELT(output, index, float_lda_model(models[index]));
      SET_STRING_ELT(
        names, index, Rf_mkChar(std::to_string(INTEGER(components)[index]).c_str())
      );
    }
    Rf_setAttrib(output, R_NamesSymbol, names);
    UNPROTECT(2);
    return output;
  } catch (const std::exception& exception) {
    Rf_error("%s", exception.what());
  } catch (...) {
    Rf_error("Unknown error in float32 PLS-LDA fitting");
  }
  return R_NilValue;
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

extern "C" SEXP _fastPLS_cpu_backend_description() {
  try {
    return Rf_mkString(fastpls::runtime::cpu_backend_description().c_str());
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

extern "C" SEXP _fastPLS_pls_labels_core_cpp(
    SEXP predictors, SEXP labels, SEXP class_count, SEXP components,
    SEXP scaling, SEXP fit, SEXP oversample, SEXP power, SEXP seed) {
  return translate_exceptions("double core PLS-SVD fitting", [&] {
    if (TYPEOF(components) != INTSXP || XLENGTH(components) < 1) {
      throw std::invalid_argument(
        "double core PLS-SVD requires integer component counts"
      );
    }
    const int classes = Rf_asInteger(class_count);
    const int scaling_code = Rf_asInteger(scaling);
    const int fit_code = Rf_asLogical(fit);
    if (scaling_code < 1 || scaling_code > 3 || fit_code == NA_LOGICAL) {
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
      Rf_asInteger(oversample), Rf_asInteger(power),
      static_cast<unsigned int>(Rf_asInteger(seed)),
      "float64_label_class_sums", backend
    );
  });
}

extern "C" SEXP _fastPLS_pls_simpls_labels_core_cpp(
    SEXP predictors, SEXP labels, SEXP class_count, SEXP components,
    SEXP scaling, SEXP fit, SEXP oversample, SEXP power, SEXP seed) {
  return translate_exceptions("double core SIMPLS fitting", [&] {
    if (TYPEOF(components) != INTSXP || XLENGTH(components) < 1) {
      throw std::invalid_argument(
        "double core SIMPLS requires integer component counts"
      );
    }
    const int classes = Rf_asInteger(class_count);
    const int scaling_code = Rf_asInteger(scaling);
    const int fit_code = Rf_asLogical(fit);
    if (scaling_code < 1 || scaling_code > 3 || fit_code == NA_LOGICAL) {
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
      Rf_asInteger(oversample), Rf_asInteger(power),
      static_cast<unsigned int>(Rf_asInteger(seed)),
      "float64_label_class_sums_blocked", backend
    );
  });
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
    const bool plssvd_weights = TYPEOF(stored_weights) == VECSXP;
    fastpls::core::ConstMatrixView<double> simpls_loadings;
    if (!plssvd_weights) {
      simpls_loadings = numeric_matrix_view(response_loadings, "model$Q");
    }
    const int return_projection = Rf_asLogical(project);
    if (x.columns() != projection.rows() ||
        center.size() != x.columns() || scale.size() != x.columns() ||
        response_mean.empty() || TYPEOF(components) != INTSXP ||
        (plssvd_weights &&
          XLENGTH(components) != XLENGTH(stored_weights)) ||
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
      if (plssvd_weights) {
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

extern "C" SEXP _fastPLS_pls_float32_labels_core_cpp(
    SEXP predictors, SEXP labels, SEXP class_count, SEXP components,
    SEXP scaling, SEXP fit, SEXP method, SEXP oversample, SEXP power,
    SEXP seed) {
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
    const int scaling_code = Rf_asInteger(scaling);
    const int fit_code = Rf_asLogical(fit);
    const int method_code = Rf_asInteger(method);
    if (classes < 2 || scaling_code < 1 || scaling_code > 3 ||
        fit_code == NA_LOGICAL ||
        (method_code != 1 && method_code != 3) ||
        XLENGTH(labels) != static_cast<R_xlen_t>(x.rows())) {
      throw std::invalid_argument(
        "float32 core PLS training dimensions or controls are invalid"
      );
    }
    const auto encoded = encoded_class_labels(
      labels, x.rows(), classes, "float32 core PLS"
    );
    fastpls::runtime::CpuLinearAlgebraF32 backend;
    if (method_code == 1) {
      return fit_plssvd_label_core(
        x, encoded, classes, components, scaling_code, fit_code,
        Rf_asInteger(oversample), Rf_asInteger(power),
        static_cast<unsigned int>(Rf_asInteger(seed)),
        "float32_label_class_sums", backend
      );
    }
    const auto prepared = fastpls::core::prepare_scaled_label_crossprod(
      x.view(), encoded.data(), encoded.size(),
      static_cast<std::size_t>(classes),
      static_cast<fastpls::core::PredictorScaling>(scaling_code), backend
    );
    return fit_simpls_label_core_prepared(
      fastpls::core::ConstMatrixView<float>(x.view()), prepared, encoded,
      classes, components, fit_code, Rf_asInteger(oversample),
      Rf_asInteger(power), static_cast<unsigned int>(Rf_asInteger(seed)),
      "float32_label_class_sums_blocked", backend
    );
  });
}
