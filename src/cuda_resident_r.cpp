#include "r_api.h"
#include "cuda_resident_api.h"

#include <R_ext/Error.h>

#include <climits>
#include <cstddef>
#include <cstring>
#include <new>

namespace {

#ifdef FASTPLS_HAS_CUDA

struct ResidentRModel {
  ResidentRModel(void* value, int observations, int predictors,
                 int feature_predictors, int responses, int components,
                 int precision_bits)
      : handle(value),
        n(observations),
        p(predictors),
        feature_p(feature_predictors),
        q(responses),
        a(components),
        precision(precision_bits) {}

  void* handle;
  int n;
  int p;
  int feature_p;
  int q;
  int a;
  int precision;
};

struct MatrixInput {
  const void* values;
  int rows;
  int columns;
};

SEXP resident_tag() {
  return Rf_install("fastPLS_cuda_resident_simpls");
}

SEXP list_element(SEXP object, const char* name) {
  if (TYPEOF(object) != VECSXP) {
    Rf_error("resident CUDA object must be a list");
  }
  SEXP names = Rf_getAttrib(object, R_NamesSymbol);
  if (TYPEOF(names) != STRSXP) {
    return R_NilValue;
  }
  for (R_xlen_t index = 0; index < XLENGTH(object); ++index) {
    if (STRING_ELT(names, index) != NA_STRING &&
        !std::strcmp(CHAR(STRING_ELT(names, index)), name)) {
      return VECTOR_ELT(object, index);
    }
  }
  return R_NilValue;
}

MatrixInput matrix_input(SEXP value, int precision, const char* name) {
  const SEXPTYPE expected = precision == 32 ? INTSXP : REALSXP;
  if (TYPEOF(value) != expected || !Rf_isMatrix(value)) {
    Rf_error(
      "%s must be a numeric matrix or float32 integer-bit matrix of the requested precision",
      name
    );
  }
  SEXP dimensions = Rf_getAttrib(value, R_DimSymbol);
  if (TYPEOF(dimensions) != INTSXP || XLENGTH(dimensions) != 2 ||
      INTEGER(dimensions)[0] < 1 || INTEGER(dimensions)[1] < 1) {
    Rf_error("%s dimensions must be positive", name);
  }
  return {
    precision == 32
      ? static_cast<const void*>(INTEGER(value))
      : static_cast<const void*>(REAL(value)),
    INTEGER(dimensions)[0], INTEGER(dimensions)[1]
  };
}

void* matrix_output(SEXP value, int precision) {
  return precision == 32
    ? static_cast<void*>(INTEGER(value))
    : static_cast<void*>(REAL(value));
}

SEXP allocate_matrix(int precision, int rows, int columns) {
  return Rf_allocMatrix(precision == 32 ? INTSXP : REALSXP, rows, columns);
}

int scalar_integer(SEXP value, const char* name) {
  const int result = Rf_asInteger(value);
  if (result == NA_INTEGER) {
    Rf_error("%s must be a finite integer", name);
  }
  return result;
}

bool scalar_logical(SEXP value, const char* name) {
  const int result = Rf_asLogical(value);
  if (result == NA_LOGICAL) {
    Rf_error("%s must be TRUE or FALSE", name);
  }
  return result == TRUE;
}

ResidentRModel* checked_state(SEXP object) {
  SEXP pointer = list_element(object, "state");
  if (pointer == R_NilValue) {
    Rf_error("resident CUDA state is missing");
  }
  if (TYPEOF(pointer) != EXTPTRSXP ||
      R_ExternalPtrTag(pointer) != resident_tag()) {
    Rf_error("invalid resident CUDA model pointer");
  }
  auto* state = static_cast<ResidentRModel*>(R_ExternalPtrAddr(pointer));
  if (state == nullptr || state->handle == nullptr) {
    Rf_error("resident CUDA state is unavailable; refit the model");
  }
  return state;
}

void finalize_resident(SEXP pointer) {
  auto* state = static_cast<ResidentRModel*>(R_ExternalPtrAddr(pointer));
  if (state != nullptr) {
    if (state->handle != nullptr) {
      fastpls_resident_simpls_destroy(state->handle);
      state->handle = nullptr;
    }
    delete state;
    R_ClearExternalPtr(pointer);
  }
}

const int* checked_prefixes(SEXP prefixes, ResidentRModel* state,
                            int* count) {
  if (TYPEOF(prefixes) != INTSXP || XLENGTH(prefixes) < 1) {
    Rf_error("ncomp must be a non-empty integer vector");
  }
  if (XLENGTH(prefixes) > INT_MAX) {
    Rf_error("ncomp is too long");
  }
  *count = static_cast<int>(XLENGTH(prefixes));
  int previous = 0;
  for (int index = 0; index < *count; ++index) {
    const int value = INTEGER(prefixes)[index];
    if (value <= previous || value > state->a) {
      Rf_error("ncomp must be strictly increasing and within the fitted path");
    }
    previous = value;
  }
  return INTEGER(prefixes);
}

SEXP allocate_array3(SEXPTYPE type, int first, int second, int third) {
  SEXP result = PROTECT(Rf_allocVector(
    type, static_cast<R_xlen_t>(first) * second * third
  ));
  SEXP dimensions = PROTECT(Rf_allocVector(INTSXP, 3));
  INTEGER(dimensions)[0] = first;
  INTEGER(dimensions)[1] = second;
  INTEGER(dimensions)[2] = third;
  Rf_setAttrib(result, R_DimSymbol, dimensions);
  UNPROTECT(2);
  return result;
}

#else

template<class... Arguments>
void ignore(Arguments&&...) {}

[[noreturn]] void unavailable(const char* operation) {
  Rf_error(
    "CUDA resident %s is unavailable in this build; no CPU fallback is performed",
    operation
  );
}

#endif

}  // namespace

extern "C" SEXP _fastPLS_cuda_resident_project_cpp(
    SEXP object, SEXP predictors, SEXP components) {
#ifdef FASTPLS_HAS_CUDA
  ResidentRModel* state = checked_state(object);
  MatrixInput input = matrix_input(predictors, state->precision, "X");
  const int prefix = scalar_integer(components, "ncomp");
  if (input.columns != state->p || prefix < 1 || prefix > state->a) {
    Rf_error("invalid resident score dimensions");
  }
  SEXP result = PROTECT(allocate_matrix(state->precision, input.rows, prefix));
  char error[1024] = {};
  if (fastpls_resident_project(
      state->handle, input.values, input.rows, prefix,
      matrix_output(result, state->precision), error, sizeof(error))) {
    UNPROTECT(1);
    Rf_error("%s", error);
  }
  UNPROTECT(1);
  return result;
#else
  ignore(object, predictors, components);
  unavailable("projection");
#endif
}

extern "C" SEXP _fastPLS_cuda_resident_response_sums_cpp(
    SEXP object, SEXP predictors, SEXP response, SEXP labels,
    SEXP components) {
#ifdef FASTPLS_HAS_CUDA
  ResidentRModel* state = checked_state(object);
  MatrixInput input = matrix_input(predictors, state->precision, "X");
  if (input.columns != state->p) {
    Rf_error("test predictor dimension differs from the fitted model");
  }
  const void* y = nullptr;
  const int* encoded = nullptr;
  if (labels != R_NilValue) {
    if (response != R_NilValue || TYPEOF(labels) != INTSXP ||
        XLENGTH(labels) != input.rows) {
      Rf_error("invalid observed class labels");
    }
    encoded = INTEGER(labels);
  } else {
    MatrixInput observed = matrix_input(response, state->precision, "Y");
    if (observed.rows != input.rows || observed.columns != state->q) {
      Rf_error("observed response dimensions differ from predictions");
    }
    y = observed.values;
  }
  SEXP result = PROTECT(allocate_matrix(state->precision, 3, state->q));
  char error[1024] = {};
  if (fastpls_resident_response_sums(
      state->handle, input.values, y, encoded, input.rows,
      scalar_integer(components, "ncomp"),
      matrix_output(result, state->precision), error, sizeof(error))) {
    UNPROTECT(1);
    Rf_error("%s", error);
  }
  UNPROTECT(1);
  return result;
#else
  ignore(object, predictors, response, labels, components);
  unavailable("metrics");
#endif
}

extern "C" SEXP _fastPLS_cuda_resident_simpls_fit_cpp(
    SEXP predictors, SEXP response, SEXP labels, SEXP classes,
    SEXP precision_value, SEXP components, SEXP scaling, SEXP oversample,
    SEXP power, SEXP seed, SEXP retain_scores, SEXP method_value,
    SEXP north, SEXP kernel, SEXP gamma, SEXP degree, SEXP coefficient) {
#ifdef FASTPLS_HAS_CUDA
  const int precision = scalar_integer(precision_value, "precision");
  if (precision != 32 && precision != 64) {
    Rf_error("precision must be 32 or 64");
  }
  MatrixInput x = matrix_input(predictors, precision, "X");
  int q = scalar_integer(classes, "classes");
  const int* encoded = nullptr;
  const void* y = nullptr;
  if (labels != R_NilValue) {
    if (response != R_NilValue || TYPEOF(labels) != INTSXP ||
        XLENGTH(labels) != x.rows || q < 2) {
      Rf_error("provide one integer class label per row and no dense response");
    }
    encoded = INTEGER(labels);
  } else {
    MatrixInput observed = matrix_input(response, precision, "Y");
    if (observed.rows != x.rows) {
      Rf_error("response and predictor row counts differ");
    }
    q = observed.columns;
    y = observed.values;
  }
  const int method = scalar_integer(method_value, "method");
  if (method != 1 && method != 3 && method != 4 && method != 5) {
    Rf_error(
      "resident core supports PLS-SVD, SIMPLS, OPLS, or nonlinear kernel PLS"
    );
  }
  const int requested = scalar_integer(components, "ncomp");
  const int scale = scalar_integer(scaling, "scaling");
  const int extra = scalar_integer(oversample, "oversample");
  const int iterations = scalar_integer(power, "power");
  const unsigned long long random_seed =
    static_cast<unsigned int>(scalar_integer(seed, "seed"));
  const int keep_scores = scalar_logical(retain_scores, "retain_scores");
  char error[1024] = {};
  void* handle = nullptr;
  if (method == 1) {
    handle = fastpls_resident_plssvd_create(
      x.values, y, encoded, precision, x.rows, x.columns, q, requested,
      scale, extra, iterations, keep_scores, random_seed, error, sizeof(error)
    );
  } else if (method == 3) {
    handle = fastpls_resident_simpls_create(
      x.values, y, encoded, precision, x.rows, x.columns, q, requested,
      scale, extra, iterations, keep_scores, random_seed, error, sizeof(error)
    );
  } else if (method == 4) {
    handle = fastpls_resident_opls_create(
      x.values, y, encoded, precision, x.rows, x.columns, q, requested,
      scale, extra, iterations, keep_scores, random_seed,
      scalar_integer(north, "north"), error, sizeof(error)
    );
  } else {
    handle = fastpls_resident_kernelpls_create(
      x.values, y, encoded, precision, x.rows, x.columns, q, requested,
      scale, extra, iterations, keep_scores, random_seed,
      scalar_integer(kernel, "kernel"), Rf_asReal(gamma),
      scalar_integer(degree, "degree"), Rf_asReal(coefficient),
      error, sizeof(error)
    );
  }
  if (handle == nullptr) {
    Rf_error("%s", error);
  }
  int effective_oversample = 0;
  int effective_power = 0;
  int refresh_block = 0;
  int refresh_block_limit = 0;
  int implicit_operator = 0;
  int predictor_crossprod_cache = 0;
  if (fastpls_resident_controls(
      handle, &effective_oversample, &effective_power, &refresh_block,
      &refresh_block_limit, &implicit_operator, &predictor_crossprod_cache,
      error, sizeof(error))) {
    fastpls_resident_simpls_destroy(handle);
    Rf_error("%s", error);
  }
  auto* state = new (std::nothrow) ResidentRModel(
    handle, x.rows, x.columns, method == 5 ? x.rows : x.columns,
    q, requested, precision
  );
  if (state == nullptr) {
    fastpls_resident_simpls_destroy(handle);
    Rf_error("unable to allocate the resident CUDA model wrapper");
  }
  SEXP pointer = PROTECT(R_MakeExternalPtr(state, resident_tag(), R_NilValue));
  R_RegisterCFinalizerEx(pointer, finalize_resident, TRUE);
  constexpr int field_count = 10;
  SEXP result = PROTECT(Rf_allocVector(VECSXP, field_count));
  SEXP names = PROTECT(Rf_allocVector(STRSXP, field_count));
  const char* const labels_out[field_count] = {
    "state", "ncomp", "precision", "resident", "effective_oversample",
    "effective_power", "refresh_block", "refresh_block_limit",
    "implicit_crosscovariance", "predictor_crossprod_cache"
  };
  SET_VECTOR_ELT(result, 0, pointer);
  SET_VECTOR_ELT(result, 1, Rf_ScalarInteger(requested));
  SET_VECTOR_ELT(result, 2, Rf_ScalarInteger(precision));
  SET_VECTOR_ELT(result, 3, Rf_ScalarLogical(TRUE));
  SET_VECTOR_ELT(result, 4, Rf_ScalarInteger(effective_oversample));
  SET_VECTOR_ELT(result, 5, Rf_ScalarInteger(effective_power));
  SET_VECTOR_ELT(result, 6, Rf_ScalarInteger(refresh_block));
  SET_VECTOR_ELT(result, 7, Rf_ScalarInteger(refresh_block_limit));
  SET_VECTOR_ELT(result, 8, Rf_ScalarLogical(implicit_operator == 1));
  SET_VECTOR_ELT(result, 9, Rf_ScalarLogical(predictor_crossprod_cache == 1));
  for (int index = 0; index < field_count; ++index) {
    SET_STRING_ELT(names, index, Rf_mkChar(labels_out[index]));
  }
  Rf_setAttrib(result, R_NamesSymbol, names);
  UNPROTECT(3);
  return result;
#else
  ignore(predictors, response, labels, classes, precision_value, components,
         scaling, oversample, power, seed, retain_scores, method_value, north,
         kernel, gamma, degree, coefficient);
  unavailable("fitting");
#endif
}

extern "C" SEXP _fastPLS_cuda_resident_export_cpp(
    SEXP object, SEXP loadings_value, SEXP variance_value,
    SEXP scores_value) {
#ifdef FASTPLS_HAS_CUDA
  ResidentRModel* state = checked_state(object);
  const bool loadings = scalar_logical(loadings_value, "loadings");
  const bool variance = scalar_logical(variance_value, "variance");
  const bool scores = scalar_logical(scores_value, "scores");
  const int rows[] = {
    state->feature_p, state->q, state->n, 1, 1, 1, state->feature_p, 1
  };
  const int columns[] = {
    state->a, state->a, state->a, state->p, state->p, state->q,
    state->a, state->a + 1
  };
  const char* const field_names[] = {
    "R", "Q", "Ttrain", "mX", "vX", "mY", "P", "predictor_ss"
  };
  const int count = 5 + static_cast<int>(scores) +
    static_cast<int>(loadings) + static_cast<int>(variance);
  SEXP result = PROTECT(Rf_allocVector(VECSXP, count));
  SEXP names = PROTECT(Rf_allocVector(STRSXP, count));
  int output_index = 0;
  for (int field = 0; field < 8; ++field) {
    if ((field == 2 && !scores) || (field == 6 && !loadings) ||
        (field == 7 && !variance)) {
      continue;
    }
    SEXP value = PROTECT(allocate_matrix(
      state->precision, rows[field], columns[field]
    ));
    char error[1024] = {};
    if (fastpls_resident_export(
        state->handle, field, matrix_output(value, state->precision),
        static_cast<std::size_t>(rows[field]) * columns[field],
        error, sizeof(error))) {
      UNPROTECT(3);
      Rf_error("%s", error);
    }
    SET_VECTOR_ELT(result, output_index, value);
    SET_STRING_ELT(names, output_index, Rf_mkChar(field_names[field]));
    ++output_index;
    UNPROTECT(1);
  }
  Rf_setAttrib(result, R_NamesSymbol, names);
  UNPROTECT(2);
  return result;
#else
  ignore(object, loadings_value, variance_value, scores_value);
  unavailable("model export");
#endif
}

extern "C" SEXP _fastPLS_cuda_resident_compact_cpp(
    SEXP object, SEXP prepare_lda) {
#ifdef FASTPLS_HAS_CUDA
  ResidentRModel* state = checked_state(object);
  char error[1024] = {};
  if (fastpls_resident_compact(
      state->handle, scalar_logical(prepare_lda, "prepare_lda"),
      error, sizeof(error))) {
    Rf_error("%s", error);
  }
  return R_NilValue;
#else
  ignore(object, prepare_lda);
  unavailable("compaction");
#endif
}

extern "C" SEXP _fastPLS_cuda_resident_classify_cpp(
    SEXP object, SEXP predictors, SEXP components, SEXP classifier_value,
    SEXP top_value) {
#ifdef FASTPLS_HAS_CUDA
  ResidentRModel* state = checked_state(object);
  MatrixInput input = matrix_input(predictors, state->precision, "X");
  const int top = scalar_integer(top_value, "top");
  if (input.columns != state->p || top < 1 || top > state->q) {
    Rf_error("invalid predictor dimension or top-k request");
  }
  SEXP result = PROTECT(Rf_allocMatrix(INTSXP, input.rows, top));
  char error[1024] = {};
  if (fastpls_resident_classify(
      state->handle, input.values, input.rows,
      scalar_integer(components, "ncomp"),
      scalar_integer(classifier_value, "classifier"), top,
      INTEGER(result), error, sizeof(error))) {
    UNPROTECT(1);
    Rf_error("%s", error);
  }
  UNPROTECT(1);
  return result;
#else
  ignore(object, predictors, components, classifier_value, top_value);
  unavailable("classification");
#endif
}

extern "C" SEXP _fastPLS_cuda_resident_classify_path_cpp(
    SEXP object, SEXP predictors, SEXP components, SEXP classifier_value,
    SEXP top_value) {
#ifdef FASTPLS_HAS_CUDA
  ResidentRModel* state = checked_state(object);
  MatrixInput input = matrix_input(predictors, state->precision, "X");
  int count = 0;
  const int* prefixes = checked_prefixes(components, state, &count);
  const int top = scalar_integer(top_value, "top");
  if (input.columns != state->p || top < 1 || top > state->q) {
    Rf_error("invalid predictor dimension, component path, or top-k request");
  }
  SEXP result = PROTECT(allocate_array3(INTSXP, input.rows, top, count));
  char error[1024] = {};
  if (fastpls_resident_classify_path(
      state->handle, input.values, input.rows, prefixes, count,
      scalar_integer(classifier_value, "classifier"), top,
      INTEGER(result), error, sizeof(error))) {
    UNPROTECT(1);
    Rf_error("%s", error);
  }
  UNPROTECT(1);
  return result;
#else
  ignore(object, predictors, components, classifier_value, top_value);
  unavailable("classification path");
#endif
}

extern "C" SEXP _fastPLS_cuda_resident_classify_response_path_cpp(
    SEXP object, SEXP predictors, SEXP components, SEXP classifier_value,
    SEXP top_value) {
#ifdef FASTPLS_HAS_CUDA
  ResidentRModel* state = checked_state(object);
  MatrixInput input = matrix_input(predictors, state->precision, "X");
  int count = 0;
  const int* prefixes = checked_prefixes(components, state, &count);
  const int top = scalar_integer(top_value, "top");
  const int classifier = scalar_integer(classifier_value, "classifier");
  if (input.columns != state->p || top < 1 || top > state->q) {
    Rf_error("invalid predictor dimension, component path, or top-k request");
  }
  if (classifier != 0 && classifier != 1) {
    Rf_error("invalid resident classifier");
  }
  SEXP labels = PROTECT(allocate_array3(INTSXP, input.rows, top, count));
  SEXP predictions = PROTECT(allocate_array3(
    state->precision == 32 ? INTSXP : REALSXP,
    input.rows, state->q, count
  ));
  char error[1024] = {};
  if (fastpls_resident_classify_response_path(
      state->handle, input.values, input.rows, prefixes, count, classifier,
      top, INTEGER(labels), matrix_output(predictions, state->precision),
      error, sizeof(error))) {
    UNPROTECT(2);
    Rf_error("%s", error);
  }
  SEXP result = PROTECT(Rf_allocVector(VECSXP, 2));
  SEXP names = PROTECT(Rf_allocVector(STRSXP, 2));
  SET_VECTOR_ELT(result, 0, labels);
  SET_VECTOR_ELT(result, 1, predictions);
  SET_STRING_ELT(names, 0, Rf_mkChar("labels"));
  SET_STRING_ELT(names, 1, Rf_mkChar("predictions"));
  Rf_setAttrib(result, R_NamesSymbol, names);
  UNPROTECT(4);
  return result;
#else
  ignore(object, predictors, components, classifier_value, top_value);
  unavailable("classification-response path");
#endif
}

extern "C" SEXP _fastPLS_cuda_resident_simpls_predict_cpp(
    SEXP object, SEXP predictors, SEXP components, SEXP classifier_value) {
#ifdef FASTPLS_HAS_CUDA
  ResidentRModel* state = checked_state(object);
  MatrixInput input = matrix_input(predictors, state->precision, "X");
  if (input.columns != state->p) {
    Rf_error("test predictor dimension differs from the fitted model");
  }
  const int classifier = scalar_integer(classifier_value, "classifier");
  if (classifier != 0 && classifier != 1) {
    Rf_error("invalid resident classifier");
  }
  SEXP result = PROTECT(allocate_matrix(
    state->precision, input.rows, state->q
  ));
  char error[1024] = {};
  const auto predict = classifier == 1
    ? fastpls_resident_lda_predict
    : fastpls_resident_simpls_predict;
  if (predict(
      state->handle, input.values, input.rows,
      scalar_integer(components, "ncomp"),
      matrix_output(result, state->precision), error, sizeof(error))) {
    UNPROTECT(1);
    Rf_error("%s", error);
  }
  UNPROTECT(1);
  return result;
#else
  ignore(object, predictors, components, classifier_value);
  unavailable("prediction");
#endif
}

extern "C" SEXP _fastPLS_cuda_resident_predict_path_cpp(
    SEXP object, SEXP predictors, SEXP components, SEXP classifier_value) {
#ifdef FASTPLS_HAS_CUDA
  ResidentRModel* state = checked_state(object);
  MatrixInput input = matrix_input(predictors, state->precision, "X");
  int count = 0;
  const int* prefixes = checked_prefixes(components, state, &count);
  const int classifier = scalar_integer(classifier_value, "classifier");
  if (input.columns != state->p) {
    Rf_error("invalid predictor dimension or empty component path");
  }
  if (classifier != 0 && classifier != 1) {
    Rf_error("invalid resident classifier");
  }
  SEXP result = PROTECT(allocate_array3(
    state->precision == 32 ? INTSXP : REALSXP,
    input.rows, state->q, count
  ));
  char error[1024] = {};
  if (fastpls_resident_predict_path(
      state->handle, input.values, input.rows, prefixes, count, classifier,
      matrix_output(result, state->precision), error, sizeof(error))) {
    UNPROTECT(1);
    Rf_error("%s", error);
  }
  UNPROTECT(1);
  return result;
#else
  ignore(object, predictors, components, classifier_value);
  unavailable("prediction path");
#endif
}
