#include "metal_resident_backend.h"

#include <cstring>
#include <memory>

namespace {

arma::fmat float_bits_to_matrix(SEXP value, const char* name) {
  if (TYPEOF(value) != INTSXP || !Rf_isMatrix(value)) {
    Rcpp::stop("%s must be a float32 integer-bit matrix", name);
  }
  Rcpp::IntegerVector dimensions = Rf_getAttrib(value, R_DimSymbol);
  if (dimensions.size() != 2 || dimensions[0] < 1 || dimensions[1] < 1) {
    Rcpp::stop("%s must have positive matrix dimensions", name);
  }
  arma::fmat result(dimensions[0], dimensions[1]);
  std::memcpy(result.memptr(), INTEGER(value), result.n_elem * sizeof(float));
  return result;
}

SEXP matrix_to_float_bits(const arma::fmat& value) {
  Rcpp::Shield<SEXP> result(
      Rf_allocMatrix(INTSXP, static_cast<int>(value.n_rows),
                     static_cast<int>(value.n_cols)));
  std::memcpy(INTEGER(result), value.memptr(), value.n_elem * sizeof(float));
  return result;
}

SEXP cube_to_float_bits(const arma::fcube& value) {
  Rcpp::Shield<SEXP> result(Rf_allocVector(INTSXP, value.n_elem));
  std::memcpy(INTEGER(result), value.memptr(), value.n_elem * sizeof(float));
  Rf_setAttrib(result, R_DimSymbol, Rcpp::IntegerVector::create(
      static_cast<int>(value.n_rows), static_cast<int>(value.n_cols),
      static_cast<int>(value.n_slices)));
  return result;
}

struct MetalResidentRModel {
  explicit MetalResidentRModel(
      std::unique_ptr<fastpls_svd::MetalResidentModel> value)
      : model(std::move(value)) {}
  std::unique_ptr<fastpls_svd::MetalResidentModel> model;
};

Rcpp::XPtr<MetalResidentRModel> checked_model(Rcpp::List object) {
  if (!object.containsElementNamed("state")) {
    Rcpp::stop("resident Metal state is missing");
  }
  SEXP pointer = object["state"];
  if (TYPEOF(pointer) != EXTPTRSXP ||
      R_ExternalPtrTag(pointer) != Rf_install("fastPLS_metal_resident")) {
    Rcpp::stop("invalid resident Metal model pointer");
  }
  Rcpp::XPtr<MetalResidentRModel> state(pointer);
  if (!state.get() || !state->model) {
    Rcpp::stop("resident Metal state is unavailable; refit the model");
  }
  return state;
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List metal_resident_simpls_fit_cpp(
    SEXP X,
    SEXP Y,
    SEXP labels,
    int classes,
    int ncomp,
    int scaling,
    int oversample,
    int power,
    int seed,
    int method = 3,
    int north = 1,
    int kernel = 2,
    double gamma = 1.0,
    int degree = 3,
    double coefficient = 1.0) {
#ifdef FASTPLS_HAS_METAL
  arma::fmat x = float_bits_to_matrix(X, "X");
  std::unique_ptr<arma::fmat> y;
  std::unique_ptr<Rcpp::IntegerVector> encoded;
  if (labels != R_NilValue) {
    if (Y != R_NilValue || TYPEOF(labels) != INTSXP ||
        Rf_xlength(labels) != static_cast<R_xlen_t>(x.n_rows) || classes < 2) {
      Rcpp::stop("provide one integer class label per row and no dense response");
    }
    encoded.reset(new Rcpp::IntegerVector(labels));
  } else {
    if (Y == R_NilValue) Rcpp::stop("Y is required for regression");
    y.reset(new arma::fmat(float_bits_to_matrix(Y, "Y")));
    if (y->n_rows != x.n_rows) {
      Rcpp::stop("X and Y must have the same number of rows");
    }
    classes = static_cast<int>(y->n_cols);
  }
  if (method != 1 && method != 3 && method != 4 && method != 5) {
    Rcpp::stop(
        "resident Metal method must be PLS-SVD, SIMPLS, OPLS, or nonlinear kernel PLS");
  }
  auto model = fastpls_svd::metal_resident_simpls_create(
      x, y.get(), encoded.get(), classes, ncomp, scaling, oversample, power,
      static_cast<unsigned int>(seed), method, north, kernel,
      static_cast<float>(gamma), degree, static_cast<float>(coefficient));
  Rcpp::XPtr<MetalResidentRModel> state(
      new MetalResidentRModel(std::move(model)), true,
      Rf_install("fastPLS_metal_resident"));
  int effective_oversample = 0;
  int effective_power = 0;
  int refresh_block = 0;
  state->model->controls(effective_oversample, effective_power, refresh_block);
  return Rcpp::List::create(
      Rcpp::_["state"] = state,
      Rcpp::_["ncomp"] = state->model->components(),
      Rcpp::_["n"] = state->model->observations(),
      Rcpp::_["p"] = state->model->predictors(),
      Rcpp::_["q"] = state->model->responses(),
      Rcpp::_["classification"] = state->model->classification(),
      Rcpp::_["precision"] = 32L,
      Rcpp::_["resident"] = true,
      Rcpp::_["implicit_crosscovariance"] =
          state->model->implicit_crosscovariance(),
      Rcpp::_["predictor_crossprod_cache"] =
          state->model->predictor_crossprod_cache(),
      Rcpp::_["host_assisted_components"] =
          state->model->host_assisted_components(),
      Rcpp::_["effective_oversample"] = effective_oversample,
      Rcpp::_["effective_power"] = effective_power,
      Rcpp::_["refresh_block"] = refresh_block);
#else
  Rcpp::stop(
      "Resident Metal fitting is unavailable in this build; no CPU fallback is performed");
#endif
}

// [[Rcpp::export]]
SEXP metal_resident_export_cpp(Rcpp::List object, int field) {
#ifdef FASTPLS_HAS_METAL
  auto state = checked_model(object);
  return matrix_to_float_bits(state->model->export_field(field));
#else
  Rcpp::stop("Resident Metal export is unavailable in this build");
#endif
}

// [[Rcpp::export]]
void metal_resident_compact_cpp(Rcpp::List object, bool prepare_lda = false) {
#ifdef FASTPLS_HAS_METAL
  auto state = checked_model(object);
  state->model->compact(prepare_lda);
#else
  Rcpp::stop(
      "Resident Metal compaction is unavailable in this build; no CPU fallback is performed");
#endif
}

// [[Rcpp::export]]
SEXP metal_resident_predict_cpp(Rcpp::List object, SEXP X, int ncomp,
                                int classifier = 0) {
#ifdef FASTPLS_HAS_METAL
  auto state = checked_model(object);
  arma::fmat x = float_bits_to_matrix(X, "X");
  if (static_cast<int>(x.n_cols) != state->model->predictors()) {
    Rcpp::stop("test predictor dimension differs from the fitted model");
  }
  if (classifier < 0 || classifier > 1) {
    Rcpp::stop("invalid resident Metal classifier");
  }
  return matrix_to_float_bits(
      state->model->predict(x, ncomp, classifier == 1));
#else
  Rcpp::stop(
      "Resident Metal prediction is unavailable in this build; no CPU fallback is performed");
#endif
}

// [[Rcpp::export]]
SEXP metal_resident_predict_path_cpp(
    Rcpp::List object, SEXP X, Rcpp::IntegerVector ncomp,
    int classifier = 0) {
#ifdef FASTPLS_HAS_METAL
  auto state = checked_model(object);
  arma::fmat x = float_bits_to_matrix(X, "X");
  if (static_cast<int>(x.n_cols) != state->model->predictors()) {
    Rcpp::stop("test predictor dimension differs from the fitted model");
  }
  if (classifier < 0 || classifier > 1) {
    Rcpp::stop("invalid resident Metal classifier");
  }
  return cube_to_float_bits(
      state->model->predict_path(x, ncomp, classifier == 1));
#else
  Rcpp::stop(
      "Resident Metal prediction path is unavailable in this build; no CPU fallback is performed");
#endif
}

// [[Rcpp::export]]
SEXP metal_resident_project_cpp(Rcpp::List object, SEXP X, int ncomp) {
#ifdef FASTPLS_HAS_METAL
  auto state = checked_model(object);
  arma::fmat x = float_bits_to_matrix(X, "X");
  if (static_cast<int>(x.n_cols) != state->model->predictors()) {
    Rcpp::stop("test predictor dimension differs from the fitted model");
  }
  return matrix_to_float_bits(state->model->project(x, ncomp));
#else
  Rcpp::stop(
      "Resident Metal projection is unavailable in this build; no CPU fallback is performed");
#endif
}

// [[Rcpp::export]]
Rcpp::IntegerMatrix metal_resident_classify_cpp(
    Rcpp::List object, SEXP X, int ncomp, int top, int classifier = 0) {
#ifdef FASTPLS_HAS_METAL
  auto state = checked_model(object);
  arma::fmat x = float_bits_to_matrix(X, "X");
  if (static_cast<int>(x.n_cols) != state->model->predictors()) {
    Rcpp::stop("test predictor dimension differs from the fitted model");
  }
  if (classifier < 0 || classifier > 1) {
    Rcpp::stop("invalid resident Metal classifier");
  }
  return Rcpp::wrap(
      state->model->classify(x, ncomp, top, classifier == 1));
#else
  Rcpp::stop(
      "Resident Metal classification is unavailable in this build; no CPU fallback is performed");
#endif
}

// [[Rcpp::export]]
Rcpp::IntegerVector metal_resident_classify_path_cpp(
    Rcpp::List object, SEXP X, Rcpp::IntegerVector ncomp, int top,
    int classifier = 0) {
#ifdef FASTPLS_HAS_METAL
  auto state = checked_model(object);
  arma::fmat x = float_bits_to_matrix(X, "X");
  if (static_cast<int>(x.n_cols) != state->model->predictors()) {
    Rcpp::stop("test predictor dimension differs from the fitted model");
  }
  if (classifier < 0 || classifier > 1) {
    Rcpp::stop("invalid resident Metal classifier");
  }
  arma::icube result = state->model->classify_path(
      x, ncomp, top, classifier == 1);
  Rcpp::IntegerVector output(result.n_elem);
  for (arma::uword index = 0; index < result.n_elem; ++index) {
    output[index] = static_cast<int>(result[index]);
  }
  output.attr("dim") = Rcpp::IntegerVector::create(
      static_cast<int>(result.n_rows), static_cast<int>(result.n_cols),
      static_cast<int>(result.n_slices));
  return output;
#else
  Rcpp::stop(
      "Resident Metal component-path classification is unavailable in this build; no CPU fallback is performed");
#endif
}

// [[Rcpp::export]]
Rcpp::List metal_resident_classify_response_path_cpp(
    Rcpp::List object, SEXP X, Rcpp::IntegerVector ncomp, int top,
    int classifier = 0) {
#ifdef FASTPLS_HAS_METAL
  auto state = checked_model(object);
  arma::fmat x = float_bits_to_matrix(X, "X");
  if (static_cast<int>(x.n_cols) != state->model->predictors()) {
    Rcpp::stop("test predictor dimension differs from the fitted model");
  }
  if (classifier < 0 || classifier > 1) {
    Rcpp::stop("invalid resident Metal classifier");
  }
  arma::icube labels;
  arma::fcube predictions;
  state->model->classify_response_path(
      x, ncomp, top, classifier == 1, labels, predictions);
  Rcpp::IntegerVector encoded(labels.n_elem);
  for (arma::uword index = 0; index < labels.n_elem; ++index) {
    encoded[index] = static_cast<int>(labels[index]);
  }
  encoded.attr("dim") = Rcpp::IntegerVector::create(
      static_cast<int>(labels.n_rows), static_cast<int>(labels.n_cols),
      static_cast<int>(labels.n_slices));
  return Rcpp::List::create(
      Rcpp::Named("labels") = encoded,
      Rcpp::Named("predictions") = cube_to_float_bits(predictions));
#else
  Rcpp::stop(
      "Resident Metal classification-response path is unavailable in this build; no CPU fallback is performed");
#endif
}

// [[Rcpp::export]]
SEXP metal_resident_response_sums_cpp(
    Rcpp::List object, SEXP X, SEXP Y, SEXP labels, int ncomp) {
#ifdef FASTPLS_HAS_METAL
  auto state = checked_model(object);
  arma::fmat x = float_bits_to_matrix(X, "X");
  if (static_cast<int>(x.n_cols) != state->model->predictors()) {
    Rcpp::stop("test predictor dimension differs from the fitted model");
  }
  std::unique_ptr<arma::fmat> y;
  std::unique_ptr<Rcpp::IntegerVector> encoded;
  if (labels != R_NilValue) {
    if (Y != R_NilValue || TYPEOF(labels) != INTSXP ||
        Rf_xlength(labels) != static_cast<R_xlen_t>(x.n_rows)) {
      Rcpp::stop("invalid observed class labels");
    }
    encoded.reset(new Rcpp::IntegerVector(labels));
  } else {
    if (Y == R_NilValue) Rcpp::stop("observed responses are required");
    y.reset(new arma::fmat(float_bits_to_matrix(Y, "Y")));
    if (y->n_rows != x.n_rows ||
        static_cast<int>(y->n_cols) != state->model->responses()) {
      Rcpp::stop("observed response dimensions differ from predictions");
    }
  }
  return matrix_to_float_bits(
      state->model->response_sums(x, y.get(), encoded.get(), ncomp));
#else
  Rcpp::stop(
      "Resident Metal metrics are unavailable in this build; no CPU fallback is performed");
#endif
}
