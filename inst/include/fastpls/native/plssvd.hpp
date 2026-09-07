// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_NATIVE_PLSSVD_HPP
#define FASTPLS_NATIVE_PLSSVD_HPP
#include <fastpls/native/simpls.hpp>

namespace fastpls { namespace native {

struct PlssvdOptions {
  int scaling = 1;
  bool fitted = false;
  bool store_coefficients = false;
  bool cache_score_gram = true;
  RsvdControls svd;
};

template<typename Scalar>
struct PlssvdModel {
  arma::Mat<Scalar> R, Q, scores, x_mean, x_scale, y_mean;
  arma::Cube<Scalar> latent_coefficients, prediction_weights, coefficients, fitted;
  arma::Col<Scalar> r2;
  arma::ivec components;
};

template<typename Scalar, typename Solver>
PlssvdModel<Scalar> fit_plssvd_with_solver(
    const arma::Mat<Scalar>& Xinput, const arma::Mat<Scalar>& Yinput,
    arma::ivec components, const PlssvdOptions& options, Solver&& solver,
    arma::Mat<Scalar>* owned_X = nullptr,
    const arma::uvec* compact_labels = nullptr, int compact_classes = 0) {
  using namespace arma;
  const bool label_response = compact_labels != nullptr;
  if (Xinput.n_rows < 2 || Xinput.n_cols < 1 ||
      (!label_response && (Yinput.n_cols < 1 || Xinput.n_rows != Yinput.n_rows)) ||
      (label_response && (compact_classes < 2 ||
                          compact_labels->n_elem != Xinput.n_rows))) {
    throw std::invalid_argument("PLS-SVD requires matching, nonempty training matrices");
  }
  if (owned_X && (owned_X->n_rows != Xinput.n_rows ||
                  owned_X->n_cols != Xinput.n_cols)) {
    throw std::invalid_argument("PLS-SVD owned predictor workspace has invalid dimensions");
  }
  if (components.is_empty()) {
    throw std::invalid_argument("ncomp must contain at least one value");
  }
  const int n = Xinput.n_rows, p = Xinput.n_cols;
  const int m = label_response ? compact_classes : Yinput.n_cols;
  const int rank_bound = std::min({n, p, m});
  for (auto& count : components) count = std::max(1, std::min(int(count), rank_bound));
  int retained = components.max();
  const int requested = components.n_elem;
  PlssvdModel<Scalar> model;
  model.components = components;
  model.x_mean.zeros(1, p);
  model.x_scale.ones(1, p);
  arma::Mat<Scalar> Xwork;
  arma::Mat<Scalar>& scaled_X = owned_X == nullptr ? Xwork : *owned_X;
  const arma::Mat<Scalar>* Xptr = &Xinput;
  if (options.scaling < 3) {
    if (owned_X == nullptr) scaled_X = Xinput;
    model.x_mean = mean(scaled_X, 0);
    scaled_X.each_row() -= model.x_mean;
    Xptr = &scaled_X;
  }
  if (options.scaling == 2) {
    model.x_scale = column_standard_deviation(scaled_X);
    scaled_X.each_row() /= model.x_scale;
  }
  const arma::Mat<Scalar>& X = *Xptr;
  arma::uvec one_hot_labels;
  const bool one_hot_response = label_response ||
    extract_one_hot_labels(Yinput, one_hot_labels);
  if (label_response) {
    one_hot_labels = *compact_labels;
    model.y_mean.zeros(1, m);
    for (arma::uword row = 0; row < one_hot_labels.n_elem; ++row) {
      if (one_hot_labels(row) >= static_cast<arma::uword>(m)) {
        throw std::invalid_argument("PLS-SVD compact class label is out of range");
      }
      model.y_mean(one_hot_labels(row)) += Scalar(1);
    }
    if (arma::accu(model.y_mean <= Scalar(0)) > 0) {
      throw std::invalid_argument("PLS-SVD compact labels contain an empty class");
    }
    model.y_mean /= static_cast<Scalar>(n);
  } else {
    model.y_mean = mean(Yinput, 0);
  }
  const bool need_centered_response = !one_hot_response ||
    (options.fitted && !label_response) ||
    !options.cache_score_gram;
  Mat<Scalar> Y;
  if (need_centered_response) {
    Y = Yinput;
    Y.each_row() -= model.y_mean;
  }
  Mat<Scalar> S = one_hot_response ?
    dummy_crossprod<Scalar>(X, one_hot_labels, -model.y_mean) : X.t() * Y;
  auto svd = solver(S, retained, rank_bound);
  Mat<Scalar> U = std::move(svd.U);
  Col<Scalar> singular = std::move(svd.s);
  Mat<Scalar> V = svd.Vt.t();
  if (options.store_coefficients) model.coefficients.zeros(p, m, requested);
  model.latent_coefficients.zeros(retained, retained, requested);
  model.prediction_weights.zeros(retained, m, requested);
  if (options.fitted) model.fitted.set_size(n, m, requested);
  model.r2.zeros(requested);
  retained = std::min(retained, int(U.n_cols));
  if (V.n_cols > 0) retained = std::min(retained, int(V.n_cols));
  if (retained < 1 || singular.n_elem < arma::uword(retained) || V.n_cols < 1) {
    throw std::runtime_error("PLS-SVD effective rank is below one after decomposition");
  }
  U = U.cols(0, retained - 1);
  if (V.n_cols > arma::uword(retained)) V = V.cols(0, retained - 1);
  model.R = std::move(U);
  model.Q = std::move(V);
  model.scores = X * model.R;
  Mat<Scalar> full_gram;
  if (options.cache_score_gram) full_gram = model.scores.t() * model.scores;
  for (int index = 0; index < requested; ++index) {
    const int count = std::min(int(components(index)), retained);
    Mat<Scalar> R = model.R.cols(0, count - 1);
    Mat<Scalar> Q = model.Q.cols(0, count - 1);
    Mat<Scalar> scores = model.scores.cols(0, count - 1);
    Mat<Scalar> gram, rhs, coeff;
    if (options.cache_score_gram) {
      gram = full_gram.submat(0, 0, count - 1, count - 1);
      rhs.zeros(count, count);
      rhs.diag() = singular.subvec(0, count - 1);
    } else {
      Mat<Scalar> response_scores = Y * Q;
      Mat<Scalar> score_transpose = scores.t();
      gram = score_transpose * scores;
      rhs = score_transpose * response_scores;
    }
    bool solved = arma::solve(coeff, gram, rhs, arma::solve_opts::likely_sympd);
    if (!solved) solved = arma::solve(coeff, gram, rhs);
    if (!solved) throw std::runtime_error("PLS-SVD latent solve failed");
    if (options.cache_score_gram) {
      model.latent_coefficients.slice(index).submat(0, 0, count - 1, count - 1) = coeff;
    } else {
      Mat<Scalar> diagonal(count, count, arma::fill::zeros), prediction_coeff;
      diagonal.diag() = singular.subvec(0, count - 1);
      bool predicted = arma::solve(prediction_coeff, gram, diagonal,
                                   arma::solve_opts::likely_sympd);
      if (!predicted) predicted = arma::solve(prediction_coeff, gram, diagonal);
      if (predicted) {
        model.latent_coefficients.slice(index).submat(0, 0, count - 1, count - 1) =
          prediction_coeff;
      }
    }
    Mat<Scalar> weights = coeff * Q.t();
    model.prediction_weights.slice(index).submat(0, 0, count - 1, m - 1) = weights;
    if (options.store_coefficients) model.coefficients.slice(index) = R * weights;
    if (options.fitted) {
      Mat<Scalar> predicted = scores * weights;
      model.r2(index) = label_response ?
        fitted_r2_labels(one_hot_labels, model.y_mean, predicted) :
        fitted_r2(Y, predicted);
      predicted.each_row() += model.y_mean;
      model.fitted.slice(index) = predicted;
    }
  }
  return model;
}

template<typename Scalar>
PlssvdModel<Scalar> fit_plssvd(
    arma::Mat<Scalar> X, arma::Mat<Scalar> Y, arma::ivec components,
    const PlssvdOptions& options = {}) {
  auto solver = [&](const arma::Mat<Scalar>& S, int rank, int) {
    return rsvd(S, rank, options.svd);
  };
  return fit_plssvd_with_solver(X, Y, std::move(components), options, solver);
}

template<typename Scalar>
arma::Mat<Scalar> predict_plssvd(const PlssvdModel<Scalar>& model,
                                arma::Mat<Scalar> X, arma::uword prefix_index) {
  if (X.n_cols != model.R.n_rows || prefix_index >= model.components.n_elem ||
      model.R.n_cols == 0 || model.components(prefix_index) < 1 ||
      prefix_index >= model.prediction_weights.n_slices ||
      model.prediction_weights.n_rows < model.R.n_cols ||
      model.prediction_weights.n_cols == 0 ||
      model.x_mean.n_rows != 1 || model.x_mean.n_cols != X.n_cols ||
      model.x_scale.n_rows != 1 || model.x_scale.n_cols != X.n_cols ||
      model.y_mean.n_rows != 1 ||
      model.y_mean.n_cols != model.prediction_weights.n_cols) {
    throw std::invalid_argument("PLS-SVD prediction dimensions or prefix index are invalid");
  }
  const arma::uword count = std::min(arma::uword(model.components(prefix_index)),
                                    model.R.n_cols);
  X.each_row() -= model.x_mean;
  X.each_row() /= model.x_scale;
  arma::Mat<Scalar> scores = X * model.R.cols(0, count - 1);
  arma::Mat<Scalar> prediction = scores *
    model.prediction_weights.slice(prefix_index).rows(0, count - 1);
  prediction.each_row() += model.y_mean;
  return prediction;
}

} } // namespace fastpls::native
#endif
