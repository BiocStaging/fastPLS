// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_NATIVE_OPLS_HPP
#define FASTPLS_NATIVE_OPLS_HPP
#include <fastpls/native/simpls.hpp>

namespace fastpls { namespace native {

template<class T>
struct OplsFilter {
  arma::Mat<T> X, W, P;
  arma::Row<T> x_mean, x_scale;
  int completed = 0;
};

template<class T>
bool leading_left_from_smaller_gram(const arma::Mat<T>& S, arma::Col<T>& w) {
  if (S.n_rows < 1 || S.n_cols < 1) return false;
  arma::Col<T> eigenvalues;
  arma::Mat<T> eigenvectors;
  if (S.n_cols <= S.n_rows) {
    if (arma::eig_sym(eigenvalues, eigenvectors, S.t() * S) &&
        eigenvalues.n_elem > 0) {
      const T value = eigenvalues(eigenvalues.n_elem - 1);
      if (std::isfinite(value) && value > T(0)) {
        w = S * eigenvectors.col(eigenvectors.n_cols - 1);
        const T norm = arma::norm(w, 2);
        if (std::isfinite(norm) && norm > T(0)) {
          w /= norm;
          return true;
        }
      }
    }
  } else if (arma::eig_sym(eigenvalues, eigenvectors, S * S.t()) &&
             eigenvalues.n_elem > 0) {
    w = eigenvectors.col(eigenvectors.n_cols - 1);
    const T norm = arma::norm(w, 2);
    if (std::isfinite(norm) && norm > T(0)) {
      w /= norm;
      return true;
    }
  }
  arma::Mat<T> U, V;
  arma::Col<T> singular;
  if (!arma::svd_econ(U, singular, V, S, "left") || U.n_cols == 0) {
    return false;
  }
  w = U.col(0);
  return true;
}

template<class T, class DirectionSolver>
OplsFilter<T> fit_opls_filter_with_solver(arma::Mat<T> X, arma::Mat<T> Y,
                                         int north, int scaling,
                                         DirectionSolver&& solve,
                                         const arma::uvec* compact_labels = nullptr,
                                         int compact_classes = 0) {
  const bool label_response = compact_labels != nullptr;
  if ((!label_response && X.n_rows != Y.n_rows) ||
      (label_response && (compact_classes < 2 ||
                          compact_labels->n_elem != X.n_rows))) {
    throw std::invalid_argument("X and Y must have the same number of rows");
  }
  if (north < 0) throw std::invalid_argument("north must be >= 0");
  OplsFilter<T> model;
  model.x_mean.zeros(X.n_cols);
  if (scaling < 3) {
    model.x_mean = arma::mean(X, 0);
    X.each_row() -= model.x_mean;
  }
  model.x_scale.ones(X.n_cols);
  if (scaling == 2) {
    model.x_scale = arma::stddev(X, 0, 0);
    for (auto& value : model.x_scale) {
      if (!std::isfinite(value) || value == T(0)) value = T(1);
    }
    X.each_row() /= model.x_scale;
  }
  arma::uvec one_hot_labels;
  const bool one_hot_response = label_response ||
    extract_one_hot_labels(Y, one_hot_labels);
  arma::Row<T> y_mean;
  if (label_response) {
    one_hot_labels = *compact_labels;
    y_mean.zeros(static_cast<arma::uword>(compact_classes));
    for (arma::uword row = 0; row < one_hot_labels.n_elem; ++row) {
      if (one_hot_labels(row) >= static_cast<arma::uword>(compact_classes)) {
        throw std::invalid_argument("OPLS compact class label is out of range");
      }
      y_mean(one_hot_labels(row)) += T(1);
    }
    if (arma::accu(y_mean <= T(0)) > 0) {
      throw std::invalid_argument("OPLS compact labels contain an empty class");
    }
    y_mean /= static_cast<T>(X.n_rows);
  } else {
    y_mean = arma::mean(Y, 0);
  }
  if (!one_hot_response) Y.each_row() -= y_mean;
  model.W.zeros(X.n_cols, static_cast<arma::uword>(north));
  model.P.zeros(X.n_cols, static_cast<arma::uword>(north));
  for (int component = 0; component < north; ++component) {
    arma::Mat<T> S = one_hot_response ?
      dummy_crossprod<T>(X, one_hot_labels, -y_mean) : X.t() * Y;
    arma::Col<T> w;
    if (!solve(S, component, w)) break;
    const T w_norm = arma::norm(w, 2);
    if (!std::isfinite(w_norm) || w_norm <= T(0)) break;
    w /= w_norm;
    arma::Col<T> t = X * w;
    const T t_ss = arma::dot(t, t);
    if (!std::isfinite(t_ss) || t_ss <= T(0)) break;
    arma::Col<T> p = X.t() * t / t_ss;
    const T ww = arma::dot(w, w);
    arma::Col<T> w_orth = p - w * (arma::dot(w, p) / ww);
    const T wo_norm = arma::norm(w_orth, 2);
    if (!std::isfinite(wo_norm) || wo_norm <= T(0)) break;
    w_orth /= wo_norm;
    arma::Col<T> t_orth = X * w_orth;
    const T to_ss = arma::dot(t_orth, t_orth);
    if (!std::isfinite(to_ss) || to_ss <= T(0)) break;
    arma::Col<T> p_orth = X.t() * t_orth / to_ss;
    X -= t_orth * p_orth.t();
    model.W.col(static_cast<arma::uword>(model.completed)) = w_orth;
    model.P.col(static_cast<arma::uword>(model.completed)) = p_orth;
    ++model.completed;
  }
  if (model.completed == 0) {
    model.W.set_size(X.n_cols, 0);
    model.P.set_size(X.n_cols, 0);
  } else if (model.completed < north) {
    model.W = model.W.cols(0, static_cast<arma::uword>(model.completed - 1));
    model.P = model.P.cols(0, static_cast<arma::uword>(model.completed - 1));
  }
  model.X = std::move(X);
  return model;
}

template<class T>
OplsFilter<T> fit_opls_filter(arma::Mat<T> X, arma::Mat<T> Y, int north,
                             int scaling = 1, RsvdControls controls = {}) {
  auto solve = [&](const arma::Mat<T>& S, int component, arma::Col<T>& w) {
    RsvdControls current = controls;
    current.seed += static_cast<unsigned int>(component);
    auto result = rsvd(S, 1, current);
    if (result.U.n_cols == 0) return false;
    w = result.U.col(0);
    return true;
  };
  return fit_opls_filter_with_solver(std::move(X), std::move(Y), north, scaling, solve);
}

template<class T>
arma::Mat<T> apply_opls_filter(arma::Mat<T> X, const arma::Row<T>& mean,
                              const arma::Row<T>& scale,
                              const arma::Mat<T>& W, const arma::Mat<T>& P) {
  if (X.n_cols != mean.n_cols || X.n_cols != scale.n_cols) {
    throw std::invalid_argument("X columns must match stored OPLS preprocessing");
  }
  X.each_row() -= mean;
  X.each_row() /= scale;
  if (W.n_cols != P.n_cols || W.n_rows != X.n_cols || P.n_rows != X.n_cols) {
    throw std::invalid_argument("Invalid OPLS orthogonal filter dimensions");
  }
  for (arma::uword component = 0; component < W.n_cols; ++component) {
    arma::Col<T> t_orth = X * W.col(component);
    X -= t_orth * P.col(component).t();
  }
  return X;
}

} }
#endif
