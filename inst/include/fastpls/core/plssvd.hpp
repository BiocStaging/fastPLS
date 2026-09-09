// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_PLSSVD_HPP
#define FASTPLS_CORE_PLSSVD_HPP

#include <fastpls/core/linalg.hpp>
#include <fastpls/core/matrix.hpp>
#include <fastpls/core/rsvd.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace fastpls {
namespace core {

struct PlssvdControls {
  RsvdControls rsvd;
};

template<class T>
struct PlssvdModel {
  Matrix<T> weights;
  Matrix<T> response_loadings;
  Matrix<T> scores;
  std::vector<Matrix<T>> prediction_weights;
  std::vector<T> singular_values;
  std::vector<int> components;
  std::size_t completed_components = 0;
};

template<class T, class Backend>
PlssvdModel<T> fit_plssvd_preprocessed(
    ConstMatrixView<T> predictors, ConstMatrixView<T> crosscov,
    const int* components, std::size_t component_count,
    const PlssvdControls& controls, Backend& backend) {
  if (predictors.empty() || crosscov.empty() || components == nullptr ||
      component_count == 0 || crosscov.rows() != predictors.columns()) {
    throw std::invalid_argument(
      "fastPLS PLS-SVD dimensions or component counts are invalid"
    );
  }
  const std::size_t rank_bound = std::min(
    crosscov.rows(), crosscov.columns()
  );
  std::size_t retained = 0;
  PlssvdModel<T> model;
  model.components.assign(components, components + component_count);
  for (const int count : model.components) {
    if (count < 1 || static_cast<std::size_t>(count) > rank_bound) {
      throw std::invalid_argument(
        "fastPLS PLS-SVD component count exceeds cross-covariance rank"
      );
    }
    retained = std::max(retained, static_cast<std::size_t>(count));
  }

  RsvdControls rsvd = controls.rsvd;
  rsvd.left_only = false;
  auto decomposition = randomized_svd(
    crosscov, static_cast<int>(retained), rsvd, backend
  );
  model.completed_components = std::min({
    decomposition.U.columns(), decomposition.Vt.rows(),
    decomposition.singular_values.size()
  });
  if (model.completed_components < retained) {
    throw std::runtime_error(
      "fastPLS PLS-SVD returned fewer components than requested"
    );
  }

  model.weights.resize(crosscov.rows(), retained);
  model.response_loadings.resize(crosscov.columns(), retained);
  model.singular_values.assign(
    decomposition.singular_values.begin(),
    decomposition.singular_values.begin() + retained
  );
  for (std::size_t component = 0; component < retained; ++component) {
    for (std::size_t predictor = 0;
         predictor < crosscov.rows(); ++predictor) {
      model.weights(predictor, component) =
        decomposition.U(predictor, component);
    }
    for (std::size_t response = 0;
         response < crosscov.columns(); ++response) {
      model.response_loadings(response, component) =
        decomposition.Vt(component, response);
    }
  }
  model.scores.resize(predictors.rows(), retained);
  backend.gemm(
    predictors, model.weights.view(), false, false, model.scores.view()
  );
  Matrix<T> full_gram(retained, retained);
  backend.gemm(
    model.scores.view(), model.scores.view(), true, false,
    full_gram.view()
  );

  model.prediction_weights.reserve(component_count);
  for (const int requested : model.components) {
    const std::size_t count = static_cast<std::size_t>(requested);
    Matrix<T> gram(count, count);
    Matrix<T> diagonal(count, count);
    for (std::size_t column = 0; column < count; ++column) {
      diagonal(column, column) = model.singular_values[column];
      for (std::size_t row = 0; row < count; ++row) {
        gram(row, column) = full_gram(row, column);
      }
    }
    Matrix<T> latent;
    if (!solve_symmetric_system(gram.view(), diagonal.view(), latent)) {
      throw std::runtime_error("fastPLS PLS-SVD latent solve failed");
    }
    Matrix<T> weights(count, crosscov.columns());
    ConstMatrixView<T> loadings(
      model.response_loadings.data(), model.response_loadings.rows(), count,
      model.response_loadings.rows()
    );
    backend.gemm(
      latent.view(), loadings, false, true, weights.view()
    );
    model.prediction_weights.push_back(std::move(weights));
  }
  return model;
}

template<class T, class Backend>
PlssvdModel<T> fit_plssvd_preprocessed(
    MatrixView<T> predictors, MatrixView<T> crosscov,
    const int* components, std::size_t component_count,
    const PlssvdControls& controls, Backend& backend) {
  return fit_plssvd_preprocessed(
    ConstMatrixView<T>(predictors), ConstMatrixView<T>(crosscov), components,
    component_count, controls, backend
  );
}

}  // namespace core
}  // namespace fastpls

#endif
