// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_NATIVE_DIRECTION_HPP
#define FASTPLS_NATIVE_DIRECTION_HPP
#include <fastpls/native/svd_types.hpp>

namespace fastpls { namespace native {

template<typename Scalar>
bool finalize_left(const arma::Mat<Scalar>& small, arma::Mat<Scalar>& U,
                   arma::Col<Scalar>& singular, arma::Mat<Scalar>& V,
                   int eigen_threshold = 4) {
  if (small.n_rows < 1 || small.n_cols < 1) return false;
  if (static_cast<int>(small.n_rows) >= eigen_threshold) {
    arma::Mat<Scalar> gram = small * small.t();
    arma::Col<Scalar> values;
    arma::Mat<Scalar> vectors;
    if (arma::eig_sym(values, vectors, gram) && values.n_elem > 0) {
      arma::uvec order = arma::sort_index(values, "descend");
      U = vectors.cols(order);
      singular = arma::sqrt(arma::clamp(values(order), Scalar(0),
                                       std::numeric_limits<Scalar>::infinity()));
      V.reset();
      return true;
    }
  }
  arma::svd_econ(U, singular, V, small, "left");
  return U.n_cols > 0;
}

template<typename Scalar>
struct DirectionWorkspace {
  arma::Mat<Scalar> Omega, Y, Z, Q, R, Bsmall, Uhat, Vhat, Qy, Ry, Qz, Rz;
  arma::Col<Scalar> shat;
  int eigen_threshold = 4;

  void fresh_sketch(arma::uword rows, arma::uword columns, unsigned int seed) {
    std::mt19937 generator(seed);
    std::normal_distribution<Scalar> normal(Scalar(0), Scalar(1));
    Omega.set_size(rows, columns);
    for (arma::uword i = 0; i < Omega.n_elem; ++i) Omega[i] = normal(generator);
  }

  void prepare_cpu_refresh(const arma::Mat<Scalar>& S, int block, int oversample,
                           int power, unsigned int seed) {
    const arma::uword width = std::min(std::min(S.n_rows, S.n_cols),
      static_cast<arma::uword>(block + std::max(oversample, 0)));
    fresh_sketch(S.n_cols, width, seed);
    Y = S * Omega;
    for (int iteration = 0; iteration < power; ++iteration) {
      arma::qr_econ(Qy, Ry, Y);
      Z = S.t() * Qy;
      arma::qr_econ(Qz, Rz, Z);
      Y = S * Qz;
    }
  }

  bool refresh(const arma::Mat<Scalar>& S, int block, int oversample,
               int power, unsigned int seed, arma::Mat<Scalar>& directions) {
    if (S.n_rows < 1 || S.n_cols < 1 || block < 1) return false;
    prepare_cpu_refresh(S, block, oversample, power, seed);
    arma::qr_econ(Q, R, Y);
    if (Q.n_cols < 1) return false;
    Bsmall = Q.t() * S;
    if (!finalize_left(Bsmall, Uhat, shat, Vhat, eigen_threshold) || Uhat.n_cols < 1) {
      return false;
    }
    directions = Q * Uhat;
    if (directions.n_cols > static_cast<arma::uword>(block)) {
      directions = directions.cols(0, static_cast<arma::uword>(block - 1));
    }
    return directions.n_cols > 0;
  }

  bool refresh_from_right_gram(const arma::Mat<Scalar>& S, const arma::Mat<Scalar>& gram,
                               int block, int oversample, int power,
                               unsigned int seed, arma::Mat<Scalar>& directions) {
    if (S.n_rows < 1 || S.n_cols < 1 || block < 1 ||
        gram.n_rows != S.n_cols || gram.n_cols != S.n_cols) return false;
    const arma::uword width = std::min(std::min(S.n_rows, S.n_cols),
      static_cast<arma::uword>(block + std::max(oversample, 0)));
    fresh_sketch(S.n_cols, width, seed);
    Z = Omega;
    for (int iteration = 0; iteration < power; ++iteration) {
      Y = gram * Z;
      arma::qr_econ(Qz, Rz, Y);
      Z = Qz;
    }
    if (power == 0) {
      arma::qr_econ(Qz, Rz, Z);
      Z = Qz;
    }
    Y = S * Z;
    if (!arma::qr_econ(Q, R, Y) || Q.n_cols < 1) return false;
    Bsmall = Q.t() * S;
    if (!finalize_left(Bsmall, Uhat, shat, Vhat, eigen_threshold) || Uhat.n_cols < 1) {
      return false;
    }
    directions = Q * Uhat;
    if (directions.n_cols > static_cast<arma::uword>(block)) {
      directions = directions.cols(0, static_cast<arma::uword>(block - 1));
    }
    return directions.n_cols > 0;
  }
};

} } // namespace fastpls::native
#endif
