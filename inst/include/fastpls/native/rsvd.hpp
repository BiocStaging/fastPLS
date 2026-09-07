// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_NATIVE_RSVD_HPP
#define FASTPLS_NATIVE_RSVD_HPP

#include <armadillo>
#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace fastpls {
namespace native {

struct RsvdControls {
  int oversample = 32;
  int power = 5;
  unsigned int seed = 1;
  bool left_only = false;
};

template<typename T>
struct SingularTriplets {
  arma::Mat<T> U;
  arma::Col<T> s;
  arma::Mat<T> Vt;
};

template<typename T>
SingularTriplets<T> dense_triplets(const arma::Mat<T>& A, int k, bool left_only) {
  SingularTriplets<T> out;
  const arma::uword rank = std::min<arma::uword>(
    std::min(A.n_rows, A.n_cols), static_cast<arma::uword>(std::max(k, 1))
  );
  if (rank == 0) return out;
  arma::Mat<T> U, V;
  arma::Col<T> s;
  if (!arma::svd_econ(U, s, V, A, left_only ? "left" : "both")) {
    throw std::runtime_error("native rSVD: dense decomposition failed");
  }
  out.U = U.cols(0, rank - 1);
  out.s = s.subvec(0, rank - 1);
  if (!left_only) out.Vt = V.cols(0, rank - 1).t();
  return out;
}

template<typename T>
SingularTriplets<T> finalize_sample(const arma::Mat<T>& A,
                                   const arma::Mat<T>& Y,
                                   int k, bool left_only) {
  SingularTriplets<T> out;
  const arma::uword rank = std::min<arma::uword>(
    std::min(A.n_rows, A.n_cols), static_cast<arma::uword>(std::max(k, 1))
  );
  if (rank == 0) return out;
  arma::Mat<T> Q, R;
  if (!arma::qr_econ(Q, R, Y)) {
    throw std::runtime_error("native rSVD: sketch orthogonalization failed");
  }
  arma::Mat<T> B = Q.t() * A;
  if (B.n_rows <= B.n_cols) {
    arma::Col<T> evals;
    arma::Mat<T> eigvec;
    if (arma::eig_sym(evals, eigvec, B * B.t())) {
      arma::uvec order = arma::sort_index(evals, "descend");
      arma::Col<T> evals_desc = evals.elem(order);
      arma::Mat<T> Uhat_desc = eigvec.cols(order);
      const T tol = std::numeric_limits<T>::epsilon() *
        static_cast<T>(std::max(B.n_rows, B.n_cols)) *
        (evals_desc.n_elem > 0 ? std::max(evals_desc(0), T(1)) : T(1));
      arma::uword usable = 0;
      while (usable < evals_desc.n_elem && evals_desc(usable) > tol) ++usable;
      usable = std::min<arma::uword>(usable, rank);
      if (usable > 0) {
        arma::Col<T> s = arma::sqrt(arma::clamp(
          evals_desc.subvec(0, usable - 1), T(0), std::numeric_limits<T>::infinity()
        ));
        arma::Mat<T> Uhat = Uhat_desc.cols(0, usable - 1);
        out.U = Q * Uhat;
        out.s = s;
        if (!left_only) {
          arma::Mat<T> Vt = Uhat.t() * B;
          Vt.each_col() /= s;
          out.Vt = std::move(Vt);
        }
        return out;
      }
    }
  }
  auto reduced = dense_triplets(B, static_cast<int>(rank), left_only);
  out.U = Q * reduced.U;
  out.s = std::move(reduced.s);
  out.Vt = std::move(reduced.Vt);
  return out;
}

template<typename T>
SingularTriplets<T> rsvd(const arma::Mat<T>& A, int k, const RsvdControls& controls) {
  const arma::uword max_rank = std::min(A.n_rows, A.n_cols);
  const arma::uword target = std::min<arma::uword>(
    max_rank, static_cast<arma::uword>(std::max(k, 1))
  );
  const arma::uword width = std::min<arma::uword>(
    max_rank, target + static_cast<arma::uword>(std::max(controls.oversample, 0))
  );
  if (width >= max_rank) {
    return dense_triplets(A, static_cast<int>(target), controls.left_only);
  }
  std::mt19937 rng(controls.seed);
  std::normal_distribution<T> normal(T(0), T(1));
  arma::Mat<T> omega(A.n_cols, width);
  for (arma::uword i = 0; i < omega.n_elem; ++i) omega[i] = normal(rng);
  arma::Mat<T> Y = A * omega;
  for (int iteration = 0; iteration < std::max(controls.power, 0); ++iteration) {
    arma::Mat<T> Qy, Ry;
    if (!arma::qr_econ(Qy, Ry, Y)) {
      throw std::runtime_error("native rSVD: forward orthogonalization failed");
    }
    arma::Mat<T> Z = A.t() * Qy;
    arma::Mat<T> Qz, Rz;
    if (!arma::qr_econ(Qz, Rz, Z)) {
      throw std::runtime_error("native rSVD: reverse orthogonalization failed");
    }
    Y = A * Qz;
  }
  return finalize_sample(A, Y, static_cast<int>(target), controls.left_only);
}

} // namespace native
} // namespace fastpls
#endif
