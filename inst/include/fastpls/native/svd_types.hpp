// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_NATIVE_SVD_TYPES_HPP
#define FASTPLS_NATIVE_SVD_TYPES_HPP

#include <fastpls/core/rsvd.hpp>

#include <armadillo>

namespace fastpls {
namespace native {

using RsvdControls = core::RsvdControls;

template<class T>
struct SingularTriplets {
  arma::Mat<T> U;
  arma::Col<T> s;
  arma::Mat<T> Vt;
};

}  // namespace native
}  // namespace fastpls

#endif
