// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#include <fastpls/core.hpp>

#include <cmath>

int main() {
  fastpls::core::Matrix<float> data(3, 2);
  data(0, 0) = 1.0f;
  data(1, 0) = 0.0f;
  data(2, 0) = -1.0f;
  data(0, 1) = 0.0f;
  data(1, 1) = 1.0f;
  data(2, 1) = 0.0f;
  auto kernel = fastpls::core::kernel_matrix_reference(
    data.view(), data.view(), fastpls::core::KernelType::radial_basis,
    0.5f, 1, 0.0f
  );
  const auto centered = fastpls::core::center_kernel_train(kernel.view());
  if (centered.column_means.size() != 3 ||
      !std::isfinite(centered.grand_mean)) {
    return 1;
  }
  fastpls::core::KernelPlsControls controls;
  controls.kernel = fastpls::core::KernelType::radial_basis;
  controls.gamma = 0.5;
  controls.simpls.components = 2;
  fastpls::core::KernelPlsModel<float> model;
  model.kernel = controls.kernel;
  if (model.kernel != fastpls::core::KernelType::radial_basis) return 1;
  fastpls::core::OplsControls opls_controls;
  opls_controls.orthogonal_components = 1;
  opls_controls.simpls.components = 2;
  fastpls::core::OplsModel<double> opls_model;
  if (opls_model.filter.completed_components != 0) return 1;
  return 0;
}
