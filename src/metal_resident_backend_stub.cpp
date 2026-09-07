#include "metal_resident_backend.h"

#include <stdexcept>

namespace fastpls_svd {

std::unique_ptr<MetalResidentModel> metal_resident_simpls_create(
    const arma::fmat&,
    const arma::fmat*,
    const Rcpp::IntegerVector*,
    int,
    int,
    int,
    int,
    int,
    unsigned int,
    int) {
  throw std::runtime_error(
      "Resident Metal fitting is unavailable in this build; no CPU fallback is performed");
}

}  // namespace fastpls_svd
