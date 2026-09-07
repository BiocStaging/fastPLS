#include "svd_iface.h"
#include <fastpls/native/rsvd.hpp>
#include <utility>

namespace fastpls_svd {
namespace {

SVDResult from_native(fastpls::native::SingularTriplets<double>&& result) {
  SVDResult out;
  out.U = std::move(result.U);
  out.s = std::move(result.s);
  out.Vt = std::move(result.Vt);
  return out;
}

} // namespace

SVDResult finalize_rsvd_from_sample(const Mat& A, const Mat& Y, int k, bool left_only) {
  return from_native(fastpls::native::finalize_sample(A, Y, k, left_only));
}

SVDResult truncated_svd_cpu_rsvd(const Mat& A, int k, const SVDOptions& opt) {
  fastpls::native::RsvdControls controls;
  controls.oversample = opt.oversample;
  controls.power = opt.power_iters;
  controls.seed = opt.seed;
  controls.left_only = opt.left_only;
  return from_native(fastpls::native::rsvd(A, k, controls));
}

} // namespace fastpls_svd
