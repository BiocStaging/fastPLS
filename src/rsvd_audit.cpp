// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#include "rsvd_audit.h"

#include <algorithm>

namespace fastpls_svd {
namespace {
thread_local fastpls::core::RSVDAuditSummary summary;
}

void reset_rsvd_audit_summary() {
  summary = fastpls::core::RSVDAuditSummary();
}

fastpls::core::RSVDAuditSummary current_rsvd_audit_summary() {
  return summary;
}

void record_rsvd_audit_case(bool certified, bool deterministic_fallback,
                            int attempts, int effective_oversample,
                            int effective_power, double triplet_residual,
                            double omitted_direction_ratio, bool failure) {
  ++summary.solves;
  if (failure) {
    ++summary.failures;
    return;
  }
  if (certified) ++summary.certified;
  if (deterministic_fallback) ++summary.deterministic_fallbacks;
  summary.max_attempts = std::max(summary.max_attempts, attempts);
  summary.max_effective_oversample = std::max(
    summary.max_effective_oversample, effective_oversample
  );
  summary.max_effective_power_iters = std::max(
    summary.max_effective_power_iters, effective_power
  );
  summary.max_triplet_residual = std::max(
    summary.max_triplet_residual, triplet_residual
  );
  summary.max_omitted_direction_ratio = std::max(
    summary.max_omitted_direction_ratio, omitted_direction_ratio
  );
}

}  // namespace fastpls_svd
