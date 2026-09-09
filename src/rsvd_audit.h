// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_RSVD_AUDIT_H
#define FASTPLS_RSVD_AUDIT_H

#include <fastpls/core/diagnostics.hpp>

namespace fastpls_svd {

void reset_rsvd_audit_summary();
fastpls::core::RSVDAuditSummary current_rsvd_audit_summary();
void record_rsvd_audit_case(bool certified, bool deterministic_fallback,
                            int attempts, int effective_oversample,
                            int effective_power, double triplet_residual,
                            double omitted_direction_ratio,
                            bool failure = false);

}  // namespace fastpls_svd

#endif
