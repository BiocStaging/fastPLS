#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace {

struct RankEntry {
    double value;
    R_xlen_t index;
};

void sort_entries(std::vector<RankEntry>& entries) {
    Rcpp::checkUserInterrupt();
    std::sort(entries.begin(), entries.end(),
        [](const RankEntry& lhs, const RankEntry& rhs) {
            return lhs.value < rhs.value;
        });
    Rcpp::checkUserInterrupt();
}

} // namespace

// [[Rcpp::export(rng = false)]]
double spearman_correlation_cpp(Rcpp::NumericVector observed,
                                Rcpp::NumericVector predicted) {
    if (observed.size() != predicted.size()) {
        Rcpp::stop("Spearman correlation requires vectors of equal length");
    }
    std::vector<RankEntry> entries;
    entries.reserve(observed.size());
    for (R_xlen_t i = 0; i < observed.size(); ++i) {
        // Match complete.obs: remove NA/NaN pairs, but retain infinities
        // because their ranks are well-defined.
        if (!ISNAN(observed[i]) && !ISNAN(predicted[i])) {
            entries.push_back({observed[i], i});
        }
    }
    if (entries.empty()) Rcpp::stop("no complete element pairs");
    if (entries.size() < 2) return NA_REAL;

    sort_entries(entries);
    if (entries.front().value == entries.back().value) return NA_REAL;
    Rcpp::NumericVector observed_ranks(observed.size(), NA_REAL);
    for (size_t start = 0; start < entries.size();) {
        size_t end = start + 1;
        while (end < entries.size() && entries[end].value == entries[start].value) ++end;
        const double rank = (static_cast<double>(start) + end + 1) / 2;
        for (size_t j = start; j < end; ++j) {
            observed_ranks[entries[j].index] = rank;
        }
        start = end;
    }
    // Reuse the sorting buffer for the other vector.
    for (RankEntry& entry : entries) entry.value = predicted[entry.index];
    sort_entries(entries);
    if (entries.front().value == entries.back().value) return NA_REAL;
    Rcpp::NumericVector predicted_ranks(predicted.size(), NA_REAL);
    for (size_t start = 0; start < entries.size();) {
        size_t end = start + 1;
        while (end < entries.size() && entries[end].value == entries[start].value) ++end;
        const double rank = (static_cast<double>(start) + end + 1) / 2;
        for (size_t j = start; j < end; ++j) {
            predicted_ranks[entries[j].index] = rank;
        }
        start = end;
    }
    // Preserve R's accumulation order and platform-specific arithmetic rather
    // than introducing a second Pearson implementation for million-value data.
    Rcpp::Function correlation = Rcpp::Environment::namespace_env("stats")["cor"];
    return Rcpp::as<double>(correlation(observed_ranks, predicted_ranks,
        Rcpp::Named("method") = "pearson", Rcpp::Named("use") = "complete.obs"));
}
