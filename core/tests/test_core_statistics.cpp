#include <fastpls/core/statistics.hpp>

#include <cassert>
#include <cmath>
#include <limits>

int main() {
  const double observed_matrix[] = {1.0, 2.0, 3.0, 2.0, 4.0, 6.0};
  const double predicted_matrix[] = {1.0, 2.5, 2.5, 2.0, 5.0, 5.0};
  const double r2 = fastpls::core::observed_mean_r2(
    fastpls::core::make_const_view(observed_matrix, 3, 2, 3),
    fastpls::core::make_const_view(predicted_matrix, 3, 2, 3)
  );
  assert(std::abs(r2 - 0.75) < 1e-15);

  const double increasing[] = {1.0, 2.0, 3.0, 4.0};
  const double decreasing[] = {4.0, 3.0, 2.0, 1.0};
  auto result = fastpls::core::spearman_correlation(
    increasing, decreasing, 4
  );
  assert(result.status == fastpls::core::CorrelationStatus::success);
  assert(std::abs(result.value + 1.0) < 1e-15);

  const double tied_left[] = {1.0, 1.0, 2.0, 3.0};
  const double tied_right[] = {1.0, 2.0, 2.0, 3.0};
  result = fastpls::core::spearman_correlation(tied_left, tied_right, 4);
  assert(result.status == fastpls::core::CorrelationStatus::success);
  assert(std::abs(result.value - 0.8333333333333334) < 1e-15);

  const double missing[] = {
    1.0, std::numeric_limits<double>::quiet_NaN(), 3.0
  };
  result = fastpls::core::spearman_correlation(missing, increasing, 3);
  assert(result.complete_pairs == 2);

  const double constant[] = {2.0, 2.0, 2.0, 2.0};
  result = fastpls::core::spearman_correlation(constant, increasing, 4);
  assert(result.status == fastpls::core::CorrelationStatus::constant_input);
}
