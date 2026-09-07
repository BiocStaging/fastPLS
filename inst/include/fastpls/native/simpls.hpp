// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_NATIVE_SIMPLS_HPP
#define FASTPLS_NATIVE_SIMPLS_HPP
#include <fastpls/native/direction.hpp>
#include <chrono>
#include <vector>

namespace fastpls { namespace native {

struct SimplsOptions {
  int scaling = 1;
  bool fitted = false;
  bool store_coefficients = false;
  bool store_scores = false;
  bool phase_timing = false;
  bool center_scores = false;
  bool reorthogonalize = false;
  bool cache_deflation = true;
  bool cache_crossprod = true;
  bool incremental_coefficients = true;
  bool randomized_directions = true;
  int crossprod_min_components = 20;
  int crossprod_max_predictors = 512;
  int crossprod_min_ratio = 8;
  int maximum_block = 64;
  RsvdControls svd;
};

struct SimplsTiming {
  double preprocess = 0, response_crosscov = 0, crossprod_cache = 0, right_gram = 0;
  double estimator = 0, direction = 0, component_update = 0;
  double coefficients = 0, fitted = 0, total = 0;
};

template<typename Scalar>
struct SimplsModel {
  arma::Mat<Scalar> R, Q, scores, x_mean, x_scale, y_mean;
  arma::Cube<Scalar> coefficients, fitted;
  arma::Col<Scalar> r2;
  arma::ivec components;
  int completed_components = 0;
  SimplsTiming timing;
};

inline int candidate_block_size(int remaining, int p, int q,
                                bool classification = false, int n = 0, int maximum = 8) {
  const double work = static_cast<double>(std::max(n, 0)) *
    static_cast<double>(std::max(p, 0)) * static_cast<double>(std::max(q, 0));
  if (q <= 1 || remaining < 4 || work < 5.0e8) return 1;
  if (classification && q <= 2048) {
    return std::max(1, std::min({maximum, remaining, p, q}));
  }
  // Sequential deflation changes the operator after every regression
  // component. Reusing a response-wide block can therefore change the model
  // materially when q is very large; keep fresh rank-one directions there.
  return 1;
}

template<typename Scalar>
bool is_one_hot_response(const arma::Mat<Scalar>& Y) {
  if (Y.n_rows == 0 || Y.n_cols <= 1) return false;
  constexpr Scalar tolerance = 1e-12;
  for (arma::uword row = 0; row < Y.n_rows; ++row) {
    int active = 0;
    for (arma::uword col = 0; col < Y.n_cols; ++col) {
      const Scalar value = Y(row, col);
      if (std::abs(value - 1.0) <= tolerance) {
        ++active;
      } else if (std::abs(value) > tolerance) {
        return false;
      }
    }
    if (active != 1) return false;
  }
  return true;
}

template<typename Scalar>
bool is_centered_one_hot_response(const arma::Mat<Scalar>& Y) {
  if (Y.n_rows == 0 || Y.n_cols <= 1) return false;
  constexpr Scalar tolerance = 1e-10;
  arma::Row<Scalar> baseline = Y.row(0);
  const arma::uword first_class = baseline.index_max();
  baseline(first_class) -= 1.0;
  if (std::abs(arma::accu(baseline) + 1.0) > tolerance) return false;
  for (arma::uword row = 0; row < Y.n_rows; ++row) {
    int active = 0;
    for (arma::uword col = 0; col < Y.n_cols; ++col) {
      const Scalar difference = Y(row, col) - baseline(col);
      if (std::abs(difference - 1.0) <= tolerance) {
        ++active;
      } else if (std::abs(difference) > tolerance) {
        return false;
      }
    }
    if (active != 1) return false;
  }
  return true;
}

template<typename Scalar>
bool extract_one_hot_labels(const arma::Mat<Scalar>& Y, arma::uvec& labels) {
  if (Y.n_rows == 0 || Y.n_cols <= 1) return false;
  constexpr Scalar tolerance = 1e-12;
  labels.set_size(Y.n_rows);
  for (arma::uword row = 0; row < Y.n_rows; ++row) {
    arma::uword active_col = 0;
    int active = 0;
    for (arma::uword col = 0; col < Y.n_cols; ++col) {
      const Scalar value = Y(row, col);
      if (std::abs(value - 1.0) <= tolerance) {
        active_col = col;
        ++active;
      } else if (std::abs(value) > tolerance) {
        labels.reset();
        return false;
      }
    }
    if (active != 1) {
      labels.reset();
      return false;
    }
    labels(row) = active_col;
  }
  return true;
}

template<typename Scalar>
arma::Mat<Scalar> dummy_crossprod(
  const arma::Mat<Scalar>& X,
  const arma::uvec& labels,
  const arma::Row<Scalar>& center_offset
) {
  arma::Mat<Scalar> out(X.n_cols, center_offset.n_elem, arma::fill::none);
  struct LabelRun {
    arma::uword begin;
    arma::uword length;
    arma::uword label;
  };
  std::vector<LabelRun> runs;
  if (labels.n_elem > 0) {
    arma::uword begin = 0;
    for (arma::uword row = 1; row <= labels.n_elem; ++row) {
      if (row == labels.n_elem || labels(row) != labels(begin)) {
        runs.push_back({begin, row - begin, labels(begin)});
        begin = row;
      }
    }
  }

  // Grouped classification data can form X'Y with one BLAS reduction per
  // contiguous label run. Shuffled labels retain the cache-friendly fallback.
  if (!runs.empty() && runs.size() <=
      std::max<std::size_t>(1024, 4 * center_offset.n_elem)) {
    out.zeros();
    arma::Col<Scalar> ones;
    std::vector<unsigned char> initialized(center_offset.n_elem, 0);
    const char transpose = 'T';
    const arma::blas_int columns = static_cast<arma::blas_int>(X.n_cols);
    const arma::blas_int leading = static_cast<arma::blas_int>(X.n_rows);
    const arma::blas_int increment = 1;
    const Scalar alpha = 1.0;
    for (const LabelRun& run : runs) {
      if (run.label >= center_offset.n_elem) {
        throw std::invalid_argument("dummy-response label is out of range");
      }
      if (ones.n_elem < run.length) {
        ones.ones(run.length);
      }
      const arma::blas_int rows = static_cast<arma::blas_int>(run.length);
      const Scalar beta = initialized[run.label] ? 1.0 : 0.0;
      arma::blas::gemv<Scalar>(
        &transpose, &rows, &columns, &alpha,
        X.memptr() + run.begin, &leading, ones.memptr(), &increment,
        &beta, out.colptr(run.label), &increment
      );
      initialized[run.label] = 1;
    }
    const arma::Col<Scalar> totals = arma::sum(out, 1);
    out += totals * center_offset;
    return out;
  }

  std::vector<Scalar> class_sums(center_offset.n_elem);
  for (arma::uword col = 0; col < X.n_cols; ++col) {
    std::fill(class_sums.begin(), class_sums.end(), 0.0);
    const Scalar* x_col = X.colptr(col);
    Scalar total = 0.0;
    for (arma::uword row = 0; row < X.n_rows; ++row) {
      const Scalar value = x_col[row];
      total += value;
      class_sums[labels(row)] += value;
    }
    for (arma::uword response = 0; response < center_offset.n_elem; ++response) {
      out(col, response) = class_sums[response] + total * center_offset(response);
    }
  }
  return out;
}

template<typename Scalar>
arma::Col<Scalar> dummy_vector_crossprod(
  const arma::Col<Scalar>& x,
  const arma::uvec& labels,
  const arma::Row<Scalar>& center_offset
) {
  arma::Col<Scalar> out(center_offset.n_elem, arma::fill::zeros);
  Scalar total = 0.0;
  for (arma::uword row = 0; row < x.n_elem; ++row) {
    const Scalar value = x(row);
    total += value;
    out(labels(row)) += value;
  }
  out += center_offset.t() * total;
  return out;
}

template<typename Scalar>
arma::Mat<Scalar> symmetric_crossprod(const arma::Mat<Scalar>& X) {
  const arma::blas_int n = static_cast<arma::blas_int>(X.n_rows);
  const arma::blas_int p = static_cast<arma::blas_int>(X.n_cols);
  arma::Mat<Scalar> out(X.n_cols, X.n_cols, arma::fill::zeros);
  const char uplo = 'U';
  const char trans = 'T';
  const Scalar alpha = 1.0;
  const Scalar beta = 0.0;
  arma::blas::syrk<Scalar>(
    &uplo, &trans, &p, &n, &alpha, X.memptr(), &n, &beta, out.memptr(), &p
  );
  for (arma::uword col = 0; col < X.n_cols; ++col) {
    for (arma::uword row = col + 1; row < X.n_cols; ++row) {
      out(row, col) = out(col, row);
    }
  }
  return out;
}

template<typename Scalar>
arma::Mat<Scalar> column_standard_deviation(const arma::Mat<Scalar>& x) {
  int nrow = x.n_rows, ncol = x.n_cols;
  arma::Mat<Scalar> out(1,ncol);
  
  for (int j = 0; j < ncol; j++) {
    Scalar mean = 0;
    Scalar M2 = 0;
    int n=0;
    Scalar delta, xx;
    for (int i = 0; i < nrow; i++) {
      n = i+1;
      xx = x(i,j);
      delta = xx - mean;
      mean += delta/n;
      M2 = M2 + delta*(xx-mean);
    }
    out(0,j) = sqrt(M2/(n-1));
  }
  return out;
}

template<typename Scalar>
Scalar fitted_r2(const arma::Mat<Scalar>& yData,const arma::Mat<Scalar>& yPred){

  Scalar TSS=0,PRESS=0;
  for(unsigned int i=0;i<yData.n_cols;i++){
    Scalar my=arma::mean(yData.col(i));
    for(unsigned int j=0;j<yData.n_rows;j++){
      Scalar b1=yPred(j,i);
      Scalar c1=yData(j,i);
      Scalar d1=c1-my;
      Scalar arg_TR=(c1-b1);
      PRESS+=arg_TR*arg_TR;
      TSS+=d1*d1;  
    }
  }
  
  Scalar R2Y=1-PRESS/TSS;
  return R2Y;
}

template<typename Scalar>
Scalar fitted_r2_labels(const arma::uvec& labels,
                        const arma::Mat<Scalar>& class_proportions,
                        const arma::Mat<Scalar>& fitted_centered) {
  if (labels.n_elem != fitted_centered.n_rows ||
      class_proportions.n_rows != 1 ||
      class_proportions.n_cols != fitted_centered.n_cols) {
    throw std::invalid_argument("label-aware fitted R2 dimensions are inconsistent");
  }
  const Scalar tss = static_cast<Scalar>(labels.n_elem) *
    (Scalar(1) - arma::dot(class_proportions.row(0), class_proportions.row(0)));
  if (!std::isfinite(tss) || tss <= Scalar(0)) {
    return std::numeric_limits<Scalar>::quiet_NaN();
  }
  Scalar response_fit_cross = Scalar(0);
  for (arma::uword row = 0; row < labels.n_elem; ++row) {
    response_fit_cross += fitted_centered(row, labels(row)) -
      arma::dot(fitted_centered.row(row), class_proportions.row(0));
  }
  const Scalar press = tss + arma::accu(arma::square(fitted_centered)) -
    Scalar(2) * response_fit_cross;
  return Scalar(1) - press / tss;
}

template<typename Scalar, typename DirectionSolver>
SimplsModel<Scalar> fit_simpls_with_solver(
    const arma::Mat<Scalar>& Xinput, const arma::Mat<Scalar>& Yinput,
    arma::ivec ncomp, const SimplsOptions& options,
    DirectionSolver&& direction_solver, arma::Mat<Scalar>* owned_X = nullptr,
    const arma::uvec* compact_labels = nullptr, int compact_classes = 0) {
  using namespace arma;
  const bool label_response = compact_labels != nullptr;
  if (Xinput.n_rows < 2 || Xinput.n_cols < 1 ||
      (!label_response && (Yinput.n_cols < 1 || Xinput.n_rows != Yinput.n_rows)) ||
      (label_response && (compact_classes < 2 ||
                          compact_labels->n_elem != Xinput.n_rows))) {
    throw std::invalid_argument("SIMPLS requires matching, nonempty training matrices");
  }
  if (owned_X && (owned_X->n_rows != Xinput.n_rows ||
                  owned_X->n_cols != Xinput.n_cols)) {
    throw std::invalid_argument("SIMPLS owned predictor workspace has invalid dimensions");
  }
  const int scaling = options.scaling;
  const bool fit = options.fitted;
  const unsigned int seed = options.svd.seed;
  using BenchClock = std::chrono::steady_clock;
  const bool benchmark_phase_timing =
    options.phase_timing;
  const auto function_started = benchmark_phase_timing ?
    BenchClock::now() : BenchClock::time_point();
  double estimator_sec = 0.0;
  double direction_sec = 0.0;
  double component_update_sec = 0.0;
  double coefficient_sec = 0.0;
  double fitted_sec = 0.0;
  const int n = Xinput.n_rows;
  const int p = Xinput.n_cols;
  const int m = label_response ? compact_classes : Yinput.n_cols;
  const bool classification_response =
    label_response || is_one_hot_response(Yinput) || is_centered_one_hot_response(Yinput);

  if (ncomp.n_elem < 1) {
    throw std::invalid_argument("ncomp must contain at least one value");
  }
  for (arma::uword i = 0; i < ncomp.n_elem; ++i) {
    if (ncomp(i) < 1) {
      ncomp(i) = 1;
    }
  }

  const int max_ncomp = max(ncomp);
  const int length_ncomp = ncomp.n_elem;

  const int center_t = options.center_scores;
  const int reorth_v = options.reorthogonalize;
  const int defl_cache = options.cache_deflation;
  const int fast_optimized = options.cache_crossprod;
  const int incremental_coefficients = options.incremental_coefficients;
  const int fast_crossprod_min_ncomp = options.crossprod_min_components;
  const int fast_crossprod_max_p = options.crossprod_max_predictors;
  const int fast_crossprod_min_n_to_p_ratio = options.crossprod_min_ratio;
  const bool randomized_solver = options.randomized_directions;
  const bool use_batched_candidate_geometry =
    randomized_solver &&
    candidate_block_size(
      max_ncomp, p, m, classification_response, n, options.maximum_block
    ) > 1;
  const bool return_ttrain = options.store_scores;
  const bool use_crossprod_cache =
    (fast_optimized == 1) &&
    !use_batched_candidate_geometry &&
    (center_t == 0) &&
    (max_ncomp >= fast_crossprod_min_ncomp) &&
    (p <= n) &&
    (n >= p * fast_crossprod_min_n_to_p_ratio) &&
    (p <= fast_crossprod_max_p);

  arma::Mat<Scalar> Xwork;
  const arma::Mat<Scalar>* Xptr = &Xinput;
  arma::Mat<Scalar> mX(1, p, fill::zeros);
  arma::Mat<Scalar>& scaled_X = owned_X == nullptr ? Xwork : *owned_X;
  if (scaling < 3) {
    if (owned_X == nullptr) scaled_X = Xinput;
    mX = mean(scaled_X, 0);
    scaled_X.each_row() -= mX;
    Xptr = &scaled_X;
  }

  arma::Mat<Scalar> vX(1, p, fill::ones);
  if (scaling == 2) {
    vX = column_standard_deviation(scaled_X);
    scaled_X.each_row() /= vX;
  }
  const arma::Mat<Scalar>& Xtrain = *Xptr;

  arma::uvec one_hot_labels;
  const bool one_hot_response = label_response ||
    extract_one_hot_labels(Yinput, one_hot_labels);
  arma::Mat<Scalar> mY;
  if (label_response) {
    one_hot_labels = *compact_labels;
    arma::Row<Scalar> counts(m, arma::fill::zeros);
    for (arma::uword row = 0; row < one_hot_labels.n_elem; ++row) {
      if (one_hot_labels(row) >= static_cast<arma::uword>(m)) {
        throw std::invalid_argument("SIMPLS compact class label is out of range");
      }
      counts(one_hot_labels(row)) += Scalar(1);
    }
    if (arma::any(counts <= Scalar(0))) {
      throw std::invalid_argument("SIMPLS compact labels contain an empty class");
    }
    mY = counts / static_cast<Scalar>(Xinput.n_rows);
  } else {
    mY = mean(Yinput, 0);
  }
  arma::Mat<Scalar> Ywork;
  const arma::Mat<Scalar>* Yptr = &Yinput;
  arma::Mat<Scalar> S;
  if (one_hot_response) {
    S = dummy_crossprod<Scalar>(Xtrain, one_hot_labels, -mY);
    if (fit && !label_response) {
      Ywork = Yinput;
      Ywork.each_row() -= mY;
      Yptr = &Ywork;
    }
  } else {
    S = Xtrain.t() * Yinput;
    const Scalar y_scale = std::max(Scalar(1), arma::abs(Yinput).max());
    const bool response_precentered = arma::abs(mY).max() <= 1e-12 * y_scale;
    if (!response_precentered) {
      S -= arma::sum(Xtrain, 0).t() * mY;
    } else {
      mY.zeros();
    }
    if (fit) {
      Ywork = Yinput;
      Ywork.each_row() -= mY;
      Yptr = &Ywork;
    }
  }
  const auto response_crosscov_done = benchmark_phase_timing ?
    BenchClock::now() : BenchClock::time_point();
  const arma::Mat<Scalar>& Ytrain = *Yptr;
  arma::Mat<Scalar> XtX_cache;
  arma::Mat<Scalar> Sxy_cache;

  arma::Mat<Scalar> RR(p, max_ncomp, fill::zeros);
  arma::Mat<Scalar> QQ(m, max_ncomp, fill::zeros);
  arma::Mat<Scalar> VV(p, max_ncomp, fill::zeros);
  const bool store_B = options.store_coefficients;
  arma::Cube<Scalar> B;
  if (store_B) {
    B.zeros(p, m, length_ncomp);
  }

  arma::Cube<Scalar> Yfit;
  arma::Col<Scalar> R2Y(length_ncomp, fill::zeros);
  arma::Mat<Scalar> Yfit_cur;
  if (fit) {
    Yfit.set_size(n, m, length_ncomp);
    Yfit_cur.zeros(n, m);
  }

  arma::Mat<Scalar> Bcur;
  if (store_B) {
    Bcur.zeros(p, m);
  }
  int i_out = 0;

  // Large dummy-response rSVD fits evaluate one candidate block at a time;
  // every accepted component still uses the sequential SIMPLS updates.
  arma::Mat<Scalar> TT;
  if (return_ttrain) {
    TT.zeros(n, max_ncomp);
  }
  if (use_crossprod_cache) {
    XtX_cache = symmetric_crossprod(Xtrain);
  }
  if (use_crossprod_cache || use_batched_candidate_geometry) {
    Sxy_cache = S;
  }
  const auto crossprod_cache_done = benchmark_phase_timing ?
    BenchClock::now() : BenchClock::time_point();
  const bool use_right_gram_refresh =
    randomized_solver && m <= p && m <= 512;
  arma::Mat<Scalar> right_gram;
  if (use_right_gram_refresh) {
    right_gram = S.t() * S;
  }
  const auto right_gram_done = benchmark_phase_timing ?
    BenchClock::now() : BenchClock::time_point();
  const auto estimator_started = benchmark_phase_timing ?
    BenchClock::now() : BenchClock::time_point();
  auto append_component = [&] (
    arma::Col<Scalar> rr,
    const int a_idx,
    const arma::Col<Scalar>* candidate_scores,
    const arma::Col<Scalar>* candidate_loadings
  ) -> bool {
    const auto component_started = benchmark_phase_timing ?
      BenchClock::now() : BenchClock::time_point();
    arma::Col<Scalar> pp;
    arma::Col<Scalar> qq;
    arma::Col<Scalar> tt;
    if (use_crossprod_cache) {
      pp = XtX_cache * rr;
      const Scalar tnorm_sq = arma::dot(rr, pp);
      if (!std::isfinite(tnorm_sq) || tnorm_sq <= 0.0) {
        return false;
      }
      const Scalar tnorm = std::sqrt(tnorm_sq);
      rr /= tnorm;
      pp /= tnorm;
      qq = Sxy_cache.t() * rr;
      if (fit || return_ttrain) {
        tt = Xtrain * rr;
      }
    } else {
      const bool has_candidate_geometry =
        candidate_scores != nullptr && candidate_loadings != nullptr;
      tt = has_candidate_geometry ? *candidate_scores : Xtrain * rr;
      if (center_t == 1) {
        tt -= arma::mean(tt);
      }
      bool score_reorthogonalized = false;
      if (reorth_v == 1 && a_idx > 0 &&
          TT.n_rows == static_cast<arma::uword>(n)) {
        auto Tprev = TT.cols(0, a_idx - 1);
        auto Rprev = RR.cols(0, a_idx - 1);
        for (int pass = 0; pass < 2; ++pass) {
          arma::Col<Scalar> projection = Tprev.t() * tt;
          tt -= Tprev * projection;
          rr -= Rprev * projection;
        }
        score_reorthogonalized = true;
      }
      const Scalar tnorm = arma::norm(tt, 2);
      if (!std::isfinite(tnorm) || tnorm <= 0.0) {
        return false;
      }
      tt /= tnorm;
      rr /= tnorm;
      if (has_candidate_geometry) {
        if (score_reorthogonalized) {
          pp = Xtrain.t() * tt;
        } else {
          pp = *candidate_loadings / tnorm;
        }
      } else {
        pp = Xtrain.t() * tt;
      }
      if (use_batched_candidate_geometry) {
        qq = Sxy_cache.t() * rr;
      } else if (one_hot_response) {
        qq = dummy_vector_crossprod<Scalar>(
          tt, one_hot_labels, -mY
        );
      } else {
        qq = Yinput.t() * tt - mY.t() * arma::accu(tt);
      }
    }
    arma::Col<Scalar> vv = pp;
    if (a_idx > 0) {
      auto Vprev = VV.cols(0, a_idx - 1);
      vv -= Vprev * (Vprev.t() * pp);
      if (reorth_v == 1) {
        vv -= Vprev * (Vprev.t() * vv);
      }
    }
    const Scalar vnorm = arma::norm(vv, 2);
    if (!std::isfinite(vnorm) || vnorm <= 0.0) {
      return false;
    }
    vv /= vnorm;

    arma::Row<Scalar> vS;
    if (defl_cache == 1 || use_right_gram_refresh) {
      vS = vv.t() * S;
      S -= vv * vS;
    } else {
      S -= vv * (vv.t() * S);
    }
    if (use_right_gram_refresh) {
      right_gram -= vS.t() * vS;
      right_gram = 0.5 * (right_gram + right_gram.t());
    }

    RR.col(a_idx) = rr;
    QQ.col(a_idx) = qq;
    VV.col(a_idx) = vv;
    if (return_ttrain && tt.n_elem == static_cast<arma::uword>(n)) {
      TT.col(a_idx) = tt;
    }
    if (benchmark_phase_timing) {
      const double elapsed = std::chrono::duration<double>(
        BenchClock::now() - component_started
      ).count();
      estimator_sec += elapsed;
      component_update_sec += elapsed;
    }
    if (store_B && incremental_coefficients == 1) {
      const auto coefficient_started = benchmark_phase_timing ?
        BenchClock::now() : BenchClock::time_point();
      Bcur += rr * qq.t();
      if (benchmark_phase_timing) {
        coefficient_sec += std::chrono::duration<double>(
          BenchClock::now() - coefficient_started
        ).count();
      }
    }
    if (fit) {
      const auto fitted_started = benchmark_phase_timing ?
        BenchClock::now() : BenchClock::time_point();
      Yfit_cur += tt * qq.t();
      if (benchmark_phase_timing) {
        fitted_sec += std::chrono::duration<double>(
          BenchClock::now() - fitted_started
        ).count();
      }
    }

    while (i_out < length_ncomp && a_idx == (ncomp(i_out) - 1)) {
      if (store_B) {
        const auto coefficient_started = benchmark_phase_timing ?
          BenchClock::now() : BenchClock::time_point();
        B.slice(i_out) = incremental_coefficients == 1 ?
          Bcur :
          RR.cols(0, a_idx) * QQ.cols(0, a_idx).t();
        if (benchmark_phase_timing) {
          coefficient_sec += std::chrono::duration<double>(
            BenchClock::now() - coefficient_started
          ).count();
        }
      }
      if (fit) {
        const auto fitted_started = benchmark_phase_timing ?
          BenchClock::now() : BenchClock::time_point();
        arma::Mat<Scalar> fitted_centered;
        if (reorth_v == 1) {
          fitted_centered =
            Xtrain * RR.cols(0, a_idx) * QQ.cols(0, a_idx).t();
        } else {
          fitted_centered = Yfit_cur;
        }
        R2Y(i_out) = label_response ?
          fitted_r2_labels(one_hot_labels, mY, fitted_centered) :
          fitted_r2(Ytrain, fitted_centered);
        arma::Mat<Scalar> yf = std::move(fitted_centered);
        yf.each_row() += mY;
        Yfit.slice(i_out) = yf;
        if (benchmark_phase_timing) {
          fitted_sec += std::chrono::duration<double>(
            BenchClock::now() - fitted_started
          ).count();
        }
      }
      ++i_out;
    }
    return true;
  };

  int a = 0;
  while (a < max_ncomp) {
    const auto direction_started = benchmark_phase_timing ?
      BenchClock::now() : BenchClock::time_point();
    const int k_block = randomized_solver ?
      candidate_block_size(
        max_ncomp - a, p, m, classification_response, n, options.maximum_block
      ) : 1;
    arma::Mat<Scalar> Ublock;
    if (!direction_solver(S, right_gram, use_right_gram_refresh, k_block,
                          static_cast<unsigned int>(seed + a), Ublock)) break;
    if (benchmark_phase_timing) {
      const double elapsed = std::chrono::duration<double>(
        BenchClock::now() - direction_started
      ).count();
      estimator_sec += elapsed;
      direction_sec += elapsed;
    }
    if (Ublock.n_cols < 1) {
      break;
    }
    const int use_cols = std::min<int>(Ublock.n_cols, k_block);
    arma::Mat<Scalar> candidate_scores;
    arma::Mat<Scalar> candidate_loadings;
    if (use_batched_candidate_geometry) {
      const arma::Mat<Scalar> directions = Ublock.cols(
        0, static_cast<arma::uword>(use_cols - 1)
      );
      candidate_scores = Xtrain * directions;
      candidate_loadings = Xtrain.t() * candidate_scores;
    }
    bool stop_now = false;
    for (int j = 0; j < use_cols && a < max_ncomp; ++j, ++a) {
      arma::Col<Scalar> candidate_score;
      arma::Col<Scalar> candidate_loading;
      const arma::Col<Scalar>* candidate_score_ptr = nullptr;
      const arma::Col<Scalar>* candidate_loading_ptr = nullptr;
      if (use_batched_candidate_geometry) {
        candidate_score = candidate_scores.col(static_cast<arma::uword>(j));
        candidate_loading = candidate_loadings.col(static_cast<arma::uword>(j));
        candidate_score_ptr = &candidate_score;
        candidate_loading_ptr = &candidate_loading;
      }
      if (!append_component(
        Ublock.col(j), a, candidate_score_ptr, candidate_loading_ptr
      )) {
        stop_now = true;
        break;
      }
    }
    if (stop_now) break;
  }

  SimplsModel<Scalar> result;
  result.R = std::move(RR);
  result.Q = std::move(QQ);
  result.scores = std::move(TT);
  result.x_mean = std::move(mX);
  result.x_scale = std::move(vX);
  result.y_mean = std::move(mY);
  result.coefficients = std::move(B);
  result.fitted = std::move(Yfit);
  result.r2 = std::move(R2Y);
  result.components = std::move(ncomp);
  result.completed_components = a;
  if (benchmark_phase_timing) {
    auto seconds = [](auto duration) { return std::chrono::duration<double>(duration).count(); };
    result.timing.preprocess = seconds(estimator_started - function_started);
    result.timing.response_crosscov = seconds(response_crosscov_done - function_started);
    result.timing.crossprod_cache = seconds(crossprod_cache_done - response_crosscov_done);
    result.timing.right_gram = seconds(right_gram_done - crossprod_cache_done);
    result.timing.estimator = estimator_sec;
    result.timing.direction = direction_sec;
    result.timing.component_update = component_update_sec;
    result.timing.coefficients = coefficient_sec;
    result.timing.fitted = fitted_sec;
    result.timing.total = seconds(BenchClock::now() - function_started);
  }
  return result;
}

template<typename Scalar>
SimplsModel<Scalar> fit_simpls(const arma::Mat<Scalar>& X, const arma::Mat<Scalar>& Y,
                              arma::ivec components, const SimplsOptions& options = {}) {
  DirectionWorkspace<Scalar> workspace;
  auto solve = [&](const arma::Mat<Scalar>& S, const arma::Mat<Scalar>& gram,
                   bool use_gram, int width, unsigned int seed, arma::Mat<Scalar>& U) {
    return use_gram ? workspace.refresh_from_right_gram(S, gram, width,
      options.svd.oversample, options.svd.power, seed, U) :
      workspace.refresh(S, width, options.svd.oversample, options.svd.power, seed, U);
  };
  return fit_simpls_with_solver(X, Y, std::move(components), options, solve);
}

template<typename Scalar>
arma::Mat<Scalar> predict_simpls(const SimplsModel<Scalar>& model,
                                arma::Mat<Scalar> X, int components) {
  if (components < 1 || components > model.completed_components ||
      X.n_cols != model.R.n_rows || model.R.n_cols < arma::uword(components) ||
      model.Q.n_cols < arma::uword(components) || model.Q.n_rows == 0 ||
      model.x_mean.n_rows != 1 || model.x_mean.n_cols != X.n_cols ||
      model.x_scale.n_rows != 1 || model.x_scale.n_cols != X.n_cols ||
      model.y_mean.n_rows != 1 || model.y_mean.n_cols != model.Q.n_rows) {
    throw std::invalid_argument("predict_simpls: invalid dimensions or component count");
  }
  X.each_row() -= model.x_mean;
  X.each_row() /= model.x_scale;
  arma::Mat<Scalar> scores = X * model.R.cols(0, components - 1);
  arma::Mat<Scalar> result = scores * model.Q.cols(0, components - 1).t();
  result.each_row() += model.y_mean;
  return result;
}

} } // namespace fastpls::native
#endif
