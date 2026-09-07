#ifndef FASTPLS_METAL_RESIDENT_BACKEND_H
#define FASTPLS_METAL_RESIDENT_BACKEND_H

#include <RcppArmadillo.h>

#include <memory>

namespace fastpls_svd {

// Persistent float32 model used by the Metal PLS path. Training matrices and
// reusable prediction workspaces remain in Metal buffers; selected large
// SIMPLS routes may update small component vectors on the host.
class MetalResidentModel {
 public:
  virtual ~MetalResidentModel() = default;

  virtual arma::fmat export_field(int field) = 0;
  virtual arma::fmat predict(const arma::fmat& x, int prefix,
                             bool use_lda) = 0;
  virtual arma::fcube predict_path(
      const arma::fmat& x, const Rcpp::IntegerVector& prefixes,
      bool use_lda) = 0;
  virtual arma::fmat project(const arma::fmat& x, int prefix) = 0;
  virtual arma::imat classify(const arma::fmat& x, int prefix, int top,
                              bool use_lda) = 0;
  virtual arma::icube classify_path(
      const arma::fmat& x, const Rcpp::IntegerVector& prefixes, int top,
      bool use_lda) = 0;
  virtual void classify_response_path(
      const arma::fmat& x, const Rcpp::IntegerVector& prefixes, int top,
      bool use_lda, arma::icube& labels, arma::fcube& predictions) = 0;
  virtual arma::fmat response_sums(const arma::fmat& x,
                                   const arma::fmat* y,
                                   const Rcpp::IntegerVector* labels,
                                   int prefix) = 0;
  virtual void controls(int& oversample, int& power, int& block) const = 0;
  virtual void compact(bool prepare_lda) = 0;
  virtual int observations() const = 0;
  virtual int predictors() const = 0;
  virtual int responses() const = 0;
  virtual int components() const = 0;
  virtual bool classification() const = 0;
  virtual bool implicit_crosscovariance() const = 0;
  virtual bool predictor_crossprod_cache() const = 0;
  virtual bool host_assisted_components() const = 0;
};

std::unique_ptr<MetalResidentModel> metal_resident_simpls_create(
    const arma::fmat& x,
    const arma::fmat* y,
    const Rcpp::IntegerVector* labels,
    int classes,
    int components,
    int scaling,
    int oversample,
    int power,
    unsigned int seed,
    int method,
    int north,
    int kernel,
    float gamma,
    int degree,
    float coefficient);

}  // namespace fastpls_svd

#endif
