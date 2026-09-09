# fastPLS C++ core

`fastpls::core` is the dependency-free C++17 numerical interface used by the
fastPLS R package. Its public types are owned column-major matrices, non-owning
matrix views, solver controls, model state, and backend concepts. No R, Rcpp,
RcppArmadillo, Armadillo, BLAS, LAPACK, CUDA, or Metal type crosses this
interface.

The headers implement:

- randomized SVD for explicit matrices and matrix operators;
- PLS-SVD and SIMPLS for float32 and float64;
- label-aware cross-products for classification;
- OPLS filtering and predictive-model composition;
- linear, radial-basis, and polynomial kernel PLS;
- pooled-covariance LDA without explicit inversion;
- fold construction, compiled cross-validation, and summary statistics.

Numerically intensive operations are supplied through a small backend concept:
matrix multiplication, economy QR, symmetric eigendecomposition, and economy
SVD. The R package provides CPU BLAS/LAPACK, CUDA, and Metal adapters privately.
This keeps third-party libraries out of the standalone ABI while allowing
platform-specific acceleration.

## Build and test

The standalone headers and tests require only a C++17 compiler and CMake:

```sh
cmake -S core -B /tmp/fastpls-core -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/fastpls-core
ctest --test-dir /tmp/fastpls-core --output-on-failure
```

Install the interface and verify it from an independent consumer project:

```sh
cmake --install /tmp/fastpls-core --prefix /tmp/fastpls-core-install
cmake -S core/tests/core_consumer -B /tmp/fastpls-core-consumer \
  -DCMAKE_PREFIX_PATH=/tmp/fastpls-core-install
cmake --build /tmp/fastpls-core-consumer
ctest --test-dir /tmp/fastpls-core-consumer --output-on-failure
```

Consumers use `find_package(fastpls_core CONFIG REQUIRED)` and link to
`fastpls::core`.

## Numerical contract

Every randomized sketch starts from its recorded seed. SIMPLS consumes a
freshly generated candidate block and applies sequential orthogonalization and
deflation to each accepted component. PLS-SVD computes one randomized dominant
subspace and reuses its component prefixes. Retention of scores, coefficients,
and fitted responses is controlled independently so prediction-only workflows
do not allocate dense output paths.

LDA computes the pooled within-class covariance from score cross-products and
class means, then solves by Cholesky factorization and triangular substitution.
The deterministic diagonal-regularization sequence is
`1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2` times the covariance trace scale.

The core is licensed under the MIT License. Platform libraries used by an
embedding application retain their own licenses.
