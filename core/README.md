# Native numerical-core extraction

Two CMake targets separate the stable interface from transitional numerical
code:

| Target | Contract |
| --- | --- |
| `fastpls::core` | Dependency-free C++17 matrix views, owned buffers, reference products, and label-aware response products; no R or third-party types cross the interface. |
| `fastpls::native` | Current optimized model algorithms using Armadillo privately; retained while algorithms are migrated behind the core interface. |

New standalone-facing APIs should use `fastpls::core` types. This keeps memory
layout and ABI under fastPLS control and allows BLAS, CUDA, Metal, or portable
kernels to remain private backend details. The R adapter is being migrated
incrementally so each step can be checked against fixed prediction and runtime
gates.

The shared numerical headers implement native rSVD, SIMPLS, PLS-SVD and
matrix-free cross-covariance products in float32 and float64. These are
extractions of the current optimized implementation, not replacements with the
earlier Eigen prototype. Every refreshed random sketch starts from its seed;
no previous latent direction initializes the next sketch.

| Header | Shared use in the current R package |
| --- | --- |
| `rsvd.hpp` | CPU double-precision explicit rSVD adapter |
| `direction.hpp`, `simpls.hpp` | CPU double-precision explicit SIMPLS and its compiled CV callers |
| `plssvd.hpp` | CPU double-precision explicit PLS-SVD and its compiled CV callers |
| `operators.hpp` | CPU float32 explicit/matrix-free products and factor-product reduction |
| `operator_rsvd.hpp` | Typed operator rSVD and checked CPU recovery without IRLBA |
| `opls.hpp` | CPU double-precision OPLS filtering and held-out filter application |
| `kernels.hpp` | CPU float64 kernels and non-Windows float32 kernel transforms/centering |
| `models.hpp` | Standalone CPU composition of the shared OPLS/kernel and SIMPLS stages |
| `lda.hpp` | CPU float64 and non-Windows CPU float32 LDA Cholesky/triangular solves |

The generic solver callbacks permit separate solver adapters without copying
the PLS engines. They do not load or depend on IRLBA. IRLBA integration has
moved to the GPL companion; the main package no longer bundles that solver.
Float32 PLS fitting, implicit float64 PLS, accelerator OPLS filtering,
classification heads, complete CV orchestration and GPU engines have not all
been extracted. This is not yet a complete standalone replacement for the R
package. GPU product wrappers remain outside these headers.

Our native headers are MIT-licensed. They use external Apache-2.0 Armadillo headers
and an external BLAS/LAPACK implementation, each retaining its own license.
Neither R nor Rcpp is required by this build. The R package adapter remains
separately coupled to RcppArmadillo; the overall package is not yet an MIT
distribution, and its DESCRIPTION license has not been changed.

```sh
cmake -S core -B /tmp/fastpls-native -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/fastpls-native
ctest --test-dir /tmp/fastpls-native --output-on-failure
```

Use `ARMADILLO_INCLUDE_DIR` and standard CMake BLAS/LAPACK selection variables
when those libraries are installed outside the default search paths. No
dependency is downloaded automatically. Tests do not require R, datasets,
IRLBA or any external PLS implementation.

The development core can also be installed for a separate C++ project:

```sh
cmake --install /tmp/fastpls-native --prefix /path/to/fastpls-native-install
cmake -S core/tests/consumer -B /tmp/fastpls-consumer \
  -DCMAKE_PREFIX_PATH=/path/to/fastpls-native-install
cmake --build /tmp/fastpls-consumer
ctest --test-dir /tmp/fastpls-consumer --output-on-failure
```

Consumers use `find_package(fastpls_native CONFIG REQUIRED)` and link to
`fastpls::native`. External Armadillo and BLAS/LAPACK are rediscovered on the
consumer's system, not bundled or tied to the producer's machine. Only the
native headers and their MIT notice are installed. `BUILD_TESTING=OFF` skips
the core tests when configuring a dependency build. The core's development
version 0.0.1 is separate from the R package version and is not a published
release claim.

For SIMPLS use `fit_simpls(X, Y, components, options)` and
`predict_simpls(model, Xtest, component_count)`. For PLS-SVD use
`fit_plssvd(X, Y, components, options)` and
`predict_plssvd(model, Xtest, prefix_index)`; the index is zero-based into the
requested component vector. Retention of dense coefficients and fitted
responses is optional. Matrix operator objects reference their input response
matrix, which must outlive the operator. Each CPU cross-covariance operator
owns its mutable predictor factor and reusable intermediate workspace.

`fit_opls_filter(X, Y, north, scaling)` provides a typed native-rSVD orthogonal
filter; `apply_opls_filter` applies its stored preprocessing and factors to new
predictors. The main R CPU float64 adapter shares the filter updates but keeps
its existing dense leading-direction callback, preserving that path's current
arithmetic. Native float32 filter tests do not establish that the R float32 or
GPU adapters already use this extracted header. Full OPLS prediction combines
the filter with the predictive PLS model; the filter alone is not that model.

`kernel_matrix` implements kernel identifiers 1 (linear), 2 (radial basis)
and 3 (polynomial). `center_kernel_train` returns the centered matrix and
training means; `center_kernel_test` reuses those means for new observations.
`kernel_from_dots` accepts an already computed dot-product matrix so existing
accelerator adapters can keep their product on the selected backend before
applying the shared host-side transform. This does not imply GPU residency
for centering or a complete standalone kernel-PLS prediction interface.

For complete standalone CPU prediction, `models.hpp` provides `fit_opls` /
`predict_opls` and `fit_kernelpls` / `predict_kernelpls` for both scalar types.
They compose the shared stages rather than duplicate PLS engines. OPLS drops
its filtered training matrix after fitting. Linear kernel PLS dispatches
directly to SIMPLS without retaining a Gram matrix or training reference;
nonlinear prediction retains the standardized reference and training kernel
means. `KernelPlsOptions` defaults to a linear kernel; set nonlinear gamma
explicitly (its native default is 1, not R's automatic gamma selection).
These wrappers return numeric multivariate predictions. They do not implement
LDA, class-label decoding or cross-validation, and are not yet replacements
for the R package's outer-model orchestration.

The shared LDA solver preserves separate double-precision LAPACK and float32
workspace-based Cholesky implementations. It does not form an inverse.
Diagonal regularization is `rho * s`, where `s = trace(covariance) / width`
when finite and positive, otherwise 1. The sequence is 1e-8, 1e-6, 1e-5,
1e-4, 1e-3, 1e-2; the first successful factorization and finite triangular
solution is retained. Failure at every level throws. Float32 workspaces are
thread-local and reused. This extracts the solve stage, not the complete
classifier fit/predict API or accelerator LDA engines.

The R CPU adapters retain their ordinary rSVD attempts and checks. When those
attempts fail, `recover_operator_rsvd` retries a fresh Gaussian sketch with a
wider subspace and more power iterations, followed by a reduced SVD. The
caller must accept the result through its residual check; failed recovery
throws rather than returning an unchecked or silently substituted solver.
Recovery allows four extra attempts and limits the conservative estimate of
its major buffers to 256 MiB. This estimate excludes inputs, existing model
storage, runtime overhead and allocator behavior; it is not a bound on RSS.
The full-width recovery route still uses the projected operator, rather than
calling a full decomposition of an implicit predictor-response matrix.
