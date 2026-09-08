# fastPLS 0.99.50

* Revalidated the source package with BiocCheck 1.49.30 before the
  Bioconductor staging update.

# fastPLS 0.99.49

* Made the CMake `fastpls::core` target independently configurable, testable
  and installable without Armadillo, BLAS or LAPACK. The current
  Armadillo-based model layer is now an explicitly optional transitional
  target.

* Added an installed-consumer test that compiles against only the public
  dependency-free headers and `fastpls::core`, providing a direct ABI and
  packaging gate for the future standalone library.

# fastPLS 0.99.48

* Moved float64 kernel-matrix construction from the generated
  RcppArmadillo interface to the hand-written R C-API and dependency-free
  kernel core.

* Added a narrow runtime BLAS adapter for double-precision matrix products,
  using Accelerate or OpenBLAS where configured and R's portable BLAS ABI on
  other package builds. This preserves optimized kernel construction without
  exposing a BLAS implementation in the standalone core ABI.

# fastPLS 0.99.47

* Added a dependency-free, templated C++17 implementation of linear,
  polynomial and radial-basis kernel transformations and train/test kernel
  centering for the future standalone core.

* Routed the transitional Armadillo kernel layer through the shared MIT core
  and moved float32 and float64 kernel-centering entry points to the hand-written
  R C-API bridge, removing four more generated Rcpp wrappers and the duplicated
  Windows implementation.

* Preserved all kernel-PLS and OPLS tests and the fixed CIFAR-100 SIMPLS/rSVD
  accuracy and prediction checksum.

# fastPLS 0.99.46

* Migrated float32 argmax, top-rank selection, column operations and
  standardization from generated Rcpp/Armadillo adapters to the dependency-free
  C++17 core and hand-written R C-API bridge.

* Moved label-aware scaled class cross-products and rSVD audit diagnostics into
  the dependency-free core, and removed their obsolete generated wrappers and
  duplicate implementations.

* Preserved the fixed CIFAR-100 float32 SIMPLS/rSVD prediction checksum and
  accuracy while further reducing the transitional Rcpp interface.

# fastPLS 0.99.45

* Moved capability detection and Spearman correlation from generated Rcpp
  adapters to a small hand-written R C-API layer. R2 and dummy-response
  helpers now use dependency-free C++ or R implementations.

* Added dependency-free C++17 statistics primitives and tests to the public
  core, and removed obsolete matrix-view fits, unused LDA helpers, and stale
  CUDA/Metal capability wrappers.

* Preserved the tested SIMPLS/rSVD component path and CIFAR-100 predictions
  while reducing the transitional Rcpp interface by five native wrappers.

# fastPLS 0.99.44

* Introduced a dependency-free C++17 core interface with non-owning matrix
  views, owned buffers, reference matrix products, and label-aware response
  cross-products. The interface contains no R, Rcpp, RcppArmadillo, or
  Armadillo types and provides the ABI-neutral foundation for the ongoing
  standalone-core migration.

* Routed the production float32 CPU matrix-product bridge through the new core
  interface while retaining Apple Accelerate, configured OpenBLAS, and the
  portable reference implementation as private execution details.

* Removed four unreachable CUDA and Metal entry points and their generated R
  wrappers. CIFAR-100 predictions remain bit-for-bit stable after these
  changes, and the native CMake suite now tests both the independent core and
  the transitional Armadillo-based numerical layer.

# fastPLS 0.99.43

* Accelerated the float32 CPU sample-matrix products used by SIMPLS and
  PLS-SVD through an explicitly configured OpenBLAS runtime on supported
  Linux installations. The native kernels honor `OPENBLAS_NUM_THREADS`, while
  macOS continues to use Apple's optimized Accelerate framework.

* Added shape-appropriate reuse of predictor cross-products, batched
  score/loading geometry, deferred training-score materialization, and
  sufficient-statistics reuse in eligible cross-validation folds. These
  changes preserve requested component and fold semantics while reducing
  repeated dense products and allocations.

* Extended numerical route diagnostics and platform tests for the optimized
  CPU, CUDA, and Metal paths.

# fastPLS 0.99.42

* Replaced the fully resident public Metal PLS route with the faster fixed
  CPU/Metal operation split. `backend = "metal"` now retains preprocessing,
  reduced decompositions, sequential component updates, LDA and prediction on
  CPU while persistent Metal workspaces execute all sample-matrix products:
  explicit cross-covariance formation, fused score/loading geometry, and
  randomized range-finder products for implicit cross-covariance operators.
  The public `metal_hybrid` name was removed, and the assignment never changes
  with dataset shape.

* Added route validation, grouped cross-validation coverage, and documentation
  for hybrid PLS-SVD, SIMPLS, OPLS, and kernel-PLS fitting. Standalone
  `fastsvd()` remains restricted to its existing backends.

# fastPLS 0.99.39

* Compacted fold-specific active labels before LDA fitting and mapped
  predictions back to the original factor levels. Cross-validation now remains
  defined when a rare class is absent from an inner training fold, while
  ordinary PLS-LDA drops unused factor levels before fitting.

* Stabilized float32 CUDA and Metal rSVD power iterations with alternating
  orthonormalization of the left and right sketches. This keeps large
  class-sum cross-covariance calculations in single precision without the
  overflow and rank collapse caused by unnormalized repeated products.

* Changed the automatic massive-cross-covariance SIMPLS-family rSVD profile
  from 10 oversampling directions and one power iteration to 12 directions and
  two iterations. Matched NMR tests across multiple seeds restored the stated
  prediction-agreement tolerance on CPU and Metal while retaining the fast
  fresh-start implementation; CUDA timing was unchanged.

* Corrected rSVD control resolution so diagnostic qualification metadata is
  generated after shape-specific controls are selected and therefore always
  describes the settings that were executed.

* Made the massive CUDA rank-one refresh execute the requested power-iteration
  count and report the same value. The automatic fast profile remains two
  iterations, so its execution path and benchmark workload are unchanged.

# fastPLS 0.99.38

* Enforced strict accelerator selection across public and native fitting,
  prediction, SVD, LDA, and cross-validation routes. Requests for unavailable
  CUDA or Metal backends now stop with an explicit error; CPU is never used as
  an implicit substitute.

* Added independent CUDA and Metal regression tests for unavailable-backend
  handling, including PLS-SVD, SIMPLS, OPLS, kernel PLS, prediction, and
  single- and double-cross-validation entry points.

# fastPLS 0.99.37

* Strengthened automatic rSVD controls for SIMPLS, OPLS, and kernel PLS.
  Ordinary problems use 32 oversampling directions and five power iterations;
  high-response regression uses 48 and six; sparse many-class classification
  uses 64 and seven. Cross-covariances larger than 512 MiB retain the separate
  10/1 massive-matrix fast path. Explicit user controls remain unchanged.

* Kept backend selection strict: unavailable CUDA or Metal requests raise an
  informative error and are never silently executed on CPU.

* Made `fastPLS_backend()` validate accelerator availability when a session
  backend is set or retrieved, so an unavailable CUDA or Metal choice fails
  immediately.

# fastPLS 0.99.36

* Made accelerator dispatch strict across fitting, prediction, SVD, LDA, and
  cross-validation. A selected CUDA or Metal backend now raises an informative
  error when unavailable instead of silently running the operation on CPU.

* Restored the fast device-resident CUDA SIMPLS path for massive
  predictor-response cross-covariance matrices, using a fresh rank-one
  randomized start for every component and the 10/1 massive-matrix profile.
  Ordinary accelerated SIMPLS-family routes used a 32/2 profile. PLS-SVD and
  standalone `fastsvd()` use the more conservative 32/5 defaults across
  backends.

* Removed obsolete benchmark and publication artifacts that were generated by
  earlier package versions. The retained publication workflow records the
  package version and exact solver controls for every result.

# fastPLS 0.99.35

* Unified randomized SIMPLS initialization across CPU, CUDA, and Metal. Every
  refresh now generates a new seeded random direction or sketch from the
  current deflated cross-covariance state.

* Replaced the package-specific session option with `options(backend = ...)`.
  Explicit function arguments retain precedence, followed by the session
  option, `FASTPLS_BACKEND`, and the CPU default.

* Added `options(cores = n)` CPU thread control. fastPLS forwards the requested
  positive integer to common BLAS and OpenMP runtimes; eligible matrix products
  can use those threads when supported by the linked numerical library.

# fastPLS 0.99.34

* Added a CUDA-specific batched randomized direction refresh for dummy-coded
  classification. The route consumes at most eight candidates through the
  sequential orthogonalization path while retaining rank-one refresh for
  regression, CPU, and Metal execution.

* Retained the faster resident Metal rank-one implementation after an
  alternative fused initializer was slower in matched CIFAR-100 testing.

* Added explicit diagnostics for backend-specific direction batching and
  audited CPU, multithreaded-BLAS CPU, CUDA, and Metal execution separately.

# fastPLS 0.99.33

* Restored an explicitly approximate accelerated-SIMPLS profile with seeded,
  component-specific randomized directions generated from the current
  deflated cross-covariance state.

* Added task-aware internal rSVD controls without adding public arguments:
  oversampling 10 with two power iterations for classification and one for
  numeric regression. Explicit controls supplied through `...` still win.

* Separated the accelerated profile from claims of deterministic de Jong
  estimator preservation in diagnostics and documentation.

# fastPLS 0.99.32

* Corrected the rSVD case audit so a near-tied retained/omitted singular-value
  boundary remains diagnostic rather than triggering repeated deterministic
  recovery. The singular-triplet residual tolerance is unchanged.

* Batched the matrix-free left/right residual products and reused the computed
  retained-plus-one Ritz boundary, avoiding repeated full operator probes on
  very wide multivariate responses.

* Added a near-tied-spectrum regression test and clarified Metal rSVD warning
  text.

# fastPLS 0.99.31

* Reformatted all R sources with four-space indentation while preserving the
  existing compact function layout and 80-character line-width policy.

# fastPLS 0.99.30

* Fixed Metal PLS dispatch after helper refactoring by validating model,
  scaling, and kernel arguments against explicit public choices. This removes
  the missing-argument error in Metal fitting and cross-validation tests.

# fastPLS 0.99.29

* Resolved the BiocCheck source-formatting notes by limiting package-facing
  lines to 80 characters and refactoring functions to at most 50 coding lines.

* Fixed float32 single-cross-validation argument forwarding and completed the
  resident Metal SIMPLS model-assembly helpers.

* Preserved the public PLS, cross-validation, metric, and permutation
  contracts. The complete package test suite passes.

# fastPLS 0.99.28

* Added direct support for `Biobase::ExpressionSet` predictor inputs in
  `pls()` and `predict()`. Assay rows are interpreted as variables and assay
  columns as samples, and are transposed internally to the sample-by-variable
  layout used by fastPLS.

* Added a Bioconductor-native classification example to the package vignette
  and expanded automated tests for fitting and predicting from
  `ExpressionSet` objects.

* Reformatted R and vignette sources to improve Bioconductor style compliance.

# fastPLS 0.99.25

* Synchronized the exported API, vignette, manual, README, and manuscript
  capability descriptions. Removed stale PCA references and deprecated the
  ignored `lda_ridge` compatibility argument; supplying it now warns, and it is
  no longer included in cross-validation tuning records.

* Standardized response-variance metrics across the public API. Training
  `R2Y`, independent-test `Q2Y`, fold-training-mean single-CV `Q2Y`, and
  outer-fold `Q2Y` now use explicit, documented denominators; dummy-response
  PLS-DA values are labelled separately from classification accuracy.
  `evaluate()` returns `NA` for Q2 when no training response is supplied rather
  than silently reproducing R2.

* Corrected finite Monte Carlo permutation inference to use
  `(b + 1) / (B + 1)`, preventing zero p-values. Grouped nested validation now
  permutes complete constraint blocks within equal-size exchangeability strata,
  holds folds and randomized-solver seeds fixed, and records failed null fits.

* Clarified that returning the sequential SIMPLS component path is standard
  behavior also provided by `pls::simpls.fit` and is not claimed as a fastPLS
  novelty.

* Defined the fastPLS contribution as compiled, shape-dependent execution and
  storage: cached deflation and cross-products, incremental coefficient and
  fitted-value updates, compact latent prediction, and implicit
  cross-covariance products.

* Added a minimally optimized compiled SIMPLS baseline and explicit asymptotic
  time/storage expressions to the implementation mapping and vignette, while
  documenting that the retained optimizations are not uniformly faster.

# fastPLS 0.99.24

* Standardized the public SIMPLS direction-refresh rule across CPU, CUDA, and
  Metal. Every component now receives a fresh rank-one IRLBA solve or
  oversampled rSVD sketch from the current deflated cross-covariance; no
  candidate block or preceding latent direction is reused.

* Removed unused experimental direction controls from the backend-control
  registry. All backends generate new randomized directions from the current
  deflated state.

* Added fitted-model diagnostics and unit tests that identify the active
  `fresh_per_component` rule and retained execution optimizations.

# fastPLS 0.99.23

* Changed the randomized-SVD default to `(oversample = 20, power = 2)` on CPU
  and CUDA. This stronger setting met all 585 CPU and 40 CUDA component-level checks across
  five prespecified random seeds in the release-candidate audits; the prior
  `(10, 2)` setting failed five of 255 screening checks and is not treated as
  qualified.

* Added an explicit warning and fitted-model diagnostic status whenever a
  user requests randomized controls that were not qualified on the
  prespecified backend validation panel. Metal randomized SVD remains marked
  as unqualified pending a dedicated multi-seed audit.

* Aligned non-exported C++ bridge defaults with the qualified CPU controls and
  expanded release tests for backend-specific dispatch and diagnostics.

# fastPLS 0.99.22

* Made the numerically qualified randomized-SVD configuration the package-wide
  default: oversampling is 10 and the number of power iterations is 2.

* Removed undocumented, matrix-shape-dependent SIMPLS overrides that could
  silently reduce randomized-SVD oversampling or power iterations. Explicit
  controls supplied through `...` still take precedence and are recorded in
  fitted-model diagnostics.

* Added release tests that verify the effective randomized-SVD defaults used by
  `fastsvd()`, `pls()`, and cross-validation.

* Strengthened benchmark provenance records with Git worktree support and
  source tree and tag identifiers.

# fastPLS 0.99.21

* Made randomized SVD the effective default throughout the public PLS and
  cross-validation APIs, including the internal cross-validation tuning grid
  and the refit path used by stored CV configurations.

* Corrected single-split permutation p-values for multi-component models so
  each permuted Q2 distribution is compared with the corresponding observed
  component rather than a recycled full Q2 vector.

* Expanded the permanent input-grid tests across all four PLS families,
  regression and classification, argmax and LDA, IRLBA and rSVD, and both
  single and double cross-validation. CPU, CUDA, Metal, and float32 grids were
  also exercised during release validation.

# fastPLS 0.99.20

* Fixed nested permutation testing with latent-space LDA. Recursive
  `pls.double.cv()` calls now receive the public `"lda"` classifier name and
  resolve it for the selected backend internally, instead of leaking the
  CPU-specific internal identifier `"lda_cpp"` through the public API.

# fastPLS 0.99.19

* Incremented the Bioconductor development version after synchronizing the
  package-specific backend configuration across both GitHub repositories.

# fastPLS 0.99.18

* Standardized backend precedence across the KODAMA ecosystem. This historical
  package-specific option was superseded by the generic session option in
  version 0.99.35.

# fastPLS 0.99.17

* Added the initial session-wide backend selector. The current selector and
  precedence rules are documented under version 0.99.35.

# fastPLS 0.99.16

* Corrected CUDA cross-validation smoke tests to request the supported rSVD
  backend explicitly instead of inheriting an incompatible IRLBA setting.

# fastPLS 0.99.15

* Corrected portable Windows float32 classification prediction so argmax
  decoding no longer calls an unavailable native single-precision kernel.

* Made float32 capability-policy tests platform-independent by explicitly
  testing Unix accelerator policies separately from Windows availability.

* Simplified the `fastcor()` example to use ten numeric rows from `iris`.

# fastPLS 0.99.14

* Incremented the Bioconductor development version to trigger refreshed
  multi-platform validation of the architecture-independent randomized-SVD
  diagnostic test.

# fastPLS 0.99.13

* Made the randomized-SVD diagnostic test architecture-independent. The test
  now accepts and verifies the documented large-residual failure state instead
  of assuming that a stochastic approximation must meet the quality threshold
  on every BLAS and CPU architecture. Runtime diagnostics remain unchanged.

# fastPLS 0.99.12

* Corrected Windows test scoping for native CPU float32 LDA. Tests that require
  unavailable single-precision BLAS/LAPACK kernels are now skipped on Windows;
  the documented runtime error and portable supported float32 routes are
  unchanged.

# fastPLS 0.99.11

* Removed retired classification and class-bias native ABI branches,
  including CPU and CUDA kernels, generated Rcpp wrappers, registrations, and
  unreachable compiled cross-validation code. Classification remains limited
  to the documented argmax and latent-space LDA heads.

* Preserved compact top-k argmax prediction through a bias-free CPU/CUDA
  implementation, including optional top-5 output.

* Removed retired classifier variants and tuning controls from active benchmark
  generators.

# fastPLS 0.99.10

* Fixed compilation of CPU-only Windows builds. An unavailable native-float32
  argmax route now raises the documented platform error through a type-correct
  integer-vector entry point instead of returning a list.

* Removed the public PCA API and its S3 methods. Principal component analysis
  remains available through dedicated R packages; `fastsvd()` remains the
  package's public standalone decomposition interface.

* Synchronized the public API and documentation around the two supported
  classification heads, argmax and latent-space LDA.

* `pls.single.cv()` and `pls.double.cv()` can now select classification models
  with `selection_metric = "balanced_accuracy"`. Nested permutation tests use
  the same selected endpoint, preventing classification analyses tuned and
  reported by balanced accuracy from being tested against dummy-response Q2.
* macOS installation now detects the system Metal frameworks even when
  `xcrun --show-sdk-path` is unavailable, and configure output reports CUDA and
  Metal status independently.
* Float32 capability reporting now distinguishes validated, experimental,
  hybrid, unavailable, and measured failed routes. The public `pls()` interface
  emits route-specific warnings or errors before allocation, and benchmark
  summaries separate input storage, baseline and incremental host RSS, sampled
  GPU use, runtime, and predictive differences from float64.
* Added public PLS, SVD, prediction, evaluation and cross-validation
  interfaces for CPU, optional CUDA and optional Apple Metal backends.
* Added optional classification heads for PLS-DA using argmax decoding or
  latent-space LDA.
* Added package datasets, examples, benchmark scripts and a single user
  vignette.
* CUDA and Metal builds are optional; CPU-only installation remains the default.
* Float32 SIMPLS now retains the latent scores already produced by the compiled
  recurrence. LDA reuses these scores instead of centering, scaling and
  projecting the full training matrix a second time, reducing peak memory and
  runtime without changing predictions.
* Float32 fitting now reports shape-based warnings for precision-sensitive
  classification, extreme multivariate responses, and nonlinear kernel routes.
  Float64 remains the numerical reference.
