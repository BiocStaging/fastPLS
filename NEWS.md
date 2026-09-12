# fastPLS 0.99.65

* Replaced the repository README with a focused installation guide for macOS,
  Windows, Ubuntu, and Fedora. The guide documents platform toolchains,
  OpenBLAS installation and verification, Apple Accelerate and Metal defaults,
  and explicit accelerator failure behavior.

* Audited the public API, vignette, and reference documentation after the CV
  selection update. Corrected the vignette to distinguish fitted-response
  `R2Y` from held-out `Q2Y` and removed avoidable condition-message diagnostics
  reported by `BiocCheck()`.

* Fixed Unix and Windows configuration so `FASTPLS_USE_OPENBLAS=1` accepts a
  detected OpenBLAS installation instead of incorrectly treating the value as
  invalid. Forced installations still fail clearly when OpenBLAS is absent.

# fastPLS 0.99.64

* Made cross-validation selection names explicit and task safe. `R2Y`, `Q2Y`,
  and `RMSD` now use the same public spelling as the PLS outputs; ambiguous
  `"r2"` and `"q2"` inputs are rejected. Selecting `R2Y` automatically enables
  the fitted-response path, while `Q2Y` remains fold-aware and predictive.
  Task-incompatible choices now fail before fitting.

* Extended `selection` to task-appropriate `evaluate()` metrics. Classification
  supports lift accuracy, macro precision/recall/F1, and kappa in addition to
  accuracy and balanced accuracy. Regression supports MAE, MAPE, RPD, and
  Pearson or Spearman correlation in addition to Q2Y and RMSD; this includes
  aggregate selection for large multivariate responses such as NMR. Signed
  bias and signed mean relative error remain evaluation outputs but are not
  exposed as one-sided tuning criteria.

* Renamed the public cross-validation tuning argument from
  `selection_metric` to the shorter `selection` in `pls.single.cv()` and
  `pls.double.cv()`. Returned audit fields `selection_metric` and
  `selection_metrics` are unchanged.

* Simplified `evaluate()` by removing `task` and `top_k`. The task is now
  always inferred from the observed and predicted values, ranked depth is
  inferred from score or ranked-label columns, and complete `predict()`
  results can be evaluated directly. Multi-component prediction results return
  a metric row and a complete evaluation for each component count.

* `predict()` now passes its predictions through `evaluate()` whenever
  `Ytest` is supplied. The complete component-wise evaluation is returned in
  `metrics` consistently across CPU, CUDA, Metal, float32, and float64 routes.

* Standardized CPU parallelism as `n.cores`. The explicit argument is
  available in `pls()`, `pls.single.cv()`, `pls.double.cv()`, `fastsvd()`,
  `predict()`, and `fastcor()` and takes precedence over the session-wide
  `options(n.cores = ...)` value. Nested CV, refitting, permutation, and
  prediction paths retain the resolved request; CUDA and Metal device
  parallelism remains controlled by their native runtimes.

* Simplified ranked classification prediction to one `top` argument. Its
  default is `NULL`, which returns one predicted class per sample; a positive
  integer requests that many ranked classes. The redundant `top5` and inactive
  `flash.block_size` arguments were removed, and regression prediction now
  warns when `top` is supplied.

* Aligned float64 and float32 ranked prediction so both use bounded row blocks
  and retain only the requested ranks unless `raw_scores = TRUE`.

* Removed the redundant `method` argument from `fastsvd()`. The function now
  selects native rSVD directly and automatically preserves float64 execution
  for ordinary R matrices or end-to-end float32 execution for `float32`
  matrices.

* Corrected the portable Windows float32 SVD dispatch so it no longer expects
  the retired solver argument.

* Shortened the `pls()` reference details to practical model, precision,
  backend, and diagnostic guidance; mathematical derivations remain outside
  the function manual.

* Removed the deprecated `lda_ridge` argument from all public PLS functions
  and the obsolete `xprod` argument from `pls.single.cv()`. LDA stabilization
  and matrix-free route selection are now exclusively automatic.

# fastPLS 0.99.63

* Removed the redundant `fastPLS_backend()` setter/getter. Backends are now
  selected through an explicit `backend =` argument or the session-wide
  `options(backend = ...)` setting.

# fastPLS 0.99.62

* Removed the redundant `svd.method` argument from `pls()`,
  `pls.single.cv()`, and `pls.double.cv()`. These functions now select the
  native rSVD implementation automatically for the requested CPU, CUDA, or
  Metal backend; rSVD accuracy controls remain available through `...`.

* Removed the solver field from cross-validation tuning output and automatic
  refit calls. Calls that still supply the retired argument fail with a clear
  migration message rather than silently ignoring it.

# fastPLS 0.99.61

* Corrected nested-CV endpoint handling so explicit R2 and Q2 selection can no
  longer substitute for one another. An R2 permutation test now uses the
  held-out R2 endpoint rather than descriptive training `R2Y`.

* Corrected nested-CV metric assembly so run-level and response-wise Q2 values
  retain their fold-training denominators. Single-CV metric paths are now named
  by component count consistently.

* Synchronized the README, vignette, and reference manual with the current
  CPU, CUDA, and Metal cross-validation coordinators.

* Made omitted prediction backends follow the same explicit argument, session
  option, environment variable, and CPU precedence as fitting functions.
  `backend = "auto"` remains the explicit model-aware prediction choice.

* Made OpenBLAS optional at installation so standard Bioconductor Linux and
  Windows builders can use the BLAS/LAPACK supplied by R. Automatic detection
  still prefers OpenBLAS, while `FASTPLS_USE_OPENBLAS=1` provides a fail-fast
  requirement for benchmark builds. Added `fastPLS_blas()` to report whether
  the package was compiled with Accelerate, OpenBLAS, or R BLAS/LAPACK.

* Removed unresolved float32 BLAS/LAPACK dependencies from fallback builds.
  Without Accelerate or OpenBLAS, float32 Gram, QR, eigenvalue, and SVD routes
  now use the package's portable compiled numerical kernels.

* Added a persistent float32 CUDA matrix workspace for compiled
  cross-validation. Device buffers, the CUDA stream, and the cuBLAS handle are
  reused across folds; strided padded matrices are transferred correctly with
  two-dimensional copies. Sample-response Gram matrices use direct
  `cublasSsyrk` and retain only the lower triangle when that is all the
  downstream fold extractor consumes. The hybrid Metal route evaluates this
  host-visible symmetric product with Accelerate SYRK.

# fastPLS 0.99.59

* Added precision- and shape-aware CPU dispatch between GEMM and direct
  SSYRK/DSYRK self-products. Symmetric routes retain one triangle when the
  caller does not require a complete matrix, avoiding unnecessary mirroring.

* Generalized bounded sample-response Gram reuse in wide-response SIMPLS
  cross-validation to all CPU platforms. The raw Gram matrix is computed once,
  each fold is recovered by principal-submatrix extraction and exact double
  centering, and centering/output workspaces are retained across folds. Wide
  gathered response matrices use aligned storage with 64-byte column padding.

* Added direct OpenBLAS discovery on Linux and Windows, with explicit
  `FASTPLS_USE_OPENBLAS=1` fail-fast behavior for controlled benchmark builds.

* Reduced repeated setup in every compiled cross-validation backend. Eligible
  CPU and Metal classification folds now recover both SIMPLS and PLS-SVD
  predictor and class moments from bounded full-data sufficient statistics;
  PLS-SVD can assemble its latent Gram matrix without materializing the
  training-score matrix. CUDA now retains configured cuBLAS, cuSOLVER, cuRAND,
  and device-pointer handles across all folds in one CV call. The Metal route
  sends small reduced products to Accelerate while retaining large assigned
  sample-matrix products on Metal. Fold membership, fold-specific
  preprocessing, component requests, seeds, predictions, and selection
  semantics are unchanged.

* Restored the fresh block randomized operator route for wide-response SIMPLS
  cross-validation when a bounded sample-response Gram matrix is not
  advantageous. Candidate blocks are generated from the current deflated
  operator and consumed through the existing sequential SIMPLS
  orthogonalization and deflation equations; no preceding component direction
  is reused.

* Reduced multivariate-regression CV work without changing its folds or model
  fits. Large implicit PLS-SVD and SIMPLS routes now compute predictor and
  response marginal moments once and recover each training fold by subtracting
  its held-out contribution. Aggregate float32 CV metrics are calculated on the
  native out-of-fold predictions before conversion to R, including an exact
  four-pass float32 Spearman rank calculation. Contiguous prediction paths are
  transferred to R with bulk conversion rather than element-wise indexing.

* Accelerated wide-response SIMPLS cross-validation on Apple Accelerate with a
  bounded sample-space response Gram matrix. When the arithmetic estimate
  predicts a gain, the native CV engine forms the response Gram matrix once,
  extracts and training-centers each fold submatrix, and reuses it throughout
  that fold's sequential direction updates. This is algebraically equivalent
  to the implicit predictor-response products and leaves fold-specific
  preprocessing, seeds, component requests, and prediction semantics
  unchanged. OpenBLAS and CUDA retain their faster implicit or resident route.

* Moved grouped fold construction, prediction evaluation, top-k summaries,
  `fastcor()` and VIP trajectories into the dependency-free C++ core. The R
  functions now validate inputs and format native results. Fold construction
  retains the existing seeded assignments while avoiding repeated scans of
  every sample for every constraint group.

* Removed the optional `RhpcBLASctl` dependency. `options(cores = n)` now uses
  an internal compiled bridge to configure loaded OpenBLAS, MKL, BLIS and
  OpenMP runtimes when they expose thread-control functions, while retaining
  environment-based requests for Apple Accelerate and other BLAS libraries.

* Reused fold matrix allocations in the compiled cross-validation engine,
  including predictor, response, held-out, and predictor-Gram workspaces.

* Made OpenBLAS the CPU numerical library on Linux and Windows.

* Completed the MIT-licensed core migration by removing the remaining Rcpp,
  RcppArmadillo and Armadillo source boundaries, generated wrappers and
  superseded numerical implementations.

* Made the dependency-free C++17 headers installable through the standalone
  `fastpls::core` CMake target and verified them from an external consumer.
  Platform BLAS/LAPACK, CUDA and Metal adapters remain private to the R package
  and do not cross the core interface.

* Removed obsolete compiled routes and bundled third-party example datasets,
  retained strict registered native calls, and preserved CPU, CUDA and Metal
  behavior across PLS-SVD, SIMPLS, OPLS, kernel PLS and LDA tests.

* Reused bounded full-data sufficient statistics in eligible SIMPLS
  classification cross-validation. Fold predictor Gram matrices and class
  predictor sums now provide the score moments required by LDA without
  materializing each training-fold score matrix; fold assignment,
  preprocessing, selected components and classifier semantics are unchanged.

* Added a resident CUDA SIMPLS cross-validation route that uploads each task
  once and retains fold gathering, standardization, fitting, projection,
  prediction and metric reduction on the device. Unsupported accelerator
  configurations continue to fail explicitly rather than falling back to CPU.

* Added resident CUDA PLS-SVD cross-validation for large multivariate
  regression responses. The automatic dispatcher retains the faster
  fold-local CUDA route for classification and smaller responses, while the
  resident route avoids repeated transfer of NMR-scale response matrices.

* Corrected numerical-rank detection in the implicit PLS-SVD rSVD operator.
  The Gram-matrix shortcut now applies the squared singular-value tolerance
  and is accepted only when it supplies the requested retained rank; otherwise
  the implementation uses the direct reduced-SVD fallback.

* Reused one fitted PLS response-score path when argmax and LDA are tuned
  together, and accumulated requested prediction prefixes in place on CPU and
  Metal to avoid rebuilding each prefix from the first component.

# fastPLS 0.99.55

* Prevented Linux BLAS symbol interposition from routing the standalone
  float32/float64 CPU matrix-product adapter through a slower BLAS than the
  OpenBLAS library selected at package configuration. The fixed CIFAR-100
  performance gate retains identical predictions and restores the prior
  same-host runtime.

* Corrected CUDA cross-validation dispatch for linear kernel PLS so that it
  reuses the intended direct SIMPLS route rather than entering the nonlinear
  kernel constructor.

# fastPLS 0.99.54

* Replaced six generated RcppArmadillo wrappers for double-precision CPU LDA
  fitting, sufficient-statistics fitting, projection and prediction with a
  hand-written R C-API boundary over the MIT core and optimized CPU GEMM.

* Removed the superseded Rcpp projection and prediction implementations.
  Old and new boundaries return the same model fields and labels, while a
  representative BLAS-backed LDA workload retained its previous median time.

# fastPLS 0.99.53

* Consolidated double- and single-precision pooled-covariance LDA training in
  the dependency-free MIT C++17 core, including a sufficient-statistics entry
  point for compiled cross-validation.

* Removed the superseded Armadillo LDA implementation and its duplicate test.
  Double-precision coefficients and discriminant scores agree with the former
  implementation to near machine precision, with identical predictions and
  deterministic regularization selection.

# fastPLS 0.99.52

* Migrated the internal CUDA matrix-product bridge from RcppArmadillo to a
  precision-aware C ABI over cuBLAS, preserving prediction behavior while
  removing the superseded Armadillo implementation.

* Added a dependency-free, templated pooled-covariance LDA implementation to
  the MIT C++17 core and routed CPU float32 LDA through the direct R C-API.
  This removes the former Windows-specific LDA restriction, the Armadillo
  float32 implementation and two generated wrappers.

# fastPLS 0.99.51

* Replaced the resident CUDA Rcpp marshalling layer with direct R C-API
  entry points while preserving the existing CUDA C ABI, model fields and
  explicit no-fallback behavior. This removes ten generated Rcpp wrappers
  from the package boundary.

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
