# fastPLS

`fastPLS` provides compiled C++, CUDA, and Apple Metal implementations of partial least squares
models for high-dimensional regression and classification. The user-facing API
is intentionally small: algorithms and implementation backends are selected
through `pls()`, `pls.single.cv()`, `pls.double.cv()`, and
`fastsvd()` instead of through low-level implementation wrappers.
The current benchmark suite compares four model families:

- `plssvd`
- `simpls`
- `opls`
- `kernelpls`

The `simpls` implementation is the optimized fastPLS SIMPLS core. Older
low-level SIMPLS tuning arguments are not part of the public API; new analyses
should use `method = "simpls"`.

The explicit package exports are `pls()`, `pls.single.cv()`, `pls.double.cv()`,
`evaluate()`, `plot.permutation()`, `ViP()`, `fastsvd()`, `fastcor()`,
`fastPLS_backend()`, `fastPLS_blas()`, `has_cuda()`, and `has_metal()`. Standard `predict()` and
`plot()` generics dispatch to registered fastPLS methods. There is no exported PCA API. The supported model families are `plssvd`,
`simpls`, `opls`, and `kernelpls`; classification uses `argmax` or latent-space
`lda`. The deprecated `lda_ridge` compatibility argument is ignored and warns
when supplied because LDA uses a fixed scale-normalized Cholesky fallback.

## Installation

Install the released package from Bioconductor:

```r
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("fastPLS")
```

The GitHub repository contains the development source and the optional
CUDA/Metal build instructions below.

## Example and benchmark data

The vignette uses datasets supplied by Biobase and base R. Prepared real
benchmark matrices such as CCLE, GTEx, and TCGA subsets are intentionally not
redistributed with fastPLS. Acquisition and data-use notes are provided in
`inst/DATA_SOURCES.md`.

## Algorithms

- `plssvd`: computes the dominant subspace of the cross-covariance
  `S = X^T Y` and reuses it for the requested component path.
- `simpls`: accelerated sequential SIMPLS. The default rSVD route is an
  explicitly approximate high-speed profile: component-wise CPU, CUDA, and
  Metal routes generate a new randomized sketch for each component. Eligible
  large dummy-coded classification paths can instead generate up to 64 fresh
  CPU, CUDA, or Metal candidates together to amortize repeated matrix
  operations and GPU launches. Each candidate is accepted only after the sequential
  SIMPLS orthogonalization and deflation update. Compact
  prediction, cached deflation products, incremental coefficient updates, and
  automatic matrix-free `xprod` further reduce computation and storage. When
  the response dimension is smaller than the predictor dimension, eligible
  rSVD fits update the response-side Gram matrix after each accepted component;
  this replaces repeated tall cross-covariance sketches with a smaller
  eigen-subspace calculation without changing the subsequent SIMPLS update.
- `opls`: supervised orthogonal filtering followed by the selected PLS core.
- `kernelpls`: linear, RBF, or polynomial kernel construction followed by the
  selected PLS core.

The CPU backend uses Apple Accelerate by default on macOS. Linux and Windows
prefer OpenBLAS when configuration finds it through `OPENBLAS_ROOT`,
`pkg-config`, or Rtools; otherwise fastPLS uses the BLAS/LAPACK supplied by R.
Set `FASTPLS_USE_OPENBLAS=1` before installation to require OpenBLAS and fail
clearly when it is unavailable. `fastPLS_blas()` reports the library selected
when the installed package was compiled. Our Linux and Windows performance
benchmarks require `identical(fastPLS_blas(), "OpenBLAS")`.
Set `options(cores = 4L)` to request four CPU threads. Eligible matrix
operations can use multiple cores when linked to a multithreaded BLAS,
for example OpenBLAS. SIMPLS deflation remains sequential, so multicore gains depend
on matrix shape. In the controlled one-, two-, and four-thread study, the
four-thread speed-up ranged from 1.03- to 1.77-fold across the three tested
matrix regimes; this is not a guarantee that additional threads help every fit.

CPU self-products are dispatched between GEMM and SYRK according to platform,
precision, and matrix shape. SYRK computes one triangle, and fastPLS mirrors it
only when a downstream operation requires a complete matrix. Owned numerical
workspaces use aligned storage; wide fold-response matrices additionally align
each column to a 64-byte boundary. For eligible wide-response cross-validation,
the raw response Gram matrix is computed once and each fold is obtained by
principal-submatrix extraction followed by exact double centering. Classification
uses compact labels and class sums rather than a dense one-hot response matrix.

For classification, factor responses are handled as PLS-DA responses. Large
response spaces use compact prediction where possible so the full coefficient
cube does not need to be stored.

For PLS-DA with LDA classification, the recommended high-accuracy/high-speed
configuration is `method = "plssvd", backend = "cuda", classifier = "lda"`.
This uses a resident CUDA path for PLS fitting, latent projection, LDA training,
and discriminant scoring; only returned arrays and R-object assembly are copied
to the host. On systems without CUDA, users can explicitly select
`method = "plssvd", backend = "cpu", classifier = "lda"` for compiled CPU
execution.

For large classification problems, such as ImageNet-scale DINOv2 feature
matrices, `method = "plssvd", backend = "cuda"` uses compact integer labels to
form class-wise cross-products without materializing a dense `n x classes`
one-hot response. The fitted model stores compact low-rank prediction factors.

## Backends

Set the fastPLS session default with `options(backend = "cuda")`, or use
`Sys.setenv(FASTPLS_BACKEND = "cuda")`. An explicit function argument always
takes precedence.

The public CPU solver is `rsvd`, a randomized SVD with Gaussian sketching and
power iterations. Very small internal decompositions may use a dense numerical
kernel when a truncated calculation is not meaningful, but no exact or IRLBA
solver is exposed through the public API.

`rsvd` is a stochastic approximation and remains the primary solver.
Standalone CPU `fastsvd()` calls in float32 and float64 use a native
case-specific audit with strengthened sketches and deterministic recovery when
needed. PLS fits always report their exact controls and structural diagnostics;
routes that invoke the audited decomposition also report its case-specific
outcome, while other routes state that no case certificate is available.
The controlled validation uses matrix-shape-specific automatic
controls and reports numerical agreement separately from successful execution.
For confirmatory coefficient or subspace interpretation, repeat the rSVD fit
across seeds and inspect the recorded diagnostics and prediction stability.
PLS-SVD and standalone `fastsvd()` use 32 oversampling directions and five
power iterations by default. Accelerated SIMPLS, OPLS, and kernel-PLS use
32/5 for ordinary shapes. Numeric responses with at least 64 columns and a
response-to-sample ratio of at least 0.2 use 48/6. Classification with at least
32 classes and no more than 20 samples per class uses 64/7. When the explicit
predictor-response cross-covariance would exceed 512 MiB, the massive-matrix
profile requests `oversample = 12` and `power = 1`; executed controls and the
refresh width are recorded separately in diagnostics.
CPU, Metal, and ordinary CUDA SIMPLS-family routes use seeded sketches of the
current deflated operator. For a massive cross-covariance, CPU, Metal, and CUDA
float64 use a fresh rank-one randomized direction for every component. CUDA
float32 can refresh up to eight fresh candidates at a time through component
64 before returning to component-wise calculations; its state remains device
resident. Large dummy-coded
  classification on CPU, CUDA, or Metal can instead refresh a small candidate
  block. Effective
controls are recorded in model diagnostics.
Very small SVD inputs automatically use a full dense decomposition inside the
selected compiled backend when the truncated route is not meaningful, but
`exact` is no longer exposed as a user-selectable PLS benchmark option.

CUDA backend:

- use `pls(..., backend = "cuda")` with `method = "plssvd"`, `"simpls"`,
  `"opls"`, or `"kernelpls"`.

On Linux and Windows, CUDA support is optional. If the CUDA Toolkit is not
available, the package builds CPU-only and CUDA requests give a clear runtime
error without running the requested operation on CPU. A CPU-only build is also
produced if an old environment has
`FASTPLS_USE_CUDA = "1"` but the toolkit is missing. To force a CPU-only build
on any machine:

```r
Sys.setenv(FASTPLS_USE_CUDA = "0")
remotes::install_github("tkcaccia/fastPLS", force = TRUE, upgrade = "never")
```

On Windows, if installation prints `package 'fastPLS' is in use and will not be
installed`, restart R before reinstalling. Windows keeps the loaded package DLL
locked, so `force = TRUE` cannot replace it while `library(fastPLS)` is active.
For a Windows machine without the CUDA Toolkit, use the CPU-only command above
and do not set `CUDA_ROOT`.

On an NVIDIA workstation, install the NVIDIA CUDA Toolkit and build with CUDA
enabled by setting `FASTPLS_USE_CUDA = "1"` and `CUDA_ROOT`.

Linux example:

```r
Sys.setenv(
  FASTPLS_USE_CUDA = "1",
  CUDA_ROOT = "/usr/local/cuda"
)
remotes::install_github("tkcaccia/fastPLS", force = TRUE, upgrade = "never")
```

Windows example:

```r
Sys.setenv(
  FASTPLS_USE_CUDA = "1",
  CUDA_ROOT = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6"
)
remotes::install_github("tkcaccia/fastPLS", force = TRUE, upgrade = "never")
```

After installation, `has_cuda()` reports whether the package was compiled with
CUDA support and can see a CUDA device at runtime.

Selecting or configuring `backend = "cuda"` without an available CUDA
build/device raises an error. fastPLS never silently changes a CUDA or Metal
request to CPU; `fastPLS_backend()` also rejects an unavailable configured
accelerator.

On macOS, the Metal backend is compiled automatically when the macOS SDK or
system Metal frameworks are available. The CUDA message "building without
CUDA" does not mean that Metal was disabled. A successful configuration prints
`Apple Metal backend enabled`. When the Xcode Metal compiler is available, the
installer also compiles and embeds the custom shader library so the first fit
does not compile kernels at runtime. Device and pipeline initialization still
make an isolated first call slower than repeated calls in one R session. After
restarting R, verify both compilation and runtime device access with:

```r
library(fastPLS)
has_metal()
```

Likewise, selecting `backend = "metal"` when Metal support is unavailable raises
an error; choose `backend = "cpu"` explicitly when CPU execution is intended.

To require Metal explicitly during a GitHub installation, use:

```r
Sys.setenv(FASTPLS_USE_METAL = "1")
remotes::install_github("tkcaccia/fastPLS", force = TRUE, upgrade = "never")
```

For automated CUDA build tests, set `FASTPLS_REQUIRE_CUDA = "1"` in addition to
`FASTPLS_USE_CUDA = "1"` if installation should fail when the CUDA Toolkit is
not found.

Compact low-rank prediction is integrated into the standard prediction path.
When latent factors are available, `predict.fastPLS()` applies streamed
low-rank products instead of materializing and multiplying by a full
coefficient matrix for every requested component count. This primarily reduces
prediction time and RAM pressure; fitting memory remains governed by the
selected model family and backend.

Use `backend = "cuda"` or `backend = "metal"` for supported accelerated PLS
runs. Standalone accelerator `fastsvd()` routes are rejected because their
reduced QR/SVD stage is not fully device-native for every matrix shape.
Single CV uses compiled fold construction and numerical loops. Nested CV uses
a reproducible fold plan constructed in R; CPU and Metal execute the nested
fold loops in compiled code, whereas CUDA uses an R coordinator around native
CUDA single-CV and outer-fit kernels. Final R-object assembly remains host-side.
No accelerator request silently substitutes a CPU estimator.

CUDA supports PLS-SVD, SIMPLS, OPLS, and linear, RBF, or polynomial kernel PLS
in float32 and float64. Metal supports the same families in float32 by assigning
fitting products involving the training sample matrix to persistent Metal
workspaces and reduced
factorizations, sequential PLS updates, LDA, and prediction to the CPU. This
assignment is independent of dataset shape. Unsafe nonlinear Gram sizes produce
an error rather than a CPU fallback.

## Current API

Main model fitting:

- `pls()`

Prediction and utilities:

- `predict()`
- `ViP()`
- `fastcor()`
- `has_cuda()`
- `has_metal()`
- `fastsvd()`
- `plot()` for `fastPLS` score plots with optional confidence
  or Hotelling's T2 ellipses

Cross-validation:

- `pls.single.cv()`
- `pls.double.cv()`

All lower-level C++, CUDA, OPLS, kernel PLS, SVD-dispatch, and KODAMA-oriented
helpers are internal implementation details. Benchmarks should use the same
public API as package users.

## Reproducible Benchmarks

Benchmark runners, numerical-validation studies, and the code that generates
manuscript tables and figures are maintained in the separate
`tkcaccia/fastPLS-extra` repository. This repository contains only the
installable package, its tests, and user documentation. Generated benchmark
results are not tracked here; a frozen evidence archive will be published
separately for the manuscript release.

## References

- de Jong, S. (1993). SIMPLS. *Chemometrics and Intelligent Laboratory Systems*.
- Halko, N., Martinsson, P.-G. and Tropp, J. A. (2011). Randomized algorithms
  for matrix decompositions. *SIAM Review*.
- Musco, C. and Musco, C. (2015). Randomized block Krylov methods for stronger
  and faster approximate singular value decomposition.
