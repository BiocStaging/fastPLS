# fastPLS

`fastPLS` provides compiled partial least-squares methods for regression and
classification. This page covers installation from GitHub on macOS, Windows,
Ubuntu, and Fedora. The package vignette documents models and usage.

## R package installer

Install `remotes` once if it is not already available:

```r
install.packages("remotes")
```

After installing the operating-system requirements below, install fastPLS in a
fresh R session:

```r
remotes::install_github(
    "tkcaccia/fastPLS",
    upgrade = "never",
    force = TRUE,
    build_vignettes = TRUE
)
```

## macOS

Install Apple's command-line developer tools:

```sh
xcode-select --install
```

The normal macOS build uses Apple Accelerate for CPU matrix operations and
enables Metal automatically when the required system frameworks are available.
No separate OpenBLAS installation is required or recommended on macOS.

To test OpenBLAS instead of Accelerate, install it with Homebrew:

```sh
brew install openblas pkg-config
```

Then install from a fresh R session:

```r
Sys.setenv(
    FASTPLS_USE_OPENBLAS = "1",
    OPENBLAS_ROOT = system("brew --prefix openblas", intern = TRUE)
)
remotes::install_github(
    "tkcaccia/fastPLS",
    upgrade = "never",
    force = TRUE,
    build_vignettes = TRUE
)
```

## Ubuntu

Install the compiler toolchain and OpenBLAS development files:

```sh
sudo apt update
sudo apt install build-essential gfortran pkg-config libopenblas-dev
```

Require OpenBLAS during installation so a missing library cannot silently use
the BLAS supplied by R:

```r
Sys.setenv(FASTPLS_USE_OPENBLAS = "1")
remotes::install_github(
    "tkcaccia/fastPLS",
    upgrade = "never",
    force = TRUE,
    build_vignettes = TRUE
)
```

## Fedora

Install the compiler toolchain and OpenBLAS development files:

```sh
sudo dnf install gcc gcc-c++ gcc-gfortran make pkgconf-pkg-config openblas-devel
```

Then require OpenBLAS when installing:

```r
Sys.setenv(FASTPLS_USE_OPENBLAS = "1")
remotes::install_github(
    "tkcaccia/fastPLS",
    upgrade = "never",
    force = TRUE,
    build_vignettes = TRUE
)
```

## Windows

Install the version of Rtools matching the installed R version. A standard
CPU-only installation can then be performed from a fresh R session with the R
command shown under **R package installer**.

OpenBLAS is optional. For x86-64 Windows, install MSYS2 and run the following
command in its **UCRT64** terminal:

```sh
pacman -S --needed mingw-w64-ucrt-x86_64-openblas
```

The usual MSYS2 location is `C:/msys64/ucrt64`. Point fastPLS to that static
OpenBLAS installation:

```r
Sys.setenv(
    FASTPLS_USE_OPENBLAS = "1",
    OPENBLAS_ROOT = "C:/msys64/ucrt64"
)
remotes::install_github(
    "tkcaccia/fastPLS",
    upgrade = "never",
    force = TRUE,
    build_vignettes = TRUE
)
```

Restart R before reinstalling an existing Windows build because Windows cannot
replace a package DLL while it is loaded.

## Verify the installation

```r
library(fastPLS)
fastPLS_blas()
has_cuda()
has_metal()
```

`fastPLS_blas()` returns `"Accelerate"`, `"OpenBLAS"`, or
`"R BLAS/LAPACK"`. Linux and Windows performance runs should verify that it
returns `"OpenBLAS"`. If OpenBLAS is not installed, fastPLS remains installable
and uses the BLAS/LAPACK supplied by R unless `FASTPLS_USE_OPENBLAS=1` was set.

CUDA is optional on Linux and Windows. A CUDA build additionally requires the
NVIDIA CUDA Toolkit and `CUDA_ROOT`; Metal is available only on macOS. Requests
for an unavailable accelerator return an error and never silently use the CPU.
