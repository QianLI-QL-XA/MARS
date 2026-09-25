# MARS

**MARS** — A second-order reduction algorithm for high-dimensional **sparse
precision matrix** estimation. The package provides both an R implementation
(`MARS`) and a Python implementation (see [Python version](#python-version)).

## What does MARS estimate?

Given `n` i.i.d. samples of a `p`-dimensional random vector with sample matrix
`X` (`p x n`, columns = samples), MARS estimates the **precision matrix**
(inverse covariance matrix) `Omega = Sigma^{-1}` under the
*l1-penalized D-trace loss*:

```text
min_{Omega in S^p}   0.5 || Omega A ||_F^2 - <Omega, I> + lambda || Omega ||_{1, off}
```

where `A` is derived from the centered sample matrix and
`||Omega||_{1, off} = sum_{i != j} |Omega_ij|`.

**Why does this matter?** For a Gaussian graphical model, the precision
matrix encodes **conditional dependence**: `Omega_ij = 0` if and only if
variables `i` and `j` are conditionally independent given all other variables.
The sparsity pattern of the estimated precision matrix is therefore a
**graph**, whose edges are the non-zero off-diagonal entries. MARS is designed
for the high-dimensional regime `p >> n` (e.g. genomics, where thousands of
genes are observed on a few dozen subjects).

The solver combines an **adaptive sieving (AS)** reduction strategy with a
**semismooth Newton augmented Lagrangian (SSNAL)** algorithm, plus a
preconditioned conjugate gradient inner solver.

## References

Qian LI, Binyan Jiang, and Defeng Sun.
["MARS: A second-order reduction algorithm for high-dimensional sparse precision matrices estimation"](https://jmlr.org/papers/v24/21-0699.html).
*Journal of Machine Learning Research*, 24 (2023) 1-44.

## Getting started (R)

### Preparation

- R (>= 3.5) with a C++ toolchain: Rtools on Windows, Xcode Command Line
  Tools on macOS, or build-essential / g++ on Linux;
- the R packages `Rcpp`, `RcppArmadillo` and `Matrix`:

```r
install.packages(c("Rcpp", "RcppArmadillo", "Matrix"))
```

### Installing MARS

From GitHub:

```r
library(devtools)
devtools::install_github("QianLI-QL/MARS")
```

Or locally, from the package source directory (where `DESCRIPTION` lives):

```r
install.packages(".", repos = NULL, type = "source")
# alternative: R CMD INSTALL .
```

### Minimal example

```r
library(MARS)
set.seed(1)
X <- matrix(runif(3000 * 50), 3000, 50)          # p = 3000, n = 50
sol <- MARS(X, stopmethod = "bigs",
            Lambdapath = seq(1, 0.5, -0.01),
            printmain = TRUE)
str(sol)
```

`MARS()` returns a list with:

- `Omegapath`: list of estimated sparse precision matrices, one per lambda
  (each a `dsCMatrix`, i.e. symmetric sparse Matrix from package `Matrix`);
- `Lambdapath`: the tuning parameters actually used;
- `timepath`: wall-clock seconds spent on each lambda.

### The `MARS()` function

```r
MARS(X, Lambdapath, stopmethod = c("bigs", "fix"), fixnumber = 1L,
     maxiter = 100L, maxlambdacheck = TRUE, printmain = FALSE,
     printsub = FALSE, stoptol = 1e-4, sigma = 1.0)
```

| Argument         | Description                                                              |
|------------------|--------------------------------------------------------------------------|
| `X`              | `p x n` sample matrix (rows = variables, columns = samples).             |
| `Lambdapath`     | Numeric vector of candidate tuning parameters (decreasing). Values above the data-dependent max-lambda are discarded when `maxlambdacheck = TRUE`. |
| `stopmethod`     | `"bigs"`: early stop when the active set saturates (`nnz > length(Lambdapath)*n` or `numAS > 4`); `"fix"`: solve every lambda in the path. |
| `fixnumber`      | With `stopmethod = "fix"`, number of leading lambdas to solve.           |
| `maxiter`        | Maximum outer (ALM) iterations.                                          |
| `maxlambdacheck` | Drop lambdas exceeding the data-dependent max-lambda.                    |
| `printmain`      | Print per-lambda progress.                                               |
| `printsub`       | Print subproblem (SSNAL) progress.                                       |
| `stoptol`        | Stopping tolerance; the solver stops once `max(primfeas, dualfeas) < 500*max(1e-6, stoptol)` **and** `eta < stoptol` (includes the `gap < tol` criterion). |
| `sigma`          | Initial ALM penalty parameter.                                           |

### Real-data example (prostate gene expression)

```r
library(MARS)
prost <- read.csv("prostmat.csv", header = FALSE)   # 6033 x 102
X1 <- as.matrix(prost[, 1:50])                      # control subjects (n = 50)
X2 <- as.matrix(prost[, 51:102])                    # cancer subjects (n = 52)
sol1 <- MARS(X1, Lambdapath = c(0.95, 0.85, 0.75, 0.65) * maxLambda(X1),
             stopmethod = "fix", maxiter = 10, stoptol = 1e-4)
sol2 <- MARS(X2, Lambdapath = c(0.95, 0.85, 0.75, 0.65) * maxLambda(X2),
             stopmethod = "fix", maxiter = 10, stoptol = 1e-4)
```

The sparsity pattern (`Omega != 0`) gives the estimated gene network; entries
`Omega_ij != 0` indicate conditional dependence between gene `i` and gene `j`
given all other genes. `maxLambda(X)` returns the data-dependent max-lambda
(the largest value for which a non-diagonal edge can enter the solution).

## Python version

A pure-Python (NumPy + optional Numba JIT) implementation of the same
algorithm ships in the repository:

```bash
pip install -e ./MARS_python
```

```python
import numpy as np
from mars import mars_path, findA, maxLambda

X = np.random.rand(3000, 50)                 # p = 3000, n = 50
lams = 0.95 * maxLambda(X)
Omegapath, Lambdapath, timepath = mars_path(X, lams, stoptol=1e-4,
                                            maxiter=10, stopmethod="fix")
```

The Python version is numerically equivalent to the R version (verified
iteration-by-iteration) and is roughly 3-4x faster than the pure-NumPy
baseline when the Numba JIT kernels are enabled. See `MARS_python/README.md`.

## Performance notes

- The R package links against the system BLAS/LAPACK. Replacing the default
  Reference BLAS with a multi-threaded BLAS (e.g. Intel MKL) can speed up the
  matrix-heavy parts 5-6x. On Windows, copy `mkl_rt.2.dll` over `Rblas.dll`
  and `Rlapack.dll` (see `MARS_matlab/MARS-R-MKL-optimization.md` for a
  worked example).
- The adaptive sieving strategy keeps the active set small in the
  high/medium-lambda regime, so memory stays low even when `p` is large.

## Notes for developers

- Core computation lives in `src/MARSc.cpp` (RcppArmadillo) and registers
  native routines via `useDynLib(MARS, .registration = TRUE)`.
- The max-lambda is computed without forming the full `p x p` Gram matrix.
- `R CMD check --as-cran` passes with no errors/warnings/notes.
