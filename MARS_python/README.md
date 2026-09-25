# MARS (Python)

Pure-Python (NumPy) implementation of **MARS** — a second-order reduction
algorithm for high-dimensional **sparse precision matrix** estimation
(*l1*-penalized D-trace loss, adaptive sieving + semismooth Newton augmented
Lagrangian, SSNAL). Optional **Numba JIT** kernels make it ~3-4x faster than
the pure-NumPy baseline while staying numerically equivalent.

This package is a faithful port of the R package `MARS`
(Qian LI, Binyan Jiang, Defeng Sun,
*JMLR* 24 (2023) 1-44). Verified iteration-by-iteration against the R version.

## Install

```bash
# from this directory
pip install -e .            # core (NumPy only)
pip install -e ".[jit]"     # + Numba JIT kernels (recommended)
```

If Numba is not installed, the code automatically falls back to pure NumPy
(`USE_NUMBA = False`).

## Quick start

```python
import numpy as np
from mars import mars_path, findA, maxLambda

rng = np.random.default_rng(0)
X = rng.standard_normal((3000, 50))          # p = 3000, n = 50
lams = 0.95 * maxLambda(X)                  # data-dependent max lambda

Omegapath, Lambdapath, timepath = mars_path(
    X, np.array([lams]), stoptol=1e-4, maxiter=10, stopmethod="fix")
Omega = Omegapath[0]                        # 3000 x 3000 sparse precision matrix
```

## API

### `mars_path(X, Lambdapath, stoptol=1e-4, maxiter=10, stopmethod="fix", printyes=False, printyessub=False, sigma=1.0, maxlambdacheck=True)`

Solve the whole lambda path.

| Argument        | Description |
|-----------------|-------------|
| `X`             | `p x n` sample matrix (rows = variables, columns = samples). |
| `Lambdapath`    | 1-D array of candidate tuning parameters (decreasing). |
| `stoptol`       | Stopping tolerance (`max(primfeas,dualfeas) < 500*max(1e-6,stoptol)` and `eta < stoptol`; includes the `gap < tol` criterion). |
| `maxiter`       | Maximum outer (ALM) iterations. |
| `stopmethod`    | `"fix"`: solve every lambda; `"bigs"`: early stop when the active set saturates. |
| `maxlambdacheck`| Drop lambdas above the data-dependent max lambda. |
| `sigma`         | Initial ALM penalty parameter. |

Returns `(Omegapath, Lambdapath, timepath)`:
`Omegapath` is a list of dense symmetric `p x p` arrays (one per lambda),
`Lambdapath` the lambdas actually used, `timepath` the per-lambda wall time.

### `findA(X)`

Build `A` from the centered sample matrix (row-center + thin SVD,
`A = U diag(s)/sqrt(n-1)`).

### `maxLambda(X)` / `findmaxlambda(A)`

Data-dependent maximum lambda. `maxLambda` takes the raw sample matrix,
`findmaxlambda` the transformed `A` (both without forming the full `p x p`
Gram matrix).

### Low-level solvers

`PMEASmainc`, `PMEASSSNCGc`, `PMEASCG`, `operatorSY`, `operatorInvLA`,
`proxBmain`, `prox_b`, `partgradient`, `findstep`, `ind2sub`, `findcd`,
`vecOmega`, `updatesigma` — direct ports of the C++ internals, exposed for
experimentation and debugging.

## Numerical equivalence with R

Verified iteration-by-iteration on the prostate data (p = 6033): the
Python solver reproduces the R solver's iterates to machine precision at the
operator level; tiny deviations (`~1e-13` relative) from BLAS threading
differences can push the *low-lambda* (ill-conditioned) solution onto a
different optimal branch, as expected for non-unique solutions. Compare
**objective values** (absolute/relative error) rather than raw solutions in
that regime.

## Performance

Real prostate data (p = 6033, control n = 50 / cancer n = 52, 3-lambda path,
sequential, Windows):

| Implementation             | control | cancer |
|----------------------------|--------:|-------:|
| R + Intel MKL              | 7.27 s  | 9.26 s |
| **Python + Numba (11 JIT kernels)** | **9.24 s** | **10.87 s** |
| Python pure NumPy          | 14.75 s | ~16 s  |

The Numba kernels cover the CG/line-search inner loop as well as `gradP`
(sparse-symmetric, reusing `G = A@A.T`), `proxBmain`, triangle scanning,
max-lambda and symmetric norms. Remaining gap to R comes from the Python
interpreted adaptive-sieving outer loop (Index/sub management).

## Files

- `mars/mars_py.py` — full implementation (single module, ~700 lines).
- `mars/__init__.py` — public API.
- `benchmark.py`, `benchmark_big.py` — synthetic benchmarks.
- `Numba-optimization-report.md`, `Numba-JIT2-optimization-report.md` — JIT
  implementation details and verification.

## Reference

Qian LI, Binyan Jiang, and Defeng Sun. "MARS: A second-order reduction
algorithm for high-dimensional sparse precision matrices estimation."
*Journal of Machine Learning Research*, 24 (2023) 1-44.
