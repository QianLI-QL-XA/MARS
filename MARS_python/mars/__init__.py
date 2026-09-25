# -*- coding: utf-8 -*-
"""MARS: sparse precision matrix estimation (Python package).

Port of the R package ``MARS`` (Qian LI, Binyan Jiang, Defeng Sun,
JMLR 24 (2023) 1-44): *l1*-penalized D-trace loss solved by
adaptive sieving + semismooth Newton augmented Lagrangian (SSNAL) + PCG.

Public API
----------
- :func:`mars_path`  : solve the whole lambda path (main entry).
- :func:`findA`      : build the transformed sample matrix A.
- :func:`findmaxlambda` / :func:`maxLambda` : data-dependent max lambda.
- :func:`MARSmainc` : solve a single lambda on a given active set.
"""
from .mars_py import (
    mars_path,
    findA,
    findmaxlambda,
    maxLambda,
    MARSmainc,
    MARSSSNCGc,
    MARSCG,
    operatorSY,
    operatorInvLA,
    proxBmain,
    prox_b,
    partgradient,
    findstep,
    ind2sub,
    findcd,
    vecOmega,
    updatesigma,
    USE_NUMBA,
)

__version__ = "0.2.0"

__all__ = [
    "mars_path",
    "findA",
    "findmaxlambda",
    "maxLambda",
    "MARSmainc",
    "MARSSSNCGc",
    "MARSCG",
    "operatorSY",
    "operatorInvLA",
    "proxBmain",
    "prox_b",
    "partgradient",
    "findstep",
    "ind2sub",
    "findcd",
    "vecOmega",
    "updatesigma",
    "USE_NUMBA",
    "__version__",
]
