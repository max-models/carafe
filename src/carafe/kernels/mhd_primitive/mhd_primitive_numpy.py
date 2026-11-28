"""The NumPy version of the ``mhd_primitive`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``mhd_primitive_pyccel.py`` (see there), vectorized
over the block.
"""

from __future__ import annotations

import numpy as np


def mhd_primitive(
    rho: np.ndarray,
    mx: np.ndarray,
    my: np.ndarray,
    mz: np.ndarray,
    energy: np.ndarray,
    bz: np.ndarray,
    az: np.ndarray,
    dx: float,
    dy: float,
    b0x: float,
    b0y: float,
    gamma: float,
    prim: np.ndarray,
    bxf: np.ndarray,
    byf: np.ndarray,
    ecc: np.ndarray,
) -> int:
    """Write the primitive variables and face fields; return the bad cells."""
    bxf[...] = (az[:, 1:] - az[:, :-1]) / dy + b0x
    byf[...] = -(az[1:, :] - az[:-1, :]) / dx + b0y
    bx = 0.5 * (bxf[:-1] + bxf[1:])
    by = 0.5 * (byf[:, :-1] + byf[:, 1:])
    u, v, w = mx / rho, my / rho, mz / rho
    p = (gamma - 1.0) * (
        energy - 0.5 * rho * (u**2 + v**2 + w**2) - 0.5 * (bx**2 + by**2 + bz**2)
    )
    for row, value in enumerate((rho, u, v, w, p, bx, by, bz)):
        prim[row] = value
    ecc[...] = bx * v - by * u
    bad = ~((rho > 0.0) & (p > 0.0)) | ~np.all(np.isfinite(prim), axis=0)
    return int(np.count_nonzero(bad))
