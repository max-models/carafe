"""The NumPy version of the ``mhd_conductive`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``mhd_conductive_pyccel.py`` (see there), vectorized
over the block with `carafe.limiters`.
"""

from __future__ import annotations

import numpy as np

from carafe.limiters import LIMITERS


def _transverse_gradient(t: np.ndarray, axis: int, spacing: float) -> np.ndarray:
    """The limited gradient along the faces normal to ``axis`` (Sharma and Hammett 2007)."""
    mc = LIMITERS["mc"]
    other = 1 - axis
    moved = np.moveaxis(t, other, 0)  # the transverse axis first
    forward = np.zeros_like(moved)
    backward = np.zeros_like(moved)
    forward[1:-1] = (moved[2:] - moved[1:-1]) / spacing
    backward[1:-1] = (moved[1:-1] - moved[:-2]) / spacing
    per_cell = np.moveaxis(mc(forward, backward), 0, other)
    index_low = [slice(None)] * 2
    index_high = [slice(None)] * 2
    n = t.shape[axis]
    index_low[axis] = slice(1, n - 2)
    index_high[axis] = slice(2, n - 1)
    return mc(per_cell[tuple(index_low)], per_cell[tuple(index_high)])


def mhd_conductive(
    prim: np.ndarray,
    bxf: np.ndarray,
    byf: np.ndarray,
    dx: float,
    dy: float,
    parallel: float,
    perpendicular: float,
    fx: np.ndarray,
    fy: np.ndarray,
) -> None:
    """Subtract the heat flux from the energy fluxes ``fx[4]`` and ``fy[4]``."""
    t = prim[4] / prim[0]
    bx, by, bz = prim[5], prim[6], prim[7]
    tiny = np.finfo(float).tiny
    # x-faces 2 .. X - 2
    bn = bxf[2:-2, :]
    bt = 0.5 * (by[1:-2] + by[2:-1])
    bzf = 0.5 * (bz[1:-2] + bz[2:-1])
    magnitude = np.maximum(np.sqrt(bn**2 + bt**2 + bzf**2), tiny)
    b_n, b_t = bn / magnitude, bt / magnitude
    normal = (t[2:-1] - t[1:-2]) / dx
    across = _transverse_gradient(t, 0, dy)
    fx[4] -= parallel * b_n * (b_n * normal + b_t * across) + perpendicular * normal
    # y-faces 2 .. Y - 2
    bn = byf[:, 2:-2]
    bt = 0.5 * (bx[:, 1:-2] + bx[:, 2:-1])
    bzf = 0.5 * (bz[:, 1:-2] + bz[:, 2:-1])
    magnitude = np.maximum(np.sqrt(bn**2 + bt**2 + bzf**2), tiny)
    b_n, b_t = bn / magnitude, bt / magnitude
    normal = (t[:, 2:-1] - t[:, 1:-2]) / dy
    across = _transverse_gradient(t, 1, dx)
    fy[4] -= parallel * b_n * (b_n * normal + b_t * across) + perpendicular * normal
