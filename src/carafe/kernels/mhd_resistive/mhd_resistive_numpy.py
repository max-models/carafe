"""The NumPy version of the ``mhd_resistive`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``mhd_resistive_pyccel.py`` (see there), vectorized
over the block.
"""

from __future__ import annotations

import numpy as np


def mhd_resistive(
    prim: np.ndarray,
    bxf: np.ndarray,
    byf: np.ndarray,
    dx: float,
    dy: float,
    eta: float,
    fx: np.ndarray,
    fy: np.ndarray,
    ez: np.ndarray,
) -> None:
    """Add the resistive parts to ``fx``, ``fy`` and ``ez``."""
    h = 2
    nx, ny = prim.shape[1] - 2 * h, prim.shape[2] - 2 * h
    bx, by, bz = prim[5], prim[6], prim[7]
    # J_z at the inner nodes: node (I, J) at [I - 1, J - 1]
    jz = (byf[1:, 1:-1] - byf[:-1, 1:-1]) / dx - (bxf[1:-1, 1:] - bxf[1:-1, :-1]) / dy
    # x-faces I = h .. h + nx between cells I - 1 and I, rows h .. h + ny - 1
    right, left, rows = slice(h, h + nx + 1), slice(h - 1, h + nx), slice(h, h + ny)
    jy = -(bz[right, rows] - bz[left, rows]) / dx
    bz_x = 0.5 * (bz[right, rows] + bz[left, rows])
    by_x = 0.5 * (by[right, rows] + by[left, rows])
    jz_x = 0.5 * (
        jz[h - 1 : h + nx, h - 1 : h + ny - 1] + jz[h - 1 : h + nx, h : h + ny]
    )
    fx[6, : nx + 1, rows] += eta * jy
    fx[4, : nx + 1, rows] += eta * (jy * bz_x - jz_x * by_x)
    # y-faces J = h .. h + ny between cells J - 1 and J, columns h .. h + nx - 1
    upper, lower, cols = slice(h, h + ny + 1), slice(h - 1, h + ny), slice(h, h + nx)
    jx = (bz[cols, upper] - bz[cols, lower]) / dy
    bz_y = 0.5 * (bz[cols, upper] + bz[cols, lower])
    bx_y = 0.5 * (bx[cols, upper] + bx[cols, lower])
    jz_y = 0.5 * (
        jz[h - 1 : h + nx - 1, h - 1 : h + ny] + jz[h : h + nx, h - 1 : h + ny]
    )
    fy[6, cols, : ny + 1] += -eta * jx
    fy[4, cols, : ny + 1] += eta * (jz_y * bx_y - jx * bz_y)
    ez += eta * jz[h - 1 : h + nx, h - 1 : h + ny]
