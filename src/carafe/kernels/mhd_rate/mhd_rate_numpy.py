"""The NumPy version of the ``mhd_rate`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``mhd_rate_pyccel.py`` (see there), vectorized over
the block.
"""

from __future__ import annotations

import numpy as np


def _fast_speed(rho, p, bn, bt1, bt2, gamma):
    a2 = gamma * p / rho
    b2 = (bn**2 + bt1**2 + bt2**2) / rho
    root = np.sqrt(np.maximum((a2 + b2) ** 2 - 4.0 * a2 * bn**2 / rho, 0.0))
    return np.sqrt(0.5 * (a2 + b2 + root))


def mhd_rate(
    rho: np.ndarray,
    mx: np.ndarray,
    my: np.ndarray,
    mz: np.ndarray,
    energy: np.ndarray,
    bz: np.ndarray,
    az: np.ndarray,
    halo: int,
    dx: float,
    dy: float,
    b0x: float,
    b0y: float,
    gamma: float,
    out: np.ndarray,
) -> None:
    """Write the largest rate and the smallest density of the block to ``out``."""
    h = halo
    block = (slice(h, rho.shape[0] - h), slice(h, rho.shape[1] - h))
    nodes = az[h : az.shape[0] - h, h : az.shape[1] - h]
    bxf = (nodes[:, 1:] - nodes[:, :-1]) / dy + b0x
    byf = -(nodes[1:, :] - nodes[:-1, :]) / dx + b0y
    bx = 0.5 * (bxf[:-1] + bxf[1:])
    by = 0.5 * (byf[:, :-1] + byf[:, 1:])
    r = rho[block]
    u, v, w = mx[block] / r, my[block] / r, mz[block] / r
    b = bz[block]
    p = (gamma - 1.0) * (
        energy[block] - 0.5 * r * (u**2 + v**2 + w**2) - 0.5 * (bx**2 + by**2 + b**2)
    )
    cx = _fast_speed(r, p, bx, by, b, gamma)
    cy = _fast_speed(r, p, by, bx, b, gamma)
    out[0] = np.max((np.abs(u) + cx) / dx + (np.abs(v) + cy) / dy)
    out[1] = np.min(r)
