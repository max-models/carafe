"""The NumPy version of the ``ct_contact`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``ct_contact_pyccel.py`` (see there), vectorized over
the nodes of the block.
"""

from __future__ import annotations

import numpy as np


def _upwind(mass, positive, negative):
    return np.where(
        mass > 0.0,
        positive,
        np.where(mass < 0.0, negative, 0.5 * (positive + negative)),
    )


def ct_contact(
    fx_by: np.ndarray,
    gy_bx: np.ndarray,
    fx_rho: np.ndarray,
    gy_rho: np.ndarray,
    ecc: np.ndarray,
    ez: np.ndarray,
) -> None:
    """Write ``E_z`` at the block's nodes to ``ez``."""
    i = np.arange(ez.shape[0])[:, None] + 2  # node columns of the haloed block
    j = np.arange(ez.shape[1])[None, :] + 2  # node rows
    fxb_lower, fxb_upper = fx_by[i - 2, j - 1], fx_by[i - 2, j]
    gyb_left, gyb_right = gy_bx[i - 1, j - 2], gy_bx[i, j - 2]
    dl2 = _upwind(
        fx_rho[i - 2, j - 1], gyb_left - ecc[i - 1, j - 1], gyb_right - ecc[i, j - 1]
    )
    dr2 = _upwind(fx_rho[i - 2, j], gyb_left - ecc[i - 1, j], gyb_right - ecc[i, j])
    dl1 = _upwind(
        gy_rho[i - 1, j - 2], -fxb_lower - ecc[i - 1, j - 1], -fxb_upper - ecc[i - 1, j]
    )
    dr1 = _upwind(gy_rho[i, j - 2], -fxb_lower - ecc[i, j - 1], -fxb_upper - ecc[i, j])
    ez[...] = 0.25 * (
        gyb_right + gyb_left - fxb_upper - fxb_lower + dl1 + dr1 + dl2 + dr2
    )
