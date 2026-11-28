"""The NumPy version of the ``mhd_sweep`` kernel, run when Pyccel is unavailable.

Same arguments and arithmetic as ``mhd_sweep_pyccel.py`` (see there), vectorized over
the block with `carafe.limiters` and `carafe.mhd.hlld_flux`.
"""

from __future__ import annotations

import numpy as np

from carafe.kernels.mhd_sweep import LIMITER_CODES
from carafe.limiters import LIMITERS


def mhd_sweep(
    prim: np.ndarray,
    bn: np.ndarray,
    axis: int,
    limiter: int,
    gamma: float,
    flux: np.ndarray,
) -> None:
    """Write the face fluxes along ``axis`` of the block to ``flux``."""
    from carafe.mhd import hlld_flux

    limit = LIMITERS[LIMITER_CODES[limiter]]
    q = prim[[0, 1 + axis, 2 - axis, 3, 4, 6 - axis, 7]]
    a = 1 + axis
    n = q.shape[a]

    def cut(start, stop):
        index = [slice(None)] * q.ndim
        index[a] = slice(start, stop)
        return q[tuple(index)]

    diff = np.diff(q, axis=a)  # diff k is q[k + 1] - q[k]
    index = [slice(None)] * q.ndim
    index[a] = slice(0, n - 3)
    backward = diff[tuple(index)]
    index[a] = slice(1, n - 2)
    forward = diff[tuple(index)]
    index[a] = slice(2, n - 1)
    further = diff[tuple(index)]
    left = cut(1, n - 2) + 0.5 * limit(backward, forward)  # cells 1 .. n - 3
    right = cut(2, n - 1) - 0.5 * limit(forward, further)  # cells 2 .. n - 2
    # first order where a reconstructed density or pressure is not positive
    bad = (left[0] <= 0.0) | (left[4] <= 0.0) | (right[0] <= 0.0) | (right[4] <= 0.0)
    if np.any(bad):
        left = np.where(bad, cut(1, n - 2), left)
        right = np.where(bad, cut(2, n - 1), right)
    normal = bn[2:-2, :] if axis == 0 else bn[:, 2:-2]
    flux[...] = hlld_flux(left, right, normal, gamma)
