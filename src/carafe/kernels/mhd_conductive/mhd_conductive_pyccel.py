"""The Braginskii heat flux of `carafe.mhd.MHD2D` through the faces of a block.

The Pyccel version (CPU) of the ``mhd_conductive`` kernel; ``mhd_conductive_numpy.py``
next to this file is the NumPy version, with the same arguments. Called by
``MHD2D._patch_fluxes`` for every patch when a conductivity is not zero.

``prim``, ``bxf`` and ``byf`` are the primitive variables and face fields of a haloed
block ``(X, Y)`` (the ``mhd_primitive`` kernel); ``fx`` and ``fy`` its face fluxes (the
``mhd_sweep`` kernel: x-face ``k + 2`` between cells ``k + 1`` and ``k + 2`` at ``fx[:,
k]``, the y-faces alike at ``fy[:, :, k]``). The energy fluxes lose ``q_n = parallel
b_n (b_n dT/dn + b_t dT/dt) + perpendicular dT/dn`` with ``T = p / rho`` and ``b`` the
unit field at the face. ``dT/dn`` is the difference across the face; ``dT/dt``, along
the face, is the monotonized-central limiter of the limited gradients of the two cells
(Sharma and Hammett 2007), zero in the first and last line of the block.
"""

from math import sqrt

import numpy as np


def mc(a: float, b: float) -> float:
    """The monotonized-central limiter of two differences."""
    slope = 0.0
    if a * b > 0.0:
        central = 0.5 * (a + b)
        limit = 2.0 * b
        if abs(a) < abs(b):
            limit = 2.0 * a
        size = min(abs(central), abs(limit))
        if central > 0.0:
            slope = size
        elif central < 0.0:
            slope = -size
    return slope


def mhd_conductive(
    prim: "float[:,:,:]",
    bxf: "float[:,:]",
    byf: "float[:,:]",
    dx: float,
    dy: float,
    parallel: float,
    perpendicular: float,
    fx: "float[:,:,:]",
    fy: "float[:,:,:]",
) -> None:
    """Subtract the heat flux from the energy fluxes ``fx[4]`` and ``fy[4]``."""
    nx = prim.shape[1]
    ny = prim.shape[2]
    tiny = 2.2250738585072014e-308
    t = np.empty((nx, ny))
    across = np.zeros((nx, ny))  # the limited gradient of each cell along the faces
    for i in range(nx):
        for j in range(ny):
            t[i, j] = prim[4, i, j] / prim[0, i, j]
    # x-faces: the gradient along y
    for i in range(nx):
        for j in range(1, ny - 1):
            across[i, j] = mc(
                (t[i, j + 1] - t[i, j]) / dy, (t[i, j] - t[i, j - 1]) / dy
            )
    for k in range(nx - 3):
        for j in range(ny):
            bn = bxf[k + 2, j]
            bt = 0.5 * (prim[6, k + 1, j] + prim[6, k + 2, j])
            bzf = 0.5 * (prim[7, k + 1, j] + prim[7, k + 2, j])
            magnitude = max(sqrt(bn * bn + bt * bt + bzf * bzf), tiny)
            b_n = bn / magnitude
            b_t = bt / magnitude
            normal = (t[k + 2, j] - t[k + 1, j]) / dx
            transverse = mc(across[k + 1, j], across[k + 2, j])
            fx[4, k, j] = fx[4, k, j] - (
                parallel * b_n * (b_n * normal + b_t * transverse)
                + perpendicular * normal
            )
    # y-faces: the gradient along x
    for i in range(nx):
        for j in range(ny):
            across[i, j] = 0.0
    for i in range(1, nx - 1):
        for j in range(ny):
            across[i, j] = mc(
                (t[i + 1, j] - t[i, j]) / dx, (t[i, j] - t[i - 1, j]) / dx
            )
    for i in range(nx):
        for k in range(ny - 3):
            bn = byf[i, k + 2]
            bt = 0.5 * (prim[5, i, k + 1] + prim[5, i, k + 2])
            bzf = 0.5 * (prim[7, i, k + 1] + prim[7, i, k + 2])
            magnitude = max(sqrt(bn * bn + bt * bt + bzf * bzf), tiny)
            b_n = bn / magnitude
            b_t = bt / magnitude
            normal = (t[i, k + 2] - t[i, k + 1]) / dy
            transverse = mc(across[i, k + 1], across[i, k + 2])
            fy[4, i, k] = fy[4, i, k] - (
                parallel * b_n * (b_n * normal + b_t * transverse)
                + perpendicular * normal
            )
