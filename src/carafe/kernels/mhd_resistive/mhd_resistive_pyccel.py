"""The resistive terms of `carafe.mhd.MHD2D` at the faces and nodes of a block.

The Pyccel version (CPU) of the ``mhd_resistive`` kernel; ``mhd_resistive_numpy.py``
next to this file is the NumPy version, with the same arguments. Called by
``MHD2D._patch_fluxes`` for every patch when the resistivity ``eta`` is not zero.

``prim``, ``bxf`` and ``byf`` are the primitive variables and face fields of a haloed
block (the ``mhd_primitive`` kernel) with a halo of two; ``fx`` and ``fy`` its face
fluxes (the ``mhd_sweep`` kernel: x-face ``k + 2`` at ``fx[:, k]``, y-face ``k + 2`` at
``fy[:, :, k]``) and ``ez`` the electric field at the block's ``(nx + 1, ny + 1)``
nodes. ``J_z`` at node ``(I, J)`` of the haloed block is ``(byf(I, J) - byf(I - 1,
J)) / dx - (bxf(I, J) - bxf(I, J - 1)) / dy``, ``J_y = -dB_z/dx`` at the x-faces and
``J_x = dB_z/dy`` at the y-faces. On the block's faces the flux of ``B_z`` gains
``eta J_y`` along x and ``-eta J_x`` along y (``B_z`` diffuses), the energy flux
``eta J x B``; ``ez`` gains ``eta J_z``.
"""

import numpy as np


def mhd_resistive(
    prim: "float[:,:,:]",
    bxf: "float[:,:]",
    byf: "float[:,:]",
    dx: float,
    dy: float,
    eta: float,
    fx: "float[:,:,:]",
    fy: "float[:,:,:]",
    ez: "float[:,:]",
) -> None:
    """Add the resistive parts to ``fx``, ``fy`` and ``ez`` (see the module)."""
    h = 2
    nx = prim.shape[1] - 2 * h
    ny = prim.shape[2] - 2 * h
    # J_z at the nodes of the haloed block: node (I, J) at jz[I, J], I, J >= 1
    jz = np.zeros((nx + 2 * h, ny + 2 * h))
    for i in range(1, nx + 2 * h):
        for j in range(1, ny + 2 * h):
            jz[i, j] = (byf[i, j] - byf[i - 1, j]) / dx - (
                bxf[i, j] - bxf[i, j - 1]
            ) / dy
    # x-faces I = h + f between cells I - 1 and I, rows J = h .. h + ny - 1
    for f in range(nx + 1):
        i = h + f
        for b in range(ny):
            j = h + b
            jy = -(prim[7, i, j] - prim[7, i - 1, j]) / dx
            bz_x = 0.5 * (prim[7, i, j] + prim[7, i - 1, j])
            by_x = 0.5 * (prim[6, i, j] + prim[6, i - 1, j])
            jz_x = 0.5 * (jz[i, j] + jz[i, j + 1])
            fx[6, f, j] = fx[6, f, j] + eta * jy
            fx[4, f, j] = fx[4, f, j] + eta * (jy * bz_x - jz_x * by_x)
    # y-faces J = h + f between cells J - 1 and J, columns I = h .. h + nx - 1
    for a in range(nx):
        i = h + a
        for f in range(ny + 1):
            j = h + f
            jx = (prim[7, i, j] - prim[7, i, j - 1]) / dy
            bz_y = 0.5 * (prim[7, i, j] + prim[7, i, j - 1])
            bx_y = 0.5 * (prim[5, i, j] + prim[5, i, j - 1])
            jz_y = 0.5 * (jz[i, j] + jz[i + 1, j])
            fy[6, i, f] = fy[6, i, f] + -eta * jx
            fy[4, i, f] = fy[4, i, f] + eta * (jz_y * bx_y - jx * bz_y)
    for a in range(nx + 1):
        for b in range(ny + 1):
            ez[a, b] = ez[a, b] + eta * jz[h + a, h + b]
