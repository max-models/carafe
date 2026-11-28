"""The primitive variables of `carafe.mhd.MHD2D` on a haloed block, and their check.

The Pyccel version (CPU) of the ``mhd_primitive`` kernel; ``mhd_primitive_numpy.py``
next to this file is the NumPy version, with the same arguments. Called by
``MHD2D._primitive`` for every patch.

``rho``, ``mx``, ``my``, ``mz``, ``energy`` and ``bz`` are the conserved fields of a
haloed block, ``(X, Y)``; ``az`` is ``A_z`` at its nodes, ``(X + 1, Y + 1)``. The
face fields are ``bxf = (A_z(i, j + 1) - A_z(i, j)) / dy + b0x`` at the x-faces,
``(X + 1, Y)``, and ``byf = -(A_z(i + 1, j) - A_z(i, j)) / dx + b0y`` at the y-faces,
``(X, Y + 1)``; the cell field is the mean of the two faces of a cell. ``prim`` gets
``(8, X, Y)``: density, the three velocities, the pressure and the three field
components, ``ecc`` the cell electric field ``B_x u_y - B_y u_x``. Returns the number of cells whose density or pressure is not positive
or whose primitive variables are not all finite.
"""


def finite(x: float) -> bool:
    """Whether ``x`` is neither infinite nor NaN."""
    return abs(x) <= 1.7976931348623157e308


def mhd_primitive(
    rho: "float[:,:]",
    mx: "float[:,:]",
    my: "float[:,:]",
    mz: "float[:,:]",
    energy: "float[:,:]",
    bz: "float[:,:]",
    az: "float[:,:]",
    dx: float,
    dy: float,
    b0x: float,
    b0y: float,
    gamma: float,
    prim: "float[:,:,:]",
    bxf: "float[:,:]",
    byf: "float[:,:]",
    ecc: "float[:,:]",
) -> int:
    """Write the primitive variables and face fields; return the bad cells (see the module)."""
    nx = rho.shape[0]
    ny = rho.shape[1]
    for i in range(nx + 1):
        for j in range(ny):
            bxf[i, j] = (az[i, j + 1] - az[i, j]) / dy + b0x
    for i in range(nx):
        for j in range(ny + 1):
            byf[i, j] = -(az[i + 1, j] - az[i, j]) / dx + b0y
    bad = 0
    for i in range(nx):
        for j in range(ny):
            r = rho[i, j]
            u = mx[i, j] / r
            v = my[i, j] / r
            w = mz[i, j] / r
            bx = 0.5 * (bxf[i, j] + bxf[i + 1, j])
            by = 0.5 * (byf[i, j] + byf[i, j + 1])
            b = bz[i, j]
            p = (gamma - 1.0) * (
                energy[i, j]
                - 0.5 * r * (u * u + v * v + w * w)
                - 0.5 * (bx * bx + by * by + b * b)
            )
            prim[0, i, j] = r
            prim[1, i, j] = u
            prim[2, i, j] = v
            prim[3, i, j] = w
            prim[4, i, j] = p
            prim[5, i, j] = bx
            prim[6, i, j] = by
            prim[7, i, j] = b
            ecc[i, j] = bx * v - by * u
            if not (r > 0.0 and p > 0.0):
                bad += 1
            elif not (
                finite(r)
                and finite(u)
                and finite(v)
                and finite(w)
                and finite(p)
                and finite(bx)
                and finite(by)
                and finite(b)
            ):
                bad += 1
    return bad
