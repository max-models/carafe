"""The largest fast-wave crossing rate of the cells of `carafe.mhd.MHD2D`'s block.

The Pyccel version (CPU) of the ``mhd_rate`` kernel; ``mhd_rate_numpy.py`` next to
this file is the NumPy version, with the same arguments. Called by
``MHD2D.stable_step`` for every patch.

The arguments are those of the ``mhd_primitive`` kernel (the conserved fields and
``A_z`` of a haloed block, the spacings, the mean field and the adiabatic index) and
the halo width ``halo``: only the block's own cells count. ``out`` gets the largest
``(|u| + c_x) / dx + (|v| + c_y) / dy`` (``c`` the fast magnetosonic speeds along x
and y) and the smallest density.
"""

from math import sqrt


def fast_speed(
    rho: float, p: float, bn: float, bt1: float, bt2: float, gamma: float
) -> float:
    """The fast magnetosonic speed along the normal."""
    a2 = gamma * p / rho
    b2 = (bn * bn + bt1 * bt1 + bt2 * bt2) / rho
    disc = (a2 + b2) * (a2 + b2) - 4.0 * a2 * (bn * bn) / rho
    root = 0.0
    if disc > 0.0:
        root = sqrt(disc)
    return sqrt(0.5 * (a2 + b2 + root))


def mhd_rate(
    rho: "float[:,:]",
    mx: "float[:,:]",
    my: "float[:,:]",
    mz: "float[:,:]",
    energy: "float[:,:]",
    bz: "float[:,:]",
    az: "float[:,:]",
    halo: int,
    dx: float,
    dy: float,
    b0x: float,
    b0y: float,
    gamma: float,
    out: "float[:]",
) -> None:
    """Write the largest rate and the smallest density of the block to ``out``."""
    nx = rho.shape[0] - 2 * halo
    ny = rho.shape[1] - 2 * halo
    largest = 0.0
    smallest = rho[halo, halo]
    for a in range(nx):
        i = a + halo
        for b in range(ny):
            j = b + halo
            r = rho[i, j]
            u = mx[i, j] / r
            v = my[i, j] / r
            w = mz[i, j] / r
            bx = 0.5 * (
                (az[i, j + 1] - az[i, j]) / dy
                + b0x
                + ((az[i + 1, j + 1] - az[i + 1, j]) / dy + b0x)
            )
            by = 0.5 * (
                -(az[i + 1, j] - az[i, j]) / dx
                + b0y
                + (-(az[i + 1, j + 1] - az[i, j + 1]) / dx + b0y)
            )
            bzc = bz[i, j]
            p = (gamma - 1.0) * (
                energy[i, j]
                - 0.5 * r * (u * u + v * v + w * w)
                - 0.5 * (bx * bx + by * by + bzc * bzc)
            )
            rate = (abs(u) + fast_speed(r, p, bx, by, bzc, gamma)) / dx + (
                abs(v) + fast_speed(r, p, by, bx, bzc, gamma)
            ) / dy
            if rate > largest:
                largest = rate
            if r < smallest:
                smallest = r
    out[0] = largest
    out[1] = smallest
