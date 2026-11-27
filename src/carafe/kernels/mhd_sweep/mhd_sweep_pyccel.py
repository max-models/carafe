"""MUSCL reconstruction and HLLD fluxes of `carafe.mhd.MHD2D` along one axis of a block.

The Pyccel version (CPU) of the ``mhd_sweep`` kernel; ``mhd_sweep_numpy.py`` next to
this file is the NumPy version, with the same arguments. Called by
``MHD2D._patch_fluxes`` once per axis of every patch.

``prim`` holds the primitive variables of a haloed block, ``(8, X, Y)``: density, the
three velocities, the pressure and the three field components (cell averages). Along
``axis`` (0 for x, 1 for y) with ``n`` cells, the faces between cells ``k + 1`` and
``k + 2`` for ``k = 0 .. n - 4`` get the HLLD flux (Miyoshi and Kusano 2005) of the
states reconstructed from both sides with the limiter ``limiter`` (0 ``minmod``, 1
``vanleer``, 2 ``mc``, 3 ``none``); where a reconstructed density or pressure is not
positive the face falls back to first order. ``bn`` is the face-normal field of the
faces along the axis (``(X + 1, Y)`` or ``(X, Y + 1)``), face ``k + 2`` of it is face
``k`` here. ``flux`` gets ``(7, n - 3, Y)`` or ``(7, X, n - 3)``: the flux of
(density, normal momentum, the two tangential momenta, energy, the two tangential
field components), the tangential directions being (y, z) along x and (x, z) along y.
"""

from math import sqrt

import numpy as np


def limited_slope(a: float, b: float, limiter: int) -> float:
    """The limited slope of a cell from its backward and forward differences."""
    slope = 0.0
    if limiter == 3:
        slope = 0.5 * (a + b)
    elif a * b > 0.0:
        if limiter == 0:
            slope = b
            if abs(a) < abs(b):
                slope = a
        elif limiter == 1:
            product = a * b
            slope = 2.0 * product / (a + b)
        else:
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


def fast_speed(
    rho: float, p: float, bn: float, bt1: float, bt2: float, gamma: float
) -> float:
    """The fast magnetosonic speed along the normal."""
    a2 = gamma * p / rho
    b2 = (bn * bn + bt1 * bt1 + bt2 * bt2) / rho
    disc = (a2 + b2) * (a2 + b2) - 4.0 * a2 * bn * bn / rho
    root = 0.0
    if disc > 0.0:
        root = sqrt(disc)
    return sqrt(0.5 * (a2 + b2 + root))


def side_state(
    w: "float[:,:]", k: int, bn: float, gamma: float, work: "float[:,:]", u: int, f: int
) -> float:
    """Conserved variables (row ``u`` of ``work``) and flux (row ``f``) of state ``k``.

    Returns the total pressure.
    """
    rho = w[0, k]
    vn = w[1, k]
    vt1 = w[2, k]
    vt2 = w[3, k]
    p = w[4, k]
    bt1 = w[5, k]
    bt2 = w[6, k]
    magnetic = 0.5 * (bn * bn + bt1 * bt1 + bt2 * bt2)
    energy = (
        p / (gamma - 1.0) + 0.5 * rho * (vn * vn + vt1 * vt1 + vt2 * vt2) + magnetic
    )
    pt = p + magnetic
    vb = vn * bn + vt1 * bt1 + vt2 * bt2
    work[u, 0] = rho
    work[u, 1] = rho * vn
    work[u, 2] = rho * vt1
    work[u, 3] = rho * vt2
    work[u, 4] = energy
    work[u, 5] = bt1
    work[u, 6] = bt2
    work[f, 0] = rho * vn
    work[f, 1] = rho * vn * vn + pt - bn * bn
    work[f, 2] = rho * vn * vt1 - bn * bt1
    work[f, 3] = rho * vn * vt2 - bn * bt2
    work[f, 4] = (energy + pt) * vn - bn * vb
    work[f, 5] = bt1 * vn - bn * vt1
    work[f, 6] = bt2 * vn - bn * vt2
    return pt


def star_state(
    w: "float[:,:]",
    k: int,
    bn: float,
    s: float,
    d: float,
    pt: float,
    sm: float,
    pts: float,
    work: "float[:,:]",
    u: int,
    us: int,
    vt: int,
) -> None:
    """The HLLD star state (row ``us`` of ``work``) of state ``k`` beside the contact.

    Row ``u`` holds the state's conserved variables; the star tangential velocity and
    field go to ``work[vt, 0:2]`` and ``work[vt, 2:4]``.
    """
    vn = w[1, k]
    rho_s = d / (s - sm)
    den = d * (s - sm) - bn * bn
    vt1 = w[2, k]
    vt2 = w[3, k]
    bt1 = w[5, k]
    bt2 = w[6, k]
    if abs(den) >= 1.0e-12 * (pt + bn * bn):
        vt1 = w[2, k] - bn * w[5, k] * (sm - vn) / den
        vt2 = w[3, k] - bn * w[6, k] * (sm - vn) / den
        bt1 = w[5, k] * (d * (s - vn) - bn * bn) / den
        bt2 = w[6, k] * (d * (s - vn) - bn * bn) / den
    vb = vn * bn + w[2, k] * w[5, k] + w[3, k] * w[6, k]
    vb_s = sm * bn + vt1 * bt1 + vt2 * bt2
    energy = ((s - vn) * work[u, 4] - pt * vn + pts * sm + bn * (vb - vb_s)) / (s - sm)
    work[us, 0] = rho_s
    work[us, 1] = rho_s * sm
    work[us, 2] = rho_s * vt1
    work[us, 3] = rho_s * vt2
    work[us, 4] = energy
    work[us, 5] = bt1
    work[us, 6] = bt2
    work[vt, 0] = vt1
    work[vt, 1] = vt2
    work[vt, 2] = bt1
    work[vt, 3] = bt2


def hlld_face(
    left: "float[:,:]",
    right: "float[:,:]",
    k: int,
    bn: float,
    gamma: float,
    work: "float[:,:]",
    flux: "float[:,:]",
) -> None:
    """The HLLD flux of face ``k`` between ``left[:, k]`` and ``right[:, k]`` into ``flux[:, k]``.

    ``work`` is scratch space of shape ``(10, 7)``: rows 0-3 the conserved variables
    and fluxes of both sides, 4-5 the star states, 6-7 the star tangential velocity
    and field, 8-9 the double-star states.
    """
    ptl = side_state(left, k, bn, gamma, work, 0, 1)
    ptr = side_state(right, k, bn, gamma, work, 2, 3)
    rl = left[0, k]
    vl = left[1, k]
    rr = right[0, k]
    vr = right[1, k]
    cf = max(
        fast_speed(rl, left[4, k], bn, left[5, k], left[6, k], gamma),
        fast_speed(rr, right[4, k], bn, right[5, k], right[6, k], gamma),
    )
    sl = min(vl, vr) - cf
    sr = max(vl, vr) + cf
    if sl >= 0.0:
        for c in range(7):
            flux[c, k] = work[1, c]
        return
    if sr < 0.0:
        for c in range(7):
            flux[c, k] = work[3, c]
        return
    dl = (sl - vl) * rl
    dr = (sr - vr) * rr
    sm = (dr * vr - dl * vl - ptr + ptl) / (dr - dl)
    pts = (dr * ptl - dl * ptr + dl * dr * (vr - vl)) / (dr - dl)
    star_state(left, k, bn, sl, dl, ptl, sm, pts, work, 0, 4, 6)
    star_state(right, k, bn, sr, dr, ptr, sm, pts, work, 2, 5, 7)
    sql = sqrt(work[4, 0])
    sqr = sqrt(work[5, 0])
    sign = 0.0
    if bn > 0.0:
        sign = 1.0
    elif bn < 0.0:
        sign = -1.0
    vt2_1 = (sql * work[6, 0] + sqr * work[7, 0] + (work[7, 2] - work[6, 2]) * sign) / (
        sql + sqr
    )
    vt2_2 = (sql * work[6, 1] + sqr * work[7, 1] + (work[7, 3] - work[6, 3]) * sign) / (
        sql + sqr
    )
    bt2_1 = (
        sql * work[7, 2]
        + sqr * work[6, 2]
        + sql * sqr * (work[7, 0] - work[6, 0]) * sign
    ) / (sql + sqr)
    bt2_2 = (
        sql * work[7, 3]
        + sqr * work[6, 3]
        + sql * sqr * (work[7, 1] - work[6, 1]) * sign
    ) / (sql + sqr)
    vb2 = sm * bn + vt2_1 * bt2_1 + vt2_2 * bt2_2
    ssl = sm - abs(bn) / sql
    ssr = sm + abs(bn) / sqr
    for side in range(2):
        us = 4 + side
        vt = 6 + side
        sq = sql
        sense = -1.0
        if side == 1:
            sq = sqr
            sense = 1.0
        vb_s = sm * bn + work[vt, 0] * work[vt, 2] + work[vt, 1] * work[vt, 3]
        rho_s = work[us, 0]
        work[8 + side, 0] = rho_s
        work[8 + side, 1] = rho_s * sm
        work[8 + side, 2] = rho_s * vt2_1
        work[8 + side, 3] = rho_s * vt2_2
        work[8 + side, 4] = work[us, 4] + sense * sq * (vb_s - vb2) * sign
        work[8 + side, 5] = bt2_1
        work[8 + side, 6] = bt2_2
    if ssl >= 0.0:
        for c in range(7):
            flux[c, k] = work[1, c] + sl * (work[4, c] - work[0, c])
    elif sm >= 0.0:
        for c in range(7):
            flux[c, k] = (
                work[1, c]
                + sl * (work[4, c] - work[0, c])
                + ssl * (work[8, c] - work[4, c])
            )
    elif ssr >= 0.0:
        for c in range(7):
            flux[c, k] = (
                work[3, c]
                + sr * (work[5, c] - work[2, c])
                + ssr * (work[9, c] - work[5, c])
            )
    else:
        for c in range(7):
            flux[c, k] = work[3, c] + sr * (work[5, c] - work[2, c])


def mhd_sweep(
    prim: "float[:,:,:]",
    bn: "float[:,:]",
    axis: int,
    limiter: int,
    gamma: float,
    flux: "float[:,:,:]",
) -> None:
    """Write the face fluxes along ``axis`` of the block to ``flux`` (see the module)."""
    n = prim.shape[1 + axis]
    lines = prim.shape[2 - axis]
    faces = n - 3
    # the primitive variables in normal form: density, normal velocity, the two
    # tangential velocities, pressure, the two tangential field components
    component = np.empty(7, dtype=np.int64)
    component[0] = 0
    component[1] = 1 + axis
    component[2] = 2 - axis
    component[3] = 3
    component[4] = 4
    component[5] = 6 - axis
    component[6] = 7
    q = np.empty((7, n))
    slope = np.zeros((7, n))
    left = np.empty((7, faces))
    right = np.empty((7, faces))
    face_flux = np.empty((7, faces))
    work = np.empty((10, 7))
    for line in range(lines):
        for c in range(7):
            if axis == 0:
                for s in range(n):
                    q[c, s] = prim[component[c], s, line]
            else:
                for s in range(n):
                    q[c, s] = prim[component[c], line, s]
            for s in range(1, n - 1):
                slope[c, s] = limited_slope(
                    q[c, s] - q[c, s - 1], q[c, s + 1] - q[c, s], limiter
                )
            for k in range(faces):
                left[c, k] = q[c, k + 1] + 0.5 * slope[c, k + 1]
                right[c, k] = q[c, k + 2] - 0.5 * slope[c, k + 2]
        for k in range(faces):
            # first order where a reconstructed density or pressure is not positive
            if (
                left[0, k] <= 0.0
                or left[4, k] <= 0.0
                or right[0, k] <= 0.0
                or right[4, k] <= 0.0
            ):
                for c in range(7):
                    left[c, k] = q[c, k + 1]
                    right[c, k] = q[c, k + 2]
            if axis == 0:
                normal = bn[k + 2, line]
            else:
                normal = bn[line, k + 2]
            hlld_face(left, right, k, normal, gamma, work, face_flux)
        for c in range(7):
            if axis == 0:
                for k in range(faces):
                    flux[c, k, line] = face_flux[c, k]
            else:
                for k in range(faces):
                    flux[c, line, k] = face_flux[c, k]
