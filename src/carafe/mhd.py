r"""MHD in two dimensions on a patchwork hierarchy.

`MHD2D` solves the equations of ideal magnetohydrodynamics in a plane ``(x, y)``
with all three components of the velocity and the magnetic field (2.5D), in
Heaviside–Lorentz units with :math:`\mu_0 = 1` (the field in units of
:math:`\sqrt{\mu_0}`):

.. math::

    \partial_t\rho + \nabla\cdot(\rho\mathbf u) = 0, \qquad
    \partial_t(\rho\mathbf u) + \nabla\cdot\big(\rho\mathbf u\mathbf u
        + p_T - \mathbf B\mathbf B\big) = 0,

    \partial_t E + \nabla\cdot\big((E + p_T)\mathbf u
        - (\mathbf u\cdot\mathbf B)\mathbf B\big) = 0, \qquad
    \partial_t\mathbf B = \nabla\times(\mathbf u\times\mathbf B),

with :math:`p_T = p + |\mathbf B|^2/2` and
:math:`E = p/(\gamma - 1) + \rho|\mathbf u|^2/2 + |\mathbf B|^2/2`. With a uniform
resistivity :math:`\eta` (a magnetic diffusivity in these units) the electric field
gains :math:`\eta\mathbf J`, :math:`\mathbf J = \nabla\times\mathbf B`, and the energy flux
:math:`\eta\,\mathbf J\times\mathbf B` (the Ohmic heating stays in the energy); the
Lundquist number of a layer of length :math:`L` is :math:`S = L v_A/\eta`.

Two further optional terms:

- *Braginskii (anisotropic) heat conduction*: the energy flux gains :math:`\mathbf q =
  -\kappa_\parallel\,\mathbf b(\mathbf b\cdot\nabla T) - \kappa_\perp\nabla T` with
  :math:`\mathbf b = \mathbf B/|\mathbf B|` and :math:`T = p/\rho`, uniform
  conductivities; the gradient across a face in the parallel term is the
  monotonized-central average of its four neighbours (Sharma and Hammett 2007),
  which keeps the temperature from overshooting where the field is oblique to the
  grid;
- *gravity*: a uniform acceleration :math:`\mathbf g`, with the sources
  :math:`\rho\mathbf g` of the momentum and :math:`\rho\mathbf u\cdot\mathbf g` of the
  energy. They are not well balanced: a hydrostatic atmosphere keeps spurious
  velocities that fall at first order with the cell size (about 6e-4 at a density
  jump of two, 32 cells across the Rayleigh–Taylor box, `MHD2D.rayleigh_taylor`).

**The mesh is patchwork's.** The model owns no grid: it allocates its fields on
an `patchwork.AMRHierarchy` and uses the hierarchy's halo exchange, flux register
and restriction. A uniform grid is a hierarchy of one level; refined patches
are patches of finer levels, fixed or regridded on the current density
(`MHD2D.current_tags`, `MHD2D.regrid`); all levels take the same time step.

**The magnetic field.** The density, the momentum, the energy and :math:`B_z`
are cell-centered conserved fields. The in-plane field is the curl of a vector
potential :math:`A_z` on the nodes, plus a uniform mean field
:math:`\mathbf B_0`:

.. math::

    B_x = B_{0x} + \partial_y A_z, \qquad B_y = B_{0y} - \partial_x A_z,

taken as the flux through every cell face (the difference of :math:`A_z` at the
face's two nodes). This is the face (Raviart–Thomas) value of the field as the
rotated gradient of a nodal (Q1) potential, the 2D form of constrained transport:
the discrete divergence of every cell is zero to round-off whatever :math:`A_z`
is, and :math:`A_z` advances with the electric field at the nodes,
:math:`\partial_t A_z = -E_z`. On a refined hierarchy the nodes a coarse and a fine
patch share carry the fine value (`patchwork.AMRHierarchy.restrict`), so the flux
through a coarse face equals the sum of the fine fluxes through it, and the
divergence of the composite mesh is zero to round-off as well. :math:`\mathbf B_0`
lets a periodic domain carry a net flux, which a periodic :math:`A_z` cannot.

**Numerics.** MUSCL reconstruction of the primitive variables (the slope limiters
of `carafe.limiters`), the HLLD Riemann solver (Miyoshi and Kusano 2005) with the
face-normal field taken from :math:`A_z`, the electric field at the nodes by the
upwind "CT-contact" average of the face fluxes (Gardiner and Stone 2005), Heun's
method (SSP-RK2), and refluxing (`patchwork.FluxRegister`) at coarse–fine faces.
Where a reconstructed density or pressure is not positive the face falls back to
first order. The resistive and conductive terms are explicit (``J_z`` at the nodes from
the face fields, ``J_x``, ``J_y`` at the faces from ``B_z``), and the step keeps within
the diffusion limit of the finest cells.

**Boundaries.** Periodic axes are those the hierarchy declares periodic. The faces
of each other axis are outflow (zero-gradient) boundaries or perfectly conducting,
reflecting walls (``boundaries={"y": "wall"}``): the halo mirrors the cells with the
normal velocity reversed, :math:`A_z` is odd about its wall value so the normal field
stays zero, the electric field along the wall is zero, and nothing but the pressure
crosses it.

patchwork is an optional dependency of carafe (the ``spatial`` extra): install it to
use this module.

**Regridding.** `MHD2D.regrid` moves the conserved fields (their integrals kept) and
interpolates :math:`A_z` bilinearly onto new patches, so the field stays
divergence-free; a state can be set again on the new patches (`set_state`), e.g. to
resolve an initial current sheet the base level cannot.

**Not yet.** Viscosity, the r–z metric and graded axes (slab and cylinder geometry only
in the sense of a Cartesian plane; no toroidal geometry); MPI is parallel over patches (the fields of a patch are not split between
ranks): place the patches with ``placement="morton"`` or ``"balanced"``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
from methodcard import ConvergenceProblem, method
from scope_profiler import profile

from carafe._profiling import _PROFILE_TAGS
from carafe.datasets import variable
from carafe.limiters import LIMITERS

if TYPE_CHECKING:
    from patchwork import AMRHierarchy, Patch

__all__ = ["CONSERVED", "MHD2D", "hlld_flux"]

#: Halo width of every field: two cells for the MUSCL stencil of a face.
HALO = 2
#: The cell-centered conserved fields, in this order.
CONSERVED = ("density", "momentum_x", "momentum_y", "momentum_z", "energy", "bz")
#: The boundary kinds of a non-periodic axis.
BOUNDARIES = ("outflow", "wall")
#: Relative size below which the HLLD star-state denominator counts as zero.
_DEGENERATE = 1.0e-12

Value = float | np.ndarray | Callable[..., Any]


# The Riemann solver --------------------------------------------------------------#
def _fast_speed(rho, p, bn, bt1, bt2, gamma):
    """The fast magnetosonic speed along the normal."""
    a2 = gamma * p / rho
    b2 = (bn**2 + bt1**2 + bt2**2) / rho
    root = np.sqrt(np.maximum((a2 + b2) ** 2 - 4.0 * a2 * bn**2 / rho, 0.0))
    return np.sqrt(0.5 * (a2 + b2 + root))


def _conserved(state, bn, gamma):
    """Conserved variables (rho, m_n, m_t1, m_t2, E, B_t1, B_t2) of a normal state."""
    rho, vn, vt1, vt2, p, bt1, bt2 = state
    energy = (
        p / (gamma - 1.0)
        + 0.5 * rho * (vn**2 + vt1**2 + vt2**2)
        + 0.5 * (bn**2 + bt1**2 + bt2**2)
    )
    return np.stack((rho, rho * vn, rho * vt1, rho * vt2, energy, bt1, bt2))


def _flux(state, cons, bn):
    """The flux along the normal of a state and its conserved variables."""
    rho, vn, vt1, vt2, p, bt1, bt2 = state
    pt = p + 0.5 * (bn**2 + bt1**2 + bt2**2)
    vb = vn * bn + vt1 * bt1 + vt2 * bt2
    return np.stack(
        (
            rho * vn,
            rho * vn**2 + pt - bn**2,
            rho * vn * vt1 - bn * bt1,
            rho * vn * vt2 - bn * bt2,
            (cons[4] + pt) * vn - bn * vb,
            bt1 * vn - bn * vt1,
            bt2 * vn - bn * vt2,
        )
    )


@method(
    equations={
        "speeds": r"S_L = \min(u_L, u_R) - \max(c_{fL}, c_{fR}), \quad S_R = \max(u_L, u_R) + \max(c_{fL}, c_{fR}), \quad S_L^* = S_M - \frac{|B_n|}{\sqrt{\rho_L^*}}, \quad S_R^* = S_M + \frac{|B_n|}{\sqrt{\rho_R^*}}",
        "contact": r"S_M = \frac{(S_R - u_R)\rho_R u_R - (S_L - u_L)\rho_L u_L - p_{TR} + p_{TL}}{(S_R - u_R)\rho_R - (S_L - u_L)\rho_L}",
        "star": r"\rho^*_K = \rho_K\frac{S_K - u_K}{S_K - S_M}, \quad \mathbf v^*_{tK} = \mathbf v_{tK} - B_n\mathbf B_{tK}\frac{S_M - u_K}{\rho_K(S_K - u_K)(S_K - S_M) - B_n^2}, \quad \mathbf B^*_{tK} = \mathbf B_{tK}\frac{\rho_K(S_K - u_K)^2 - B_n^2}{\rho_K(S_K - u_K)(S_K - S_M) - B_n^2}",
        "double_star": r"\mathbf v^{**}_t = \frac{\sqrt{\rho_L^*}\mathbf v^*_{tL} + \sqrt{\rho_R^*}\mathbf v^*_{tR} + (\mathbf B^*_{tR} - \mathbf B^*_{tL})\operatorname{sgn}B_n}{\sqrt{\rho_L^*} + \sqrt{\rho_R^*}}",
    },
    assumptions=(
        "ideal MHD, a gamma-law gas; the normal field is one value on both sides of the face",
        "where the star denominator vanishes (B_t = 0 and B_n^2 = rho (S - u)(S - S_M)) the tangential velocity and field keep their values",
    ),
    references=(
        "[Miyoshi and Kusano, J. Comput. Phys. 208, 315 (2005)](https://doi.org/10.1016/j.jcp.2005.02.017)",
    ),
)
def hlld_flux(
    left: np.ndarray, right: np.ndarray, bn: np.ndarray, gamma: float
) -> np.ndarray:
    """The HLLD flux through faces between two states.

    Args:
        left: The state left of each face, ``(7, ...)``: density, normal velocity,
            the two tangential velocities, pressure and the two tangential field
            components.
        right: The state right of each face, the same way.
        bn: The normal field at each face, ``(...)``.
        gamma: The adiabatic index.

    Returns:
        The flux of (density, normal momentum, the two tangential momenta, energy,
        the two tangential field components), ``(7, ...)``.
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    bn = np.asarray(bn, dtype=float)
    ul, ur = _conserved(left, bn, gamma), _conserved(right, bn, gamma)
    fl, fr = _flux(left, ul, bn), _flux(right, ur, bn)
    rl, vl, pl = left[0], left[1], left[4]
    rr, vr, pr = right[0], right[1], right[4]
    ptl = pl + 0.5 * (bn**2 + left[5] ** 2 + left[6] ** 2)
    ptr = pr + 0.5 * (bn**2 + right[5] ** 2 + right[6] ** 2)
    cf = np.maximum(
        _fast_speed(rl, pl, bn, left[5], left[6], gamma),
        _fast_speed(rr, pr, bn, right[5], right[6], gamma),
    )
    sl = np.minimum(vl, vr) - cf
    sr = np.maximum(vl, vr) + cf
    dl, dr = (sl - vl) * rl, (sr - vr) * rr
    sm = (dr * vr - dl * vl - ptr + ptl) / (dr - dl)
    pts = (dr * ptl - dl * ptr + dl * dr * (vr - vl)) / (dr - dl)

    def star(state, cons, s, d, pt):
        vn = state[1]
        rho_s = d / (s - sm)
        den = d * (s - sm) - bn**2
        degenerate = np.abs(den) < _DEGENERATE * (pt + bn**2)
        safe = np.where(degenerate, 1.0, den)
        vt = np.where(
            degenerate, state[2:4], state[2:4] - bn * state[5:7] * (sm - vn) / safe
        )
        bt = np.where(
            degenerate, state[5:7], state[5:7] * (d * (s - vn) - bn**2) / safe
        )
        vb = vn * bn + state[2] * state[5] + state[3] * state[6]
        vb_s = sm * bn + vt[0] * bt[0] + vt[1] * bt[1]
        energy = ((s - vn) * cons[4] - pt * vn + pts * sm + bn * (vb - vb_s)) / (s - sm)
        return (
            np.stack(
                (rho_s, rho_s * sm, rho_s * vt[0], rho_s * vt[1], energy, bt[0], bt[1])
            ),
            vt,
            bt,
        )

    usl, vtl, btl = star(left, ul, sl, dl, ptl)
    usr, vtr, btr = star(right, ur, sr, dr, ptr)
    sql, sqr = np.sqrt(usl[0]), np.sqrt(usr[0])
    sign = np.sign(bn)
    vt2 = (sql * vtl + sqr * vtr + (btr - btl) * sign) / (sql + sqr)
    bt2 = (sql * btr + sqr * btl + sql * sqr * (vtr - vtl) * sign) / (sql + sqr)
    vb2 = sm * bn + vt2[0] * bt2[0] + vt2[1] * bt2[1]

    def double_star(us, vt, bt, sq, side):
        vb_s = sm * bn + vt[0] * bt[0] + vt[1] * bt[1]
        energy = us[4] + side * sq * (vb_s - vb2) * sign
        return np.stack(
            (us[0], us[0] * sm, us[0] * vt2[0], us[0] * vt2[1], energy, bt2[0], bt2[1])
        )

    uddl = double_star(usl, vtl, btl, sql, -1.0)
    uddr = double_star(usr, vtr, btr, sqr, 1.0)
    ssl = sm - np.abs(bn) / sql
    ssr = sm + np.abs(bn) / sqr
    fsl = fl + sl * (usl - ul)
    fsr = fr + sr * (usr - ur)
    return np.select(
        [sl >= 0.0, ssl >= 0.0, sm >= 0.0, ssr >= 0.0, sr >= 0.0],
        [fl, fsl, fsl + ssl * (uddl - usl), fsr + ssr * (uddr - usr), fsr],
        fr,
    )


# Convergence problem -------------------------------------------------------------#
def _alfven_problem() -> ConvergenceProblem:
    """The oblique circularly polarized Alfvén wave after one period (Tóth 2000)."""

    def error(h: float) -> float:
        cells = round(1.0 / h)
        model, exact = MHD2D.alfven_wave(cells, limiter="none")
        model.run(model.period)
        by = model.dataset()["magnetic_field_y"].values
        return float(np.mean(np.abs(by - exact["magnetic_field_y"].values)))

    return ConvergenceProblem(error, steps=[1 / 8, 1 / 16, 1 / 32])


# Helpers -------------------------------------------------------------------------#
def _ct_contact(fx_by, gy_bx, fx_rho, gy_rho, ecc, nx, ny):
    """``E_z`` at the nodes of a block by the CT-contact average (Gardiner and Stone 2005).

    ``fx_*`` are x-face fluxes of the haloed block at x-faces ``2 .. X - 2`` (array
    index ``k`` is x-face ``k + 2``) and all rows; ``gy_*`` y-face fluxes at all
    columns and y-faces ``2 .. Y - 2``; ``ecc`` the cell electric field ``B_x u_y - B_y
    u_x``. Returns ``E_z`` at the block's ``(nx + 1, ny + 1)`` nodes.
    """
    h = HALO
    # node (i, j) of the block is node (i + h, j + h) of the haloed block; the x-face
    # at column i + h is fx index i + h - 2, the y-face at row j + h is gy index j + h - 2
    ci = np.arange(nx + 1)[:, None] + h  # node columns, haloed
    rj = np.arange(ny + 1)[None, :] + h  # node rows, haloed

    def fxb(col, row):
        return fx_by[col - 2, row]

    def gyb(col, row):
        return gy_bx[col, row - 2]

    def fxr(col, row):
        return fx_rho[col - 2, row]

    def gyr(col, row):
        return gy_rho[col, row - 2]

    def upwind(mass, positive, negative):
        return np.where(
            mass > 0.0,
            positive,
            np.where(mass < 0.0, negative, 0.5 * (positive + negative)),
        )

    i, j = ci, rj
    dl2 = upwind(
        fxr(i, j - 1), gyb(i - 1, j) - ecc[i - 1, j - 1], gyb(i, j) - ecc[i, j - 1]
    )
    dr2 = upwind(fxr(i, j), gyb(i - 1, j) - ecc[i - 1, j], gyb(i, j) - ecc[i, j])
    dl1 = upwind(
        gyr(i - 1, j), -fxb(i, j - 1) - ecc[i - 1, j - 1], -fxb(i, j) - ecc[i - 1, j]
    )
    dr1 = upwind(gyr(i, j), -fxb(i, j - 1) - ecc[i, j - 1], -fxb(i, j) - ecc[i, j])
    return 0.25 * (
        gyb(i, j) + gyb(i - 1, j) - fxb(i, j) - fxb(i, j - 1) + dl1 + dr1 + dl2 + dr2
    )


def _outflow_cells(array: np.ndarray, axis: int, lower: bool, upper: bool) -> None:
    """Zero-gradient halo cells at the domain's faces along ``axis``."""
    h = HALO
    moved = np.moveaxis(array, axis, 0)
    if lower:
        moved[:h] = moved[h]
    if upper:
        moved[-h:] = moved[-h - 1]


def _outflow_nodes(array: np.ndarray, axis: int, lower: bool, upper: bool) -> None:
    """Halo nodes of ``A_z`` extrapolated linearly at the domain's faces along ``axis``.

    The field component that the gradient along ``axis`` gives (the tangential
    field of the boundary face) then continues unchanged into the halo.
    """
    h = HALO
    moved = np.moveaxis(array, axis, 0)
    if lower:
        edge, inner = moved[h].copy(), moved[h + 1].copy()
        for k in range(1, h + 1):
            moved[h - k] = edge + k * (edge - inner)
    if upper:
        edge, inner = moved[-h - 1].copy(), moved[-h - 2].copy()
        for k in range(1, h + 1):
            moved[-h - 1 + k] = edge + k * (edge - inner)


def _wall_cells(
    array: np.ndarray, axis: int, lower: bool, upper: bool, sign: float = 1.0
) -> None:
    """Mirrored halo cells at walls along ``axis``; ``sign`` -1 for the normal momentum."""
    h = HALO
    moved = np.moveaxis(array, axis, 0)
    if lower:
        for k in range(h):
            moved[h - 1 - k] = sign * moved[h + k]
    if upper:
        n = moved.shape[0]
        for k in range(h):
            moved[n - h + k] = sign * moved[n - h - 1 - k]


def _wall_nodes(array: np.ndarray, axis: int, lower: bool, upper: bool) -> None:
    """Halo nodes of ``A_z`` odd about the wall nodes: zero normal field, even tangential."""
    h = HALO
    moved = np.moveaxis(array, axis, 0)
    if lower:
        wall = moved[h].copy()
        for k in range(1, h + 1):
            moved[h - k] = 2.0 * wall - moved[h + k]
    if upper:
        n = moved.shape[0]
        wall = moved[n - h - 1].copy()
        for k in range(1, h + 1):
            moved[n - h - 1 + k] = 2.0 * wall - moved[n - h - 1 - k]


def _transverse_gradient(t: np.ndarray, axis: int, spacing: float) -> np.ndarray:
    """Limited gradient across the faces normal to ``axis`` (Sharma and Hammett 2007).

    For the faces between cells ``k - 1`` and ``k`` along ``axis`` (``k = 2 .. n -
    2``, as `MHD2D._reconstruct`), the derivative along the other axis: the
    monotonized-central limiter of the four one-sided differences of the two cells.
    Zero in the first and last row across, which no block face uses.
    """
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


# The model -----------------------------------------------------------------------#
@method(
    equations={
        "mass": r"\partial_t\rho + \nabla\cdot(\rho\mathbf u) = 0",
        "momentum": r"\partial_t(\rho\mathbf u) + \nabla\cdot\big(\rho\mathbf u\mathbf u + p_T\mathbf I - \mathbf B\mathbf B\big) = 0, \qquad p_T = p + \tfrac12|\mathbf B|^2",
        "energy": r"\partial_t E + \nabla\cdot\big((E + p_T)\mathbf u - (\mathbf u\cdot\mathbf B)\mathbf B\big) = 0, \qquad E = \frac{p}{\gamma - 1} + \tfrac12\rho|\mathbf u|^2 + \tfrac12|\mathbf B|^2",
        "field": r"B_x = B_{0x} + \partial_y A_z, \quad B_y = B_{0y} - \partial_x A_z, \quad \partial_t A_z = -E_z = u_x B_y - u_y B_x - \eta J_z, \quad \partial_t B_z + \nabla\cdot(B_z\mathbf u - u_z\mathbf B) = \eta\nabla^2 B_z",
        "resistive": r"\mathbf E \mathrel{+}= \eta\mathbf J, \quad \mathbf J = \nabla\times\mathbf B, \quad \text{energy flux} \mathrel{+}= \eta\,\mathbf J\times\mathbf B",
        "conduction": r"\mathbf q = -\kappa_\parallel\mathbf b(\mathbf b\cdot\nabla T) - \kappa_\perp\nabla T, \quad T = p/\rho, \quad \partial_y T|_{x\text{-face}} = \mathrm{MC}\big(\mathrm{MC}(\delta^+_{i-1}, \delta^-_{i-1}), \mathrm{MC}(\delta^+_i, \delta^-_i)\big)",
        "gravity": r"\partial_t(\rho\mathbf u) \mathrel{+}= \rho\mathbf g, \qquad \partial_t E \mathrel{+}= \rho\mathbf u\cdot\mathbf g",
        "walls": r"\text{mirrored halo}, \ u_n \to -u_n, \quad A_z(w - k) = 2A_z(w) - A_z(w + k), \quad E_z|_w = 0",
        "ct_contact": r"E_z = \tfrac14\big(\textstyle\sum_\text{faces} E_z^f\big) + \tfrac14\big(\textstyle\sum_\text{4 half-faces} (E_z^f - E_z^c)_\text{upwind}\big)",
    },
    order=2,
    assumptions=(
        r"MHD in the plane with three velocity and field components; Heaviside–Lorentz units ($\mu_0 = 1$); uniform resistivity $\eta$, explicit",
        r"in-plane field from a nodal $A_z$: zero discrete divergence per cell and on the composite mesh, to round-off",
        r"MUSCL on the primitives, HLLD, CT-contact electric field, Heun (SSP-RK2) at CFL 0.4; refluxing at coarse–fine faces",
        r"Cartesian, uniform levels, one time step for all levels; periodic, outflow or conducting-wall faces; regridding interpolates $A_z$ bilinearly",
        r"uniform $\kappa_\parallel$, $\kappa_\perp$ and $\mathbf g$; Sharma–Hammett limited transverse gradients; gravity not well balanced",
        r"the order is that of a smooth Alfvén wave with the unlimited reconstruction",
    ),
    references=(
        "[Gardiner and Stone, J. Comput. Phys. 205, 509 (2005)](https://doi.org/10.1016/j.jcp.2004.11.016)",
        "[Tóth, J. Comput. Phys. 161, 605 (2000)](https://doi.org/10.1006/jcph.2000.6519)",
        "[Sharma and Hammett, J. Comput. Phys. 227, 123 (2007)](https://doi.org/10.1016/j.jcp.2007.07.026)",
    ),
    components=(hlld_flux,),
    problem=_alfven_problem,
)
class MHD2D:
    """MHD (2.5D) on the patches of a patchwork hierarchy; see the module docstring.

    Args:
        hierarchy: The 2D Cartesian hierarchy the fields live on. Its periodic axes
            are periodic; the faces of the others follow ``boundaries``.
        gamma: The adiabatic index.
        mean_field: The uniform in-plane field ``(B0x, B0y)`` added to the curl of
            ``A_z``.
        resistivity: The uniform resistivity ``eta`` (a magnetic diffusivity); 0 for
            ideal MHD.
        limiter: The slope limiter of the reconstruction (`carafe.limiters`):
            ``"minmod"``, ``"vanleer"`` (default), ``"mc"`` or ``"none"``.
        cfl: The Courant number.
        conduction: The heat conductivities ``(kappa_parallel, kappa_perp)`` along
            and across the field, in units of energy flux per gradient of ``T = p /
            rho``.
        gravity: The uniform acceleration ``(g_x, g_y)``.
        boundaries: The kind of the faces of each non-periodic axis, ``{"x": ...,
            "y": ...}``: ``"outflow"`` (the default) or ``"wall"``.
        prefix: The prefix of the names of the fields the model allocates on the
            hierarchy (``prefix + "density"``, ..., ``prefix + "az"``).
    """

    def __init__(
        self,
        hierarchy: AMRHierarchy,
        *,
        gamma: float = 5.0 / 3.0,
        mean_field: tuple[float, float] = (0.0, 0.0),
        resistivity: float = 0.0,
        limiter: str = "vanleer",
        cfl: float = 0.4,
        conduction: tuple[float, float] = (0.0, 0.0),
        gravity: tuple[float, float] = (0.0, 0.0),
        boundaries: dict[str, str] | None = None,
        prefix: str = "mhd_",
    ) -> None:
        """Allocate the fields on ``hierarchy``; set the state with `set_state`."""
        if hierarchy.ndim != 2:
            raise ValueError("MHD2D needs a 2D hierarchy")
        if hierarchy.cylindrical or hierarchy.graded:
            raise NotImplementedError(
                "MHD2D is Cartesian on uniform levels: no r–z metric, no graded axes"
            )
        if limiter not in LIMITERS:
            raise ValueError(f"limiter is one of {sorted(LIMITERS)}")
        if not 0.0 < cfl <= 1.0:
            raise ValueError("cfl must be in (0, 1]")
        if gamma <= 1.0:
            raise ValueError("gamma must be larger than 1")
        if not resistivity >= 0.0:
            raise ValueError("resistivity must be non-negative")
        if not min(conduction) >= 0.0:
            raise ValueError("the conductivities must be non-negative")
        kinds = {"x": "outflow", "y": "outflow"}
        for axis_name, kind in (boundaries or {}).items():
            if axis_name not in kinds or kind not in BOUNDARIES:
                raise ValueError(
                    f"boundaries maps 'x' or 'y' to one of {BOUNDARIES}, "
                    f"got {axis_name!r}: {kind!r}"
                )
            axis = "xy".index(axis_name)
            if hierarchy.periodic[axis]:
                raise ValueError(f"the hierarchy is periodic along {axis_name}")
            if kind == "wall" and mean_field[axis] != 0.0:
                raise ValueError(
                    f"a wall normal to {axis_name} needs a zero mean field along it"
                )
            kinds[axis_name] = kind
        self.hierarchy = hierarchy
        self.gamma = float(gamma)
        self.mean_field = (float(mean_field[0]), float(mean_field[1]))
        self.resistivity = float(resistivity)
        self.limiter = limiter
        self.cfl = float(cfl)
        self.conduction = (float(conduction[0]), float(conduction[1]))
        self.gravity = (float(gravity[0]), float(gravity[1]))
        self.walls = tuple(kinds[name] == "wall" for name in "xy")
        self.names = {name: prefix + name for name in CONSERVED}
        self.potential = prefix + "az"
        for name in CONSERVED:
            hierarchy.allocate_field(
                self.names[name], halo=HALO, split=None, conservative=True
            )
        hierarchy.allocate_field(
            self.potential, halo=HALO, split=None, centering="node"
        )
        from patchwork import FluxRegister

        self.periodic = tuple(hierarchy.periodic)
        self.register = FluxRegister(hierarchy)
        self.time = 0.0
        self.steps = 0
        #: The period of the wave `alfven_wave` sets up (NaN otherwise).
        self.period = float("nan")
        self._ready = False

    # Construction of common cases ---------------------------------------------#
    @classmethod
    def alfven_wave(
        cls, cells: int, *, amplitude: float = 0.1, **options: Any
    ) -> tuple[MHD2D, xr.Dataset]:
        """The oblique circularly polarized Alfvén wave (Tóth 2000), and its exact state.

        A periodic box of ``2 cells x cells`` cells over ``[0, 2] x [0, 1]``; the
        wave travels at the angle ``atan(2)`` to ``x`` with wavelength ``2 /
        sqrt(5)``, density 1, pressure 0.1 and a parallel field of 1, so the
        Alfvén speed is 1. The exact solution after a whole number of periods
        (`period`) is the initial state, which the second item returns.
        """
        from patchwork import AMRHierarchy

        hierarchy = AMRHierarchy(
            (2 * cells, cells), ((0.0, 2.0), (0.0, 1.0)), periodic=(True, True)
        )
        angle = np.arctan2(2.0, 1.0)
        kx, ky = np.cos(angle), np.sin(angle)
        wavelength = 2.0 / np.sqrt(5.0)
        k = 2.0 * np.pi / wavelength

        def phase(x, y):
            return k * (kx * x + ky * y)

        def perp(x, y):  # the in-plane perpendicular field and velocity amplitude
            return amplitude * np.sin(phase(x, y))

        def velocity(x, y):
            b_perp = perp(x, y)
            bz = amplitude * np.cos(phase(x, y))
            # a wave travelling along +k: v_perp = -B_perp / sqrt(rho)
            return (b_perp * ky, -b_perp * kx, -bz)

        model = cls(hierarchy, mean_field=(kx, ky), **options)
        model.set_state(
            1.0,
            velocity,
            0.1,
            vector_potential=lambda x, y: amplitude / k * np.cos(phase(x, y)),
            bz=lambda x, y: amplitude * np.cos(phase(x, y)),
        )
        model.period = float(wavelength)
        exact = model.dataset()
        return model, exact

    @classmethod
    def orszag_tang(cls, cells: int, **options: Any) -> MHD2D:
        """The Orszag–Tang vortex on the periodic unit square, ``cells`` per side.

        Density ``25 / (36 pi)``, pressure ``5 / (12 pi)``, velocity
        ``(-sin 2 pi y, sin 2 pi x)``, field ``(-sin 2 pi y, sin 4 pi x) / sqrt(4 pi)``
        (Gardiner and Stone 2005), ``gamma = 5/3``.
        """
        from patchwork import AMRHierarchy

        hierarchy = AMRHierarchy(
            (cells, cells), ((0.0, 1.0), (0.0, 1.0)), periodic=(True, True)
        )
        model = cls(hierarchy, **options)
        b0 = 1.0 / np.sqrt(4.0 * np.pi)
        model.set_state(
            25.0 / (36.0 * np.pi),
            lambda x, y: (-np.sin(2 * np.pi * y), np.sin(2 * np.pi * x), 0.0 * x),
            5.0 / (12.0 * np.pi),
            vector_potential=lambda x, y: (
                b0
                * (
                    np.cos(4 * np.pi * x) / (4 * np.pi)
                    + np.cos(2 * np.pi * y) / (2 * np.pi)
                )
            ),
        )
        return model

    @classmethod
    def rayleigh_taylor(
        cls,
        cells: int,
        *,
        field: float = 0.0,
        amplitude: float = 0.01,
        **options: Any,
    ) -> MHD2D:
        """The single-mode Rayleigh–Taylor instability, ``cells x 3 cells`` cells.

        Heavy gas (density 2) on light (density 1) over ``[-0.25, 0.25] x [-0.75,
        0.75]``, periodic in ``x``, walls in ``y``, gravity ``-0.1`` along ``y``,
        hydrostatic pressure ``2.5 - 0.1 rho y``, ``gamma = 1.4``, a velocity
        perturbation ``v_y = amplitude (1 + cos 4 pi x)(1 + cos 3 pi y) / 4`` and a
        uniform field ``field`` along ``x`` (the setup of Stone and Gardiner, *ApJS*
        178, 137, 2008). A field along the interface stabilises the modes with
        ``k B^2 > (rho_h - rho_l) g`` roughly.
        """
        from patchwork import AMRHierarchy

        hierarchy = AMRHierarchy(
            (cells, 3 * cells),
            ((-0.25, 0.25), (-0.75, 0.75)),
            periodic=(True, False),
        )
        options = {"gamma": 1.4, **options}
        model = cls(
            hierarchy,
            mean_field=(field, 0.0),
            gravity=(0.0, -0.1),
            boundaries={"y": "wall"},
            **options,
        )

        def density(x, y):
            return np.where(y > 0.0, 2.0, 1.0)

        def velocity(x, y):
            vy = amplitude * (1 + np.cos(4 * np.pi * x)) * (1 + np.cos(3 * np.pi * y))
            return (0.0 * x, vy / 4.0, 0.0 * x)

        model.set_state(density, velocity, lambda x, y: 2.5 - 0.1 * density(x, y) * y)
        return model

    # State ------------------------------------------------------------------#
    @staticmethod
    def _evaluate(value: Value, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if callable(value):
            return np.broadcast_to(np.asarray(value(x, y), dtype=float), x.shape)
        return np.broadcast_to(np.asarray(value, dtype=float), x.shape)

    def _cell_field(
        self, patch: Patch, az: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centered in-plane field of a patch from its nodal ``A_z`` (no halo)."""
        dx, dy = self.hierarchy.spacing(patch.level)
        bxf = (az[:, 1:] - az[:, :-1]) / dy + self.mean_field[0]
        byf = -(az[1:, :] - az[:-1, :]) / dx + self.mean_field[1]
        return 0.5 * (bxf[:-1] + bxf[1:]), 0.5 * (byf[:, :-1] + byf[:, 1:])

    def set_state(
        self,
        density: Value,
        velocity: Value | tuple[Value, Value, Value],
        pressure: Value,
        *,
        vector_potential: Value = 0.0,
        bz: Value = 0.0,
    ) -> None:
        """Set the state on every patch.

        Each argument is a number, an array of the patch's shape (only for a
        hierarchy of one patch) or a function of the cell-centre coordinates
        ``(x, y)`` (of the node coordinates for ``vector_potential``).
        ``velocity`` is a function returning the three components, or a tuple of
        three values. The in-plane field is ``mean_field`` plus the curl of
        ``vector_potential``; ``bz`` is the out-of-plane field.
        """
        hierarchy = self.hierarchy
        gamma = self.gamma
        for patch in hierarchy.local_patches:
            x, y = patch.local_cell_centers(hierarchy, self.names["density"])
            xn, yn = patch.local_node_coordinates(hierarchy, self.potential)
            rho = self._evaluate(density, x, y)
            if callable(velocity):
                u, v, w = (np.broadcast_to(c, x.shape) for c in velocity(x, y))
            elif isinstance(velocity, tuple | list) and len(velocity) == 3:
                u, v, w = (self._evaluate(c, x, y) for c in velocity)
            else:
                raise ValueError("velocity is a function or three values")
            p = self._evaluate(pressure, x, y)
            if np.any(rho <= 0.0) or np.any(p <= 0.0):
                raise ValueError("the density and the pressure must be positive")
            az = self._evaluate(vector_potential, xn, yn)
            patch.field(self.potential).local[...] = az
            bx, by = self._cell_field(patch, az)
            bzv = self._evaluate(bz, x, y)
            values = {
                "density": rho,
                "momentum_x": rho * u,
                "momentum_y": rho * v,
                "momentum_z": rho * w,
                "energy": p / (gamma - 1.0)
                + 0.5 * rho * (u**2 + v**2 + w**2)
                + 0.5 * (bx**2 + by**2 + bzv**2),
                "bz": bzv,
            }
            for name, value in values.items():
                patch.field(self.names[name]).local[...] = value
        for name in self.names.values():
            hierarchy.restrict(name)
        hierarchy.restrict(self.potential)
        self.time = 0.0
        self.steps = 0
        self._ready = True

    def _check_ready(self) -> None:
        if not self._ready:
            raise RuntimeError("set the state first (set_state)")

    # Halos and boundaries ---------------------------------------------------#
    def _fill_halos(self) -> None:
        """Halos from the hierarchy, then the outflow faces and walls of the domain."""
        hierarchy = self.hierarchy
        for name in self.names.values():
            hierarchy.exchange_halos(name)
        hierarchy.exchange_halos(self.potential)
        for patch in hierarchy.local_patches:
            shape = hierarchy.level_shape(patch.level)
            for axis in range(2):
                if self.periodic[axis]:
                    continue
                lower = patch.box[axis][0] == 0
                upper = patch.box[axis][1] == shape[axis]
                if not (lower or upper):
                    continue
                potential = patch.field(self.potential).local_with_halos
                if not self.walls[axis]:
                    for name in self.names.values():
                        _outflow_cells(
                            patch.field(name).local_with_halos, axis, lower, upper
                        )
                    _outflow_nodes(potential, axis, lower, upper)
                    continue
                normal = ("momentum_x", "momentum_y")[axis]
                for key, name in self.names.items():
                    _wall_cells(
                        patch.field(name).local_with_halos,
                        axis,
                        lower,
                        upper,
                        -1.0 if key == normal else 1.0,
                    )
                _wall_nodes(potential, axis, lower, upper)

    # Primitive variables -----------------------------------------------------#
    def _primitive(self, patch: Patch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Primitive variables of a patch with its halo, and the face fields."""
        dx, dy = self.hierarchy.spacing(patch.level)
        cons = np.stack(
            [patch.field(self.names[name]).local_with_halos for name in CONSERVED]
        )
        az = patch.field(self.potential).local_with_halos
        bxf = (az[:, 1:] - az[:, :-1]) / dy + self.mean_field[0]
        byf = -(az[1:, :] - az[:-1, :]) / dx + self.mean_field[1]
        bx = 0.5 * (bxf[:-1] + bxf[1:])
        by = 0.5 * (byf[:, :-1] + byf[:, 1:])
        rho = cons[0]
        u, v, w = cons[1] / rho, cons[2] / rho, cons[3] / rho
        bz = cons[5]
        p = (self.gamma - 1.0) * (
            cons[4] - 0.5 * rho * (u**2 + v**2 + w**2) - 0.5 * (bx**2 + by**2 + bz**2)
        )
        prim = np.stack((rho, u, v, w, p, bx, by, bz))
        return prim, bxf, byf

    # The right-hand side -------------------------------------------------------#
    def _reconstruct(self, q: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
        """Left and right states at the faces along ``axis`` (array axis 1 + axis).

        For ``n`` cells along the axis, the faces between cells ``k - 1`` and
        ``k`` for ``k = 2 .. n - 2``.
        """
        limit = LIMITERS[self.limiter]
        a = 1 + axis

        def cut(start, stop):
            index = [slice(None)] * q.ndim
            index[a] = slice(start, stop)
            return q[tuple(index)]

        n = q.shape[a]
        diff = np.diff(q, axis=a)  # diff k is q[k + 1] - q[k]
        index = [slice(None)] * q.ndim
        index[a] = slice(0, n - 3)
        backward = diff[tuple(index)]
        index[a] = slice(1, n - 2)
        forward = diff[tuple(index)]
        index[a] = slice(2, n - 1)
        further = diff[tuple(index)]
        slope_left = limit(backward, forward)  # cells 1 .. n - 3
        slope_right = limit(forward, further)  # cells 2 .. n - 2
        left = cut(1, n - 2) + 0.5 * slope_left
        right = cut(2, n - 1) - 0.5 * slope_right
        # first order where a reconstructed density or pressure is not positive
        bad = (
            (left[0] <= 0.0) | (left[4] <= 0.0) | (right[0] <= 0.0) | (right[4] <= 0.0)
        )
        if np.any(bad):
            left = np.where(bad, cut(1, n - 2), left)
            right = np.where(bad, cut(2, n - 1), right)
        return left, right

    def _current(self, bxf: np.ndarray, byf: np.ndarray, level: int) -> np.ndarray:
        """``J_z`` at the inner nodes of a haloed block: node ``(I, J)`` at ``[I - 1, J - 1]``."""
        dx, dy = self.hierarchy.spacing(level)
        return (byf[1:, 1:-1] - byf[:-1, 1:-1]) / dx - (
            bxf[1:-1, 1:] - bxf[1:-1, :-1]
        ) / dy

    def _resistive(self, prim, bxf, byf, patch: Patch) -> dict[str, Any]:
        """The resistive parts of ``E_z`` at the block's nodes and of the face fluxes.

        ``eta J_z`` at the nodes; the flux of ``B_z`` gains ``eta J_y`` along x and
        ``-eta J_x`` along y (``B_z`` diffuses), the energy flux ``eta J x B``.
        """
        eta = self.resistivity
        h = HALO
        nx, ny = patch.shape
        dx, dy = self.hierarchy.spacing(patch.level)
        _, _, _, _, _, bx, by, bz = prim
        jz = self._current(bxf, byf, patch.level)
        # x-faces I = h .. h + nx between cells I - 1 and I, rows h .. h + ny - 1
        right, left, rows = slice(h, h + nx + 1), slice(h - 1, h + nx), slice(h, h + ny)
        jy = -(bz[right, rows] - bz[left, rows]) / dx
        bz_x = 0.5 * (bz[right, rows] + bz[left, rows])
        by_x = 0.5 * (by[right, rows] + by[left, rows])
        jz_x = 0.5 * (
            jz[h - 1 : h + nx, h - 1 : h + ny - 1] + jz[h - 1 : h + nx, h : h + ny]
        )
        # y-faces J = h .. h + ny between cells J - 1 and J, columns h .. h + nx - 1
        upper, lower, cols = (
            slice(h, h + ny + 1),
            slice(h - 1, h + ny),
            slice(h, h + nx),
        )
        jx = (bz[cols, upper] - bz[cols, lower]) / dy
        bz_y = 0.5 * (bz[cols, upper] + bz[cols, lower])
        bx_y = 0.5 * (bx[cols, upper] + bx[cols, lower])
        jz_y = 0.5 * (
            jz[h - 1 : h + nx - 1, h - 1 : h + ny] + jz[h : h + nx, h - 1 : h + ny]
        )
        return {
            "ez": eta * jz[h - 1 : h + nx, h - 1 : h + ny],
            "x": {"bz": eta * jy, "energy": eta * (jy * bz_x - jz_x * by_x)},
            "y": {"bz": -eta * jx, "energy": eta * (jz_y * bx_y - jx * bz_y)},
        }

    def _wall_sides(self, patch: Patch) -> tuple[tuple[bool, bool], ...]:
        """Whether the lower and upper faces of the patch along each axis are walls."""
        shape = self.hierarchy.level_shape(patch.level)
        return tuple(
            (
                self.walls[axis] and patch.box[axis][0] == 0,
                self.walls[axis] and patch.box[axis][1] == shape[axis],
            )
            for axis in range(2)
        )

    def _conductive(self, prim, bxf, byf, fx, fy, dx, dy) -> None:
        """Add the Braginskii heat flux to the energy fluxes ``fx`` and ``fy``."""
        parallel, perpendicular = self.conduction
        t = prim[4] / prim[0]
        bx, by, bz = prim[5], prim[6], prim[7]
        tiny = np.finfo(float).tiny
        # x-faces 2 .. X - 2
        bn = bxf[2:-2, :]
        bt = 0.5 * (by[1:-2] + by[2:-1])
        bzf = 0.5 * (bz[1:-2] + bz[2:-1])
        magnitude = np.sqrt(bn**2 + bt**2 + bzf**2)
        b_n, b_t = bn / np.maximum(magnitude, tiny), bt / np.maximum(magnitude, tiny)
        normal = (t[2:-1] - t[1:-2]) / dx
        across = _transverse_gradient(t, 0, dy)
        fx[4] -= parallel * b_n * (b_n * normal + b_t * across) + perpendicular * normal
        # y-faces 2 .. Y - 2
        bn = byf[:, 2:-2]
        bt = 0.5 * (bx[:, 1:-2] + bx[:, 2:-1])
        bzf = 0.5 * (bz[:, 1:-2] + bz[:, 2:-1])
        magnitude = np.sqrt(bn**2 + bt**2 + bzf**2)
        b_n, b_t = bn / np.maximum(magnitude, tiny), bt / np.maximum(magnitude, tiny)
        normal = (t[:, 2:-1] - t[:, 1:-2]) / dy
        across = _transverse_gradient(t, 1, dx)
        fy[4] -= parallel * b_n * (b_n * normal + b_t * across) + perpendicular * normal

    def _patch_fluxes(self, patch: Patch) -> dict[str, Any]:
        """The face fluxes of a patch's cells and the electric field at its nodes."""
        prim, bxf, byf = self._primitive(patch)
        if (
            np.any(prim[0] <= 0.0)
            or np.any(prim[4] <= 0.0)
            or not np.all(np.isfinite(prim))
        ):
            raise FloatingPointError(
                f"non-positive density or pressure on the level-{patch.level} patch "
                f"{patch.box} at t = {self.time}"
            )
        nx, ny = patch.shape
        h = HALO
        gamma = self.gamma
        rho, u, v, w, p, bx, by, bz = prim
        # x-sweep: normal x, tangential (y, z)
        qx = np.stack((rho, u, v, w, p, by, bz))
        left, right = self._reconstruct(qx, 0)
        fx = hlld_flux(left, right, bxf[2:-2, :], gamma)  # faces 2 .. X - 2, all rows
        # y-sweep: normal y, tangential (x, z)
        qy = np.stack((rho, v, u, w, p, bx, bz))
        left, right = self._reconstruct(qy, 1)
        fy = hlld_flux(
            left, right, byf[:, 2:-2], gamma
        )  # all columns, faces 2 .. Y - 2
        # fx[k] at x-face k + 2 of the haloed block, i.e. block face k + 2 - h
        ecc = bx * v - by * u
        fx_by, gy_bx = fx[5], fy[5]  # x-flux of B_y = -E_z, y-flux of B_x = +E_z
        fx_rho, gy_rho = fx[0], fy[0]
        ez = _ct_contact(fx_by, gy_bx, fx_rho, gy_rho, ecc, nx, ny)
        resistive = self._resistive(prim, bxf, byf, patch) if self.resistivity else None
        if resistive is not None:
            ez = ez + resistive["ez"]
        if max(self.conduction) > 0.0:
            dx, dy = self.hierarchy.spacing(patch.level)
            self._conductive(prim, bxf, byf, fx, fy, dx, dy)
        walls = self._wall_sides(patch)
        if walls[0][0]:
            ez[0, :] = 0.0
        if walls[0][1]:
            ez[-1, :] = 0.0
        if walls[1][0]:
            ez[:, 0] = 0.0
        if walls[1][1]:
            ez[:, -1] = 0.0
        sx = slice(h, h + ny)  # block rows (x-faces)
        sy = slice(h, h + nx)  # block columns (y-faces)
        xf = slice(0, nx + 1)  # block x-faces 0 .. nx: fx index k = face + h - 2
        yf = slice(0, ny + 1)
        flux_x = {
            "density": fx[0, xf, sx],
            "momentum_x": fx[1, xf, sx],
            "momentum_y": fx[2, xf, sx],
            "momentum_z": fx[3, xf, sx],
            "energy": fx[4, xf, sx],
            "bz": fx[6, xf, sx],
        }
        flux_y = {
            "density": fy[0, sy, yf],
            "momentum_x": fy[2, sy, yf],
            "momentum_y": fy[1, sy, yf],
            "momentum_z": fy[3, sy, yf],
            "energy": fy[4, sy, yf],
            "bz": fy[6, sy, yf],
        }
        if resistive is not None:
            for name in ("bz", "energy"):
                flux_x[name] = flux_x[name] + resistive["x"][name]
                flux_y[name] = flux_y[name] + resistive["y"][name]
        # nothing but the pressure crosses a wall
        tangential = ("momentum_y", "momentum_x")
        for axis, flux in enumerate((flux_x, flux_y)):
            for side, at in ((0, 0), (1, -1)):
                if not walls[axis][side]:
                    continue
                for name in ("density", "energy", "bz", tangential[axis], "momentum_z"):
                    part = flux[name] = flux[name].copy()
                    if axis == 0:
                        part[at, :] = 0.0
                    else:
                        part[:, at] = 0.0
        return {"x": flux_x, "y": flux_y, "ez": ez}

    def _advance(self, dt: float, scale: float) -> None:
        """One forward-Euler stage of every patch, recorded with ``scale``."""
        hierarchy = self.hierarchy
        self._fill_halos()
        fluxes = {
            id(patch): self._patch_fluxes(patch) for patch in hierarchy.local_patches
        }
        gx, gy = self.gravity
        sources = {}
        if gx or gy:
            for patch in hierarchy.local_patches:
                rho = patch.field(self.names["density"]).local.copy()
                mx = patch.field(self.names["momentum_x"]).local
                my = patch.field(self.names["momentum_y"]).local
                sources[id(patch)] = (rho * gx, rho * gy, (mx * gx + my * gy).copy())
        for name in CONSERVED:
            self.register.update(
                self.names[name],
                lambda patch, name=name: (
                    fluxes[id(patch)]["x"][name],
                    fluxes[id(patch)]["y"][name],
                ),
                dt,
                scale=scale,
            )
        for patch in hierarchy.local_patches:
            patch.field(self.potential).local[...] -= dt * fluxes[id(patch)]["ez"]
            if gx or gy:
                force_x, force_y, work = sources[id(patch)]
                patch.field(self.names["momentum_x"]).local[...] += dt * force_x
                patch.field(self.names["momentum_y"]).local[...] += dt * force_y
                patch.field(self.names["energy"]).local[...] += dt * work

    # Time stepping -------------------------------------------------------------#
    def stable_step(self) -> float:
        """The largest stable step: ``cfl`` times the fast-wave crossing time of a cell.

        With resistivity or conduction, also ``cfl`` times the explicit diffusion
        limit ``1 / (2 D (dx^-2 + dy^-2))`` of the finest cells, ``D`` the largest of
        ``eta`` and ``(gamma - 1) kappa / rho``.
        """
        self._check_ready()
        hierarchy = self.hierarchy
        smallest = np.inf
        for patch in hierarchy.local_patches:
            dx, dy = hierarchy.spacing(patch.level)
            az = patch.field(self.potential).local
            bx, by = self._cell_field(patch, az)
            cons = [patch.field(self.names[name]).local for name in CONSERVED]
            rho = cons[0]
            u, v, w = cons[1] / rho, cons[2] / rho, cons[3] / rho
            bz = cons[5]
            p = (self.gamma - 1.0) * (
                cons[4]
                - 0.5 * rho * (u**2 + v**2 + w**2)
                - 0.5 * (bx**2 + by**2 + bz**2)
            )
            cx = _fast_speed(rho, p, bx, by, bz, self.gamma)
            cy = _fast_speed(rho, p, by, bx, bz, self.gamma)
            rate = np.max((np.abs(u) + cx) / dx + (np.abs(v) + cy) / dy)
            smallest = min(smallest, 1.0 / rate)
            diffusivity = max(
                self.resistivity,
                (self.gamma - 1.0) * max(self.conduction) / float(np.min(rho)),
            )
            if diffusivity > 0.0:
                smallest = min(smallest, 0.5 / (diffusivity * (dx**-2 + dy**-2)))
        from maybempi import MPI

        return self.cfl * float(hierarchy.comm.allreduce(smallest, op=MPI.MIN))

    @profile("carafe.mhd_step", tags=_PROFILE_TAGS)
    def step(self, dt: float | None = None) -> float:
        """Advance by one step of Heun's method (SSP-RK2); return the step taken."""
        self._check_ready()
        if dt is None:
            dt = self.stable_step()
        hierarchy = self.hierarchy
        names = [*self.names.values(), self.potential]
        start = {
            (id(patch), name): patch.field(name).local.copy()
            for patch in hierarchy.local_patches
            for name in names
        }
        self._advance(dt, 0.5 * dt)
        hierarchy.restrict(self.potential)
        self._advance(dt, 0.5 * dt)
        for patch in hierarchy.local_patches:
            for name in names:
                array = patch.field(name).local
                array[...] = 0.5 * (array + start[(id(patch), name)])
        for name in self.names.values():
            self.register.reflux(name)
        hierarchy.restrict(self.potential)
        self.time += dt
        self.steps += 1
        return dt

    @profile("carafe.mhd_run", tags=_PROFILE_TAGS)
    def run(self, duration: float, *, max_steps: int = 1_000_000) -> int:
        """Advance by ``duration`` (the last step shortened to land on it); return the steps."""
        self._check_ready()
        if not np.isfinite(duration) or duration < 0.0:
            raise ValueError("duration must be a finite, non-negative time")
        end = self.time + duration
        taken = 0
        while self.time < end * (1.0 - 1e-14) and taken < max_steps:
            dt = min(self.stable_step(), end - self.time)
            self.step(dt)
            taken += 1
        return taken

    # Refinement ----------------------------------------------------------------#
    def current_tags(self, threshold: float, levels: int) -> dict[int, np.ndarray]:
        """Tag the cells where ``|J_z|`` times the cell size exceeds ``threshold``.

        That is the change of the in-plane field across a cell. Levels ``0 .. levels -
        1`` that exist are tagged (so at most ``levels`` refined levels, one more per
        regrid), from the patches this rank holds; the result is for `regrid`.
        """
        self._check_ready()
        hierarchy = self.hierarchy
        self._fill_halos()
        h = HALO
        tags = {
            level: np.zeros(hierarchy.level_shape(level), dtype=bool)
            for level in range(min(levels, hierarchy.max_level + 1))
        }
        for patch in hierarchy.local_patches:
            if patch.level not in tags:
                continue
            nx, ny = patch.shape
            dx, dy = hierarchy.spacing(patch.level)
            _, bxf, byf = self._primitive(patch)
            jz = np.abs(self._current(bxf, byf, patch.level))
            # cell (C, R) has the nodes C, C + 1 and R, R + 1: jz index minus one
            cell = 0.25 * sum(
                jz[h - 1 + a : h - 1 + a + nx, h - 1 + b : h - 1 + b + ny]
                for a in (0, 1)
                for b in (0, 1)
            )
            (i0, i1), (j0, j1) = patch.box
            tags[patch.level][i0:i1, j0:j1] |= cell * max(dx, dy) > threshold
        return tags

    def regrid(self, tags: dict[int, Any], **options: Any) -> None:
        """Regrid on ``tags`` (`patchwork.AMRHierarchy.regrid`, which takes ``options``).

        The conserved fields keep their integrals, ``A_z`` is interpolated (the field
        stays divergence-free); the flux register is rebuilt for the new hierarchy.
        """
        self._check_ready()
        from patchwork import FluxRegister

        hierarchy = self.hierarchy.regrid(tags, **options)
        hierarchy.restrict(self.potential)
        self.hierarchy = hierarchy
        self.register = FluxRegister(hierarchy)

    # Diagnostics ---------------------------------------------------------------#
    def leaf_values(self) -> dict[str, np.ndarray]:
        """The primitive variables on the leaf cells of the hierarchy (collective).

        One value per leaf cell, in the order of `patchwork.CompositeCells`: ``x``,
        ``y`` (the centres), ``level``, the variables of `dataset` and ``current_z``
        (the mean of the corners).
        """
        self._check_ready()
        from patchwork import CompositeCells

        hierarchy = self.hierarchy
        cells = CompositeCells(hierarchy)
        h = HALO
        names = (
            "density",
            "velocity_x",
            "velocity_y",
            "velocity_z",
            "pressure",
            "magnetic_field_x",
            "magnetic_field_y",
            "magnetic_field_z",
            "current_z",
        )
        result = {name: np.empty(cells.num_cells) for name in names}
        self._fill_halos()
        for number, patch in enumerate(hierarchy.patches):
            mine = cells.patch == number
            values = None
            if patch.is_local:
                nx, ny = patch.shape
                prim, bxf, byf = self._primitive(patch)
                jz = self._current(bxf, byf, patch.level)
                corner = 0.25 * sum(
                    jz[h - 1 + a : h - 1 + a + nx, h - 1 + b : h - 1 + b + ny]
                    for a in (0, 1)
                    for b in (0, 1)
                )
                block = (slice(h, h + nx), slice(h, h + ny))
                values = np.stack([*(q[block] for q in prim), corner])
            # the patch's ranks hold it whole (split=None): share it from its first
            values = hierarchy.comm.bcast(values, root=patch.ranks[0])
            origin = np.array([lo for lo, _ in patch.box])
            index = tuple((cells.index[mine] - origin).T)
            for name, value in zip(names, values, strict=True):
                result[name][mine] = value[index]
        levels = np.array([patch.level for patch in hierarchy.patches])
        return {
            "x": cells.centers[:, 0],
            "y": cells.centers[:, 1],
            "level": levels[cells.patch],
            **result,
        }

    # Diagnostics, per level ------------------------------------------------------#
    def divergence(self) -> float:
        """The largest ``|div B|`` times the cell size over the leaf cells of all patches.

        The net flux out of each cell over its perimeter: zero to round-off by
        construction, on each patch and (`composite_divergence`) across patches.
        """
        self._check_ready()
        largest = 0.0
        for patch in self.hierarchy.local_patches:
            dx, dy = self.hierarchy.spacing(patch.level)
            az = patch.field(self.potential).local
            bxf = (az[:, 1:] - az[:, :-1]) / dy + self.mean_field[0]
            byf = -(az[1:, :] - az[:-1, :]) / dx + self.mean_field[1]
            div = (bxf[1:] - bxf[:-1]) / dx + (byf[:, 1:] - byf[:, :-1]) / dy
            largest = max(largest, float(np.max(np.abs(div))) * min(dx, dy))
        from maybempi import MPI

        return float(self.hierarchy.comm.allreduce(largest, op=MPI.MAX))

    def composite_divergence(self) -> float:
        """The largest net magnetic flux out of a coarse leaf cell next to a finer patch.

        The flux through a coarse face is the difference of ``A_z`` at its two
        nodes; the fine faces over it add up to the difference at the same
        nodes, which the fine patch shares (restricted). Returned relative to
        the largest face flux: zero to round-off.
        """
        self._check_ready()
        hierarchy = self.hierarchy
        ratio = hierarchy.refinement_ratio
        worst, scale = 0.0, 0.0
        for fine in hierarchy.patches:
            if fine.level == 0:
                continue
            coarse_patches = hierarchy.patches_at_level(fine.level - 1)
            fine_az = hierarchy.gather(fine, self.potential)
            scale = max(scale, float(np.max(np.abs(np.diff(fine_az, axis=0)))))
            scale = max(scale, float(np.max(np.abs(np.diff(fine_az, axis=1)))))
            for coarse in coarse_patches:
                coarse_az = hierarchy.gather(coarse, self.potential)
                # the coarse nodes on the fine patch's boundary
                (fx0, fx1), (fy0, fy1) = fine.box
                for i in range(fx0 // ratio, fx1 // ratio + 1):
                    for j in range(fy0 // ratio, fy1 // ratio + 1):
                        on_edge = i in (fx0 // ratio, fx1 // ratio) or j in (
                            fy0 // ratio,
                            fy1 // ratio,
                        )
                        ci, cj = i - coarse.box[0][0], j - coarse.box[1][0]
                        if not on_edge or not (
                            0 <= ci < coarse_az.shape[0]
                            and 0 <= cj < coarse_az.shape[1]
                        ):
                            continue
                        fi, fj = i * ratio - fx0, j * ratio - fy0
                        worst = max(worst, abs(coarse_az[ci, cj] - fine_az[fi, fj]))
        return worst / scale if scale else 0.0

    def totals(self) -> dict[str, float]:
        """Integrals of the conserved fields over the domain, from the finest cells."""
        self._check_ready()
        return {name: self.hierarchy.integral(self.names[name]) for name in CONSERVED}

    def dataset(self, level: int = 0) -> xr.Dataset:
        """The primitive variables of ``level`` as a dataset over its cell centres.

        The patches of the level are gathered into one array of the level's
        shape, NaN where the level has no patch. Collective.
        """
        self._check_ready()
        hierarchy = self.hierarchy
        nx, ny = hierarchy.level_shape(level)
        dx, dy = hierarchy.spacing(level)
        x = hierarchy.lower_bound[0] + (np.arange(nx) + 0.5) * dx
        y = hierarchy.lower_bound[1] + (np.arange(ny) + 0.5) * dy
        fields = {name: np.full((nx, ny), np.nan) for name in (*CONSERVED, "bx", "by")}
        for patch in hierarchy.patches_at_level(level):
            (i0, i1), (j0, j1) = patch.box
            for name in CONSERVED:
                fields[name][i0:i1, j0:j1] = hierarchy.gather(patch, self.names[name])
            az = hierarchy.gather(patch, self.potential)
            bx, by = self._cell_field(patch, az)
            fields["bx"][i0:i1, j0:j1] = bx
            fields["by"][i0:i1, j0:j1] = by
        rho = fields["density"]
        u = fields["momentum_x"] / rho
        v = fields["momentum_y"] / rho
        w = fields["momentum_z"] / rho
        b2 = fields["bx"] ** 2 + fields["by"] ** 2 + fields["bz"] ** 2
        p = (self.gamma - 1.0) * (
            fields["energy"] - 0.5 * rho * (u**2 + v**2 + w**2) - 0.5 * b2
        )
        dims = ("x", "y")
        data = {
            "density": variable(dims, rho, r"$\rho$"),
            "velocity_x": variable(dims, u, r"$u_x$"),
            "velocity_y": variable(dims, v, r"$u_y$"),
            "velocity_z": variable(dims, w, r"$u_z$"),
            "pressure": variable(dims, p, r"$p$"),
            "magnetic_field_x": variable(dims, fields["bx"], r"$B_x$"),
            "magnetic_field_y": variable(dims, fields["by"], r"$B_y$"),
            "magnetic_field_z": variable(dims, fields["bz"], r"$B_z$"),
            "magnetic_pressure": variable(dims, 0.5 * b2, r"$|\mathbf{B}|^2/2$"),
        }
        coords = {
            "x": variable("x", x, r"$x$"),
            "y": variable("y", y, r"$y$"),
        }
        return xr.Dataset(
            data,
            coords=coords,
            attrs={"time": self.time, "level": level, "gamma": self.gamma},
        )
