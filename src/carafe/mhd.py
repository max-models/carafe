r"""Ideal and resistive MHD in two dimensions on a patchwork hierarchy.

`MHD2D` solves the compressible, single-fluid MHD equations in a periodic box of the
plane, with all three components of the velocity and of the field, in units with
$\mu_0 = 1$ (the field in units of $\sqrt{\mu_0}$, so the magnetic pressure is $B^2/2$):

$$
\partial_t\rho + \nabla\cdot(\rho\mathbf v) = 0, \qquad
\partial_t(\rho\mathbf v) + \nabla\cdot\big(\rho\mathbf v\mathbf v
    + (p + \tfrac12 B^2)\mathsf I - \mathbf B\mathbf B\big) = 0,
$$

$$
\partial_t E + \nabla\cdot\big((E + p + \tfrac12 B^2)\mathbf v
    - (\mathbf v\cdot\mathbf B)\mathbf B + \eta\,\mathbf J\times\mathbf B\big) = 0,
\qquad
\partial_t\mathbf B = -\nabla\times\mathbf E, \qquad
\mathbf E = -\mathbf v\times\mathbf B + \eta\mathbf J, \qquad
\mathbf J = \nabla\times\mathbf B,
$$

with $E = p/(\gamma - 1) + \tfrac12\rho v^2 + \tfrac12 B^2$ and a uniform resistivity
$\eta$ (zero: ideal MHD). The Lundquist number of a layer of length $L$ is
$S = L v_A / \eta$.

**The field.** The in-plane field is the curl of the vector potential $A_z$ plus a
uniform field $\mathbf B_0$ (``mean_field``, which a periodic potential cannot carry),

$$
B_x = B_{0x} + \partial_y A_z, \qquad B_y = B_{0y} - \partial_x A_z, \qquad
\partial_t A_z = -E_z,
$$

and $A_z$ sits on the nodes of the patches. The field through a face is the difference of
$A_z$ at its two ends over its length, so $\nabla\cdot\mathbf B = 0$ holds in every cell to
round-off: this is constrained transport (Evans and Hawley 1988) with the
electromotive force at the cell corners, written for the potential. On a patch hierarchy
the nodes that patches share have one value (`patchwork.AMRHierarchy.restrict` gives the
coarse nodes under a finer patch the fine values), so a coarse face on the edge of a finer
level carries the sum of the fluxes of its fine faces: the divergence vanishes on the
composite mesh too, and a regrid, which interpolates $A_z$ bilinearly onto new patches,
keeps it. In 3D the potential would be the edge field of patchwork's face space; patchwork
has no edge-centred patch fields yet. $B_z$ is a conserved cell quantity.

**Numerics.** Finite volumes on the cells of each patch: MUSCL reconstruction of the
primitive variables ($\rho$, $\mathbf v$, $p$, the tangential field) with a slope
limiter, the normal field taken from the face, the HLLD flux (Miyoshi and Kusano 2005), and
the corner electromotive force of Gardiner and Stone (2005, the $E_z^c$ upwinding by the
mass flux). Resistivity adds $\eta J_z$ at the nodes, $\eta(J_x, J_y)$ to the flux of
$B_z$ and $\eta\,\mathbf J\times\mathbf B$ to the energy flux, explicitly. Heun's method
(SSP-RK2) with one time step for all levels; the fluxes through coarse–fine faces are
matched by `patchwork.FluxRegister` (refluxing), so the integrals of $\rho$, $\rho\mathbf v$,
$E$ and $B_z$ change by round-off only. The step is limited by the fast speed (CFL) and by
the resistive diffusion of the finest cells.

**Scope.** patchwork is an optional dependency (``pip install "plasma-carafe[spatial]"``),
imported when a model is made. Periodic boxes in the plane (slab geometry); no viscosity, conduction or Hall
term yet. The ``"carafe.mhd_step"`` profiling region times each step.

**MPI.** The model runs on any hierarchy over a communicator and gives the results of one
rank to round-off. Each rank's block of a patch needs at least three nodes along the split
axis (the halo of the potential): with many small patches, place each on one rank
(``AMRHierarchy(..., placement="morton")`` or ``"balanced"``) rather than splitting every
patch over all ranks (``"all"``, the default).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from methodcard import ConvergenceProblem, method
from scope_profiler import profile

from carafe._profiling import _PROFILE_TAGS
from carafe.gasdynamics import LIMITERS

if TYPE_CHECKING:
    from patchwork import AMRHierarchy, Patch

__all__ = ["CONSERVED", "MHD2D", "harris_sheets", "hlld_flux", "sheet_diagnostics"]

#: The cell fields of the conserved variables, in this order.
CONSERVED = ("rho", "mx", "my", "mz", "energy", "bz")
#: The node field of the vector potential.
POTENTIAL = "az"
#: Halo width of every field (MUSCL needs two cells beyond a face).
HALO = 2
_DENSITY_FLOOR = 1e-12
_PRESSURE_FLOOR = 1e-12

Function = Callable[[np.ndarray, np.ndarray], Any]


# -- the Riemann solver ---------------------------------------------------------------


def _fast_speed(rho, p, bn, bt1, bt2, gamma):
    gp = gamma * p
    b2 = bn * bn + bt1 * bt1 + bt2 * bt2
    disc = np.sqrt(np.maximum((gp + b2) ** 2 - 4.0 * gp * bn * bn, 0.0))
    return np.sqrt(np.maximum((gp + b2 + disc) / (2.0 * rho), 0.0))


@method(
    equations={
        "speeds": r"S_L = \min(u_L, u_R) - \max(c_{f,L}, c_{f,R}), \quad S_R = \max(u_L, u_R) + \max(c_{f,L}, c_{f,R}), \quad S_M = \frac{(S_R - u_R)\rho_R u_R - (S_L - u_L)\rho_L u_L - p_{T,R} + p_{T,L}}{(S_R - u_R)\rho_R - (S_L - u_L)\rho_L}",
        "alfven": r"S_L^* = S_M - \frac{|B_n|}{\sqrt{\rho_L^*}}, \qquad S_R^* = S_M + \frac{|B_n|}{\sqrt{\rho_R^*}}, \qquad \rho_\alpha^* = \rho_\alpha\frac{S_\alpha - u_\alpha}{S_\alpha - S_M}",
        "flux": r"F = F_L,\ F_L^*,\ F_L^{**},\ F_R^{**},\ F_R^*,\ F_R \quad\text{as } 0 \text{ lies in } (-\infty, S_L),\ (S_L, S_L^*),\ (S_L^*, S_M),\ (S_M, S_R^*),\ (S_R^*, S_R),\ (S_R, \infty)",
    },
    assumptions=(
        "four intermediate states: the total pressure and the normal velocity constant across the fan, the density only jumps at the fast waves",
        "exact for isolated contacts, tangential and rotational discontinuities; the degenerate star states (no normal field) fall back to the outer ones",
    ),
    references=(
        "[Miyoshi and Kusano, J. Comput. Phys. 208, 315 (2005)](https://doi.org/10.1016/j.jcp.2005.02.017)",
    ),
)
def hlld_flux(
    left: Sequence[np.ndarray],
    right: Sequence[np.ndarray],
    bn: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, ...]:
    """The HLLD flux (Miyoshi and Kusano 2005) through faces of normal field ``bn``.

    ``left`` and ``right`` are the primitive states ``(rho, vn, vt1, vt2, p, bt1, bt2)``
    on the two sides of each face (normal and two tangential components, a right-handed
    order is not needed). Returns the fluxes of ``(rho, rho vn, rho vt1, rho vt2, E, bt1,
    bt2)`` along the normal, with ``E`` the total energy density and the magnetic
    pressure ``B^2 / 2``.
    """
    rl, ul, vl, wl, pl, byl, bzl = (np.asarray(q, dtype=float) for q in left)
    rr, ur, vr, wr, pr, byr, bzr = (np.asarray(q, dtype=float) for q in right)
    bx = np.asarray(bn, dtype=float)
    bx2 = bx * bx

    def conserved(r, u, v, w, p, by, bz):
        b2 = bx2 + by * by + bz * bz
        energy = p / (gamma - 1.0) + 0.5 * r * (u * u + v * v + w * w) + 0.5 * b2
        return energy, p + 0.5 * b2

    el, ptl = conserved(rl, ul, vl, wl, pl, byl, bzl)
    er, ptr = conserved(rr, ur, vr, wr, pr, byr, bzr)

    def flux(r, u, v, w, by, bz, e, pt):
        vb = u * bx + v * by + w * bz
        return (
            r * u,
            r * u * u + pt - bx2,
            r * u * v - bx * by,
            r * u * w - bx * bz,
            (e + pt) * u - bx * vb,
            by * u - bx * v,
            bz * u - bx * w,
        )

    fl = flux(rl, ul, vl, wl, byl, bzl, el, ptl)
    fr = flux(rr, ur, vr, wr, byr, bzr, er, ptr)
    cfl = _fast_speed(rl, pl, bx, byl, bzl, gamma)
    cfr = _fast_speed(rr, pr, bx, byr, bzr, gamma)
    cmax = np.maximum(cfl, cfr)
    sl = np.minimum(ul, ur) - cmax
    sr = np.maximum(ul, ur) + cmax
    dl = (sl - ul) * rl
    dr = (sr - ur) * rr
    denom = dr - dl
    sm = (dr * ur - dl * ul - ptr + ptl) / denom
    pts = (dr * ptl - dl * ptr + dl * dr * (ur - ul)) / denom

    def star(r, u, v, w, by, bz, e, pt, s):
        rs = r * (s - u) / (s - sm)
        d = r * (s - u) * (s - sm) - bx2
        small = np.abs(d) < 1e-12 * np.maximum(pt, 1e-300)
        safe = np.where(small, 1.0, d)
        vs = np.where(small, v, v - bx * by * (sm - u) / safe)
        ws = np.where(small, w, w - bx * bz * (sm - u) / safe)
        factor = np.where(small, 1.0, (r * (s - u) ** 2 - bx2) / safe)
        bys = by * factor
        bzs = bz * factor
        vb = u * bx + v * by + w * bz
        vbs = sm * bx + vs * bys + ws * bzs
        es = ((s - u) * e - pt * u + pts * sm + bx * (vb - vbs)) / (s - sm)
        return rs, vs, ws, bys, bzs, es

    rsl, vsl, wsl, bysl, bzsl, esl = star(rl, ul, vl, wl, byl, bzl, el, ptl, sl)
    rsr, vsr, wsr, bysr, bzsr, esr = star(rr, ur, vr, wr, byr, bzr, er, ptr, sr)
    sqrtl, sqrtr = np.sqrt(rsl), np.sqrt(rsr)
    ssl = sm - np.abs(bx) / sqrtl
    ssr = sm + np.abs(bx) / sqrtr
    sign = np.sign(bx)
    total = sqrtl + sqrtr
    vss = (sqrtl * vsl + sqrtr * vsr + (bysr - bysl) * sign) / total
    wss = (sqrtl * wsl + sqrtr * wsr + (bzsr - bzsl) * sign) / total
    byss = (sqrtl * bysr + sqrtr * bysl + sqrtl * sqrtr * (vsr - vsl) * sign) / total
    bzss = (sqrtl * bzsr + sqrtr * bzsl + sqrtl * sqrtr * (wsr - wsl) * sign) / total
    vbss = sm * bx + vss * byss + wss * bzss
    vbsl = sm * bx + vsl * bysl + wsl * bzsl
    vbsr = sm * bx + vsr * bysr + wsr * bzsr
    essl = esl - sqrtl * (vbsl - vbss) * sign
    essr = esr + sqrtr * (vbsr - vbss) * sign

    ul_ = (rl, rl * ul, rl * vl, rl * wl, el, byl, bzl)
    ur_ = (rr, rr * ur, rr * vr, rr * wr, er, byr, bzr)
    usl = (rsl, rsl * sm, rsl * vsl, rsl * wsl, esl, bysl, bzsl)
    usr = (rsr, rsr * sm, rsr * vsr, rsr * wsr, esr, bysr, bzsr)
    ussl = (rsl, rsl * sm, rsl * vss, rsl * wss, essl, byss, bzss)
    ussr = (rsr, rsr * sm, rsr * vss, rsr * wss, essr, byss, bzss)

    result = []
    for k in range(7):
        f_sl = fl[k] + sl * (usl[k] - ul_[k])
        f_sr = fr[k] + sr * (usr[k] - ur_[k])
        f_ssl = f_sl + ssl * (ussl[k] - usl[k])
        f_ssr = f_sr + ssr * (ussr[k] - usr[k])
        result.append(
            np.select(
                [sl > 0.0, ssl >= 0.0, sm >= 0.0, ssr >= 0.0, sr >= 0.0],
                [fl[k], f_sl, f_ssl, f_ssr, f_sr],
                default=fr[k],
            )
        )
    return tuple(result)


# -- the model ------------------------------------------------------------------------


class _Block:
    """The arrays of one rank's block of a patch, in the patch's global indices.

    Cell ``(i, j)`` of the patch's level is at ``cells[:, i - ci, j - cj]`` and node
    ``(i, j)`` at ``nodes[i - ni, j - nj]``; ``bounds`` are the cells of the block,
    ``node_bounds`` the nodes it owns.
    """

    def __init__(self, patch: Patch) -> None:
        self.patch = patch
        arrays = [patch.field(name) for name in CONSERVED]
        layout = arrays[0].layout
        self.bounds = tuple(
            (patch.box[a][0] + b0, patch.box[a][0] + b1)
            for a, (b0, b1) in enumerate(layout.index_bounds)
        )
        self.cells = np.stack([np.asarray(a.local_with_halos) for a in arrays])
        potential = patch.field(POTENTIAL)
        nlayout = potential.layout
        self.node_bounds = tuple(
            (patch.box[a][0] + b0, patch.box[a][0] + b1)
            for a, (b0, b1) in enumerate(nlayout.index_bounds)
        )
        self.nodes = np.asarray(potential.local_with_halos)
        self.ci = tuple(lo - HALO for lo, _ in self.bounds)
        self.ni = tuple(
            lo - h for (lo, _), h in zip(self.node_bounds, nlayout.halo, strict=True)
        )


@method(
    equations={
        "mass": r"\partial_t\rho + \nabla\cdot(\rho\mathbf v) = 0",
        "momentum": r"\partial_t(\rho\mathbf v) + \nabla\cdot\big(\rho\mathbf v\mathbf v + (p + \tfrac12B^2)\mathsf I - \mathbf B\mathbf B\big) = 0",
        "energy": r"\partial_t E + \nabla\cdot\big((E + p + \tfrac12B^2)\mathbf v - (\mathbf v\cdot\mathbf B)\mathbf B + \eta\mathbf J\times\mathbf B\big) = 0, \qquad E = \frac{p}{\gamma-1} + \tfrac12\rho v^2 + \tfrac12B^2",
        "field": r"B_x = B_{0x} + \partial_y A_z,\quad B_y = B_{0y} - \partial_x A_z,\quad \partial_t A_z = -E_z,\quad \partial_t B_z = -(\nabla\times\mathbf E)_z, \quad \mathbf E = -\mathbf v\times\mathbf B + \eta\nabla\times\mathbf B",
    },
    order=2,
    assumptions=(
        r"units with $\mu_0 = 1$; periodic boxes in the plane; uniform resistivity $\eta$, explicit",
        r"$A_z$ on the patch nodes, so $\nabla\cdot\mathbf B = 0$ to round-off on the composite mesh and across regrids",
        r"MUSCL on the primitives (van Leer by default), HLLD, Gardiner–Stone corner EMF, Heun (SSP-RK2), one step for all levels, refluxing",
        "the order is that of the circularly polarised Alfvén wave with the unlimited reconstruction",
    ),
    references=(
        "[Miyoshi and Kusano, J. Comput. Phys. 208, 315 (2005)](https://doi.org/10.1016/j.jcp.2005.02.017)",
        "[Gardiner and Stone, J. Comput. Phys. 205, 509 (2005)](https://doi.org/10.1016/j.jcp.2004.11.016)",
        "[Evans and Hawley, Astrophys. J. 332, 659 (1988)](https://doi.org/10.1086/166684)",
        "[Tóth, J. Comput. Phys. 161, 605 (2000)](https://doi.org/10.1006/jcph.2000.6519)",
    ),
    components=(hlld_flux,),
    problem=lambda: _alfven_problem(),
)
class MHD2D:
    """Compressible MHD in a periodic plane on a patchwork hierarchy; see the module.

    Args:
        hierarchy: A 2D `patchwork.AMRHierarchy`, periodic along both axes. The model
            allocates its fields on it (the conserved cell fields `CONSERVED` and the
            node field ``"az"``, all with a halo of two).
        gamma: The adiabatic index.
        resistivity: The uniform resistivity ``eta`` (a magnetic diffusivity in these
            units); 0 for ideal MHD.
        mean_field: The uniform in-plane field ``(B0x, B0y)``.
        limiter: ``"minmod"``, ``"vanleer"`` (default), ``"mc"`` or ``"none"`` (central
            slopes, for convergence tests only).
        cfl: The Courant number of the fast waves (and the share of the explicit
            diffusion limit the resistive step may use).

    Attributes:
        hierarchy: The current hierarchy (`regrid` replaces it).
        time: The time reached.
        steps: The number of steps taken.
    """

    def _use(self, hierarchy: AMRHierarchy) -> None:
        from patchwork import CompositeCells, FluxRegister

        self.hierarchy = hierarchy
        self._cells = CompositeCells(hierarchy)
        self._register = FluxRegister(hierarchy, self._cells)

    def __init__(
        self,
        hierarchy: AMRHierarchy,
        *,
        gamma: float = 5.0 / 3.0,
        resistivity: float = 0.0,
        mean_field: tuple[float, float] = (0.0, 0.0),
        limiter: str = "vanleer",
        cfl: float = 0.4,
    ) -> None:
        """Allocate the fields on ``hierarchy`` (the state is zero until `set_state`)."""
        if hierarchy.ndim != 2:
            raise ValueError("MHD2D needs a 2D hierarchy")
        if not all(hierarchy.periodic):
            raise ValueError("MHD2D supports periodic boxes only (both axes periodic)")
        if hierarchy.cylindrical or getattr(hierarchy, "graded", False):
            raise ValueError("MHD2D needs a uniform Cartesian hierarchy")
        if limiter not in LIMITERS:
            raise ValueError(
                f"limiter must be one of {sorted(LIMITERS)}, got {limiter!r}"
            )
        if resistivity < 0:
            raise ValueError("resistivity must be non-negative")
        self.gamma = float(gamma)
        self.resistivity = float(resistivity)
        self.mean_field = (float(mean_field[0]), float(mean_field[1]))
        self.limiter = limiter
        self.cfl = float(cfl)
        self.time = 0.0
        self.steps = 0
        for name in CONSERVED:
            if name not in hierarchy.field_names:
                hierarchy.allocate_field(name, halo=HALO, conservative=True)
        if POTENTIAL not in hierarchy.field_names:
            # one node more than the cells: a block of a patch split over ranks may
            # start one node after its first cell
            hierarchy.allocate_field(POTENTIAL, halo=HALO + 1, centering="node")
        self._use(hierarchy)

    def _face_field(self, block: _Block) -> tuple[np.ndarray, np.ndarray]:
        """``Bx`` on the x-faces and ``By`` on the y-faces of the block's storage.

        ``bxf[i, j]``: x-face at node column ``i``, cell row ``j``; ``byf[i, j]``: y-face
        at cell column ``i``, node row ``j``; both indexed like the cells, offset
        ``block.ci`` (a face at node ``i`` has index ``i - ci``).
        """
        dx, dy = self.hierarchy.spacing(block.patch.level)
        a = block.nodes
        ncx, ncy = block.cells.shape[1:]
        # the nodes in a frame aligned with the cells: node i at frame index i - ci
        frame = np.full((ncx + 1, ncy + 1), np.nan)
        source, target = [], []
        for axis, size in enumerate(frame.shape):
            first = max(block.ci[axis], block.ni[axis])
            last = min(block.ci[axis] + size, block.ni[axis] + a.shape[axis])
            target.append(slice(first - block.ci[axis], last - block.ci[axis]))
            source.append(slice(first - block.ni[axis], last - block.ni[axis]))
        frame[tuple(target)] = a[tuple(source)]
        if np.isnan(frame).any():  # a limiter would turn it into a zero slope
            raise RuntimeError("the potential's halo does not cover the cells' halo")
        bxf = np.empty((ncx + 1, ncy))
        byf = np.empty((ncx, ncy + 1))
        bxf[...] = self.mean_field[0] + (frame[:, 1:] - frame[:, :-1]) / dy
        byf[...] = self.mean_field[1] - (frame[1:, :] - frame[:-1, :]) / dx
        return bxf, byf

    def _cell_field(self, block: _Block) -> tuple[np.ndarray, np.ndarray]:
        bxf, byf = self._face_field(block)
        return 0.5 * (bxf[1:] + bxf[:-1]), 0.5 * (byf[:, 1:] + byf[:, :-1])

    # -- state ----------------------------------------------------------------------

    def set_state(
        self,
        *,
        density: float | Function,
        velocity: Sequence[float | Function] = (0.0, 0.0, 0.0),
        pressure: float | Function,
        potential: float | Function = 0.0,
        bz: float | Function = 0.0,
    ) -> None:
        """Set the state from numbers or functions of ``(x, y)``.

        ``density``, ``velocity`` (three components), ``pressure`` and ``bz`` at the cell
        centres, ``potential`` (``A_z``) at the nodes; the in-plane field is the mean
        field plus the curl of the potential. The energy is set from the pressure and
        the field of the cells (the mean of their faces).
        """
        mesh = self.hierarchy

        def value(source, x, y):
            result = source(x, y) if callable(source) else source
            return np.broadcast_to(np.asarray(result, dtype=float), x.shape)

        for patch in mesh.local_patches:
            x, y = patch.local_node_coordinates(mesh, POTENTIAL)
            patch.field(POTENTIAL).local[...] = value(potential, x, y)
        mesh.restrict(POTENTIAL)
        mesh.exchange_halos(POTENTIAL)
        for patch in mesh.local_patches:
            x, y = patch.local_cell_centers(mesh, "rho")
            rho = value(density, x, y)
            v = [value(c, x, y) for c in velocity]
            p = value(pressure, x, y)
            b3 = value(bz, x, y)
            block = _Block(patch)
            bx, by = self._cell_field(block)
            (b0, b1), (c0, c1) = block.bounds
            sx = slice(b0 - block.ci[0], b1 - block.ci[0])
            sy = slice(c0 - block.ci[1], c1 - block.ci[1])
            bx, by = bx[sx, sy], by[sx, sy]
            energy = (
                p / (self.gamma - 1.0)
                + 0.5 * rho * (v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
                + 0.5 * (bx**2 + by**2 + b3**2)
            )
            for name, q in zip(
                CONSERVED,
                (rho, rho * v[0], rho * v[1], rho * v[2], energy, b3),
                strict=True,
            ):
                patch.field(name).local[...] = q
        for name in CONSERVED:
            mesh.restrict(name)
            mesh.exchange_halos(name)

    def _primitive(self, block: _Block) -> tuple[np.ndarray, ...]:
        rho, mx, my, mz, energy, bz = block.cells
        bx, by = self._cell_field(block)
        rho = np.maximum(rho, _DENSITY_FLOOR)
        vx, vy, vz = mx / rho, my / rho, mz / rho
        p = (self.gamma - 1.0) * (
            energy
            - 0.5 * rho * (vx * vx + vy * vy + vz * vz)
            - 0.5 * (bx * bx + by * by + bz * bz)
        )
        p = np.where(np.isnan(p), p, np.maximum(p, _PRESSURE_FLOOR))
        return rho, vx, vy, vz, p, bx, by, bz

    # -- fluxes and the electromotive force --------------------------------------------

    def _reconstruct(self, q: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
        """Left and right states at the faces between neighbours along ``axis``.

        Face ``k`` (between cells ``k - 1`` and ``k`` of the storage) for ``k`` from 2 to
        ``n - 2``: arrays of ``n - 3`` faces along ``axis``.
        """
        limiter = LIMITERS[self.limiter]
        q = np.moveaxis(q, axis, 0)
        diff = q[1:] - q[:-1]
        slope = limiter(diff[:-1], diff[1:])  # cells 1 .. n - 2
        left = q[1:-2] + 0.5 * slope[:-1]  # cells 1 .. n - 3, upper side
        right = q[2:-1] - 0.5 * slope[1:]  # cells 2 .. n - 2, lower side
        return np.moveaxis(left, 0, axis), np.moveaxis(right, 0, axis)

    def _riemann(self, prim, normal_field, axis):
        """Fluxes through the faces normal to ``axis``, storage faces 2 .. n - 2."""
        rho, vx, vy, vz, p, bx, by, bz = prim
        if axis == 0:
            order = (rho, vx, vy, vz, p, by, bz)
        else:
            order = (rho, vy, vx, vz, p, bx, bz)
        states = [self._reconstruct(q, axis) for q in order]
        left = [s[0] for s in states]
        right = [s[1] for s in states]
        # fall back to the cell values where the reconstruction is not positive
        bad = (left[0] <= 0) | (left[4] <= 0) | (right[0] <= 0) | (right[4] <= 0)
        if np.any(bad):
            take = [slice(None)] * 2
            take[axis] = slice(1, -2)
            lo = tuple(take)
            take[axis] = slice(2, -1)
            hi = tuple(take)
            for k, q in enumerate(order):
                left[k] = np.where(bad, q[lo], left[k])
                right[k] = np.where(bad, q[hi], right[k])
        take = [slice(None)] * 2
        take[axis] = slice(2, -2)
        bn = normal_field[tuple(take)]
        return hlld_flux(left, right, bn, self.gamma)

    def _resistive_fluxes(self, x_frame, y_frame, prim, jz, dx, dy) -> None:
        """Add ``eta (Jx, Jy)`` to the flux of ``B_z`` and ``eta J x B`` to the energy flux."""
        eta = self.resistivity
        _, _, _, _, _, bx, by, bz = prim
        ncx, ncy = bz.shape
        # x-faces 1 .. ncx - 1 of the frame (between cells k - 1, k)
        jy = -(bz[1:] - bz[:-1]) / dx
        bz_face = 0.5 * (bz[1:] + bz[:-1])
        by_face = 0.5 * (by[1:] + by[:-1])
        jz_xface = np.full((ncx - 1, ncy), np.nan)
        jz_xface[:, 1:-1] = 0.5 * (jz[:, 1:] + jz[:, :-1])
        x_frame[6][1:-1] += eta * jy  # flux of Bz along x: E_y = eta J_y
        x_frame[4][1:-1] += eta * (jy * bz_face - jz_xface * by_face)
        jx = (bz[:, 1:] - bz[:, :-1]) / dy
        bz_face = 0.5 * (bz[:, 1:] + bz[:, :-1])
        bx_face = 0.5 * (bx[:, 1:] + bx[:, :-1])
        jz_yface = np.full((ncx, ncy - 1), np.nan)
        jz_yface[1:-1, :] = 0.5 * (jz[1:, :] + jz[:-1, :])
        y_frame[6][:, 1:-1] -= eta * jx  # flux of Bz along y: -E_x = -eta J_x
        y_frame[4][:, 1:-1] += eta * (jz_yface * bx_face - jx * bz_face)

    def _fluxes_unguarded(self, block: _Block) -> dict[str, Any]:
        mesh = self.hierarchy
        dx, dy = mesh.spacing(block.patch.level)
        prim = self._primitive(block)
        rho, vx, vy, vz, p, bx, by, bz = prim
        bxf, byf = self._face_field(block)
        # x-faces: storage faces 2 .. ncx - 2 (node index i = ci + k), all rows
        fx = self._riemann(prim, bxf, 0)
        fy = self._riemann(prim, byf, 1)
        ncx, ncy = block.cells.shape[1:]
        # flux arrays on a full face frame (nan where not computed)
        x_frame = [np.full((ncx + 1, ncy), np.nan) for _ in range(7)]
        y_frame = [np.full((ncx, ncy + 1), np.nan) for _ in range(7)]
        for k in range(7):
            x_frame[k][2 : ncx - 1] = fx[k]
            y_frame[k][:, 2 : ncy - 1] = fy[k]
        # E_z on the faces: -F_x(By) and G_y(Bx); in the y-sweep the first tangential
        # field is Bx
        ex_face = -x_frame[5]
        ey_face = y_frame[5]
        e_cell = vy * bx - vx * by
        mass_x = x_frame[0]
        mass_y = y_frame[0]
        # corner (node) values for nodes 1 .. ncx - 1 of the storage frame
        e4 = 0.25 * (
            ex_face[1:-1, 1:]
            + ex_face[1:-1, :-1]
            + ey_face[1:, 1:-1]
            + ey_face[:-1, 1:-1]
        )

        def upwind(flux, a, b):
            return np.where(flux > 0, a, np.where(flux < 0, b, 0.5 * (a + b)))

        # (dE/dy) on x-face (i, row j) upwinded from cells i - 1 and i
        # up: between the y-face below row j and the centre of row j
        up_cells = e_cell - ey_face[:, :-1]  # cell (c, j): E_c - E(y-face below)
        dn_cells = ey_face[:, 1:] - e_cell  # cell (c, j): E(y-face above) - E_c
        d_up = upwind(mass_x[1:-1, 1:], up_cells[:-1, 1:], up_cells[1:, 1:])
        d_dn = upwind(mass_x[1:-1, :-1], dn_cells[:-1, :-1], dn_cells[1:, :-1])
        right_cells = e_cell - ex_face[:-1, :]  # cell (i, r): E_c - E(x-face left)
        left_cells = ex_face[1:, :] - e_cell  # cell (i, r): E(x-face right) - E_c
        d_right = upwind(mass_y[1:, 1:-1], right_cells[1:, :-1], right_cells[1:, 1:])
        d_left = upwind(mass_y[:-1, 1:-1], left_cells[:-1, :-1], left_cells[:-1, 1:])
        # integrated from the four face centres to the corner (GS05 eq. 41): in a flow
        # along x alone it is the upwind E_z of the x-faces
        ez = e4 + 0.25 * (d_dn - d_up) + 0.25 * (d_left - d_right)
        # node frame: storage node k (k = 1 .. ncx - 1) is the corner of cells k - 1, k
        jz = None
        if self.resistivity > 0:
            jz = (byf[1:, 1:-1] - byf[:-1, 1:-1]) / dx - (
                bxf[1:-1, 1:] - bxf[1:-1, :-1]
            ) / dy
            ez = ez + self.resistivity * jz
            self._resistive_fluxes(x_frame, y_frame, prim, jz, dx, dy)
        return {"x": x_frame, "y": y_frame, "ez": ez}

    def _fluxes(self, block: _Block) -> dict[str, Any]:
        """Fluxes of the conserved fields through the block's faces, and ``E_z`` at its nodes.

        Computed on the whole storage of the block; the points next to its outer edge,
        which lack neighbours, are NaN and never used.
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            return self._fluxes_unguarded(block)

    def _interior(self, block: _Block, arrays):
        (b0, b1), (c0, c1) = block.bounds
        sx = slice(b0 - block.ci[0], b1 - block.ci[0])
        sy = slice(c0 - block.ci[1], c1 - block.ci[1])
        return tuple(q[sx, sy] for q in arrays)

    # -- time stepping -------------------------------------------------------------------

    def stable_step(self) -> float:
        """The largest stable time step over the hierarchy (collective)."""
        smallest = np.inf
        for patch in self.hierarchy.local_patches:
            block = _Block(patch)
            rho, vx, vy, vz, p, bx, by, bz = self._interior(
                block, self._primitive(block)
            )
            dx, dy = self.hierarchy.spacing(patch.level)
            cx = _fast_speed(rho, p, bx, by, bz, self.gamma)
            cy = _fast_speed(rho, p, by, bx, bz, self.gamma)
            rate = np.max((np.abs(vx) + cx) / dx + (np.abs(vy) + cy) / dy)
            dt = self.cfl / rate
            if self.resistivity > 0:
                dt = min(dt, self.cfl * 0.5 / (self.resistivity * (dx**-2 + dy**-2)))
            smallest = min(smallest, dt)
        from maybempi import MPI

        return float(self.hierarchy.comm.allreduce(smallest, op=MPI.MIN))

    def _stage(self, dt: float) -> None:
        """One forward-Euler stage of every field, recording the fluxes with ``dt / 2``."""
        mesh = self.hierarchy
        cache: dict[int, dict[str, Any]] = {}
        blocks: dict[int, _Block] = {}
        for patch in mesh.local_patches:
            block = _Block(patch)
            blocks[id(patch)] = block
            cache[id(patch)] = self._fluxes(block)

        # map the conserved fields to the flux components of each sweep:
        # x-sweep: (rho, mx, my, mz, E, By, Bz); y-sweep: (rho, my, mx, mz, E, Bx, Bz)
        x_index = {"rho": 0, "mx": 1, "my": 2, "mz": 3, "energy": 4, "bz": 6}
        y_index = {"rho": 0, "mx": 2, "my": 1, "mz": 3, "energy": 4, "bz": 6}
        for name in CONSERVED:

            def fluxes(patch: Patch, name: str = name):
                block = blocks[id(patch)]
                (b0, b1), (c0, c1) = block.bounds
                ox, oy = block.ci
                fx = cache[id(patch)]["x"][x_index[name]]
                fy = cache[id(patch)]["y"][y_index[name]]
                return (
                    fx[b0 - ox : b1 - ox + 1, c0 - oy : c1 - oy],
                    fy[b0 - ox : b1 - ox, c0 - oy : c1 - oy + 1],
                )

            self._register.update(name, fluxes, dt, scale=0.5 * dt)
        for patch in mesh.local_patches:
            block = blocks[id(patch)]
            ez = cache[id(patch)]["ez"]  # storage nodes 1 .. ncx - 1
            (n0, n1), (m0, m1) = block.node_bounds
            ox, oy = block.ci
            values = ez[n0 - ox - 1 : n1 - ox - 1, m0 - oy - 1 : m1 - oy - 1]
            patch.field(POTENTIAL).local[...] -= dt * values

    def _exchange(self) -> None:
        mesh = self.hierarchy
        mesh.restrict(POTENTIAL)
        mesh.exchange_halos(POTENTIAL)
        for name in CONSERVED:
            mesh.restrict(name)
            mesh.exchange_halos(name)

    @profile("carafe.mhd_step", tags=_PROFILE_TAGS)
    def step(self, dt: float | None = None) -> float:
        """Advance every level by one step of Heun's method; returns the step taken."""
        if dt is None:
            dt = self.stable_step()
        mesh = self.hierarchy
        names = (*CONSERVED, POTENTIAL)
        start = {
            (id(patch), name): patch.field(name).local.copy()
            for patch in mesh.local_patches
            for name in names
        }
        for _ in range(2):
            self._stage(dt)
            self._exchange()
        for patch in mesh.local_patches:
            for name in names:
                array = patch.field(name).local
                array[...] = 0.5 * (start[(id(patch), name)] + array)
        for name in CONSERVED:
            self._register.reflux(name)
        self._exchange()
        self.time += dt
        self.steps += 1
        return dt

    def run(
        self, until: float, *, callback: Callable[[MHD2D], None] | None = None
    ) -> None:
        """Step until ``until``, the last step shortened to land on it."""
        while self.time < until * (1 - 1e-12):
            dt = min(self.stable_step(), until - self.time)
            self.step(dt)
            if callback is not None:
                callback(self)

    # -- refinement ----------------------------------------------------------------------

    def current_tags(self, threshold: float, levels: int) -> dict[int, np.ndarray]:
        """Tag the cells where ``|J_z|`` times the cell size exceeds ``threshold``.

        The change of the in-plane field across a cell; levels ``0 .. levels - 1`` are
        tagged (so at most ``levels`` refined levels), from the patches this rank
        holds. The result is for `regrid`.
        """
        mesh = self.hierarchy
        tags = {
            level: np.zeros(mesh.level_shape(level), dtype=bool)
            for level in range(min(levels, mesh.max_level + 1))
        }
        for patch in mesh.local_patches:
            if patch.level not in tags:
                continue
            block = _Block(patch)
            dx, dy = mesh.spacing(patch.level)
            bxf, byf = self._face_field(block)
            jz = (byf[1:, 1:-1] - byf[:-1, 1:-1]) / dx - (
                bxf[1:-1, 1:] - bxf[1:-1, :-1]
            ) / dy
            # cell (k, l) of the storage has the nodes k, k + 1: jz nodes are 1 .. n - 1
            cell = 0.25 * (
                np.abs(jz[:-1, :-1])
                + np.abs(jz[1:, :-1])
                + np.abs(jz[:-1, 1:])
                + np.abs(jz[1:, 1:])
            )
            (b0, b1), (c0, c1) = block.bounds
            ox, oy = block.ci
            inside = cell[b0 - ox - 1 : b1 - ox - 1, c0 - oy - 1 : c1 - oy - 1]
            tags[patch.level][b0:b1, c0:c1] |= inside * max(dx, dy) > threshold
        return tags

    def regrid(self, tags: Mapping[int, Any], **options: Any) -> None:
        """Regrid the hierarchy on ``tags`` (see `patchwork.AMRHierarchy.regrid`).

        The conserved fields keep their integrals, the potential is interpolated
        (bilinear), so the field stays divergence-free; the flux register is rebuilt.
        """
        new = self.hierarchy.regrid(tags, **options)
        self._use(new)
        new.restrict(POTENTIAL)
        self._exchange()

    # -- diagnostics ----------------------------------------------------------------------

    def totals(self) -> dict[str, float]:
        """The integrals of the conserved fields over the domain (collective)."""
        return {name: float(self.hierarchy.integral(name)) for name in CONSERVED}

    def leaf_values(self) -> dict[str, np.ndarray]:
        """The primitive variables on the leaf cells (gathered on every rank).

        ``x``, ``y`` (cell centres), ``level``, ``rho``, ``vx``, ``vy``,
        ``vz``, ``p``, ``bx``, ``by``, ``bz`` and ``jz`` (the mean of the corners).
        """
        mesh = self.hierarchy
        for name in ("bx_cell", "by_cell", "jz_cell", "p_cell"):
            if name not in mesh.field_names:
                mesh.allocate_field(name, halo=0)
        for patch in mesh.local_patches:
            block = _Block(patch)
            prim = self._primitive(block)
            dx, dy = mesh.spacing(patch.level)
            bxf, byf = self._face_field(block)
            jz = (byf[1:, 1:-1] - byf[:-1, 1:-1]) / dx - (
                bxf[1:-1, 1:] - bxf[1:-1, :-1]
            ) / dy
            cell = 0.25 * (jz[:-1, :-1] + jz[1:, :-1] + jz[:-1, 1:] + jz[1:, 1:])
            (b0, b1), (c0, c1) = block.bounds
            ox, oy = block.ci
            interior = self._interior(block, prim)
            patch.field("bx_cell").local[...] = interior[5]
            patch.field("by_cell").local[...] = interior[6]
            patch.field("p_cell").local[...] = interior[4]
            patch.field("jz_cell").local[...] = cell[
                b0 - ox - 1 : b1 - ox - 1, c0 - oy - 1 : c1 - oy - 1
            ]
        cells = self._cells
        get = cells.from_patches
        centres = cells.centers
        levels = np.array([p.level for p in mesh.patches])[cells.patch]
        rho = get("rho")
        result = {
            "x": centres[:, 0],
            "y": centres[:, 1],
            "level": levels,
            "rho": rho,
            "vx": get("mx") / rho,
            "vy": get("my") / rho,
            "vz": get("mz") / rho,
            "p": get("p_cell"),
            "bx": get("bx_cell"),
            "by": get("by_cell"),
            "bz": get("bz"),
            "jz": get("jz_cell"),
        }
        return result


# -- the convergence problem of the method card ----------------------------------------------


def alfven_wave(
    n: int, *, levels: int = 0, limiter: str = "none", cfl: float = 0.4
) -> tuple[MHD2D, Callable[[np.ndarray, float], np.ndarray]]:
    """A circularly polarised Alfvén wave along ``x`` (Tóth 2000) on an ``n x 4`` box.

    ``levels``: refined patches over the middle of the box. Returns the model and the
    exact ``B_y(x, t)``; the wave travels at ``v_A = 1`` and returns after ``t = 1``.
    """
    from patchwork import AMRHierarchy

    mesh = AMRHierarchy(
        (n, 4), bounds=((0.0, 1.0), (0.0, 4.0 / n)), periodic=(True, True)
    )
    for level in range(1, levels + 1):
        r = 2**level
        mesh.add_patch(level, ((n * r // 4, 3 * n * r // 4), (0, 4 * r)))
    model = MHD2D(
        mesh, gamma=5.0 / 3.0, mean_field=(1.0, 0.0), limiter=limiter, cfl=cfl
    )
    amp = 0.1
    k = 2.0 * np.pi

    def by_exact(x, t):
        return amp * np.sin(k * (x - t))

    model.set_state(
        density=1.0,
        velocity=(
            0.0,
            lambda x, y: -amp * np.sin(k * x),
            lambda x, y: -amp * np.cos(k * x),
        ),
        pressure=0.1,
        potential=lambda x, y: amp * np.cos(k * x) / k,
        bz=lambda x, y: amp * np.cos(k * x),
    )
    return model, by_exact


def _alfven_problem() -> ConvergenceProblem:
    def error(h: float) -> float:
        model, exact = alfven_wave(round(1.0 / h))
        model.run(0.5)
        leaf = model.leaf_values()
        return float(np.mean(np.abs(leaf["by"] - exact(leaf["x"], 0.5))))

    return ConvergenceProblem(error, steps=[1 / 16, 1 / 32, 1 / 64])


# -- reconnection: two Harris sheets in a periodic box ----------------------------------------


def harris_sheets(
    hierarchy: AMRHierarchy,
    *,
    width: float,
    beta: float = 0.2,
    perturbation: float = 1e-3,
    noise: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Two Harris current sheets at ``y = Ly / 4`` and ``3 Ly / 4`` of a periodic box.

    Returns the arguments of `MHD2D.set_state` (functions of ``x, y``), so a model can
    take them again on a regridded hierarchy, whose patches then resolve the sheets
    instead of interpolating them from the base level.

    ``B_x = tanh((y - y1) / w) - tanh((y - y2) / w) - 1`` (unit field, ``|B_x| = 1`` far
    from the sheets, the sheets of opposite current), uniform density 1 and the pressure
    of the balance ``p + B_x^2 / 2 = (1 + beta) / 2``, so the Alfvén speed is 1 and the
    Lundquist number of a sheet of the box's length is ``S = Lx / eta``. The sheets are
    perturbed by ``perturbation`` times the box's longest mode in the potential (one
    X-point per sheet) and, with ``noise``, by random modes up to a tenth of the length
    (a seed for plasmoids).
    """
    (x0, x1), (y0, y1) = hierarchy.bounds
    lx, ly = x1 - x0, y1 - y0
    first, second = y0 + 0.25 * ly, y0 + 0.75 * ly
    rng = np.random.default_rng(seed)
    modes = np.arange(2, 11)
    phases = rng.uniform(0.0, 2.0 * np.pi, (2, modes.size))
    amplitudes = noise * rng.standard_normal((2, modes.size)) / modes

    def potential(x, y):
        # B_x = dA/dy; the sheets' B_x has zero mean over y, so A is periodic
        a = width * (
            np.log(np.cosh((y - first) / width)) - np.log(np.cosh((y - second) / width))
        ) - (y - y0)
        for sheet, (centre, sign) in enumerate(((first, 1.0), (second, -1.0))):
            shape = np.exp(-(((y - centre) / (2.0 * width)) ** 2))
            wave = perturbation * np.cos(2.0 * np.pi * (x - x0) / lx)
            for mode, phase, amp in zip(
                modes, phases[sheet], amplitudes[sheet], strict=True
            ):
                wave = wave + amp * np.cos(2.0 * np.pi * mode * (x - x0) / lx + phase)
            a = a + sign * wave * shape
        return a

    def pressure(x, y):
        bx = np.tanh((y - first) / width) - np.tanh((y - second) / width) - 1.0
        return 0.5 * (1.0 + beta) - 0.5 * bx * bx

    return {"density": 1.0, "pressure": pressure, "potential": potential}


def sheet_diagnostics(
    model: MHD2D, leaf: Mapping[str, np.ndarray] | None = None
) -> dict:
    """The reconnected flux and the plasmoids of the lower sheet of `harris_sheets`.

    Along the line ``y = Ly / 4``: the reconnected flux ``psi = (1/2) int |B_y| dx`` (per
    unit length out of the plane), the number of O-points (where ``B_y`` changes sign
    from positive to negative along ``x``: field lines close around them) and of
    X-points, and the largest ``|J_z|`` in the box. ``leaf``: the `MHD2D.leaf_values`
    already computed.
    """
    mesh = model.hierarchy
    leaf = model.leaf_values() if leaf is None else leaf
    (x0, x1), (y0, y1) = mesh.bounds
    line = y0 + 0.25 * (y1 - y0)
    finest = mesh.spacing(mesh.max_level)[0]
    x = x0 + (np.arange(round((x1 - x0) / finest)) + 0.5) * finest
    # the cells on both sides of the line, averaged
    below = model._cells.locate(np.array([x, np.full_like(x, line - 1e-9)]))
    above = model._cells.locate(np.array([x, np.full_like(x, line + 1e-9)]))
    by = 0.5 * (leaf["by"][below] + leaf["by"][above])
    psi = 0.5 * float(np.sum(np.abs(by)) * finest)
    sign = np.sign(by)
    turns = sign * np.roll(sign, -1) < 0
    # with B_x growing across the sheet (B_x = dA/dy), A has a maximum at an O-point:
    # B_y = -dA/dx turns from positive to negative there
    o_points = int(np.sum(turns & (sign > 0)))
    x_points = int(np.sum(turns & (sign < 0)))
    return {
        "time": model.time,
        "psi": psi,
        "o_points": o_points,
        "x_points": x_points,
        "max_current": float(np.max(np.abs(leaf["jz"]))),
        "leaf_cells": int(leaf["rho"].size),
    }
