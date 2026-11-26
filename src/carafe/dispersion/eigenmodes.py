"""Nonlocal linear eigenmodes of the gradient-drift instability across a 1D profile.

The local relation of `carafe.dispersion` treats the profile as uniform over a
wavelength. Where the density falls over a few millimetres, as in the anode layer of a
magnetron, the radial structure of a mode is part of the problem: this module solves the
linearised equations on the whole profile ``x`` as an eigenvalue problem in ``omega`` at
each azimuthal wavenumber ``k_y``.

Geometry and signs follow `carafe.dispersion`: a slab ``(x, y)`` with ``x`` across
the field toward the anode, ``y`` along ``E x B`` for ``E_0 < 0`` and ``B > 0``, the model
field ``B(x) z``; perturbations ``f(x) exp(i k_y y - i omega t)``, so ``Im omega > 0``
grows and ``Re omega / k_y`` is the phase speed along ``+y``.

- **Electrons**: an isothermal drift fluid with collisions (the perpendicular momentum
  balance without inertia, solved for the velocity) or, with ``inertia=True``, the full
  momentum balance with the equilibrium flow. The fluxes are kept in divergence form, so
  the compressibility of every drift in ``B(x) z`` is automatic; ``curl_free_field`` adds
  the second ``-v_x B'/B`` of a vacuum field (the 3D arch). The ionisation source is
  removed with the equilibrium continuity, leaving ``nu_iz (a_i - a_e)`` in the ion
  equation only (the consistent form).
- **Gyro-averaged electrons** (``gyroaverage=True``): the electron unknown is the
  gyrocentre density, driven by ``Gamma_0^(1/2) delta phi`` with the Padé forms
  ``Gamma_0 = (1 + b)^-1``, ``Gamma_0^(1/2) = (1 + b/2)^-1`` of the operator
  ``b = k_y^2 rho_e^2 - d/dx rho_e^2 d/dx``; the particle density adds the polarisation
  shielding ``(1 - Gamma_0) e delta phi / T_e``. This gives the finite-Larmor-radius cutoff
  the fluid lacks (the lower-hybrid restoring force at ``k rho_e`` of order one).
- **Ions**: a cold unmagnetised fluid with the equilibrium flow ``v_0(x)``, its
  shear, a charge-exchange collision frequency and the ionisation coupling.
- **Closure**: Poisson's equation (``delta phi`` eliminated, a standard eigenproblem) or
  quasineutrality (a generalised eigenproblem).

Second-order finite differences on a uniform grid, ``a_e = a_i = delta phi = 0`` at both
ends, first-order upwinding of the flows (so the growth converges at first order: at 200
points on Boeuf's state it is 3 % below the limit), and a dense eigensolve: 1-3 s per ``k_y``
at 200 points, about 15 s for the quasineutral (generalised) problem. The growing roots below ``0.1 omega_ce`` with smooth eigenfunctions are the
drift branch (`Eigenmodes.drift`); `NonlocalDispersion.scan` follows it over ``k_y``.

`Equilibrium.boeuf` is the 1D state of Boeuf's DC magnetron (arXiv:2211.17051, his figure
2, digitized in `BOEUF_FIGURE_2`): on it the drift fluid gives one growing branch, peaked
in the anode layer at 18 mm, with ``6.3e6`` /s and ``+7.2`` km/s at his 1.6 cm wavelength.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import xarray as xr
from methodcard import method
from plasmacoll.constants import ATOMIC_MASS, ELECTRON_MASS, ELEMENTARY_CHARGE
from scipy import linalg
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline, PchipInterpolator

from carafe.datasets import variable

__all__ = [
    "BOEUF_FIGURE_2",
    "Eigenmodes",
    "Equilibrium",
    "NonlocalDispersion",
]

E = ELEMENTARY_CHARGE
EPSILON_0 = 8.8541878128e-12
ARGON_MASS = 39.948 * ATOMIC_MASS - ELECTRON_MASS

#: Boeuf's 1D DC magnetron state (arXiv:2211.17051, figure 2), digitized from the curves of
#: the arXiv version to about 2 % of full scale. Columns: ``x`` [cm] (cathode at 0, anode at
#: 2), ``n_e`` and ``n_i`` [1e15 m^-3], ``E`` [1e4 V/m], ``T_e`` [eV], ionisation rate
#: ``n_e nu_iz`` [1e21 m^-3 s^-1]. Checks (trapezoidal rule on these points): the integral of
#: ``E`` is -261 V (260 V applied), and ``e`` times the integral of the ionisation rate is
#: 5.18 A/m^2 (his 5.1 A/m^2).
BOEUF_FIGURE_2 = np.array(
    [
        (0.00, 0.55, 1.36, -4.81, 6.04, 0.05),
        (0.05, 0.45, 1.45, -4.20, 7.97, 0.22),
        (0.10, 0.74, 1.55, -3.38, 7.49, 0.28),
        (0.20, 1.39, 1.71, -2.34, 6.77, 0.35),
        (0.30, 1.75, 1.83, -2.02, 6.92, 0.45),
        (0.40, 1.88, 1.94, -1.91, 7.47, 0.63),
        (0.50, 1.99, 2.05, -1.82, 8.12, 0.87),
        (0.60, 2.11, 2.17, -1.74, 8.67, 1.18),
        (0.70, 2.23, 2.29, -1.60, 9.27, 1.55),
        (0.80, 2.32, 2.41, -1.47, 9.81, 1.92),
        (0.90, 2.41, 2.49, -1.31, 10.35, 2.31),
        (1.00, 2.45, 2.54, -1.16, 10.75, 2.61),
        (1.10, 2.45, 2.54, -1.01, 11.08, 2.82),
        (1.20, 2.41, 2.48, -0.87, 11.28, 2.94),
        (1.30, 2.29, 2.35, -0.74, 11.48, 2.94),
        (1.40, 2.08, 2.15, -0.63, 11.83, 2.84),
        (1.50, 1.80, 1.87, -0.54, 11.98, 2.59),
        (1.60, 1.49, 1.56, -0.43, 11.98, 2.23),
        (1.70, 1.14, 1.21, -0.32, 11.63, 1.74),
        (1.80, 0.78, 0.85, -0.21, 11.38, 1.22),
        (1.90, 0.43, 0.50, -0.10, 11.50, 0.73),
        (2.00, 0.08, 0.20, 0.00, 15.74, 0.21),
    ]
)


def _ion_flow(x: np.ndarray, source: np.ndarray, density: np.ndarray) -> np.ndarray:
    """``-int_x^L S dx' / n``: the ion flow toward the cathode fed by the source ``S``."""
    reverse = cumulative_trapezoid(source[::-1], -x[::-1], initial=0.0)[::-1]
    return -reverse / density


@dataclass(frozen=True)
class Equilibrium:
    """A 1D equilibrium on a uniform grid ``x`` (m): the coefficients of the eigenproblem.

    All fields are arrays on ``x``: ``density`` (m^-3) and its gradient (m^-4), ``e_field``
    ``E_0`` (V/m, signed along ``x``), ``b_field`` (T, along ``z``) and its gradient (T/m),
    ``electron_temperature`` (eV) and its gradient (eV/m), ``ionization_frequency``
    ``nu_iz = S / n`` (1/s), ``ion_flow`` ``v_0`` (m/s, along ``x``) and
    ``electron_collision_frequency`` (electron-neutral momentum transfer, 1/s). Build one
    with `boeuf`, `model_state` or `from_profiles`, and change it with `scaled` or
    `dataclasses.replace`.
    """

    x: np.ndarray
    density: np.ndarray
    density_gradient: np.ndarray
    e_field: np.ndarray
    b_field: np.ndarray
    b_field_gradient: np.ndarray
    electron_temperature: np.ndarray
    temperature_gradient: np.ndarray
    ionization_frequency: np.ndarray
    ion_flow: np.ndarray
    electron_collision_frequency: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        """Check the grid and broadcast scalars to it."""
        x = np.asarray(self.x, dtype=float)
        if x.ndim != 1 or x.size < 5:
            raise ValueError("x is a 1D grid of at least 5 points")
        h = np.diff(x)
        if np.any(h <= 0.0) or np.ptp(h) > 1e-6 * h.mean():
            raise ValueError("x must be uniform and increasing")
        object.__setattr__(self, "x", x)
        for name in (
            "density",
            "density_gradient",
            "e_field",
            "b_field",
            "b_field_gradient",
            "electron_temperature",
            "temperature_gradient",
            "ionization_frequency",
            "ion_flow",
            "electron_collision_frequency",
        ):
            values = np.broadcast_to(np.asarray(getattr(self, name), float), x.shape)
            object.__setattr__(self, name, values.copy())

    @classmethod
    def from_profiles(
        cls,
        x: np.ndarray,
        density: np.ndarray,
        e_field: np.ndarray | float,
        b_field: np.ndarray | float,
        electron_temperature: np.ndarray | float,
        ionization_frequency: np.ndarray | float = 0.0,
        electron_collision_frequency: np.ndarray | float = 0.0,
        density_gradient: np.ndarray | None = None,
        b_field_gradient: np.ndarray | None = None,
        temperature_gradient: np.ndarray | None = None,
        ion_flow: np.ndarray | None = None,
    ) -> Equilibrium:
        """An equilibrium from profiles on ``x`` (units as in the class docstring).

        Gradients not given are centred differences (`numpy.gradient`); the ion flow not
        given is the one fed by the source, ``v_0 = -int_x^L nu_iz n dx' / n`` (all ions
        born beyond ``x`` stream toward the cathode at ``x = 0``).
        """
        x = np.asarray(x, dtype=float)

        def on_grid(f: np.ndarray | float) -> np.ndarray:
            return np.broadcast_to(np.asarray(f, float), x.shape).copy()

        density, e_field, b_field = on_grid(density), on_grid(e_field), on_grid(b_field)
        te, nu_iz = on_grid(electron_temperature), on_grid(ionization_frequency)
        return cls(
            x=x,
            density=density,
            density_gradient=(
                np.gradient(density, x)
                if density_gradient is None
                else density_gradient
            ),
            e_field=e_field,
            b_field=b_field,
            b_field_gradient=(
                np.gradient(b_field, x)
                if b_field_gradient is None
                else b_field_gradient
            ),
            electron_temperature=te,
            temperature_gradient=(
                np.gradient(te, x)
                if temperature_gradient is None
                else temperature_gradient
            ),
            ionization_frequency=nu_iz,
            ion_flow=_ion_flow(x, nu_iz * density, density)
            if ion_flow is None
            else ion_flow,
            electron_collision_frequency=on_grid(electron_collision_frequency),
        )

    @classmethod
    def boeuf(
        cls,
        x: np.ndarray | None = None,
        electron_collision_frequency: float = 4.5e7,
        b_max: float = 0.12,
        b_length: float = 6e-3,
        smoothing: str = "pchip",
    ) -> Equilibrium:
        """Boeuf's DC magnetron state (`BOEUF_FIGURE_2`) on ``x`` (m, default 3-20 mm, 200 points).

        ``n_e``, ``E``, ``T_e`` and the source are interpolated (``smoothing="pchip"``,
        monotone, or ``"spline"``, a cubic spline), with their analytic derivatives; the
        field is his ``B = b_max exp(-x / b_length)`` (T, m); the ion flow is fed by the
        source through the ion density column. The electron-neutral collision frequency
        (1/s) is a constant: ``4.5e7`` is the momentum-transfer rate of argon at 0.6 Pa
        and 11 eV, Boeuf quotes ``2e7``.
        """
        if smoothing not in ("pchip", "spline"):
            raise ValueError("smoothing is 'pchip' or 'spline'")
        x = np.linspace(3e-3, 0.02, 200) if x is None else np.asarray(x, dtype=float)
        xd, ne, ni, ex, te, s = BOEUF_FIGURE_2.T
        xd = xd * 1e-2
        interp = PchipInterpolator if smoothing == "pchip" else CubicSpline
        n_sp, e_sp, t_sp = interp(xd, ne * 1e15), interp(xd, ex * 1e4), interp(xd, te)
        s_sp, ni_sp = interp(xd, s * 1e21), interp(xd, ni * 1e15)
        density = n_sp(x)
        source = np.clip(s_sp(x), 0.0, None)
        flow = -np.array([s_sp.integrate(xi, xd[-1]) for xi in x]) / ni_sp(x)
        b = b_max * np.exp(-x / b_length)
        return cls(
            x=x,
            density=density,
            density_gradient=n_sp.derivative()(x),
            e_field=e_sp(x),
            b_field=b,
            b_field_gradient=-b / b_length,
            electron_temperature=t_sp(x),
            temperature_gradient=t_sp.derivative()(x),
            ionization_frequency=source / density,
            ion_flow=flow,
            electron_collision_frequency=np.full_like(x, electron_collision_frequency),
        )

    @classmethod
    def model_state(
        cls,
        x: np.ndarray,
        peak_density: float,
        b_max: float,
        b_length: float,
        e_field: float,
        electron_temperature: float,
        density_length: float,
        peak_position: float = 3e-3,
        ionization_frequency: float = 1e5,
        electron_collision_frequency: float = 1e7,
        width: float = 0.5e-3,
    ) -> Equilibrium:
        """A model magnetron state: a density plateau and an exponential fall to the anode.

        ``n = peak_density`` (m^-3) up to ``peak_position`` (m) and
        ``exp(-(x - peak_position) / density_length)`` beyond (``density_length > 0``, m),
        blended over ``width`` (m); a field ``e_field`` (V/m, < 0) at ``x = 0`` falling
        linearly to zero at the anode ``x[-1]``; ``B = b_max exp(-x / b_length)``;
        isothermal ``electron_temperature`` (eV); a source ``nu_iz n`` and the ion flow it
        feeds. Not an equilibrium of any transport model: a state with the measured scales
        of a discharge, for the eigenproblem.
        """
        x = np.asarray(x, dtype=float)
        blend = 0.5 * (1.0 + np.tanh((x - peak_position) / width))
        tail = np.exp(-np.clip(x - peak_position, 0.0, None) / density_length)
        density = peak_density * ((1.0 - blend) + tail * blend)
        b = b_max * np.exp(-x / b_length)
        return cls.from_profiles(
            x,
            density,
            e_field * (1.0 - x / x[-1]),
            b,
            electron_temperature,
            ionization_frequency=ionization_frequency,
            electron_collision_frequency=electron_collision_frequency,
            b_field_gradient=-b / b_length,
        )

    def scaled(
        self,
        density: float = 1.0,
        b_field: float = 1.0,
        b_floor: float = 0.0,
        electron_collision_frequency: float = 1.0,
    ) -> Equilibrium:
        """The same state with the density, the field and the collisions scaled.

        The density (and the source with it, so ``nu_iz`` and the ion flow are unchanged)
        times ``density``; ``B`` times ``b_field`` plus a uniform ``b_floor`` (T), which
        weakens the relative field gradient; ``nu_en`` times ``electron_collision_frequency``.
        In the linear problem the density enters only through Poisson's equation, so
        scaling it by ``1 / s`` is scaling ``epsilon_0`` by ``s``.
        """
        return replace(
            self,
            density=self.density * density,
            density_gradient=self.density_gradient * density,
            b_field=self.b_field * b_field + b_floor,
            b_field_gradient=self.b_field_gradient * b_field,
            electron_collision_frequency=self.electron_collision_frequency
            * electron_collision_frequency,
        )

    @property
    def omega_ce(self) -> np.ndarray:
        """Electron cyclotron frequency (rad/s)."""
        return E * self.b_field / ELECTRON_MASS

    @property
    def gyroradius(self) -> np.ndarray:
        """Thermal electron gyroradius ``sqrt(T_e m_e) / (e B)`` (m)."""
        return np.sqrt(self.electron_temperature * E * ELECTRON_MASS) / (
            E * self.b_field
        )

    @property
    def debye_length(self) -> np.ndarray:
        """Electron Debye length (m)."""
        return np.sqrt(EPSILON_0 * self.electron_temperature / (E * self.density))

    @property
    def density_length(self) -> np.ndarray:
        """Signed density gradient length ``n / n'`` (m), negative where ``n`` falls."""
        with np.errstate(divide="ignore"):
            return self.density / self.density_gradient

    def to_dataset(self) -> xr.Dataset:
        """The profiles, labelled for plotting, on ``x``."""
        return xr.Dataset(
            {
                "density": variable("x", self.density, "$n_0$", "m^-3"),
                "e_field": variable("x", self.e_field, "$E_0$", "V/m"),
                "b_field": variable("x", self.b_field, "$B$", "T"),
                "electron_temperature": variable(
                    "x", self.electron_temperature, "$T_e$", "eV"
                ),
                "ionization_frequency": variable(
                    "x", self.ionization_frequency, r"$\nu_{iz}$", "1/s"
                ),
                "ion_flow": variable("x", self.ion_flow, "$v_{i0}$", "m/s"),
                "density_length": variable("x", self.density_length, "$L_n$", "m"),
                "gyroradius": variable("x", self.gyroradius, r"$\rho_e$", "m"),
                "debye_length": variable("x", self.debye_length, r"$\lambda_D$", "m"),
            },
            coords={"x": variable("x", self.x, "$x$", "m")},
        )


@dataclass(frozen=True)
class Eigenmodes:
    """Eigenvalues and eigenfunctions at one ``k_y``, the fastest growing first.

    ``omega`` (rad/s) has one entry per mode; ``potential``, ``electron_density`` and
    ``ion_density`` are ``(mode, x)``, normalised per mode so that
    ``e delta phi / T_e = 1`` (real) at the peak of ``|delta phi|``: ``potential`` is
    ``e delta phi / T_e(x_peak)`` and the densities are ``delta n / n_0`` on that scale
    (with ``gyroaverage``, ``electron_density`` is the gyrocentre density).
    ``omega_limit`` is the default bound of `drift` (``0.1 min omega_ce``).
    """

    ky: float
    equilibrium: Equilibrium = field(repr=False)
    omega: np.ndarray
    potential: np.ndarray = field(repr=False)
    electron_density: np.ndarray = field(repr=False)
    ion_density: np.ndarray = field(repr=False)
    omega_limit: float = np.inf

    def select(self, keep: np.ndarray) -> Eigenmodes:
        """The modes where ``keep`` (a mask or indices) holds, in that order."""
        return replace(
            self,
            omega=self.omega[keep],
            potential=self.potential[keep],
            electron_density=self.electron_density[keep],
            ion_density=self.ion_density[keep],
        )

    @property
    def roughness(self) -> np.ndarray:
        """``|| second difference of a_e || / || a_e ||`` per mode: near 2 for grid-scale modes."""
        a = self.electron_density
        norm = np.linalg.norm(a, axis=-1) + 1e-300
        return np.linalg.norm(np.diff(a, 2, axis=-1), axis=-1) / norm

    def drift(
        self, omega_max: float | None = None, roughness: float = 0.5
    ) -> Eigenmodes:
        """The drift branch: growing, ``|omega| < omega_max``, smooth; the fastest first.

        ``omega_max`` (rad/s) defaults to ``0.1 min omega_ce``: the drift ordering needs
        the frequency well below the gyration, which also drops the fast electron roots of
        the inertial model. Modes whose ``a_e`` is rougher than ``roughness`` (see
        `roughness`) are numerical, at the grid scale. A mode on a boundary (the peak of
        ``|delta phi|`` at the first or last interior point) is kept: check
        `peak_position` for wall-attached modes. Growing means ``Im omega`` above
        ``1e-9`` of the largest ``|omega|`` below ``omega_max``: below that it is the
        round-off of the eigensolver (about the machine precision times the spectrum's
        scale), whose sign depends on the LAPACK build.
        """
        limit = self.omega_limit if omega_max is None else omega_max
        inside = np.isfinite(self.omega) & (np.abs(self.omega) < limit)
        scale = np.max(np.abs(self.omega[inside]), initial=0.0)
        keep = (
            inside & (np.imag(self.omega) > 1e-9 * scale) & (self.roughness < roughness)
        )
        return self.select(np.flatnonzero(keep))

    @property
    def growth_rate(self) -> float:
        """``Im omega`` of the first mode (1/s); ``nan`` without modes."""
        return float(self.omega[0].imag) if self.omega.size else np.nan

    @property
    def phase_speed(self) -> float:
        """``Re omega / k_y`` of the first mode (m/s, along ``+y``); ``nan`` without modes."""
        return float(self.omega[0].real / self.ky) if self.omega.size else np.nan

    @property
    def peak_position(self) -> np.ndarray:
        """``x`` of the peak of ``|delta phi|`` per mode (m)."""
        return self.equilibrium.x[np.argmax(np.abs(self.potential), axis=-1)]

    @property
    def radial_wavenumber(self) -> np.ndarray:
        """``|| delta phi' || / || delta phi ||`` per mode (1/m): the radial scale."""
        phi = self.potential
        dphi = np.gradient(phi, self.equilibrium.x, axis=-1)
        return np.linalg.norm(dphi, axis=-1) / (np.linalg.norm(phi, axis=-1) + 1e-300)

    def to_dataset(self) -> xr.Dataset:
        """``omega`` and the eigenfunctions on ``(mode, x)``."""
        mode = np.arange(self.omega.size)
        x = self.equilibrium.x
        return xr.Dataset(
            {
                "growth_rate": variable(
                    "mode", np.imag(self.omega), r"$\gamma$", "1/s"
                ),
                "frequency": variable(
                    "mode", np.real(self.omega), r"$\omega_r$", "rad/s"
                ),
                "phase_speed": variable(
                    "mode", np.real(self.omega) / self.ky, "$v_{ph}$", "m/s"
                ),
                "potential": variable(
                    ("mode", "x"), self.potential, r"$e\delta\phi/T_e$"
                ),
                "electron_density": variable(
                    ("mode", "x"), self.electron_density, r"$\delta n_e/n_0$"
                ),
                "ion_density": variable(
                    ("mode", "x"), self.ion_density, r"$\delta n_i/n_0$"
                ),
            },
            coords={"mode": mode, "x": variable("x", x, "$x$", "m")},
            attrs={"ky": self.ky},
        )

    def __len__(self) -> int:
        """The number of modes."""
        return self.omega.size


def _derivative(n: int, h: float) -> np.ndarray:
    """Centred first derivative, one-sided at the ends."""
    d1 = (np.eye(n, k=1) - np.eye(n, k=-1)) / (2.0 * h)
    d1[0, :2] = (-1.0 / h, 1.0 / h)
    d1[-1, -2:] = (-1.0 / h, 1.0 / h)
    return d1


def _div_grad(p: np.ndarray, h: float) -> np.ndarray:
    """``d/dx (p d/dx .)`` with zero ghost values at both ends (face values averaged)."""
    faces = 0.5 * (p[1:] + p[:-1])
    minus = np.concatenate(([p[0]], faces))
    plus = np.concatenate((faces, [p[-1]]))
    return (np.diag(-(minus + plus)) + np.diag(faces, 1) + np.diag(faces, -1)) / h**2


def _upwind(v: np.ndarray, h: float) -> np.ndarray:
    """``v d/dx`` with first-order upwinding; zero inflow values."""
    positive = v >= 0.0
    lower = np.where(positive[1:], -v[1:], 0.0)
    upper = np.where(positive[:-1], 0.0, v[:-1])
    return (np.diag(np.abs(v)) + np.diag(lower, -1) + np.diag(upper, 1)) / h


@method(
    equations={
        "electron_continuity": r"-i\omega\, a_e + u_{0x} a_e' + i k_y u_{0y} a_e + \frac{1}{n_0}\left[\frac{d}{dx}(n_0\,\delta u_x) + i k_y n_0\,\delta u_y\right] = 0",
        "electron_momentum": r"m_e(-i\omega + \mathbf u_0\cdot\nabla + \nu_{en})\,\delta\mathbf u + m_e(\delta\mathbf u\cdot\nabla)\mathbf u_0 = e\nabla\psi - T_e\nabla a_e - e\,\delta\mathbf u\times\mathbf B \quad (\text{no inertia: the left side is } m_e\nu_{en}\delta\mathbf u)",
        "ion_continuity": r"-i\omega\, a_i + v_0 a_i' + \frac{1}{n_0}\frac{d}{dx}(n_0\,\delta v_x) + i k_y\,\delta v_y + \nu_{iz}(a_i - a_e) = 0",
        "ion_momentum": r"(-i\omega + v_0\partial_x + \nu_{in})\,\delta\mathbf v + v_0'\,\delta v_x\hat x = -\frac{e}{M}\nabla\delta\phi",
        "poisson": r"\left(\frac{d^2}{dx^2} - k_y^2\right)\delta\phi = -\frac{e n_0}{\varepsilon_0}\left[a_i - \Gamma_0^{1/2} a_e - (1 - \Gamma_0)\frac{e\,\delta\phi}{T_e}\right] \quad \text{or quasineutral } [\ldots] = 0",
        "gyroaverage": r"\psi = \Gamma_0^{1/2}\delta\phi, \quad \Gamma_0 = (1 + b)^{-1}, \quad \Gamma_0^{1/2} = (1 + b/2)^{-1}, \quad b = k_y^2\rho_e^2 - \frac{d}{dx}\rho_e^2\frac{d}{dx} \quad (\text{fluid: } \Gamma_0 = 1)",
        "equilibrium_flow": r"\mathbf u_0 = \frac{m_e\nu_{en}\,\mathbf F_0 + e B\,\mathbf F_0\times\hat z}{(m_e\nu_{en})^2 + (eB)^2}, \quad F_{0x} = -eE_0 - \frac{T_e n_0'}{n_0} - T_e'",
    },
    assumptions=(
        r"slab $(x, y)$ with the model field $B(x)\hat z$, perturbations $f(x)\,e^{i k_y y - i\omega t}$; `curl_free_field` adds $-(B'/B)\,n_0\,\delta u_{x,E\times B}$ to the electron flux divergence",
        r"isothermal electrons; the ionisation source removed with the equilibrium continuity (it appears only as $\nu_{iz}(a_i - a_e)$); $\delta\nu_{iz} = 0$",
        r"cold unmagnetised ions with the flow $v_0 = -\int_x^L S\,dx'/n_i$",
        r"`gyroaverage`: Padé $\Gamma_0$ of linear gyrokinetics at $k_\parallel = 0$, $\omega \ll \omega_{ce}$; the magnetic drift in its fluid form",
        r"second-order differences on a uniform grid, $a_e = a_i = \delta\phi = 0$ at both ends, first-order upwinding of the flows; dense eigensolve",
    ),
    references=("[Boeuf, arXiv:2211.17051](https://arxiv.org/abs/2211.17051)",),
)
@dataclass(frozen=True)
class NonlocalDispersion:
    """The linear eigenproblem on a 1D ``equilibrium`` and the options of the model.

    ``inertia`` keeps the electron inertia (six unknowns per point instead of four);
    ``electron_collisions=False`` drops the electron-neutral collisions (and the Pedersen
    flow); ``ion_flow`` and ``ionization`` switch the ion flow and the ionisation coupling;
    ``ion_collision_frequency`` (1/s) damps the ion momentum; ``curl_free_field`` adds the
    vacuum field's compressibility; ``quasineutral`` replaces Poisson's equation;
    ``gyroaverage`` uses the Padé gyro-averaged electron response, with ``rho_e`` scaled by
    ``gyroradius_scale`` (``-> 0`` recovers the fluid); ``ion_mass`` (kg).
    """

    equilibrium: Equilibrium
    inertia: bool = False
    electron_collisions: bool = True
    ion_flow: bool = True
    ionization: bool = True
    ion_collision_frequency: float = 0.0
    curl_free_field: bool = False
    quasineutral: bool = False
    gyroaverage: bool = False
    gyroradius_scale: float = 1.0
    ion_mass: float = ARGON_MASS

    @property
    def unknowns(self) -> tuple[str, ...]:
        """The unknowns per grid point, in the order of the state vector."""
        base = ("a_e", "a_i", "v_x", "v_y")
        return base + ("u_x", "u_y") if self.inertia else base

    def _operators(
        self, ky: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(S, F, C, D)``: ``-i omega u = S u + F phi`` and the closure ``C u + D phi = 0``."""
        eq = self.equilibrium
        x = eq.x
        n = x.size
        h = x[1] - x[0]
        d1 = _derivative(n, h)
        eye = np.eye(n)
        n0, b = eq.density, eq.b_field
        t0 = eq.electron_temperature * E  # J
        dt0 = eq.temperature_gradient * E
        nu = eq.electron_collision_frequency * float(self.electron_collisions)
        nu_iz = eq.ionization_frequency * float(self.ionization)
        v0 = eq.ion_flow * float(self.ion_flow)
        inv_n0 = np.diag(1.0 / n0)
        # the gyro-average: Gh = Gamma_0^(1/2), and 1 - Gamma_0 for the polarisation density
        if self.gyroaverage:
            rho2 = self.gyroradius_scale**2 * t0 * ELECTRON_MASS / (E * b) ** 2
            b_op = ky**2 * np.diag(rho2) - _div_grad(rho2, h)
            gh = np.linalg.inv(eye + 0.5 * b_op)
            polarization = b_op @ np.linalg.inv(eye + b_op)
        else:
            gh = eye
            polarization = np.zeros((n, n))
        # equilibrium electron flow (drift ordering) and the perpendicular mobilities
        f0x = -E * eq.e_field - t0 * eq.density_gradient / n0 - dt0
        den = (ELECTRON_MASS * nu) ** 2 + (E * b) ** 2
        c1, c2 = ELECTRON_MASS * nu / den, E * b / den  # Pedersen, Hall
        u0x, u0y = c1 * f0x, c2 * f0x
        advect = _upwind(u0x, h) + 1j * ky * np.diag(u0y)
        # the perturbed force per electron, F = e grad psi - T grad a, split into a and psi
        fx_a, fx_psi = -np.diag(t0) @ d1, E * d1
        fy_a, fy_psi = -1j * ky * np.diag(t0), 1j * ky * E * eye
        blocks = len(self.unknowns)
        s = np.zeros((blocks * n, blocks * n), dtype=complex)
        f = np.zeros((blocks * n, n), dtype=complex)
        sl = [slice(k * n, (k + 1) * n) for k in range(blocks)]
        ae, ai, vx, vy = sl[:4]
        if not self.inertia:
            # d/dx(n0 du_x) in conservative form: du_x = c1 F_x - c2 F_y
            exb = -1j * ky * d1 @ np.diag(n0 * c2)  # acts on (e psi - T a)
            if self.curl_free_field:
                exb = exb + 1j * ky * np.diag(eq.b_field_gradient / b * n0 * c2)
            kx_a = -_div_grad(n0 * c1 * t0, h) - exb @ np.diag(t0)
            kx_psi = E * _div_grad(n0 * c1, h) + E * exb
            # i k_y n0 du_y, du_y = c1 F_y + c2 F_x
            ky_a = 1j * ky * np.diag(n0) @ (np.diag(c1) @ fy_a + np.diag(c2) @ fx_a)
            ky_psi = (
                1j * ky * np.diag(n0) @ (np.diag(c1) @ fy_psi + np.diag(c2) @ fx_psi)
            )
            s[ae, ae] = -advect - inv_n0 @ (kx_a + ky_a)
            f[ae] = -inv_n0 @ (kx_psi + ky_psi) @ gh
        else:
            ux, uy = sl[4], sl[5]
            s[ae, ae] = -advect
            s[ae, ux] = -inv_n0 @ d1 @ np.diag(n0)
            s[ae, uy] = -1j * ky * eye
            s[ux, ae] = fx_a / ELECTRON_MASS
            f[ux] = fx_psi @ gh / ELECTRON_MASS
            s[ux, ux] = -advect - np.diag(d1 @ u0x) - np.diag(nu)
            s[ux, uy] = -np.diag(eq.omega_ce)
            s[uy, ae] = fy_a / ELECTRON_MASS
            f[uy] = fy_psi @ gh / ELECTRON_MASS
            s[uy, ux] = np.diag(eq.omega_ce) - np.diag(d1 @ u0y)
            s[uy, uy] = -advect - np.diag(nu)
        # cold ions with the flow v0, the source coupling and charge exchange
        ion_advect = _upwind(v0, h)
        nu_in = self.ion_collision_frequency
        s[ai, ai] = -ion_advect - np.diag(nu_iz)
        s[ai, ae] = np.diag(nu_iz)
        s[ai, vx] = -inv_n0 @ d1 @ np.diag(n0)
        s[ai, vy] = -1j * ky * eye
        s[vx, vx] = -ion_advect - np.diag(d1 @ v0) - nu_in * eye
        s[vy, vy] = -ion_advect - nu_in * eye
        f[vx] = -(E / self.ion_mass) * d1
        f[vy] = -(E / self.ion_mass) * 1j * ky * eye
        # the closure: Poisson (charge density) or quasineutrality, C u + D phi = 0
        c = np.zeros((n, blocks * n), dtype=complex)
        c[:, ai] = eye
        c[:, ae] = -gh
        d = -np.diag(E / t0) @ polarization
        if not self.quasineutral:
            laplace = _div_grad(np.ones(n), h) - ky**2 * eye
            c = np.diag(E * n0 / EPSILON_0) @ c
            d = laplace + np.diag(E * n0 / EPSILON_0) @ d
        return s, f, c, d

    def _system(self, ky: float) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """``(A, B, Phi)``: the pencil of `matrices` and ``delta phi = Phi u``."""
        s, f, c, d = self._operators(ky)
        if not self.quasineutral:
            potential = -np.linalg.solve(d, c)
            return s + f @ potential, None, potential
        size, n = s.shape[0], d.shape[0]
        a = np.block([[s, f], [c, d]])
        b = np.diag(np.concatenate((np.ones(size), np.zeros(n)))).astype(complex)
        potential = np.zeros((n, size + n), dtype=complex)
        potential[:, size:] = np.eye(n)
        return a, b, potential

    def matrices(self, ky: float) -> tuple[np.ndarray, np.ndarray | None]:
        """``(A, B)`` of ``A u = -i omega B u`` at ``k_y`` (1/m); ``B`` is ``None`` if standard.

        With Poisson's equation ``delta phi`` is eliminated and ``u`` holds the blocks of
        `unknowns`, each on the grid; quasineutral, ``delta phi`` is an extra last block
        and ``B`` is singular there (a generalised problem with infinite eigenvalues).
        """
        a, b, _ = self._system(ky)
        return a, b

    def solve(self, ky: float) -> Eigenmodes:
        """All finite eigenvalues and their eigenfunctions at ``k_y`` (1/m), fastest first."""
        eq = self.equilibrium
        n = eq.x.size
        a, b, potential = self._system(ky)
        lam, vec = linalg.eig(a, b)
        with np.errstate(invalid="ignore"):
            omega = 1j * lam
        finite = np.isfinite(omega)
        order = np.flatnonzero(finite)[np.argsort(-omega[finite].imag)]
        omega, vec = omega[order], vec[:, order]
        phi = (potential @ vec).T
        a_e, a_i = vec[:n].T, vec[n : 2 * n].T
        # per mode: e delta phi / T_e = 1 (real) at the peak of |delta phi|
        peak = np.argmax(np.abs(phi), axis=-1)
        scale = 1.0 / phi[np.arange(omega.size), peak]
        te_peak = eq.electron_temperature[peak]
        return Eigenmodes(
            ky=float(ky),
            equilibrium=eq,
            omega=omega,
            potential=phi * scale[:, None],
            electron_density=a_e * (scale * te_peak)[:, None],
            ion_density=a_i * (scale * te_peak)[:, None],
            omega_limit=0.1 * float(eq.omega_ce.min()),
        )

    def scan(
        self,
        ky: np.ndarray,
        omega_max: float | None = None,
        roughness: float = 0.5,
    ) -> xr.Dataset:
        """The fastest mode of the drift branch (`Eigenmodes.drift`) over ``k_y`` (1/m).

        Returns, on ``ky`` (with the ``wavelength`` ``2 pi / k_y`` as a coordinate):
        ``growth_rate`` (1/s), ``frequency`` (rad/s), ``phase_speed`` (m/s), the number of
        drift-branch ``modes``, and at the peak of ``|delta phi|``: ``peak_position`` (m),
        ``radial_wavenumber`` (1/m, `Eigenmodes.radial_wavenumber`), ``ky_gyroradius``
        ``k_y rho_e``, ``ky_debye_length`` ``k_y lambda_D`` and ``density_length`` (m);
        and the normalised eigenfunctions ``potential`` and ``electron_density`` on
        ``(ky, x)`` (complex). Without a growing mode the values are ``nan`` (zero modes).
        """
        eq = self.equilibrium
        ky = np.atleast_1d(np.asarray(ky, dtype=float))
        names = (
            "growth_rate",
            "frequency",
            "peak_position",
            "radial_wavenumber",
            "ky_gyroradius",
            "ky_debye_length",
            "density_length",
        )
        values = {name: np.full(ky.size, np.nan) for name in names}
        count = np.zeros(ky.size, dtype=int)
        potential = np.full((ky.size, eq.x.size), np.nan + 0j)
        density = np.full((ky.size, eq.x.size), np.nan + 0j)
        for j, k in enumerate(ky):
            modes = self.solve(float(k)).drift(omega_max, roughness)
            count[j] = len(modes)
            if not len(modes):
                continue
            peak = int(np.argmax(np.abs(modes.potential[0])))
            values["growth_rate"][j] = modes.omega[0].imag
            values["frequency"][j] = modes.omega[0].real
            values["peak_position"][j] = eq.x[peak]
            values["radial_wavenumber"][j] = modes.radial_wavenumber[0]
            values["ky_gyroradius"][j] = k * eq.gyroradius[peak]
            values["ky_debye_length"][j] = k * eq.debye_length[peak]
            values["density_length"][j] = eq.density_length[peak]
            potential[j] = modes.potential[0]
            density[j] = modes.electron_density[0]
        labels = {
            "growth_rate": (r"$\gamma$", "1/s"),
            "frequency": (r"$\omega_r$", "rad/s"),
            "peak_position": (r"$x_{\rm peak}$", "m"),
            "radial_wavenumber": (r"$k_{x,\rm eff}$", "1/m"),
            "ky_gyroradius": (r"$k_y\rho_e$", ""),
            "ky_debye_length": (r"$k_y\lambda_D$", ""),
            "density_length": (r"$L_n$", "m"),
        }
        data = {name: variable("ky", values[name], *labels[name]) for name in names}
        data["phase_speed"] = variable(
            "ky", values["frequency"] / ky, "$v_{ph}$", "m/s"
        )
        data["modes"] = variable("ky", count, "growing modes")
        data["potential"] = variable(("ky", "x"), potential, r"$e\delta\phi/T_e$")
        data["electron_density"] = variable(("ky", "x"), density, r"$\delta n_e/n_0$")
        return xr.Dataset(
            data,
            coords={
                "ky": variable("ky", ky, "$k_y$", "1/m"),
                "wavelength": variable("ky", 2 * np.pi / ky, r"$\lambda$", "m"),
                "x": variable("x", eq.x, "$x$", "m"),
            },
        )
