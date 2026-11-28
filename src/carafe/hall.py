"""A 1D axial model of a Hall thruster: neutrals, ions with momentum, quasineutral electrons.

The discharge channel and the near plume along the thruster axis ``z``, the anode at
``z = 0``, the channel exit at ``z = L_ch`` and the cathode line at ``z_c``. The
model is the quasineutral fluid model of Boeuf and Garrigues (1998) in the form of
HallThruster.jl (Marks and Jorns 2023) and of the LANDMARK 1D fluid test case:

- neutrals at a fixed axial speed ``u_n``, ionized by the electrons::

      dn_n/dt + u_n dn_n/dz = -n_e n_n k_iz(T_e)

- singly charged ions, isothermal at ``T_i``, accelerated by the axial field::

      dn_i/dt + d(n_i u_i)/dz = n_e n_n k_iz
      d(n_i u_i)/dt + d(n_i u_i^2 + n_i e T_i / M)/dz = (e / M) n_i E + n_e n_n k_iz u_n

- quasineutral electrons (``n_e = n_i``) whose axial flux follows the cross-field
  Ohm's law with the mobility ``mu = (e / m nu) / (1 + Omega^2)``, ``Omega = e B / m nu``,
  and ``nu`` the classical (electron-neutral and electron-ion) plus the anomalous
  collision frequency ``nu_ano`` (a closure: `TwoZoneBohm` or any callable of an
  `ElectronContext`)::

      n_e u_e = -mu n_e (E + (1 / n_e) d(n_e T_e)/dz)

- current continuity: the discharge current density ``J = e n (u_i - u_e)`` is uniform
  in ``z``, and ``int_0^z_c E dz = V_d`` fixes it::

      J = e [V_d + int (u_i / mu + (1 / n) d(n T_e)/dz) dz] / int dz / (n mu)

- the electron energy ``(3/2) n T_e``, with the enthalpy flux, the conduction
  ``q = -(5/2) mu n T_e dT_e/dz``, the ohmic heating ``-n u_e E``, the ionization,
  excitation and elastic losses and the wall losses ``n W(T_e)`` (`ExponentialWallLoss`)::

      d(3/2 n T_e)/dt + d(5/2 n u_e T_e + q)/dz = -n u_e E - n n_n sum_j k_j eps_j - n W

Discretisation: finite volumes on uniform cells. The ions take a MUSCL reconstruction
of ``n`` and ``u`` (`carafe.limiters`) and the Rusanov flux with the ion acoustic
speed, the neutrals the upwind flux, both with the second-order SSP Runge-Kutta scheme
at a fixed field and ionization frequency over a step (explicit, CFL-limited). The
electron energy is implicit (backward Euler, upwind enthalpy flux), so the step is not
limited by the conduction. The anode takes the ions leaving at the Bohm speed or
faster and returns them as neutrals; the electron temperature is fixed at the anode
and the cathode end. Every step conserves the heavy particles to round-off: the
anode inflow ``mdot / M`` minus what leaves through the outer boundary.

Xenon data: the rate coefficients are PLACEHOLDER fits (`xenon_reactions`, Goebel and
Katz 2008, appendix E), to be replaced by plasmacoll's xenon cross sections through
`carafe.reactions.Reaction.from_process` once plasmacoll has them. The wall loss is the
LANDMARK / HallThruster.jl exponential form, a PLACEHOLDER for a sheath model with
secondary emission.

Not yet: ion losses to the walls, double ions, an anode sheath, a cathode in the domain
with a plume beyond it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from methodcard import method
from plasmacoll.constants import ELECTRON_MASS, ELEMENTARY_CHARGE
from plasmacoll.units import convert_units, unit_field
from scipy.linalg import solve_banded
from scope_profiler import profile

from carafe._profiling import _PROFILE_TAGS
from carafe.datasets import variable
from carafe.limiters import LIMITERS
from carafe.reactions import Reaction
from carafe.species import Species

__all__ = [
    "XENON",
    "XENON_ION",
    "ElectronContext",
    "ExponentialWallLoss",
    "GaussianField",
    "HallState",
    "HallThruster1D",
    "TwoZoneBohm",
    "xenon_reactions",
]

E = ELEMENTARY_CHARGE
G0 = 9.80665
#: The xenon atom and its singly charged ion.
XENON = Species.atomic("Xe", 131.293)
XENON_ION = Species.atomic("Xe+", 131.293, 1)
#: Floors of the plasma density (m^-3) and the electron temperature (eV).
DENSITY_FLOOR = 1.0e12
TE_FLOOR = 0.1


# -- PLACEHOLDER xenon data ---------------------------------------------------------
# Maxwellian rate coefficients of electron impact on xenon, the fits of Goebel and Katz,
# "Fundamentals of Electric Propulsion" (2008), appendix E. They stand in for
# plasmacoll's xenon cross sections (Reaction.from_process) until plasmacoll has xenon;
# swap them by passing other `reactions` to HallThruster1D.

#: Ionization energy of xenon (eV).
XENON_IONIZATION_ENERGY = 12.13
#: Lumped excitation energy of xenon (eV), Goebel and Katz's ``U*``.
XENON_EXCITATION_ENERGY = 8.32


def _mean_speed(te: np.ndarray | float) -> np.ndarray:
    """The mean speed of Maxwellian electrons at ``te`` (eV), m/s."""
    return np.sqrt(8.0 * E * np.asarray(te, dtype=float) / (np.pi * ELECTRON_MASS))


def xenon_ionization_rate(te: np.ndarray | float) -> np.ndarray:
    """``k_iz(T_e)`` of ``e + Xe -> 2e + Xe+`` (m^3/s), PLACEHOLDER (Goebel and Katz)."""
    te = np.maximum(np.asarray(te, dtype=float), TE_FLOOR)
    sigma = (-1.031e-4 * te**2 + 6.386 * np.exp(-12.127 / te)) * 1.0e-20
    return np.maximum(sigma, 0.0) * _mean_speed(te)


def xenon_excitation_rate(te: np.ndarray | float) -> np.ndarray:
    """``k_ex(T_e)`` of the lumped excitation of xenon (m^3/s), PLACEHOLDER (Goebel and Katz)."""
    te = np.maximum(np.asarray(te, dtype=float), TE_FLOOR)
    return 1.931e-19 * np.exp(-11.6 / te) / np.sqrt(te) * _mean_speed(te)


def xenon_elastic_rate(te: np.ndarray | float) -> np.ndarray:
    """``k_el(T_e)`` of elastic electron-xenon collisions (m^3/s), PLACEHOLDER (Goebel and Katz)."""
    te = np.maximum(np.asarray(te, dtype=float), TE_FLOOR)
    quarter = te / 4.0
    sigma = 6.6e-19 * (quarter - 0.1) / (1.0 + quarter**1.6)
    return np.maximum(sigma, 1.0e-21) * _mean_speed(te)


def xenon_reactions() -> dict[str, Reaction]:
    """The PLACEHOLDER xenon reactions of the model: ionization, excitation, elastic."""
    return {
        "ionization": Reaction(
            ("e", "Xe"),
            ("e", "e", "Xe+"),
            xenon_ionization_rate,
            electron_energy_loss=XENON_IONIZATION_ENERGY,
        ),
        "excitation": Reaction(
            ("e", "Xe"),
            ("e", "Xe*"),
            xenon_excitation_rate,
            electron_energy_loss=XENON_EXCITATION_ENERGY,
        ),
        "elastic": Reaction(
            ("e", "Xe"), ("e", "Xe"), xenon_elastic_rate, elastic_mass=XENON.mass
        ),
    }


# -- closures and the field -----------------------------------------------------------


@dataclass(frozen=True)
class ElectronContext:
    """What a closure of the electrons sees, on the cells.

    Attributes:
        z: The cell centres (m).
        magnetic_field: The radial field ``B`` (T).
        density: The plasma density ``n_e = n_i`` (m^-3).
        electron_temperature: ``T_e`` (eV).
        ion_velocity: ``u_i`` (m/s).
        neutral_density: ``n_n`` (m^-3).
        channel_length: ``L_ch`` (m), where the channel ends.
    """

    z: np.ndarray
    magnetic_field: np.ndarray
    density: np.ndarray
    electron_temperature: np.ndarray
    ion_velocity: np.ndarray
    neutral_density: np.ndarray
    channel_length: float

    @property
    def inside(self) -> np.ndarray:
        """Whether a cell is inside the channel."""
        return self.z < self.channel_length

    @property
    def cyclotron_frequency(self) -> np.ndarray:
        """The electron cyclotron frequency ``e B / m_e`` (rad/s)."""
        return E * self.magnetic_field / ELECTRON_MASS


#: A closure: the anomalous collision frequency (1/s), or the wall loss (eV/s per electron).
Closure = Callable[[ElectronContext], np.ndarray]


@dataclass(frozen=True)
class TwoZoneBohm:
    """The two-zone Bohm closure ``nu_ano = c omega_ce``, ``c`` inside and outside the channel.

    Attributes:
        inside: ``c`` in the channel (HallThruster.jl's default 1/160).
        outside: ``c`` beyond the exit (1/16, Bohm's value).
        transition: The width (m) of a smooth step between them (a ``tanh``); 0 is a jump.
    """

    inside: float = 1.0 / 160.0
    outside: float = 1.0 / 16.0
    transition: float = 1.0e-3

    def __call__(self, context: ElectronContext) -> np.ndarray:
        """``nu_ano`` on the cells (1/s)."""
        offset = context.z - context.channel_length
        if self.transition > 0.0:
            share = 0.5 * (1.0 + np.tanh(offset / self.transition))
        else:
            share = (offset >= 0.0).astype(float)
        coefficient = self.inside + (self.outside - self.inside) * share
        return coefficient * context.cyclotron_frequency


@dataclass(frozen=True)
class ExponentialWallLoss:
    """The electron energy lost to the channel walls, ``W = a nu_w T_e exp(-U / T_e)``.

    The form of the LANDMARK 1D fluid test case and of HallThruster.jl's
    ``ConstantSheathPotential`` (eV/s per electron). PLACEHOLDER for a sheath model with
    secondary emission (the 1D PIC of the wall sheath).

    Attributes:
        frequency: ``nu_w`` (1/s).
        energy: ``U`` (eV), the sheath potential the electrons must cross.
        inside: ``a`` in the channel.
        outside: ``a`` beyond the exit.
    """

    frequency: float = 1.0e7
    energy: float = 20.0
    inside: float = 1.0
    outside: float = 1.0

    def __call__(self, context: ElectronContext) -> np.ndarray:
        """``W`` on the cells (eV/s per electron)."""
        te = np.maximum(context.electron_temperature, TE_FLOOR)
        coefficient = np.where(context.inside, self.inside, self.outside)
        return coefficient * self.frequency * te * np.exp(-self.energy / te)


@dataclass(frozen=True)
class GaussianField:
    """A radial field peaking at ``location``, Gaussian with its width on either side.

    Attributes:
        peak: The largest field (T).
        location: Where it peaks (m), near the channel exit.
        width_inside: The Gaussian width towards the anode (m).
        width_outside: The width towards the plume (m).
    """

    peak: float = 0.02
    location: float = 0.025
    width_inside: float = 0.011
    width_outside: float = 0.018

    def __call__(self, z: np.ndarray) -> np.ndarray:
        """``B(z)`` (T)."""
        z = np.asarray(z, dtype=float)
        width = np.where(z < self.location, self.width_inside, self.width_outside)
        return self.peak * np.exp(-0.5 * ((z - self.location) / width) ** 2)


# -- the state --------------------------------------------------------------------------


@dataclass
class HallState:
    """The thruster at one time, and the history of the run that led to it.

    Attributes:
        time: The time (s).
        z: The cell centres (m).
        neutral_density: ``n_n`` (m^-3).
        ion_density: ``n_i = n_e`` (m^-3).
        ion_velocity: ``u_i`` (m/s).
        electron_temperature: ``T_e`` (eV).
        potential: ``phi`` (V), ``V_d`` at the anode and 0 at the cathode line.
        electric_field: ``E = -d phi / dz`` (V/m).
        anomalous_frequency: ``nu_ano`` (1/s).
        magnetic_field: ``B`` (T).
        discharge_current: ``I_d`` (A).
        thrust: The ions' momentum flux through the outer boundary times the area (N).
        ion_current: The ion current through the outer boundary (A).
        history: The run's samples: ``time``, ``discharge_current``, ``ion_current``
            and ``thrust``, arrays.
        mass_flow: The anode mass flow (kg/s).
        discharge_voltage: ``V_d`` (V).
    """

    time: float
    z: np.ndarray
    neutral_density: np.ndarray
    ion_density: np.ndarray
    ion_velocity: np.ndarray
    electron_temperature: np.ndarray
    potential: np.ndarray
    electric_field: np.ndarray
    anomalous_frequency: np.ndarray
    magnetic_field: np.ndarray
    discharge_current: float
    thrust: float
    ion_current: float
    mass_flow: float
    discharge_voltage: float
    history: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def specific_impulse(self) -> float:
        """``I_sp = T / (mdot g0)`` (s)."""
        return self.thrust / (self.mass_flow * G0)

    @property
    def anode_efficiency(self) -> float:
        """``T^2 / (2 mdot I_d V_d)``."""
        power = self.discharge_current * self.discharge_voltage
        return self.thrust**2 / (2.0 * self.mass_flow * power)

    def average(self, name: str, start: float = 0.5) -> float:
        """The mean of a history entry over the run after ``start`` (a fraction of it)."""
        times = self.history["time"]
        keep = times >= times[0] + start * (times[-1] - times[0])
        return float(np.mean(self.history[name][keep]))

    def to_dataset(self) -> xr.Dataset:
        """The profiles on ``z`` (mm), the history on ``time`` (us) and the performance."""
        coords = {
            "z": ("z", self.z * 1e3, {"long_name": "$z$", "units": "mm"}),
        }
        data = {
            "neutral_density": variable(
                "z", self.neutral_density, "neutral density $n_n$", "m^-3"
            ),
            "ion_density": variable(
                "z", self.ion_density, "plasma density $n_i$", "m^-3"
            ),
            "ion_velocity": variable(
                "z", self.ion_velocity * 1e-3, "ion velocity $u_i$", "km/s"
            ),
            "electron_temperature": variable(
                "z", self.electron_temperature, "electron temperature $T_e$", "eV"
            ),
            "potential": variable("z", self.potential, r"potential $\phi$", "V"),
            "electric_field": variable(
                "z", self.electric_field * 1e-3, "electric field $E_z$", "kV/m"
            ),
            "anomalous_frequency": variable(
                "z", self.anomalous_frequency, r"$\nu_{ano}$", "s^-1"
            ),
            "magnetic_field": variable(
                "z", self.magnetic_field * 1e3, "magnetic field $B_r$", "mT"
            ),
            "discharge_current": variable(
                (), self.discharge_current, "discharge current $I_d$", "A"
            ),
            "thrust": variable((), self.thrust * 1e3, "thrust $T$", "mN"),
            "ion_current": variable((), self.ion_current, "ion current $I_i$", "A"),
            "specific_impulse": variable(
                (), self.specific_impulse, r"specific impulse $I_{sp}$", "s"
            ),
            "anode_efficiency": variable(
                (), self.anode_efficiency, r"anode efficiency $\eta_a$"
            ),
        }
        if self.history:
            coords["time"] = (
                "time",
                self.history["time"] * 1e6,
                {"long_name": "$t$", "units": "us"},
            )
            data["discharge_current_history"] = variable(
                "time",
                self.history["discharge_current"],
                "discharge current $I_d$",
                "A",
            )
            data["ion_current_history"] = variable(
                "time", self.history["ion_current"], "ion current $I_i$", "A"
            )
            data["thrust_history"] = variable(
                "time", self.history["thrust"] * 1e3, "thrust $T$", "mN"
            )
        return xr.Dataset(
            data,
            coords=coords,
            attrs={
                "time": self.time,
                "mass_flow": self.mass_flow,
                "discharge_voltage": self.discharge_voltage,
            },
        )


# -- the model ----------------------------------------------------------------------------


@method(
    equations={
        "neutrals": r"\partial_t n_n + u_n \partial_z n_n = -n_e n_n k_{iz}(T_e)",
        "ion_continuity": r"\partial_t n_i + \partial_z (n_i u_i) = n_e n_n k_{iz}",
        "ion_momentum": r"\partial_t (n_i u_i) + \partial_z (n_i u_i^2 + n_i e T_i / M) = \frac{e}{M} n_i E + n_e n_n k_{iz} u_n",
        "ohm": r"n_e u_e = -\mu_\perp n_e \left(E + \frac{1}{n_e} \partial_z (n_e T_e)\right), \qquad \mu_\perp = \frac{e}{m_e \nu} \frac{1}{1 + \Omega_e^2}, \quad \nu = \nu_{en} + \nu_{ei} + \nu_{ano}",
        "current": r"J = e n (u_i - u_e) = \text{const}, \qquad \int_0^{z_c} E\, dz = V_d",
        "energy": r"\partial_t \left(\tfrac{3}{2} n T_e\right) + \partial_z \left(\tfrac{5}{2} n u_e T_e - \tfrac{5}{2} \mu_\perp n T_e \partial_z T_e\right) = -n u_e E - n n_n \sum_j k_j \varepsilon_j - n W(T_e)",
        "performance": r"T = M A n_i u_i^2 \big|_{\text{exit}}, \quad I_{sp} = \frac{T}{\dot m g_0}, \quad \eta_a = \frac{T^2}{2 \dot m I_d V_d}",
    },
    assumptions=(
        "one dimension along the axis, quasineutral, singly charged ions isothermal at $T_i$",
        "neutrals at a fixed speed; the ions reaching the anode return as neutrals",
        r"the anomalous collision frequency a closure, two-zone Bohm by default",
        r"xenon rate coefficients and the wall loss $W$ are PLACEHOLDERS (Goebel and Katz fits; the LANDMARK exponential form)",
    ),
    references=(
        "J.-P. Boeuf and L. Garrigues, J. Appl. Phys. 84, 3541 (1998)",
        "T. A. Marks and B. A. Jorns, HallThruster.jl, J. Open Source Softw. 8, 4672 (2023)",
        "D. M. Goebel and I. Katz, Fundamentals of Electric Propulsion (2008), appendix E",
    ),
)
@dataclass
class HallThruster1D:
    """A 1D axial Hall thruster model; see the module docstring.

    Attributes:
        discharge_voltage: ``V_d`` between anode and cathode (V).
        mass_flow: The anode mass flow of xenon (kg/s).
        channel_length: ``L_ch``, from the anode to the exit (m).
        length: The domain, from the anode into the plume (m).
        area: The channel cross section (m^2); the SPT-100's annulus by default.
        cathode: ``z_c``, the cathode line (m); the domain end if ``None``.
        cells: The number of cells.
        magnetic_field: ``B(z)`` (T), a callable of the positions.
        anomalous: ``nu_ano`` (1/s), a callable of an `ElectronContext`.
        wall_loss: ``W`` (eV/s per electron), a callable of an `ElectronContext`.
        reactions: ``{"ionization", "excitation", "elastic"}`` reactions of the
            electrons with the gas (`xenon_reactions` by default).
        neutral_speed: ``u_n`` (m/s).
        ion_temperature: ``T_i`` (eV).
        anode_temperature: ``T_e`` at the anode (eV).
        cathode_temperature: ``T_e`` at the domain end (eV).
        limiter: The MUSCL limiter of the ions (a key of `carafe.limiters.LIMITERS`).
        cfl: The Courant number of the ion step.

    The physical settings take astropy Quantities or strings such as ``"5 mg/s"``.
    """

    discharge_voltage: float = unit_field("V", 300.0)
    mass_flow: float = unit_field("kg / s", 5.0e-6)
    channel_length: float = unit_field("m", 0.025)
    length: float = unit_field("m", 0.05)
    area: float = unit_field("m^2", np.pi * (0.05**2 - 0.0345**2))
    cathode: float | None = unit_field("m", None)
    cells: int = 100
    magnetic_field: Callable[[np.ndarray], np.ndarray] = field(
        default_factory=GaussianField
    )
    anomalous: Closure = field(default_factory=TwoZoneBohm)
    wall_loss: Closure = field(default_factory=ExponentialWallLoss)
    reactions: Mapping[str, Reaction] = field(default_factory=xenon_reactions)
    neutral_speed: float = unit_field("m / s", 150.0)
    ion_temperature: float = unit_field("eV", 0.1)
    anode_temperature: float = unit_field("eV", 3.0)
    cathode_temperature: float = unit_field("eV", 3.0)
    limiter: str = "minmod"
    cfl: float = 0.7

    def __post_init__(self) -> None:
        """Check the settings and build the mesh."""
        convert_units(self)
        if self.cathode is None:
            self.cathode = self.length
        if not 0.0 < self.channel_length < self.length:
            raise ValueError("need 0 < channel_length < length")
        if not 0.0 < self.cathode <= self.length:
            raise ValueError("the cathode must be in the domain")
        if self.cells < 4:
            raise ValueError("need at least 4 cells")
        if self.mass_flow <= 0.0 or self.area <= 0.0 or self.neutral_speed <= 0.0:
            raise ValueError("mass_flow, area and neutral_speed must be > 0")
        if self.limiter not in LIMITERS:
            raise ValueError(f"limiter must be one of {sorted(LIMITERS)}")
        missing = {"ionization", "excitation", "elastic"} - set(self.reactions)
        if missing:
            raise ValueError(f"reactions missing {sorted(missing)}")
        self.mass = XENON.mass
        self.h = self.length / self.cells
        self.faces = np.arange(self.cells + 1) * self.h
        self.z = 0.5 * (self.faces[:-1] + self.faces[1:])
        self.b = np.asarray(self.magnetic_field(self.z), dtype=float)
        #: the share of every cell between the anode and the cathode line
        self._weights = np.clip((self.cathode - self.faces[:-1]) / self.h, 0.0, 1.0)
        #: the neutrals entering at the anode (m^-2 s^-1)
        self.inflow = self.mass_flow / (self.mass * self.area)

    # -- the state ---------------------------------------------------------------------

    def initial_state(self) -> HallState:
        """A start: neutrals from the anode, a plasma near the exit, a guess of ``u_i`` and ``T_e``."""
        z, lch = self.z, self.channel_length
        anode_density = self.inflow / self.neutral_speed
        burnt = 0.5 * (1.0 + np.tanh((z - 0.8 * lch) / (0.1 * lch)))
        nn = anode_density * (1.0 - 0.95 * burnt)
        ni = 0.02 * anode_density * np.exp(-(((z - 0.8 * lch) / (0.4 * lch)) ** 2))
        ni += 1.0e-3 * anode_density
        te = self.anode_temperature + 20.0 * np.exp(-(((z - lch) / (0.3 * lch)) ** 2))
        bohm = np.sqrt(E * self.anode_temperature / self.mass)
        exit_speed = 0.8 * np.sqrt(2.0 * E * self.discharge_voltage / self.mass)
        ramp = np.clip(z / (1.2 * lch), 0.0, 1.0)
        ui = -bohm + (exit_speed + bohm) * ramp
        return self.state(nn, ni, ni * ui, te, time=0.0)

    def _mobility(
        self, nn: np.ndarray, ni: np.ndarray, te: np.ndarray, anomalous: np.ndarray
    ) -> np.ndarray:
        """The cross-field mobility ``mu`` (m^2/(V s)) on the cells."""
        elastic = nn * self.reactions["elastic"].coefficients(te, 0.0)
        # electron-ion collisions, NRL formulary (ln Lambda for T_e > 10 eV, at least 1)
        log_lambda = np.maximum(
            24.0 - np.log(np.sqrt(ni * 1.0e-6) / np.maximum(te, TE_FLOOR)), 1.0
        )
        coulomb = 2.91e-12 * ni * log_lambda * np.maximum(te, TE_FLOOR) ** -1.5
        nu = elastic + coulomb + anomalous
        hall = E * self.b / (ELECTRON_MASS * nu)
        return E / (ELECTRON_MASS * nu) / (1.0 + hall**2)

    def _electrons(
        self, nn: np.ndarray, ni: np.ndarray, gi: np.ndarray, te: np.ndarray
    ) -> dict:
        """The current density, field, potential and electron flux of the unknowns."""
        anomalous = np.asarray(
            self.anomalous(self.context(nn, ni, gi, te)), dtype=float
        )
        mu = self._mobility(nn, ni, te, anomalous)
        pressure_gradient = np.gradient(ni * te, self.h) / ni
        resistance = 1.0 / (ni * mu)
        w = self._weights * self.h
        current = (
            E
            * (
                self.discharge_voltage
                + np.sum(w * (gi * resistance + pressure_gradient))
            )
            / np.sum(w * resistance)
        )
        efield = (current / E - gi) * resistance - pressure_gradient
        phi_faces = self.discharge_voltage - np.concatenate(
            ([0.0], np.cumsum(efield * self.h))
        )
        return {
            "current": current,
            "field": efield,
            "potential": 0.5 * (phi_faces[:-1] + phi_faces[1:]),
            "electron_flux": gi - current / E,
            "mobility": mu,
            "anomalous": anomalous,
        }

    def state(
        self,
        neutral_density: np.ndarray,
        ion_density: np.ndarray,
        ion_flux: np.ndarray,
        electron_temperature: np.ndarray,
        *,
        time: float = 0.0,
        history: dict[str, np.ndarray] | None = None,
    ) -> HallState:
        """The `HallState` of the unknowns, with the electrons solved for them."""
        electrons = self._electrons(
            neutral_density, ion_density, ion_flux, electron_temperature
        )
        ui = ion_flux / ion_density
        exit_flux = ion_flux[-1]
        return HallState(
            time=time,
            z=self.z.copy(),
            neutral_density=neutral_density.copy(),
            ion_density=ion_density.copy(),
            ion_velocity=ui,
            electron_temperature=electron_temperature.copy(),
            potential=electrons["potential"],
            electric_field=electrons["field"],
            anomalous_frequency=electrons["anomalous"],
            magnetic_field=self.b.copy(),
            discharge_current=electrons["current"] * self.area,
            thrust=self.mass * self.area * exit_flux * max(ui[-1], 0.0),
            ion_current=E * self.area * exit_flux,
            mass_flow=self.mass_flow,
            discharge_voltage=self.discharge_voltage,
            history=dict(history or {}),
        )

    # -- the electrons ------------------------------------------------------------------

    def context(
        self, nn: np.ndarray, ni: np.ndarray, gi: np.ndarray, te: np.ndarray
    ) -> ElectronContext:
        """The `ElectronContext` of the unknowns."""
        return ElectronContext(
            z=self.z,
            magnetic_field=self.b,
            density=ni,
            electron_temperature=te,
            ion_velocity=gi / ni,
            neutral_density=nn,
            channel_length=self.channel_length,
        )

    # -- the heavy species ----------------------------------------------------------------

    def _heavy_rhs(
        self,
        nn: np.ndarray,
        ni: np.ndarray,
        gi: np.ndarray,
        efield: np.ndarray,
        ionization_frequency: np.ndarray,
        sound: np.ndarray,
        anode_bohm: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``d/dt`` of ``n_n``, ``n_i`` and ``n_i u_i``, and the heavy fluxes through the ends."""
        ui = gi / ni
        density = np.concatenate(([ni[0], ni[0]], ni, [ni[-1], ni[-1]]))
        anode_speed = min(ui[0], -anode_bohm)
        velocity = np.concatenate(([anode_speed, anode_speed], ui, [ui[-1], ui[-1]]))
        limiter = LIMITERS[self.limiter]

        def faces(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            slope = limiter(values[1:-1] - values[:-2], values[2:] - values[1:-1])
            centre = values[1:-1]
            return (centre + 0.5 * slope)[:-1], (centre - 0.5 * slope)[1:]

        n_left, n_right = faces(density)
        u_left, u_right = faces(velocity)
        speed = np.concatenate(([sound[0]], sound, [sound[-1]]))
        wave = np.maximum(np.abs(u_left) + speed[:-1], np.abs(u_right) + speed[1:])
        thermal = E * self.ion_temperature / self.mass
        mass_left, mass_right = n_left * u_left, n_right * u_right
        momentum_left = mass_left * u_left + n_left * thermal
        momentum_right = mass_right * u_right + n_right * thermal
        mass_flux = 0.5 * (mass_left + mass_right) - 0.5 * wave * (n_right - n_left)
        momentum_flux = 0.5 * (momentum_left + momentum_right) - 0.5 * wave * (
            mass_right - mass_left
        )
        neutral_flux = np.empty(self.cells + 1)
        neutral_flux[0] = self.inflow - mass_flux[0]  # the anode returns the ions
        neutral_flux[1:] = self.neutral_speed * nn
        ionization = ionization_frequency * nn
        dnn = -np.diff(neutral_flux) / self.h - ionization
        dni = -np.diff(mass_flux) / self.h + ionization
        dgi = (
            -np.diff(momentum_flux) / self.h
            + E / self.mass * ni * efield
            + ionization * self.neutral_speed
        )
        ends = np.array(
            [neutral_flux[0] + mass_flux[0], neutral_flux[-1] + mass_flux[-1]]
        )
        return dnn, dni, dgi, ends

    # -- the electron energy ----------------------------------------------------------------

    def _energy(
        self,
        te: np.ndarray,
        n_old: np.ndarray,
        n_new: np.ndarray,
        nn: np.ndarray,
        electrons: dict,
        context: ElectronContext,
        dt: float,
    ) -> np.ndarray:
        """``T_e`` after an implicit step of the electron energy."""
        h = self.h
        flux = electrons["electron_flux"]
        kappa = 2.5 * electrons["mobility"] * n_new * te
        face_flux = np.concatenate(
            ([flux[0]], 0.5 * (flux[1:] + flux[:-1]), [flux[-1]])
        )
        face_kappa = np.concatenate(
            ([kappa[0]], 0.5 * (kappa[1:] + kappa[:-1]), [kappa[-1]])
        )
        forward = 2.5 * np.maximum(face_flux, 0.0)
        backward = 2.5 * np.minimum(face_flux, 0.0)
        # the face flux is left * T_left + right * T_right + constant
        left = forward + face_kappa / h
        right = backward - face_kappa / h
        constant = np.zeros(self.cells + 1)
        ta, tc = self.anode_temperature, self.cathode_temperature
        left[0] = 0.0
        right[0] = backward[0] - 2.0 * face_kappa[0] / h
        constant[0] = (forward[0] + 2.0 * face_kappa[0] / h) * ta
        left[-1] = forward[-1] + 2.0 * face_kappa[-1] / h
        right[-1] = 0.0
        constant[-1] = (backward[-1] - 2.0 * face_kappa[-1] / h) * tc

        losses = np.zeros(self.cells)
        for reaction in self.reactions.values():
            rate = reaction.coefficients(te, 0.0)
            if reaction.elastic_mass is not None:
                per_event = 3.0 * ELECTRON_MASS / reaction.elastic_mass * te
            else:
                per_event = reaction.electron_energy_loss
            losses += nn * rate * per_event
        losses += np.asarray(self.wall_loss(context), dtype=float)
        # losses per electron and eV of T_e, implicit in T_e (keeps it positive)
        sink = n_new * losses / np.maximum(te, TE_FLOOR)
        heating = -flux * electrons["field"]

        diagonal = 1.5 * n_new / dt + sink + (left[1:] - right[:-1]) / h
        upper = right[1:-1] / h
        lower = -left[1:-1] / h
        rhs = 1.5 * n_old * te / dt + heating
        rhs[0] += constant[0] / h
        rhs[-1] -= constant[-1] / h
        bands = np.zeros((3, self.cells))
        bands[0, 1:] = upper
        bands[1] = diagonal
        bands[2, :-1] = lower
        return np.maximum(solve_banded((1, 1), bands, rhs), TE_FLOOR)

    # -- stepping ---------------------------------------------------------------------------

    def stable_step(self, state: HallState) -> float:
        """The largest ion step (s): ``cfl h / max(|u_i| + c_s)``."""
        sound = np.sqrt(
            E * (state.electron_temperature + self.ion_temperature) / self.mass
        )
        return self.cfl * self.h / float(np.max(np.abs(state.ion_velocity) + sound))

    def _step(
        self, nn: np.ndarray, ni: np.ndarray, gi: np.ndarray, te: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, dict]:
        """One step of every equation; returns the unknowns, ``dt``, the end fluxes, the electrons."""
        electrons = self._electrons(nn, ni, gi, te)
        sound = np.sqrt(E * (te + self.ion_temperature) / self.mass)
        dt = self.cfl * self.h / float(np.max(np.abs(gi / ni) + sound))
        frequency = ni * self.reactions["ionization"].coefficients(te, 0.0)
        bohm = float(np.sqrt(E * te[0] / self.mass))
        args = (electrons["field"], frequency, sound, bohm)
        d1 = self._heavy_rhs(nn, ni, gi, *args)
        nn1, ni1, gi1 = nn + dt * d1[0], ni + dt * d1[1], gi + dt * d1[2]
        ni1 = np.maximum(ni1, DENSITY_FLOOR)
        d2 = self._heavy_rhs(nn1, ni1, gi1, *args)
        nn2 = 0.5 * (nn + nn1 + dt * d2[0])
        ni2 = np.maximum(0.5 * (ni + ni1 + dt * d2[1]), DENSITY_FLOOR)
        gi2 = 0.5 * (gi + gi1 + dt * d2[2])
        ends = 0.5 * (d1[3] + d2[3])
        context = self.context(nn, ni, gi, te)
        te2 = self._energy(te, ni, ni2, nn, electrons, context, dt)
        return nn2, ni2, gi2, te2, dt, ends, electrons

    @profile("carafe.hall_run", tags=_PROFILE_TAGS)
    def run(
        self,
        duration: float,
        state: HallState | None = None,
        *,
        samples: int = 200,
    ) -> HallState:
        """Step the thruster for ``duration`` (s) from ``state`` (`initial_state` if ``None``).

        Args:
            duration: The time to run (s).
            state: The start; the history of a state continues.
            samples: How many times to sample the discharge current, the ion current
                and the thrust into the history, evenly in time.

        Returns:
            The state at the end, with the history of the run and of ``state``.
        """
        state = self.initial_state() if state is None else state
        nn = state.neutral_density.copy()
        ni = np.maximum(state.ion_density, DENSITY_FLOOR)
        gi = ni * state.ion_velocity
        te = state.electron_temperature.copy()
        time, end = state.time, state.time + duration
        history = {key: list(values) for key, values in state.history.items()}
        for key in ("time", "discharge_current", "ion_current", "thrust"):
            history.setdefault(key, [])
        #: the heavy particles in and out through the ends, per unit area (m^-2)
        self.boundary_budget = np.zeros(2)
        interval = duration / samples
        next_sample = time + interval
        while time < end:
            nn, ni, gi, te, dt, ends, electrons = self._step(nn, ni, gi, te)
            time += dt
            self.boundary_budget += dt * ends
            if time >= next_sample or time >= end:
                next_sample += interval
                history["time"].append(time)
                history["discharge_current"].append(electrons["current"] * self.area)
                history["ion_current"].append(E * self.area * gi[-1])
                history["thrust"].append(
                    self.mass * self.area * gi[-1] * max(gi[-1] / ni[-1], 0.0)
                )
        arrays = {key: np.asarray(values) for key, values in history.items()}
        return self.state(nn, ni, gi, te, time=time, history=arrays)
