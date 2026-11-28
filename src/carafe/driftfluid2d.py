"""A 2D axial-azimuthal drift-fluid model of a magnetron: gradient-drift instability and spokes.

The plane is ``(x, y)``: ``x`` axial from the cathode (``x = 0``) to the anode
(``x = length_x``), ``y`` azimuthal and periodic, the magnetic field ``B_z(x)`` out
of the plane (the curvature is neglected, as in the 2D PIC benchmarks of Charoy et
al. 2019 and Boeuf 2022/2023). Species:

- **electrons**, inertialess and magnetised, with the full Hall tensor. From
  ``0 = -e (E + v x B) - grad(n T_e)/n - m nu v``:

      Gamma_e = -mu_perp [[1, -h], [h, 1]] (n E + grad(n T_e)),
      h = omega_ce / nu,   mu_perp = e nu / (m (nu^2 + omega_ce^2)),

  so the E x B drift (``h >> 1``: ``mu_perp h = 1/B``), the diamagnetic drift and the
  gradient-B drift (``d_x mu_perp`` in the divergence of the antisymmetric part) all
  come out of the tensor; ``nu = k_en N + alpha omega_ce / 16`` adds a Bohm-like
  anomalous collision frequency when ``bohm > 0``;
- **ions**, cold or isothermal, unmagnetised, with inertia and friction on the gas
  (``dv/dt + v.grad v = e E / M - nu_in v - T_i grad n / (M n)``). Inertia matters:
  the ion response sets the lower-hybrid / ion-sound frequency of the instability,
  which a drift-diffusion ion does not have;
- **neutral gas** ``N``, depleted by ionization and refilled at the rate
  ``refill_rate`` (the gas flow and the sputter wind), the slow variable of an
  ionization wave;
- the **electron energy** ``w = 3/2 n T_e`` with Joule heating, the enthalpy flux of
  Hagelaar and Pitchford (2005) and three loss channels (elastic, ionization,
  excitation), or a fixed ``T_e``;
- optionally (``metal``, a `carafe.magnetron.TargetMaterial`) the **sputtered metal**:
  neutral atoms ``M`` sputtered from the cathode by the ion fluxes into it,
  ``Gamma_M = sum_s Y_s(E_s) |Gamma_s|`` over ``Ar+``, ``M+`` and ``M2+``
  (self-sputtering by the metal ions), with the yields functions of the ion energy at the
  cathode (`carafe.sputtering`: Bohdansky's formula; ``E_s = Z_s e`` times the cathode
  fall, see ``ion_energy``), carried away from the cathode as a beam at the speed
  ``metal_speed`` or diffusing (``metal_transport``), ionized by electron impact (the Lotz
  rate by default; its energy is an electron energy loss) and by charge exchange with
  ``Ar+``, excited and scattered by the electrons (``metal_collisions``: an energy loss and
  a share of the collision frequency of the mobility); the singly charged **metal ions**
  ``M+``, a second ion fluid with its own mass and the momentum equation of the argon ions;
  and the doubly charged ``M2+`` (``double_ionization``, ``e + M+ -> 2e + M2+``), a third
  of charge 2; all in Poisson's equation and in the current. The cathode then emits
  ``sum_s gamma_s |Gamma_s|`` electrons (unless ``cathode_current_density`` is given): the
  secondary yield depends on the ion species. A singly charged ion of most metals carries
  too little potential energy to release an electron from its own target (it needs
  ``0.78 E_iz > 2 W``), so ``gamma_M+`` is zero and a discharge that turns to
  self-sputtering keeps its secondary emission only through the ``M2+``, which do emit
  (Anders 2008; Anders et al., arXiv:1305.5453). The discharge current density at the
  cathode is ``sum_s (Z_s + gamma_s) e |Gamma_s|`` (`DriftFluid2D.cathode_budget`).

Closures (what the 1D runaway diagnosis changed; see the changelog): the electron
momentum-transfer rate is a function of ``T_e`` (not a constant), excitation is an explicit
loss channel (replacing a lumped 30 eV per ionization), the fitted rates are held at their
end values outside ``0.5 .. 100 eV`` so no rate grows with a runaway ``T_e``, the local
energy loss is treated implicitly (the explicit update is stiff: the heating of a
weakly magnetised, high-field cell is 1e11 eV/s per electron), and the cathode emits a
*prescribed net electron current* ``cathode_current_density`` as in Boeuf's PIC (a fixed
``gamma`` is an avalanche whenever ``gamma M >= 1``, with ``M`` about 170 here).

Formulation: **full Poisson, semi-implicit**, not quasi-neutral. The cathode
sheath and the plasma-wall boundaries are then part of the solution and no
sheath boundary model is needed, and the discharge is free to leave quasi-neutrality.
The dielectric relaxation through the electron conductivity (the stiff part: the
Hall current does not relax charge but ``mu_perp`` does) is removed by solving, in
every stage, the Poisson equation for the potential at the end of the stage,
``div((eps0 + dt sigma_e) grad phi) = rho - dt div(J_known)`` with the electron
conductivity tensor ``sigma_e = e n mu_perp [[1, -h], [h, 1]]`` (Lapenta's implicit
permittivity). With the same discrete fluxes this makes the Poisson equation of the
new densities hold exactly. The step is then limited by the advection (E x B drift
up to 1e6 m/s at the cathode) and not by ``eps0 / sigma``. Time stepping is the
SSP Runge-Kutta 3 scheme (each stage an implicit-permittivity Euler step).

Discretisation: finite volumes on cells; the E x B advection of the electrons uses
third-order upwind-biased face values (a first-order upwind scheme has a numerical
diffusion ``v dy / 2`` that damps the spoke wavelengths); the ions use first-order
upwind (positive, monotone). Electron boundaries: the cathode emits a prescribed net
current (``cathode_current_density``) or ``gamma`` electrons per incident ion, the anode
absorbs (outflow only); ions leave freely (outflow only). The potential is ``-voltage`` at the cathode and 0 at the
anode.

Verification (``tests/unit/test_driftfluid2d.py``): a y-uniform state stays uniform to
round-off; a passive blob drifts at ``E/B`` to 3e-6; one Euler stage satisfies the
discrete Poisson equation exactly; ``stationary_sources`` hold an equilibrium; every term
of the discrete operator matches its analytic form to under 1 % (electron Hall flux with
a density gradient, ion continuity and momentum); and in a uniform plasma periodic in
``x`` (``periodic_x``, ``field_bc``) the time-domain growth rate and phase speed of an
``x``-uniform mode agree with `uniform_plasma_roots`, the full-Poisson local
dispersion relation, to under 4 % (0.5 % in the eigenvalues of the semi-discrete operator, at
32 cells per wavelength, converging with resolution). What is not verified: the
gradient-drift growth rate in a bounded box. The local WKB relation
`gradient_drift_roots` assumes a mode uniform along ``x``, which electrodes forbid:
the eigenmodes of a box are waves along ``x`` with their own rates, several times
different from the local ones (the quasi-neutral form is also far off, because the
E x B-driven branch sits on a near cancellation that the few-percent Debye correction moves).

Linear stability: `DriftFluid2D.linear_modes` builds the Jacobian of the discretised operator
about a y-uniform state for one azimuthal mode, x-resolved, so the electrodes, the gradients, the Hall
tensor and the energy equation are in it, and returns its eigenvalues and eigenvectors. Its leading
eigenvalue is meant to agree with `uniform_plasma_roots` for a uniform plasma and with a
time-domain run fitted in its exponential-growth window (tests in ``test_driftfluid2d.py``). On the
relaxed 1D state of Boeuf's case (gallery study 07) the two agree to 5 % in the growth rate, and the
leading mode is not Boeuf's: it is driven by the ionization response (frozen, the rate falls tenfold),
grows at 1-3e6 /s at all wavelengths and hardly propagates (a few km/s, either sign), while his
linear phase is a gradient-drift wave at 1e7 /s, 1.6 cm, +17 km/s. The cause is the 1D state:
its density falls towards the anode over ``L_n`` of about -5 mm at ``x = 1.8`` cm (converged
in ``nx``) against his -2.2 mm, and `boeuf_dispersion_roots` with that ``L_n`` grows a
hundred times slower.

Analytic reference: `uniform_plasma_roots` and `gradient_drift_roots` are the
local dispersion relations of this system (without and with a density gradient), and
`boeuf_dispersion_roots` is Boeuf's (with finite Larmor radius and the grad-B drift).

PLACEHOLDERS, to be checked against the sources before any physical claim: the three
argon rate fits (ionization, excitation, momentum transfer: quoted from memory of
Lieberman and Lichtenberg, *Principles of Plasma Discharges and Materials Processing*;
the momentum rate gives 1.3e-13 m^3/s at 3 eV, close to the ``nu_en = 2e7 /s`` that Boeuf
quotes at 0.6 Pa; replace all three with rates from plasmacoll's cross sections), the
ion friction, the single lumped excitation level (11.56 eV) and ``secondary_emission``.

PLACEHOLDERS of the metal: the factor ``Q`` of the sputter yields, the outer-shell
electron counts of the Lotz rates, the excitation level and the charge-exchange rate of the
`TargetMaterial` (see `carafe.magnetron.MATERIALS`; the yields to take from Eckstein's fits,
Yamamura and Tawara 1996, SRIM or Depla and Mahieu 2008); the secondary yields come from
Baragiola's empirical formula (a factor of two); the Lotz rates (a factor of two), the
lumped excitation rate (`lumped_excitation_rate`), ``metal_momentum_rate``, ``metal_speed``,
``metal_diffusion`` and the metal ions' friction (``metal_ion_friction_rate``, by default
the argon ``ion_friction_rate``).

Not yet: electron inertia, an electron-ion or ion-ion collision operator, the curvature of
the race track, a gas temperature equation. Of the metal: the sputter wind on the gas
(only through ``refill_rate``), the energy distribution and the angular spread of the
sputtered atoms (a beam along ``x`` or diffusion, nothing between), the redeposition of
metal ions on the anode (they leave there), the distribution of the ion energies at the
cathode (one energy per species and ``y``), the kinetic secondary emission, ``Ar2+`` and
the momentum the ions gain or lose in ionization.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import xarray as xr
from methodcard import ConvergenceProblem, Rule, method
from plasmacoll.constants import ATOMIC_MASS, ELECTRON_MASS, ELEMENTARY_CHARGE
from plasmacoll.units import convert_units, to_array, to_unit, unit_field, unit_metadata
from scope_profiler import profile

from carafe._profiling import _PROFILE_TAGS
from carafe.datasets import variable
from carafe.magnetron import TargetMaterial, lotz_ionization
from carafe.sputtering import BohdanskyYield, potential_emission_yield

__all__ = [
    "DriftFluid2D",
    "DriftFluid2DState",
    "LinearModes",
    "ModeFit",
    "Run",
    "SpokeReport",
    "argon_excitation_rate",
    "argon_ionization_rate",
    "argon_momentum_rate",
    "boeuf_dispersion_roots",
    "exponential_field",
    "fit_mode",
    "gradient_drift_roots",
    "lumped_excitation_rate",
    "mode_amplitudes",
    "spoke_report",
    "uniform_plasma_roots",
]

E = ELEMENTARY_CHARGE
EPSILON_0 = 8.8541878128e-12
#: Floor of the densities (m^-3) and of the electron temperature (eV).
N_FLOOR = 1.0e3
TE_FLOOR = 0.1
TE_CEIL = 200.0


#: Electron temperatures (eV) between which the fitted rates are evaluated; outside they
#: are held at the end values, so no rate can run away with a runaway ``T_e``.
RATE_TE_RANGE = (0.5, 100.0)


def _bounded(te: np.ndarray | float) -> np.ndarray | float:
    return np.clip(te, *RATE_TE_RANGE)


def _fit_rules(a: str, power: str, energy: str) -> dict:
    """The SymPy rule of an Arrhenius-like fit ``a T^power exp(-energy / T)``."""
    import sympy

    te = sympy.Symbol("te", positive=True)
    rule = (
        sympy.Float(a) * te ** sympy.Float(power) * sympy.exp(-sympy.Float(energy) / te)
    )
    return {"rate": Rule(rule, ranges={"te": (1.0, 50.0)})}


def _momentum_rules() -> dict:
    """The SymPy rule of `argon_momentum_rate` (for the tests)."""
    import sympy

    te = sympy.Symbol("te", positive=True)
    log = sympy.log(te)
    rule = sympy.Float("2.336e-14") * sympy.exp(
        sympy.Float("1.609") * log
        + sympy.Float("0.0618") * log**2
        - sympy.Float("0.1171") * log**3
    )
    return {"rate": Rule(rule, ranges={"te": (1.0, 50.0)})}


@method(
    equations={
        "rate": r"K_\mathrm{iz} = 2.34\times10^{-14}\, T_e^{0.59}\, e^{-17.44/T_e} \quad [\mathrm{m^3/s}]",
    },
    assumptions=(
        r"PLACEHOLDER: the fit as remembered from Lieberman and Lichtenberg, the same as `carafe.magnetron.argon_ionization`",
        r"$T_e$ held within 0.5–100 eV (`RATE_TE_RANGE`)",
    ),
    symbolic=lambda: _fit_rules("2.34e-14", "0.59", "17.44"),
)
def argon_ionization_rate(te: np.ndarray | float) -> np.ndarray | float:
    """Ionization rate coefficient of argon at a Maxwellian ``T_e`` (eV), m^3/s.

    PLACEHOLDER: ``2.34e-14 T_e^0.59 exp(-17.44 / T_e)``, the fit as remembered from
    Lieberman and Lichtenberg's textbook; check it against the plasmacoll cross
    section before use. ``T_e`` is bounded to `RATE_TE_RANGE`.
    """
    te = _bounded(te)
    return 2.34e-14 * te**0.59 * np.exp(-17.44 / te)


@method(
    equations={
        "rate": r"K_\mathrm{ex} = 2.48\times10^{-14}\, T_e^{0.33}\, e^{-12.78/T_e} \quad [\mathrm{m^3/s}], \qquad \varepsilon_\mathrm{ex} = 11.56\ \mathrm{eV}",
    },
    assumptions=(
        r"PLACEHOLDER: one lumped level, the fit of `carafe.magnetron.argon_excitation`",
        r"$T_e$ held within 0.5–100 eV (`RATE_TE_RANGE`)",
    ),
    symbolic=lambda: _fit_rules("2.48e-14", "0.33", "12.78"),
)
def argon_excitation_rate(te: np.ndarray | float) -> np.ndarray | float:
    """Excitation rate coefficient of argon (lumped, ``11.56 eV``), m^3/s.

    PLACEHOLDER: ``2.48e-14 T_e^0.33 exp(-12.78 / T_e)``, as remembered from Lieberman
    and Lichtenberg; check against plasmacoll.
    """
    te = _bounded(te)
    return 2.48e-14 * te**0.33 * np.exp(-12.78 / te)


@method(
    equations={
        "rate": r"k_{en} = 2.336\times10^{-14} \exp\!\left(1.609 \ln T_e + 0.0618 \ln^2 T_e - 0.1171 \ln^3 T_e\right) \quad [\mathrm{m^3/s}]",
    },
    assumptions=(
        r"PLACEHOLDER: the elastic fit of `carafe.magnetron.argon_elastic` as a momentum-transfer rate; $1.3\times10^{-13}$ m³/s at 3 eV",
        r"$T_e$ held within 0.5–100 eV (`RATE_TE_RANGE`)",
    ),
    symbolic=_momentum_rules,
)
def argon_momentum_rate(te: np.ndarray | float) -> np.ndarray | float:
    """Electron-argon momentum-transfer rate coefficient at ``T_e`` (eV), m^3/s.

    PLACEHOLDER: ``2.336e-14 T_e^1.609 exp(0.0618 ln(T_e)^2 - 0.1171 ln(T_e)^3)``, as
    remembered from Lieberman and Lichtenberg; it gives 1.3e-13 m^3/s at 3 eV, close to the
    ``nu_en = 2e7 /s`` at 0.6 Pa that Boeuf quotes (``k = 1.4e-13``).
    """
    log = np.log(_bounded(te))
    return 2.336e-14 * np.exp(1.609 * log + 0.0618 * log**2 - 0.1171 * log**3)


def lumped_excitation_rate(
    excitation_energy: float, coefficient: float = 1.0e-13
) -> Callable[[np.ndarray | float], np.ndarray | float]:
    """``K_ex = coefficient exp(-excitation_energy / T_e)`` (m^3/s), a lumped excitation.

    The crudest rate of a single level of ``excitation_energy`` (eV): a constant times the
    share ``exp(-E_ex / T_e)`` of a Maxwellian above the threshold (``T_e`` in eV, bounded to
    `RATE_TE_RANGE`). PLACEHOLDER for the metal atoms: ``coefficient = 1e-13`` m^3/s is a
    resonance cross section of about 1e-19 m^2 times the mean electron speed at a few eV;
    replace by a Maxwellian average of the metal's excitation cross sections (LXCat).
    """

    def rate(te: np.ndarray | float) -> np.ndarray | float:
        return coefficient * np.exp(-excitation_energy / _bounded(te))

    return rate


def exponential_field(b_max: float, scale: float) -> Callable[[np.ndarray], np.ndarray]:
    """``B(x) = b_max exp(-x / scale)``: the profile of Boeuf (2022), 120 mT and ``0.3 gap``.

    ``b_max`` (T) and ``scale`` (m) take Quantities or strings such as ``"120 mT"`` too.
    """
    b_max = to_unit(b_max, "T", "b_max")
    scale = to_unit(scale, "m", "scale")

    def field_of_x(x: np.ndarray) -> np.ndarray:
        return b_max * np.exp(-np.asarray(x) / scale)

    return field_of_x


@dataclass
class DriftFluid2DState:
    """The fields on the ``(nx, ny)`` cells.

    Attributes:
        x: Cell centres along ``x`` (m).
        y: Cell centres along ``y`` (m).
        electron_density: ``n_e`` (m^-3).
        ion_density: ``n_i`` (m^-3).
        ion_velocity: ``(v_x, v_y)`` of the ions, shape ``(2, nx, ny)`` (m/s).
        gas_density: The neutral density ``N`` (m^-3).
        electron_temperature: ``T_e`` (eV).
        potential: ``phi`` (V).
        time: Time (s).
        metal_density: The sputtered metal atoms ``M`` (m^-3), ``None`` without a metal.
        metal_ion_density: The metal ions ``M+`` (m^-3), ``None`` without a metal.
        metal_ion_velocity: ``(v_x, v_y)`` of the metal ions, shape ``(2, nx, ny)`` (m/s),
            ``None`` without a metal.
        metal_ion2_density: The doubly charged metal ions ``M2+`` (m^-3), ``None`` without
            them.
        metal_ion2_velocity: ``(v_x, v_y)`` of the ``M2+`` (m/s), ``None`` without them.
    """

    x: np.ndarray
    y: np.ndarray
    electron_density: np.ndarray
    ion_density: np.ndarray
    ion_velocity: np.ndarray
    gas_density: np.ndarray
    electron_temperature: np.ndarray
    potential: np.ndarray
    time: float = 0.0
    metal_density: np.ndarray | None = None
    metal_ion_density: np.ndarray | None = None
    metal_ion_velocity: np.ndarray | None = None
    metal_ion2_density: np.ndarray | None = None
    metal_ion2_velocity: np.ndarray | None = None

    def copy(self) -> DriftFluid2DState:
        """A deep copy."""

        def copied(value):
            return None if value is None else value.copy()

        return DriftFluid2DState(
            self.x,
            self.y,
            self.electron_density.copy(),
            self.ion_density.copy(),
            self.ion_velocity.copy(),
            self.gas_density.copy(),
            self.electron_temperature.copy(),
            self.potential.copy(),
            self.time,
            copied(self.metal_density),
            copied(self.metal_ion_density),
            copied(self.metal_ion_velocity),
            copied(self.metal_ion2_density),
            copied(self.metal_ion2_velocity),
        )

    def to_dataset(self) -> xr.Dataset:
        """The state as a labelled dataset on ``(x, y)``."""
        dims = ("x", "y")
        data = {
            "electron_density": variable(
                dims, self.electron_density, "electron density", "m^-3"
            ),
            "ion_density": variable(dims, self.ion_density, "ion density", "m^-3"),
            "ion_velocity_x": variable(
                dims, self.ion_velocity[0], "ion velocity along x", "m/s"
            ),
            "ion_velocity_y": variable(
                dims, self.ion_velocity[1], "ion velocity along y", "m/s"
            ),
            "gas_density": variable(dims, self.gas_density, "gas density", "m^-3"),
            "electron_temperature": variable(
                dims, self.electron_temperature, "electron temperature", "eV"
            ),
            "potential": variable(dims, self.potential, "potential", "V"),
        }
        if self.metal_density is not None:
            data["metal_density"] = variable(
                dims, self.metal_density, "metal atom density", "m^-3"
            )
        if self.metal_ion_density is not None and self.metal_ion_velocity is not None:
            data["metal_ion_density"] = variable(
                dims, self.metal_ion_density, "metal ion density", "m^-3"
            )
            data["metal_ion_velocity_x"] = variable(
                dims, self.metal_ion_velocity[0], "metal ion velocity along x", "m/s"
            )
            data["metal_ion_velocity_y"] = variable(
                dims, self.metal_ion_velocity[1], "metal ion velocity along y", "m/s"
            )
        if self.metal_ion2_density is not None and self.metal_ion2_velocity is not None:
            data["metal_ion2_density"] = variable(
                dims,
                self.metal_ion2_density,
                "doubly charged metal ion density",
                "m^-3",
            )
            data["metal_ion2_velocity_x"] = variable(
                dims,
                self.metal_ion2_velocity[0],
                "doubly charged metal ion velocity along x",
                "m/s",
            )
            data["metal_ion2_velocity_y"] = variable(
                dims,
                self.metal_ion2_velocity[1],
                "doubly charged metal ion velocity along y",
                "m/s",
            )
        coords = {
            "x": ("x", self.x, {"long_name": "x (axial)", "units": "m"}),
            "y": ("y", self.y, {"long_name": "y (azimuthal)", "units": "m"}),
        }
        return xr.Dataset(data, coords=coords, attrs={"time": self.time})


@dataclass
class ModeFit:
    """The growth and the phase speed of one azimuthal mode over a time window.

    Attributes:
        mode: The azimuthal mode number ``m`` (``k = 2 pi m / length_y``).
        growth_rate: ``gamma`` (1/s) from the log of the amplitude.
        frequency: ``omega`` (rad/s) of ``exp(i (k y - omega t))``; positive moves along ``+y``.
        phase_speed: ``omega / k`` (m/s), along ``+y`` (the E x B direction for the
            geometry of the module).
    """

    mode: int
    growth_rate: float
    frequency: float
    phase_speed: float


@dataclass
class Run:
    """The record of `DriftFluid2D.run`.

    Attributes:
        times: Recording times (s).
        probe: The recorded field along ``y`` at ``probe_x``, shape ``(nt, ny)``.
        probe_x: The ``x`` of the probe (m), the cell centre nearest the request.
        amplitudes: Complex Fourier coefficients of ``probe / mean`` per mode
            ``0 .. ny // 2``, shape ``(nt, ny // 2 + 1)``.
        state: The final state.
        steps: The number of steps taken.
        length_y: The azimuthal period (m).
        series: Further records, shape ``(nt, ny)``: ``"density_cathode"`` (the cell
            next to the cathode), ``"potential"``, ``"e_y"`` and ``"v_exb_y"`` (the
            azimuthal E x B velocity) at the probe.
    """

    times: np.ndarray
    probe: np.ndarray
    probe_x: float
    amplitudes: np.ndarray
    state: DriftFluid2DState
    steps: int
    length_y: float = 0.0
    series: dict[str, np.ndarray] = field(default_factory=dict)

    def to_dataset(self) -> xr.Dataset:
        """The space-time map of the probe and the mode amplitudes."""
        ny = self.probe.shape[1]
        y = self.state.y
        data = {
            "probe": variable(
                ("time", "y"), self.probe, "electron density at the probe", "m^-3"
            ),
            "mode_amplitude": variable(
                ("time", "mode"), np.abs(self.amplitudes), "|n_m / n_0|"
            ),
        }
        coords = {
            "time": ("time", self.times, {"long_name": "time", "units": "s"}),
            "y": ("y", y, {"long_name": "y (azimuthal)", "units": "m"}),
            "mode": ("mode", np.arange(ny // 2 + 1), {"long_name": "mode number"}),
        }
        return xr.Dataset(
            data, coords=coords, attrs={"probe_x": self.probe_x, "steps": self.steps}
        )


@dataclass
class SpokeReport:
    """What a run says about its azimuthal structure, in the terms of the experiments.

    Attributes:
        mode: The dominant azimuthal mode number ``m >= 1`` (largest time-averaged amplitude).
        wavelength: ``length_y / m`` (m).
        growth_rate: Growth rate (1/s) of that mode over the window.
        phase_velocity: Phase velocity (m/s) along ``+y``.
        exb_velocity: The azimuthal E x B velocity (m/s) at the probe, averaged over the
            window and ``y``.
        relative_direction: ``+1`` if the mode moves along E x B, ``-1`` against it.
        speed_ratio: ``|phase_velocity| / |exb_velocity|``.
        cathode_modulation: Peak-to-peak density modulation next to the cathode,
            ``(max - min) / (2 mean)`` along ``y``, averaged over the window.
        potential_hump: ``max_y(phi) - mean_y(phi)`` at the probe (V), window average.
        azimuthal_field: ``max_y |E_y|`` at the probe (V/m), window average.
        window: The ``(t0, t1)`` (s) of the analysis.
    """

    mode: int
    wavelength: float
    growth_rate: float
    phase_velocity: float
    exb_velocity: float
    relative_direction: int
    speed_ratio: float
    cathode_modulation: float
    potential_hump: float
    azimuthal_field: float
    window: tuple[float, float]


def fit_mode(
    times: np.ndarray,
    amplitudes: np.ndarray,
    mode: int,
    length_y: float,
    window: tuple[float, float] | None = None,
) -> ModeFit:
    """Fit the growth rate and the phase speed of mode ``mode`` in a time window.

    Args:
        times: The recording times (s).
        amplitudes: `mode_amplitudes` at those times.
        mode: The mode number ``m``.
        length_y: The azimuthal period (m).
        window: ``(t0, t1)`` to fit in; the whole record by default.
    """
    times = np.asarray(times, dtype=float)
    select = np.ones(times.shape, bool)
    if window is not None:
        select = (times >= window[0]) & (times <= window[1])
    if select.sum() < 3:
        raise ValueError("need at least 3 records in the window")
    t = times[select]
    a = amplitudes[select, mode]
    growth = np.polyfit(t, np.log(np.abs(a)), 1)[0]
    omega = -np.polyfit(t, np.unwrap(np.angle(a)), 1)[0]
    k = 2.0 * np.pi * mode / length_y
    return ModeFit(mode, float(growth), float(omega), float(omega / k))


def spoke_report(run: Run, window: tuple[float, float] | None = None) -> SpokeReport:
    """The `SpokeReport` of a run over a time window (the last half by default)."""
    t = run.times
    window = (0.5 * (t[0] + t[-1]), t[-1]) if window is None else window
    select = (t >= window[0]) & (t <= window[1])
    modes = np.arange(1, run.amplitudes.shape[1])
    mean_amp = np.abs(run.amplitudes[select][:, 1:]).mean(axis=0)
    mode = int(modes[np.argmax(mean_amp)])
    fit = fit_mode(t, run.amplitudes, mode, run.length_y, window)
    v_exb = float(run.series["v_exb_y"][select].mean())
    n_c = run.series["density_cathode"][select]
    modulation = float(
        np.mean((n_c.max(axis=1) - n_c.min(axis=1)) / (2.0 * n_c.mean(axis=1)))
    )
    phi = run.series["potential"][select]
    hump = float(np.mean(phi.max(axis=1) - phi.mean(axis=1)))
    e_y = float(np.mean(np.abs(run.series["e_y"][select]).max(axis=1)))
    direction = int(np.sign(fit.phase_speed) * np.sign(v_exb)) if v_exb else 0
    return SpokeReport(
        mode,
        run.length_y / mode,
        fit.growth_rate,
        fit.phase_speed,
        v_exb,
        direction,
        abs(fit.phase_speed) / abs(v_exb) if v_exb else float("nan"),
        modulation,
        hump,
        e_y,
        window,
    )


def mode_amplitudes(values: np.ndarray) -> np.ndarray:
    """Complex azimuthal Fourier coefficients of ``values / mean`` along the last axis.

    Mode ``m`` is the coefficient of ``exp(-i k y)`` over ``ny``: a perturbation
    ``a cos(k y - omega t)`` has ``a / 2 exp(-i omega t)``, the phase falls in time
    for a wave that moves along ``+y``.
    """
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=-1, keepdims=True)
    return np.fft.rfft(values / mean - 1.0, axis=-1) / values.shape[-1]


def _plasma_parameters(
    b_field: float, electron_collision_frequency: float, ion_collision_frequency: float
):
    """``(h, mu_perp)`` of the electrons from the field and the collision frequency."""
    omega_ce = E * b_field / ELECTRON_MASS
    nu = electron_collision_frequency
    return omega_ce / nu, E * nu / (ELECTRON_MASS * (nu**2 + omega_ce**2))


@method(
    equations={
        "electrons": r"s\, a_e = -\left(i\mathbf k\cdot\mathbf v_{0e} + \mu_\perp k^2 T_e\right) a_e + \mu_\perp k^2\, \delta\phi, \qquad \mathbf v_{0e} = -\mu_\perp E_0 (1, h)",
        "ions": r"s\, a_i = -i\mathbf k\cdot\mathbf v_{0i}\, a_i - i\mathbf k\cdot\mathbf v, \qquad s\,\mathbf v = -(\nu_i + i\mathbf k\cdot\mathbf v_{0i})\mathbf v - i\frac{e}{M}\mathbf k\,\delta\phi, \qquad v_{0i} = \frac{e E_0}{M\nu_i}",
        "poisson": r"\varepsilon_0 k^2\, \delta\phi = e n_0 (a_i - a_e)",
    },
    assumptions=(
        r"a uniform plasma in uniform $E_0\hat x$ and $B_z$, constant $T_e$, the model's Hall tensor (its antisymmetric part drops out of $\mathbf k\cdot R\,\mathbf k$), cold ions with friction",
        r"$\delta n = n_0 a\, e^{i\mathbf k\cdot\mathbf r + st}$; $\mathrm{Re}\,s > 0$ grows; the eigenvalues of the $4\times4$ matrix",
    ),
)
def uniform_plasma_roots(
    kx: complex,
    ky: float,
    density: float,
    e_field: float,
    b_field: float,
    electron_temperature: float,
    ion_mass: float,
    ion_collision_frequency: float,
    electron_collision_frequency: float,
) -> np.ndarray:
    """Eigenvalues ``s`` of ``exp(i (k . r) + s t)`` for a uniform plasma, full Poisson.

    Uniform ``n_0``, uniform ``E_0 = E_0 x`` and ``B_z``, constant ``T_e``, magnetised
    inertialess electrons (the model's Hall tensor), cold ions with friction (steady
    drift ``v_0i = e E_0 / (M nu_i)`` along ``x``) and Poisson's equation
    ``eps0 k^2 delta phi = e n_0 (a_i - a_e)``, with ``delta n = n_0 a``:

        s a_e = -(i k.v_0e + mu_perp k^2 T_e) a_e + mu_perp k^2 delta phi,
        s a_i = -i k.v_0i a_i - i k.v,
        s v   = -(nu_i + i k.v_0i) v - i (e / M) k delta phi,

    where ``v_0e = -mu_perp E_0 (1, h)``. (The antisymmetric Hall part of the tensor drops
    out of ``k . R k = k^2``.) Returns the three roots ``s``; ``Re s > 0`` grows and
    ``omega = -Im s`` is the frequency for ``k_y > 0``. For ``k_x = 0`` and a large
    density this tends to the quasi-neutral relation, whose roots near an
    E x B-driven resonance are very sensitive to the few-percent Debye correction.
    """
    h, mu_perp = _plasma_parameters(
        b_field, electron_collision_frequency, ion_collision_frequency
    )
    k = np.array([kx, ky], dtype=complex)
    k2 = k @ k
    v0e = -mu_perp * e_field * np.array([1.0, h])
    v0i = np.array([E * e_field / (ion_mass * ion_collision_frequency), 0.0])
    poisson = E * density / (EPSILON_0 * k2)
    nu_i = ion_collision_frequency
    matrix = np.zeros((4, 4), complex)  # (a_e, a_i, v_x, v_y)
    matrix[0, 0] = -(1j * (k @ v0e) + mu_perp * k2 * electron_temperature)
    matrix[0, 0] -= mu_perp * k2 * poisson
    matrix[0, 1] = mu_perp * k2 * poisson
    matrix[1, 1] = -1j * (k @ v0i)
    matrix[1, 2], matrix[1, 3] = -1j * k[0], -1j * k[1]
    for row, component in ((2, 0), (3, 1)):
        matrix[row, row] = -(nu_i + 1j * (k @ v0i))
        matrix[row, 0] = 1j * (E / ion_mass) * k[component] * poisson
        matrix[row, 1] = -1j * (E / ion_mass) * k[component] * poisson
    return np.linalg.eigvals(matrix)


@method(
    equations={
        "electrons": r"s\, a_e = \left(\frac{v_{0e}}{L} - i k h v_{0e}\right) a_e + K\,(T_e a_e - \delta\phi), \qquad K = i k\left(\frac{\mu_\perp h}{L} + i k\mu_\perp\right), \qquad v_{0e} = -\mu_\perp\left(E_0 - \frac{T_e}{L}\right)",
        "ions": r"s\, a_i = \frac{v_{0i}}{L} a_i - i k v, \qquad s\, v = -\nu_i v - i\frac{e}{M} k\,\delta\phi",
        "closure": r"\varepsilon_0 k^2\,\delta\phi = e n_0 (a_i - a_e) \ (\text{Poisson}) \quad \text{or} \quad a_e = a_i \ (\text{quasi-neutral})",
    },
    assumptions=(
        r"$n_0(x) = n_0 e^{-x/L}$ in uniform $E_0$ and $B$; WKB, $k_x = 0$, $kL$ large; $s = -i\omega$",
        r"a bounded box forbids the mode uniform along $x$: the growth rate of a box is not this one",
    ),
)
def gradient_drift_roots(
    k: float,
    gradient_length: float,
    e_field: float,
    b_field: float,
    electron_temperature: float,
    ion_mass: float,
    ion_collision_frequency: float,
    electron_collision_frequency: float,
    density: float | None = None,
) -> np.ndarray:
    """Roots ``omega`` of the local gradient-drift (Simon-Hoh) dispersion relation.

    The system of the module on an equilibrium ``n_0(x) = n_0 exp(-x / L)`` in a
    uniform ``E_0 = E_0 x`` and a uniform ``B_z = B``, perturbed by
    ``delta n = n_0(x) a exp(i (k y - omega t))`` with ``delta phi`` uniform in ``x``
    (WKB, ``k_x = 0``, ``k L`` large), ``T_e`` constant, cold ions with friction
    ``nu_i`` (so with the steady drift ``v_0i = e E_0 / (M nu_i)``). With
    ``s = -i omega``, ``K = i k (mu_perp h / L + i k mu_perp)``, ``v_0e = -mu_perp (E_0 - T_e / L)``
    along ``x`` and ``h v_0e`` along ``y``:

        s a_e = (v_0e / L - i k h v_0e) a_e + K (T_e a_e - delta phi),
        s a_i = (v_0i / L) a_i - i k v,        s v = -nu_i v - i (e / M) k delta phi,

    with ``eps0 k^2 delta phi = e n_0 (a_i - a_e)`` (Poisson) when ``density`` (``n_0``,
    m^-3) is given, and ``a_e = a_i`` (quasi-neutral, two roots) when it is not.
    The electron part is the continuity equation with the Hall tensor (the E x B
    advection, the density gradient and the diamagnetic term in ``T_e``), the ion part
    the unmagnetised ion continuity and momentum. Valid when the mode varies slowly
    along ``x`` on the scale ``L``; ``T_e`` in eV, ``E_0`` in V/m.

    Returns:
        The complex roots ``omega``; ``Im omega > 0`` grows, ``Re omega / k`` is
        the phase speed along ``+y``.
    """
    h, mu_perp = _plasma_parameters(
        b_field, electron_collision_frequency, ion_collision_frequency
    )
    nu_i = ion_collision_frequency
    v0i = E * e_field / (ion_mass * nu_i)
    v0e = -mu_perp * (e_field - electron_temperature / gradient_length)
    big_k = 1j * k * (mu_perp * h / gradient_length + 1j * k * mu_perp)
    if density is not None:
        poisson = E * density / (EPSILON_0 * k**2)
        matrix = np.zeros((3, 3), complex)  # (a_e, a_i, v)
        matrix[0, 0] = v0e / gradient_length - 1j * k * h * v0e
        matrix[0, 0] += big_k * (electron_temperature + poisson)
        matrix[0, 1] = -big_k * poisson
        matrix[1, 1] = v0i / gradient_length
        matrix[1, 2] = -1j * k
        matrix[2, 2] = -nu_i
        matrix[2, 0] = 1j * (E / ion_mass) * k * poisson
        matrix[2, 1] = -1j * (E / ion_mass) * k * poisson
        return 1j * np.linalg.eigvals(matrix)
    m_over = ion_mass / (E * k**2)
    # polynomial in s: -K m (s - v0i/L)(s + nu_i) + (1 - ...) s + const
    quad = -big_k * m_over
    lin = quad * (nu_i - v0i / gradient_length) + 1.0
    const = (
        quad * (-v0i / gradient_length) * nu_i
        - v0e / gradient_length
        + 1j * k * h * v0e
        - big_k * electron_temperature
    )
    s = np.roots([quad, lin, const])
    return 1j * s


@method(
    equations={
        "relation": r"\left(k^2\lambda_D^2\omega^2 - k^2 c_s^2\right)\left((1 + a)\omega - \omega_0 - \omega_D - a\omega_0 + i a\nu\right) + a\omega^3 + \left(\omega_* - \omega_D - a\omega_0 + i a\nu\right)\omega^2 = 0",
        "frequencies": r"a = k^2\rho_e^2, \qquad \omega_0 = -\frac{k E_0}{B}, \qquad \omega_* = -\frac{k T_e}{B L_n}, \qquad \omega_D = -\frac{2 k T_e}{B L_B}",
    },
    assumptions=(
        r"magnetised electrons with collisions and finite Larmor radius; unmagnetised cold collisionless ions; Poisson's equation",
        r"$L_n = n/n'$, $L_B = B/B'$ signed (negative when falling towards the anode); `grad_b=False` drops $\omega_D$",
        r"the drift fluid in the model field $B(x)\hat z$ gives half of $\omega_D$ (`carafe.dispersion`)",
    ),
    references=(
        "[Boeuf, arXiv:2211.17051](https://arxiv.org/abs/2211.17051), Eq. (1)",
    ),
)
def boeuf_dispersion_roots(
    k: float,
    density: float,
    e_field: float,
    b_field: float,
    electron_temperature: float,
    density_length: float,
    field_length: float,
    electron_collision_frequency: float,
    ion_mass: float = 39.948 * ATOMIC_MASS - ELECTRON_MASS,
    grad_b: bool = True,
) -> np.ndarray:
    """Roots ``omega`` of Boeuf's local gradient-drift relation, Eq. (1) of arXiv:2211.17051.

    The local relation of the PIC study the spoke runs are compared with: magnetised electrons
    with collisions and finite Larmor radius (``k^2 rho_e^2``), the E x B, diamagnetic and
    grad-B drift frequencies ``omega_0 = -k E_0 / B``, ``omega_* = -k T_e / (B L_n)``,
    ``omega_D = -2 k T_e / (B L_B)``, unmagnetised cold collisionless ions and Poisson's
    equation (``k^2 lambda_D^2``):

        (k^2 lambda_D^2 omega^2 - k^2 c_s^2) ((1 + a) omega - omega_0 - omega_D - a omega_0
        + i a nu) + a omega^3 + (omega_* - omega_D - a omega_0 + i a nu) omega^2 = 0,

    with ``a = k^2 rho_e^2``. ``density_length`` and ``field_length`` are the signed lengths
    ``n / n'`` and ``B / B'`` (negative when the quantity falls towards the anode), ``T_e``
    in eV, ``E_0`` in V/m. ``grad_b=False`` drops ``omega_D``. With Boeuf's parameters at
    ``x = 1.8`` cm (``n_0 = 0.8e15`` m^-3, ``E_0 = -2200`` V/m, ``B = 6`` mT, ``T_e = 11.4``
    eV, ``L_n = -2.2`` mm, ``L_B = -6`` mm, ``nu = 2e7`` /s) the largest growth rate is
    1.0e7 /s at a wavelength of 1.6 cm, moving at 17 km/s along ``+E x B`` (he quotes 1.1e7
    /s at ``k = 400`` /m and 17 km/s).

    Returns:
        The four complex roots ``omega``; ``Im omega > 0`` grows and ``Re omega / k`` is the
        phase speed along ``+y`` (the E x B direction for ``E_0 < 0``, ``B > 0``).
    """
    cs2 = E * electron_temperature / ion_mass
    omega_ce = E * b_field / ELECTRON_MASS
    rho2 = E * electron_temperature / ELECTRON_MASS / omega_ce**2
    lambda2 = EPSILON_0 * electron_temperature / (density * E)
    nu = electron_collision_frequency
    w0 = -k * e_field / b_field
    ws = -k * electron_temperature / (b_field * density_length)
    wd = -2.0 * k * electron_temperature / (b_field * field_length) if grad_b else 0.0
    a = k * k * rho2
    p1 = np.poly1d([k * k * lambda2, 0.0, -k * k * cs2])
    p2 = np.poly1d([1.0 + a, -w0 - wd - a * w0 + 1j * a * nu])
    p3 = np.poly1d([a, ws - wd - a * w0 + 1j * a * nu, 0.0, 0.0])
    return np.roots((p1 * p2 + p3).coeffs)


def _upwind3(
    n: np.ndarray, positive: np.ndarray, axis: int, periodic_x: bool = False
) -> np.ndarray:
    """Third-order upwind-biased face values; ``positive`` selects the flow direction.

    Along ``axis = 1`` (periodic) face ``j`` lies between cells ``j - 1`` and ``j``. Along
    ``axis = 0`` the result has ``nx + 1`` faces; the faces next to the boundary
    take the central average and the boundary faces the adjacent cell.
    """
    if axis == 1 or periodic_x:
        left = np.roll(n, 1, axis=axis)
        left2 = np.roll(n, 2, axis=axis)
        right = n
        right2 = np.roll(n, -1, axis=axis)
        pos = (-left2 + 5.0 * left + 2.0 * right) / 6.0
        neg = (2.0 * left + 5.0 * right - right2) / 6.0
        if axis == 1:
            return np.where(positive, pos, neg)
        face = np.where(positive[: n.shape[0]], pos, neg)
        return np.vstack([face, face[:1]])
    nx = n.shape[0]
    faces = np.empty((nx + 1, n.shape[1]))
    faces[0] = n[0]
    faces[nx] = n[-1]
    faces[1:nx] = 0.5 * (n[:-1] + n[1:])
    if nx >= 4:
        pos = (-n[:-3] + 5.0 * n[1:-2] + 2.0 * n[2:-1]) / 6.0
        neg = (2.0 * n[1:-2] + 5.0 * n[2:-1] - n[3:]) / 6.0
        # faces 2 .. nx - 2 lie between cells (i - 1, i)
        faces[2 : nx - 1] = np.where(positive[2 : nx - 1], pos, neg)
    return faces


@dataclass
class LinearModes:
    """The eigenmodes of `DriftFluid2D.linear_modes`, the most unstable first.

    Attributes:
        mode: The azimuthal mode number ``m``.
        wavenumber: ``k = 2 pi m / length_y`` (1/m).
        eigenvalues: ``s = gamma - i omega`` of ``exp(i k y + s t)``.
        profiles: The eigenvectors along ``x`` per unknown (``"ne"``, ``"ni"``, ...), shape
            ``(nx, number of eigenvalues)``.
        x: The cell centres (m).
        matrix: The complex Jacobian (rows and columns: the unknowns in blocks of ``nx``).
    """

    mode: int
    wavenumber: float
    eigenvalues: np.ndarray
    profiles: dict[str, np.ndarray]
    x: np.ndarray
    matrix: np.ndarray

    @property
    def growth_rates(self) -> np.ndarray:
        """``gamma`` of every eigenvalue (1/s)."""
        return np.real(self.eigenvalues)

    @property
    def frequencies(self) -> np.ndarray:
        """``omega = -Im s`` of every eigenvalue (rad/s)."""
        return -np.imag(self.eigenvalues)

    @property
    def phase_speeds(self) -> np.ndarray:
        """``omega / k`` (m/s), positive along ``+y``."""
        return self.frequencies / self.wavenumber

    def leading(self) -> tuple[float, float, float]:
        """``(gamma, omega, phase speed)`` of the most unstable eigenvalue."""
        return (
            float(self.growth_rates[0]),
            float(self.frequencies[0]),
            float(self.phase_speeds[0]),
        )


def _ssp_problem() -> ConvergenceProblem:
    """A uniform plasma, periodic in x and y, growing by ionization while its ions spin up.

    With a constant ionization frequency and a gas that does not deplete,
    ``n = n_0 e^(nu t)``; the ions relax to ``e E_0 / (M nu_i)`` with ``1 - e^(-nu_i t)``.
    Both rates are ``2 / T`` and the steps resolve the drift (``dt`` below the CFL step).
    """
    gas, field, duration = 1.43e20, -2200.0, 6.0e-8
    rate = 2.0 / duration
    model = DriftFluid2D(
        0.0048,
        0.016,
        4,
        4,
        voltage=0.0,
        magnetic_field=0.02,
        gas_density=gas,
        electron_temperature=5.0,
        momentum_rate=2e7 / gas,
        ion_friction_rate=rate / gas,
        ionization_rate=lambda te: rate / gas + 0.0 * te,
        excitation_rate=None,
        refill_rate=0.0,
        depletion=0.0,
        secondary_emission=0.0,
        ion_temperature=1e-4,
        field_bc=field,
        periodic_x=True,
    )
    drift = E * field / (model.ion_mass * rate)

    def error(dt: float) -> dict[str, float]:
        state = model.initial_state(1e15)
        steps = round(duration / dt)
        for _ in range(steps):
            state = model.step(state, duration / steps)
        density = state.electron_density.mean() / (1e15 * np.exp(2.0))
        velocity = state.ion_velocity[0].mean() / (drift * (1.0 - np.exp(-2.0)))
        return {"density": abs(density - 1.0), "velocity": abs(velocity - 1.0)}

    return ConvergenceProblem(
        error, steps=[duration / 16, duration / 32, duration / 64]
    )


@method(
    equations={
        "electrons": r"\boldsymbol\Gamma_e = -\mu_\perp\begin{pmatrix}1 & -h \\ h & 1\end{pmatrix}\big(n\mathbf E + \nabla(n T_e)\big), \qquad h = \frac{\omega_{ce}}{\nu}, \quad \mu_\perp = \frac{e\nu}{m_e(\nu^2 + \omega_{ce}^2)}",
        "collisions": r"\nu = k_{en}(T_e) N + \alpha\frac{\omega_{ce}}{16}\left[\frac{1}{1 + (B/B_*)^4}\right]",
        "ions": r"\partial_t n_i + \nabla\cdot(n_i\mathbf v) = \nu_\mathrm{iz} N n_e, \qquad \partial_t\mathbf v + \mathbf v\cdot\nabla\mathbf v = \frac{e}{M}\mathbf E - k_{in} N\mathbf v - \frac{e T_i}{M n_i}\nabla n_i",
        "gas": r"\partial_t N = -d\,\nu_\mathrm{iz} N n_e + d\,k_{cx} M n_i + \nu_r (N_0 - N)",
        "energy": r"\partial_t w = -\nabla\cdot\mathbf q - s\,\boldsymbol\Gamma_e\cdot\mathbf E - 3 m_e\left(\frac{\nu_{en}}{M} + \frac{\nu_{eM}}{M_\mathrm{M}}\right) n_e (T_e - T_i) - \varepsilon_\mathrm{iz} K_\mathrm{iz} N n_e - \varepsilon_\mathrm{ex} K_\mathrm{ex} N n_e - \left(\varepsilon_\mathrm{M} K_\mathrm{M} + \varepsilon_\mathrm{M}^* K^*_\mathrm{M}\right) M n_e - \varepsilon_\mathrm{M^+} K_\mathrm{M^+} n_{M^+} n_e",
        "poisson": r"\varepsilon_0\nabla^2\phi = -e(n_i + n_{M^+} + 2 n_{M^{2+}} - n_e), \qquad \nu = k_{en} N + k_{eM} M + \nu_B",
        "metal": r"\partial_t M + \nabla\cdot\boldsymbol\Gamma_M = -K_\mathrm{M}(T_e) M n_e - k_{cx} M n_i, \qquad \boldsymbol\Gamma_M = v_M M\,\hat x \ \text{or} \ -D_M\nabla M, \qquad \Gamma_M\big|_{x=0} = \sum_s Y_s(E_s)|\Gamma_s|, \quad E_s = Z_s e\,\Delta\phi_c",
        "metal_ions": r"\partial_t n_{M^+} + \nabla\cdot(n_{M^+}\mathbf v_{M^+}) = K_\mathrm{M} M n_e + k_{cx} M n_i - K_\mathrm{M^+} n_{M^+} n_e, \qquad \partial_t n_{M^{2+}} + \nabla\cdot(n_{M^{2+}}\mathbf v_{M^{2+}}) = K_\mathrm{M^+} n_{M^+} n_e, \qquad \partial_t\mathbf v_{s} + \mathbf v_{s}\cdot\nabla\mathbf v_{s} = \frac{Z_s e}{M_\mathrm{M}}\mathbf E - k_{iM} N\mathbf v_{s} - \frac{e T_i}{M_\mathrm{M} n_{s}}\nabla n_{s}",
        "emission": r"\Gamma_e\big|_{x=0} = \sum_s \gamma_s|\Gamma_s| \quad \text{or the prescribed } j_{e0}/e, \qquad j_c = \sum_s (Z_s + \gamma_s)\, e|\Gamma_s|",
        "upwind3": r"n^+_{j-1/2} = \frac{-n_{j-2} + 5 n_{j-1} + 2 n_j}{6}, \qquad n^-_{j-1/2} = \frac{2 n_{j-1} + 5 n_j - n_{j+1}}{6}",
    },
    assumptions=(
        r"the plane $(x, y)$, $B = B_z(x)$, periodic $y$; inertialess electrons; the energy flux the same operator on $w$ with $\tfrac53$",
        r"cell-centred finite volumes on a uniform grid: electrons third-order upwind-biased (by the lagged drift), ions and $\mathbf v\cdot\nabla\mathbf v$ first-order upwind",
        r"$\mu_\perp$, $h$ on the faces from the face gas and $B(x_f)$; rates held at their 0.5 and 100 eV values outside",
        r"no electron inertia, no electron-ion or ion-ion collisions, no race-track curvature, no gas temperature",
        r"the metal (opt-in, `metal`): atoms a beam along $x$ at `metal_speed` or diffusing (`metal_transport`), outflow at the anode; $M^+$ and $M^{2+}$ (`double_ionization`); electron ionization, excitation and elastic collisions with $M$ (`metal_collisions`); yields of the ion energy, one energy per species and $y$; PLACEHOLDER rates and yield factors",
    ),
    references=(
        "[Hagelaar and Pitchford, Plasma Sources Sci. Technol. 14, 722 (2005)](https://doi.org/10.1088/0963-0252/14/4/011)",
        "[Boeuf, arXiv:2211.17051](https://arxiv.org/abs/2211.17051)",
        "[Anders, Appl. Phys. Lett. 92, 201501 (2008)](https://doi.org/10.1063/1.2936307)",
    ),
    components=(
        argon_ionization_rate,
        argon_excitation_rate,
        argon_momentum_rate,
        uniform_plasma_roots,
        boeuf_dispersion_roots,
        lotz_ionization,
        BohdanskyYield,
        potential_emission_yield,
    ),
)
@dataclass
class DriftFluid2D:
    """The magnetron plane of the module docstring.

    Attributes:
        length_x: The cathode-anode gap (m).
        length_y: The azimuthal period (m).
        nx: Cells along ``x``.
        ny: Cells along ``y`` (at least 4).
        voltage: The discharge voltage (V): the cathode is at ``-voltage``, the anode at 0.
        magnetic_field: ``B_z(x)`` in T: a callable of the cell positions, or a number.
        gas_density: The neutral density at rest ``N_0`` (m^-3).
        electron_temperature: ``T_e`` held fixed (eV), or ``None`` for the energy equation.
        momentum_rate: Electron-neutral momentum-transfer rate coefficient ``k_en`` (m^3/s),
            ``nu_en = k_en N``: a number, or a function of ``T_e`` (eV; the default is the
            PLACEHOLDER `argon_momentum_rate`).
        ion_friction_rate: Ion-neutral momentum-transfer rate coefficient (m^3/s).
            PLACEHOLDER.
        ion_mass_u: The ion mass in atomic mass units (argon).
        ion_temperature: ``T_i`` in eV (also the gas temperature).
        ionization_rate: Rate coefficient (m^3/s) as a function of ``T_e`` (eV).
        excitation_rate: Excitation rate coefficient (m^3/s) as a function of ``T_e``, or
            ``None`` for none; ``excitation_energy_ev`` is the energy it costs (eV), and
            ``ionization_energy_ev`` the energy an ionization costs. Both are channels
            of the electron energy loss, with the elastic loss ``3 (m / M) nu_en (T_e - T_i)``.
        secondary_emission: Electrons emitted per ion hitting the cathode. PLACEHOLDER.
        cathode_current_density: If given (A/m^2), the cathode emits this net electron current
            density, whatever the ion flux, instead of ``secondary_emission`` per ion. This is the
            condition of Boeuf's PIC (``j_e0 = 0.03 A/m^2``, the electrons returning to the cathode
            re-emitted): a fixed ``gamma`` makes the discharge an avalanche whenever
            ``gamma M >= 1`` (``M`` the multiplication), a fixed emitted current gives a
            stable steady state with ``gamma_net = j_e0 / j_ion``.
        secondary_energy_ev: Temperature of the emitted electrons (eV).
        refill_rate: Relaxation rate (1/s) of the gas towards ``N_0`` (the gas flow and
            the sputter wind): the explicit parameter, with ``depletion``, of the neutral dynamics.
        depletion: Fraction of the ionization events that remove a neutral (1: full
            depletion, 0: a frozen gas).
        field_bc: ``None`` for the electrodes at ``-voltage`` and 0 V. A number ``E_0`` (V/m)
            replaces them by a fixed axial field ``E_x = E_0`` at both ends (a pinned
            potential), so that a perturbation can be uniform along ``x``: the setting
            for comparing with the local dispersion relation.
        heating_scale: Factor on the Joule heating of the electron energy (a diagnostic switch).
        floor_added: Running total of the density (summed over cells, m^-3) that the floor
            ``N_FLOOR`` on ``ne``, ``ni`` and ``N`` created in the Euler stages, a diagnostic of
            how much charge the limiter invents; reset it to zero to start counting.
        anode_thermal_flux: Adds the thermal flux of an absorbing wall, ``n v_bar / 2`` with
            ``v_bar = sqrt(8 e T_e / (pi m))`` (Hagelaar et al. 2000, reflection 0), to the drift
            flux of the electrons at the anode, which otherwise only leave at the drift rate; the
            outflow is then the larger of the two, and the density next to the anode falls to
            what carries the current. Tested for Boeuf's case without success: the 1D discharge
            then decays and runs away (no sheath barrier for the wall flux), so it is off.
        cathode_absorbs: The cathode also takes the electrons the field drives back into it
            (the drift of the cell next to it, where it points into the cathode), on top of
            the emitted flux. Without it the cathode emits its fixed flux whatever the field,
            and once the field at the cathode reverses (the potential in front of it falls
            below the cathode's) the returned electrons pile up there without limit.
        anode_sheath: Adds the thermal flux of a full Maxwellian, ``n v_bar / 4``, held back by
            the Boltzmann factor of the potential drop over the half cell next to the anode,
            ``exp(-max(phi_N - phi_anode, 0) / T_e)`` with ``phi_N - phi_anode = E_x dx / 2``:
            an electron-repelling anode sheath where the plasma sits above the anode, the
            saturation flux where it sits below. Replaces ``anode_thermal_flux``.
        diamagnetic: Factor on the electron pressure-gradient (diamagnetic) force in the fluxes
            and the Poisson solve (a diagnostic switch for the linear analysis; 1 is the model).
        periodic_x: Periodic along ``x`` as well, for a uniform plasma in a uniform field
            ``field_bc`` (required) and a uniform ``B``: pure Fourier modes ``(k_x, k_y)``
            exist, which is what the local dispersion relation describes.
        bohm: ``alpha`` of the anomalous frequency ``alpha omega_ce / 16`` (0 off).
        bohm_below: If given (T), the anomalous frequency acts only where the field is weak:
            times ``1 / (1 + (B / bohm_below)^4)``. A diagnostic of the anode region, where the
            electrons' Larmor radius reaches the gradient lengths (a few mm at a few mT) and
            the classical cross-field transport of the fluid is too small; the cathode
            region, where the confinement makes the discharge, keeps its classical value.
        external_sources: Volume sources added to the time derivative of an unknown
            (``"ne"``, ``"ni"``, ``"N"``, ``"vx"``, ``"vy"``, ``"w"``, and with a metal
            ``"M"``, ``"nm"``, ``"vmx"``, ``"vmy"``), numbers or arrays on the cells, per unit
            time. `stationary_sources` returns the ones that make a given state stationary.
        metal: The target material (a `carafe.magnetron.TargetMaterial`, e.g.
            ``MATERIALS["Ti"]``), or ``None`` (the default) for argon alone. With a metal the
            state carries the metal atoms and ions, the cathode is sputtered by
            ``argon_yield(E)`` atoms per ``Ar+`` and ``self_yield(E)`` per ``M+`` and
            ``M2+`` (at their energy ``E``, see ``ion_energy``), and emits
            ``argon_secondary`` electrons per ``Ar+`` (see ``argon_emission``),
            ``self_secondary`` per ``M+`` and ``double_secondary`` per ``M2+``;
            ``charge_exchange`` is the rate of ``Ar+ + M -> Ar + M+``. PLACEHOLDER: the
            yields and rates of `MATERIALS`.
        metal_speed: The axial speed (m/s) at which the sputtered atoms leave the cathode and
            cross the gap: a beam along ``x``, absorbed at the anode. PLACEHOLDER: between the
            thermal speed of the gas (about 400 m/s) and that of a sputtered atom of a few eV
            (several km/s); it sets the metal density for a given sputtered flux.
        metal_ionization_rate: Electron-impact ionization rate coefficient of the metal
            (m^3/s) as a function of ``T_e`` (eV); by default the Lotz rate of the material
            (`carafe.magnetron.lotz_ionization`, good to a factor of about two). Each
            ionization costs the electrons the material's ``ionization_energy``.
        ion_energy: The energy of the ions at the cathode that the sputter yields are
            evaluated at, per ``y``: ``"fall"`` (the default) ``Z e`` times the cathode fall,
            the largest potential along ``x`` above the cathode's (an ion born at the
            potential maximum, without collisions); ``"kinetic"`` the kinetic energy
            ``M v_x^2 / 2`` of the ion fluid in the cell next to the cathode (with the
            friction of the fluid: lower); or a number, a fixed fall in V (``Z e`` times it).
        double_ionization: Follow the doubly charged metal ions ``M2+`` (``e + M+ -> 2e +
            M2+``), a third ion fluid of charge 2. They are the efficient emitters: a
            singly charged ion of most metals emits no electron from its own target, a
            doubly charged one does (``double_secondary``), so a self-sputtering discharge
            keeps its secondary electrons only through them (Anders 2008). ``False`` gives
            the model with ``M+`` alone.
        metal_double_ionization_rate: Rate coefficient (m^3/s) of ``e + M+ -> 2e + M2+`` as
            a function of ``T_e``; by default the Lotz rate of the material's
            ``second_ionization_energy`` and ``ion_outer_electrons`` (PLACEHOLDER). Each costs
            the electrons the second ionization energy.
        metal_collisions: Electron collisions with the metal atoms other than the
            ionization: excitation of a lumped level (``metal_excitation_rate``, an energy
            loss of the material's ``excitation_energy``) and elastic momentum transfer
            (``metal_momentum_rate``, in the collision frequency of the mobility and the
            elastic energy loss ``3 (m / M_M) nu_eM (T_e - T_i)``).
        metal_excitation_rate: Rate coefficient (m^3/s) of the excitation of ``M`` as a
            function of ``T_e``; by default `metal_excitation_rate` of the material's level
            (PLACEHOLDER).
        metal_momentum_rate: Electron-metal momentum-transfer rate coefficient (m^3/s), a
            number or a function of ``T_e``. PLACEHOLDER: 1e-13 m^3/s, a cross section of
            1e-19 m^2 at the mean speed of 3 eV electrons; to take from the LXCat data of the
            metal.
        metal_transport: The closure of the metal atoms: ``"beam"`` (the default), a beam
            along ``x`` at ``metal_speed`` (the sputtered atoms fly ballistically across
            the gap, as at a mean free path longer than the gap), or ``"diffusion"``, atoms
            thermalized at once that diffuse with ``metal_diffusion`` in ``x`` and ``y``
            (the high-pressure limit), absorbed at the anode.
        metal_diffusion: The diffusion coefficient (m^2/s) of the metal atoms in the gas for
            ``metal_transport="diffusion"``. PLACEHOLDER: 0.02 m^2/s, of the order of copper in
            argon at 0.6 Pa (``D p`` about 0.01 m^2 Pa/s); to compute from a binary
            diffusion coefficient (`carafe.gasdynamics.binary_diffusion`) at the gas density.
        argon_emission: Which secondary yield the ``Ar+`` have with a metal: ``"material"``
            (the default) the material's ``argon_secondary``, the yield of argon on that
            target; ``"model"`` the model's ``secondary_emission``, so that adding the metal
            leaves the emission by argon as it was.
        metal_ion_friction_rate: The ion-neutral momentum-transfer rate coefficient of the
            metal ions in the gas (m^3/s); ``None`` (the default) takes the argon ions'
            ``ion_friction_rate``. PLACEHOLDER either way: the polarization (Langevin) rate of
            the metal ion in argon is the source to compute it from.

    The physical settings take astropy Quantities or strings such as ``"2 cm"``,
    ``"20 mT"`` or ``"300 K"`` too, converted to the units above (a magnetic field
    or momentum rate that is a function is kept as it is).
    """

    length_x: float = field(metadata=unit_metadata("m"))
    length_y: float = field(metadata=unit_metadata("m"))
    nx: int
    ny: int
    voltage: float = unit_field("V", 260.0)
    magnetic_field: Callable[[Any], Any] | float = 0.0
    gas_density: float = unit_field("m^-3", 1.45e20)
    electron_temperature: float | None = unit_field("eV", 5.0)
    momentum_rate: float | Callable[[Any], Any] = argon_momentum_rate
    ion_friction_rate: float = unit_field("m^3 / s", 1.0e-15)
    ion_mass_u: float = unit_field("u", 39.948)
    ion_temperature: float = unit_field("eV", 0.025)
    ionization_rate: Callable[[Any], Any] = argon_ionization_rate
    excitation_rate: Callable[[Any], Any] | None = argon_excitation_rate
    ionization_energy_ev: float = unit_field("eV", 15.76)
    excitation_energy_ev: float = unit_field("eV", 11.56)
    secondary_emission: float = 0.06
    cathode_current_density: float | None = unit_field("A / m^2", None)
    secondary_energy_ev: float = unit_field("eV", 1.0)
    refill_rate: float = unit_field("1 / s", 1.0e4)
    depletion: float = 1.0
    field_bc: float | None = unit_field("V / m", None)
    periodic_x: bool = False
    heating_scale: float = 1.0
    diamagnetic: float = 1.0
    anode_thermal_flux: bool = False
    anode_sheath: bool = False
    cathode_absorbs: bool = False
    floor_added: dict = field(default_factory=lambda: {"ne": 0.0, "ni": 0.0, "N": 0.0})
    _rate_override: np.ndarray | None = field(default=None, repr=False)
    _frequency_override: np.ndarray | None = field(default=None, repr=False)
    _lu: spla.SuperLU | None = field(default=None, repr=False)
    bohm: float = 0.0
    bohm_below: float | None = unit_field("T", None)
    external_sources: dict[str, np.ndarray] = field(default_factory=dict)
    metal: TargetMaterial | None = None
    metal_speed: float = unit_field("m / s", 1.0e3)
    metal_ionization_rate: Callable[[Any], Any] | None = None
    ion_energy: str | float = "fall"
    double_ionization: bool = True
    metal_double_ionization_rate: Callable[[Any], Any] | None = None
    metal_collisions: bool = True
    metal_excitation_rate: Callable[[Any], Any] | None = None
    metal_momentum_rate: float | Callable[[Any], Any] = 1.0e-13
    metal_transport: str = "beam"
    metal_diffusion: float = unit_field("m^2 / s", 0.02)
    argon_emission: str = "material"
    metal_ion_friction_rate: float | None = unit_field("m^3 / s", None)

    def _setup_metal(self, metal: TargetMaterial) -> None:
        """Check the options of the metal and fill in its default rates."""
        if self.metal_transport not in ("beam", "diffusion"):
            raise ValueError('metal_transport must be "beam" or "diffusion"')
        if self.metal_diffusion <= 0.0:
            raise ValueError("metal_diffusion must be > 0")
        if self.argon_emission not in ("material", "model"):
            raise ValueError('argon_emission must be "material" or "model"')
        if isinstance(self.ion_energy, str):
            if self.ion_energy not in ("fall", "kinetic"):
                raise ValueError('ion_energy must be "fall", "kinetic" or a number')
        elif self.ion_energy < 0.0:
            raise ValueError("a fixed ion_energy (the fall in V) must be >= 0")
        if self.double_ionization and self.metal_double_ionization_rate is None:
            if metal.second_ionization_energy is None:
                raise ValueError(
                    "double_ionization needs the material's second_ionization_energy"
                )
            self.metal_double_ionization_rate = lotz_ionization(
                metal.second_ionization_energy, metal.ion_outer_electrons
            )
        if self.metal_collisions and self.metal_excitation_rate is None:
            if metal.excitation_energy is None:
                raise ValueError(
                    "metal_collisions needs the material's excitation_energy"
                )
            self.metal_excitation_rate = lumped_excitation_rate(metal.excitation_energy)
        if not callable(self.metal_momentum_rate):
            self.metal_momentum_rate = to_unit(
                self.metal_momentum_rate, "m^3 / s", "metal_momentum_rate"
            )
        friction = self.metal_ion_friction_rate
        self._metal_friction = self.ion_friction_rate if friction is None else friction

    def _build_poisson_pattern(self) -> None:
        """Fix the sparsity of the Poisson matrix: its entries are linear in the face coefficients.

        ``A = -(DIVX Cxx GXF + DIVX Cxy MGY + DIVY Cyy GYF + DIVY Cyx GXYF)`` with diagonal
        ``C``: ``data = B @ concat(cxx, cxy, cyy, cyx)`` for a fixed ``B``.
        """
        terms = [
            (self._divx, self._gxf),
            (self._divx, self._gyxf_masked),
            (self._divy, self._gyf),
            (self._divy, self._gxyf),
        ]
        sizes = [t[0].shape[1] for t in terms]
        offsets = np.r_[0, np.cumsum(sizes)]
        ncell = self.nx * self.ny
        entries = []  # (row, col, coefficient index, value)
        for k, (div, grad) in enumerate(terms):
            div_csc = div.tocsc()
            grad_csr = grad.tocsr()
            for face in range(div.shape[1]):
                drows = div_csc.indices[div_csc.indptr[face] : div_csc.indptr[face + 1]]
                dvals = div_csc.data[div_csc.indptr[face] : div_csc.indptr[face + 1]]
                lo, hi = grad_csr.indptr[face], grad_csr.indptr[face + 1]
                for r, dv in zip(drows, dvals, strict=True):
                    for c, gv in zip(
                        grad_csr.indices[lo:hi], grad_csr.data[lo:hi], strict=True
                    ):
                        entries.append((r, c, offsets[k] + face, -dv * gv))
        rows, cols, idx, vals = (np.array(x) for x in zip(*entries, strict=True))
        key = rows * ncell + cols
        unique, position = np.unique(key, return_inverse=True)
        self._pat_rows = unique // ncell
        self._pat_cols = unique % ncell
        self._pat_b = sp.csr_matrix(
            (vals, (position, idx)), shape=(len(unique), offsets[-1])
        )

    def _build_periodic_x_operators(self, ix, iy, d_yc, d_yf, a_yf, div_y) -> None:
        """The operators of a domain periodic in ``x``: face ``nx`` repeats face 0."""
        nx, ny, dx = self.nx, self.ny, self.dx
        faces = np.arange(nx + 1)
        right, left = faces % nx, (faces - 1) % nx
        d_xf = sp.csr_matrix(
            (
                np.r_[np.full(nx + 1, 1.0 / dx), np.full(nx + 1, -1.0 / dx)],
                (np.r_[faces, faces], np.r_[right, left]),
            ),
            shape=(nx + 1, nx),
        )
        a_xf = sp.csr_matrix(
            (
                np.full(2 * (nx + 1), 0.5),
                (np.r_[faces, faces], np.r_[right, left]),
            ),
            shape=(nx + 1, nx),
        )
        cells = np.arange(nx)
        d_xc = sp.csr_matrix(
            (
                np.r_[np.full(nx, 0.5 / dx), np.full(nx, -0.5 / dx)],
                (np.r_[cells, cells], np.r_[(cells + 1) % nx, (cells - 1) % nx]),
            ),
            shape=(nx, nx),
        )
        div_x = sp.csr_matrix(
            (
                np.r_[np.full(nx, -1.0 / dx), np.full(nx, 1.0 / dx)],
                (np.r_[cells, cells], np.r_[cells, cells + 1]),
            ),
            shape=(nx, nx + 1),
        )
        self._gxf = sp.kron(d_xf, iy, format="csr")
        self._gyxf_masked = sp.kron(a_xf, d_yc, format="csr")
        self._gyf = sp.kron(ix, d_yf, format="csr")
        self._gxyf = sp.kron(d_xc, a_yf, format="csr")
        self._dxc = sp.kron(d_xc, iy, format="csr")
        self._dyc = sp.kron(ix, d_yc, format="csr")
        self._divx = sp.kron(div_x, iy, format="csr")
        self._divy = sp.kron(ix, div_y, format="csr")
        self._pyxf = sp.kron(a_xf, d_yc, format="csr")
        self._axf = sp.kron(a_xf, iy, format="csr")
        self._ayf = sp.kron(ix, a_yf, format="csr")
        self._shape_xf = (nx + 1, ny)
        self._build_poisson_pattern()

    # -- operators ------------------------------------------------------------------

    def _build_operators(self) -> None:
        nx, ny, dx, dy = self.nx, self.ny, self.dx, self.dy
        ix, iy = sp.identity(nx, format="csr"), sp.identity(ny, format="csr")
        shift = sp.csr_matrix(
            (np.ones(ny), (np.arange(ny), (np.arange(ny) - 1) % ny)), shape=(ny, ny)
        )  # (shift f)_j = f_{j-1}
        shift_up = shift.T.tocsr()  # f_{j+1}
        d_yc = (shift_up - shift) / (2.0 * dy)
        d_yf = (iy - shift) / dy
        a_yf = (iy + shift) / 2.0
        div_y = (shift_up - iy) / dy
        # x: cells -> faces (nx + 1 faces), Dirichlet potentials as affine terms
        rows, cols, vals = [], [], []
        if self.periodic_x:
            return self._build_periodic_x_operators(ix, iy, d_yc, d_yf, a_yf, div_y)
        for f in range(1, nx):
            rows += [f, f]
            cols += [f, f - 1]
            vals += [1.0 / dx, -1.0 / dx]
        if self.field_bc is None:
            rows += [0, nx]
            cols += [0, nx - 1]
            vals += [2.0 / dx, -2.0 / dx]
        d_xf = sp.csr_matrix((vals, (rows, cols)), shape=(nx + 1, nx))
        rows, cols, vals = [], [], []
        for f in range(1, nx):
            rows += [f, f]
            cols += [f, f - 1]
            vals += [0.5, 0.5]
        rows += [0, nx]
        cols += [0, nx - 1]
        vals += [1.0, 1.0]
        a_xf = sp.csr_matrix((vals, (rows, cols)), shape=(nx + 1, nx))
        div_x = sp.csr_matrix(
            (
                np.r_[np.full(nx, -1.0 / dx), np.full(nx, 1.0 / dx)],
                (
                    np.r_[np.arange(nx), np.arange(nx)],
                    np.r_[np.arange(nx), np.arange(nx) + 1],
                ),
            ),
            shape=(nx, nx + 1),
        )
        # x-derivative at the cell centres with Dirichlet ghosts (potential)
        rows, cols, vals = [], [], []
        for i in range(1, nx - 1):
            rows += [i, i]
            cols += [i + 1, i - 1]
            vals += [0.5 / dx, -0.5 / dx]
        rows += [0, 0, nx - 1, nx - 1]
        cols += [1, 0, nx - 1, nx - 2]
        if self.field_bc is None:
            vals += [0.5 / dx, 0.5 / dx, -0.5 / dx, -0.5 / dx]
        else:  # fixed field: the ghost cell extends the potential linearly
            vals += [0.5 / dx, -0.5 / dx, 0.5 / dx, -0.5 / dx]
        d_xc = sp.csr_matrix((vals, (rows, cols)), shape=(nx, nx))
        mask = np.ones(nx + 1)
        mask[[0, nx]] = 0.0
        self._gxf = sp.kron(d_xf, iy, format="csr")
        self._gyxf_masked = sp.diags(np.repeat(mask, ny)) @ sp.kron(
            a_xf, d_yc, format="csr"
        )
        self._gyf = sp.kron(ix, d_yf, format="csr")
        self._gxyf = sp.kron(d_xc, a_yf, format="csr")
        self._dxc = sp.kron(d_xc, iy, format="csr")
        self._dyc = sp.kron(ix, d_yc, format="csr")
        self._divx = sp.kron(div_x, iy, format="csr")
        self._divy = sp.kron(ix, div_y, format="csr")
        self._pyxf = sp.kron(a_xf, d_yc, format="csr")  # d_y p averaged to x-faces
        self._axf = sp.kron(a_xf, iy, format="csr")
        self._ayf = sp.kron(ix, a_yf, format="csr")
        self._shape_xf = (nx + 1, ny)
        self._build_poisson_pattern()

    def __post_init__(self) -> None:
        """Build the grid and the sparse operators."""
        convert_units(self)
        if not callable(self.magnetic_field):
            self.magnetic_field = to_unit(self.magnetic_field, "T", "magnetic_field")
        if not callable(self.momentum_rate):
            self.momentum_rate = to_unit(self.momentum_rate, "m^3 / s", "momentum_rate")
        if self.nx < 4 or self.ny < 4:
            raise ValueError("need at least 4 cells in each direction")
        if self.periodic_x and self.field_bc is None:
            raise ValueError("periodic_x needs field_bc, the uniform axial field")
        if self.metal is not None:
            if self.periodic_x:
                raise ValueError("a metal needs the cathode: not with periodic_x")
            if self.metal_speed <= 0.0:
                raise ValueError("metal_speed must be > 0")
            self.metal_mass = self.metal.mass_u * ATOMIC_MASS - ELECTRON_MASS
            if self.metal_ionization_rate is None:
                self.metal_ionization_rate = lotz_ionization(
                    self.metal.ionization_energy, self.metal.outer_electrons
                )
            self._setup_metal(self.metal)
        self.dx = self.length_x / self.nx
        self.dy = self.length_y / self.ny
        self.x = (np.arange(self.nx) + 0.5) * self.dx
        self.xf = np.arange(self.nx + 1) * self.dx
        self.y = (np.arange(self.ny) + 0.5) * self.dy
        self.ion_mass = self.ion_mass_u * ATOMIC_MASS - ELECTRON_MASS
        bfun = self.magnetic_field
        if not callable(bfun):
            value = float(bfun)

            def uniform_field(x):
                return np.full_like(np.asarray(x, dtype=float), value)

            bfun = uniform_field

        self._bc = np.asarray(bfun(self.x), dtype=float)
        self._bf = np.asarray(bfun(self.xf), dtype=float)
        self._k0 = 1.0 if callable(self.momentum_rate) else float(self.momentum_rate)
        self._omega_xf = (E * self._bf / ELECTRON_MASS)[:, None]
        self._omega_yf = (E * self._bc / ELECTRON_MASS)[:, None]
        self._build_operators()

    def _affine(self) -> tuple[np.ndarray, np.ndarray]:
        """Constant parts of the face gradients of the potential from the boundary values."""
        if self.periodic_x:
            assert self.field_bc is not None
            bx = np.full(self.nx + 1, -self.field_bc)
            bc = np.full(self.nx, -self.field_bc)
            return np.repeat(bx, self.ny), np.repeat(bc, self.ny)
        if self.field_bc is not None:
            bx = np.zeros(self.nx + 1)
            bx[[0, -1]] = -self.field_bc
            bc = np.zeros(self.nx)
            bc[[0, -1]] = -0.5 * self.field_bc
            return np.repeat(bx, self.ny), np.repeat(bc, self.ny)
        phi_c, phi_a = -self.voltage, 0.0
        bx = np.zeros(self.nx + 1)
        bx[0] = -2.0 * phi_c / self.dx
        bx[-1] = 2.0 * phi_a / self.dx
        bc = np.zeros(self.nx)
        # ghost = 2 phi_b - phi_edge, so d phi/dx(edge cell) gets phi_b / dx with the sign of the side
        bc[0] = -phi_c / self.dx
        bc[-1] = phi_a / self.dx
        return np.repeat(bx, self.ny), np.repeat(bc, self.ny)

    # -- states ---------------------------------------------------------------------

    def initial_state(
        self,
        density: float | np.ndarray = 1.0e15,
        electron_temperature: float | None = None,
        gas_density: float | np.ndarray | None = None,
        potential: np.ndarray | None = None,
        metal_density: float | np.ndarray = 0.0,
        metal_ion_density: float | np.ndarray = 0.0,
        metal_ion2_density: float | np.ndarray = 0.0,
    ) -> DriftFluid2DState:
        """A quasi-neutral plasma of ``density`` (a number or an ``(nx, ny)`` array).

        The potential is the vacuum one (linear between the electrodes) unless given.
        The densities (m^-3), ``T_e`` (eV) and potential (V) take Quantities or strings
        such as ``"1e15 m^-3"`` too. With a ``metal``, ``density`` is that of the argon ions,
        the metal atoms and ions start at ``metal_density``, ``metal_ion_density`` and (with
        ``double_ionization``) ``metal_ion2_density`` (none by default) at rest, and the
        electrons at the charge density of the ions.
        """
        density = to_array(density, "m^-3", "density")
        electron_temperature = to_array(
            electron_temperature, "eV", "electron_temperature"
        )
        gas_density = to_array(gas_density, "m^-3", "gas_density")
        potential = to_array(potential, "V", "potential")
        shape = (self.nx, self.ny)
        n = np.broadcast_to(np.asarray(density, dtype=float), shape).copy()
        metal = {}
        ne = n
        if self.metal is not None:
            metal_density = to_array(metal_density, "m^-3", "metal_density")
            metal_ion_density = to_array(metal_ion_density, "m^-3", "metal_ion_density")
            nm = np.broadcast_to(np.asarray(metal_ion_density, dtype=float), shape)
            metal = {
                "metal_density": np.broadcast_to(
                    np.asarray(metal_density, dtype=float), shape
                ).copy(),
                "metal_ion_density": nm.copy(),
                "metal_ion_velocity": np.zeros((2, *shape)),
            }
            ne = n + nm
            if self.double_ionization:
                metal_ion2_density = to_array(
                    metal_ion2_density, "m^-3", "metal_ion2_density"
                )
                nm2 = np.broadcast_to(
                    np.asarray(metal_ion2_density, dtype=float), shape
                )
                metal["metal_ion2_density"] = nm2.copy()
                metal["metal_ion2_velocity"] = np.zeros((2, *shape))
                ne = ne + 2.0 * nm2
        te = electron_temperature
        if te is None:
            te = self.electron_temperature if self.electron_temperature else 3.0
        gas = self.gas_density if gas_density is None else gas_density
        if potential is None and self.field_bc is not None:
            potential = np.broadcast_to(
                (-self.field_bc * self.x)[:, None], shape
            ).copy()
        if potential is None:
            potential = np.broadcast_to(
                (-self.voltage * (1.0 - self.x / self.length_x))[:, None], shape
            ).copy()
        return DriftFluid2DState(
            self.x,
            self.y,
            ne,
            n.copy(),
            np.zeros((2, *shape)),
            np.broadcast_to(np.asarray(gas, dtype=float), shape).copy(),
            np.broadcast_to(np.asarray(te, dtype=float), shape).copy(),
            np.asarray(potential, dtype=float).copy(),
            **metal,
        )

    def _bohm_nu(self, omega):
        nu = self.bohm * np.abs(omega) / 16.0
        if self.bohm_below is not None:
            field = ELECTRON_MASS * np.abs(omega) / E
            nu = nu / (1.0 + (field / self.bohm_below) ** 4)
        return nu

    # -- transport coefficients -------------------------------------------------------

    def _collisions(self, gas_xf, gas_yf):
        """``(mu_perp, h)`` on the x- and y-faces."""
        out = []
        for gas, omega in ((gas_xf, self._omega_xf), (gas_yf, self._omega_yf)):
            nu = np.maximum(self._k0 * gas + self._bohm_nu(omega), 1.0)
            out.append((E * nu / (ELECTRON_MASS * (nu**2 + omega**2)), omega / nu))
        return out

    def _metal_momentum(self, te: np.ndarray) -> np.ndarray | float:
        """``k_eM(T_e)``, the electron-metal momentum-transfer rate coefficient (m^3/s)."""
        rate = self.metal_momentum_rate
        return rate(te) if callable(rate) else rate

    def _collision_gas(
        self, gas: np.ndarray, te: np.ndarray, metal: np.ndarray | None = None
    ) -> np.ndarray:
        """The gas density times ``k_en(T_e) / k_0``: what the collision frequency scales with.

        With the metal atom density ``metal`` (and ``metal_collisions``) their momentum
        transfer ``k_eM M / k_0`` is added.
        """
        out = gas * self.momentum_rate(te) if callable(self.momentum_rate) else gas
        if metal is not None and self.metal is not None and self.metal_collisions:
            out = out + metal * self._metal_momentum(te) / self._k0
        return out

    def _energy_loss(
        self,
        ne: np.ndarray,
        te: np.ndarray,
        gas: np.ndarray,
        gas_c,
        metal=None,
        metal_ions=None,
    ):
        """Electron energy loss rate (eV m^-3 s^-1): elastic, ionization and excitation.

        ``gas_c`` is the argon part of `_collision_gas`. With the metal atom density
        ``metal`` its ionization energy is lost too, and with ``metal_collisions`` its
        excitation and elastic losses; with the ``M+`` density ``metal_ions`` (and
        ``double_ionization``) the second ionization energy.
        """
        elastic = (
            3.0
            * ELECTRON_MASS
            / self.ion_mass
            * self._k0
            * gas_c
            * ne
            * (te - self.ion_temperature)
        )
        loss = elastic + self.ionization_energy_ev * self.ionization_rate(te) * gas * ne
        if self.excitation_rate is not None:
            loss = (
                loss + self.excitation_energy_ev * self.excitation_rate(te) * gas * ne
            )
        if metal is not None and self.metal is not None:
            assert self.metal_ionization_rate is not None
            loss = loss + (
                self.metal.ionization_energy
                * self.metal_ionization_rate(te)
                * metal
                * ne
            )
            if self.metal_collisions:
                assert self.metal_excitation_rate is not None
                assert self.metal.excitation_energy is not None
                loss = loss + (
                    self.metal.excitation_energy
                    * self.metal_excitation_rate(te)
                    * metal
                    * ne
                )
                loss = loss + (
                    3.0
                    * ELECTRON_MASS
                    / self.metal_mass
                    * self._metal_momentum(te)
                    * metal
                    * ne
                    * (te - self.ion_temperature)
                )
        if metal_ions is not None and self.metal is not None:
            assert self.metal_double_ionization_rate is not None
            assert self.metal.second_ionization_energy is not None
            loss = loss + (
                self.metal.second_ionization_energy
                * self.metal_double_ionization_rate(te)
                * metal_ions
                * ne
            )
        return loss

    def _dx_cell(self, f: np.ndarray) -> np.ndarray:
        """Central ``d f / dx`` on cells, one-sided half-steps at the ends."""
        if self.periodic_x:
            return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * self.dx)
        d = np.empty_like(f)
        d[1:-1] = (f[2:] - f[:-2]) / (2.0 * self.dx)
        d[0] = (f[1] - f[0]) / (2.0 * self.dx)
        d[-1] = (f[-1] - f[-2]) / (2.0 * self.dx)
        return d

    def _fields(self, phi: np.ndarray):
        """The electric field on the faces and the cells from the potential ``(nx, ny)``."""
        flat = phi.ravel()
        bx, bc = self._affine()
        shape_xf = self._shape_xf
        ex_xf = -(self._gxf @ flat + bx).reshape(shape_xf)
        ey_xf = -(self._gyxf_masked @ flat).reshape(shape_xf)
        ey_yf = -(self._gyf @ flat).reshape(self.nx, self.ny)
        ex_yf = -(self._gxyf @ flat + bc).reshape(self.nx, self.ny)
        ecx = -(self._dxc @ flat + bc).reshape(self.nx, self.ny)
        ecy = -(self._dyc @ flat).reshape(self.nx, self.ny)
        return ex_xf, ey_xf, ex_yf, ey_yf, ecx, ecy

    def _face_gas(self, gas: np.ndarray):
        gas_xf = (self._axf @ gas.ravel()).reshape(self._shape_xf)
        gas_yf = 0.5 * (gas + np.roll(gas, 1, axis=1))
        return gas_xf, gas_yf

    def _pressure_gradients(self, p: np.ndarray):
        """``grad p`` at the x-faces ``(dx, dy)`` and the y-faces ``(dx, dy)``."""
        nx = self.nx
        dpx_xf = np.empty(self._shape_xf)
        if self.periodic_x:
            dpx_xf[:nx] = (p - np.roll(p, 1, axis=0)) / self.dx
            dpx_xf[nx] = dpx_xf[0]
        else:
            dpx_xf[1:nx] = (p[1:] - p[:-1]) / self.dx
            dpx_xf[0] = (p[1] - p[0]) / self.dx
            dpx_xf[nx] = (p[-1] - p[-2]) / self.dx
        dpy_xf = (self._pyxf @ p.ravel()).reshape(self._shape_xf)
        dpy_yf = (p - np.roll(p, 1, axis=1)) / self.dy
        d = self._dx_cell(p)
        dpx_yf = 0.5 * (d + np.roll(d, 1, axis=1))
        scale = self.diamagnetic
        return scale * dpx_xf, scale * dpy_xf, scale * dpx_yf, scale * dpy_yf

    def _face_densities(self, n, mu_xf, h_xf, mu_yf, h_yf, fields):
        """Upwind-biased densities on the faces, by the sign of the E x B-like drift."""
        ex_xf, ey_xf, ex_yf, ey_yf = fields[:4]
        vn_x = -mu_xf * (ex_xf - h_xf * ey_xf)
        vn_y = -mu_yf * (ey_yf + h_yf * ex_yf)
        n_xf = np.maximum(_upwind3(n, vn_x > 0.0, 0, self.periodic_x), 0.0)
        n_yf = np.maximum(_upwind3(n, vn_y > 0.0, 1), 0.0)
        return n_xf, n_yf

    def _electron_like_flux(self, n, te, fields, gas, scale=1.0, lagged=None):
        """``Gamma = -scale mu_perp [[1, -h], [h, 1]] (n E + grad(n T_e))`` on the faces."""
        ex_xf, ey_xf, ex_yf, ey_yf = fields[:4]
        gas_xf, gas_yf = self._face_gas(gas)
        (mu_xf, h_xf), (mu_yf, h_yf) = self._collisions(gas_xf, gas_yf)
        n_xf, n_yf = self._face_densities(
            n, mu_xf, h_xf, mu_yf, h_yf, lagged if lagged is not None else fields
        )
        dpx_xf, dpy_xf, dpx_yf, dpy_yf = self._pressure_gradients(n * te)
        fx_xf = n_xf * ex_xf + dpx_xf
        fy_xf = n_xf * ey_xf + dpy_xf
        fy_yf = n_yf * ey_yf + dpy_yf
        fx_yf = n_yf * ex_yf + dpx_yf
        gam_x = -scale * mu_xf * (fx_xf - h_xf * fy_xf)
        gam_y = -scale * mu_yf * (fy_yf + h_yf * fx_yf)
        return (
            gam_x,
            gam_y,
            (n_xf, n_yf, mu_xf, h_xf, mu_yf, h_yf, dpx_xf, dpy_xf, dpx_yf, dpy_yf),
        )

    def _ion_fluxes(self, ni, vel):
        """Upwind ion fluxes on the faces; outflow only through the electrodes."""
        vx, vy = vel
        nx = self.nx
        vxf = np.empty(self._shape_xf)
        if self.periodic_x:
            vxf[:nx] = 0.5 * (vx + np.roll(vx, 1, axis=0))
            vxf[nx] = vxf[0]
            nxf = np.where(vxf[:nx] > 0.0, np.roll(ni, 1, axis=0), ni)
            nxf = np.vstack([nxf, nxf[:1]])
            vyf = 0.5 * (vy + np.roll(vy, 1, axis=1))
            nyf = np.where(vyf > 0.0, np.roll(ni, 1, axis=1), ni)
            return nxf * vxf, nyf * vyf
        vxf[1:nx] = 0.5 * (vx[1:] + vx[:-1])
        vxf[0], vxf[nx] = vx[0], vx[-1]
        nxf = np.where(vxf > 0.0, np.vstack([ni[:1], ni]), np.vstack([ni, ni[-1:]]))
        gam_x = nxf * vxf
        gam_x[0] = ni[0] * np.minimum(vx[0], 0.0)
        gam_x[nx] = ni[-1] * np.maximum(vx[-1], 0.0)
        vyf = 0.5 * (vy + np.roll(vy, 1, axis=1))
        nyf = np.where(vyf > 0.0, np.roll(ni, 1, axis=1), ni)
        return gam_x, nyf * vyf

    def _cathode_flux(
        self,
        ion_flux: np.ndarray,
        ne=None,
        te=None,
        gas=None,
        ex=None,
        ey=None,
        metal_ion_fluxes=None,
    ) -> np.ndarray:
        """Electron flux into the domain at the cathode, for the ion flux ``ion_flux`` (< 0).

        With a ``metal`` the secondary yield depends on the species: `argon_secondary_yield`
        per ``Ar+`` of ``ion_flux``, ``self_secondary`` per ``M+`` and ``double_secondary``
        per ``M2+`` of ``metal_ion_fluxes`` (``{"M+": flux, "M2+": flux}`` on the cathode
        face, < 0). With ``cathode_absorbs`` (and the cell state ``ne``, ``te``, ``gas`` and the field
        ``ex``, ``ey`` on the cathode face) the drift of the first cell into the cathode,
        where it points that way, is taken off.
        """
        if self.cathode_current_density is not None:
            emitted = np.full_like(ion_flux, self.cathode_current_density / E)
        else:
            emitted = -self.argon_secondary_yield * ion_flux
            if self.metal is not None and metal_ion_fluxes is not None:
                for name, flux in metal_ion_fluxes.items():
                    emitted = emitted - self.metal.secondary_yield(name) * flux
        if not self.cathode_absorbs or ne is None or te is None or gas is None:
            return emitted
        gas_xf, gas_yf = self._face_gas(gas)
        (mu_xf, h_xf), _ = self._collisions(gas_xf, gas_yf)
        dpx_xf, dpy_xf, _, _ = self._pressure_gradients(ne * te)
        fx = ne[0] * ex + dpx_xf[0]
        fy = ne[0] * ey + dpy_xf[0]
        drift = -mu_xf[0] * (fx - h_xf[0] * fy)
        return emitted + np.minimum(drift, 0.0)

    @property
    def argon_secondary_yield(self) -> float:
        """Electrons emitted per ``Ar+``: the material's or ``secondary_emission``.

        See ``argon_emission``; without a metal always ``secondary_emission``.
        """
        if self.metal is None or self.argon_emission == "model":
            return self.secondary_emission
        return self.metal.argon_secondary

    def _solve_linear(self, matrix, rhs: np.ndarray) -> np.ndarray:
        """Solve ``matrix x = rhs``, refining with the last LU factors while they still work.

        The matrix changes by a few percent per stage, so the previous factorisation is a
        very good preconditioner; the iteration runs to a relative residual of 1e-12
        (the solution is as exact as a fresh factorisation) and refactorises when it
        does not get there in a few sweeps.
        """
        scale = np.linalg.norm(rhs) + 1e-300
        if self._lu is not None:
            x = self._lu.solve(rhs)
            for _ in range(10):
                residual = rhs - matrix @ x
                if np.linalg.norm(residual) <= 1e-12 * scale:
                    return x
                x = x + self._lu.solve(residual)
        lu = spla.splu(matrix, permc_spec="COLAMD")
        self._lu = lu
        return lu.solve(rhs)

    def _anode_flux(self, ne, te, gas, ex, ey):
        """Electron flux out through the anode face: the drift of the adjacent cell, outflow only.

        With ``anode_thermal_flux`` the thermal flux of an absorbing wall is added, with
        ``anode_sheath`` that of a full Maxwellian behind the Boltzmann factor of the drop
        to the anode. ``ex``, ``ey`` are the field components on the anode face (length ``ny``).
        """
        gas_xf, gas_yf = self._face_gas(gas)
        (mu_xf, h_xf), _ = self._collisions(gas_xf, gas_yf)
        dpx_xf, dpy_xf, _, _ = self._pressure_gradients(ne * te)
        fx = ne[-1] * ex + dpx_xf[-1]
        fy = ne[-1] * ey + dpy_xf[-1]
        flux = -mu_xf[-1] * (fx - h_xf[-1] * fy)
        mean_speed = np.sqrt(8.0 * E * te[-1] / (np.pi * ELECTRON_MASS))
        if self.anode_sheath:
            # the plasma's potential above the anode over the half cell: E_x dx / 2
            drop = np.maximum(0.5 * self.dx * ex, 0.0)
            barrier = np.exp(-drop / np.maximum(te[-1], TE_FLOOR))
            flux = np.maximum(flux, 0.0) + 0.25 * ne[-1] * mean_speed * barrier
        elif self.anode_thermal_flux:
            flux = flux + 0.5 * ne[-1] * mean_speed
        return np.maximum(flux, 0.0)

    # -- the potential ----------------------------------------------------------------

    @profile("carafe.DriftFluid2D.solve_potential", tags=_PROFILE_TAGS)
    def _solve_potential(
        self, ne, ni, te, gas, dtau, gam_ix, gam_iy, lagged, metal_ions=None
    ):
        """Implicit-permittivity Poisson solve for the potential at the end of an Euler step.

        ``metal_ions`` maps the metal ion species (``"M+"``, ``"M2+"``) to ``(Z, n,
        Gamma_x, Gamma_y)``, or is ``None``.
        """
        nx, ny = self.nx, self.ny
        metal_flux_x = None
        ion_x, ion_y, ions = gam_ix, gam_iy, ni
        if metal_ions is not None:
            metal_flux_x = {}
            for name, (charge, nm, gam_mx, gam_my) in metal_ions.items():
                metal_flux_x[name] = gam_mx[0]
                ion_x = ion_x + charge * gam_mx
                ion_y = ion_y + charge * gam_my
                ions = ions + charge * nm
        gas_xf, gas_yf = self._face_gas(gas)
        (mu_xf, h_xf), (mu_yf, h_yf) = self._collisions(gas_xf, gas_yf)
        n_xf, n_yf = self._face_densities(ne, mu_xf, h_xf, mu_yf, h_yf, lagged)
        dpx_xf, dpy_xf, dpx_yf, dpy_yf = self._pressure_gradients(ne * te)
        sig_x = E * n_xf * mu_xf
        sig_y = E * n_yf * mu_yf
        # no conduction through the electrodes: their electron flux is explicit
        if not self.periodic_x:
            sig_x[[0, nx]] = 0.0
        sxx = EPSILON_0 + dtau * sig_x
        sxy = -dtau * sig_x * h_xf
        if not self.periodic_x:
            sxy[[0, nx]] = 0.0
        syy = EPSILON_0 + dtau * sig_y
        syx = dtau * sig_y * h_yf
        # known current: ions, electron pressure part, and the electrode boundary fluxes
        jk_x = E * ion_x + E * mu_xf * (dpx_xf - h_xf * dpy_xf)
        jk_y = E * ion_y + E * mu_yf * (dpy_yf + h_yf * dpx_yf)
        if not self.periodic_x:
            anode = self._anode_flux(ne, te, gas, lagged[0][-1], lagged[1][-1])
            cathode = self._cathode_flux(
                gam_ix[0],
                ne,
                te,
                gas,
                lagged[0][0],
                lagged[1][0],
                metal_ion_fluxes=metal_flux_x,
            )
            jk_x[0] = E * ion_x[0] - E * cathode
            jk_x[nx] = E * ion_x[nx] - E * anode
        div_jk = self._divx @ jk_x.ravel() + self._divy @ jk_y.ravel()
        rho = E * (ions - ne).ravel()
        src_e = self.external_sources.get("ne", 0.0)
        src_i = self.external_sources.get("ni", 0.0)
        if metal_ions is not None:
            src_i = src_i + self.external_sources.get("nm", 0.0)
            if "M2+" in metal_ions:
                src_i = src_i + 2.0 * self.external_sources.get("nm2", 0.0)
        rho = rho + dtau * E * np.ravel(np.broadcast_to(src_i - src_e, ne.shape))
        bx, bc = self._affine()
        coefficients = np.concatenate(
            [sxx.ravel(), sxy.ravel(), syy.ravel(), syx.ravel()]
        )
        matrix = sp.csc_matrix(
            (self._pat_b @ coefficients, (self._pat_rows, self._pat_cols)),
            shape=(nx * ny, nx * ny),
        )
        rhs = (
            rho
            - dtau * div_jk
            + self._divx @ (sxx.ravel() * bx)
            + self._divy @ (syx.ravel() * bc)
        )
        if self.field_bc is not None:  # pure Neumann: pin the potential of cell 0
            matrix = matrix.tolil()
            matrix[0, 0] = matrix[0, 0] * (1.0 + 1e8)
            matrix = matrix.tocsc()
            rhs[0] = 0.0
        return self._solve_linear(matrix, rhs).reshape(nx, ny)

    def _advect(self, f, vx, vy):
        """``v . grad f`` by first-order upwind; zero gradient at the electrodes."""
        if self.periodic_x:
            fm, fp = np.roll(f, 1, axis=0), np.roll(f, -1, axis=0)
        else:
            fm = np.vstack([f[:1], f[:-1]])
            fp = np.vstack([f[1:], f[-1:]])
        dfx = np.where(vx > 0.0, (f - fm) / self.dx, (fp - f) / self.dx)
        dfy = np.where(
            vy > 0.0,
            (f - np.roll(f, 1, axis=1)) / self.dy,
            (np.roll(f, -1, axis=1) - f) / self.dy,
        )
        return vx * dfx + vy * dfy

    def _ion_momentum(
        self, n, vx, vy, gas, mass, ecx, ecy, charge: int = 1, friction=None
    ):
        """``d v / d t`` of an ion fluid of ``mass`` and ``charge``.

        The field, friction (the rate coefficient ``friction``, ``ion_friction_rate`` by
        default), pressure and upwind advection.
        """
        n_safe = np.maximum(n, N_FLOOR)
        friction = self.ion_friction_rate if friction is None else friction
        nu_i = friction * gas
        c = charge * E / mass
        grad_nx = self._dx_cell(n)
        grad_ny = (np.roll(n, -1, axis=1) - np.roll(n, 1, axis=1)) / (2.0 * self.dy)
        t_over_m = self.ion_temperature * E / mass
        adv_x = self._advect(vx, vx, vy)
        adv_y = self._advect(vy, vx, vy)
        # semi-implicit friction: dv/dt = F - nu v, evaluated for the Euler step dtau
        dvx = c * ecx - t_over_m * grad_nx / n_safe - adv_x - nu_i * vx
        dvy = c * ecy - t_over_m * grad_ny / n_safe - adv_y - nu_i * vy
        return dvx, dvy

    def _cathode_fall(self, phi: np.ndarray) -> np.ndarray:
        """The cathode fall along ``y`` (V): the largest potential along ``x`` above the cathode's.

        With the electrodes the anode (0 V) counts too, so the vacuum fall is ``voltage``.
        """
        top = phi.max(axis=0)
        if self.field_bc is None:
            cathode = np.full(self.ny, -self.voltage)
            top = np.maximum(top, 0.0)
        else:  # the potential extrapolated to the cathode face
            cathode = 1.5 * phi[0] - 0.5 * phi[1]
        return np.maximum(top - cathode, 0.0)

    def _ion_energies(
        self, phi: np.ndarray, velocities: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """The energies (eV) of the ions at the cathode along ``y``, per species (``ion_energy``).

        ``velocities`` maps ``"Ar+"``, ``"M+"``, ``"M2+"`` to the ``v_x`` of their fluid.
        """
        charges = {"Ar+": 1, "M+": 1, "M2+": 2}
        if self.ion_energy == "kinetic":
            masses = {
                "Ar+": self.ion_mass,
                "M+": self.metal_mass,
                "M2+": self.metal_mass,
            }
            return {
                name: 0.5 * masses[name] * np.minimum(vx[0], 0.0) ** 2 / E
                for name, vx in velocities.items()
            }
        if self.ion_energy == "fall":
            fall = self._cathode_fall(phi)
        else:
            fall = np.full(self.ny, float(self.ion_energy))
        return {name: charges[name] * fall for name in velocities}

    def _sputtered_flux(
        self, fluxes: dict[str, np.ndarray], energies: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Metal atoms sputtered into the gap per unit area and time.

        ``sum_s Y_s(E_s) |Gamma_s|`` from the ion fluxes into the cathode (< 0) and their
        energies (eV), per species ``"Ar+"``, ``"M+"``, ``"M2+"``.
        """
        assert self.metal is not None
        total = np.zeros(self.ny)
        for name, flux in fluxes.items():
            total = total - self.metal.sputter_yield(name, energies[name]) * flux
        return total

    def _metal_fluxes(self, atoms, sputtered):
        """Fluxes of the metal atoms on the x- and y-faces: the sputtered flux in at the cathode.

        ``"beam"``: upwind along ``x`` at ``metal_speed``, none along ``y``;
        ``"diffusion"``: ``-D grad M``, absorbed at the anode (``M = 0`` on its face).
        """
        flux = np.empty(self._shape_xf)
        flux[0] = sputtered
        if self.metal_transport == "beam":
            flux[1:] = self.metal_speed * atoms
            return flux, np.zeros_like(atoms)
        d = self.metal_diffusion
        flux[1:-1] = -d * (atoms[1:] - atoms[:-1]) / self.dx
        flux[-1] = 2.0 * d * atoms[-1] / self.dx
        flux_y = -d * (atoms - np.roll(atoms, 1, axis=1)) / self.dy
        return flux, flux_y

    def _metal_rhs(self, u, du, te, gam_ix, metal_ions, ecx, ecy, energies):
        """Add the metal atoms and ions to the time derivatives ``du`` (in place).

        ``metal_ions`` maps ``"M+"`` (and ``"M2+"``) to ``(Z, n, Gamma_x, Gamma_y)``;
        ``energies`` are those of the ions at the cathode (`_ion_energies`).
        """
        assert self.metal is not None and self.metal_ionization_rate is not None
        atoms, nm, ne, ni = u["M"], u["nm"], u["ne"], u["ni"]
        ionized = self.metal_ionization_rate(te) * atoms * ne
        exchanged = self.metal.charge_exchange * atoms * ni
        fluxes = {"Ar+": gam_ix[0]}
        fluxes.update({name: ion[2][0] for name, ion in metal_ions.items()})
        sputtered = self._sputtered_flux(fluxes, energies)
        beam_x, beam_y = self._metal_fluxes(atoms, sputtered)
        div_atoms = (beam_x[1:] - beam_x[:-1]) / self.dx
        if self.metal_transport != "beam":
            div_atoms = div_atoms + (np.roll(beam_y, -1, axis=1) - beam_y) / self.dy
        du["ne"] = du["ne"] + ionized
        du["ni"] = du["ni"] - exchanged
        du["N"] = du["N"] + self.depletion * exchanged
        du["M"] = -div_atoms - ionized - exchanged
        friction = self._metal_friction
        for name, (charge, n, gam_mx, gam_my) in metal_ions.items():
            div_m = (self._divx @ gam_mx.ravel() + self._divy @ gam_my.ravel()).reshape(
                ne.shape
            )
            key = "nm" if name == "M+" else "nm2"
            du[key] = -div_m
            vx, vy = u["v" + key[1:] + "x"], u["v" + key[1:] + "y"]
            du["v" + key[1:] + "x"], du["v" + key[1:] + "y"] = self._ion_momentum(
                n, vx, vy, u["N"], self.metal_mass, ecx, ecy, charge, friction
            )
        du["nm"] = du["nm"] + ionized + exchanged
        if "M2+" in metal_ions:
            assert self.metal_double_ionization_rate is not None
            doubled = self.metal_double_ionization_rate(te) * nm * ne
            du["ne"] = du["ne"] + doubled
            du["nm"] = du["nm"] - doubled
            du["nm2"] = du["nm2"] + doubled

    # -- the right-hand side -----------------------------------------------------------

    def _rhs(self, u: dict, phi_prev: np.ndarray, dtau: float):
        """Time derivatives of the unknowns and the new potential for an Euler step ``dtau``."""
        ne, ni, vx, vy, gas = u["ne"], u["ni"], u["vx"], u["vy"], u["N"]
        energy = "w" in u
        te = (
            np.clip(u["w"] / (1.5 * np.maximum(ne, N_FLOOR)), TE_FLOOR, TE_CEIL)
            if energy
            else np.full_like(ne, self.electron_temperature)
        )
        metal = "M" in u
        gas_argon = self._collision_gas(gas, te)
        gas_c = self._collision_gas(gas, te, u["M"]) if metal else gas_argon
        prev = self._fields(phi_prev)
        gam_ix, gam_iy = self._ion_fluxes(ni, (vx, vy))
        metal_ions = metal_flux_x = None
        if metal:
            metal_ions = {}
            for name, key, charge in (("M+", "nm", 1), ("M2+", "nm2", 2)):
                if key in u:
                    velocity = (u["v" + key[1:] + "x"], u["v" + key[1:] + "y"])
                    gam_mx, gam_my = self._ion_fluxes(u[key], velocity)
                    metal_ions[name] = (charge, u[key], gam_mx, gam_my)
            metal_flux_x = {name: ion[2][0] for name, ion in metal_ions.items()}
        phi = self._solve_potential(
            ne, ni, te, gas_c, dtau, gam_ix, gam_iy, prev, metal_ions
        )
        fields = self._fields(phi)
        gam_x, gam_y, _ = self._electron_like_flux(ne, te, fields, gas_c, lagged=prev)
        if not self.periodic_x:
            gam_x[0] = self._cathode_flux(
                gam_ix[0],
                ne,
                te,
                gas_c,
                prev[0][0],
                prev[1][0],
                metal_ion_fluxes=metal_flux_x,
            )
            # explicit in the field, as in the Poisson solve, so the discrete charge
            # balance is exact
            gam_x[-1] = self._anode_flux(ne, te, gas_c, prev[0][-1], prev[1][-1])
        ecx, ecy = fields[4], fields[5]
        rate = self.ionization_rate(te) * gas * ne
        if self._rate_override is not None:
            rate = self._rate_override
        elif self._frequency_override is not None:
            rate = self._frequency_override * ne
        div_e = (self._divx @ gam_x.ravel() + self._divy @ gam_y.ravel()).reshape(
            ne.shape
        )
        div_i = (self._divx @ gam_ix.ravel() + self._divy @ gam_iy.ravel()).reshape(
            ne.shape
        )
        du = {"ne": -div_e + rate, "ni": -div_i + rate}
        du["N"] = -self.depletion * rate + self.refill_rate * (self.gas_density - gas)
        du["vx"], du["vy"] = self._ion_momentum(
            ni, vx, vy, gas, self.ion_mass, ecx, ecy
        )
        if metal:
            assert metal_ions is not None
            velocities = {"Ar+": vx}
            for name, key in (("M+", "nm"), ("M2+", "nm2")):
                if key in u:
                    velocities[name] = u["v" + key[1:] + "x"]
            energies = self._ion_energies(phi_prev, velocities)
            self._metal_rhs(u, du, te, gam_ix, metal_ions, ecx, ecy, energies)
        if energy:
            q_x, q_y, _ = self._electron_like_flux(
                ne * te * 1.5, te, fields, gas_c, scale=5.0 / 3.0, lagged=prev
            )
            if not self.periodic_x:
                # the emitted electrons at their energy, the returned ones at 2 T_e
                emitted = self._cathode_flux(gam_ix[0], metal_ion_fluxes=metal_flux_x)
                q_x[0] = 2.0 * self.secondary_energy_ev * emitted + 2.0 * te[0] * (
                    gam_x[0] - emitted
                )
                q_x[-1] = 2.0 * te[-1].mean() * np.maximum(gam_x[-1], 0.0)
            div_q = (self._divx @ q_x.ravel() + self._divy @ q_y.ravel()).reshape(
                ne.shape
            )
            gcx = 0.5 * (gam_x[:-1] + gam_x[1:])
            gcy = 0.5 * (gam_y + np.roll(gam_y, -1, axis=1))
            joule = -(gcx * ecx + gcy * ecy)
            metal_atoms, metal_ions_1 = u.get("M"), u.get("nm") if "nm2" in u else None
            loss = self._energy_loss(ne, te, gas, gas_argon, metal_atoms, metal_ions_1)
            du["w"] = -div_q + self.heating_scale * joule - loss
            # d(loss)/dw for the implicit treatment of the loss in `_euler`
            te_up = te * 1.001
            dloss = (
                self._energy_loss(
                    ne,
                    te_up,
                    gas,
                    self._collision_gas(gas, te_up),
                    metal_atoms,
                    metal_ions_1,
                )
                - loss
            ) / (1.5 * np.maximum(ne, N_FLOOR) * 0.001 * te)
            du["_dloss"] = np.maximum(dloss, 0.0)
        for key, source in self.external_sources.items():
            if key in du:
                du[key] = du[key] + source
        return du, phi

    # -- time stepping -----------------------------------------------------------------

    def _pack(self, state: DriftFluid2DState) -> dict:
        u = {
            "ne": state.electron_density.copy(),
            "ni": state.ion_density.copy(),
            "vx": state.ion_velocity[0].copy(),
            "vy": state.ion_velocity[1].copy(),
            "N": state.gas_density.copy(),
        }
        if self.electron_temperature is None:
            u["w"] = 1.5 * state.electron_density * state.electron_temperature
        if self.metal is not None:
            if state.metal_density is None or state.metal_ion_density is None:
                raise ValueError("a model with a metal needs a state with the metal")
            velocity = state.metal_ion_velocity
            if velocity is None:
                velocity = np.zeros((2, *state.electron_density.shape))
            u["M"] = state.metal_density.copy()
            u["nm"] = state.metal_ion_density.copy()
            u["vmx"] = velocity[0].copy()
            u["vmy"] = velocity[1].copy()
            if self.double_ionization:
                if state.metal_ion2_density is None:
                    raise ValueError(
                        "a model with double_ionization needs a state with the M2+"
                    )
                velocity = state.metal_ion2_velocity
                if velocity is None:
                    velocity = np.zeros((2, *state.electron_density.shape))
                u["nm2"] = state.metal_ion2_density.copy()
                u["vm2x"] = velocity[0].copy()
                u["vm2y"] = velocity[1].copy()
        return u

    @profile("carafe.DriftFluid2D.linear_modes", tags=_PROFILE_TAGS)
    @method(
        equations={
            "jacobian": r"J_{:,j} = \frac{\mathcal R(u + \epsilon_j \cos(k y)\, e_j) + i\,\mathcal R(u + \epsilon_j\sin(k y)\, e_j) - (1 + i)\,\mathcal R(u)}{\epsilon_j} \text{ projected on } \frac{e^{-iky}}{n_y}",
            "modes": r"J\,\hat u = s\,\hat u \quad (\texttt{numpy.linalg.eig}), \qquad \epsilon_j = 10^{-6}|u_j| \ (\text{velocities } 10^{-6}(\max|v| + 1))",
        },
        assumptions=(
            r"about a $y$-uniform state made stationary by sources of its own residual; one azimuthal mode, resolved in $x$",
            r"the potential evaluated twice, so the lagged field is the state's own; `frozen` holds $T_e$, the ionization rate or the gas",
        ),
    )
    def linear_modes(
        self,
        state: DriftFluid2DState,
        mode: int,
        frozen: tuple[str, ...] = (),
        epsilon: float = 1e-6,
    ) -> LinearModes:
        """Linear stability of a y-uniform state to the azimuthal mode ``mode``, x-resolved.

        The Jacobian of the semi-discrete operator (the time derivative of every unknown,
        with the potential from Poisson's equation) about ``state``, restricted to
        perturbations ``exp(i k y)``, ``k = 2 pi mode / length_y``: one column per unknown and
        cell, by finite differences with a cosine and a sine perturbation (the operator is
        real and invariant along ``y``, so the response to ``exp(i k y)`` stays in that
        mode). The electrodes, the gradients, the Hall tensor and the energy equation are
        all in it; nothing is x-uniform or local. The state is made stationary by
        sources of its own residual (as `stationary_sources`); ``external_sources``
        are restored afterwards.

        ``frozen`` switches off responses, to see what each does: ``"temperature"`` (``T_e``
        held at its value, no energy perturbation), ``"ionization"`` (the ionization source held
        at its base value), ``"gas"`` (the neutral density not perturbed),
        ``"ionization_frequency"`` (the ionization frequency ``k(T_e) N`` held at its base value,
        so the source still follows ``n_e``: the consistent form of the nonlocal solvers, where
        the source perturbation is ``nu_iz dn_e`` and the equilibrium continuity removes it from
        the density equation; ``"ionization"`` removes the ``dn_e`` part too).

        With ``s = gamma - i omega`` the eigenvalues, ``exp(i k y + s t)``, ``gamma > 0``
        grows and ``omega / k > 0`` moves along ``+y`` (the E x B direction when ``E_x < 0``
        and ``B_z > 0``).
        """
        ny, nx = self.ny, self.nx
        for name in ("electron_density", "ion_density", "electron_temperature"):
            value = getattr(state, name)
            if np.abs(value - value[:, :1]).max() > 1e-9 * np.abs(value).max():
                raise ValueError(f"the state must be uniform along y ({name})")
        k = 2.0 * np.pi * mode / self.length_y
        saved = (
            self.external_sources,
            self.electron_temperature,
            self._rate_override,
            self._frequency_override,
        )
        try:
            if "temperature" in frozen and self.electron_temperature is None:
                self.electron_temperature = state.electron_temperature.copy()
            if "ionization_frequency" in frozen and "ionization" not in frozen:
                self._frequency_override = (
                    self.ionization_rate(state.electron_temperature) * state.gas_density
                )
            if "ionization" in frozen:
                self._rate_override = None
                self._rate_override = (
                    self.ionization_rate(state.electron_temperature)
                    * state.gas_density
                    * state.electron_density
                )
            self.external_sources = {}
            u0 = self._pack(state)
            phi0 = state.potential

            def evaluate(u):
                # the anode flux and the upwind direction use the field of the previous
                # stage: take the one the state itself gives, so the lag is consistent
                _, phi = self._rhs(u, phi0, 0.0)
                return self._rhs(u, phi, 0.0)[0]

            du0 = evaluate(u0)
            self.external_sources = {k_: -v for k_, v in du0.items() if k_ in u0}
            base = evaluate(u0)
            keys = [name for name in u0 if not (name == "N" and "gas" in frozen)]
            cos, sin = np.cos(k * self.y), np.sin(k * self.y)
            phase = np.exp(-1j * k * self.y)
            matrix = np.zeros((len(keys) * nx, len(keys) * nx), complex)
            for column_key, key in enumerate(keys):
                scale = np.abs(u0[key][:, 0]) * epsilon
                scale = np.where(scale > 0.0, scale, epsilon)
                if key in ("vx", "vy"):
                    scale = np.full(nx, epsilon * (np.abs(u0[key]).max() + 1.0))
                for i in range(nx):
                    responses = []
                    for pattern in (cos, sin):
                        u = dict(u0)
                        u[key] = u0[key].copy()
                        u[key][i, :] += scale[i] * pattern
                        responses.append(evaluate(u))
                    for row_key, name in enumerate(keys):
                        response = (
                            (responses[0][name] - base[name])
                            + 1j * (responses[1][name] - base[name])
                        ) / scale[i]
                        matrix[
                            row_key * nx : (row_key + 1) * nx, column_key * nx + i
                        ] = (response @ phase) / ny
        finally:
            (
                self.external_sources,
                self.electron_temperature,
                self._rate_override,
                self._frequency_override,
            ) = saved
        values, vectors = np.linalg.eig(matrix)
        order = np.argsort(-np.real(values))
        profiles = {
            name: vectors[row * nx : (row + 1) * nx][:, order]
            for row, name in enumerate(keys)
        }
        return LinearModes(
            mode, k, values[order], profiles, self.x.copy(), matrix=matrix
        )

    def stationary_sources(self, state: DriftFluid2DState) -> dict[str, np.ndarray]:
        """The external sources that make ``state`` stationary (well-balanced equilibria).

        Sets ``external_sources`` to ``-d u / d t`` of the state without them and returns
        it, so that the state is a steady state of the discrete equations and only a
        perturbation evolves; the way to test linear growth on an equilibrium that
        the physical sources would not hold.
        """
        self.external_sources = {}
        u = self._pack(state)
        du, _ = self._rhs(u, state.potential, 0.0)
        self.external_sources = {key: -value for key, value in du.items() if key in u}
        return self.external_sources

    def _unpack(self, u: dict, phi: np.ndarray, time: float) -> DriftFluid2DState:
        if "w" in u:
            te = np.clip(
                u["w"] / (1.5 * np.maximum(u["ne"], N_FLOOR)), TE_FLOOR, TE_CEIL
            )
        else:
            te = np.full_like(u["ne"], self.electron_temperature)
        metal = {}
        if "M" in u:
            metal = {
                "metal_density": u["M"],
                "metal_ion_density": u["nm"],
                "metal_ion_velocity": np.array([u["vmx"], u["vmy"]]),
            }
            if "nm2" in u:
                metal["metal_ion2_density"] = u["nm2"]
                metal["metal_ion2_velocity"] = np.array([u["vm2x"], u["vm2y"]])
        return DriftFluid2DState(
            self.x,
            self.y,
            u["ne"],
            u["ni"],
            np.array([u["vx"], u["vy"]]),
            u["N"],
            te,
            phi,
            time,
            **metal,
        )

    def _euler(self, u, phi_prev, dt):
        du, phi = self._rhs(u, phi_prev, dt)
        out = {}
        for key, value in u.items():
            if key == "w":
                # backward Euler on the (stiff, local) energy loss, linearised in w
                new = value + dt * du["w"] / (1.0 + dt * du["_dloss"])
                new = np.maximum(new, 1.5 * TE_FLOOR * N_FLOOR)
            else:
                new = value + dt * du[key]
                if key in ("ne", "ni", "N"):
                    clipped = np.maximum(new, N_FLOOR)
                    self.floor_added[key] += float((clipped - new).sum())
                    new = clipped
                elif key in ("M", "nm", "nm2"):  # the metal may vanish: floored at zero
                    clipped = np.maximum(new, 0.0)
                    added = float((clipped - new).sum())
                    self.floor_added[key] = self.floor_added.get(key, 0.0) + added
                    new = clipped
            out[key] = new
        return out, phi

    @profile("carafe.DriftFluid2D.step", tags=_PROFILE_TAGS)
    @method(
        equations={
            "ssp_rk3": r"u^{(1)} = \mathcal E(u^n), \qquad u^{(2)} = \tfrac34 u^n + \tfrac14\mathcal E(u^{(1)}), \qquad u^{n+1} = \tfrac13 u^n + \tfrac23\mathcal E(u^{(2)})",
            "implicit_permittivity": r"-\nabla\cdot\big((\varepsilon_0 + \Delta t\,\boldsymbol\sigma_e)\nabla\phi\big) = \rho - \Delta t\,\nabla\cdot\mathbf J_\mathrm{known}, \qquad \boldsymbol\sigma_e = e n\mu_\perp\begin{pmatrix}1 & -h \\ h & 1\end{pmatrix}",
            "energy_loss": r"w^{n+1} = w + \frac{\Delta t\,\dot w}{1 + \Delta t\,\partial_w P_\mathrm{loss}}",
        },
        order={"density": 3, "velocity": 3},
        assumptions=(
            r"each stage $\mathcal E$ an Euler step with its own potential; electrode fluxes explicit in the previous stage's field, so the discrete Poisson equation of the new densities holds exactly",
            r"`splu(COLAMD)` of the fixed pattern, reused with refinement to $10^{-12}$; densities floored at $10^3$ m$^{-3}$ (`floor_added`)",
            r"the order is that of a uniform plasma (ionization and ion friction, rates $2/T$); the transport is limited by the CFL step",
        ),
        problem=_ssp_problem,
    )
    def step(self, state: DriftFluid2DState, dt: float) -> DriftFluid2DState:
        """Advance by ``dt`` with SSP-RK3; the potential of the result is that of the last stage."""
        u0 = self._pack(state)
        phi = state.potential
        u1, phi = self._euler(u0, phi, dt)
        u2, phi = self._euler(u1, phi, dt)
        u2 = {k: 0.75 * u0[k] + 0.25 * u2[k] for k in u0}
        u3, phi = self._euler(u2, phi, dt)
        u3 = {k: u0[k] / 3.0 + 2.0 / 3.0 * u3[k] for k in u0}
        return self._unpack(u3, phi, state.time + dt)

    @method(
        equations={
            "advection": r"\Delta t = \frac{\mathrm{cfl}}{(v_x + c_s)/\Delta x + (v_y + c_s)/\Delta y}, \qquad c_s = \sqrt{\frac{e(\max T_e + T_i)}{M}}",
            "diffusion": r"\Delta t \le \frac{\mathrm{cfl}/2}{D\,(1/\Delta x^2 + 1/\Delta y^2)}, \qquad D = \tfrac53\max\mu_\perp\max T_e",
        },
        assumptions=(
            r"$v$ the largest face drift or ion speed per axis; $\max T_e$ over the cells with $n_e > 10^{-3}\max n_e$",
        ),
    )
    def suggest_timestep(self, state: DriftFluid2DState, cfl: float = 0.4) -> float:
        """A step with Courant number ``cfl``: electron drift, ions, and diffusion.

        The diffusive limit is that of the (explicit) energy and density diffusion with
        ``D = (5/3) mu_perp T_e``, which becomes the binding one where ``B`` is small
        and the electrons are hot.
        """
        fields = self._fields(state.potential)
        gas_c = self._collision_gas(
            state.gas_density, state.electron_temperature, state.metal_density
        )
        gas_xf, gas_yf = self._face_gas(gas_c)
        (mu_xf, h_xf), (mu_yf, h_yf) = self._collisions(gas_xf, gas_yf)
        vx = np.abs(mu_xf * (fields[0] - h_xf * fields[1]))
        vy = np.abs(mu_yf * (fields[3] + h_yf * fields[2]))
        speed_x = max(vx.max(), np.abs(state.ion_velocity[0]).max())
        speed_y = max(vy.max(), np.abs(state.ion_velocity[1]).max())
        if self.metal is not None and state.metal_ion_velocity is not None:
            metal_v = np.abs(state.metal_ion_velocity)
            speed_x = max(speed_x, metal_v[0].max())
            speed_y = max(speed_y, metal_v[1].max())
            if self.metal_transport == "beam":
                speed_x = max(speed_x, self.metal_speed)
            if state.metal_ion2_velocity is not None:
                metal_v = np.abs(state.metal_ion2_velocity)
                speed_x = max(speed_x, metal_v[0].max())
                speed_y = max(speed_y, metal_v[1].max())
        te = np.max(state.electron_temperature)
        sound = np.sqrt(E * (te + self.ion_temperature) / self.ion_mass)
        step = cfl / ((speed_x + sound) / self.dx + (speed_y + sound) / self.dy)
        mu_max = max(mu_xf.max(), mu_yf.max())
        # cells without electrons (the T_e of w / n is meaningless there) do not set the limit
        populated = state.electron_density > 1e-3 * state.electron_density.max()
        diffusivity = 5.0 / 3.0 * mu_max * state.electron_temperature[populated].max()
        if self.metal is not None and self.metal_transport == "diffusion":
            diffusivity = max(diffusivity, self.metal_diffusion)
        diffusive = 0.5 * cfl / (diffusivity * (1.0 / self.dx**2 + 1.0 / self.dy**2))
        return min(step, diffusive)

    @profile("carafe.DriftFluid2D.run", tags=_PROFILE_TAGS)
    def run(
        self,
        state: DriftFluid2DState,
        t_end: float,
        dt: float | None = None,
        cfl: float = 0.4,
        record_every: float | None = None,
        probe_x: float | None = None,
        callback: Callable[[DriftFluid2DState], None] | None = None,
    ) -> Run:
        """Advance ``state`` by ``t_end`` seconds and record the electron density along ``y``.

        Args:
            state: The initial state.
            t_end: The duration (s).
            dt: A fixed step (s); by default the step is re-chosen from ``cfl``
                every 20 steps.
            cfl: The Courant number of the chosen step.
            record_every: The recording interval (s); ``t_end / 200`` by default.
            probe_x: Where along ``x`` to record (m); mid-gap by default.
            callback: Called with the state at every recording.

        The times and ``probe_x`` take Quantities or strings such as ``"10 us"`` too.
        """
        t_end = to_unit(t_end, "s", "t_end")
        dt = to_unit(dt, "s", "dt")
        record_every = to_unit(record_every, "s", "record_every")
        probe_x = to_unit(probe_x, "m", "probe_x")
        record_every = t_end / 200.0 if record_every is None else record_every
        probe_x = 0.5 * self.length_x if probe_x is None else probe_x
        index = int(np.argmin(np.abs(self.x - probe_x)))
        times, probe = [], []
        series: dict[str, list] = {
            "density_cathode": [],
            "potential": [],
            "e_y": [],
            "v_exb_y": [],
        }

        def record(st: DriftFluid2DState) -> None:
            times.append(st.time)
            probe.append(st.electron_density[index].copy())
            fields = self._fields(st.potential)
            series["density_cathode"].append(st.electron_density[0].copy())
            series["potential"].append(st.potential[index].copy())
            series["e_y"].append(fields[5][index].copy())
            series["v_exb_y"].append(-fields[4][index] / self._bc[index])

        next_record = state.time
        t_stop = state.time + t_end
        steps = 0
        step_size = dt if dt is not None else self.suggest_timestep(state, cfl)
        while state.time < t_stop - 1e-15 * t_end:
            if state.time >= next_record - 1e-12 * record_every:
                record(state)
                next_record += record_every
                if callback is not None:
                    callback(state)
            if dt is None and steps % 20 == 0:
                step_size = self.suggest_timestep(state, cfl)
            h = min(step_size, t_stop - state.time, next_record - state.time)
            h = max(h, 1e-3 * step_size)
            state = self.step(state, h)
            steps += 1
            if not np.all(np.isfinite(state.potential)):
                raise FloatingPointError(f"the run diverged at t = {state.time:.3e} s")
        record(state)
        probe_arr = np.array(probe)
        return Run(
            np.array(times),
            probe_arr,
            float(self.x[index]),
            mode_amplitudes(probe_arr),
            state,
            steps,
            self.length_y,
            {key: np.array(value) for key, value in series.items()},
        )

    # -- diagnostics ---------------------------------------------------------------------

    def exb_velocity(self, state: DriftFluid2DState) -> np.ndarray:
        """The cell-centre E x B drift ``(E x B) / B^2``, shape ``(2, nx, ny)`` (m/s)."""
        ecx, ecy = self._fields(state.potential)[4:6]
        b = self._bc[:, None]
        return np.array([ecy / b, -ecx / b])

    def electric_field(self, state: DriftFluid2DState) -> np.ndarray:
        """The cell-centre electric field ``(E_x, E_y)``, shape ``(2, nx, ny)`` (V/m)."""
        return np.array(self._fields(state.potential)[4:6])

    def electron_flux(self, state: DriftFluid2DState):
        """The electron flux on the x-faces ``(nx + 1, ny)`` and the y-faces ``(nx, ny)`` (m^-2 s^-1)."""
        fields = self._fields(state.potential)
        gx, gy, _ = self._electron_like_flux(
            state.electron_density,
            state.electron_temperature,
            fields,
            self._collision_gas(
                state.gas_density, state.electron_temperature, state.metal_density
            ),
        )
        return gx, gy

    def cathode_budget(self, state: DriftFluid2DState) -> xr.Dataset:
        """What hits and leaves the cathode, per species and along ``y``.

        Per ion species (``"Ar+"``, and with a metal ``"M+"`` and ``"M2+"``): the flux into
        the cathode ``ion_flux`` (m^-2 s^-1, positive), the energy ``ion_energy`` (eV) and
        the sputter yield ``sputter_yield`` at it, the atoms sputtered ``sputtered``, the
        secondary yield ``secondary_yield`` and the current densities (A/m^2) of the ions,
        ``ion_current = Z e Gamma``, and of their secondary electrons,
        ``electron_current = gamma e Gamma`` (zero with a prescribed
        ``cathode_current_density``). Along ``y``: ``emitted``, the electrons emitted (the
        prescribed current or ``sum gamma Gamma``, without the absorption of
        ``cathode_absorbs``), ``current_density`` (A/m^2), the discharge current density
        at the cathode ``e (sum Z Gamma + emitted)`` (``sum (Z + gamma) e Gamma``), and
        ``self_sputtering_share``, the share of the sputtered atoms that the metal ions
        sputter. The fluxes are those the next step starts from (the ion velocities and the
        potential of the state).
        """
        species = {"Ar+": (1, state.ion_density, state.ion_velocity)}
        if self.metal is not None and state.metal_ion_density is not None:
            assert state.metal_ion_velocity is not None
            species["M+"] = (1, state.metal_ion_density, state.metal_ion_velocity)
            if state.metal_ion2_density is not None:
                assert state.metal_ion2_velocity is not None
                species["M2+"] = (
                    2,
                    state.metal_ion2_density,
                    state.metal_ion2_velocity,
                )
        fluxes = {
            name: self._ion_fluxes(n, (v[0], v[1]))[0][0]
            for name, (_, n, v) in species.items()
        }
        names = list(species)
        zero = np.zeros(self.ny)
        if self.metal is not None:
            energies = self._ion_energies(
                state.potential, {name: v[0] for name, (_, _, v) in species.items()}
            )
            yields = {n: self.metal.sputter_yield(n, energies[n]) for n in names}
            gammas = {
                n: self.argon_secondary_yield
                if n == "Ar+"
                else self.metal.secondary_yield(n)
                for n in names
            }
        else:
            energies = {"Ar+": zero}
            yields = {"Ar+": zero}
            gammas = {"Ar+": self.secondary_emission}
        metal_fluxes = {n: fluxes[n] for n in names if n != "Ar+"}
        emitted = self._cathode_flux(fluxes["Ar+"], metal_ion_fluxes=metal_fluxes)
        if self.cathode_current_density is not None:
            gammas = dict.fromkeys(names, 0.0)
        charges = np.array([species[n][0] for n in names], dtype=float)[:, None]
        flux = np.array([-fluxes[n] for n in names])
        energy = np.array([np.broadcast_to(energies[n], (self.ny,)) for n in names])
        sputter = np.array([np.broadcast_to(yields[n], (self.ny,)) for n in names])
        gamma = np.array([gammas[n] for n in names], dtype=float)
        sputtered = sputter * flux
        total = sputtered.sum(axis=0)
        from_metal = sputtered[1:].sum(axis=0)
        share = np.divide(from_metal, total, out=np.zeros(self.ny), where=total > 0.0)
        dims = ("species", "y")
        data = {
            "ion_flux": variable(dims, flux, "ion flux into the cathode", "m^-2 s^-1"),
            "ion_energy": variable(dims, energy, "ion energy at the cathode", "eV"),
            "sputter_yield": variable(dims, sputter, "sputter yield $Y$"),
            "sputtered": variable(dims, sputtered, "sputtered atoms", "m^-2 s^-1"),
            "secondary_yield": variable(
                ("species",), gamma, r"secondary electron yield $\gamma$"
            ),
            "ion_current": variable(
                dims, E * charges * flux, r"ion current density $Z e \Gamma$", "A/m^2"
            ),
            "electron_current": variable(
                dims,
                E * gamma[:, None] * flux,
                r"secondary electron current density $\gamma e \Gamma$",
                "A/m^2",
            ),
            "emitted": variable(("y",), emitted, "emitted electron flux", "m^-2 s^-1"),
            "current_density": variable(
                ("y",),
                E * ((charges * flux).sum(axis=0) + emitted),
                "discharge current density at the cathode",
                "A/m^2",
            ),
            "self_sputtering_share": variable(
                ("y",), share, "self-sputtering share of the sputtered atoms"
            ),
        }
        coords = {
            "species": ("species", names, {"long_name": "ion species"}),
            "y": ("y", self.y, {"long_name": "y (azimuthal)", "units": "m"}),
        }
        return xr.Dataset(data, coords=coords, attrs={"time": state.time})

    def charge_residual(self, state: DriftFluid2DState) -> float:
        """Largest ``|eps0 div E - rho| / (e n)``: how far the state is from Poisson's equation."""
        ex_xf, _, _, ey_yf, _, _ = self._fields(state.potential)
        div_e = (self._divx @ ex_xf.ravel() + self._divy @ ey_yf.ravel()).reshape(
            state.potential.shape
        )
        ions = state.ion_density
        if state.metal_ion_density is not None:
            ions = ions + state.metal_ion_density
        if state.metal_ion2_density is not None:
            ions = ions + 2.0 * state.metal_ion2_density
        rho = E * (ions - state.electron_density)
        scale = E * np.maximum(state.electron_density, N_FLOOR)
        return float(np.max(np.abs(EPSILON_0 * div_e - rho) / scale))
