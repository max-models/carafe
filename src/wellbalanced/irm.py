"""The ionization region model (IRM) of a HiPIMS discharge.

A time-dependent, volume-averaged model of the ionization region (IR) of a pulsed
sputtering magnetron: the annulus above the racetrack, between the radii ``r_1``
and ``r_2`` and the heights ``z_1`` (the cathode sheath) and ``z_2`` above the
target. It follows the structure of the IRM of Raadu et al., *Plasma Sources Sci.
Technol.* 20, 065007 (2011), Huo et al., *Plasma Sources Sci. Technol.* 26, 055003
(2017) and, for a copper target, Gudmundsson et al., *Surf. Coat. Technol.* 442,
128189 (2022). The measured discharge current ``I_D(t)`` and voltage ``V_D(t)`` are
inputs; the back-attraction probability ``beta_t`` and the share ``f`` of the
discharge voltage that falls over the IR are the fitting parameters, found by
matching the model's own discharge current to the measured one
(`IonizationRegion.fit`).

**Species.** Two electron groups: the thermal (cold) Maxwellian electrons ``e``, with
an energy equation for ``T_e``, and the hot electrons ``e_h``, the secondary
electrons accelerated in the cathode sheath, a density at a fixed temperature
``T_H``. Argon: cold ground state ``Ar`` (the filling gas), warm ``Ar_w`` (the ions
that leave for the bulk plasma and return neutralized), hot ``Ar_h`` (the ions
neutralized and reflected at the target, and the gas kicked by the sputtered
atoms), the metastables ``Ar_m``, ``Ar+`` and ``Ar2+``. The metal ``M`` (the target
material), ``M+`` and ``M2+``. Quasi-neutrality gives the thermal electrons,
``n_e = n_Ar+ + 2 n_Ar2+ + n_M+ + 2 n_M2+ - n_h``.

**Volume reactions** (`wellbalanced.Reaction`, with rate coefficients from the
functions of `IRMRates`): electron-impact ionization of every argon group, of the
metastables, of ``Ar+``, of ``M`` and ``M+``; excitation of argon into the
metastables and into radiating levels; elastic collisions on argon; quenching of the
metastables by electrons (superelastic); charge exchange ``Ar+ + M -> Ar + M+``;
Penning ionization ``Ar_m + M -> Ar + M+ + e``; and the gas kicked out of the cold
group by the sputtered atoms, ``Ar + M -> Ar_h + M`` (the sputter wind as
collisions). The hot electrons ionize and excite argon and ionize the metal at the
rates of a Maxwellian at ``T_H``; they become thermal electrons when they have
lost their mean energy ``3/2 T_H`` in these collisions.

**Boundaries.** An ion ``i`` leaves the IR for the bulk (the diffusion region)
through the area ``S_BP`` (the top and the two side walls) at the Bohm speed
``u_B = sqrt(e T_e / M_i)``; a share ``beta_t`` of the ions that leave goes back to
the target instead, so the flux to the target is ``beta_t / (1 - beta_t)`` times the
flux to the bulk. Neutrals leave through ``S_BP`` at a quarter of their mean speed
at their temperature; the cold gas flows in from the bulk at the filling density,
``n_0 v_C / 4 S_BP``, so with no plasma ``n_Ar = n_0``. Metastables reaching the
target (area ``S_RT``) are quenched to ``Ar``. At the target an ion of charge ``Z``
arrives with the energy ``Z e (1 - f) V_D`` (the cathode fall), sputters
``Y = a E^b`` metal atoms (per ion species) and emits ``gamma`` secondary electrons,
which become hot electrons; the argon ions return as ``Ar_h`` and the metal ions
stay on the target. Argon ions that leave for the bulk return as ``Ar_w`` with the
probability ``warm_return``.

**Energy.** The thermal electrons are heated ohmically by the current through the
IR, ``P = I_D V_IR`` with ``V_IR = f V_D``; they lose the energy of the inelastic and
elastic collisions and ``2 T_e`` per electron that leaves for the bulk. The
electrons that leave are those that quasi-neutrality does not keep:
``Gamma_e = S_e - dn_e/dt`` with ``S_e`` every electron made in the volume, which
makes the discharge current continuous: the current at the target,
``I_D = e (sum_i Z_i Gamma_i^t + Gamma_SE)``, equals the current into the bulk,
``e (Gamma_e + Gamma_h - sum_i Z_i Gamma_i^bulk)``, at every instant.

**Outputs.** The densities, ``T_e``, the model's discharge current, the sputtered
flux, ``alpha_t`` (the share of the sputtered atoms that are ionized and return to
the target), the ionized flux fraction of the metal leaving for the bulk, the
self-sputtering share and the rarefaction of the argon.

**Data.** Only the atomic masses and ionization energies, and the argon metastable
energy, are data. Everything else is a **PLACEHOLDER** of the right order of
magnitude, to be replaced before a comparison with an experiment:

- the argon rate coefficients are Lieberman and Lichtenberg's fits (`argon_ionization`,
  `argon_excitation`, `argon_elastic`), with a fixed share of the excitation into
  the metastables; the IRM uses the argon reaction set of Huo et al. / Rudolph et al.
  (cross sections averaged over a Maxwellian);
- the metal ionization (``M -> M+`` and ``M+ -> M2+``), the ionization of ``Ar+``
  and of the metastables are the Lotz approximation (`lotz_ionization`); for copper
  the IRM uses the cross sections of Freund et al., *Phys. Rev. A* 41, 3575 (1990);
- the quenching, Penning, charge-exchange and gas-kick rate coefficients;
- the sputter yield fits ``a E^b`` and the secondary electron yields of every ion
  (`IRMTarget`), the temperatures of the sputtered, hot and warm atoms and of the
  hot electrons, and the geometry (`IonizationRegion`).

An `IonizationRegion` with an unverified target warns.

Not in the model: the energy equation of the hot electrons (their temperature is
fixed), gas heating, the excitation of the metal, the radial structure of the IR and
the spokes; the two-zone (IR and diffusion region) and Ar2+ charge-exchange
extensions of the later IRM versions.

All temperatures are in eV, densities in m^-3, areas in m^2, currents in A,
voltages in V and times in s.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import xarray as xr
from methodcard import method
from plasmacoll.constants import BOLTZMANN, ELECTRON_MASS, ELEMENTARY_CHARGE
from plasmacoll.units import convert_units, to_array, to_unit, unit_field
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scope_profiler import profile

from wellbalanced._profiling import _PROFILE_TAGS
from wellbalanced.datasets import variable
from wellbalanced.magnetron import (
    ARGON_EXCITATION_ENERGY,
    ARGON_IONIZATION_ENERGY,
    COPPER,
    TargetMaterial,
    argon_elastic,
    argon_excitation,
    argon_ionization,
    lotz_ionization,
)
from wellbalanced.reactions import Reaction
from wellbalanced.species import Species

__all__ = [
    "ARGON_METASTABLE_ENERGY",
    "ARGON_SECOND_IONIZATION_ENERGY",
    "COPPER_TARGET",
    "IRMFit",
    "IRMRates",
    "IRMTarget",
    "IonizationRegion",
    "SputterYield",
    "waveform",
]

E = ELEMENTARY_CHARGE

#: The energy of the argon metastables (the 1s5 level) in eV.
ARGON_METASTABLE_ENERGY = 11.55

#: The second ionization energy of argon (``Ar+ -> Ar2+``) in eV.
ARGON_SECOND_IONIZATION_ENERGY = 27.63

#: The share of the argon excitation (Lieberman and Lichtenberg's fit) that goes
#: into the metastables. PLACEHOLDER.
METASTABLE_SHARE = 0.2

#: The names of the argon and electron species.
ARGON = "Ar"
WARM_ARGON = "Ar_w"
HOT_ARGON = "Ar_h"
METASTABLE = "Ar_m"
ARGON_ION = "Ar+"
ARGON_ION2 = "Ar2+"
HOT_ELECTRONS = "e_h"
#: The argon ground-state groups.
ARGON_GROUND = (ARGON, WARM_ARGON, HOT_ARGON)


def waveform(
    times: Sequence[float] | np.ndarray, values: Sequence[float] | np.ndarray
) -> Callable[[float], float]:
    """A measured waveform as a function of the time: linear between the samples.

    ``times`` in s (or Quantities or strings such as ``"[0, 10, 20] us"``) and the
    values in their unit (A or V); zero outside the samples.
    """
    times = np.asarray(to_array(times, "s", "times"), dtype=float)
    values = np.asarray(values, dtype=float)
    if times.shape != values.shape or times.ndim != 1 or len(times) < 2:
        raise ValueError("times and values must be 1D arrays of the same length >= 2")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing")

    def sampled(t: float) -> float:
        return float(np.interp(t, times, values, left=0.0, right=0.0))

    return sampled


@dataclass(frozen=True)
class SputterYield:
    """A sputter yield ``Y = a E^b`` (atoms per ion), ``E`` the ion energy in eV.

    ``a`` is in atoms per ion per eV^b. Zero for ``E <= 0``.
    """

    a: float
    b: float

    def __post_init__(self) -> None:
        """Validate the fit."""
        if self.a < 0.0 or self.b <= 0.0:
            raise ValueError("a sputter yield needs a >= 0 and b > 0")

    def __call__(self, energy: float) -> float:
        """The yield at the ion energy ``energy`` (eV)."""
        return self.a * max(float(energy), 0.0) ** self.b


@dataclass(frozen=True)
class IRMTarget:
    """The target of the IRM: the metal and its response to every ion.

    Attributes:
        material: The metal (name, mass, first ionization energy, outer electrons).
        second_ionization_energy: ``M+ -> M2+`` in eV.
        argon_yield: The sputter yield of ``Ar+`` (``Ar2+`` at twice the energy).
        metal_yield: The sputter yield of ``M+`` (``M2+`` at twice the energy).
        secondary: Secondary electrons emitted per ion, by role: ``"Ar+"``,
            ``"Ar2+"``, ``"M+"``, ``"M2+"``.
        sputtered_temperature: The temperature in eV whose mean speed the sputtered
            atoms leave the region at.
        verified: ``False`` while the yields are placeholders.

    The energies take Quantities or strings such as ``"20.29 eV"`` too.
    """

    material: TargetMaterial = COPPER
    second_ionization_energy: float = unit_field("eV", 20.29)
    argon_yield: SputterYield = SputterYield(0.07, 0.55)
    metal_yield: SputterYield = SputterYield(0.07, 0.55)
    secondary: Mapping[str, float] = field(
        default_factory=lambda: {"Ar+": 0.07, "Ar2+": 0.3, "M+": 0.0, "M2+": 0.05}
    )
    sputtered_temperature: float = unit_field("eV", 2.0)
    verified: bool = False

    def __post_init__(self) -> None:
        """Validate the target."""
        convert_units(self)
        object.__setattr__(self, "secondary", dict(self.secondary))
        if set(self.secondary) != {"Ar+", "Ar2+", "M+", "M2+"}:
            raise ValueError('secondary needs the yields of "Ar+", "Ar2+", "M+", "M2+"')
        if min(self.secondary.values()) < 0.0:
            raise ValueError("a secondary electron yield must be >= 0")
        if self.second_ionization_energy <= self.material.ionization_energy:
            raise ValueError("the second ionization energy must exceed the first")
        if self.sputtered_temperature <= 0.0:
            raise ValueError("sputtered_temperature must be > 0")

    @property
    def name(self) -> str:
        """The element, e.g. ``"Cu"``."""
        return self.material.name


#: A copper target. The second ionization energy is data; the sputter yield fits
#: (about 2 atoms per ion at 500 eV), the secondary electron yields (Ar+ 0.07,
#: Ar2+ 0.3 by potential emission, Cu+ none, Cu2+ 0.05) and the temperature of the
#: sputtered atoms are PLACEHOLDERS: Gudmundsson et al. (2022) take them from
#: TRIM/the literature for copper.
COPPER_TARGET = IRMTarget()


def _metastable_excitation(te: np.ndarray | float) -> np.ndarray | float:
    """Excitation of argon into the metastables, a share of the total. PLACEHOLDER."""
    return METASTABLE_SHARE * argon_excitation(te)


def _radiative_excitation(te: np.ndarray | float) -> np.ndarray | float:
    """Excitation of argon into the radiating levels, the rest. PLACEHOLDER."""
    return (1.0 - METASTABLE_SHARE) * argon_excitation(te)


#: The ionization of the metastables, Lotz at 4.21 eV (one 4s electron). PLACEHOLDER.
_metastable_ionization = lotz_ionization(
    ARGON_IONIZATION_ENERGY - ARGON_METASTABLE_ENERGY, 1
)
#: The ionization of Ar+, Lotz at 27.63 eV (five 3p electrons). PLACEHOLDER.
_argon_ion_ionization = lotz_ionization(ARGON_SECOND_IONIZATION_ENERGY, 5)


@dataclass(frozen=True)
class IRMRates:
    """The rate coefficients of the IRM in m^3/s.

    Every field is a function of the temperature in eV (``T_e`` for the thermal
    electrons, ``T_H`` for the hot ones, ``T_g`` for heavy particles) or a number.
    The defaults are stand-ins (PLACEHOLDER): Lieberman and Lichtenberg's argon fits
    for the argon set of Huo et al. / Rudolph et al., and the Lotz approximation
    for the ionization of the metal (Freund et al. 1990 for copper), of the
    metastables and of ``Ar+``; ``metal_ionization`` and ``metal_ion_ionization``
    default (``None``) to Lotz at the target's ionization energies.

    Attributes:
        argon_ionization: ``e + Ar -> 2e + Ar+`` (every ground-state group).
        argon_metastable: ``e + Ar -> e + Ar_m``.
        argon_excitation: ``e + Ar -> e + Ar*`` into radiating levels (an energy loss).
        argon_elastic: Elastic electron collisions on argon.
        metastable_ionization: ``e + Ar_m -> 2e + Ar+``.
        metastable_quenching: ``e + Ar_m -> e + Ar`` (superelastic).
        argon_ion_ionization: ``e + Ar+ -> 2e + Ar2+``.
        metal_ionization: ``e + M -> 2e + M+``.
        metal_ion_ionization: ``e + M+ -> 2e + M2+``.
        charge_exchange: ``Ar+ + M -> Ar + M+``.
        penning: ``Ar_m + M -> Ar + M+ + e``.
        gas_kick: ``Ar + M -> Ar_h + M``, the cold gas kicked out by sputtered atoms.
    """

    argon_ionization: Callable | float = argon_ionization
    argon_metastable: Callable | float = _metastable_excitation
    argon_excitation: Callable | float = _radiative_excitation
    argon_elastic: Callable | float = argon_elastic
    metastable_ionization: Callable | float = _metastable_ionization
    metastable_quenching: Callable | float = 2.0e-13
    argon_ion_ionization: Callable | float = _argon_ion_ionization
    metal_ionization: Callable | float | None = None
    metal_ion_ionization: Callable | float | None = None
    charge_exchange: Callable | float = 1.0e-15
    penning: Callable | float = 5.0e-16
    gas_kick: Callable | float = 3.0e-16


@method(
    equations={
        "geometry": r"V = \pi (r_2^2 - r_1^2)(z_2 - z_1), \qquad S_\mathrm{RT} = \pi (r_2^2 - r_1^2), \qquad S_\mathrm{BP} = S_\mathrm{RT} + 2\pi (r_1 + r_2)(z_2 - z_1)",
        "quasineutrality": r"n_e = n_{\mathrm{Ar}^+} + 2 n_{\mathrm{Ar}^{2+}} + n_{\mathrm{M}^+} + 2 n_{\mathrm{M}^{2+}} - n_h",
        "ions": r"\Gamma_i^\mathrm{BP} = n_i u_{B,i} S_\mathrm{BP}, \qquad \Gamma_i^t = \frac{\beta_t}{1 - \beta_t}\, \Gamma_i^\mathrm{BP}, \qquad u_{B,i} = \sqrt{e T_e / M_i}",
        "neutrals": r"\frac{dn_X}{dt} \mathrel{-}= \frac{\bar v_X}{4} \frac{S_\mathrm{BP}}{V} n_X, \qquad \frac{dn_\mathrm{Ar}}{dt} \mathrel{+}= \frac{\bar v_\mathrm{Ar}}{4} \frac{S_\mathrm{BP}}{V} n_0, \qquad \bar v_X = \sqrt{\frac{8 e T_X}{\pi m_X}}",
        "target": r"E_i = Z_i e (1 - f) V_D, \qquad \Gamma_\mathrm{sput} = \sum_i Y_i(E_i)\, \Gamma_i^t, \qquad \Gamma_\mathrm{SE} = \sum_i \gamma_i \Gamma_i^t, \qquad Y = a E^b",
        "hot_electrons": r"\frac{dn_h}{dt} = \frac{\Gamma_\mathrm{SE}}{V} - \frac{\sum_j k_j(T_H)\, n_h n_j \varepsilon_j}{\tfrac32 T_H} - \frac{n_h}{\tau_h}",
        "current": r"I_D^\mathrm{model} = e \left(\sum_i Z_i \Gamma_i^t + \Gamma_\mathrm{SE}\right) = e\left(\Gamma_e + \Gamma_h - \sum_i Z_i \Gamma_i^\mathrm{BP}\right)",
        "energy": r"\frac{d}{dt}\left(\tfrac32 n_e T_e\right) = \frac{I_D f V_D}{e V} - \sum_\mathrm{cold} R_j \varepsilon_j - 2 T_e \frac{\Gamma_e}{V}, \qquad \frac{\Gamma_e}{V} = S_e - \frac{dn_e}{dt}",
        "outputs": r"\alpha_t = \frac{\sum_{M^{Z+}} \Gamma_{M^{Z+}}^t}{\Gamma_\mathrm{sput}}, \qquad F_\mathrm{flux} = \frac{\Gamma_{M^+}^\mathrm{BP} + \Gamma_{M^{2+}}^\mathrm{BP}}{\Gamma_M^\mathrm{BP} + \Gamma_{M^+}^\mathrm{BP} + \Gamma_{M^{2+}}^\mathrm{BP}}",
    },
    assumptions=(
        r"a uniform annulus over the racetrack; the measured $I_D(t)$, $V_D(t)$ drive it, $\beta_t \in [0, 1)$ and $f \in (0, 1)$ are fitted",
        r"two electron groups: Maxwellian thermal electrons with an energy equation, hot electrons at a fixed $T_H$ that thermalize after losing $\tfrac32 T_H$",
        r"ions leave at the Bohm speed of the thermal electrons (no factor $Z$); a share $\beta_t$ of them returns to the target",
        r"Ar ions return as hot Ar from the target and as warm Ar from the bulk (probability `warm_return`); metal ions stick",
        r"newborn electrons carry no energy; electrons leave for the bulk with $2 T_e$; the cold gas flows in at the filling density",
        r"rate coefficients, yields, secondary yields, temperatures of the atom groups and the geometry are PLACEHOLDERS",
    ),
    references=(
        "[Raadu et al., Plasma Sources Sci. Technol. 20, 065007 (2011)](https://doi.org/10.1088/0963-0252/20/6/065007)",
        "[Huo et al., Plasma Sources Sci. Technol. 26, 055003 (2017)](https://doi.org/10.1088/1361-6595/aa5f0a)",
        "[Gudmundsson et al., Surf. Coat. Technol. 442, 128189 (2022)](https://doi.org/10.1016/j.surfcoat.2022.128189)",
    ),
    components=(
        Reaction,
        argon_ionization,
        argon_excitation,
        argon_elastic,
        lotz_ionization,
    ),
)
@dataclass(frozen=True)
class IonizationRegion:
    """The ionization region model of a HiPIMS discharge; see the module docstring.

    The defaults are placeholders of the order of a 2 inch magnetron in argon at
    1 Pa, not a particular device.

    Attributes:
        target: The target and its yields.
        rates: The rate coefficients.
        inner_radius: ``r_1``, the inner radius of the annulus in m.
        outer_radius: ``r_2``, its outer radius in m.
        sheath_height: ``z_1``, the height of the cathode sheath, where the IR starts, in m.
        height: ``z_2``, the height of the top of the IR above the target in m.
        pressure: The argon filling pressure in Pa, at ``gas_temperature``.
        gas_temperature: The temperature of the cold gas and the metastables in K.
        back_attraction: ``beta_t``, the probability that an ion leaving the IR
            returns to the target (a fitting parameter, in [0, 1)).
        voltage_fraction: ``f``, the share of the discharge voltage over the IR,
            ``V_IR = f V_D`` (a fitting parameter, in (0, 1)).
        hot_temperature: ``T_H``, the fixed temperature of the hot electrons in eV.
        hot_escape_time: The time in s in which hot electrons leave the IR
            unthermalized; ``None``: they stay until thermalized.
        warm_temperature: The temperature of the warm argon in eV.
        hot_argon_temperature: The temperature of the hot argon in eV.
        warm_return: The share of the argon ions leaving for the bulk that return
            as warm argon.

    The physical settings take Quantities or strings such as ``"1 Pa"`` or
    ``"10 mm"`` too.
    """

    target: IRMTarget = COPPER_TARGET
    rates: IRMRates = IRMRates()
    inner_radius: float = unit_field("m", 0.008)
    outer_radius: float = unit_field("m", 0.020)
    sheath_height: float = unit_field("m", 0.001)
    height: float = unit_field("m", 0.011)
    pressure: float = unit_field("Pa", 1.0)
    gas_temperature: float = unit_field("K", 300.0)
    back_attraction: float = 0.9
    voltage_fraction: float = 0.15
    hot_temperature: float = unit_field("eV", 100.0)
    hot_escape_time: float | None = unit_field("s", None)
    warm_temperature: float = unit_field("eV", 0.1)
    hot_argon_temperature: float = unit_field("eV", 1.0)
    warm_return: float = 1.0

    def _index(self) -> None:
        """The unknowns, the ions and the reactions, computed once."""
        species = self.species
        metal, ion, ion2 = self.metal
        unknown = [name for name in species if name != "e"]
        temperatures = {
            ARGON: BOLTZMANN * self.gas_temperature / E,
            WARM_ARGON: self.warm_temperature,
            HOT_ARGON: self.hot_argon_temperature,
            METASTABLE: BOLTZMANN * self.gas_temperature / E,
            metal: self.target.sputtered_temperature,
        }
        # the quarter of the mean speed through S_BP per volume, 1/s
        exit_rate = {
            name: 0.25
            * np.sqrt(8.0 * E * temperature / (np.pi * species[name].mass))
            * self.bulk_area
            / self.volume
            for name, temperature in temperatures.items()
        }
        ions = {
            ARGON_ION: "Ar+",
            ARGON_ION2: "Ar2+",
            ion: "M+",
            ion2: "M2+",
        }
        object.__setattr__(self, "_unknown", unknown)
        object.__setattr__(self, "_exit_rate", exit_rate)
        object.__setattr__(self, "_ions", ions)
        tg = BOLTZMANN * self.gas_temperature / E
        # (reaction, hot, the coefficient of a hot or heavy-particle reaction, which
        # does not change, else None, the net change as items)
        table = []
        for reaction in self.reactions():
            hot = HOT_ELECTRONS in reaction.reactants
            fixed = None
            if hot:
                fixed = reaction.coefficient(self.hot_temperature, tg)
            elif reaction.temperature == "gas":
                fixed = reaction.coefficient(tg, tg)
            table.append((reaction, hot, fixed, tuple(reaction.stoichiometry.items())))
        object.__setattr__(self, "_reactions", table)

    def __post_init__(self) -> None:
        """Validate the region, and warn about placeholder data."""
        convert_units(self)
        if not 0.0 <= self.inner_radius < self.outer_radius:
            raise ValueError("need 0 <= inner_radius < outer_radius")
        if not 0.0 <= self.sheath_height < self.height:
            raise ValueError("need 0 <= sheath_height < height")
        positive = (
            self.pressure,
            self.gas_temperature,
            self.hot_temperature,
            self.warm_temperature,
            self.hot_argon_temperature,
        )
        if min(positive) <= 0.0:
            raise ValueError("the pressure and the temperatures must be > 0")
        if not 0.0 <= self.back_attraction < 1.0:
            raise ValueError("back_attraction must be in [0, 1)")
        if not 0.0 < self.voltage_fraction < 1.0:
            raise ValueError("voltage_fraction must be in (0, 1)")
        if not 0.0 <= self.warm_return <= 1.0:
            raise ValueError("warm_return must be in [0, 1]")
        if self.hot_escape_time is not None and self.hot_escape_time <= 0.0:
            raise ValueError("hot_escape_time must be > 0")
        if not self.target.verified:
            warnings.warn(
                f"{self.target.name}: the IRM rate coefficients, sputter and secondary "
                "electron yields are placeholders, not data",
                UserWarning,
                stacklevel=2,
            )
        self._index()

    # -- geometry and species ---------------------------------------------------------

    @property
    def target_area(self) -> float:
        """``S_RT``, the racetrack area under the IR in m^2."""
        return np.pi * (self.outer_radius**2 - self.inner_radius**2)

    @property
    def length(self) -> float:
        """``z_2 - z_1``, the height of the IR in m."""
        return self.height - self.sheath_height

    @property
    def volume(self) -> float:
        """The volume of the IR in m^3."""
        return self.target_area * self.length

    @property
    def bulk_area(self) -> float:
        """``S_BP``, the area towards the bulk plasma (top and sides) in m^2."""
        sides = 2.0 * np.pi * (self.inner_radius + self.outer_radius) * self.length
        return self.target_area + sides

    @property
    def gas_density(self) -> float:
        """``n_0``, the argon filling density in m^-3."""
        return self.pressure / (BOLTZMANN * self.gas_temperature)

    @property
    def metal(self) -> tuple[str, str, str]:
        """The names of the metal, its ion and its doubly charged ion."""
        name = self.target.name
        return name, f"{name}+", f"{name}2+"

    @property
    def species(self) -> dict[str, Species]:
        """Every species by name, the two electron groups included."""
        argon = 39.948
        m = self.target.material
        metal, ion, ion2 = self.metal
        return {
            "e": Species("e", ELECTRON_MASS, -1),
            HOT_ELECTRONS: Species(HOT_ELECTRONS, ELECTRON_MASS, -1),
            ARGON: Species.atomic(ARGON, argon),
            WARM_ARGON: Species.atomic(WARM_ARGON, argon),
            HOT_ARGON: Species.atomic(HOT_ARGON, argon),
            METASTABLE: Species.atomic(METASTABLE, argon),
            ARGON_ION: Species.atomic(ARGON_ION, argon, 1),
            ARGON_ION2: Species.atomic(ARGON_ION2, argon, 2),
            metal: Species.atomic(metal, m.mass_u),
            ion: Species.atomic(ion, m.mass_u, 1),
            ion2: Species.atomic(ion2, m.mass_u, 2),
        }

    def reactions(self) -> list[Reaction]:
        """The volume reactions, thermal and hot electrons and heavy particles."""
        r = self.rates
        m = self.target.material
        metal, ion, ion2 = self.metal
        argon_mass = self.species[ARGON].mass
        metal_ionization = r.metal_ionization
        if metal_ionization is None:
            metal_ionization = lotz_ionization(m.ionization_energy, m.outer_electrons)
        metal_ion_ionization = r.metal_ion_ionization
        if metal_ion_ionization is None:
            metal_ion_ionization = lotz_ionization(
                self.target.second_ionization_energy, m.outer_electrons
            )
        reactions = []
        for gas in ARGON_GROUND:
            reactions += [
                Reaction(
                    ("e", gas),
                    ("e", "e", ARGON_ION),
                    r.argon_ionization,
                    electron_energy_loss=ARGON_IONIZATION_ENERGY,
                    name=f"e + {gas} ionization",
                ),
                Reaction(
                    ("e", gas),
                    ("e", METASTABLE),
                    r.argon_metastable,
                    electron_energy_loss=ARGON_METASTABLE_ENERGY,
                    name=f"e + {gas} metastable excitation",
                ),
                Reaction(
                    ("e", gas),
                    ("e", gas),
                    r.argon_excitation,
                    electron_energy_loss=ARGON_EXCITATION_ENERGY,
                    name=f"e + {gas} excitation",
                ),
                Reaction(
                    ("e", gas),
                    ("e", gas),
                    r.argon_elastic,
                    elastic_mass=argon_mass,
                    name=f"e + {gas} elastic",
                ),
                Reaction(
                    (HOT_ELECTRONS, gas),
                    (HOT_ELECTRONS, "e", ARGON_ION),
                    r.argon_ionization,
                    electron_energy_loss=ARGON_IONIZATION_ENERGY,
                    name=f"e_h + {gas} ionization",
                ),
                Reaction(
                    (HOT_ELECTRONS, gas),
                    (HOT_ELECTRONS, METASTABLE),
                    r.argon_metastable,
                    electron_energy_loss=ARGON_METASTABLE_ENERGY,
                    name=f"e_h + {gas} metastable excitation",
                ),
                Reaction(
                    (HOT_ELECTRONS, gas),
                    (HOT_ELECTRONS, gas),
                    r.argon_excitation,
                    electron_energy_loss=ARGON_EXCITATION_ENERGY,
                    name=f"e_h + {gas} excitation",
                ),
            ]
        penning_energy = ARGON_METASTABLE_ENERGY - m.ionization_energy
        reactions += [
            Reaction(
                ("e", METASTABLE),
                ("e", "e", ARGON_ION),
                r.metastable_ionization,
                electron_energy_loss=ARGON_IONIZATION_ENERGY - ARGON_METASTABLE_ENERGY,
                name="e + Ar_m ionization",
            ),
            Reaction(
                ("e", METASTABLE),
                ("e", ARGON),
                r.metastable_quenching,
                electron_energy_loss=-ARGON_METASTABLE_ENERGY,
                name="e + Ar_m quenching",
            ),
            Reaction(
                ("e", ARGON_ION),
                ("e", "e", ARGON_ION2),
                r.argon_ion_ionization,
                electron_energy_loss=ARGON_SECOND_IONIZATION_ENERGY,
                name="e + Ar+ ionization",
            ),
            Reaction(
                ("e", metal),
                ("e", "e", ion),
                metal_ionization,
                electron_energy_loss=m.ionization_energy,
                name=f"e + {metal} ionization",
            ),
            Reaction(
                (HOT_ELECTRONS, metal),
                (HOT_ELECTRONS, "e", ion),
                metal_ionization,
                electron_energy_loss=m.ionization_energy,
                name=f"e_h + {metal} ionization",
            ),
            Reaction(
                ("e", ion),
                ("e", "e", ion2),
                metal_ion_ionization,
                electron_energy_loss=self.target.second_ionization_energy,
                name=f"e + {ion} ionization",
            ),
            Reaction(
                (ARGON_ION, metal),
                (ARGON, ion),
                r.charge_exchange,
                temperature="gas",
                name="charge exchange",
            ),
            Reaction(
                (METASTABLE, metal),
                (ARGON, ion, "e"),
                r.penning,
                temperature="gas",
                # the electron is born with the excess energy
                electron_energy_loss=-max(penning_energy, 0.0),
                name="Penning ionization",
            ),
            Reaction(
                (ARGON, metal),
                (HOT_ARGON, metal),
                r.gas_kick,
                temperature="gas",
                name="gas kick",
            ),
        ]
        return reactions

    # -- the balances -----------------------------------------------------------------

    def initial_state(self, ion_density: float = 1.0e15) -> dict[str, float]:
        """A starting state: the filling gas, a seed plasma of ``ion_density`` (m^-3), ``T_e``."""
        ion_density = to_unit(ion_density, "m^-3", "ion_density")
        metal, ion, ion2 = self.metal
        tiny = 1.0e-6 * ion_density
        return {
            ARGON: self.gas_density,
            WARM_ARGON: tiny,
            HOT_ARGON: tiny,
            METASTABLE: tiny,
            ARGON_ION: ion_density,
            ARGON_ION2: tiny,
            metal: tiny,
            ion: tiny,
            ion2: tiny,
            HOT_ELECTRONS: tiny,
            "Te": 2.0,
        }

    def _vector(self, state: Mapping[str, float]) -> np.ndarray:
        """The state vector (logarithms) of densities by name and ``"Te"``."""
        values = [max(float(state[name]), 1.0e-30) for name in self._unknown]
        return np.log([*values, float(state.get("Te", 2.0))])

    def _terms(
        self, y: np.ndarray, t: float, current: Any, voltage: Any
    ) -> dict[str, Any]:
        """Every term of the balances at the state ``y`` and the time ``t``."""
        species = self.species
        metal, ion, ion2 = self.metal
        volume = self.volume
        n = {
            name: float(np.exp(np.clip(y[k], -600.0, 600.0)))
            for k, name in enumerate(self._unknown)
        }
        tg = BOLTZMANN * self.gas_temperature / E
        te = max(float(np.exp(np.clip(y[-1], -100.0, 100.0))), tg)
        charge = sum(species[name].charge * n[name] for name in self._ions)
        n["e"] = charge - n[HOT_ELECTRONS]
        ne = max(n["e"], 1.0)
        vd = float(voltage(t)) if callable(voltage) else float(voltage)
        f = self.voltage_fraction
        beta = self.back_attraction

        change = dict.fromkeys(n, 0.0)
        rates: dict[str, float] = {}
        cold_energy = 0.0  # eV m^-3 s^-1 the thermal electrons lose
        hot_energy = 0.0  # eV m^-3 s^-1 the hot electrons lose
        cache: dict[int, float] = {}  # coefficients at T_e by rate function
        for reaction, hot, fixed, stoichiometry in self._reactions:
            if fixed is None:
                key = id(reaction.rate)
                if key not in cache:
                    cache[key] = reaction.coefficient(te, tg)
                rate = cache[key]
            else:
                rate = fixed
            for name in reaction.reactants:
                rate *= ne if name == "e" else n[name]
            rates[reaction.name] = rate
            for name, count in stoichiometry:
                change[name] += count * rate
            loss = reaction.electron_loss(te, tg) * rate
            if hot:
                hot_energy += loss
            else:
                cold_energy += loss
        # the hot electrons thermalize after losing their mean energy
        thermalized = hot_energy / (1.5 * self.hot_temperature)
        change[HOT_ELECTRONS] -= thermalized
        change["e"] += thermalized
        escaped = 0.0
        if self.hot_escape_time is not None:
            escaped = n[HOT_ELECTRONS] / self.hot_escape_time
            change[HOT_ELECTRONS] -= escaped

        # ions: to the bulk at the Bohm speed, a share beta to the target
        to_bulk: dict[str, float] = {}  # 1/s
        to_target: dict[str, float] = {}
        for name in self._ions:
            bohm = np.sqrt(E * te / species[name].mass)
            to_bulk[name] = n[name] * bohm * self.bulk_area
            to_target[name] = beta / (1.0 - beta) * to_bulk[name]
            change[name] -= (to_bulk[name] + to_target[name]) / volume
        energy = {name: species[name].charge * (1.0 - f) * vd for name in self._ions}
        target = self.target
        yields = {
            ARGON_ION: target.argon_yield(energy[ARGON_ION]),
            ARGON_ION2: target.argon_yield(energy[ARGON_ION2]),
            ion: target.metal_yield(energy[ion]),
            ion2: target.metal_yield(energy[ion2]),
        }
        sputtered = sum(yields[name] * to_target[name] for name in self._ions)
        secondary = sum(
            target.secondary[role] * to_target[name]
            for name, role in self._ions.items()
        )
        change[metal] += sputtered / volume
        change[HOT_ELECTRONS] += secondary / volume
        # argon ions return as hot argon from the target, as warm argon from the bulk
        warm = self.warm_return * (to_bulk[ARGON_ION] + to_bulk[ARGON_ION2])
        change[HOT_ARGON] += (to_target[ARGON_ION] + to_target[ARGON_ION2]) / volume
        change[WARM_ARGON] += warm / volume

        # neutrals: out through S_BP, the cold gas in; metastables quenched at the target
        neutral_out = {}
        for name, rate in self._exit_rate.items():
            neutral_out[name] = rate * n[name] * volume  # 1/s
            change[name] -= rate * n[name]
        inflow = self._exit_rate[ARGON] * self.gas_density * volume  # 1/s
        change[ARGON] += inflow / volume
        quench = self._exit_rate[METASTABLE] * n[METASTABLE] / self.bulk_area
        quench *= self.target_area
        change[METASTABLE] -= quench
        change[ARGON] += quench

        # the thermal electrons: what quasi-neutrality does not keep leaves
        made = change["e"]  # every thermal electron the volume makes, m^-3 s^-1
        dne = sum(species[name].charge * change[name] for name in self._ions)
        dne -= change[HOT_ELECTRONS]
        electrons_out = made - dne  # m^-3 s^-1
        change["e"] = dne

        model_current = E * (
            sum(species[name].charge * to_target[name] for name in self._ions)
            + secondary
        )
        if current is None:
            drive = model_current
        else:
            drive = float(current(t)) if callable(current) else float(current)
        ohmic = drive * f * vd  # W
        # d(3/2 n_e T_e)/dt = Q - 2 T_e Gamma_e, newborn electrons without energy
        net = ohmic / (E * volume) - cold_energy - 2.0 * te * electrons_out
        dte = (net - 1.5 * te * dne) / (1.5 * ne)
        return {
            "densities": n,
            "te": te,
            "change": change,
            "rates": rates,
            "to_bulk": to_bulk,
            "to_target": to_target,
            "neutral_out": neutral_out,
            "inflow": inflow,
            "warm_return": warm,
            "sputtered": sputtered,
            "secondary": secondary,
            "electrons_out": electrons_out * volume,
            "hot_escaped": escaped * volume,
            "model_current": model_current,
            "current": drive,
            "voltage": vd,
            "ohmic": ohmic,
            "dte": dte,
        }

    def _derivative(
        self, y: np.ndarray, t: float, current: Any, voltage: Any
    ) -> np.ndarray:
        """``dy/dt`` in the logarithms of the densities and ``T_e``."""
        terms = self._terms(y, t, current, voltage)
        n, change = terms["densities"], terms["change"]
        rates = [change[name] / n[name] for name in self._unknown]
        rates.append(terms["dte"] / terms["te"])
        return np.array(rates)

    @staticmethod
    def _drive(current: Any, voltage: Any) -> tuple[Any, Any]:
        """Numbers in A and V, functions of the time, or ``None`` for the current."""
        if current is not None and not callable(current):
            current = to_unit(current, "A", "current")
        if not callable(voltage):
            voltage = to_unit(voltage, "V", "voltage")
        return current, voltage

    def diagnostics(
        self, state: Mapping[str, float], t: float, current: Any, voltage: Any
    ) -> dict[str, Any]:
        """Every term of the balances for a state (densities by name and ``"Te"``).

        ``current`` and ``voltage`` as in `evolve`. Returns the densities, ``Te``,
        ``change`` (``dn/dt`` by species, m^-3 s^-1), the reaction rates, the ion
        fluxes ``to_bulk`` and ``to_target`` and the neutral fluxes ``neutral_out``
        (1/s), the cold gas ``inflow`` and the ``warm_return`` (1/s), the
        ``sputtered`` and ``secondary`` electron fluxes (1/s), ``electrons_out``
        and ``hot_escaped`` (1/s, to the bulk), the ``model_current`` and the
        driving ``current`` (A), the ``voltage`` (V), the ``ohmic`` power (W) and
        the outputs ``alpha_t``, ``ionized_flux_fraction``,
        ``self_sputtering_share``, ``rarefaction`` (cold argon over ``n_0``) and
        ``argon_rarefaction`` (all argon atoms over ``n_0``).
        """
        current, voltage = self._drive(current, voltage)
        terms = self._terms(self._vector(state), t, current, voltage)
        metal, ion, ion2 = self.metal
        n = terms["densities"]
        to_target, to_bulk = terms["to_target"], terms["to_bulk"]
        sputtered = terms["sputtered"]
        metal_ions_back = to_target[ion] + to_target[ion2]
        metal_out = terms["neutral_out"][metal] + to_bulk[ion] + to_bulk[ion2]
        fall = (1.0 - self.voltage_fraction) * terms["voltage"]
        self_sputtered = sum(
            self.target.metal_yield(self.species[k].charge * fall) * to_target[k]
            for k in (ion, ion2)
        )
        argon = sum(n[name] for name in (*ARGON_GROUND, METASTABLE))
        terms.update(
            alpha_t=metal_ions_back / sputtered if sputtered > 0.0 else 0.0,
            ionized_flux_fraction=(to_bulk[ion] + to_bulk[ion2]) / metal_out
            if metal_out > 0.0
            else 0.0,
            self_sputtering_share=self_sputtered / sputtered
            if sputtered > 0.0
            else 0.0,
            rarefaction=n[ARGON] / self.gas_density,
            argon_rarefaction=argon / self.gas_density,
        )
        terms["Te"] = terms.pop("te")
        terms.pop("dte")
        return terms

    def _series(
        self, times: np.ndarray, columns: list[np.ndarray], current: Any, voltage: Any
    ) -> xr.Dataset:
        """The states at ``times`` as a dataset."""
        names = [*self._unknown, "e"]
        rows = []
        for t, y in zip(times, columns, strict=True):
            state = dict(zip([*self._unknown, "Te"], np.exp(y), strict=True))
            rows.append(self.diagnostics(state, float(t), current, voltage))
        data = xr.Dataset(
            {
                "density": variable(
                    ("t", "species"),
                    [[row["densities"][s] for s in names] for row in rows],
                    "density",
                    "m^-3",
                ),
                "Te": variable(
                    "t", [row["Te"] for row in rows], "electron temperature", "eV"
                ),
            },
            coords={
                "t": ("t", times, {"long_name": "time", "units": "s"}),
                "species": ("species", names, {"long_name": "species"}),
            },
        )
        for key, long_name, units in (
            ("current", "discharge current (input)", "A"),
            ("model_current", "discharge current of the model", "A"),
            ("voltage", "discharge voltage", "V"),
            ("ohmic", "ohmic heating power", "W"),
            ("sputtered", "sputtered atoms", "s^-1"),
            ("secondary", "secondary electrons", "s^-1"),
            ("electrons_out", "thermal electrons to the bulk", "s^-1"),
            ("alpha_t", r"ionized and back-attracted share $\alpha_t$", ""),
            ("ionized_flux_fraction", "ionized flux fraction of the metal", ""),
            ("self_sputtering_share", "self-sputtering share", ""),
            ("rarefaction", "cold argon over the filling density", ""),
            ("argon_rarefaction", "argon atoms over the filling density", ""),
        ):
            data[key] = variable("t", [row[key] for row in rows], long_name, units)
        data.attrs.update(
            back_attraction=self.back_attraction,
            voltage_fraction=self.voltage_fraction,
            metal=self.target.name,
        )
        return data

    # -- solving ----------------------------------------------------------------------

    @profile("wellbalanced.IonizationRegion.evolve", tags=_PROFILE_TAGS)
    def evolve(
        self,
        times: Sequence[float] | np.ndarray,
        current: float | Callable[[float], float] | None,
        voltage: float | Callable[[float], float],
        initial: Mapping[str, float] | None = None,
        *,
        breakpoints: Sequence[float] = (),
        rtol: float = 1.0e-6,
    ) -> xr.Dataset:
        """The IR driven by ``I_D(t)`` and ``V_D(t)``, as a dataset over the time ``t``.

        ``current`` and ``voltage`` are numbers (A, V), Quantities, strings such
        as ``"20 A"``, or functions of the time in s (e.g. a measured `waveform`).
        ``current=None`` heats the electrons with the model's own discharge current,
        a discharge driven by the voltage alone; its current is a waveform the
        model reproduces exactly (synthetic data for `fit`). The integration
        (implicit, in the logarithms of the densities and ``T_e``, so they stay
        positive) starts at ``times[0]`` from ``initial`` (`initial_state` by
        default) and stops at every ``breakpoints`` time, where the drive jumps.
        """
        times = np.asarray(to_array(times, "s", "times"), dtype=float)
        if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be a strictly increasing 1D sequence")
        current, voltage = self._drive(current, voltage)
        breakpoints = np.atleast_1d(to_array(breakpoints, "s", "breakpoints"))
        edges = sorted({float(b) for b in breakpoints if times[0] < b < times[-1]})
        edges = [float(times[0]), *edges, float(times[-1])]
        y = self._vector(self.initial_state() if initial is None else initial)
        columns = [y]
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            inside = times[(times > lo) & (times <= hi)]
            run = solve_ivp(
                lambda t, y: self._derivative(y, t, current, voltage),
                (lo, hi),
                y,
                method="BDF",
                t_eval=inside,
                rtol=rtol,
                atol=1.0e-9,
            )
            if not run.success:
                raise RuntimeError(f"the time integration failed: {run.message}")
            columns += list(run.y.T)
            y = run.y[:, -1] if run.y.size else y
        return self._series(times, columns, current, voltage)

    @profile("wellbalanced.IonizationRegion.fit", tags=_PROFILE_TAGS)
    @method(
        equations={
            "residual": r"r_k(\beta_t, f) = \frac{I_D^\mathrm{model}(t_k; \beta_t, f) - I_D(t_k)}{\max_k I_D(t_k)}",
            "fit": r"(\beta_t, f) = \arg\min \sum_k r_k^2, \qquad \beta_t \in [0, \beta_\max],\ f \in [f_\min, f_\max]",
        },
        assumptions=(
            r"the model is driven by the measured $I_D(t)$ and $V_D(t)$, interpolated linearly between the samples",
            r"`scipy.optimize.least_squares` (trust region reflective) with finite-difference Jacobians",
            r"the other parameters of the region are held fixed",
        ),
        references=(
            "[Huo et al., Plasma Sources Sci. Technol. 26, 055003 (2017)](https://doi.org/10.1088/1361-6595/aa5f0a)",
        ),
    )
    def fit(
        self,
        times: Sequence[float] | np.ndarray,
        current: Sequence[float] | np.ndarray,
        voltage: Sequence[float] | np.ndarray,
        initial: Mapping[str, float] | None = None,
        *,
        guess: tuple[float, float] | None = None,
        bounds: tuple[tuple[float, float], tuple[float, float]] = (
            (0.0, 0.99),
            (0.02, 0.9),
        ),
        breakpoints: Sequence[float] = (),
        tolerance: float = 1.0e-6,
    ) -> IRMFit:
        """Fit ``beta_t`` and ``f`` to a measured discharge current.

        ``times`` (s), ``current`` (A) and ``voltage`` (V) are the samples of the
        measured waveforms. The model, driven by them, runs from ``initial`` for
        each trial ``(beta_t, f)``; the least-squares fit makes its own discharge
        current match ``current`` at the samples. ``guess`` starts it (this
        region's values by default); ``bounds`` are those of ``beta_t`` and ``f``;
        ``tolerance`` the relative change of the parameters and of the squared
        mismatch at which the fit stops.
        """
        times = np.asarray(to_array(times, "s", "times"), dtype=float)
        measured = np.asarray(current, dtype=float)
        drive_current = waveform(times, measured)
        drive_voltage = waveform(times, np.asarray(voltage, dtype=float))
        scale = max(float(np.max(np.abs(measured))), 1.0e-30)
        start = guess or (self.back_attraction, self.voltage_fraction)
        (b_lo, b_hi), (f_lo, f_hi) = bounds

        def region(x: np.ndarray) -> IonizationRegion:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return dataclasses.replace(
                    self, back_attraction=float(x[0]), voltage_fraction=float(x[1])
                )

        def residual(x: np.ndarray) -> np.ndarray:
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    data = region(x).evolve(
                        times,
                        drive_current,
                        drive_voltage,
                        initial,
                        breakpoints=breakpoints,
                    )
            except RuntimeError:
                return np.full(len(times), 10.0)
            return (data.model_current.values - measured) / scale

        result = least_squares(
            residual,
            np.clip(start, [b_lo, f_lo], [b_hi, f_hi]),
            bounds=([b_lo, f_lo], [b_hi, f_hi]),
            x_scale=[0.1, 0.1],
            diff_step=1.0e-4,
            xtol=tolerance,
            ftol=tolerance,
            gtol=tolerance,
        )
        fitted = region(result.x)
        data = fitted.evolve(
            times, drive_current, drive_voltage, initial, breakpoints=breakpoints
        )
        rms = float(np.sqrt(np.mean(result.fun**2)))
        return IRMFit(
            back_attraction=float(result.x[0]),
            voltage_fraction=float(result.x[1]),
            region=fitted,
            data=data,
            relative_rms=rms,
            success=bool(result.success),
            evaluations=int(result.nfev),
        )


@dataclass(frozen=True)
class IRMFit:
    """The result of `IonizationRegion.fit`.

    Attributes:
        back_attraction: The fitted ``beta_t``.
        voltage_fraction: The fitted ``f``.
        region: The region with the fitted parameters.
        data: Its run on the measured waveforms (`IonizationRegion.evolve`).
        relative_rms: The rms of the current mismatch over the peak current.
        success: Whether the least-squares solver converged.
        evaluations: The runs of the model the fit took (without the Jacobians).
    """

    back_attraction: float
    voltage_fraction: float
    region: IonizationRegion
    data: xr.Dataset
    relative_rms: float
    success: bool
    evaluations: int
