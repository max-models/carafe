"""The response of a sputtering target to the ions: sputter yields and secondary electrons.

Shared by the models of the magnetron target (`carafe.magnetron.TargetMaterial`, used by
`carafe.magnetron.Magnetron`, `carafe.driftfluid2d.DriftFluid2D` and
`carafe.irm.IonizationRegion`):

- **sputter yields** ``Y(E)``, atoms sputtered per ion of kinetic energy ``E`` (eV) at
  normal incidence, as callables of the energy: `BohdanskyYield` (the Matsunami-type
  formula of Bohdansky with the Kr–C nuclear stopping and the empirical threshold),
  `PowerLawYield` (``Y = a E^b`` above a threshold, the fits of the ionization region
  models) and `ConstantYield` (one number, whatever the energy). A doubly charged ion
  of the cathode fall ``V`` arrives with ``2 e V``: the yield depends on the energy and
  the projectile atom, not on the charge;
- the **potential emission** yield `potential_emission_yield`, the secondary electrons
  per ion from the empirical formula of Baragiola et al., ``gamma = 0.032 (0.78 E_iz -
  2 W)``: an ion emits only if ``0.78 E_iz > 2 W``, which a singly charged metal ion on
  its own target does not satisfy and a doubly charged one (``E_iz`` its second ionization
  energy) does (Anders 2008).

PLACEHOLDERS: the factor ``Q`` of every `BohdanskyYield` of `carafe.magnetron.MATERIALS`
(anchored at one yield at 500 eV, see there); Bohdansky's formula itself is good to some
tens of percent above a few times the threshold and poorer near it. Replace by the fitted
``Q`` and ``E_th`` of Eckstein (*Sputtering by Particle Bombardment*, Topics Appl. Phys. 110,
2007), the Yamamura–Tawara tables (*At. Data Nucl. Data Tables* 62, 149, 1996), SRIM runs
or the measurements collected by Depla and Mahieu (*Reactive Sputter Deposition*, 2008).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from methodcard import Rule, method

__all__ = [
    "BohdanskyYield",
    "ConstantYield",
    "PowerLawYield",
    "SputterYield",
    "as_yield",
    "bohdansky_threshold",
    "krc_nuclear_stopping",
    "potential_emission_yield",
    "reduced_energy",
]

#: ``e^2 / (4 pi eps0)`` in eV Angstrom.
_E2 = 14.399645
#: The Lindhard screening length ``(9 pi^2 / 128)^(1/3) a_0`` in Angstrom.
_LINDHARD = 0.46850


@runtime_checkable
class SputterYield(Protocol):
    """A sputter yield: atoms per ion as a function of the ion energy (eV)."""

    def __call__(self, energy: np.ndarray | float) -> np.ndarray | float:
        """The yield at ``energy`` (eV)."""
        ...


@dataclass(frozen=True)
class ConstantYield:
    """The same yield at every energy: ``Y = value`` (a number of atoms per ion)."""

    value: float

    def __post_init__(self) -> None:
        """Validate the yield."""
        if self.value < 0.0:
            raise ValueError("a sputter yield must be >= 0")

    def __call__(self, energy: np.ndarray | float) -> np.ndarray | float:
        """``value``, shaped as ``energy``."""
        if np.ndim(energy):
            return np.full(np.shape(energy), self.value)
        return self.value


@dataclass(frozen=True)
class PowerLawYield:
    """A sputter yield ``Y = a E^b`` above ``threshold`` (eV), zero below.

    ``a`` is in atoms per ion per eV^b. The fits of the ionization region models
    (Raadu et al. 2011, Gudmundsson et al. 2022) have no threshold.
    """

    a: float
    b: float
    threshold: float = 0.0

    def __post_init__(self) -> None:
        """Validate the fit."""
        if self.a < 0.0 or self.b <= 0.0 or self.threshold < 0.0:
            raise ValueError("a sputter yield needs a >= 0, b > 0 and threshold >= 0")

    def __call__(self, energy: np.ndarray | float) -> np.ndarray | float:
        """The yield at the ion energy ``energy`` (eV)."""
        e = np.asarray(energy, dtype=float)
        out = np.where(e > self.threshold, self.a * np.maximum(e, 0.0) ** self.b, 0.0)
        return out if out.ndim else float(out)


def as_yield(value: SputterYield | float) -> SputterYield:
    """A yield model from a yield model or a number (`ConstantYield`)."""
    if callable(value):
        return value
    return ConstantYield(float(value))


def reduced_energy(
    energy: np.ndarray | float,
    projectile_z: int,
    projectile_mass_u: float,
    target_z: int,
    target_mass_u: float,
) -> np.ndarray | float:
    """Lindhard's reduced energy ``eps = E M_2 / (M_1 + M_2) a_L / (Z_1 Z_2 e^2)``.

    ``a_L = 0.4685 A / sqrt(Z_1^(2/3) + Z_2^(2/3))``; ``E`` in eV.
    """
    z1, z2 = projectile_z, target_z
    screening = _LINDHARD / np.sqrt(z1 ** (2.0 / 3.0) + z2 ** (2.0 / 3.0))
    mass = target_mass_u / (projectile_mass_u + target_mass_u)
    return np.asarray(energy, dtype=float) * mass * screening / (z1 * z2 * _E2)


def krc_nuclear_stopping(epsilon: np.ndarray | float) -> np.ndarray | float:
    """The reduced nuclear stopping of the Kr–C potential (Eckstein's fit).

    ``s_n = 0.5 ln(1 + 1.2288 eps) / (eps + 0.1728 sqrt(eps) + 0.008 eps^0.1504)``.
    """
    eps = np.maximum(np.asarray(epsilon, dtype=float), 1e-300)
    return (
        0.5
        * np.log1p(1.2288 * eps)
        / (eps + 0.1728 * np.sqrt(eps) + 0.008 * eps**0.1504)
    )


def bohdansky_threshold(
    projectile_mass_u: float, target_mass_u: float, surface_binding_energy: float
) -> float:
    """Bohdansky's empirical sputter threshold (eV) at normal incidence.

    ``E_th = U_s / (g (1 - g))`` for ``M_1 / M_2 <= 0.2`` (light ions),
    ``E_th = 8 U_s (M_1 / M_2)^(2/5)`` otherwise; ``g = 4 M_1 M_2 / (M_1 + M_2)^2`` the
    energy transfer factor, ``U_s`` the surface binding energy (eV).
    """
    m1, m2 = projectile_mass_u, target_mass_u
    if m1 / m2 <= 0.2:
        transfer = 4.0 * m1 * m2 / (m1 + m2) ** 2
        return surface_binding_energy / (transfer * (1.0 - transfer))
    return 8.0 * surface_binding_energy * (m1 / m2) ** 0.4


def _bohdansky_rules() -> dict:
    """The SymPy rule of `BohdanskyYield` (for the tests)."""
    import sympy

    e, q, th = sympy.symbols("energy q threshold", positive=True)
    z1, m1, z2, m2 = 18, 39.948, 29, 63.546
    screening = sympy.Float(_LINDHARD) / sympy.sqrt(
        sympy.Integer(z1) ** sympy.Rational(2, 3)
        + sympy.Integer(z2) ** sympy.Rational(2, 3)
    )
    eps = e * sympy.Float(m2 / (m1 + m2)) * screening / (z1 * z2 * sympy.Float(_E2))
    stopping = (
        sympy.Rational(1, 2)
        * sympy.log(1 + sympy.Float("1.2288") * eps)
        / (
            eps
            + sympy.Float("0.1728") * sympy.sqrt(eps)
            + sympy.Float("0.008") * eps ** sympy.Float("0.1504")
        )
    )
    rule = q * stopping * (1 - (th / e) ** sympy.Rational(2, 3)) * (1 - th / e) ** 2
    return {
        "yield": Rule(
            rule,
            implementation=lambda energy, q, threshold: BohdanskyYield(
                z1, m1, z2, m2, 3.49, q, threshold
            )(energy),
            ranges={
                "energy": (100.0, 1000.0),
                "q": (1.0, 10.0),
                "threshold": (5.0, 30.0),
            },
        )
    }


@method(
    equations={
        "yield": r"Y(E) = Q\, s_n^\mathrm{KrC}(\varepsilon)\left[1 - \left(\frac{E_\mathrm{th}}{E}\right)^{2/3}\right]\left(1 - \frac{E_\mathrm{th}}{E}\right)^2 \quad (E > E_\mathrm{th})",
        "reduced_energy": r"\varepsilon = E\,\frac{M_2}{M_1 + M_2}\,\frac{a_L}{Z_1 Z_2 e^2}, \qquad a_L = \frac{0.4685\ \text{Å}}{\sqrt{Z_1^{2/3} + Z_2^{2/3}}}",
        "stopping": r"s_n^\mathrm{KrC} = \frac{0.5\ln(1 + 1.2288\,\varepsilon)}{\varepsilon + 0.1728\sqrt\varepsilon + 0.008\,\varepsilon^{0.1504}}",
        "threshold": r"E_\mathrm{th} = \frac{U_s}{\gamma(1 - \gamma)}\ \left(\tfrac{M_1}{M_2} \le 0.2\right), \qquad E_\mathrm{th} = 8 U_s \left(\frac{M_1}{M_2}\right)^{2/5}\ \text{otherwise}, \qquad \gamma = \frac{4 M_1 M_2}{(M_1 + M_2)^2}",
    },
    assumptions=(
        r"normal incidence on a clean, amorphous target; the ion's charge does not matter, only its energy",
        r"$Q$ a fitted factor per projectile–target pair (PLACEHOLDER in `MATERIALS`: anchored at one yield at 500 eV); $U_s$ the sublimation energy",
        r"good to some tens of percent above a few times $E_\mathrm{th}$, poorer near the threshold",
    ),
    references=(
        "[Bohdansky, Nucl. Instrum. Methods B 2, 587 (1984)](https://doi.org/10.1016/0168-583X(84)90271-4)",
        "Eckstein, in Sputtering by Particle Bombardment, Topics Appl. Phys. 110 (2007)",
        "[Yamamura and Tawara, At. Data Nucl. Data Tables 62, 149 (1996)](https://doi.org/10.1006/adnd.1996.0005)",
    ),
    symbolic=_bohdansky_rules,
)
@dataclass(frozen=True)
class BohdanskyYield:
    """The sputter yield of Bohdansky's formula, a function of the ion energy (eV).

    ``Y = Q s_n(eps) (1 - (E_th / E)^(2/3)) (1 - E_th / E)^2`` above the threshold, zero
    below, with the Kr–C nuclear stopping `krc_nuclear_stopping` of the reduced energy
    `reduced_energy`.

    Attributes:
        projectile_z: The atomic number of the ion.
        projectile_mass_u: The ion mass in u.
        target_z: The atomic number of the target.
        target_mass_u: The target mass in u.
        surface_binding_energy: ``U_s`` in eV (the sublimation energy).
        q: The fitted factor ``Q`` (atoms per ion).
        threshold: ``E_th`` in eV; `bohdansky_threshold` of the masses and ``U_s`` if
            ``None``.
    """

    projectile_z: int
    projectile_mass_u: float
    target_z: int
    target_mass_u: float
    surface_binding_energy: float
    q: float
    threshold: float | None = None

    def __post_init__(self) -> None:
        """Validate the parameters and fill in the threshold."""
        if min(self.projectile_mass_u, self.target_mass_u) <= 0.0:
            raise ValueError("the masses must be > 0")
        if self.surface_binding_energy <= 0.0 or self.q < 0.0:
            raise ValueError("a sputter yield needs U_s > 0 and Q >= 0")
        if self.threshold is None:
            object.__setattr__(
                self,
                "threshold",
                bohdansky_threshold(
                    self.projectile_mass_u,
                    self.target_mass_u,
                    self.surface_binding_energy,
                ),
            )

    @classmethod
    def anchored(
        cls,
        projectile_z: int,
        projectile_mass_u: float,
        target_z: int,
        target_mass_u: float,
        surface_binding_energy: float,
        energy: float,
        value: float,
    ) -> BohdanskyYield:
        """The yield whose ``Q`` gives ``value`` atoms per ion at ``energy`` (eV)."""
        unit = cls(
            projectile_z,
            projectile_mass_u,
            target_z,
            target_mass_u,
            surface_binding_energy,
            1.0,
        )
        return cls(
            projectile_z,
            projectile_mass_u,
            target_z,
            target_mass_u,
            surface_binding_energy,
            value / float(unit(energy)),
        )

    def __call__(self, energy: np.ndarray | float) -> np.ndarray | float:
        """The yield at the ion energy ``energy`` (eV)."""
        e = np.asarray(energy, dtype=float)
        threshold = float(self.threshold or 0.0)
        safe = np.maximum(e, threshold, out=None) + 1e-300
        ratio = threshold / safe
        eps = reduced_energy(
            safe,
            self.projectile_z,
            self.projectile_mass_u,
            self.target_z,
            self.target_mass_u,
        )
        value = (
            self.q
            * krc_nuclear_stopping(eps)
            * (1.0 - ratio ** (2.0 / 3.0))
            * (1.0 - ratio) ** 2
        )
        out = np.where(e > threshold, value, 0.0)
        return out if out.ndim else float(out)


def _emission_rules() -> dict:
    """The SymPy rule of `potential_emission_yield` (for the tests)."""
    import sympy

    i, w = sympy.symbols("ionization_energy work_function", positive=True)
    rule = sympy.Float("0.032") * (sympy.Float("0.78") * i - 2 * w)
    return {
        "gamma": Rule(
            rule,
            ranges={"ionization_energy": (15.0, 25.0), "work_function": (3.0, 5.0)},
        )
    }


@method(
    equations={
        "gamma": r"\gamma = 0.032\,(0.78\, E_\mathrm{iz} - 2W) \quad (0.78\,E_\mathrm{iz} > 2W), \qquad 0 \text{ otherwise}",
    },
    assumptions=(
        r"potential emission only (no kinetic emission: ion energies below about 1 keV); clean metal surfaces",
        r"$E_\mathrm{iz}$ the energy the neutralization releases: the first ionization energy for a singly charged ion, the second for a doubly charged one",
    ),
    references=(
        "[Baragiola et al., Surf. Sci. 90, 240 (1979)](https://doi.org/10.1016/0039-6028(79)90341-8)",
        "[Anders, Appl. Phys. Lett. 92, 201501 (2008)](https://doi.org/10.1063/1.2936307)",
    ),
    symbolic=_emission_rules,
)
def potential_emission_yield(
    ionization_energy: np.ndarray | float, work_function: np.ndarray | float
) -> np.ndarray | float:
    """Secondary electrons per ion by potential emission (Baragiola's empirical formula).

    ``gamma = 0.032 (0.78 E_iz - 2 W)`` where positive, else zero; ``E_iz`` (eV) the
    ionization energy of the incoming ion's last charge state, ``W`` (eV) the work function
    of the target. For a singly charged metal ion on its own target ``0.78 E_iz < 2 W``: no
    emission; for argon or a doubly charged metal ion, a few tenths at most.
    """
    gamma = 0.032 * (
        0.78 * np.asarray(ionization_energy, dtype=float)
        - 2.0 * np.asarray(work_function, dtype=float)
    )
    out = np.maximum(gamma, 0.0)
    return out if np.ndim(out) else float(out)
