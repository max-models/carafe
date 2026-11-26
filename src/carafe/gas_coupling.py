"""The drift fluid and the gas dynamics together: depletion and refill from a gas model.

`DriftFluid2D` carries the neutral gas as a density ``N`` that ionization depletes and that
relaxes to its fill value at a fixed ``refill_rate``: a placeholder for the gas flow and the
sputter wind. `GasCoupledDriftFluid` replaces that relaxation by `GasDynamics`, the
compressible argon (and metal) gas in front of the target, by operator splitting on windows
of length ``window``:

1. the plasma advances over the window with the gas density of the gas model, depleting it by
   its own ionization (``depletion``) but with no refill;
2. the gas model advances over the same window with the plasma's ionization averaged over it
   as a sink of argon (atoms removed per m^3 and s), the ions that reach the target returned
   as argon atoms from the target wall (``gas_return_flux``), and, if ``ion_exchange`` is
   set, the friction of the ions on the gas (`IonExchange` with the frequency
   ``ion_friction_rate * n_i`` and the window-averaged ion velocity);
3. the plasma's ``N`` is set to the argon number density of the gas model.

The gas model's axis is the plasma's ``x`` (the target, i.e. the cathode, at ``x = 0``), with the
same length and number of cells; its ``y`` has either the plasma's cells or one cell (then the
plasma's sources are averaged along ``y`` and its ``N`` is uniform along ``y``). The gas model's
boundaries are the user's: a wall at the target (the left end) is needed for the return of
the ions, a reservoir (`Boundary.inflow`) at the right end is the refill.

The splitting is first order in ``window``; the gas time (the depletion time ``1 / (k_iz n_e)``
and the refill through the gap, tens of microseconds in a DC magnetron, microseconds at HiPIMS
density) has to be long against it. The ionization, the ion flux to the target and the ion
velocity are sampled at the end of every plasma step and weighted with its length.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from methodcard import method
from plasmacoll.constants import BOLTZMANN, ELEMENTARY_CHARGE
from scope_profiler import profile

from carafe._profiling import _PROFILE_TAGS
from carafe.driftfluid2d import DriftFluid2D, DriftFluid2DState
from carafe.gasdynamics import GasDynamics, GasState, IonExchange

__all__ = ["CoupledRecord", "GasCoupledDriftFluid"]

_EV_IN_K = ELEMENTARY_CHARGE / BOLTZMANN


@dataclass
class CoupledRecord:
    """What a coupled run records once per window.

    Attributes:
        time: The time at the end of each window (s).
        anode_current: The electron current density into the anode, mean over ``y`` (A/m^2).
        ionized: The argon atoms the plasma ionized in each window, per m^2 of target
            (integrated along ``x``, averaged along ``y``).
        returned: The argon atoms the target returned in each window, per m^2 of target.
        gas_minimum: The smallest argon density over the domain (m^-3).
        gas_mean: The mean argon density (m^-3).
    """

    time: list[float] = field(default_factory=list)
    anode_current: list[float] = field(default_factory=list)
    ionized: list[float] = field(default_factory=list)
    returned: list[float] = field(default_factory=list)
    gas_minimum: list[float] = field(default_factory=list)
    gas_mean: list[float] = field(default_factory=list)

    def as_arrays(self) -> dict[str, np.ndarray]:
        """The records as arrays."""
        return {key: np.asarray(value) for key, value in vars(self).items()}


@dataclass
class GasCoupledDriftFluid:
    """A `DriftFluid2D` whose neutral gas is a `GasDynamics` model; see the module docstring.

    Args:
        plasma: The drift-fluid model. Its ``refill_rate`` is set to 0: the gas model refills.
        gas: The gas model, of the plasma's length and ``nx``, with ``cells_y`` 1 or the
            plasma's ``ny``, its state set (e.g. `GasDynamics.set_uniform`).
        window: The splitting interval (s).
        return_ions: Return the ions that reach the target as argon atoms from the left wall.
        ion_exchange: Add the friction of the ions on the gas.
        cfl: The Courant number of the plasma steps.
    """

    plasma: DriftFluid2D
    gas: GasDynamics
    window: float
    return_ions: bool = True
    ion_exchange: bool = False
    cfl: float = 0.4
    record: CoupledRecord = field(default_factory=CoupledRecord)

    def __post_init__(self) -> None:
        """Check that the grids match and hand the refill to the gas model."""
        plasma, gas = self.plasma, self.gas
        if gas.nx != plasma.nx or not np.isclose(gas.length, plasma.length_x):
            raise ValueError("the gas model needs the plasma's length_x and nx")
        if gas.ny not in (1, plasma.ny):
            raise ValueError("the gas model has 1 cell along y or the plasma's ny")
        if gas.ny > 1 and not np.isclose(gas.width, plasma.length_y):
            raise ValueError("a 2D gas model needs the plasma's length_y as its width")
        if self.window <= 0.0:
            raise ValueError("window > 0")
        if self.return_ions and gas.boundaries["left"].kind != "wall":
            raise ValueError("returning the ions needs a wall at the target (left)")
        gas._check_ready()
        plasma.refill_rate = 0.0

    # -- the exchange between the grids ------------------------------------------------

    def _to_gas(
        self, value: np.ndarray, weight: np.ndarray | None = None
    ) -> np.ndarray:
        """A plasma field ``(nx, ny)`` on the gas grid (a ``weight``-mean along ``y`` for 1 cell)."""
        if self.gas.ny > 1:
            return value
        if weight is None:
            return value.mean(axis=1, keepdims=True)
        total = weight.sum(axis=1, keepdims=True)
        return np.where(
            total > 0.0,
            (value * weight).sum(axis=1, keepdims=True) / np.where(total > 0, total, 1),
            0.0,
        )

    def gas_number(self) -> np.ndarray:
        """The argon number density of the gas model on the plasma's grid (m^-3)."""
        number = self.gas.state().gas_number
        return np.broadcast_to(number, (self.plasma.nx, self.plasma.ny)).copy()

    def with_gas(self, state: DriftFluid2DState) -> DriftFluid2DState:
        """``state`` with its gas density replaced by the gas model's."""
        out = state.copy()
        out.gas_density = self.gas_number()
        return out

    def ionization(self, state: DriftFluid2DState) -> np.ndarray:
        """The ionization rate ``k_iz(T_e) N n_e`` of ``state`` (m^-3 s^-1)."""
        return (
            self.plasma.ionization_rate(state.electron_temperature)
            * state.gas_density
            * state.electron_density
        )

    @staticmethod
    def target_ion_flux(state: DriftFluid2DState) -> np.ndarray:
        """The ion flux into the target along ``y``, ``n_i max(-v_x, 0)`` of the first cell."""
        return state.ion_density[0] * np.maximum(-state.ion_velocity[0, 0], 0.0)

    # -- time stepping -----------------------------------------------------------------

    @profile("carafe.GasCoupledDriftFluid.advance", tags=_PROFILE_TAGS)
    @method(
        equations={
            "sink": r"S_g = \frac{1}{\Delta}\int_{t}^{t + \Delta} k_\mathrm{iz}(T_e)\,N n_e\,dt",
            "return": r"\Phi_\mathrm{return} = \frac{1}{\Delta}\int_{t}^{t + \Delta} n_i \max(-v_{i,x}, 0)\big|_{x_0}\,dt",
            "exchange": r"\nu_{gi} = k_\mathrm{fr}\,\bar n_i, \qquad \bar{\mathbf u}_i = \frac{\int n_i\mathbf v_i\,dt}{\int n_i\,dt}",
            "gas": r"N(t + \Delta) = \frac{\rho_g(t + \Delta)}{m_g}",
        },
        order=1,
        assumptions=(
            r"Lie splitting on windows $\Delta$: the plasma with the gas frozen except its own depletion, then the gas with the plasma's window averages",
            r"the gas time ($1/(k_\mathrm{iz} n_e)$, the refill through the gap) long against $\Delta$; sources sampled at the end of each plasma step",
            r"the gas grid is the plasma's along $x$, with one cell or the plasma's cells along $y$ (one cell: $y$-averaged sources, ion velocity weighted by $n_i$)",
        ),
    )
    def advance(self, state: DriftFluid2DState) -> DriftFluid2DState:
        """One window: the plasma, then the gas with the plasma's averages, then ``N``."""
        plasma, gas = self.plasma, self.gas
        state = self.with_gas(state)
        t_stop = state.time + self.window
        ionized = np.zeros_like(state.electron_density)
        target = np.zeros(plasma.ny)
        momentum = np.zeros_like(state.ion_velocity)
        density = np.zeros_like(state.ion_density)
        steps = 0
        step = plasma.suggest_timestep(state, self.cfl)
        while state.time < t_stop * (1.0 - 1e-14):
            if steps % 20 == 0:
                step = plasma.suggest_timestep(state, self.cfl)
            h = min(step, t_stop - state.time)
            state = plasma.step(state, h)
            steps += 1
            if not np.all(np.isfinite(state.potential)):
                raise FloatingPointError(
                    f"the plasma diverged at t = {state.time:.3e} s"
                )
            ionized += h * self.ionization(state)
            target += h * self.target_ion_flux(state)
            momentum += h * state.ion_density * state.ion_velocity
            density += h * state.ion_density
        rate = ionized / self.window
        gas.sinks["gas"] = self._to_gas(rate)
        if self.return_ions:
            flux = target / self.window
            gas.boundaries["left"].gas_return_flux = (
                flux if gas.ny > 1 else float(flux.mean())
            )
        if self.ion_exchange:
            mean_n = density / self.window
            safe = np.where(density > 0.0, density, 1.0)
            velocity = np.where(density > 0.0, momentum / safe, 0.0)
            gas.exchange = [
                IonExchange(
                    "gas",
                    self._to_gas(plasma.ion_friction_rate * mean_n),
                    (
                        self._to_gas(velocity[0], mean_n),
                        self._to_gas(velocity[1], mean_n),
                    ),
                    plasma.ion_temperature * _EV_IN_K,
                )
            ]
        gas.run(gas.time + self.window)
        state = self.with_gas(state)
        gx, _ = plasma.electron_flux(state)
        number = self.gas.state().gas_number
        area = plasma.dx * plasma.dy
        self.record.time.append(state.time)
        self.record.anode_current.append(float(ELEMENTARY_CHARGE * gx[-1].mean()))
        self.record.ionized.append(float(ionized.sum() * area / plasma.length_y))
        self.record.returned.append(
            float(target.sum() * plasma.dy / plasma.length_y)
            if self.return_ions
            else 0.0
        )
        self.record.gas_minimum.append(float(number.min()))
        self.record.gas_mean.append(float(number.mean()))
        return state

    @profile("carafe.GasCoupledDriftFluid.run", tags=_PROFILE_TAGS)
    def run(
        self, state: DriftFluid2DState, duration: float
    ) -> tuple[DriftFluid2DState, GasState]:
        """Advance by ``duration`` (a whole number of windows, rounded up); the final states."""
        windows = max(1, int(np.ceil(duration / self.window - 1e-9)))
        for _ in range(windows):
            state = self.advance(state)
        return state, self.gas.state()
