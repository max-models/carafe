"""Wind efficiency.

The global model's sputter wind, with a wind efficiency derived from the compressible gas.

The global model (:class:`wellbalanced.Magnetron`) rarefies the argon with a sputter wind
that pushes ``eta`` argon atoms out of the ionization region per sputtered atom. With a
constant ``eta = 0.3`` the rarefaction at 100 A is far stronger than that of the argon as a
compressible gas in front of the target (``examples/sputter_wind_gas_dynamics.py``).
`wellbalanced.magnetron.wind_efficiency_from_gas_dynamics` runs the gas
(`wellbalanced.magnetron.sputter_wind_gas`: titanium sputtered cold into 2 cm of argon at 1
Pa, refilled from the far end) to its steady state at six sputtered fluxes and sets
``eta(Gamma)`` so that the global model's argon deficit is the gas's mean deficit. The first
figure is ``eta`` against the discharge current; the second the rarefaction of the global
model with the constant and with the derived ``eta``, and, as a check between the points
``eta`` was derived at, the gas run at the sputtered flux the global model finds with it.

The gas model is the continuum (Euler) limit at Knudsen numbers near 1, so a kinetic (DSMC)
check is still due; its diffusion, conduction and side-wall cooling, the yields and the
refill time are placeholders, and the derived ``eta`` holds for this refill time and region
height.
"""

import dataclasses
import warnings

import numpy as np
import plasma_plots

from wellbalanced import MATERIALS, Magnetron
from wellbalanced.magnetron import sputter_wind_gas, wind_efficiency_from_gas_dynamics

warnings.filterwarnings("ignore", message=".*placeholders, not data")

CURRENTS = (1.0, 3.0, 10.0, 30.0, 100.0)


def voltage(current: float) -> float:
    """A placeholder current-voltage characteristic, 400 V at 0.1 A rising slowly."""
    return 400.0 + 60.0 * np.log10(1.0 + current)


constant = Magnetron(material=MATERIALS["Ti"], wind_efficiency=0.3)
constant = constant.calibrate_heating(2.0, voltage(2.0))
efficiency = wind_efficiency_from_gas_dynamics(constant)
derived = dataclasses.replace(constant, wind_efficiency=efficiency)

print("wind efficiency from the gas dynamics (cold sputtered atoms)")
print("  flux [1/m^2/s]   mean argon deficit   eta")
for flux, deficit, eta in zip(
    efficiency.flux, efficiency.deficit, efficiency.efficiency, strict=True
):
    print(f"  {flux:12.2e}   {deficit:18.3f}   {eta:5.3f}")

rows = []
for current in CURRENTS:
    row = {"current": current}
    for label, magnetron in (("constant", constant), ("derived", derived)):
        state = magnetron.steady_state(current, voltage(current))
        diagnostics = magnetron.diagnostics(state)
        row[f"{label}_rarefaction"] = diagnostics["rarefaction"]
        row[f"{label}_eta"] = diagnostics["wind_efficiency"]
        row[f"{label}_flux"] = (
            magnetron.target_currents(state)["sputtered"] / magnetron.target_area
        )
    gas = sputter_wind_gas(derived, row["derived_flux"]).steady_state(
        tolerance=1.0e-4, max_time=1.0e-2
    )
    row["gas_rarefaction"] = float(np.mean(gas.gas_number)) / derived.gas_density
    rows.append(row)

print("argon density over the filling density against the discharge current")
print("  I [A]   eta    global, eta = 0.3   global, derived eta   gas dynamics")
for row in rows:
    print(
        f"  {row['current']:5.0f}   {row['derived_eta']:5.3f}"
        f"   {row['constant_rarefaction']:17.3f}   {row['derived_rarefaction']:19.3f}"
        f"   {row['gas_rarefaction']:12.3f}"
    )


with plasma_plots.figure(
    1, 2, title="Sputter wind of titanium into argon at 1 Pa, cold atoms"
) as fig:
    fig[0].plot(
        CURRENTS, [r["constant_eta"] for r in rows], marker="s", label="constant"
    )
    fig[0].plot(CURRENTS, [r["derived_eta"] for r in rows], marker="o", label="derived")
    fig[0].set_xscale("log")
    fig[0].set_xlabel("discharge current [A]")
    fig[0].set_ylabel("wind efficiency")
    fig[0].set_title("argon atoms pushed out per sputtered atom")
    fig[0].legend()
    for key, label, marker in (
        ("constant_rarefaction", "global model, eta = 0.3", "s"),
        ("derived_rarefaction", "global model, derived eta", "o"),
        ("gas_rarefaction", "gas dynamics", "^"),
    ):
        fig[1].plot(CURRENTS, [r[key] for r in rows], marker=marker, label=label)
    fig[1].set_xscale("log")
    fig[1].set_xlabel("discharge current [A]")
    fig[1].set_ylabel("argon density over the filling density")
    fig[1].set_title("rarefaction")
    fig[1].legend()
