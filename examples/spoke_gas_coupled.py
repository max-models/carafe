"""Gas-coupled drift fluid.

The 1D state of Boeuf's DC magnetron (argon at 0.6 Pa, 2 cm gap, 260 V) relaxed with its neutral
gas from the gas dynamics (`GasCoupledDriftFluid`): the plasma ionizes the argon, the ions that
reach the target come back as argon atoms, and a reservoir at the anode end refills the gap
through the compressible gas. Compared with the drift fluid's own closure, a relaxation of the
gas towards its fill density at the fixed rate 2e4 /s.

The first figure is the argon density across the gap at the end of the run with each closure;
the second the anode current against time. At this DC density the depletion is small (the
depletion time is tens of microseconds, the run a few), so the two agree: the coupling matters at
HiPIMS density and over the gas time of about 100 us, which this short run does not reach.
"""

from __future__ import annotations

import numpy as np
import plasma_plots
import xarray as xr

from wellbalanced import DriftFluid2D, GasDynamics
from wellbalanced.driftfluid2d import exponential_field
from wellbalanced.gas_coupling import GasCoupledDriftFluid
from wellbalanced.gasdynamics import Boundary

BOLTZMANN = 1.380649e-23
GAP, PERIOD, PRESSURE, B_MAX = 0.02, 0.08, 0.6, 0.12
NX, DURATION, WINDOW = 16, 4e-6, 0.25e-6


def build(**options) -> DriftFluid2D:
    """The magnetron plane of the gallery's study 07, y-uniform (4 cells along y)."""
    return DriftFluid2D(
        length_x=GAP,
        length_y=PERIOD,
        nx=NX,
        ny=4,
        voltage=260.0,
        magnetic_field=exponential_field(B_MAX, 0.3 * GAP),
        gas_density=PRESSURE / (BOLTZMANN * 300.0),
        electron_temperature=None,
        cathode_current_density=0.03,
        **options,
    )


def coupled_run():
    """The drift fluid with the gas dynamics."""
    gas = GasDynamics(
        GAP,
        NX,
        boundaries={
            "left": Boundary.wall(300.0),
            "right": Boundary.inflow(PRESSURE, 300.0),
        },
    )
    gas.set_uniform(PRESSURE, 300.0)
    coupled = GasCoupledDriftFluid(build(), gas, window=WINDOW, ion_exchange=True)
    state = coupled.plasma.initial_state(1.0e15, electron_temperature=3.0)
    state, _ = coupled.run(state, DURATION)
    record = coupled.record.as_arrays()
    return coupled.plasma, state, record["time"], record["anode_current"]


def relaxation_run():
    """The drift fluid with its own refill closure, sampled at the same windows."""
    model = build(refill_rate=2.0e4)
    state = model.initial_state(1.0e15, electron_temperature=3.0)
    times, current = [], []
    for _ in range(int(round(DURATION / WINDOW))):
        state = model.run(state, WINDOW, record_every=WINDOW).state
        gx, _ = model.electron_flux(state)
        times.append(state.time)
        current.append(1.602176634e-19 * gx[-1].mean())
    return model, state, np.array(times), np.array(current)


runs = {"gas dynamics": coupled_run(), "relaxation": relaxation_run()}
fill = PRESSURE / (BOLTZMANN * 300.0)
argon = xr.concat(
    [
        xr.DataArray(
            state.gas_density[:, 0] / fill,
            coords={"x": model.x * 1e3},
            dims="x",
            name="argon",
        ).expand_dims(closure=[name])
        for name, (model, state, _, _) in runs.items()
    ],
    dim="closure",
)
argon.attrs["long_name"] = "argon density over the fill density"
argon.x.attrs.update(long_name="distance from the cathode", units="mm")
current = xr.concat(
    [
        xr.DataArray(j, coords={"t": t * 1e6}, dims="t", name="current").expand_dims(
            closure=[name]
        )
        for name, (_, _, t, j) in runs.items()
    ],
    dim="closure",
)
current.attrs.update(long_name="anode current density", units="A/m^2")
current.t.attrs.update(long_name="time", units="us")
for name, (_, state, _, j) in runs.items():
    print(
        f"{name:>13}: j_anode = {j[-1]:.2f} A/m2, argon minimum "
        f"{state.gas_density.min() / fill:.4f} of the fill"
    )

with plasma_plots.figure(1, 1, title="Argon across the gap") as fig:
    argon.plasma.plot.profiles(x="x", over="closure", ax=fig[0])
with plasma_plots.figure(1, 1, title="Anode current") as fig:
    current.plasma.plot.profiles(x="t", over="closure", ax=fig[0])
