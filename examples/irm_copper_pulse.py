"""Ionization region model.

The ionization region model (IRM) of a HiPIMS pulse on a copper target, fitted to a
discharge current.

The ionization region over the racetrack, driven by a discharge current ``I_D(t)`` and
voltage ``V_D(t)``: two electron groups (thermal electrons heated ohmically by the share
``f`` of the voltage over the region, and hot secondary electrons from the target), cold,
warm and hot argon, metastables, ``Ar+``, ``Ar2+``, and copper with ``Cu+`` and ``Cu2+``.
The back-attraction probability ``beta_t`` and ``f`` are fitted so that the model's own
discharge current matches the given one, which is the way the IRM is fitted to a measured
pulse. Here the "measured" pulse is synthetic (a 100 us pulse rising to 30 A at 550 V), and
every rate coefficient, yield and the geometry are placeholders (see ``wellbalanced.irm``):
the procedure is the IRM's, the numbers are not data. The figure shows the current, the
densities, the electron temperature and the outputs of the IRM: the share ``alpha_t`` of the
sputtered atoms that are ionized and return to the target, the ionized flux fraction of the
copper leaving the region and the rarefaction of the argon.
"""

import warnings

import numpy as np
import plasma_plots

from wellbalanced.irm import IonizationRegion

warnings.filterwarnings("ignore", message=".*placeholders, not data")

on = 100.0e-6
times = np.linspace(0.0, on, 101)
# a synthetic HiPIMS pulse: the current rises to 30 A, the voltage sags a little
measured_current = 30.0 * (1.0 - np.exp(-times / 30.0e-6))
measured_voltage = 550.0 - 50.0 * times / on

region = IonizationRegion(back_attraction=0.8, voltage_fraction=0.15)
fit = region.fit(
    times,
    measured_current,
    measured_voltage,
    region.initial_state(1.0e17),
    tolerance=1.0e-4,
)
data = fit.data
print(
    f"beta_t = {fit.back_attraction:.3f}, f = {fit.voltage_fraction:.3f} "
    f"(rms mismatch {fit.relative_rms:.1%} of the peak current, "
    f"{fit.evaluations} runs); at the end of the pulse: alpha_t = "
    f"{float(data.alpha_t[-1]):.2f}, ionized flux fraction "
    f"{float(data.ionized_flux_fraction[-1]):.2f}, argon at "
    f"{float(data.rarefaction[-1]):.2f} of the filling density"
)

data = data.isel(t=slice(1, None))  # the seed state at t = 0 is not a discharge
with plasma_plots.figure(2, 2, title="Cu HiPIMS pulse, IRM fitted") as fig:
    data.model_current.plasma.plot.timeseries(
        logy=False,
        reference={"given": data.current},
        title="discharge current",
        ax=fig[0],
    )
    species = ["e", "Ar", "Ar+", "Ar_m", "Cu", "Cu+", "Cu2+"]
    data.density.sel(species=species).plasma.plot.profiles(
        x="t", over="species", logy=True, ax=fig[1]
    )
    data.Te.plasma.plot.timeseries(logy=False, title="electron temperature", ax=fig[2])
    data.alpha_t.plasma.plot.timeseries(
        data.ionized_flux_fraction,
        data.rarefaction,
        logy=False,
        title="IRM outputs",
        ax=fig[3],
    )
