"""Ionization region model.

The ionization region model (IRM) of a HiPIMS pulse on a copper target, fitted to a
discharge current: case I of Gudmundsson et al., Surf. Coat. Technol. 442, 128189
(2022), a 2 inch target in argon and a 40 us pulse to 20 A.

The ionization region over the racetrack, driven by a discharge current ``I_D(t)`` and
voltage ``V_D(t)``: two electron groups (thermal electrons heated ohmically by the share
``f`` of the voltage over the region, and hot secondary electrons from the target), cold,
warm and hot argon, metastables, ``Ar+``, ``Ar2+``, and copper with ``Cu+`` and ``Cu2+``.
The back-attraction probability ``beta_t`` and ``f`` are fitted so that the model's own
discharge current matches the given one, which is the way the IRM is fitted to a measured
pulse; ``beta_t`` is zero in the afterglow.

The case, its waveforms, the geometry of the region (Table 4 of the paper), the pressure and
the published results (Table 5), is the one file ``examples/data/irm_copper_case1.csv``.
Its waveforms are placeholders until the digitized ``I_D(t)`` and ``V_D(t)`` of Fig. 2 of
the paper replace them, and every rate coefficient and yield is a placeholder too (see
``carafe.irm``): the procedure is the IRM's, the numbers are not data, and the table
printed at the end compares them with the paper only to show where the comparison goes.
The figures show the current, its carriers at the target, the densities, the electron
temperature and ``alpha_t``, ``beta_t`` and the ionized flux fraction over the time.
"""

import warnings
from pathlib import Path

import plasma_plots

from carafe.irm import load_case

warnings.filterwarnings("ignore", message=".*placeholders, not data")

case = load_case(Path(__file__).parent / "data" / "irm_copper_case1.csv")
# f between 5 and 50 %, beta_t below 0.95: the range of the IRM papers' maps
fit = case.fit(bounds=((0.0, 0.95), (0.05, 0.5)), tolerance=1.0e-3)
averages = fit.averages
print(
    f"{case.name}: beta_t = {fit.back_attraction:.3f}, f = {fit.voltage_fraction:.3f} "
    f"(rms mismatch {fit.relative_rms:.1%} of the peak current, {fit.evaluations} runs)"
)
print(f"{'':24s} {'model':>8s} {'paper':>8s}")
for key, value in averages.items():
    paper = case.reference.get(key)
    print(f"{key:24s} {value:8.3f} {'' if paper is None else f'{paper:8.3f}'}")

data = fit.data.isel(t=slice(1, None))  # the seed state at t = 0 is not a discharge
with plasma_plots.figure(5, 1, title=f"{case.name}, IRM fitted") as fig:
    data.model_current.plasma.plot.timeseries(
        logy=False,
        reference={"given": data.current},
        title="discharge current",
        ax=fig[0],
    )
    data.target_current.plasma.plot.profiles(
        x="t", over="carrier", logy=False, ax=fig[1]
    )
    species = ["e", "Ar", "Ar_h", "Ar_m", "Ar+", "Cu", "Cu+", "Cu2+"]
    data.density.sel(species=species).plasma.plot.profiles(
        x="t", over="species", logy=True, ax=fig[2]
    )
    data.Te.plasma.plot.timeseries(logy=False, title="electron temperature", ax=fig[3])
    data.alpha_t.plasma.plot.timeseries(
        data.back_attraction,
        data.ionized_flux_fraction,
        data.rarefaction,
        logy=False,
        title="IRM outputs",
        ax=fig[4],
    )
