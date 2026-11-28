"""The ionization region model: geometry, conservation, limits, current and the fit."""

from __future__ import annotations

import dataclasses
from collections import Counter

import numpy as np
import pytest
from plasmacoll.constants import ATOMIC_MASS, BOLTZMANN, ELEMENTARY_CHARGE

from carafe.irm import (
    COPPER_TARGET,
    IonizationRegion,
    IRMRates,
    IRMTarget,
    waveform,
)
from carafe.magnetron import COPPER, lotz_ionization
from carafe.sputtering import ConstantYield, PowerLawYield

E = ELEMENTARY_CHARGE
pytestmark = pytest.mark.filterwarnings("ignore:.*placeholders, not data")

ON = 50.0e-6
TIMES = np.linspace(0.0, 80.0e-6, 17)
ARGON = ("Ar", "Ar_w", "Ar_h", "Ar_m", "Ar+", "Ar2+")


def current(t: float) -> float:
    """A HiPIMS-like current: a 25 us ramp to 20 A, held to the end of the pulse."""
    return 20.0 * min(t / 25.0e-6, 1.0) if t < ON else 0.0


def voltage(t: float) -> float:
    """500 V during the pulse."""
    return 500.0 if t < ON else 0.0


@pytest.fixture(scope="module")
def region() -> IonizationRegion:
    return IonizationRegion(back_attraction=0.5, voltage_fraction=0.15)


@pytest.fixture(scope="module")
def pulse(region: IonizationRegion):
    return region.evolve(TIMES, current, voltage, breakpoints=[ON])


def states(region: IonizationRegion, data):
    """The states of a run, as densities by name and ``Te``."""
    for k in range(len(data.t)):
        state = {
            str(s): float(data.density.isel(t=k).sel(species=s))
            for s in data.species.values
            if s != "e"
        }
        state["Te"] = float(data.Te.isel(t=k))
        yield float(data.t[k]), state


def mean_speed(temperature_ev: float, mass_u: float) -> float:
    return np.sqrt(8.0 * E * temperature_ev / (np.pi * mass_u * ATOMIC_MASS))


def test_the_target_takes_the_yields_and_the_second_ionization_of_the_material() -> (
    None
):
    target = IRMTarget()
    assert target.second_ionization == pytest.approx(20.29)
    assert target.sputter_yield("Ar", 500.0) == pytest.approx(2.0)
    assert target.sputter_yield("M", 500.0) == pytest.approx(2.2)
    custom = IRMTarget(argon_yield=PowerLawYield(0.07, 0.55))
    assert custom.sputter_yield("Ar", 400.0) == pytest.approx(0.07 * 400.0**0.55)
    with pytest.raises(ValueError, match="second ionization"):
        IRMTarget(second_ionization_energy=5.0)
    bare = dataclasses.replace(COPPER, second_ionization_energy=None)
    with pytest.raises(ValueError, match="no second ionization"):
        IRMTarget(material=bare)


def test_a_waveform_interpolates_the_samples() -> None:
    sampled = waveform([0.0, 1.0e-6, 2.0e-6], [0.0, 10.0, 20.0])
    assert sampled(0.5e-6) == pytest.approx(5.0)
    assert sampled(3.0e-6) == 0.0
    with pytest.raises(ValueError, match="increasing"):
        waveform([0.0, 0.0], [1.0, 2.0])


def test_the_target_and_the_region_are_validated() -> None:
    with pytest.raises(ValueError, match="secondary"):
        IRMTarget(secondary={"Ar+": 0.1})
    with pytest.raises(ValueError, match="second ionization"):
        IRMTarget(second_ionization_energy=5.0)
    with pytest.warns(UserWarning, match="placeholders"):
        IonizationRegion()
    for bad in (
        {"back_attraction": 1.0},
        {"voltage_fraction": 0.0},
        {"inner_radius": 0.03},
        {"warm_return": 2.0},
        {"hot_escape_time": -1.0},
    ):
        with pytest.raises(ValueError):
            IonizationRegion(**bad)


def test_the_geometry_is_the_annulus_over_the_racetrack() -> None:
    region = IonizationRegion(
        inner_radius="10 mm", outer_radius="20 mm", sheath_height=0.0, height="1 cm"
    )
    area = np.pi * (0.02**2 - 0.01**2)
    assert region.target_area == pytest.approx(area)
    assert region.volume == pytest.approx(area * 0.01)
    assert region.bulk_area == pytest.approx(area + 2.0 * np.pi * 0.03 * 0.01)
    assert region.gas_density == pytest.approx(1.0 / (BOLTZMANN * 300.0))


def test_the_reactions_conserve_charge_and_nuclei(region: IonizationRegion) -> None:
    charge = {name: s.charge for name, s in region.species.items()}
    argon = {name: 1 for name in ARGON}
    metal = {"Cu": 1, "Cu+": 1, "Cu2+": 1}
    for reaction in region.reactions():
        for weights in (charge, argon, metal):
            before = sum(weights.get(s, 0) for s in reaction.reactants)
            after = sum(weights.get(s, 0) for s in reaction.products)
            assert before == after, reaction.name
        # the hot electrons are never made or lost in the volume
        counts = Counter(reaction.products)
        counts.subtract(Counter(reaction.reactants))
        assert counts["e_h"] == 0


def test_without_a_discharge_the_gas_stays_at_the_filling_density() -> None:
    region = IonizationRegion()
    data = region.evolve(TIMES, 0.0, 0.0, region.initial_state(1.0e10))
    assert data.rarefaction.values == pytest.approx(1.0, abs=1.0e-6)
    assert float(data.density.sel(species="e")[-1]) < 1.0e10


def test_the_states_stay_positive(pulse) -> None:
    density = pulse.density.values
    assert np.all(np.isfinite(density)) and np.all(density > 0.0)
    assert np.all(pulse.Te.values > 0.0)
    for key in ("alpha_t", "ionized_flux_fraction", "self_sputtering_share"):
        values = pulse[key].values
        assert np.all((values >= 0.0) & (values <= 1.0)), key
    # a pulse of 20 A makes a dense, rarefied plasma with metal ions
    peak = pulse.sel(t=0.9 * ON, method="nearest")
    assert float(peak.density.sel(species="e")) > 1.0e18
    assert float(peak.rarefaction) < 0.9
    assert float(peak.density.sel(species="Cu+")) > 0.0
    assert float(peak.alpha_t) > 0.0
    assert pulse.ohmic.values == pytest.approx(
        pulse.current.values * 0.15 * pulse.voltage.values
    )


def test_argon_is_conserved_up_to_the_flows_across_the_boundary(
    region: IonizationRegion, pulse
) -> None:
    v = region.volume
    area = region.bulk_area
    temperature = {
        "Ar": BOLTZMANN * 300.0 / E,
        "Ar_w": region.warm_temperature,
        "Ar_h": region.hot_argon_temperature,
        "Ar_m": BOLTZMANN * 300.0 / E,
    }
    for t, state in states(region, pulse):
        terms = region.diagnostics(state, t, current, voltage)
        n = terms["densities"]
        total = v * sum(terms["change"][s] for s in ARGON)  # 1/s
        inflow = 0.25 * mean_speed(temperature["Ar"], 39.948) * region.gas_density
        out = sum(0.25 * mean_speed(temperature[s], 39.948) * n[s] for s in temperature)
        bohm = {
            s: np.sqrt(E * terms["Te"] / region.species[s].mass)
            for s in ("Ar+", "Ar2+")
        }
        ions_out = sum(n[s] * bohm[s] for s in bohm)
        # the ions that leave for the bulk come back as warm argon (warm_return = 1)
        expected = (
            area * (inflow - out - ions_out) + region.warm_return * area * ions_out
        )
        scale = area * inflow
        assert total == pytest.approx(expected, abs=1.0e-9 * scale)


def test_the_metal_is_conserved_up_to_sputtering_and_the_losses(
    region: IonizationRegion, pulse
) -> None:
    v = region.volume
    for t, state in states(region, pulse):
        terms = region.diagnostics(state, t, current, voltage)
        metal = ("Cu", "Cu+", "Cu2+")
        total = v * sum(terms["change"][s] for s in metal)
        out = terms["neutral_out"]["Cu"] + sum(
            terms["to_bulk"][s] + terms["to_target"][s] for s in metal[1:]
        )
        expected = terms["sputtered"] - out
        scale = max(terms["sputtered"], out, 1.0)
        assert total == pytest.approx(expected, abs=1.0e-9 * scale)


@pytest.mark.parametrize("escape", [None, 1.0e-7])
def test_the_discharge_current_is_continuous(escape) -> None:
    region = IonizationRegion(
        back_attraction=0.5, voltage_fraction=0.15, hot_escape_time=escape
    )
    data = region.evolve(TIMES, current, voltage, breakpoints=[ON])
    for t, state in states(region, data):
        terms = region.diagnostics(state, t, current, voltage)
        species = region.species
        ions = ("Ar+", "Ar2+", "Cu+", "Cu2+")
        at_target = E * (
            sum(species[s].charge * terms["to_target"][s] for s in ions)
            + terms["secondary"]
        )
        into_bulk = E * (
            terms["electrons_out"]
            + terms["hot_escaped"]
            - sum(species[s].charge * terms["to_bulk"][s] for s in ions)
        )
        assert terms["model_current"] == pytest.approx(at_target, rel=1.0e-12)
        assert into_bulk == pytest.approx(at_target, rel=1.0e-8, abs=1.0e-12)


def test_without_sputtering_the_metal_plays_no_part() -> None:
    def run(factor: float):
        target = dataclasses.replace(
            COPPER_TARGET,
            argon_yield=ConstantYield(0.0),
            metal_yield=ConstantYield(0.0),
        )
        rate = lotz_ionization(7.726, 1)
        rates = IRMRates(metal_ionization=lambda te: factor * rate(te))
        region = IonizationRegion(target=target, rates=rates, back_attraction=0.5)
        return region.evolve(TIMES, current, voltage, breakpoints=[ON])

    base, faster = run(1.0), run(10.0)
    assert np.all(base.sputtered.values == 0.0)
    assert np.all(base.alpha_t.values == 0.0)
    metal = base.density.sel(species=["Cu", "Cu+", "Cu2+"]).sum("species")
    assert np.all(metal.values < 1.0e-6 * base.density.sel(species="e").values.max())
    for name in ("Ar", "Ar+", "Ar_m", "e"):
        np.testing.assert_allclose(
            base.density.sel(species=name), faster.density.sel(species=name), rtol=1e-4
        )
    np.testing.assert_allclose(base.Te, faster.Te, rtol=1e-4)


def test_the_fit_recovers_the_parameters_of_synthetic_data() -> None:
    # a discharge driven by the voltage alone: its current is a waveform that the
    # region with these parameters reproduces exactly when driven by it
    true = IonizationRegion(back_attraction=0.5, voltage_fraction=0.18)
    on = 10.0e-6
    times = np.linspace(0.0, 20.0e-6, 81)
    initial = true.initial_state(1.0e18)
    synthetic = true.evolve(
        times, None, lambda t: 400.0 if t < on else 0.0, initial, breakpoints=[on]
    )
    assert float(synthetic.model_current.max()) > 10.0  # a nonlinear pulse
    assert float(synthetic.rarefaction.min()) < 0.8
    fit = IonizationRegion().fit(
        times,
        synthetic.model_current.values,
        synthetic.voltage.values,
        initial,
        guess=(0.55, 0.16),
        breakpoints=[on],
        tolerance=1.0e-4,
    )
    assert fit.success
    assert fit.back_attraction == pytest.approx(0.5, abs=5.0e-3)
    assert fit.voltage_fraction == pytest.approx(0.18, abs=2.0e-3)
    assert fit.relative_rms < 2.0e-3
    assert fit.region.back_attraction == fit.back_attraction
