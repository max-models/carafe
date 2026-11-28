"""The ionization region model: geometry, conservation, limits, current and the fit."""

from __future__ import annotations

import dataclasses
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from plasmacoll.constants import ATOMIC_MASS, BOLTZMANN, ELEMENTARY_CHARGE

from carafe.irm import (
    COPPER_TARGET,
    IonizationRegion,
    IRMRates,
    IRMTarget,
    SputterYield,
    load_case,
    pulse_averages,
    waveform,
)
from carafe.magnetron import lotz_ionization

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


def test_the_sputter_yield_is_a_power_law() -> None:
    fit = SputterYield(0.07, 0.55)
    assert fit(400.0) == pytest.approx(0.07 * 400.0**0.55)
    assert fit(0.0) == 0.0 and fit(-5.0) == 0.0
    with pytest.raises(ValueError, match="sputter yield"):
        SputterYield(0.1, 0.0)


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
        {"pulse_end": -1.0},
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
            argon_yield=SputterYield(0.0, 0.5),
            metal_yield=SputterYield(0.0, 0.5),
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


@pytest.fixture(scope="module")
def afterglow() -> tuple[IonizationRegion, object]:
    """A pulse that ends at ``ON``, with no back-attraction in the afterglow."""
    region = IonizationRegion(back_attraction=0.5, voltage_fraction=0.15, pulse_end=ON)
    times = np.linspace(0.0, 120.0e-6, 61)
    return region, region.evolve(times, current, voltage)


def test_the_current_at_the_target_is_its_carriers(pulse) -> None:
    assert list(pulse.carrier.values) == ["Ar+", "Ar2+", "Cu+", "Cu2+", "SE"]
    np.testing.assert_allclose(
        pulse.target_current.sum("carrier"), pulse.model_current, rtol=1.0e-12
    )
    np.testing.assert_allclose(
        pulse.target_current.isel(carrier=slice(0, 4)).values,
        E * pulse.ion_flux_target.values * np.array([1, 2, 1, 2]),
        rtol=1.0e-12,
    )
    # argon ions carry most of the current, with secondary electrons
    peak = pulse.target_current.sel(t=0.9 * ON, method="nearest")
    assert float(peak.sel(carrier="Ar+")) > float(peak.sel(carrier="Ar2+"))
    assert float(peak.sel(carrier="SE")) > 0.0


def test_after_the_pulse_no_ion_returns_to_the_target(afterglow) -> None:
    region, data = afterglow
    assert region.back_attraction_at(0.5 * ON) == 0.5
    assert region.back_attraction_at(2.0 * ON) == 0.0
    after = data.where(data.t > ON + 1.0e-9, drop=True)
    assert np.all(after.back_attraction.values == 0.0)
    assert np.all(after.ion_flux_target.values == 0.0)
    assert np.all(after.model_current.values == 0.0)
    # the metal keeps leaving for the bulk until the region is empty
    assert float(after.ion_flux_bulk.sel(ion="Cu+")[0]) > 0.0
    metal = data.density.sel(species="Cu")
    assert float(metal[-1]) < 1.0e-2 * float(metal.max())


def test_the_electron_power_balances(region: IonizationRegion) -> None:
    # start at 5 us, past the seed state
    start = region.evolve([0.0, 5.0e-6], current, voltage)
    *_, (_, initial) = states(region, start)
    times = np.linspace(5.0e-6, 45.0e-6, 401)
    data = region.evolve(times, current, voltage, initial, rtol=1.0e-9)
    power = data.electron_power
    np.testing.assert_allclose(
        power.sel(term="ohmic"),
        power.sel(term=["collisions", "escape", "stored"]).sum("term"),
        rtol=1.0e-9,
    )
    # the stored power is d(3/2 n_e T_e e V)/dt
    energy = 1.5 * E * region.volume * data.density.sel(species="e") * data.Te
    derivative = np.gradient(energy.values, times)
    inner = slice(10, -10)
    scale = float(power.sel(term="ohmic").max())
    np.testing.assert_allclose(
        derivative[inner],
        power.sel(term="stored").values[inner],
        atol=1.0e-2 * scale,
    )
    assert np.all(power.sel(term=["ohmic", "escape"]).values >= 0.0)
    assert np.all(data.hot_power.values >= 0.0)


def test_the_pulse_averages_are_those_of_the_irm_papers(afterglow) -> None:
    region, data = afterglow
    averages = pulse_averages(data)
    for key in ("alpha_t", "beta_t", "alpha", "ionized_flux_fraction"):
        assert 0.0 < averages[key] < 1.0, key
    assert averages["alpha_t"] == pytest.approx(averages["alpha"] * averages["beta_t"])
    # the afterglow, with beta_t = 0, lowers the average below the pulse's value
    assert averages["beta_t"] < averages["back_attraction"] == 0.5
    assert averages["voltage_fraction"] == 0.15
    assert averages["rarefaction_min"] == pytest.approx(float(data.rarefaction.min()))
    assert averages["argon_rarefaction_min"] >= averages["rarefaction_min"]
    # alpha = alpha_t / beta_t at every instant of the pulse
    pulse = data.where((data.t > 0.0) & (data.t < ON), drop=True)
    np.testing.assert_allclose(
        pulse.alpha_t, pulse.alpha * pulse.back_attraction, rtol=1.0e-10
    )


def test_without_sputtering_the_averages_are_zero() -> None:
    target = dataclasses.replace(
        COPPER_TARGET,
        argon_yield=SputterYield(0.0, 0.5),
        metal_yield=SputterYield(0.0, 0.5),
    )
    region = IonizationRegion(target=target, back_attraction=0.5, pulse_end=ON)
    data = region.evolve(TIMES, current, voltage)
    averages = pulse_averages(data)
    assert averages["alpha_t"] == averages["alpha"] == 0.0
    assert np.all(data.alpha.values == 0.0)


CASE = """# A test pulse.
# name = test case
# source = synthetic
# placeholder = {placeholder}
# inner_radius = 8 mm
# pressure = 1 Pa
# pulse_end = 10 us
# back_attraction = 0.5
# voltage_fraction = 0.18
# reference.alpha_t = 0.5
time_us,current_A,voltage_V
"""


def write_case(path: Path, times, current, voltage, placeholder: str = "false"):
    rows = "".join(
        f"{1.0e6 * t:.6g},{i:.9g},{v:.9g}\n"
        for t, i, v in zip(times, current, voltage, strict=True)
    )
    path.write_text(CASE.format(placeholder=placeholder) + rows)
    return path


def test_a_case_is_read_from_one_file(tmp_path: Path) -> None:
    path = write_case(tmp_path / "case.csv", [0.0, 5.0e-6], [0.0, 10.0], [400.0, 400.0])
    case = load_case(path)
    assert case.name == "test case" and case.source == "synthetic"
    assert not case.placeholder
    np.testing.assert_allclose(case.times, [0.0, 5.0e-6])
    np.testing.assert_allclose(case.current, [0.0, 10.0])
    assert case.reference == {"alpha_t": 0.5}
    region = case.region(height="2 cm")
    assert region.inner_radius == pytest.approx(0.008)
    assert region.pressure == 1.0 and region.height == pytest.approx(0.02)
    assert region.pulse_end == pytest.approx(10.0e-6)
    assert region.voltage_fraction == 0.18
    with pytest.warns(UserWarning, match="placeholders"):
        load_case(write_case(tmp_path / "p.csv", [0, 1e-6], [0, 1], [1, 1], "true"))
    (tmp_path / "bad.csv").write_text("# volume = 3\ntime_us,current_A,voltage_V\n")
    with pytest.raises(ValueError, match="unknown keys volume"):
        load_case(tmp_path / "bad.csv")
    (tmp_path / "header.csv").write_text("# name = x\nt,I,V\n0,0,0\n")
    with pytest.raises(ValueError, match="header"):
        load_case(tmp_path / "header.csv")


def test_the_copper_case_ships_with_the_examples() -> None:
    path = Path(__file__).parents[4] / "examples" / "data" / "irm_copper_case1.csv"
    with pytest.warns(UserWarning, match="placeholders"):
        case = load_case(path)
    assert case.placeholder
    region = case.region()
    # Table 4 of Gudmundsson et al. (2022), the 2 inch target
    assert (region.inner_radius, region.outer_radius) == pytest.approx((0.006, 0.02))
    assert (region.sheath_height, region.height) == pytest.approx((0.002, 0.028))
    assert region.pulse_end == pytest.approx(40.0e-6)
    assert set(case.reference) <= {
        "alpha_t",
        "beta_t",
        "alpha",
        "ionized_flux_fraction",
        "rarefaction_min",
        "argon_rarefaction_min",
        "back_attraction",
        "voltage_fraction",
    }
    assert case.current.max() == pytest.approx(20.0, rel=0.05)


def test_a_case_fits_its_pulse(tmp_path: Path) -> None:
    true = IonizationRegion(back_attraction=0.5, voltage_fraction=0.18, pulse_end=10e-6)
    times = np.linspace(0.0, 15.0e-6, 31)
    initial = true.initial_state(1.0e17)
    synthetic = true.evolve(times, None, lambda t: 400.0 if t < 10e-6 else 0.0, initial)
    case = load_case(
        write_case(
            tmp_path / "case.csv",
            times,
            synthetic.model_current.values,
            synthetic.voltage.values,
        )
    )
    fit = case.fit(guess=(0.45, 0.17), tolerance=1.0e-3)
    assert fit.back_attraction == pytest.approx(0.5, abs=0.03)
    assert fit.voltage_fraction == pytest.approx(0.18, abs=0.02)
    assert fit.averages["back_attraction"] == fit.back_attraction
