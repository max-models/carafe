"""The magnetron global model: rates, balances, limits and pulses (placeholder yields)."""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest
from plasmacoll.constants import ELECTRON_MASS, ELEMENTARY_CHARGE
from scipy.integrate import quad

from wellbalanced import ELECTRON, Chamber, GlobalModel, Reaction
from wellbalanced.magnetron import (
    MATERIALS,
    TITANIUM,
    Magnetron,
    TargetMaterial,
    WindEfficiency,
    argon_elastic,
    argon_excitation,
    argon_ionization,
    lotz_ionization,
    pulse,
    sputter_wind_gas,
    wind_efficiency_from_gas_dynamics,
)

E = ELEMENTARY_CHARGE
pytestmark = pytest.mark.filterwarnings("ignore:.*placeholders, not data")


def test_the_argon_fits_have_their_textbook_values() -> None:
    te = 3.0
    assert argon_ionization(te) == pytest.approx(
        2.34e-14 * te**0.59 * np.exp(-17.44 / te)
    )
    assert argon_excitation(te) == pytest.approx(
        2.48e-14 * te**0.33 * np.exp(-12.78 / te)
    )
    # the elastic rate is of order 1e-13 m^3/s at a few eV
    assert 3.0e-14 < argon_elastic(te) < 3.0e-13
    # the fits do not overflow at the wild values an integrator tries
    assert np.isfinite(argon_elastic(1.0e8)) and np.isfinite(argon_ionization(1.0e-9))


@pytest.mark.parametrize("te", [1.0, 3.0, 10.0])
def test_the_lotz_rate_is_the_maxwellian_average_of_its_cross_section(
    te: float,
) -> None:
    ionization_energy, electrons = 6.83, 2

    def sigma(energy: float) -> float:  # m^2, energy in eV
        return (
            electrons
            * 4.5e-18
            * np.log(energy / ionization_energy)
            / (energy * ionization_energy)
        )

    def integrand(energy: float) -> float:
        speed = np.sqrt(2.0 * energy * E / ELECTRON_MASS)
        weight = 2.0 * np.sqrt(energy / np.pi) * te**-1.5 * np.exp(-energy / te)
        return sigma(energy) * speed * weight

    expected, _ = quad(integrand, ionization_energy, np.inf)
    rate = lotz_ionization(ionization_energy, electrons)
    assert rate(te) == pytest.approx(expected, rel=1e-6)


def test_the_material_table_holds_the_metals_of_the_review() -> None:
    assert set(MATERIALS) == {"Al", "Cr", "Cu", "Ti", "Au", "Mo", "Nb"}
    assert MATERIALS["Ti"].ionization_energy == pytest.approx(6.83, abs=0.01)
    assert all(not m.verified for m in MATERIALS.values())
    with pytest.raises(ValueError, match="yield"):
        TargetMaterial("X", 10.0, 7.0, 1, -1.0, 0.0, 0.1, 0.1, 1e-15)


def test_a_placeholder_material_warns() -> None:
    with pytest.warns(UserWarning, match="placeholders"):
        Magnetron()


def test_validation() -> None:
    with pytest.raises(ValueError, match="h_factor"):
        Magnetron(h_factor=0.0)
    with pytest.raises(ValueError, match="heating_fraction"):
        Magnetron(heating_fraction=-1.0)
    with pytest.raises(ValueError, match="on_time"):
        pulse(1.0, 2.0, 1.0)


def test_the_pulse_waveform() -> None:
    wave = pulse(100.0, 1.0e-4, 1.0e-3)
    assert wave(0.0) == 100.0 and wave(0.5e-4) == 100.0
    assert wave(2.0e-4) == 0.0 and wave(1.0e-3 + 0.5e-4) == 100.0


def test_a_dc_state_balances_every_species_and_the_metal_is_conserved() -> None:
    magnetron = Magnetron(wind_efficiency=0.3)
    state = magnetron.steady_state(2.0, 500.0)
    assert state.converged
    for name, rate in state.particle_balance.items():
        if name != "e":
            assert abs(rate) < 1.0e-6 * state.densities[name] * 1.0e5, name
    # the sputtered atoms leave as neutrals or as ions: nothing is created or lost
    currents = magnetron.target_currents(state)
    sputtered = currents["sputtered"] / magnetron.volume
    metal = magnetron.species["metal"].name
    metal_ion = magnetron.species["metal_ion"].name
    leaving = state.wall_losses[metal] + state.wall_losses[metal_ion]
    assert sputtered == pytest.approx(leaving, rel=1e-6)
    # quasi-neutrality and the power balance of the electrons
    assert state.densities["e"] == pytest.approx(
        state.densities["Ar+"] + state.densities["Ti+"]
    )
    assert sum(state.power.values()) == pytest.approx(
        0.0, abs=1.0e-6 * state.power["absorbed"]
    )


def test_without_sputtering_it_is_the_argon_global_model() -> None:
    """Zero yields and no metal: the plasma is that of the plain argon model."""
    inert = dataclasses.replace(
        TITANIUM, argon_yield=0.0, self_yield=0.0, charge_exchange=0.0
    )
    magnetron = Magnetron(material=inert, seed=0.0)
    drive = magnetron.heating_power(1.0, 500.0)
    # the metal has no source: start it empty-ish and let it ionize away
    state = magnetron.steady_state(
        1.0, 500.0, {**magnetron.guess(), "Ti": 1.0e5, "Ti+": 1.0e3}
    )
    s = magnetron.species
    plain = GlobalModel(
        species=[ELECTRON, s["gas"], s["gas_ion"]],
        reactions=[
            r
            for r in magnetron.reactions()
            if "Ti" not in r.name and r.name != "charge exchange"
        ],
        chamber=Chamber(
            volume=magnetron.volume,
            wall_area=magnetron.chamber.wall_area,
            ion_loss_area=magnetron.target_area + magnetron.side_area,
            h_factor=magnetron.h_factor,
        ),
        power=drive,
        feed={"Ar": magnetron.gas_density * magnetron.volume / magnetron.refill_time},
        ion_wall_products={"Ar+": {"Ar": 1}},
        gas_temperature=magnetron.model(1.0, 500.0).gas_temperature,
    ).steady_state(
        {"Ar": magnetron.gas_density, "Ar+": 1.0e17, "Te": 3.0},
    )
    # the metal has decayed away (its relative rates never settle: 0/0), the argon has
    assert plain.converged
    assert state.densities["Ti"] + state.densities["Ti+"] < 1.0
    assert abs(state.particle_balance["Ar+"]) < 1.0e-6 * state.densities["Ar+"]
    assert state.electron_temperature == pytest.approx(
        plain.electron_temperature, rel=1e-3
    )
    assert state.densities["Ar+"] == pytest.approx(plain.densities["Ar+"], rel=1e-3)
    assert (
        magnetron.diagnostics(state)["self_sputtering_share"] != 1.0
    )  # defined, no 0/0


def test_the_spoke_diagnostics_against_the_current() -> None:
    """Rarefaction, the metal share and self-sputtering grow with the current."""
    magnetron = Magnetron(wind_efficiency=0.3)
    data = magnetron.current_scan([0.1, 1.0, 10.0, 50.0], 500.0)
    assert bool(data.converged.all())
    assert data.electron_density.values[0] < data.electron_density.values[-1]
    assert np.all(np.diff(data.rarefaction.values) < 0.0)
    assert data.rarefaction.values[0] > 0.99
    assert np.all(np.diff(data.metal_to_argon.values) > 0.0)
    assert np.all(np.diff(data.metal_ion_fraction.values) >= -1.0e-9)
    assert np.all(np.diff(data.self_sputtering_share.values) >= -1.0e-9)
    assert data.density.dims[0] == "current"
    assert data.current.attrs["units"] == "A"


def test_the_sputter_wind_is_what_rarefies_the_gas() -> None:
    quiet = Magnetron(wind_efficiency=0.0).steady_state(20.0, 600.0)
    windy = Magnetron(wind_efficiency=1.0).steady_state(20.0, 600.0)
    assert windy.densities["Ar"] < 0.9 * quiet.densities["Ar"]


def test_the_heating_fraction_is_calibrated_to_a_measured_current() -> None:
    magnetron = Magnetron(wind_efficiency=0.3)
    fitted = magnetron.calibrate_heating(2.0, 500.0)
    state = fitted.steady_state(2.0, 500.0)
    assert fitted.target_currents(state)["I_D"] == pytest.approx(2.0, rel=1e-4)
    with pytest.raises(ValueError, match="reproduces"):
        # a current no heating can reach: the model has no ions to carry it
        magnetron.calibrate_heating(1.0e6, 500.0)


def test_a_pulse_reaches_a_periodic_state() -> None:
    magnetron = Magnetron(wind_efficiency=0.3).calibrate_heating(2.0, 500.0)
    data = magnetron.pulse(
        20.0, 600.0, 1.0e-4, 5.0e-4, periods=12, points=60, tolerance=1.0e-4
    )
    assert data.attrs["periodicity_error"] < 1.0e-4
    assert data.attrs["periods"] < 12
    # periodic: the state after one period is the state at its start
    density = data.density.sel(species="e")
    assert float(density[-1]) == pytest.approx(float(density[0]), rel=1.0e-3)
    # the current rises in the pulse and decays in the afterglow
    current = data.I_D.values
    peak = int(np.argmax(current))
    assert 0 < peak < len(current) - 1
    assert current[peak] > 100.0 * current[0]
    assert data.t.attrs["units"] == "s"
    # the afterglow does not cool the electrons below the gas
    gas_ev = 300.0 * 1.380649e-23 / E
    assert float(data.electron_temperature.min()) >= gas_ev * (1.0 - 1.0e-6)


def _first_state(magnetron: Magnetron, data) -> dict[str, float]:
    """The densities at the start of a pulse dataset, as an initial guess."""
    start = data.isel(t=0)
    densities = {
        str(s): float(start.density.sel(species=s)) for s in data.species.values
    }
    return {**densities, "Te": float(start.electron_temperature)}


def test_a_pulse_restarted_from_its_periodic_state_stays_periodic() -> None:
    magnetron = Magnetron(wind_efficiency=0.3).calibrate_heating(2.0, 500.0)
    kwargs = {"periods": 12, "points": 40, "tolerance": 1.0e-4}
    first = magnetron.pulse(20.0, 600.0, 1.0e-4, 5.0e-4, **kwargs)
    again = magnetron.pulse(
        20.0, 600.0, 1.0e-4, 5.0e-4, **kwargs, initial=_first_state(magnetron, first)
    )
    assert again.attrs["periods"] <= 2
    np.testing.assert_allclose(again.I_D.values, first.I_D.values, rtol=2.0e-2)


def test_a_dc_scan_continues_from_state_to_state() -> None:
    magnetron = Magnetron()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        states = magnetron.scan("pressure", [0.5, 1.0, 2.0], 2.0, 500.0)
    assert all(s.converged for s in states)
    # a higher pressure holds a denser argon region
    assert states[0].densities["Ar"] < states[-1].densities["Ar"]


def test_reactions_are_the_listed_ones() -> None:
    reactions = Magnetron().reactions()
    assert all(isinstance(r, Reaction) for r in reactions)
    names = {r.name for r in reactions}
    assert {"Ar ionization", "Ti ionization", "charge exchange"} <= names


# -- the limits of the model ------------------------------------------------------------


def _inert() -> TargetMaterial:
    """Titanium that sputters nothing and exchanges no charge."""
    return dataclasses.replace(
        TITANIUM, argon_yield=0.0, self_yield=0.0, charge_exchange=0.0
    )


def test_without_sputtering_the_wind_does_nothing_and_the_argon_is_the_plain_one() -> (
    None
):
    """Zero yields: no metal, any wind efficiency, and the argon of the plain model."""
    guess = {**Magnetron(material=_inert()).guess(), "Ti": 1.0e5, "Ti+": 1.0e3}
    states = [
        Magnetron(material=_inert(), wind_efficiency=wind).steady_state(
            1.0, 500.0, guess
        )
        for wind in (0.0, 1.0, WindEfficiency((1.0e21, 1.0e23), (2.0, 0.5)))
    ]
    reference = states[0]
    for state in states:
        assert state.densities["Ti"] + state.densities["Ti+"] < 1.0
        for name in ("Ar", "Ar+"):
            assert state.densities[name] == pytest.approx(
                reference.densities[name], rel=1e-9
            )
    # without the wind the neutral argon is the filling density: the ions come back
    # as neutrals and only neutrals flow out (plus the seed's pairs, tau S_seed)
    magnetron = Magnetron(material=_inert())
    expected = magnetron.gas_density + magnetron.refill_time * magnetron.seed
    assert reference.densities["Ar"] == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    "wind", [0.0, 0.3, WindEfficiency((1.0e21, 1.0e22, 1.0e23), (1.0, 0.7, 0.1))]
)
def test_dc_particle_budgets_close(wind) -> None:
    """Ionization equals the ion losses and the argon budget gives the rarefaction."""
    magnetron = Magnetron(wind_efficiency=wind)
    state = magnetron.steady_state(20.0, 600.0)
    assert state.converged
    rates, walls = state.reaction_rates, state.wall_losses
    seed = magnetron.seed
    # the pairs the volume makes leave as ions through the target and the sides
    # (charge exchange swaps an ion for an ion)
    ionization = rates["Ar ionization"] + rates["Ti ionization"] + seed
    assert ionization == pytest.approx(walls["Ar+"] + walls["Ti+"], rel=1e-6)
    assert walls["e"] == pytest.approx(walls["Ar+"] + walls["Ti+"], rel=1e-9)
    # each ion separately: Ar+ by ionization and the seed, lost to the walls and to
    # charge exchange; Ti+ from ionization and charge exchange
    exchange = rates["charge exchange"]
    assert rates["Ar ionization"] + seed == pytest.approx(
        walls["Ar+"] + exchange, rel=1e-6
    )
    assert rates["Ti ionization"] + exchange == pytest.approx(walls["Ti+"], rel=1e-6)
    # the argon budget: fed n_fill / tau, out n_Ar / tau and eta Gamma A_t / V, the
    # ions return as neutrals, so 1 - n_Ar / n_fill = eta tau Gamma / (h_r n_fill)
    flux = magnetron.target_currents(state)["sputtered"] / magnetron.target_area
    eta = magnetron.wind_efficiency_at(state)
    fill, tau = magnetron.gas_density, magnetron.refill_time
    expected = 1.0 - eta * tau * flux / (magnetron.region_height * fill)
    expected += tau * seed / fill
    assert magnetron.diagnostics(state)["rarefaction"] == pytest.approx(
        expected, rel=1e-6
    )
    if callable(wind):
        assert eta == pytest.approx(wind(flux))
        assert 0.1 < eta < 1.0


def test_the_discharge_current_is_the_ion_current_to_the_target_with_its_secondaries() -> (
    None
):
    """``I_D = sum (1 + gamma_i) e Gamma_i A_t``, with the target's share of the ion losses."""
    magnetron = Magnetron(wind_efficiency=0.3)
    state = magnetron.steady_state(20.0, 600.0)
    m = magnetron.material
    share = magnetron.target_area / (magnetron.target_area + magnetron.side_area)
    to_target = {
        name: E * state.wall_losses[name] * share * magnetron.volume
        for name in ("Ar+", "Ti+")
    }
    currents = magnetron.target_currents(state)
    assert currents["I_Ar+"] == pytest.approx(to_target["Ar+"], rel=1e-12)
    assert currents["I_Ti+"] == pytest.approx(to_target["Ti+"], rel=1e-12)
    expected = (1.0 + m.argon_secondary) * to_target["Ar+"]
    expected += (1.0 + m.self_secondary) * to_target["Ti+"]
    assert currents["I_D"] == pytest.approx(expected, rel=1e-12)
    # and the sputtered flux is what the ions to the target sputter
    sputtered = (m.argon_yield * to_target["Ar+"] + m.self_yield * to_target["Ti+"]) / E
    assert currents["sputtered"] == pytest.approx(sputtered, rel=1e-12)


# -- the wind efficiency from the gas dynamics --------------------------------------------


def test_a_constant_wind_efficiency_function_is_the_number() -> None:
    number = Magnetron(wind_efficiency=0.3).steady_state(20.0, 600.0)
    function = Magnetron(
        wind_efficiency=WindEfficiency((1.0e21, 1.0e23), (0.3, 0.3))
    ).steady_state(20.0, 600.0)
    for name, density in number.densities.items():
        assert function.densities[name] == pytest.approx(density, rel=1e-8)


def test_the_wind_efficiency_interpolates_in_the_logs() -> None:
    eta = WindEfficiency((1.0e21, 1.0e22, 1.0e23), (1.0, 0.5, 0.05), (0.1, 0.2, 0.3))
    assert eta(1.0e22) == pytest.approx(0.5)
    # constant outside the points
    assert eta(1.0e18) == pytest.approx(1.0) and eta(1.0e25) == pytest.approx(0.05)
    assert eta(0.0) == pytest.approx(1.0)
    values = eta(np.geomspace(1.0e21, 1.0e23, 9))
    assert np.all(np.diff(values) < 0.0)  # monotone between monotone points
    # a power law between two points
    pair = WindEfficiency((1.0e21, 1.0e23), (1.0, 0.01))
    assert pair(1.0e22) == pytest.approx(0.1)
    assert WindEfficiency((1.0e22,), (0.4,))(3.0e20) == pytest.approx(0.4)
    with pytest.raises(ValueError, match="increasing"):
        WindEfficiency((1.0e22, 1.0e21), (1.0, 1.0))
    with pytest.raises(ValueError, match=">= 0"):
        WindEfficiency((1.0e21, 1.0e22), (1.0, -1.0))
    with pytest.raises(ValueError, match="same length"):
        WindEfficiency((1.0e21, 1.0e22), (1.0,))
    with pytest.raises(ValueError, match="wind_efficiency"):
        Magnetron(wind_efficiency=-0.1)


def test_the_wind_efficiency_from_the_gas_dynamics_matches_its_deficit() -> None:
    """``eta`` makes the 0D deficit the gas's mean deficit; a coarse, cheap gas."""
    magnetron = Magnetron()
    fluxes = (3.0e21, 3.0e22)

    def gas(flux: float):
        return sputter_wind_gas(magnetron, flux, cells=12)

    eta = wind_efficiency_from_gas_dynamics(magnetron, fluxes, gas=gas, tolerance=1e-3)
    assert eta.flux == fluxes
    deficits = np.array(eta.deficit)
    assert np.all((deficits > 0.0) & (deficits < 1.0))
    assert deficits[1] > deficits[0]  # more sputtering, more rarefaction
    assert eta.efficiency[1] < eta.efficiency[0]  # but less per sputtered atom
    for flux, deficit, value in zip(fluxes, deficits, eta.efficiency, strict=True):
        zero_d = value * magnetron.refill_time * flux
        zero_d /= magnetron.region_height * magnetron.gas_density
        assert zero_d == pytest.approx(deficit, rel=1e-12)
    # the gas: the target emits the flux into argon at the filling pressure
    model = gas(fluxes[0])
    assert model.boundaries["left"].metal_flux == fluxes[0]
    assert model.boundaries["right"].pressure == magnetron.pressure
    assert model.length == magnetron.region_height
