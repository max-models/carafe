"""The 1D Hall thruster model: conservation, current, limits and the SPT-100 regime.

The xenon rates and the wall loss are the PLACEHOLDER fits of `carafe.hall`.
"""

from __future__ import annotations

import numpy as np
import pytest
from plasmacoll.constants import ELEMENTARY_CHARGE

from carafe import HallThruster1D, Reaction
from carafe.hall import (
    XENON,
    ElectronContext,
    ExponentialWallLoss,
    GaussianField,
    TwoZoneBohm,
    xenon_elastic_rate,
    xenon_excitation_rate,
    xenon_ionization_rate,
    xenon_reactions,
)

E = ELEMENTARY_CHARGE


def no_ionization() -> dict[str, Reaction]:
    reactions = xenon_reactions()
    reactions["ionization"] = Reaction(
        ("e", "Xe"), ("e", "e", "Xe+"), 0.0, electron_energy_loss=12.13
    )
    return reactions


@pytest.fixture(scope="module")
def spt100():
    """SPT-100-like inputs (300 V, 5 mg/s) on a coarse mesh, run past the start-up."""
    model = HallThruster1D(
        discharge_voltage=300.0, mass_flow="5 mg/s", cells=60, length=0.05
    )
    return model, model.run(4.0e-4, samples=200)


def test_the_xenon_fits_have_the_right_size() -> None:
    te = np.array([2.0, 5.0, 10.0, 20.0, 50.0])
    ionization = xenon_ionization_rate(te)
    assert np.all(np.diff(ionization) > 0.0)
    assert 1.0e-14 < xenon_ionization_rate(10.0) < 1.0e-13
    assert 1.0e-14 < xenon_excitation_rate(10.0) < 1.0e-13
    assert 1.0e-14 < xenon_elastic_rate(10.0) < 1.0e-12
    assert xenon_ionization_rate(0.5) < 1.0e-20


def test_settings_are_checked() -> None:
    with pytest.raises(ValueError, match="channel_length"):
        HallThruster1D(channel_length=0.06, length=0.05)
    with pytest.raises(ValueError, match="cathode"):
        HallThruster1D(cathode=0.07)
    with pytest.raises(ValueError, match="cells"):
        HallThruster1D(cells=3)
    with pytest.raises(ValueError, match="mass_flow"):
        HallThruster1D(mass_flow=0.0)
    with pytest.raises(ValueError, match="limiter"):
        HallThruster1D(limiter="superbee")
    with pytest.raises(ValueError, match="missing"):
        HallThruster1D(reactions={"ionization": xenon_reactions()["ionization"]})


def test_units_are_converted() -> None:
    model = HallThruster1D(mass_flow="5 mg/s", channel_length="2.5 cm")
    assert model.mass_flow == pytest.approx(5.0e-6)
    assert model.channel_length == pytest.approx(0.025)
    assert model.cathode == model.length


def test_the_closures() -> None:
    z = np.linspace(0.0, 0.05, 11)
    field = GaussianField(peak=0.02, location=0.025)
    b = field(z)
    assert b.max() == pytest.approx(0.02)
    assert b[0] < b[5] and b[-1] < b[5]
    context = ElectronContext(
        z=z,
        magnetic_field=b,
        density=np.full_like(z, 1.0e17),
        electron_temperature=np.full_like(z, 20.0),
        ion_velocity=np.zeros_like(z),
        neutral_density=np.full_like(z, 1.0e19),
        channel_length=0.025,
    )
    jump = TwoZoneBohm(inside=0.01, outside=0.1, transition=0.0)(context)
    omega = E * b / 9.1093837e-31
    np.testing.assert_allclose(jump[:5], 0.01 * omega[:5], rtol=1e-6)
    np.testing.assert_allclose(jump[5:], 0.1 * omega[5:], rtol=1e-6)
    smooth = TwoZoneBohm(inside=0.01, outside=0.1)(context)
    assert smooth[0] == pytest.approx(jump[0], rel=1e-6)
    loss = ExponentialWallLoss(inside=1.0, outside=0.0)(context)
    assert loss[0] == pytest.approx(1.0e7 * 20.0 * np.exp(-1.0))
    assert loss[-1] == 0.0


def test_heavy_particles_are_conserved() -> None:
    """The inventory changes by the anode inflow minus the outflow, to round-off."""
    model = HallThruster1D(cells=40)
    start = model.initial_state()
    end = model.run(2.0e-5, start, samples=10)
    before = np.sum(start.neutral_density + start.ion_density) * model.h
    after = np.sum(end.neutral_density + end.ion_density) * model.h
    inflow, outflow = model.boundary_budget
    assert inflow == pytest.approx(model.inflow * (end.time - start.time), rel=1e-12)
    assert after - before == pytest.approx(inflow - outflow, rel=1e-9)


def test_the_current_is_the_same_everywhere_and_the_voltage_is_applied() -> None:
    """Ohm's law with the field and the profiles gives ``I_d`` on every cell."""
    model = HallThruster1D(cells=50, cathode=0.04)
    state = model.run(5.0e-6, samples=5)
    ni, te = state.ion_density, state.electron_temperature
    mu = model._mobility(
        state.neutral_density, ni, te, state.anomalous_frequency
    )
    electron_flux = -mu * (ni * state.electric_field + np.gradient(ni * te, model.h))
    current = E * model.area * (ni * state.ion_velocity - electron_flux)
    np.testing.assert_allclose(current, state.discharge_current, rtol=1e-10)
    inside = np.clip((model.cathode - model.faces[:-1]) / model.h, 0.0, 1.0)
    applied = np.sum(inside * state.electric_field) * model.h
    assert applied == pytest.approx(model.discharge_voltage, rel=1e-12)
    assert state.potential[0] == pytest.approx(300.0, abs=10.0)


def test_without_ionization_and_voltage_the_gas_streams_through() -> None:
    """No ionization, no voltage: the neutrals fill the channel at ``mdot / (M A u_n)``."""
    model = HallThruster1D(
        discharge_voltage=0.0,
        reactions=no_ionization(),
        neutral_speed=2000.0,
        cells=40,
    )
    start = model.initial_state()
    state = model.run(1.0e-4, start, samples=20)
    uniform = model.mass_flow / (XENON.mass * model.area * model.neutral_speed)
    assert state.ion_density.sum() < 0.5 * start.ion_density.sum()
    recombined = (state.neutral_density - uniform) / uniform
    assert np.all(recombined > -1e-6)
    assert np.max(recombined) < 0.02
    assert abs(state.discharge_current) < 0.1
    assert state.thrust < 1.0e-3


def test_spt100_lands_in_the_range_of_the_thruster(spt100) -> None:
    """300 V and 5 mg/s: a few amperes, about 100 mN, ions near 20 km/s."""
    model, state = spt100
    current = state.average("discharge_current")
    thrust = state.average("thrust")
    assert 4.0 < current < 8.0
    assert 0.08 < thrust < 0.12
    assert 0.8 < state.average("ion_current") * XENON.mass / (E * model.mass_flow) < 1.05
    assert 15.0e3 < state.ion_velocity[-1] < 22.0e3
    at_exit = np.interp(model.channel_length, state.z, state.ion_velocity)
    assert 10.0e3 < at_exit < 19.0e3
    assert 15.0 < state.electron_temperature.max() < 60.0
    assert 1200.0 < state.specific_impulse < 2400.0
    assert 0.3 < state.anode_efficiency < 0.8


def test_the_dataset_and_a_continued_run(spt100) -> None:
    model, state = spt100
    data = state.to_dataset()
    for name in ("neutral_density", "ion_velocity", "electron_temperature", "potential"):
        assert data[name].dims == ("z",)
        assert "$" in data[name].attrs["long_name"]
    assert data.discharge_current_history.dims == ("time",)
    assert float(data.thrust) == pytest.approx(state.thrust * 1e3)
    assert data.z.attrs["units"] == "mm"
    more = model.run(2.0e-6, state, samples=4)
    assert len(more.history["time"]) == len(state.history["time"]) + 4
    assert more.time == pytest.approx(state.time + 2.0e-6, rel=1e-3)
    assert model.stable_step(more) > 0.0
