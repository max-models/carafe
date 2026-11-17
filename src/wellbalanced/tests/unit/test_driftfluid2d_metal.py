"""The sputtered metal of the 2D drift fluid: opt-in, conservative, species-dependent emission.

The material numbers are the placeholders of `wellbalanced.magnetron.MATERIALS`; the tests check
the bookkeeping, not the physics of the yields.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from wellbalanced.driftfluid2d import DriftFluid2D, exponential_field
from wellbalanced.magnetron import MATERIALS, lotz_ionization

E = 1.602176634e-19
TITANIUM = MATERIALS["Ti"]


def _model(**settings) -> DriftFluid2D:
    return DriftFluid2D(
        0.02,
        0.016,
        16,
        8,
        voltage=260.0,
        magnetic_field=exponential_field(0.12, 0.006),
        **settings,
    )


def _bombarded_state(model: DriftFluid2D):
    """A plasma whose argon and metal ions both stream into the cathode."""
    x = model.x[:, None] * np.ones((1, model.ny))
    state = model.initial_state(
        1e15,
        electron_temperature=4.0,
        metal_density=1e16 * np.exp(-x / 0.005),
        metal_ion_density=2e14,
    )
    state.ion_velocity[0] = -1.0e4
    state.metal_ion_velocity[0] = -5.0e3
    return state


def test_a_metal_without_yields_or_atoms_reproduces_argon_alone_exactly() -> None:
    """With zero yields and no metal, the argon fields are bit for bit those without a metal."""
    gamma = 0.07
    metal = dataclasses.replace(
        TITANIUM, argon_yield=0.0, self_yield=0.0, argon_secondary=gamma
    )
    plain = _model(electron_temperature=None, secondary_emission=gamma)
    with_metal = _model(electron_temperature=None, metal=metal)
    x, y = np.meshgrid(plain.x, plain.y, indexing="ij")
    density = (
        1e15
        * (1.0 + 0.1 * np.cos(2 * np.pi * y / 0.016))
        * np.exp(-(((x - 0.01) / 0.005) ** 2))
        + 1e13
    )
    a = plain.initial_state(density, electron_temperature=4.0)
    b = with_metal.initial_state(density, electron_temperature=4.0)
    dt = plain.suggest_timestep(a)
    for _ in range(10):
        a = plain.step(a, dt)
        b = with_metal.step(b, dt)
    for name in (
        "electron_density",
        "ion_density",
        "ion_velocity",
        "gas_density",
        "electron_temperature",
        "potential",
    ):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert np.all(b.metal_density == 0.0)
    assert np.all(b.metal_ion_density == 0.0)
    assert a.metal_density is None and "metal_density" not in a.to_dataset()


def test_the_sputtered_atoms_are_the_yields_times_the_ion_fluxes_into_the_cathode() -> (
    None
):
    """One stage: the atoms gained are ``Y_Ar Gamma_Ar+ + Y_M Gamma_M+`` less the anode outflow."""
    metal = dataclasses.replace(TITANIUM, charge_exchange=0.0)
    model = _model(
        electron_temperature=5.0, metal=metal, metal_ionization_rate=lambda te: 0.0 * te
    )
    state = _bombarded_state(model)
    budget = model.cathode_budget(state)
    argon = budget["argon_ions"]
    metal_ions = budget["metal_ions"]
    assert argon == pytest.approx(1e15 * 1e4)
    assert metal_ions == pytest.approx(2e14 * 5e3)
    sputtered = metal.argon_yield * argon + metal.self_yield * metal_ions
    assert budget["sputtered"] == pytest.approx(sputtered)
    assert budget["self_sputtering_share"] == pytest.approx(
        metal.self_yield * metal_ions / sputtered
    )
    dt = 1e-10
    u0 = model._pack(state)
    u1, _ = model._euler(u0, state.potential, dt)
    gained = (u1["M"] - u0["M"]).sum(axis=0) * model.dx
    anode = model.metal_speed * u0["M"][-1]
    assert gained == pytest.approx(dt * (sputtered - anode), rel=1e-10)


def test_ionization_and_charge_exchange_conserve_the_metal_and_the_charge() -> None:
    """Atoms plus ions change only by the boundary fluxes; Poisson holds after a stage."""
    model = _model(electron_temperature=None, metal=TITANIUM)
    state = _bombarded_state(model)
    dt = 1e-10
    u0 = model._pack(state)
    u1, phi = model._euler(u0, state.potential, dt)
    gam_mx, _ = model._ion_fluxes(u0["nm"], (u0["vmx"], u0["vmy"]))
    sputtered = model.cathode_budget(state)["sputtered"]
    inflow = sputtered - model.metal_speed * u0["M"][-1] + gam_mx[0] - gam_mx[-1]
    total = (u1["M"] + u1["nm"] - u0["M"] - u0["nm"]).sum(axis=0) * model.dx
    assert total == pytest.approx(dt * inflow, rel=1e-9)
    # the metal is ionized: the ions grow by more than their own transport
    ionized = (u1["nm"] - u0["nm"]).sum() * model.dx + dt * (
        gam_mx[-1] - gam_mx[0]
    ).sum()
    assert ionized > 0.0
    assert model.charge_residual(model._unpack(u1, phi, dt)) < 1e-6


def test_the_emission_depends_on_the_ion_species() -> None:
    """``gamma_Ar Gamma_Ar+ + gamma_M Gamma_M+``; a prescribed current overrides both."""
    metal = dataclasses.replace(TITANIUM, argon_secondary=0.1, self_secondary=0.02)
    model = _model(metal=metal)
    argon = np.full(model.ny, -1e18)
    ions = np.full(model.ny, -5e17)
    assert model._cathode_flux(argon, metal_ion_flux=ions) == pytest.approx(
        0.1 * 1e18 + 0.02 * 5e17
    )
    # a metal whose singly charged ions emit nothing: only the argon ions count
    model.metal = dataclasses.replace(metal, self_secondary=0.0)
    assert model._cathode_flux(argon, metal_ion_flux=ions) == pytest.approx(0.1 * 1e18)
    model.metal = dataclasses.replace(metal, argon_secondary=0.0)
    assert model._cathode_flux(argon, metal_ion_flux=ions) == pytest.approx(0.02 * 5e17)
    # the same through the state: a self-sputtering cathode emits by gamma_M alone
    model.metal = metal
    state = _bombarded_state(model)
    state.ion_velocity[0] = 0.0
    budget = model.cathode_budget(state)
    assert budget["argon_ions"] == pytest.approx(0.0)
    assert budget["emitted"] == pytest.approx(0.02 * budget["metal_ions"])
    assert budget["self_sputtering_share"] == pytest.approx(1.0)
    model.cathode_current_density = 0.03
    assert model.cathode_budget(state)["emitted"] == pytest.approx(0.03 / E)


def test_the_metal_ionization_costs_the_electrons_its_energy() -> None:
    model = _model(electron_temperature=None, metal=TITANIUM)
    rate = lotz_ionization(TITANIUM.ionization_energy, TITANIUM.outer_electrons)
    assert model.metal_ionization_rate(4.0) == pytest.approx(rate(4.0))
    shape = (model.nx, model.ny)
    ne, te = np.full(shape, 1e15), np.full(shape, 4.0)
    gas = np.full(shape, model.gas_density)
    gas_c = model._collision_gas(gas, te)
    atoms = np.full(shape, 1e18)
    extra = model._energy_loss(ne, te, gas, gas_c, atoms) - model._energy_loss(
        ne, te, gas, gas_c
    )
    assert extra == pytest.approx(TITANIUM.ionization_energy * rate(4.0) * 1e18 * 1e15)


def test_a_metal_needs_the_electrodes_and_a_state_with_the_metal() -> None:
    with pytest.raises(ValueError, match="periodic_x"):
        DriftFluid2D(0.01, 0.01, 4, 4, field_bc=-1e3, periodic_x=True, metal=TITANIUM)
    with pytest.raises(ValueError, match="metal_speed"):
        _model(metal=TITANIUM, metal_speed=0.0)
    state = _model().initial_state(1e15)
    with pytest.raises(ValueError, match="state with the metal"):
        _model(metal=TITANIUM).step(state, 1e-10)


def test_a_short_1d_run_with_a_titanium_target_sputters_and_ionizes_the_metal() -> None:
    gas = 0.6 / (1.380649e-23 * 300.0)
    model = DriftFluid2D(
        0.02,
        0.008,
        16,
        4,
        voltage=260.0,
        magnetic_field=exponential_field(0.12, 0.006),
        gas_density=gas,
        electron_temperature=None,
        metal=TITANIUM,
    )
    state = model.initial_state(1e15, electron_temperature=3.0)
    run = model.run(state, 2e-7)
    final = run.state
    assert np.all(np.isfinite(final.potential))
    assert final.electron_temperature.max() < 40.0
    assert final.electron_density.max() < 5e15
    # the atoms come from the cathode and thin out towards the anode
    profile = final.metal_density.mean(axis=1)
    assert profile[0] > profile[-1] > 0.0
    assert final.metal_ion_density.max() > 0.0
    budget = model.cathode_budget(final)
    assert np.all(budget["sputtered"] > 0.0)
    assert np.all(
        (budget["self_sputtering_share"] >= 0.0)
        & (budget["self_sputtering_share"] < 1.0)
    )
    assert model.charge_residual(final) < 1e-5
    data = final.to_dataset()
    assert {"metal_density", "metal_ion_density", "metal_ion_velocity_x"} <= set(data)
    copy = final.copy()
    copy.metal_density[:] = 0.0
    assert final.metal_density.max() > 0.0
