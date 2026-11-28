"""The sputtered metal of the 2D drift fluid: opt-in, conservative, species-dependent emission.

The material numbers are the placeholders of `carafe.magnetron.MATERIALS`; the tests check
the bookkeeping and the limits, not the physics of the yields.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from carafe.driftfluid2d import DriftFluid2D, exponential_field, lumped_excitation_rate
from carafe.magnetron import MATERIALS, lotz_ionization

E = 1.602176634e-19
TITANIUM = MATERIALS["Ti"]
COPPER = MATERIALS["Cu"]
FIELDS = (
    "electron_density",
    "ion_density",
    "ion_velocity",
    "gas_density",
    "electron_temperature",
    "potential",
)


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
    """A plasma whose argon and metal ions all stream into the cathode."""
    x = model.x[:, None] * np.ones((1, model.ny))
    state = model.initial_state(
        1e15,
        electron_temperature=4.0,
        metal_density=1e16 * np.exp(-x / 0.005),
        metal_ion_density=2e14,
        metal_ion2_density=5e13,
    )
    state.ion_velocity[0] = -1.0e4
    state.metal_ion_velocity[0] = -5.0e3
    if state.metal_ion2_velocity is not None:
        state.metal_ion2_velocity[0] = -7.0e3
    return state


def _relaxed_density(model: DriftFluid2D) -> np.ndarray:
    x, y = np.meshgrid(model.x, model.y, indexing="ij")
    return (
        1e15
        * (1.0 + 0.1 * np.cos(2 * np.pi * y / 0.016))
        * np.exp(-(((x - 0.01) / 0.005) ** 2))
        + 1e13
    )


def test_a_metal_without_yields_or_atoms_reproduces_argon_alone_exactly() -> None:
    """With zero yields and no metal, the argon fields are bit for bit those without a metal."""
    gamma = 0.07
    metal = dataclasses.replace(
        TITANIUM, argon_yield=0.0, self_yield=0.0, argon_secondary=gamma
    )
    plain = _model(electron_temperature=None, secondary_emission=gamma)
    with_metal = _model(electron_temperature=None, metal=metal)
    density = _relaxed_density(plain)
    a = plain.initial_state(density, electron_temperature=4.0)
    b = with_metal.initial_state(density, electron_temperature=4.0)
    dt = plain.suggest_timestep(a)
    for _ in range(10):
        a = plain.step(a, dt)
        b = with_metal.step(b, dt)
    for name in FIELDS:
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert np.all(b.metal_density == 0.0)
    assert np.all(b.metal_ion_density == 0.0)
    assert np.all(b.metal_ion2_density == 0.0)
    assert a.metal_density is None and "metal_density" not in a.to_dataset()
    assert {"metal_ion2_density", "metal_ion2_velocity_x"} <= set(b.to_dataset())


def test_without_double_ionization_the_model_is_the_one_with_singly_charged_ions() -> (
    None
):
    """A zero ``M+ -> M2+`` rate gives bit for bit the model without ``M2+``."""
    density = _relaxed_density(_model())
    runs = []
    for settings in (
        {"double_ionization": False},
        {"metal_double_ionization_rate": lambda te: 0.0 * te},
    ):
        model = _model(electron_temperature=None, metal=TITANIUM, **settings)
        state = model.initial_state(
            density,
            electron_temperature=4.0,
            metal_density=1e17,
            metal_ion_density=1e13,
        )
        dt = model.suggest_timestep(state)
        for _ in range(5):
            state = model.step(state, dt)
        runs.append(state)
    single, zero_rate = runs
    assert single.metal_ion2_density is None
    assert np.all(zero_rate.metal_ion2_density == 0.0)
    for name in (*FIELDS, "metal_density", "metal_ion_density"):
        assert np.array_equal(getattr(single, name), getattr(zero_rate, name)), name
    with pytest.raises(ValueError, match="with the M2\\+"):
        _model(metal=TITANIUM).step(dataclasses.replace(single), 1e-10)


def test_the_sputtered_atoms_are_the_yields_at_the_ion_energies_times_the_fluxes() -> (
    None
):
    """One stage: atoms gained ``sum_s Y_s(E_s) Gamma_s`` less the anode outflow."""
    metal = dataclasses.replace(COPPER, charge_exchange=0.0)
    model = _model(
        electron_temperature=5.0,
        metal=metal,
        metal_ionization_rate=lambda te: 0.0 * te,
        metal_double_ionization_rate=lambda te: 0.0 * te,
    )
    state = _bombarded_state(model)
    budget = model.cathode_budget(state)
    flux = budget.ion_flux.sel(species=["Ar+", "M+", "M2+"]).values[:, 0]
    assert flux == pytest.approx([1e15 * 1e4, 2e14 * 5e3, 5e13 * 7e3])
    # the vacuum potential: the fall is the voltage, doubled for M2+
    energy = budget.ion_energy.values[:, 0]
    assert energy == pytest.approx([260.0, 260.0, 520.0])
    expected = (
        metal.sputter_yield("Ar+", 260.0) * flux[0]
        + metal.sputter_yield("M+", 260.0) * flux[1]
        + metal.sputter_yield("M+", 520.0) * flux[2]
    )
    sputtered = budget.sputtered.sum("species").values
    assert sputtered == pytest.approx(np.full(model.ny, expected))
    share = budget.self_sputtering_share.values
    assert share == pytest.approx((expected - budget.sputtered.values[0]) / expected)
    dt = 1e-10
    u0 = model._pack(state)
    u1, _ = model._euler(u0, state.potential, dt)
    gained = (u1["M"] - u0["M"]).sum(axis=0) * model.dx
    anode = model.metal_speed * u0["M"][-1]
    assert gained == pytest.approx(dt * (expected - anode), rel=1e-10)


def test_the_ion_energy_follows_the_fall_the_fluid_or_a_number() -> None:
    model = _model(metal=COPPER)
    state = _bombarded_state(model)
    # a potential hump of 20 V above the anode: the fall is 280 V
    state.potential = state.potential.copy()
    state.potential[model.nx // 2] = 20.0
    energy = model.cathode_budget(state).ion_energy
    assert energy.sel(species="Ar+").values == pytest.approx(280.0)
    assert energy.sel(species="M2+").values == pytest.approx(560.0)
    model.ion_energy = "kinetic"
    energy = model.cathode_budget(state).ion_energy
    kinetic = 0.5 * model.metal_mass * 7.0e3**2 / E
    assert energy.sel(species="M2+").values == pytest.approx(kinetic)
    assert energy.sel(species="Ar+").values == pytest.approx(
        0.5 * model.ion_mass * 1.0e8 / E
    )
    model.ion_energy = 100.0
    energy = model.cathode_budget(state).ion_energy
    assert energy.values[:, 0] == pytest.approx([100.0, 100.0, 200.0])
    # a pinned field instead of the electrodes: the cathode potential is extrapolated
    pinned = _model(metal=COPPER, field_bc=-1.0e4)
    fall = pinned._cathode_fall(pinned.initial_state(1e15).potential)
    assert fall == pytest.approx(1.0e4 * (pinned.length_x - pinned.dx / 2), rel=1e-9)


def test_the_cathode_current_is_the_sum_over_the_species_of_z_plus_gamma() -> None:
    """``j = sum_s (Z_s + gamma_s) e Gamma_s``, and the step emits ``sum gamma Gamma``."""
    model = _model(metal=COPPER)
    state = _bombarded_state(model)
    budget = model.cathode_budget(state)
    flux = budget.ion_flux.values
    charge = np.array([1.0, 1.0, 2.0])[:, None]
    gamma = np.array(
        [COPPER.argon_secondary, COPPER.self_secondary, COPPER.double_secondary]
    )[:, None]
    assert budget.secondary_yield.values == pytest.approx(gamma[:, 0])
    expected = E * ((charge + gamma) * flux).sum(axis=0)
    assert budget.current_density.values == pytest.approx(expected, rel=1e-12)
    parts = (budget.ion_current + budget.electron_current).sum("species")
    assert parts.values == pytest.approx(expected, rel=1e-12)
    assert budget.emitted.values == pytest.approx((gamma * flux).sum(axis=0))
    # Cu+ emits nothing, Cu2+ is the efficient emitter
    assert COPPER.self_secondary == 0.0 and COPPER.double_secondary > 0.15
    # the electrons the step lets into the domain at the cathode
    u = model._pack(state)
    gam_ix, _ = model._ion_fluxes(u["ni"], (u["vx"], u["vy"]))
    metal = {
        "M+": model._ion_fluxes(u["nm"], (u["vmx"], u["vmy"]))[0][0],
        "M2+": model._ion_fluxes(u["nm2"], (u["vm2x"], u["vm2y"]))[0][0],
    }
    emitted = model._cathode_flux(gam_ix[0], metal_ion_fluxes=metal)
    assert emitted == pytest.approx(budget.emitted.values)
    model.cathode_current_density = 0.03
    budget = model.cathode_budget(state)
    assert budget.emitted.values == pytest.approx(0.03 / E)
    assert np.all(budget.electron_current.values == 0.0)
    assert budget.current_density.values == pytest.approx(
        E * (charge * flux).sum(axis=0) + 0.03
    )


def test_ionization_and_charge_exchange_conserve_the_metal_and_the_charge() -> None:
    """Atoms plus ions change only by the boundary fluxes; Poisson holds after a stage."""
    model = _model(electron_temperature=None, metal=TITANIUM)
    state = _bombarded_state(model)
    dt = 1e-10
    u0 = model._pack(state)
    u1, phi = model._euler(u0, state.potential, dt)
    gam_mx, _ = model._ion_fluxes(u0["nm"], (u0["vmx"], u0["vmy"]))
    gam_m2x, _ = model._ion_fluxes(u0["nm2"], (u0["vm2x"], u0["vm2y"]))
    sputtered = model.cathode_budget(state).sputtered.sum("species").values
    inflow = (
        sputtered
        - model.metal_speed * u0["M"][-1]
        + gam_mx[0]
        - gam_mx[-1]
        + gam_m2x[0]
        - gam_m2x[-1]
    )
    total = (u1["M"] + u1["nm"] + u1["nm2"] - u0["M"] - u0["nm"] - u0["nm2"]).sum(
        axis=0
    ) * model.dx
    assert total == pytest.approx(dt * inflow, rel=1e-9)
    # the metal is ionized twice: the M2+ grow by more than their own transport
    doubled = (u1["nm2"] - u0["nm2"]).sum() * model.dx + dt * (
        gam_m2x[-1] - gam_m2x[0]
    ).sum()
    rate = model.metal_double_ionization_rate(4.0)
    ne = 1e15 + 2e14 + 2 * 5e13
    cells = model.nx * model.ny
    assert doubled == pytest.approx(dt * rate * 2e14 * ne * cells * model.dx, rel=1e-6)
    assert model.charge_residual(model._unpack(u1, phi, dt)) < 1e-6


def test_diffusing_atoms_keep_their_budget() -> None:
    """``metal_transport="diffusion"``: the atoms change by the sputtered flux and the anode."""
    model = _model(
        electron_temperature=5.0,
        metal=dataclasses.replace(TITANIUM, charge_exchange=0.0),
        metal_ionization_rate=lambda te: 0.0 * te,
        metal_transport="diffusion",
    )
    state = _bombarded_state(model)
    dt = 1e-10
    u0 = model._pack(state)
    u1, _ = model._euler(u0, state.potential, dt)
    sputtered = model.cathode_budget(state).sputtered.sum("species").values
    anode = 2.0 * model.metal_diffusion * u0["M"][-1] / model.dx
    gained = (u1["M"] - u0["M"]).sum() * model.dx
    assert gained == pytest.approx(dt * (sputtered - anode).sum(), rel=1e-10)
    # the beam's speed no longer limits the step, the diffusion might
    fast = dataclasses.replace(model, metal_speed=1e9)
    assert fast.suggest_timestep(state) == pytest.approx(model.suggest_timestep(state))


def test_the_emission_depends_on_the_ion_species() -> None:
    """``sum_s gamma_s Gamma_s``; a prescribed current overrides all."""
    metal = dataclasses.replace(
        TITANIUM, argon_secondary=0.1, self_secondary=0.02, double_secondary=0.3
    )
    model = _model(metal=metal)
    argon = np.full(model.ny, -1e18)
    ions = {"M+": np.full(model.ny, -5e17), "M2+": np.full(model.ny, -1e17)}
    assert model._cathode_flux(argon, metal_ion_fluxes=ions) == pytest.approx(
        0.1 * 1e18 + 0.02 * 5e17 + 0.3 * 1e17
    )
    model.metal = dataclasses.replace(metal, self_secondary=0.0, double_secondary=0.0)
    assert model._cathode_flux(argon, metal_ion_fluxes=ions) == pytest.approx(
        0.1 * 1e18
    )
    # Ar+ with the model's own yield instead of the material's
    model.metal = metal
    model.argon_emission = "model"
    assert model.argon_secondary_yield == model.secondary_emission
    assert model._cathode_flux(argon) == pytest.approx(model.secondary_emission * 1e18)
    model.argon_emission = "material"
    # the same through the state: a self-sputtering cathode emits by its metal ions alone
    state = _bombarded_state(model)
    state.ion_velocity[0] = 0.0
    budget = model.cathode_budget(state)
    assert budget.ion_flux.sel(species="Ar+").values == pytest.approx(0.0)
    metal_flux = budget.ion_flux.values[1:]
    assert budget.emitted.values == pytest.approx(
        0.02 * metal_flux[0] + 0.3 * metal_flux[1]
    )
    assert budget.self_sputtering_share.values == pytest.approx(1.0)


def test_the_metal_atoms_cost_the_electrons_ionization_excitation_and_elastic_energy() -> (
    None
):
    model = _model(electron_temperature=None, metal=TITANIUM)
    rate = lotz_ionization(TITANIUM.ionization_energy, TITANIUM.outer_electrons)
    assert model.metal_ionization_rate(4.0) == pytest.approx(rate(4.0))
    shape = (model.nx, model.ny)
    ne, te = np.full(shape, 1e15), np.full(shape, 4.0)
    gas = np.full(shape, model.gas_density)
    gas_c = model._collision_gas(gas, te)
    atoms, ions = np.full(shape, 1e18), np.full(shape, 1e16)
    base = model._energy_loss(ne, te, gas, gas_c)
    extra = model._energy_loss(ne, te, gas, gas_c, atoms) - base
    excitation = lumped_excitation_rate(TITANIUM.excitation_energy)(4.0)
    elastic = 3.0 * 9.1093837e-31 / model.metal_mass * 1e-13 * (4.0 - 0.025)
    expected = (
        (
            TITANIUM.ionization_energy * rate(4.0)
            + TITANIUM.excitation_energy * excitation
            + elastic
        )
        * 1e18
        * 1e15
    )
    assert extra == pytest.approx(expected, rel=1e-6)
    second = lotz_ionization(
        TITANIUM.second_ionization_energy, TITANIUM.ion_outer_electrons
    )
    doubled = model._energy_loss(ne, te, gas, gas_c, None, ions) - base
    assert doubled == pytest.approx(
        TITANIUM.second_ionization_energy * second(4.0) * 1e16 * 1e15
    )
    # and they slow the electrons: the collision frequency grows by k_eM M
    with_atoms = model._collision_gas(gas, te, atoms)
    assert with_atoms - gas_c == pytest.approx(1e18 * 1e-13 / model._k0)
    off = _model(metal=TITANIUM, metal_collisions=False)
    assert np.array_equal(
        off._collision_gas(gas, te, atoms), off._collision_gas(gas, te)
    )
    assert off._energy_loss(ne, te, gas, gas_c, atoms) == pytest.approx(
        base + TITANIUM.ionization_energy * rate(4.0) * 1e18 * 1e15
    )


def test_the_metal_ions_take_their_own_friction() -> None:
    state = _bombarded_state(_model(metal=TITANIUM))
    rates = []
    for friction in (None, 3e-15):
        model = _model(metal=TITANIUM, metal_ion_friction_rate=friction)
        u = model._pack(state)
        du, _ = model._rhs(u, state.potential, 1e-12)
        rates.append(du["vmx"])
    # friction decelerates the inflow: the stronger one adds (3e-15 - 1e-15) N |v|
    gas = state.gas_density
    assert rates[1] - rates[0] == pytest.approx(
        -(2e-15) * gas * state.metal_ion_velocity[0]
    )


def test_a_metal_needs_the_electrodes_valid_options_and_a_state_with_the_metal() -> (
    None
):
    with pytest.raises(ValueError, match="periodic_x"):
        DriftFluid2D(0.01, 0.01, 4, 4, field_bc=-1e3, periodic_x=True, metal=TITANIUM)
    with pytest.raises(ValueError, match="metal_speed"):
        _model(metal=TITANIUM, metal_speed=0.0)
    for settings, match in (
        ({"metal_transport": "jet"}, "metal_transport"),
        ({"metal_diffusion": 0.0}, "metal_diffusion"),
        ({"argon_emission": "x"}, "argon_emission"),
        ({"ion_energy": "hot"}, "ion_energy"),
        ({"ion_energy": -1.0}, "ion_energy"),
    ):
        with pytest.raises(ValueError, match=match):
            _model(metal=TITANIUM, **settings)
    bare = dataclasses.replace(
        TITANIUM, second_ionization_energy=None, excitation_energy=None
    )
    with pytest.raises(ValueError, match="second_ionization_energy"):
        _model(metal=bare)
    with pytest.raises(ValueError, match="excitation_energy"):
        _model(metal=bare, double_ionization=False)
    _model(metal=bare, double_ionization=False, metal_collisions=False)
    state = _model().initial_state(1e15)
    with pytest.raises(ValueError, match="state with the metal"):
        _model(metal=TITANIUM).step(state, 1e-10)
    plain = _model().cathode_budget(state)
    assert list(plain.species.values) == ["Ar+"]


def test_a_short_1d_run_with_a_copper_target_sputters_and_ionizes_the_metal_twice() -> (
    None
):
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
        metal=COPPER,
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
    assert final.metal_ion2_density.max() > 0.0
    budget = model.cathode_budget(final)
    assert np.all(budget.sputtered.sum("species").values > 0.0)
    share = budget.self_sputtering_share.values
    assert np.all((share >= 0.0) & (share < 1.0))
    assert model.charge_residual(final) < 1e-5
    data = final.to_dataset()
    assert {"metal_density", "metal_ion_density", "metal_ion2_density"} <= set(data)
    copy = final.copy()
    copy.metal_ion2_density[:] = 0.0
    assert final.metal_ion2_density.max() > 0.0
