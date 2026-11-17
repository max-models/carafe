"""Tests of the drift fluid coupled to the gas dynamics."""

import numpy as np
import pytest

from wellbalanced import DriftFluid2D
from wellbalanced.driftfluid2d import exponential_field
from wellbalanced.gas_coupling import GasCoupledDriftFluid
from wellbalanced.gasdynamics import ARGON_MASS, Boundary, GasDynamics

BOLTZMANN = 1.380649e-23
GAP, PERIOD, PRESSURE = 0.02, 0.08, 0.6


def plasma(nx=12, ny=4, **options):
    """A coarse version of Boeuf's DC magnetron plane."""
    return DriftFluid2D(
        length_x=GAP,
        length_y=PERIOD,
        nx=nx,
        ny=ny,
        voltage=260.0,
        magnetic_field=exponential_field(0.12, 0.3 * GAP),
        gas_density=PRESSURE / (BOLTZMANN * 300.0),
        electron_temperature=None,
        cathode_current_density=0.03,
        **options,
    )


def gas(nx=12, ny=1, right=None, **options):
    """Argon at the plasma's pressure, a wall at the target."""
    model = GasDynamics(
        GAP,
        nx,
        cells_y=ny,
        width=PERIOD if ny > 1 else 1.0,
        boundaries={
            "left": Boundary.wall(300.0),
            "right": right or Boundary.wall(300.0),
        },
        **options,
    )
    model.set_uniform(PRESSURE, 300.0)
    return model


def _outflow_left():
    model = GasDynamics(
        GAP, 12, boundaries={"left": Boundary.outflow(), "right": Boundary.outflow()}
    )
    model.set_uniform(PRESSURE, 300.0)
    return model


def test_grids_must_match():
    with pytest.raises(ValueError, match="length_x and nx"):
        GasCoupledDriftFluid(plasma(), gas(nx=10), window=1e-7)
    with pytest.raises(ValueError, match="1 cell along y"):
        GasCoupledDriftFluid(plasma(ny=4), gas(ny=3), window=1e-7)
    with pytest.raises(ValueError, match="window"):
        GasCoupledDriftFluid(plasma(), gas(), window=0.0)
    with pytest.raises(ValueError, match="wall at the target"):
        GasCoupledDriftFluid(plasma(), _outflow_left(), window=1e-7)


def test_refill_handed_to_the_gas_model():
    model = plasma(refill_rate=2e4)
    GasCoupledDriftFluid(model, gas(), window=1e-7)
    assert model.refill_rate == 0.0


def test_the_gas_loses_what_the_plasma_ionizes():
    """Closed gas, no return: the argon lost is the ionization recorded, to rounding."""
    coupled = GasCoupledDriftFluid(plasma(), gas(), window=5e-8, return_ions=False)
    before = coupled.gas.totals()["gas"] / ARGON_MASS
    state = coupled.plasma.initial_state(1.0e16, electron_temperature=5.0)
    state, _ = coupled.run(state, 1.5e-7)
    after = coupled.gas.totals()["gas"] / ARGON_MASS
    ionized = np.sum(coupled.record.ionized)
    assert ionized > 0.0
    assert before - after == pytest.approx(ionized, rel=1e-9)
    assert len(coupled.record.time) == 3
    assert np.allclose(state.gas_density, coupled.gas_number())


def test_returned_ions_go_back_into_the_gas():
    """With the return the gas loses the ionization less what the target sends back."""
    coupled = GasCoupledDriftFluid(plasma(), gas(), window=5e-8, ion_exchange=True)
    before = coupled.gas.totals()["gas"] / ARGON_MASS
    state = coupled.plasma.initial_state(1.0e16, electron_temperature=5.0)
    coupled.run(state, 1.0e-7)
    after = coupled.gas.totals()["gas"] / ARGON_MASS
    record = coupled.record.as_arrays()
    assert record["returned"].sum() > 0.0
    assert before - after == pytest.approx(
        record["ionized"].sum() - record["returned"].sum(), rel=1e-6
    )


def test_two_dimensional_gas():
    coupled = GasCoupledDriftFluid(plasma(ny=4), gas(ny=4), window=5e-8)
    state = coupled.plasma.initial_state(1.0e16, electron_temperature=5.0)
    state, gas_state = coupled.run(state, 5e-8)
    assert gas_state.gas_number.shape == (12, 4)
    assert np.allclose(state.gas_density, gas_state.gas_number)
    assert gas_state.gas_number.min() < PRESSURE / (BOLTZMANN * 300.0)
