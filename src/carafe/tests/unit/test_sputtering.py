"""The sputter and secondary electron yields of `carafe.sputtering` and the materials."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from carafe.magnetron import MATERIALS, YIELD_ANCHOR_ENERGY, TargetMaterial
from carafe.sputtering import (
    BohdanskyYield,
    ConstantYield,
    PowerLawYield,
    as_yield,
    bohdansky_threshold,
    krc_nuclear_stopping,
    potential_emission_yield,
    reduced_energy,
)

COPPER = MATERIALS["Cu"]


def test_the_reduced_energy_and_the_krc_stopping() -> None:
    # Ar on Cu: eps = E M2/(M1+M2) a_L/(Z1 Z2 e^2) with a_L = 0.4685 A/sqrt(Z1^2/3 + Z2^2/3)
    a_l = 0.4685 / np.sqrt(18 ** (2 / 3) + 29 ** (2 / 3))
    expected = 1000.0 * 63.546 / (39.948 + 63.546) * a_l / (18 * 29 * 14.4)
    assert reduced_energy(1000.0, 18, 39.948, 29, 63.546) == pytest.approx(
        expected, rel=1e-4
    )
    # the Kr-C stopping peaks near eps = 0.3 and falls like ln(eps)/(2 eps) beyond
    eps = np.logspace(-4, 2, 200)
    stopping = krc_nuclear_stopping(eps)
    assert 0.1 < eps[np.argmax(stopping)] < 1.0
    assert stopping.max() == pytest.approx(0.39, abs=0.01)
    assert krc_nuclear_stopping(100.0) == pytest.approx(
        0.5 * np.log(1 + 122.88) / (100 + 1.728 + 0.008 * 100**0.1504)
    )


def test_bohdanskys_threshold_for_light_and_heavy_ions() -> None:
    # heavy: 8 U_s (M1/M2)^0.4; self-sputtering 8 U_s
    assert bohdansky_threshold(63.546, 63.546, 3.49) == pytest.approx(8 * 3.49)
    assert bohdansky_threshold(39.948, 63.546, 3.49) == pytest.approx(
        8 * 3.49 * (39.948 / 63.546) ** 0.4
    )
    # light (M1/M2 <= 0.2): U_s / (g (1 - g))
    g = 4 * 4.0 * 63.546 / (4.0 + 63.546) ** 2
    assert bohdansky_threshold(4.0, 63.546, 3.49) == pytest.approx(3.49 / (g * (1 - g)))


def test_the_bohdansky_yield_vanishes_below_the_threshold_and_rises_above() -> None:
    fit = BohdanskyYield.anchored(18, 39.948, 29, 63.546, 3.49, 500.0, 2.0)
    assert fit(500.0) == pytest.approx(2.0)
    assert fit.threshold == pytest.approx(bohdansky_threshold(39.948, 63.546, 3.49))
    energies = np.array([0.0, 5.0, fit.threshold, 50.0, 100.0, 300.0, 1000.0])
    values = fit(energies)
    assert isinstance(values, np.ndarray) and values.shape == energies.shape
    assert np.all(values[:3] == 0.0)
    assert np.all(np.diff(values[2:]) > 0.0)
    # sub-linear above a few hundred eV, as the measured yields
    assert fit(1000.0) / fit(500.0) < 1.6
    assert isinstance(fit(400.0), float)
    # an explicit threshold and the scaling with Q
    pinned = dataclasses.replace(fit, threshold=40.0, q=2.0 * fit.q)
    assert pinned(39.0) == 0.0
    assert pinned(800.0) > 2.0 * fit(800.0) * 0.9
    with pytest.raises(ValueError, match="masses"):
        BohdanskyYield(18, 0.0, 29, 63.546, 3.49, 1.0)
    with pytest.raises(ValueError, match="U_s"):
        BohdanskyYield(18, 39.948, 29, 63.546, 0.0, 1.0)


def test_the_power_law_and_the_constant_yields() -> None:
    fit = PowerLawYield(0.07, 0.55)
    assert fit(400.0) == pytest.approx(0.07 * 400.0**0.55)
    assert fit(0.0) == 0.0 and fit(-5.0) == 0.0
    assert PowerLawYield(0.07, 0.55, threshold=30.0)(29.0) == 0.0
    assert fit(np.array([100.0, 400.0])).shape == (2,)
    with pytest.raises(ValueError, match="sputter yield"):
        PowerLawYield(0.1, 0.0)
    constant = ConstantYield(0.4)
    assert constant(300.0) == 0.4
    assert np.all(constant(np.array([1.0, 2.0])) == 0.4)
    with pytest.raises(ValueError, match="sputter yield"):
        ConstantYield(-1.0)
    assert as_yield(0.3)(10.0) == 0.3 and as_yield(fit) is fit


def test_potential_emission_needs_twice_the_work_function() -> None:
    # Baragiola: 0.032 (0.78 E_iz - 2 W), zero below
    assert potential_emission_yield(15.76, 4.65) == pytest.approx(
        0.032 * (0.78 * 15.76 - 9.3)
    )
    assert potential_emission_yield(7.726, 4.65) == 0.0
    values = potential_emission_yield(np.array([7.0, 20.0]), 4.65)
    assert values[0] == 0.0 and values[1] > 0.1


def test_the_materials_take_their_yields_from_the_formulas() -> None:
    for material in MATERIALS.values():
        anchor = YIELD_ANCHOR_ENERGY
        assert material.sputter_yield("Ar+", anchor) > 0.3
        assert material.sputter_yield("M+", anchor) > 0.3
        # M2+ sputter as the metal atom at their energy
        assert material.sputter_yield("M2+", 2 * anchor) == pytest.approx(
            material.sputter_yield("M+", 2 * anchor)
        )
        assert material.sputter_yield("Ar+", 10.0) == 0.0
        # a singly charged metal ion emits nothing from its own target, a doubly charged one does
        assert material.secondary_yield("M+") == 0.0
        assert material.secondary_yield("M2+") > 0.0
        assert 0.05 < material.secondary_yield("Ar+") < 0.15
        assert material.second_ionization_energy > material.ionization_energy
    assert COPPER.sputter_yield("Ar+", 500.0) == pytest.approx(2.0)
    assert COPPER.secondary_yield("M2+") == pytest.approx(
        0.032 * (0.78 * 20.29 - 2 * 4.65)
    )
    with pytest.raises(ValueError, match="unknown ion"):
        COPPER.sputter_yield("He+", 100.0)
    with pytest.raises(ValueError, match="unknown ion"):
        COPPER.secondary_yield("He+")
    with pytest.raises(ValueError, match="second ionization"):
        dataclasses.replace(COPPER, second_ionization_energy=1.0)
    with pytest.raises(ValueError, match="secondary"):
        dataclasses.replace(COPPER, double_secondary=-0.1)
    # numbers still work, as yields independent of the energy
    plain = TargetMaterial("X", 10.0, 7.0, 1, 0.5, 0.2, 0.1, 0.0, 1e-15)
    assert plain.sputter_yield("Ar+", 123.0) == 0.5
    assert plain.second_ionization_energy is None
