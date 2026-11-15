"""The nonlocal eigenproblem: its limits, convergence, and Boeuf's profile."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from wellbalanced.dispersion import (
    BOEUF_FIGURE_2,
    ColdIons,
    Eigenmodes,
    Equilibrium,
    LocalDispersion,
    NonlocalDispersion,
    Plasma,
)

LY = 0.08  # Boeuf's 8 cm box
M5 = 2 * np.pi * 5 / LY  # his 1.6 cm


def _boeuf(points: int, **kwargs) -> Equilibrium:
    return Equilibrium.boeuf(np.linspace(3e-3, 0.02, points), **kwargs)


def _slab(length: float, points: int, density_length: float = -3e-3) -> Equilibrium:
    """A uniform-gradient slab: the coefficients of the local relation are constant."""
    x = np.linspace(0.0, length, points)
    n = 1e16 * np.exp((x - length / 2) / density_length)
    return Equilibrium.from_profiles(
        x, n, -1e4, 0.1, 3.0, density_gradient=n / density_length
    )


def test_boeuf_table_checks() -> None:
    # the integral of E is the applied voltage, e times the integral of S the current
    x, _, _, e_field, _, source = BOEUF_FIGURE_2.T
    voltage = np.trapezoid(e_field * 1e4, x * 1e-2)
    current = 1.602176634e-19 * np.trapezoid(source * 1e21, x * 1e-2)
    assert voltage == pytest.approx(-260.6, abs=0.1)
    assert current == pytest.approx(5.18, abs=0.02)


def test_boeuf_equilibrium_at_the_anode_layer() -> None:
    eq = Equilibrium.boeuf(np.linspace(3e-3, 0.02, 171))  # a point at 18 mm
    j = np.argmin(abs(eq.x - 0.018))
    assert eq.x[j] == pytest.approx(0.018)
    assert eq.density[j] == pytest.approx(0.78e15, rel=1e-6)
    assert eq.b_field[j] == pytest.approx(0.12 * np.exp(-3), rel=1e-6)
    assert eq.electron_temperature[j] == pytest.approx(11.4, rel=0.01)
    assert eq.density_length[j] == pytest.approx(-2.2e-3, rel=0.1)
    assert eq.ionization_frequency[j] == pytest.approx(1.56e6, rel=0.02)
    assert eq.ion_flow[j] == pytest.approx(-1.7e3, rel=0.05)
    assert eq.gyroradius[j] == pytest.approx(1.35e-3, rel=0.01)
    spline = Equilibrium.boeuf(eq.x, smoothing="spline")
    assert spline.density[j] == pytest.approx(eq.density[j], rel=0.02)
    ds = eq.to_dataset()
    assert ds["density"].attrs["units"] == "m^-3"
    assert set(ds.data_vars) >= {"debye_length", "ion_flow", "e_field"}


def test_equilibrium_inputs_are_checked() -> None:
    with pytest.raises(ValueError, match="smoothing"):
        Equilibrium.boeuf(smoothing="linear")
    with pytest.raises(ValueError, match="uniform"):
        Equilibrium.from_profiles(np.array([0, 1, 2, 4, 5.0]), 1e16, -1e4, 0.1, 3.0)
    with pytest.raises(ValueError, match="1D grid"):
        Equilibrium.from_profiles(np.arange(3.0), 1e16, -1e4, 0.1, 3.0)


def test_from_profiles_ion_flow_and_scaling() -> None:
    x = np.linspace(0.0, 0.01, 101)
    eq = Equilibrium.from_profiles(x, 1e16, -1e4, 0.05, 3.0, ionization_frequency=1e5)
    # uniform n and nu_iz: v_0 = -nu_iz (L - x)
    np.testing.assert_allclose(eq.ion_flow, -1e5 * (0.01 - x), atol=1e-9)
    np.testing.assert_allclose(eq.b_field_gradient, 0.0, atol=1e-12)
    scaled = eq.scaled(
        density=10.0, b_field=2.0, b_floor=0.01, electron_collision_frequency=3.0
    )
    np.testing.assert_allclose(scaled.density, 1e17)
    np.testing.assert_allclose(scaled.b_field, 0.11)
    np.testing.assert_allclose(scaled.ion_flow, eq.ion_flow)
    np.testing.assert_allclose(scaled.debye_length, eq.debye_length / np.sqrt(10))


def test_model_state() -> None:
    x = np.linspace(5e-4, 0.015, 120)
    eq = Equilibrium.model_state(x, 1e19, 0.05, 0.01, -1e4, 3.0, 5e-3)
    assert eq.density[0] == pytest.approx(1e19, rel=1e-3)
    j = np.argmin(abs(x - 0.01))
    assert eq.density_length[j] == pytest.approx(-5e-3, rel=0.02)
    assert eq.e_field[-1] == 0.0
    np.testing.assert_allclose(eq.ionization_frequency, 1e5)
    assert np.all(eq.ion_flow <= 0.0)


def test_uniform_plasma_has_no_growing_mode() -> None:
    eq = Equilibrium.from_profiles(np.linspace(0.0, 0.015, 60), 1e16, -1e4, 0.1, 3.0)
    assert len(NonlocalDispersion(eq).solve(500.0).drift()) == 0


def test_local_limit_on_a_uniform_gradient_slab() -> None:
    # with a uniform L_n, E, B and T_e the only nonlocal coefficient left is n_0 in
    # Poisson's equation; the fastest modes then satisfy the local relation at the
    # eigenfunction's own radial wavenumber
    eq = _slab(0.006, 80)
    ky = 2 * np.pi / 0.016
    modes = NonlocalDispersion(eq, ion_flow=False).solve(ky).drift()
    assert len(modes) > 10  # a family of radial modes, not one
    plasma = Plasma(1e16, -1e4, 0.1, 3.0, -3e-3)
    for j in range(3):
        kx = modes.radial_wavenumber[j]
        local = LocalDispersion(plasma, ColdIons()).roots(kx, ky)
        assert abs(modes.omega[j] - local.omega[0]) < 0.1 * abs(local.omega[0])


def test_grid_convergence_is_first_order() -> None:
    growth = [
        NonlocalDispersion(_boeuf(points)).solve(M5).drift().growth_rate
        for points in (60, 120, 240)
    ]
    order = np.log2((growth[1] - growth[0]) / (growth[2] - growth[1]))
    assert 0.6 < order < 1.6  # first-order upwinding of the flows
    assert abs(growth[2] - growth[1]) < 0.03 * growth[2]


def test_boeuf_regression() -> None:
    # the drift fluid at N = 200: one growing branch peaked at 18 mm, 1.6 cm at +7.2 km/s
    eq = _boeuf(200)
    modes = NonlocalDispersion(eq).solve(M5).drift()
    assert len(modes) == 1
    assert modes.growth_rate == pytest.approx(6.27e6, rel=0.01)
    assert modes.phase_speed == pytest.approx(7.16e3, rel=0.01)
    assert modes.peak_position[0] == pytest.approx(0.018, abs=3e-4)
    assert modes.radial_wavenumber[0] == pytest.approx(620, rel=0.05)
    # the vacuum field's compressibility nearly kills it, the gyro-average does
    curl_free = NonlocalDispersion(eq, curl_free_field=True).solve(M5).drift()
    assert curl_free.growth_rate == pytest.approx(2.3e5, rel=0.05)
    assert len(NonlocalDispersion(eq, gyroaverage=True).solve(M5).drift()) == 0


def test_boeuf_with_electron_inertia() -> None:
    eq = _boeuf(200)
    model = NonlocalDispersion(eq, inertia=True)
    assert model.unknowns == ("a_e", "a_i", "v_x", "v_y", "u_x", "u_y")
    modes = model.solve(M5).drift()
    assert modes.growth_rate == pytest.approx(2.32e6, rel=0.01)
    assert modes.phase_speed == pytest.approx(4.9e3, rel=0.02)


def test_collisions_set_the_phase_speed() -> None:
    eq = _boeuf(100)
    with_nu = NonlocalDispersion(eq).solve(M5).drift()
    without = NonlocalDispersion(eq, electron_collisions=False).solve(M5).drift()
    assert without.phase_speed < 0.2 * with_nu.phase_speed
    assert without.growth_rate > with_nu.growth_rate
    # the ion flow and the ionisation coupling add growth, not speed
    for option in ("ion_flow", "ionization"):
        off = NonlocalDispersion(eq, **{option: False}).solve(M5).drift()
        assert 0.7 < off.growth_rate / with_nu.growth_rate < 0.95
    damped = NonlocalDispersion(eq, ion_collision_frequency=3.6e5).solve(M5).drift()
    assert damped.growth_rate < with_nu.growth_rate


def test_gyroaverage_reduces_to_the_fluid() -> None:
    eq = _boeuf(100)
    fluid = NonlocalDispersion(eq).solve(M5).drift()
    small = NonlocalDispersion(eq, gyroaverage=True, gyroradius_scale=1e-4)
    gyro = small.solve(M5).drift()
    assert gyro.omega[0] == pytest.approx(fluid.omega[0], rel=1e-6)


def test_quasineutral_is_the_dense_limit_of_poisson() -> None:
    eq = _boeuf(60).scaled(density=1e4)
    poisson = NonlocalDispersion(eq).solve(M5).drift()
    neutral_model = NonlocalDispersion(eq, quasineutral=True)
    a, b = neutral_model.matrices(M5)
    assert b is not None and a.shape == (300, 300)
    neutral = neutral_model.solve(M5).drift()
    assert neutral.omega[0] == pytest.approx(poisson.omega[0], rel=2e-3)
    # the eigenfunctions are normalised to e phi / T_e = 1 at the peak
    peak = np.argmax(abs(neutral.potential[0]))
    assert neutral.potential[0, peak] == pytest.approx(1.0)
    a_poisson, b_poisson = NonlocalDispersion(eq).matrices(M5)
    assert b_poisson is None and a_poisson.shape == (240, 240)


def test_eigenmodes_selection_and_dataset() -> None:
    modes = NonlocalDispersion(_boeuf(60)).solve(M5)
    assert isinstance(modes, Eigenmodes)
    assert np.all(np.diff(modes.omega.imag) <= 0.0)  # fastest first
    drift = modes.drift(omega_max=1e9, roughness=0.5)
    assert 0 < len(drift) < len(modes)
    assert np.all(drift.roughness < 0.5)
    assert len(modes.drift(omega_max=1.0)) == 0
    assert np.isnan(modes.drift(omega_max=1.0).growth_rate)
    assert np.isnan(modes.drift(omega_max=1.0).phase_speed)
    ds = drift.to_dataset()
    assert ds.sizes == {"mode": len(drift), "x": 60}
    assert ds["growth_rate"].attrs["units"] == "1/s"


def test_scan_follows_the_drift_branch() -> None:
    eq = _boeuf(60)
    ky = 2 * np.pi * np.array([3, 5, 40]) / LY
    scan = NonlocalDispersion(eq).scan(ky)
    assert isinstance(scan, xr.Dataset)
    assert np.all(scan["growth_rate"][:2] > 0)
    assert scan["modes"][2] == 0 and np.isnan(scan["growth_rate"][2])  # k lambda_D >> 1
    np.testing.assert_allclose(scan["wavelength"], LY / np.array([3, 5, 40]))
    np.testing.assert_allclose(scan["phase_speed"][:2], scan["frequency"][:2] / ky[:2])
    assert 0.016 < float(scan["peak_position"][1]) < 0.019
    assert 0.1 < float(scan["ky_gyroradius"][1]) < 1.0
    assert scan["potential"].dims == ("ky", "x")
    assert float(abs(scan["potential"][1]).max()) == pytest.approx(1.0)
