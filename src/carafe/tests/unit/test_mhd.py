"""Ideal MHD on a patchwork hierarchy: Riemann solver, convergence, div B, refinement."""

import numpy as np
import pytest
from patchwork import AMRHierarchy

from carafe.mhd import MHD2D, _outflow_nodes, hlld_flux

GAMMA = 5.0 / 3.0


def _state(rho, vn, vt1, vt2, p, bt1, bt2, size=1):
    return np.array([[rho], [vn], [vt1], [vt2], [p], [bt1], [bt2]], float) * np.ones(
        size
    )


def _physical_flux(state, bn, gamma=GAMMA):
    rho, vn, vt1, vt2, p, bt1, bt2 = state
    b2 = bn**2 + bt1**2 + bt2**2
    pt = p + 0.5 * b2
    energy = p / (gamma - 1) + 0.5 * rho * (vn**2 + vt1**2 + vt2**2) + 0.5 * b2
    vb = vn * bn + vt1 * bt1 + vt2 * bt2
    return np.array(
        [
            rho * vn,
            rho * vn**2 + pt - bn**2,
            rho * vn * vt1 - bn * bt1,
            rho * vn * vt2 - bn * bt2,
            (energy + pt) * vn - bn * vb,
            bt1 * vn - bn * vt1,
            bt2 * vn - bn * vt2,
        ]
    )


# The Riemann solver ----------------------------------------------------------------#
@pytest.mark.parametrize("bn", [0.0, 0.3, -1.2])
def test_hlld_is_consistent(bn):
    state = _state(1.3, 0.4, -0.2, 0.7, 0.9, 0.5, -0.8)
    flux = hlld_flux(state, state, np.array([bn]), GAMMA)
    np.testing.assert_allclose(flux, _physical_flux(state, bn), atol=1e-13)


def test_hlld_upwinds_supersonic_flow():
    left = _state(1.0, 20.0, 0.1, 0.0, 1.0, 0.2, 0.1)
    right = _state(0.5, 19.0, -0.3, 0.2, 0.4, -0.1, 0.0)
    flux = hlld_flux(left, right, np.array([0.5]), GAMMA)
    np.testing.assert_allclose(flux, _physical_flux(left, 0.5), atol=1e-12)
    left[1], right[1] = -19.0, -20.0
    flux = hlld_flux(left, right, np.array([0.5]), GAMMA)
    np.testing.assert_allclose(flux, _physical_flux(right, 0.5), atol=1e-12)


def test_hlld_mirror_symmetry():
    """Mirroring the problem (x -> -x) negates the normal flux of mass and energy."""
    left = _state(1.0, 0.3, 0.1, -0.2, 1.0, 0.6, 0.2)
    right = _state(0.3, -0.1, 0.4, 0.1, 0.2, -0.5, 0.3)
    bn = np.array([0.7])
    flux = hlld_flux(left, right, bn, GAMMA)
    mirror = np.array([[1], [-1], [1], [1], [1], [1], [1]])
    mirrored = hlld_flux(right * mirror, left * mirror, -bn, GAMMA)
    # the normal momentum flux is even under the mirror, every other flux odd
    expected = flux * np.array([[-1], [1], [-1], [-1], [-1], [-1], [-1]])
    np.testing.assert_allclose(mirrored, expected, atol=1e-13)


def test_hlld_resolves_an_isolated_contact():
    """A contact (density jump only, B_n != 0) is a stationary solution: no mass flux."""
    left = _state(1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0)
    right = _state(0.2, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0)
    flux = hlld_flux(left, right, np.array([0.8]), GAMMA)
    assert abs(flux[0, 0]) < 1e-14
    np.testing.assert_allclose(flux[1, 0], 1.0 + 0.5 * 0.25 - 0.5 * 0.64)


# Whole runs --------------------------------------------------------------------------#
def test_alfven_wave_converges_at_second_order():
    errors = []
    for cells in (8, 16):
        model, exact = MHD2D.alfven_wave(cells, limiter="none")
        model.run(model.period)
        by = model.dataset()["magnetic_field_y"].values
        errors.append(np.mean(np.abs(by - exact["magnetic_field_y"].values)))
        assert model.divergence() < 1e-14
    assert np.log2(errors[0] / errors[1]) > 1.85


def test_orszag_tang_keeps_div_b_zero_and_conserves():
    model = MHD2D.orszag_tang(32)
    before = model.totals()
    model.run(0.1)
    after = model.totals()
    assert model.divergence() < 1e-14
    for name in before:
        assert abs(after[name] - before[name]) < 1e-13
    assert float(model.dataset().pressure.min()) > 0.0


def _brio_wu(axis, cells=100):
    if axis == 0:
        hierarchy = AMRHierarchy(
            (cells, 2), ((0.0, 1.0), (0.0, 2.0 / cells)), periodic=(False, True)
        )
    else:
        hierarchy = AMRHierarchy(
            (2, cells), ((0.0, 2.0 / cells), (0.0, 1.0)), periodic=(True, False)
        )
    mean = (0.75, 0.0) if axis == 0 else (0.0, 0.75)
    model = MHD2D(hierarchy, gamma=2.0, mean_field=mean)

    def s(x, y):
        return x if axis == 0 else y

    # the transverse field +1 left, -1 right, from |s - 1/2|; B_y = -dA/dx, B_x = dA/dy
    sign = 1.0 if axis == 0 else -1.0
    model.set_state(
        lambda x, y: np.where(s(x, y) < 0.5, 1.0, 0.125),
        (0.0, 0.0, 0.0),
        lambda x, y: np.where(s(x, y) < 0.5, 1.0, 0.1),
        vector_potential=lambda x, y: sign * np.abs(s(x, y) - 0.5),
    )
    model.run(0.1)
    return model


def test_brio_wu_along_x_and_y_agree():
    along_x = _brio_wu(0)
    along_y = _brio_wu(1)
    rx = along_x.dataset().density.values[:, 0]
    ry = along_y.dataset().density.values[0, :]
    np.testing.assert_allclose(rx, ry, atol=1e-12)
    # the transverse field keeps its far values; the solution is bounded and positive
    by = along_x.dataset().magnetic_field_y.values[:, 0]
    assert by[0] == pytest.approx(1.0) and by[-1] == pytest.approx(-1.0)
    assert 0.11 < rx.min() and rx.max() <= 1.0 + 1e-12
    # the compound wave and the slow shock leave a plateau of about 0.68 at x ~ 0.5
    assert 0.6 < rx[50] < 0.75
    assert along_x.divergence() < 1e-14


def test_uniform_state_stays_uniform_with_outflow_faces():
    hierarchy = AMRHierarchy((8, 6), ((0.0, 1.0), (0.0, 1.0)))
    model = MHD2D(hierarchy, mean_field=(0.3, -0.2), limiter="mc")
    model.set_state(1.0, (0.2, -0.1, 0.05), 0.5, bz=0.4)
    model.run(0.2)
    data = model.dataset()
    np.testing.assert_allclose(data.density, 1.0, atol=1e-13)
    np.testing.assert_allclose(data.magnetic_field_x, 0.3, atol=1e-13)
    np.testing.assert_allclose(data.magnetic_field_z, 0.4, atol=1e-13)
    np.testing.assert_allclose(data.pressure, 0.5, atol=1e-12)


def test_outflow_nodes_continue_the_gradient():
    array = np.zeros((7, 3))
    array[2:5] = np.arange(3)[:, None] * 2.0 + 1.0
    _outflow_nodes(array, 0, True, True)
    np.testing.assert_allclose(array[:, 0], [-3, -1, 1, 3, 5, 7, 9])


def _refined(cells=16):
    hierarchy = AMRHierarchy(
        (2 * cells, cells), ((-1.0, 1.0), (-0.5, 0.5)), periodic=(True, True)
    )
    hierarchy.add_patch(
        1, ((cells - cells // 2, cells + cells // 2), (cells // 2, 3 * cells // 2))
    )
    model = MHD2D(hierarchy, limiter="minmod")
    model.set_state(
        lambda x, y: 1.0 + 0.5 * np.exp(-((x + 0.3) ** 2 + y**2) / 0.02),
        (2.0, 1.0, 0.3),
        lambda x, y: 1.0 + 0.2 * np.cos(np.pi * x),
        vector_potential=lambda x, y: 0.1 * np.maximum(0.3 - np.hypot(x, y), 0.0),
        bz=lambda x, y: 0.1 * np.sin(np.pi * x),
    )
    return model


def test_refined_patch_conserves_and_keeps_div_b_zero():
    model = _refined()
    before = model.totals()
    model.run(0.2)
    after = model.totals()
    for name in before:
        assert abs(after[name] - before[name]) < 1e-12
    assert model.divergence() < 1e-14
    assert model.composite_divergence() < 1e-14
    # the fine patch and the coarse level carry the same solution where both exist
    fine = model.dataset(1)
    assert np.isnan(fine.density.values).any() and np.nanmin(fine.density) > 0.0


def test_refined_patch_follows_the_uniform_solution():
    """A refined patch changes a smooth solution only at the truncation error."""
    refined = _refined()
    refined.run(0.1)
    hierarchy = AMRHierarchy(
        (32, 16), ((-1.0, 1.0), (-0.5, 0.5)), periodic=(True, True)
    )
    uniform = MHD2D(hierarchy, limiter="minmod")
    uniform.set_state(
        lambda x, y: 1.0 + 0.5 * np.exp(-((x + 0.3) ** 2 + y**2) / 0.02),
        (2.0, 1.0, 0.3),
        lambda x, y: 1.0 + 0.2 * np.cos(np.pi * x),
        vector_potential=lambda x, y: 0.1 * np.maximum(0.3 - np.hypot(x, y), 0.0),
        bz=lambda x, y: 0.1 * np.sin(np.pi * x),
    )
    uniform.run(0.1)
    difference = refined.dataset().density - uniform.dataset().density
    assert float(np.abs(difference).max()) < 0.05


# Errors ------------------------------------------------------------------------------#
def test_rejects_unsupported_hierarchies_and_options():
    with pytest.raises(ValueError, match="2D"):
        MHD2D(AMRHierarchy((8,)))
    with pytest.raises(NotImplementedError, match="Cartesian"):
        MHD2D(AMRHierarchy((4, 4), ((0.0, 1.0), (0.0, 1.0)), geometry="cylindrical"))
    with pytest.raises(ValueError, match="limiter"):
        MHD2D(AMRHierarchy((4, 4)), limiter="superbee")
    with pytest.raises(ValueError, match="cfl"):
        MHD2D(AMRHierarchy((4, 4)), cfl=1.5)
    with pytest.raises(ValueError, match="gamma"):
        MHD2D(AMRHierarchy((4, 4)), gamma=1.0)


def test_needs_a_positive_state_first():
    model = MHD2D(AMRHierarchy((4, 4), periodic=(True, True)))
    with pytest.raises(RuntimeError, match="set_state"):
        model.step()
    with pytest.raises(ValueError, match="positive"):
        model.set_state(1.0, (0.0, 0.0, 0.0), -1.0)
    with pytest.raises(ValueError, match="three"):
        model.set_state(1.0, (0.0, 0.0), 1.0)
    model.set_state(1.0, (0.0, 0.0, 0.0), 1.0)
    with pytest.raises(ValueError, match="duration"):
        model.run(model.period)


def test_reports_a_negative_pressure():
    model = MHD2D(AMRHierarchy((8, 8), periodic=(True, True)))
    model.set_state(1.0, (0.0, 0.0, 0.0), 1e-3)
    for patch in model.hierarchy.patches:
        patch.field(model.names["energy"]).local[3, 3] = -1.0
    with pytest.raises(FloatingPointError, match="pressure"):
        model.step(1e-3)


def test_falls_back_to_first_order_where_slopes_would_go_negative():
    """Unlimited slopes at a strong jump would reconstruct a negative density."""
    hierarchy = AMRHierarchy(
        (32, 2), ((0.0, 1.0), (0.0, 1 / 16)), periodic=(False, True)
    )
    model = MHD2D(hierarchy, limiter="none")
    model.set_state(
        lambda x, y: np.where(x < 0.5, 1.0, 0.05),
        (0.0, 0.0, 0.0),
        lambda x, y: np.where(x < 0.5, 1.0, 0.05),
    )
    model.step(1e-3)
    assert float(model.dataset().density.min()) > 0.0


def test_interior_patch_of_a_bounded_domain():
    hierarchy = AMRHierarchy((8, 8), ((0.0, 1.0), (0.0, 1.0)))
    hierarchy.add_patch(1, ((4, 12), (4, 12)))
    model = MHD2D(hierarchy)
    model.set_state(
        1.0, (0.1, 0.0, 0.0), 1.0, vector_potential=lambda x, y: 0.1 * x * y
    )
    before = model.totals()
    model.step()
    assert model.time > 0.0 and model.steps == 1
    assert model.composite_divergence() < 1e-14
    assert abs(model.totals()["density"] - before["density"]) < 0.01


# Conduction, gravity and walls ----------------------------------------#
def _hot_spot(field, conduction, cells=64, t=2e-3):
    hierarchy = AMRHierarchy(
        (cells, cells), ((-1.0, 1.0), (-1.0, 1.0)), periodic=(True, True)
    )
    model = MHD2D(hierarchy, mean_field=field, conduction=conduction, limiter="mc")
    model.set_state(
        1.0,
        (0.0, 0.0, 0.0),
        lambda x, y: 1.0 + 0.01 * np.exp(-(x**2 + y**2) / 0.02),
    )
    before = model.totals()["energy"]
    model.run(t)
    data = model.dataset()
    return model, data, (data.pressure / data.density).values, before


def test_conduction_along_an_aligned_field_is_one_dimensional_diffusion():
    kappa, t = 10.0, 2e-3
    model, data, temperature, before = _hot_spot((1.0, 0.0), (kappa, 0.0), t=t)
    chi = (5.0 / 3.0 - 1.0) * kappa
    s2 = 0.01 + 2.0 * chi * t
    x, y = data.x.values[:, None], data.y.values[None, :]
    exact = 1.0 + 0.01 * np.sqrt(0.01 / s2) * np.exp(-(x**2) / (2 * s2) - y**2 / 0.02)
    assert np.abs(temperature - exact).max() < 2e-5
    assert abs(model.totals()["energy"] - before) < 1e-13


def test_oblique_conduction_stays_on_the_field_lines_without_overshoot():
    field = (1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0))
    _, _, temperature, _ = _hot_spot(field, (10.0, 0.0))
    assert temperature.min() > 1.0 - 1e-14  # the limited transverse gradient
    np.testing.assert_allclose(temperature, temperature.T, atol=1e-15)
    along, across = temperature[40, 40] - 1.0, temperature[40, 24] - 1.0
    assert along > 100.0 * across


def test_isotropic_conduction_without_a_field():
    kappa, t = 10.0, 2e-3
    _, data, temperature, _ = _hot_spot((0.0, 0.0), (0.0, kappa), t=t)
    s2 = 0.01 + 2.0 * (2.0 / 3.0) * kappa * t
    x, y = data.x.values[:, None], data.y.values[None, :]
    exact = 1.0 + 0.01 * (0.01 / s2) * np.exp(-(x**2 + y**2) / (2 * s2))
    assert np.abs(temperature - exact).max() < 3e-5


def test_free_fall_gains_the_momentum_of_gravity():
    model = MHD2D(AMRHierarchy((8, 8), periodic=(True, True)), gravity=(0.2, -0.5))
    model.set_state(2.0, (0.0, 0.0, 0.0), 1.0)
    model.run(0.3)
    totals = model.totals()
    assert totals["momentum_x"] == pytest.approx(2.0 * 0.2 * 0.3, rel=1e-12)
    assert totals["momentum_y"] == pytest.approx(-2.0 * 0.5 * 0.3, rel=1e-12)


def test_walls_conserve_mass_and_energy_and_keep_the_normal_field_zero():
    hierarchy = AMRHierarchy((24, 24))
    model = MHD2D(hierarchy, boundaries={"x": "wall", "y": "wall"})
    model.set_state(
        1.0,
        (0.3, -0.2, 0.1),
        lambda x, y: 1.0 + 2.0 * np.exp(-((x - 0.3) ** 2 + (y - 0.6) ** 2) / 0.005),
        vector_potential=lambda x, y: 0.1 * np.sin(np.pi * x) * np.sin(np.pi * y),
        bz=0.2,
    )
    before = model.totals()
    model.run(0.3)
    after = model.totals()
    for name in ("density", "energy", "momentum_z", "bz"):
        assert abs(after[name] - before[name]) < 1e-13
    az = hierarchy.gather(hierarchy.patches[0], model.potential)
    for edge in (az[0], az[-1], az[:, 0], az[:, -1]):
        assert np.abs(edge).max() < 1e-15  # A_z constant along a wall: B_n = 0


def test_rayleigh_taylor_grows_and_a_strong_field_holds_it():
    free = MHD2D.rayleigh_taylor(16)
    free.run(4.0)
    held = MHD2D.rayleigh_taylor(16, field=0.3)
    held.run(4.0)
    growing = float(np.abs(free.dataset().velocity_y).max())
    stable = float(np.abs(held.dataset().velocity_y).max())
    assert growing > 0.015 and stable < 0.006  # from 0.01 at t = 0
    assert free.divergence() < 1e-14


def test_rejects_bad_boundaries_and_coefficients():
    periodic = AMRHierarchy((4, 4), periodic=(True, False))
    with pytest.raises(ValueError, match="periodic"):
        MHD2D(periodic, boundaries={"x": "wall"})
    with pytest.raises(ValueError, match="boundaries"):
        MHD2D(AMRHierarchy((4, 4)), boundaries={"z": "wall"})
    with pytest.raises(ValueError, match="boundaries"):
        MHD2D(AMRHierarchy((4, 4)), boundaries={"y": "inflow"})
    with pytest.raises(ValueError, match="mean field"):
        MHD2D(AMRHierarchy((4, 4)), boundaries={"y": "wall"}, mean_field=(0.0, 1.0))
    with pytest.raises(ValueError, match="non-negative"):
        MHD2D(AMRHierarchy((4, 4)), conduction=(-1.0, 0.0))


# The test problems of mhd-sandbox ----------------------------------------------------#
def test_rotor_keeps_its_point_symmetry_and_div_b_zero():
    model = MHD2D.rotor(32)
    model.run(0.05)
    data = model.dataset()
    density = data.density.values
    np.testing.assert_allclose(density, density[::-1, ::-1], rtol=1e-10)
    assert float(data.pressure.min()) > 0.0
    assert model.divergence() < 1e-14


def test_blast_wave_conserves_and_runs_along_the_field():
    model = MHD2D.blast_wave(32)
    before = model.totals()
    model.run(0.1)
    after = model.totals()
    for name in before:
        assert abs(after[name] - before[name]) < 1e-12
    data = model.dataset()
    pressure = data.pressure.values
    np.testing.assert_allclose(pressure, pressure[::-1, ::-1], rtol=1e-10)
    # along the field (the diagonal x = y) the blast reaches further than across it
    x, y = np.meshgrid(data.x.values, data.y.values, indexing="ij")
    hot = pressure > 1.0  # the hot core
    along = np.abs(x + y)[hot].max()
    across = np.abs(x - y)[hot].max()
    assert along > 1.5 * across
    assert model.divergence() < 1e-14
