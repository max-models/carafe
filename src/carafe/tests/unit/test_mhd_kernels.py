"""The compiled MHD2D kernels against their NumPy versions."""

import cunumpy as xp
import numpy as np
import pytest

from carafe.kernels.ct_contact import kernel as ct_contact
from carafe.kernels.mhd_conductive import kernel as mhd_conductive
from carafe.kernels.mhd_primitive import kernel as mhd_primitive
from carafe.kernels.mhd_rate import kernel as mhd_rate
from carafe.kernels.mhd_resistive import kernel as mhd_resistive
from carafe.kernels.mhd_sweep import LIMITER_CODES
from carafe.kernels.mhd_sweep import kernel as mhd_sweep
from carafe.limiters import LIMITERS

GAMMA = 5.0 / 3.0
SHAPE = (13, 9)


def _block(seed=1):
    """Primitive variables of a haloed block with a near-vacuum cell and a zero B_n."""
    rng = np.random.default_rng(seed)
    nx, ny = SHAPE
    prim = rng.normal(size=(8, nx, ny))
    prim[0] = rng.uniform(0.5, 2.0, SHAPE)
    prim[4] = rng.uniform(0.1, 1.0, SHAPE)
    prim[1, :, 6] += 30.0  # supersonic rows, both ways along y
    prim[2, 8, :] -= 30.0
    prim[4, 5, 3] = 1.0e-3  # its slopes would give a negative pressure
    bxf = rng.normal(size=(nx + 1, ny))
    byf = rng.normal(size=(nx, ny + 1))
    bxf[4, 2] = byf[2, 4] = 0.0
    return prim, bxf, byf


def _both(kernel, shape, *args):
    results = []
    for implementation in ("pyccel", "numpy"):
        with xp.kernels.use_host_kernel_implementation(implementation):
            out = np.full(shape, np.nan)
            kernel(*args, out)
            results.append(out)
    return results


def test_the_compiled_kernels_are_used():
    assert mhd_sweep.selected() == "pyccel"
    assert ct_contact.selected() == "pyccel"
    for kernel in (mhd_primitive, mhd_rate, mhd_resistive, mhd_conductive):
        assert kernel.selected() == "pyccel"
    assert set(LIMITER_CODES) == set(LIMITERS)


@pytest.mark.parametrize("limiter", range(len(LIMITER_CODES)))
@pytest.mark.parametrize("axis", [0, 1])
def test_sweep_matches_numpy(axis, limiter):
    prim, bxf, byf = _block()
    nx, ny = SHAPE
    shape = (7, nx - 3, ny) if axis == 0 else (7, nx, ny - 3)
    bn = (bxf, byf)[axis]
    compiled, numpy = _both(mhd_sweep, shape, prim, bn, axis, limiter, GAMMA)
    assert np.all(np.isfinite(compiled))
    np.testing.assert_allclose(compiled, numpy, rtol=1e-12, atol=1e-12)


def test_ct_contact_matches_numpy():
    rng = np.random.default_rng(2)
    nx, ny = SHAPE
    fx = rng.normal(size=(7, nx - 3, ny))
    fy = rng.normal(size=(7, nx, ny - 3))
    fx[0, 3, 4] = fy[0, 2, 5] = 0.0  # no mass flux: the mean of both sides
    ecc = rng.normal(size=SHAPE)
    compiled, numpy = _both(
        ct_contact, (nx - 3, ny - 3), fx[5], fy[5], fx[0], fy[0], ecc
    )
    np.testing.assert_allclose(compiled, numpy, rtol=0, atol=1e-14)


def test_orszag_tang_steps_agree():
    from carafe.mhd import MHD2D

    states = []
    for implementation in ("pyccel", "numpy"):
        with xp.kernels.use_host_kernel_implementation(implementation):
            model = MHD2D.orszag_tang(32)
            for _ in range(5):
                model.step()
            states.append(model.dataset())
    for name in states[0].data_vars:
        np.testing.assert_allclose(
            states[0][name].values, states[1][name].values, rtol=1e-11, atol=1e-12
        )


def _conserved(seed=3):
    """Conserved fields and nodal A_z of a haloed block with a positive pressure."""
    rng = np.random.default_rng(seed)
    nx, ny = SHAPE
    rho = rng.uniform(0.5, 2.0, SHAPE)
    momenta = [rng.normal(size=SHAPE) * rho for _ in range(3)]
    bz = rng.normal(size=SHAPE)
    az = 0.1 * rng.normal(size=(nx + 1, ny + 1))
    energy = 20.0 + rng.uniform(size=SHAPE)
    return [rho, *momenta, energy, bz], az


def _primitive_both(fields, az):
    nx, ny = SHAPE
    results = []
    for implementation in ("pyccel", "numpy"):
        with xp.kernels.use_host_kernel_implementation(implementation):
            prim = np.full((8, nx, ny), np.nan)
            bxf = np.full((nx + 1, ny), np.nan)
            byf = np.full((nx, ny + 1), np.nan)
            ecc = np.full(SHAPE, np.nan)
            bad = mhd_primitive(
                *fields, az, 0.1, 0.2, 0.3, -0.4, GAMMA, prim, bxf, byf, ecc
            )
            results.append((bad, prim, bxf, byf, ecc))
    return results


def test_primitive_matches_numpy():
    fields, az = _conserved()
    compiled, numpy = _primitive_both(fields, az)
    assert compiled[0] == numpy[0] == 0
    for a, b in zip(compiled[1:], numpy[1:], strict=True):
        assert np.all(np.isfinite(a))
        np.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize(
    ("row", "value"),
    [(0, -1.0), (4, -1.0e3), (1, np.inf), (2, np.nan), (5, -np.inf), (0, np.nan)],
)
def test_primitive_counts_the_bad_cells(row, value):
    fields, az = _conserved()
    fields[row][3, 4] = value  # a cell with a bad density, pressure or momentum
    fields[row][7, 1] = value
    for bad, *_ in _primitive_both(fields, az):
        assert bad == 2


def test_rate_matches_numpy():
    fields, az = _conserved()
    results = []
    for implementation in ("pyccel", "numpy"):
        with xp.kernels.use_host_kernel_implementation(implementation):
            out = np.full(2, np.nan)
            mhd_rate(*fields, az, 2, 0.1, 0.2, 0.3, -0.4, GAMMA, out)
            results.append(out)
    np.testing.assert_allclose(results[0], results[1], rtol=1e-14)
    assert results[0][1] == np.min(fields[0][2:-2, 2:-2])


def test_resistive_matches_numpy():
    prim, bxf, byf = _block()
    nx, ny = SHAPE
    rng = np.random.default_rng(4)
    fx0 = rng.normal(size=(7, nx - 3, ny))
    fy0 = rng.normal(size=(7, nx, ny - 3))
    ez0 = rng.normal(size=(nx - 3, ny - 3))
    results = []
    for implementation in ("pyccel", "numpy"):
        with xp.kernels.use_host_kernel_implementation(implementation):
            fx, fy, ez = fx0.copy(), fy0.copy(), ez0.copy()
            mhd_resistive(prim, bxf, byf, 0.1, 0.2, 0.05, fx, fy, ez)
            results.append((fx, fy, ez))
    for a, b, start in zip(*results, (fx0, fy0, ez0), strict=True):
        assert not np.allclose(a, start)
        np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-13)


def test_conductive_matches_numpy():
    prim, bxf, byf = _block()
    prim[5:8, 6, 4] = 0.0  # a cell without field
    bxf[6:8, 4] = byf[6, 4:6] = 0.0
    nx, ny = SHAPE
    rng = np.random.default_rng(5)
    fx0 = rng.normal(size=(7, nx - 3, ny))
    fy0 = rng.normal(size=(7, nx, ny - 3))
    results = []
    for implementation in ("pyccel", "numpy"):
        with xp.kernels.use_host_kernel_implementation(implementation):
            fx, fy = fx0.copy(), fy0.copy()
            mhd_conductive(prim, bxf, byf, 0.1, 0.2, 0.3, 0.01, fx, fy)
            results.append((fx, fy))
    for a, b, start in zip(*results, (fx0, fy0), strict=True):
        assert not np.allclose(a[4], start[4])
        np.testing.assert_array_equal(np.delete(a, 4, axis=0), np.delete(start, 4, 0))
        np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize(
    "options",
    [
        {"resistivity": 1e-3, "conduction": (1e-2, 1e-3)},
        {"resistivity": 1e-3, "conduction": (1e-3, 0.0), "field": 0.05},
    ],
)
def test_resistive_conductive_steps_agree(options):
    from carafe.mhd import MHD2D

    states = []
    for implementation in ("pyccel", "numpy"):
        with xp.kernels.use_host_kernel_implementation(implementation):
            if "field" in options:  # walls and gravity
                model = MHD2D.rayleigh_taylor(8, **options)
            else:
                model = MHD2D.orszag_tang(16, **options)
            for _ in range(3):
                model.step()
            states.append(model.dataset())
    for name in states[0].data_vars:
        np.testing.assert_allclose(
            states[0][name].values, states[1][name].values, rtol=1e-11, atol=1e-12
        )
