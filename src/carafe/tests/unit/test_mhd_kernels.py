"""The compiled MHD2D kernels against their NumPy versions."""

import cunumpy as xp
import numpy as np
import pytest

from carafe.kernels.ct_contact import kernel as ct_contact
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
