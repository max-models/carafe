"""MHD2D on refined and regridded patches, and with resistivity."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("patchwork")

from patchwork import AMRHierarchy  # noqa: E402

from carafe.mhd import CONSERVED, MHD2D  # noqa: E402

AMPLITUDE = 0.1
K = 2.0 * np.pi


def alfven_along_x(n: int, levels: int) -> MHD2D:
    """A circularly polarised Alfvén wave along ``x`` on ``n x 4`` cells; ``v_A = 1``."""
    mesh = AMRHierarchy((n, 4), ((0.0, 1.0), (0.0, 4.0 / n)), periodic=(True, True))
    for level in range(1, levels + 1):
        r = 2**level
        mesh.add_patch(level, ((n * r // 4, 3 * n * r // 4), (0, 4 * r)))
    model = MHD2D(mesh, mean_field=(1.0, 0.0), limiter="none")
    model.set_state(
        1.0,
        (
            0.0,
            lambda x, y: -AMPLITUDE * np.sin(K * x),
            lambda x, y: -AMPLITUDE * np.cos(K * x),
        ),
        0.1,
        vector_potential=lambda x, y: AMPLITUDE * np.cos(K * x) / K,
        bz=lambda x, y: AMPLITUDE * np.cos(K * x),
    )
    return model


def alfven_error(n: int, levels: int) -> float:
    model = alfven_along_x(n, levels)
    model.run(0.5)
    leaf = model.leaf_values()
    exact = AMPLITUDE * np.sin(K * (leaf["x"] - 0.5))
    return float(np.mean(np.abs(leaf["magnetic_field_y"] - exact)))


def test_the_alfven_wave_converges_at_second_order_across_a_refined_patch() -> None:
    errors = np.array([alfven_error(n, 1) for n in (16, 32)])
    assert np.log2(errors[0] / errors[1]) > 1.85
    # the patch makes the wave more accurate, not less
    assert errors[1] < alfven_error(32, 0)


def orszag_tang(n: int, **options) -> MHD2D:
    mesh = AMRHierarchy((n, n), ((0.0, 1.0), (0.0, 1.0)), periodic=(True, True))
    model = MHD2D(mesh, **options)
    model.set_state(
        25.0 / (36.0 * np.pi),
        (lambda x, y: -np.sin(K * y), lambda x, y: np.sin(K * x), 0.0),
        5.0 / (12.0 * np.pi),
        vector_potential=lambda x, y: (
            (np.cos(2 * K * x) / (2 * K) + np.cos(K * y) / K) / np.sqrt(4 * np.pi)
        ),
        bz=lambda x, y: 0.1 * np.cos(K * x),
    )
    return model


def test_regrids_conserve_and_keep_the_field_divergence_free() -> None:
    model = orszag_tang(16, resistivity=1e-3)
    start = model.totals()
    for _ in range(3):
        before = model.totals()
        model.regrid(model.current_tags(0.1, levels=2), max_patch_size=16)
        after = model.totals()
        for name in CONSERVED:
            assert abs(after[name] - before[name]) < 1e-14
        assert model.composite_divergence() < 1e-12
        for _ in range(4):
            model.step()
    assert model.hierarchy.max_level == 2
    end = model.totals()
    for name in CONSERVED:
        assert abs(end[name] - start[name]) < 1e-13
    assert model.divergence() < 1e-12
    assert model.composite_divergence() < 1e-12
    leaf = model.leaf_values()
    assert np.all(leaf["pressure"] > 0) and np.all(leaf["density"] > 0)
    assert np.unique(leaf["level"]).max() == 2


def test_the_leaf_values_are_those_of_a_uniform_level() -> None:
    model = orszag_tang(16)
    leaf = model.leaf_values()
    data = model.dataset()
    order = np.lexsort((leaf["y"], leaf["x"]))
    np.testing.assert_allclose(
        leaf["density"][order], data["density"].values.ravel(), rtol=1e-14
    )
    np.testing.assert_allclose(
        leaf["magnetic_field_y"][order],
        data["magnetic_field_y"].values.ravel(),
        rtol=1e-14,
        atol=1e-15,
    )


@pytest.mark.parametrize("levels", [0, 1])
def test_resistivity_damps_the_field_at_the_diffusion_rate(levels) -> None:
    n, eta, amp = 16, 0.02, 1e-3
    mesh = AMRHierarchy((n, n), ((0.0, 1.0), (0.0, 1.0)), periodic=(True, True))
    if levels:
        mesh.add_patch(1, ((n // 2, 3 * n // 2), (n // 2, 3 * n // 2)))
    model = MHD2D(mesh, resistivity=eta, limiter="none")
    model.set_state(
        1.0,
        (0.0, 0.0, 0.0),
        1.0,
        vector_potential=lambda x, y: amp * np.cos(K * x) / K,
        bz=lambda x, y: amp * np.cos(K * y),
    )
    start = model.totals()
    model.run(0.25)
    leaf = model.leaf_values()
    decay = np.exp(-eta * K * K * model.time)
    by = np.mean(np.abs(leaf["magnetic_field_y"] - amp * np.sin(K * leaf["x"]) * decay))
    bz = np.mean(np.abs(leaf["magnetic_field_z"] - amp * np.cos(K * leaf["y"]) * decay))
    # second order on one level; next to a coarse-fine edge the current of the
    # interpolated potential is first order
    assert by < (6e-3 if levels else 3e-3) * amp * decay
    assert bz < 4e-3 * amp * decay
    # Ohmic heating stays in the energy
    assert abs(model.totals()["energy"] - start["energy"]) < 1e-12


def test_the_step_keeps_within_the_diffusion_limit() -> None:
    mesh = AMRHierarchy((16, 16), ((0.0, 1.0), (0.0, 1.0)), periodic=(True, True))
    model = MHD2D(mesh, resistivity=1.0)
    model.set_state(1.0, (0.0, 0.0, 0.0), 1e-3)
    dx = 1.0 / 16
    assert model.stable_step() == pytest.approx(model.cfl * 0.5 / (2.0 / dx**2))


def test_a_negative_resistivity_is_refused() -> None:
    mesh = AMRHierarchy((8, 8), periodic=(True, True))
    with pytest.raises(ValueError, match="resistivity"):
        MHD2D(mesh, resistivity=-1.0)
