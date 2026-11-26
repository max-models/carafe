"""MHD2D: the HLLD flux, conservation and div B on patches, convergence, resistivity."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("patchwork")

from patchwork import AMRHierarchy  # noqa: E402

from carafe.mhd import CONSERVED, MHD2D, alfven_wave, hlld_flux  # noqa: E402

GAMMA = 5.0 / 3.0


def physical_flux(state, bn):
    rho, u, v, w, p, by, bz = state
    b2 = bn * bn + by * by + bz * bz
    energy = p / (GAMMA - 1) + 0.5 * rho * (u * u + v * v + w * w) + 0.5 * b2
    pt = p + 0.5 * b2
    vb = u * bn + v * by + w * bz
    return (
        rho * u,
        rho * u * u + pt - bn * bn,
        rho * u * v - bn * by,
        rho * u * w - bn * bz,
        (energy + pt) * u - bn * vb,
        by * u - bn * v,
        bz * u - bn * w,
    )


@pytest.mark.parametrize(
    "state",
    [
        (1.0, 0.3, -0.2, 0.1, 0.8, 0.5, -0.4),
        (0.125, -2.0, 0.0, 0.0, 0.1, -1.0, 0.0),
    ],
)
def test_hlld_of_equal_states_is_the_physical_flux(state) -> None:
    left = [np.array([q]) for q in state]
    flux = hlld_flux(left, left, np.array([0.75]), GAMMA)
    np.testing.assert_allclose(
        np.concatenate(flux), np.array(physical_flux(state, 0.75)), atol=1e-14
    )


def test_hlld_keeps_a_contact_and_a_tangential_discontinuity_at_rest() -> None:
    # contact (normal field): density jumps, the rest is continuous
    left = [np.array([q]) for q in (1.0, 0.0, 0.2, 0.0, 1.0, 0.3, 0.1)]
    right = [np.array([q]) for q in (0.125, 0.0, 0.2, 0.0, 1.0, 0.3, 0.1)]
    flux = hlld_flux(left, right, np.array([0.75]), GAMMA)
    assert abs(flux[0][0]) < 1e-14
    # tangential discontinuity (no normal field): total pressure in balance
    left = [np.array([q]) for q in (1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0)]
    right = [np.array([q]) for q in (0.2, 0.0, 0.0, 0.0, 1.4, 0.0, np.sqrt(0.2))]
    flux = hlld_flux(left, right, np.array([0.0]), GAMMA)
    assert abs(flux[0][0]) < 1e-14
    assert abs(flux[1][0] - 1.5) < 1e-14


def alfven_error(n: int, levels: int) -> float:
    model, exact = alfven_wave(n, levels=levels)
    model.run(0.5)
    leaf = model.leaf_values()
    return float(np.mean(np.abs(leaf["by"] - exact(leaf["x"], 0.5))))


def test_the_alfven_wave_converges_at_second_order_across_a_refined_patch() -> None:
    errors = np.array([alfven_error(n, 1) for n in (16, 32)])
    assert np.log2(errors[0] / errors[1]) > 1.85
    # the patch makes the wave more accurate, not less
    assert errors[1] < alfven_error(32, 0)


def orszag_tang(n: int, levels: int = 0, **options) -> MHD2D:
    mesh = AMRHierarchy((n, n), bounds=((0, 1), (0, 1)), periodic=(True, True))
    if levels:
        # level-1 cells: x in (0.5, 1), y in (0.25, 1)
        mesh.add_patch(1, ((n, 2 * n), (n // 2, 2 * n)))
    model = MHD2D(mesh, **options)
    k = 2.0 * np.pi
    model.set_state(
        density=25.0 / (36.0 * np.pi),
        velocity=(lambda x, y: -np.sin(k * y), lambda x, y: np.sin(k * x), 0.0),
        pressure=5.0 / (12.0 * np.pi),
        potential=lambda x, y: (
            (np.cos(2 * k * x) / (2 * k) + np.cos(k * y) / k) / np.sqrt(4 * np.pi)
        ),
        bz=lambda x, y: 0.1 * np.cos(k * x),
    )
    return model


def shared_node_mismatch(model: MHD2D) -> float:
    """The largest difference of A_z between patches at nodes they share.

    Zero means that every coarse face on the edge of a finer level carries the sum of
    the fluxes of its fine faces: div B = 0 on the composite mesh.
    """
    mesh = model.hierarchy
    seen: dict[tuple[int, int], float] = {}
    worst = 0.0
    for patch in mesh.local_patches:
        values = patch.field("az").local
        scale = mesh.refinement_ratio ** (mesh.max_level - patch.level)
        (i0, _), (j0, _) = patch.box
        for (i, j), value in np.ndenumerate(values):
            key = ((i0 + i) * scale, (j0 + j) * scale)
            if key in seen:
                worst = max(worst, abs(seen[key] - value))
            else:
                seen[key] = value
    return worst


def test_steps_and_regrids_conserve_and_keep_the_field_divergence_free() -> None:
    model = orszag_tang(16, resistivity=1e-3)
    start = model.totals()
    for _ in range(3):
        before = model.totals()
        model.regrid(model.current_tags(0.1, levels=2), max_patch_size=16)
        after = model.totals()
        for name in CONSERVED:
            assert abs(after[name] - before[name]) < 1e-14
        assert shared_node_mismatch(model) == 0.0
        for _ in range(4):
            model.step()
    assert model.hierarchy.max_level == 2
    end = model.totals()
    for name in CONSERVED:
        assert abs(end[name] - start[name]) < 1e-13
    assert shared_node_mismatch(model) == 0.0
    leaf = model.leaf_values()
    assert np.all(leaf["p"] > 0) and np.all(leaf["rho"] > 0)


def test_a_refined_patch_is_closer_to_the_fine_mesh_than_the_coarse_one() -> None:
    fine, coarse = orszag_tang(32), orszag_tang(16)
    refined = orszag_tang(16, levels=1)
    dt = min(fine.stable_step(), refined.stable_step())
    for _ in range(15):
        for model in (fine, coarse, refined):
            model.step(dt)

    def at_fine_centres(model):
        """A model's leaf density at the centres of the fine run's cells.

        Centres in units of 1/64: fine cells at odd numbers, coarse cells at 4 i + 2.
        """
        leaf, target = model.leaf_values(), fine.leaf_values()
        values = {
            (round(x * 64), round(y * 64)): v
            for x, y, v in zip(leaf["x"], leaf["y"], leaf["rho"], strict=True)
        }
        result = np.array(
            [
                values.get(
                    (round(x * 64), round(y * 64)),
                    values.get((4 * int(x * 16) + 2, 4 * int(y * 16) + 2)),
                )
                for x, y in zip(target["x"], target["y"], strict=True)
            ]
        )
        return result, target

    covered = None
    errors = {}
    for name, model in (("refined", refined), ("coarse", coarse)):
        values, target = at_fine_centres(model)
        if covered is None:
            covered = (target["x"] > 0.5) & (target["y"] > 0.25)
        errors[name] = np.mean(np.abs(values - target["rho"])[covered])
    assert errors["refined"] < 0.5 * errors["coarse"]


@pytest.mark.parametrize("levels", [0, 1])
def test_resistivity_damps_the_field_at_the_diffusion_rate(levels) -> None:
    n, eta, k, amp = 16, 0.02, 2 * np.pi, 1e-3
    mesh = AMRHierarchy((n, n), bounds=((0, 1), (0, 1)), periodic=(True, True))
    if levels:
        mesh.add_patch(1, ((n // 2, 3 * n // 2), (n // 2, 3 * n // 2)))
    model = MHD2D(mesh, resistivity=eta, limiter="none")
    model.set_state(
        density=1.0,
        pressure=1.0,
        potential=lambda x, y: amp * np.cos(k * x) / k,
        bz=lambda x, y: amp * np.cos(k * y),
    )
    start = model.totals()
    model.run(0.25)
    leaf = model.leaf_values()
    decay = np.exp(-eta * k * k * model.time)
    by = np.mean(np.abs(leaf["by"] - amp * np.sin(k * leaf["x"]) * decay))
    bz = np.mean(np.abs(leaf["bz"] - amp * np.cos(k * leaf["y"]) * decay))
    # second order on one level; next to the coarse-fine edge the current of the
    # interpolated potential is first order
    assert by < (6e-3 if levels else 3e-3) * amp * decay
    assert bz < 4e-3 * amp * decay
    # Ohmic heating stays in the energy
    assert abs(model.totals()["energy"] - start["energy"]) < 1e-12


def test_a_shock_tube_stays_positive_and_conservative_on_patches() -> None:
    n = 64
    mesh = AMRHierarchy((n, 4), bounds=((0, 1), (0, 4 / n)), periodic=(True, True))
    mesh.add_patch(1, ((n // 2 - 16, n // 2 + 16), (0, 8)))
    model = MHD2D(mesh, gamma=2.0, mean_field=(0.75, 0.0))

    def inside(x, y):
        return (x > 0.25) & (x < 0.75)

    # Brio and Wu's states twice (periodic): B_y = 1 inside, -1 outside, B_y = -A'
    model.set_state(
        density=lambda x, y: np.where(inside(x, y), 1.0, 0.125),
        pressure=lambda x, y: np.where(inside(x, y), 1.0, 0.1),
        potential=lambda x, y: np.where(
            x < 0.25, x, np.where(x < 0.75, 0.5 - x, x - 1.0)
        ),
    )
    start = model.totals()
    model.run(0.05)
    leaf = model.leaf_values()
    assert np.all(np.isfinite(leaf["rho"])) and np.all(leaf["rho"] > 0.1)
    assert np.all(leaf["p"] > 0)
    end = model.totals()
    for name in CONSERVED:
        assert abs(end[name] - start[name]) < 1e-13


def test_unsupported_meshes_and_options_are_refused() -> None:
    with pytest.raises(ValueError, match="periodic"):
        MHD2D(AMRHierarchy((8, 8), periodic=(True, False)))
    mesh = AMRHierarchy((8, 8), periodic=(True, True))
    with pytest.raises(ValueError, match="limiter"):
        MHD2D(mesh, limiter="superbee")
    with pytest.raises(ValueError, match="resistivity"):
        MHD2D(AMRHierarchy((8, 8), periodic=(True, True)), resistivity=-1.0)
    with pytest.raises(ValueError, match="2D"):
        MHD2D(AMRHierarchy((8,), periodic=(True,)))
