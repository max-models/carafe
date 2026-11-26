r"""Plasmoids on refined patches.

Two Harris current sheets in a periodic box reconnect through resistivity (`MHD2D`,
`harris_sheets`, the state set again on the patches after each of the first regrids); the patches follow the current density and are regridded every few
steps. The same case runs on the uniform mesh of the finest level, and the script compares
the two: the reconnected flux ``psi(t)`` of the lower sheet, its O-points (plasmoids), the
leaf cells and the wall time per step.

The defaults are a quick look (a minute): a thick sheet at a Lundquist number of 2000, which
tears into one island. Plasmoid chains need ``S`` above about 10^4 and a thin sheet, e.g.
``--production``: ``S = 4e4``, three refined levels, noise to seed the plasmoids, to
``t = 60`` (hours; run it yourself, under ``mpirun`` with ``--placement morton``). To profile
a run::

    scope-profiler run --enable-tag carafe --enable-tag patchwork -o prof.h5 \
        examples/plasmoid_reconnection.py --no-uniform
"""

from __future__ import annotations

import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
from maybempi import MPI
from patchwork import AMRHierarchy, CompositeCells, plot_patches

from carafe.mhd import MHD2D, harris_sheets, sheet_diagnostics

QUICK = {
    "base": (64, 32),
    "levels": 2,
    "width": 0.08,
    "eta": 2e-3,
    "perturbation": 0.05,
    "noise": 0.0,
    "t_end": 0.6,
    "every": 0.05,
}
PRODUCTION = {
    "base": (256, 128),
    "levels": 3,
    "width": 0.02,
    "eta": 1e-4,
    "perturbation": 1e-3,
    "noise": 1e-4,
    "t_end": 60.0,
    "every": 0.5,
}
BOX = ((0.0, 4.0), (0.0, 2.0))


def run(case: dict, *, refined: bool, args: argparse.Namespace) -> dict:
    """One run of ``case``: refined patches, or the uniform mesh of the finest level."""
    base = case["base"]
    levels = case["levels"]
    shape = base if refined else tuple(n * 2**levels for n in base)
    mesh = AMRHierarchy(
        shape,
        bounds=BOX,
        periodic=(True, True),
        comm=MPI.COMM_WORLD,
        placement=args.placement,
    )
    state = harris_sheets(
        mesh,
        width=case["width"],
        perturbation=case["perturbation"],
        noise=case["noise"],
    )
    model = MHD2D(mesh, resistivity=case["eta"])
    model.set_state(**state)

    def regrid() -> None:
        tags = model.current_tags(args.threshold, levels=levels)
        model.regrid(tags, buffer=2, max_patch_size=args.max_patch)

    if refined:
        for _ in range(levels):  # one level more per regrid, the sheets set on it
            regrid()
            model.set_state(**state)
    history = [sheet_diagnostics(model)]
    wall, steps = 0.0, 0
    next_output = case["every"]
    while model.time < case["t_end"] * (1 - 1e-12):
        start = time.perf_counter()
        if refined and steps % args.regrid_every == 0 and steps > 0:
            regrid()
        dt = min(model.stable_step(), next_output - model.time)
        model.step(dt)
        wall += time.perf_counter() - start
        steps += 1
        if model.time >= next_output * (1 - 1e-12):
            history.append(sheet_diagnostics(model))
            next_output += case["every"]
    return {
        "model": model,
        "history": history,
        "steps": steps,
        "wall_per_step": wall / max(steps, 1),
        "patches": len(model.hierarchy.patches),
    }


def plot_current(model) -> None:
    """|J_z| on the leaf cells, with the patch outlines."""
    leaf = model.leaf_values()
    mesh = model.hierarchy
    finest = mesh.spacing(mesh.max_level)
    (x0, x1), (y0, y1) = mesh.bounds
    x = x0 + (np.arange(round((x1 - x0) / finest[0])) + 0.5) * finest[0]
    y = y0 + (np.arange(round((y1 - y0) / finest[1])) + 0.5) * finest[1]
    xx, yy = np.meshgrid(x, y, indexing="ij")
    cells = CompositeCells(mesh).locate(np.array([xx.ravel(), yy.ravel()]))
    current = np.abs(leaf["current_z"][cells]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    image = ax.pcolormesh(x, y, current.T, cmap="magma", shading="auto")
    plot_patches(mesh, show_grid=False, ax=ax)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    fig.colorbar(image, ax=ax, label=r"$|J_z|$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_aspect("equal")
    ax.set_title(
        f"t = {model.time:.2f}: {len(mesh.patches)} patches on levels 0-{mesh.max_level}"
    )


def plot_flux(results: dict) -> None:
    """The reconnected flux of every run against time."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for name, result in results.items():
        history = result["history"]
        ax.plot(
            [h["time"] for h in history],
            [h["psi"] for h in history],
            label=f"{name} ({result['leaf_cells']} cells)",
        )
    ax.set_xlabel(r"$t\ (L / v_A)$")
    ax.set_ylabel(r"reconnected flux $\psi$")
    ax.legend()


def main() -> None:
    """Run the case refined and uniform, print the comparison, plot."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--production", action="store_true")
    parser.add_argument(
        "--no-uniform", action="store_true", help="skip the uniform run"
    )
    parser.add_argument("--t-end", type=float, help="override the case's end time")
    parser.add_argument(
        "--threshold", type=float, default=0.08, help="|J| dx to refine"
    )
    parser.add_argument("--regrid-every", type=int, default=10)
    parser.add_argument("--max-patch", type=int, default=32)
    parser.add_argument("--placement", default="morton")
    parser.add_argument("--output", help="write the histories and costs to this JSON")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    case = dict(PRODUCTION if args.production else QUICK)
    if args.t_end is not None:
        case["t_end"] = args.t_end

    results = {"refined": run(case, refined=True, args=args)}
    if not args.no_uniform:
        results["uniform"] = run(case, refined=False, args=args)
    for result in results.values():
        result["leaf_cells"] = result["history"][-1]["leaf_cells"]

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        print(
            f"{'run':10s} {'cells':>8s} {'patches':>8s} {'steps':>6s} "
            f"{'ms/step':>8s} {'psi':>9s} {'O-points':>8s}"
        )
        for name, result in results.items():
            last = result["history"][-1]
            print(
                f"{name:10s} {result['leaf_cells']:8d} {result['patches']:8d} "
                f"{result['steps']:6d} {1e3 * result['wall_per_step']:8.1f} "
                f"{last['psi']:9.5f} {last['o_points']:8d}"
            )
        if "uniform" in results:
            a = np.array([h["psi"] for h in results["refined"]["history"]])
            b = np.array([h["psi"] for h in results["uniform"]["history"]])
            n = min(a.size, b.size)
            growth = np.max(np.abs(b[:n] - b[0]))
            print(
                f"psi: refined - uniform at most {np.max(np.abs(a[:n] - b[:n])):.2e} "
                f"(the uniform run's change {growth:.2e})"
            )
    if args.output and rank == 0:
        summary = {
            name: {k: v for k, v in result.items() if k != "model"}
            for name, result in results.items()
        }
        with open(args.output, "w") as handle:
            json.dump({"case": case, "runs": summary}, handle, indent=1)
    if not args.no_plot:
        plot_current(results["refined"]["model"])
        plot_flux(results)
        if rank == 0:
            plt.show()


if __name__ == "__main__":
    main()
