"""MHD2D on all ranks against the same run on one rank (run under ``mpirun``).

Each rank builds the hierarchy twice: once over the world communicator (the patches
spread by ``placement``, the first one split over all ranks) and once on its own
(``COMM_SELF``). Both take the same regrids and steps; the leaf values must agree to
round-off and the integrals stay those of the start. Prints ``MPI_JOB_OK``.
"""

from __future__ import annotations

import sys

import numpy as np
from maybempi import MPI
from patchwork import AMRHierarchy

from carafe.mhd import CONSERVED, MHD2D

WORLD = MPI.COMM_WORLD


def run(comm, placement):
    mesh = AMRHierarchy(
        (32, 32),
        bounds=((0, 1), (0, 1)),
        periodic=(True, True),
        comm=comm,
        placement=placement,
    )
    model = MHD2D(mesh, resistivity=1e-3)
    k = 2 * np.pi
    model.set_state(
        density=25 / (36 * np.pi),
        velocity=(lambda x, y: -np.sin(k * y), lambda x, y: np.sin(k * x), 0.0),
        pressure=5 / (12 * np.pi),
        potential=lambda x, y: (
            (np.cos(2 * k * x) / (2 * k) + np.cos(k * y) / k) / np.sqrt(4 * np.pi)
        ),
        bz=lambda x, y: 0.1 * np.cos(k * x),
    )
    start = model.totals()
    for _ in range(2):
        model.regrid(model.current_tags(0.1, levels=2), max_patch_size=16)
        for _ in range(3):
            model.step()
    leaf = model.leaf_values()
    order = np.lexsort((leaf["y"], leaf["x"]))
    end = model.totals()
    drift = max(abs(end[name] - start[name]) for name in CONSERVED)
    return np.stack([leaf[key][order] for key in ("rho", "bx", "by", "bz", "p")]), drift


def main() -> None:
    placement = sys.argv[1] if len(sys.argv) > 1 else "morton"
    spread, drift = run(WORLD, placement)
    alone, _ = run(MPI.COMM_SELF, placement)
    difference = float(np.max(np.abs(spread - alone)))
    if WORLD.Get_rank() == 0:
        print(f"difference {difference:.2e}, drift {drift:.2e}")
        if difference < 1e-12 and drift < 1e-13:
            print("MPI_JOB_OK")


if __name__ == "__main__":
    main()
