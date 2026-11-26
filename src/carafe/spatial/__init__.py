"""Spatially resolved models: the inductive RF field of a driver, in a cut, in r–z or in 3D.

Everything that resolves the plane lives here, on one cell grid
(a tensor `carafe.spatial.grid.CellGrid`, or the refined patches of a
`carafe.spatial.patches.PatchGrid`; the only place the geometry and its
metric enter): the magnetoquasistatic field of coil windings, conductors and a
prescribed plasma (`carafe.spatial.inductive`), the plasma region
(`carafe.spatial.regions`) and the coupling of the field to a
power-dependent plasma (`carafe.spatial.coupling`), and the driver plasma as a
multi-species fluid heated by that field (`carafe.spatial.fluid`,
`Fluid2D`). The 3D field (`carafe.spatial.inductive3d`, step R4) is on
the edges of a Cartesian `carafe.spatial.edges.EdgeGrid`, or of refined
patches (`carafe.spatial.patch_edges.PatchEdgeGrid`, patchwork's edge space). The neutral gas
(H and H2 as two Navier–Stokes fluids, step R3.5: `carafe.spatial.neutrals`,
`NeutralFluid2D`) is on the plasma's cells and coupled to it by
`couple_neutrals`, and the subpackage is the unit a later
split of carafe moves as a whole (the magnetron's
`carafe.DriftFluid2D` is a candidate to join it then).

Shapes are anything with ``contains_points`` (simgeometry shapes, an optional
dependency: ``pip install "carafe-plasma[spatial]"``); simgeometry is imported only for the
cut cells of surface-impedance conductors (`carafe.spatial.cutcell`). patchwork, for `carafe.spatial.patches.PatchGrid`, is in the
same extra and imported when a patch grid is made.
"""

from carafe.spatial.coupling import (
    CoupledState,
    DriverCoupling,
    FluidPlasma,
    PlasmaProfile,
    fluid_plasma,
    global_model_plasma,
)
from carafe.spatial.edges import EdgeGrid
from carafe.spatial.fluid import Fluid2D, Fluid2DState
from carafe.spatial.grid import CellGrid, TransportFaces, stretched_edges
from carafe.spatial.inductive import (
    Conductor,
    InductiveField2D,
    InductiveSolution,
    InductiveState,
    InductiveTimeStepper,
    Winding,
)
from carafe.spatial.inductive3d import (
    Coil3D,
    Conductor3D,
    Feed3D,
    ImpedanceConductor3D,
    InductiveField3D,
    InductiveSolution3D,
)
from carafe.spatial.neutrals import (
    Collision,
    CoupledNeutrals,
    Inlet,
    NeutralFluid2D,
    NeutralSpecies,
    NeutralState,
    Opening,
    couple_neutrals,
    hard_sphere,
)
from carafe.spatial.patch_edges import PatchEdgeGrid
from carafe.spatial.patches import PatchGrid
from carafe.spatial.regions import PlasmaRegion

__all__ = [
    "CellGrid",
    "Coil3D",
    "Collision",
    "Conductor",
    "Conductor3D",
    "CoupledNeutrals",
    "CoupledState",
    "DriverCoupling",
    "EdgeGrid",
    "Feed3D",
    "ImpedanceConductor3D",
    "Fluid2D",
    "Fluid2DState",
    "FluidPlasma",
    "Inlet",
    "InductiveField2D",
    "InductiveField3D",
    "InductiveSolution",
    "InductiveSolution3D",
    "InductiveState",
    "InductiveTimeStepper",
    "NeutralFluid2D",
    "NeutralSpecies",
    "NeutralState",
    "Opening",
    "PatchEdgeGrid",
    "PatchGrid",
    "PlasmaProfile",
    "PlasmaRegion",
    "TransportFaces",
    "Winding",
    "couple_neutrals",
    "fluid_plasma",
    "global_model_plasma",
    "hard_sphere",
    "stretched_edges",
]
