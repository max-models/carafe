"""The electric field at the nodes of a block by the CT-contact average.

The Pyccel version (CPU) of the ``ct_contact`` kernel (Gardiner and Stone 2005);
``ct_contact_numpy.py`` next to this file is the NumPy version, with the same
arguments. Called by ``MHD2D._patch_fluxes`` for every patch.

``fx_by`` and ``fx_rho`` are the x-face fluxes of ``B_y`` and the density of a haloed
block at x-faces ``2 .. X - 2`` (array index ``k`` is x-face ``k + 2``) and all rows;
``gy_bx`` and ``gy_rho`` the y-face fluxes of ``B_x`` and the density at all columns
and y-faces ``2 .. Y - 2``; ``ecc`` the cell electric field ``B_x u_y - B_y u_x`` of
the haloed block. ``ez`` gets ``E_z`` at the block's ``(nx + 1, ny + 1)`` nodes; node
``(a, b)`` of the block is node ``(a + 2, b + 2)`` of the haloed block (a halo of two).
"""


def upwind(mass: float, positive: float, negative: float) -> float:
    """``positive`` or ``negative`` by the sign of the mass flux, their mean at zero."""
    value = 0.5 * (positive + negative)
    if mass > 0.0:
        value = positive
    elif mass < 0.0:
        value = negative
    return value


def ct_contact(
    fx_by: "float[:,:]",
    gy_bx: "float[:,:]",
    fx_rho: "float[:,:]",
    gy_rho: "float[:,:]",
    ecc: "float[:,:]",
    ez: "float[:,:]",
) -> None:
    """Write ``E_z`` at the block's nodes to ``ez`` (see the module)."""
    for a in range(ez.shape[0]):
        i = a + 2
        for b in range(ez.shape[1]):
            j = b + 2
            fxb_lower = fx_by[i - 2, j - 1]
            fxb_upper = fx_by[i - 2, j]
            gyb_left = gy_bx[i - 1, j - 2]
            gyb_right = gy_bx[i, j - 2]
            dl2 = upwind(
                fx_rho[i - 2, j - 1],
                gyb_left - ecc[i - 1, j - 1],
                gyb_right - ecc[i, j - 1],
            )
            dr2 = upwind(
                fx_rho[i - 2, j], gyb_left - ecc[i - 1, j], gyb_right - ecc[i, j]
            )
            dl1 = upwind(
                gy_rho[i - 1, j - 2],
                -fxb_lower - ecc[i - 1, j - 1],
                -fxb_upper - ecc[i - 1, j],
            )
            dr1 = upwind(
                gy_rho[i, j - 2], -fxb_lower - ecc[i, j - 1], -fxb_upper - ecc[i, j]
            )
            ez[a, b] = 0.25 * (
                gyb_right + gyb_left - fxb_upper - fxb_lower + dl1 + dr1 + dl2 + dr2
            )
