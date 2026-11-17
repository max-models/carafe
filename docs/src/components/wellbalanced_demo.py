"""The live demo of the docs: a dispersion explorer for the gradient-drift instability.

Boeuf's local relation of the magnetron's ionisation region
(`wellbalanced.driftfluid2d.boeuf_dispersion_roots`), solved over the azimuthal wavenumber
for the plasma parameters of the sliders. It runs in the browser (Pyodide); `run` takes the
values of the sliders and returns the figures.
"""

import numpy as np

from wellbalanced.driftfluid2d import boeuf_dispersion_roots

WAVENUMBERS = np.linspace(20.0, 3000.0, 150)  # 1/m


def _figure(y: np.ndarray, peak: float, k_peak: float, title: str) -> dict:
    """A line of ``y`` against the wavenumber, with the fastest mode marked."""
    return {
        "data": [
            {"x": WAVENUMBERS, "y": y, "mode": "lines", "name": "fastest root"},
            {
                "x": [k_peak],
                "y": [peak],
                "mode": "markers",
                "marker": {"size": 10},
                "name": "fastest mode",
            },
        ],
        "layout": {
            "xaxis": {"title": {"text": "$k_y$ (1/m)"}},
            "yaxis": {"title": {"text": title}},
            "showlegend": False,
        },
    }


def run(
    b_field=6.0,
    e_field=-2200.0,
    electron_temperature=11.4,
    density=0.8e15,
    density_length=-2.2,
    field_length=-6.0,
    collision_frequency=2e7,
    grad_b="on",
):
    """Growth rate and phase speed of the gradient-drift mode against ``k_y``.

    The defaults are Boeuf's DC magnetron at ``x = 1.8`` cm (arXiv:2211.17051), where the
    fastest mode grows at about 1e7 /s at a wavelength of 1.6 cm.

    Args:
        b_field: Magnetic field ``B`` (mT).
        e_field: Axial electric field ``E_0`` (V/m), negative towards the cathode.
        electron_temperature: Electron temperature ``T_e`` (eV).
        density: Plasma density ``n_0`` (m^-3).
        density_length: Signed density gradient length ``L_n = n / n'`` (mm).
        field_length: Signed field gradient length ``L_B = B / B'`` (mm).
        collision_frequency: Electron collision frequency ``nu`` (1/s).
        grad_b: ``"on"`` keeps the grad-B drift frequency ``omega_D``, ``"off"`` drops it.

    Returns:
        The growth rate and phase speed figures, the fastest mode's numbers and a sentence.
    """
    roots = np.array(
        [
            boeuf_dispersion_roots(
                k,
                density,
                e_field,
                b_field * 1e-3,
                electron_temperature,
                density_length * 1e-3,
                field_length * 1e-3,
                collision_frequency,
                grad_b=grad_b == "on",
            )
            for k in WAVENUMBERS
        ]
    )
    fastest = roots[np.arange(len(WAVENUMBERS)), np.argmax(roots.imag, axis=1)]
    growth = fastest.imag
    # the phase speed of a growing mode only: a damped root is not seen
    speed = np.where(growth > 0, fastest.real / WAVENUMBERS, np.nan)
    i = int(np.argmax(growth))
    k_peak, gamma = WAVENUMBERS[i], growth[i]
    figures = [
        _figure(growth * 1e-6, gamma * 1e-6, k_peak, "growth rate (1/µs)"),
        _figure(speed * 1e-3, speed[i] * 1e-3, k_peak, "phase speed (km/s)"),
    ]
    exb = -e_field / (b_field * 1e-3)
    if gamma <= 0:
        return {
            "figure": figures,
            "numbers": {"E×B speed (km/s)": exb * 1e-3},
            "text": "Stable: no root grows at these parameters.",
        }
    return {
        "figure": figures,
        "numbers": {
            "Largest growth rate (1/µs)": gamma * 1e-6,
            "Its wavelength (cm)": 2 * np.pi / k_peak * 1e2,
            "Its phase speed (km/s)": speed[i] * 1e-3,
            "E×B speed (km/s)": exb * 1e-3,
        },
        "text": f"The fastest mode grows by a factor e in {1e6 / gamma:.3g} µs.",
    }
