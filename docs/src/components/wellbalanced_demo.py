"""The live demo of the docs: a dispersion explorer for the gradient-drift instability.

The local dispersion relation of the magnetron's ionisation region, solved over the azimuthal
wavenumber for the plasma parameters of the sliders, by one of three models: Boeuf's cold-ion
relation (`wellbalanced.driftfluid2d.boeuf_dispersion_roots`), and the local theory of
`wellbalanced.dispersion` with kinetic ions, in the 2D model field ``B(x) z`` or in a curl-free
(3D, vacuum) field, whose drifts are twice as compressible. It runs in the browser (Pyodide);
`run` takes the values of the sliders and returns the figures.
"""

import numpy as np

from wellbalanced.dispersion import KineticIons, LocalDispersion, Plasma
from wellbalanced.driftfluid2d import boeuf_dispersion_roots

# 1/m, log-spaced: the local theory has a narrow peak at a few 100 /m
WAVENUMBERS = np.geomspace(30.0, 3000.0, 100)
MODELS = {
    "Boeuf": "Boeuf, cold ions",
    "2D field": "kinetic ions, 2D field",
    "3D field": "kinetic ions, 3D field",
}


def _boeuf(b_field, e_field, te, density, ln, lb, nu, grad_b):
    """Growth rate and frequency of Boeuf's fastest root (1/s, rad/s)."""
    roots = np.array(
        [
            boeuf_dispersion_roots(
                k, density, e_field, b_field, te, ln, lb, nu, grad_b=grad_b
            )
            for k in WAVENUMBERS
        ]
    )
    fastest = roots[np.arange(len(WAVENUMBERS)), np.argmax(roots.imag, axis=1)]
    return fastest.imag, fastest.real


def _local(b_field, e_field, te, density, ln, lb, nu, curl_free):
    """Growth rate and frequency of the drift branch of the local theory (1/s, rad/s)."""
    plasma = Plasma(
        density=density,
        e_field=e_field,
        b_field=b_field,
        electron_temperature=te,
        density_length=ln,
        field_length=lb,
        electron_collision_frequency=nu,
        curl_free_field=curl_free,
    )
    scan = LocalDispersion(plasma, ions=KineticIons()).scan(WAVENUMBERS)
    return scan["growth_rate"].values, scan["frequency"].values


def _figure(lines: dict, title: str, scale: float) -> dict:
    """One line per model against the wavenumber, its fastest mode marked."""
    data = []
    for name, (y, i) in lines.items():
        data.append({"x": WAVENUMBERS, "y": y * scale, "mode": "lines", "name": name})
        if i is not None:
            data.append(
                {
                    "x": [WAVENUMBERS[i]],
                    "y": [y[i] * scale],
                    "mode": "markers",
                    "marker": {"size": 9},
                    "name": f"{name}: fastest",
                    "showlegend": False,
                }
            )
    return {
        "data": data,
        "layout": {
            "xaxis": {"title": {"text": "$k_y$ (1/m)"}, "type": "log"},
            "yaxis": {"title": {"text": title}},
            "showlegend": len(lines) > 1,
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
    model="Boeuf",
    grad_b="on",
):
    """Growth rate and phase speed of the gradient-drift mode against ``k_y``.

    The defaults are Boeuf's DC magnetron at ``x = 1.8`` cm (arXiv:2211.17051), where his
    fastest mode grows at about 1e7 /s at a wavelength of 1.6 cm.

    Args:
        b_field: Magnetic field ``B`` (mT).
        e_field: Axial electric field ``E_0`` (V/m), negative towards the cathode.
        electron_temperature: Electron temperature ``T_e`` (eV).
        density: Plasma density ``n_0`` (m^-3).
        density_length: Signed density gradient length ``L_n = n / n'`` (mm).
        field_length: Signed field gradient length ``L_B = B / B'`` (mm).
        collision_frequency: Electron collision frequency ``nu`` (1/s).
        model: ``"Boeuf"``, ``"2D field"``, ``"3D field"`` (see ``MODELS``) or ``"all three"``.
        grad_b: ``"on"`` keeps the grad-B drift frequency ``omega_D`` in Boeuf's relation,
            ``"off"`` drops it (the local theory always keeps the drifts).

    Returns:
        The growth rate and phase speed figures, the fastest mode's numbers and a sentence.
    """
    args = (
        b_field * 1e-3,
        float(e_field),
        float(electron_temperature),
        float(density),
        density_length * 1e-3,
        field_length * 1e-3,
        float(collision_frequency),
    )
    solve = {
        "Boeuf": lambda: _boeuf(*args, grad_b=grad_b == "on"),
        "2D field": lambda: _local(*args, curl_free=False),
        "3D field": lambda: _local(*args, curl_free=True),
    }
    chosen = list(MODELS) if model == "all three" else [model]
    growth_lines, speed_lines, fastest = {}, {}, {}
    for key in chosen:
        growth, frequency = solve[key]()
        growth = np.where(np.isfinite(growth), growth, np.nan)
        # the phase speed of a growing mode only: a damped root is not seen
        speed = np.where(growth > 0, frequency / WAVENUMBERS, np.nan)
        i = int(np.nanargmax(growth)) if np.any(growth > 0) else None
        growth_lines[MODELS[key]] = (growth, i)
        speed_lines[MODELS[key]] = (speed, i)
        fastest[key] = None if i is None else (growth[i], WAVENUMBERS[i], speed[i])
    figures = [
        _figure(growth_lines, "growth rate (1/µs)", 1e-6),
        _figure(speed_lines, "phase speed (km/s)", 1e-3),
    ]
    exb = -e_field / (b_field * 1e-3)
    numbers = {}
    for key, mode in fastest.items():
        label = "" if len(chosen) == 1 else f" ({MODELS[key]})"
        if mode is None:
            numbers[f"Largest growth rate (1/µs){label}"] = "stable"
            continue
        gamma, k, speed = mode
        numbers[f"Largest growth rate (1/µs){label}"] = gamma * 1e-6
        numbers[f"Its wavelength (cm){label}"] = 2 * np.pi / k * 1e2
        if len(chosen) == 1:
            numbers["Its phase speed (km/s)"] = speed * 1e-3
    numbers["E×B speed (km/s)"] = exb * 1e-3
    growing = [mode for mode in fastest.values() if mode is not None]
    if not growing:
        text = "Stable: no root grows at these parameters."
    elif len(chosen) == 1:
        text = f"The fastest mode grows by a factor e in {1e6 / growing[0][0]:.3g} µs."
    else:
        text = (
            "Boeuf counts the grad-B drift twice; the drift fluid in the 2D model field "
            "B(x) z counts it once, and the curl-free field of a 3D magnetron adds the "
            "compressibility that the 2D field misses."
        )
    return {"figure": figures, "numbers": numbers, "text": text}
