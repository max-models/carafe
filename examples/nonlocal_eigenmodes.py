"""Nonlocal eigenmodes.

The gradient-drift instability on the 1D state of Boeuf's DC magnetron (arXiv:2211.17051,
figure 2) as an eigenvalue problem across the whole profile, with
:class:`wellbalanced.dispersion.NonlocalDispersion`:

1. growth rate and phase speed of the drift branch against the azimuthal wavelength (an
   8 cm box, ``m = 1-12``), with the drift-fluid electrons and with the gyro-averaged
   (Padé) electron response, which removes the mode: the anode layer is a few electron
   gyroradii wide;
2. the eigenfunction at Boeuf's 1.6 cm: one smooth hump of ``|delta phi|`` in the anode
   layer, the density perturbation narrower.
"""

import matplotlib.pyplot as plt
import numpy as np

from wellbalanced.dispersion import Equilibrium, NonlocalDispersion

equilibrium = Equilibrium.boeuf(np.linspace(3e-3, 0.02, 150))
m = np.arange(1, 13)
ky = 2 * np.pi * m / 0.08

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for options, label in (({}, "drift fluid"), ({"gyroaverage": True}, "gyro-averaged")):
    scan = NonlocalDispersion(equilibrium, **options).scan(ky)
    wavelength = scan["wavelength"] * 1e2
    axes[0].plot(
        wavelength, np.nan_to_num(scan["growth_rate"]) * 1e-6, "o-", label=label
    )
    axes[1].plot(wavelength, scan["phase_speed"] * 1e-3, "o-", label=label)
axes[0].set(xlabel="wavelength (cm)", ylabel="growth rate (1/µs)", title="drift branch")
axes[0].legend()
axes[1].axhline(8.0, color="gray", ls=":", label="Boeuf's PIC")
axes[1].set(xlabel="wavelength (cm)", ylabel="phase speed (km/s)", title="+E x B")
axes[1].legend()

modes = NonlocalDispersion(equilibrium).solve(2 * np.pi / 0.016).drift()
x = equilibrium.x * 1e3
for values, label in (
    (modes.potential[0], r"$|\delta\phi|$"),
    (modes.electron_density[0], r"$|\delta n_e / n_0|$"),
):
    axes[2].plot(x, abs(values) / abs(values).max(), label=label)
axes[2].set(xlabel="x (mm, anode at 20)", ylabel="normalised", title="1.6 cm mode")
axes[2].legend()

fig.tight_layout()
fig.savefig("nonlocal_eigenmodes.png", dpi=150)
print(f"m = 5: {modes.growth_rate:.2e} /s, {modes.phase_speed / 1e3:+.1f} km/s")
