"""The mhd_sweep kernel: Pyccel and NumPy versions with the same arguments.

``kernel(...)`` runs the Pyccel version, or the NumPy version if Pyccel is unavailable
(see ``cunumpy.kernels.set_host_kernel_implementation`` to choose).
"""

from carafe.kernels import KERNEL_PACKAGE

# host kernels only: no CUDA version, so CuPy arrays fall back to the host one
kernel = KERNEL_PACKAGE.from_folder(__name__, missing_cuda="fallback")

#: The slope limiters by their code in the kernel's ``limiter`` argument.
LIMITER_CODES = ("minmod", "vanleer", "mc", "none")
