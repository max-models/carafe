"""Slope limiters of the MUSCL reconstruction, shared by the finite-volume models.

Each limiter takes the backward and forward differences ``a`` and ``b`` of a cell and
returns its limited slope (per cell, not per length): ``minmod``, ``vanleer`` and
``mc`` (monotonized central) are TVD; ``none`` is the central slope, second order on
smooth flow but not monotone.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

__all__ = ["LIMITERS"]


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(a * b > 0.0, np.where(np.abs(a) < np.abs(b), a, b), 0.0)


def _vanleer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    product = a * b
    return np.where(
        product > 0.0, 2.0 * product / np.where(product > 0, a + b, 1.0), 0.0
    )


def _mc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    central = 0.5 * (a + b)
    limit = 2.0 * np.where(np.abs(a) < np.abs(b), a, b)
    return np.where(
        a * b > 0.0, np.sign(central) * np.minimum(np.abs(central), np.abs(limit)), 0.0
    )


def _central(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * (a + b)


LIMITERS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "minmod": _minmod,
    "vanleer": _vanleer,
    "mc": _mc,
    "none": _central,
}
