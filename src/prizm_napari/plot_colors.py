"""
Shared plotting helpers for categorical colors that stay visually distinct.
"""

from __future__ import annotations

import colorsys

import matplotlib
import numpy as np


def _ordered_cmap_colors(name: str, n: int = 20) -> list[tuple[float, float, float]]:
    rgba = matplotlib.colormaps[name](np.linspace(0.0, 1.0, n))
    order = list(range(0, n, 2)) + list(range(1, n, 2))
    return [tuple(float(x) for x in rgba[i, :3]) for i in order]


_DISTINCT_BASE = (
    _ordered_cmap_colors("tab20")
    + _ordered_cmap_colors("tab20b")
    + _ordered_cmap_colors("tab20c")
)

_NEUTRAL_BASE = np.asarray(
    [
        (0.18, 0.18, 0.18),
        (0.34, 0.34, 0.34),
        (0.52, 0.52, 0.52),
        (0.70, 0.70, 0.70),
    ],
    dtype=float,
)


def distinct_categorical_colors(n: int) -> np.ndarray:
    """
    Return n RGB colors with high categorical separation.

    The palette uses reordered matplotlib tab families first, then falls back to
    evenly distributed HSV colors if more colors are needed.
    """
    n = max(0, int(n))
    if n == 0:
        return np.zeros((0, 3), dtype=float)

    if n <= len(_DISTINCT_BASE):
        return np.asarray(_DISTINCT_BASE[:n], dtype=float)

    colors = list(_DISTINCT_BASE)
    extra_needed = n - len(colors)
    golden = 0.618033988749895
    sat_cycle = (0.82, 0.72, 0.62)
    val_cycle = (0.92, 0.82, 0.72)
    for i in range(extra_needed):
        hue = (0.11 + golden * i) % 1.0
        sat = sat_cycle[i % len(sat_cycle)]
        val = val_cycle[(i // len(sat_cycle)) % len(val_cycle)]
        colors.append(colorsys.hsv_to_rgb(hue, sat, val))
    return np.asarray(colors[:n], dtype=float)


def neutral_categorical_colors(n: int) -> np.ndarray:
    """
    Return n neutral grayscale RGB colors for control-like groups.
    """
    n = max(0, int(n))
    if n == 0:
        return np.zeros((0, 3), dtype=float)
    if n <= len(_NEUTRAL_BASE):
        return _NEUTRAL_BASE[:n].copy()
    values = np.linspace(0.18, 0.82, n, dtype=float)
    return np.column_stack([values, values, values])
