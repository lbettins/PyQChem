"""Silica host framework builders."""

from __future__ import annotations

import numpy as np

from pyqchem.structure.host_framework import HostFramework


def build_silica_tetrahedron(center: np.ndarray | None = None, scale: float = 1.0) -> HostFramework:
    """
    Build a minimal tetrahedral SiO4 host motif as a fixed framework.

    Parameters
    ----------
    center : np.ndarray | None
        Center of the tetrahedron in Angstrom. Defaults to origin.
    scale : float
        Bond-length scale factor.

    Returns
    -------
    HostFramework
        Fixed SiO4-like tetrahedral host.
    """
    center = np.zeros(3) if center is None else np.asarray(center, dtype=float)
    si_o = 1.62 * scale
    tet_dirs = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=float,
    )
    tet_dirs /= np.linalg.norm(tet_dirs, axis=1, keepdims=True)
    symbols = ["Si"]
    positions = [center.copy()]
    for direction in tet_dirs:
        symbols.append("O")
        positions.append(center + si_o * direction)
    return HostFramework.from_positions(symbols, np.array(positions), name="SiO4_host")
