"""Lebedev quadrature nodes on the unit sphere."""

from __future__ import annotations

import numpy as np


def lebedev_sphere(npoints: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return Lebedev quadrature nodes as spherical coordinates.

    Parameters
    ----------
    npoints : int
        Number of Lebedev points (must be a supported order, e.g. 6, 14, 26, 38).

    Returns
    -------
    theta : np.ndarray
        Polar angle from the north pole (+z), in `[0, pi]`, shape `(npoints,)`.
    phi : np.ndarray
        Azimuthal angle in `[-pi, pi]`, shape `(npoints,)`.
    weights : np.ndarray
        Quadrature weights summing to 4 pi, shape `(npoints,)`.
    """
    from pyscf.dft import LebedevGrid

    grid = np.asarray(LebedevGrid.MakeAngularGrid(npoints), dtype=float)
    xyz = grid[:, :3]
    weights = grid[:, 3] * 4.0 * np.pi
    theta = np.arccos(np.clip(xyz[:, 2], -1.0, 1.0))
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    return theta, phi, weights


def supported_lebedev_orders() -> list[int]:
    """
    List supported Lebedev point counts.

    Returns
    -------
    list[int]
        Allowed values for `npoints` in `lebedev_sphere`.
    """
    from pyscf.dft import LebedevGrid

    return sorted(int(n) for n in LebedevGrid.LEBEDEV_NGRID)
