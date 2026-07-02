"""Rigid translation of molecular geometries."""

from __future__ import annotations

import numpy as np


def translation_cube_grid(n_points_per_axis: int, half_width_a: float) -> np.ndarray:
    """
    Build a cubic grid of `(delta_x, delta_y, delta_z)` displacements.

    Parameters
    ----------
    n_points_per_axis : int
        Number of samples along each axis; total points is `n_points_per_axis**3`.
    half_width_a : float
        Half-width of the cube in Angstrom (`[-half_width, +half_width]` per axis).

    Returns
    -------
    np.ndarray
        Displacements with shape `(n_points_per_axis**3, 3)`.
    """
    if n_points_per_axis < 2:
        raise ValueError("n_points_per_axis must be at least 2")
    if half_width_a <= 0.0:
        raise ValueError("half_width_a must be positive")

    axis = np.linspace(-half_width_a, half_width_a, n_points_per_axis)
    delta_x, delta_y, delta_z = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.column_stack([delta_x.ravel(), delta_y.ravel(), delta_z.ravel()])


def apply_translation(positions: np.ndarray, delta_x: float, delta_y: float, delta_z: float) -> np.ndarray:
    """
    Translate all atoms by a Cartesian displacement vector.

    Parameters
    ----------
    positions : np.ndarray
        Cartesian coordinates, shape `(n_atoms, 3)`.
    delta_x : float
        Displacement along x in Angstrom.
    delta_y : float
        Displacement along y in Angstrom.
    delta_z : float
        Displacement along z in Angstrom.

    Returns
    -------
    np.ndarray
        Translated coordinates, shape `(n_atoms, 3)`.
    """
    displacement = np.array([delta_x, delta_y, delta_z], dtype=float)
    return positions + displacement
