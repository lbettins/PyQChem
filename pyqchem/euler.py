"""ZYZ Euler angles for molecular orientation."""

from __future__ import annotations

import numpy as np

from pyqchem.internal_coords import _principal_axes


def rotation_matrix_z(angle: float) -> np.ndarray:
    """
    Build a rotation matrix about the laboratory z-axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        Shape `(3, 3)`.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def rotation_matrix_y(angle: float) -> np.ndarray:
    """
    Build a rotation matrix about the y-axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        Shape `(3, 3)`.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def rotation_matrix_zyz(phi: float, theta: float, chi: float) -> np.ndarray:
    """
    Build the ZYZ active rotation matrix `R_z(phi) R_y(theta) R_z(chi)`.

    Parameters
    ----------
    phi : float
        Azimuthal angle in radians.
    theta : float
        Polar angle from the north pole in radians.
    chi : float
        Intrinsic spin angle in radians.

    Returns
    -------
    np.ndarray
        Shape `(3, 3)`.
    """
    from scipy.spatial.transform import Rotation as SciRotation

    return SciRotation.from_euler("zyz", [chi, theta, phi]).as_matrix()


def euler_zyz_from_matrix(rotation: np.ndarray) -> tuple[float, float, float]:
    """
    Extract ZYZ Euler angles `(phi, theta, chi)` from a rotation matrix.

    Parameters
    ----------
    rotation : np.ndarray
        Shape `(3, 3)`.

    Returns
    -------
    phi : float
        Azimuthal angle in radians.
    theta : float
        Polar angle from the north pole in radians.
    chi : float
        Intrinsic spin angle in radians.
    """
    from scipy.spatial.transform import Rotation as SciRotation

    chi, theta, phi = SciRotation.from_matrix(np.asarray(rotation, dtype=float)).as_euler("zyz")
    return float(phi), float(theta), float(chi)


def euler_angles_from_geometry(
    positions: np.ndarray,
    masses: np.ndarray,
    reference_axes: np.ndarray,
) -> tuple[float, float, float]:
    """
    Compute `(phi, theta, chi)` for a geometry relative to a reference frame.

    Parameters
    ----------
    positions : np.ndarray
        Shape `(n_atoms, 3)`.
    masses : np.ndarray
        Atomic masses in amu, shape `(n_atoms,)`.
    reference_axes : np.ndarray
        Reference principal axes, shape `(3, 3)`.

    Returns
    -------
    phi : float
        Azimuthal angle in radians.
    theta : float
        Polar angle from the north pole in radians.
    chi : float
        Intrinsic spin angle in radians.
    """
    current_axes = _principal_axes(positions, masses)
    rotation = current_axes @ reference_axes.T
    return euler_zyz_from_matrix(rotation)


def apply_euler_rotation(
    positions: np.ndarray,
    masses: np.ndarray,
    phi: float,
    theta: float,
    chi: float,
) -> np.ndarray:
    """
    Rotate a molecule about its center of mass using ZYZ Euler angles.

    Parameters
    ----------
    positions : np.ndarray
        Cartesian coordinates, shape `(n_atoms, 3)`.
    masses : np.ndarray
        Atomic masses in amu, shape `(n_atoms,)`.
    phi : float
        Azimuthal angle in radians.
    theta : float
        Polar angle from the north pole in radians.
    chi : float
        Intrinsic spin angle in radians.

    Returns
    -------
    np.ndarray
        Rotated coordinates, shape `(n_atoms, 3)`.
    """
    com = np.average(positions, axis=0, weights=masses)
    relative = positions - com
    rotation = rotation_matrix_zyz(phi, theta, chi)
    return com + relative @ rotation.T
