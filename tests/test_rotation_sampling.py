"""Tests for Euler-angle rotation sampling."""

from __future__ import annotations

import numpy as np
import pytest

from pyqchem.euler import apply_euler_rotation, euler_zyz_from_matrix, rotation_matrix_zyz
from pyqchem.internal_coords import InternalCoordinateKind, InternalCoordinateSystem
from pyqchem.lebedev import lebedev_sphere, supported_lebedev_orders
from pyqchem.sampling import InternalModeSampler, SamplingSettings, chi_grid
from pyqchem.xyz import build_water


def _dummy_hessian(n_atoms: int, scale: float = 0.02) -> np.ndarray:
    """Build a positive-definite Cartesian Hessian for testing."""
    n = 3 * n_atoms
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(n, n))
    return scale * (mat.T @ mat + np.eye(n))


def test_lebedev_theta_and_phi_ranges() -> None:
    """Lebedev nodes should cover theta in [0, pi] and phi in [-pi, pi]."""
    theta, phi, weights = lebedev_sphere(26)
    assert np.all(theta >= 0.0)
    assert np.all(theta <= np.pi)
    assert np.all(phi >= -np.pi)
    assert np.all(phi <= np.pi)
    assert weights.size == 26
    assert weights.sum() == pytest.approx(4.0 * np.pi, rel=1e-6)


def test_chi_grid_spans_minus_pi_to_pi() -> None:
    """Chi should be equally spaced from -pi to pi."""
    chi = chi_grid(37)
    assert chi[0] == pytest.approx(-np.pi)
    assert chi[-1] == pytest.approx(np.pi)
    assert chi.size == 37


def test_euler_identity_rotation() -> None:
    """Zero Euler angles should leave geometry unchanged."""
    water = build_water()
    masses = np.array([15.999, 1.008, 1.008])
    rotated = apply_euler_rotation(water.positions, masses, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(rotated, water.positions, atol=1e-10)


def test_euler_roundtrip_matrix() -> None:
    """Extracted Euler angles should reconstruct the same rotation matrix."""
    phi, theta, chi = 0.3, 1.1, -0.8
    rotation = rotation_matrix_zyz(phi, theta, chi)
    phi2, theta2, chi2 = euler_zyz_from_matrix(rotation)
    np.testing.assert_allclose(
        rotation_matrix_zyz(phi2, theta2, chi2),
        rotation,
        atol=1e-10,
    )


def test_sample_rotations_grid_size() -> None:
    """Rotation samples should equal Lebedev nodes times chi count."""
    water = build_water()
    ics = InternalCoordinateSystem(water, include_rotations=True)
    analysis = ics.project_hessian(_dummy_hessian(len(water.atoms)), angstrom_hessian=True)
    settings = SamplingSettings(lebedev_points=6, n_chi=8)
    sampler = InternalModeSampler(water, analysis)
    result = sampler.sample_rotations(settings)
    assert len(result.samples) == 6 * 8
    assert result.samples[0].theta >= 0.0
    assert 6 in supported_lebedev_orders()


def test_rotation_primitives_use_theta_phi_chi_labels() -> None:
    """Internal rotation primitives should be labeled rot_theta, rot_phi, rot_chi."""
    water = build_water()
    ics = InternalCoordinateSystem(water, include_rotations=True)
    labels = [p.label for p in ics.primitives if p.kind == InternalCoordinateKind.ROTATION]
    assert labels == ["rot_theta", "rot_phi", "rot_chi"]
