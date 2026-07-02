"""Tests for cubic translation sampling."""

from __future__ import annotations

import numpy as np
import pytest

from pyqchem.internal_coords import InternalCoordinateSystem
from pyqchem.sampling import InternalModeSampler, SamplingSettings
from pyqchem.translation import apply_translation, translation_cube_grid
from pyqchem.xyz import build_water


def _dummy_hessian(n_atoms: int, scale: float = 0.02) -> np.ndarray:
    """Build a positive-definite Cartesian Hessian for testing."""
    n = 3 * n_atoms
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(n, n))
    return scale * (mat.T @ mat + np.eye(n))


def test_translation_cube_grid_shape_and_bounds() -> None:
    """A cubic grid should contain n**3 points within +/- half_width."""
    n = 5
    half_width = 0.4
    grid = translation_cube_grid(n, half_width)
    assert grid.shape == (n**3, 3)
    assert np.max(np.abs(grid)) == pytest.approx(half_width)
    assert grid[0, 0] == pytest.approx(-half_width)
    assert grid[-1, 0] == pytest.approx(half_width)


def test_apply_translation_shifts_all_atoms() -> None:
    """Translation should add the same vector to every atom."""
    water = build_water()
    delta = np.array([0.2, -0.1, 0.3])
    translated = apply_translation(water.positions, *delta)
    for row in translated - water.positions:
        np.testing.assert_allclose(row, delta)


def test_sample_translations_cube_size() -> None:
    """Translation sampling should produce n_translation**3 samples."""
    water = build_water()
    ics = InternalCoordinateSystem(water, include_rotations=False)
    analysis = ics.project_hessian(_dummy_hessian(len(water.atoms)), angstrom_hessian=True)
    settings = SamplingSettings(n_translation=4, translation_window_a=0.3)
    sampler = InternalModeSampler(water, analysis, reference_energy_hartree=-76.0)
    result = sampler.sample_translations(settings)
    assert len(result.samples) == 4**3
    assert result.samples[0].delta_x == pytest.approx(-0.3)


def test_harmonic_translation_minimum_at_origin() -> None:
    """Harmonic translation energy should be lowest at zero displacement."""
    water = build_water()
    ics = InternalCoordinateSystem(water, include_rotations=False)
    analysis = ics.project_hessian(_dummy_hessian(len(water.atoms)), angstrom_hessian=True)
    settings = SamplingSettings(
        n_translation=3,
        translation_window_a=0.2,
        translation_force_constant_au=0.05,
    )
    sampler = InternalModeSampler(water, analysis, reference_energy_hartree=-76.0)
    result = sampler.sample_translations(settings)
    energies = np.array([sample.energy_hartree for sample in result.samples])
    origin_idx = np.argmin(np.linalg.norm(np.array([[s.delta_x, s.delta_y, s.delta_z] for s in result.samples]), axis=1))
    assert energies[origin_idx] == pytest.approx(min(energies))
