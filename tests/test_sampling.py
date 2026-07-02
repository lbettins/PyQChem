"""Tests for internal-coordinate energy sampling."""

from __future__ import annotations

import numpy as np
import pytest

from pyqchem.internal_coords import InternalCoordinateKind, InternalCoordinateSystem
from pyqchem.sampling import (
    EnergyMethod,
    InternalModeSampler,
    SamplingSettings,
    coordinate_grid,
    positions_at_coordinate_value,
)
from pyqchem.xyz import build_water


def _dummy_hessian(n_atoms: int, scale: float = 0.02) -> np.ndarray:
    """Build a positive-definite Cartesian Hessian for testing."""
    n = 3 * n_atoms
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(n, n))
    return scale * (mat.T @ mat + np.eye(n))


def test_torsion_grid_spans_full_dihedral_range() -> None:
    """Torsion scans should cover the configured dihedral range."""
    from pyqchem.internal_coords import PrimitiveInternalCoordinate

    torsion = PrimitiveInternalCoordinate(
        kind=InternalCoordinateKind.TORSION,
        indices=(0, 1, 2, 3),
        value=0.0,
        equilibrium=0.0,
        label="phi(0-1-2-3)",
    )
    settings = SamplingSettings(n_points=37, torsion_start_rad=-np.pi, torsion_end_rad=np.pi)
    grid = coordinate_grid(torsion, settings)
    assert grid[0] == pytest.approx(-np.pi)
    assert grid[-1] == pytest.approx(np.pi)
    assert grid.size == 37


def test_water_stretch_slice_has_harmonic_minimum_at_equilibrium() -> None:
    """Harmonic stretch slices should be lowest at zero displacement."""
    water = build_water()
    ics = InternalCoordinateSystem(water, include_rotations=False)
    analysis = ics.project_hessian(_dummy_hessian(len(water.atoms)), angstrom_hessian=True)
    sampler = InternalModeSampler(water, analysis, reference_energy_hartree=-76.0)
    stretch_idx = next(
        idx for idx, p in enumerate(analysis.primitives) if p.kind == InternalCoordinateKind.STRETCH
    )
    slice_ = sampler.sample_primitive(stretch_idx, SamplingSettings(n_points=11))
    displacements = np.array([sample.displacement for sample in slice_.samples])
    energies = np.array([sample.energy_hartree for sample in slice_.samples])
    assert displacements[np.argmin(energies)] == pytest.approx(0.0, abs=1e-6)
    assert slice_.kind == InternalCoordinateKind.STRETCH


def test_sample_all_groups_by_kind() -> None:
    """sample_all should return stretch and bend slices for water."""
    water = build_water()
    ics = InternalCoordinateSystem(water, include_rotations=False)
    analysis = ics.project_hessian(_dummy_hessian(len(water.atoms)), angstrom_hessian=True)
    sampler = InternalModeSampler(water, analysis)
    result = sampler.sample_all(SamplingSettings(n_points=5))
    assert len(result.by_kind[InternalCoordinateKind.STRETCH]) == 2
    assert len(result.by_kind[InternalCoordinateKind.BEND]) == 1
    assert all(len(slice_.samples) == 5 for slice_ in result.slices)


def test_positions_at_stretch_updates_bond_length() -> None:
    """Setting a stretch target should produce the requested bond length."""
    water = build_water()
    ics = InternalCoordinateSystem(water, include_rotations=False)
    stretch = next(p for p in ics.primitives if p.kind == InternalCoordinateKind.STRETCH)
    target = stretch.equilibrium + 0.05
    new_positions = positions_at_coordinate_value(water.positions, stretch, target, ics)
    i, j = stretch.indices
    length = np.linalg.norm(new_positions[j] - new_positions[i])
    assert length == pytest.approx(target, abs=1e-6)


def test_torsion_positions_change_dihedral() -> None:
    """Torsion sampling geometry should move the dihedral toward the target."""
    from pyqchem.internal_coords import _dihedral
    from pyqchem.structure import Atom, Molecule

    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 1.0, 0.5],
        ],
        dtype=float,
    )
    molecule = Molecule(
        atoms=[
            Atom("C", positions[0]),
            Atom("C", positions[1]),
            Atom("C", positions[2]),
            Atom("H", positions[3]),
        ]
    )
    ics = InternalCoordinateSystem(molecule, include_rotations=False)
    torsion = next((p for p in ics.primitives if p.kind == InternalCoordinateKind.TORSION), None)
    assert torsion is not None
    target = np.pi / 2.0
    new_positions = positions_at_coordinate_value(molecule.positions, torsion, target, ics)
    i, j, k, l = torsion.indices
    assert _dihedral(new_positions, i, j, k, l) == pytest.approx(target, abs=1e-3)
