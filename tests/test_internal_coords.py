"""Tests for internal-coordinate conversion and dynamics."""

from __future__ import annotations

import numpy as np
import pytest

from pyqchem.dynamics import EmbeddedDynamics, DynamicsSettings
from pyqchem.internal_coords import InternalCoordinateKind, InternalCoordinateSystem
from pyqchem.structure import EmbeddedSystem, HostFramework
from pyqchem.xyz import build_water


def _dummy_hessian(n_atoms: int, scale: float = 0.02) -> np.ndarray:
    """Build a positive-definite Cartesian Hessian for testing."""
    n = 3 * n_atoms
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(n, n))
    return scale * (mat.T @ mat + np.eye(n))


def test_water_internal_coordinates_include_stretch_and_bend() -> None:
    """Water should decompose into O-H stretches and H-O-H bend coordinates."""
    water = build_water()
    ics = InternalCoordinateSystem(water, include_rotations=True)
    kinds = {p.kind for p in ics.primitives}
    assert InternalCoordinateKind.STRETCH in kinds
    assert InternalCoordinateKind.BEND in kinds
    assert len([p for p in ics.primitives if p.kind == InternalCoordinateKind.STRETCH]) == 2


def test_cartesian_internal_roundtrip_linear() -> None:
    """Wilson B-matrix should relate Cartesian and internal displacements linearly."""
    water = build_water()
    ics = InternalCoordinateSystem(water, include_rotations=False)
    q0 = ics.cartesian_to_internal(water.positions)
    dq = np.zeros_like(q0)
    if dq.size:
        dq[0] = 1e-4
    new_positions = ics.internal_displacement_to_cartesian(dq, water.positions)
    dx_flat = (new_positions - water.positions).reshape(-1)
    dq_linear = ics.wilson_b_matrix(water.positions) @ dx_flat
    np.testing.assert_allclose(dq_linear, dq, atol=1e-5)


def test_hessian_projection_assigns_frequencies() -> None:
    """Projected internal modes should carry non-negative Hessian-derived frequencies."""
    water = build_water()
    ics = InternalCoordinateSystem(water, include_rotations=True)
    hess = _dummy_hessian(len(water.atoms))
    analysis = ics.project_hessian(hess, angstrom_hessian=True)
    assert len(analysis.modes) == len(ics.primitives)
    assert all(mode.frequency_cm1 >= 0.0 for mode in analysis.modes)
    grouped = ics.modes_by_kind(analysis)
    assert len(grouped[InternalCoordinateKind.STRETCH]) >= 1


def test_internal_dynamics_preserves_host() -> None:
    """Internal-coordinate dynamics must leave the host framework fixed."""
    water = build_water()
    host = HostFramework.from_positions(
        ["Si", "O"],
        np.array([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]]),
    )
    system = EmbeddedSystem(substrate=water, host=host)
    hess = _dummy_hessian(len(water.atoms))
    engine = EmbeddedDynamics(system, hess, DynamicsSettings(timestep_fs=0.5), angstrom_hessian=True, seed=1)
    initial_host = system.host.positions.copy()
    engine.run(5)
    np.testing.assert_allclose(system.host.positions, initial_host)
    assert engine.internal_analysis.modes
