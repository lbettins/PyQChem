"""Tests for PyQChem."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyqchem.dynamics import EmbeddedDynamics, DynamicsSettings
from pyqchem.statmech import StatisticalMechanicsEstimator
from pyqchem.structure import EmbeddedSystem, HostFramework, build_silica_tetrahedron
from pyqchem.xyz import build_water, read_xyz, write_xyz


def _dummy_hessian(n_atoms: int, scale: float = 0.02) -> np.ndarray:
    """Build a positive-definite Cartesian Hessian for testing."""
    n = 3 * n_atoms
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(n, n))
    return scale * (mat.T @ mat + np.eye(n))


def test_water_geometry() -> None:
    """Water should have one oxygen and two hydrogens."""
    water = build_water()
    assert water.symbols.count("O") == 1
    assert water.symbols.count("H") == 2
    assert water.positions.shape == (3, 3)


def test_xyz_roundtrip(tmp_path: Path) -> None:
    """XYZ write/read should preserve coordinates and mobility flags."""
    water = build_water()
    water.atoms[0].substrate = True
    water.atoms[1].substrate = False
    path = tmp_path / "water.xyz"
    write_xyz(water, path)
    loaded = read_xyz(path)
    assert loaded.symbols == water.symbols
    np.testing.assert_allclose(loaded.positions, water.positions)
    assert loaded.atoms[1].substrate is False


def test_embedded_system_slices() -> None:
    """Combined embedded system should preserve host immobility."""
    water = build_water()
    host = build_silica_tetrahedron(center=np.array([3.0, 0.0, 0.0]))
    system = EmbeddedSystem(substrate=water, host=host)
    combined = system.as_combined_molecule()
    assert len(combined.atoms) == len(water.atoms) + len(host.atoms)
    assert combined.fixed_indices() == list(range(len(water.atoms), len(combined.atoms)))


def test_host_positions_remain_fixed_during_dynamics() -> None:
    """Host framework atoms must not move during embedded dynamics."""
    water = build_water()
    host = HostFramework.from_positions(
        ["Si", "O"],
        np.array([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0]]),
    )
    system = EmbeddedSystem(substrate=water, host=host)
    initial_host = system.host.positions.copy()
    engine = EmbeddedDynamics(
        system,
        _dummy_hessian(len(water.atoms)),
        DynamicsSettings(timestep_fs=0.5),
        angstrom_hessian=True,
        seed=1,
    )
    engine.run(5)
    np.testing.assert_allclose(system.host.positions, initial_host)


def test_translational_partition_scales_with_volume() -> None:
    """Translational Q should double when volume doubles."""
    estimator = StatisticalMechanicsEstimator(
        molecular_mass_amu=18.015,
        principal_moments_amu_a2=np.array([0.0, 0.0, 0.0]),
    )
    q1 = estimator.translational_partition(300.0, 1.0e-24)
    q2 = estimator.translational_partition(300.0, 2.0e-24)
    assert q2 == pytest.approx(2.0 * q1)


def test_vibrational_partition_increases_with_temperature() -> None:
    """Higher temperature should increase the vibrational partition function."""
    freqs = np.array([1500.0, 3500.0, 3600.0])
    q_cold = StatisticalMechanicsEstimator.vibrational_partition(freqs, 100.0)
    q_hot = StatisticalMechanicsEstimator.vibrational_partition(freqs, 600.0)
    assert q_hot > q_cold


@pytest.mark.slow
def test_water_dft_pipeline() -> None:
    """Full DFT + statmech pipeline should run for a small water molecule."""
    pytest.importorskip("pyscf")
    from pyqchem.dft import DFTCalculator, DFTSettings
    from pyqchem.orchestrator import SimulationConfig, SimulationOrchestrator

    water = build_water()
    host = build_silica_tetrahedron(center=water.positions[0] + 2.0)
    system = EmbeddedSystem(substrate=water, host=host)
    config = SimulationConfig(
        dft_settings=DFTSettings(basis="sto-3g", functional="B3LYP"),
        dynamics_steps=2,
        output_dir=None,
    )
    result = SimulationOrchestrator(config).run(system)
    assert result.initial_dft.converged
    assert result.initial_dft.energy_hartree < 0.0
    assert result.frequencies.frequencies_cm1.size > 0
    assert result.thermodynamics.q_total > 0.0
    assert len(result.dynamics_trajectory) == 2
