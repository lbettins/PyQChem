"""Command-line interface for PyQChem."""

from __future__ import annotations

import argparse
from pathlib import Path

from pyqchem.dft import DFTSettings
from pyqchem.dynamics import DynamicsSettings
from pyqchem.orchestrator import SimulationConfig, SimulationOrchestrator
from pyqchem.structure import EmbeddedSystem, build_silica_tetrahedron
from pyqchem.xyz import build_water, write_xyz


def main() -> None:
    """Run a demo embedded-water simulation pipeline."""
    parser = argparse.ArgumentParser(description="PyQChem molecular simulator")
    parser.add_argument("--temperature", type=float, default=300.0, help="Temperature in Kelvin")
    parser.add_argument("--steps", type=int, default=20, help="Embedded dynamics steps")
    parser.add_argument("--basis", type=str, default="sto-3g", help="Gaussian basis set")
    parser.add_argument("--functional", type=str, default="B3LYP", help="DFT functional")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Trajectory output directory")
    args = parser.parse_args()

    water = build_water()
    host = build_silica_tetrahedron(center=water.positions[0] + 2.5, scale=1.0)
    system = EmbeddedSystem(substrate=water, host=host)

    config = SimulationConfig(
        temperature_k=args.temperature,
        dft_settings=DFTSettings(functional=args.functional, basis=args.basis, verbose=0),
        dynamics_settings=DynamicsSettings(temperature_k=args.temperature),
        dynamics_steps=args.steps,
        output_dir=args.output_dir,
    )
    orchestrator = SimulationOrchestrator(config)

    initial_xyz = args.output_dir / "initial.xyz"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_xyz(system.as_combined_molecule(), initial_xyz)

    result = orchestrator.run(system)

    print("=== PyQChem demo ===")
    print(f"Initial geometry written to: {initial_xyz}")
    print(f"DFT energy (Ha): {result.initial_dft.energy_hartree:.6f}")
    print(f"SCF converged: {result.initial_dft.converged}")
    print(f"Vibrational modes (cm-1): {result.frequencies.frequencies_cm1[:6]}")
    print(f"Partition function Q: {result.thermodynamics.q_total:.3e}")
    print(f"Entropy (J/mol/K): {result.thermodynamics.entropy_j_mol_k:.2f}")
    print(f"Gibbs energy (kJ/mol): {result.thermodynamics.gibbs_kj_mol:.2f}")
    print(f"Dynamics frames: {len(result.dynamics_trajectory)} written to {args.output_dir}")


if __name__ == "__main__":
    main()
