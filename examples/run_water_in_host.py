"""Example: embed water in a silica host and run the full pipeline."""

from pathlib import Path

from pyqchem.orchestrator import SimulationConfig, SimulationOrchestrator
from pyqchem.structure import EmbeddedSystem, build_silica_tetrahedron
from pyqchem.xyz import build_water, write_xyz


def main() -> None:
    """
    Generate XYZ coordinates, run DFT, estimate thermodynamics, and evolve dynamics.

    Outputs are written to `examples/output/`.
    """
    water = build_water()
    host = build_silica_tetrahedron(center=water.positions[0] + 2.5)
    system = EmbeddedSystem(substrate=water, host=host)

    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_xyz(system.as_combined_molecule(), output_dir / "initial.xyz")

    config = SimulationConfig(dynamics_steps=10, output_dir=output_dir)
    result = SimulationOrchestrator(config).run(system)

    print(f"DFT energy (Ha): {result.initial_dft.energy_hartree:.6f}")
    print(f"Entropy (J/mol/K): {result.thermodynamics.entropy_j_mol_k:.2f}")


if __name__ == "__main__":
    main()
