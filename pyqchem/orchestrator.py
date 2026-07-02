"""End-to-end simulation orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pyqchem.dft import DFTCalculator, DFTResult, DFTSettings, FrequencyResult
from pyqchem.dynamics import DynamicsSettings, DynamicsState, EmbeddedDynamics
from pyqchem.statmech import StatisticalMechanicsEstimator, ThermodynamicProperties
from pyqchem.structure import EmbeddedSystem, Molecule
from pyqchem.xyz import write_xyz


@dataclass
class SimulationConfig:
    """
    Configuration for a full PyQChem pipeline run.

    Parameters
    ----------
    temperature_k : float
        Thermodynamic and dynamics temperature.
    volume_m3 : float
        Simulation volume for translational partition functions.
    symmetry_number : int
        Rotational symmetry number for the substrate molecule.
    dft_settings : DFTSettings
        Ground-state DFT configuration.
    dynamics_settings : DynamicsSettings
        Embedded Langevin dynamics settings.
    dynamics_steps : int
        Number of spacetime evolution steps after the initial electronic structure pass.
    output_dir : Path | None
        Optional directory for XYZ trajectory snapshots.
    """

    temperature_k: float = 300.0
    volume_m3: float = 1.0e-24
    symmetry_number: int = 2
    dft_settings: DFTSettings = field(default_factory=DFTSettings)
    dynamics_settings: DynamicsSettings = field(default_factory=DynamicsSettings)
    dynamics_steps: int = 10
    output_dir: Path | None = None


@dataclass
class SimulationResult:
    """
    Aggregated output from an orchestrated simulation.

    Parameters
    ----------
    initial_dft : DFTResult
        Ground-state energy at the starting geometry.
    frequencies : FrequencyResult
        Harmonic vibrational analysis.
    thermodynamics : ThermodynamicProperties
        Statistical mechanical estimates.
    dynamics_trajectory : list[DynamicsState]
        Substrate-atom trajectory inside the fixed host.
    """

    initial_dft: DFTResult
    frequencies: FrequencyResult
    thermodynamics: ThermodynamicProperties
    dynamics_trajectory: list[DynamicsState]


class SimulationOrchestrator:
    """
    Coordinate XYZ generation, DFT, statistical mechanics, and embedded dynamics.

    Parameters
    ----------
    config : SimulationConfig
        Pipeline configuration.
    """

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()

    def generate_xyz(self, system: EmbeddedSystem, path: Path | str) -> Path:
        """
        Write the combined embedded system to XYZ format.

        Parameters
        ----------
        system : EmbeddedSystem
            Substrate molecule plus fixed host.
        path : Path | str
            Output path.

        Returns
        -------
        Path
            Written XYZ file path.
        """
        path = Path(path)
        write_xyz(system.as_combined_molecule(), path, comment=system.as_combined_molecule().name)
        return path

    def run_dft(self, substrate: Molecule) -> tuple[DFTResult, FrequencyResult]:
        """
        Run ground-state DFT and harmonic frequency analysis on the substrate region.

        Parameters
        ----------
        substrate : Molecule
            Substrate molecule geometry.

        Returns
        -------
        tuple[DFTResult, FrequencyResult]
            Ground-state energy and vibrational data.
        """
        calculator = DFTCalculator(self.config.dft_settings)
        dft = calculator.ground_state_energy(substrate)
        freqs = calculator.harmonic_frequencies(substrate)
        return dft, freqs

    def estimate_statmech(
        self,
        substrate: Molecule,
        dft: DFTResult,
        freqs: FrequencyResult,
    ) -> ThermodynamicProperties:
        """
        Estimate RRHO thermodynamic properties for the substrate molecule.

        Parameters
        ----------
        substrate : Molecule
            Substrate molecule geometry.
        dft : DFTResult
            Ground-state energy reference.
        freqs : FrequencyResult
            Vibrational frequencies.

        Returns
        -------
        ThermodynamicProperties
            Thermodynamic estimates at `config.temperature_k`.
        """
        estimator = StatisticalMechanicsEstimator.from_geometry(substrate.symbols, substrate.positions)
        return estimator.estimate(
            temperature_k=self.config.temperature_k,
            frequencies_cm1=freqs.frequencies_cm1,
            volume_m3=self.config.volume_m3,
            symmetry_number=self.config.symmetry_number,
            dft_result=dft,
            frequency_result=freqs,
        )

    def evolve_embedded(
        self,
        system: EmbeddedSystem,
        hessian_cart: np.ndarray,
        n_steps: int | None = None,
    ) -> list[DynamicsState]:
        """
        Evolve the substrate molecule in internal coordinates while the host stays fixed.

        Parameters
        ----------
        system : EmbeddedSystem
            Embedded system to evolve.
        hessian_cart : np.ndarray
            Cartesian Hessian from DFT for mode frequencies and force constants.
        n_steps : int | None
            Override for number of dynamics steps.

        Returns
        -------
        list[DynamicsState]
            Trajectory of substrate atom states.
        """
        steps = self.config.dynamics_steps if n_steps is None else n_steps
        engine = EmbeddedDynamics(
            system,
            hessian_cart,
            self.config.dynamics_settings,
            angstrom_hessian=False,
        )
        trajectory = engine.run(steps)
        if self.config.output_dir is not None:
            out_dir = Path(self.config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for state in trajectory:
                write_xyz(
                    system.as_combined_molecule(),
                    out_dir / f"frame_{state.step:05d}.xyz",
                    comment=f"t={state.time_fs:.2f} fs",
                )
        return trajectory

    def run(self, system: EmbeddedSystem) -> SimulationResult:
        """
        Execute the full pipeline on an embedded system.

        Steps:
        1. Ground-state DFT on the substrate molecule
        2. Harmonic frequency analysis
        3. RRHO thermodynamic property estimation
        4. Internal-coordinate Langevin evolution of substrate atoms in the fixed host

        Parameters
        ----------
        system : EmbeddedSystem
            Embedded substrate molecule and host framework.

        Returns
        -------
        SimulationResult
            Aggregated simulation outputs.
        """
        dft, freqs = self.run_dft(system.substrate)
        thermo = self.estimate_statmech(system.substrate, dft, freqs)
        trajectory = self.evolve_embedded(system, freqs.hessian)
        return SimulationResult(
            initial_dft=dft,
            frequencies=freqs,
            thermodynamics=thermo,
            dynamics_trajectory=trajectory,
        )
