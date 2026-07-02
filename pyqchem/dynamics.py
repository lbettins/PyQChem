"""Internal-coordinate Langevin dynamics for embedded substrate molecules."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pyqchem.internal_coords import (
    InternalCoordinateAnalysis,
    InternalCoordinateKind,
    InternalCoordinateSystem,
    InternalMode,
)
from pyqchem.structure import EmbeddedSystem

HARTREE_TO_KJ_MOL = 2625.499638


@dataclass
class DynamicsSettings:
    """
    Langevin dynamics parameters in internal-coordinate space.

    Parameters
    ----------
    timestep_fs : float
        Integration timestep in femtoseconds.
    temperature_k : float
        Target bath temperature.
    friction_ps : float
        Friction coefficient in 1/ps applied to each internal mode.
    host_spring_k : float
        Harmonic spring constant (kJ/mol/Angstrom^2) tethering substrate atoms to host.
    host_cutoff_a : float
        Cutoff radius in Angstrom for host-substrate interactions.
    min_frequency_cm1 : float
        Minimum frequency assigned to soft or null modes for stable integration.
    """

    timestep_fs: float = 1.0
    temperature_k: float = 300.0
    friction_ps: float = 1.0
    host_spring_k: float = 50.0
    host_cutoff_a: float = 4.0
    min_frequency_cm1: float = 50.0


@dataclass
class InternalDynamicsState:
    """
    State during internal-coordinate dynamics.

    Parameters
    ----------
    mode_displacements : np.ndarray
        Normal-mode amplitudes in internal space, shape `(n_modes,)`.
    mode_velocities : np.ndarray
        Normal-mode velocities, shape `(n_modes,)`.
    internal_values : np.ndarray
        Current primitive internal coordinate values.
    cartesian_positions : np.ndarray
        Substrate-atom Cartesian coordinates, shape `(n_substrate, 3)`.
    step : int
        Integrator step counter.
    time_fs : float
        Elapsed simulation time in femtoseconds.
    """

    mode_displacements: np.ndarray
    mode_velocities: np.ndarray
    internal_values: np.ndarray
    cartesian_positions: np.ndarray
    step: int = 0
    time_fs: float = 0.0


@dataclass
class DynamicsState:
    """
    Cartesian view of the dynamics state for trajectory output.

    Parameters
    ----------
    positions : np.ndarray
        Shape `(n_substrate, 3)` in Angstrom.
    velocities : np.ndarray
        Cartesian velocities estimated from internal mode velocities.
    internal_state : InternalDynamicsState
        Full internal-coordinate state.
    step : int
        Integrator step counter.
    time_fs : float
        Elapsed simulation time in femtoseconds.
    """

    positions: np.ndarray
    velocities: np.ndarray
    internal_state: InternalDynamicsState
    step: int = 0
    time_fs: float = 0.0


class EmbeddedDynamics:
    """
    Evolve a substrate molecule in separable internal coordinates.

    Normal modes are obtained by projecting the molecular Hessian into the
    Wilson internal-coordinate basis. Each mode is classified as a stretch,
    bend, torsion, or rotation and integrated with its own harmonic frequency.

    Parameters
    ----------
    system : EmbeddedSystem
        Embedded substrate molecule and fixed host framework.
    hessian_cart : np.ndarray
        Cartesian Hessian from DFT, shape `(3 * n_atoms, 3 * n_atoms)`.
    settings : DynamicsSettings
        Integrator and bath parameters.
    angstrom_hessian : bool
        Whether `hessian_cart` uses Angstrom (False for PySCF Bohr-based Hessians).
    seed : int
        Random seed for stochastic forces.
    """

    def __init__(
        self,
        system: EmbeddedSystem,
        hessian_cart: np.ndarray,
        settings: DynamicsSettings | None = None,
        angstrom_hessian: bool = False,
        seed: int = 0,
    ) -> None:
        self.system = system
        self.settings = settings or DynamicsSettings()
        self.rng = np.random.default_rng(seed)
        self._substrate_idx = system.substrate.substrate_indices()
        self._ics = InternalCoordinateSystem(system.substrate, include_rotations=True)
        self._analysis = self._ics.project_hessian(hessian_cart, angstrom_hessian=angstrom_hessian)
        self._mode_matrix = np.column_stack([mode.displacement_vector for mode in self._analysis.modes])
        self._equilibrium_internal = self._analysis.internal_values.copy()
        self._omega_au = self._mode_frequencies_au()
        self._x0_cartesian = system.substrate.positions.copy()

        substrate_positions = system.substrate.positions[self._substrate_idx]
        internal_state = InternalDynamicsState(
            mode_displacements=np.zeros(len(self._analysis.modes)),
            mode_velocities=np.zeros(len(self._analysis.modes)),
            internal_values=self._analysis.internal_values.copy(),
            cartesian_positions=substrate_positions.copy(),
        )
        self.state = DynamicsState(
            positions=substrate_positions.copy(),
            velocities=np.zeros_like(substrate_positions),
            internal_state=internal_state,
        )

    @property
    def internal_analysis(self) -> InternalCoordinateAnalysis:
        """Internal-coordinate decomposition with Hessian-derived mode frequencies."""
        return self._analysis

    def modes_by_kind(self) -> dict[InternalCoordinateKind, list[InternalMode]]:
        """Group separable modes into stretches, bends, torsions, and rotations."""
        return self._ics.modes_by_kind(self._analysis)

    def _mode_frequencies_au(self) -> np.ndarray:
        """Return mode frequencies in atomic units from Hessian force constants."""
        cm1_to_au = 2.998e10 * 100.0 * 6.62607015e-34 / 4.3597447222071e-18 / (2.0 * np.pi)
        min_omega = self.settings.min_frequency_cm1 * cm1_to_au
        omegas: list[float] = []
        for mode in self._analysis.modes:
            if mode.force_constant_au > 1e-8:
                omega = float(np.sqrt(mode.force_constant_au))
            else:
                omega = min_omega
            if mode.kind == InternalCoordinateKind.ROTATION and mode.frequency_cm1 < 1.0:
                omega = min_omega
            omegas.append(max(omega, min_omega))
        return np.array(omegas, dtype=float)

    def _host_cartesian_forces(self, positions: np.ndarray) -> np.ndarray:
        """Compute Cartesian repulsion forces from the fixed host on substrate atoms."""
        host_positions = self.system.host.positions
        forces = np.zeros_like(positions)
        cutoff = self.settings.host_cutoff_a
        k = self.settings.host_spring_k * 0.001
        for i, pos in enumerate(positions):
            deltas = host_positions - pos
            distances = np.linalg.norm(deltas, axis=1)
            mask = (distances > 0.0) & (distances < cutoff)
            if not np.any(mask):
                continue
            dirs = deltas[mask] / distances[mask, None]
            magnitudes = k * (cutoff - distances[mask]) / cutoff
            forces[i] += np.sum(dirs * magnitudes[:, None], axis=0)
        return forces

    def _cartesian_to_internal_forces(self, cartesian_forces: np.ndarray) -> np.ndarray:
        """
        Project Cartesian forces onto internal normal modes.

        Parameters
        ----------
        cartesian_forces : np.ndarray
            Shape `(n_substrate, 3)`.

        Returns
        -------
        np.ndarray
            Generalized forces in mode space, shape `(n_modes,)`.
        """
        full_force = np.zeros(3 * len(self.system.substrate.atoms))
        substrate_forces = cartesian_forces
        for local_idx, atom_idx in enumerate(self._substrate_idx):
            full_force[3 * atom_idx : 3 * atom_idx + 3] = substrate_forces[local_idx]
        b_mat = self._analysis.b_matrix
        g_mat = self._analysis.g_matrix
        g_inv = np.linalg.pinv(g_mat, rcond=1e-8)
        generalized = b_mat @ full_force
        return self._mode_matrix.T @ (g_inv @ generalized)

    def _mode_displacement_to_internal(self, mode_displacements: np.ndarray) -> np.ndarray:
        """Expand normal-mode amplitudes into primitive internal displacements."""
        return self._mode_matrix @ mode_displacements

    def _harmonic_internal_forces(self, mode_displacements: np.ndarray) -> np.ndarray:
        """Restoring forces from Hessian-derived mode force constants."""
        return -(self._omega_au**2) * mode_displacements

    def step(self) -> DynamicsState:
        """
        Advance one Langevin step in internal normal-mode coordinates.

        Returns
        -------
        DynamicsState
            Updated Cartesian and internal state.
        """
        settings = self.settings
        dt_fs = settings.timestep_fs
        dt_au = dt_fs * 41.341374575751
        friction = settings.friction_ps * 1.0e-3
        k_b_au = 3.166811563e-6

        q = self.state.internal_state.mode_displacements
        qdot = self.state.internal_state.mode_velocities

        f_harm = self._harmonic_internal_forces(q)
        cart_forces = self._host_cartesian_forces(self.state.positions)
        f_host = self._cartesian_to_internal_forces(cart_forces)
        f_total = f_harm + f_host

        noise = self.rng.normal(size=q.shape) * np.sqrt(2.0 * friction * k_b_au * settings.temperature_k * dt_au)
        qddot = f_total - friction * qdot + noise
        qdot = qdot + qddot * dt_au
        q = q + qdot * dt_au

        dq_internal = self._mode_displacement_to_internal(q)
        target_internal = self._equilibrium_internal + dq_internal
        delta_internal = dq_internal
        new_full = self._ics.internal_displacement_to_cartesian(delta_internal, self._x0_cartesian)
        new_substrate = new_full[self._substrate_idx]
        self.system.substrate.positions[self._substrate_idx] = new_substrate

        internal_state = InternalDynamicsState(
            mode_displacements=q,
            mode_velocities=qdot,
            internal_values=target_internal,
            cartesian_positions=new_substrate.copy(),
            step=self.state.step + 1,
            time_fs=self.state.time_fs + dt_fs,
        )
        cart_vel = (new_substrate - self.state.positions) / max(dt_fs, 1e-12)
        self.state = DynamicsState(
            positions=new_substrate.copy(),
            velocities=cart_vel,
            internal_state=internal_state,
            step=internal_state.step,
            time_fs=internal_state.time_fs,
        )
        return self.state

    def run(self, n_steps: int) -> list[DynamicsState]:
        """
        Run multiple internal-coordinate dynamics steps.

        Parameters
        ----------
        n_steps : int
            Number of integration steps.

        Returns
        -------
        list[DynamicsState]
            State after each step.
        """
        return [self.step() for _ in range(n_steps)]
