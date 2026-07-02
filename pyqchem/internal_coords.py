"""Internal coordinates, Wilson B-matrix, and Hessian-derived mode frequencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from pyqchem.structure import Molecule

COVALENT_RADII_A: dict[str, float] = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "Si": 1.11,
    "S": 1.05,
    "F": 0.57,
    "Cl": 1.02,
}

DEFAULT_MASS_AMU: dict[str, float] = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "Si": 28.086,
    "S": 32.06,
    "F": 18.998,
    "Cl": 35.45,
}


class InternalCoordinateKind(str, Enum):
    """Classification of separable internal coordinates."""

    STRETCH = "stretch"
    BEND = "bend"
    TORSION = "torsion"
    ROTATION = "rotation"


@dataclass
class PrimitiveInternalCoordinate:
    """
    One primitive internal coordinate and its reference value.

    Parameters
    ----------
    kind : InternalCoordinateKind
        Coordinate type (stretch, bend, torsion, or overall rotation).
    indices : tuple[int, ...]
        Atom indices defining the coordinate (2, 3, or 4 atoms).
    value : float
        Current coordinate value (Angstrom for stretches, radians otherwise).
    equilibrium : float
        Reference equilibrium value at the start of dynamics.
    label : str
        Human-readable identifier.
    """

    kind: InternalCoordinateKind
    indices: tuple[int, ...]
    value: float
    equilibrium: float
    label: str = ""


@dataclass
class InternalMode:
    """
    A normal mode expressed in the internal-coordinate basis.

    Parameters
    ----------
    kind : InternalCoordinateKind
        Dominant primitive type for this mode.
    frequency_cm1 : float
        Harmonic frequency from the projected molecular Hessian.
    force_constant_au : float
        Force constant in atomic units (Hartree / rad^2 or Hartree / bohr^2).
    displacement_vector : np.ndarray
        Unit displacement in internal-coordinate space, shape `(n_internal,)`.
    primitive_contributions : dict[str, float]
        Absolute projection weights onto primitive labels.
    """

    kind: InternalCoordinateKind
    frequency_cm1: float
    force_constant_au: float
    displacement_vector: np.ndarray
    primitive_contributions: dict[str, float] = field(default_factory=dict)


@dataclass
class InternalCoordinateAnalysis:
    """
    Full internal-coordinate decomposition with separable mode frequencies.

    Parameters
    ----------
    primitives : list[PrimitiveInternalCoordinate]
        Primitive stretch, bend, torsion, and rotation coordinates.
    modes : list[InternalMode]
        Normal modes in internal space with Hessian-derived frequencies.
    b_matrix : np.ndarray
        Wilson B-matrix, shape `(n_internal, 3 * n_atoms)`.
    g_matrix : np.ndarray
        Kinetic metric `G = B M^{-1} B^T`, shape `(n_internal, n_internal)`.
    internal_values : np.ndarray
        Current internal coordinate values.
    """

    primitives: list[PrimitiveInternalCoordinate]
    modes: list[InternalMode]
    b_matrix: np.ndarray
    g_matrix: np.ndarray
    internal_values: np.ndarray


def _bond_pairs(symbols: list[str], positions: np.ndarray, scale: float = 1.2) -> list[tuple[int, int]]:
    """
    Detect covalent bonds from interatomic distances.

    Parameters
    ----------
    symbols : list[str]
        Element symbols.
    positions : np.ndarray
        Shape `(n_atoms, 3)` in Angstrom.
    scale : float
        Distance multiplier on summed covalent radii.

    Returns
    -------
    list[tuple[int, int]]
        Unique bonded atom pairs with i < j.
    """
    n_atoms = len(symbols)
    pairs: list[tuple[int, int]] = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            ri = COVALENT_RADII_A.get(symbols[i], 0.8)
            rj = COVALENT_RADII_A.get(symbols[j], 0.8)
            dist = np.linalg.norm(positions[j] - positions[i])
            if dist <= scale * (ri + rj):
                pairs.append((i, j))
    return pairs


def _unit(v: np.ndarray) -> np.ndarray:
    """Return a normalized copy of vector v."""
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return np.zeros_like(v)
    return v / norm


def _bond_length(positions: np.ndarray, i: int, j: int) -> float:
    """Bond length between atoms i and j in Angstrom."""
    return float(np.linalg.norm(positions[j] - positions[i]))


def _bond_angle(positions: np.ndarray, i: int, j: int, k: int) -> float:
    """Bond angle at atom j in radians."""
    v1 = positions[i] - positions[j]
    v2 = positions[k] - positions[j]
    cos_theta = np.clip(np.dot(_unit(v1), _unit(v2)), -1.0, 1.0)
    return float(np.arccos(cos_theta))


def _dihedral(positions: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    """Dihedral angle for atoms i-j-k-l in radians."""
    b1 = positions[j] - positions[i]
    b2 = positions[k] - positions[j]
    b3 = positions[l] - positions[k]
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    x = np.dot(_unit(n1), _unit(n2))
    y = np.dot(np.cross(_unit(n1), _unit(n2)), _unit(b2))
    return float(np.arctan2(y, x))


def _principal_axes(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """
    Principal rotation axes as a 3x3 orthonormal matrix.

    Parameters
    ----------
    positions : np.ndarray
        Shape `(n_atoms, 3)`.
    masses : np.ndarray
        Atomic masses in amu.

    Returns
    -------
    np.ndarray
        Columns are principal axes.
    """
    com = np.average(positions, axis=0, weights=masses)
    rel = positions - com
    inertia = np.zeros((3, 3), dtype=float)
    for mass, pos in zip(masses, rel):
        r2 = np.dot(pos, pos)
        inertia += mass * (r2 * np.eye(3) - np.outer(pos, pos))
    _, axes = np.linalg.eigh(inertia)
    return axes


class InternalCoordinateSystem:
    """
    Build and convert between Cartesian and separable internal coordinates.

    Parameters
    ----------
    molecule : Molecule
        Molecular geometry whose substrate atoms define the coordinate system.
    include_rotations : bool
        If True, append three overall rotation coordinates (Euler angles).
    """

    def __init__(self, molecule: Molecule, include_rotations: bool = True) -> None:
        self.molecule = molecule
        self.include_rotations = include_rotations
        self.symbols = molecule.symbols
        self.masses = np.array(
            [DEFAULT_MASS_AMU.get(sym, 12.0) for sym in self.symbols],
            dtype=float,
        )
        self.bonds = _bond_pairs(self.symbols, molecule.positions)
        self._reference_positions = molecule.positions.copy()
        self._reference_axes = _principal_axes(self._reference_positions, self.masses)
        self.primitives = self._build_primitives(molecule.positions)

    def _build_primitives(self, positions: np.ndarray) -> list[PrimitiveInternalCoordinate]:
        """Construct stretch, bend, torsion, and optional rotation primitives."""
        primitives: list[PrimitiveInternalCoordinate] = []
        for i, j in self.bonds:
            value = _bond_length(positions, i, j)
            primitives.append(
                PrimitiveInternalCoordinate(
                    kind=InternalCoordinateKind.STRETCH,
                    indices=(i, j),
                    value=value,
                    equilibrium=value,
                    label=f"r({i}-{j})",
                )
            )

        seen_bends: set[tuple[int, int, int]] = set()
        for j in range(len(self.symbols)):
            neighbors = [
                other
                for a, b in self.bonds
                if j in (a, b)
                for other in (a, b)
                if other != j
            ]
            for idx_a, i in enumerate(neighbors):
                for k in neighbors[idx_a + 1 :]:
                    key = (min(i, k), j, max(i, k))
                    if key in seen_bends:
                        continue
                    seen_bends.add(key)
                    value = _bond_angle(positions, i, j, k)
                    primitives.append(
                        PrimitiveInternalCoordinate(
                            kind=InternalCoordinateKind.BEND,
                            indices=(i, j, k),
                            value=value,
                            equilibrium=value,
                            label=f"theta({i}-{j}-{k})",
                        )
                    )

        for j, k in self.bonds:
            i_candidates = [a if a != j else b for a, b in self.bonds if j in (a, b)]
            l_candidates = [a if a != k else b for a, b in self.bonds if k in (a, b)]
            for i in i_candidates:
                for l in l_candidates:
                    if len({i, j, k, l}) < 4 or i == l:
                        continue
                    value = _dihedral(positions, i, j, k, l)
                    label = f"phi({i}-{j}-{k}-{l})"
                    if any(p.kind == InternalCoordinateKind.TORSION and p.label == label for p in primitives):
                        continue
                    primitives.append(
                        PrimitiveInternalCoordinate(
                            kind=InternalCoordinateKind.TORSION,
                            indices=(i, j, k, l),
                            value=value,
                            equilibrium=value,
                            label=label,
                        )
                    )

        if self.include_rotations:
            for angle_name in ("theta", "phi", "chi"):
                primitives.append(
                    PrimitiveInternalCoordinate(
                        kind=InternalCoordinateKind.ROTATION,
                        indices=(),
                        value=0.0,
                        equilibrium=0.0,
                        label=f"rot_{angle_name}",
                    )
                )
        return primitives

    def wilson_b_matrix(self, positions: np.ndarray | None = None) -> np.ndarray:
        """
        Build the Wilson B-matrix mapping Cartesian displacements to internal ones.

        Parameters
        ----------
        positions : np.ndarray | None
            Shape `(n_atoms, 3)`. Defaults to current molecule positions.

        Returns
        -------
        np.ndarray
            Matrix with shape `(n_internal, 3 * n_atoms)`.
        """
        positions = self.molecule.positions if positions is None else positions
        n_atoms = len(positions)
        rows: list[np.ndarray] = []
        for primitive in self.primitives:
            row = np.zeros(3 * n_atoms, dtype=float)
            if primitive.kind == InternalCoordinateKind.STRETCH:
                i, j = primitive.indices
                rij = positions[j] - positions[i]
                r = np.linalg.norm(rij)
                if r > 1e-12:
                    e = rij / r
                    row[3 * i : 3 * i + 3] = -e
                    row[3 * j : 3 * j + 3] = e
            elif primitive.kind == InternalCoordinateKind.BEND:
                i, j, k = primitive.indices
                vji = positions[i] - positions[j]
                vjk = positions[k] - positions[j]
                rji = np.linalg.norm(vji)
                rjk = np.linalg.norm(vjk)
                if rji > 1e-12 and rjk > 1e-12:
                    e_ji = vji / rji
                    e_jk = vjk / rjk
                    cos_theta = np.clip(np.dot(e_ji, e_jk), -1.0, 1.0)
                    sin_theta = np.sqrt(max(1.0 - cos_theta**2, 1e-12))
                    row[3 * i : 3 * i + 3] = (e_jk - cos_theta * e_ji) / (rji * sin_theta)
                    row[3 * k : 3 * k + 3] = (e_ji - cos_theta * e_jk) / (rjk * sin_theta)
                    row[3 * j : 3 * j + 3] = -row[3 * i : 3 * i + 3] - row[3 * k : 3 * k + 3]
            elif primitive.kind == InternalCoordinateKind.TORSION:
                i, j, k, l = primitive.indices
                b1 = positions[j] - positions[i]
                b2 = positions[k] - positions[j]
                b3 = positions[l] - positions[k]
                n1 = np.cross(b1, b2)
                n2 = np.cross(b2, b3)
                n1_u = _unit(n1)
                n2_u = _unit(n2)
                b2_u = _unit(b2)
                r1 = np.linalg.norm(b1)
                r2 = np.linalg.norm(b2)
                r3 = np.linalg.norm(b3)
                if r1 > 1e-12 and r2 > 1e-12 and r3 > 1e-12:
                    row[3 * i : 3 * i + 3] = (r2 / r1) * np.cross(n1_u, b1) / np.dot(n1, n1_u)
                    row[3 * j : 3 * j + 3] = np.cross(n1_u, b2) / r2 - (r2 / r1 - 1.0) * np.cross(n1_u, b1) / r1
                    row[3 * k : 3 * k + 3] = np.cross(n2_u, b2) / r2 - (r3 / r2 - 1.0) * np.cross(n2_u, b3) / r3
                    row[3 * l : 3 * l + 3] = (r2 / r3) * np.cross(n2_u, b3) / np.dot(n2, n2_u)
            elif primitive.kind == InternalCoordinateKind.ROTATION:
                from pyqchem.euler import apply_euler_rotation, euler_angles_from_geometry

                component = primitive.label.removeprefix("rot_")
                phi, theta, chi = euler_angles_from_geometry(positions, self.masses, self._reference_axes)
                step = 1e-5
                delta = {
                    "phi": (step, 0.0, 0.0),
                    "theta": (0.0, step, 0.0),
                    "chi": (0.0, 0.0, step),
                }[component]
                bumped = apply_euler_rotation(
                    positions,
                    self.masses,
                    phi + delta[0],
                    theta + delta[1],
                    chi + delta[2],
                )
                row = (bumped - positions).reshape(-1) / step
            rows.append(row)
        return np.vstack(rows)

    def cartesian_to_internal(self, positions: np.ndarray | None = None) -> np.ndarray:
        """
        Evaluate all primitive internal coordinate values.

        Parameters
        ----------
        positions : np.ndarray | None
            Shape `(n_atoms, 3)`.

        Returns
        -------
        np.ndarray
            Internal coordinate values, shape `(n_internal,)`.
        """
        positions = self.molecule.positions if positions is None else positions
        values: list[float] = []
        for primitive in self.primitives:
            if primitive.kind == InternalCoordinateKind.STRETCH:
                i, j = primitive.indices
                values.append(_bond_length(positions, i, j))
            elif primitive.kind == InternalCoordinateKind.BEND:
                i, j, k = primitive.indices
                values.append(_bond_angle(positions, i, j, k))
            elif primitive.kind == InternalCoordinateKind.TORSION:
                i, j, k, l = primitive.indices
                values.append(_dihedral(positions, i, j, k, l))
            else:
                from pyqchem.euler import euler_angles_from_geometry

                phi, theta, chi = euler_angles_from_geometry(positions, self.masses, self._reference_axes)
                angle_map = {"rot_theta": theta, "rot_phi": phi, "rot_chi": chi}
                values.append(float(angle_map[primitive.label]))
        return np.array(values, dtype=float)

    def g_matrix(self, positions: np.ndarray | None = None) -> np.ndarray:
        """
        Compute the Wilson G matrix `G = B M^{-1} B^T`.

        Parameters
        ----------
        positions : np.ndarray | None
            Shape `(n_atoms, 3)`.

        Returns
        -------
        np.ndarray
            G-matrix with shape `(n_internal, n_internal)`.
        """
        b_mat = self.wilson_b_matrix(positions)
        inv_mass = np.repeat(1.0 / self.masses, 3)
        return b_mat @ np.diag(inv_mass) @ b_mat.T

    def internal_displacement_to_cartesian(
        self,
        dq: np.ndarray,
        positions: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Map an internal-coordinate displacement to Cartesian coordinates.

        Uses the linearized relation `dx = B^T G^{-1} dq`.

        Parameters
        ----------
        dq : np.ndarray
            Internal displacement, shape `(n_internal,)`.
        positions : np.ndarray | None
            Reference geometry, shape `(n_atoms, 3)`.

        Returns
        -------
        np.ndarray
            Updated Cartesian coordinates, shape `(n_atoms, 3)`.
        """
        positions = self.molecule.positions.copy() if positions is None else positions.copy()
        b_mat = self.wilson_b_matrix(positions)
        g_mat = self.g_matrix(positions)
        g_inv = np.linalg.pinv(g_mat, rcond=1e-8)
        inv_mass = np.repeat(1.0 / self.masses, 3)
        dx = inv_mass * (b_mat.T @ (g_inv @ dq))
        return positions + dx.reshape(-1, 3)

    def project_hessian(
        self,
        hessian_cart: np.ndarray,
        angstrom_hessian: bool = True,
    ) -> InternalCoordinateAnalysis:
        """
        Project the Cartesian Hessian into internal coordinates and diagonalize.

        Each separable mode is classified as stretch, bend, torsion, or rotation
        according to its largest projection onto primitive coordinates.

        Parameters
        ----------
        hessian_cart : np.ndarray
            Cartesian Hessian, shape `(3 * n_atoms, 3 * n_atoms)`.
            If `angstrom_hessian` is True, units are Hartree/Angstrom^2.
        angstrom_hessian : bool
            Whether the input Hessian uses Angstrom rather than Bohr.

        Returns
        -------
        InternalCoordinateAnalysis
            Primitives, separable modes, and conversion matrices.
        """
        positions = self.molecule.positions
        b_mat = self.wilson_b_matrix(positions)
        g_mat = self.g_matrix(positions)
        n_atoms = len(self.molecule.atoms)

        hess = np.asarray(hessian_cart, dtype=float)
        if hess.shape != (3 * n_atoms, 3 * n_atoms):
            raise ValueError(f"Hessian shape {hess.shape} incompatible with {n_atoms} atoms")

        if angstrom_hessian:
            hess = hess / (0.529177210903**2)

        mass_inv_sqrt = np.repeat(self.masses ** (-0.5), 3)
        mass_weighted_hess = np.diag(mass_inv_sqrt) @ hess @ np.diag(mass_inv_sqrt)

        g_inv_sqrt: np.ndarray
        try:
            chol = np.linalg.cholesky(g_mat + 1e-10 * np.eye(g_mat.shape[0]))
            g_inv_sqrt = np.linalg.inv(chol).T
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(g_mat)
            g_inv_sqrt = eigvecs @ np.diag(np.where(eigvals > 1e-10, eigvals ** (-0.5), 0.0)) @ eigvecs.T

        l_mat = g_inv_sqrt @ b_mat @ np.diag(mass_inv_sqrt)
        internal_hess = l_mat @ mass_weighted_hess @ l_mat.T
        force_constants, eigenvectors = np.linalg.eigh(internal_hess)

        au2hz = (4.3597447222071e-18 / (1.66053906660e-27 * (5.29177210903e-11**2))) ** 0.5 / (2.0 * np.pi)
        modes: list[InternalMode] = []
        for fc, vec in zip(force_constants, eigenvectors.T):
            if fc <= 1e-8:
                freq_cm1 = 0.0
            else:
                freq_au = np.sqrt(fc)
                freq_cm1 = float(freq_au * au2hz / 299792458.0 * 100.0)

            contributions: dict[str, float] = {}
            for weight, primitive in zip(np.abs(vec), self.primitives):
                contributions[primitive.label] = float(weight)
            dominant = max(self.primitives, key=lambda p: contributions.get(p.label, 0.0))
            modes.append(
                InternalMode(
                    kind=dominant.kind,
                    frequency_cm1=freq_cm1,
                    force_constant_au=float(max(fc, 0.0)),
                    displacement_vector=vec / max(np.linalg.norm(vec), 1e-12),
                    primitive_contributions=contributions,
                )
            )

        internal_values = self.cartesian_to_internal(positions)
        return InternalCoordinateAnalysis(
            primitives=self.primitives,
            modes=modes,
            b_matrix=b_mat,
            g_matrix=g_mat,
            internal_values=internal_values,
        )

    def modes_by_kind(self, analysis: InternalCoordinateAnalysis) -> dict[InternalCoordinateKind, list[InternalMode]]:
        """
        Group separable modes by coordinate kind.

        Parameters
        ----------
        analysis : InternalCoordinateAnalysis
            Result of `project_hessian`.

        Returns
        -------
        dict[InternalCoordinateKind, list[InternalMode]]
            Modes grouped into stretches, bends, torsions, and rotations.
        """
        grouped: dict[InternalCoordinateKind, list[InternalMode]] = {kind: [] for kind in InternalCoordinateKind}
        for mode in analysis.modes:
            grouped[mode.kind].append(mode)
        return grouped
