"""One-dimensional energy sampling along internal coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from pyqchem.dft import DFTCalculator, DFTResult, DFTSettings
from pyqchem.euler import apply_euler_rotation
from pyqchem.internal_coords import (
    InternalCoordinateAnalysis,
    InternalCoordinateKind,
    InternalCoordinateSystem,
    PrimitiveInternalCoordinate,
    _bond_angle,
    _bond_length,
    _dihedral,
    _unit,
)
from pyqchem.lebedev import lebedev_sphere
from pyqchem.structure import Molecule
from pyqchem.translation import apply_translation, translation_cube_grid


HARTREE_TO_KJ_MOL = 2625.499638


class EnergyMethod(str, Enum):
    """Energy evaluation backend for coordinate slices."""

    HARMONIC = "harmonic"
    DFT = "dft"


@dataclass
class SamplingSettings:
    """
    Grid settings for 1-D internal coordinate scans.

    Parameters
    ----------
    n_points : int
        Number of samples per primitive coordinate.
    stretch_window_a : float
        Half-width in Angstrom for bond-length scans about equilibrium.
    bend_window_rad : float
        Half-width in radians for angle scans about equilibrium.
    torsion_start_rad : float
        Start of the dihedral scan in radians.
    torsion_end_rad : float
        End of the dihedral scan in radians.
    lebedev_points : int
        Number of Lebedev nodes for `(theta, phi)` on the sphere.
    n_chi : int
        Number of equally spaced chi samples on `[-pi, pi]`.
    n_translation : int
        Number of samples per axis for the translation cube (`n_translation**3` points).
    translation_window_a : float
        Half-width in Angstrom for each translation axis.
    translation_force_constant_au : float
        Isotropic harmonic force constant for translations in atomic units.
    energy_method : EnergyMethod
        Whether to use the Hessian harmonic surface or DFT.
    """

    n_points: int = 36
    stretch_window_a: float = 0.15
    bend_window_rad: float = 0.35
    torsion_start_rad: float = -np.pi
    torsion_end_rad: float = np.pi
    lebedev_points: int = 26
    n_chi: int = 36
    n_translation: int = 11
    translation_window_a: float = 0.5
    translation_force_constant_au: float = 0.0
    energy_method: EnergyMethod = EnergyMethod.HARMONIC


@dataclass
class CoordinateSample:
    """
    One point on a 1-D internal coordinate slice.

    Parameters
    ----------
    coordinate_value : float
        Absolute internal coordinate (Angstrom or radians).
    displacement : float
        Displacement from equilibrium along this coordinate.
    energy_hartree : float
        Energy at this geometry in Hartree.
    energy_kj_mol : float
        Energy at this geometry in kJ/mol.
    positions : np.ndarray
        Cartesian geometry, shape `(n_atoms, 3)`.
    """

    coordinate_value: float
    displacement: float
    energy_hartree: float
    energy_kj_mol: float
    positions: np.ndarray


@dataclass
class InternalCoordinateSlice:
    """
    Energy samples along a single primitive internal coordinate.

    Parameters
    ----------
    primitive : PrimitiveInternalCoordinate
        Primitive coordinate being scanned.
    kind : InternalCoordinateKind
        Coordinate class (stretch, bend, torsion, rotation).
    equilibrium_value : float
        Equilibrium value of the coordinate.
    force_constant_au : float
        Hessian-derived force constant for this coordinate.
    reference_energy_hartree : float
        Reference energy at equilibrium.
    samples : list[CoordinateSample]
        Ordered samples along the coordinate.
    """

    primitive: PrimitiveInternalCoordinate
    kind: InternalCoordinateKind
    equilibrium_value: float
    force_constant_au: float
    reference_energy_hartree: float
    samples: list[CoordinateSample] = field(default_factory=list)


@dataclass
class SamplingResult:
    """
    Collection of 1-D slices grouped by internal coordinate kind.

    Parameters
    ----------
    slices : list[InternalCoordinateSlice]
        Stretch, bend, and torsion scans.
    by_kind : dict[InternalCoordinateKind, list[InternalCoordinateSlice]]
        Slices grouped into stretches, bends, and torsions.
    rotations : RotationSamplingResult | None
        Euler-angle rotation samples on a Lebedev sphere x chi grid.
    translations : TranslationSamplingResult | None
        Cubic-grid translation samples in `(delta_x, delta_y, delta_z)`.
    """

    slices: list[InternalCoordinateSlice]
    by_kind: dict[InternalCoordinateKind, list[InternalCoordinateSlice]]
    rotations: RotationSamplingResult | None = None
    translations: TranslationSamplingResult | None = None


@dataclass
class RotationSample:
    """
    One Euler-angle rotation sample.

    Parameters
    ----------
    theta : float
        Polar angle from the north pole in `[0, pi]`.
    phi : float
        Azimuthal angle in `[-pi, pi]`.
    chi : float
        Intrinsic spin angle in `[-pi, pi]`.
    lebedev_weight : float
        Lebedev quadrature weight for the `(theta, phi)` node.
    energy_hartree : float
        Energy at this orientation in Hartree.
    energy_kj_mol : float
        Energy at this orientation in kJ/mol.
    positions : np.ndarray
        Cartesian geometry, shape `(n_atoms, 3)`.
    """

    theta: float
    phi: float
    chi: float
    lebedev_weight: float
    energy_hartree: float
    energy_kj_mol: float
    positions: np.ndarray


@dataclass
class RotationSamplingResult:
    """
    Energy samples on a Lebedev `(theta, phi)` sphere with equally spaced chi.

    Parameters
    ----------
    samples : list[RotationSample]
        All rotation samples.
    lebedev_points : int
        Number of Lebedev nodes used for `(theta, phi)`.
    n_chi : int
        Number of chi samples on `[-pi, pi]`.
    reference_energy_hartree : float
        Reference energy at `(0, 0, 0)` Euler angles.
    """

    samples: list[RotationSample]
    lebedev_points: int
    n_chi: int
    reference_energy_hartree: float


@dataclass
class TranslationSample:
    """
    One cubic-grid translation sample.

    Parameters
    ----------
    delta_x : float
        Displacement along x in Angstrom.
    delta_y : float
        Displacement along y in Angstrom.
    delta_z : float
        Displacement along z in Angstrom.
    displacement : np.ndarray
        Displacement vector, shape `(3,)`.
    energy_hartree : float
        Energy at this geometry in Hartree.
    energy_kj_mol : float
        Energy at this geometry in kJ/mol.
    positions : np.ndarray
        Cartesian geometry, shape `(n_atoms, 3)`.
    """

    delta_x: float
    delta_y: float
    delta_z: float
    displacement: np.ndarray
    energy_hartree: float
    energy_kj_mol: float
    positions: np.ndarray


@dataclass
class TranslationSamplingResult:
    """
    Energy samples on a cubic translation grid.

    Parameters
    ----------
    samples : list[TranslationSample]
        All translation samples.
    n_points_per_axis : int
        Number of grid points along each axis.
    window_a : float
        Half-width of the cube in Angstrom.
    reference_energy_hartree : float
        Reference energy at zero displacement.
    """

    samples: list[TranslationSample]
    n_points_per_axis: int
    window_a: float
    reference_energy_hartree: float


def _rotate_point(point: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate a point about an axis through an origin.

    Parameters
    ----------
    point : np.ndarray
        Point to rotate, shape `(3,)`.
    origin : np.ndarray
        Axis origin, shape `(3,)`.
    axis : np.ndarray
        Rotation axis, shape `(3,)`.
    angle : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        Rotated point, shape `(3,)`.
    """
    axis_u = _unit(axis)
    rel = point - origin
    rotated = (
        rel * np.cos(angle)
        + np.cross(axis_u, rel) * np.sin(angle)
        + axis_u * np.dot(axis_u, rel) * (1.0 - np.cos(angle))
    )
    return origin + rotated


def _apply_stretch(positions: np.ndarray, i: int, j: int, target_length: float) -> np.ndarray:
    """
    Set the bond length between atoms i and j.

    Parameters
    ----------
    positions : np.ndarray
        Shape `(n_atoms, 3)`.
    i : int
        First atom index.
    j : int
        Second atom index.
    target_length : float
        Target bond length in Angstrom.

    Returns
    -------
    np.ndarray
        Updated positions.
    """
    updated = positions.copy()
    bond = updated[j] - updated[i]
    length = np.linalg.norm(bond)
    direction = np.array([1.0, 0.0, 0.0]) if length < 1e-12 else bond / length
    updated[j] = updated[i] + target_length * direction
    return updated


def _apply_bend(positions: np.ndarray, i: int, j: int, k: int, target_angle: float) -> np.ndarray:
    """
    Set the bond angle at atom j between atoms i-j-k.

    Parameters
    ----------
    positions : np.ndarray
        Shape `(n_atoms, 3)`.
    i, j, k : int
        Atom indices defining the angle at j.
    target_angle : float
        Target angle in radians.

    Returns
    -------
    np.ndarray
        Updated positions.
    """
    updated = positions.copy()
    v_ji = updated[i] - updated[j]
    v_jk = updated[k] - updated[j]
    axis = np.cross(v_ji, v_jk)
    if np.linalg.norm(axis) < 1e-12:
        axis = np.array([0.0, 0.0, 1.0])
    current = _bond_angle(updated, i, j, k)
    updated[i] = _rotate_point(updated[i], updated[j], axis, target_angle - current)
    return updated


def _apply_torsion(positions: np.ndarray, i: int, j: int, k: int, l: int, target_dihedral: float) -> np.ndarray:
    """
    Set the dihedral angle for atoms i-j-k-l by rotating atom i about the j-k axis.

    Parameters
    ----------
    positions : np.ndarray
        Shape `(n_atoms, 3)`.
    i, j, k, l : int
        Atom indices defining the dihedral.
    target_dihedral : float
        Target dihedral in radians.

    Returns
    -------
    np.ndarray
        Updated positions.
    """
    updated = positions.copy()
    axis = updated[k] - updated[j]
    current = _dihedral(updated, i, j, k, l)
    delta = target_dihedral - current
    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
    updated[l] = _rotate_point(updated[l], updated[k], axis, delta)
    return updated


def positions_at_coordinate_value(
    positions: np.ndarray,
    primitive: PrimitiveInternalCoordinate,
    target_value: float,
    ics: InternalCoordinateSystem,
) -> np.ndarray:
    """
    Build Cartesian coordinates with one primitive internal coordinate set to a target.

    Parameters
    ----------
    positions : np.ndarray
        Reference geometry, shape `(n_atoms, 3)`.
    primitive : PrimitiveInternalCoordinate
        Primitive coordinate to vary.
    target_value : float
        Target value for that coordinate.
    ics : InternalCoordinateSystem
        Internal coordinate system for the molecule.

    Returns
    -------
    np.ndarray
        Updated Cartesian coordinates, shape `(n_atoms, 3)`.
    """
    if primitive.kind == InternalCoordinateKind.STRETCH:
        i, j = primitive.indices
        return _apply_stretch(positions, i, j, target_value)
    if primitive.kind == InternalCoordinateKind.BEND:
        i, j, k = primitive.indices
        return _apply_bend(positions, i, j, k, target_value)
    if primitive.kind == InternalCoordinateKind.TORSION:
        i, j, k, l = primitive.indices
        return _apply_torsion(positions, i, j, k, l, target_value)

    raise ValueError(
        "Rotation coordinates require full (phi, theta, chi); use apply_euler_rotation() or sample_rotations()."
    )


def coordinate_grid(primitive: PrimitiveInternalCoordinate, settings: SamplingSettings) -> np.ndarray:
    """
    Build the 1-D sample grid for a primitive internal coordinate.

    Parameters
    ----------
    primitive : PrimitiveInternalCoordinate
        Primitive coordinate to scan.
    settings : SamplingSettings
        Scan settings.

    Returns
    -------
    np.ndarray
        Sample values along the coordinate.
    """
    if settings.n_points < 2:
        raise ValueError("n_points must be at least 2")

    if primitive.kind == InternalCoordinateKind.STRETCH:
        low = primitive.equilibrium - settings.stretch_window_a
        high = primitive.equilibrium + settings.stretch_window_a
    elif primitive.kind == InternalCoordinateKind.BEND:
        low = primitive.equilibrium - settings.bend_window_rad
        high = primitive.equilibrium + settings.bend_window_rad
    elif primitive.kind == InternalCoordinateKind.TORSION:
        low = settings.torsion_start_rad
        high = settings.torsion_end_rad
    else:
        raise ValueError(
            "Rotation coordinates use Lebedev (theta, phi) and equally spaced chi; call sample_rotations()."
        )

    return np.linspace(low, high, settings.n_points)


def chi_grid(n_chi: int) -> np.ndarray:
    """
    Build equally spaced intrinsic spin samples on `[-pi, pi]`.

    Parameters
    ----------
    n_chi : int
        Number of chi samples.

    Returns
    -------
    np.ndarray
        Chi values in radians.
    """
    if n_chi < 2:
        raise ValueError("n_chi must be at least 2")
    return np.linspace(-np.pi, np.pi, n_chi)


class InternalModeSampler:
    """
    Sample energies along 1-D slices of primitive internal coordinates.

    Each stretch, bend, torsion, or rotation coordinate is scanned independently
    while other coordinates remain at their equilibrium values. Energies may be
    evaluated harmonically from the projected Hessian or with DFT.

    Parameters
    ----------
    molecule : Molecule
        Equilibrium molecular geometry.
    analysis : InternalCoordinateAnalysis
        Internal-coordinate decomposition with Hessian-derived modes.
    reference_energy_hartree : float
        Reference energy at equilibrium.
    dft_settings : DFTSettings | None
        DFT settings used when `energy_method` is `dft`.
    """

    def __init__(
        self,
        molecule: Molecule,
        analysis: InternalCoordinateAnalysis,
        reference_energy_hartree: float = 0.0,
        dft_settings: DFTSettings | None = None,
    ) -> None:
        self.molecule = molecule
        self.analysis = analysis
        self.reference_energy_hartree = reference_energy_hartree
        self.ics = InternalCoordinateSystem(molecule, include_rotations=True)
        self.ics.primitives = analysis.primitives
        self._dft = DFTCalculator(dft_settings)
        self._primitive_force_constants = self._compute_primitive_force_constants()
        from pyqchem.internal_coords import DEFAULT_MASS_AMU

        self._masses = np.array(
            [DEFAULT_MASS_AMU.get(sym, 12.0) for sym in molecule.symbols],
            dtype=float,
        )
        self._rotation_force_constants = self._rotation_force_constant_map()
        self._translation_force_constant_au = 0.0

    def _compute_primitive_force_constants(self) -> np.ndarray:
        """
        Estimate diagonal force constants for each primitive coordinate.

        Returns
        -------
        np.ndarray
            Force constants in atomic units, shape `(n_primitives,)`.
        """
        n_primitives = len(self.analysis.primitives)
        force_constants = np.zeros(n_primitives, dtype=float)
        for mode in self.analysis.modes:
            if mode.force_constant_au <= 0.0:
                continue
            for idx, weight in enumerate(np.abs(mode.displacement_vector)):
                force_constants[idx] += mode.force_constant_au * weight**2
        return force_constants

    def _rotation_force_constant_map(self) -> dict[str, float]:
        """
        Map rot_theta, rot_phi, and rot_chi labels to force constants.

        Returns
        -------
        dict[str, float]
            Force constants in atomic units for each Euler angle.
        """
        constants = {"rot_theta": 0.0, "rot_phi": 0.0, "rot_chi": 0.0}
        for idx, primitive in enumerate(self.analysis.primitives):
            if primitive.kind == InternalCoordinateKind.ROTATION:
                constants[primitive.label] = float(self._primitive_force_constants[idx])
        return constants

    def _harmonic_rotation_energy(self, phi: float, theta: float, chi: float) -> float:
        """
        Harmonic energy for an Euler-angle orientation relative to equilibrium.

        Parameters
        ----------
        phi : float
            Azimuthal angle in radians.
        theta : float
            Polar angle from the north pole in radians.
        chi : float
            Intrinsic spin angle in radians.

        Returns
        -------
        float
            Energy in Hartree.
        """
        energy = self.reference_energy_hartree
        angle_map = {"rot_theta": theta, "rot_phi": phi, "rot_chi": chi}
        for label, angle in angle_map.items():
            force_constant = self._rotation_force_constants[label]
            energy += 0.5 * force_constant * angle**2
        return energy

    def _harmonic_translation_energy(self, delta_x: float, delta_y: float, delta_z: float) -> float:
        """
        Harmonic energy for a rigid translation relative to equilibrium.

        Parameters
        ----------
        delta_x : float
            Displacement along x in Angstrom.
        delta_y : float
            Displacement along y in Angstrom.
        delta_z : float
            Displacement along z in Angstrom.

        Returns
        -------
        float
            Energy in Hartree.
        """
        force_constant = self._translation_force_constant_au
        displacement_sq = delta_x**2 + delta_y**2 + delta_z**2
        return self.reference_energy_hartree + 0.5 * force_constant * displacement_sq

    def _harmonic_energy(self, primitive_idx: int, displacement: float) -> float:
        """
        Harmonic energy for a displacement along one primitive coordinate.

        Parameters
        ----------
        primitive_idx : int
            Index of the scanned primitive.
        displacement : float
            Displacement from equilibrium.

        Returns
        -------
        float
            Energy in Hartree relative to reference.
        """
        force_constant = self._primitive_force_constants[primitive_idx]
        return self.reference_energy_hartree + 0.5 * force_constant * displacement**2

    def _dft_energy(self, positions: np.ndarray) -> float:
        """
        DFT ground-state energy for a Cartesian geometry.

        Parameters
        ----------
        positions : np.ndarray
            Shape `(n_atoms, 3)`.

        Returns
        -------
        float
            Energy in Hartree.
        """
        trial = self.molecule.copy()
        trial.positions = positions
        result: DFTResult = self._dft.ground_state_energy(trial)
        return result.energy_hartree

    def sample_primitive(
        self,
        primitive_idx: int,
        settings: SamplingSettings | None = None,
    ) -> InternalCoordinateSlice:
        """
        Sample energies along one primitive internal coordinate.

        Parameters
        ----------
        primitive_idx : int
            Index into `analysis.primitives`.
        settings : SamplingSettings | None
            Scan grid and energy method.

        Returns
        -------
        InternalCoordinateSlice
            Energy samples along the coordinate.
        """
        settings = settings or SamplingSettings()
        primitive = self.analysis.primitives[primitive_idx]
        if primitive.kind == InternalCoordinateKind.ROTATION:
            raise ValueError(
                "Rotation coordinates use Lebedev (theta, phi) and equally spaced chi; call sample_rotations()."
            )
        x0 = self.molecule.positions.copy()
        grid = coordinate_grid(primitive, settings)
        samples: list[CoordinateSample] = []

        for value in grid:
            positions = positions_at_coordinate_value(x0, primitive, float(value), self.ics)
            displacement = float(value - primitive.equilibrium)
            if settings.energy_method == EnergyMethod.DFT:
                energy = self._dft_energy(positions)
            else:
                energy = self._harmonic_energy(primitive_idx, displacement)
            samples.append(
                CoordinateSample(
                    coordinate_value=float(value),
                    displacement=displacement,
                    energy_hartree=energy,
                    energy_kj_mol=energy * HARTREE_TO_KJ_MOL,
                    positions=positions,
                )
            )

        return InternalCoordinateSlice(
            primitive=primitive,
            kind=primitive.kind,
            equilibrium_value=primitive.equilibrium,
            force_constant_au=float(self._primitive_force_constants[primitive_idx]),
            reference_energy_hartree=self.reference_energy_hartree,
            samples=samples,
        )

    def sample_rotations(self, settings: SamplingSettings | None = None) -> RotationSamplingResult:
        """
        Sample energies on a Lebedev `(theta, phi)` sphere with equally spaced chi.

        Parameters
        ----------
        settings : SamplingSettings | None
            Lebedev order, chi count, and energy method.

        Returns
        -------
        RotationSamplingResult
            Energies for each `(theta, phi, chi)` grid point.
        """
        settings = settings or SamplingSettings()
        theta_nodes, phi_nodes, weights = lebedev_sphere(settings.lebedev_points)
        chi_nodes = chi_grid(settings.n_chi)
        x0 = self.molecule.positions.copy()
        samples: list[RotationSample] = []

        for theta, phi, weight in zip(theta_nodes, phi_nodes, weights):
            for chi in chi_nodes:
                positions = apply_euler_rotation(x0, self._masses, float(phi), float(theta), float(chi))
                if settings.energy_method == EnergyMethod.DFT:
                    energy = self._dft_energy(positions)
                else:
                    energy = self._harmonic_rotation_energy(float(phi), float(theta), float(chi))
                samples.append(
                    RotationSample(
                        theta=float(theta),
                        phi=float(phi),
                        chi=float(chi),
                        lebedev_weight=float(weight),
                        energy_hartree=energy,
                        energy_kj_mol=energy * HARTREE_TO_KJ_MOL,
                        positions=positions,
                    )
                )

        return RotationSamplingResult(
            samples=samples,
            lebedev_points=settings.lebedev_points,
            n_chi=settings.n_chi,
            reference_energy_hartree=self.reference_energy_hartree,
        )

    def sample_translations(self, settings: SamplingSettings | None = None) -> TranslationSamplingResult:
        """
        Sample energies on a cubic grid of `(delta_x, delta_y, delta_z)` displacements.

        Parameters
        ----------
        settings : SamplingSettings | None
            Cube size, grid density, and energy method.

        Returns
        -------
        TranslationSamplingResult
            Energies for each point in the translation cube.
        """
        settings = settings or SamplingSettings()
        self._translation_force_constant_au = settings.translation_force_constant_au
        displacements = translation_cube_grid(settings.n_translation, settings.translation_window_a)
        x0 = self.molecule.positions.copy()
        samples: list[TranslationSample] = []

        for delta_x, delta_y, delta_z in displacements:
            positions = apply_translation(x0, float(delta_x), float(delta_y), float(delta_z))
            if settings.energy_method == EnergyMethod.DFT:
                energy = self._dft_energy(positions)
            else:
                energy = self._harmonic_translation_energy(float(delta_x), float(delta_y), float(delta_z))
            displacement = np.array([delta_x, delta_y, delta_z], dtype=float)
            samples.append(
                TranslationSample(
                    delta_x=float(delta_x),
                    delta_y=float(delta_y),
                    delta_z=float(delta_z),
                    displacement=displacement,
                    energy_hartree=energy,
                    energy_kj_mol=energy * HARTREE_TO_KJ_MOL,
                    positions=positions,
                )
            )

        return TranslationSamplingResult(
            samples=samples,
            n_points_per_axis=settings.n_translation,
            window_a=settings.translation_window_a,
            reference_energy_hartree=self.reference_energy_hartree,
        )

    def sample_all(self, settings: SamplingSettings | None = None) -> SamplingResult:
        """
        Sample 1-D energy slices for every primitive internal coordinate.

        Parameters
        ----------
        settings : SamplingSettings | None
            Scan grid and energy method.

        Returns
        -------
        SamplingResult
            All slices grouped by coordinate kind.
        """
        settings = settings or SamplingSettings()
        non_rotation_indices = [
            idx for idx, p in enumerate(self.analysis.primitives) if p.kind != InternalCoordinateKind.ROTATION
        ]
        slices = [self.sample_primitive(idx, settings) for idx in non_rotation_indices]
        grouped: dict[InternalCoordinateKind, list[InternalCoordinateSlice]] = {
            kind: [] for kind in InternalCoordinateKind
        }
        for slice_ in slices:
            grouped[slice_.kind].append(slice_)
        rotations = self.sample_rotations(settings)
        translations = self.sample_translations(settings)
        grouped[InternalCoordinateKind.ROTATION] = []
        return SamplingResult(slices=slices, by_kind=grouped, rotations=rotations, translations=translations)

    def sample_by_kind(
        self,
        kind: InternalCoordinateKind,
        settings: SamplingSettings | None = None,
    ) -> list[InternalCoordinateSlice]:
        """
        Sample only primitives of one coordinate kind.

        Parameters
        ----------
        kind : InternalCoordinateKind
            Stretch, bend, torsion, or rotation.
        settings : SamplingSettings | None
            Scan grid and energy method.

        Returns
        -------
        list[InternalCoordinateSlice]
            Slices for matching primitives.
        """
        settings = settings or SamplingSettings()
        if kind == InternalCoordinateKind.ROTATION:
            raise ValueError("Use sample_rotations() for Euler-angle rotation grids.")
        indices = [idx for idx, p in enumerate(self.analysis.primitives) if p.kind == kind]
        return [self.sample_primitive(idx, settings) for idx in indices]
