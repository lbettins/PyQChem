"""Molecular fragment representation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyqchem.structure.atom import Atom


@dataclass
class Molecule:
    """
    Collection of atoms representing a molecular fragment.

    Parameters
    ----------
    atoms : list[Atom]
        Atoms in the molecule.
    name : str
        Optional label.
    charge : int
        Net charge for electronic structure calculations.
    spin : int
        Spin multiplicity minus one (2S).
    """

    atoms: list[Atom]
    name: str = "molecule"
    charge: int = 0
    spin: int = 0

    @property
    def symbols(self) -> list[str]:
        """Element symbols in atom order."""
        return [atom.symbol for atom in self.atoms]

    @property
    def positions(self) -> np.ndarray:
        """Cartesian coordinates with shape `(n_atoms, 3)`."""
        return np.array([atom.position for atom in self.atoms], dtype=float)

    @positions.setter
    def positions(self, coords: np.ndarray) -> None:
        coords = np.asarray(coords, dtype=float)
        if coords.shape != (len(self.atoms), 3):
            raise ValueError(f"Expected shape ({len(self.atoms)}, 3), got {coords.shape}")
        for atom, pos in zip(self.atoms, coords):
            atom.position = pos.copy()

    def substrate_indices(self) -> list[int]:
        """Indices of atoms that may evolve in time."""
        return [idx for idx, atom in enumerate(self.atoms) if atom.substrate]

    def fixed_indices(self) -> list[int]:
        """Indices of frozen framework atoms."""
        return [idx for idx, atom in enumerate(self.atoms) if not atom.substrate]

    def subset(self, indices: list[int]) -> Molecule:
        """
        Extract a sub-molecule by atom indices.

        Parameters
        ----------
        indices : list[int]
            Atom indices to retain.

        Returns
        -------
        Molecule
            Sub-molecule preserving charge and spin metadata.
        """
        return Molecule(
            atoms=[self.atoms[i].copy() for i in indices],
            name=f"{self.name}_subset",
            charge=self.charge,
            spin=self.spin,
        )

    def copy(self) -> Molecule:
        """Return a deep copy."""
        return Molecule(
            atoms=[atom.copy() for atom in self.atoms],
            name=self.name,
            charge=self.charge,
            spin=self.spin,
        )
