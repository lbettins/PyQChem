"""Embedded substrate molecule in a fixed host framework."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyqchem.structure.atom import Atom
from pyqchem.structure.host_framework import HostFramework
from pyqchem.structure.molecule import Molecule


@dataclass
class EmbeddedSystem:
    """
    Substrate molecule embedded in a fixed host framework.

    The combined system can be written as XYZ and evolved in time while
    host atoms remain fixed.

    Parameters
    ----------
    substrate : Molecule
        Evolving quantum or classical region.
    host : HostFramework
        Fixed external framework.
    """

    substrate: Molecule
    host: HostFramework

    def combined_atoms(self) -> list[Atom]:
        """All atoms in substrate-then-host order."""
        substrate_atoms = [atom.copy() for atom in self.substrate.atoms]
        for atom in substrate_atoms:
            atom.region = "substrate"
        host_atoms = [atom.copy() for atom in self.host.atoms]
        return substrate_atoms + host_atoms

    def as_combined_molecule(self) -> Molecule:
        """Return a single Molecule view of the embedded system."""
        return Molecule(
            atoms=self.combined_atoms(),
            name=f"{self.substrate.name}@{self.host.name}",
            charge=self.substrate.charge,
            spin=self.substrate.spin,
        )

    def substrate_slice(self) -> slice:
        """Slice object addressing substrate atoms in the combined system."""
        return slice(0, len(self.substrate.atoms))

    def host_slice(self) -> slice:
        """Slice object addressing host atoms in the combined system."""
        start = len(self.substrate.atoms)
        return slice(start, start + len(self.host.atoms))

    def update_substrate_positions(self, coords: np.ndarray) -> None:
        """
        Update substrate atom coordinates in place.

        Parameters
        ----------
        coords : np.ndarray
            Shape `(n_substrate, 3)` new positions in Angstrom.
        """
        coords = np.asarray(coords, dtype=float)
        substrate_indices = self.substrate.substrate_indices()
        if coords.shape != (len(substrate_indices), 3):
            raise ValueError(
                f"Expected ({len(substrate_indices)}, 3) substrate coordinates, got {coords.shape}"
            )
        for idx, pos in zip(substrate_indices, coords):
            self.substrate.atoms[idx].position = pos.copy()
