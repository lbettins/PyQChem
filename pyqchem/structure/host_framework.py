"""Fixed external host framework representation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyqchem.structure.atom import Atom


@dataclass
class HostFramework:
    """
    Fixed external structure that does not evolve in dynamics.

    Parameters
    ----------
    atoms : list[Atom]
        Host atoms; all are forced to `substrate=False`.
    name : str
        Framework label, e.g. `zeolite` or `surface`.
    """

    atoms: list[Atom]
    name: str = "host"

    def __post_init__(self) -> None:
        for atom in self.atoms:
            atom.substrate = False
            atom.region = "host"

    @property
    def positions(self) -> np.ndarray:
        """Host atom coordinates with shape `(n_host, 3)`."""
        return np.array([atom.position for atom in self.atoms], dtype=float)

    @classmethod
    def from_positions(
        cls,
        symbols: list[str],
        positions: np.ndarray,
        name: str = "host",
    ) -> HostFramework:
        """
        Build a host framework from symbols and coordinates.

        Parameters
        ----------
        symbols : list[str]
            Element symbols.
        positions : np.ndarray
            Shape `(n_atoms, 3)` in Angstrom.
        name : str
            Framework label.

        Returns
        -------
        HostFramework
            Frozen host structure.
        """
        positions = np.asarray(positions, dtype=float)
        atoms = [
            Atom(symbol=symbol, position=pos.copy(), substrate=False, region="host")
            for symbol, pos in zip(symbols, positions)
        ]
        return cls(atoms=atoms, name=name)
