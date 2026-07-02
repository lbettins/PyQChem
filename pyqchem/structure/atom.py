"""Single atom representation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Atom:
    """
    Single atom with Cartesian coordinates in Angstrom.

    Parameters
    ----------
    symbol : str
        Element symbol.
    position : np.ndarray
        Shape `(3,)` Cartesian coordinates in Angstrom.
    substrate : bool
        If False, the atom is frozen in dynamics and treated as host framework.
    region : str
        Logical region label, e.g. `substrate` or `host`.
    """

    symbol: str
    position: np.ndarray
    substrate: bool = True
    region: str = "substrate"

    def copy(self) -> Atom:
        """Return a deep copy of this atom."""
        return Atom(
            symbol=self.symbol,
            position=self.position.copy(),
            substrate=self.substrate,
            region=self.region,
        )
