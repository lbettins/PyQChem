"""Tests for example adsorbate–zeolite structures."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyqchem.structure import EmbeddedSystem, HostFramework, Molecule
from pyqchem.xyz import read_xyz

STRUCTURES_DIR = Path(__file__).resolve().parents[1] / "examples" / "structures"

EXPECTED = {
    "methane_cha.xyz": {"nads": 5, "adsorbate": ["H", "C", "H", "H", "H"]},
    "methanol_mfi.xyz": {"nads": 6, "adsorbate": ["H", "H", "C", "H", "H", "O"]},
    "ethane_cha.xyz": {"nads": 8, "adsorbate": ["H", "C", "H", "C", "H", "H", "H", "H"]},
    "ethanol_mfi.xyz": {"nads": 9, "adsorbate": ["C", "H", "C", "H", "H", "O", "H", "H", "H"]},
    "propane_mfi.xyz": {"nads": 11, "adsorbate": ["C", "H", "C", "H", "H", "C", "H", "H", "H", "H", "H"]},
}


@pytest.mark.parametrize("filename,meta", EXPECTED.items())
def test_example_structure_tags(filename: str, meta: dict) -> None:
    """
    Example XYZs should load with adsorbate substrate tags and a zeolite host.

    Parameters
    ----------
    filename : str
        XYZ filename under `examples/structures/`.
    meta : dict
        Expected adsorbate size and element order.
    """
    path = STRUCTURES_DIR / filename
    assert path.is_file()
    mol = read_xyz(path)
    nads = meta["nads"]
    assert [atom.symbol for atom in mol.atoms[:nads]] == meta["adsorbate"]
    assert all(atom.substrate for atom in mol.atoms[:nads])
    assert all(not atom.substrate for atom in mol.atoms[nads:])
    assert any(atom.symbol == "Al" for atom in mol.atoms[nads:])
    assert any(atom.symbol == "Si" for atom in mol.atoms[nads:])


def test_example_structure_as_embedded_system() -> None:
    """Tagged example XYZ should split cleanly into an EmbeddedSystem."""
    combined = read_xyz(STRUCTURES_DIR / "methanol_mfi.xyz")
    substrate = Molecule(
        atoms=[atom.copy() for atom in combined.atoms if atom.substrate],
        name="methanol",
    )
    host = HostFramework(
        atoms=[atom.copy() for atom in combined.atoms if not atom.substrate],
        name="mfi",
    )
    system = EmbeddedSystem(substrate=substrate, host=host)
    assert len(system.substrate.atoms) == 6
    assert len(system.host.atoms) == len(combined.atoms) - 6
    assert system.substrate.symbols.count("O") == 1
    assert system.substrate.symbols.count("C") == 1
