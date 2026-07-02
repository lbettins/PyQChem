"""Read, write, and build XYZ coordinate files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyqchem.structure import Atom, Molecule


def write_xyz(molecule: Molecule, path: Path | str, comment: str = "PyQChem") -> None:
    """
    Write a molecule to an XYZ file.

    Parameters
    ----------
    molecule : Molecule
        Structure to serialize.
    path : Path | str
        Output file path.
    comment : str
        Second line comment in the XYZ file.
    """
    path = Path(path)
    lines = [str(len(molecule.atoms)), comment]
    for atom in molecule.atoms:
        x, y, z = atom.position
        substrate_flag = " substrate" if atom.substrate else " fixed"
        lines.append(f"{atom.symbol:>2s} {x:12.8f} {y:12.8f} {z:12.8f}{substrate_flag}")
    path.write_text("\n".join(lines) + "\n")


def read_xyz(path: Path | str) -> Molecule:
    """
    Read an XYZ file into a Molecule.

    Lines may optionally include trailing `substrate` or `fixed` markers.

    Parameters
    ----------
    path : Path | str
        Input XYZ file.

    Returns
    -------
    Molecule
        Parsed structure.
    """
    path = Path(path)
    raw_lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    n_atoms = int(raw_lines[0])
    atoms: list[Atom] = []
    for line in raw_lines[2 : 2 + n_atoms]:
        parts = line.split()
        symbol = parts[0]
        position = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)
        substrate = True
        if len(parts) >= 5:
            substrate = parts[4].lower() != "fixed"
        atoms.append(Atom(symbol=symbol, position=position, substrate=substrate))
    return Molecule(atoms=atoms)


def molecule_from_smiles(
    smiles: str,
    seed: int = 0,
    substrate: bool = True,
) -> Molecule:
    """
    Build a 3D structure from a SMILES string using RDKit.

    Parameters
    ----------
    smiles : str
        SMILES representation of the molecule.
    seed : int
        Random seed for conformer embedding.
    substrate : bool
        Whether generated atoms are tagged as substrate.

    Returns
    -------
    Molecule
        Embedded 3D geometry in Angstrom.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise ImportError(
            "RDKit is required for SMILES input. Install with: pip install 'pyqchem[smiles]'"
        ) from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(f"Failed to embed conformer for SMILES: {smiles}")
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    atoms = [
        Atom(
            symbol=conf.GetAtomWithIdx(idx).GetSymbol(),
            position=np.array(conf.GetAtomPosition(idx), dtype=float),
            substrate=substrate,
            region="substrate",
        )
        for idx in range(mol.GetNumAtoms())
    ]
    return Molecule(atoms=atoms)


def build_water() -> Molecule:
    """
    Return a pre-equilibrated water molecule in Angstrom.

    Returns
    -------
    Molecule
        H2O geometry with O at the origin.
    """
    oh = 0.9572
    hoh = np.deg2rad(104.52)
    o = Atom("O", np.array([0.0, 0.0, 0.0]))
    h1 = Atom("H", np.array([oh * np.sin(hoh / 2.0), 0.0, oh * np.cos(hoh / 2.0)]))
    h2 = Atom("H", np.array([-oh * np.sin(hoh / 2.0), 0.0, oh * np.cos(hoh / 2.0)]))
    return Molecule(atoms=[o, h1, h2])
