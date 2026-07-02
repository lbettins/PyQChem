"""Density functional theory ground-state calculations via PySCF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyqchem.structure import Molecule


@dataclass
class DFTSettings:
    """
    Electronic structure settings for a ground-state calculation.

    Parameters
    ----------
    functional : str
        DFT functional name understood by PySCF, e.g. `B3LYP` or `PBE`.
    basis : str
        Gaussian basis set, e.g. `sto-3g` or `6-31g`.
    verbose : int
        PySCF log verbosity.
    """

    functional: str = "B3LYP"
    basis: str = "sto-3g"
    verbose: int = 0


@dataclass
class DFTResult:
    """
    Output from a ground-state DFT calculation.

    Parameters
    ----------
    energy_hartree : float
        Total electronic energy in Hartree.
    converged : bool
        Whether the SCF procedure converged.
    n_iterations : int
        Number of SCF cycles.
    method : str
        Human-readable method label.
    """

    energy_hartree: float
    converged: bool
    n_iterations: int
    method: str


@dataclass
class FrequencyResult:
    """
    Harmonic vibrational analysis from a numerical Hessian.

    Parameters
    ----------
    frequencies_cm1 : np.ndarray
        Real vibrational frequencies in cm^-1 after removing translations/rotations.
    hessian : np.ndarray
        Cartesian Hessian in Hartree/Bohr^2.
    zpe_hartree : float
        Zero-point vibrational energy in Hartree.
    """

    frequencies_cm1: np.ndarray
    hessian: np.ndarray
    zpe_hartree: float


class DFTCalculator:
    """
    Run ground-state DFT and harmonic frequency analysis with PySCF.

    Parameters
    ----------
    settings : DFTSettings
        Functional and basis configuration.
    """

    def __init__(self, settings: DFTSettings | None = None) -> None:
        self.settings = settings or DFTSettings()

    def ground_state_energy(
        self,
        molecule: Molecule,
        atom_indices: list[int] | None = None,
    ) -> DFTResult:
        """
        Compute the ground-state energy of a molecule or substructure.

        Parameters
        ----------
        molecule : Molecule
            Full or partial molecular geometry in Angstrom.
        atom_indices : list[int] | None
            If provided, only these atoms are included in the QM calculation.

        Returns
        -------
        DFTResult
            Ground-state energy and convergence metadata.
        """
        from pyscf import gto, scf

        target = molecule if atom_indices is None else molecule.subset(atom_indices)
        mol = gto.Mole()
        mol.atom = [(symbol, tuple(pos)) for symbol, pos in zip(target.symbols, target.positions)]
        mol.unit = "Angstrom"
        mol.basis = self.settings.basis
        mol.charge = target.charge
        mol.spin = target.spin
        mol.verbose = self.settings.verbose
        mol.build()

        mf = scf.RHF(mol)
        mf = mf.density_fit()
        mf.xc = self.settings.functional
        energy = mf.kernel()
        return DFTResult(
            energy_hartree=float(energy),
            converged=bool(mf.converged),
            n_iterations=int(getattr(mf, "cycle", 0) or 0),
            method=f"DFT/{self.settings.functional}/{self.settings.basis}",
        )

    def harmonic_frequencies(
        self,
        molecule: Molecule,
        atom_indices: list[int] | None = None,
    ) -> FrequencyResult:
        """
        Estimate harmonic vibrational frequencies from a PySCF Hessian.

        Parameters
        ----------
        molecule : Molecule
            Geometry in Angstrom.
        atom_indices : list[int] | None
            Optional subset of atoms for the QM region.

        Returns
        -------
        FrequencyResult
            Vibrational frequencies, Hessian, and zero-point energy.
        """
        from pyscf import gto, hessian, scf
        from pyscf.hessian import thermo

        target = molecule if atom_indices is None else molecule.subset(atom_indices)
        mol = gto.Mole()
        mol.atom = [(symbol, tuple(pos)) for symbol, pos in zip(target.symbols, target.positions)]
        mol.unit = "Angstrom"
        mol.basis = self.settings.basis
        mol.charge = target.charge
        mol.spin = target.spin
        mol.verbose = self.settings.verbose
        mol.build()

        mf = scf.RHF(mol)
        mf = mf.density_fit()
        mf.xc = self.settings.functional
        mf.kernel()
        hess = hessian.RHF(mf).kernel()
        analysis = thermo.harmonic_analysis(mol, hess)
        freq_wavenumber = np.real(analysis["freq_wavenumber"])
        vib_mask = freq_wavenumber > 1.0
        freq_cm1 = freq_wavenumber[vib_mask]
        freq_au = np.real(analysis["freq_au"])[vib_mask]
        zpe_hartree = float(0.5 * np.sum(np.abs(freq_au)))
        hess_flat = np.asarray(hess, dtype=float).reshape(3 * len(target.atoms), 3 * len(target.atoms))
        return FrequencyResult(
            frequencies_cm1=freq_cm1,
            hessian=hess_flat,
            zpe_hartree=zpe_hartree,
        )
