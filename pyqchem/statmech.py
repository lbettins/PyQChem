"""Statistical mechanical property estimation from molecular data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import Avogadro, Boltzmann, Planck, gas_constant, speed_of_light

from pyqchem.dft import DFTResult, FrequencyResult


@dataclass
class ThermodynamicProperties:
    """
    Canonical-ensemble thermodynamic estimates at a single temperature.

    Parameters
    ----------
    temperature_k : float
        Temperature in Kelvin.
    q_total : float
        Total molecular partition function (unitless).
    q_trans : float
        Translational partition function.
    q_rot : float
        Rotational partition function.
    q_vib : float
        Vibrational partition function.
    internal_energy_kj_mol : float
        Internal energy relative to the ground electronic state.
    enthalpy_kj_mol : float
        Enthalpy estimate.
    entropy_j_mol_k : float
        Entropy.
    gibbs_kj_mol : float
        Gibbs free energy.
    heat_capacity_j_mol_k : float
        Constant-pressure heat capacity estimate.
    zpe_kj_mol : float
        Zero-point vibrational energy.
    """

    temperature_k: float
    q_total: float
    q_trans: float
    q_rot: float
    q_vib: float
    internal_energy_kj_mol: float
    enthalpy_kj_mol: float
    entropy_j_mol_k: float
    gibbs_kj_mol: float
    heat_capacity_j_mol_k: float
    zpe_kj_mol: float


class StatisticalMechanicsEstimator:
    """
    Estimate partition functions and thermodynamic properties.

    Uses the ideal-gas rigid-rotor harmonic-oscillator (RRHO) model.
    """

    def __init__(self, molecular_mass_amu: float, principal_moments_amu_a2: np.ndarray) -> None:
        """
        Parameters
        ----------
        molecular_mass_amu : float
            Total molecular mass in atomic mass units.
        principal_moments_amu_a2 : np.ndarray
            Principal moments of inertia in amu*Angstrom^2.
        """
        self.molecular_mass_amu = molecular_mass_amu
        self.principal_moments_amu_a2 = np.asarray(principal_moments_amu_a2, dtype=float)

    @classmethod
    def from_geometry(cls, symbols: list[str], positions: np.ndarray) -> StatisticalMechanicsEstimator:
        """
        Build an estimator from atomic symbols and coordinates.

        Parameters
        ----------
        symbols : list[str]
            Element symbols.
        positions : np.ndarray
            Shape `(n_atoms, 3)` in Angstrom.

        Returns
        -------
        StatisticalMechanicsEstimator
            Estimator with mass and inertia tensor derived from geometry.
        """
        from pyscf import gto

        masses = np.array([gto.charge(sym) for sym in symbols], dtype=float)
        center = np.average(positions, axis=0, weights=masses)
        rel = positions - center
        inertia = np.zeros((3, 3), dtype=float)
        for mass, pos in zip(masses, rel):
            r2 = np.dot(pos, pos)
            inertia += mass * (r2 * np.eye(3) - np.outer(pos, pos))
        moments = np.sort(np.linalg.eigvalsh(inertia))
        return cls(molecular_mass_amu=float(masses.sum()), principal_moments_amu_a2=moments)

    def translational_partition(self, temperature_k: float, volume_m3: float) -> float:
        """
        Compute the translational partition function.

        Parameters
        ----------
        temperature_k : float
            Temperature in Kelvin.
        volume_m3 : float
            Volume of the simulation box in m^3.

        Returns
        -------
        float
            Translational partition function.
        """
        mass_kg = self.molecular_mass_amu * 1.66053906660e-27
        lambda_th = Planck / np.sqrt(2.0 * np.pi * mass_kg * Boltzmann * temperature_k)
        return volume_m3 / (lambda_th**3)

    def rotational_partition(self, temperature_k: float, symmetry_number: int = 1) -> float:
        """
        Compute the rigid-rotor rotational partition function.

        Parameters
        ----------
        temperature_k : float
            Temperature in Kelvin.
        symmetry_number : int
            Rotational symmetry number.

        Returns
        -------
        float
            Rotational partition function.
        """
        amu_to_kg = 1.66053906660e-27
        ang_to_m = 1.0e-10
        moments_kg_m2 = self.principal_moments_amu_a2 * amu_to_kg * ang_to_m**2
        positive = moments_kg_m2[moments_kg_m2 > 1.0e-30]
        if len(positive) == 0:
            return 1.0
        if len(positive) == 1:
            theta = (Planck**2) / (8.0 * (np.pi**2) * positive[0] * Boltzmann)
            return temperature_k / (symmetry_number * theta)
        if len(positive) == 2:
            theta_a, theta_b = (Planck**2) / (8.0 * (np.pi**2) * positive * Boltzmann)
            return np.sqrt(np.pi * temperature_k**3 / (symmetry_number**2 * theta_a * theta_b))
        theta_a, theta_b, theta_c = (Planck**2) / (8.0 * (np.pi**2) * positive * Boltzmann)
        return np.sqrt(np.pi * temperature_k**3 / (symmetry_number * theta_a * theta_b * theta_c))

    @staticmethod
    def vibrational_partition(frequencies_cm1: np.ndarray, temperature_k: float) -> float:
        """
        Compute the harmonic vibrational partition function.

        Parameters
        ----------
        frequencies_cm1 : np.ndarray
            Vibrational frequencies in cm^-1.
        temperature_k : float
            Temperature in Kelvin.

        Returns
        -------
        float
            Vibrational partition function.
        """
        freqs = np.asarray(frequencies_cm1, dtype=float)
        freqs = freqs[freqs > 1.0]
        if freqs.size == 0:
            return 1.0
        x = Planck * speed_of_light * 100.0 * freqs / (Boltzmann * temperature_k)
        return float(np.prod(1.0 / (1.0 - np.exp(-x))))

    @staticmethod
    def vibrational_internal_energy(frequencies_cm1: np.ndarray, temperature_k: float) -> float:
        """
        Vibrational internal energy in J per molecule.

        Parameters
        ----------
        frequencies_cm1 : np.ndarray
            Vibrational frequencies in cm^-1.
        temperature_k : float
            Temperature in Kelvin.

        Returns
        -------
        float
            Vibrational internal energy in J.
        """
        freqs = np.asarray(frequencies_cm1, dtype=float)
        freqs = freqs[freqs > 1.0]
        if freqs.size == 0:
            return 0.0
        x = Planck * speed_of_light * 100.0 * freqs / (Boltzmann * temperature_k)
        return float(np.sum(Boltzmann * temperature_k * x / (np.exp(x) - 1.0)))

    @staticmethod
    def vibrational_heat_capacity(frequencies_cm1: np.ndarray, temperature_k: float) -> float:
        """
        Vibrational heat capacity at constant volume in J/K per molecule.

        Parameters
        ----------
        frequencies_cm1 : np.ndarray
            Vibrational frequencies in cm^-1.
        temperature_k : float
            Temperature in Kelvin.

        Returns
        -------
        float
            Vibrational heat capacity in J/K.
        """
        freqs = np.asarray(frequencies_cm1, dtype=float)
        freqs = freqs[freqs > 1.0]
        if freqs.size == 0:
            return 0.0
        x = Planck * speed_of_light * 100.0 * freqs / (Boltzmann * temperature_k)
        ex = np.exp(x)
        return float(np.sum(Boltzmann * (x**2) * ex / ((ex - 1.0) ** 2)))

    def estimate(
        self,
        temperature_k: float,
        frequencies_cm1: np.ndarray,
        volume_m3: float,
        symmetry_number: int = 1,
        dft_result: DFTResult | None = None,
        frequency_result: FrequencyResult | None = None,
    ) -> ThermodynamicProperties:
        """
        Estimate RRHO thermodynamic properties at a temperature.

        Parameters
        ----------
        temperature_k : float
            Temperature in Kelvin.
        frequencies_cm1 : np.ndarray
            Vibrational frequencies in cm^-1.
        volume_m3 : float
            Simulation volume in m^3.
        symmetry_number : int
            Rotational symmetry number.
        dft_result : DFTResult | None
            Optional ground-state energy reference.
        frequency_result : FrequencyResult | None
            Optional frequency result for zero-point energy.

        Returns
        -------
        ThermodynamicProperties
            Estimated thermodynamic quantities per mole.
        """
        q_trans = self.translational_partition(temperature_k, volume_m3)
        q_rot = self.rotational_partition(temperature_k, symmetry_number)
        q_vib = self.vibrational_partition(frequencies_cm1, temperature_k)
        q_total = q_trans * q_rot * q_vib

        u_vib = self.vibrational_internal_energy(frequencies_cm1, temperature_k)
        cv_vib = self.vibrational_heat_capacity(frequencies_cm1, temperature_k)
        zpe_j = 0.0
        if frequency_result is not None:
            zpe_j = frequency_result.zpe_hartree * 4.3597447222071e-18

        u_total_j = 1.5 * Boltzmann * temperature_k + u_vib + zpe_j
        if dft_result is not None:
            u_total_j += dft_result.energy_hartree * 4.3597447222071e-18 / Avogadro

        s_trans = Boltzmann * (np.log(q_trans) + 2.5)
        s_rot = Boltzmann * np.log(q_rot) if q_rot > 0 else 0.0
        freqs = np.asarray(frequencies_cm1, dtype=float)
        freqs = freqs[freqs > 1.0]
        if freqs.size:
            x = Planck * speed_of_light * 100.0 * freqs / (Boltzmann * temperature_k)
            s_vib = Boltzmann * np.sum(x / (np.exp(x) - 1.0) - np.log(1.0 - np.exp(-x)))
        else:
            s_vib = 0.0
        entropy = s_trans + s_rot + s_vib

        u_kj_mol = u_total_j * Avogadro / 1000.0
        h_kj_mol = u_kj_mol + gas_constant * temperature_k / 1000.0
        g_kj_mol = h_kj_mol - temperature_k * entropy * Avogadro / 1000.0
        cp_j_mol_k = (2.5 + cv_vib / Boltzmann) * gas_constant

        return ThermodynamicProperties(
            temperature_k=temperature_k,
            q_total=q_total,
            q_trans=q_trans,
            q_rot=q_rot,
            q_vib=q_vib,
            internal_energy_kj_mol=u_kj_mol,
            enthalpy_kj_mol=h_kj_mol,
            entropy_j_mol_k=entropy * Avogadro,
            gibbs_kj_mol=g_kj_mol,
            heat_capacity_j_mol_k=cp_j_mol_k,
            zpe_kj_mol=zpe_j * Avogadro / 1000.0,
        )
