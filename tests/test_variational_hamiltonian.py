"""Unit tests for the variational Wigner-basis Hamiltonian."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.constants import gas_constant

from pyqchem.variational_hamiltonian import (
    basis_dimension,
    build_hamiltonian,
    converge_lmax,
    diagonalize,
    series_lmax_from_coeff_count,
    solve_variational_hamiltonian,
    thermodynamics_from_eigenvalues,
)


def test_basis_dimension() -> None:
    """Basis size matches sum_l (2l+1)^2."""
    assert basis_dimension(0) == 1
    assert basis_dimension(1) == 1 + 9
    assert basis_dimension(2) == 1 + 9 + 25
    assert basis_dimension(3) == 1 + 9 + 25 + 49


def test_series_lmax_from_coeff_count() -> None:
    """Coefficient counts map back to Lmax."""
    assert series_lmax_from_coeff_count(0) == 0
    assert series_lmax_from_coeff_count(1) == 0
    assert series_lmax_from_coeff_count(basis_dimension(2)) == 2
    with pytest.raises(ValueError):
        series_lmax_from_coeff_count(2)


def test_hamiltonian_is_hermitian() -> None:
    """Kinetic plus a small potential yields a Hermitian matrix."""
    moments = np.array([2.5, 3.0, 3.5])
    # series Lmax=1 has 1 + 9 = 10 coefficients
    coeffs = np.zeros(basis_dimension(1), dtype=np.complex128)
    coeffs[0] = 0.01 + 0.0j
    coeffs[4] = 0.002 - 0.001j
    h = build_hamiltonian(lmax=2, moments_amu_a2=moments, coeffs=coeffs, series_lmax=1)
    assert h.shape == (basis_dimension(2), basis_dimension(2))
    assert torch.allclose(h, h.conj().T, atol=1e-12)


def test_free_rotor_spherical_top_spectrum() -> None:
    """Spherical free-rotor eigenvalues match B l(l+1) with (2l+1)^2 degeneracy."""
    i = 4.0
    moments = np.array([i, i, i])
    lmax = 3
    h = build_hamiltonian(lmax=lmax, moments_amu_a2=moments, coeffs=None)
    eigvals = diagonalize(h).detach().cpu().numpy()

    from pyqchem.variational_hamiltonian import _rotational_constants_hartree

    a, b, c, _kappa = _rotational_constants_hartree(i, i, i)
    assert abs(a - b) < 1e-30 and abs(b - c) < 1e-30

    expected: list[float] = []
    for el in range(lmax + 1):
        expected.extend([b * el * (el + 1)] * ((2 * el + 1) ** 2))
    expected_arr = np.array(expected, dtype=float)
    order = np.argsort(expected_arr)
    np.testing.assert_allclose(eigvals, expected_arr[order], rtol=0, atol=1e-10)


def test_thermodynamics_sanity() -> None:
    """Free-rotor thermo is positive and satisfies H = U + RT."""
    moments = np.array([3.0, 3.0, 3.0])
    result = solve_variational_hamiltonian(
        lmax=4,
        moments_amu_a2=moments,
        temperature_k=300.0,
        coeffs=None,
        symmetry_number=1.0,
    )
    assert result.q > 0.0
    assert result.entropy_j_mol_k > 0.0
    assert result.heat_capacity_j_mol_k > 0.0
    assert result.zpe_hartree == pytest.approx(result.eigenvalues_hartree.min())
    assert result.enthalpy_kj_mol == pytest.approx(
        result.internal_energy_kj_mol + gas_constant * 300.0 / 1000.0,
        rel=0,
        abs=1e-9,
    )


def test_thermodynamics_from_eigenvalues_cp_gt_cv() -> None:
    """Cp exceeds the fluctuation Cv by R."""
    moments = np.array([2.0, 3.0, 4.0])
    h = build_hamiltonian(lmax=3, moments_amu_a2=moments, coeffs=None)
    eigvals = diagonalize(h)
    thermo = thermodynamics_from_eigenvalues(eigvals, temperature_k=298.15, lmax=3)
    assert thermo.heat_capacity_j_mol_k > gas_constant


def test_converge_lmax_free_rotor() -> None:
    """Lmax convergence returns a result at or above the start cutoff."""
    # Small moments so high-l Boltzmann factors decay before the matrix grows large.
    moments = np.array([0.5, 0.5, 0.5])
    result = converge_lmax(
        lmax_start=1,
        moments_amu_a2=moments,
        temperature_k=100.0,
        coeffs=None,
        max_lmax=10,
    )
    assert result.lmax >= 1
    assert result.q > 0.0
