"""Variational Wigner-basis rotational Hamiltonian and thermodynamics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import factorial, sqrt

import numpy as np
import torch
from scipy.constants import Avogadro, Boltzmann, gas_constant

# Physical constants matching wigner-basis-schrodinger/src/hamiltonian.cpp
_EHARTREE = 4.35974434e-18  # J/Hartree
_AMU = 1.660538921e-27  # kg/amu
_HBAR = 1.054571726e-34  # J.s
_HBAR1 = _HBAR / _EHARTREE  # Hartree.s
_HBAR2 = _HBAR * 1e20 / _AMU  # amu.Angstrom^2 / s

# Convergence thresholds matching the C++ driver
_DQ_TOL = 1e-4
_DS_TOL = 1e-2  # cal/(mol.K) in C++; here applied to entropy in J/(mol.K) after converting criterion
_DU_TOL = 1e-3  # kcal/mol in C++; here applied to internal energy in kJ/mol after converting criterion
_DS_TOL_SI = _DS_TOL * 4.184  # cal -> J
_DU_TOL_SI = _DU_TOL * 4.184  # kcal -> kJ


@lru_cache(maxsize=None)
def wigner_3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    """
    Evaluate a Wigner 3j symbol for integer angular momenta.

    Parameters
    ----------
    j1, j2, j3 : int
        Angular-momentum quantum numbers.
    m1, m2, m3 : int
        Projection quantum numbers.

    Returns
    -------
    float
        Value of (j1 j2 j3 ; m1 m2 m3).
    """
    if m1 + m2 + m3 != 0:
        return 0.0
    if abs(m1) > j1 or abs(m2) > j2 or abs(m3) > j3:
        return 0.0
    if j3 > j1 + j2 or j3 < abs(j1 - j2):
        return 0.0

    t1 = j2 - m1 - j3
    t2 = j1 + m2 - j3
    t3 = j1 + j2 - j3
    t4 = j1 - m1
    t5 = j2 + m2

    t_min = max(0, t1, t2)
    t_max = min(t3, t4, t5)
    if t_min > t_max:
        return 0.0

    # Edmonds / Racah series (exact for modest integer j used here).
    sign = (-1) ** (j1 - j2 - m3)
    prefactor = sqrt(
        factorial(j1 + j2 - j3)
        * factorial(j1 - j2 + j3)
        * factorial(-j1 + j2 + j3)
        / factorial(j1 + j2 + j3 + 1)
        * factorial(j1 + m1)
        * factorial(j1 - m1)
        * factorial(j2 + m2)
        * factorial(j2 - m2)
        * factorial(j3 + m3)
        * factorial(j3 - m3)
    )

    total = 0.0
    for t in range(t_min, t_max + 1):
        denom = (
            factorial(t)
            * factorial(t3 - t)
            * factorial(t4 - t)
            * factorial(t5 - t)
            * factorial(t - t1)
            * factorial(t - t2)
        )
        total += (-1) ** t / denom
    return float(sign * prefactor * total)


@dataclass
class VariationalThermoResult:
    """
    Canonical thermodynamics from a variational rotational spectrum.

    Parameters
    ----------
    temperature_k : float
        Temperature in Kelvin.
    lmax : int
        Symmetric-top basis cutoff used for the Hamiltonian.
    eigenvalues_hartree : np.ndarray
        Energy eigenvalues in Hartree, ascending.
    q : float
        Rotational partition function Q = Q0 / sigma.
    zpe_hartree : float
        Lowest eigenvalue (zero-point energy) in Hartree.
    internal_energy_kj_mol : float
        Mean internal energy in kJ/mol.
    enthalpy_kj_mol : float
        Enthalpy H = U + RT in kJ/mol.
    entropy_j_mol_k : float
        Entropy in J/(mol.K).
    heat_capacity_j_mol_k : float
        Constant-pressure heat capacity Cp = Cv + R in J/(mol.K).
    """

    temperature_k: float
    lmax: int
    eigenvalues_hartree: np.ndarray
    q: float
    zpe_hartree: float
    internal_energy_kj_mol: float
    enthalpy_kj_mol: float
    entropy_j_mol_k: float
    heat_capacity_j_mol_k: float


def basis_dimension(lmax: int) -> int:
    """
    Return the dimension of the |lmk> basis up to `lmax`.

    Parameters
    ----------
    lmax : int
        Maximum angular momentum quantum number (inclusive).

    Returns
    -------
    int
        Number of basis states: sum_{l=0}^{lmax} (2l+1)^2.
    """
    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    return int((lmax + 1) * (2 * lmax + 1) * (2 * lmax + 3) // 3)


def series_lmax_from_coeff_count(n_coeffs: int) -> int:
    """
    Infer potential-expansion Lmax from the number of a_LMK coefficients.

    Parameters
    ----------
    n_coeffs : int
        Length of the flat coefficient array ordered by nested L, M, K.

    Returns
    -------
    int
        Lmax such that basis_dimension(Lmax) == n_coeffs.

    Raises
    ------
    ValueError
        If `n_coeffs` does not match any Lmax dimension.
    """
    if n_coeffs == 0:
        return 0
    lmax = 0
    while basis_dimension(lmax) < n_coeffs:
        lmax += 1
    if basis_dimension(lmax) != n_coeffs:
        raise ValueError(
            f"coeff count {n_coeffs} is not a valid Wigner series dimension "
            f"(expected form sum_L (2L+1)^2)"
        )
    return lmax


def _rotational_constants_hartree(ix: float, iy: float, iz: float) -> tuple[float, float, float, float]:
    """
    Convert principal moments (amu.Angstrom^2) to A, B, C, kappa in Hartree.

    Parameters
    ----------
    ix, iy, iz : float
        Principal moments of inertia in amu.Angstrom^2 (C++ I[0], I[1], I[2]).

    Returns
    -------
    tuple[float, float, float, float]
        (A, B, C, kappa) with A = hbar^2/(2 Iy), B = hbar^2/(2 Iz), C = hbar^2/(2 Ix).
    """
    a = _HBAR1 * _HBAR2 / (2.0 * iy)
    b = _HBAR1 * _HBAR2 / (2.0 * iz)
    c = _HBAR1 * _HBAR2 / (2.0 * ix)
    if a == b and a == c:
        kappa = 0.0
    else:
        kappa = (2.0 * b - (a + c)) / (a - c)
    return a, b, c, kappa


def _basis_indices(lmax: int) -> list[tuple[int, int, int]]:
    """
    Enumerate |lmk> quantum numbers in C++ loop order.

    Parameters
    ----------
    lmax : int
        Maximum angular momentum.

    Returns
    -------
    list[tuple[int, int, int]]
        List of (l, m, k) for each basis index.
    """
    return [(el, m, k) for el in range(lmax + 1) for m in range(-el, el + 1) for k in range(-el, el + 1)]


def build_hamiltonian(
    lmax: int,
    moments_amu_a2: np.ndarray | torch.Tensor,
    coeffs: np.ndarray | torch.Tensor | list[complex] | None = None,
    series_lmax: int | None = None,
) -> torch.Tensor:
    """
    Build the dense complex Hermitian rotational Hamiltonian in the |lmk> basis.

    Parameters
    ----------
    lmax : int
        Symmetric-top basis cutoff.
    moments_amu_a2 : np.ndarray | torch.Tensor
        Length-3 principal moments (Ix, Iy, Iz) in amu.Angstrom^2.
    coeffs : np.ndarray | torch.Tensor | list[complex] | None
        Flat complex a_LMK potential coefficients (nested L, M, K). Empty or None
        selects the free-rotor (kinetic-only) Hamiltonian.
    series_lmax : int | None
        Potential expansion Lmax. If None and coeffs are provided, inferred from
        the coefficient count. Ignored when coeffs are empty.

    Returns
    -------
    torch.Tensor
        Complex Hermitian matrix of shape `(n, n)` with n = basis_dimension(lmax),
        energies in Hartree.
    """
    moments = np.asarray(moments_amu_a2, dtype=float).reshape(3)
    ix, iy, iz = float(moments[0]), float(moments[1]), float(moments[2])
    a_const, _b_const, c_const, kappa = _rotational_constants_hartree(ix, iy, iz)

    if coeffs is None:
        a_vec: list[complex] = []
    else:
        raw = np.asarray(coeffs, dtype=np.complex128).ravel()
        a_vec = [complex(z) for z in raw]

    if len(a_vec) == 0:
        series_l = 0
    elif series_lmax is None:
        series_l = series_lmax_from_coeff_count(len(a_vec))
    else:
        series_l = int(series_lmax)
        expected = basis_dimension(series_l)
        if expected != len(a_vec):
            raise ValueError(
                f"series_lmax={series_l} expects {expected} coeffs, got {len(a_vec)}"
            )

    n = basis_dimension(lmax)
    states = _basis_indices(lmax)
    h = torch.zeros((n, n), dtype=torch.complex128)

    # Kinetic (diagonal + Delta k = +/- 2) — O(n), matching the C++ special cases.
    for i, (el, _m, k) in enumerate(states):
        kinetic = 0.5 * (a_const + c_const) * el * (el + 1) + 0.5 * (a_const - c_const) * kappa * k * k
        h[i, i] += complex(kinetic, 0.0)
        if k + 2 <= el:
            coupling = (
                0.25
                * (c_const - a_const)
                * sqrt(el * (el + 1) - k * (k + 1))
                * sqrt(el * (el + 1) - (k + 1) * (k + 2))
            )
            h[i + 2, i] += complex(coupling, 0.0)
            h[i, i + 2] += complex(coupling, 0.0)

    if series_l == 0:
        return h

    # Potential from a_LMK via Wigner 3j (Hermitian fill of lower triangle).
    for i, (el, m, k) in enumerate(states):
        for j, (ell, mm, kk) in enumerate(states):
            if j > i:
                continue
            vij = 0j
            ind = 0
            sign = (-1.0) ** (mm + kk)
            for L in range(series_l + 1):
                for M in range(-L, L + 1):
                    wlm = wigner_3j(el, L, ell, m, M, -mm)
                    for K in range(-L, L + 1):
                        if wlm == 0.0:
                            ind += 1
                            continue
                        wlk = wigner_3j(el, L, ell, k, K, -kk)
                        val = sqrt(2 * ell + 1) * sqrt(2 * el + 1) * sign * wlm * wlk
                        vij += a_vec[ind] * val
                        ind += 1
            if ind != len(a_vec):
                raise RuntimeError(
                    f"potential assembly consumed {ind} of {len(a_vec)} coefficients"
                )
            if i == j:
                h[i, j] += complex(vij.real, 0.0)
            else:
                h[i, j] += vij
                h[j, i] += complex(vij.real, -vij.imag)

    return h


def diagonalize(h: torch.Tensor) -> torch.Tensor:
    """
    Diagonalize a Hermitian Hamiltonian and return ascending eigenvalues.

    Parameters
    ----------
    h : torch.Tensor
        Complex Hermitian matrix.

    Returns
    -------
    torch.Tensor
        Real eigenvalue vector (Hartree), sorted ascending.
    """
    eigvals, _ = torch.linalg.eigh(h)
    return eigvals.real


def thermodynamics_from_eigenvalues(
    eigvals: torch.Tensor | np.ndarray,
    temperature_k: float,
    symmetry_number: float = 1.0,
    ref_energy: float = 0.0,
    lmax: int = 0,
) -> VariationalThermoResult:
    """
    Compute canonical rotational thermodynamics from energy eigenvalues.

    Parameters
    ----------
    eigvals : torch.Tensor | np.ndarray
        Energy eigenvalues in Hartree.
    temperature_k : float
        Temperature in Kelvin.
    symmetry_number : float
        Rotational symmetry number sigma.
    ref_energy : float
        Reference energy subtracted from eigenvalues before Boltzmann weighting.
    lmax : int
        Basis cutoff recorded on the result (informational).

    Returns
    -------
    VariationalThermoResult
        Partition function, ZPE, U, H, S, and Cp.
    """
    if temperature_k <= 0.0:
        raise ValueError("temperature_k must be positive")
    if symmetry_number <= 0.0:
        raise ValueError("symmetry_number must be positive")

    energies = np.asarray(
        eigvals.detach().cpu().numpy() if isinstance(eigvals, torch.Tensor) else eigvals,
        dtype=float,
    ).ravel()
    e = energies - ref_energy
    beta = _EHARTREE / (Boltzmann * temperature_k)  # Hartree^-1

    # Stabilize exponentials relative to the ground state.
    e0 = float(e.min())
    e_shift = e - e0
    w = np.exp(-beta * e_shift)
    q0_shift = float(np.sum(w))
    u_shift = float(np.sum(e_shift * w) / q0_shift)
    e2_shift = float(np.sum((e_shift**2) * w) / q0_shift)
    u_hartree = u_shift + e0
    var_hartree = e2_shift - u_shift**2

    # True Q0 = exp(-beta*e0) * q0_shift; F and S use the shifted partition function.
    q = q0_shift * np.exp(-beta * e0) / symmetry_number
    f_shift = -(1.0 / beta) * np.log(q0_shift / symmetry_number)
    s_hartree_per_k = (u_shift - f_shift) / temperature_k

    # Cv from energy fluctuations; convert Hartree -> J, then to per mole.
    k_b_hartree = Boltzmann / _EHARTREE
    cv_hartree_per_k = var_hartree / (k_b_hartree * temperature_k**2)
    cv_j_mol_k = cv_hartree_per_k * _EHARTREE * Avogadro
    cp_j_mol_k = cv_j_mol_k + gas_constant

    u_kj_mol = u_hartree * _EHARTREE * Avogadro / 1000.0
    h_kj_mol = u_kj_mol + gas_constant * temperature_k / 1000.0
    s_j_mol_k = s_hartree_per_k * _EHARTREE * Avogadro

    return VariationalThermoResult(
        temperature_k=temperature_k,
        lmax=lmax,
        eigenvalues_hartree=energies,
        q=q,
        zpe_hartree=float(energies[0]),
        internal_energy_kj_mol=u_kj_mol,
        enthalpy_kj_mol=h_kj_mol,
        entropy_j_mol_k=s_j_mol_k,
        heat_capacity_j_mol_k=cp_j_mol_k,
    )


def solve_variational_hamiltonian(
    lmax: int,
    moments_amu_a2: np.ndarray | torch.Tensor,
    temperature_k: float,
    coeffs: np.ndarray | torch.Tensor | list[complex] | None = None,
    series_lmax: int | None = None,
    symmetry_number: float = 1.0,
    ref_energy: float = 0.0,
) -> VariationalThermoResult:
    """
    Build, diagonalize, and evaluate thermodynamics of the variational Hamiltonian.

    Parameters
    ----------
    lmax : int
        Symmetric-top basis cutoff.
    moments_amu_a2 : np.ndarray | torch.Tensor
        Principal moments (Ix, Iy, Iz) in amu.Angstrom^2.
    temperature_k : float
        Temperature in Kelvin.
    coeffs : np.ndarray | torch.Tensor | list[complex] | None
        Complex Wigner a_LMK coefficients; None/empty for free rotor.
    series_lmax : int | None
        Potential expansion Lmax (inferred from coeffs if omitted).
    symmetry_number : float
        Rotational symmetry number.
    ref_energy : float
        Reference energy in Hartree.

    Returns
    -------
    VariationalThermoResult
        Eigenvalues and thermodynamic properties at `temperature_k`.
    """
    h = build_hamiltonian(lmax, moments_amu_a2, coeffs=coeffs, series_lmax=series_lmax)
    eigvals = diagonalize(h)
    return thermodynamics_from_eigenvalues(
        eigvals,
        temperature_k=temperature_k,
        symmetry_number=symmetry_number,
        ref_energy=ref_energy,
        lmax=lmax,
    )


def converge_lmax(
    lmax_start: int,
    moments_amu_a2: np.ndarray | torch.Tensor,
    temperature_k: float,
    coeffs: np.ndarray | torch.Tensor | list[complex] | None = None,
    series_lmax: int | None = None,
    symmetry_number: float = 1.0,
    ref_energy: float = 0.0,
    max_lmax: int = 40,
) -> VariationalThermoResult:
    """
    Raise the basis cutoff until partition-function thermodynamics converge.

    Uses the same relative-change thresholds as wigner-basis-schrodinger
    (`|dQ| < 1e-4`, `|dS| < 1e-2 cal/(mol.K)`, `|dU| < 1e-3 kcal/mol`),
    applied after converting S/U tolerances to SI.

    Parameters
    ----------
    lmax_start : int
        Initial basis cutoff.
    moments_amu_a2 : np.ndarray | torch.Tensor
        Principal moments (Ix, Iy, Iz) in amu.Angstrom^2.
    temperature_k : float
        Temperature in Kelvin.
    coeffs : np.ndarray | torch.Tensor | list[complex] | None
        Complex Wigner a_LMK coefficients; None/empty for free rotor.
    series_lmax : int | None
        Potential expansion Lmax (inferred from coeffs if omitted).
    symmetry_number : float
        Rotational symmetry number.
    ref_energy : float
        Reference energy in Hartree.
    max_lmax : int
        Hard upper bound on the basis cutoff.

    Returns
    -------
    VariationalThermoResult
        Result at the converged `lmax`.

    Raises
    ------
    RuntimeError
        If convergence is not reached by `max_lmax`.
    """
    if lmax_start < 0:
        raise ValueError("lmax_start must be non-negative")

    q_prev = 0.0
    u_prev = 0.0
    s_prev = 0.0
    result: VariationalThermoResult | None = None

    for lmax in range(lmax_start, max_lmax + 1):
        result = solve_variational_hamiltonian(
            lmax=lmax,
            moments_amu_a2=moments_amu_a2,
            temperature_k=temperature_k,
            coeffs=coeffs,
            series_lmax=series_lmax,
            symmetry_number=symmetry_number,
            ref_energy=ref_energy,
        )
        dq = abs(result.q - q_prev)
        du = abs(result.internal_energy_kj_mol - u_prev)
        ds = abs(result.entropy_j_mol_k - s_prev)
        if lmax > lmax_start and dq < _DQ_TOL and ds < _DS_TOL_SI and du < _DU_TOL_SI:
            return result
        q_prev = result.q
        u_prev = result.internal_energy_kj_mol
        s_prev = result.entropy_j_mol_k

    assert result is not None
    raise RuntimeError(
        f"lmax convergence failed by max_lmax={max_lmax} "
        f"(last lmax={result.lmax}, dQ={dq}, dU={du}, dS={ds})"
    )
