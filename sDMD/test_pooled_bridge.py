"""
Tests for ``sDMDc.from_pooled_pairs`` and ``prime_stacks``.

Validates pooled least-squares alignment and streaming handoff (mathematical soundness).
"""

from __future__ import annotations

import numpy as np
import pytest

from sDMD.sDMD import Stacker, Delayer, sDMD_base, sDMDc


def _ls_AB(Z_minus: np.ndarray, Z_plus: np.ndarray, U_minus: np.ndarray) -> np.ndarray:
    """Return [A B] with Z_plus = [A B] @ vstack(Z_minus, U_minus)."""
    V = np.vstack([Z_minus, U_minus])
    return Z_plus @ np.linalg.pinv(V, rcond=1e-12)


def _extract_AB(obj: sDMDc) -> tuple[np.ndarray, np.ndarray]:
    """Physical (nx,nx), (nx,nu) blocks consistent with library properties."""
    nx, nu = obj.nx, obj.nu
    a_eff = obj.Uy @ obj.A @ obj.Ux[:nx, :].T
    b_eff = obj.CB[:nx, :nu] if obj.B.shape[1] >= nu else obj.B[:nx, :]
    return a_eff, b_eff


def _sequential_init_arrays(X: np.ndarray, U: np.ndarray, s: int, f: int) -> tuple[np.ndarray, np.ndarray]:
    """Replicate sDMDc.__init__ construction of (XU_hank, Y_hank) passed to sDMD_base."""
    nx, T = X.shape
    nu = U.shape[0]
    xstack0 = Stacker(nx, s)
    xstack1 = Delayer(nx * s, f)
    ustack = Delayer(nu, f)
    y_h = np.hstack([xstack0.update(x) for x in X.T])
    x_h = np.hstack([xstack1.update(x) for x in y_h.T])
    u_h = np.hstack([ustack.update(x) for x in U.T])
    xu = np.vstack([x_h, u_h])
    y_h = y_h[:, s + f :]
    xu = xu[:, s + f :]
    return xu, y_h


def test_from_pooled_matches_direct_lstsq() -> None:
    rng = np.random.default_rng(42)
    nx, nu, n_p = 4, 2, 60
    z_minus = rng.standard_normal((nx, n_p))
    u_minus = rng.standard_normal((nu, n_p))
    a_true = rng.standard_normal((nx, nx))
    b_true = rng.standard_normal((nx, nu))
    m_true = np.hstack([a_true, b_true])
    z_plus = m_true @ np.vstack([z_minus, u_minus])

    m_ls = _extract_AB(
        sDMDc.from_pooled_pairs(z_minus, z_plus, u_minus, rmin=nx + nu, rmax=50, s=1, f=1)
    )
    a_ls, b_ls = m_ls[0], m_ls[1]
    np.testing.assert_allclose(a_ls, a_true, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(b_ls, b_true, rtol=1e-10, atol=1e-10)

    m_dir = _ls_AB(z_minus, z_plus, u_minus)
    a_dir, b_dir = m_dir[:, :nx], m_dir[:, nx:]
    np.testing.assert_allclose(a_ls, a_dir, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(b_ls, b_dir, rtol=1e-10, atol=1e-10)


def test_from_pooled_same_state_as_sequential_when_same_Xin_Yin() -> None:
    """If (Z^-,U^-) and Z^+ equal sequential trim output, init state matches."""
    rng = np.random.default_rng(0)
    nx, nu, T = 3, 1, 10
    X = rng.standard_normal((nx, T))
    U = rng.standard_normal((nu, T))
    xu_seq, y_seq = _sequential_init_arrays(X, U, s=1, f=1)
    z_minus = xu_seq[:nx, :]
    u_minus = xu_seq[nx:, :]
    z_plus = y_seq

    seq = sDMDc(X, U, rmin=nx + nu, rmax=40, s=1, f=1)
    pooled = sDMDc.from_pooled_pairs(z_minus, z_plus, u_minus, rmin=nx + nu, rmax=40, s=1, f=1)

    np.testing.assert_allclose(pooled.Ux, seq.Ux, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(pooled.Uy, seq.Uy, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(pooled.Q, seq.Q, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(pooled.Pinvx, seq.Pinvx, rtol=1e-10, atol=1e-10)


def test_full_pooled_differs_from_sequential_on_same_raw_trajectory() -> None:
    """Sequential s=f=1 init omits the first physical pair; full pooled LS does not."""
    rng = np.random.default_rng(1)
    nx, nu, T = 3, 2, 12
    X = rng.standard_normal((nx, T))
    U = rng.standard_normal((nu, T))
    z_full_minus = X[:, :-1]
    z_full_plus = X[:, 1:]
    u_full_minus = U[:, :-1]

    seq = sDMDc(X, U, rmin=nx + nu, rmax=50, s=1, f=1)
    pooled_full = sDMDc.from_pooled_pairs(
        z_full_minus, z_full_plus, u_full_minus, rmin=nx + nu, rmax=50, s=1, f=1
    )
    a_seq, b_seq = _extract_AB(seq)
    a_full, b_full = _extract_AB(pooled_full)
    assert np.linalg.norm(a_seq - a_full) > 1e-6
    assert np.linalg.norm(b_seq - b_full) > 1e-6


def test_multi_segment_concat_sequential_differs_from_pooled_hstack() -> None:
    """Naive concat of segments into one (X,U) does not match pooled valid pairs."""
    rng = np.random.default_rng(2)
    nx, nu = 2, 1
    T1, T2 = 8, 9
    x1 = rng.standard_normal((nx, T1))
    u1 = rng.standard_normal((nu, T1))
    x2 = rng.standard_normal((nx, T2))
    u2 = rng.standard_normal((nu, T2))

    zm1, zp1, um1 = x1[:, :-1], x1[:, 1:], u1[:, :-1]
    zm2, zp2, um2 = x2[:, :-1], x2[:, 1:], u2[:, :-1]
    z_m = np.hstack([zm1, zm2])
    z_p = np.hstack([zp1, zp2])
    u_m = np.hstack([um1, um2])

    m_pooled = _ls_AB(z_m, z_p, u_m)

    X_bad = np.hstack([x1, x2])
    U_bad = np.hstack([u1, u2])
    seq_bad = sDMDc(X_bad, U_bad, rmin=nx + nu, rmax=30, s=1, f=1)
    m_bad = np.hstack(_extract_AB(seq_bad))

    assert m_bad.shape == m_pooled.shape
    assert np.linalg.norm(m_bad - m_pooled) > 0.05


def test_prime_stacks_enables_update_without_rank_one_during_warmup() -> None:
    """After priming, one update runs; Q should change only via update path."""
    rng = np.random.default_rng(3)
    nx, nu, T = 2, 1, 15
    X = rng.standard_normal((nx, T))
    U = rng.standard_normal((nu, T))
    z_m, z_p, u_m = X[:, :-1], X[:, 1:], U[:, :-1]

    obj = sDMDc.from_pooled_pairs(z_m, z_p, u_m, rmin=nx + nu, rmax=40, s=1, f=1)
    q0 = obj.Q.copy()
    x_last = X[:, -1:]
    u_last = U[:, -1:]
    obj.prime_stacks(x_last, u_last)
    q_after_prime = obj.Q.copy()
    np.testing.assert_allclose(q_after_prime, q0, rtol=0, atol=0)

    obj.update(X[:, -1].reshape(-1, 1), U[:, -1].reshape(-1, 1))
    assert np.linalg.norm(obj.Q - q0) > 1e-12


def test_prime_stacks_insufficient_columns_raises() -> None:
    obj = sDMDc.from_pooled_pairs(
        np.zeros((2, 5)), np.zeros((2, 5)), np.zeros((1, 5)), rmin=3, rmax=10, s=1, f=1
    )
    with pytest.raises(ValueError, match="warmup columns"):
        obj.prime_stacks(np.zeros((2, 0)), np.zeros((1, 0)))
