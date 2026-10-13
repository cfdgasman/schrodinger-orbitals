"""Time-independent Schrodinger equation as a sparse eigenvalue problem (atomic units: hbar = m_e = e = 1).

    H psi = E psi,    H = -1/2 lap + V
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.special import eval_genlaguerre, factorial, sph_harm_y


def second_derivative(n, h, order=2):
    """Dirichlet second-derivative matrix on n interior points."""
    if order == 2:
        return sp.diags([np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)], [-1, 0, 1]) / h**2
    c = [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]
    return sp.diags([c[0] * np.ones(n - 2), c[1] * np.ones(n - 1), c[2] * np.ones(n),
                     c[3] * np.ones(n - 1), c[4] * np.ones(n - 2)], [-2, -1, 0, 1, 2]) / h**2


# ---------------------------------------------------------------- 1D harmonic oscillator


def harmonic_oscillator(n_points=400, L=10.0, order=2, k=8):
    """V = x^2/2 on (-L, L); exact E_n = n + 1/2."""
    x = np.linspace(-L, L, n_points + 2)[1:-1]
    h = x[1] - x[0]
    H = -0.5 * second_derivative(n_points, h, order) + sp.diags(0.5 * x**2)
    E, psi = eigsh(H.tocsc(), k=k, sigma=0.0)
    idx = np.argsort(E)
    return E[idx], psi[:, idx] / np.sqrt(h), x


# ---------------------------------------------------------------- hydrogen, radial


def hydrogen_radial(l, n_points=4000, r_max=80.0, k=4):
    """u(r) = r R(r):  -1/2 u'' + [l(l+1)/(2r^2) - 1/r] u = E u,  u(0) = u(r_max) = 0.  Exact E_n = -1/(2 n^2)."""
    r = np.linspace(0, r_max, n_points + 2)[1:-1]
    h = r[1] - r[0]
    V = l * (l + 1) / (2 * r**2) - 1 / r
    H = -0.5 * second_derivative(n_points, h) + sp.diags(V)
    E, u = eigsh(H.tocsc(), k=k, sigma=-0.6)
    idx = np.argsort(E)
    u = u[:, idx] / np.sqrt(h)
    u *= np.sign(u[np.argmax(np.abs(u), axis=0), np.arange(u.shape[1])])
    return E[idx], u, r


def radial_exact(n, l, r):
    """Normalised hydrogen radial function R_nl(r) (Bohr radius a0 = 1)."""
    rho = 2 * r / n
    norm = np.sqrt((2 / n) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
    return norm * np.exp(-rho / 2) * rho**l * eval_genlaguerre(n - l - 1, 2 * l + 1, rho)


def real_orbital(n, l, m, x, y, z):
    """Real hydrogen orbital psi_nlm (m > 0: cos-type, m < 0: sin-type)."""
    r = np.sqrt(x * x + y * y + z * z) + 1e-300
    theta = np.arccos(np.clip(z / r, -1, 1))
    phi = np.arctan2(y, x)
    Y = sph_harm_y(l, abs(m), theta, phi)
    if m > 0:
        Y = np.sqrt(2) * (-1) ** m * Y.real
    elif m < 0:
        Y = np.sqrt(2) * (-1) ** m * Y.imag
    else:
        Y = Y.real
    return radial_exact(n, l, r) * Y


# ---------------------------------------------------------------- hydrogen, full 3D grid


def hydrogen_3d(n=40, L=24.0, k=10, method="lanczos", seed=0):
    """Cartesian 7-point Hamiltonian on a cube (-L/2, L/2)^3 with a cell-centred grid (no node at r = 0).
    Returns the k lowest eigenpairs and the grid coordinates.

    method = "lanczos": ARPACK implicitly restarted Lanczos (fast, but a single-vector Krylov method
                        can miss copies of exactly degenerate eigenvalues unless k is generous)
    method = "lobpcg":  block LOBPCG with an ILU preconditioner (a block of k vectors captures
                        a whole degenerate subspace)"""
    x = -L / 2 + (np.arange(n) + 0.5) * L / n
    h = x[1] - x[0]
    D2 = second_derivative(n, h)
    I = sp.identity(n)
    lap = sp.kron(sp.kron(D2, I), I) + sp.kron(sp.kron(I, D2), I) + sp.kron(sp.kron(I, I), D2)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r = np.sqrt(X**2 + Y**2 + Z**2).ravel()
    H = (-0.5 * lap + sp.diags(-1 / r)).tocsr()
    if method == "lobpcg":
        from scipy.sparse.linalg import LinearOperator, lobpcg, spilu

        ilu = spilu((H + sp.identity(H.shape[0])).tocsc(), drop_tol=1e-3, fill_factor=5)
        X0 = np.random.default_rng(seed).standard_normal((H.shape[0], k))
        E, psi = lobpcg(H, X0, M=LinearOperator(H.shape, matvec=ilu.solve), largest=False, tol=1e-7, maxiter=600)
    else:
        E, psi = eigsh(H, k=k, which="SA", tol=1e-8)
    idx = np.argsort(E)
    return E[idx], psi[:, idx].reshape(n, n, n, k) / h**1.5, x
