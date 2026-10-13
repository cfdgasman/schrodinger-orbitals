import numpy as np
import pytest

from quantum import harmonic_oscillator, hydrogen_3d, hydrogen_radial, radial_exact, real_orbital


def test_oscillator_energies_and_order():
    e = []
    for n in (200, 400):
        E, _, _ = harmonic_oscillator(n, order=4)
        e.append(np.abs(E - (np.arange(8) + 0.5)).max())
    assert e[1] < 1e-4
    assert np.log2(e[0] / e[1]) == pytest.approx(4, abs=0.2)


def test_oscillator_eigenvectors_orthonormal():
    _, psi, x = harmonic_oscillator(300, order=2)
    h = x[1] - x[0]
    assert np.allclose(psi.T @ psi * h, np.eye(psi.shape[1]), atol=1e-10)


@pytest.mark.parametrize("l", [0, 1, 2])
def test_hydrogen_radial_energies(l):
    E, _, _ = hydrogen_radial(l, n_points=3000, r_max=120.0, k=2)
    exact = [-1 / (2 * n * n) for n in (l + 1, l + 2)]
    assert np.allclose(E, exact, atol=5e-4)


def test_exact_radial_functions_are_normalised():
    r = np.linspace(0, 200, 200_001)
    for n, l in ((1, 0), (2, 1), (3, 2), (4, 0)):
        assert np.trapezoid((r * radial_exact(n, l, r)) ** 2, r) == pytest.approx(1.0, abs=1e-6)


def test_real_orbitals_are_orthogonal():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-30, 30, (400_000, 3))
    vol = 60.0**3
    a = real_orbital(2, 1, 1, *pts.T)
    b = real_orbital(2, 1, -1, *pts.T)
    assert abs(np.mean(a * b) * vol) < 0.03
    assert np.mean(a * a) * vol == pytest.approx(1.0, abs=0.05)


def test_3d_grid_ground_state_and_2p_degeneracy():
    # a block eigensolver captures the whole degenerate 2p subspace
    # (single-vector Lanczos with k = 5..8 finds only two of the three copies on this grid)
    E, _, _ = hydrogen_3d(24, 24.0, 6, method="lobpcg")
    assert -0.5 < E[0] < -0.40  # discretisation error of the Coulomb cusp, O(h^2)-like
    assert np.ptp(E[1:4]) < 1e-5  # the 2p triplet is exactly degenerate on the cubic grid
