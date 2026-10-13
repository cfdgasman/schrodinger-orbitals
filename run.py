"""Schrodinger equation as a sparse eigenproblem: oscillator, hydrogen (radial and full 3D), orbital gallery."""

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from quantum import harmonic_oscillator, hydrogen_3d, hydrogen_radial, radial_exact, real_orbital


def oscillator():
    print("1D harmonic oscillator, max |E_n - (n + 1/2)| over the 8 lowest states")
    print("| points | h | 2nd-order FD | 4th-order FD |\n|---|---|---|---|")
    ns = [100, 200, 400, 800]
    e2, e4, hs = [], [], []
    for n in ns:
        E2, _, x = harmonic_oscillator(n, order=2)
        E4, _, _ = harmonic_oscillator(n, order=4)
        exact = np.arange(8) + 0.5
        e2.append(np.abs(E2 - exact).max())
        e4.append(np.abs(E4 - exact).max())
        hs.append(x[1] - x[0])
        print(f"| {n} | {hs[-1]:.4f} | {e2[-1]:.2e} | {e4[-1]:.2e} |")
    print(f"observed orders: {np.log2(e2[-2] / e2[-1]):.2f} and {np.log2(e4[-2] / e4[-1]):.2f}")
    E, psi, x = harmonic_oscillator(400, order=4)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.plot(x, 0.5 * x**2, "k-", lw=1)
    for k in range(6):
        a1.plot(x, E[k] + 0.6 * psi[:, k] * np.sign(psi[np.argmax(np.abs(psi[:, k])), k]), lw=1.5)
        a1.axhline(E[k], color="grey", lw=0.5, ls=":")
    a1.set(xlim=(-5, 5), ylim=(0, 6.5), xlabel="x", ylabel="E,  ψ (offset)", title="Harmonic oscillator: V = x²/2, E_n = n + ½")
    a2.loglog(hs, e2, "o-", label="2nd-order FD")
    a2.loglog(hs, e4, "s-", label="4th-order FD")
    a2.loglog(hs, e2[0] * (np.array(hs) / hs[0]) ** 2, "k:", lw=0.8)
    a2.loglog(hs, e4[0] * (np.array(hs) / hs[0]) ** 4, "k--", lw=0.8)
    a2.set(xlabel="h", ylabel="max energy error", title="Convergence (slopes 2 and 4)")
    a2.grid(alpha=0.3, which="both")
    a2.legend()
    fig.tight_layout()
    fig.savefig("docs/oscillator.png", dpi=120)
    plt.close(fig)


def radial():
    print("\nHydrogen radial equation (4000 points, r_max = 150): E_nl vs -1/(2n^2)")
    print("| state | computed | exact | error |\n|---|---|---|---|")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    for ax, l in zip(axes, (0, 1, 2)):
        E, u, r = hydrogen_radial(l, n_points=4000, r_max=150.0, k=3)
        for j in range(3):
            n = l + 1 + j
            name = f"{n}{'spd'[l]}"
            print(f"| {name} | {E[j]:.7f} | {-1 / (2 * n * n):.7f} | {abs(E[j] + 1 / (2 * n * n)):.1e} |")
            P = u[:, j] ** 2
            ax.plot(r, P, lw=2, alpha=0.6, color=f"C{j}", label=f"{name}: FD")
            ax.plot(r, (r * radial_exact(n, l, r)) ** 2, "k--", lw=0.8)
        ax.set(xlim=(0, 30 + 10 * l), xlabel="r (Bohr)", ylabel="r² |R_nl|²", title=f"l = {l} (dashed: exact)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("docs/radial.png", dpi=120)
    plt.close(fig)


def rotate_subspace(V, funcs):
    """Rotate a degenerate eigenvector block V (N x m) to best match the target functions (N x m)."""
    C = V.T @ funcs  # overlaps
    U, _, Wt = np.linalg.svd(C)
    return V @ (U @ Wt)


def full_3d():
    print("\nHydrogen on a 3D Cartesian grid (7-point Laplacian, -1/r potential, lowest 14 states)")
    print("| grid spacing h | unknowns | E(1s) | E(2p) | E(2s) | solve time |\n|---|---|---|---|---|---|")
    for n in (32, 48, 64):
        t0 = time.perf_counter()
        E, psi, x = hydrogen_3d(n, 24.0, 14)
        h = x[1] - x[0]
        print(f"| {h:.3f} | {n**3:,} | {E[0]:.5f} | {E[1]:.5f} | {E[4]:.5f} | {time.perf_counter() - t0:.0f} s |")
    print("exact: E(1s) = -0.5, E(n=2) = -0.125, E(n=3) = -0.05556")
    print("all 14 energies at h = 0.375:", " ".join(f"{e:.4f}" for e in E))

    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    j = len(x) // 2
    # identify blocks by energy clustering
    blocks, cur = [], [0]
    for k in range(1, len(E)):
        if abs(E[k] - E[cur[-1]]) < 2e-4:
            cur.append(k)
        else:
            blocks.append(cur)
            cur = [k]
    blocks.append(cur)
    panels = []
    for b in blocks:
        V = psi[..., b].reshape(-1, len(b))
        if len(b) == 3:  # p-like triplet -> px, py, pz
            V = rotate_subspace(V, np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
                                * np.exp(-np.sqrt(X**2 + Y**2 + Z**2).ravel() / 2)[:, None])
        for i, k in enumerate(b):
            panels.append((E[k], V[:, i].reshape(X.shape)))
    names = ["1s", "2p_x", "2p_y", "2p_z", "2s"]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    for ax, (e, f), name in zip(axes.ravel(), panels[:5], names):
        # y = 0 plane (x right, z up); 2p_y has its node there, so show it in the z = 0 plane (x right, y up)
        sl = f[:, :, j] if name == "2p_y" else f[:, j, :]
        m = np.abs(sl).max()
        ax.imshow(sl.T, origin="lower", extent=(x[0], x[-1], x[0], x[-1]), cmap="RdBu_r", vmin=-m, vmax=m)
        ax.set(xticks=[], yticks=[], title=f"{name}: E = {e:.4f}" + (" (z = 0 plane)" if name == "2p_y" else ""))
    fig.suptitle("Lowest eigenvectors of the 3D grid Hamiltonian (h = 0.375, 262 144 unknowns), slice y = 0, colour = sign of ψ", y=1.0)
    fig.tight_layout()
    fig.savefig("docs/grid_orbitals.png", dpi=110)
    plt.close(fig)


def orbital_gallery():
    rng = np.random.default_rng(0)
    orbitals = [(1, 0, 0, "1s"), (2, 1, 0, "2p_z"), (3, 2, 0, "3d_z²"), (3, 2, 2, "3d_x²−y²"),
                (4, 3, 0, "4f_z³"), (3, 1, 1, "3p_x")]
    clouds = []
    for n, l, m, name in orbitals:
        R = 4 * n * n
        pts = rng.uniform(-R, R, (400_000, 3))
        dens = real_orbital(n, l, m, *pts.T)
        p = dens**2
        keep = rng.random(len(p)) < p / p.max()
        clouds.append((name, pts[keep][:3000], np.sign(dens[keep][:3000]), R))
    fig = plt.figure(figsize=(12, 7.4))
    axes = [fig.add_subplot(2, 3, i + 1, projection="3d") for i in range(6)]
    arts = []
    for ax, (name, P, sgn, R) in zip(axes, clouds):
        arts.append(ax.scatter(*P.T, c=sgn, cmap="coolwarm", s=1.5, alpha=0.6))
        ax.set(xlim=(-R * 0.7, R * 0.7), ylim=(-R * 0.7, R * 0.7), zlim=(-R * 0.7, R * 0.7), title=name)
        ax.set_axis_off()
    fig.suptitle("Hydrogen orbitals: points sampled from |ψ_nlm|², coloured by the sign of ψ")
    fig.tight_layout()
    fig.savefig("docs/orbitals.png", dpi=110)

    def update(k):
        for ax in axes:
            ax.view_init(elev=20, azim=3 * k)
        return arts

    FuncAnimation(fig, update, frames=120).save("docs/orbitals.gif", writer=PillowWriter(fps=15), dpi=55)
    plt.close(fig)


if __name__ == "__main__":
    oscillator()
    radial()
    full_3d()
    orbital_gallery()
