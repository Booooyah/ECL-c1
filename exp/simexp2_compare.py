import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.ECLC1 import ECLC1
from models.baselines import GaussianKernelJacobian, MarkovRandomWalk


# ================= Data Generation Engine =================
def fix_orientation(coords, simplices, normals):
    simplices = np.copy(simplices)
    for i in range(len(simplices)):
        p0, p1, p2 = coords[simplices[i, 0]], coords[simplices[i, 1]], coords[simplices[i, 2]]
        n = normals[simplices[i, 0]]
        if np.dot(np.cross(p1 - p0, p2 - p0), n) < 0:
            simplices[i, 1], simplices[i, 2] = simplices[i, 2], simplices[i, 1]
    return simplices


def inject_macro_source(coords, normals, V_base, center, radius=1.0):
    V_new = np.copy(V_base)
    for i in range(len(coords)):
        d = coords[i] - center
        dist = np.linalg.norm(d)
        if dist < radius:
            t = d - np.dot(d, normals[i]) * normals[i]
            t_norm = np.linalg.norm(t)
            t = t / t_norm if t_norm > 1e-5 else np.zeros_like(t)
            w = 0.5 * (1 + np.cos(np.pi * dist / radius))
            V_new[i] = w * t + (1 - w) * V_base[i]
    V_new -= np.sum(V_new * normals, axis=1)[:, np.newaxis] * normals
    norms = np.linalg.norm(V_new, axis=1, keepdims=True)
    return np.where(norms > 0, V_new / norms, V_new)


def apply_noise(coords, V, normals, noise_std):
    V_noisy = V + np.random.randn(*V.shape) * noise_std
    V_noisy -= np.sum(V_noisy * normals, axis=1)[:, np.newaxis] * normals
    norms = np.linalg.norm(V_noisy, axis=1, keepdims=True)
    return np.where(norms > 0, V_noisy / norms, V_noisy)


def generate_hemisphere(noise_std=0.8):
    np.random.seed(42)
    u = np.linspace(-2.2, 2.2, 60)
    v = np.linspace(-2.2, 2.2, 60)
    U, V = np.meshgrid(u, v)
    U, V = U.flatten(), V.flatten()
    mask = (U ** 2 + V ** 2) < 2.0 ** 2
    U, V = U[mask], V[mask]
    U += np.random.randn(len(U)) * 0.015
    V += np.random.randn(len(V)) * 0.015
    simplices = Delaunay(np.column_stack((U, V))).simplices
    z = np.sqrt(np.maximum(2.0 ** 2 - U ** 2 - V ** 2, 0))
    coords = np.column_stack((U, V, z))
    normals = coords / 2.0
    simplices = fix_orientation(coords, simplices, normals)

    # Divergent base flow field
    V_base = np.column_stack((U, V, np.zeros_like(U)))
    V_base -= np.sum(V_base * normals, axis=1)[:, np.newaxis] * normals
    norms = np.linalg.norm(V_base, axis=1, keepdims=True)
    V_base = np.where(norms > 0, V_base / norms, np.array([1.0, 0.0, 0.0]))

    # Inject an eccentric macroscopic genuine singularity
    center_pt = np.array([0.9, 0.0, np.sqrt(max(2.0 ** 2 - 0.9 ** 2, 0))])
    V_inj = inject_macro_source(coords, normals, V_base, center=center_pt, radius=0.8)
    return coords, apply_noise(coords, V_inj, normals, noise_std), normals, simplices


def generate_hyperbolic_saddle(noise_std=0.8):
    np.random.seed(42)
    u = np.linspace(-2.5, 2.5, 60)
    v = np.linspace(-2.5, 2.5, 60)
    U, V = np.meshgrid(u, v)
    U, V = U.flatten(), V.flatten()
    mask = (U ** 2 + V ** 2) < 2.3 ** 2
    U, V = U[mask], V[mask]
    U += np.random.randn(len(U)) * 0.015
    V += np.random.randn(len(V)) * 0.015
    simplices = Delaunay(np.column_stack((U, V))).simplices
    z = 0.5 * (U ** 2 - V ** 2)
    coords = np.column_stack((U, V, z))

    normals = np.column_stack((-U, V, np.ones_like(U)))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    simplices = fix_orientation(coords, simplices, normals)

    # Saddle base flow field
    V_base = np.column_stack((U, -V, np.zeros_like(U)))
    V_base -= np.sum(V_base * normals, axis=1)[:, np.newaxis] * normals
    norms = np.linalg.norm(V_base, axis=1, keepdims=True)
    V_base = np.where(norms > 0, V_base / norms, np.array([1.0, 0.0, 0.0]))

    # Inject an eccentric macroscopic genuine singularity
    center_pt = np.array([0.8, 0.8, 0.5 * (0.8 ** 2 - 0.8 ** 2)])
    V_inj = inject_macro_source(coords, normals, V_base, center=center_pt, radius=0.8)
    return coords, apply_noise(coords, V_inj, normals, noise_std), normals, simplices


# ================= Unified Plotting Engine =================
def plot_manifold_subplot(ax, coords, simplices, V_plot, sings, title, elev, azim):
    ax.set_title(title, fontsize=20, fontweight='bold', pad=8)
    ax.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2], triangles=simplices, color='whitesmoke', alpha=0.3,
                    edgecolor='darkgray', linewidth=0.1)

    sub_idx = np.random.choice(coords.shape[0], size=min(450, coords.shape[0]), replace=False)
    max_range = np.max([np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), np.ptp(coords[:, 2])]) / 2.0

    # Reduce arrow length to prevent occlusion of topological singularities
    ax.quiver(coords[sub_idx, 0], coords[sub_idx, 1], coords[sub_idx, 2],
              V_plot[sub_idx, 0], V_plot[sub_idx, 1], V_plot[sub_idx, 2],
              length=max_range * 0.12, normalize=True, color='mediumseagreen', alpha=0.5, linewidth=0.6)

    if sings:
        f_p, f_c = np.array([s[0] for s in sings]), np.array([s[1] for s in sings])
        if len(f_p[f_c > 0]) > 0:
            ax.scatter(f_p[f_c > 0, 0], f_p[f_c > 0, 1], f_p[f_c > 0, 2], color='red', s=350, marker='*',
                       edgecolors='black', linewidths=1.2, zorder=10)
        if len(f_p[f_c < 0]) > 0:
            ax.scatter(f_p[f_c < 0, 0], f_p[f_c < 0, 1], f_p[f_c < 0, 2], color='blue', s=300, marker='X',
                       edgecolors='white', linewidths=1.2, zorder=10)

        for pt, chg in zip(f_p, f_c):
            z_off = max_range * 0.1 if chg > 0 else -max_range * 0.1
            label = "+1" if chg > 0 else "-1"
            ax.text(pt[0], pt[1], pt[2] + z_off, label, color='black', fontsize=9, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.2'), zorder=100,
                    ha='center', va='center', clip_on=False)

    mid_x, mid_y, mid_z = np.mean(coords[:, 0]), np.mean(coords[:, 1]), np.mean(coords[:, 2])
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.axis('off')
    ax.view_init(elev=elev, azim=azim)


def main():
    print("=" * 80)
    print(" SimExp 2 Compare: Evaluating Boundary Immunity on Open Manifolds")
    print("=" * 80)

    # Instantiate the three methodological paradigms
    stat_baseline = GaussianKernelJacobian(bandwidth=0.4, speed_threshold=0.25)
    ml_baseline = MarkovRandomWalk(beta=15.0, top_percentile=98.5)
    ecl_model = ECLC1(tau=2.0, cooling_iterations=15, dt=0.2, fdr_alpha=0.05)

    datasets = [
        ("Hemisphere Bowl (Outward Flow)", generate_hemisphere, 35, 45, 0),
        ("Hyperbolic Saddle (Open Saddle)", generate_hyperbolic_saddle, 35, 45, 1)
    ]

    fig = plt.figure(figsize=(24, 11))

    for name, generator, elev, azim, row_idx in datasets:
        print(f"\n[*] Probing boundary vulnerabilities on: {name.split(' ')[0]} (sigma=0.8) ...")
        coords, V_noisy, normals, simplices = generator()

        print("  -> [Baseline 1] Gaussian Jacobian (Asymmetric kernel truncation and derivative amplification)...")
        _, sings_stat, V_smooth, _ = stat_baseline.fit(coords, V_noisy, normals, simplices)

        print("  -> [Baseline 2] Markov Random Walk (Structural edge pooling of probability mass)...")
        _, sings_ml, _, _ = ml_baseline.fit(coords, V_noisy, normals, simplices)

        print("  -> [Proposed] ECL-c1 (Absolute boundary immunity via Cech closed-cocycle integration)...")
        _, sings_ecl, V_cooled, _ = ecl_model.fit(coords, V_noisy, normals, simplices)

        # Column 1: Raw
        ax1 = fig.add_subplot(2, 4, row_idx * 4 + 1, projection='3d')
        plot_manifold_subplot(ax1, coords, simplices, V_noisy, [], f"{name}\n1. Raw Noisy Vector Field", elev, azim)

        # Column 2: Stats
        ax2 = fig.add_subplot(2, 4, row_idx * 4 + 2, projection='3d')
        plot_manifold_subplot(ax2, coords, simplices, V_smooth, sings_stat,
                              f"2. Spatial Stats (Gaussian+Jacobian)\nIdentified N={len(sings_stat)}",
                              elev, azim)

        # Column 3: ML
        ax3 = fig.add_subplot(2, 4, row_idx * 4 + 3, projection='3d')
        plot_manifold_subplot(ax3, coords, simplices, V_noisy, sings_ml,
                              f"3. Graph ML (Markov Walk)\nIdentified N={len(sings_ml)}", elev,
                              azim)

        # Column 4: ECL
        ax4 = fig.add_subplot(2, 4, row_idx * 4 + 4, projection='3d')
        plot_manifold_subplot(ax4, coords, simplices, V_cooled, sings_ecl,
                              f"4. Ours (ECL Framework)\nIdentified N={len(sings_ecl)}", elev, azim)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'simexp2_compare.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[*] Boundary immunity benchmarking plot successfully generated. Saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
