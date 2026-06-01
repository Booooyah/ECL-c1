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


# ================= 数据生成引擎 =================
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


def get_grid_simplices(n_u, n_v):
    simplices = []
    for i in range(n_u):
        for j in range(n_v):
            p0, p1 = i * n_v + j, ((i + 1) % n_u) * n_v + j
            p2, p3 = i * n_v + (j + 1) % n_v, ((i + 1) % n_u) * n_v + (j + 1) % n_v
            if (i + j) % 2 == 0:
                simplices.extend([[p0, p1, p2], [p1, p3, p2]])
            else:
                simplices.extend([[p0, p1, p3], [p0, p3, p2]])
    return np.array(simplices)


def compute_vertex_normals(coords, simplices, center_guides=None):
    normals = np.zeros_like(coords)
    v0, v1, v2 = coords[simplices[:, 0]], coords[simplices[:, 1]], coords[simplices[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    if center_guides is not None:
        c_guides = (center_guides[simplices[:, 0]] + center_guides[simplices[:, 1]] + center_guides[
            simplices[:, 2]]) / 3.0
        dots = np.sum(face_normals * c_guides, axis=1)
        face_normals[dots < 0] *= -1
    np.add.at(normals, simplices[:, 0], face_normals)
    np.add.at(normals, simplices[:, 1], face_normals)
    np.add.at(normals, simplices[:, 2], face_normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.where(norms > 0, normals / norms, normals)


def generate_torus(n_u=60, n_v=30, noise_std=0.8):
    np.random.seed(42)
    R_t, r_t = 2.0, 0.8
    u = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
    v = np.linspace(0, 2 * np.pi, n_v, endpoint=False)
    U, V = np.meshgrid(u, v, indexing='ij')
    U, V = U.flatten() + np.random.randn(n_u * n_v) * 0.02, V.flatten() + np.random.randn(n_u * n_v) * 0.02
    coords = np.column_stack(
        ((R_t + r_t * np.cos(V)) * np.cos(U), (R_t + r_t * np.cos(V)) * np.sin(U), r_t * np.sin(V)))
    simplices = get_grid_simplices(n_u, n_v)
    center_guides = coords - np.column_stack((R_t * np.cos(U), R_t * np.sin(U), np.zeros_like(U)))
    normals = compute_vertex_normals(coords, simplices, center_guides=center_guides)
    V_base = np.column_stack((-np.sin(U), np.cos(U), np.zeros_like(U)))
    V_base -= np.sum(V_base * normals, axis=1)[:, np.newaxis] * normals
    norms = np.linalg.norm(V_base, axis=1, keepdims=True)
    V_base = np.where(norms > 0, V_base / norms, V_base)
    V_inj = inject_macro_source(coords, normals, V_base, center=np.array([R_t + r_t, 0, 0]), radius=1.5)
    V_inj = inject_macro_source(coords, normals, V_inj, center=np.array([-(R_t + r_t), 0, 0]), radius=1.5)
    return coords, apply_noise(coords, V_inj, normals, noise_std), normals, simplices


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

    V_base = np.column_stack((U, V, np.zeros_like(U)))
    V_base -= np.sum(V_base * normals, axis=1)[:, np.newaxis] * normals
    norms = np.linalg.norm(V_base, axis=1, keepdims=True)
    V_base = np.where(norms > 0, V_base / norms, np.array([1.0, 0.0, 0.0]))

    center_pt = np.array([0.9, 0.0, np.sqrt(max(2.0 ** 2 - 0.9 ** 2, 0))])
    V_inj = inject_macro_source(coords, normals, V_base, center=center_pt, radius=0.8)
    return coords, apply_noise(coords, V_inj, normals, noise_std), normals, simplices


# ================= 统一制图引擎 =================
def plot_manifold_subplot(ax, coords, simplices, V_plot, sings, title, elev, azim):
    ax.set_title(title, fontsize=20, fontweight='bold', pad=8)
    ax.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2], triangles=simplices, color='whitesmoke', alpha=0.3,
                    edgecolor='darkgray', linewidth=0.1)

    sub_idx = np.random.choice(coords.shape[0], size=min(450, coords.shape[0]), replace=False)
    max_range = np.max([np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), np.ptp(coords[:, 2])]) / 2.0

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
    print(" Ablation Study: 喂给对手完美降噪场 (V_cooled)，看其能否起死回生")
    print("=" * 80)

    # 正常参数的基线方法
    stat_baseline = GaussianKernelJacobian(bandwidth=0.35, speed_threshold=0.25)
    ml_baseline = MarkovRandomWalk(beta=15.0, top_percentile=98.5)
    ecl_model = ECLC1(tau=2.0, cooling_iterations=15, dt=0.2, fdr_alpha=0.05)

    datasets = [
        ("Torus (Saddle Test)", generate_torus, 45, 45, 0),
        ("Hemisphere (Boundary Test)", generate_hemisphere, 35, 45, 1)
    ]

    fig = plt.figure(figsize=(24, 11))

    for name, generator, elev, azim, row_idx in datasets:
        print(f"\n[*] 激荡流形: {name.split(' ')[0]} ($\sigma=0.8$) ...")
        coords, V_noisy, normals, simplices = generator()

        # 使用 ECL 生成完美的冷却流场 V_cooled
        print("  -> [ECL] 生成完美的宏观平滑流场 (V_cooled) ...")
        _, sings_ecl, V_cooled, _ = ecl_model.fit(coords, V_noisy, normals, simplices)

        # 🌟 核心杀招：将完美的 V_cooled 喂给传统方法，看它们接不接得住！
        print("  -> [Ablation 1] 将 V_cooled 喂给 Stats (Gaussian+Jacobian)...")
        _, sings_stat_cooled, V_smooth_cooled, _ = stat_baseline.fit(coords, V_cooled, normals, simplices)

        print("  -> [Ablation 2] 将 V_cooled 喂给 Graph ML (Markov Walk)...")
        _, sings_ml_cooled, _, _ = ml_baseline.fit(coords, V_cooled, normals, simplices)

        # 列 1: 展示完美的 V_cooled (证明不是平滑没做好)
        ax1 = fig.add_subplot(2, 4, row_idx * 4 + 1, projection='3d')
        plot_manifold_subplot(ax1, coords, simplices, V_cooled, [], f"{name}\n1. Shared Input: ECL Cooled Field", elev,
                              azim)

        # 列 2: Stats (Cooled Input)
        ax2 = fig.add_subplot(2, 4, row_idx * 4 + 2, projection='3d')
        plot_manifold_subplot(ax2, coords, simplices, V_smooth_cooled, sings_stat_cooled,
                              f"2. Stats (Given Cooled Input)\nIdentified N={len(sings_stat_cooled)}",
                              elev, azim)

        # 列 3: ML (Cooled Input)
        ax3 = fig.add_subplot(2, 4, row_idx * 4 + 3, projection='3d')
        plot_manifold_subplot(ax3, coords, simplices, V_cooled, sings_ml_cooled,
                              f"3. Graph ML (Given Cooled Input)\nIdentified N={len(sings_ml_cooled)}",
                              elev, azim)

        # 列 4: ECL (Full Framework)
        ax4 = fig.add_subplot(2, 4, row_idx * 4 + 4, projection='3d')
        plot_manifold_subplot(ax4, coords, simplices, V_cooled, sings_ecl,
                              f"4. Ours (Full ECL Framework)\nIdentified N={len(sings_ecl)}", elev,
                              azim)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'simexp3_ablation.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[*] 极其震撼的消融实验大图出炉！请审阅: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()