import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.ECLC1 import ECLC1
from models.baselines import GaussianKernelJacobian, MarkovRandomWalk


# ================= 数据生成引擎 =================
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


def generate_bumpy_sphere(n_points=1600, noise_std=0.8):
    np.random.seed(42)
    indices = np.arange(0, n_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_points)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    x_u, y_u, z_u = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    unit_coords = np.column_stack((x_u, y_u, z_u))
    simplices = ConvexHull(unit_coords).simplices
    R = 1.0 + 0.12 * np.sin(5 * theta) * np.sin(4 * phi)
    coords = unit_coords * R[:, np.newaxis]
    normals = compute_vertex_normals(coords, simplices, center_guides=coords)
    V_base = np.column_stack((-x_u * z_u, -y_u * z_u, x_u ** 2 + y_u ** 2))
    V_base -= np.sum(V_base * normals, axis=1)[:, np.newaxis] * normals
    norms = np.linalg.norm(V_base, axis=1, keepdims=True)
    V_base = np.where(norms > 0, V_base / norms, V_base)
    equator_idx = np.argmin(np.abs(z_u))
    V_inj = inject_macro_source(coords, normals, V_base, center=coords[equator_idx], radius=1.0)
    return coords, apply_noise(coords, V_inj, normals, noise_std), normals, simplices


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


# ================= 统一制图引擎 =================
def plot_manifold_subplot(ax, coords, simplices, V_plot, sings, title, elev, azim):
    ax.set_title(title, fontsize=20, fontweight='bold', pad=8)
    ax.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2], triangles=simplices, color='whitesmoke', alpha=0.3,
                    edgecolor='darkgray', linewidth=0.1)

    sub_idx = np.random.choice(coords.shape[0], size=min(600, coords.shape[0]), replace=False)
    max_range = np.max([np.ptp(coords[:, 0]), np.ptp(coords[:, 1]), np.ptp(coords[:, 2])]) / 2.0

    ax.quiver(coords[sub_idx, 0], coords[sub_idx, 1], coords[sub_idx, 2],
              V_plot[sub_idx, 0], V_plot[sub_idx, 1], V_plot[sub_idx, 2],
              length=max_range * 0.15, normalize=True, color='mediumseagreen', alpha=0.4, linewidth=0.5)

    if sings:
        f_p, f_c = np.array([s[0] for s in sings]), np.array([s[1] for s in sings])
        if len(f_p[f_c > 0]) > 0:
            ax.scatter(f_p[f_c > 0, 0], f_p[f_c > 0, 1], f_p[f_c > 0, 2], color='red', s=400, marker='*',
                       edgecolors='black', linewidths=1.2, zorder=10)
        if len(f_p[f_c < 0]) > 0:
            ax.scatter(f_p[f_c < 0, 0], f_p[f_c < 0, 1], f_p[f_c < 0, 2], color='blue', s=350, marker='X',
                       edgecolors='white', linewidths=1.2, zorder=10)

        for pt, chg in zip(f_p, f_c):
            z_off = max_range * 0.15 if chg > 0 else -max_range * 0.15
            label = "Source/Sink (+1)" if chg > 0 else "Saddle (-1)"
            ax.text(pt[0], pt[1], pt[2] + z_off, f"{chg:+d}", color='black', fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'), zorder=100,
                    ha='center', va='center')

    mid_x, mid_y, mid_z = np.mean(coords[:, 0]), np.mean(coords[:, 1]), np.mean(coords[:, 2])
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.axis('off')
    ax.view_init(elev=elev, azim=azim)


def main():
    print("=" * 80)
    print(" SimExp 1 Compare: 跨界方法盲测对比 (ECL vs Spatial Stats vs Graph ML)")
    print("=" * 80)

    # 1. 实例化三大门派算法
    stat_baseline = GaussianKernelJacobian(bandwidth=0.35, speed_threshold=0.25)
    ml_baseline = MarkovRandomWalk(beta=15.0, top_percentile=98.5)
    ecl_model = ECLC1(tau=2.0, cooling_iterations=15, dt=0.2, fdr_alpha=0.05)

    datasets = [
        ("Bumpy Sphere ($S^2$)", generate_bumpy_sphere, 20, 45, 0),
        ("Torus ($T^2$)", generate_torus, 45, 45, 1)
    ]

    fig = plt.figure(figsize=(24, 11))

    for name, generator, elev, azim, row_idx in datasets:
        print(f"\n[*] 激荡流形: {name} ($\sigma=0.8$) ...")
        coords, V_noisy, normals, simplices = generator()

        print("  -> [Baseline 1] Gaussian Jacobian (统计平滑求导派)...")
        _, sings_stat, V_smooth, _ = stat_baseline.fit(coords, V_noisy, normals, simplices)

        print("  -> [Baseline 2] Markov Random Walk (图机器学习派)...")
        _, sings_ml, _, _ = ml_baseline.fit(coords, V_noisy, normals, simplices)

        print("  -> [Proposed] ECL-c1 (代数拓扑与经验假设检验)...")
        _, sings_ecl, V_cooled, _ = ecl_model.fit(coords, V_noisy, normals, simplices)

        # 列 1: Raw
        ax1 = fig.add_subplot(2, 4, row_idx * 4 + 1, projection='3d')
        plot_manifold_subplot(ax1, coords, simplices, V_noisy, [], f"{name}\n1. Raw Noisy Vector Field", elev, azim)

        # 列 2: Stats
        ax2 = fig.add_subplot(2, 4, row_idx * 4 + 2, projection='3d')
        plot_manifold_subplot(ax2, coords, simplices, V_smooth, sings_stat,
                              f"2. Spatial Stats (Gaussian+Jacobian)\nIdentified N={len(sings_stat)}",
                              elev, azim)

        # 列 3: ML
        ax3 = fig.add_subplot(2, 4, row_idx * 4 + 3, projection='3d')
        plot_manifold_subplot(ax3, coords, simplices, V_noisy, sings_ml,
                              f"3. Graph ML (Markov Walk)\nIdentified N={len(sings_ml)}", elev,
                              azim)

        # 列 4: ECL
        ax4 = fig.add_subplot(2, 4, row_idx * 4 + 4, projection='3d')
        plot_manifold_subplot(ax4, coords, simplices, V_cooled, sings_ecl,
                              f"4. Ours (ECL Framework)\nIdentified N={len(sings_ecl)}", elev, azim)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'simexp1_compare.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[*] 世纪对决大图已出炉！请审阅: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()