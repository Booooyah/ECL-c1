import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# 引入 ECL 核心模型
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.ECLC1 import ECLC1


def generate_flat_mesh(grid_size=45):
    np.random.seed(42)
    u = np.linspace(-2.5, 2.5, grid_size)
    v = np.linspace(-2.5, 2.5, grid_size)
    U, V = np.meshgrid(u, v)
    U = U.flatten() + np.random.randn(grid_size ** 2) * 0.015
    V = V.flatten() + np.random.randn(grid_size ** 2) * 0.015
    coords = np.column_stack((U, V, np.zeros_like(U)))
    simplices = Delaunay(coords[:, :2]).simplices
    normals = np.zeros_like(coords)
    normals[:, 2] = 1.0

    simplices_fixed = np.copy(simplices)
    for i in range(len(simplices_fixed)):
        p0, p1, p2 = coords[simplices_fixed[i, 0]], coords[simplices_fixed[i, 1]], coords[simplices_fixed[i, 2]]
        if np.dot(np.cross(p1 - p0, p2 - p0), normals[simplices_fixed[i, 0]]) < 0:
            simplices_fixed[i, 1], simplices_fixed[i, 2] = simplices_fixed[i, 2], simplices_fixed[i, 1]
    return coords, simplices_fixed, normals


def get_pure_noise_field(coords, normals, noise_std=1.0):
    V = np.random.randn(*coords.shape) * noise_std
    V -= np.sum(V * normals, axis=1)[:, np.newaxis] * normals
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    return np.where(norms > 0, V / norms, V)


def get_dipole_field(coords, normals, noise_std=0.5):
    x, y = coords[:, 0], coords[:, 1]
    c1, c2 = np.array([-1.2, 0.0, 0.0]), np.array([1.2, 0.0, 0.0])

    dx1, dy1 = x - c1[0], y - c1[1]
    r1_sq = dx1 ** 2 + dy1 ** 2 + 1e-2
    v1_x, v1_y = dx1 / r1_sq, dy1 / r1_sq

    dx2, dy2 = x - c2[0], y - c2[1]
    r2_sq = dx2 ** 2 + dy2 ** 2 + 1e-2
    v2_x, v2_y = -dx2 / r2_sq, -dy2 / r2_sq

    V_base = np.column_stack((v1_x + v2_x, v1_y + v2_y, np.zeros_like(x)))
    norms = np.linalg.norm(V_base, axis=1, keepdims=True)
    V_base = np.where(norms > 0, V_base / norms, np.array([1.0, 0.0, 0.0]))

    V_noisy = V_base + np.random.randn(*coords.shape) * noise_std
    V_noisy -= np.sum(V_noisy * normals, axis=1)[:, np.newaxis] * normals
    norms_n = np.linalg.norm(V_noisy, axis=1, keepdims=True)
    return np.where(norms_n > 0, V_noisy / norms_n, V_noisy), [c1, c2]


def extract_cooled_clusters(model, coords, V, normals, simplices):
    V_cooled = model.vector_field_cooling_manifold(coords, V, normals, simplices)
    _, cooled_sings = model.intrinsic_dec_3d_c1(coords, V_cooled, normals, simplices)

    if not cooled_sings: return []
    N_raw = len(cooled_sings)
    adj_matrix = np.zeros((N_raw, N_raw), dtype=bool)
    for i in range(N_raw):
        cent_i, _, L_bar_i = cooled_sings[i]
        for j in range(i + 1, N_raw):
            cent_j, _, L_bar_j = cooled_sings[j]
            if np.linalg.norm(cent_i - cent_j) <= model.tau * max(L_bar_i, L_bar_j):
                adj_matrix[i, j] = adj_matrix[j, i] = True

    visited = np.zeros(N_raw, dtype=bool)
    clusters = []
    for i in range(N_raw):
        if not visited[i]:
            comp, stack = [], [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for n in np.where(adj_matrix[node])[0]:
                    if not visited[n]:
                        visited[n] = True
                        stack.append(n)
            cluster_charge = sum([cooled_sings[idx][1] for idx in comp])
            if cluster_charge != 0:
                cluster_cents = [cooled_sings[idx][0] for idx in comp]
                clusters.append((np.mean(cluster_cents, axis=0), cluster_charge))
    return clusters


def empirical_hypothesis_test(clusters, emp_lambda_opp, alpha=0.05):
    if not clusters: return []
    clusters_with_pval = []
    for i, (cent_i, chg_i) in enumerate(clusters):
        min_dist = float('inf')
        for j, (cent_j, chg_j) in enumerate(clusters):
            if chg_i * chg_j < 0:
                dist = np.linalg.norm(cent_i - cent_j)
                if dist < min_dist: min_dist = dist
        if min_dist == float('inf'):
            p_value = 0.0
        else:
            exponent = -emp_lambda_opp * np.pi * (min_dist ** 2)
            p_value = 0.0 if exponent < -700 else np.exp(exponent)
        clusters_with_pval.append((cent_i, chg_i, p_value))

    p_values = np.array([c[2] for c in clusters_with_pval])
    N_tests = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    thresholds = (np.arange(1, N_tests + 1) / N_tests) * alpha

    valid_count = 0
    for i in range(N_tests - 1, -1, -1):
        if sorted_p_values[i] <= thresholds[i]:
            valid_count = i + 1
            break

    if valid_count == 0: return []
    valid_indices = sorted_indices[:valid_count]
    return [(clusters_with_pval[i][0], clusters_with_pval[i][1], clusters_with_pval[i][2]) for i in valid_indices]


def main():
    print("=" * 80)
    print(" SimExp 5: 终极量化评估 (经验零浴校准 & 物理熔断预警)")
    print("=" * 80)

    # 修改为 1x3 大宽图
    fig = plt.figure(figsize=(24, 6), facecolor='white')
    model = ECLC1(tau=2.0, cooling_iterations=15, dt=0.2, fdr_alpha=0.05)

    coords_null, simplices_null, normals_null = generate_flat_mesh(50)
    total_area = model._compute_surface_area(coords_null, simplices_null)

    # ================= Exp A =================
    print("\n[*] 执行 Exp A: 提取【冷却后】的残余奇点，校准经验本底密度...")
    distances_r = []
    total_opp_charges = 0
    mc_trials_A = 40

    for trial in tqdm(range(mc_trials_A), desc="Empirical Null Calibration"):
        np.random.seed(100 + trial)
        V_null = get_pure_noise_field(coords_null, normals_null, noise_std=1.0)
        clusters = extract_cooled_clusters(model, coords_null, V_null, normals_null, simplices_null)
        opp_count = sum(1 for c in clusters if c[1] < 0)
        total_opp_charges += opp_count

        for i, (cent_i, chg_i) in enumerate(clusters):
            min_dist = float('inf')
            for j, (cent_j, chg_j) in enumerate(clusters):
                if chg_i * chg_j < 0:
                    dist = np.linalg.norm(cent_i - cent_j)
                    if dist < min_dist: min_dist = dist
            if min_dist != float('inf'):
                distances_r.append(min_dist)

    r_squared = np.array(distances_r) ** 2
    emp_lambda_opp = max(total_opp_charges / mc_trials_A, 1) / total_area

    ax1 = fig.add_subplot(1, 3, 1)
    if len(r_squared) > 0:
        ax1.hist(r_squared, bins=15, density=True, color='skyblue', edgecolor='white', alpha=0.85,
                 label='Empirical Cooled Noise')
        x_vals = np.linspace(0, max(r_squared), 200)
        pdf_theoretical = emp_lambda_opp * np.pi * np.exp(-emp_lambda_opp * np.pi * x_vals)
        ax1.plot(x_vals, pdf_theoretical, color='red', linewidth=3, linestyle='--',
                 label=rf'Empirical PDF ($\lambda_{{emp}}$)')

    ax1.set_title("A. Spatial Poisson Null Distribution (After Cooling)", fontsize=20, fontweight='bold')
    ax1.set_xlabel("Squared Nearest-Opposite Distance ($r^2$)", fontsize=13)
    ax1.set_ylabel("Probability Density Function", fontsize=13)
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # ================= Exp B =================
    print("\n[*] 执行 Exp B: 基于动态经验密度的 FDR 防御与 BKT 密度熔断测试...")
    noise_levels = np.linspace(0.0, 3.5, 8)
    mc_trials_B = 25

    tpr_means, tpr_stds, fdr_means, fdr_stds = [], [], [], []
    bkt_density_means, bkt_density_stds = [], []  # 🌟 新增：监控原始 BKT 密度！

    num_simplices = len(simplices_null)

    for sigma in tqdm(noise_levels, desc="Noise Sweep"):
        tpr_list, fdr_list, bkt_list = [], [], []
        for trial in range(mc_trials_B):
            np.random.seed(int(sigma * 100) + trial)
            V_noisy, true_centers = get_dipole_field(coords_null, normals_null, noise_std=sigma)

            # 🌟 核心新增：提取未经任何冷却处理的纯原始伪影密度 (Raw BKT Density)
            _, raw_sings = model.intrinsic_dec_3d_c1(coords_null, V_noisy, normals_null, simplices_null)
            bkt_density = len(raw_sings) / num_simplices
            bkt_list.append(bkt_density * 100)  # 转为百分比

            clusters = extract_cooled_clusters(model, coords_null, V_noisy, normals_null, simplices_null)
            final_sings = empirical_hypothesis_test(clusters, emp_lambda_opp, alpha=0.05)

            matched_true = set()
            FP = 0
            for pt, chg, _ in final_sings:
                dists = [np.linalg.norm(pt[:2] - tc[:2]) for tc in true_centers]
                min_idx = np.argmin(dists)
                if dists[min_idx] < 0.8 and min_idx not in matched_true:
                    matched_true.add(min_idx)
                else:
                    FP += 1

            TP = len(matched_true)
            tpr_list.append(TP / 2.0)
            fdr_list.append(FP / (TP + FP) if (TP + FP) > 0 else 0.0)

        tpr_means.append(np.mean(tpr_list))
        tpr_stds.append(np.std(tpr_list))
        fdr_means.append(np.mean(fdr_list))
        fdr_stds.append(np.std(fdr_list))
        bkt_density_means.append(np.mean(bkt_list))
        bkt_density_stds.append(np.std(bkt_list))

    tpr_means, tpr_stds = np.array(tpr_means), np.array(tpr_stds)
    fdr_means, fdr_stds = np.array(fdr_means), np.array(fdr_stds)
    bkt_density_means = np.array(bkt_density_means)
    bkt_density_stds = np.array(bkt_density_stds)

    # --- Panel B: TPR vs FDR ---
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("B. Statistical Phase Transition Limits", fontsize=20, fontweight='bold')
    ax2.set_xlabel(r"Gaussian Noise Std ($\sigma$)", fontsize=13)

    ax2.plot(noise_levels, tpr_means, color='forestgreen', marker='o', linewidth=2.5, label='Statistical Power (TPR)')
    ax2.fill_between(noise_levels, np.clip(tpr_means - tpr_stds, 0, 1), np.clip(tpr_means + tpr_stds, 0, 1),
                     color='forestgreen', alpha=0.2)
    ax2.set_ylabel("True Positive Rate (Power)", color='forestgreen', fontsize=13, fontweight='bold')
    ax2.set_ylim(-0.05, 1.05)

    ax2_r = ax2.twinx()
    ax2_r.plot(noise_levels, fdr_means, color='crimson', marker='s', linewidth=2.5, label='Empirical FDR')
    ax2_r.fill_between(noise_levels, np.clip(fdr_means - fdr_stds, 0, 1), np.clip(fdr_means + fdr_stds, 0, 1),
                       color='crimson', alpha=0.2)
    ax2_r.axhline(0.05, color='black', linestyle='--', linewidth=2, label=r'Target FDR Bound ($\alpha = 0.05$)')
    ax2_r.set_ylabel("False Discovery Rate (FDR)", color='crimson', fontsize=13, fontweight='bold')
    ax2_r.set_ylim(-0.05, 1.05)

    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_r.get_legend_handles_labels()
    ax2_r.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center left', fontsize=12)

    # --- Panel C: Raw BKT Density ---
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title(r"C. Diagnostic: Raw BKT Density ($\rho_{\text{raw}}$)", fontsize=20, fontweight='bold')
    ax3.set_xlabel(r"Gaussian Noise Std ($\sigma$)", fontsize=13)

    ax3.plot(noise_levels, bkt_density_means, color='darkorange', marker='D', linewidth=2.5, label=r'Raw BKT Density')
    ax3.fill_between(noise_levels, np.clip(bkt_density_means - bkt_density_stds, 0, 100),
                     bkt_density_means + bkt_density_stds, color='darkorange', alpha=0.2)

    ax3.axhline(12.5, color='dimgray', linestyle='-.', linewidth=2, label='Thermodynamic Max Limit (12.5%)')
    ax3.axhline(3.0, color='purple', linestyle='--', linewidth=2, label='Avalanche Warning Threshold (3.0%)')

    ax3.set_ylabel("Density of Spurious Singularities (%)", color='darkorange', fontsize=13, fontweight='bold')
    ax3.set_ylim(-0.5, 14)
    ax3.legend(loc='lower right', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'simexp_stats.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[*] 完美的 1x3 经验统计校准与预警图已生成！请查看: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()