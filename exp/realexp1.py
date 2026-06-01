import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import warnings

warnings.filterwarnings("ignore")

try:
    import scvelo as scv
    import scanpy as sc
except ImportError:
    print("[!] 缺少生物信息学依赖库，请运行: pip install scvelo scanpy")
    sys.exit(1)

# 引入我们的降维打击核武器 ECL-c1
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.ECLC1 import ECLC1


def get_manifold_simplices(coords_2d, percentile=97):
    """
    【关键几何正则化：流形剪枝】
    单细胞 UMAP 投影本质上是非凸的散乱点云。传统的 Delaunay 剖分会跨越 UMAP 上的“聚类空白带”
    生成极长的伪边缘。我们通过边长分位数截断，还原出只在细胞稠密区连接的真实拓扑流形。
    """
    tri = Delaunay(coords_2d)
    simplices = tri.simplices

    max_lens = np.zeros(len(simplices))
    for i, t in enumerate(simplices):
        pts = coords_2d[t]
        l0 = np.linalg.norm(pts[0] - pts[1])
        l1 = np.linalg.norm(pts[1] - pts[2])
        l2 = np.linalg.norm(pts[2] - pts[0])
        max_lens[i] = max(l0, l1, l2)

    threshold = np.percentile(max_lens, percentile)
    return simplices[max_lens <= threshold]


def fix_orientation(coords, simplices, normals):
    """固定单纯形的法线朝向 (确保 Z 轴向上，满足右手螺旋定则)"""
    simplices = np.copy(simplices)
    for i in range(len(simplices)):
        p0, p1, p2 = coords[simplices[i, 0]], coords[simplices[i, 1]], coords[simplices[i, 2]]
        n = normals[simplices[i, 0]]
        if np.dot(np.cross(p1 - p0, p2 - p0), n) < 0:
            simplices[i, 1], simplices[i, 2] = simplices[i, 2], simplices[i, 1]
    return simplices


def prepare_pancreas_data():
    print("[*] 正在下载并处理单细胞胰腺发育数据集 (Pancreas Dataset)...")
    print("    (采用 Scanpy 原生管线硬核预处理，规避一切 API 弃用报错)")
    scv.settings.verbosity = 1

    # 1. 加载内置数据集 (通常自带 X_umap 和 clusters 标签)
    adata = scv.datasets.pancreas()

    # 2. 基因过滤与数学归一化 (防弹级混用)
    scv.pp.filter_genes(adata, min_shared_counts=20)
    scv.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    try:
        scv.pp.filter_genes_dispersion(adata, n_top_genes=2000)
    except Exception:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='cell_ranger')

    # 3. 拓扑降维与近邻图构建 (强制移交给底层 scanpy 处理)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)

    # scvelo 0.4.0 移除了 scv.tl.umap，必须使用 scanpy.tl.umap
    # 且只有在数据集没有自带 umap 时才重新计算，保留经典论文视角
    if 'X_umap' not in adata.obsm:
        sc.tl.umap(adata)

    # 4. RNA 动力学推断 (scVelo 本职物理工作)
    # 此时 neighbors 已经由 scanpy 算好，传 None 自动继承
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)
    scv.tl.velocity(adata)
    scv.tl.velocity_graph(adata)

    # 5. 流场嵌入 (在 UMAP 上生成 2D 向量场)
    scv.tl.velocity_embedding(adata, basis='umap')

    return adata


def main():
    print("=" * 80)
    print(" RealExp1: ECL 框架在真实单细胞 RNA 速率场中的降维打击 (Pancreas)")
    print("=" * 80)

    # 1. 获取并处理数据
    adata = prepare_pancreas_data()

    # 2. 提取经验底空间与观测场
    # UMAP 是 2D 的，为了适配 ECLC1 统一的三维几何引擎，我们给 Z 轴补 0
    umap_coords = adata.obsm['X_umap']
    V_umap = adata.obsm['velocity_umap']

    # 剔除由于技术误差导致速度估计为 NaN 的无效细胞
    valid_mask = ~np.isnan(V_umap).any(axis=1)
    umap_coords = umap_coords[valid_mask]
    V_umap = V_umap[valid_mask]

    N = umap_coords.shape[0]
    coords = np.column_stack((umap_coords, np.zeros(N)))
    V = np.column_stack((V_umap, np.zeros(N)))

    # 强制归一化流场：提取纯粹的相位差信息，屏蔽速率幅度的极度异方差干扰
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    V_normalized = np.where(norms > 1e-8, V / norms, np.zeros_like(V))

    # 构造法线与三角面 (平面流形的法线统一为 [0, 0, 1])
    normals = np.zeros((N, 3))
    normals[:, 2] = 1.0
    simplices = get_manifold_simplices(umap_coords, percentile=96)
    simplices = fix_orientation(coords, simplices, normals)

    print(f"\n[*] 经验底空间构造完成: {len(coords)} 个细胞节点, {len(simplices)} 个三角单纯形")

    # =====================================================================
    # 核心高光时刻：调用 ECL 模型！
    # 因为单细胞基因测序数据极其嘈杂，我们适度增加冷却迭代与相消阈值，并坚持 FDR=0.05！
    # =====================================================================
    model = ECLC1(tau=3.0, cooling_iterations=30, dt=0.2, fdr_alpha=0.05)

    print("\n[*] 正在启动 ECL-c1 统计拓扑推断引擎...")
    c1_filtered, final_sings, V_cooled, raw_sings = model.fit(coords, V_normalized, normals, simplices)

    print(f"\n" + "=" * 40 + " 推断战报 " + "=" * 40)
    print(f"👻 [微观幻觉] Stage 1 纯代数算子在测序噪声中迷失，识别出了 {len(raw_sings)} 个高频量子涨落。")
    print(f"🌟 [宏观本源] Stage 2 统计相消与泊松零浴洗礼后，仅保留了 {len(final_sings)} 个具备绝对显著性的生命奇点！")

    for pt, chg, pval in final_sings:
        pt_type = "干细胞起源 (+1 Source)" if chg > 0 else "终末分化状态 (-1 Sink/Saddle)"
        print(f"    => {pt_type}: 坐标({pt[0]:.2f}, {pt[1]:.2f}) | Poisson P-value = {pval:.2e}")
    print("=" * 90)

    # =====================================================================
    # 绘制顶刊级对比图表
    # =====================================================================
    print("\n[*] 正在渲染对比图表...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    plt.subplots_adjust(wspace=0.1)

    # 获取细胞颜色 (生物学 Ground Truth)
    clusters = adata.obs['clusters'][valid_mask]
    cluster_names = adata.obs['clusters'].cat.categories
    cluster_colors = adata.uns['clusters_colors'] if 'clusters_colors' in adata.uns else plt.cm.tab20.colors
    color_map = {name: color for name, color in zip(cluster_names, cluster_colors)}
    cell_colors = [color_map[c] for c in clusters]

    # --- 图 1: 传统生物学流线图 (Ground Truth) ---
    ax1 = axes[0]
    scv.pl.velocity_embedding_stream(adata, basis='umap', color='clusters',
                                     title="", ax=ax1, show=False, legend_loc='best', alpha=0.5)
    ax1.set_title(f"1. Biological Ground Truth\n(scVelo Streamplot)", fontsize=20, fontweight='bold', pad=15)

    # --- 图 2: Stage 1 Naive DEC 的灾难现场 ---
    ax2 = axes[1]
    ax2.set_title(f"2. Stage 1: Naive DEC (Zero Statistical Tolerance)\nFalse Illusions N={len(raw_sings)}",
                  fontsize=20, fontweight='bold', pad=15)
    ax2.scatter(umap_coords[:, 0], umap_coords[:, 1], c='whitesmoke', s=10, edgecolors='lightgray', alpha=0.5)
    ax2.triplot(umap_coords[:, 0], umap_coords[:, 1], simplices, color='whitesmoke', alpha=0.3, linewidth=0.5)

    if raw_sings:
        r_p, r_c = np.array([s[0] for s in raw_sings]), np.array([s[1] for s in raw_sings])
        ax2.scatter(r_p[r_c > 0, 0], r_p[r_c > 0, 1], c='coral', marker='x', s=15, alpha=0.8, label='Micro Source (+1)')
        ax2.scatter(r_p[r_c < 0, 0], r_p[r_c < 0, 1], c='cornflowerblue', marker='x', s=15, alpha=0.8,
                    label='Micro Sink (-1)')
    ax2.legend(loc='lower left', frameon=True)

    # --- 图 3: ECL-c1 统计拓扑绝杀 ---
    ax3 = axes[2]
    net_c1 = sum([chg for _, chg, _ in final_sings])
    ax3.set_title(
        f"3. Stage 2: ECL Inference (Topology + Statistics)\nSurvived Macroscopic States N={len(final_sings)}, Net $c_1$={net_c1}",
        fontsize=20, fontweight='bold', pad=15)

    ax3.scatter(umap_coords[:, 0], umap_coords[:, 1], c=cell_colors, s=15, alpha=0.4, edgecolors='none')

    sub_idx = np.random.choice(umap_coords.shape[0], size=min(1200, umap_coords.shape[0]), replace=False)
    ax3.quiver(umap_coords[sub_idx, 0], umap_coords[sub_idx, 1],
               V_cooled[sub_idx, 0], V_cooled[sub_idx, 1],
               scale=45, color='darkslategray', alpha=0.5, width=0.003)

    if final_sings:
        f_p, f_c = np.array([s[0] for s in final_sings]), np.array([s[1] for s in final_sings])
        ax3.scatter(f_p[f_c > 0, 0], f_p[f_c > 0, 1], color='red', s=550, marker='*', edgecolors='black',
                    linewidths=1.5, zorder=10)
        ax3.scatter(f_p[f_c < 0, 0], f_p[f_c < 0, 1], color='blue', s=450, marker='X', edgecolors='white',
                    linewidths=1.5, zorder=10)

        max_range_x = np.ptp(umap_coords[:, 0])
        max_range_y = np.ptp(umap_coords[:, 1])
        mid_x = np.mean(umap_coords[:, 0])

        for pt, chg, pval in final_sings:
            p_text = "P~0" if pval < 1e-6 else f"P={pval:.1e}"

            # 智能排版与生物学纠错：利用 UMAP 大局观（右侧起点，左侧终点），一上一下，一左一右推开
            if chg > 0:
                if pt[0] > mid_x:  # 位于流形右侧的干细胞集群 (Ductal)
                    label_text = "Origin (+1)"
                    x_off = max_range_x * 0.08  # 标签往右推
                    y_off = max_range_y * 0.06  # 标签往上拉
                else:  # 位于流形左侧的终末归宿 (Alpha/Beta)
                    label_text = "Fate (+1)"
                    x_off = max_range_x * 0.08  # 标签往右推
                    y_off = -max_range_y * 0.08  # 标签往下沉（彻底避开左侧的鞍点！）
            else:  # 位于中左侧的分岔口 Bifurcation (-1)
                label_text = "Bifurcation (-1)"
                x_off = -max_range_x * 0.10  # 标签向左推
                y_off = max_range_y * 0.06  # 标签向上拉

            # 采用顶级排版的 annotate（带箭头的引线），文字框不再紧贴奇点
            ax3.annotate(f"{label_text}\n({p_text})",
                         xy=(pt[0], pt[1]),
                         xytext=(pt[0] + x_off, pt[1] + y_off),
                         arrowprops=dict(arrowstyle="-|>", color="dimgray", lw=1.5, alpha=0.8, shrinkA=0, shrinkB=6),
                         color='black', fontsize=11, fontweight='bold',
                         bbox=dict(facecolor='white', alpha=0.95, edgecolor='gray', boxstyle='round,pad=0.3'),
                         zorder=100, ha='center', va='center')

    # 统一视野边界
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([]);
        ax.set_yticks([])
        ax.spines['top'].set_visible(False);
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False);
        ax.spines['left'].set_visible(False)
        if ax != ax1:
            ax.set_xlim(ax1.get_xlim());
            ax.set_ylim(ax1.get_ylim())

    # =====================================================================
    # 🔬 定量佐证：用物理坐标反查生物学金标准
    # =====================================================================
    print("\n" + "=" * 80)
    print(" 终极定量佐证：拓扑奇点与真实生物学标签的完美交汇")
    print("=" * 80)

    scv.tl.velocity_pseudotime(adata)
    pseudotime = adata.obs['velocity_pseudotime'].values
    clusters = adata.obs['clusters'].values

    print(f"{'拓扑推断物理属性':<22} | {'靶中真实细胞类别 (GT)':<22} | {'发育伪时间 (0~1)'}")
    print("-" * 75)

    for i, (pt, chg, pval) in enumerate(final_sings):
        dists = np.linalg.norm(umap_coords - pt[:2], axis=1)
        nearest_idx = np.argmin(dists)

        # 获取奇点周围 15 个近邻细胞的向量，计算物理散度 (判断流入还是流出)
        neighbor_indices = np.argsort(dists)[1:16]
        divergence = 0
        for n_idx in neighbor_indices:
            vec_radial = umap_coords[n_idx] - pt[:2]
            vec_flow = V_normalized[n_idx, :2]
            divergence += np.dot(vec_radial, vec_flow)

        if chg > 0:
            if divergence > 0:
                role = "🎯 生命起源 (Source +1)"
            else:
                role = "🛑 终末归宿 (Sink +1)"
        else:
            role = "✂️ 命运分岔 (Saddle -1)"

        cell_type = str(clusters[nearest_idx])
        ptime = pseudotime[nearest_idx]
        print(f"{role:<20} | {cell_type:<25} | {ptime:.3f}")
    print("=" * 75)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'realexp1_scvelo_pancreas.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[*] 实战图表已成功保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()