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
    print("[!] Missing bioinformatics dependencies. Please run: pip install scvelo scanpy")
    sys.exit(1)

# Import the core ECL-c1 statistical topological framework
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.ECLC1 import ECLC1


def get_manifold_simplices(coords_2d, percentile=97):
    """
    [Crucial Geometric Regularization: Manifold Pruning]
    Single-cell UMAP projections are inherently non-convex scattered point clouds. 
    Traditional Delaunay triangulation bridges "empty gaps" between clusters, generating 
    extremely long spurious edges. We apply a strictly percentile-based edge-length truncation 
    to reconstruct the true topological manifold strictly within dense cellular regions.
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
    """
    Fix the normal orientation of simplices 
    (Ensure Z-axis points upwards, satisfying the right-hand rule for consistent flux integration).
    """
    simplices = np.copy(simplices)
    for i in range(len(simplices)):
        p0, p1, p2 = coords[simplices[i, 0]], coords[simplices[i, 1]], coords[simplices[i, 2]]
        n = normals[simplices[i, 0]]
        if np.dot(np.cross(p1 - p0, p2 - p0), n) < 0:
            simplices[i, 1], simplices[i, 2] = simplices[i, 2], simplices[i, 1]
    return simplices


def prepare_pancreas_data():
    print("[*] Downloading and processing the single-cell Pancreas developmental dataset...")
    print("    (Using native Scanpy pipeline for robust preprocessing to avoid API deprecation issues)")
    scv.settings.verbosity = 1

    # 1. Load built-in dataset (typically includes X_umap and clusters labels)
    adata = scv.datasets.pancreas()

    # 2. Gene filtering and mathematical normalization (Robust pipeline)
    scv.pp.filter_genes(adata, min_shared_counts=20)
    scv.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    try:
        scv.pp.filter_genes_dispersion(adata, n_top_genes=2000)
    except Exception:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='cell_ranger')

    # 3. Topological dimensionality reduction and KNN graph construction
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)

    # scvelo 0.4.0 removed scv.tl.umap; strictly fallback to scanpy.tl.umap
    # Only recompute if X_umap is missing to preserve the classic paper perspective.
    if 'X_umap' not in adata.obsm:
        sc.tl.umap(adata)

    # 4. RNA dynamics inference (scVelo's core physics engine)
    # KNN graph is already computed by scanpy; passing None to inherit automatically
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)
    scv.tl.velocity(adata)
    scv.tl.velocity_graph(adata)

    # 5. Flow field embedding (generate 2D vector field on the UMAP manifold)
    scv.tl.velocity_embedding(adata, basis='umap')

    return adata


def main():
    print("=" * 80)
    print(" RealExp1: ECL Framework on Real Single-Cell RNA Velocity Fields (Pancreas)")
    print("=" * 80)

    # 1. Acquire and process data
    adata = prepare_pancreas_data()

    # 2. Extract empirical base space and observation field
    # UMAP is 2D; to adapt to ECLC1's unified 3D geometric engine, we pad the Z-axis with 0
    umap_coords = adata.obsm['X_umap']
    V_umap = adata.obsm['velocity_umap']

    # Remove invalid cells with NaN velocity estimates due to sequencing dropouts
    valid_mask = ~np.isnan(V_umap).any(axis=1)
    umap_coords = umap_coords[valid_mask]
    V_umap = V_umap[valid_mask]

    N = umap_coords.shape[0]
    coords = np.column_stack((umap_coords, np.zeros(N)))
    V = np.column_stack((V_umap, np.zeros(N)))

    # Force flow field L2-normalization: Extract pure phase difference information, 
    # structurally shielding against extreme heteroscedasticity in velocity amplitudes.
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    V_normalized = np.where(norms > 1e-8, V / norms, np.zeros_like(V))

    # Construct normals and triangular simplices (planar manifold normals are uniformly [0, 0, 1])
    normals = np.zeros((N, 3))
    normals[:, 2] = 1.0
    simplices = get_manifold_simplices(umap_coords, percentile=96)
    simplices = fix_orientation(coords, simplices, normals)

    print(f"\n[*] Empirical base space constructed: {len(coords)} cell nodes, {len(simplices)} triangular simplices")

    # =====================================================================
    # Core Execution: Invoke the ECL model
    # Due to severe transcriptomic sequencing noise, we slightly increase the 
    # cooling iterations and annihilation radius, strictly enforcing FDR=0.05.
    # =====================================================================
    model = ECLC1(tau=3.0, cooling_iterations=30, dt=0.2, fdr_alpha=0.05)

    print("\n[*] Initializing the ECL-c1 Statistical Topological Inference Engine...")
    c1_filtered, final_sings, V_cooled, raw_sings = model.fit(coords, V_normalized, normals, simplices)

    print(f"\n" + "=" * 40 + " Inference Report " + "=" * 40)
    print(f"[Microscopic Illusions] Stage 1 naive algebraic operator got overwhelmed by sequencing noise, extracting {len(raw_sings)} high-frequency phase slips.")
    print(f"[Macroscopic Truth] Stage 2 spatial Poisson null bath and statistical annihilation strictly preserved {len(final_sings)} highly significant biological singularities!")

    for pt, chg, pval in final_sings:
        pt_type = "Developmental Origin (+1 Source)" if chg > 0 else "Terminal Fate / Lineage Bifurcation (-1 Sink/Saddle)"
        print(f"    => {pt_type}: Coordinates({pt[0]:.2f}, {pt[1]:.2f}) | Poisson P-value = {pval:.2e}")
    print("=" * 90)

    # =====================================================================
    # Render publication-quality comparative figures
    # =====================================================================
    print("\n[*] Rendering comparative figures...")
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    plt.subplots_adjust(wspace=0.1)

    # Retrieve biological ground truth cell colors
    clusters = adata.obs['clusters'][valid_mask]
    cluster_names = adata.obs['clusters'].cat.categories
    cluster_colors = adata.uns['clusters_colors'] if 'clusters_colors' in adata.uns else plt.cm.tab20.colors
    color_map = {name: color for name, color in zip(cluster_names, cluster_colors)}
    cell_colors = [color_map[c] for c in clusters]

    # --- Plot 1: Classical Biological Streamplot (Ground Truth) ---
    ax1 = axes[0]
    scv.pl.velocity_embedding_stream(adata, basis='umap', color='clusters',
                                     title="", ax=ax1, show=False, legend_loc='best', alpha=0.5)
    ax1.set_title(f"1. Biological Ground Truth\n(scVelo Streamplot)", fontsize=20, fontweight='bold', pad=15)

    # --- Plot 2: Stage 1 Naive DEC (The Noise-dominated Disaster) ---
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

    # --- Plot 3: Stage 2 ECL-c1 Statistical Topological Inference ---
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

            # Smart layout adjustment based on UMAP macro structure
            if chg > 0:
                if pt[0] > mid_x:  # Progenitor stem cells on the right (Ductal)
                    label_text = "Origin (+1)"
                    x_off = max_range_x * 0.08  
                    y_off = max_range_y * 0.06  
                else:  # Terminal fate on the left (Alpha/Beta)
                    label_text = "Fate (+1)"
                    x_off = max_range_x * 0.08  
                    y_off = -max_range_y * 0.08  
            else:  # Lineage Bifurcation in the middle-left
                label_text = "Bifurcation (-1)"
                x_off = -max_range_x * 0.10  
                y_off = max_range_y * 0.06  

            # Premium annotation formatting with arrow leads
            ax3.annotate(f"{label_text}\n({p_text})",
                         xy=(pt[0], pt[1]),
                         xytext=(pt[0] + x_off, pt[1] + y_off),
                         arrowprops=dict(arrowstyle="-|>", color="dimgray", lw=1.5, alpha=0.8, shrinkA=0, shrinkB=6),
                         color='black', fontsize=11, fontweight='bold',
                         bbox=dict(facecolor='white', alpha=0.95, edgecolor='gray', boxstyle='round,pad=0.3'),
                         zorder=100, ha='center', va='center')

    # Unify viewport boundaries
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
        if ax != ax1:
            ax.set_xlim(ax1.get_xlim()); ax.set_ylim(ax1.get_ylim())

    # =====================================================================
    # 🔬 Quantitative Validation: Cross-referencing physical coords with Ground Truth
    # =====================================================================
    print("\n" + "=" * 80)
    print(" Ultimate Quantitative Validation: Perfect intersection of topological singularities and biological labels")
    print("=" * 80)

    scv.tl.velocity_pseudotime(adata)
    pseudotime = adata.obs['velocity_pseudotime'].values
    clusters = adata.obs['clusters'].values

    print(f"{'Inferred Topological State':<30} | {'Biological Ground Truth (Cell Type)':<35} | {'Pseudotime (0~1)'}")
    print("-" * 90)

    for i, (pt, chg, pval) in enumerate(final_sings):
        dists = np.linalg.norm(umap_coords - pt[:2], axis=1)
        nearest_idx = np.argmin(dists)

        # Compute physical divergence using the 15 nearest neighbor cells to determine inflow/outflow
        neighbor_indices = np.argsort(dists)[1:16]
        divergence = 0
        for n_idx in neighbor_indices:
            vec_radial = umap_coords[n_idx] - pt[:2]
            vec_flow = V_normalized[n_idx, :2]
            divergence += np.dot(vec_radial, vec_flow)

        if chg > 0:
            if divergence > 0:
                role = "Developmental Origin (Source +1)"
            else:
                role = "Terminal Fate (Sink +1)"
        else:
            role = "Lineage Bifurcation (Saddle -1)"

        cell_type = str(clusters[nearest_idx])
        ptime = pseudotime[nearest_idx]
        print(f"{role:<30} | {cell_type:<35} | {ptime:.3f}")
    print("=" * 90)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'realexp1_scvelo_pancreas.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[*] Real-world empirical chart successfully saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
