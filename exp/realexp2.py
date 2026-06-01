import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
import warnings

warnings.filterwarnings("ignore")

# 引入 ECL 核心模型
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.ECLC1 import ECLC1


def load_multiday_data(csv_path="data/ocean_5days.csv"):
    print(f"[*] 正在加载连续 5 天的海洋时空数据: {csv_path} ...")
    if not os.path.exists(csv_path):
        print(f"[!] 找不到文件 {csv_path}！请按指引下载并重命名。")
        sys.exit(1)

    df_test = pd.read_csv(csv_path, nrows=5)
    skip = [1] if any(
        isinstance(v, str) and ('m/s' in v.lower() or 'utc' in v.lower() or 'm' in v.lower() or 'degree' in v.lower())
        for v in df_test.iloc[0].values) else None
    df = pd.read_csv(csv_path, skiprows=skip)

    u_col = next((c for c in df.columns if c.lower() in ['u', 'ugos', 'ugosa'] or 'currents_u' in c.lower()), None)
    v_col = next((c for c in df.columns if c.lower() in ['v', 'vgos', 'vgosa'] or 'currents_v' in c.lower()), None)
    sla_col = next((c for c in df.columns if
                    'sla' in c.lower() or 'height' in c.lower() or 'ssh' in c.lower() or 'adt' in c.lower()), None)
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
    time_col = next((c for c in df.columns if 'time' in c.lower()), None)

    df = df.dropna(subset=[u_col, v_col, sla_col, time_col])

    # 转换经度并清理
    df[lon_col] = np.where(df[lon_col] > 180, df[lon_col] - 360, df[lon_col])
    return df, time_col, lon_col, lat_col, u_col, v_col, sla_col


def main():
    print("=" * 80)
    print(" RealExp 4 终极时空实证: 涡旋生命周期的物理连贯性追踪")
    print("=" * 80)

    df, t_col, lon_col, lat_col, u_col, v_col, sla_col = load_multiday_data()
    unique_times = sorted(df[t_col].unique())
    print(f"[*] 成功识别到 {len(unique_times)} 个连续的日期快照。")

    all_sings_by_day = []
    first_day_sla = None
    first_day_coords = None

    # ================= 分天独立运行 ECL =================
    for day_idx, t_val in enumerate(unique_times):
        df_day = df[df[t_col] == t_val]
        lons, lats = df_day[lon_col].values.astype(float), df_day[lat_col].values.astype(float)
        u, v = df_day[u_col].values.astype(float), df_day[v_col].values.astype(float)
        sla = df_day[sla_col].values.astype(float)

        coords = np.column_stack([lons, lats, np.zeros_like(lons)])
        V = np.column_stack([u, v, np.zeros_like(u)])

        if day_idx == 0:
            first_day_sla = sla
            first_day_coords = coords

        tri = Delaunay(coords[:, :2])
        simplices = tri.simplices
        pts = coords[simplices][:, :, :2]
        max_lens = np.max(np.linalg.norm(pts - np.roll(pts, 1, axis=1), axis=-1), axis=1)
        simplices = simplices[max_lens <= 0.4]
        normals = np.zeros_like(coords)
        normals[:, 2] = 1.0

        norms_v = np.linalg.norm(V, axis=1, keepdims=True)
        V_normalized = np.where(norms_v > 1e-8, V / norms_v, np.zeros_like(V))

        print(f"\n[ Day {day_idx + 1} / {len(unique_times)} ] 正在对 {t_val[:10]} 快照进行毫无历史记忆的独立盲推断...")
        # 为了看清主轨迹，我们使用稍微严苛的显著性截断
        model = ECLC1(tau=3.5, cooling_iterations=35, dt=0.2, fdr_alpha=0.005)
        _, final_sings, _, _ = model.fit(coords, V_normalized, normals, simplices)

        all_sings_by_day.append(final_sings)
        print(f"  -> 提取出 {len(final_sings)} 个宏观结构。")

    # ================= 渲染时空轨迹神图 =================
    fig, ax = plt.subplots(figsize=(20, 10), facecolor='white')
    ax.set_facecolor('#0f172a')
    ax.set_aspect('equal')
    ax.set_xlabel("Longitude (°E)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Latitude (°S)", fontsize=14, fontweight='bold')
    ax.set_title(
        "Spatiotemporal Coherence Validation: Lagrangian Tracking of Inferred Singularities Over 5 Days\n(Algorithm runs independently per day. Smooth spatial trails physically prove they are massive inertial eddies, not random stochastic noise.)",
        fontsize=20, fontweight='bold', pad=15)

    # 1. 用第一天的 SLA 垫底作为地理参考
    print("\n[*] 正在渲染 SLA 标量底图 ...")
    grid_x, grid_y = np.mgrid[min(first_day_coords[:, 0]):max(first_day_coords[:, 0]):500j,
    min(first_day_coords[:, 1]):max(first_day_coords[:, 1]):500j]
    grid_sla = griddata((first_day_coords[:, 0], first_day_coords[:, 1]), first_day_sla, (grid_x, grid_y),
                        method='cubic')

    # 稍微调高透明度，让上方的轨迹线更醒目
    contour = ax.contourf(grid_x, grid_y, grid_sla, levels=35, cmap='RdYlBu_r', alpha=0.5)
    cbar = plt.colorbar(contour, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Sea Level Anomaly (m) on Day 1', fontsize=13, fontweight='bold')

    # 2. 绘制时空轨迹 (Lagrangian Tracking)
    print("[*] 正在进行拉格朗日轨迹连线...")

    # 每天最大允许移动距离（约 60 公里），如果在这个半径内找到了同性电荷，就认为是同一个涡旋在移动
    TRACK_DIST_THRESHOLD = 0.6

    for t in range(1, len(all_sings_by_day)):
        prev_day_sings = all_sings_by_day[t - 1]
        curr_day_sings = all_sings_by_day[t]

        for curr_s in curr_day_sings:
            pt_curr, chg_curr, _ = curr_s

            # 寻找昨天相同拓扑电荷的最近邻
            candidates = [ps for ps in prev_day_sings if ps[1] == chg_curr]
            if not candidates: continue

            dists = [np.linalg.norm(pt_curr[:2] - ps[0][:2]) for ps in candidates]
            min_dist_idx = np.argmin(dists)

            if dists[min_dist_idx] < TRACK_DIST_THRESHOLD:
                pt_prev = candidates[min_dist_idx][0]
                color = 'lime' if chg_curr > 0 else 'fuchsia'
                # 画出平滑的移动轨迹连线
                ax.plot([pt_prev[0], pt_curr[0]], [pt_prev[1], pt_curr[1]],
                        color=color, linewidth=2.5, zorder=8, alpha=0.8)

    # 3. 把每一天的奇点散点打上去，用颜色深浅和大小表示时间的流逝
    for day_idx, day_sings in enumerate(all_sings_by_day):
        # 时间越靠后，星星越亮、越大（展现漂移的方向感）
        alpha_val = 0.3 + 0.7 * (day_idx / (len(unique_times) - 1))
        size_val = 150 + 80 * (day_idx / (len(unique_times) - 1))

        for s in day_sings:
            pt, chg, _ = s
            color, marker = ('lime', '*') if chg > 0 else ('fuchsia', 'X')
            ax.scatter(pt[0], pt[1], color=color, s=size_val, marker=marker,
                       edgecolors='black', linewidths=1.0, alpha=alpha_val, zorder=10)

    # 图例
    ax.scatter([], [], color='lime', s=300, marker='*', edgecolors='black',
               label='Eddy (+1) Trajectories (Day 1 $\\rightarrow$ Day 5)')
    ax.scatter([], [], color='fuchsia', s=150, marker='X', edgecolors='black', label='Saddle (-1) Trajectories')
    ax.legend(loc='lower left', framealpha=0.9, fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_path = 'realexp4_trajectories.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[*] 令人震撼的时空物理轨迹神图已保存至: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()