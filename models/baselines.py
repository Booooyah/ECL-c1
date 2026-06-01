import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
import warnings

warnings.filterwarnings("ignore")


class GaussianKernelJacobian:
    """
    【Baseline 1: 空间统计学流派 (Spatial Statistics)】
    机制: 高斯平滑 -> 寻找速度驻点 -> 最小二乘法估计雅可比矩阵(Jacobian)。
    死穴: 强行求偏导会指数级放大高频噪声，引发满屏的假阳性爆炸 (Derivative Amplification)。
    """

    def __init__(self, bandwidth=0.35, speed_threshold=0.25):
        self.bandwidth = bandwidth
        self.speed_threshold = speed_threshold

    def fit(self, coords, V, normals, simplices):
        N = coords.shape[0]
        tree = cKDTree(coords)
        V_smooth = np.zeros_like(V)

        # 高斯核平滑
        for i in range(N):
            idx = tree.query_ball_point(coords[i], r=self.bandwidth * 3)
            if len(idx) > 0:
                dists = np.linalg.norm(coords[idx] - coords[i], axis=1)
                weights = np.exp(-(dists ** 2) / (2 * self.bandwidth ** 2))
                weights /= (np.sum(weights) + 1e-12)
                V_smooth[i] = np.average(V[idx], axis=0, weights=weights)
            else:
                V_smooth[i] = V[i]

        V_smooth -= np.sum(V_smooth * normals, axis=1)[:, np.newaxis] * normals
        speed = np.linalg.norm(V_smooth, axis=1)

        adj = [[] for _ in range(N)]
        for tri in simplices:
            adj[tri[0]].extend([tri[1], tri[2]])
            adj[tri[1]].extend([tri[0], tri[2]])
            adj[tri[2]].extend([tri[0], tri[1]])
        adj = [list(set(neighbors)) for neighbors in adj]

        candidates = [i for i in range(N) if
                      speed[i] < self.speed_threshold and all(speed[i] <= speed[n] for n in adj[i])]

        final_sings = []
        for c_idx in candidates:
            idx = tree.query_ball_point(coords[c_idx], r=self.bandwidth * 1.5)
            if len(idx) < 5: continue

            dX = coords[idx] - coords[c_idx]
            dV = V_smooth[idx] - V_smooth[c_idx]

            n = normals[c_idx]
            x_ax = np.array([1.0, 0.0, 0.0])
            x_ax -= np.dot(x_ax, n) * n
            if np.linalg.norm(x_ax) < 1e-3:
                x_ax = np.array([0.0, 1.0, 0.0])
                x_ax -= np.dot(x_ax, n) * n
            x_ax /= np.linalg.norm(x_ax)
            y_ax = np.cross(n, x_ax)

            dX_2d = np.column_stack((np.dot(dX, x_ax), np.dot(dX, y_ax)))
            dV_2d = np.column_stack((np.dot(dV, x_ax), np.dot(dV, y_ax)))

            try:
                J, _, _, _ = np.linalg.lstsq(dX_2d, dV_2d, rcond=None)
                evals = np.linalg.eigvals(J.T)
                r1, r2 = np.real(evals)
                charge = 1 if r1 * r2 > 0 else (-1 if r1 * r2 < 0 else 0)
                if charge != 0: final_sings.append((coords[c_idx], charge, 1.0))
            except:
                continue

        filtered_sings = []
        for s in final_sings:
            if not any(np.linalg.norm(s[0] - fs[0]) < self.bandwidth for fs in filtered_sings):
                filtered_sings.append(s)
        return sum([c for _, c, _ in filtered_sings]), filtered_sings, V_smooth, []


class MarkovRandomWalk:
    """
    【Baseline 2: 图机器学习流派 (Graph ML)】
    机制: 构建转移概率矩阵 -> 跑马尔可夫稳态游走找聚集点。
    死穴: 彻头彻尾的“鞍点瞎子”。绝对无法在数学上发现分岔路口 (-1 电荷)。
    """

    def __init__(self, beta=10.0, top_percentile=98.5):
        self.beta = beta
        self.top_percentile = top_percentile

    def _build_P(self, coords, V, simplices, reverse=False):
        N = coords.shape[0]
        row, col, data = [], [], []
        edges = set()
        for tri in simplices:
            edges.add(tuple(sorted([tri[0], tri[1]])));
            edges.add(tuple(sorted([tri[1], tri[2]])));
            edges.add(tuple(sorted([tri[2], tri[0]])))

        V_work = -V if reverse else V
        for i, j in edges:
            d_ij = coords[j] - coords[i]
            dist = np.linalg.norm(d_ij)
            if dist < 1e-8: continue

            v_norm, v_norm_j = np.linalg.norm(V_work[i]), np.linalg.norm(V_work[j])
            cos_ij = np.dot(V_work[i], d_ij) / (v_norm * dist) if v_norm > 1e-8 else 0
            cos_ji = np.dot(V_work[j], -d_ij) / (v_norm_j * dist) if v_norm_j > 1e-8 else 0

            row.extend([i, j]);
            col.extend([j, i]);
            data.extend([np.exp(self.beta * cos_ij), np.exp(self.beta * cos_ji)])

        row.extend(range(N));
        col.extend(range(N));
        data.extend(np.ones(N))
        W = csr_matrix((data, (row, col)), shape=(N, N))
        row_sums = np.array(W.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        return csr_matrix((1.0 / row_sums, (range(N), range(N))), shape=(N, N)).dot(W)

    def fit(self, coords, V, normals, simplices):
        N = coords.shape[0]
        adj = [[] for _ in range(N)]
        for tri in simplices:
            adj[tri[0]].extend([tri[1], tri[2]]);
            adj[tri[1]].extend([tri[0], tri[2]]);
            adj[tri[2]].extend([tri[0], tri[1]])
        adj = [list(set(neighbors)) for neighbors in adj]

        def pagerank(P):
            pi = np.ones(N) / N
            for _ in range(100):
                pi_new = P.T.dot(pi)
                if np.linalg.norm(pi_new - pi) < 1e-6: break
                pi = pi_new
            return pi

        pi_fwd = pagerank(self._build_P(coords, V, simplices, False))
        pi_bwd = pagerank(self._build_P(coords, V, simplices, True))

        tf, tb = np.percentile(pi_fwd, self.top_percentile), np.percentile(pi_bwd, self.top_percentile)

        final_sings = []
        for i in range(N):
            if pi_fwd[i] > tf and all(pi_fwd[i] >= pi_fwd[n] for n in adj[i]):
                final_sings.append((coords[i], 1, 1.0))
            elif pi_bwd[i] > tb and all(pi_bwd[i] >= pi_bwd[n] for n in adj[i]):
                final_sings.append((coords[i], 1, 1.0))

        filtered = []
        for s in final_sings:
            if not any(np.linalg.norm(s[0] - f[0]) < 0.3 for f in filtered): filtered.append(s)
        return sum([c for _, c, _ in filtered]), filtered, V, []