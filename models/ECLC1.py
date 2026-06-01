import numpy as np


class ECLC1:
    """
    Empirical Spectral Cohomology Learning for 1st Chern Class (ECL-c1)
    【完美终极版】普适热力学零浴生存模型 (Universal Null Bath Model)
    彻底修复了 DFS 漏算根节点的黑洞 Bug，重铸 FDR=0.05 统计黄金法则！
    """

    def __init__(self, tau=2.0, cooling_iterations=15, dt=0.2, fdr_alpha=0.05):
        self.tau = tau
        self.cooling_iterations = cooling_iterations
        self.dt = dt
        self.fdr_alpha = fdr_alpha  # 极其自信地回归 0.05

    @staticmethod
    def _wrap_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def vector_field_cooling_manifold(self, coords, V, normals, simplices):
        N = coords.shape[0]
        adj = [[] for _ in range(N)]
        for tri in simplices:
            adj[tri[0]].extend([tri[1], tri[2]])
            adj[tri[1]].extend([tri[0], tri[2]])
            adj[tri[2]].extend([tri[0], tri[1]])
        adj = [list(set(neighbors)) for neighbors in adj]

        V_cooled = np.copy(V)
        for _ in range(self.cooling_iterations):
            V_new = np.zeros_like(V_cooled)
            for i in range(N):
                neighbors = adj[i]
                if len(neighbors) > 0:
                    v_avg = np.mean(V_cooled[neighbors], axis=0)
                    v_step = V_cooled[i] + self.dt * (v_avg - V_cooled[i])
                    n_vec = normals[i]
                    v_tangent = v_step - np.dot(v_step, n_vec) * n_vec
                    norm_val = np.linalg.norm(v_tangent)
                    V_new[i] = v_tangent / norm_val if norm_val > 1e-12 else V_cooled[i]
                else:
                    V_new[i] = V_cooled[i]
            V_cooled = V_new
        return V_cooled

    def intrinsic_dec_3d_c1(self, coords, V, normals, simplices):
        N = coords.shape[0]
        X, Y, phi = np.zeros((N, 3)), np.zeros((N, 3)), np.zeros(N)

        for i in range(N):
            n = normals[i]
            x = np.array([1.0, 0.0, 0.0])
            x = x - np.dot(x, n) * n
            if np.linalg.norm(x) < 1e-3:
                x = np.array([0.0, 1.0, 0.0])
                x = x - np.dot(x, n) * n
            x /= np.linalg.norm(x)
            y = np.cross(n, x)
            X[i], Y[i] = x, y
            phi[i] = np.arctan2(np.dot(V[i], y), np.dot(V[i], x))

        c1_charge = 0
        singularity_data = []

        for tri in simplices:
            i, j, k = tri
            v1, v2 = coords[j] - coords[i], coords[k] - coords[i]
            if np.dot(np.cross(v1, v2), normals[i]) < 0:
                j, k = k, j

            def get_delta_phi(u, v_idx):
                e = coords[v_idx] - coords[u]
                e_u = e - np.dot(e, normals[u]) * normals[u]
                alpha_u = np.arctan2(np.dot(e_u, Y[u]), np.dot(e_u, X[u]))
                e_v = e - np.dot(e, normals[v_idx]) * normals[v_idx]
                alpha_v = np.arctan2(np.dot(e_v, Y[v_idx]), np.dot(e_v, X[v_idx]))
                return self._wrap_angle(phi[v_idx] - phi[u] - (alpha_v - alpha_u))

            flux = get_delta_phi(i, j) + get_delta_phi(j, k) + get_delta_phi(k, i)
            local_charge = int(np.round(flux / (2 * np.pi)))

            if local_charge != 0:
                c1_charge += local_charge
                centroid = (coords[i] + coords[j] + coords[k]) / 3.0
                L_bar = (np.linalg.norm(coords[i] - coords[j]) +
                         np.linalg.norm(coords[j] - coords[k]) +
                         np.linalg.norm(coords[k] - coords[i])) / 3.0
                singularity_data.append((centroid, local_charge, L_bar))

        return c1_charge, singularity_data

    def _compute_surface_area(self, coords, simplices):
        v0, v1, v2 = coords[simplices[:, 0]], coords[simplices[:, 1]], coords[simplices[:, 2]]
        return 0.5 * np.sum(np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1))

    def topological_hypothesis_test(self, cooled_sings, simplices, total_area):
        if not cooled_sings: return 0, []

        N_raw = len(cooled_sings)
        adj_matrix = np.zeros((N_raw, N_raw), dtype=bool)

        for i in range(N_raw):
            cent_i, _, L_bar_i = cooled_sings[i]
            for j in range(i + 1, N_raw):
                cent_j, _, L_bar_j = cooled_sings[j]
                if np.linalg.norm(cent_i - cent_j) <= self.tau * max(L_bar_i, L_bar_j):
                    adj_matrix[i, j] = adj_matrix[j, i] = True

        visited = np.zeros(N_raw, dtype=bool)
        clusters = []

        for i in range(N_raw):
            if not visited[i]:
                comp = []
                stack = [i]
                visited[i] = True
                while stack:
                    node = stack.pop()
                    comp.append(node)  # <=== ✅【致命黑洞 Bug 终极修复】：堂堂正正收编根节点！
                    for n in np.where(adj_matrix[node])[0]:
                        if not visited[n]:
                            visited[n] = True
                            stack.append(n)

                cluster_charge = sum([cooled_sings[idx][1] for idx in comp])
                if cluster_charge != 0:
                    cluster_cents = [cooled_sings[idx][0] for idx in comp]
                    clusters.append((np.mean(cluster_cents, axis=0), cluster_charge))

        if not clusters: return 0, []

        # ================= 阶段 2：普适热力学零浴生存模型 =================
        N_triangles = len(simplices)
        lambda_null = (N_triangles / 8.0) / total_area if total_area > 0 else 1.0

        clusters_with_pval = []
        for i, (cent_i, chg_i) in enumerate(clusters):
            min_dist = float('inf')
            for j, (cent_j, chg_j) in enumerate(clusters):
                if chg_i * chg_j < 0:
                    dist = np.linalg.norm(cent_i - cent_j)
                    if dist < min_dist:
                        min_dist = dist

            if min_dist == float('inf'):
                p_value = 0.0  # 孤立本征奇点，大自然赋予的绝对豁免权！
            else:
                exponent = -lambda_null * np.pi * (min_dist ** 2)
                p_value = 0.0 if exponent < -700 else np.exp(exponent)

            clusters_with_pval.append((cent_i, chg_i, p_value))

        # ================= 阶段 3：严苛的 FDR 截断 =================
        p_values = np.array([c[2] for c in clusters_with_pval])
        N_tests = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]

        thresholds = (np.arange(1, N_tests + 1) / N_tests) * self.fdr_alpha
        valid_count = 0
        for i in range(N_tests - 1, -1, -1):
            if sorted_p_values[i] <= thresholds[i]:
                valid_count = i + 1
                break

        if valid_count == 0: return 0, []

        valid_indices = sorted_indices[:valid_count]
        final_singularities = [(clusters_with_pval[i][0], clusters_with_pval[i][1], clusters_with_pval[i][2]) for i in
                               valid_indices]

        c1_filtered = sum(charge for _, charge, _ in final_singularities)
        return c1_filtered, final_singularities

    def fit(self, coords, V, normals, simplices):
        total_area = self._compute_surface_area(coords, simplices)
        _, raw_sings = self.intrinsic_dec_3d_c1(coords, V, normals, simplices)
        V_cooled = self.vector_field_cooling_manifold(coords, V, normals, simplices)
        _, cooled_sings = self.intrinsic_dec_3d_c1(coords, V_cooled, normals, simplices)

        c1_filtered, final_sings = self.topological_hypothesis_test(cooled_sings, simplices, total_area)
        return c1_filtered, final_sings, V_cooled, raw_sings