import numpy as np
import trimesh
from matplotlib import cm, colors
from scipy.spatial import cKDTree
from tools.utils import (
    extract_np,
    gen_coefficients_np,
    load_input,
)
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, conf, dataset, dataname, stage):
        super().__init__()
        self.stage = stage
        self.conf = conf
        self.data_dir = conf.get_string("general.data_dir") + dataset + "/"

        self.train_num_points = conf.get_int("train.train_num_points")
        self.datalength = conf.get_int("train.datalength")
        self.metric = conf.get_string("train.metric")
        self.time_sum = conf.get_int("train.time_sum")
        self.points, gt_normals, _, self.error_pointcloud, self.error_stats = load_input(self.data_dir, dataname)
        self.normals = []
        self.gt_normals = gt_normals
        self.num_points = self.points.shape[0]
        self.normalize()
        self.sigmas = self.sample_gaussian_noise_around_shape()
        self.bbox = np.array([np.min(self.points - 0.05, axis=0), np.max(self.points + 0.05, axis=0)])

    def __len__(self):
        return self.datalength

    def __getitem__(self, index):
        alphas = gen_coefficients_np(self.time_sum, schedule=self.metric)
        alphas_cumsum = np.clip(np.cumsum(alphas, axis=0), 0, 1)
        point_idxes_permutation = np.random.permutation(self.points.shape[0])
        train_num_points = self.train_num_points
        # point_idxes = point_idxes_permutation[:train_num_points]
        # # Gaussian noise points
        # sample_surface = self.points[point_idxes]
        # sample_sigmas = self.sigmas[point_idxes]
        # theta_guassian = 0.25

        if self.stage == 1:
            point_idxes = point_idxes_permutation[:train_num_points]
            # Gaussian noise points
            sample_surface = self.points[point_idxes]
            sample_sigmas = self.sigmas[point_idxes]
            theta_guassian = 0.25
            noise = np.random.normal(0.0, 1.0, size=(train_num_points, 3)).astype(np.float32)
            sample = sample_surface + theta_guassian * sample_sigmas * noise
            sample_near_index = self.kd_tree.query(sample, k=1)[1]
            sample_near = self.points[sample_near_index]
            # sample_near = self.emd_projection(sample)
        else:
            num_gaussian = int(self.train_num_points * 0.0)
            num_alongNormal = self.train_num_points - num_gaussian
            point_gaussian_idxes = point_idxes_permutation[:num_gaussian]
            point_alongNormal_idxes = point_idxes_permutation[num_gaussian : self.train_num_points]
            # Gaussian noise points
            sample_gaussian_surface = self.points[point_gaussian_idxes]
            sample_gaussian_sigmas = self.sigmas[point_gaussian_idxes]
            theta_guassian = 0.25
            noise_gaussian = np.random.normal(0.0, 1.0, size=(num_gaussian, 3)).astype(np.float32)
            sample_gaussian = sample_gaussian_surface + theta_guassian * sample_gaussian_sigmas * noise_gaussian
            # sample_gaussian_near = self.bilateral_projection(sample_gaussian)
            _, nearest_idx = self.kd_tree.query(sample_gaussian, k=1)
            sample_gaussian_near = self.points[nearest_idx]
            # sample_gaussian_near = self.emd_projection(sample_gaussian)
            # sample_gaussian_near = sample_gaussian_surface
            # Along normal points
            sample_alongNormal_surface = self.points[point_alongNormal_idxes]
            sample_alongNormal_sigmas = self.sigmas[point_alongNormal_idxes]
            sample_alongNormal_normals = self.normals[point_alongNormal_idxes]
            theta_alongNormal = 0.25
            noise_alongNormal = np.random.normal(0.0, 1.0, size=(num_alongNormal, 1)).astype(np.float32)  # (N,1)
            sample_alongNormal = (
                sample_alongNormal_surface
                + theta_alongNormal * sample_alongNormal_sigmas * noise_alongNormal * sample_alongNormal_normals
            )
            sample_alongNormal_near = sample_alongNormal_surface
            sample = np.concatenate([sample_gaussian, sample_alongNormal], axis=0)
            sample_near = np.concatenate([sample_gaussian_near, sample_alongNormal_near], axis=0)
        res = sample - sample_near  # (N,3)
        time = np.random.randint(0, self.time_sum, size=(train_num_points, 1))

        sample_time = sample_near + extract_np(alphas_cumsum, time) * res
        return {
            "sample": sample,
            "sample_time": sample_time,
            "sample_near": sample_near,
            "time": time,
            "res": res,
        }

    def sample_gaussian_noise_around_shape(self):
        self.kd_tree = cKDTree(self.points)
        # query each point for sigma
        dist, _ = self.kd_tree.query(self.points, k=51, workers=-1)
        sigmas = dist[:, -1:]
        return sigmas.astype(np.float32)

    def normalize(self):
        self.shape_scale = np.max(
            [
                np.max(self.points[:, 0]) - np.min(self.points[:, 0]),
                np.max(self.points[:, 1]) - np.min(self.points[:, 1]),
                np.max(self.points[:, 2]) - np.min(self.points[:, 2]),
            ]
        )
        self.shape_center = [
            (np.max(self.points[:, 0]) + np.min(self.points[:, 0])) / 2,
            (np.max(self.points[:, 1]) + np.min(self.points[:, 1])) / 2,
            (np.max(self.points[:, 2]) + np.min(self.points[:, 2])) / 2,
        ]
        self.points = self.points - self.shape_center
        self.points = self.points / self.shape_scale
        self.points = self.points.astype(np.float32)

    def get_error_pointcloud(self):
        return self.error_pointcloud

    def get_new_points(self, extra_points, extra_normals):
        K = 1  # 最近邻数量

        # --- Step 1: 建树并查询 ---
        self.extra_kdtree = cKDTree(extra_points)
        dist, idx = self.extra_kdtree.query(self.points, k=K)

        if K == 1:
            dist = dist[:, np.newaxis]  # shape (N,) → (N, 1)
            idx = idx[:, np.newaxis]  # shape (N,) → (N, 1)

        # --- Step 2: 取邻居法向量 ---
        neighbor_normals = extra_normals[idx]  # shape: (N, K, 3)

        # --- Step 3: 定向对齐 ---
        ref_normals = neighbor_normals[:, 0, :]  # 参考方向
        sign = np.sign(np.sum(neighbor_normals * ref_normals[:, None, :], axis=2))
        sign[sign == 0] = 1
        aligned_normals = neighbor_normals * sign[..., None]

        # --- Step 4: 距离加权平均 ---
        weights = 1.0 / (dist + 1e-8)
        weights = weights / weights.sum(axis=1, keepdims=True)
        avg_normals = np.sum(aligned_normals * weights[..., None], axis=1)

        # --- Step 5: 单位化 ---
        norm = np.linalg.norm(avg_normals, axis=1, keepdims=True) + 1e-12
        avg_normals = avg_normals / norm
        self.normals = avg_normals.astype(np.float32)

        # --- Step 6: 计算法线误差 ---
        ni = self.normals.copy()
        n_gt = self.gt_normals.copy()
        ni /= np.linalg.norm(ni, axis=1, keepdims=True) + 1e-12
        n_gt /= np.linalg.norm(n_gt, axis=1, keepdims=True) + 1e-12
        cos_val = np.sum(ni * n_gt, axis=1)
        cos_val = np.clip(np.abs(cos_val), 0.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_val))  # 每个点的法线夹角误差

        # --- Step 7: 统计指标 ---
        mean_error = np.mean(angle_deg)  # 平均误差
        median_error = np.median(angle_deg)  # 中位误差
        std_error = np.std(angle_deg)  # 标准差
        max_error = np.max(angle_deg)  # 最大误差

        normal_error_stats = {
            "mean": float(mean_error),
            "median": float(median_error),
            "std": float(std_error),
            "max": float(max_error),
        }

        # --- Step 8: 颜色映射 (Matplotlib + Numpy) ---
        vmax_deg = 60.0
        norm = colors.Normalize(vmin=0.0, vmax=float(vmax_deg))
        cmap = cm.get_cmap("jet")
        col = cmap(norm(np.clip(angle_deg, 0.0, vmax_deg)))[:, :3]
        col_uint8 = (col * 255).astype(np.uint8)

        # --- Step 9: 创建 Trimesh 点云对象 ---
        estimate_error_pointcloud = trimesh.points.PointCloud(vertices=self.points, colors=col_uint8)

        # --- Step 10: 更新点云并返回 ---
        # 不加点云
        # self.normals = np.concatenate([self.normals, extra_normals], axis=0)
        # self.points = np.concatenate([self.points, extra_points], axis=0)

        self.num_points = self.points.shape[0]
        self.sigmas = self.sample_gaussian_noise_around_shape()
        self.kd_tree = cKDTree(self.points)

        # 返回误差可视化点云 + 统计指标
        return estimate_error_pointcloud, normal_error_stats

    def set_point_cloud(self, points, normals, stage):
        self.stage = stage
        self.points = points
        self.normals = normals
        self.num_points = self.points.shape[0]
        self.normalize()
        self.sigmas = self.sample_gaussian_noise_around_shape()
        self.bbox = np.array([np.min(self.points - 0.05, axis=0), np.max(self.points + 0.05, axis=0)])

    def bilateral_projection(self, query, k=10, sigma_p=None, sigma_n=None):
        M = query.shape[0]
        dists, idxs = self.kd_tree.query(query, k=k)
        dists = dists.astype(np.float32)  # (M,k)
        neigh_pos = self.points[idxs]  # (M,k,3)
        neigh_n = self.normals[idxs]  # (M,k,3)

        # 空间权重 ws
        if sigma_p is None:
            sig_p = dists[:, -1][:, None]  # 每个查询点的局部尺度
        else:
            sig_p = np.full((M, 1), float(sigma_p), dtype=np.float32)
        wp = np.exp(-(dists**2) / (sig_p**2))  # (M,k)
        sum_wp = wp.sum(axis=1, keepdims=True)

        # 参考法线,先对查询点
        # 加权均值(坐标)
        center = (wp[..., None] * neigh_pos).sum(axis=1) / sum_wp  # (M,3)
        # 获得加权PCA的法线
        X = neigh_pos - center[:, None, :]  # (M,k,3)
        J = np.einsum("mk,mki,mkj->mij", wp, X, X)  # (M,3,3)
        _, evecs = np.linalg.eigh(J)  # 批处理特征分解,默认升序
        n_ref = evecs[..., 0]  # 最小特征向量(M,3)
        # 定向参考法线
        # 1. 加权平均邻域法线
        mean_n = (wp[..., None] * neigh_n).sum(axis=1)  # (M,3)

        # 2. 与 PCA 法线点积求符号
        dots = (n_ref * mean_n).sum(axis=1, keepdims=True)  # (M,1)
        sign = np.where(dots >= 0.0, 1.0, -1.0).astype(np.float32)
        n_ref = n_ref * sign
        n_ref /= np.linalg.norm(n_ref, axis=1, keepdims=True)  # (M,3)
        neigh_n /= np.linalg.norm(neigh_n, axis=2, keepdims=True)
        # 残差 r_j = (q - x_j)·n_j(用每个邻居自己的法线)
        r = ((query[:, None, :] - neigh_pos) * neigh_n).sum(axis=2)  # (M,k)

        # 法线权重 wn
        if sigma_n is None:
            sig_n = np.full((M, 1), 0.5236, dtype=np.float32)  # 30度
        else:
            sig_n = np.full((M, 1), float(sigma_n), dtype=np.float32)

        dot = np.sum(n_ref[:, None, :] * neigh_n, axis=-1)
        dot = np.clip(dot, -1.0, 1.0)
        wn = np.exp(-(((1.0 - dot) / (1.0 - np.cos(sig_n))) ** 2))
        wn = np.maximum(wn, 1e-6)
        w = wp * wn
        w_sum = w.sum(axis=1)  # (M,)

        # 计算位移
        dis = (w * r).sum(axis=1) / w_sum  # (M,)

        proj = query - dis[:, None] * n_ref  # (M,3)
        return proj.astype(np.float32)

    def bilateral_projection_protection(self, query, k=10, sigma_p=None, sigma_n=None):
        M = query.shape[0]
        dists, idxs = self.kd_tree.query(query, k=k)
        dists = dists.astype(np.float32)  # (M,k)
        neigh_pos = self.points[idxs]  # (M,k,3)
        neigh_n = self.normals[idxs]  # (M,k,3)

        # 空间权重 ws
        if sigma_p is None:
            sig_p = dists[:, -1][:, None]  # 每个查询点的局部尺度
        else:
            sig_p = np.full((M, 1), float(sigma_p), dtype=np.float32)
        wp = np.exp(-(dists**2) / (sig_p**2))  # (M,k)
        sum_wp = wp.sum(axis=1, keepdims=True)

        # 参考法线,先对查询点
        # 加权均值(坐标)
        center = (wp[..., None] * neigh_pos).sum(axis=1) / sum_wp  # (M,3)
        # 获得加权PCA的法线
        X = neigh_pos - center[:, None, :]  # (M,k,3)
        J = np.einsum("mk,mki,mkj->mij", wp, X, X)  # (M,3,3)
        _, evecs = np.linalg.eigh(J)  # 批处理特征分解,默认升序
        n_ref = evecs[..., 0]  # 最小特征向量(M,3)
        # 定向参考法线
        # 1. 加权平均邻域法线
        mean_n = (wp[..., None] * neigh_n).sum(axis=1)  # (M,3)

        # 2. 与 PCA 法线点积求符号
        dots = (n_ref * mean_n).sum(axis=1, keepdims=True)  # (M,1)
        sign = np.where(dots >= 0.0, 1.0, -1.0).astype(np.float32)
        n_ref = n_ref * sign
        n_ref /= np.linalg.norm(n_ref, axis=1, keepdims=True)  # (M,3)
        neigh_n /= np.linalg.norm(neigh_n, axis=2, keepdims=True)
        # 残差 r_j = (q - x_j)·n_j(用每个邻居自己的法线)
        r = ((query[:, None, :] - neigh_pos) * neigh_n).sum(axis=2)  # (M,k)

        # 法线权重 wn
        if sigma_n is None:
            # sig_n = np.full((M, 1), float(0.2618), dtype=np.float32)  # 15度
            sig_n = np.full((M, 1), 0.5236, dtype=np.float32)  # 30度
        else:
            sig_n = np.full((M, 1), float(sigma_n), dtype=np.float32)

        dot = np.sum(n_ref[:, None, :] * neigh_n, axis=-1)
        dot = np.clip(dot, -1.0, 1.0)
        wn = np.exp(-(((1.0 - dot) / (1.0 - np.cos(sig_n))) ** 2))
        wn = np.maximum(wn, 1e-6)
        w = wp * wn
        w_sum = w.sum(axis=1)  # (M,)

        # 计算位移
        dis = (w * r).sum(axis=1) / w_sum  # (M,)

        proj = query - dis[:, None] * n_ref  # (M,3)
        # 计算最近邻点和距离
        nn_pos = neigh_pos[:, 0, :]  # (M,3)
        nn_dist = dists[:, 0]  # (M,)

        # 如果计算的投影距离绝对值大于最近点欧氏距离,则采用最近点作为投影
        mask = np.abs(dis) > nn_dist
        proj[mask] = nn_pos[mask]
        # # 输出一步赋值最近点的个数
        # count = np.sum(mask)
        # print(f"一步赋值最近点的个数: {count}")

        return proj.astype(np.float32)

    def average_projection(self, query, k=10, sigma_p=None):
        M = query.shape[0]
        dists, idxs = self.kd_tree.query(query, k=k)
        dists = dists.astype(np.float32)  # (M,k)
        neigh_pos = self.points[idxs]  # (M,k,3)

        # 空间权重 ws
        if sigma_p is None:
            sig_p = dists[:, -1][:, None]  # 每个查询点的局部尺度
        else:
            sig_p = np.full((M, 1), float(sigma_p), dtype=np.float32)
        wp = np.exp(-(dists**2) / (sig_p**2))  # (M,k)
        sum_wp = wp.sum(axis=1, keepdims=True)

        # 参考法线,先对查询点
        # 加权均值(坐标)
        projection = (wp[..., None] * neigh_pos).sum(axis=1) / sum_wp  # (M,3)

        proj = projection  # (M,3)
        return proj.astype(np.float32)
