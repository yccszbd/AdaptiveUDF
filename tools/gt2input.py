#!/usr/bin/env python3
"""
Mesh → Point-cloud batch sampler
支持命令行配置采样方式、点数、并行度
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from tools.utils import normalize_mesh, start_process_pool

# ------------------------------------------------------------------
# 采样策略 dispatcher
# ------------------------------------------------------------------


def sample_random_surface(mesh: trimesh.Trimesh, count: int) -> tuple[np.ndarray, np.ndarray]:
    """纯随机:随机面 + 随机重心坐标,一行 Open3D 搞定"""
    # trimesh → open3d
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices), triangles=o3d.utility.Vector3iVector(mesh.faces)
    )
    # 计算顶点法向(可选,但采样时会插值面法向)
    o3d_mesh.compute_vertex_normals()

    pcd = o3d_mesh.sample_points_uniformly(number_of_points=count)  # ← 核心一行
    pts = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)  # 已插值好
    return pts, normals


def sample_uniform_surface(mesh: trimesh.Trimesh, count: int) -> tuple[np.ndarray, np.ndarray]:
    """均匀表面采样(面积加权)"""
    pts, face_idx = trimesh.sample.sample_surface(mesh, count)
    normals = mesh.face_normals[face_idx]
    return pts, normals


def sample_poisson_disk(mesh: trimesh.Trimesh, count: int) -> tuple[np.ndarray, np.ndarray]:
    """泊松盘采样(trimesh 自带)"""
    pts, face_idx = trimesh.sample.sample_surface_even(mesh, count)
    normals = mesh.face_normals[face_idx]
    return pts, normals


def sample_poisson_fps(mesh: trimesh.Trimesh, count: int) -> tuple[np.ndarray, np.ndarray]:
    """
    1. 先用泊松盘采 2*count 个点
    2. 再用最远点采样降回到 count
    """
    pts, face_idx = trimesh.sample.sample_surface_even(mesh, count * 2)
    # 简单最远点采样
    selected = [np.random.randint(0, len(pts))]
    dists = np.full(len(pts), np.inf)
    for _ in range(count - 1):
        last = selected[-1]
        dists = np.minimum(dists, np.linalg.norm(pts - pts[last], axis=1))
        selected.append(int(np.argmax(dists)))
    pts = pts[selected]
    normals = mesh.face_normals[face_idx[selected]]
    return pts, normals


SAMPLER = {
    "random": sample_random_surface,
    "uniform": sample_uniform_surface,
    "poisson": sample_poisson_disk,
    "poisson_fps": sample_poisson_fps,
}


# ------------------------------------------------------------------
# 单文件处理
# ------------------------------------------------------------------
def data_no_filter(
    mesh_path: Path, input_path: Path, sample_num: int, sample_mode: str, with_normal: bool = True
) -> None:
    print(f"[PID {os.getpid()}] Processing: {mesh_path}")

    try:
        mesh = trimesh.load_mesh(mesh_path, process=False, maintain_order=False, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                print(f"Empty scene: {mesh_path}")
                return
            mesh = mesh.to_mesh()

        mesh = normalize_mesh(mesh)

        # 采样
        pts, normals = SAMPLER[sample_mode](mesh, sample_num)

        # 保存
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if with_normal:  # ← 新增判断
            pcd.normals = o3d.utility.Vector3dVector(normals)

        out_file = input_path / f"{mesh_path.stem}.ply"
        o3d.io.write_point_cloud(str(out_file), pcd)

        print(f"Saved -> {out_file}")
    except Exception as e:
        print(f"Error on {mesh_path}: {e}")


# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch mesh → point-cloud sampler")
    parser.add_argument("--sample_mode", choices=SAMPLER.keys(), default="uniform", help="Sampling strategy")
    parser.add_argument("--sample_num", type=int, default=100_000, help="Number of points to sample")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel processes")
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="/home/ycc/data/DiffusionUDF不同版本/data/shapenetCars100000/ground_truth",
        help="Directory containing ground-truth meshes",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/ycc/data/DiffusionUDF不同版本/data/shapenetCars100000/input",
        help="Directory to save point clouds",
    )

    parser.add_argument("--without_normal", action="store_true", help="If set, do NOT write normals to output PLY")
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_files = sorted(p for p in gt_dir.iterdir() if p.suffix.lower() in {".ply", ".obj", ".off"})

    call_params = [(m, out_dir, args.sample_num, args.sample_mode, not args.without_normal) for m in mesh_files]

    start_process_pool(data_no_filter, call_params, args.workers)


if __name__ == "__main__":
    import os

    main()
