from pathlib import Path

import open3d as o3d
import trimesh

# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from tools.utils import normalize_mesh, start_process_pool


def data_no_filter(mesh_path, input_path, sample_pt_num=10000):
    print(f"Processing: {mesh_path}")

    try:
        mesh = trimesh.load_mesh(mesh_path, process=False, force="mesh")
        print(f"Loaded mesh type: {type(mesh)}")

        if isinstance(mesh, trimesh.Scene):
            print(f"Scene contains {len(mesh.geometry)} geometries")
            if len(mesh.geometry) == 0:
                print(f"Empty scene: {mesh_path}")
                return
            # 合并 Scene 中的所有几何体为单个 Trimesh
            mesh = mesh.to_mesh()
            print(f"Merged mesh type: {type(mesh)}")

        mesh = normalize_mesh(mesh)
        print(f"Normalized mesh type: {type(mesh)}")

        pts, idx = trimesh.sample.sample_surface(mesh, count=sample_pt_num)
        normals = mesh.face_normals[idx]

        pts_o3d = o3d.geometry.PointCloud()
        pts_o3d.points = o3d.utility.Vector3dVector(pts)
        pts_o3d.normals = o3d.utility.Vector3dVector(normals)

        f_name = Path(mesh_path).stem  # 取文件名(不含扩展名)
        out_path = Path(input_path) / f"{f_name}.ply"  # 使用 / 拼接 + f-string
        o3d.io.write_point_cloud(str(out_path), pts_o3d)  # open3d 仍需字符串路径

        return
    except Exception as e:
        print(e)
        print("error", mesh_path)


if __name__ == "__main__":
    gt_path = Path("/home/ycc/data/DiffusionUDF不同版本/data/shapenetCars100000/ground_truth")
    gt_path.mkdir(parents=True, exist_ok=True)

    input_path = Path("/home/ycc/data/DiffusionUDF不同版本/data/shapenetCars100000/input")
    input_path.mkdir(parents=True, exist_ok=True)

    num_processes = 16
    sample_pt_num = 100000

    call_params: list[tuple[Path, Path, int]] = [
        (mesh_path, input_path, sample_pt_num)
        for mesh_path in sorted(gt_path.iterdir())
        if mesh_path.suffix in {".ply", ".obj", ".off"}
    ]

    start_process_pool(data_no_filter, call_params, num_processes)
