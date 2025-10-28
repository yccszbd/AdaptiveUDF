# tools/slice.py
import os
import numpy as np
import torch
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# --------------------------------------------------
# 1. 内部小工具
# --------------------------------------------------
def _load_normalize_mesh(path_prefix):
    for ext in (".ply", ".obj", ".xyz"):
        path = path_prefix + ext
        if os.path.isfile(path):
            return trimesh.load_mesh(path)
    raise FileNotFoundError(f"找不到模型: {path_prefix}.[ply|obj|xyz]")


def _auto_bbox(field_func, mesh=None, device="cuda", n_samples=100000):
    # ### GRAD-MOD ① ###  去掉内部 torch.no_grad()
    pts = torch.rand(n_samples, 3, device=device) * 2 - 1
    vals = field_func(pts)  # 可能带梯度
    mask = (
        (vals < 0.05) if vals.numel() > 100 else torch.ones_like(vals, dtype=torch.bool)
    )
    if mask.sum() < 10:
        fallback = np.array([[-1, -1, -1], [1, 1, 1]], dtype=np.float32)
        return fallback[0], fallback[1]
    pts = pts[mask.squeeze()]
    return pts.amin(0).cpu().detach().numpy(), pts.amax(0).cpu().detach().numpy()


def _judge_field_type(field_func, device="cuda"):
    # ### GRAD-MOD ② ###  去掉 torch.no_grad()
    pts = torch.randn(5000, 3, device=device)
    v = field_func(pts)
    return "sdf" if (v < 0).any() else "udf"


def _adaptive_levels(
    vmin,
    vmax,
    dense_threshold=0.05,
    dense_step=0.005,
    sparse_step=0.02,
    field_type="udf",
):
    """
    根据 SDF/UDF 值自适应生成等值线层级
    参数：
        dense_threshold: 绝对值小于该值视为高密度区
        dense_step: 高密度区每隔多少值画一条线
        sparse_step: 低密度区每隔多少值画一条线
        field_type: "sdf" 或 "udf"
    """
    z = 0.0
    levels = []

    # 确保方向正确
    if field_type == "sdf":
        neg = np.arange(z, vmin, -sparse_step)[::-1]  # 负方向稀疏
        pos = np.arange(z, vmax, sparse_step)[1:]  # 正方向稀疏
        dense_neg = np.arange(z, vmin, -dense_step)[::-1]
        dense_pos = np.arange(z, vmax, dense_step)[1:]
    else:  # UDF
        neg = []
        pos = np.arange(vmin, vmax, sparse_step)
        dense_pos = np.arange(vmin, min(vmax, dense_threshold), dense_step)
        dense_neg = []

    # 高密度区
    dense_levels = []
    if field_type == "sdf":
        dense_levels += [l for l in dense_neg if abs(l) <= dense_threshold]
        dense_levels += [l for l in dense_pos if abs(l) <= dense_threshold]
    else:
        dense_levels += [l for l in dense_pos if l <= dense_threshold]

    # 低密度区
    sparse_levels = []
    if field_type == "sdf":
        sparse_levels += [l for l in neg if abs(l) > dense_threshold]
        sparse_levels += [l for l in pos if abs(l) > dense_threshold]
    else:
        sparse_levels += [l for l in pos if l > dense_threshold]

    levels = np.unique(np.concatenate([dense_levels, sparse_levels]))
    return levels


def _should_flip(mesh, axis_id, a1, a2):
    """
    根据 mesh 长宽比与体素网格长宽比自动决定是否翻转
    返回 (flip_x, flip_y) 的 bool 元组
    """
    if mesh is None:
        return False, False
    extents = mesh.extents
    len1, len2 = extents[a1], extents[a2]
    if axis_id == 0:  # X 切片
        return (len1 < len2), False
    elif axis_id == 1:  # Y 切片
        return False, (len1 > len2)
    else:  # Z 切片
        return False, False


def _auto_flip_vote(
    field_func,
    mesh,
    axis,
    axis_id,
    a1,
    a2,
    extent_plt,
    lr_none,
    ud_none,
    device,
    th,
    pct,
    n,
):
    """返回 (lr_best, ud_best) 两个 bool"""
    if mesh is None:
        return False, False

    plane_normal = [0, 0, 0]
    plane_normal[axis_id] = 1
    plane_origin = [0, 0, 0]
    # 先切一层轴心 0 面拿轮廓
    plane_origin[axis_id] = 0.0
    sec = mesh.section(plane_normal=plane_normal, plane_origin=plane_origin)
    if sec is None or len(sec.vertices) == 0:
        return False, False

    # 在所有轮廓上均匀随机采 n 个点
    # 把 Path3D 所有线段打散成顶点再随机采
    verts = np.vstack([ent.discrete(sec.vertices) for ent in sec.entities])
    if verts.shape[0] < 50:
        return False, False
    # 随机采 n 个点（有放回）
    idx = np.random.choice(verts.shape[0], size=min(n, verts.shape[0]), replace=True)
    pts_3d = verts[idx]
    if pts_3d.shape[0] < 50:  # 太少直接放弃
        return False, False

    # 映射到图像坐标
    coord1, coord2 = pts_3d[:, a1], pts_3d[:, a2]
    # 四种翻转组合
    combos = [(False, False), (True, False), (False, True), (True, True)]
    scores = []
    for lr, ud in combos:
        c1, c2 = coord1.copy(), coord2.copy()
        if lr:
            c1 = extent_plt[0] + extent_plt[1] - c1
        if ud:
            c2 = extent_plt[2] + extent_plt[3] - c2
        # 把图像坐标再变回 3D 查询点（z=0 面）
        q = torch.zeros(c1.shape[0], 3, device=device)
        q[:, a1] = torch.as_tensor(c1, device=device)
        q[:, a2] = torch.as_tensor(c2, device=device)
        q[:, axis_id] = 0.0
        vals = field_func(q).abs().squeeze()
        # 合格率
        ok_rate = (vals < th).float().mean().item()
        scores.append(ok_rate)
    best_idx = int(np.argmax(scores))
    if scores[best_idx] < pct:  # 连最好都达不到要求 -> 放弃翻转
        return False, False
    return combos[best_idx]


# --------------------------------------------------
# 2. 单方向切片
# --------------------------------------------------
def save_field_slices(
    field_func,  # lambda pts: network(pts) -> (N,1)/(N,)
    bbox_min=None,
    bbox_max=None,  # 可传 None，自动估计
    gt_mesh_prefix=None,
    pred_mesh_prefix=None,
    out_dir=".",
    n_slices=7,
    axis="z",
    levels=None,
    resolution=512,
    colormap="turbo",
    clim_percentile=(1, 99),
    dpi=150,
    contour_colors="black",
    contour_linewidths=0.4,
    fill_contour=True,
    alpha=0.8,
    customcm=-1,
    device="cuda",
    field_type="auto",  # "sdf"/"udf"/"auto"
    dense_threshold=0.05,  # 高密度区阈值
    dense_step=0.005,  # 高密度区步长
    sparse_step=0.02,  # 低密度区步长
    manual_flip_xyz=None,  # 新格式：{"x": {"lr":None|Bool, "ud":None|Bool}, ...}
    auto_flip_th=0.02,  # 自动投票阈值：|SDF|/|UDF| < th 认为“在面上”
    auto_flip_pct=0.6,  # 多少比例点在面上就判合格
    auto_flip_n=200,  # 每轴随机采样点数
    extent_ratio=1.0,  # ① 显示范围缩放
    extra_planes=None,  # ② 强制额外截面
):
    os.makedirs(out_dir, exist_ok=True)
    if manual_flip_xyz is None:
        manual_flip_xyz = {ax: {"lr": None, "ud": None} for ax in ["x", "y", "z"]}
    # ---- 2.1 自动识别 SDF/UDF ----
    if field_type == "auto":
        field_type = _judge_field_type(field_func, device)
        print(f"[slice] 自动判断为 {field_type.upper()} 场")

    # ---- 2.2 自动估计 bbox ----
    if bbox_min is None or bbox_max is None:
        pred_mesh = _load_normalize_mesh(pred_mesh_prefix) if pred_mesh_prefix else None
        bbox_min, bbox_max = _auto_bbox(field_func, pred_mesh, device)
        print(f"[slice] 自动估计 bbox: min={bbox_min}, max={bbox_max}")

    axis_id = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    extent = np.array(bbox_max) - np.array(bbox_min)
    axis_len = extent[axis_id]

    # ---- 2.2b 新增：在已得 bbox 基础上同向缩放 ----
    if extent_ratio != 1.0:
        center = (bbox_min + bbox_max) / 2
        half = (bbox_max - bbox_min) / 2 * extent_ratio
        bbox_min, bbox_max = center - half, center + half
        print(
            f"[slice] extent_ratio={extent_ratio} -> 缩放后 bbox min={bbox_min}, max={bbox_max}"
        )

    # ---- 2.3 切片高度 ----
    if levels is None:
        ratios = np.linspace(0, 1, n_slices + 2)[1:-1]
        levels = [float(bbox_min[axis_id] + r * axis_len) for r in ratios]
    else:
        levels = [float(l) for l in np.asarray(levels).reshape(-1)]

    # ---- 2.3b 新增：把 extra_planes 并入 levels（不去重也行，sorted 自动去重） ----
    if extra_planes is not None and len(extra_planes):
        levels = sorted(set(levels) | set(map(float, extra_planes)))
        print(f"[slice] extra_planes {extra_planes} 已并入 -> 共 {len(levels)} 层")

    # ---- 2.4 加载 mesh ----
    pred_mesh = _load_normalize_mesh(pred_mesh_prefix) if pred_mesh_prefix else None

    # ---- 2.5 绘图网格 ----
    ids = [0, 1, 2]
    ids.remove(axis_id)
    a1, a2 = ids
    len1, len2 = extent[a1], extent[a2]
    if len1 > len2:
        res1, res2 = resolution, max(int(resolution * len2 / len1), 1)
    else:
        res2, res1 = resolution, max(int(resolution * len1 / len2), 1)
    xx = torch.linspace(float(bbox_min[a1]), float(bbox_max[a1]), res1)
    yy = torch.linspace(float(bbox_min[a2]), float(bbox_max[a2]), res2)
    XX, YY = torch.meshgrid(xx, yy, indexing="ij")
    XX, YY = XX.to(device), YY.to(device)

    """
    # ---- 2.6 单轴翻转策略 ----
    # 统一转成 dict，缺省用 None（代表自动）
    if manual_flip_xyz is None:
        manual_flip_xyz = {"x":None, "y":None, "z":None}
    elif isinstance(manual_flip_xyz, (list, tuple)):
        manual_flip_xyz = dict(zip(["x","y","z"], manual_flip_xyz))

    flip_flag = manual_flip_xyz.get(axis, None)   # 当前轴的策略
    if flip_flag is None:                         # 自动
        flip_x, flip_y = _should_flip(pred_mesh, axis_id, a1, a2)
    else:                                         # 手动
        flip_x = flip_y = flip_flag
    """

    # --------------------------------------------------
    # 2.6b 解析翻转策略（自动投票只跑一次）
    # --------------------------------------------------
    # 1) 先统一转成字典结构
    if manual_flip_xyz is None:
        manual_flip_xyz = {ax: {"lr": None, "ud": None} for ax in ["x", "y", "z"]}

    # 2) 缓存 key 提前生成，但投票结果在第一次循环里填
    if not hasattr(save_field_slices, "_flip_cache"):
        save_field_slices._flip_cache = {}
    cache_key = (axis, id(pred_mesh) if pred_mesh else "dummy")

    # 先占位，第一次循环再算
    flip_lr = flip_ud = None

    # ---- 2.7 逐层推理 & 绘图 ----
    for lvl_idx, lvl in enumerate(levels):
        pts = torch.zeros(res1, res2, 3, device=device)
        pts[..., a1] = XX
        pts[..., a2] = YY
        pts[..., axis_id] = lvl
        pts = pts.reshape(-1, 3)

        vals = []
        batch = 100000
        for i in range(0, pts.shape[0], batch):
            v = field_func(pts[i : i + batch]).reshape(-1)
            vals.append(v)
        vals = torch.cat(vals).reshape(res1, res2).cpu().detach().numpy()

        # ---- 第一次拿到 extent_plt 后立刻投票 ----
        need_swap = len2 > len1  # 与 vals.T 的条件保持一致
        if need_swap:
            extent_plt = [yy[0].item(), yy[-1].item(), xx[0].item(), xx[-1].item()]
            img_arr = vals
        else:
            extent_plt = [xx[0].item(), xx[-1].item(), yy[0].item(), yy[-1].item()]
            img_arr = vals.T

        if lvl_idx == 0 and cache_key not in save_field_slices._flip_cache:
            # 此时 extent_plt 已确定，可以安全投票
            lr_need = manual_flip_xyz[axis]["lr"] is None
            ud_need = manual_flip_xyz[axis]["ud"] is None
            if lr_need or ud_need:
                lr_best, ud_best = _auto_flip_vote(
                    field_func,
                    pred_mesh,
                    axis,
                    axis_id,
                    a1,
                    a2,
                    extent_plt,
                    lr_need,
                    ud_need,
                    device,
                    auto_flip_th,
                    auto_flip_pct,
                    auto_flip_n,
                )
                if lr_need:
                    manual_flip_xyz[axis]["lr"] = lr_best
                if ud_need:
                    manual_flip_xyz[axis]["ud"] = ud_best
            # 写缓存
            save_field_slices._flip_cache[cache_key] = {
                "lr": manual_flip_xyz[axis]["lr"],
                "ud": manual_flip_xyz[axis]["ud"],
            }

        # 取出最终策略（后续楼层直接复用）
        flip_lr = save_field_slices._flip_cache[cache_key]["lr"]
        flip_ud = save_field_slices._flip_cache[cache_key]["ud"]

        # ---- 应用翻转到图像 ----
        if flip_lr:
            extent_plt[0], extent_plt[1] = extent_plt[1], extent_plt[0]
            img_arr = img_arr[:, ::-1]
        if flip_ud:
            extent_plt[2], extent_plt[3] = extent_plt[3], extent_plt[2]
            img_arr = img_arr[::-1, :]

        vmin, vmax = np.percentile(img_arr, clim_percentile)
        # 只在 SDF 时把 colorbar 中心固定在 0
        if field_type == "sdf":
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max  # 强制对称
        else:  # UDF 保持原样
            vmin, vmax = np.percentile(img_arr, clim_percentile)
        c_levels = _adaptive_levels(
            vmin,
            vmax,
            dense_threshold=dense_threshold,
            dense_step=dense_step,
            sparse_step=sparse_step,
            field_type=field_type,
        )

        # ... 后面继续你原来的绘图、contour、colorbar、savefig 逻辑（无需改动）
        # （只把 flip_x / flip_y 换成 flip_lr / flip_ud 即可）

        # ↓↓↓ 补下面这几行 ↓↓↓
        fig_w = 6.0 if len2 > len1 else 6.0 * len1 / len2
        fig_h = 6.0 * len2 / len1 if len2 > len1 else 6.0
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_aspect("equal", adjustable="box")
        im = ax.imshow(
            img_arr,
            origin="lower",
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            extent=extent_plt,
            alpha=alpha,
        )
        # ↑↑↑ 到这里 ax 已生成 ↑↑↑

        # 后面你原来的 contour、colorbar、savefig 逻辑保持不变
        if fill_contour:
            ax.contourf(
                img_arr,
                levels=c_levels,
                cmap=colormap,
                extent=extent_plt,
                alpha=alpha,
                origin="lower",
            )
        ax.contour(
            img_arr,
            levels=c_levels,
            colors=contour_colors,
            linewidths=contour_linewidths,
            extent=extent_plt,
            origin="lower",
        )

        # ---- 2.8 绘制 mesh 截面 ----
        if pred_mesh is not None:
            plane_normal = [0, 0, 0]
            plane_normal[axis_id] = 1
            plane_origin = [0, 0, 0]
            plane_origin[axis_id] = float(lvl)
            sec = pred_mesh.section(
                plane_normal=plane_normal, plane_origin=plane_origin
            )
            if sec is not None:
                for ent in sec.entities:
                    path = ent.discrete(sec.vertices)
                    if path.shape[0] == 0:
                        continue
                    c1, c2 = path[:, a1], path[:, a2]  # 原始坐标

                    # 1) 同步 90° 旋转（和图像 vals.T 保持一致）
                    if need_swap:
                        c1, c2 = c2, c1

                    # 2) 翻转轴也要同步 swap！
                    flip_lr_now = flip_lr
                    flip_ud_now = flip_ud
                    if need_swap:  # 图像轴顺序已换，翻转定义跟着换
                        flip_lr_now, flip_ud_now = flip_ud, flip_lr

                    # 3) 应用翻转
                    if flip_lr_now:
                        c1 = extent_plt[0] + extent_plt[1] - c1
                    if flip_ud_now:
                        c2 = extent_plt[2] + extent_plt[3] - c2

                    ax.plot(c1, c2, color="black", lw=1.5, zorder=11)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title(f"{field_type.upper()} slice {axis}={lvl:.3f}")
        suffix = f"custum{customcm}" if customcm != -1 else colormap
        fname = os.path.join(out_dir, f"{suffix}_{axis}_{lvl:.3f}.png")
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"[slice] Saved {len(levels)} {field_type.upper()} slices -> {out_dir}")


# --------------------------------------------------
# 3. 一键三方向 + 多 colormap
# --------------------------------------------------
def save_all_slice_views(
    field_func,
    bbox_min=None,
    bbox_max=None,
    gt_mesh_prefix=None,
    pred_mesh_prefix=None,
    out_root=".",
    iter_step=0,
    device="cuda",
    field_type="auto",
    dense_threshold=0.05,  # 高密度区阈值
    dense_step=0.005,  # 高密度区步长
    sparse_step=0.02,  # 低密度区步长
    manual_flip_xyz={
        "x": {"lr": None, "ud": None},  # 全自动
        "y": {"lr": True, "ud": None},  # y 轴左右强制翻，上下自动
        "z": {"lr": False, "ud": False},  # z 轴完全强制不翻
    },
    auto_flip_th=0.02,  # |SDF|/|UDF| < 0.02 认为“在面上”
    auto_flip_pct=0.6,  # 60 % 点合格就采纳
    auto_flip_n=300,  # 每轴采 300 点
    extent_ratio=1.0,
    extra_planes=[0],
):
    """
    一键生成 x/y/z 三个方向、多种 colormap 的全部切片。
    所有参数含义与 save_field_slices 相同。
    """
    axes = ["x", "y", "z"]
    out_dir = os.path.join(out_root, str(iter_step))

    def _once(cmap, cm_id):
        for ax in axes:
            save_field_slices(
                field_func,
                bbox_min,
                bbox_max,
                gt_mesh_prefix,
                pred_mesh_prefix,
                out_dir,
                axis=ax,
                n_slices=9,
                resolution=512,
                colormap=cmap,
                contour_colors="black",
                contour_linewidths=0.2,
                fill_contour=True,
                device=device,
                field_type=field_type,
                dense_threshold=dense_threshold,
                dense_step=dense_step,
                sparse_step=sparse_step,
                manual_flip_xyz=manual_flip_xyz,
                auto_flip_th=auto_flip_th,
                auto_flip_pct=auto_flip_pct,
                auto_flip_n=auto_flip_n,
                customcm=cm_id,
                extent_ratio=extent_ratio,
                extra_planes=extra_planes,
            )

    _once("Oranges", -1)
    # _once("coolwarm",-1)
    # cmap1 = mcolors.LinearSegmentedColormap.from_list("OrangeWhite",[(0,"white"),(1,"orange")])
    # _once(cmap1,1)
    # cmap2 = mcolors.LinearSegmentedColormap.from_list(
    #     "BlueWhite", [(0, "white"), (1, "blue")]
    # )
    # _once(cmap2, 2)
    # cmap3 = mcolors.LinearSegmentedColormap.from_list("BlueGreen",[(0,"green"),(0.01,"white"),(1,"blue")])
    # _once(cmap3,3)
    # cmap_sdf_rwb = mcolors.LinearSegmentedColormap.from_list("SDF_RWB", [(0.0, "red"), (0.5, "white"), (1.0, "blue")])
    # cmap_sdf_rwb = mcolors.LinearSegmentedColormap.from_list(
    #     "SDF_RWB",
    #     [(0.0, (1.0, 0.0, 0.0)), (0.5, (1.0, 1.0, 1.0)), (1.0, (0.0, 0.0, 1.0))],
    # )
    # _once(cmap_sdf_rwb, 4)  # 12 专门留给 SDF
