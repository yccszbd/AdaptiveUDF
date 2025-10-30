import argparse  # noqa: INP001
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from models.dataset import Dataset
from models.fields import CAPUDFNetwork
from pyhocon import ConfigFactory
from tools.log2csv import parse_log_to_table
from tools.logger import get_root_logger, print_log
from tools.slice import save_all_slice_views
from tools.surface_extraction import surface_extraction
from tools.utils import (
    conf_log,
    count_parameters,
    eval_pointcloud,
    normalize_mesh,
    remove_far,
    set_seed,
    setCUDA,
)

# from tools.slice import save_all_slice_views
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm


def extract_fields(bound_min, bound_max, resolution, query_func, grad_func):
    N = 32
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    g = np.zeros([resolution, resolution, resolution, 3], dtype=np.float32)
    # with torch.no_grad():
    for xi, xs in tqdm(enumerate(X), total=len(X), desc="Calculate field"):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = torch.meshgrid(xs, ys, zs)

                pts = torch.cat(
                    [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                    dim=-1,
                ).cuda()

                grad = grad_func(pts).reshape(len(xs), len(ys), len(zs), 3).detach().cpu().numpy()
                val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                u[
                    xi * N : xi * N + len(xs),
                    yi * N : yi * N + len(ys),
                    zi * N : zi * N + len(zs),
                ] = val
                g[
                    xi * N : xi * N + len(xs),
                    yi * N : yi * N + len(ys),
                    zi * N : zi * N + len(zs),
                ] = grad

    return u, g


def extract_geometry(
    bound_min,
    bound_max,
    resolution,
    threshold,
    query_func,
    grad_func,
):
    print(f"Extracting mesh with resolution: {resolution}")
    u, g = extract_fields(bound_min, bound_max, resolution, query_func, grad_func)
    b_max = bound_max.detach().cpu().numpy()
    b_min = bound_min.detach().cpu().numpy()
    mesh = surface_extraction(u, g, threshold, b_max, b_min, resolution)

    return mesh


class Runner:
    def __init__(self, args, conf_path):
        set_seed(123)
        self.device = setCUDA(args.gpu)
        # Configuration
        self.conf_path = Path(conf_path)
        conf_text = self.conf_path.read_text()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.dir = args.dir
        self.dataname = args.dataname
        self.base_exp_dir = Path(self.conf["general.base_exp_dir"]) / args.dir / args.dataname
        self.base_exp_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_stage_1 = Dataset(self.conf, self.dir, self.dataname, stage=1)
        self.dataset_stage_2 = Dataset(self.conf, self.dir, self.dataname, stage=2)
        self.GT_points = torch.from_numpy(self.dataset_stage_1.points).to(self.device)
        self.batch_size = self.conf.get_int("train.batch_size")
        self.dataloader_stage_1 = torch.utils.data.DataLoader(
            self.dataset_stage_1,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        self.dataloader_stage_2 = torch.utils.data.DataLoader(
            self.dataset_stage_2,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

        self.datalength = self.conf.get_int("train.datalength")
        # self.epochs = self.conf.get_int("train.epochs")
        self.epochs_stage_1 = self.conf.get_int("train.epochs_stage_1")
        self.epochs_stage_2 = self.conf.get_int("train.epochs_stage_2")
        self.epochs = self.epochs_stage_1 + self.epochs_stage_2

        self.learning_rate_stage1 = self.conf.get_float("train.learning_rate_stage1")
        self.learning_rate_stage2 = self.conf.get_float("train.learning_rate_stage2")

        self.warm_up_end_stage1 = self.conf.get_int("train.warm_up_end_stage1")
        self.warm_up_end_stage2 = self.conf.get_int("train.warm_up_end_stage2")

        self.metric = self.conf.get_string("train.metric")
        self.epoch_step = 0

        # Training parameters
        self.time_sum = self.conf.get_int("train.time_sum")
        self.report_freq = self.conf.get_int("train.report_freq")
        self.batch_size = self.conf.get_int("train.batch_size")
        # 产生额外点云的数量
        self.extra_points_rate = self.conf.get_int("train.extra_points_rate")
        self.noise_range = self.conf.get_float("train.noise_range")

        # Networks
        self.udf_network = CAPUDFNetwork(**self.conf["model.udf_network"]).to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.udf_network.parameters(), lr=self.learning_rate_stage1)

    def train(self):
        log_dir = self.base_exp_dir / "log"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"{self.epochs}_{self.time_sum}epoch_.log"
        self.writer = SummaryWriter(log_dir=log_dir)
        logger = get_root_logger(log_file=log_file, name="outs")
        self.logger = logger
        if self.conf.get_string("train.load_ckpt") != "None":
            self.load_checkpoint(self.conf.get_string("train.load_ckpt"))
        print_log(
            f"Message: {self.conf.get_string('train.message')}",
            logger=self.logger,
        )
        print_log(
            f"Dataset: Bounding Box:{self.dataset_stage_1.bbox} \n Center:{self.dataset_stage_1.shape_center} \n Scale:{self.dataset_stage_1.shape_scale}",
            logger=self.logger,
        )
        (self.base_exp_dir / "pointcloud").mkdir(parents=True, exist_ok=True)
        self.error_stats = self.dataset_stage_1.error_stats
        self.error_pointcloud = self.dataset_stage_1.error_pointcloud
        if len(self.error_pointcloud.vertices) == 0:
            print_log("No Normals loaded.", logger=self.logger)
        else:
            error_pointcloud_path = self.base_exp_dir / "pointcloud" / "pca_error_point_cloud.ply"
            self.error_pointcloud.export(error_pointcloud_path)
            print_log(
                f"pca Error point cloud saved at {error_pointcloud_path}.",
                logger=self.logger,
            )
            print_log("pca_Normal Error Stats:")
            print_log(f"Mean={self.error_stats['mean']:.6f}", logger=self.logger)
            print_log(f"Median={self.error_stats['median']:.6f}", logger=self.logger)
            print_log(f"Std={self.error_stats['std']:.6f}", logger=self.logger)
            print_log(f"Max={self.error_stats['max']:.6f}", logger=self.logger)

        n_parameters = count_parameters(self.udf_network)
        print_log(
            f"Number of parameters in UDF network: {n_parameters}",
            logger=self.logger,
        )
        iter_sum = self.epochs * self.datalength
        print_log(
            f"Number of iter in training: {iter_sum}",
            logger=self.logger,
        )
        print_log(f"Training {self.dataname} with Conf {args.conf}", logger=self.logger)
        conf_log(self.conf, logger=self.logger)

        # len(dataset)是一个iter中的总batch数
        # get item是获取一个batch
        # dataloader的batchsize是每次取多少batch
        # batch是最大的范围,表示一共训练多少轮
        # iter是从dataloader中取数据的次数
        # 一个batch有多少个iter由dataloader的长度决定,len(dataloader)=len(dataset)/batchsize
        self.stage = 1 if self.epoch_step < self.epochs_stage_1 else 2
        while self.epoch_step < self.epochs_stage_1:
            # For each batch in the dataloader
            # Stage 1 生成新的点
            for _, data in enumerate(self.dataloader_stage_1):
                # data [4,5000,3]
                self.update_learning_rate(self.epoch_step, self.epochs_stage_1, self.epochs_stage_2)
                (
                    sample,
                    sample_time,
                    sample_near,
                    time,
                    res,
                ) = (
                    data["sample"].to(self.device),
                    data["sample_time"].to(self.device),
                    data["sample_near"].to(self.device),
                    data["time"].to(self.device),
                    data["res"].to(self.device),
                )
                # sample_gaussian_moved
                sample_time.requires_grad = True
                pred_gradients = self.udf_network.gradient(sample_time, time)  # 4*5000x3
                pred_res_udf = self.udf_network.res_udf(sample_time, time)  # 4*5000x1
                pred_grad_norm = F.normalize(pred_gradients, dim=-1)  # 4*5000x3
                pred_res = pred_res_udf * pred_grad_norm
                loss_res = F.l1_loss(pred_res, res)
                losses = {
                    "res": loss_res,
                }

                # Calculate total loss
                total_loss = sum(losses.values())
                losses["total"] = total_loss

                # Log losses to tensorboard
                for loss_name, loss_value in losses.items():
                    self.writer.add_scalar(f",Loss/{loss_name}", loss_value.item(), self.epoch_step)

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
            self.epoch_step += 1
            if self.epoch_step % self.report_freq == 0:
                for loss_name, loss_value in losses.items():
                    self.writer.add_scalar(f"Loss/{loss_name}", loss_value.item(), self.epoch_step)
                # Create dynamic loss string from dictionary
                loss_str = ", ".join([f"loss:{name} = {value:.6e}" for name, value in losses.items()])
                print_log(
                    "stage-1{}_{} epoch:{:8>d} {} lr={:.6e}".format(
                        self.dataname,
                        self.time_sum,
                        self.epoch_step,
                        loss_str,
                        self.optimizer.param_groups[0]["lr"],
                    ),
                    logger=logger,
                )
        self.save_checkpoint()
        if self.epoch_step == self.epochs_stage_1:
            self.stage = 2
            print_log("进入第二阶段训练", logger=self.logger)
            # 先生成过量点云
            extra_points, extra_normals = self.get_extra_points(
                int(self.extra_points_rate * self.GT_points.shape[0]),
                self.noise_range,
            )
            # 基于泊松盘采集均匀的点云
            # idx = pcu.downsample_point_cloud_poisson_disk(
            #     all_extra_points,
            #     num_samples=int(self.extra_points_rate * self.GT_points.shape[0]),
            # )
            # extra_points = all_extra_points[idx]
            # extra_normals = all_extra_normals[idx]
            (estimate_error_pointcloud, estimate_error_stats) = self.dataset_stage_2.get_new_pointcloud(
                extra_points, extra_normals
            )
            estimate_error_pointcloud_path = (
                self.base_exp_dir / "pointcloud" / f"estimate_error_cloud{self.epoch_step}_{self.time_sum}epoch.ply"
            )
            # 创建点云对象
            estimate_error_pointcloud.export(estimate_error_pointcloud_path)
            print_log(
                f"Estimate_Error_Point cloud saved successfully at epoch {self.epoch_step}. File path: {estimate_error_pointcloud_path}",
                logger=self.logger,
            )
            print_log("estimate_Normal Error Stats:")
            print_log(f"Mean={estimate_error_stats['mean']:.6f}", logger=self.logger)
            print_log(f"Median={estimate_error_stats['median']:.6f}", logger=self.logger)
            print_log(f"Std={estimate_error_stats['std']:.6f}", logger=self.logger)
            print_log(f"Max={estimate_error_stats['max']:.6f}", logger=self.logger)
            new_point_cloud = trimesh.Trimesh(
                vertices=self.dataset_stage_2.points, vertex_normals=self.dataset_stage_2.normals
            )
            new_point_cloud_path = (
                self.base_exp_dir / "pointcloud" / f"new_point_cloud{self.epoch_step}_{self.time_sum}epoch.ply"
            )
            new_point_cloud.export(new_point_cloud_path)
            print_log(
                f"new_Point cloud saved successfully at epoch {self.epoch_step}. File path: {new_point_cloud_path}",
                logger=self.logger,
            )
            self.epoch_step += 1
            self.save_checkpoint()
            self.extract_mesh(
                resolution=args.mcube_resolution,
                threshold=0.005,
                point_gt=self.GT_points,
                epoch_step=self.epoch_step,
                time_sum=self.time_sum,
            )
            self.evaluate()

        while self.epoch_step < self.epochs:
            # data [4,5000,3]
            for _, data in enumerate(self.dataloader_stage_2):
                self.update_learning_rate(self.epoch_step, self.epochs_stage_1, self.epochs_stage_2)
                (
                    sample,  # noqa: F841
                    sample_time,
                    sample_near,  # noqa: F841
                    time,
                    res,
                ) = (
                    data["sample"].to(self.device),
                    data["sample_time"].to(self.device),
                    data["sample_near"].to(self.device),
                    data["time"].to(self.device),
                    data["res"].to(self.device),
                )
                # sample_gaussian_moved
                sample_time.requires_grad = True
                pred_gradients = self.udf_network.gradient(sample_time, time)  # 4*5000x3
                pred_res_udf = self.udf_network.res_udf(sample_time, time)  # 4*5000x1
                pred_grad_norm = F.normalize(pred_gradients, dim=-1)  # 4*5000x3
                pred_res = pred_res_udf * pred_grad_norm
                loss_res = F.l1_loss(pred_res, res)
                losses = {
                    "res": loss_res,
                }

                # Calculate total loss
                total_loss = sum(losses.values())
                losses["total"] = total_loss

                # Log losses to tensorboard
                for loss_name, loss_value in losses.items():
                    self.writer.add_scalar(f",Loss/{loss_name}", loss_value.item(), self.epoch_step)

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
            self.epoch_step += 1
            if self.epoch_step % self.report_freq == 0:
                for loss_name, loss_value in losses.items():
                    self.writer.add_scalar(f"Loss/{loss_name}", loss_value.item(), self.epoch_step)
                # Create dynamic loss string from dictionary
                loss_str = ", ".join([f"loss:{name} = {value:.6e}" for name, value in losses.items()])
                print_log(
                    "stage-2{}_{} epoch:{:8>d} {} lr={:.6e}".format(
                        self.dataname,
                        self.time_sum,
                        self.epoch_step,
                        loss_str,
                        self.optimizer.param_groups[0]["lr"],
                    ),
                    logger=logger,
                )
        self.save_checkpoint()
        self.gen_cube_pointcloud(self.epoch_step, self.time_sum)

        self.extract_mesh(
            resolution=args.mcube_resolution,
            threshold=0.005,
            point_gt=self.GT_points,
            epoch_step=self.epoch_step,
            time_sum=self.time_sum,
        )
        self.evaluate()
        parse_log_to_table(log_file)

        save_all_slice_views(
            field_func=lambda pts: self.udf_network.predict(pts, self.time_sum, self.metric)[0],
            pred_mesh_prefix=self.base_exp_dir / "mesh" / f"{self.epoch_step}_{self.time_sum}epoch_mesh",
            out_root=self.base_exp_dir / "slice",
            iter_step=self.epoch_step,
            device=self.device,
            field_type="auto",  # ② 自动识别 sdf/udf
            dense_threshold=0.02,  # 高密度区阈值
            dense_step=0.005,  # 高密度区步长
            sparse_step=0.02,  # 低密度区步长
            manual_flip_xyz=None,
            auto_flip_th=0.02,  # |SDF|/|UDF| < 0.02 认为“在面上”
            auto_flip_pct=0.6,  # 60 % 点合格就采纳
            auto_flip_n=300,  # 每轴采 300 点
            extent_ratio=1.05,  # 包围盒大小放大或缩小的比例,1为和原本相同
            extra_planes=[0],  # 当x,y,z=数组中的值时,必定输出相应的截面
        )

    def extract_mesh(
        self,
        resolution=64,
        threshold=0.005,
        point_gt=None,
        epoch_step=0,
        time_sum=0,
    ):
        bound_min = torch.tensor(self.dataset.bbox[0], dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.bbox[1], dtype=torch.float32)
        out_dir = self.base_exp_dir / "mesh"
        out_dir.mkdir(parents=True, exist_ok=True)

        mesh = extract_geometry(
            bound_min,
            bound_max,
            resolution=resolution,
            threshold=threshold,
            query_func=lambda pts: self.udf_network.predict(pts, self.time_sum, self.metric)[0],
            grad_func=lambda pts: self.udf_network.predict(pts, self.time_sum, self.metric)[1],
        )
        if self.conf.get_float("train.far") > 0:
            mesh = remove_far(point_gt.detach().cpu().numpy(), mesh, self.conf.get_float("train.far"))

        mesh_path = out_dir / f"{epoch_step}_{time_sum}epoch_mesh.obj"
        mesh.export(mesh_path)

        print_log(
            f"Mesh saved successfully at epoch {epoch_step}_{time_sum}epoch. File path: {mesh_path}",
            logger=self.logger,
        )

    def gen_cube_pointcloud(self, epoch_step, time_sum):
        device = self.device  # 使用的GPU设备
        points_batch_size = 100000
        resolution = 64  # 网格分辨率
        bounding_box = 1.1  # 边界框大小

        # 创建三维网格点
        p = torch.linspace(-bounding_box / 2, bounding_box / 2, resolution)
        px, py, pz = torch.meshgrid([p, p, p])
        points = torch.stack([px, py, pz], 3)
        points = points.view(-1, 3)  # 展平为点列表
        p_split = torch.split(points, points_batch_size)  # 按批次分割点
        perd_points_all = []  # 存储预测位移点
        # 遍历每批次点,计算预测的距离和位移
        for pi in p_split:
            samples = pi.clone().to(device)
            # 将点移动到设备
            # samples = pi.clone().to(device)
            _, _, perd_points = self.udf_network.predict(samples, self.time_sum, self.metric)  # 预测位移
            perd_points_all.append(perd_points.squeeze(0).detach().cpu())

        # 合并所有批次的结果
        perd_points_all = torch.cat(perd_points_all, dim=0)

        # 将点转换为numpy数组
        perd_points_np = perd_points_all.cpu().numpy()

        cube_pointcloud_path = self.base_exp_dir / "pointcloud" / f"cube_point_cloud{epoch_step}_{time_sum}epoch.ply"

        # 创建点云对象
        trimesh.Trimesh(vertices=perd_points_np, process=False).export(cube_pointcloud_path)
        print_log(
            f"Cube_Point cloud saved successfully at epoch {epoch_step}. File path: {cube_pointcloud_path}",
            logger=self.logger,
        )

    # def update_learning_rate(self, epoch_step):
    #     warn_up = self.warm_up_end
    #     max_epoch = self.epochs
    #     init_lr = self.learning_rate
    #     lr = (
    #         (epoch_step / warn_up)
    #         if epoch_step < warn_up
    #         else 0.5 * (math.cos((epoch_step - warn_up) / (max_epoch - warn_up) * math.pi) + 1)
    #     )
    #     lr = lr * init_lr

    #     for g in self.optimizer.param_groups:
    #         g["lr"] = lr
    def update_learning_rate(self, epoch_step, epochs_stage_1, epochs_stage_2):
        """
        多阶段学习率调度:
        - 阶段1: 从 epoch_step=0 到 epoch_step=epochs_stage_1-1
            warmup -> 余弦下降
            初始学习率: self.learning_rate_stage1
        - 阶段2: 从 epoch_step=epochs_stage_1 到 epochs_stage_1+epochs_stage_2-1
            warmup -> 余弦下降
            初始学习率: self.learning_rate_stage2

        参数
        ----
        epoch_step: int
            当前的全局 epoch 序号,比如第几轮训练了
        epochs_stage_1: int
            第一阶段的总 epoch 数
        epochs_stage_2: int
            第二阶段的总 epoch 数
        """

        # ====== 配置两个阶段的超参数 ======
        # 每个阶段各自的初始lr
        lr_stage1 = self.learning_rate_stage1  # 需要在 __init__ 里定义
        lr_stage2 = self.learning_rate_stage2  # 需要在 __init__ 里定义

        # 每个阶段各自的 warmup 轮数
        warmup_stage1 = min(self.warm_up_end_stage1, epochs_stage_1)
        warmup_stage2 = min(self.warm_up_end_stage2, epochs_stage_2)

        # ====== 判定当前属于哪个阶段 ======
        if epoch_step < epochs_stage_1:
            # ------- 阶段 1 -------
            local_epoch = epoch_step  # 阶段内的第几轮
            max_epoch_this_stage = epochs_stage_1
            warmup_this_stage = warmup_stage1
            base_lr = lr_stage1

        else:
            # ------- 阶段 2 -------
            local_epoch = epoch_step - epochs_stage_1  # 把计数重置到该阶段起点
            max_epoch_this_stage = epochs_stage_2
            warmup_this_stage = warmup_stage2
            base_lr = lr_stage2

        # ====== 计算该阶段内的学习率 ======
        if local_epoch < warmup_this_stage and warmup_this_stage > 0:
            # 线性 warmup: 从 0 -> base_lr
            lr_factor = float(local_epoch) / float(warmup_this_stage)
        else:
            # 余弦退火:
            # local_epoch 从 warmup_this_stage ... max_epoch_this_stage-1
            # 先把它平移到从0开始计
            t = local_epoch - warmup_this_stage
            T = max_epoch_this_stage - warmup_this_stage
            # 防止除0 (比如没有warmup 或阶段长度==warmup长度)
            if T <= 0:
                lr_factor = 1.0
            else:
                # 0.5 * (1 + cos(pi * t / T))
                lr_factor = 0.5 * (1.0 + math.cos(math.pi * t / T))

        lr = base_lr * lr_factor

        # ====== 写回 optimizer ======
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def load_checkpoint(self, checkpoint_name):
        checkpoint_path = self.base_exp_dir / "checkpoints" / checkpoint_name
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )
        print(checkpoint_path)
        self.stage = checkpoint["stage"]
        if self.stage == 1:
            self.dataset = self.dataset_stage_1
        else:
            self.dataset = self.dataset_stage_2
        self.udf_network.load_state_dict(checkpoint["udf_network_fine"])
        self.dataset.set_point_cloud(checkpoint["points"], checkpoint["normals"])
        self.epoch_step = checkpoint["epoch_step"]
        print_log(
            f"Checkpoint loaded successfully at epoch {self.epoch_step}.",
            logger=self.logger,
        )

    def save_checkpoint(self):
        if self.stage == 1:
            self.dataset = self.dataset_stage_1
        else:
            self.dataset = self.dataset_stage_2
        checkpoint = {
            "udf_network_fine": self.udf_network.state_dict(),
            "epoch_step": self.epoch_step,
            "points": self.dataset.points,
            "normals": self.dataset.normals,
            "stage": self.stage,
        }
        checkpoint_dir = self.base_exp_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            checkpoint,
            checkpoint_dir / f"ckpt_{self.epoch_step:0>6d}_{self.time_sum}epoch.pth",
        )
        print_log(
            f"Checkpoint saved successfully at epoch {self.epoch_step}.",
            logger=self.logger,
        )

    def get_extra_points(self, extra_points, noise_range):
        batch_size = 10000
        all_extra_points = []
        all_extra_normals = []
        num_collected_points = 0
        # 计算当前批次需要生成的点数
        # 确保总数不会超过 target_points
        while num_collected_points < extra_points:
            current_batch_size = min(batch_size, extra_points - num_collected_points)
            current_indices = np.random.choice(self.dataset_stage_1.points.shape[0], current_batch_size, replace=True)
            # Gaussian noise points
            sample_surface = self.dataset_stage_1.points[current_indices]
            sample_sigmas = self.dataset_stage_1.sigmas[current_indices]
            theta_guassian = 0.25 * noise_range
            noise = np.random.normal(0.0, 1.0, size=(current_indices.shape[0], 3)).astype(np.float32)
            sample = sample_surface + theta_guassian * sample_sigmas * noise
            sample = torch.from_numpy(sample).to(self.device).float()
            sample.requires_grad = True
            _, sample_normal, sample_near = self.udf_network.predict(sample, self.time_sum, self.metric)
            sample_near = sample_near.detach().cpu().numpy()
            sample_normal = sample_normal.detach().cpu().numpy()
            gt_kd_tree = self.dataset_stage_1.kd_tree
            distances, _ = gt_kd_tree.query(sample_near, p=2, distance_upper_bound=0.003)
            sample_near = sample_near[distances < 0.003]
            sample_normal = sample_normal[distances < 0.003]
            all_extra_points.append(sample_near)
            all_extra_normals.append(sample_normal)
            num_collected_points += sample_near.shape[0]
        all_extra_points = np.concatenate(all_extra_points, axis=0)
        all_extra_normals = np.concatenate(all_extra_normals, axis=0)
        return all_extra_points, all_extra_normals

    def evaluate(self, sample_num=100000):
        data_dir = Path(self.conf.get_string("general.data_dir"))
        gt_base = data_dir / self.dir / "ground_truth" / self.dataname
        if (gt_base.with_suffix(".ply")).is_file():
            gt_file = gt_base.with_suffix(".ply")
        elif (gt_base.with_suffix(".obj")).is_file():
            gt_file = gt_base.with_suffix(".obj")
        elif (gt_base.with_suffix(".xyz")).is_file():
            gt_file = gt_base.with_suffix(".xyz")
        else:
            raise FileNotFoundError(f"找不到 ground truth 文件: {gt_base}.[ply|obj|xyz]")
        pred_file = self.base_exp_dir / "mesh" / f"{self.epoch_step}_{self.time_sum}epoch_mesh.obj"
        print_log(f"pred_file:{pred_file}", logger=self.logger)
        print_log(f"gt_file:{gt_file}", logger=self.logger)
        gt_mesh = trimesh.load_mesh(gt_file)
        pred_mesh = trimesh.load_mesh(pred_file)
        gt_mesh = normalize_mesh(gt_mesh)
        total_size = (gt_mesh.bounds[1] - gt_mesh.bounds[0]).max()
        centers = (gt_mesh.bounds[1] + gt_mesh.bounds[0]) / 2
        pred_mesh.apply_scale(total_size)
        pred_mesh.apply_translation(centers)

        # pred_mesh = normalize_mesh(pred_mesh)

        # sample point for rec
        pts_pred, idx = pred_mesh.sample(sample_num, return_index=True)
        normals_rec = pred_mesh.face_normals[idx]
        # sample point for gt
        pts_gt = None
        normals_gt = None
        if isinstance(gt_mesh, trimesh.PointCloud):
            sample_num = min(sample_num, gt_mesh.vertices.shape[0])
            idx = np.random.choice(gt_mesh.vertices.shape[0], sample_num, replace=False)
            pts_gt = gt_mesh.vertices[idx]
            normals_gt = None
        elif isinstance(gt_mesh, trimesh.Trimesh):
            pts_gt, idx = gt_mesh.sample(sample_num, return_index=True)
            normals_gt = gt_mesh.face_normals[idx]
        elif isinstance(gt_mesh, trimesh.Scene):
            # 多个子网格,合并为一个大网格再采样
            combined = gt_mesh.to_geometry()  # 返回一个合并的 Trimesh 对象
            pts_gt, idx = combined.sample(sample_num, return_index=True)
            pts_gt = pts_gt.astype(np.float32)
            normals_gt = combined.face_normals[idx]

        (
            normals_correctness,
            chamferL1_mean,
            chamferL1_median,
            chamferL2_mean,
            chamferL2_median,
            f_score_001,
            f_score_0005,
        ) = eval_pointcloud(pts_pred, pts_gt, normals_rec, normals_gt)

        print_log(f"dataset:{self.dir}", logger=self.logger)
        print_log(f"dataname:{self.dataname}", logger=self.logger)
        print_log(f"time_sum:{self.time_sum}", logger=self.logger)
        print_log(
            f"normals_correctness:{normals_correctness * 100:.4g}%",
            logger=self.logger,
        )
        print_log(f"chamferL1_mean:{chamferL1_mean * 1000:.4e}", logger=self.logger)
        print_log(
            f"chamferL1_median:{chamferL1_median * 1000:.4e}",
            logger=self.logger,
        )
        print_log(f"chamferL2_mean:{chamferL2_mean * 1000:.4e}", logger=self.logger)
        print_log(
            f"chamferL2_median:{chamferL2_median * 1000:.4e}",
            logger=self.logger,
        )
        print_log(f"F_score_0.01:{f_score_001 * 100:.4g}%", logger=self.logger)
        print_log(f"F_score_0.005:{f_score_0005 * 100:.4g}%", logger=self.logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="./confs/ndf.conf")
    parser.add_argument("--mcube_resolution", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dir", type=str, default="test")
    parser.add_argument("--dataname", type=str, default="demo")
    args = parser.parse_args()

    runner = Runner(args, args.conf)

    runner.train()
