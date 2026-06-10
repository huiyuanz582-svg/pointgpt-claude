import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

import numpy as np
import psutil
import sys
import gc
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from torchvision import transforms

import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os
from extensions.chamfer_dist import ChamferDistanceL1,ChamferDistanceL2
import pytorch3d
import pytorch3d.loss
from utils.p2m_loss import compute_p2m, compute_p2m_train

train_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudTranslate(),
        # data_transforms.PointcloudJitter(),
        # data_transforms.PointcloudRandomInputDropout(),
        # data_transforms.RandomHorizontalFlip(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

test_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudTranslate(),
        # data_transforms.PointcloudScaleAndTranslate(),
    ]
)


class DenoiseMetrics:
    def __init__(self, cd=0.0, p2m=0.0):
        self.cd = cd    # Chamfer Distance
        self.p2m = p2m  # Point-to-Mesh Distance

    def better_than(self, other):
        if other.cd == 0.0:
            return True
        # CD 和 P2M 均衡评判（CD~1.7, P2M~5~7，量级差约 3 倍，系数 0.3 使两者贡献相当）
        score_self  = self.cd  + 0.3 * self.p2m
        score_other = other.cd + 0.3 * other.p2m
        return score_self < score_other

    def state_dict(self):
        _dict = dict()
        _dict['cd'] = self.cd
        _dict['p2m'] = self.p2m
        return _dict

def load_off(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    assert lines[0].strip() == 'OFF'
    n_verts, n_faces, _ = map(int, lines[1].split())

    verts = np.array(
        [list(map(float, lines[i + 2].split()))
         for i in range(n_verts)],
        dtype=np.float32
    )

    faces = np.array(
        [list(map(int, lines[i + 2 + n_verts].split()[1:4]))
         for i in range(n_faces)],
        dtype=np.int64
    )

    return verts, faces

_mesh_normal_cache = {}  # {(name, split): (face_normals, face_centers_norm)}

def compute_mesh_normals_for_pcl(pcl_clean, name, split='train'):
    """
    对 clean 点云中每个点，查询最近 mesh 面的法向。
    pcl_clean: [N, 3] tensor（已归一化到单位球）
    返回: normals [N, 3] tensor（与 pcl_clean 同设备）
    """
    cache_key = (name, split)
    if cache_key not in _mesh_normal_cache:
        off_path = os.path.join(
            'data/ScoreDenoise/PUNet/meshes', split, f'{name}.off')
        verts, faces = load_off(off_path)
        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0).astype(np.float32)
        norms = np.linalg.norm(cross, axis=1, keepdims=True).clip(min=1e-8)
        face_normals = cross / norms                                  # [F, 3]

        face_centers = (v0 + v1 + v2) / 3.0                          # [F, 3]
        fc_center = (face_centers.min(axis=0) + face_centers.max(axis=0)) / 2
        fc_scale = np.linalg.norm(face_centers - fc_center, axis=1).max().clip(min=1e-8)
        face_centers_norm = (face_centers - fc_center) / fc_scale     # [F, 3]

        _mesh_normal_cache[cache_key] = (face_normals, face_centers_norm)

    face_normals, face_centers_norm = _mesh_normal_cache[cache_key]
    pts_np = pcl_clean.detach().cpu().numpy()                         # [N, 3]

    from sklearn.neighbors import NearestNeighbors
    nn_model = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(face_centers_norm)
    _, face_idx = nn_model.kneighbors(pts_np)                         # [N, 1]
    face_idx = face_idx[:, 0]

    normals_np = face_normals[face_idx].astype(np.float32)            # [N, 3]
    return torch.from_numpy(normals_np).to(pcl_clean.device)


def normalize_unit_sphere(pc, radius=1.0):
    """
    Args:
        pc: A batch of point clouds, (B, N, 3).
    """
    ## Center
    p_max = pc.max(dim=-2, keepdim=True)[0]
    p_min = pc.min(dim=-2, keepdim=True)[0]
    center = (p_max + p_min) / 2    # (B, 1, 3)
    pc = pc - center
    ## Scale
    scale = (pc ** 2).sum(dim=-1, keepdim=True).sqrt().max(dim=-2, keepdim=True)[0] / radius  # (B, N, 1)
    pc = pc / scale
    return pc, center, scale

def normalize_pcl(pc, center, scale):
    return (pc - center) / scale


def sor_filter(pts_tensor, nb_neighbors=20, std_ratio=2.0):
    """Statistical Outlier Removal，输入输出均为 [N, 3] CPU tensor"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_tensor.cpu().numpy())
    pcd_filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    filtered = torch.from_numpy(np.asarray(pcd_filtered.points)).float()
    return filtered


@torch.no_grad()
def patch_based_denoise(base_model, pcl_noisy, noise_std_t, patch_size=1024,
                        seed_ratio=3, patch_batch=4,
                        num_steps=1, step_size=1.0, decay=0.95):
    """
    完整点云的 patch-based 推理：把 N 点云切成覆盖全部点的重叠 1024-patch，
    逐 patch 去噪后按覆盖次数平均拼回完整点云。

    - pcl_noisy:   [N, 3]，归一化空间的完整噪声点云
    - noise_std_t: [1] 或标量，该点云的噪声水平 σ
    - 与训练一致：每个 patch 是 1024 点，喂给 model 后内部 group_divider 覆盖 200%
    - seed_ratio:  FPS 种子数 = seed_ratio * N / patch_size，控制重叠/覆盖冗余

    多步 Langevin 退火推理（num_steps>1 时启用）：
        x ← x_noisy
        for t in range(num_steps):
            ε   = model(x, σ_t)               # 每步重新估计
            x   ← x + step_size · σ_t · ε     # 走一小步
            σ_t ← σ_t · decay                 # 逐步退火
    - num_steps=1, step_size=1.0 时退化为原单步 Tweedie：x̂ = x + σ·ε（fallback）
    - patch 内部做迭代（group_divider 是逐 patch 工作的），最后再拼回

    返回 [N, 3] 去噪后完整点云。
    """
    device = pcl_noisy.device
    N = pcl_noisy.shape[0]
    patch_size = min(patch_size, N)

    # FPS 选种子点，数量足够让 patch 并集覆盖所有点
    num_seeds = max(1, int(seed_ratio * N / patch_size))
    seeds = misc.fps(pcl_noisy.unsqueeze(0), num_seeds)[0]          # [S, 3]

    # 每个种子 KNN 取 patch_size 个点（用 noisy 空间距离，与训练切 patch 一致）
    dist = torch.cdist(seeds, pcl_noisy)                            # [S, N]
    patch_idx = dist.topk(patch_size, dim=1, largest=False).indices  # [S, patch_size]
    patches = pcl_noisy[patch_idx]                                  # [S, patch_size, 3]

    sigma0 = noise_std_t.view(-1)[0] if torch.is_tensor(noise_std_t) else float(noise_std_t)
    sigma0 = float(sigma0)

    # 分批喂模型去噪
    denoised_patches = []
    for i in range(0, patches.shape[0], patch_batch):
        # 每个 patch 批次前硬保护：逼近上限就主动退出，绝不让服务器崩
        if i > 0 and gpu_mem_ratio() > 0.90:
            torch.cuda.empty_cache()
            if gpu_mem_ratio() > 0.90:
                print('[测试内存保护] patch 推理中 GPU 显存逼近上限，为防崩溃主动退出；'
                      '请在 yaml 调小 test_patch_batch 后重跑')
                sys.exit(0)
        x = patches[i:i + patch_batch]                            # [b, patch_size, 3]
        sigma_t = sigma0
        for _ in range(num_steps):
            ns = torch.full((x.shape[0],), float(sigma_t), device=device)
            # 'val' 路径返回 x + σ_t·ε，反解出 ε 再按 step_size 走一小步
            out = base_model(x, None, 'val', '', noise_std=ns)    # [b, patch_size, 3]
            eps = (out - x) / sigma_t                             # 估计的 ε
            x = x + step_size * sigma_t * eps                     # Langevin 退火步
            sigma_t = sigma_t * decay                             # σ 逐步退火
        denoised_patches.append(x.detach())
        del x, out, eps, ns
    denoised_patches = torch.cat(denoised_patches, dim=0)          # [S, patch_size, 3]

    # 按覆盖次数平均拼回完整点云
    accum = torch.zeros(N, 3, device=device)
    count = torch.zeros(N, 1, device=device)
    flat_idx = patch_idx.reshape(-1)                               # [S*patch_size]
    accum.index_add_(0, flat_idx, denoised_patches.reshape(-1, 3))
    count.index_add_(0, flat_idx, torch.ones(flat_idx.shape[0], 1, device=device))
    # 未被任何 patch 覆盖的点回退为原始 noisy（理论上 seed_ratio>=1 时不会发生）
    uncovered = (count < 0.5).squeeze(-1)
    denoised = accum / count.clamp(min=1.0)
    denoised[uncovered] = pcl_noisy[uncovered]
    return denoised


@torch.no_grad()
def local_surface_projection(pcl, k=16, num_iters=1, blend=1.0, chunk=4096):
    """
    局部表面投影后处理：对每个点用 k-NN 做局部 PCA，估计局部表面法向，
    把点沿法向投影到由邻域质心确定的局部切平面上，压低点到真实表面的距离(P2M)。

    - pcl:       [N, 3]，去噪后点云（归一化空间）
    - k:         邻域点数
    - num_iters: 迭代次数（多次迭代收敛到更平滑表面）
    - blend:     投影强度 [0,1]，1=完全投影，<1=部分投影（保细节）
    - chunk:     分块大小，控制显存

    返回 [N, 3] 投影后点云。
    """
    device = pcl.device
    N = pcl.shape[0]
    x = pcl.clone()
    for _ in range(num_iters):
        new_x = torch.empty_like(x)
        for s in range(0, N, chunk):
            q = x[s:s + chunk]                                   # [c, 3]
            d = torch.cdist(q, x)                                # [c, N]
            knn_idx = d.topk(k, dim=1, largest=False).indices    # [c, k]
            nb = x[knn_idx]                                      # [c, k, 3]
            centroid = nb.mean(dim=1, keepdim=True)              # [c, 1, 3]
            cen = nb - centroid                                  # [c, k, 3]
            cov = cen.transpose(1, 2) @ cen / k                  # [c, 3, 3]
            # 最小特征向量 = 局部表面法向
            evals, evecs = torch.linalg.eigh(cov)                # 升序
            normal = evecs[:, :, 0]                              # [c, 3]
            # 把点投影到过质心、法向为 normal 的平面：x' = x - (n·(x-c))·n
            diff = (q - centroid.squeeze(1))                     # [c, 3]
            dist_along_n = (diff * normal).sum(dim=1, keepdim=True)  # [c, 1]
            proj = q - blend * dist_along_n * normal             # [c, 3]
            new_x[s:s + chunk] = proj
        x = new_x
    return x


def check_memory_and_exit(base_model, optimizer, epoch, metrics, best_metrics, args, logger,
                          threshold_gpu=0.72, threshold_cpu=78.0):
    """检查 GPU 显存和 CPU 内存，超阈值时保存检查点并主动退出，防止服务器崩溃

    实测（A100-80GB）：CPU 125GB 仅占 15GB 完全不是瓶颈，真正风险是 GPU 显存尖峰
    （L 模型验证阶段一度冲到 57GB ≈ 71%）。故 GPU 阈值收紧到 0.72，在尖峰逼近时
    及时保存退出，避免 bs 偏大时冲破 80GB 导致 GPU OOM / 服务器卡死。
    本函数在 epoch 内每 mem_check_interval 个 batch、以及验证前后被调用。
    """
    should_exit = False
    reason = ''

    if torch.cuda.is_available():
        # 用整卡已用/总量来判断，与 nvidia-smi 显示一致
        import subprocess
        try:
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total',
                 '--format=csv,noheader,nounits', '--id=0'],
                encoding='utf-8'
            ).strip().split(',')
            used_mb = int(result[0].strip())
            total_mb = int(result[1].strip())
            gpu_ratio = used_mb / total_mb
            
        except Exception:
            # fallback：用 PyTorch 自身统计
            reserved = torch.cuda.memory_reserved(0)
            total = torch.cuda.get_device_properties(0).total_memory
            gpu_ratio = reserved / total
        if gpu_ratio > threshold_gpu:
            reason = f'GPU 显存使用率 {gpu_ratio:.1%} 超过阈值 {threshold_gpu:.0%}'
            should_exit = True

    if not should_exit:
        cpu_percent = psutil.virtual_memory().percent
        if cpu_percent > threshold_cpu:
            reason = f'CPU 内存使用率 {cpu_percent:.1f}% 超过阈值 {threshold_cpu:.0f}%'
            should_exit = True

    if should_exit:
        print_log(f'[内存保护] {reason}，保存检查点后主动退出', logger=logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)
        sys.exit(0)


def gpu_mem_ratio():
    """返回当前 GPU(0) 显存使用率 [0,1]，读取失败返回 0.0。仅用于测试路径软监控（不退出）"""
    if not torch.cuda.is_available():
        return 0.0
    import subprocess
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total',
             '--format=csv,noheader,nounits', '--id=0'],
            encoding='utf-8'
        ).strip().split(',')
        return int(result[0].strip()) / int(result[1].strip())
    except Exception:
        try:
            reserved = torch.cuda.memory_reserved(0)
            total = torch.cuda.get_device_properties(0).total_memory
            return reserved / total
        except Exception:
            return 0.0


def run_net(args, config, train_writer=None, val_writer=None):

    logger = get_logger(args.log_name)

    # 1. 构建数据集
    train_sampler,train_dataloader = builder.dataset_builder(args, config.dataset.train)
    _,test_dataloader  = builder.dataset_builder(args, config.dataset.val)  
    

    # 2. 构建模型
    base_model = builder.model_builder(config.model)

    # Score-based 去噪：encoder 不冻结
    # 理由：预训练 encoder 学的是"为自回归生成 patch 内点"的特征，跟"估计 score 场"的目标不完全对齐
    # 整体微调（小 lr）让 encoder 适配 score 任务，比纯冻结好

    # parameter setting
    start_epoch = 0
    best_metrics = DenoiseMetrics(0.0)
    best_metrics_vote = DenoiseMetrics(0.0)
    metrics = DenoiseMetrics(0.0)

    # 加载预训练
    if args.ckpts is not None:
        base_model.load_model_from_ckpt(args.ckpts)
        # Score-based 关键步骤：重新初始化 generator 输出头
        # 预训练的输出头学的是"生成绝对坐标"，直接用会让 noisy + σ·ε 大幅漂移
        # ε-prediction target 的每点 norm ≈ √3 ≈ 1.7（标准正态）；增益头经过 ln_f 后
        # 输出幅度 ≈ std·√fan_in ≈ std·√384。std=0.1 → 起点 ≈ 2.0，与 target 量级对齐，
        # 避免之前 std=0.01（起点 ≈ 0.2，仅 target 的 ~12%）导致点几乎不动、幅度学不上来。
        with torch.no_grad():
            nn.init.normal_(base_model.generator_blocks.increase_dim[0].weight, std=0.1)
            if base_model.generator_blocks.increase_dim[0].bias is not None:
                nn.init.zeros_(base_model.generator_blocks.increase_dim[0].bias)
        print_log('[ε-prediction] Re-initialized generator output head (std=0.1)', logger=logger)
    else:
        print_log('Training from scratch', logger=logger)
    
    if args.use_gpu:
        base_model.to(args.local_rank)
    
    # DDP 多GPU
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(
            base_model, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    # trainval
    # training
    base_model.zero_grad()
    mesh_surface_cache = {}
    # P2M 训练 loss 权重（0 = 关闭，纯 ε-prediction）；从 config.p2m_weight 读
    p2m_weight = float(getattr(config, 'p2m_weight', 0.0))
    if p2m_weight > 0:
        print_log(f'[Training] P2M loss 已启用，权重 = {p2m_weight}', logger=logger)
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        # 计算平均值
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss'])
        p2m_meter = AverageMeter(['p2m'])

        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader) #375
        npoints = config.npoints #2048
       

        for idx, (pcl_noisy, pcl_clean, noise_std, center, scale, name) in enumerate(train_dataloader):

            num_iter += 1
            n_itr = epoch * n_batches + idx

            data_time.update(time.time() - batch_start_time)
            pcl_noisy = pcl_noisy.cuda()
            pcl_clean = pcl_clean.cuda()
            if noise_std is not None:
                noise_std_t = noise_std.cuda()
            else:
                # 训练时 PairedPatchDataset 始终提供 noise_std，走到这里是 bug
                raise RuntimeError('training noise_std is None — AddNoise transform 是否被意外移除？')

            loss1, denoised_norm = base_model(pcl_noisy, pcl_clean, 'train', name,
                               epoch=epoch, max_epoch=config.max_epoch,
                               noise_std=noise_std_t)
            _loss = loss1.mean()  # ε-MSE 标量

            # P2M 训练 loss（可微，单向 point→face）：把预测点还原到 mesh 世界坐标系再算
            if p2m_weight > 0:
                p2m_terms = []
                B = denoised_norm.shape[0]
                for b in range(B):
                    try:
                        c_b = center[b].to(denoised_norm.device).view(1, 3)
                        s_b = scale[b].to(denoised_norm.device).view(1, 1)
                        pred_world = denoised_norm[b] * s_b + c_b   # [N,3] 还原世界坐标
                        p2m_b = compute_p2m_train(pred_world, name[b], split='train')
                        p2m_terms.append(p2m_b)
                    except Exception as e:
                        if idx == 0 and b == 0:
                            print_log(f'[P2M] 跳过 (mesh 缺失或 pytorch3d 问题): {e}', logger=logger)
                if len(p2m_terms) > 0:
                    p2m_loss = torch.stack(p2m_terms).mean() * 1e4  # 量级对齐 test 的 P2M
                    _loss = _loss + p2m_weight * p2m_loss
                    p2m_meter.update([p2m_loss.item()])

            # loss, acc = base_model.module.get_loss_acc(ret, label)

            # _loss = loss + 3 * loss1 #loss下游任务的损失 loss1生成任务的损失

            try:
                _loss = _loss.mean()     # 永远先变成标量
                _loss.backward()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print_log('OOM，清理显存并跳过当前 batch', logger=logger)
                    del loss1, _loss
                    torch.cuda.empty_cache()
                    base_model.zero_grad()
                    continue
                else:
                    raise e



            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(
                        base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss1, args)
                losses.update([ loss.item()])
            else:
                try:
                    losses.update([loss1.item()])
                except:
                    losses.update([loss1.mean().item()])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar(
                    'Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            # epoch 内周期性内存检查：L 模型内存峰值在 epoch 中间，epoch 末尾才查会来不及救
            # 每 mem_check_interval 个 batch 查一次，超阈值立刻保存 ckpt-last 并退出
            mem_check_interval = int(getattr(config, 'mem_check_interval', 50))
            if mem_check_interval > 0 and idx > 0 and idx % mem_check_interval == 0:
                check_memory_and_exit(base_model, optimizer, epoch, best_metrics, best_metrics, args, logger)

            # if idx % 10 == 0:
            #     print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss+Acc = %s lr = %.6f' %
            #                 (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
            #                 ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)

        p2m_avg_str = (' P2M = %.4f' % p2m_meter.avg(0)) if (p2m_weight > 0 and p2m_meter.count(0) > 0) else ''
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s%s lr = %.6f' %
                  (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()], p2m_avg_str, optimizer.param_groups[0]['lr']), logger=logger)
    
        if epoch % args.val_freq == 0 and epoch != 0:
            # 验证前清缓存 + 查显存：验证阶段是 GPU 显存尖峰来源，跑之前先确保有余量
            torch.cuda.empty_cache()
            check_memory_and_exit(base_model, optimizer, epoch, best_metrics, best_metrics, args, logger)
            # Validate the current model
            metrics = validate(base_model, test_dataloader,
                               epoch, val_writer, args, config, logger=logger)

            better = metrics.better_than(best_metrics)
            # Save ckeckpoints
            if better:
                best_metrics = metrics
                builder.save_checkpoint(
                    base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger=logger)
                print_log(
                    "--------------------------------------------------------------------------------------------", logger=logger)
            
            # 这里是针对去噪任务的代码，使用cd距离来衡量性能
            if args.vote:
                # 如果 CD 小于阈值或比当前 best 更好
                if metrics.cd < 0.01 or (better and metrics.cd < 0.02):  # 这里 0.01/0.02 是示例阈值，可根据实际情况调整
                    metrics_vote = validate_vote(
                        base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)
                    if metrics_vote.better_than(best_metrics_vote):
                        best_metrics_vote = metrics_vote
                        print_log(
                            "****************************************************************************************",
                            logger=logger)
                        builder.save_checkpoint(
                            base_model, optimizer, epoch, metrics, best_metrics_vote, 'ckpt-best_vote', args, logger=logger)


        builder.save_checkpoint(base_model, optimizer, epoch,
                                metrics, best_metrics, 'ckpt-last', args, logger=logger)
        check_memory_and_exit(base_model, optimizer, epoch, metrics, best_metrics, args, logger)
        # if (config.max_epoch - epoch) < 10:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


# validate10k  ----------------------------------------------------------------------------------------
def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=None):
    base_model.eval()  # set model to eval mode
    npoints = config.npoints

    total_cd = 0.0
    total_cd_10k = 0.0
    total_batches = 0
    total_p2m = 0.0
    total_cd_noisy_baseline = 0.0

    with torch.no_grad():
        for idx, (pcl_noisy, pcl_clean, noise_std, center, scale, name) in enumerate(test_dataloader):
            pcl_noisy = pcl_noisy.cuda()
            pcl_clean = pcl_clean.cuda()
            if noise_std is not None:
                noise_std_t = noise_std.cuda()
            else:
                # PairedEvalDataset 不含 noise_std；从 config.TEST_NOISE fallback
                _test_sigma = getattr(config.dataset.test._base_, 'TEST_NOISE', 0.01)
                noise_std_t = torch.full(
                    (pcl_noisy.shape[0],), _test_sigma, dtype=torch.float32
                ).cuda()
            # val 用 1024-point patch（dataloader flag='train'），base_model 直接去噪，覆盖 200%
            denoised = base_model(pcl_noisy, pcl_clean, 'val', name, noise_std=noise_std_t)

            # 归一化空间 CD
            batch_cd = ChamferDistanceL2().cuda()(denoised, pcl_clean) * 1e4
            # noisy baseline CD（参考：去噪前 vs 去噪后）
            batch_cd_noisy = ChamferDistanceL2().cuda()(pcl_noisy, pcl_clean) * 1e4
            total_cd_noisy_baseline = total_cd_noisy_baseline + batch_cd_noisy

            # val 阶段只用 CD 监控（patch 无法对齐完整 mesh）；P2M 留到最终 test
            total_cd_10k += batch_cd
            total_batches += 1

    if args.distributed:
        total_cd_10k = dist_utils.gather_tensor(total_cd_10k.cuda() if torch.is_tensor(total_cd_10k) else torch.tensor(total_cd_10k).cuda(), args)
        total_batches = dist_utils.gather_tensor(torch.tensor(total_batches).cuda(), args)
        if args.local_rank != 0:
            return None

    avg_cd = total_cd_10k / total_batches
    avg_cd_noisy = total_cd_noisy_baseline / total_batches

    print_log('[Validation] EPOCH: %d  Chamfer Distance = %.6f  (Noisy baseline CD = %.6f, delta = %.6f)' %
        (epoch, avg_cd, avg_cd_noisy, avg_cd_noisy - avg_cd), logger=logger)

    if args.distributed:
        torch.cuda.synchronize()

    if val_writer is not None:
        val_writer.add_scalar('Metric/CD', avg_cd, epoch)

    # val 只用 CD 排 checkpoint（P2M 置 0，better_than 退化为按 CD 比较）
    return DenoiseMetrics(avg_cd, 0.0)

def validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=None, times=10):
    print_log(f"[VALIDATION_VOTE] epoch {epoch}", logger=logger)
    base_model.eval()  # set model to eval mode

    cd_list = []
    npoints = config.npoints

    with torch.no_grad():
        for idx, (pcl_noisy, pcl_clean, noise_std, center, scale, name) in enumerate(test_dataloader):
            pcl_noisy = pcl_noisy.cuda()
            pcl_clean = pcl_clean.cuda()
            if noise_std is not None:
                noise_std_t = noise_std.cuda()
            else:
                # PairedEvalDataset 不含 noise_std；从 config.TEST_NOISE fallback
                _test_sigma = getattr(config.dataset.test._base_, 'TEST_NOISE', 0.01)
                noise_std_t = torch.full(
                    (pcl_noisy.shape[0],), _test_sigma, dtype=torch.float32
                ).cuda()
            center_t = center[0].to(pcl_noisy.device)
            scale_t = scale[0].to(pcl_noisy.device)

            local_cd = []
            for _ in range(times):
                denoised = base_model(pcl_noisy, pcl_clean, 'val', name, noise_std=noise_std_t)
                denoised_filtered = sor_filter(denoised[0])
                denoised = denoised_filtered.unsqueeze(0).to(pcl_noisy.device)
                denoised_world = denoised * scale_t + center_t
                clean_world = pcl_clean * scale_t + center_t

                _, center1, scale1 = normalize_unit_sphere(clean_world)
                pcl_clean_norm = normalize_pcl(clean_world, center1, scale1)
                denoised_norm = normalize_pcl(denoised_world, center1, scale1)
                cd = ChamferDistanceL2().cuda()(denoised_norm, pcl_clean_norm) * 1e4
                local_cd.append(cd.item())

            avg_cd = sum(local_cd) / len(local_cd)
            cd_list.append(avg_cd)

        # 整个验证集 CD
        cd_all = sum(cd_list) / len(cd_list)
        print_log('[Validation_vote] EPOCH: %d  CD_vote = %.6f' %
                  (epoch, cd_all), logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/CD_vote', cd_all, epoch)

    return DenoiseMetrics(cd_all)


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)
    _,test_dataloader  = builder.dataset_builder(args, config.dataset.test)
    base_model = builder.model_builder(config.model)

    # load checkpoints
    # for finetuned transformer
    builder.load_model(base_model, args.ckpts, logger=logger)
    # base_model.load_model_from_ckpt(args.ckpts) # for BERT
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config, logger=logger)
# -----------------------------------------------------------------------------------------------



# 10k测试------------------------------------------------------------------------------------------------
def test(base_model, test_dataloader, args, config, logger=None):
    npoints = config.npoints
    base_model.eval()  # set model to eval mode

    total_cd = 0.0
    total_cd_10k = 0.0
    total_batches = 0
    total_p2m = 0.0

    # Langevin 多步推理超参（从 config 读，缺省退化为单步 Tweedie）
    langevin = getattr(config, 'langevin', None)
    if langevin is not None:
        lv_steps = int(getattr(langevin, 'num_steps', 1))
        lv_step_size = float(getattr(langevin, 'step_size', 1.0))
        lv_decay = float(getattr(langevin, 'decay', 0.95))
    else:
        lv_steps, lv_step_size, lv_decay = 1, 1.0, 0.95
    print_log(f'[Inference] Langevin: num_steps={lv_steps}, step_size={lv_step_size}, decay={lv_decay}'
              + (' (单步 Tweedie fallback)' if lv_steps <= 1 else ''), logger=logger)

    # 测试 patch_batch（从 config 读）：L 模型显存吃紧，10k 用 4、50k 建议 2，缺省 4
    test_patch_batch = int(getattr(config, 'test_patch_batch', 4))
    print_log(f'[Inference] test_patch_batch={test_patch_batch}', logger=logger)

    # ChamferDistanceL2 只实例化一次，避免每次投票都新建 CUDA 模块（防显存碎片累积）
    cd_metric = ChamferDistanceL2().cuda()

    # 局部表面投影后处理超参（从 config.surface_projection 读，缺省关闭）
    sp_cfg = getattr(config, 'surface_projection', None)
    if sp_cfg is not None and bool(getattr(sp_cfg, 'enable', False)):
        sp_enable = True
        sp_k = int(getattr(sp_cfg, 'k', 16))
        sp_iters = int(getattr(sp_cfg, 'num_iters', 1))
        sp_blend = float(getattr(sp_cfg, 'blend', 1.0))
        print_log(f'[Inference] Surface projection: k={sp_k}, num_iters={sp_iters}, blend={sp_blend}', logger=logger)
    else:
        sp_enable, sp_k, sp_iters, sp_blend = False, 16, 1, 1.0

    with torch.no_grad():
        vote_times = 1   # 关闭多次投票：单次推理即可，降一倍以上显存/计算压力，防崩机
        for idx, (pcl_noisy, pcl_clean, noise_std, center, scale, name) in enumerate(test_dataloader):
            torch.cuda.empty_cache()   # L 模型测试每个样本前清缓存，防显存尖峰崩机
            gc.collect()               # 同步回收 CPU 端 numpy/o3d 临时对象（50k 可视化吃 CPU）
            # 硬显存保护：宁可中断测试也绝不让服务器崩。
            # 先清缓存再读真实占用；若清完仍逼近上限，立刻打印已完成结果并安全退出。
            _mr = gpu_mem_ratio()
            _cpu = psutil.virtual_memory().percent
            if _mr > 0.88 or _cpu > 90.0:
                print_log(f'[测试内存保护] 样本 {idx} 前 GPU={_mr:.1%} / CPU={_cpu:.1f}% 逼近上限，'
                          f'为防服务器崩溃主动中断测试。已完成 {total_batches} 个样本：'
                          f'CD={ (total_cd/total_batches) if total_batches>0 else float("nan"):.6f} '
                          f'p2m={ (total_p2m/total_batches) if total_batches>0 else float("nan"):.6f}。'
                          f'请在 yaml 调小 test_patch_batch 后重跑', logger=logger)
                sys.exit(0)
            save_path = 'test_test_result_1'
            save_path2 = 'visualiza-result10k-1'
            save_path3 = 'finetune_scoredenoise_L'
            PointGPTtype = 'PointGPT-Change'
            center_t = center[0]
            scale_t = scale[0]

            pcl_noisy_world = pcl_noisy * scale_t + center_t
            pcl_clean_world = pcl_clean * scale_t + center_t

            # 保存噪声点（原始坐标系）
            filename_noisy = f'experiments/{save_path3}/{PointGPTtype}/{save_path}/{save_path2}/noisy-result/txt/{name[0]}.xyz'
            os.makedirs(os.path.dirname(filename_noisy), exist_ok=True)
            np.savetxt(filename_noisy, pcl_noisy_world[0].cpu().numpy(), fmt='%.8f')
            # 保存干净点（原始坐标系）
            filename_clean = f'experiments/{save_path3}/{PointGPTtype}/{save_path}/{save_path2}/clean-result/txt/{name[0]}.xyz'
            os.makedirs(os.path.dirname(filename_clean), exist_ok=True)
            np.savetxt(filename_clean, pcl_clean_world[0].cpu().numpy(), fmt='%.8f')

            pcl_noisy = pcl_noisy.cuda()
            pcl_clean = pcl_clean.cuda()
            if noise_std is not None:
                noise_std_t = noise_std.cuda()
            else:
                # PairedEvalDataset 不含 noise_std；从 config.TEST_NOISE fallback
                _test_sigma = getattr(config.dataset.test._base_, 'TEST_NOISE', 0.01)
                noise_std_t = torch.full(
                    (pcl_noisy.shape[0],), _test_sigma, dtype=torch.float32
                ).cuda()
            center_gpu = center_t.to(pcl_noisy.device)
            scale_gpu = scale_t.to(pcl_noisy.device)

            # 用于可视化的点 50k原始坐标系
            P_gt = pcl_clean_world.squeeze(0).clone()
            P_gt = P_gt.detach().cpu().numpy()
            P_ny = pcl_noisy_world.squeeze(0).clone()
            P_ny = P_ny.detach().cpu().numpy()

            
            best_cd = float('inf')
            best_p2m = None
            best_denoised = None
            for _ in range(vote_times):
                # patch-based 推理：完整点云切 1024-patch 逐个去噪再拼回，覆盖全部点
                denoised_full = patch_based_denoise(
                    base_model, pcl_noisy[0], noise_std_t,
                    patch_batch=test_patch_batch,
                    num_steps=lv_steps, step_size=lv_step_size, decay=lv_decay,
                )  # [N, 3]
                denoised_10k = denoised_full.unsqueeze(0)
                denoised_filtered = sor_filter(denoised_10k[0])
                # 局部表面投影后处理（仅在 config.surface_projection.enable 时启用）
                if sp_enable:
                    denoised_filtered = local_surface_projection(
                        denoised_filtered, k=sp_k, num_iters=sp_iters, blend=sp_blend)
                denoised_10k = denoised_filtered.unsqueeze(0).to(denoised_10k.device)
                denoised_world = denoised_10k * scale_gpu + center_gpu
                clean_world = pcl_clean * scale_gpu + center_gpu
                p2m_loss = compute_p2m(denoised_world[0], name[0], 'test') * 1e4

                _, center1, scale1 = normalize_unit_sphere(clean_world)
                pcl_clean_10k_norm = (clean_world - center1) / scale1
                denoised_10k_norm = (denoised_world - center1) / scale1
                batch_cd = cd_metric(denoised_10k_norm, pcl_clean_10k_norm) * 1e4

                cur_cd = batch_cd.item()
                if cur_cd < best_cd:
                    best_cd = cur_cd
                    best_p2m = p2m_loss.item() if isinstance(p2m_loss, torch.Tensor) else float(p2m_loss)
                    best_denoised = denoised_world.detach().clone()

                # 每次投票后释放本轮中间张量并清缓存，防 50k×Langevin 多步显存碎片累积崩机
                del denoised_full, denoised_10k, denoised_filtered, denoised_world
                del clean_world, denoised_10k_norm, pcl_clean_10k_norm, batch_cd, p2m_loss
                torch.cuda.empty_cache()

            denoised_10k = best_denoised
            batch_cd = float(best_cd)
            p2m_loss = best_p2m
           
# 可视化去噪效果--------------------------------------------------------------------------------------------
        # 噪声点------------------------------------------------------------
            nn = NearestNeighbors(n_neighbors=1).fit(P_gt)
            dists_noisy, _ = nn.kneighbors(P_ny)
            dists_noisy = dists_noisy.reshape(-1)
           
            # 归一化误差（使颜色更集中）
            max_noisy = np.percentile(dists_noisy, 99)   # 去除极端大值
            norm_noisy = np.clip(dists_noisy / max_noisy, 0, 1)
            # ---- Step 3: 蓝→黄 colormap ----
            # 蓝色 (0,0,1)
            # 黄色 (1,1,0)
            # 插值：color = (1-t)*blue + t*yellow
            blue = np.array([0.0, 0.0, 1.0])
            yellow = np.array([1.0, 1.0, 0.0])
            colors_noisy = (1 - norm_noisy[:, None]) * blue + norm_noisy[:, None] * yellow

            # ---- Step 4: 保存为 PLY ----
            pcd_noisy = o3d.geometry.PointCloud()
            pcd_noisy.points = o3d.utility.Vector3dVector(P_ny)
            pcd_noisy.colors = o3d.utility.Vector3dVector(colors_noisy)
            output_ply_noisy = f"experiments/{save_path3}/{PointGPTtype}/{save_path}/{save_path2}/noisy-result/blue_yellow/{name[0]}.ply"
            os.makedirs(os.path.dirname(output_ply_noisy), exist_ok=True)
            o3d.io.write_point_cloud(output_ply_noisy, pcd_noisy)
            print("Saved colored PLY:", output_ply_noisy)
        # 去噪后效果------------------------------------------------------------------
           # ---- Step 2: 最近邻误差计算 ----
            nn = NearestNeighbors(n_neighbors=1).fit(P_gt)
            dists_denoise, _ = nn.kneighbors(denoised_10k[0].cpu().numpy())
            dists_denoise = dists_denoise.reshape(-1)

            # 归一化误差（使颜色更集中）
            max_denoise = np.percentile(dists_denoise, 99)   # 去除极端大值
            norm_denoise = np.clip(dists_denoise / max_denoise, 0, 1)

            blue = np.array([0.0, 0.0, 1.0])
            yellow = np.array([1.0, 1.0, 0.0])
            colors_denoise = (1 - norm_denoise[:, None]) * blue + norm_denoise[:, None] * yellow

            pcd_denoise = o3d.geometry.PointCloud()
            pcd_denoise.points = o3d.utility.Vector3dVector(denoised_10k[0].cpu().numpy())
            pcd_denoise.colors = o3d.utility.Vector3dVector(colors_denoise)
            output_ply_denoise = f"experiments/{save_path3}/{PointGPTtype}/{save_path}/{save_path2}/denoise-result/blue_yellow/{name[0]}.ply"
            os.makedirs(os.path.dirname(output_ply_denoise), exist_ok=True)
            o3d.io.write_point_cloud(output_ply_denoise, pcd_denoise)
            print("Saved colored PLY:", output_ply_denoise)
# 可视化去噪效果--------------------------------------------------------------------------------------------

            # 保存去噪后的点
            filename = f'experiments/{save_path3}/{PointGPTtype}/{save_path}/{save_path2}/denoise-result/txt/{name[0]}.xyz'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            np.savetxt(filename, denoised_10k[0].cpu().numpy(), fmt='%.8f')
            # print(pcl_denoised_original,pcl_denoised_original.shape,'point-------------------------------')
            total_cd += batch_cd
            total_p2m += p2m_loss
            
            total_batches += 1

    avg_cd = total_cd / total_batches
    avg_p2m = total_p2m / total_batches
    

    # print_log(' Chamfer Distance = %.6f' %
    #         ( avg_cd), logger=logger)
    print_log(' Chamfer Distance = %.6f p2m =  %.6f' %
            ( avg_cd,avg_p2m), logger=logger)

    print_log(f"[TEST_VOTE] 每个样本在 {vote_times} 次中取最优结果后保存", logger=logger)


# 50k-----------------------------------------------------------------------------------------------
def test_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=None, times=10):

    base_model.eval()  # set model to eval mode
    npoints = config.npoints
    cd_list = []
    p2m_list = []
    cd_list_10k = []

    with torch.no_grad():
        for idx, (pcl_noisy, pcl_clean, noise_std, center, scale, name) in enumerate(test_dataloader):
            pcl_noisy = pcl_noisy.cuda()
            pcl_clean = pcl_clean.cuda()
            if noise_std is not None:
                noise_std_t = noise_std.cuda()
            else:
                # PairedEvalDataset 不含 noise_std；从 config.TEST_NOISE fallback
                _test_sigma = getattr(config.dataset.test._base_, 'TEST_NOISE', 0.01)
                noise_std_t = torch.full(
                    (pcl_noisy.shape[0],), _test_sigma, dtype=torch.float32
                ).cuda()
            center_t = center[0].to(pcl_noisy.device)
            scale_t = scale[0].to(pcl_noisy.device)
            local_cd = []
            local_p2m = []
            for kk in range(times):
                denoised_10k = base_model(pcl_noisy, pcl_clean, 'val', name, noise_std=noise_std_t)

                denoised_filtered = sor_filter(denoised_10k[0])
                denoised_10k = denoised_filtered.unsqueeze(0).to(denoised_10k.device)
                denoised_world = denoised_10k * scale_t + center_t
                clean_world = pcl_clean * scale_t + center_t

                p2m_loss = compute_p2m(denoised_world[0], name[0], 'test') * 1e4
                
               
                _, center1, scale1 = normalize_unit_sphere(clean_world)

                # pc_cat = torch.cat([pcl_clean, denoised_10k], dim=1)

                # pc_cat_norm, center1, scale1 = normalize_unit_sphere(
                #     pc_cat
                # )

                # 分别归一化（保持一模一样的 center / scale）
                pcl_clean_10k_norm = (clean_world - center1) / scale1
                denoised_10k_norm = (denoised_world - center1) / scale1
               

                cd = ChamferDistanceL2().cuda()(denoised_10k_norm,pcl_clean_10k_norm)* 1e4
                
                # cd = pytorch3d.loss.chamfer_distance(denoised_10k_norm, pcl_clean_10k_norm)[0].item()* 1e4

                

                local_cd.append(cd.item())
                local_p2m.append(p2m_loss)

            avg_cd = sum(local_cd) / len(local_cd)
            avg_p2m = sum(local_p2m) / len(local_p2m)
            
            cd_list.append(avg_cd)
            p2m_list.append(avg_p2m)
            

        # 整个验证集 CD
        cd_all = sum(cd_list) / len(cd_list)
        p2m_all = sum(p2m_list) / len(p2m_list)
        
        

        # print_log('[Validation_vote] EPOCH: %d  CD_vote = %.6f' %
        #           (epoch, cd_all), logger=logger)
        print_log('[Validation_vote] EPOCH: %d  CD_vote = %.6f P2M_vote = %.6f' %
                  (epoch, cd_all,p2m_all), logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/CD_vote', cd_all, epoch)

    return DenoiseMetrics(cd_all, 0.0)
