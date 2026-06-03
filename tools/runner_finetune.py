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
from utils.p2m_loss import compute_p2m

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


def check_memory_and_exit(base_model, optimizer, epoch, metrics, best_metrics, args, logger,
                          threshold_gpu=0.80, threshold_cpu=85.0):
    """检查 GPU 显存和 CPU 内存，超阈值时保存检查点并主动退出，防止服务器崩溃"""
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

            loss1 = base_model(pcl_noisy, pcl_clean, 'train', name,
                               epoch=epoch, max_epoch=config.max_epoch,
                               noise_std=noise_std_t)
            _loss = loss1 #用于去噪，下游任务的损失和生成任务的损失是一样的

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

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                  (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()], optimizer.param_groups[0]['lr']), logger=logger)
    
        if epoch % args.val_freq == 0 and epoch != 0:
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
            center_t = center[0].to(pcl_noisy.device)
            scale_t = scale[0].to(pcl_noisy.device)

            # 模型输出在归一化空间（单步去噪）
            denoised_10k_norm = base_model(pcl_noisy, pcl_clean, 'val', name, noise_std=noise_std_t)

            # 归一化空间 CD
            batch_cd_10k = ChamferDistanceL2().cuda()(denoised_10k_norm, pcl_clean) * 1e4
            # noisy baseline：输入直接当输出，CD 应比 denoised 大；用于判断模型是否真的在降噪
            batch_cd_noisy = ChamferDistanceL2().cuda()(pcl_noisy, pcl_clean) * 1e4
            total_cd_noisy_baseline = total_cd_noisy_baseline + batch_cd_noisy if idx > 0 else batch_cd_noisy

            # 反归一化到世界坐标系算 P2M
            denoised_world = denoised_10k_norm * scale_t + center_t
            p2m_loss = compute_p2m(denoised_world[0], name[0], 'test') * 1e4

            total_cd_10k += batch_cd_10k
            total_p2m += p2m_loss
            total_batches += 1

    if args.distributed:
        # 用真正参与累加的 total_cd_10k / total_p2m
        total_cd_10k = dist_utils.gather_tensor(total_cd_10k.cuda() if torch.is_tensor(total_cd_10k) else torch.tensor(total_cd_10k).cuda(), args)
        total_p2m = dist_utils.gather_tensor(total_p2m.cuda() if torch.is_tensor(total_p2m) else torch.tensor(total_p2m).cuda(), args)
        total_batches = dist_utils.gather_tensor(torch.tensor(total_batches).cuda(), args)
        if args.local_rank != 0:
            return None

    avg_cd = total_cd_10k / total_batches
    avg_p2m = total_p2m / total_batches
    avg_cd_noisy = total_cd_noisy_baseline / total_batches

    print_log('[Validation] EPOCH: %d  Chamfer Distance = %.6f  P2M = %.6f  (Noisy baseline CD = %.6f, delta = %.6f)' %
        (epoch, avg_cd, avg_p2m, avg_cd_noisy, avg_cd_noisy - avg_cd), logger=logger)

    if args.distributed:
        torch.cuda.synchronize()

    if val_writer is not None:
        val_writer.add_scalar('Metric/CD', avg_cd, epoch)

    return DenoiseMetrics(avg_cd, avg_p2m)

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

    with torch.no_grad():
        vote_times = 5
        for idx, (pcl_noisy, pcl_clean, noise_std, center, scale, name) in enumerate(test_dataloader):
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
                denoised_10k = base_model(pcl_noisy, pcl_clean, 'val', name, noise_std=noise_std_t)
                denoised_filtered = sor_filter(denoised_10k[0])
                denoised_10k = denoised_filtered.unsqueeze(0).to(denoised_10k.device)
                denoised_world = denoised_10k * scale_gpu + center_gpu
                clean_world = pcl_clean * scale_gpu + center_gpu
                p2m_loss = compute_p2m(denoised_world[0], name[0], 'test') * 1e4

                _, center1, scale1 = normalize_unit_sphere(clean_world)
                pcl_clean_10k_norm = (clean_world - center1) / scale1
                denoised_10k_norm = (denoised_world - center1) / scale1
                batch_cd = ChamferDistanceL2().cuda()(denoised_10k_norm, pcl_clean_10k_norm) * 1e4

                if batch_cd.item() < best_cd:
                    best_cd = batch_cd.item()
                    best_p2m = p2m_loss.item() if isinstance(p2m_loss, torch.Tensor) else float(p2m_loss)
                    best_denoised = denoised_world.clone()

            denoised_10k = best_denoised
            batch_cd = torch.tensor(best_cd, device=pcl_noisy.device)
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
