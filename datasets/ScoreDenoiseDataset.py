import random
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import pytorch_lightning as pl
from utils import misc
import numpy as np
import os, sys
import torch
from pointnet2_ops import pointnet2_utils
# 这里替换成 PointGPT 内置 KNN
from knn_cuda import KNN

from .build import DATASETS
from utils.logger import *
from datasets.scoredenoise.transforms import standard_train_transforms
from datasets.scoredenoise.transforms import NormalizeUnitSphere, AddNoise, NoisyJitter, NoisyPointDropout, RandomScale, RandomRotate, CleanScaleTranslateCPU,CleanPointcloudTranslateCPU,CleanPointcloudRotateY_CPU

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


# 加载原始点云（干净） 单个gpu版本
class PointCloudDataset(Dataset):

    def __init__(self, root, dataset, split, resolution, transform=None, pcl_dir=None,
                 exclude_names=None, include_names=None):
        super().__init__()
        # pcl_dir 若给定则直接用它（泛化实验换数据集，绕过 PUNet 固定路径）；否则按默认拼接
        self.pcl_dir = pcl_dir if pcl_dir is not None else os.path.join(root, dataset, 'pointclouds', split, resolution)
        self.transform = transform
        self.pointclouds = []
        self.pointcloud_names = []
        # exclude_names / include_names：按 shape 名过滤（用于 train/val 划分：训练排除验证 shape，
        # 验证只取这些 shape）。二者可同时为 None（不过滤）。
        for fn in os.listdir(self.pcl_dir):
            if fn[-3:] != 'xyz':
                continue
            nm = fn[:-4]
            if exclude_names is not None and nm in exclude_names:
                continue
            if include_names is not None and nm not in include_names:
                continue
            pcl_path = os.path.join(self.pcl_dir, fn)
            if not os.path.exists(pcl_path):
                raise FileNotFoundError('File not found: %s' % pcl_path)
            pcl = torch.FloatTensor(np.loadtxt(pcl_path, dtype=np.float32))
            self.pointclouds.append(pcl)
            self.pointcloud_names.append(nm)

        print(f'[INFO] Loaded dataset {dataset} - {resolution} ({len(self.pointclouds)} clouds)')

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {
            'pcl_clean': self.pointclouds[idx].clone(), 
            'name': self.pointcloud_names[idx]
        }
        if self.transform is not None:
            data = self.transform(data)
        return data
    
    # 加载去噪一次后的50k噪声点云
class PointCloudDataset_noise(Dataset):

    def __init__(self, root, resolution):
        super().__init__()
        self.pcl_dir = os.path.join(root, resolution)
        self.pointclouds = []
        self.pointcloud_names = []
        for fn in os.listdir(self.pcl_dir):
            if fn[-3:] != 'xyz':
                continue
            pcl_path = os.path.join(self.pcl_dir, fn)
            if not os.path.exists(pcl_path):
                raise FileNotFoundError('File not found: %s' % pcl_path)
            pcl = torch.FloatTensor(np.loadtxt(pcl_path, dtype=np.float32))
            self.pointclouds.append(pcl)
            self.pointcloud_names.append(fn[:-4])
        
        print(f'[INFO] Loaded dataset {root} - {resolution}')

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {
            'pcl_clean': self.pointclouds[idx].clone(), 
            'name': self.pointcloud_names[idx]
        }
        return data

class PointCloudNoisyDataset(Dataset):
    def __init__(self, noisy_dir):
        super().__init__()
        self.pcl_dir = noisy_dir
        self.pointclouds = []
        self.pointcloud_names = []
        for fn in os.listdir(self.pcl_dir):
            if not fn.endswith('xyz'):
                continue
            pcl_path = os.path.join(self.pcl_dir, fn)
            if not os.path.exists(pcl_path):
                raise FileNotFoundError('File not found: %s' % pcl_path)
            pcl = torch.FloatTensor(np.loadtxt(pcl_path, dtype=np.float32))
            self.pointclouds.append(pcl)
            self.pointcloud_names.append(fn[:-4])
        print(f'[INFO] Loaded noisy dataset: {self.pcl_dir}')

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        return {
            'pcl_noisy': self.pointclouds[idx].clone(),
            'name': self.pointcloud_names[idx]
        }
# 加载成对点云数据集（干净 + 有噪声）
class PairedPointCloudDataset(Dataset):
    def __init__(self, clean_dataset, noise_dataset):
        super().__init__()
        
        self.clean_dict = {
            clean_dataset[idx]['name']: clean_dataset[idx]['pcl_clean']
            for idx in range(len(clean_dataset))
        }

        self.noise_dict = {
            noise_dataset[idx]['name']: noise_dataset[idx]['pcl_clean']
            for idx in range(len(noise_dataset))
        }

        # 名字取交集
        self.names = sorted(
            set(self.clean_dict.keys()) & set(self.noise_dict.keys())
        )

        if len(self.names) == 0:
            raise RuntimeError("No matched clean/noisy point clouds found.")

        print(f'[INFO] PairedPointCloudDataset size = {len(self.names)}')

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        data = {
            'pcl_clean': self.clean_dict[name].clone(),
            'pcl_noisy': self.noise_dict[name].clone(),
            'name': name
        }
        return data

class PairedEvalDataset(Dataset):
    """
    测试配对数据集：
    - clean: PUNet test（干净点云）
    - noisy: examples test（已加噪点云）
    归一化基准采用 noisy 自身统计量。
    """
    def __init__(self, clean_dataset, noisy_dataset):
        super().__init__()
        self.clean_dict = {
            clean_dataset[idx]['name']: clean_dataset[idx]['pcl_clean']
            for idx in range(len(clean_dataset))
        }
        self.noisy_dict = {
            noisy_dataset[idx]['name']: noisy_dataset[idx]['pcl_noisy']
            for idx in range(len(noisy_dataset))
        }
        self.names = sorted(
            set(self.clean_dict.keys()) & set(self.noisy_dict.keys())
        )
        if len(self.names) == 0:
            raise RuntimeError("No matched clean/noisy point clouds found.")
        print(f'[INFO] PairedEvalDataset size = {len(self.names)}')

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        pcl_clean_world = self.clean_dict[name].clone()
        pcl_noisy_world = self.noisy_dict[name].clone()

        # ✅ 修复: 使用干净点云确定归一化基准(与训练一致)
        pcl_clean_norm, center, scale = NormalizeUnitSphere.normalize(pcl_clean_world)
        pcl_noisy_norm = (pcl_noisy_world - center) / scale

        return {
            'pcl_noisy': pcl_noisy_norm,
            'pcl_clean': pcl_clean_norm,
            'center': center,
            'scale': scale,
            'name': name
        }


class HeldOutValDataset(Dataset):
    """验证集：从训练集留出的【完整点云】+ 固定种子加噪。

    - 每个 (点云, 噪声级) 生成一条样本，返回完整云（供 runner 用 test 同款 pipeline 评测）。
    - 噪声用 per-sample 固定种子 → 每个 epoch 复现同一份噪声，保证跨 epoch 的 val 分数可比、
      checkpoint 排序稳定（若每次随机加噪，val 分数会抖动，选模型不可靠）。
    - 归一化基准取自干净点云自身（与 train/test 一致）。P2M 的 mesh split 由 runner 按
      VAL_NUM 决定：VAL_NUM<=0（验证看 test）→ meshes/test/；VAL_NUM>0（留出 train shape）→ meshes/train/。
    """
    def __init__(self, clean_dataset, noise_levels, seed=2024):
        super().__init__()
        self.samples = []  # (clean_world[N,3], name, sigma, sample_seed)
        for i in range(len(clean_dataset)):
            item = clean_dataset[i]
            for j, sigma in enumerate(noise_levels):
                self.samples.append(
                    (item['pcl_clean'], item['name'], float(sigma), int(seed) + i * 131 + j))
        print(f'[INFO] HeldOutValDataset: {len(clean_dataset)} clouds × {len(noise_levels)} '
              f'noise level(s) = {len(self.samples)} val samples (levels={list(noise_levels)})')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clean_world, name, sigma, s = self.samples[idx]
        clean_norm, center, scale = NormalizeUnitSphere.normalize(clean_world.clone())
        g = torch.Generator().manual_seed(s)
        noise = torch.randn(clean_norm.shape, generator=g) * sigma
        noisy_norm = clean_norm + noise
        return {
            'pcl_noisy': noisy_norm,
            'pcl_clean': clean_norm,
            'center': center,
            'scale': scale,
            'name': name,
            'noise_std': sigma,
        }

# 加载原始点云（干净） 多个gpu版本
# class PointCloudDataset(Dataset):

#     def __init__(self, root, dataset, split, resolution, transform=None):
#         super().__init__()
#         self.pcl_dir = os.path.join(root, dataset, 'pointclouds', split, resolution)
#         self.transform = transform
#         self.pointcloud_paths = []
#         self.pointcloud_names = []
#         for fn in os.listdir(self.pcl_dir):
#             if fn[-3:] != 'xyz':
#                 continue
#             pcl_path = os.path.join(self.pcl_dir, fn)
#             if not os.path.exists(pcl_path):
#                 raise FileNotFoundError('File not found: %s' % pcl_path)
#             self.pointcloud_paths.append(pcl_path)
#             self.pointcloud_names.append(fn[:-4])
        
#         print(f'[INFO] Loaded dataset {dataset} - {resolution}')

#     def __len__(self):
#         return len(self.pointcloud_paths)

#     def __getitem__(self, idx):
#         # 运行时读取
#         pcl = np.loadtxt(self.pointcloud_paths[idx], dtype=np.float32)
#         pcl = torch.from_numpy(pcl)

#         data = {
#             'pcl_clean': pcl, 
#             'name': self.pointcloud_names[idx]
#         }
#         if self.transform is not None:
#             data = self.transform(data)
#         return data
# 加载测试集干净点云
class PointCloudTestDataset(Dataset):

    def __init__(self, root, dataset,split, transform=None):
        super().__init__()
        self.pcl_dir = os.path.join(root, dataset,split)
        self.transform = transform
        self.pointclouds = []
        self.pointcloud_names = []
        self.split = split
        for fn in os.listdir(self.pcl_dir):
            if fn[-3:] != 'xyz':
                continue
            pcl_path = os.path.join(self.pcl_dir, fn)
            if not os.path.exists(pcl_path):
                raise FileNotFoundError('File not found: %s' % pcl_path)
            pcl = torch.FloatTensor(np.loadtxt(pcl_path, dtype=np.float32))
            self.pointclouds.append(pcl)
            self.pointcloud_names.append(fn[:-4])
        
        print(f'[INFO] Loaded dataset {dataset} - {split}')

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {
            'pcl_clean': self.pointclouds[idx].clone(), 
            'name': self.pointcloud_names[idx]
        }
        if self.transform is not None:
            data = self.transform(data)
        return data

def make_patches_for_pcl_pair(pcl_A, pcl_B, patch_size, num_patches, ratio):
    pcl_A = pcl_A.cuda()
    pcl_B = pcl_B.cuda()
    device = pcl_A.device
    N_A = pcl_A.size(0)
    N_B = pcl_B.size(0)
    
    patch_size_A = min(patch_size, N_A)
    patch_size_B = min(int(ratio*patch_size), N_B)
    num_patches = min(num_patches, N_A)
    
    seed_idx = torch.randperm(N_A, device=device)[:num_patches]
    seed_pnts = pcl_A[seed_idx].unsqueeze(0)  # (1, P, 3)
    
    knn_op_A = KNN(k=patch_size_A, transpose_mode=False)
    _, idx_A = knn_op_A(pcl_A.unsqueeze(0), seed_pnts)  # (1, P, M)
    pat_A = pcl_A[idx_A[0]]  # (P, M, 3)
    
    knn_op_B = KNN(k=patch_size_B, transpose_mode=False)
    _, idx_B = knn_op_B(pcl_B.unsqueeze(0), seed_pnts)
    pat_B = pcl_B[idx_B[0]]  # (P, rM, 3)
    
    return pat_A, pat_B

def knn_pt(query, ref, k):
    """
    query: (B, Q, 3)
    ref:   (B, N, 3)
    return: idx (B, Q, k)
    """
    # dist: (B, Q, N)
    dist = torch.cdist(query, ref)
    idx = dist.topk(k, largest=False).indices
    return idx

# 将点云选取一个中心点，找到其邻近的patch_size个点，组成一个patch
def make_patches_for_pcl(pcl, patch_size, num_patches):
    # 将点云数据移到gpu上
    # pcl = pcl.cuda()
    device = pcl.device
    # 获取点云数量
    N = pcl.size(0)
    # 防止数组越界
    patch_size = min(patch_size, N)
    num_patches = min(num_patches, N)
    # 随机选取16个中心点
    seed_idx = torch.randperm(N, device=device)[:num_patches]
    seed_pnts = pcl[seed_idx].unsqueeze(0)
    # knn_op = KNN(k=patch_size, transpose_mode=False)
    # _, idx = knn_op(pcl.unsqueeze(0), seed_pnts)

    idx = knn_pt(seed_pnts, pcl.unsqueeze(0), patch_size)
    pat = pcl[idx[0]]  # (16, 2048, 3)
    return pat



def denoise_collate_fn(batch):
    noisy_list = []
    clean_list = []
    center_list = []
    scale_list=[]
    name_list = []
    for item in batch:
        assert "pcl_clean" in item
        assert "pcl_noisy" in item

        assert item["pcl_clean"].shape == item["pcl_noisy"].shape

        noisy_list.append(item["pcl_noisy"])
        clean_list.append(item["pcl_clean"])
        name_list.append(item["name"])
        
        center_list.append(item["center"])
        scale_list.append(item["scale"])

        
    noisy = torch.stack(noisy_list, dim=0).float()
    clean = torch.stack(clean_list, dim=0).float()

    
    return noisy, clean,center_list,scale_list,name_list

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def denoise_collate_fn_test(batch):
    noisy_list = []
    clean_list = []
    name_list = []
    center_list = []
    scale_list = []
    noise_std_list = []

    for item in batch:
        assert "pcl_clean" in item
        assert "pcl_noisy" in item
        assert item["pcl_clean"].shape == item["pcl_noisy"].shape

        noisy_list.append(item["pcl_noisy"])
        clean_list.append(item["pcl_clean"])
        name_list.append(item["name"])
        center_list.append(item["center"])
        scale_list.append(item["scale"])
        noise_std_list.append(item.get('noise_std', None))

    noisy = torch.stack(noisy_list, dim=0).float()
    clean = torch.stack(clean_list, dim=0).float()

    # noise_std: train/val 走 transform 一定有；test 走 PairedEvalDataset 没有，置 None 由模型 fallback
    if all(s is None for s in noise_std_list):
        noise_std = None
    else:
        noise_std = torch.tensor(noise_std_list, dtype=torch.float32)

    return noisy, clean, noise_std, center_list, scale_list, name_list
    
def global_sample(pcl, target_n=10000, mode='fps'):
    """
    pcl: Tensor [N, 3]
    return: Tensor [target_n, 3]
    """
    N = pcl.shape[0]

    if N == target_n:
        return pcl

    if N > target_n:
        if mode == 'random':
            idx = torch.randperm(N, device=pcl.device)[:target_n]
            return pcl[idx]
        elif mode == 'fps':
            # fps 接受 [1, N, 3]
            pcl_b = pcl.unsqueeze(0)
            idx = misc.fps(pcl_b, target_n)[0]  # [target_n, 3]
            return idx
        else:
            raise ValueError("mode must be 'random' or 'fps'")
    else:
        # N < target_n：重复采样（一般不会发生）
        idx = torch.randint(0, N, (target_n,), device=pcl.device)
        return pcl[idx]



def cpu_fps(points, k):
    """
    points: [N, 3]  (CPU Tensor)
    return: [k, 3]
    """

    N = points.shape[0]
    assert k <= N

    pts = points.float()

    # 记录已选点的最小距离
    dist = torch.full((N,), float('inf'))

    # 随机挑一个起点
    farthest = torch.randint(0, N, (1,)).item()

    idxs = []

    for _ in range(k):
        idxs.append(farthest)

        cur = pts[farthest].unsqueeze(0)        # [1, 3]
        d = torch.sum((pts - cur)**2, dim=1)    # [N]

        # 记录“到已选集合的最小距离”
        dist = torch.minimum(dist, d)

        # 选择目前距离最远的点
        farthest = torch.argmax(dist).item()

    idxs = torch.tensor(idxs, dtype=torch.long)
    return pts[idxs]
   
# 结构保持型采样（Voxel Grid Downsample）
def voxel_then_fps_to_fixed(pcl, target_n=10000, max_iter=8):

    """
    pcl: (N, 3) torch.Tensor
    return: (1024, 3)
    """

    N = pcl.shape[0]
    if N == target_n:
        return pcl

    p_min = pcl.min(0)[0]
    p_max = pcl.max(0)[0]
    bbox = p_max - p_min
    volume = bbox.prod()

    # 初始 voxel size：按 1024 目标估计
    voxel_size = (volume / target_n) ** (1 / 3)
    pcl_shift = pcl - p_min

    sampled = pcl

    for _ in range(max_iter):
        vox = torch.floor(pcl_shift / voxel_size)

        key = (
            vox[:, 0] * 73856093 +
            vox[:, 1] * 19349663 +
            vox[:, 2] * 83492791
        )

        # -------- 兼容所有 PyTorch 版本 --------
        sorted_key, order = torch.sort(key)

        mask = torch.ones_like(sorted_key, dtype=torch.bool)
        mask[1:] = sorted_key[1:] != sorted_key[:-1]

        uniq_idx = order[mask]
        sampled = pcl[uniq_idx]
        M = sampled.shape[0]

        # -------- 自适应 voxel 调整 --------
        if M > target_n:
            voxel_size *= 1.1
        else:
            voxel_size *= 0.9

    # -------- 精确裁剪到 1024 --------
    if M > target_n:
        pts = cpu_fps(sampled, target_n)
        return pts

    if M < target_n:
        pad = torch.randint(0, M, (target_n - M,), device=pcl.device)
        sampled = torch.cat([sampled, sampled[pad]], dim=0)

    return sampled

# 从点云中切出小 patch，并添加噪声，生成训练样本
class PairedPatchDataset(Dataset):

    def __init__(self, datasets, patch_ratio, on_the_fly=True, patch_size=1000, num_patches=1000,
                 noise_min=0, noise_max=0, transform=None, flag='train', oversample_factor=1):
        super().__init__()
        self.datasets = datasets
        self.all_data = []  # 长度 ~120，包含了 10k/30k/50k 三个分辨率的所有干净点云
        for dset in datasets:
            for i in range(len(dset)):
                self.all_data.append(dset[i])

        self.len_datasets = len(self.all_data)
        self.patch_ratio = patch_ratio
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.on_the_fly = on_the_fly
        self.transform = transform
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.patches = []
        self.flag = flag
        # 训练时给每张点云重复采样 oversample_factor 次（每次随机 seed + 随机 σ），
        # 增加每 epoch 的优化步数；val/test 走 flag != 'train' 路径，oversample 应保持 1
        self.oversample_factor = oversample_factor if flag == 'train' else 1
        # print(self.len_datasets * self.num_patches,'+++++++++++++++++++++++++++++++++++++++++++++') 40
        # Initialize
        if not on_the_fly:
            self.make_patches()
        

    def make_patches(self):
        for dataset in tqdm(self.datasets, desc='MakePatch'):
            for data in tqdm(dataset):
                pat_noisy, pat_clean = make_patches_for_pcl_pair(
                    data['pcl_noisy'],
                    data['pcl_clean'],
                    patch_size=self.patch_size,
                    num_patches=self.num_patches,
                    ratio=self.patch_ratio
                )   # (P, M, 3), (P, rM, 3)
                for i in range(pat_noisy.size(0)):
                    self.patches.append((pat_noisy[i], pat_clean[i], ))

    def __len__(self):
        return len(self.all_data) * self.oversample_factor

    def __getitem__(self, idx):
        # all_data 是 10k/30k/50k 三个分辨率拼起来的所有干净点云
        # oversample 时同一个 pcl 会被采样 oversample_factor 次，每次随机种子点 + 随机 σ
        if self.on_the_fly:
            pcl_data = self.all_data[idx % len(self.all_data)]
            data = {
                'pcl_clean': pcl_data['pcl_clean'].clone(),
                'name': pcl_data['name'],
            }
            if self.transform is not None:
                data = self.transform(data)
                # 注意：score-based 训练需要 noise_std 透传到模型，所以不再 pop

            # 训练阶段：随机种子点 + KNN 切固定大小 patch，让不同分辨率样本可以同 batch
            if self.flag == 'train':
                N = data['pcl_clean'].shape[0]
                patch_size = min(self.patch_size, N)
                seed = torch.randint(0, N, (1,)).item()
                seed_pnt = data['pcl_noisy'][seed:seed + 1]               # [1, 3]
                dist = ((data['pcl_noisy'] - seed_pnt) ** 2).sum(dim=-1)  # [N]
                # 用 noisy 空间的 KNN（推理时一致），同步索引保证 clean/noisy 点对齐
                idx_knn = dist.topk(patch_size, largest=False).indices
                data['pcl_clean'] = data['pcl_clean'][idx_knn]
                data['pcl_noisy'] = data['pcl_noisy'][idx_knn]
        else:
            data = {
                'pcl_noisy': self.patches[idx][0].clone(),
                'pcl_clean': self.patches[idx][1].clone(),
            }
        return data


class ScoreDenoise(pl.LightningDataModule):
# class ScoreDenoise():

    # 数据加载部分
    def __init__(self,args, config):
        super(ScoreDenoise, self).__init__()
        self.root = config.ROOT
        self.dataset = config.DATASET
        self.resolutions = config.RESOLUTIONS
        # if num_points ==10000:
        #     self.resolutions = config.RESOLUTIONS_10k
        # elif num_points ==30000:
        #     self.resolutions = config.RESOLUTIONS_30k
        # else:
        #     self.resolutions = config.RESOLUTIONS_50k
        self.noise_min = config.NOISE_MIN
        self.noise_max = config.NOISE_MAX
        self.patch_size = config.PATCH_SIZE
        self.num_patches = config.NUM_PATCHES
        self.train_batch_size = config.TRAIN_BATCH_SIZE
        self.num_workers = config.NUM_WORKERS
        self.val_noise = config.VAL_NOISE
        self.aug_rotate = config.AUG_ROTATE
        # 训练过采样倍数：每张点云每 epoch 切 oversample 次随机 patch（不同种子+不同σ）
        self.train_oversample = getattr(config, 'TRAIN_OVERSAMPLE', 1)
        # σ 对数均匀采样开关（EDM 式）：范围仍为 [NOISE_MIN, NOISE_MAX] 不变，仅改采样分布
        self.noise_log_uniform = getattr(config, 'NOISE_LOG_UNIFORM', False)
        # 测试分辨率和带噪目录（从 ScoreDenoise.yaml 读，默认 10k/1%）
        self.test_resolution = getattr(config, 'TEST_RESOLUTION', '10000_poisson')
        self.test_noisy_dir  = getattr(config, 'TEST_NOISY_DIR',
                                       f'PUNet_{self.test_resolution}_0.01')
        # TEST_NOISY_PATH（可选）：泛化实验换测试集时，直接指向任意带噪 .xyz 文件夹
        # （相对运行目录/仓库根 或 绝对路径）；设了就覆盖上面默认的 examples 路径构造。
        # clean 仍用 PUNet test（按 TEST_RESOLUTION 配对+归一化+P2M），故带噪 .xyz 要与之同名同分辨率。
        self.test_noisy_path = getattr(config, 'TEST_NOISY_PATH', None)
        # TEST_CLEAN_PATH（可选）：换整个数据集（如 PCNet）时，干净点云根目录（其下应有 <分辨率>/ 子目录）
        # 干净点云目录 = <TEST_CLEAN_PATH>/<TEST_RESOLUTION>；缺省用 PUNet 默认路径。
        self.test_clean_path = getattr(config, 'TEST_CLEAN_PATH', None)
        # ===== 验证集划分（从训练集留出，独立于 test，防泄漏）=====
        # 原来没有 val 文件夹时 val_dataloader 会静默回退到 test split → 用测试集选模型（泄漏）。
        # 现在改为：确定性地从 train 里留出 VAL_NUM 个 shape 作验证，并把它们从训练集排除。
        # VAL_NUM<=0（默认）: 不留出，验证看 test split → 与 ScoreDenoise/PD-Flow 对齐（train 全用）。
        # VAL_NUM>0: 从 train 留出这么多 shape 作验证（方法学更干净，但训练数据变少、偏离 baseline 协议）。
        self.val_num = int(getattr(config, 'VAL_NUM', 0))
        self.val_split_seed = int(getattr(config, 'VAL_SPLIT_SEED', 2024))
        # 验证分辨率：单一，与主基准对齐（留出 shape 在所有分辨率都排除出训练）
        self.val_resolution = getattr(config, 'VAL_RESOLUTION', '10000_poisson')
        # 验证噪声级：固定、可复现的列表；缺省用单个 VAL_NOISE。想更鲁棒可设 [0.01,0.02,0.03]
        _levels = getattr(config, 'VAL_NOISE_LEVELS', None)
        self.val_noise_levels = [float(x) for x in _levels] if _levels else [float(self.val_noise)]
        self._val_names_cache = None
        self.args = args

    def _get_val_names(self):
        """确定性地从 train 里挑 VAL_NUM 个 shape 名作验证集（固定 seed → train/val 划分可复现）。
        返回 shape 名集合；训练时排除这些名、验证时只取这些名（跨所有分辨率一致）。"""
        if self._val_names_cache is not None:
            return self._val_names_cache
        if self.val_num <= 0:
            # 不留出：训练用满 train split，验证看 test（对齐 baseline）
            self._val_names_cache = set()
            return self._val_names_cache
        train_dir = os.path.join(self.root, self.dataset, 'pointclouds', 'train', self.resolutions[0])
        names = sorted([fn[:-4] for fn in os.listdir(train_dir) if fn.endswith('xyz')])
        n_val = min(self.val_num, max(0, len(names) - 1))  # 至少给训练留 1 个
        rng = random.Random(self.val_split_seed)
        rng.shuffle(names)
        val_names = set(names[:n_val])
        self._val_names_cache = val_names
        print(f'[Split] 从训练集留出 {len(val_names)} 个 shape 作验证（其余训练）: {sorted(val_names)}')
        return val_names
    
# 训练数据加载函数 划分成小点云块 添加噪声、旋转、缩放
    def train_dataloader(self):
        # 构建数据增强流程
        transforms = [
            # RandomScale([0.8, 1.2]),
            # RandomRotate(degrees=30,axis=0),      # 注释掉：旋转会导致点云与mesh坐标系不对齐，使p2m_loss失效
            # RandomRotate(degrees=30,axis=1),
            # RandomRotate(degrees=30,axis=2),
            # CleanScaleTranslateCPU(),             # 注释掉：缩放/平移同样会破坏与mesh的对齐关系
            # CleanPointcloudTranslateCPU(),
            # CleanPointcloudRotateY_CPU(),         # 注释掉：同上
        ]
       
        transforms = Compose(transforms)
    
        # 训练时排除留作验证的 shape（跨所有分辨率都排除，防泄漏）
        val_names = self._get_val_names()
        pc_datasets = [
            PointCloudDataset(root=self.root, dataset=self.dataset, split='train', resolution=resl,transform=transforms,
                              exclude_names=val_names)
            for resl in self.resolutions
        ]
        # print(pc_datasets[2][39]['pcl_clean'].shape,'pc_datasets---------------------------------------------------')
        # return
        
        # Score-based 训练的噪声模型：只用 AddNoise (纯高斯)
        # 不再用 NoisyJitter / NoisyPointDropout —— 它们违反 DSM 的"noisy = clean + N(0,σ²)"假设：
        #   * NoisyJitter 引入额外 σ_extra 噪声，σ²-加权 target 失配
        #   * NoisyPointDropout 把 3% 的 noisy 点全部替换成 noisy[0]，
        #     这些点的 target_score = (clean[i] - noisy[0])/σ² 量级可达 1e4，
        #     单点 loss 贡献 ~ σ²·target² ~ 1e5，主导整个 batch loss
        noise_tran = Compose([
            NormalizeUnitSphere(),
            AddNoise(self.noise_min, self.noise_max, log_uniform=self.noise_log_uniform),
        ])
        # 构建成对 patch 数据集
            # PairedPatchDataset 会把点云随机分割成多个 patch；
            # 每个 patch 会被动态加噪（on_the_fly=True）；
            # 每次取 patch → 输出 (pcl_clean, pcl_noisy) 成对样本；
            # 传入的 transform (AddNoise) 控制噪声生成。
        # train_dset包含120个索引，0-39是10k，40-79是30k，80-119是50k
        train_dset = PairedPatchDataset(
            datasets=pc_datasets, patch_size=self.patch_size, num_patches=self.num_patches,
            patch_ratio=1.0, on_the_fly=True, noise_min=self.noise_min, noise_max=self.noise_max,
            transform=noise_tran, flag='train', oversample_factor=self.train_oversample,
        )
        # print(train_dset[0]['pcl_noisy'].shape,train_dset[0]['pcl_clean'].shape,'----------------------------')
        # return
        if self.args.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(
                train_dset, shuffle=True)
            dataloader = DataLoader(train_dset, batch_size=self.train_batch_size, num_workers=self.num_workers,drop_last=True,collate_fn=denoise_collate_fn_test,worker_init_fn=worker_init_fn,sampler=sampler)
        else:
            sampler = None
            dataloader = DataLoader(train_dset, batch_size=self.train_batch_size,shuffle=True, num_workers=self.num_workers,drop_last=True,collate_fn=denoise_collate_fn_test)

        return sampler,dataloader
# 验证数据加载函数：返回【从训练集留出的完整点云】+ 固定噪声（独立于 test，防泄漏）。
# runner 的 validate() 会对这些完整云跑 test 同款 pipeline（整云切块→Langevin→拼接→SOR），
# 算 CD+P2M 并按 cd+0.3·p2m 选 ckpt-best —— 选模型标准 = 报告标准，选出的模型最贴近 test 最优。
# 验证集不切分片（DDP 下也让每个 rank 跑完整 val，样本少、且噪声固定→各 rank 结果一致，
# rank0 的 checkpoint 决策才正确）。
    def val_dataloader(self):
        if self.val_num > 0:
            # 从训练集留出验证 shape（方法学更干净，但训练数据变少、且与 baseline 协议不同）；
            # 这些 shape 已在 train_dataloader 里被排除出训练。P2M 用 meshes/train/。
            val_names = self._get_val_names()
            clean_dset = PointCloudDataset(
                root=self.root, dataset=self.dataset, split='train',
                resolution=self.val_resolution, include_names=val_names)
        else:
            # VAL_NUM<=0（默认）：验证用 test split 干净云 + 固定噪声，与 ScoreDenoise / PD-Flow 对齐
            #（它们都是 train 全用、验证看 test）。训练数据量与选模型协议都对齐 baseline → 公平对比。
            #  P2M 用 meshes/test/（validate() 里按 VAL_NUM 决定 split）。
            clean_dset = PointCloudDataset(
                root=self.root, dataset=self.dataset, split='test',
                resolution=self.val_resolution)
        val_dset = HeldOutValDataset(clean_dset, self.val_noise_levels, seed=self.val_split_seed)
        dataloader = DataLoader(
            val_dset, batch_size=1, shuffle=False, num_workers=self.num_workers,
            drop_last=False, collate_fn=denoise_collate_fn_test)
        return None, dataloader
    def test_dataloader(self):
        # 从 config 读分辨率和带噪目录，不想改代码时只改 ScoreDenoise.yaml
        test_resolution = getattr(self, 'test_resolution', '10000_poisson')
        test_noisy_dir_name = getattr(self, 'test_noisy_dir', f'PUNet_{test_resolution}_0.01')
        # 干净点云：TEST_CLEAN_PATH 若设置则用 <它>/<分辨率>（换数据集，如 PCNet）；否则 PUNet 默认
        test_clean_path = getattr(self, 'test_clean_path', None)
        if test_clean_path:
            clean_pcl_dir = os.path.join(test_clean_path, test_resolution)
            clean_dset = PointCloudDataset(
                root=self.root, dataset='PUNet', split='test', resolution=test_resolution,
                pcl_dir=clean_pcl_dir
            )
            print(f'[INFO] TEST_CLEAN_PATH 生效，干净点云取自: {clean_pcl_dir}')
        else:
            clean_dset = PointCloudDataset(
                root=self.root, dataset='PUNet', split='test', resolution=test_resolution
            )
        # 带噪目录：TEST_NOISY_PATH 若设置则直接用它（泛化实验换测试集，指向任意文件夹）；
        # 否则用默认 <root>/examples/pointclouds/test/<TEST_NOISY_DIR>
        test_noisy_path = getattr(self, 'test_noisy_path', None)
        if test_noisy_path:
            noisy_dir = test_noisy_path
            print(f'[INFO] TEST_NOISY_PATH 生效，带噪点云取自: {noisy_dir}')
        else:
            noisy_dir = os.path.join(
                self.root, 'examples', 'pointclouds', 'test', test_noisy_dir_name
            )
        noisy_dset = PointCloudNoisyDataset(noisy_dir=noisy_dir)
        val_dset = PairedEvalDataset(clean_dataset=clean_dset, noisy_dataset=noisy_dset)

        if self.args.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(
                val_dset, shuffle=False)
            dataloader = DataLoader(val_dset, batch_size=1,num_workers=self.num_workers, drop_last=False,collate_fn=denoise_collate_fn_test,worker_init_fn=worker_init_fn,sampler=sampler)
        else:
            sampler = None
            dataloader = DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=self.num_workers,drop_last=False,collate_fn=denoise_collate_fn_test)
        return sampler,dataloader
