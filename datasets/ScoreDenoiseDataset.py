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

    def __init__(self, root, dataset, split, resolution, transform=None):
        super().__init__()
        self.pcl_dir = os.path.join(root, dataset, 'pointclouds', split, resolution)
        self.transform = transform
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
        
        print(f'[INFO] Loaded dataset {dataset} - {resolution}')

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

    return noisy, clean, center_list, scale_list, name_list
    
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

    def __init__(self, datasets, patch_ratio, on_the_fly=True,patch_size=1000, num_patches=1000, noise_min=0, noise_max=0,transform=None,flag='train'):
        super().__init__()
        self.datasets = datasets
        self.all_data = [] #长度120，包含了10k，30k，50k的数据
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
        self.flag=flag
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
        return len(self.all_data)

    def __getitem__(self, idx):
        # all_data 是 10k/30k/50k 三个分辨率拼起来的所有干净点云
        if self.on_the_fly:
            pcl_data = self.all_data[idx]
            data = {
                'pcl_clean': pcl_data['pcl_clean'].clone(),
                'name': pcl_data['name'],
            }
            if self.transform is not None:
                data = self.transform(data)
                data.pop('noise_std', None)

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
        self.args = args
    
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
    
        pc_datasets = [
            PointCloudDataset(root=self.root, dataset=self.dataset, split='train', resolution=resl,transform=transforms)
            for resl in self.resolutions
        ]
        # print(pc_datasets[2][39]['pcl_clean'].shape,'pc_datasets---------------------------------------------------')
        # return
        
        # 定义噪声添加变换
        noise_tran = Compose([
            NormalizeUnitSphere(),
            AddNoise(self.noise_min, self.noise_max),
            NoisyJitter(sigma=0.002, clip=0.01),
            NoisyPointDropout(max_dropout_ratio=0.03),
        ])
        # 构建成对 patch 数据集
            # PairedPatchDataset 会把点云随机分割成多个 patch；
            # 每个 patch 会被动态加噪（on_the_fly=True）；
            # 每次取 patch → 输出 (pcl_clean, pcl_noisy) 成对样本；
            # 传入的 transform (AddNoise) 控制噪声生成。
        # train_dset包含120个索引，0-39是10k，40-79是30k，80-119是50k
        train_dset = PairedPatchDataset(datasets=pc_datasets,patch_size=self.patch_size, num_patches=self.num_patches, patch_ratio=1.0, on_the_fly=True,noise_min=self.noise_min, noise_max=self.noise_max,transform=noise_tran)
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
# 验证数据加载函数 固定噪声水平、不旋转、不缩放、不切割成小点云块，将整个点云数据输入模型测试，方便计算性能指标
    def val_dataloader(self):
        val_split = 'val'
        val_dir = os.path.join(self.root, self.dataset, 'pointclouds', val_split)
        if not os.path.isdir(val_dir):
            print(f"[WARN] split '{val_split}' 不存在，验证集回退到 test split")
            val_split = 'test'

        val_dset = [
            PointCloudDataset(root=self.root, dataset=self.dataset, split=val_split, resolution=resl)
            for resl in ['10000_poisson']
        ]
        # print(val_dset[0][19],'pc_datasets---------------------------------------------------')
        # return
        transform = standard_train_transforms(noise_std_min=self.val_noise, noise_std_max=self.val_noise, rotate=False, scale_d=0.0)
        # val_dset = PointCloudDataset(root=self.root, dataset=self.dataset, split='test', resolution='50000_poisson')
        val_dset = PairedPatchDataset(datasets=val_dset,patch_size=self.patch_size, num_patches=self.num_patches, patch_ratio=1.0, on_the_fly=True,noise_min=self.noise_min, noise_max=self.noise_max,transform=transform,flag='test')
        
        # print(val_dset[0],'=======================================val_test')
        # return
        if self.args.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(
                val_dset, shuffle=False)
            dataloader = DataLoader(val_dset, batch_size=1, num_workers=self.num_workers,drop_last=False,collate_fn=denoise_collate_fn_test,worker_init_fn=worker_init_fn,sampler=sampler)
        else:
            # val_dset = PairedPatchDataset(datasets=val_dset, patch_ratio=1.0, on_the_fly=True,noise_min=self.val_noise, noise_max=self.val_noise)
            sampler = None
            dataloader = DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=self.num_workers, drop_last=False,collate_fn=denoise_collate_fn_test)
            # dataloader = DataLoader(val_dset, batch_size=self.train_batch_size, shuffle=False, num_workers=self.num_workers, drop_last=False,collate_fn=denoise_collate_fn)
        return sampler,dataloader
    def test_dataloader(self):
        clean_dset = PointCloudDataset(
            root=self.root, dataset='PUNet', split='test', resolution='10000_poisson'
        )
        noisy_dir = os.path.join(
            self.root, 'examples', 'pointclouds', 'test', 'PUNet_10000_poisson_0.01'
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
