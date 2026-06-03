# 在训练前动态生成带噪点云，并对输入数据进行归一化、旋转、缩放等预处理
import math
import random
import numbers
from numpy.core.fromnumeric import size
import torch
import numpy as np
from torchvision.transforms import Compose

# -----------------------------scoredenoise数据集只用到了以下处理---------------------------------
# 把点云归一化到单位球内（防止尺度不一致）
class NormalizeUnitSphere(object):

    def __init__(self):
        super().__init__()

    @staticmethod
    def normalize(pcl, center=None, scale=None):
        """
        Args:
            pcl:  The point cloud to be normalized, (N, 3)
        """
        if center is None:
            p_max = pcl.max(dim=0, keepdim=True)[0]
            p_min = pcl.min(dim=0, keepdim=True)[0]
            center = (p_max + p_min) / 2    # (1, 3)
        pcl = pcl - center
        if scale is None:
            scale = (pcl ** 2).sum(dim=1, keepdim=True).sqrt().max(dim=0, keepdim=True)[0]  # (1, 1)
        pcl = pcl / scale
        return pcl, center, scale

    def __call__(self, data):
        assert 'pcl_noisy' not in data, 'Point clouds must be normalized before applying noise perturbation.'
        pcl_clean_norm, center, scale = self.normalize(data['pcl_clean'])
        data['pcl_clean'] = pcl_clean_norm
        data['center'] = center
        data['scale'] = scale
        return data

# 各向同性高斯噪声 常见的白噪声模型
# class AddNoise(object):

#     def __init__(self, noise_std_min, noise_std_max):
#         super().__init__()
#         self.noise_std_min = noise_std_min
#         self.noise_std_max = noise_std_max

#     def __call__(self, data):
#         print('=====================================')
#         noise_std = random.uniform(self.noise_std_min, self.noise_std_max)
        
#         data['pcl_noisy'] = data['pcl_clean'] + torch.randn_like(data['pcl_clean']) * noise_std
#         if 'pcl_clean_50k' in data:
#             data['pcl_noisy_50k'] = data['pcl_clean_50k'] + torch.randn_like(data['pcl_clean_50k']) * noise_std * data['scale'].item()
#         data['noise_std'] = noise_std
#         return data

# 随机缩放 
class RandomScale(object):

    def __init__(self, scales):
        assert isinstance(scales, (tuple, list)) and len(scales) == 2
        self.scales = scales

    def __call__(self, data):
        scale = random.uniform(*self.scales)
        data['pcl_clean'] = data['pcl_clean'] * scale
        if 'pcl_noisy' in data:
            data['pcl_noisy'] = data['pcl_noisy'] * scale
        return data

# 随机旋转
class RandomRotate(object):

    def __init__(self, degrees=15.0, axis=0):
        if isinstance(degrees, numbers.Number):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axis = axis

    def __call__(self, data):
        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if self.axis == 0:
            matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
        elif self.axis == 1:
            matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
        else:
            matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        matrix = torch.tensor(matrix)

        device = data['pcl_clean'].device
        matrix = matrix.to(device)
        data['pcl_clean'] = torch.matmul(data['pcl_clean'], matrix)

        if 'pcl_noisy' in data:
            data['pcl_noisy'] = torch.matmul(data['pcl_noisy'], matrix)

        return data
# 随机旋转
# class RandomRotate(object):

#     def __init__(self, degrees=180.0, axis=0):
#         if isinstance(degrees, numbers.Number):
#             degrees = (-abs(degrees), abs(degrees))
#         assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
#         self.degrees = degrees
#         self.axis = axis

#     def __call__(self, data):
#         degree = math.pi * random.uniform(*self.degrees) / 180.0
#         sin, cos = math.sin(degree), math.cos(degree)

#         if self.axis == 0:
#             matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
#         elif self.axis == 1:
#             matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
#         else:
#             matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
#         matrix = torch.tensor(matrix)

#         data['pcl_clean'] = torch.matmul(data['pcl_clean'], matrix)
#         if 'pcl_noisy' in data:
#             data['pcl_noisy'] = torch.matmul(data['pcl_noisy'], matrix)

#         return data

# 平移+缩放点云
class CleanScaleTranslateCPU(object):
    def __init__(self, scale_low=0.9, scale_high=1.1, translate_range=0.02):
    # def __init__(self, scale_low=2. / 3., scale_high=3. / 2., translate_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, sample):
        # sample 是 dict
        clean = sample["pcl_clean"].cpu()

        # 兼容 [N,3] 或 [B,N,3]
        if clean.dim() == 2:
            clean = clean.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        B = clean.shape[0]

        # (B,1,3) 生成批量随机增强参数
        scales = torch.empty(B, 1, 3).uniform_(self.scale_low, self.scale_high)
        shifts = torch.empty(B, 1, 3).uniform_(-self.translate_range, self.translate_range)

        # 不做 in-place，避免 autograd 问题
        clean = clean * scales + shifts

        if squeeze_back:
            clean = clean.squeeze(0)

        sample["pcl_clean"] = clean
        return sample

# 平移点云
class CleanPointcloudTranslateCPU(object):
    def __init__(self, translate_range=0.2):
        self.translate_range = translate_range

    def __call__(self, sample):
        # 只处理干净点云
        pc = sample["pcl_clean"]

        # 兼容 [N,3] / [B,N,3]
        if pc.dim() == 2:
            pc = pc.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        bsize = pc.size(0)

        # (B,1,3) — 每个样本一个平移向量
        shifts = torch.empty(bsize, 1, 3).uniform_(
            -self.translate_range, self.translate_range
        )

        # 不做 in-place，避免 autograd 问题
        pc = pc + shifts

        if squeeze_back:
            pc = pc.squeeze(0)

        sample["pcl_clean"] = pc
        return sample

# 绕 Y 轴旋转点云
class CleanPointcloudRotateY_CPU(object):
    def __call__(self, sample):
        """
        sample: dict
            {
                'pcl_clean': [N,3] 或 [B,N,3]
                ... 其它字段保持不变
            }
        """
        pc = sample["pcl_clean"]

        # 统一成 [B,N,3] 处理
        if pc.dim() == 2:
            pc = pc.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False

        B = pc.size(0)

        rotated = []
        for i in range(B):
            # 随机角度
            angle = np.random.uniform() * 2 * np.pi
            cosv = np.cos(angle)
            sinv = np.sin(angle)

            # 绕 Y 轴旋转矩阵
            R = torch.tensor([
                [cosv, 0,  sinv],
                [0,    1,  0   ],
                [-sinv,0,  cosv]
            ], dtype=torch.float32)

            # (N,3) @ (3,3)
            pi = torch.matmul(pc[i], R)
            rotated.append(pi)

        pc = torch.stack(rotated, dim=0)

        if squeeze_back:
            pc = pc.squeeze(0)

        # 只更新干净点云
        sample["pcl_clean"] = pc
        return sample

# ----------------------------------以下处理没用到-----------------------------------------

# 各向异性高斯噪声 协方差可控，方向性噪声
# non-isotropic Gaussian noise
class AddCovNoise(object):

    def __init__(self, cov, std_factor=1.0):
        super().__init__()
        self.cov = torch.FloatTensor(cov)
        self.std_factor = std_factor

    def __call__(self, data):
        num_points = data['pcl_clean'].shape[0]
        noise = np.random.multivariate_normal(np.zeros(3), self.cov.numpy(), num_points) # (N, 3)
        noise = torch.FloatTensor(noise).to(data['pcl_clean'])
        data['pcl_noisy'] = data['pcl_clean'] + noise * self.std_factor
        data['noise_std'] = self.std_factor
        return data

# 单方向噪声 仅在 x 轴添加扰动
# Uni-directional Noise
class AddUniDirectional(object):

    def __init__(self, std_factor=1.0):
        super().__init__()
        self.std_factor = std_factor

    def __call__(self, data):
        num_points = data['pcl_clean'].shape[0]
        noise = np.random.normal(loc=0.0, scale=self.std_factor, size=(num_points,))
        noise = torch.FloatTensor(noise).to(data['pcl_clean'])
        data['pcl_noisy'] = torch.clone(data['pcl_clean'])
        data['pcl_noisy'][:, 0] += noise
        data['noise_std'] = self.std_factor
        return data

# 拉普拉斯噪声 稀疏、尖峰噪声
# Laplace noise
class AddLaplacianNoise(object):

    def __init__(self, noise_std_min, noise_std_max):
        super().__init__()
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max

    def __call__(self, data):
        noise_std = random.uniform(self.noise_std_min, self.noise_std_max)
        noise = torch.FloatTensor(np.random.laplace(0, noise_std, size=data['pcl_clean'].shape)).to(data['pcl_clean'])
        data['pcl_noisy'] = data['pcl_clean'] + noise
        data['noise_std'] = noise_std
        return data


# uniform noise 单个gpu版本
class AddNoise(object):
    def __init__(self, noise_std_min, noise_std_max):
        super().__init__()
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max

    def __call__(self, data):
        # 随机噪声标准差
        noise_std = random.uniform(self.noise_std_min, self.noise_std_max)
        pcl_clean = data['pcl_clean'].float().contiguous()

        if pcl_clean.numel() == 0:
            raise ValueError("pcl_clean is empty!")

        # 归一化坐标系内加噪
        noise = torch.randn_like(pcl_clean) * noise_std
        data['pcl_noisy'] = pcl_clean + noise
        data['noise_std'] = noise_std
        return data


class NoisyJitter(object):
    """
    仅对 noisy 点云做轻量抖动，保持 clean/mesh 坐标系不变。
    """
    def __init__(self, sigma=0.002, clip=0.01):
        super().__init__()
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data):
        if 'pcl_noisy' not in data:
            return data
        jitter = torch.randn_like(data['pcl_noisy']) * self.sigma
        jitter = torch.clamp(jitter, -self.clip, self.clip)
        data['pcl_noisy'] = data['pcl_noisy'] + jitter
        return data


class NoisyPointDropout(object):
    """
    仅对 noisy 点云做随机点替换（PointNet 常见增强），提升抗稀疏异常能力。
    """
    def __init__(self, max_dropout_ratio=0.03):
        super().__init__()
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, data):
        if 'pcl_noisy' not in data:
            return data
        pcl_noisy = data['pcl_noisy']
        num_points = pcl_noisy.shape[0]
        dropout_ratio = random.uniform(0.0, self.max_dropout_ratio)
        if dropout_ratio <= 0.0 or num_points <= 1:
            return data
        drop_idx = torch.rand(num_points, device=pcl_noisy.device) < dropout_ratio
        if drop_idx.any():
            pcl_noisy = pcl_noisy.clone()
            # 避免 source/target 共享存储导致的重叠写入报错
            keep_point = pcl_noisy[0].clone()
            pcl_noisy[drop_idx] = keep_point
            data['pcl_noisy'] = pcl_noisy
        return data
    # 多个gpu版本
# class AddNoise(object):
#     def __init__(self, noise_std_min, noise_std_max):
#         self.noise_std_min = noise_std_min
#         self.noise_std_max = noise_std_max

#     def __call__(self, data):
#         # 随机噪声标准差
#         noise_std = random.uniform(self.noise_std_min, self.noise_std_max)

#         pcl_clean = data['pcl_clean'].float().contiguous()  # 保持在 CPU

#         # 添加噪声
#         noise = torch.randn_like(pcl_clean) * noise_std
#         data['pcl_noisy'] = pcl_clean + noise

#         data['pcl_clean'] = pcl_clean  # 保证原点云也在 GPU
#         data['noise_std'] = noise_std
#         return data

# 球面均匀噪声 模拟扫描误差
class AddUniformBallNoise(object):
    
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def __call__(self, data):
        N = data['pcl_clean'].shape[0]
        phi = np.random.uniform(0, 2*np.pi, size=N)
        costheta = np.random.uniform(-1, 1, size=N)
        u = np.random.uniform(0, 1, size=N)
        theta = np.arccos(costheta)
        r = self.scale * u ** (1/3)

        noise = np.zeros([N, 3])
        noise[:, 0] = r * np.sin(theta) * np.cos(phi)
        noise[:, 1] = r * np.sin(theta) * np.sin(phi)
        noise[:, 2] = r * np.cos(theta)
        noise = torch.FloatTensor(noise).to(data['pcl_clean'])
        data['pcl_noisy'] = data['pcl_clean'] + noise
        return data

# 离散方向噪声 模拟采样突变或量化误差
# discrete noise
class AddDiscreteNoise(object):

    def __init__(self, scale, prob=0.1):
        super().__init__()
        self.scale = scale
        self.prob = prob
        self.template = np.array([
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ], dtype=np.float32)

    def __call__(self, data):
        num_points = data['pcl_clean'].shape[0]
        uni_rand = np.random.uniform(size=num_points)
        noise = np.zeros([num_points, 3])
        for i in range(self.template.shape[0]):
            idx = np.logical_and(0.1*i <= uni_rand, uni_rand < 0.1*(i+1))
            noise[idx] = self.template[i].reshape(1, 3)
        noise = torch.FloatTensor(noise).to(data['pcl_clean'])
        # print(data['pcl_clean'])
        # print(self.scale)
        data['pcl_noisy'] = data['pcl_clean'] + noise * self.scale
        data['noise_std'] = self.scale
        return data





def standard_train_transforms(noise_std_min, noise_std_max, scale_d=0.2, rotate=True):
    # Score-based 噪声模型：只用 NormalizeUnitSphere + AddNoise
    # NoisyJitter / NoisyPointDropout 已移除（违反 DSM 假设，详见 ScoreDenoiseDataset.train_dataloader 注释）
    transforms = [
        NormalizeUnitSphere(),
        AddNoise(noise_std_min=noise_std_min, noise_std_max=noise_std_max),
        # RandomScale([1.0-scale_d, 1.0+scale_d]),
    ]
    if rotate:
        transforms += [
            RandomRotate(axis=0),
            RandomRotate(axis=1),
            RandomRotate(axis=2),
        ]
    return Compose(transforms)
