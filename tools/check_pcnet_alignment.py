# -*- coding: utf-8 -*-
"""PCNet 泛化测试数据对齐自检（只查数据/协议，不跑模型）。

用法（仓库根目录下执行）：
    python tools/check_pcnet_alignment.py                       # 10k / 1%
    python tools/check_pcnet_alignment.py --res 50000 --noise 0.03
    python tools/check_pcnet_alignment.py --clean-dir ... --noisy-dir ... --mesh-root ...

检查项与判读：
  ① clean↔mesh 坐标系：clean 点云对 mesh 的 P2M(×1e4)，应 ≈0（Poisson 采样自表面，
     通常 <0.05）。明显偏大 = clean 点云与 mesh 不同源/不同坐标系 → 所有方法在你这套
     数据上的 P2M 都被抬高一个底数，与论文报告的数字不可比。
  ② 加噪约定：由 noisy−clean 位移反推每坐标 σ（相对单位球半径，score-denoise 系约定），
     应 ≈ 名义值（0.01/0.02/0.03）。偏差 >20% = 加噪脚本的归一化约定不一致，
     实际测的噪声比表格标称的更大/更小。
  ③ 配对完整性：clean / noisy / mesh 三方同名文件数量（PCNet test 应为 10 个 shape）。
  ④ SOR 敏感性：对 clean 云做 test() 同参的 SOR(20, 2.0)，统计删点数与删点后 P2M。
     PCNet 多为带尖锐边角的 CAD 形状，边角点易被 SOR 当离群点误删；双向 P2M 的
     face→point 项会因覆盖空洞被抬高（对照方法都不做后处理滤波）。
     若"SOR 后 clean P2M"相对①明显升高，说明 SOR 在这套数据上是减分项。
"""
import os
import sys
import argparse

import numpy as np
import torch

# 仓库根加入 path，便于 python tools/check_pcnet_alignment.py 直接跑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.p2m_loss as p2ml           # noqa: E402
from utils.p2m_loss import compute_p2m  # noqa: E402


def load_xyz(path):
    return torch.from_numpy(np.loadtxt(path, dtype=np.float32))


def unit_sphere_radius(pc):
    """与 NormalizeUnitSphere/normalize_sphere 同款：bbox 中心 + 最大半径"""
    c = (pc.max(0)[0] + pc.min(0)[0]) / 2
    r = (pc - c).norm(dim=1).max().clamp_min(1e-8)
    return c, r


def sor_keep(pts, nb_neighbors=20, std_ratio=2.0):
    """与 runner_finetune.sor_filter 同参的 SOR，返回保留的点"""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.numpy().astype(np.float64))
    _, keep_idx = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pts[np.asarray(keep_idx)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--res', default='10000', help='10000 / 50000')
    ap.add_argument('--noise', default='0.01', help='名义噪声级 0.01/0.02/0.03')
    ap.add_argument('--clean-dir', default=None)
    ap.add_argument('--noisy-dir', default=None)
    ap.add_argument('--mesh-root', default='data/PCNet/meshes')
    args = ap.parse_args()

    clean_dir = args.clean_dir or f'data/PCNet/pointclouds/test/{args.res}_poisson'
    noisy_dir = args.noisy_dir or f'data/PCNet/noisy/PCNet_{args.res}_poisson_{args.noise}'
    p2ml._MESH_ROOT = args.mesh_root
    nominal = float(args.noise)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    names_c = {f[:-4] for f in os.listdir(clean_dir) if f.endswith('xyz')}
    names_n = {f[:-4] for f in os.listdir(noisy_dir) if f.endswith('xyz')}
    mesh_dir = os.path.join(args.mesh_root, 'test')
    names_m = {os.path.splitext(f)[0] for f in os.listdir(mesh_dir)
               if f.endswith('.off')}
    names = sorted(names_c & names_n & names_m)

    print(f'clean: {clean_dir}')
    print(f'noisy: {noisy_dir}')
    print(f'mesh : {mesh_dir}\n')
    print(f'③ 配对检查: clean={len(names_c)}  noisy={len(names_n)}  '
          f'mesh={len(names_m)}  三方同名={len(names)}')
    missing = (names_c | names_n | names_m) - set(names)
    if missing:
        print(f'   ⚠ 未三方配对的名字（会被测试静默丢弃）: {sorted(missing)}')
    print()

    p2m_clean_list, sigma_list, removed_list, p2m_sor_list = [], [], [], []
    n_total = 0
    for nm in names:
        clean = load_xyz(os.path.join(clean_dir, nm + '.xyz'))
        noisy = load_xyz(os.path.join(noisy_dir, nm + '.xyz'))
        _, r = unit_sphere_radius(clean)
        n_total = clean.shape[0]

        # ① clean 对 mesh 的 P2M（应≈0）
        p2m_clean = compute_p2m(clean.to(dev), nm, 'test').item() * 1e4

        # ② 反推噪声：成对（同名同序）位移；若行数不同回退最近邻（只作下界参考）
        if noisy.shape == clean.shape:
            d = (noisy - clean).norm(dim=1)
            tag = '成对'
        else:
            d = torch.cdist(noisy.to(dev), clean.to(dev)).min(dim=1).values.cpu()
            tag = 'NN下界'
        # 每坐标 σ = RMS(位移范数)/√3，再除单位球半径 → 与名义 σ 同一约定
        sigma_est = float((d.pow(2).mean() / 3).sqrt() / r)

        # ④ SOR 敏感性（作用在 clean 上，隔离模型因素）
        kept = sor_keep(clean)
        removed = clean.shape[0] - kept.shape[0]
        p2m_sor = compute_p2m(kept.to(dev), nm, 'test').item() * 1e4

        p2m_clean_list.append(p2m_clean)
        sigma_list.append(sigma_est)
        removed_list.append(removed)
        p2m_sor_list.append(p2m_sor)
        print(f'{nm:16s} ①cleanP2M={p2m_clean:8.4f}  ②σ[{tag}]={sigma_est:.5f}'
              f' (名义 {nominal})  ④SOR删点={removed:4d}  SOR后cleanP2M={p2m_sor:8.4f}')

    if not names:
        print('没有可检查的配对样本，请核对路径')
        return

    print('\n===== 汇总（判读见文件头注释）=====')
    m_clean = float(np.mean(p2m_clean_list))
    m_sigma = float(np.mean(sigma_list))
    m_rm = float(np.mean(removed_list))
    m_sor = float(np.mean(p2m_sor_list))
    print(f'① clean P2M 均值 = {m_clean:.4f} ×1e4'
          f'   → ≳0.1 说明 clean/mesh 不对齐，全体 P2M 虚高该底数')
    print(f'② 反推 σ 均值 = {m_sigma:.5f}（名义 {nominal}，偏差 '
          f'{(m_sigma / nominal - 1) * 100:+.1f}%）→ |偏差|>20% 说明加噪约定不一致')
    print(f'④ SOR 平均删点 = {m_rm:.1f}/{n_total}，SOR 后 clean P2M 均值 = {m_sor:.4f}'
          f'（相对①变化 {m_sor - m_clean:+.4f}）'
          f' → 明显升高说明 SOR 在删边角点、抬高 face→point 项')


if __name__ == '__main__':
    main()
