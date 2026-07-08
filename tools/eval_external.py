# -*- coding: utf-8 -*-
"""用【本仓库 test() 完全同款】的 CD / P2M 评测任意方法的去噪输出。

动机：让所有对比方法过同一把尺子。本仓库 test() 的 CD = ChamferDistanceL2（按干净云
单位球归一化），P2M = compute_p2m（双向 point↔face，mesh 归一化到单位球），均 ×1e4。
把别的方法的去噪 .xyz 也用这套算，整张对比表就都在同一口径下，最公平。

与 test() 的唯一区别：**不加 SOR / 不做任何后处理** —— 别人的输出是其最终结果，
按原样评测（你自己的方法也应报告无 SOR 的数，见 sor_enable 开关）。

用法（仓库根目录、你的训练环境下跑，需要已编译的 extensions.chamfer_dist + pytorch3d）：
    python tools/eval_external.py \
        --clean_dir data/PCNet/pointclouds/test/10000_poisson \
        --mesh_root data/PCNet/meshes \
        --pred_dir /path/methodA/10k_0.01  /path/methodB/10k_0.01 \
        --out_csv pcnet_10k_0.01_external.csv

- --clean_dir : 干净 GT 点云目录（按分辨率，如 .../test/10000_poisson）
- --mesh_root : mesh 根目录，其下应有 test/<name>.off（compute_p2m 用 <root>/test/<name>.off）
- --pred_dir  : 一个或多个方法的去噪 .xyz 目录；方法名默认取目录 basename
- 文件按同名（去 .xyz 后缀）与 clean/mesh 配对；点数可与 GT 不同（CD/P2M 都支持）
- .xyz 若多于 3 列（含法向等），只取前 3 列

前提：别人的去噪点云必须与干净 GT 在**同一世界坐标系**（多数去噪 repo 的输出都做了
反归一化到原始坐标，如 IterativePFN test.py 的 `pcl * scale + center`）。若某 shape 的
CD 异常巨大，多半是坐标系不一致（对方在单位球空间保存），需先对齐再评。
"""
import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from extensions.chamfer_dist import ChamferDistanceL2   # noqa: E402
import utils.p2m_loss as p2ml                            # noqa: E402
from utils.p2m_loss import compute_p2m                   # noqa: E402


def normalize_unit_sphere(pc, radius=1.0):
    """与 runner_finetune.normalize_unit_sphere 完全一致。pc: (B, N, 3)。"""
    p_max = pc.max(dim=-2, keepdim=True)[0]
    p_min = pc.min(dim=-2, keepdim=True)[0]
    center = (p_max + p_min) / 2                      # (B,1,3)
    pc = pc - center
    scale = (pc ** 2).sum(dim=-1, keepdim=True).sqrt().max(dim=-2, keepdim=True)[0] / radius  # (B,1,1)
    return pc / scale, center, scale


def load_xyz_dir(d):
    """读一个目录下所有 .xyz → {name: [N,3] tensor}（只取前 3 列）。"""
    out = {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith('.xyz'):
            continue
        arr = np.loadtxt(os.path.join(d, fn), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        out[fn[:-4]] = torch.from_numpy(arr[:, :3]).float()
    return out


def eval_one_method(pred_dir, clean, mesh_names, cd_metric, device, verbose=True):
    """对一个方法目录评测，返回 (mean_cd, mean_p2m, n)。"""
    pred = load_xyz_dir(pred_dir)
    names = sorted(set(pred) & set(clean) & set(mesh_names))
    missing = (set(pred) & set(clean)) - set(mesh_names)
    if missing:
        print(f'  ⚠ 缺 mesh，跳过 P2M 但仍算 CD: {sorted(missing)}')
    if not names and not missing:
        print(f'  ⚠ {pred_dir} 与 clean/mesh 无同名配对，检查文件名/分辨率')
    eval_names = sorted(set(pred) & set(clean))  # CD 只需 clean，P2M 另判 mesh

    cds, p2ms = [], []
    for nm in eval_names:
        clean_w = clean[nm].unsqueeze(0).to(device)      # [1,N,3]
        pred_w = pred[nm].unsqueeze(0).to(device)        # [1,M,3]
        # CD：按干净云单位球归一化两者，再 ChamferDistanceL2 ×1e4（与 test() 一致）
        _, c1, s1 = normalize_unit_sphere(clean_w)
        cd = cd_metric((pred_w - c1) / s1, (clean_w - c1) / s1).item() * 1e4
        cds.append(cd)
        # P2M：世界坐标直接喂 compute_p2m（内部按 mesh 归一化），×1e4
        p2m = float('nan')
        if nm in mesh_names:
            try:
                p2m = compute_p2m(pred_w[0], nm, 'test').item() * 1e4
                p2ms.append(p2m)
            except Exception as e:
                print(f'  ⚠ {nm} P2M 失败: {e}')
        if verbose:
            print(f'    {nm:16s} CD={cd:9.4f}  P2M={p2m:9.4f}  (pred {pred[nm].shape[0]} pts)')

    mcd = float(np.mean(cds)) if cds else float('nan')
    mp2m = float(np.mean(p2ms)) if p2ms else float('nan')
    return mcd, mp2m, len(cds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clean_dir', required=True, help='干净 GT 点云目录（按分辨率）')
    ap.add_argument('--mesh_root', required=True, help='mesh 根目录（其下 test/<name>.off）')
    ap.add_argument('--pred_dir', required=True, nargs='+', help='一个或多个方法的去噪 .xyz 目录')
    ap.add_argument('--out_csv', default=None, help='可选：把各方法均值写入 CSV')
    ap.add_argument('--quiet', action='store_true', help='不打印逐 shape 明细')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    p2ml._MESH_ROOT = args.mesh_root
    cd_metric = ChamferDistanceL2().to(device) if device == 'cuda' else ChamferDistanceL2()

    clean = load_xyz_dir(args.clean_dir)
    mesh_dir = os.path.join(args.mesh_root, 'test')
    mesh_names = {os.path.splitext(f)[0] for f in os.listdir(mesh_dir) if f.endswith('.off')}
    print(f'clean: {args.clean_dir}  ({len(clean)} 云)')
    print(f'mesh : {mesh_dir}  ({len(mesh_names)} 个)\n')

    rows = []
    for pd in args.pred_dir:
        name = os.path.basename(pd.rstrip('/'))
        print(f'======== {name} ({pd}) ========')
        mcd, mp2m, n = eval_one_method(pd, clean, mesh_names, cd_metric, device,
                                       verbose=not args.quiet)
        print(f'  >>> {name}: CD={mcd:.4f}  P2M={mp2m:.4f}  ({n} 云)\n')
        rows.append((name, mcd, mp2m, n))

    print('======== 汇总（本仓库口径 CD/P2M，×1e4，无 SOR）========')
    print(f'{"method":24s} {"CD":>10s} {"P2M":>10s} {"n":>4s}')
    for name, mcd, mp2m, n in rows:
        print(f'{name:24s} {mcd:10.4f} {mp2m:10.4f} {n:4d}')

    if args.out_csv:
        import csv
        with open(args.out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['method', 'CD', 'P2M', 'n'])
            w.writerows(rows)
        print(f'\n已写入 {args.out_csv}')


if __name__ == '__main__':
    main()
