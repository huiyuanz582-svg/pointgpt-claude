# -*- coding: utf-8 -*-
"""单文件·可移植版：给对比方法的去噪 .xyz 加同款 SOR 后，用与本仓库 test() 一致的
CD / P2M 口径评测。**不依赖 extensions/chamfer_dist（无需编译 CUDA 扩展）**，方便在
放着别人去噪结果的另一台服务器上直接跑。

口径对齐说明（与 utils/p2m_loss.compute_p2m、runner_finetune.test() 一致）：
- CD  = ChamferDistanceL2：两方向最近邻【平方】距离的均值之和，
        在【按干净云单位球归一化】的空间里计算，再 ×1e4。
        本文件用 scipy KDTree 复现该公式（数值上等价于本仓库的 CUDA ChamferDistanceL2）。
        若本机恰好能 import extensions.chamfer_dist，则优先用真扩展（更保险）。
- P2M = pytorch3d.loss.point_mesh_face_distance：mesh 归一化到单位球、预测点施加同一变换，
        双向 point↔face，再 ×1e4。与仓库 compute_p2m 完全同一调用。
- SOR = open3d remove_statistical_outlier(nb=20, std=2.0)，与 runner sor_filter 一致。
        SOR 对全局缩放/平移不变，故在世界坐标系对预测点加 SOR，与仓库在归一化空间加 SOR
        删除的是同一批点，口径不冲突。

依赖（另一台服务器需要）：
    必需： numpy, scipy, open3d
    P2M 需： torch, pytorch3d, trimesh   —— 缺这三个则只算 CD（打印 P2M=nan）
    （都装不了时，至少 numpy+scipy+open3d 能给出加 SOR 后的 CD）

用法（示例：Laplace 50k-3%，给所有对比方法加同款 SOR）：
    python eval_sor_standalone.py \
        --clean_dir /data/PUNet/pointclouds/test/50000_poisson \
        --mesh_root /data/PUNet/meshes \
        --pred_dir /preds/IterativePFN/lap_50k_0.03 /preds/PGD/lap_50k_0.03 ... \
        --sor --out_csv lap_50k_0.03_withSOR.csv
    # 去掉 --sor 即为无后处理评测（可对照）。

前提：别人的去噪点云须与干净 GT 在【同一世界坐标系】（多数 repo 已反归一化到原始坐标）。
公平性：加 SOR 前确保别人的 .xyz 是其【最终去噪输出且未含其自身后处理】，否则叠加 SOR =
双重后处理，对该方法不公平；论文中注明 "identical SOR applied to all methods"。
建议：先在【服务器 A（有完整仓库）】上对某 1~2 个 shape 用 tools/eval_external.py 跑一遍，
与本文件在 B 上的结果核对 CD/P2M 是否一致，确认口径无偏差后再信任整张表。
"""
import os
import argparse
import numpy as np
from scipy.spatial import cKDTree


# ----------------------------- IO -----------------------------
def load_xyz_dir(d):
    """读目录下所有 .xyz → {name: [N,3] float32 ndarray}（只取前 3 列）。"""
    out = {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith('.xyz'):
            continue
        arr = np.loadtxt(os.path.join(d, fn), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        out[fn[:-4]] = arr[:, :3].astype(np.float32)
    return out


# ----------------------------- SOR -----------------------------
def sor_filter(pts_np, nb_neighbors=20, std_ratio=2.0):
    """与 runner_finetune.sor_filter 同款（open3d）。SOR 对全局缩放/平移不变。"""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_np.astype(np.float64))
    pcd_f, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return np.asarray(pcd_f.points, dtype=np.float32)


# ----------------------------- CD -----------------------------
def normalize_unit_sphere_np(pc):
    """与 runner_finetune.normalize_unit_sphere 一致（单个云 [N,3]）。返回归一化点、center、scale。"""
    p_max = pc.max(axis=0, keepdims=True)
    p_min = pc.min(axis=0, keepdims=True)
    center = (p_max + p_min) / 2.0                                  # [1,3]
    pc0 = pc - center
    scale = np.sqrt((pc0 ** 2).sum(axis=1)).max()                   # 标量, radius=1
    return pc0 / scale, center, scale


def chamfer_l2_kdtree(pred, clean):
    """ChamferDistanceL2 = mean(min_sq pred->clean) + mean(min_sq clean->pred)。
    pred/clean: [·,3]（已在同一归一化空间）。返回标量（未 ×1e4）。"""
    d_pc, _ = cKDTree(clean).query(pred)     # 每个 pred 到最近 clean 的欧氏距离
    d_cp, _ = cKDTree(pred).query(clean)     # 每个 clean 到最近 pred 的欧氏距离
    return float((d_pc ** 2).mean() + (d_cp ** 2).mean())


# ----------------------------- P2M -----------------------------
_P2M_OK = None
def _try_import_p2m():
    """惰性导入 P2M 依赖。返回 (torch, trimesh, point_mesh_face_distance, Meshes, Pointclouds) 或 None。"""
    global _P2M_OK
    if _P2M_OK is not None:
        return _P2M_OK
    try:
        import torch
        import trimesh
        from pytorch3d.structures import Meshes, Pointclouds
        from pytorch3d.loss import point_mesh_face_distance
        _P2M_OK = (torch, trimesh, point_mesh_face_distance, Meshes, Pointclouds)
    except Exception as e:
        print(f'[P2M] 依赖缺失（torch/pytorch3d/trimesh），仅算 CD：{e}')
        _P2M_OK = False
    return _P2M_OK


def compute_p2m_world(pred_world_np, mesh_path):
    """与 utils/p2m_loss.compute_p2m 同一算法：mesh 归一化到单位球、pred 施加同变换，
    双向 point↔face 距离。pred_world_np: [N,3]（世界坐标）。返回标量（未 ×1e4）。"""
    dep = _try_import_p2m()
    if not dep:
        return float('nan')
    torch, trimesh, point_mesh_face_distance, Meshes, Pointclouds = dep
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mesh = trimesh.load(mesh_path)
    verts = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32)).to(device)
    faces = torch.from_numpy(np.asarray(mesh.faces, dtype=np.int64)).to(device)
    # normalize mesh 到单位球（与 p2m_loss.normalize_sphere 一致）
    p_max = verts.max(0)[0]; p_min = verts.min(0)[0]
    center = (p_max + p_min) / 2.0
    v = verts - center
    scale = (v ** 2).sum(-1).sqrt().max()
    verts_n = v / scale
    pcl = torch.from_numpy(pred_world_np.astype(np.float32)).to(device)
    pcl_n = (pcl - center) / scale
    meshes = Meshes([verts_n], [faces])
    pcls = Pointclouds([pcl_n])
    return float(point_mesh_face_distance(meshes, pcls).item())


# 可选：若本机恰好有完整仓库+已编译扩展，优先用真 ChamferDistanceL2（更保险）
def _try_real_chamfer():
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        import torch
        from extensions.chamfer_dist import ChamferDistanceL2
        m = ChamferDistanceL2().cuda() if torch.cuda.is_available() else ChamferDistanceL2()

        def _cd(pred, clean):   # pred/clean: [·,3] numpy（归一化空间）
            pt = torch.from_numpy(pred).float().unsqueeze(0)
            ct = torch.from_numpy(clean).float().unsqueeze(0)
            if torch.cuda.is_available():
                pt, ct = pt.cuda(), ct.cuda()
            return float(m(pt, ct).item())
        print('[CD] 检测到 extensions.chamfer_dist，使用真 ChamferDistanceL2')
        return _cd
    except Exception:
        return None


def eval_one_method(pred_dir, clean, mesh_names, mesh_root, cd_fn,
                    use_sor, sor_nb, sor_std, verbose=True):
    pred = load_xyz_dir(pred_dir)
    eval_names = sorted(set(pred) & set(clean))
    if not eval_names:
        print(f'  ⚠ {pred_dir} 与 clean 无同名配对，检查文件名/分辨率')
    cds, p2ms = [], []
    for nm in eval_names:
        p_world = pred[nm]
        n0 = p_world.shape[0]
        if use_sor:
            p_world = sor_filter(p_world, sor_nb, sor_std)
        # CD：按干净云单位球归一化两者
        clean_w = clean[nm]
        _, c1, s1 = normalize_unit_sphere_np(clean_w)
        cd = cd_fn((p_world - c1) / s1, (clean_w - c1) / s1) * 1e4
        cds.append(cd)
        # P2M：世界坐标直接算
        p2m = float('nan')
        if nm in mesh_names:
            mp = os.path.join(mesh_root, 'test', nm + '.off')
            p2m = compute_p2m_world(p_world, mp) * 1e4
            if p2m == p2m:   # 非 nan
                p2ms.append(p2m)
        if verbose:
            tag = f'(SOR {n0}->{p_world.shape[0]})' if use_sor else f'({p_world.shape[0]} pts)'
            print(f'    {nm:16s} CD={cd:9.4f}  P2M={p2m:9.4f}  {tag}')
    mcd = float(np.mean(cds)) if cds else float('nan')
    mp2m = float(np.mean(p2ms)) if p2ms else float('nan')
    return mcd, mp2m, len(cds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clean_dir', required=True, help='干净 GT 点云目录（按分辨率）')
    ap.add_argument('--mesh_root', required=True, help='mesh 根目录，其下 test/<name>.off')
    ap.add_argument('--pred_dir', required=True, nargs='+', help='一个或多个方法的去噪 .xyz 目录')
    ap.add_argument('--sor', action='store_true', help='加同款 SOR 后再评（Laplace/各向异性公平对照）')
    ap.add_argument('--sor_nb', type=int, default=20)
    ap.add_argument('--sor_std', type=float, default=2.0)
    ap.add_argument('--out_csv', default=None)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    clean = load_xyz_dir(args.clean_dir)
    mesh_dir = os.path.join(args.mesh_root, 'test')
    mesh_names = {os.path.splitext(f)[0] for f in os.listdir(mesh_dir) if f.endswith('.off')} \
        if os.path.isdir(mesh_dir) else set()
    cd_fn = _try_real_chamfer() or chamfer_l2_kdtree
    print(f'clean: {args.clean_dir} ({len(clean)} 云)  mesh: {mesh_dir} ({len(mesh_names)} 个)')
    if args.sor:
        print(f'[SOR] 启用 nb={args.sor_nb} std={args.sor_std}（对每个方法先滤波再评）')
    print()

    rows = []
    for pd in args.pred_dir:
        name = os.path.basename(pd.rstrip('/'))
        print(f'======== {name} ========')
        mcd, mp2m, n = eval_one_method(pd, clean, mesh_names, args.mesh_root, cd_fn,
                                       args.sor, args.sor_nb, args.sor_std,
                                       verbose=not args.quiet)
        print(f'  >>> {name}: CD={mcd:.4f}  P2M={mp2m:.4f}  ({n} 云)\n')
        rows.append((name, mcd, mp2m, n))

    tag = f'加同款 SOR({args.sor_nb}/{args.sor_std})' if args.sor else '无 SOR'
    print(f'======== 汇总（CD/P2M ×1e4，{tag}）========')
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
