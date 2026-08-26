import torch
import os
import pytorch3d
from pytorch3d.loss import point_mesh_face_distance
from pytorch3d.structures import Meshes, Pointclouds
import trimesh
import numpy as np

def normalize_sphere(pc, radius=1.0):
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
def point_mesh_bidir_distance_single_unit_sphere(pcl, verts, faces):
    """
    Args:
        pcl:    (N, 3).
        verts:  (M, 3).
        faces:  LongTensor, (T, 3).
    Returns:
        Squared pointwise distances, (N, ).
    """
    assert pcl.dim() == 2 and verts.dim() == 2 and faces.dim() == 2, 'Batch is not supported.'
    
    # Normalize mesh
    verts, center, scale = normalize_sphere(verts.unsqueeze(0))
    verts = verts[0]
    # Normalize pcl
    pcl = normalize_pcl(pcl.unsqueeze(0), center=center, scale=scale)
    pcl = pcl[0]

    # print('%.6f %.6f' % (verts.abs().max().item(), pcl.abs().max().item()))

    # Convert them to pytorch3d structures
    pcls = pytorch3d.structures.Pointclouds([pcl])
    meshes = pytorch3d.structures.Meshes([verts], [faces])
    return pytorch3d.loss.point_mesh_face_distance(meshes, pcls)

_MESH_ROOT = os.environ.get(
    'PUNET_MESH_ROOT', 'data/ScoreDenoise/PUNet/meshes')

def compute_p2m(pred_points, name, type):
    # 跟随预测点所在设备。原先硬编码为默认 ``cuda``，当阶段 0 通过
    # --device 选择非 0 号卡时会产生跨设备张量错误。
    device = pred_points.device
    split = 'train' if type == 'train' else 'test'
    verts, faces = _load_mesh_cached(name, split, device)

    loss = point_mesh_bidir_distance_single_unit_sphere(
        pred_points,
        verts,
        faces
    )

    return loss


# ============================================================================
# 训练用 P2M loss（可微，单向 point→face）
# ----------------------------------------------------------------------------
# 训练时输入是 1024-point patch，bidir 的 face→point 方向在完整 mesh 上没意义
# （绝大多数面远离小 patch），所以只用 point→face：每个预测点到最近 mesh 面的距离。
# 用 pytorch3d 底层 point_face_distance，对预测点可微。
# ============================================================================
_train_mesh_cache = {}  # (mesh_root, name, split) -> CPU (verts, faces)


def _load_mesh_cached(name, split, device):
    # 把 root 纳入 key：不同实验可能在同一进程中切换 TEST_MESH_ROOT。
    mesh_root = os.path.abspath(_MESH_ROOT)
    key = (mesh_root, name, split)
    if key not in _train_mesh_cache:
        off_path = os.path.join(mesh_root, split, name + ".off")
        mesh = trimesh.load(off_path)
        verts = torch.from_numpy(mesh.vertices.astype(np.float32))
        faces = torch.from_numpy(mesh.faces.astype(np.int64))
        _train_mesh_cache[key] = (verts, faces)
    verts, faces = _train_mesh_cache[key]
    return verts.to(device), faces.to(device)


def compute_p2m_train(pred_points_world, name, split='train'):
    """
    可微 P2M 训练 loss（单向 point→face，均方距离）。

    Args:
        pred_points_world: [N, 3]，预测去噪点（mesh 世界坐标系，与 .off 顶点同坐标）
        name:  点云名（对应 meshes/<split>/<name>.off）
        split: 'train'
    Returns:
        标量：均方 point→face 距离（已在 mesh 单位球归一化空间内，量级与 test 的 P2M 可比）
    """
    from pytorch3d.loss.point_mesh_distance import point_face_distance

    device = pred_points_world.device
    verts, faces = _load_mesh_cached(name, split, device)

    # 把 mesh 顶点归一化到单位球，并对预测点施加相同变换（与 test 的 compute_p2m 一致）
    verts_n, center, scale = normalize_sphere(verts.unsqueeze(0))
    verts_n = verts_n[0]
    pcl_n = normalize_pcl(pred_points_world.unsqueeze(0), center=center, scale=scale)[0]

    tris = verts_n[faces]  # [T, 3, 3]
    points_first_idx = torch.zeros(1, dtype=torch.long, device=device)
    tris_first_idx = torch.zeros(1, dtype=torch.long, device=device)
    # 返回每个点到最近面的平方距离，[N]
    dists = point_face_distance(pcl_n, points_first_idx, tris, tris_first_idx, pcl_n.shape[0])
    return dists.mean()
