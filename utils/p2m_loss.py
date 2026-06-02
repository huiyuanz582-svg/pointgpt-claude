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

def compute_p2m(pred_points, name,type):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if type == 'train':
        off_path = os.path.join('/home/sapi/zhy/point-gpt-denoise-1024/data/ScoreDenoise/PUNet/meshes/train', name + ".off")
    else:
        off_path = os.path.join('/home/sapi/zhy/point-gpt-denoise-1024/data/ScoreDenoise/PUNet/meshes/test', name + ".off")
    mesh = trimesh.load(off_path)
    verts = torch.from_numpy(mesh.vertices.astype(np.float32)).to(device)
    faces = torch.from_numpy(mesh.faces.astype(np.int64)).to(device)

    loss = point_mesh_bidir_distance_single_unit_sphere(
        pred_points,
        verts,
        faces
    )

    return loss