import numpy as np
import trimesh
import torch
from torch_geometric.data import Data

# Expands triangles in object space
def expand_trig_np(verts):
    pt1 = verts[:, 0, :]
    pt2 = verts[:, 1, :]
    pt3 = verts[:, 2, :]
    line1 = pt1 - pt2
    line2 = pt2 - pt3
    line3 = pt3 - pt1

    line1_len = np.linalg.norm(line1, axis=1, keepdims=True)
    line2_len = np.linalg.norm(line2, axis=1, keepdims=True)
    line3_len = np.linalg.norm(line3, axis=1, keepdims=True)

    sum_len = np.clip(line1_len + line2_len + line3_len, a_min=1e-9, a_max=None)
    incenter = (line1_len * pt3 + line2_len * pt1 + line3_len * pt2) / sum_len

    unsqrt = np.clip((-line1_len + line2_len + line3_len) * (line1_len - line2_len + line3_len) * (line1_len + line2_len - line3_len) / sum_len, a_min=1e-9, a_max=10)
    radius = 0.5*np.sqrt(unsqrt)

    eps = 2e-2
    pt1_off = pt1 - incenter
    pt2_off = pt2 - incenter
    pt3_off = pt3 - incenter

    off_ratio = np.clip(eps / radius, 0.0, 4.0)
    pt1_new = off_ratio * pt1_off + pt1
    pt2_new = off_ratio * pt2_off + pt2
    pt3_new = off_ratio * pt3_off + pt3
    new_vert = np.concatenate((pt1_new, pt2_new, pt3_new), axis=0)
    new_face_flat = np.arange(start=0, stop=3 * len(verts), dtype=np.int)
    new_face = np.reshape(new_face_flat, (3, len(verts))).T
    return new_vert, new_face

# Normalize shape
def scale_to_unit_sphere(mesh):
    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False, maintain_order=True)

# Convert to list of pytorch geometric objects
# Allow us to use data parallel

def make_list_parallel(input_list_all):
    data_list = []
    for k in range(len(input_list_all)):
        input_list = input_list_all[k]
        data_exp = Data()
        data_exp.radial_depth_loc = torch.from_numpy(input_list["depth_xyz"][0]).to(device="cuda", non_blocking=True, dtype=torch.float32)
        data_exp.radial_depth_idx_tri = torch.from_numpy(input_list["depth_tri"][0]).to(device="cuda", non_blocking=True,dtype=torch.long)
        data_exp.radial_depth_idx_ray = torch.from_numpy(input_list["depth_ray"][0]).to(device="cuda", non_blocking=True,dtype=torch.long)

        data_exp.ortho_depth_loc = torch.from_numpy(input_list["depth_xyz"][1]).to(device="cuda", non_blocking=True,dtype=torch.float32)
        data_exp.ortho_depth_idx_tri = torch.from_numpy(input_list["depth_tri"][1]).to(device="cuda", non_blocking=True,dtype=torch.long)
        data_exp.ortho_depth_idx_ray = torch.from_numpy(input_list["depth_ray"][1]).to(device="cuda", non_blocking=True,dtype=torch.long)

        data_exp.radial_prob_loc = torch.from_numpy(np.concatenate(input_list["prob_xyz_radial"], axis=0)).to(device="cuda", non_blocking=True, dtype=torch.float32)
        data_exp.radial_prob_idx_tri = torch.from_numpy(np.concatenate(input_list["prob_tri_radial"], axis=0)).to(device="cuda", non_blocking=True, dtype=torch.long)
        data_exp.radial_prob_idx_ray = torch.from_numpy(np.concatenate(input_list["prob_ray_radial"], axis=0)).to(device="cuda", non_blocking=True, dtype=torch.long)
        data_exp.radial_offsets = torch.LongTensor([0] + [len(_) for _ in input_list["prob_ray_radial"]])
        # We save the offsets for radial, for hit 1,2,3

        data_exp.ortho_prob_loc = torch.from_numpy(np.concatenate(input_list["prob_xyz_ortho"], axis=0)).to(device="cuda",non_blocking=True,dtype=torch.float32)
        data_exp.ortho_prob_idx_tri = torch.from_numpy(np.concatenate(input_list["prob_tri_ortho"], axis=0)).to(device="cuda", non_blocking=True, dtype=torch.long)
        data_exp.ortho_prob_idx_ray = torch.from_numpy(np.concatenate(input_list["prob_ray_ortho"], axis=0)).to(device="cuda", non_blocking=True, dtype=torch.long)
        data_exp.ortho_offsets = torch.LongTensor([0] + [len(_) for _ in input_list["prob_ray_ortho"]])
        # We save the offsets for ortho, for hit 1,2,3

        # data_exp.verts_in = torch.from_numpy(input_list["verts"]).to(device="cuda", non_blocking=True, dtype=torch.float32)
        data_exp.verts_in = input_list["verts"]
        # data_exp.tri_in = torch.from_numpy(input_list["faces"]).to(device="cuda", non_blocking=True, dtype=torch.long)
        data_exp.tri_in = input_list["faces"]
        # any constant is fine, used to distribute the objects to gpus
        data_exp.num_nodes = 100
        data_list.append(data_exp)
    return data_list
