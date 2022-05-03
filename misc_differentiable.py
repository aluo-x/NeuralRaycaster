import torch
def compute_bary_new(intersect_loc, intersect_idx, verts_arr, faces_arr):
    # intersect_loc is of shape (I, 3), gives us exactly where we intersect the surface
    # intersect_idx is of shape (I)
    # verts[faces_arr] is of shape (FACE, 3, 3) or (FACE, PT, XYZ)
    intersect_face_loc = verts_arr[faces_arr[intersect_idx]]
    #     intersect_face_loc = face_loc[intersect_idx]
    # should be of shape I, 3, 3
    a = intersect_face_loc[:, 0, :]
    b = intersect_face_loc[:, 1, :]
    c = intersect_face_loc[:, 2, :]
    v0 = b - a
    v1 = c - a
    v2 = intersect_loc - a

    d00 = torch.sum(v0*v0, dim=1)
    d01 = torch.sum(v0*v1, dim=1)
    d11 = torch.sum(v1*v1, dim=1)
    d20 = torch.sum(v2*v0, dim=1)
    d21 = torch.sum(v2*v1, dim=1)
    # d00 = torch.einsum('ij,ij->i',v0,v0)
    # d01 = torch.einsum('ij,ij->i',v0,v1)
    # d11 = torch.einsum('ij,ij->i',v1,v1)
    # d20 = torch.einsum('ij,ij->i',v2,v0)
    # d21 = torch.einsum('ij,ij->i',v2,v1)
    denom = d00 * d11 - d01 * d01
    # colinear verts :(
    bad = torch.abs(denom)<1e-9
    denom[bad] = 1e-9
    inv_denom = 1/denom
    v = (d11 * d20 - d01 * d21) * inv_denom
    w = (d00 * d21 - d01 * d20) * inv_denom
    u = 1.0 - v - w
    return (u, v, w)

# Based on code from pytorch3d
def point_line_distance(p, v0, v1):
    v1v0 = v1 - v0
    l2 = torch.sum(v1v0*v1v0, dim=1)
    t = torch.clamp(torch.sum(v1v0*(p-v0), dim=1)/torch.clamp(l2, min=1e-9), min=0.0, max=1.0)
    p_proj = v0 + t[:, None] * v1v0
    delta_p = p_proj - p
    return torch.sum(delta_p*delta_p, dim=1)

def point_triangle_distance(p, v0, v1, v2):
    e01_dist = point_line_distance(p, v0, v1)
    e02_dist = point_line_distance(p, v0, v2)
    e12_dist = point_line_distance(p, v1, v2)
    edge_dists_min = torch.min(torch.min(e01_dist, e02_dist), e12_dist)
    return edge_dists_min