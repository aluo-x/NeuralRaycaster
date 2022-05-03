import torch
from sph_related.misc_differentiable import compute_bary_new, point_triangle_distance

# This works with multi-GPU
class diff_render_blend(torch.nn.Module):
    def __init__(self, znear=0.01, zfar=2.5, sigma=1e-4, eps=1e-10, sgrid_obj=None, radius=2.0, max_hits=4):
        super().__init__()
        if sgrid_obj is None:
            print("Please provide an sgrid object")
            exit()
        # self.sgrid_torch = sgrid_obj/torch.linalg.norm(sgrid_obj, dim=1, keepdim=True) * radius
        self.register_buffer("sgrid_torch", sgrid_obj/torch.linalg.norm(sgrid_obj, dim=1, keepdim=True) * radius, persistent=False)
        sgrid_ortho = sgrid_obj * 1.0
        # make a copy

        ortho_mask = sgrid_ortho[:,1] > 0.0
        sgrid_ortho[ortho_mask, 1] = 2.0
        sgrid_ortho[~ortho_mask, 1] = -2.0
        self.register_buffer("ortho_torch", sgrid_ortho, persistent=False)

        self.sgrid_shape = list(self.sgrid_torch.shape)[:-1]
        self.znear = znear
        self.zfar = zfar
        self.sigma = sigma
        self.eps = eps
        self.max_hits = max_hits

    def forward(self, input_stuff):
        # input_stuff_list = input_stuff.to_data_list()
        # input_stuff_list = [input_stuff]
        input_stuff_list = input_stuff
        # We split the input into the original chunks
        output_list = []
        cur_device = self.sgrid_torch.device
        hits = self.max_hits
        sgrid_shape = self.sgrid_shape

        depth_collector = []
        silhouettes_collector = []
        for data_idx in range(len(input_stuff_list)):
            silhouettes_container = []
            total_out_depths = []
            cur_data = input_stuff_list[data_idx]
            verts_list = cur_data.verts_in
            faces_list = cur_data.tri_in

            depth_return_container = torch.zeros([2] + sgrid_shape, dtype=torch.float).to(device=cur_device, non_blocking=True)+2.0
            cur_offset = 0

            for projection_type in ["radial", "ortho"]:
                # compute the differentiable depth
                hits_radial_pos_list = getattr(cur_data, "{}_depth_loc".format(projection_type))
                hits_radial_index_tri_list = getattr(cur_data, "{}_depth_idx_tri".format(projection_type))
                hits_radial_index_index_ray = getattr(cur_data, "{}_depth_idx_ray".format(projection_type))

                input_xyz = hits_radial_pos_list
                index_tri = hits_radial_index_tri_list
                index_ray = hits_radial_index_index_ray
                x_b, y_b, z_b = compute_bary_new(input_xyz, index_tri, verts_list, faces_list)
                bary_weights = torch.stack([x_b, y_b, z_b], 1)[:, :, None]
                new_loc = torch.sum(bary_weights * verts_list[faces_list[hits_radial_index_tri_list]], 1)

                if projection_type == "radial":
                    z_dist = torch.linalg.norm(self.sgrid_torch[index_ray] - new_loc, dim=1)
                else:
                    z_dist = torch.linalg.norm(self.ortho_torch[index_ray] - new_loc, dim=1)
                depth_return_container[cur_offset][hits_radial_index_index_ray] = z_dist
                cur_offset += 1
            for projection_type in ["radial", "ortho"]:
            # for projection_type in ["radial"]:
                hits_radial_pos_list = getattr(cur_data, "{}_prob_loc".format(projection_type))
                hits_radial_index_tri_list = getattr(cur_data, "{}_prob_idx_tri".format(projection_type))
                hits_radial_index_index_ray = getattr(cur_data, "{}_prob_idx_ray".format(projection_type))
                index_ray2 = getattr(cur_data, "{}_depth_idx_ray".format(projection_type))
                slices_radial_idx = getattr(cur_data, "{}_offsets".format(projection_type))
                # xy_dist_radial_container = -torch.ones([hits] + sgrid_shape, dtype=torch.float).to(device=cur_device, non_blocking=True)*0.0
                xy_dist_radial_container = torch.zeros([hits] + sgrid_shape, dtype=torch.float).to(device=cur_device, non_blocking=True) * 0.0
                mask_radial = torch.zeros([hits] + sgrid_shape, dtype=torch.float).to(device=cur_device, non_blocking=True)
                trig_coords = verts_list[faces_list[hits_radial_index_tri_list]]
                face_trig_dist = point_triangle_distance(hits_radial_pos_list, trig_coords[:, 0], trig_coords[:, 1], trig_coords[:, 2])
                # get rid of the last dimension which is 1
                for k_hit_idx in range(len(slices_radial_idx)-1):
                    start = slices_radial_idx[k_hit_idx]
                    end = slices_radial_idx[k_hit_idx + 1]
                    if end - start == 0:
                        continue
                    hit_rays = hits_radial_index_index_ray[start:end]
                    mask_radial[k_hit_idx][hit_rays] = 1.0
                    xy_dist_radial_container[k_hit_idx][hit_rays] = face_trig_dist[start:end]
                prob_map = torch.exp(-xy_dist_radial_container/5e-5) * mask_radial
                alpha = torch.prod((1.0 - prob_map), dim=0)
                silhouette = 1.0 - alpha
                silhouette[index_ray2] = 1.0
                silhouettes_container.append(silhouette)
            silhouettes_return_container = torch.stack(silhouettes_container, dim=0)
            depth_collector.append(depth_return_container)
            silhouettes_collector.append(silhouettes_return_container)
        batched_depth = torch.stack(depth_collector, dim=0)
        batched_silhouette = torch.stack(silhouettes_collector, dim=0)
        final_out = torch.cat((batched_depth, batched_silhouette), dim=1)
        return final_out