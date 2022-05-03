# from pygsp.graphs.nngraphs.spherehealpix import SphereHealpix
# grid_object_sphere = SphereHealpix(nside=32, n_neighbors=8)
# grid_object_sphere = SphereHealpix(subdivisions=32, k=8, nest=True)
# pygsp has many different versions.... API is different
# See https://github.com/deepsphere/deepsphere-pytorch/blob/master/notebooks/demo_visualizations.ipynb for visualization

# for k in range(len(stuff)):
#     cur_stuff = stuff[k]
#     data_exp = torch_geometric.data.Data()
#     data_exp.radial_depth_loc = torch.from_numpy(cur_stuff["depth_xyz"][0]).to(device="cuda", non_blocking=True,
#                                                                                dtype=torch.float32)
#     data_exp.radial_depth_idx_tri = torch.from_numpy(cur_stuff["depth_tri"][0]).to(device="cuda", non_blocking=True,
#                                                                                    dtype=torch.long)
#     data_exp.radial_depth_idx_ray = torch.from_numpy(cur_stuff["depth_ray"][0]).to(device="cuda", non_blocking=True,
#                                                                                    dtype=torch.long)
#
#     data_exp.ortho_depth_loc = torch.from_numpy(cur_stuff["depth_xyz"][1]).to(device="cuda", non_blocking=True,
#                                                                               dtype=torch.float32)
#     data_exp.ortho_depth_idx_tri = torch.from_numpy(cur_stuff["depth_tri"][1]).to(device="cuda", non_blocking=True,
#                                                                                   dtype=torch.long)
#     data_exp.ortho_depth_idx_ray = torch.from_numpy(cur_stuff["depth_ray"][1]).to(device="cuda", non_blocking=True,
#                                                                                   dtype=torch.long)
#
#     data_exp.radial_prob_loc = torch.from_numpy(np.concatenate(cur_stuff["prob_xyz_radial"], axis=0)).to(device="cuda",
#                                                                                                          non_blocking=True,
#                                                                                                          dtype=torch.float32)
#     data_exp.radial_prob_idx_tri = torch.from_numpy(np.concatenate(cur_stuff["prob_tri_radial"], axis=0)).to(
#         device="cuda", non_blocking=True, dtype=torch.long)
#     data_exp.radial_prob_idx_ray = torch.from_numpy(np.concatenate(cur_stuff["prob_ray_radial"], axis=0)).to(
#         device="cuda", non_blocking=True, dtype=torch.long)
#     data_exp.radial_offsets = torch.LongTensor([0] + [len(_) for _ in cur_stuff["prob_ray_radial"]])
#
#     data_exp.ortho_prob_loc = torch.from_numpy(np.concatenate(cur_stuff["prob_xyz_ortho"], axis=0)).to(device="cuda",
#                                                                                                        non_blocking=True,
#                                                                                                        dtype=torch.float32)
#     data_exp.ortho_prob_idx_tri = torch.from_numpy(np.concatenate(cur_stuff["prob_tri_ortho"], axis=0)).to(
#         device="cuda", non_blocking=True, dtype=torch.long)
#     data_exp.ortho_prob_idx_ray = torch.from_numpy(np.concatenate(cur_stuff["prob_ray_ortho"], axis=0)).to(
#         device="cuda", non_blocking=True, dtype=torch.long)
#     data_exp.ortho_offsets = torch.LongTensor([0] + [len(_) for _ in cur_stuff["prob_ray_ortho"]])
#     data_exp.verts_in = torch.from_numpy(verts).to(device="cuda", non_blocking=True, dtype=torch.float32)
#     data_exp.tri_in = torch.from_numpy(faces).to(device="cuda", non_blocking=True, dtype=torch.long)
#
#     data_exp.num_nodes = 100
#     data_list.append(data_exp)

# blending_obj = diff_render_blend(sgrid_obj=torch.from_numpy(grid_obj).float())
# blending_obj.cuda()
# blending_obj = torch_geometric.nn.DataParallel(blending_obj, device_ids=[0, 1])
# torch.cuda.synchronize()
# t0 = time.time()
# for k in range(10):
#     output = blending_obj(data_list)
# torch.cuda.synchronize()
# print(time.time() - t0, "TIME")
# val = output[0][2]
# val = val.detach().cpu().numpy()
# # plt.scatter(x=grid_obj2.lon, y=-grid_obj2.lat, c=val, cmap="Greys", alpha=0.5)
# plt.scatter(x=grid_obj2[0], y=grid_obj2[1], c=val, cmap="jet", alpha=0.5)
# plt.colorbar()
# plt.show()
