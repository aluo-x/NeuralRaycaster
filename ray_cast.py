from sph_related.embree_rays import TrimeshShapeModel, intersect
from sph_related.misc import expand_trig_np
import numpy as np
from trimesh.intersections import planes_lines

def ray_trace_new_embree(input_data):
    verts_list, faces_list, ray_grid = input_data[0], input_data[1], input_data[2]
    face_verts = verts_list[faces_list]
    tnear_cur = None
    # verts_list, faces_list = expand_trig_np(face_verts)
    shape_obj = TrimeshShapeModel(verts_list*1.0, np.copy(faces_list))
    intersection_xyz=[]
    index_tri=[]
    index_ray=[]

    radial_origin = ray_grid * 2.0
    radial_dir = -radial_origin / np.linalg.norm(radial_origin, axis=1, keepdims=True)

    ortho_origin = ray_grid * 1.0
    ortho_dir = ray_grid * 1.0
    y_value = ortho_origin[:, 1]
    lt0 = y_value > 0
    ortho_dir[lt0] = np.array([0, -1, 0])
    ortho_dir[~lt0] = np.array([0, 1, 0])
    ortho_origin[lt0, 1] = 2.0
    ortho_origin[~lt0, 1] = -2.0
    # Set to 2.0 for numerical stability, some shapes are too close to 1.0 :(

    return_dict = dict()

    for intersect_mode in ["radial", "ortho"]:
    # for intersect_mode in ["radial"]:
        if intersect_mode == "radial":
            cur_dir = radial_dir
            current_origin = radial_origin
        elif intersect_mode == "ortho":
            cur_dir = ortho_dir
            current_origin = ortho_origin

        # bad = np.zeros(current_origin.shape[0], dtype=np.bool)
        # output = intersect(verts_list*1.0,np.copy(faces_list), current_origin, cur_dir, tnear_override=tnear_cur)
        output = shape_obj.intersect(current_origin, cur_dir, tnear_override=tnear_cur)
        # bad[output<-0.5] = 1
        bad = output<-0.5
        good = ~bad
        good_outputs = output[good]
        valid_vert = face_verts[good_outputs]
        pt0 = valid_vert[:,0]
        line1 = valid_vert[:,1]-pt0
        line2 = valid_vert[:,2]-pt0
        good_normals = np.cross(line1, line2)
        good_normals = good_normals/np.clip(np.linalg.norm(good_normals, axis=1, keepdims=True), a_min=1e-10, a_max=None)
        good_ray = cur_dir[good]
        new_origins, valid = planes_lines(plane_origins=pt0, plane_normals=good_normals,line_origins=current_origin[good], line_directions=good_ray)
        good_idx = np.where(good)[0][valid]
        intersection_xyz.append(new_origins)
        index_tri.append(good_outputs[valid])
        index_ray.append(good_idx)
    shape_obj.scene.release()
    shape_obj.device.release()
    return_dict["depth_xyz"] = intersection_xyz
    return_dict["depth_tri"] = index_tri
    return_dict["depth_ray"] = index_ray

    expanded_verts_list, expanded_faces_list = expand_trig_np(face_verts)
    shape_obj2 = TrimeshShapeModel(expanded_verts_list, expanded_faces_list)

    for intersect_mode in ["radial", "ortho"]:
        intersection_xyz = []
        index_tri = []
        index_ray = []
        tnear_cur = None
        if intersect_mode == "radial":
            cur_dir = radial_dir
            current_origin = radial_origin
        elif intersect_mode == "ortho":
            cur_dir = ortho_dir
            current_origin = ortho_origin

        bad = np.zeros(current_origin.shape[0], dtype=np.bool)

        for hits in range(3):
            output = shape_obj2.intersect(current_origin, cur_dir, tnear_override=tnear_cur)
            # output = intersect(expanded_verts_list, expanded_faces_list, current_origin, cur_dir, tnear_override=tnear_cur)
            bad[output < -0.5] = 1
            good = ~bad
            good_outputs = output[good]
            valid_vert = face_verts[good_outputs]
            pt0 = valid_vert[:, 0]
            line1 = valid_vert[:, 1] - pt0
            line2 = valid_vert[:, 2] - pt0
            good_normals = np.cross(line1, line2)
            good_normals = good_normals / np.clip(np.linalg.norm(good_normals, axis=1, keepdims=True), a_min=1e-10, a_max=None)
            good_ray = cur_dir[good]
            new_origins, valid = planes_lines(plane_origins=pt0, plane_normals=good_normals, line_origins=current_origin[good], line_directions=good_ray)
            good_idx = np.where(good)[0][valid]
            tnear_cur = np.full(current_origin.shape[0], 1000.0, dtype=np.float)
            tnear_cur[good_idx] = np.linalg.norm(new_origins - current_origin[good_idx], axis=1) + 2e-5

            intersection_xyz.append(new_origins)
            index_tri.append(good_outputs[valid])
            index_ray.append(good_idx)

        return_dict["prob_xyz_{}".format(intersect_mode)] = intersection_xyz
        return_dict["prob_tri_{}".format(intersect_mode)] = index_tri
        return_dict["prob_ray_{}".format(intersect_mode)] = index_ray
    shape_obj2.scene.release()
    shape_obj2.device.release()

    return return_dict