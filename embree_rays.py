import embree
import numpy as np

def get_centroids(V, F):
    return V[F].mean(axis=1)


def get_cross_products(V, F):
    vert = V[F]
    V0 = vert[:, 0, :]
    C = np.cross(vert[:, 1, :] - V0, vert[:, 2, :] - V0)
    return C


def get_face_areas(V, F):
    C = get_cross_products(V, F)
    C_norms = np.sqrt(np.sum(C**2, axis=1))
    A = C_norms/2
    return A


def get_surface_normals(V, F):
    C = get_cross_products(V, F)
    C_norms = np.sqrt(np.sum(C**2, axis=1))
    N = C/C_norms.reshape(C.shape[0], 1)
    return N


def get_surface_normals_and_face_areas(V, F):
    C = get_cross_products(V, F)
    C_norms = np.clip(np.sqrt(np.sum(C**2.0, axis=1)), a_min=1e-10, a_max=None)
    print(C_norms)
    exit()
    N = C/C_norms.reshape(C.shape[0], 1)
    A = C_norms/2
    return N, A

def intersect(V, F, origins, dirs, tnear_override=None):
    device = embree.Device()
    geometry = device.make_geometry(embree.GeometryType.Triangle)
    scene = device.make_scene()
    scene.set_flags(4)
    vertex_buffer = geometry.set_new_buffer(
        embree.BufferType.Vertex,  # buf_type
        0,  # slot
        embree.Format.Float3,  # fmt
        3 * np.dtype('float32').itemsize,  # byte_stride
        V.shape[0],  # item_count
    )
    vertex_buffer[:] = V[:]
    index_buffer = geometry.set_new_buffer(
        embree.BufferType.Index,  # buf_type
        0,  # slot
        embree.Format.Uint3,  # fmt
        3 * np.dtype('uint32').itemsize,  # byte_stride,
        F.shape[0]
    )
    index_buffer[:] = F[:]
    geometry.commit()
    scene.attach_geometry(geometry)
    geometry.release()
    scene.commit()
    m = origins.shape[0]
    if dirs.shape[0] != m:
        raise Exception('origins and dirs need the same number of rows')
    rayhit = embree.RayHit1M(m)
    context = embree.IntersectContext()
    rayhit.org[:] = origins
    rayhit.dir[:] = dirs
    if not (tnear_override is None):
        rayhit.tnear[:] = tnear_override[:]
    else:
        rayhit.tnear[:] = 0
    rayhit.tfar[:] = np.inf
    rayhit.flags[:] = 0
    rayhit.geom_id[:] = embree.INVALID_GEOMETRY_ID

    scene.intersect1M(context, rayhit)

    I = rayhit.prim_id.copy().astype(np.intp)
    I[rayhit.geom_id == embree.INVALID_GEOMETRY_ID] = -1
    scene.release()
    device.release()
    return I


class TrimeshShapeModel:
    def __init__(self, V, F, P=None, N=None, A=None):
        self.dtype = V.dtype

        self.V = V
        self.F = F

        # if N is None and A is None:
        #     N, A = get_surface_normals_and_face_areas(V, F)
        # elif A is None:
        #     if N.shape[0] != F.shape[0]:
        #         raise Exception(
        #             'must pass same number of surface normals as faces (got ' +
        #             '%d faces and %d normals' % (F.shape[0], N.shape[0])
        #         )
        #     A = get_face_areas(V, F)
        # elif N is None:
        #     N = get_surface_normals(V, F)

        # self.P = get_centroids(V, F)
        # self.N = N
        # self.A = A

        # assert self.P.dtype == self.dtype
        # assert self.N.dtype == self.dtype
        # assert self.A.dtype == self.dtype

        self._make_scene()

    def _make_scene(self):
        '''Set up an Embree scene. This function allocates some memory that
        Embree manages, and loads vertices and index lists for the
        faces. In Embree parlance, this function creates a "device",
        which manages a "scene", which has one "geometry" in it, which
        is our mesh.

        '''
        self.device = embree.Device()
        geometry = self.device.make_geometry(embree.GeometryType.Triangle)
        self.scene = self.device.make_scene()
        self.scene.set_flags(4)
        vertex_buffer = geometry.set_new_buffer(
            embree.BufferType.Vertex, # buf_type
            0, # slot
            embree.Format.Float3, # fmt
            3*np.dtype('float32').itemsize, # byte_stride
            self.V.shape[0], # item_count
        )
        vertex_buffer[:] = self.V[:]
        index_buffer = geometry.set_new_buffer(
            embree.BufferType.Index, # buf_type
            0, # slot
            embree.Format.Uint3, # fmt
            3*np.dtype('uint32').itemsize, # byte_stride,
            self.F.shape[0]
        )
        index_buffer[:] = self.F[:]
        geometry.commit()
        self.scene.attach_geometry(geometry)
        geometry.release()
        self.scene.commit()

        # This is the only variable we need to retain a reference to
        # (I think)

    def __reduce__(self):
        return (self.__class__, (self.V, self.F))

    # @property
    # def num_faces(self):
    #     return self.P.shape[0]

    # def check_vis_1_to_N(self, i, J, eps=None):
    #     if eps is None:
    #         eps = 1e3*np.finfo(np.float32).resolution
    #
    #     D = self.P[J] - self.P[i]
    #     D /= np.sqrt(np.sum(D**2, axis=1)).reshape(D.shape[0], 1)
    #     P = self.P[i] + eps*D
    #
    #     rayhit = embree.RayHit1M(len(J))
    #     context = embree.IntersectContext()
    #     rayhit.org[:] = P
    #     rayhit.dir[:] = D
    #     rayhit.tnear[:] = 0
    #     rayhit.tfar[:] = np.inf
    #     rayhit.flags[:] = 0
    #     rayhit.geom_id[:] = embree.INVALID_GEOMETRY_ID
    #
    #     self.scene.intersect1M(context, rayhit)
    #
    #     return np.logical_and(
    #         rayhit.geom_id != embree.INVALID_GEOMETRY_ID,
    #         rayhit.prim_id == J
    #     )

    def intersect(self, origins, dirs, tnear_override=None):
        m = origins.shape[0]
        if dirs.shape[0] != m:
            raise Exception('origins and dirs need the same number of rows')

        rayhit = embree.RayHit1M(m)
        context = embree.IntersectContext()
        rayhit.org[:] = origins
        rayhit.dir[:] = dirs
        if not (tnear_override is None):
            rayhit.tnear[:]=tnear_override[:]
        else:
            rayhit.tnear[:] = 0
        rayhit.tfar[:] = np.inf
        rayhit.flags[:] = 0
        rayhit.geom_id[:] = embree.INVALID_GEOMETRY_ID

        self.scene.intersect1M(context, rayhit)

        I = rayhit.prim_id.copy().astype(np.intp)
        I[rayhit.geom_id == embree.INVALID_GEOMETRY_ID] = -1
        # self.scene.release()
        # self.device.release()
        return I

    # def get_direct_irradiance(self, F0, dir_sun, eps=None):
    #     if eps is None:
    #         eps = 1e3*np.finfo(np.float32).resolution
    #
    #     # Here, we use Embree directly to find the indices of triangles
    #     # which are directly illuminated (I_sun) or not (I_shadow).
    #     ray = embree.Ray1M(self.num_faces)
    #     ray.org[:] = self.P + eps*self.N
    #     ray.dir[:] = dir_sun
    #     ray.tnear[:] = 0
    #     ray.tfar[:] = np.inf
    #     ray.flags[:] = 0
    #     context = embree.IntersectContext()
    #     self.scene.occluded1M(context, ray)
    #
    #     # Determine which rays escaped (i.e., can see the sun)
    #     I = np.isposinf(ray.tfar)
    #
    #     # Compute the direct irradiance
    #     E = np.zeros(self.num_faces, dtype=self.dtype)
    #     E[I] = F0*np.maximum(0, self.N[I]@dir_sun)
    #
    #     return E

