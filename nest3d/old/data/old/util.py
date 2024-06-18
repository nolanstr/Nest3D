import numpy as np


def compute_ND(scan, crystal_normal, plane_normal=None):
    crystal_normals = np.einsum("nij,j->ni", scan.R, crystal_normal.flatten())

    if not isinstance(plane_normal, np.ndarray):
        return crystal_normals
    else:
        projections = plane_normal - np.einsum("nj,j->n", crystal_normals,
                                       plane_normal).reshape((-1,1)) * plane_normal
        return projections



