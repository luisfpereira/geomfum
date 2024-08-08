"""Adapted functions from pyFM."""

import numpy as np
import scipy


def get_orientation_op(
    grad_field, vertices, faces, normals, per_vert_area, rotated=False
):
    """
    Compute the linear orientation operator associated to a gradient field grad(f).

    This operator computes g -> < grad(f) x grad(g), n> (given at each vertex) for any function g
    In practice, we compute < n x grad(f), grad(g) > for simpler computation.

    Parameters
    --------------------------------
    grad_field    :
        (n_f,3) gradient field on the mesh
    vertices      :
        (n_v,3) coordinates of vertices
    faces         :
        (n_f,3) indices of vertices for each face
    normals       :
        (n_f,3) normals coordinate for each face
    per_vert_area :
        (n_v,) voronoi area for each vertex
    rotated       : bool
        whether gradient field is already rotated by n x grad(f)

    Returns
    --------------------------
    operator : sparse.csc_matrix or list[sparse.csc_matrix], shape=[n_vertices, n_verticess]
        (n_v,n_v) orientation operator.

    Notes
    -----
    * vectorized version of ``pyFm.geometry.get_orientation_op``.
    """
    n_vertices = per_vert_area.shape[0]
    per_vert_area = np.asarray(per_vert_area)

    v1 = vertices[faces[:, 0]]  # (n_f,3)
    v2 = vertices[faces[:, 1]]  # (n_f,3)
    v3 = vertices[faces[:, 2]]  # (n_f,3)

    # Define (normalized) gradient directions for each barycentric coordinate on each face
    # Remove normalization since it will disappear later on after multiplcation
    Jc1 = np.cross(normals, v3 - v2) / 2
    Jc2 = np.cross(normals, v1 - v3) / 2
    Jc3 = np.cross(normals, v2 - v1) / 2

    # Rotate the gradient field
    if rotated:
        rot_field = grad_field
    else:
        rot_field = np.cross(normals, grad_field)  # (n_f,3)

    I = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    J = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])

    # Compute pairwise dot products between the gradient directions
    # and the gradient field
    Sij = (
        1
        / 3
        * np.concatenate(
            [
                np.einsum("ij,...ij->...i", Jc2, rot_field),
                np.einsum("ij,...ij->...i", Jc3, rot_field),
                np.einsum("ij,...ij->...i", Jc1, rot_field),
            ],
            axis=-1,
        )
    )

    Sji = (
        1
        / 3
        * np.concatenate(
            [
                np.einsum("ij,...ij->...i", Jc1, rot_field),
                np.einsum("ij,...ij->...i", Jc2, rot_field),
                np.einsum("ij,...ij->...i", Jc3, rot_field),
            ],
            axis=-1,
        )
    )

    In = np.concatenate([I, J, I, J])
    Jn = np.concatenate([J, I, I, J])
    Sn = np.concatenate([Sij, Sji, -Sij, -Sji], axis=-1)

    inv_area = scipy.sparse.diags(
        1 / per_vert_area, shape=(n_vertices, n_vertices), format="csc"
    )

    if Sn.ndim == 1:
        W = scipy.sparse.coo_matrix(
            (Sn, (In, Jn)), shape=(n_vertices, n_vertices)
        ).tocsc()

        return inv_area @ W

    out = []
    for Sn_ in Sn:
        W = scipy.sparse.coo_matrix(
            (Sn_, (In, Jn)), shape=(n_vertices, n_vertices)
        ).tocsc()
        out.append(inv_area @ W)

    return out
