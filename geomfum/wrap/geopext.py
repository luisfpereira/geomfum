import numpy as np
import scipy

try:
    import geopext
except ModuleNotFoundError:
    pass

from geomfum.laplacian._base import BaseLaplacianFinder


class GeopextMeshLaplacianFinder(BaseLaplacianFinder):
    """Algorithm to find the Laplacian of a mesh."""

    def __call__(self, shape):
        """Apply algorithm.

        Parameters
        ----------
        shape : TriangleMesh
            Mesh.

        Returns
        -------
        laplace_matrix : scipy.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Laplace matrix.
        mass_matrix : scipy.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Diagonal lumped mass matrix.
        """
        laplace_dict, mass_vec = geopext.mesh_laplacian(
            shape.vertices, shape.faces.ravel().astype(np.uintp)
        )

        indices_i = []
        indices_j = []
        values = []
        for (index_i, index_j), value in laplace_dict.items():
            indices_i.append(index_i)
            indices_j.append(index_j)

            values.append(-value)

        laplace_matrix = scipy.sparse.coo_matrix(
            (values, (indices_i, indices_j)), shape=(shape.n_vertices, shape.n_vertices)
        ).tocsc()

        indices = range(shape.n_vertices)
        mass_matrix = scipy.sparse.coo_matrix(
            (mass_vec, (indices, indices)), shape=(shape.n_vertices, shape.n_vertices)
        ).tocsc()

        return laplace_matrix, mass_matrix
