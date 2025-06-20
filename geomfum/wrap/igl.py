"""igl wrapper."""

import igl

import geomfum.backend as xgs
from geomfum.laplacian import BaseLaplacianFinder


class IglMeshLaplacianFinder(BaseLaplacianFinder):
    """Algorithm to find the Laplacian of a mesh."""

    def __call__(self, shape):
        """Apply algorithm.

        Parameters
        ----------
        shape : TriangleMesh
            Mesh.

        Returns
        -------
        stiffness_matrix : sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Stiffness matrix.
        mass_matrix : sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Diagonal lumped mass matrix.
        """
        return (
            xgs.sparse.from_scipy_csc(-igl.cotmatrix(shape.vertices, shape.faces)),
            xgs.sparse.from_scipy_csc(
                igl.massmatrix(shape.vertices, shape.faces, igl.MASSMATRIX_TYPE_VORONOI)
            ),
        )
