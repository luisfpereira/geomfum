"""Laplacian-related algorithms for meshes."""

import igl
import pyFM
import robust_laplacian

from ._base import BaseLaplacianFinder


class RobustMeshLaplacianFinder(BaseLaplacianFinder):
    """Algorithm to find the Laplacian of a mesh.

    Parameters
    ----------
    mollify_factor : float
        Amount of intrinsic mollification to perform.
    """

    def __init__(self, mollify_factor=1e-5):
        self.mollify_factor = mollify_factor

    def __call__(self, shape):
        """Apply algorithm.

        Parameters
        ----------
        shape : TriangleMesh
            Mesh.

        Returns
        -------
        laplace_matrix : scipy.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Laplacian matrix.
        mass_matrix : scipy.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Diagonal lumped mass matrix.
        """
        return robust_laplacian.mesh_laplacian(
            shape.vertices, shape.faces, mollify_factor=self.mollify_factor
        )


class PyfmMeshLaplacianFinder(BaseLaplacianFinder):
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
            Laplacian matrix.
        mass_matrix : scipy.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Diagonal lumped mass matrix.
        """
        return (
            pyFM.mesh.laplacian.cotangent_weights(shape.vertices, shape.faces),
            pyFM.mesh.laplacian.dia_area_mat(shape.vertices, shape.faces),
        )


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
        laplace_matrix : scipy.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Laplacian matrix.
        mass_matrix : scipy.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Diagonal lumped mass matrix.
        """
        return (
            -igl.cotmatrix(shape.vertices, shape.faces),
            igl.massmatrix(shape.vertices, shape.faces, igl.MASSMATRIX_TYPE_VORONOI),
        )
