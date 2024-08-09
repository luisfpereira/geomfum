"""Laplacian-related algorithms for point clouds."""

import robust_laplacian

from ._base import BaseLaplacianFinder


class RobustPointCloudLaplacianFinder(BaseLaplacianFinder):
    """Algorithm to find the Laplacian of a point cloud.

    Parameters
    ----------
    mollify_factor : float
        Amount of intrinsic mollification to perform.
    n_neighbors : float
        Number of nearest neighbors to use when constructing local triangulations.
    """

    def __init__(self, mollify_factor=1e-5, n_neighbors=30):
        self.mollify_factor = mollify_factor
        self.n_neighbors = n_neighbors

    def __call__(self, shape):
        """Apply algorithm.

        Parameters
        ----------
        shape : PointCloud
            Point cloud.

        Returns
        -------
        laplace_matrix : scipy.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            "Weak" Laplace matrix.
        mass_matrix : scipy.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Diagonal lumped mass matrix.
        """
        return robust_laplacian.point_cloud_laplacian(
            shape.vertices,
            mollify_factor=self.mollify_factor,
            n_neighbors=self.n_neighbors,
        )
