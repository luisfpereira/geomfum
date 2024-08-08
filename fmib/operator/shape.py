import abc

import igl
import pyFM
import robust_laplacian


class ShapeOperator(abc.ABC):
    def __call__(self, shape):
        pass


class RobustMeshLaplacian(ShapeOperator):
    """Mesh Laplacian.

    Parameters
    ----------
    mollify_factor : float
        Amount of intrinsic mollification to perform.
    """

    def __init__(self, mollify_factor=1e-5):
        self.mollify_factor = mollify_factor

    def __call__(self, shape):
        """Operator on shape.

        Parameters
        ----------
        shape : TriangleMesh

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


class PyfmMeshLaplacian(ShapeOperator):
    def __call__(self, shape):
        return (
            pyFM.mesh.laplacian.cotangent_weights(shape.vertices, shape.faces),
            pyFM.mesh.laplacian.dia_area_mat(shape.vertices, shape.faces),
        )


class IglMeshLaplacian(ShapeOperator):
    def __call__(self, shape):
        """Operator on shape.

        Parameters
        ----------
        shape : TriangleMesh

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


class RobustPointCloudLaplacian(ShapeOperator):
    """Point cloud Laplacian.

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
        """Operator on shape.

        Parameters
        ----------
        shape : TriangleMesh

        Returns
        -------
        laplace_matrix : scipy.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            "Weak" Laplacian matrix.
        mass_matrix : scipy.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Diagonal lumped mass matrix.
        """
        return robust_laplacian.point_cloud_laplacian(
            shape.points,
            mollify_factor=self.mollify_factor,
            n_neighbors=self.n_neighbors,
        )


class ShapeLaplacian:
    """Shape Laplacian.

    Parameters
    ----------
    mesh : bool
        If mesh or point cloud.
    which : str
        One of: robust, pyfm
    """

    # TODO: use register instead?
    _MAP = {
        (True, "robust"): RobustMeshLaplacian,
        (True, "pyfm"): PyfmMeshLaplacian,
        (True, "igl"): IglMeshLaplacian,
        (False, "robust"): RobustPointCloudLaplacian,
    }

    def __new__(cls, mesh=True, which="robust", **kwargs):
        return cls._MAP[(mesh, which)](**kwargs)
