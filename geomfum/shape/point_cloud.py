"""Definition of point cloud."""

from geomfum.io import load_mesh
from ._base import Shape


class PointCloud(Shape):
    """Point cloud.

    Parameters
    ----------
    vertices : array-like, shape=[n_vertices, 3]
        Vertices of the point cloud.
    """

    def __init__(self, vertices):
        super().__init__(is_mesh=False)
        self.vertices = vertices

    @classmethod
    def from_file(cls, filename):
        """Instantiate given a file.

        Returns
        -------
        mesh : TriangleMesh
            A triangle mesh.
        """
        vertices, _ = load_mesh(filename)
        return cls(vertices)

    @property
    def n_vertices(self):
        """Number of vertices.

        Returns
        -------
        n_vertices : int
        """
        return self.vertices.shape[0]