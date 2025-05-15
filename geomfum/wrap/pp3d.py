"""

Wrap file for potpourri3d functions

https://github.com/nmwsharp/potpourri3d
by Nicholas Sharp.

"""

import numpy as np
import potpourri3d as pp3d

from geomfum.metric.mesh import HeatDistanceMetric


class Pp3dHeatDistanceMetric(HeatDistanceMetric):
    """Heat distance metric between vertices of a mesh.

    Parameters
    ----------
    shape : Shape
        Shape.

    References
    ----------
    "The Heat Method for Distance Computation, Communications of the ACM (2017),
    Keenan Crane, Clarisse Weischedel, Max Wardetzky"
    """

    def __init__(self, shape):
        super().__init__(shape)
        self.solver = pp3d.MeshHeatMethodDistanceSolver(shape.vertices, shape.faces)

    def dist_matrix(self):
        """Distance between mesh vertices.

        Returns
        -------
        dist_matrix : array-like, shape=[n_vertices, n_vertices]
            Distance matrix.

        Notes
        -----
        slow
        """
        dist_mat = np.empty((self._shape.n_vertices, self._shape.n_vertices))
        for i in range(self._shape.n_vertices):
            dist_mat[i] = self.solver.compute_distance(i)

        return dist_mat

    def _dist_from_source_single(self, source_point):
        """Distance between mesh vertices.

        Parameters
        ----------
        source_point : array-like, shape=()
            Index of source point.

        Returns
        -------
        dist : array-like, shape=[n_vertices]
            Distance.
        target_point : array-like, shape=[n_vertices,]
            Target index.
        """
        dist = self.solver.compute_distance(source_point.item())

        target_point = np.arange(self._shape.n_vertices)

        return dist, target_point

    def _dist_single(self, point_a, point_b):
        """Distance between mesh vertices.

        Parameters
        ----------
        point_a : array-like, shape=()
            Index of source point.
        point_b : array-like, shape=()
            Index of target point.

        Returns
        -------
        dist : numeric
            Distance.
        """
        dist = self.solver.compute_distance(point_a)[point_b]

        return dist
