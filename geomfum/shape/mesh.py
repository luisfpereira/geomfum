"""Definition of triangle mesh."""

import numpy as np
import scipy

from geomfum.io import load_mesh
from geomfum.metric.mesh import HeatDistanceMetric
from geomfum.operator import (
    FaceDivergenceOperator,
    FaceOrientationOperator,
    FaceValuedGradient,
)

from ._base import Shape


class TriangleMesh(Shape):
    """Triangle mesh.

    Parameters
    ----------
    vertices : array-like, shape=[n_vertices, 3]
        Vertices of the mesh.
    faces : array-like, shape=[n_faces, 3]
        Faces of the mesh.
    """

    def __init__(self, vertices, faces,):
        super().__init__(is_mesh=True)
        self.vertices = np.asarray(vertices)
        self.faces = np.asarray(faces)


        self._edges = None
        self._face_normals = None
        self._face_areas = None
        self._vertex_areas = None
        self._dist_matrix = None
        self.metric = None
    
        self._at_init()

    def _at_init(self):
        self.equip_with_operator(
            "face_valued_gradient", FaceValuedGradient.from_registry
        )
        self.equip_with_operator(
            "face_divergence", FaceDivergenceOperator.from_registry
        )
        self.equip_with_operator(
            "face_orientation_operator", FaceOrientationOperator.from_registry
        )

    @classmethod
    def from_file(cls, filename,):
        """Instantiate given a file.

        Parameters
        ----------
        filename : str
            Path to the mesh file.     

        Returns
        -------
        mesh : TriangleMesh
            A triangle mesh.
        """
        vertices, faces = load_mesh(filename)
        return cls(vertices, faces)

    @property
    def n_vertices(self):
        """Number of vertices.

        Returns
        -------
        n_vertices : int
        """
        return self.vertices.shape[0]

    @property
    def n_faces(self):
        """Number of faces.

        Returns
        -------
        n_faces : int
        """
        return self.faces.shape[0]

    @property
    def edges(self):
        """Edges of the mesh.

        Returns
        -------
        edges : array-like, shape=[n_edges, 2]
        """
        if self._edges is None:
            I = np.concatenate([self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]])
            J = np.concatenate([self.faces[:, 1], self.faces[:, 2], self.faces[:, 0]])

            In = np.concatenate([I, J])
            Jn = np.concatenate([J, I])
            Vn = np.ones_like(In)

            M = scipy.sparse.csr_matrix(
                (Vn, (In, Jn)), shape=(self.n_vertices, self.n_vertices)
            ).tocoo()

            edges0 = M.row
            edges1 = M.col

            indices = M.col > M.row

            self._edges = np.concatenate(
                [edges0[indices, None], edges1[indices, None]], axis=1
            )
        return self._edges

    @property
    def face_normals(self):
        """Compute face normals of a triangular mesh.

        Returns
        -------
        normals : array-like, shape=[n_faces, 3]
            Normalized per-face normals.
        """
        if self._face_normals is None:
            v1 = self.vertices[self.faces[:, 0]]
            v2 = self.vertices[self.faces[:, 1]]
            v3 = self.vertices[self.faces[:, 2]]

            normals = np.cross(v2 - v1, v3 - v1)
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)

            self._face_normals = normals

        return self._face_normals

    @property
    def face_areas(self):
        """Compute per-face areas.

        Returns
        -------
        face_areas : array-like, shape=[n_faces]
            Per-face areas.
        """
        if self._face_areas is None:
            v1 = self.vertices[self.faces[:, 0]]
            v2 = self.vertices[self.faces[:, 1]]
            v3 = self.vertices[self.faces[:, 2]]
            self._face_areas = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)

        return self._face_areas

    @property
    def vertex_areas(self):
        """Compute per-vertex areas.

        Area of a vertex, approximated as one third of the sum of the area of its adjacent triangles.

        Returns
        -------
        vertex_areas : array-like, shape=[n_vertices]
            Per-vertex areas.
        """
        if self._vertex_areas is None:
            # THIS IS JUST A TRICK TO BE FASTER THAN NP.ADD.AT
            I = np.concatenate([self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]])
            J = np.zeros_like(I)

            V = np.tile(self.face_areas / 3, 3)

            self._vertex_areas = np.array(
                scipy.sparse.coo_matrix(
                    (V, (I, J)), shape=(self.n_vertices, 1)
                ).todense()
            ).flatten()

        return self._vertex_areas

    @property   #ToDo
    def dist_matrix(self):
        """Compute metric distance matrix.

        Returns
        -------
        d_matrix : array-like, shape=[n_vertices, n_vertices]
            Metric distance matrix.
        """
        if self._d_matrix is None:
            if self.metric is None:
                raise ValueError("Metric is not set.")
            self._d_matrix = self.metric.dist_matrix()
        return self._d_matrix

    def equip_with_metric(self, metric):
        """Set the metric for the mesh.

        Parameters
        ----------
        metric : class
            A metric class to use for the mesh.
        """
        if metric == HeatDistanceMetric:
            self.metric = metric.from_registry(which="pp3d",shape = self)
        else:
            self.metric = metric(self)
        self._dist_matrix = None
