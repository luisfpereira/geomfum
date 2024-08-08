import abc
import logging

import numpy as np
import scipy

from fmib.operator.functional import (
    FaceOrientationOperator,
    FaceValuedGradient,
)


class Shape(abc.ABC):
    def __init__(self):
        # TODO: create automated way for computing this?
        # TODO: should this be handled as e.g. laplacian.<>
        self.basis = None

        # TODO: empty np instead?
        # TODO: add function to add them
        self.landmarks = []

    def equip_with_operator(self, name, Operator, allow_overwrite=True, **kwargs):
        name_exists = hasattr(self, name)
        if name_exists:
            if allow_overwrite:
                logging.warning(f"Overriding {name}.")
            else:
                raise ValueError(f"{name} already exists")

        operator = Operator(self, **kwargs)
        setattr(self, name, operator)

        return self


class TriangleMesh(Shape):
    def __init__(self, vertices, faces):
        super().__init__()
        self.vertices = vertices
        self.faces = faces

        self._face_normals = None
        self._face_areas = None
        self._vertex_areas = None

        self._at_init()

    def _at_init(self):
        self.equip_with_operator("face_valued_gradient", FaceValuedGradient)
        self.equip_with_operator("face_orientation_operator", FaceOrientationOperator)

    @property
    def n_vertices(self):
        return self.vertices.shape[0]

    @property
    def n_faces(self):
        return self.faces.shape[0]

    @property
    def face_normals(self):
        """
        Compute face normals of a triangular mesh

        Returns
        -------
        normals : np.ndarray
            (m,3) array of normalized per-face normals
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
        """
        Compute per-face areas of a triangular mesh

        Parameters
        -----------------------------
        vertices : np.ndarray
            (n,3) array of vertices coordinates
        faces    : np.ndarray
            (m,3) array of vertex indices defining faces

        Returns
        -----------------------------
        faces_areas : np.ndarray
            (m,) array of per-face areas
        """
        if self._face_areas is None:
            v1 = self.vertices[self.faces[:, 0]]
            v2 = self.vertices[self.faces[:, 1]]
            v3 = self.vertices[self.faces[:, 2]]
            self._face_areas = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)

        return self._face_areas

    @property
    def vertex_areas(self):
        """
        Compute per-vertex areas of a triangular mesh.
        Area of a vertex, approximated as one third of the sum of the area of its adjacent triangles.

        Parameters
        ----------
        vertices    : np.ndarray
            (n,3) array of vertices coordinates
        faces       : np.ndarray
            (m,3) array of vertex indices defining faces
        faces_areas : np.ndarray, optional
            (m,) array of per-face areas

        Returns
        -------
        vert_areas : np.ndarray
            (n,) array of per-vertex areas
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


class PointCloud(Shape):
    def __init__(self, points):
        super().__init__()
        self.points = points
