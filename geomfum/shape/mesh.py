"""Definition of triangle mesh."""

import numpy as np
import scipy

from geomfum.io import load_mesh
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

    def __init__(self, vertices, faces):
        super().__init__(is_mesh=True)
        self.vertices = np.asarray(vertices)
        self.faces = np.asarray(faces)

        self._edges = None
        self._face_normals = None
        self._face_areas = None
        self._vertex_areas = None
        self._vertex_normals = None
        self._vertex_tangent_frames = None
        self._edge_tangent_vectors = None
        self._gradient_matrix = None
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
    def from_file(cls, filename):
        """Instantiate given a file.

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
    def vertex_normals(self):
        """Compute vertex normals of a triangular mesh.

        Returns
        -------
        normals : array-like, shape=[n_vertices, 3]
            Normalized per-vertex normals.
        """
        if self._vertex_normals is None:
            I = np.concatenate([self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]])
            J = np.zeros(len(I))

            normals_repeated = np.vstack([self.face_normals] * 3)

            vertex_normals = np.zeros_like(self.vertices)
            for c in range(3):
                V = normals_repeated[:, c]

                vertex_normals[:, c] = (
                    scipy.sparse.coo_matrix((V, (I, J)), shape=(self.n_vertices, 1))
                    .toarray()
                    .flatten()
                )

            vertex_normals = vertex_normals / (
                np.linalg.norm(vertex_normals, axis=1, keepdims=True) + 1e-12
            )

            self._vertex_normals = vertex_normals

        return self._vertex_normals

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

    @property
    def vertex_tangent_frames(self):
        """Compute vertex tangent frame.

        Returns
        -------
        tangent_frame : array-like, shape=[n_vertices, 3, 3]
            Tangent frame of the mesh, where:
            - [n_vertices, 0, :] are the X basis vectors
            - [n_vertices, 1, :] are the Y basis vectors
            - [n_vertices, 2, :] are the vertex normals
        """
        if self._vertex_tangent_frames is None:
            normals = self.vertex_normals
            tangent_frame = np.zeros((self.n_vertices, 3, 3))

            tangent_frame[:, 2, :] = normals

            basis_cand1 = np.tile([1, 0, 0], (self.n_vertices, 1))
            basis_cand2 = np.tile([0, 1, 0], (self.n_vertices, 1))

            dot_products = np.sum(normals * basis_cand1, axis=1, keepdims=True)
            basis_x = np.where(np.abs(dot_products) < 0.9, basis_cand1, basis_cand2)

            normal_projections = (
                np.sum(basis_x * normals, axis=1, keepdims=True) * normals
            )
            basis_x = basis_x - normal_projections

            basis_x_norm = np.linalg.norm(basis_x, axis=1, keepdims=True)
            basis_x = basis_x / (basis_x_norm + 1e-12)

            basis_y = np.cross(normals, basis_x)

            tangent_frame[:, 0, :] = basis_x
            tangent_frame[:, 1, :] = basis_y

            self._vertex_tangent_frames = tangent_frame

        return self._vertex_tangent_frames

    @property
    def edge_tangent_vectors(self):
        """Compute edge tangent vectors.

        Returns
        -------
        edge_tangent_vectors : array-like, shape=[n_edges, 2]
            Tangent vectors of the edges, projected onto the local tangent plane.
            Each vector has x and y components in the local tangent frame.
        """
        if self._edge_tangent_vectors is None:
            edges = self.edges
            frames = self.vertex_tangent_frames

            edge_vecs = self.vertices[edges[:, 1], :] - self.vertices[edges[:, 0], :]

            basis_x = frames[edges[:, 0], 0, :]
            basis_y = frames[edges[:, 0], 1, :]

            # Project edge vectors onto the local tangent plane
            comp_x = np.sum(edge_vecs * basis_x, axis=1)
            comp_y = np.sum(edge_vecs * basis_y, axis=1)

            self._edge_tangent_vectors = np.stack((comp_x, comp_y), axis=-1)

        return self._edge_tangent_vectors

    @property
    def gradient_matrix(self):
        # TODO: Implement this as operator
        """Compute the gradient operator as a complex sparse matrix.

        This code locally fits a linear function to the scalar values at each vertex and its neighbors, extracts the gradient in the tangent plane, and assembles the global sparse matrix that acts as the discrete gradient operator on the mesh.

        Returns
        -------
        grad_op : scipy.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Complex sparse matrix representing the gradient operator.
            The real part corresponds to the X component in the local tangent frame,
            and the imaginary part corresponds to the Y component.
        """
        if self._gradient_matrix is None:
            # Build a list of outgoing edges for each vertex (neighbor list)
            outgoing_edges_per_vertex = [[] for _ in range(self.n_vertices)]
            for edge_index in range(self.edges.shape[0]):
                tail_ind = self.edges[edge_index, 0]
                tip_ind = self.edges[edge_index, 1]
                if tip_ind != tail_ind:
                    outgoing_edges_per_vertex[tail_ind].append(edge_index)

            row_inds = []
            col_inds = []
            data_vals = []
            eps_reg = 1e-5  # Regularization for numerical stability

            # For each vertex, fit a local linear function 'f' to its neighbors
            for vertex_idx in range(self.n_vertices):
                num_neighbors = len(outgoing_edges_per_vertex[vertex_idx])

                # Skip isolated vertices
                if num_neighbors == 0:
                    continue

                # Set up the least squares system for the local neighborhood
                lhs_mat = np.zeros((num_neighbors, 2))  # Edge tangent vectors
                rhs_mat = np.zeros(
                    (num_neighbors, num_neighbors + 1)
                )  # Finite Difference matrix rhs_mat[i,j] = f(j) - f(i)
                lookup_vertices_idx = [vertex_idx]

                # for each row of the rhs_mat, we have the following:
                # - rhs_mat[i, 0] = -f(center) (the value at the center vertex)
                # - rhs_mat[i, i + 1] = +f(neighbor) (the value at the neighbor vertex)
                # - rhs_mat[i, j] = 0 for j != 0, i + 1 (no other values)
                for neighbor_index in range(num_neighbors):
                    edge_index = outgoing_edges_per_vertex[vertex_idx][neighbor_index]
                    neigbor_vertex_idx = self.edges[edge_index, 1]
                    lookup_vertices_idx.append(neigbor_vertex_idx)

                    edge_vec = self.edge_tangent_vectors[edge_index][:]

                    lhs_mat[neighbor_index][:] = edge_vec
                    rhs_mat[neighbor_index][0] = -1
                    rhs_mat[neighbor_index][neighbor_index + 1] = 1

                # Solve
                lhs_T = lhs_mat.T
                lhs_inv = np.linalg.inv(lhs_T @ lhs_mat + eps_reg * np.eye(2)) @ lhs_T
                sol_mat = lhs_inv @ rhs_mat
                sol_coefs = (sol_mat[0, :] + 1j * sol_mat[1, :]).T

                for i_neigh in range(num_neighbors + 1):
                    i_glob = lookup_vertices_idx[i_neigh]
                    row_inds.append(vertex_idx)
                    col_inds.append(i_glob)
                    data_vals.append(sol_coefs[i_neigh])

            # Assemble the global sparse gradient operator matrix
            row_inds = np.array(row_inds)
            col_inds = np.array(col_inds)
            data_vals = np.array(data_vals)

            self._gradient_matrix = scipy.sparse.coo_matrix(
                (data_vals, (row_inds, col_inds)),
                shape=(self.n_vertices, self.n_vertices),
            ).tocsc()
        return self._gradient_matrix
