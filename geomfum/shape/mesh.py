"""Definition of triangle mesh."""

import geomstats.backend as gs
import scipy

import geomfum.backend as xgs
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
        self.vertices = gs.asarray(vertices)
        self.faces = gs.asarray(faces)

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
            vind012 = gs.concatenate(
                [self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]]
            )
            vind120 = gs.concatenate(
                [self.faces[:, 1], self.faces[:, 2], self.faces[:, 0]]
            )
            edges = gs.stack(
                [
                    gs.concatenate([vind012, vind120]),
                    gs.concatenate([vind120, vind012]),
                ],
                axis=-1,
            )
            edges = gs.unique(edges, axis=0)
            self._edges = edges[edges[:, 1] > edges[:, 0]]

        return self._edges

    @property
    def face_vertex_coords(self):
        """Extract vertex coordinates corresponding to each face.

        Returns
        -------
        vertices : array-like, shape=[{n_faces}, n_per_face_vertex, 3]
            Coordinates of the ith vertex of that face.
        """
        return gs.stack(
            [
                self.vertices[self.faces[:, 0]],
                self.vertices[self.faces[:, 1]],
                self.vertices[self.faces[:, 2]],
            ],
            axis=-2,
        )

    @property
    def face_normals(self):
        """Compute face normals of a triangular mesh.

        Returns
        -------
        normals : array-like, shape=[n_faces, 3]
            Per-face normals.
        """
        if self._face_normals is None:
            face_vertex_coords = self.face_vertex_coords
            self._face_normals = gs.cross(
                face_vertex_coords[:, 1, :] - face_vertex_coords[:, 0, :],
                face_vertex_coords[:, 2, :] - face_vertex_coords[:, 0, :],
            )

        return self._face_normals

    @property
    def unit_face_normals(self):
        """Compute face normals of a triangular mesh.

        Returns
        -------
        normals : array-like, shape=[n_faces, 3]
            Per-face normals.
        """
        return self.face_normals / gs.linalg.norm(
            self.face_normals, axis=1, keepdims=True
        )

    @property
    def face_areas(self):
        """Compute per-face areas.

        Returns
        -------
        face_areas : array-like, shape=[n_faces]
            Per-face areas.
        """
        if self._face_areas is None:
            self._face_areas = 0.5 * gs.linalg.norm(self.face_normals, axis=1)

        return self._face_areas

    @property
    def vertex_normals(self):
        """Compute vertex normals of a triangular mesh.

        Returns
        -------
        normals : array-like, shape=[n_vertices, 3]
            Normalized per-vertex normals.
        """
        if self._vertex_normals is None:
            I = gs.concatenate([self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]])
            J = gs.zeros(len(I))

            normals_repeated = gs.vstack([self.face_normals] * 3)

            vertex_normals = gs.zeros_like(self.vertices)
            for c in range(3):
                V = normals_repeated[:, c]

                vertex_normals[:, c] = gs.array(
                    scipy.sparse.coo_matrix((V, (I, J)), shape=(self.n_vertices, 1))
                    .toarray()
                    .flatten()
                )

            vertex_normals = vertex_normals / (
                gs.linalg.norm(vertex_normals, axis=1, keepdims=True) + 1e-12
            )

            self._vertex_normals = vertex_normals

        return self._vertex_normals

    @property
    def vertex_areas(self):
        """Compute per-vertex areas.

        Area of a vertex, approximated as one third of the sum of the area of its adjacent triangles.

        Returns
        -------
        vertex_areas : array-like, shape=[n_vertices]
            Per-vertex areas.
        """
        area = self.face_areas

        id_vertices = gs.broadcast_to(gs.reshape(self.faces, (-1,)), self.n_faces * 3)
        val = gs.reshape(
            gs.broadcast_to(gs.expand_dims(area, axis=-1), (self.n_faces, 3)),
            (-1,),
        )
        incident_areas = xgs.scatter_sum_1d(
            index=id_vertices,
            src=val,
        )
        return incident_areas / 3.0

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
            tangent_frame = gs.zeros((self.n_vertices, 3, 3))

            tangent_frame[:, 2, :] = normals

            basis_cand1 = gs.tile([1, 0, 0], (self.n_vertices, 1))
            basis_cand2 = gs.tile([0, 1, 0], (self.n_vertices, 1))

            dot_products = gs.sum(normals * basis_cand1, axis=1, keepdims=True)
            basis_x = gs.where(gs.abs(dot_products) < 0.9, basis_cand1, basis_cand2)

            normal_projections = (
                gs.sum(basis_x * normals, axis=1, keepdims=True) * normals
            )
            basis_x = basis_x - normal_projections

            basis_x_norm = gs.linalg.norm(basis_x, axis=1, keepdims=True)
            basis_x = basis_x / (basis_x_norm + 1e-12)

            basis_y = gs.cross(normals, basis_x)

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
            comp_x = gs.sum(edge_vecs * basis_x, axis=1)
            comp_y = gs.sum(edge_vecs * basis_y, axis=1)

            self._edge_tangent_vectors = gs.stack((comp_x, comp_y), axis=-1)

        return self._edge_tangent_vectors

    @property
    def gradient_matrix(self):
        # TODO: Implement this as operator
        """Compute the gradient operator as a complex sparse matrix.

        The gradient operator maps scalar functions defined on vertices to
        vector fields in the tangent plane at each vertex.

        Returns
        -------
        grad_op : scipy.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Complex sparse matrix representing the gradient operator.
            The real part corresponds to the X component in the local tangent frame,
            and the imaginary part corresponds to the Y component.
        """
        if self._gradient_matrix is None:
            verts = self.vertices
            edges = self.edges.T  # Transpose to match the [2, E] format expected
            edge_tangent_vecs = self.edge_tangent_vectors

            # Build outgoing neighbor lists
            V = verts.shape[0]
            vert_edge_outgoing = [[] for _ in range(V)]
            for e in range(edges.shape[1]):
                tail_ind = edges[0, e]
                tip_ind = edges[1, e]
                if tip_ind != tail_ind:
                    vert_edge_outgoing[tail_ind].append(e)

            # Build local inversion matrix for each vertex
            row_inds = []
            col_inds = []
            data_vals = []
            eps_reg = 1e-5

            for iv in range(V):
                n_neigh = len(vert_edge_outgoing[iv])

                # Skip vertices with no outgoing edges
                if n_neigh == 0:
                    continue

                lhs_mat = gs.zeros((n_neigh, 2))
                rhs_mat = gs.zeros((n_neigh, n_neigh + 1))
                ind_lookup = [iv]

                for i_neigh in range(n_neigh):
                    ie = vert_edge_outgoing[iv][i_neigh]
                    jv = edges[1, ie]
                    ind_lookup.append(jv)

                    edge_vec = edge_tangent_vecs[ie][:]
                    w_e = 1.0

                    lhs_mat[i_neigh][:] = w_e * edge_vec
                    rhs_mat[i_neigh][0] = w_e * (-1)
                    rhs_mat[i_neigh][i_neigh + 1] = w_e * 1

                lhs_T = lhs_mat.T
                lhs_inv = gs.linalg.inv(lhs_T @ lhs_mat + eps_reg * gs.eye(2)) @ lhs_T

                sol_mat = lhs_inv @ rhs_mat
                sol_coefs = (sol_mat[0, :] + 1j * sol_mat[1, :]).T

                for i_neigh in range(n_neigh + 1):
                    i_glob = ind_lookup[i_neigh]

                    row_inds.append(iv)
                    col_inds.append(i_glob)
                    data_vals.append(sol_coefs[i_neigh])

            # Build the sparse matrix
            row_inds = gs.array(row_inds)
            col_inds = gs.array(col_inds)
            data_vals = gs.array(data_vals)

            self._gradient_matrix = scipy.sparse.coo_matrix(
                (data_vals, (row_inds, col_inds)), shape=(V, V)
            ).tocsc()

        return xgs.sparse.from_scipy_csc(self._gradient_matrix)
