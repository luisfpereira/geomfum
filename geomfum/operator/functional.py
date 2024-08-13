"""Functional operators."""

import abc

import numpy as np
import pyFM.mesh.geometry

import geomfum._pyfm


class FunctionalOperator(abc.ABC):
    """Functional operator."""

    # TODO: move to operator
    def __init__(self, shape):
        self._shape = shape

    @abc.abstractmethod
    def __call__(self, point):
        """Apply operator.

        Parameters
        ----------
        point : array-like, shape=[..., n_vertices]
        """
        # TODO: update docstrings


class VectorFieldOperator(abc.ABC):
    """Vector field operator."""

    # TODO: really needed?
    def __init__(self, shape):
        self._shape = shape

    @abc.abstractmethod
    def __call__(self, vector):
        """Apply operator.

        Parameters
        ----------
        point : array-like, shape=[..., n_faces, 3]
        """
        # TODO: update docstrings


class FaceValuedGradient(FunctionalOperator):
    """Gradient of a function on a mesh.

    Computes the gradient of a function on f using linear
    interpolation between vertices.
    """

    def __call__(self, point):
        """Apply operator.

        Parameters
        ----------
        point : array-like, shape=[..., n_vertices]
            Function value on each vertex.

        Returns
        -------
        gradient : array-like, shape=[..., n_faces]
            Gradient of the function on each face.
        """
        gradient = pyFM.mesh.geometry.grad_f(
            point.T,
            self._shape.vertices,
            self._shape.faces,
            self._shape.face_normals,
            face_areas=self._shape.face_areas,
        )
        if gradient.ndim > 2:
            return np.moveaxis(gradient, 0, 1)

        return gradient


class FaceDivergenceOperator(VectorFieldOperator):
    """Divergence of a function on a mesh."""

    def __call__(self, vector):
        """Divergence of a vector field on a mesh.

        Parameters
        ----------
        vector : array-like, shape=[..., n_faces, 3]
            Vector field on the mesh.

        Returns
        -------
        divergence : array-like, shape=[..., n_vertices]
            Divergence of the vector field on each vertex.
        """
        if vector.ndim > 2:
            vector = np.moveaxis(vector, 0, 1)

        div = pyFM.mesh.geometry.div_f(
            vector,
            self._shape.vertices,
            self._shape.faces,
            self._shape.face_normals,
            vert_areas=self._shape.vertex_areas,
        )
        if div.ndim > 1:
            return np.moveaxis(div, 0, 1)

        return div


class FaceOrientationOperator(VectorFieldOperator):
    r"""Orientation operator associated to a gradient field.

    For a given function :math:`g` on the vertices, this operator linearly computes
    :math:`< \grad(f) x \grad(g)`, n> for each vertex by averaging along the adjacent
    faces.
    In practice, we compute :math:`< n x \grad(f), \grad(g) >` for simpler computation.
    """

    def __call__(self, vector):
        """Apply operator.

        Parameters
        ----------
        vector : array-like, shape=[..., n_faces, 3]
            Gradient field on the mesh.

        Returns
        -------
        operator : scipy.sparse.csc_matrix or list[sparse.csc_matrix], shape=[n_vertices, n_vertices]
            Orientation operator.
        """
        return geomfum._pyfm.get_orientation_op(
            vector,
            self._shape.vertices,
            self._shape.faces,
            self._shape.face_normals,
            self._shape.vertex_areas,
        )
