import abc

import numpy as np
import pyFM.mesh.geometry

import fmib._pyfm


class FunctionalOperator(abc.ABC):
    # TODO: need to disambiguate properly with ShapeOperator
    # TODO: call it FunctionalOperator instead?

    # TODO: move to operator
    def __init__(self, shape):
        self._shape = shape

    @abc.abstractmethod
    def __call__(self, point):
        pass


class VectorFieldOperator(abc.ABC):
    # TODO: really needed?
    def __init__(self, shape):
        self._shape = shape

    @abc.abstractmethod
    def __call__(self, vector):
        pass


class FaceValuedGradient(FunctionalOperator):
    """
    computes the gradient of a function on f using linear
    interpolation between vertices.
    """

    def __call__(self, point):
        """Apply operator.

        Parameters
        ----------
        point : array-like, shape=[..., n_vertices]
            Function value on each vertex.

        Returns
        --------------------------
        gradient : array-like, shape=[..., n_faces]
            Gradient of the function on each face.
        """
        gradient = pyFM.mesh.geometry.grad_f(
            point.T, self._shape.vertices, self._shape.faces, self._shape.face_normals
        )
        if gradient.ndim > 2:
            return np.moveaxis(gradient, 0, 1)

        return gradient


class FaceOrientationOperator(VectorFieldOperator):
    """
    Compute the orientation operator associated to a gradient field gradf.

    For a given function g on the vertices, this operator linearly computes
    < grad(f) x grad(g), n> for each vertex by averaging along the adjacent faces.
    In practice, we compute < n x grad(f), grad(g) > for simpler computation.

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
        return fmib._pyfm.get_orientation_op(
            vector,
            self._shape.vertices,
            self._shape.faces,
            self._shape.face_normals,
            self._shape.vertex_areas,
        )
