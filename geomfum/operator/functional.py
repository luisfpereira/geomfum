"""Functional operators."""

import abc

from geomfum._registry import (
    FaceDivergenceOperatorRegistry,
    FaceOrientationOperatorRegistry,
    FaceValuedGradientRegistry,
    WhichRegistryMixins,
)


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


class FaceValuedGradient(WhichRegistryMixins, FunctionalOperator):
    """Gradient of a function on a mesh.

    Computes the gradient of a function on f using linear
    interpolation between vertices.
    """

    _Registry = FaceValuedGradientRegistry


class FaceDivergenceOperator(WhichRegistryMixins, VectorFieldOperator):
    """Divergence of a function on a mesh."""

    _Registry = FaceDivergenceOperatorRegistry


class FaceOrientationOperator(WhichRegistryMixins, VectorFieldOperator):
    r"""Orientation operator associated to a gradient field.

    For a given function :math:`g` on the vertices, this operator linearly computes
    :math:`< \grad(f) x \grad(g)`, n> for each vertex by averaging along the adjacent
    faces.
    In practice, we compute :math:`< n x \grad(f), \grad(g) >` for simpler computation.
    """

    _Registry = FaceOrientationOperatorRegistry
