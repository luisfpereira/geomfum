"""Hierarchical objects."""

import abc

from geomfum._registry import HierarchicalMeshRegistry, WhichRegistryMixins
from geomfum.basis import EigenBasis


class HierarchicalShape(abc.ABC):
    """Hierarchical shape.

    Parameters
    ----------
    low : Shape
        Low-resolution shape.
    high : Shape
        High-resolution shape.
    """

    def __init__(self, low, high):
        self.low = low
        self.high = high

    @abc.abstractmethod
    def scalar_low_high(self, scalar):
        """Transfer scalar from low-resolution to high.

        Parameters
        ----------
        scalar : array-like, shape=[..., low.n_vertices]
            Scalar map on the low-resolution shape.

        Returns
        -------
        high_scalar : array-like, shape=[..., high.n_vertices]
            Scalar map on the high-resolution shape.
        """

    def extend_basis(self, set_as_basis=True):
        """Extend basis.

        See section 3.3. of [MBMR2023]_ for details.

        References
        ----------
        .. [MBMR2023] Filippo Maggioli, Daniele Baieri, Simone Melzi, and Emanuele Rodolà.
           “ReMatching: Low-Resolution Representations for Scalable Shape
            Correspondence.” arXiv, October 30, 2023.
            https://doi.org/10.48550/arXiv.2305.09274.
        """
        hvecs = self.scalar_low_high(self.low.basis.full_vecs.T).T

        basis = EigenBasis(self.low.basis.full_vals, hvecs)

        if set_as_basis:
            self.high.basis = basis

        return basis


class HierarchicalMesh(WhichRegistryMixins, HierarchicalShape):
    """Hierarchical mesh.

    Parameters
    ----------
    low : TriangleMesh
        Low resolution shape.
    high : TriangleMesh
        High resolution shape.
    """

    _Registry = HierarchicalMeshRegistry
