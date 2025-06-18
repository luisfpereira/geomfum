"""Conversion between pointwise and functional maps."""

import abc

import geomstats.backend as gs
import geomfum.backend as xgs
import scipy
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
import geomfum.wrap as _wrap  # noqa (for register)
from geomfum._registry import (
    SinkhornNeighborFinderRegistry,
    WhichRegistryMixins,
    TorchNeighborFinderRegistry,
)


class BaseP2pFromFmConverter(abc.ABC):
    """Pointwise map from functional map."""


class P2pFromFmConverter(BaseP2pFromFmConverter):
    """Pointwise map from functional map.

    Parameters
    ----------
    neighbor_finder : NeighborFinder
        Nearest neighbor finder.
    adjoint : bool
        Whether to use adjoint method.
    bijective : bool
        Whether to use bijective method. Check [VM2023]_.

    References
    ----------
    .. [OCSBG2012] Maks Ovsjanikov, Mirela Ben-Chen, Justin Solomon,
        Adrian Butscher, and Leonidas Guibas.
        “Functional Maps: A Flexible Representation of Maps between
        Shapes.” ACM Transactions on Graphics 31, no. 4 (2012): 30:1-30:11.
        https://doi.org/10.1145/2185520.2185526.
    .. [VM2023] Giulio Viganò  Simone Melzi. “Adjoint Bijective ZoomOut:
        Efficient Upsampling for Learned Linearly-Invariant Embedding.”
        The Eurographics Association, 2023. https://doi.org/10.2312/stag.20231293.
    """

    def __init__(self, neighbor_finder=None, adjoint=False, bijective=False):
        if neighbor_finder is None:
            neighbor_finder = NearestNeighbors(
                n_neighbors=1, leaf_size=40, algorithm="kd_tree", n_jobs=1
            )
        if neighbor_finder.n_neighbors > 1:
            raise ValueError("Expects `n_neighors = 1`.")

        self.neighbor_finder = neighbor_finder
        self.adjoint = adjoint
        self.bijective = bijective

    def __call__(self, fmap_matrix, basis_a, basis_b):
        """Convert functional map.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.

        Returns
        -------
        p2p : array-like, shape=[{n_vertices_b, n_vertices_a}]
            Pointwise map. ``bijective`` controls shape.
        """
        k2, k1 = fmap_matrix.shape

        if self.adjoint:
            emb1 = basis_a.full_vecs[:, :k1]
            emb2 = basis_b.full_vecs[:, :k2] @ fmap_matrix

        else:
            emb1 = basis_a.full_vecs[:, :k1] @ fmap_matrix.T
            emb2 = basis_b.full_vecs[:, :k2]

        if self.bijective:
            emb1, emb2 = emb2, emb1

        # TODO: update neighbor finder instead
        self.neighbor_finder.fit(xgs.to_device(emb1, "cpu"))
        p2p_21 = self.neighbor_finder.kneighbors(
            xgs.to_device(emb2, "cpu"), return_distance=False
        )

        return gs.from_numpy(p2p_21[:, 0])


class BaseNeighborFinder(abc.ABC):
    """Base class for a Neighbor finder.

    A simplified blueprint of ``sklearn.NearestNeighbors`` implementation.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors.
    """

    def __init__(self, n_neighbors=1):
        self.n_neighbors = 1

    @abc.abstractmethod
    def fit(self, X, y=None):
        """Store the reference points.

        Parameters
        ----------
        X : array-like, shape=[n_points_x, n_features]
            Reference points.
        y : Ignored
        """

    @abc.abstractmethod
    def kneighbors(self, X, return_distance=True):
        """Find k nearest neighbors using Sinkhorn regularization.

        Parameters
        ----------
        X : array-like, shape=[n_points_y, n_features]
            Query points.
        return_distance : bool
            Whether to return the distances.

        Returns
        -------
        distances : array-like, shape=[n_points_y, n_neighbors]
            Distances to the nearest neighbors, only present if
            ``return_distance is True``.
        indices : array-like, shape=[n_points_y, n_neighbors]
            Indices of the nearest neighbors.
        """


class TorchNeighborFinder(WhichRegistryMixins):
    """Torch-based neighbor finder."""

    _Registry = TorchNeighborFinderRegistry


class SinkhornNeighborFinder(WhichRegistryMixins):
    """Sinkhorn neighbor finder.

    Finds neighbors based on the solution of optimal transport (OT) maps
    computed with Sinkhorn regularization.

    References
    ----------
    .. [Cuturi2013] Marco Cuturi. “Sinkhorn Distances: Lightspeed Computation
        of Optimal Transport.”
        Advances in Neural Information Processing Systems (NIPS), 2013.
        http://marcocuturi.net/SI.html
    """

    _Registry = SinkhornNeighborFinderRegistry


class SinkhornP2pFromFmConverter(P2pFromFmConverter):
    """Pointwise map from functional map using Sinkhorn filters.

    Parameters
    ----------
    neighbor_finder : SinkhornKNeighborsFinder
        Nearest neighbor finder.
    adjoint : bool
        Whether to use adjoint method.
    bijective : bool
        Whether to use bijective method. Check [VM2023]_.

    References
    ----------
    .. [PRMWO2021] Gautam Pai, Jing Ren, Simone Melzi, Peter Wonka, and Maks Ovsjanikov.
        "Fast Sinkhorn Filters: Using Matrix Scaling for Non-Rigid Shape Correspondence
        with Functional Maps." Proceedings of the IEEE/CVF Conference on Computer Vision
        and Pattern Recognition (CVPR), 2021, pp. 11956-11965.
        https://hal.science/hal-03184936/document
    """

    def __init__(
        self,
        neighbor_finder=None,
        adjoint=False,
        bijective=False,
    ):
        if neighbor_finder is None:
            neighbor_finder = SinkhornNeighborFinder.from_registry(which="pot")

        super().__init__(
            neighbor_finder=neighbor_finder,
            adjoint=adjoint,
            bijective=bijective,
        )


class BaseFmFromP2pConverter(abc.ABC):
    """Functional map from pointwise map."""


class FmFromP2pConverter(BaseFmFromP2pConverter):
    """Functional map from pointwise map.

    Parameters
    ----------
    pseudo_inverse : bool
        Whether to solve using pseudo-inverse.
    """

    # TODO: add subsampling
    def __init__(self, pseudo_inverse=False):
        self.pseudo_inverse = pseudo_inverse

    def __call__(self, p2p, basis_a, basis_b):
        """Convert point to point map.

        Parameters
        ----------
        p2p : array-like, shape=[n_vertices_b]
            Poinwise map.

        Returns
        -------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        """
        evects1_pb = basis_a.vecs[p2p, :]

        if self.pseudo_inverse:
            return basis_b.vecs.T @ (basis_b._shape.laplacian.mass_matrix @ evects1_pb)

        return gs.from_numpy(scipy.linalg.lstsq(basis_b.vecs, evects1_pb)[0])


class FmFromP2pBijectiveConverter(BaseFmFromP2pConverter):
    """Bijective functional map from pointwise map method.

    References
    ----------
    .. [VM2023] Giulio Viganò  Simone Melzi. “Adjoint Bijective ZoomOut:
        Efficient Upsampling for Learned Linearly-Invariant Embedding.”
        The Eurographics Association, 2023. https://doi.org/10.2312/stag.20231293.
    """

    def __call__(self, p2p, basis_a, basis_b):
        """Convert point to point map.

        Parameters
        ----------
        p2p : array-like, shape=[n_vertices_a]
            Pointwise map.

        Returns
        -------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        """
        evects2_pb = basis_b.vecs[p2p, :]
        return gs.from_numpy(scipy.linalg.lstsq(evects2_pb, basis_a.vecs)[0])
