"""Sampling methods."""

import abc

import numpy as np
from sklearn.neighbors import NearestNeighbors

import geomfum.wrap as _wrap  # noqa (for register)
from geomfum._registry import (
    FarthestPointSamplerRegistry,
    PoissonSamplerRegistry,
    WhichRegistryMixins,
)


class BaseSampler(abc.ABC):
    """Sampler."""

    @abc.abstractmethod
    def sample(self, shape):
        """Sample shape."""


class PoissonSampler(WhichRegistryMixins):
    """Poisson disk sampling."""

    _Registry = PoissonSamplerRegistry


class FarthestPointSampler(WhichRegistryMixins):
    """Farthest point Euclidean sampling."""

    _Registry = FarthestPointSamplerRegistry


class VertexProjectionSampler(BaseSampler):
    """Sample by projecting samples to the closest vertex.

    Uses nearest neighbor to get indices of sample coordinates
    resulting from another sampler.

    Parameters
    ----------
    min_n_samples : int
        Minimum number of samples to target.
        Ignored if ``sampler`` is not None.
        Not guaranteed if ``unique`` is True.
    sampler : BaseSampler
        Coordinates sampler.
    neighbor_finder : sklearn.NearestNeighbors
        Nearest neighbors finder.
    unique : bool
        Whether to remove duplicates.
    """

    def __init__(
        self, min_n_samples=100, sampler=None, neighbor_finder=None, unique=False
    ):
        super().__init__()
        if sampler is None:
            sampler = PoissonSampler.from_registry(min_n_samples=min_n_samples)

        if neighbor_finder is None:
            neighbor_finder = NearestNeighbors(
                n_neighbors=1, leaf_size=40, algorithm="kd_tree", n_jobs=1
            )
        if neighbor_finder.n_neighbors > 1:
            raise ValueError("Expects `n_neighors = 1`.")

        self.neighbor_finder = neighbor_finder
        self.sampler = sampler
        self.unique = unique

    def sample(self, shape):
        """Sample using Poisson disk sampling.

        Parameters
        ----------
        shape : Shape
            Shape to be sampled.

        Returns
        -------
        samples : array-like, shape=[n_samples]
            Vertex indices of samples.
        """
        sampled_points = self.sampler.sample(shape)

        self.neighbor_finder.fit(shape.vertices)
        _, neighbor_indices = self.neighbor_finder.kneighbors(sampled_points)

        if self.unique:
            return np.unique(neighbor_indices)

        return np.squeeze(neighbor_indices)
