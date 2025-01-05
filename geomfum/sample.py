import abc

import numpy as np
from sklearn.neighbors import NearestNeighbors

import geomfum.wrap as _wrap  # noqa (for register)
from geomfum._registry import PoissonSamplerRegistry, FpSamplerRegistry, WhichRegistryMixins


class BaseSampler(abc.ABC):
    @abc.abstractmethod
    def sample(self, shape):
        pass


class PoissonSampler(WhichRegistryMixins):
    _Registry = PoissonSamplerRegistry

class FpSampler(BaseSampler):
    _Registry= FpSamplerRegistry

class NearestNeighborsIndexSampler(BaseSampler):
    """

    Parameters
    ----------
    min_n_samples : int
        Minimum number of samples to target.
        Ignored if ``sampler is not None``.
    """

    # uses nearest neighbor to get indices of sample coordinates
    # resulting from another sampler

    # TODO: find better naming as this is confusing

    def __init__(self, min_n_samples=100, sampler=None, neighbor_finder=None):
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

    def sample(self, shape):
        # returns array[index]
        sampled_points = self.sampler.sample(shape)

        self.neighbor_finder.fit(shape.vertices)
        _, neighbor_indices = self.neighbor_finder.kneighbors(sampled_points)

        return np.unique(neighbor_indices)

class FarthestPointSampler(BaseSampler):
    def __init__(self, n_samples=100):
        self.n_samples = n_samples
        self.sampler = FpSampler.from_registry(min_n_samples=n_samples)

    def sample(self, shape):
        # returns array[index]
        sampled_points = self.sampler.sample(shape)
        
        return np.unique(sampled_points)