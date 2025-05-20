import abc

import numpy as np
from sklearn.neighbors import NearestNeighbors

import geomfum.wrap as _wrap  # noqa (for register)
from geomfum._registry import (
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

class FarthestPointSampler(BaseSampler):
    """Farthest point Euclidean sampling.

    Parameters
    ----------
    min_n_samples : int
        Minimum number of samples to target.
    dist_func : callable, optional 
        A callable distance function from a point if None, uses the default distance function (Euclidean distance).
    """

    def __init__(self, min_n_samples, dist_func = None):
        super().__init__()
        self.min_n_samples = min_n_samples
        self.dist_func = dist_func

    def sample(self, shape, first_point=None):
        """Sample using farthest point sampling.

        Parameters
        ----------
        shape : TriangleMesh
            Mesh.
        first_point : int, optional
            Index of the first point to sample. If None, samples randomly.

        Returns
        -------
        samples : array-like, shape=[n_samples, 3]
            Coordinates of samples.
        """
        if self.dist_func is None:
            def dist_func(i):
                return np.linalg.norm(shape.vertices - shape.vertices[i, None, :], axis=1)
            self.dist_func = dist_func
        
        return self.farthest_point_sampling_call(
            self.dist_func,
            self.min_n_samples,
            points_pool=shape.n_vertices,
            first_index = first_point,            
        )
    
    def farthest_point_sampling_call(self, dist_func, k, points_pool=None, first_index=None,):
        """Sample points using farthest point sampling.

        Parameters
        ----------
        d_func   : callable - for index i, d_func(i) is a (points_pool,) array of geodesic distance to
                other points
        k        : int - number of points to sample
        points_pool : Number of points. If not specified, checks d_func(0)
        first_index : int - index of the first point to sample. If None, samples randomly

        Returns
        -------
        fps : (k,) array of indices of sampled points
        """
        if dist_func is None:
            raise ValueError("d_func should be a callable") 
        if first_index is None:
            rng = np.random.default_rng()
            inds = [rng.integers(points_pool).item(0)]
        else:
            inds = [first_index]

        if points_pool is None:
            points_pool = dist_func(0)[0].shape
        else:
            assert points_pool > 0
        dists = dist_func(inds[0])

        iterable = range(k-1)
        for i in iterable:
            if i == k-1:
                continue
            newid = np.argmax(dists[0])
            inds.append(newid)
            new_dists = dist_func(newid)
            minimum_dists = np.minimum(dists[0], new_dists[0])
            dists = (minimum_dists, dists[1])
            
        return np.asarray(inds)


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
            raise ValueError("Expects `n_neighbors = 1`.")

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
