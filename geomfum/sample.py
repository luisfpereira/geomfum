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
    """Farthest point sampling.

    Parameters
    ----------
    min_n_samples : int
        Minimum number of samples to target.
    metric : class, optional 
        A metric class, if None uses the default metric class (Euclidean distance).
    """

    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

    def sample(self, shape, first_point=None, points_pool=None):
        """
        Perform farthest point sampling on a mesh.

        Parameters
        ----------
        shape : TriangleMesh
            The mesh to sample points from.
        first_point : int, optional
            Index of the initial point to start sampling from. If None, a random point is chosen.
        points_pool : array-like or int, optional
            Pool of candidate points to sample from. If None, all vertices in the mesh are used.

        Returns
        -------
        samples : array-like of shape (n_samples, 3)
            Coordinates of the sampled points.
        """
        if points_pool is None:
            return self._farthest_point_sampling_call(
                shape,
                points_pool = shape.n_vertices,
                first_index = first_point,
            )
        else:
            return self._farthest_point_sampling_call_sub(
                shape,
                points_pool = points_pool,
                first_index = first_point,
            )
   
    def _farthest_point_sampling_call(self, mesh, points_pool= None, first_index=None,):
        """Sample points using farthest point sampling.

        Parameters
        ----------
        mesh : TriangleMesh
            Mesh to sample from.
        points_pool : array, optional
            The set of points (all mesh vertices) from which to sample. 
        first_index : int 
            Index of the first point to sample. If None, samples randomly

        Returns
        -------
        fps : (k,) array of indices of sampled points
        """
        dist_func = mesh.metric.dist_from_source
        if mesh.metric is None:
            raise ValueError("d_func should be a callable")

        if first_index is None:
            rng = np.random.default_rng()
            inds = [rng.integers(points_pool).item(0)]
        else:
            inds = [first_index]

        dists = dist_func(inds[0])
        iterable = range(self.n_samples-1)
        for i in iterable:
            if i == self.n_samples-1:
                continue
            newid = np.argmax(dists[0])
            inds.append(newid)
            new_dists = dist_func(newid)
            minimum_dists = np.minimum(dists[0], new_dists[0])
            dists = (minimum_dists, dists[1])

        return np.asarray(inds)
    
    def _farthest_point_sampling_call_sub(self, mesh, points_pool, first_index=None) :       
        """
        Sample points using farthest point sampling.

        Parameters
        ----------
        mesh : TriangleMesh
            Mesh to sample from.
        points_pool : array
            The subset of points from which to sample. 
        first_index : int 
            Index of the first point to sample. If None, samples randomly

        Returns
        -------
        fps : (k,) array of indices of sampled points
        """
        dist_func = mesh.metric.dist_from_source
        sub_points = np.array(points_pool)

        if first_index is None:
            rng = np.random.default_rng()
            start_subid = rng.choice(sub_points)
            inds = [start_subid]
        else:
            inds = [first_index]

        dists = dist_func(inds[0])[0][sub_points]

        iterable = range(self.n_samples - 1)
        for i in iterable:
            if i == self.n_samples - 1:
                continue

            new_subid = np.argmax(dists)
            newid = sub_points[new_subid]
            inds.append(newid)
            dists = np.minimum(dists, dist_func(newid)[0][sub_points])

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
