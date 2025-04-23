import abc

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path


def to_nx_edge_graph(shape):
    # TODO: move to utils? circular imports
    vertex_a, vertex_b = shape.edges.T
    lengths = EuclideanMetric(shape).dist(vertex_a, vertex_b)

    weighted_edges = [
        (int(vertex_a_), int(vertex_b_), length)
        for vertex_a_, vertex_b_, length in zip(vertex_a, vertex_b, lengths)
    ]

    graph = nx.Graph()
    graph.add_weighted_edges_from(weighted_edges)

    return graph


def _is_single_index(index):
    return isinstance(index, int) or index.ndim == 0


class BaseMetric(abc.ABC):
    # TODO: may need to do intermediate abstractions
    def __init__(self, shape):
        self._shape = shape

    @abc.abstractmethod
    def dist(self, point_a, point_b):
        """Distance between points.

        Parameters
        ----------
        point_a : array-like, shape=[...]
            Index Point.
        point_b : array-like, shape=[...]
            Index Point.

        Returns
        -------
        dist : array-like, shape=[...,]
            Distance.
        """
    @abc.abstractmethod
    def dist_matrix(self):
        """Distance between all the points of a shape.

        Returns
        -------
        dist_matrix : array-like, shape=[n_vertices, n_vertices]
            Distance matrix.
        """

class EuclideanMetric(BaseMetric):
    def dist(self, point_a, point_b=None):
        """Distance between mesh vertices.

        Parameters
        ----------
        point_a : array-like, shape=[...]
            Index of source point.
        point_b : array-like, shape=[...]
            Index of target point.

        Returns
        -------
        dist : array-like, shape=[...,]
            Distance.
        """
        # TODO: add euclidean with cutoff
        vertices = self._shape.vertices
    
        if point_b is None:
            diff = vertices[point_a][:,np.newaxis,:] - vertices[np.newaxis, :, :]
            return np.linalg.norm(diff, axis=diff.ndim - 1), np.arange(self._shape.n_vertices)
        else:
            diff = vertices[point_a] - vertices[point_b]
            return np.linalg.norm(diff, axis=diff.ndim - 1)        

    def dist_matrix(self):
        """Distance between mesh vertices.
        
        Returns
        -------
        dist_matrix : array-like, shape=[n_vertices, n_vertices]
            Distance matrix.
        """
        return self.dist(np.arange(self._shape.n_vertices))[0]


class Dijkstra(BaseMetric):
    """Shortest path on edge graph of mesh with single source Dijkstra.

    Parameters
    ----------
    shape : Shape
        Shape.
    cutoff : float
        Length (sum of edge weights) at which the search is stopped.
    """

    def __init__(self, shape, cutoff=None):
        self.cutoff = cutoff

        super().__init__(shape)
        self._graph = to_nx_edge_graph(shape)

    def _dist_no_target_single(self, source_point):
        """Distance between mesh vertices.

        Parameters
        ----------
        source_point : array-like, shape=()
            Index of source point.

        Returns
        -------
        dist : array-like, shape=[n_targets]
            Distance.
        target_point : array-like, shape=[n_targets,]
            Target index.
        """
        dist_dict = nx.single_source_dijkstra_path_length(
            self._graph, source_point, cutoff=self.cutoff, weight="weight"
        )
        return np.array(list(dist_dict.values())), np.array(list(dist_dict.keys()))

    def _dist_no_target(self, source_point):
        """Distance between mesh vertices.

        Parameters
        ----------
        source_point : array-like, shape=[...]
            Index of source point.

        Returns
        -------
        dist : array-like, shape=[...,] or list[array-like]
            Distance.
        target_point : array-like, shape=[n_targets,] or list[array-like]
            Target index.
        """
        if _is_single_index(source_point):
            return self._dist_no_target_single(source_point)

        out = [
            self._dist_no_target_single(source_index_) for source_index_ in source_point
        ]
        return list(zip(*out))

    def _dist_target_single(self, point_a, point_b):
        """Distance between mesh vertices.

        Parameters
        ----------
        point_a : array-like, shape=()
            Index of source point.
        point_b : array-like, shape=()
            Index of target point.

        Returns
        -------
        dist : numeric
            Distance.
        """
        try:
            dist, _ = nx.single_source_dijkstra(
                self._graph,
                point_a,
                target=point_b,
                cutoff=None,
                weight="weight",
            )
        except nx.NetworkXNoPath:
            dist = np.inf
        return dist

    def _dist_target(self, point_a, point_b):
        """Distance between mesh vertices.

        Parameters
        ----------
        point_a : array-like, shape=[...]
            Index of source point.
        point_b : array-like, shape=[...]
            Index of target point.

        Returns
        -------
        dist : array-like, shape=[...,]
            Distance.
        """
        if _is_single_index(point_a) and _is_single_index(point_b):
            return self._dist_target_single(point_a, point_b)

        point_a, point_b = np.broadcast_arrays(point_a, point_b)
        return np.stack(
            [
                self._dist_target_single(point_a_, point_b_)
                for point_a_, point_b_ in zip(point_a, point_b)
            ]
        )

    def dist(self, point_a, point_b=None):
        """Distance between mesh vertices.

        Parameters
        ----------
        point_a : array-like, shape=[...]
            Index of source point.
        point_b : array-like, shape=[...]
            Index of target point.

        Returns
        -------
        dist : array-like, shape=[...,] or list[array-like]
            Distance.
        point_b : array-like, shape=[n_targets,] or list[array-like]
            Target index. If `point_b` is None.
        """
        if point_b is None:
            return self._dist_no_target(point_a)
        return self._dist_target(point_a, point_b)

    def dist_matrix(self):
        """Distance between mesh vertices.

        Returns
        -------
        dist_matrix : array-like, shape=[n_vertices, n_vertices]
            Distance matrix.
        """
        adj_matrix = nx.to_scipy_sparse_array(self._graph, weight='weight', format='csr')

        adj_matrix=adj_matrix.tolil()
        
        return shortest_path(adj_matrix, directed=False)



class FixedNeighborsDijkstra(Dijkstra):
    """Shortest path on edge graph of mesh with Dijkstra.

    Parameters
    ----------
    shape : Shape
        Shape.
    n_neighbors : int
        Number of neighbors to return when ``point_b is None``.
    neighbors_ratio : float
        Neighbors ratio to use to consider decreasing cuttoff.
    cutoff_decr_ratio : float
        Ratio to consider to proceed with cuttoff decrease.
    cutoff_incr_ratio : float
        Ratio use to update cutoff when not enough neighbors.
    """

    def __init__(
        self,
        shape,
        n_neighbors=5,
        neighbors_ratio=5,
        cutoff_decr_ratio=0.9,
        cutoff_incr_ratio=2.0,
    ):
        super().__init__(shape, cutoff=self._initial_cuttoff(shape, n_neighbors))

        self.n_neighbors = n_neighbors
        self.neighbors_ratio = neighbors_ratio
        self.cutoff_decr_ratio = cutoff_decr_ratio
        self.cutoff_incr_ratio = cutoff_incr_ratio

    @staticmethod
    def _initial_cuttoff(shape, n_neighbors):
        index_a, index_b = np.random.randint(0, high=shape.n_vertices, size=2)
        dist = EuclideanMetric(shape).dist(index_a, index_b)

        ratio = np.pow(n_neighbors / shape.n_vertices, 1 / 2.2)
        return ratio * dist

    def _dist_no_target_single(self, source_point):
        """Distance between mesh vertices.

        Parameters
        ----------
        source_point : array-like, shape=()
            Index of source point.

        Returns
        -------
        dist : array-like, shape=[n_neighbors]
            Distance.
        target_point : array-like, shape=[n_neighors,]
            Target index.
        """
        while True:
            dist, target = super()._dist_no_target_single(source_point)
            if target.size > self.n_neighbors:
                if target.size > round(self.n_neighbors * self.neighbors_ratio):
                    self.cutoff *= self.cutoff_decr_ratio

                return dist[: self.n_neighbors], target[: self.n_neighbors]

            self.cutoff *= self.cutoff_incr_ratio
