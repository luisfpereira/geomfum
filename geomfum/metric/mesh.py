import abc

import networkx as nx
import numpy as np


def to_nx_edge_graph(shape):
    # TODO: move to utils? circular imports
    edges = shape.edges
    vertex_a, vertex_b = edges.T
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
            Point.
        point_b : array-like, shape=[...]
            Other point.

        Returns
        -------
        dist : array-like, shape=[...,]
            Distance.
        """


class EuclideanMetric(BaseMetric):
    def dist(self, point_a, point_b):
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
        # TODO: do against all if target index is None?
        # TODO: add euclidean with cutoff
        vertices = self._shape.vertices
        diff = vertices[point_a] - vertices[point_b]
        return np.linalg.norm(diff, axis=diff.ndim - 1)


class SingleSourceDijkstra(BaseMetric):
    # TODO: fixed n_neighbors distance
    # TODO: use two random points to check initial cutoff
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
        # TODO: reconsider behavior with inf

        try:
            dist, _ = nx.single_source_dijkstra(
                self._graph,
                point_a,
                target=point_b,
                cutoff=self.cutoff,
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
