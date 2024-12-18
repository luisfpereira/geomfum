import networkx as nx
import numpy as np


class EuclideanMetric:
    def __init__(self, shape):
        self._shape = shape

    def dist(self, source_index, target_index):
        """Distance between mesh vertices.

        Parameters
        ----------
        source_index : array-like, shape=[...]
            Index of source point.
        target_index : array-like, shape=[...]
            Index of target point.

        Returns
        -------
        dist : array-like, shape=[...,]
            Distance.
        """
        # TODO: do against all if target index is None?
        # TODO: add euclidean with cutoff
        vertices = self._shape.vertices
        diff = vertices[source_index] - vertices[target_index]
        return np.linalg.norm(diff, axis=diff.ndim - 1)


class SingleSourceDijkstra:
    def __init__(self, shape, cutoff=None):
        self.cutoff = cutoff

        self._shape = shape
        self._euc_metric = EuclideanMetric(shape)
        self._graph = self._to_nx_edge_graph()

    def _to_nx_edge_graph(self):
        edges = self._shape.edges
        vertex_a, vertex_b = edges.T
        lengths = self._euc_metric.dist(vertex_a, vertex_b)

        weighted_edges = list(zip(vertex_a, vertex_b, lengths))

        graph = nx.Graph()
        graph.add_weighted_edges_from(weighted_edges)

        return graph

    def _dist_no_target_single(self, source_index):
        """Distance between mesh vertices.

        Parameters
        ----------
        source_index : array-like, shape=()
            Index of source point.

        Returns
        -------
        dist : array-like, shape=[n_targets]
            Distance.
        target_index : array-like, shape=[n_targets,]
            Target index.
        """
        dist_dict = nx.single_source_dijkstra_path_length(
            self._graph, source_index, cutoff=self.cutoff, weight="weight"
        )
        return np.array(list(dist_dict.values())), np.array(list(dist_dict.keys()))

    def _dist_no_target(self, source_index):
        """Distance between mesh vertices.

        Parameters
        ----------
        source_index : array-like, shape=[...]
            Index of source point.

        Returns
        -------
        dist : array-like, shape=[...,] or list[array-like]
            Distance.
        target_index : array-like, shape=[n_targets,] or list[array-like]
            Target index.
        """
        if source_index.ndim == 0:
            return self._dist_no_target_single(source_index)

        out = [
            self._dist_no_target_single(source_index_) for source_index_ in source_index
        ]
        return list(zip(*out))

    def _dist_target_single(self, source_index, target_index):
        """Distance between mesh vertices.

        Parameters
        ----------
        source_index : array-like, shape=()
            Index of source point.
        target_index : array-like, shape=()
            Index of target point.

        Returns
        -------
        dist : numeric
            Distance.
        """
        try:
            dist, _ = nx.single_source_dijkstra(
                self._graph,
                source_index,
                target=target_index,
                cutoff=self.cutoff,
                weight="weight",
            )
        except nx.NetworkXNoPath:
            dist = np.inf
        return dist

    def _dist_target(self, source_index, target_index):
        """Distance between mesh vertices.

        Parameters
        ----------
        source_index : array-like, shape=[...]
            Index of source point.
        target_index : array-like, shape=[...]
            Index of target point.

        Returns
        -------
        dist : array-like, shape=[...,]
            Distance.
        """
        if source_index.ndim == 0 and target_index.ndim == 0:
            return self._dist_target_single(source_index, target_index)

        source_index, target_index = np.broadcast_arrays(source_index, target_index)
        return np.stack(
            [
                self._dist_target_single(source_index_, target_index_)
                for source_index_, target_index_ in zip(source_index, target_index)
            ]
        )

    def dist(self, source_index, target_index=None):
        """Distance between mesh vertices.

        Parameters
        ----------
        source_index : array-like, shape=[...]
            Index of source point.
        target_index : array-like, shape=[...]
            Index of target point.

        Returns
        -------
        dist : array-like, shape=[...,] or list[array-like]
            Distance.
        target_index : array-like, shape=[n_targets,] or list[array-like]
            Target index. If `target_index` is None.
        """
        if target_index is None:
            return self._dist_no_target(source_index)

        return self._dist_target(source_index, target_index)
