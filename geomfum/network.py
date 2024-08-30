"""Tools to handle collections of shapes with functional map framework."""

import networkx as nx


class FunctionalMapNetwork:
    """Functional map network.

    Parameters
    ----------
    shapes : list[Shape]
        Vertices of the graph.
    """

    def __init__(self, shapes):
        # TODO: add iso as flag
        self.shapes = tuple(shapes)
        self.conn = {}

    def add_edge(self, i, j, fmap):
        """Add edge.

        Parameters
        ----------
        i : int
            Initial vertex of edge.
        j : int
            End vertex of edge.
        fmap : array-like
            Functional map between S_i and S_j.

        Returns
        -------
        self : FunctionalMapNetwork
        """
        self.conn[(i, j)] = fmap
        return self

    @property
    def edges(self):
        """Get all edges.

        Returns
        -------
        edges : list[tuple[int; 2]]
        """
        return list(self.conn.keys())

    def get_fmap(self, i, j):
        """Get functional map for a given edge.

        Parameters
        ----------
        i : int
            Initial vertex of edge.
        j : int
            End vertex of edge.

        Returns
        -------
        self : FunctionalMapNetwork
        """
        return self.conn[(i, j)]

    def to_networkx(self):
        """Get networkx representation.

        Keeps only network topology.

        Returns
        -------
        graph : nx.Digraph
            Topology of network.
        """
        return nx.DiGraph(self.edges)

    def simply_cycles(self):
        """Find simple cycles (elementary circuits) of the network.

        Returns
        -------
        cycles : list[list[int]]
            Cycles represented as a list of nodes.
        """
        return nx.simple_cycles(self.to_networkx())

    def edges_starting_at_index(self, i):
        """Get all the edges starting at shape i.

        Returns
        -------
        edges : list[tuple[int; 2]]
        """
        return filter(self.edges, lambda edge: edge[0] == i)

    def edges_ending_at_index(self, i):
        """Get all the edges ending at shape i.

        Returns
        -------
        edges : list[tuple[int; 2]]
        """
        return filter(self.edges, lambda edge: edge[1] == i)
