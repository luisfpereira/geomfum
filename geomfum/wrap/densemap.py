"""Wrapper for DenseMap library"""

from densemaps.torch import maps

from geomfum.convert import BaseNeighborFinder


class DenseMapNeighborFinder(BaseNeighborFinder):
    """DenseMap neighbor finder.

    Parameters
    ----------
    n_neighbors : int, default=15
        Number of neighbors to find.
    """

    def __init__(self):
        super().__init__(n_neighbors=1)
        self.X_ = None

    def fit(self, X):
        """Store the reference points."""
        self.X_ = X

    def kneighbors(self, Y, return_distance=True):
        """Find k-nearest neighbors."""
        P21 = maps.KernelDistMap(self.X_, Y, blur=1e-1)

        p2p_21 = P21.get_nn()  # I can get the (N2,) vertex to vertex map

        if return_distance:
            return p2p_21, (self.X_ - Y[p2p_21]).norm(dim=-1)
        return p2p_21[:, None]
