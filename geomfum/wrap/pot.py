"""Python Optimal Trasport wrapper."""

import numpy as np
import ot

from geomfum.convert import BaseNeighborFinder


class PotSinkhornNeighborFinder(BaseNeighborFinder):
    """Neighbor finder based on Optimal Transport maps computed with Sinkhorn regularization.

    Parameters
    ----------
    n_neighbors : int, default=1
        Number of neighbors to find.
    lambd : float, default=1e-1
        Regularization parameter for Sinkhorn algorithm.
    method : str, default="sinkhorn"
        Method to use for Sinkhorn algorithm.
    max_iter : int, default=100
        Maximum number of iterations for Sinkhorn algorithm.

    References
    ----------
    .. [Cuturi2013] Marco Cuturi. "Sinkhorn Distances: Lightspeed Computation of Optimal Transport."
        Advances in Neural Information Processing Systems (NIPS), 2013.
        http://marcocuturi.net/SI.html
    """

    def __init__(self, n_neighbors=1, lambd=1e-1, method="sinkhorn", max_iter=100):
        self.n_neighbors = n_neighbors
        self.lambd = lambd
        self.max_iter = max_iter
        self.method = method
        self.X = None

    def fit(self, X):
        """Store the reference points.

        Parameters
        ----------
        X : array-like, shape=[n_points_x, n_features]
            Reference points.
        """
        self.X = X
        return self

    def kneighbors(self, Y):
        """Find k nearest neighbors using Sinkhorn regularization.

        Parameters
        ----------
        Y : array-like, shape=[n_points_y, n_features]
            Query points.

        Returns
        -------
        distances : array-like, shape=[n_points_y, n_neighbors]
            Distances to the nearest neighbors.
        indices : array-like, shape=[n_points_y, n_neighbors]
            Indices of the nearest neighbors.
        """
        M = np.exp(-self.lambd * ot.dist(Y, self.X))

        n, m = M.shape
        a = np.ones(n) / n
        b = np.ones(m) / m

        Gs = ot.sinkhorn(a, b, M, self.lambd, self.method, self.max_iter)

        indices = np.argsort(Gs, axis=1)[:, : self.n_neighbors]

        distances = np.array([M[i, indices[i]] for i in range(Y.shape[0])])

        return distances, indices
