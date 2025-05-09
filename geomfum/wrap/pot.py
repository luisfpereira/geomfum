""" python optimal transport wrapper """


import abc
import ot
from geomfum.convert import SinkhornNeighborFinder
from numpy import np


class POTSinkhornNeighborFinder(SinkhornNeighborFinder):
    """This function implements a nieghbour finder based on the solution of OT maps computed with Sinkhorn regularization.
    
    References
    ----------
    .. [Cuturi2013] Marco Cuturi. “Sinkhorn Distances: Lightspeed Computation of Optimal Transport.”
    Advances in Neural Information Processing Systems (NIPS), 2013.
    http://marcocuturi.net/SI.html

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors.
    epsilon : float
        Regularization parameter for Sinkhorn algorithm.
    max_iter : int
        Maximum number of iterations for Sinkhorn algorithm.
    """

    def __init__(self, n_neighbors=1, lambd=1e-1):
        self.n_neighbors = n_neighbors
        self.lambd = lambd
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
      
        M = ot.dist(Y, self.X)
        n, m = M.shape
        
        a = np.ones(n) / n
        b = np.ones(m) / m

        Gs = ot.sinkhorn(a, b, M, self.lambd)   
             
        indices = np.argsort(Gs, axis=1)[:, :self.n_neighbors]
        
        distances = np.array([M[i, indices[i]] for i in range(Y.shape[0])])
        
        return distances, indices