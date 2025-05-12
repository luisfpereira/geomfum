"""Conversion between pointwise and functional maps."""

import abc
import scipy
from sklearn.neighbors import NearestNeighbors
from geomfum._registry import SinkhornNeighborFinderRegistry, WhichRegistryMixins
import numpy as np

class BaseP2pFromFmConverter(abc.ABC):
    """Pointwise map from functional map."""

class P2pFromFmConverter(BaseP2pFromFmConverter):
    """Pointwise map from functional map.

    Parameters
    ----------
    neighbor_finder : NeighborFinder
        Nearest neighbor finder.
    adjoint : bool
        Whether to use adjoint method.
    bijective : bool
        Whether to use bijective method. Check [VM2023]_.

    References
    ----------
    .. [OCSBG2012] Maks Ovsjanikov, Mirela Ben-Chen, Justin Solomon,
        Adrian Butscher, and Leonidas Guibas.
        “Functional Maps: A Flexible Representation of Maps between
        Shapes.” ACM Transactions on Graphics 31, no. 4 (2012): 30:1-30:11.
        https://doi.org/10.1145/2185520.2185526.
    .. [VM2023] Giulio Viganò  Simone Melzi. “Adjoint Bijective ZoomOut:
        Efficient Upsampling for Learned Linearly-Invariant Embedding.”
        The Eurographics Association, 2023. https://doi.org/10.2312/stag.20231293.
    """

    def __init__(self, neighbor_finder=None, adjoint=False, bijective=False):
        if neighbor_finder is None:
            neighbor_finder = NearestNeighbors(
                n_neighbors=1, leaf_size=40, algorithm="kd_tree", n_jobs=1
            )
        if neighbor_finder.n_neighbors > 1:
            raise ValueError("Expects `n_neighors = 1`.")

        self.neighbor_finder = neighbor_finder
        self.adjoint = adjoint
        self.bijective = bijective

    def __call__(self, fmap_matrix_12, basis_a, basis_b):
        """Convert functional map.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.

        Returns
        -------
        p2p : array-like, shape=[{n_vertices_b, n_vertices_a}]
            Pointwise map. ``bijective`` controls shape.
        """
        k2, k1 = fmap_matrix_12.shape

        if self.adjoint:
            emb1 = basis_a.full_vecs[:, :k1]
            emb2 = basis_b.full_vecs[:, :k2] @ fmap_matrix_12

        else:
            emb1 = basis_a.full_vecs[:, :k1] @ fmap_matrix_12.T
            emb2 = basis_b.full_vecs[:, :k2]

        if self.bijective:
            emb1, emb2 = emb2, emb1

        self.neighbor_finder.fit(emb1)
        _, p2p_21 = self.neighbor_finder.kneighbors(emb2)

    
        return p2p_21[:, 0]
    
class BijectiveP2pFromFmConverter(BaseP2pFromFmConverter):
 
    """ Bijective pointwise maps from functional maps.
    
    Parameters
    ----------
    neighbor_finder : NeighborFinder
        Nearest neighbor finder.
    adjoint : bool
        Whether to use adjoint method.

    References
    ----------
    .. [RMOW2023] Jing Ren, Simone Melzi, Maks Ovsjanikov, Peter Wonka.
        "MapTree: Recovering Multiple Solutions in the Space of Maps."
        ACM Transactions on Graphics 42, no. 4 (2023): 1-15.
    """

    def __init__(self, neighbor_finder=None, adjoint=False):
        if neighbor_finder is None:
            neighbor_finder = NearestNeighbors(
                n_neighbors=1, leaf_size=40, algorithm="kd_tree", n_jobs=1
            )
        if neighbor_finder.n_neighbors > 1:
            raise ValueError("Expects `n_neighors = 1`.")

        self.neighbor_finder = neighbor_finder
        self.adjoint = adjoint

    def __call__(self, fmap_matrix_ab, fmap_matrix_ba, basis_a, basis_b):
        """Convert functional map.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.

        Returns
        -------
        p2p : array-like, shape=[{n_vertices_b, n_vertices_a}]
            Pointwise map. ``bijective`` controls shape.
        """
        k2, k1 = fmap_matrix_ab.shape
        
        emb1=[]
        emb2=[]

        if self.adjoint:
            emb1.append( basis_a.full_vecs[:, :k1] )
            emb2.append( basis_b.full_vecs[:, :k2] @ fmap_matrix_ab )
            
            emb1.append( basis_a.full_vecs[:, :k1] @ fmap_matrix_ba )
            emb2.append( basis_b.full_vecs[:, :k2] )

        else:
            emb1.append( basis_a.full_vecs[:, :k1] @ fmap_matrix_ba.T )
            emb2.append( basis_b.full_vecs[:, :k2] )
            
            emb1.append( basis_a.full_vecs[:, :k1] )
            emb2.append( basis_b.full_vecs[:, :k2] @ fmap_matrix_ab.T )
        
        emb1= np.concatenate(emb1, axis=1)
        emb2= np.concatenate(emb2, axis=1)
        
        self.neighbor_finder.fit(emb1)
        _, p2p_21 = self.neighbor_finder.kneighbors(emb2)

        emb1, emb2 = emb2, emb1

        self.neighbor_finder.fit(emb1)
        _, p2p_12 = self.neighbor_finder.kneighbors(emb2)
        
        return p2p_21[:, 0], p2p_12[:, 0]


class DiscreteOptimizationP2pFromFmConverter(BaseP2pFromFmConverter):
    """Discrete optimization pointwise map from functional map.

    Parameters
    ----------
    neighbor_finder : NeighborFinder
        Nearest neighbor finder.
    energies : list of str
        Energies to use. Options are 'ortho', 'adjoint', 'conformal', and 'descriptors'.
        
    References
    ----------
    .. [RMWO2021] Jing Ren, Simone Melzi, Peter Wonka, Maks Ovsjanikov.
        “Discrete Optimization for Shape Matching.” Eurographics Symposium
        on Geometry Processing 2021, K. Crane and J. Digne (Guest Editors),
        Volume 40 (2021), Number 5. 
    """

    def __init__(self, neighbor_finder=None, energies=['ortho','adjoint','conformal','descriptors']):
        if neighbor_finder is None:
            neighbor_finder = NearestNeighbors(
                n_neighbors=1, leaf_size=40, algorithm="kd_tree", n_jobs=1
            )
        if neighbor_finder.n_neighbors > 1:
            raise ValueError("Expects `n_neighors = 1`.")

        self.neighbor_finder = neighbor_finder
        self.energies = energies
        
    def __call__(self, fmap_matrix_12, fmap_matrix_21, basis_a, basis_b, descr_a, descr_b):
        """Convert functional map.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.

        Returns
        -------
        p2p : array-like, shape=[{n_vertices_b, n_vertices_a}]
            Pointwise map. ``bijective`` controls shape.
        """
        k2, k1 = fmap_matrix_12.shape        

        emb1 = []
        emb2 = []

        if  'ortho' in self.energies:
            emb1.append(basis_a.full_vecs[:, :k1] @ fmap_matrix_12.T)
            emb2.append(basis_b.full_vecs[:, :k2])
        if 'adjoint' in self.energies:
            emb1.append(basis_a.full_vecs[:, :k1])
            emb2.append(basis_b.full_vecs[:, :k2] @ fmap_matrix_12)
        if 'conformal' in self.energies:
            emb1.append(basis_a.full_vecs[:, :k1] @ (basis_a.full_vals[:k1][:, None] * fmap_matrix_12.T))
            emb2.append(basis_b.full_vecs[:, :k2])
        if 'descriptors' in self.energies:
            emb1.append(basis_a.full_vecs[:, :k1] @ basis_a.project(descr_a).T)
            emb2.append(basis_b.full_vecs[:, :k2] @ basis_b.project(descr_b).T)
        if 'bijective' in self.energies:
            emb1.append(basis_a.full_vecs[:, :k1] @ fmap_matrix_21)
            emb2.append(basis_b.full_vecs[:, :k2])
        
        emb1 = np.concatenate(emb1, axis=1)
        emb2 = np.concatenate(emb2, axis=1)
        
    
        self.neighbor_finder.fit(emb1)
        _, p2p_21 = self.neighbor_finder.kneighbors(emb2)


        emb1, emb2 = emb2, emb1

        self.neighbor_finder.fit(emb1)
        _, p2p_12 = self.neighbor_finder.kneighbors(emb2)
        
        return p2p_21[:, 0], p2p_12[:, 0]

class SmoothP2pFromFmConverter(BaseP2pFromFmConverter):
    """Smooth pointwise map from functional map.

    Parameters
    ----------
    neighbor_finder : NeighborFinder
        Nearest neighbor finder.

    References
    ----------
    .. [MRSO2022] R. Magnet, J. Ren, O. Sorkine-Hornung, and M. Ovsjanikov.
        "Smooth NonRigid Shape Matching via Effective Dirichlet Energy Optimization."
        In 2022 International Conference on 3D Vision (3DV).
    """

    def __init__(self, neighbor_finder=None, adjoint=False):
        if neighbor_finder is None:
            neighbor_finder = NearestNeighbors(
                n_neighbors=1, leaf_size=40, algorithm="kd_tree", n_jobs=1
            )
        if neighbor_finder.n_neighbors > 1:
            raise ValueError("Expects `n_neighors = 1`.")

        self.neighbor_finder = neighbor_finder
        self.adjoint = adjoint
    def __call__(self, fmap_matrix12, fmap_matrix21, displ21, displ12 , mesh_a,mesh_b):
        """Convert functional map.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.

        Returns
        -------
        p2p : array-like, shape=[{n_vertices_b, n_vertices_a}]
            Pointwise map. ``bijective`` controls shape.
        """
        vert_a = mesh_a.vertices
        vert_b = mesh_b.vertices
        basis_a = mesh_a.basis
        basis_b = mesh_b.basis
        
        k2, k1 = fmap_matrix12.shape 
        emb1=[]
        emb2=[]

        if self.adjoint:
            emb1.append( basis_a.full_vecs[:, :k1] )
            emb2.append( basis_b.full_vecs[:, :k2] @ fmap_matrix12 )
            
            emb1.append( basis_a.full_vecs[:, :k1] @ fmap_matrix21 )
            emb2.append( basis_b.full_vecs[:, :k2] )

        else:
            emb1.append( basis_a.full_vecs[:, :k1] @ fmap_matrix12.T )
            emb2.append( basis_b.full_vecs[:, :k2] )
            
            emb1.append( basis_a.full_vecs[:, :k1] )
            emb2.append( basis_b.full_vecs[:, :k2] @ fmap_matrix12.T )
            
            
        emb1.append(vert_a)
        emb2.append(vert_b+displ21)
        
        emb1.append(vert_a+displ12)
        emb2.append(vert_b)
                    
        emb1 = np.concatenate(emb1, axis=1)
        emb2 = np.concatenate(emb2, axis=1)
        
        self.neighbor_finder.fit(emb1)
        _, p2p_21 = self.neighbor_finder.kneighbors(emb2)

        emb1, emb2 = emb2, emb1

        self.neighbor_finder.fit(emb1)
        _, p2p_12 = self.neighbor_finder.kneighbors(emb2)
        
        return p2p_21[:, 0], p2p_12[:, 0]


class BaseSinkhornNeighborFinder(abc.ABC):
    """Base class for a synkhorn Neighbor finder to find Neighbors based on the solution of OT maps computed with Sinkhorn regularization.
    
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
    @abc.abstractmethod
    def fit(self, X):
        """Store the reference points.
        
        Parameters
        ----------
        X : array-like, shape=[n_points_x, n_features]
            Reference points.
        """
        pass
    @abc.abstractmethod
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
        pass
    
    
class SinkhornNeighborFinder(WhichRegistryMixins):
    """Sinkhorn neighbor finder to find Neighbors based on the solution of OT maps computed with Sinkhorn regularization."""
    _Registry = SinkhornNeighborFinderRegistry
    def __init__(self, n_neighbors=1, lambd=1e-1, tau=1e-9, max_iter=100):
        self.n_neighbors = n_neighbors
        self.lambd = lambd
        self.max_iter = max_iter
        self.tau = tau
        self.X = None

    def fit(self, X):
        self.X = X
        return self

    def kneighbors(self, Y):
        M = np.linalg.norm(Y[:, None, :] - self.X[None, :, :], axis=2)
        n, m = M.shape

        a = np.ones(n) / n
        b = np.ones(m) / m

        Gs = self._sinkhorn_numpy(a, b, M, lambd=self.lambd, max_iter=self.max_iter, tau=self.tau)

        indices = np.argsort(Gs, axis=1)[:, :self.n_neighbors]
        distances = np.array([M[i, indices[i]] for i in range(Y.shape[0])])

        return distances, indices

    def _sinkhorn_numpy(self,a, b, M, lambd=1e-2, max_iter=100, tau=1e-9):
        K = np.exp(-M / lambd)
        u = np.ones_like(a)
        v = np.ones_like(b)

        for _ in range(max_iter):
            K_v = K @ v
            K_v[K_v < tau] = tau  # Avoid division by zero
            u = a / K_v

            K_T_u = K.T @ u
            K_T_u[K_T_u < tau] = tau
            v = b / K_T_u

        P = u[:, None] * K * v[None, :]
        return P
        
    
class SinkhornP2pFromFmConverter(P2pFromFmConverter):
    """Pointwise map from functional map using Sinkhorn filters.

    Parameters
    ----------
    adjoint : bool
        Whether to use adjoint method.
    bijective : bool
        Whether to use bijective method. Check [VM2023]_.
    epsilon : float
        Regularization parameter for Sinkhorn algorithm.
    max_iter : int
        Maximum number of iterations for Sinkhorn algorithm.
    epsilon0 : float
        Initial regularization parameter for epsilon scaling.
    tau : float
        Threshold for numerical stability in log-domain calculations.
        
    References
    ----------
    .. [PRMWO2021] Gautam Pai, Jing Ren, Simone Melzi, Peter Wonka, and Maks Ovsjanikov.
        "Fast Sinkhorn Filters: Using Matrix Scaling for Non-Rigid Shape Correspondence
        with Functional Maps." Proceedings of the IEEE/CVF Conference on Computer Vision
        and Pattern Recognition (CVPR), 2021, pp. 11956-11965.
        https://hal.science/hal-03184936/document
    """

    def __init__(self, sinkhorn_neigbor_finder=None, adjoint=False, bijective=False):
        
        neighbor_finder = sinkhorn_neigbor_finder
        super().__init__(neighbor_finder, adjoint, bijective)
        
        
class BaseFmFromP2pConverter(abc.ABC):
    """Functional map from pointwise map."""


class FmFromP2pConverter(BaseFmFromP2pConverter):
    """Functional map from pointwise map.

    Parameters
    ----------
    pseudo_inverse : bool
        Whether to solve using pseudo-inverse.
    """

    # TODO: add subsampling
    def __init__(self, pseudo_inverse=False):
        self.pseudo_inverse = pseudo_inverse

    def __call__(self, p2p, basis_a, basis_b):
        """Convert point to point map.

        Parameters
        ----------
        p2p : array-like, shape=[n_vertices_b]
            Poinwise map.

        Returns
        -------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        """
        evects1_pb = basis_a.vecs[p2p, :]

        if self.pseudo_inverse:
            return basis_b.vecs.T @ (basis_b._shape.laplacian.mass_matrix @ evects1_pb)

        return scipy.linalg.lstsq(basis_b.vecs, evects1_pb)[0]


class FmFromP2pBijectiveConverter(BaseFmFromP2pConverter):
    """Bijective functional map from pointwise map method.

    References
    ----------
    .. [VM2023] Giulio Viganò  Simone Melzi. “Adjoint Bijective ZoomOut:
        Efficient Upsampling for Learned Linearly-Invariant Embedding.”
        The Eurographics Association, 2023. https://doi.org/10.2312/stag.20231293.
    """

    def __call__(self, p2p, basis_a, basis_b):
        """Convert point to point map.

        Parameters
        ----------
        p2p : array-like, shape=[n_vertices_a]
            Pointwise map.

        Returns
        -------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        """
        evects2_pb = basis_b.vecs[p2p, :]
        return scipy.linalg.lstsq(evects2_pb, basis_a.vecs)[0]



class BaseDisplacementFromP2pConverter(abc.ABC):
    """Base class to obtain a displacement from a permutations map."""
    

class DisplacementFromP2pConverter(BaseDisplacementFromP2pConverter):
    """Displacement from pointwise map.

    Parameters
    ----------
    neighbor_finder : NeighborFinder
        Nearest neighbor finder.
    """
    def __init__(self):
        pass
        
    def __call__(self, p2p, mesh_a, mesh_b):
        """Convert pointwise map to displacement.    
    
        Parameters
        ----------
        p2p : array-like, shape=[n_vertices_a]
            Pointwise map.

        Returns
        -------
        disp : array-like, shape=[n_vertices_a, 3]
            Functional map matrix.
        """
        
        return  mesh_a.vertices[p2p]-mesh_b.vertices

class DirichletDisplacementFromP2pConverter(BaseDisplacementFromP2pConverter):
    """Displacement from pointwise map.

    Parameters
    ----------
    neighbor_finder : NeighborFinder
        Nearest neighbor finder.
        
    References
    ----------
    .. [MRSO2022] R. Magnet, J. Ren, O. Sorkine-Hornung, and M. Ovsjanikov.
        "Smooth NonRigid Shape Matching via Effective Dirichlet Energy Optimization."
        In 2022 International Conference on 3D Vision (3DV).
    """
    
    def __init__(self, weight=1e3):
        self.weight = weight
    def __call__(self, p2p, mesh_a, mesh_b, stiffness_matrix_b, mass_matrix_b):
        """Convert pointwise map to displacement.    
    
        Parameters
        ----------
        p2p : array-like, shape=[n_vertices_a]
            Pointwise map.
        mesh_a : Mesh
            Mesh A.
        mesh_b : Mesh
            Mesh B.
        stiffness_matrix_a : array-like, shape=[n_vertices_a, n_vertices_a]
            Stiffness matrix of mesh A.
        mass_matrix_a : array-like, shape=[n_vertices_a, n_vertices_a]
            Mass matrix of mesh A.

        Returns
        -------
        disp : array-like, shape=[n_vertices_a, 3]
            Functional map matrix.
        """
        
        target = scipy.sparse.linalg.spsolve(stiffness_matrix_b + self.weight *mass_matrix_b, mass_matrix_b @ mesh_a.vertices[p2p])
        
        return target - mesh_b.vertices
