"""Conversion between pointwise and functional maps."""

import abc

import scipy
from sklearn.neighbors import NearestNeighbors
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

    def __call__(self, fmap_matrix, basis_a, basis_b):
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
        k2, k1 = fmap_matrix.shape

        if self.adjoint:
            emb1 = basis_a.full_vecs[:, :k1]
            emb2 = basis_b.full_vecs[:, :k2] @ fmap_matrix

        else:
            emb1 = basis_a.full_vecs[:, :k1] @ fmap_matrix.T
            emb2 = basis_b.full_vecs[:, :k2]

        if self.bijective:
            emb1, emb2 = emb2, emb1

        self.neighbor_finder.fit(emb1)
        _, p2p_21 = self.neighbor_finder.kneighbors(emb2)

        return p2p_21[:, 0]

class DiscreteOptimizationP2pFromFmConverter(BaseP2pFromFmConverter):
    """Discrete optimization pointwise map from functional map.

    Parameters
    ----------
    neighbor_finder : NeighborFinder
        Nearest neighbor finder.
    bijective : bool
        Whether to use bijective method. Check [VM2023]_.

    References
    ----------
    .. [RMWO2021] Jing Ren, Simone Melzi, Peter Wonka, Maks Ovsjanikov.
        “Discrete Optimization for Shape Matching.” Eurographics Symposium
        on Geometry Processing 2021, K. Crane and J. Digne (Guest Editors),
        Volume 40 (2021), Number 5. 
    """

    def __init__(self, neighbor_finder=None, adjoint=False, bijective=False, energies=['ortho','adjoint','conformal','descriptors']):
        if neighbor_finder is None:
            neighbor_finder = NearestNeighbors(
                n_neighbors=1, leaf_size=40, algorithm="kd_tree", n_jobs=1
            )
        if neighbor_finder.n_neighbors > 1:
            raise ValueError("Expects `n_neighors = 1`.")

        self.neighbor_finder = neighbor_finder
        self.bijective = bijective
        self.energies = energies
        
    def __call__(self, fmap_matrix, basis_a, basis_b, descr_a, descr_b):
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
        k2, k1 = fmap_matrix.shape        
        #embedding concatenation
        emb_a = []
        emb_b = []

        if  'ortho' in self.energies:
            emb_a.append(basis_a.full_vecs[:, :k1] @ fmap_matrix.T)
            emb_b.append(basis_b.full_vecs[:, :k2])
        if 'adjoint' in self.energies:
            emb_a.append(basis_a.full_vecs[:, :k1])
            emb_b.append(basis_b.full_vecs[:, :k2] @ fmap_matrix)
        if 'conformal' in self.energies:
            emb_a.append(basis_a.full_vecs[:, :k1] @ (basis_a.full_vals[:k1][:, None] * fmap_matrix.T))
            emb_b.append(basis_b.full_vecs[:, :k2])
        if 'descriptors' in self.energies:
            emb_a.append(basis_a.full_vecs[:, :k1] @ basis_a.project(descr_a).T)
            emb_b.append(basis_b.full_vecs[:, :k2] @ basis_b.project(descr_b).T)

        #TODO: add bijective zoomout
        
        emb1 = np.concatenate(emb_a, axis=1)
        emb2 = np.concatenate(emb_b, axis=1)
        
        
        if self.bijective:
            emb1, emb2 = emb2, emb1

        self.neighbor_finder.fit(emb1)
        _, p2p_21 = self.neighbor_finder.kneighbors(emb2)

        return p2p_21[:, 0]

class SmoothP2pFromFmConverter(BaseP2pFromFmConverter):
    """Smooth pointwise map from functional map.

    Parameters
    ----------
    neighbor_finder : NeighborFinder
        Nearest neighbor finder.
    bijective : bool
        Whether to use bijective method. Check [VM2023]_.

    References
    ----------
    .. [MRSO2022] R. Magnet, J. Ren, O. Sorkine-Hornung, and M. Ovsjanikov.
        "Smooth NonRigid Shape Matching via Effective Dirichlet Energy Optimization."
        In 2022 International Conference on 3D Vision (3DV).
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
        
    def __call__(self, fmap_matrix,displ, mesh_a,mesh_b):
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
        
        k2, k1 = fmap_matrix.shape 
        #embedding concatenation
        emb_a = []
        emb_b = []

        if  self.adjoint:
            emb_a.append(basis_a.full_vecs[:, :k1])
            emb_b.append(basis_b.full_vecs[:, :k2] @ fmap_matrix)
        else:
            emb_a.append(basis_a.full_vecs[:, :k1] @ fmap_matrix.T)
            emb_b.append(basis_b.full_vecs[:, :k2])

        emb_a.append(vert_a)
        emb_b.append(vert_b+displ)
        #TODO: add bijective zoomout
        
        emb1 = np.concatenate(emb_a, axis=1)
        emb2 = np.concatenate(emb_b, axis=1)
        
        
        if self.bijective:
            emb1, emb2 = emb2, emb1

        self.neighbor_finder.fit(emb1)
        _, p2p_21 = self.neighbor_finder.kneighbors(emb2)

        return p2p_21[:, 0]

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
    
    def __init__(self, weight=1.0):
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

