"""Laplacian-related algorithms."""

import abc

import geomfum.wrap as _wrap  # noqa (for register)
from geomfum._registry import LaplacianFinderRegistry, MeshWhichRegistryMixins
from geomfum.basis import LaplaceEigenBasis
from geomfum.numerics.eig import ScipyEigsh
import numpy as np
import scipy.sparse as sparse


class BaseLaplacianFinder(abc.ABC):
    """Algorithm to find the Laplacian."""

    @abc.abstractmethod
    def __call__(self, shape):
        """Apply algorithm.

        Parameters
        ----------
        shape : Shape
            Shape.

        Returns
        -------
        stiffness_matrix : array-like, shape=[n_vertices, n_vertices]
            Stiffness matrix.
        mass_matrix : array-like, shape=[n_vertices, n_vertices]
            Mass matrix.
        """


class LaplacianFinder(MeshWhichRegistryMixins, BaseLaplacianFinder):
    """Algorithm to find the Laplacian."""

    _Registry = LaplacianFinderRegistry

    def __call__(self, shape):
        """Apply algorithm.

        Parameters
        ----------
        shape : TriangleMesh
            Mesh.

        Returns
        -------
        stiffness_matrix : scipy.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Stiffness matrix.
        mass_matrix : scipy.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Diagonal lumped mass matrix.
        """
        
        v1 = shape.vertices[shape.faces[:, 0]]  
        v2 = shape.vertices[shape.faces[:, 1]]
        v3 = shape.vertices[shape.faces[:, 2]]

        u1 = v3 - v2
        u2 = v1 - v3
        u3 = v2 - v1

        L1 = np.linalg.norm(u1, axis=1)
        L2 = np.linalg.norm(u2, axis=1)
        L3 = np.linalg.norm(u3, axis=1)


        A1 = np.einsum("ij,ij->i", -u2, u3) / (L2 * L3)
        A2 = np.einsum("ij,ij->i", u1, -u3) / (L1 * L3) 
        A3 = np.einsum("ij,ij->i", -u1, u2) / (L1 * L2)

        I = np.concatenate([shape.faces[:, 0], shape.faces[:, 1], shape.faces[:, 2]])
        J = np.concatenate([shape.faces[:, 1], shape.faces[:, 2], shape.faces[:, 0]])
        S = np.concatenate([A3, A1, A2])
        epsilon = 1e-8
        S = 0.5 * S / np.sqrt(1 - S**2 + epsilon)

        In = np.concatenate([I, J, I, J])
        Jn = np.concatenate([J, I, I, J])
        Sn = np.concatenate([-S, -S, S, S])

        W = sparse.coo_matrix((Sn, (In, Jn)), shape=(shape.n_vertices, shape.n_vertices)).tocsc()

        return (
            W, sparse.dia_matrix((shape.vertex_areas, 0), shape=(shape.n_vertices, shape.n_vertices))
        )


class LaplacianSpectrumFinder:
    """Algorithm to find Laplacian spectrum.

    Parameters
    ----------
    spectrum_size : int
        Spectrum size. Ignored if ``eig_solver`` is not None.
    nonzero : bool
        Remove zero zero eigenvalue.
    fix_sign : bool
        Wheather to have all the first components with positive sign.
    laplacian_finder : BaseLaplacianFinder
        Algorithm to find the Laplacian. Ignored if Laplace and mass matrices
        were already computed.
    eig_solver : EigSolver
        Eigen solver.
    """

    def __init__(
        self,
        spectrum_size=100,
        nonzero=False,
        fix_sign=False,
        laplacian_finder=None,
        eig_solver=None,
    ):
        if eig_solver is None:
            eig_solver = ScipyEigsh(spectrum_size=spectrum_size, sigma=-0.01)

        self.nonzero = nonzero
        self.fix_sign = fix_sign
        self.laplacian_finder = laplacian_finder
        self.eig_solver = eig_solver

    @property
    def spectrum_size(self):
        """Spectrum size.

        Returns
        -------
        spectrum_size : int
            Spectrum size.
        """
        return self.eig_solver.spectrum_size

    @spectrum_size.setter
    def spectrum_size(self, spectrum_size):
        """Set spectrum size.

        Parameters
        ----------
        spectrum_size : int
            Spectrum size.
        """
        self.eig_solver.spectrum_size = spectrum_size

    def __call__(self, shape, as_basis=True, recompute=False):
        """Apply algorithm.

        Parameters
        ----------
        shape : Shape
            Shape.
        as_basis : bool
            Whether return basis or eigenvals/vecs.
        recompute : bool
            Whether to recompute Laplacian if information is cached.

        Returns
        -------
        eigenvals : array-like, shape=[spectrum_size]
            Eigenvalues. (If ``basis is False``.)
        eigenvecs : array-like, shape=[n_vertices, spectrum_size]
            Eigenvectors. (If ``basis is False``.)
        basis : LaplaceEigenBasis
            A basis. (If ``basis is True``.)
        """
        stiffness_matrix, mass_matrix = shape.laplacian.find(
            self.laplacian_finder, recompute=recompute
        )

        eigenvals, eigenvecs = self.eig_solver(stiffness_matrix, M=mass_matrix)

        if self.nonzero:
            eigenvals = eigenvals[1:]
            eigenvecs = eigenvecs[:, 1:]

        if self.fix_sign:
            indices = eigenvecs[0, :] < 0
            eigenvals[indices] *= -1
            eigenvecs[:, indices] *= -1

        if as_basis:
            return LaplaceEigenBasis(shape, eigenvals, eigenvecs)

        return eigenvals, eigenvecs
