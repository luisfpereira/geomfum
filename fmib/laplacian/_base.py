import abc

from fmib.basis import LaplaceEigenBasis
from fmib.numerics.eig import ScipyEigsh

from ._registry import LaplacianFinderRegistry


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
        laplace_matrix : array-like, shape=[n_vertices, n_vertices]
            Laplacian matrix.
        mass_matrix : array-like, shape=[n_vertices, n_vertices]
            Diagonal lumped mass matrix.
        """


class LaplacianFinder:
    """Algorithm to find the Laplacian.

    Parameters
    ----------
    mesh : bool
        If mesh or point cloud.
    which : str
        Which algorithm/library to use.
    """

    def __new__(cls, mesh=True, which="robust", **kwargs):
        return LaplacianFinderRegistry.MAP[(mesh, which)](**kwargs)


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
        Algorithm to find the Laplacian.
    eig_solver : EigSolver
        Eigen solver.
    """

    # TODO: complete docstrings

    def __init__(
        self,
        spectrum_size=100,
        nonzero=False,
        fix_sign=False,
        laplacian_finder=None,
        eig_solver=None,
    ):
        if laplacian_finder is None:
            laplacian_finder = LaplacianFinder()

        if eig_solver is None:
            eig_solver = ScipyEigsh(spectrum_size=spectrum_size, sigma=-0.01)

        self.nonzero = nonzero
        self.fix_sign = fix_sign
        self.laplace_operator = laplacian_finder
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

    def __call__(self, shape, as_basis=True):
        """Apply algorithm.

        Parameters
        ----------
        shape : Shape
            Shape.
        as_basis : bool
            Whether return basis or eigenvals/vecs.

        Returns
        -------
        eigenvals : array-like, shape=[spectrum_size]
            Eigenvalues. (If ``basis is False``.)
        eigenvecs : array-like, shape=[n_vertices, spectrum_size]
            Eigenvectors. (If ``basis is False``.)
        basis : LaplaceEigenBasis
            A basis. (If ``basis is True``.)
        """
        # TODO: fix here, do it in shape
        laplace_matrix, mass_matrix = self.laplace_operator(shape)

        eigenvals, eigenvecs = self.eig_solver(laplace_matrix, M=mass_matrix)

        if self.nonzero:
            eigenvals = eigenvals[1:]
            eigenvecs = eigenvecs[:, 1:]

        if self.fix_sign:
            indices = eigenvecs[0, :] < 0
            eigenvals[indices] *= -1
            eigenvecs[:, indices] *= -1

        if as_basis:
            return LaplaceEigenBasis(eigenvals, eigenvecs, laplace_matrix, mass_matrix)

        return eigenvals, eigenvecs
