from fmib.basis import LaplaceEigenBasis
from fmib.numerics.eig import ScipyEigsh
from fmib.operator import ShapeLaplacian


class LaplacianSpectrumFinder:
    """

    Parameters
    ----------
    spectrum_size : int
        Ignored if ``eig_solver`` is not None.
    nonzero : bool
        Remove zero zero eigenvalue.
    fix_sign : bool
        Wheather to have all the first components with positive sign.
    cache : bool
        If to cache intermediate matrices.
    """

    def __init__(
        self,
        spectrum_size=100,
        nonzero=False,
        fix_sign=False,
        laplace_operator=None,
        eig_solver=None,
    ):
        if laplace_operator is None:
            laplace_operator = ShapeLaplacian()

        if eig_solver is None:
            eig_solver = ScipyEigsh(spectrum_size=spectrum_size, sigma=-0.01)

        self.nonzero = nonzero
        self.fix_sign = fix_sign
        self.laplace_operator = laplace_operator
        self.eig_solver = eig_solver

    def __call__(self, shape, as_basis=True):
        """
        Returns
        -------
        eigenvals
        eigenvecs
        """
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
