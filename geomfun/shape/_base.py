"""Base shape."""

import abc
import logging

import numpy as np

from geomfun.basis import LaplaceEigenBasis
from geomfun.laplacian import LaplacianFinder, LaplacianSpectrumFinder


class Shape(abc.ABC):
    def __init__(self, is_mesh):
        self.is_mesh = is_mesh

        self.basis = None

        self.landmark_indices = None

        self._laplace_matrix = None
        self._mass_matrix = None

    def equip_with_operator(self, name, Operator, allow_overwrite=True, **kwargs):
        """Equip with operator."""
        name_exists = hasattr(self, name)
        if name_exists:
            if allow_overwrite:
                logging.warning(f"Overriding {name}.")
            else:
                raise ValueError(f"{name} already exists")

        operator = Operator(self, **kwargs)
        setattr(self, name, operator)

        return self

    def set_landmarks(self, landmark_indices, append=False):
        """Set landmarks.

        Parameters
        ----------
        landmark_indices : array-like, shape=[n_landmarks]
            Landmarks.
        append : bool
            Whether to append landmarks to already-existing ones.
        """
        if append:
            self.landmark_indices = np.stack(self.landmark_indices, landmark_indices)

        else:
            self.landmark_indices = landmark_indices

        return self

    @property
    def laplace_matrix(self):
        """Laplace matrix.

        Returns
        -------
        laplace_matrix : array-like, shape=[n_vertices, n_vertices]
            Laplace matrix.
        """
        if self._laplace_matrix is None:
            self.find_laplacian()

        return self._laplace_matrix

    @property
    def mass_matrix(self):
        """Mass matrix.

        Returns
        -------
        mass_matrix : array-like, shape=[n_vertices, n_vertices]
            Diagonal lumped mass matrix.
        """
        if self._mass_matrix is None:
            self.find_laplacian()

        return self._mass_matrix

    def find_laplacian(self, laplacian_finder=None, recompute=False):
        """Find Laplacian.

        Parameters
        ----------
        laplacian_finder : BaseLaplacianFinder
            Algorithm to find the Laplacian.
        recompute : bool
            Whether to recompute Laplacian if information is cached.

        Returns
        -------
        laplace_matrix : array-like, shape=[n_vertices, n_vertices]
            Laplace matrix.
        mass_matrix : array-like, shape=[n_vertices, n_vertices]
            Diagonal lumped mass matrix.
        """
        if (
            not recompute
            and self._laplace_matrix is not None
            and self._mass_matrix is None
        ):
            return self._laplace_matrix, self._mass_matrix

        if laplacian_finder is None:
            laplacian_finder = LaplacianFinder(mesh=self.is_mesh, which="robust")

        self._laplace_matrix, self._mass_matrix = laplacian_finder(self)

        return self._laplace_matrix, self._mass_matrix

    def find_laplacian_spectrum(
        self,
        spectrum_size=100,
        laplacian_spectrum_finder=None,
        set_as_basis=True,
        recompute=False,
    ):
        """Find Laplacian spectrum.

        Parameters
        ----------
        spectrum_size : int
            Spectrum size. Ignored if ``laplacian_spectrum_finder`` is not None.
        laplacian_spectrum_finder : LaplacianSpectrumFinder
            Algorithm to find Laplacian spectrum.
        set_as_basis : bool
            Whether to set spectrum as basis.
        recompute : bool
            Whether to recompute spectrum if information is cached in basis.

        Returns
        -------
        eigenvals : array-like, shape=[spectrum_size]
            Eigenvalues.
        eigenvecs : array-like, shape=[n_vertices, spectrum_size]
            Eigenvectors.
        """
        if (
            not recompute
            and self.basis is not None
            and isinstance(self.basis, LaplaceEigenBasis)
        ):
            if set_as_basis:
                return self.basis

            return self.basis.full_vals, self.basis.full_vecs

        if laplacian_spectrum_finder is None:
            laplacian_spectrum_finder = LaplacianSpectrumFinder(
                spectrum_size=spectrum_size,
                nonzero=False,
                fix_sign=False,
            )

        if set_as_basis:
            self.basis = laplacian_spectrum_finder(self, as_basis=True)
            return self.basis.full_vals, self.basis.full_vecs

        return laplacian_spectrum_finder(self, as_basis=False)
