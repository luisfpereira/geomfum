"""Functional map refinement machinery."""

import abc
import logging

import numpy as np
import scipy

from geomfum.convert import FmFromP2pConverter, P2pFromFmConverter


class Refiner(abc.ABC):
    """Functional map refiner."""


class IcpRefiner(Refiner):
    """Standard ICP algorithm.

    Parameters
    ----------
    nit : int
        Number of iterations.
    atol : float
        Convergence tolerance.
    p2p_from_fm_converter : P2pFromFmConverter
        Pointwise map from functional map.
    fm_from_p2p_converter : FmFromP2pConverter
        Functional map from pointwise map
    """

    def __init__(
        self,
        nit=10,
        atol=1e-4,
        p2p_from_fm_converter=None,
        fm_from_p2p_converter=None,
    ):
        if p2p_from_fm_converter is None:
            p2p_from_fm_converter = P2pFromFmConverter()

        if fm_from_p2p_converter is None:
            fm_from_p2p_converter = FmFromP2pConverter()

        self.nit = nit
        self.atol = atol
        self.p2p_from_fm_converter = p2p_from_fm_converter
        self.fm_from_p2p_converter = fm_from_p2p_converter

    def iter(self, fmap_matrix, basis_a, basis_b):
        """Refiner iteration.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        basis_a : Eigenbasis.
            Basis.
        basis_b: Eigenbasis.
            Basis.

        Returns
        -------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Refined functional map matrix.
        """
        k2, k1 = fmap_matrix.shape
        p2p_21 = self.p2p_from_fm_converter(fmap_matrix, basis_a, basis_b)
        fmap_matrix_icp = self.fm_from_p2p_converter(
            p2p_21, basis_a.truncate(k1), basis_b.truncate(k2)
        )
        U, _, VT = scipy.linalg.svd(fmap_matrix_icp)
        return U @ np.eye(k2, k1) @ VT

    def __call__(self, fmap_matrix, basis_a, basis_b):
        """Apply refiner.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        basis_a : Eigenbasis.
            Basis.
        basis_b: Eigenbasis.
            Basis.

        Returns
        -------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Refined functional map matrix.
        """
        # TODO: make it general?

        for _ in range(self.nit):
            new_fmap_matrix = self.iter(fmap_matrix, basis_a, basis_b)

            if (
                self.atol is not None
                and np.amax(np.abs(new_fmap_matrix - fmap_matrix)) < self.atol
            ):
                break

            fmap_matrix = new_fmap_matrix

        else:
            logging.warning(f"Maximum number of iterations reached: {self.nit}")

        return new_fmap_matrix
