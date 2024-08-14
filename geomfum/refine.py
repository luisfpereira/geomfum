"""Functional map refinement machinery."""

import abc
import logging

import numpy as np
import scipy

from geomfum.convert import FmFromP2pConverter, P2pFromFmConverter


class Refiner(abc.ABC):
    """Functional map refiner."""

    @abc.abstractmethod
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


class IdentityRefiner(Refiner):
    """A dummy refiner."""

    def __call__(self, fmap_matrix, basis_a=None, basis_b=None):
        """Apply refiner.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        basis_a : Eigenbasis.
            Basis. Ignored.
        basis_b: Eigenbasis.
            Basis. Ignored.

        Returns
        -------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Refined functional map matrix.
        """
        return fmap_matrix


class SvdRefiner(Refiner):
    """Refinement using singular value decomposition.

    References
    ----------
    .. [OCSBG2012] Maks Ovsjanikov, Mirela Ben-Chen, Justin Solomon,
        Adrian Butscher, and Leonidas Guibas.
        “Functional Maps: A Flexible Representation of Maps between
        Shapes.” ACM Transactions on Graphics 31, no. 4 (2012): 30:1-30:11.
        https://doi.org/10.1145/2185520.2185526.
    """

    # TODO: find better name

    def __call__(self, fmap_matrix, basis_a=None, basis_b=None):
        """Apply refiner.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        basis_a : Eigenbasis.
            Basis. Ignored.
        basis_b: Eigenbasis.
            Basis. Ignored.

        Returns
        -------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Refined functional map matrix.
        """
        k2, k1 = fmap_matrix.shape
        U, _, VT = scipy.linalg.svd(fmap_matrix)
        return U @ np.eye(k2, k1) @ VT


class IterativeRefiner(Refiner):
    """Iterative refinement of functional map.

    At each iteration, it computes a pointwise map,
    converts it back to a functional map, and (optionally)
    furthers refines it.

    Parameters
    ----------
    nit : int
        Number of iterations.
    atol : float
        Convergence tolerance.
    p2p_from_fm_converter : P2pFromFmConverter
        Pointwise map from functional map.
    fm_from_p2p_converter : FmFromP2pConverter
        Functional map from pointwise map.
    iter_refiner : Refiner
        Refinement algorithm that runs within each iteration.
    """

    def __init__(
        self,
        nit=10,
        atol=1e-4,
        p2p_from_fm_converter=None,
        fm_from_p2p_converter=None,
        iter_refiner=None,
    ):
        super().__init__()
        if p2p_from_fm_converter is None:
            p2p_from_fm_converter = P2pFromFmConverter()

        if fm_from_p2p_converter is None:
            fm_from_p2p_converter = FmFromP2pConverter()

        if iter_refiner is None:
            iter_refiner = IdentityRefiner()

        self.nit = nit
        self.atol = atol
        self.p2p_from_fm_converter = p2p_from_fm_converter
        self.fm_from_p2p_converter = fm_from_p2p_converter
        self.iter_refiner = iter_refiner

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

        fmap_matrix = self.fm_from_p2p_converter(
            p2p_21, basis_a.truncate(k1), basis_b.truncate(k2)
        )
        return self.iter_refiner(fmap_matrix, basis_a, basis_b)

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


class IterativeSvdRefiner(IterativeRefiner):
    """Iterative refinement of functional map using SVD.

    Parameters
    ----------
    nit : int
        Number of iterations.
    atol : float
        Convergence tolerance.
    p2p_from_fm_converter : P2pFromFmConverter
        Pointwise map from functional map.
    fm_from_p2p_converter : FmFromP2pConverter
        Functional map from pointwise map.

    References
    ----------
    .. [OCSBG2012] Maks Ovsjanikov, Mirela Ben-Chen, Justin Solomon,
        Adrian Butscher, and Leonidas Guibas.
        “Functional Maps: A Flexible Representation of Maps between
        Shapes.” ACM Transactions on Graphics 31, no. 4 (2012): 30:1-30:11.
        https://doi.org/10.1145/2185520.2185526.
    """

    # TODO: find better name

    def __init__(
        self,
        nit=10,
        atol=1e-4,
        p2p_from_fm_converter=None,
        fm_from_p2p_converter=None,
    ):
        super().__init__(
            nit, atol, p2p_from_fm_converter, fm_from_p2p_converter, SvdRefiner()
        )
