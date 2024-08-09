"""Spectral descriptors."""

import abc

import numpy as np
import pyFM.signatures

from ._base import Descriptor


class SpectralDescriptor(Descriptor, abc.ABC):
    """Spectral descriptor.

    Parameters
    ----------
    n_domain : int
        Number of domain points. Ignored if ``domain`` is not a callable.
    domain : callable or array-like, shape=[n_domain]
        Method to compute domain points (``f(basis, n_domain)``) or
        domain points.
    """

    # TODO: handle landmarks?
    # TODO: make general implementation

    def __init__(self, n_domain, domain):
        self.n_domain = n_domain
        self.domain = domain

    @abc.abstractmethod
    def __call__(self, basis, domain=None):
        """Compute descriptor.

        Parameters
        ----------
        basis : Eigenbasis.
            Basis.
        domain : array-like, shape=[n_domain]
            Domain points for computation.
        """


class PyfmHeatKernelSignature(SpectralDescriptor):
    """Heat kernel signature using pyFM.

    Parameters
    ----------
    scaled : bool
        Whether to scale for each time value.
    n_domain : int
        Number of time points. Ignored if ``domain`` is not a callable.
    domain : callable or array-like, shape=[n_domain]
        Method to compute time points (``f(basis, n_domain)``) or
        time points.
    """

    def __init__(self, scaled=True, n_domain=3, domain=None):
        super().__init__(n_domain, domain or self.default_domain)
        self.scaled = scaled

    def default_domain(self, basis, n_domain):
        """Compute default domain.

        Parameters
        ----------
        basis : Eigenbasis.
            Basis.
        n_domain : int
            Number of time points.

        Returns
        -------
        domain : array-like, shape=[n_domain]
            Time points.
        """
        abs_ev = np.sort(np.abs(basis.vals))
        index = 1 if np.isclose(abs_ev[0], 0.0) else 0
        return np.geomspace(
            4 * np.log(10) / abs_ev[-1], 4 * np.log(10) / abs_ev[index], n_domain
        )

    def __call__(self, basis, domain=None):
        """Compute descriptor.

        Parameters
        ----------
        basis : Eigenbasis.
            Basis.
        domain : array-like, shape=[n_domain]
            Time points.

        Returns
        -------
        descr : array-like, shape=[n_domain, n_vertices]
            Descriptor.
        """
        if domain is None:
            domain = (
                self.domain(basis, self.n_domain)
                if callable(self.domain)
                else self.domain
            )

        return pyFM.signatures.HKS(basis.vals, basis.vecs, domain, scaled=self.scaled).T


class HeatKernelSignature:
    """Heat kernel signature.

    Parameters
    ----------
    which : str
        One of: pyfm
    """

    # TODO: add registry
    _MAP = {"pyfm": PyfmHeatKernelSignature}

    def __new__(cls, which="pyfm", **kwargs):
        """Create new instance."""
        return cls._MAP[which](**kwargs)


class PyfmWaveKernelSignature(SpectralDescriptor):
    """Wave kernel signature using pyFM.

    Parameters
    ----------
    scaled : bool
        Whether to scale for each energy value.
    sigma : float
        Standard deviation.
    n_domain : int
        Number of energy points. Ignored if ``domain`` is not a callable.
    domain : callable or array-like, shape=[n_domain]
        Method to compute energy points (``f(basis, n_domain)``) or
        energy points.
    """

    def __init__(self, scaled=True, sigma=None, n_domain=3, domain=None):
        super().__init__(n_domain, domain or self.default_domain)

        self.scaled = scaled
        self.sigma = None

    def default_sigma(self, e_min, e_max, n_domain):
        """Compute default sigma.

        Parameters
        ----------
        e_min : float
            Minimum energy.
        e_max : float
            Maximum energy.
        n_domain : int
            Number of energy points.

        Returns
        -------
        sigma : float
            Standard deviation.
        """
        return 7 * (e_max - e_min) / n_domain

    def default_domain(self, basis, n_domain):
        """Compute default domain.

        Parameters
        ----------
        basis : Eigenbasis.
            Basis.
        n_domain : int
            Number of energy points to use.

        Returns
        -------
        domain : array-like, shape=[n_domain]
        """
        abs_ev = np.sort(np.abs(basis.vals))
        index = 1 if np.isclose(abs_ev[0], 0.0) else 0

        e_min, e_max = np.log(abs_ev[index]), np.log(abs_ev[-1])

        sigma = (
            self.default_sigma(e_min, e_max, n_domain)
            if self.sigma is None
            else self.sigma
        )

        e_min += 2 * sigma
        e_max -= 2 * sigma

        energy = np.linspace(e_min, e_max, n_domain)

        return energy, sigma

    def __call__(self, basis, domain=None):
        """Compute descriptor.

        Parameters
        ----------
        basis : Eigenbasis.
            Basis.
        domain : array-like, shape=[n_domain]
            Energy points for computation.

        Returns
        -------
        descr : array-like, shape=[n_domain, n_vertices]
            Descriptor.
        """
        sigma = None
        if domain is None:
            if callable(self.domain):
                energy, sigma = self.domain(basis, self.n_domain)
            else:
                energy = self.domain

        if sigma is None:
            # TODO: need to verify this
            sigma = (
                self.default_sigma(np.amin(domain), np.amax(domain), len(domain))
                if self.sigma is None
                else self.sigma
            )

        return pyFM.signatures.WKS(
            basis.vals, basis.vecs, energy, sigma, scaled=self.scaled
        ).T


class WaveKernelSignature:
    """Wave kernel signature.

    Parameters
    ----------
    which : str
        One of: pyfm
    """

    _MAP = {"pyfm": PyfmWaveKernelSignature}

    def __new__(cls, which="pyfm", **kwargs):
        """Create new instance."""
        return cls._MAP[which](**kwargs)
