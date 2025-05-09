"""Spectral descriptors."""

import numpy as np

import geomfum.linalg as la
from geomfum._registry import (
    HeatKernelSignatureRegistry,
    WaveKernelSignatureRegistry,
    WhichRegistryMixins,
)

from ._base import SpectralDescriptor


def hks_default_domain(shape, n_domain):
    """Compute HKS default domain.

    Parameters
    ----------
    shape : Shape.
        Shape with basis.
    n_domain : int
        Number of time points.

    Returns
    -------
    domain : array-like, shape=[n_domain]
        Time points.
    """
    nonzero_vals = shape.basis.nonzero_vals
    return np.geomspace(
        4 * np.log(10) / nonzero_vals[-1], 4 * np.log(10) / nonzero_vals[0], n_domain
    )


class WksDefaultDomain:
    """Compute WKS domain.

    Parameters
    ----------
    shape : Shape.
        Shape with basis.
    n_domain : int
        Number of energy points to use.
    n_overlap : int
        Controls Gaussian overlap. Ignored if ``sigma`` is not None.
    n_trans : int
        Number of standard deviations to translate energy bound by.
    """

    def __init__(self, n_domain, sigma=None, n_overlap=7, n_trans=2):
        self.n_domain = n_domain
        self.sigma = sigma
        self.n_overlap = n_overlap
        self.n_trans = n_trans

    def __call__(self, shape):
        """Compute WKS domain.

        Parameters
        ----------
        shape : Shape.
            Shape with basis.

        Returns
        -------
        domain : array-like, shape=[n_domain]
        sigma : float
            Standard deviation.
        """
        nonzero_vals = shape.basis.nonzero_vals

        e_min, e_max = np.log(nonzero_vals[0]), np.log(nonzero_vals[-1])

        sigma = (
            self.n_overlap * (e_max - e_min) / self.n_domain
            if self.sigma is None
            else self.sigma
        )

        e_min += self.n_trans * sigma
        e_max -= self.n_trans * sigma

        energy = np.linspace(e_min, e_max, self.n_domain)

        return energy, sigma


class HeatKernelSignature(WhichRegistryMixins, SpectralDescriptor):
    """Heat kernel signature.

    Parameters
    ----------
    scale : bool
        Whether to scale weights to sum to one.
    n_domain : int
        Number of domain points. Ignored if ``domain`` is not None.
    domain : callable or array-like, shape=[n_domain]
        Method to compute domain points (``f(shape)``) or
        domain points.
    """

    _Registry = HeatKernelSignatureRegistry

    def __init__(self, scale=True, n_domain=3, domain=None):
        super().__init__(
            domain or (lambda shape: hks_default_domain(shape, n_domain=n_domain)),
            use_landmarks=False,
        )
        self.scale = scale

    def __call__(self, shape):
        """Compute descriptor.

        Parameters
        ----------
        shape : Shape.
            Shape with basis.

        Returns
        -------
        descr : array-like, shape=[n_domain, n_vertices]
            Descriptor.
        """
        domain = self.domain(shape) if callable(self.domain) else self.domain

        vals_term = np.exp(-la.scalarvecmul(domain, shape.basis.vals))
        vecs_term = np.square(shape.basis.vecs)

        if self.scale:
            vals_term = la.scale_to_unit_sum(vals_term)

        return np.einsum("...j,ij->...i", vals_term, vecs_term)


class WaveKernelSignature(WhichRegistryMixins, SpectralDescriptor):
    """Wave kernel signature."""

    _Registry = WaveKernelSignatureRegistry

    def __init__(self, scale=True, sigma=None, n_domain=3, domain=None):
        super().__init__(
            domain or WksDefaultDomain(n_domain=n_domain, sigma=sigma),
            use_landmarks=False,
        )
        self.scale = scale
        self.sigma = sigma

    def __call__(self, shape):
        """Compute descriptor.

        Parameters
        ----------
        shape : Shape.
            Shape with basis.

        Returns
        -------
        descr : array-like, shape=[n_domain, n_vertices]
            Descriptor.
        """
        if callable(self.domain):
            # TODO: document domain better
            domain, sigma = self.domain(shape)
        else:
            domain = self.domain
            sigma = self.sigma

        exp_arg = -np.square(np.log(shape.basis.nonzero_vals) - domain[:, None]) / (
            2 * sigma**2
        )
        vals_term = np.exp(exp_arg)
        vecs_term = np.square(shape.basis.nonzero_vecs)

        if self.scale:
            vals_term = la.scale_to_unit_sum(vals_term)

        return np.einsum("...j,ij->...i", vals_term, vecs_term)
