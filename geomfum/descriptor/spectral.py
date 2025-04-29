"""Spectral descriptors."""

import numpy as np

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
    abs_ev = np.sort(np.abs(shape.basis.vals))
    index = 1 if np.isclose(abs_ev[0], 0.0) else 0
    return np.geomspace(
        4 * np.log(10) / abs_ev[-1], 4 * np.log(10) / abs_ev[index], n_domain
    )


class HeatKernelSignature(WhichRegistryMixins, SpectralDescriptor):
    """Heat kernel signature."""

    _Registry = HeatKernelSignatureRegistry

    def __init__(self, n_domain=3, domain=None):
        super().__init__(n_domain, domain or hks_default_domain, use_landmarks=False)

    def __call__(self, shape, domain=None):
        """Compute descriptor.

        Parameters
        ----------
        shape : Shape.
            Shape with basis.
        domain : array-like, shape=[n_domain]
            Time points.

        Returns
        -------
        descr : array-like, shape=[n_domain, n_vertices]
            Descriptor.
        """
        if domain is None:
            domain = (
                self.domain(shape, self.n_domain)
                if callable(self.domain)
                else self.domain
            )

        time_term = np.exp(-np.einsum("i,...->...i", shape.basis.vals, domain))
        space_term = np.square(shape.basis.vecs)

        return np.einsum("...j,ij->...i", time_term, space_term)


class WaveKernelSignature(WhichRegistryMixins):
    """Wave kernel signature."""

    _Registry = WaveKernelSignatureRegistry
