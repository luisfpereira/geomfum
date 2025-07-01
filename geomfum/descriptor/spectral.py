"""Spectral descriptors."""

import geomstats.backend as gs

import geomfum.backend as xgs
import geomfum.linalg as la
from geomfum._registry import (
    HeatKernelSignatureRegistry,
    WaveKernelSignatureRegistry,
    LandmarkHeatKernelSignatureRegistry,
    LandmarkWaveKernelSignatureRegistry,
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
    device = getattr(nonzero_vals, "device", None)

    return xgs.to_device(
        xgs.geomspace(
            4 * gs.log(10) / nonzero_vals[-1],
            4 * gs.log(10) / nonzero_vals[0],
            n_domain,
        ),
        device,
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
        device = getattr(nonzero_vals, "device", None)

        e_min, e_max = gs.log(nonzero_vals[0]), gs.log(nonzero_vals[-1])

        sigma = (
            self.n_overlap * (e_max - e_min) / self.n_domain
            if self.sigma is None
            else self.sigma
        )

        e_min += self.n_trans * sigma
        e_max -= self.n_trans * sigma

        energy = xgs.to_device(gs.linspace(e_min, e_max, self.n_domain), device)

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
            domain or (lambda shape: hks_default_domain(shape, n_domain=n_domain))
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

        vals_term = gs.exp(-la.scalarvecmul(domain, shape.basis.vals))
        vecs_term = xgs.square(shape.basis.vecs)

        if self.scale:
            vals_term = la.scale_to_unit_sum(vals_term)

        return gs.einsum("...j,ij->...i", vals_term, vecs_term)


class WaveKernelSignature(WhichRegistryMixins, SpectralDescriptor):
    """Wave kernel signature."""

    _Registry = WaveKernelSignatureRegistry

    def __init__(self, scale=True, sigma=None, n_domain=3, domain=None):
        super().__init__(
            domain or WksDefaultDomain(n_domain=n_domain, sigma=sigma),
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

        exp_arg = -xgs.square(gs.log(shape.basis.nonzero_vals) - domain[:, None]) / (
            2 * xgs.square(sigma)
        )
        vals_term = gs.exp(exp_arg)
        vecs_term = xgs.square(shape.basis.nonzero_vecs)

        if self.scale:
            vals_term = la.scale_to_unit_sum(vals_term)

        return gs.einsum("...j,ij->...i", vals_term, vecs_term)


class LandmarkHeatKernelSignature(SpectralDescriptor):
    """Landmark-based Heat Kernel Signature.

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

    _Registry = LandmarkHeatKernelSignatureRegistry

    def __init__(self, scale=True, n_domain=3, domain=None):
        super().__init__(
            domain or (lambda shape: hks_default_domain(shape, n_domain=n_domain)),
        )
        self.scale = scale

    def __call__(self, shape):
        """Compute landmark-based HKS descriptor.

        Parameters
        ----------
        shape : Shape.
            Shape with basis and landmark_indices.

        Returns
        -------
        descr : array-like, shape=[n_landmarks * n_domain, n_vertices]
            Landmark-based HKS descriptor.
        """
        if not hasattr(shape, "landmark_indices") or shape.landmark_indices is None:
            raise AttributeError(
                "Shape must have 'landmark_indices' set for LandmarkHeatKernelSignature."
            )

        domain = self.domain(shape) if callable(self.domain) else self.domain
        evals = shape.basis.vals
        evects = shape.basis.vecs
        landmarks = shape.landmark_indices

        # coefs: (n_domain, n_eigen)
        coefs = gs.exp(-gs.outer(domain, evals))
        # weighted_evects: (n_domain, n_landmarks, n_eigen)
        weighted_evects = evects[landmarks][None, :, :] * coefs[:, None, :]
        # landmarks_HKS: (n_landmarks, n_domain, n_vertices)
        landmarks_HKS = gs.einsum("tpk,nk->ptn", weighted_evects, evects)

        if self.scale:
            inv_scaling = coefs.sum(1)  # (n_domain,)
            landmarks_HKS = (1 / inv_scaling)[None, :, None] * landmarks_HKS

        # reshape to (n_landmarks * n_domain, n_vertices)
        descr = gs.reshape(
            landmarks_HKS,
            (landmarks_HKS.shape[0] * landmarks_HKS.shape[1], evects.shape[0]),
        )
        return descr


class LandmarkWaveKernelSignature(SpectralDescriptor):
    """Landmark-based Wave Kernel Signature.

    Parameters
    ----------
    scale : bool
        Whether to scale weights to sum to one.
    sigma : float
        Standard deviation for the Gaussian.
    n_domain : int
        Number of domain points. Ignored if ``domain`` is not None.
    domain : callable or array-like, shape=[n_domain]
        Method to compute domain points (``f(shape)``) or
        domain points.
    """

    _Registry = LandmarkWaveKernelSignatureRegistry

    def __init__(self, scale=True, sigma=None, n_domain=3, domain=None):
        super().__init__(
            domain or WksDefaultDomain(n_domain=n_domain, sigma=sigma),
        )
        self.scale = scale
        self.sigma = sigma

    def __call__(self, shape):
        """Compute landmark-based WKS descriptor.

        Parameters
        ----------
        shape : Shape.
            Shape with basis and landmark_indices.

        Returns
        -------
        descr : array-like, shape=[n_landmarks * n_domain, n_vertices]
            Landmark-based WKS descriptor.
        """
        if not hasattr(shape, "landmark_indices") or shape.landmark_indices is None:
            raise AttributeError(
                "Shape must have 'landmark_indices' set for LandmarkWaveKernelSignature."
            )

        if callable(self.domain):
            domain, sigma = self.domain(shape)
        else:
            domain = self.domain
            sigma = self.sigma

        if sigma is None or sigma <= 0:
            raise ValueError(f"Sigma should be positive! Given value: {sigma}")

        evals = shape.basis.nonzero_vals
        evects = shape.basis.nonzero_vecs
        landmarks = shape.landmark_indices

        # coefs: (n_domain, n_eigen)
        coefs = gs.exp(
            -xgs.square(domain[:, None] - gs.log(gs.abs(evals))[None, :])
            / (2 * sigma**2)
        )
        # weighted_evects: (n_domain, n_landmarks, n_eigen)
        weighted_evects = evects[None, landmarks, :] * coefs[:, None, :]
        # landmarks_WKS: (n_landmarks, n_domain, n_vertices)
        landmarks_WKS = gs.einsum("tpk,nk->ptn", weighted_evects, evects)

        if self.scale:
            inv_scaling = coefs.sum(1)  # (n_domain,)
            landmarks_WKS = (1 / inv_scaling)[None, :, None] * landmarks_WKS

        # reshape to (n_landmarks * n_domain, n_vertices)
        descr = gs.reshape(
            landmarks_WKS,
            (landmarks_WKS.shape[0] * landmarks_WKS.shape[1], evects.shape[0]),
        )
        return descr
