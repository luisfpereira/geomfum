import numpy as np
import pyFM.signatures

from geomfun.descriptor._base import SpectralDescriptor
from geomfun.laplacian._base import BaseLaplacianFinder


class PyfmMeshLaplacianFinder(BaseLaplacianFinder):
    """Algorithm to find the Laplacian of a mesh."""

    def __call__(self, shape):
        """Apply algorithm.

        Parameters
        ----------
        shape : TriangleMesh
            Mesh.

        Returns
        -------
        laplace_matrix : scipy.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Laplace matrix.
        mass_matrix : scipy.sparse.csc_matrix, shape=[n_vertices, n_vertices]
            Diagonal lumped mass matrix.
        """
        return (
            pyFM.mesh.laplacian.cotangent_weights(shape.vertices, shape.faces),
            pyFM.mesh.laplacian.dia_area_mat(shape.vertices, shape.faces),
        )


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
    use_landmarks : bool
        Whether to use landmarks.
    """

    def __init__(self, scaled=True, n_domain=3, domain=None, use_landmarks=False):
        super().__init__(
            n_domain, domain or self.default_domain, use_landmarks=use_landmarks
        )
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

        if self.use_landmarks:
            return pyFM.signatures.lm_HKS(
                basis.vals,
                basis.vecs,
                basis.landmark_indices,
                domain,
                scaled=self.scaled,
            ).T

        return pyFM.signatures.HKS(basis.vals, basis.vecs, domain, scaled=self.scaled).T


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
    use_landmarks : bool
        Whether to use landmarks.
    """

    def __init__(
        self, scaled=True, sigma=None, n_domain=3, domain=None, use_landmarks=False
    ):
        super().__init__(
            n_domain, domain or self.default_domain, use_landmarks=use_landmarks
        )

        self.scaled = scaled
        self.sigma = sigma

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
        descr : array-like, shape=[{n_domain, n_landmarks*n_domain}, n_vertices]
            Descriptor.
        """
        sigma = None
        if domain is None:
            if callable(self.domain):
                domain, sigma = self.domain(basis, self.n_domain)
            else:
                domain = self.domain

        if sigma is None:
            # TODO: simplify sigma
            # TODO: need to verify this
            sigma = (
                self.default_sigma(np.amin(domain), np.amax(domain), len(domain))
                if self.sigma is None
                else self.sigma
            )

        if self.use_landmarks:
            return pyFM.signatures.lm_WKS(
                basis.vals,
                basis.vecs,
                basis.landmark_indices,
                domain,
                sigma,
                scaled=self.scaled,
            ).T

        return pyFM.signatures.WKS(
            basis.vals, basis.vecs, domain, sigma, scaled=self.scaled
        ).T
