import abc

import numpy as np
import pyFM.signatures

from ._base import Descriptor


class SpectralDescriptor(Descriptor, abc.ABC):
    # TODO: homogenize signatures
    # TODO: handle landmarks?
    # TODO: make general implementation

    def __init__(self, n_domain, domain_func):
        self.n_domain = n_domain
        self.domain_func = domain_func


class PyfmHeatKernelSignature(SpectralDescriptor):
    """Heat kernel signature using pyFM.

    n_domain : int
        Number of times. Ignored if ``domain`` is not ``None``.
    """

    def __init__(self, scaled=True, n_domain=3, domain_func=None):
        super().__init__(n_domain, domain_func or self.default_domain)
        self.scaled = scaled

    def default_domain(self, basis, n_domain):
        abs_ev = np.sort(np.abs(basis.vals))
        index = 1 if np.isclose(abs_ev[0], 0.0) else 0
        return np.geomspace(
            4 * np.log(10) / abs_ev[-1], 4 * np.log(10) / abs_ev[index], n_domain
        )

    def __call__(self, basis, domain=None):
        """ """
        if domain is None:
            domain = self.domain_func(basis, self.n_domain)

        return pyFM.signatures.HKS(basis.vals, basis.vecs, domain, scaled=self.scaled)


class HeatKernelSignature:
    """Heat kernel signature.

    Parameters
    ----------
    which : str
        One of: pyfm
    """

    _MAP = {"pyfm": PyfmHeatKernelSignature}

    def __new__(cls, which="pyfm", **kwargs):
        return cls._MAP[which](**kwargs)


class PyfmWaveKernelSignature(SpectralDescriptor):
    """Wave kernel signature using pyFM.

    Parameters
    ----------
    n_domain : int
        Number of energies. Ignored if ``domain`` is not ``None``.
    """

    def __init__(self, scaled=True, sigma=None, n_domain=3, domain_func=None):
        super().__init__(n_domain, domain_func or self.domain_func)

        self.scaled = scaled
        self.sigma = None

    def default_sigma(self, e_min, e_max, n_domain):
        return 7 * (e_max - e_min) / n_domain

    def domain_func(self, basis, n_domain):
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
        """
        n_time : int
            Ignored if ``time`` is not ``None``.
        """
        # TODO: handle landmarks?
        if domain is None:
            energy, sigma = self.domain_func(basis, self.n_domain)

        elif self.sigma is None:
            sigma = self.default_sigma(np.amin(domain), np.amax(domain), len(domain))

        return pyFM.signatures.WKS(
            basis.vals, basis.vecs, energy, sigma, scaled=self.scaled
        )


class WaveKernelSignature:
    """Wave kernel signature.

    Parameters
    ----------
    which : str
        One of: pyfm
    """

    _MAP = {"pyfm": PyfmWaveKernelSignature}

    def __new__(cls, which="pyfm", **kwargs):
        return cls._MAP[which](**kwargs)
