import random

import numpy as np
import pyFM.signatures as sg
import pytest
from geomstats.test.parametrizers import DataBasedParametrizer

from geomfun.descriptor.spectral import HeatKernelSignature, WaveKernelSignature
from tests.cases.pyfm import SpectralDescriptorCmpCase

from .data.spectral import SpectralDescriptorCmpData


def _pyfm_spectral(num_T, k, mesh_func, func_domain):
    def spectral(_cls, mesh, domain=None):
        if domain is None:
            return mesh_func(
                mesh,
                num_T,
                k=k,
            )
        else:
            return func_domain(mesh.eigenvalues, mesh.eigenvectors, domain, scaled=True)

    return spectral


def _pyfm_wks(num_T, k, sigma):
    def spectral(_cls, mesh, domain=None):
        if domain is None:
            abs_ev = sorted(np.abs(mesh.eigenvalues))
            e_min, e_max = np.log(abs_ev[1]), np.log(abs_ev[-1])
            e_min += 2 * sigma
            e_max -= 2 * sigma
            domain = np.linspace(e_min, e_max, num_T)

            return sg.WKS(
                mesh.eigenvalues, mesh.eigenvectors, domain, sigma, scaled=True
            )
        else:
            return sg.WKS(
                mesh.eigenvalues, mesh.eigenvectors, domain, sigma, scaled=True
            )

    return spectral


@pytest.fixture(
    scope="class",
    params=["hks", "wks"],
)
def spectral_descriptors(request):
    descr_type = request.param

    n_domain = random.randint(2, 5)
    request.cls.spectrum_size = spectrum_size = random.randint(3, 5)

    if descr_type == "hks":
        descriptor = HeatKernelSignature(
            scaled=True,
            n_domain=n_domain,
        )

        pyfm_descriptor = _pyfm_spectral(
            num_T=n_domain,
            k=spectrum_size,
            mesh_func=sg.mesh_HKS,
            func_domain=sg.HKS,
        )

    elif descr_type == "wks":
        sigma = np.random.uniform(low=0.1, high=2.0, size=1)[0]
        descriptor = WaveKernelSignature(
            scaled=True,
            n_domain=n_domain,
            sigma=sigma,
        )
        pyfm_descriptor = _pyfm_wks(
            num_T=n_domain,
            k=spectrum_size,
            sigma=sigma,
        )
    else:
        raise ValueError(f"Unknown descriptor type: {descr_type}")

    request.cls.descriptor = descriptor
    request.cls.pyfm_descriptor = pyfm_descriptor


@pytest.mark.usefixtures("spectral_descriptors")
class TestSpectralDescriptorCmp(
    SpectralDescriptorCmpCase, metaclass=DataBasedParametrizer
):
    testing_data = SpectralDescriptorCmpData()
