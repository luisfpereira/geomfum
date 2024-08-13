import random

import numpy as np
import pyFM.signatures as sg
import pytest
from geomstats.test.parametrizers import DataBasedParametrizer

from geomfun.descriptor.spectral import HeatKernelSignature, WaveKernelSignature
from tests.cases.pyfm import SpectralDescriptorCmpCase

from .data.spectral import SpectralDescriptorCmpData


def _pyfm_hks(num_T, k, use_landmarks=False):
    def hks(_cls, mesh, domain=None):
        landmarks = mesh.landmark_indices if use_landmarks else None

        if domain is None:
            return sg.mesh_HKS(
                mesh,
                num_T=num_T,
                landmarks=landmarks,
                k=k,
            )

        else:
            if use_landmarks:
                return sg.lm_HKS(
                    mesh.eigenvalues[:k],
                    mesh.eigenvectors[:, :k],
                    landmarks,
                    domain,
                    scaled=True,
                )

            return sg.HKS(
                mesh.eigenvalues[:k], mesh.eigenvectors[:, :k], domain, scaled=True
            )

    return hks


def _pyfm_wks(num_T, k, sigma, use_landmarks=False):
    def wks(_cls, mesh, domain=None):
        landmarks = mesh.landmark_indices if use_landmarks else None

        if domain is None:
            abs_ev = sorted(np.abs(mesh.eigenvalues[:k]))
            e_min, e_max = np.log(abs_ev[1]), np.log(abs_ev[-1])
            e_min += 2 * sigma
            e_max -= 2 * sigma
            domain = np.linspace(e_min, e_max, num_T)

        if use_landmarks:
            return sg.lm_WKS(
                mesh.eigenvalues[:k],
                mesh.eigenvectors[:, :k],
                landmarks,
                domain,
                sigma,
                scaled=True,
            )

        return sg.WKS(
            mesh.eigenvalues[:k],
            mesh.eigenvectors[:, :k],
            domain,
            sigma,
            scaled=True,
        )

    return wks


@pytest.fixture(
    scope="class",
    params=[
        ("hks", False),
        ("hks", True),
        ("wks", False),
        ("wks", True),
    ],
)
def spectral_descriptors(request):
    descr_type, use_landmarks = request.param

    n_domain = random.randint(2, 5)
    request.cls.spectrum_size = spectrum_size = random.randint(3, 5)

    testing_data = request.cls.testing_data

    # TODO: make trigger later
    testing_data.set_spectrum(spectrum_size)

    if use_landmarks:
        testing_data.set_landmarks()

    if descr_type == "hks":
        descriptor = HeatKernelSignature(
            scaled=True, n_domain=n_domain, use_landmarks=use_landmarks
        )

        pyfm_descriptor = _pyfm_hks(
            num_T=n_domain, k=spectrum_size, use_landmarks=use_landmarks
        )

    elif descr_type == "wks":
        sigma = np.random.uniform(low=0.1, high=2.0, size=1)[0]
        descriptor = WaveKernelSignature(
            scaled=True, n_domain=n_domain, sigma=sigma, use_landmarks=use_landmarks
        )
        pyfm_descriptor = _pyfm_wks(
            num_T=n_domain, k=spectrum_size, sigma=sigma, use_landmarks=use_landmarks
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
