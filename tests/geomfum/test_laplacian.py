"""Test Laplacian-related machinery for meshes."""

import random

import pytest
from polpo.testing import DataBasedParametrizer

from geomfum.laplacian import LaplacianFinder, LaplacianSpectrumFinder
from tests.cases.cmp import (
    LaplacianFinderCmpCase,
    LaplacianSpectrumFinderCmpCase,
)

from .data.laplacian import LaplacianFinderCmpData, LaplacianSpectrumFinderCmpData


@pytest.mark.redundant
@pytest.mark.usefixtures("data_check")
class TestLaplacianFinderCmp(LaplacianFinderCmpCase, metaclass=DataBasedParametrizer):
    """Laplacian finder comparison.

    Notes
    -----
    Redundant with ``TestLaplacianSpectrumFinderCmp``.
    """

    finder_a = LaplacianFinder.from_registry(which="pyfm")
    finder_b = LaplacianFinder.from_registry(which="igl")

    testing_data = LaplacianFinderCmpData()


@pytest.fixture(
    scope="class",
    params=[
        ("default", "pyfm"),
        ("pyfm", "igl"),
        ("pyfm", "robust"),
        ("igl", "robust"),
    ],
)
def spectrum_finders(request):
    which_a, which_b = request.param

    spectrum_size = random.randint(2, 5)
    if which_a == "default":
        request.cls.finder_a = LaplacianSpectrumFinder(
            spectrum_size=spectrum_size,
            laplacian_finder=LaplacianFinder(),
        )
    else:
        request.cls.finder_a = LaplacianSpectrumFinder(
            spectrum_size=spectrum_size,
            laplacian_finder=LaplacianFinder.from_registry(which=which_a),
        )

    if which_b == "default":
        request.cls.finder_b = LaplacianSpectrumFinder(
            spectrum_size=spectrum_size,
            laplacian_finder=LaplacianFinder(),
        )
    else:
        request.cls.finder_b = LaplacianSpectrumFinder(
            spectrum_size=spectrum_size,
            laplacian_finder=LaplacianFinder.from_registry(which=which_b),
        )


@pytest.mark.usefixtures("data_check", "spectrum_finders")
class TestLaplacianSpectrumFinderCmp(
    LaplacianSpectrumFinderCmpCase, metaclass=DataBasedParametrizer
):
    """Laplacian spectrum finder comparison."""

    testing_data = LaplacianSpectrumFinderCmpData()
