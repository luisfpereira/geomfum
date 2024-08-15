"""Test Laplacian-related machinery for meshes."""

import random

import pytest
from geomstats.test.parametrizers import DataBasedParametrizer

from geomfum.laplacian import LaplacianFinder, LaplacianSpectrumFinder
from tests.cases.laplacian import (
    LaplacianFinderCmpCase,
    LaplacianSpectrumFinderCmpCase,
)

from .data.mesh import LaplacianFinderCmpData, LaplacianSpectrumFinderCmpData


@pytest.mark.skip
@pytest.mark.redundant
class TestLaplacianFinderCmp(LaplacianFinderCmpCase, metaclass=DataBasedParametrizer):
    """Laplacian finder comparison.

    Notes
    -----
    Redundant with ``TestLaplacianSpectrumFinderCmp``.
    """

    finder_a = LaplacianFinder.from_registry(which="robust")
    finder_b = LaplacianFinder.from_registry(which="pyfm")

    testing_data = LaplacianFinderCmpData()


@pytest.fixture(
    scope="class",
    params=[("robust", "pyfm"), ("robust", "igl")],
)
def spectrum_finders(request):
    which_a, which_b = request.param

    spectrum_size = random.randint(2, 5)
    request.cls.finder_a = LaplacianSpectrumFinder(
        spectrum_size=spectrum_size,
        laplacian_finder=LaplacianFinder.from_registry(which=which_a),
    )

    request.cls.finder_b = LaplacianSpectrumFinder(
        spectrum_size=spectrum_size,
        laplacian_finder=LaplacianFinder.from_registry(which=which_b),
    )


@pytest.mark.usefixtures("spectrum_finders")
class TestLaplacianSpectrumFinderCmp(
    LaplacianSpectrumFinderCmpCase, metaclass=DataBasedParametrizer
):
    """Laplacian spectrum finder comparison."""

    testing_data = LaplacianSpectrumFinderCmpData()
