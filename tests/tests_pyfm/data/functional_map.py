import random

import numpy as np
from geomstats.test.data import TestData
from pyFM.mesh import TriMesh

from geomfun.shape import TriangleMesh
from tests.utils import DATA_DIR


class WeightedFactorCmpTestData(TestData):
    trials = 1

    _filenames = ["cat-00.off", "lion-00.off"]
    shape_a, shape_b = [
        TriangleMesh.from_file(f"{DATA_DIR}/{filename}") for filename in _filenames
    ]
    pyfm_shape_a, pyfm_shape_b = [
        TriMesh(shape.vertices, shape.faces) for shape in (shape_a, shape_b)
    ]

    _spectrum_size_a = random.randint(3, 5)
    _spectrum_size_b = random.randint(3, 5)

    # TODO: define better when to trigger this
    shape_a.find_laplacian_spectrum(spectrum_size=_spectrum_size_a)
    shape_b.find_laplacian_spectrum(spectrum_size=_spectrum_size_b)

    pyfm_shape_a.eigenvalues = shape_a.basis.vals
    pyfm_shape_a.eigenvectors = shape_a.basis.vecs
    pyfm_shape_b.eigenvalues = shape_a.basis.vals
    pyfm_shape_b.eigenvectors = shape_a.basis.vecs

    def factor_value_random_test_data(self):
        return self.generate_tests([dict()])

    def factor_gradient_random_test_data(self):
        return self.generate_tests([dict()])

    def generate_random_descriptors(self, num_descr=None):
        num_descr = num_descr or random.randint(1, 6)
        descr_a = np.random.uniform(size=(num_descr, self.shape_a.n_vertices))
        descr_b = np.random.uniform(size=(num_descr, self.shape_b.n_vertices))

        return descr_a, descr_b
