from geomstats.test.data import TestData

from tests.utils import get_meshes_from_indices


class LaplacianFinderCmpData(TestData):
    _indices = ["cat-00"]
    shapes = get_meshes_from_indices(_indices, target_reduction=0.95)

    tolerances = {"matrices_cmp": {"atol": 1e-3}}

    def matrices_cmp_test_data(self):
        data = [dict(shape=shape) for shape in self.shapes]
        return self.generate_tests(data)


class LaplacianSpectrumFinderCmpData(TestData):
    _indices = ["cat-00"]
    shapes = get_meshes_from_indices(_indices, target_reduction=0.6)

    tolerances = {"eigenvals_cmp": {"atol": 5e-1}}

    def eigenvals_cmp_test_data(self):
        data = [dict(shape=shape) for shape in self.shapes]
        return self.generate_tests(data)
