from geomstats.test.data import TestData

from geomfum.dataset import NotebooksDataset
from geomfum.shape import TriangleMesh

_DATASET = NotebooksDataset()


class LaplacianFinderCmpData(TestData):
    _indices = ["cat-00"]
    shapes = [
        TriangleMesh.from_file(_DATASET.get_filename(index)) for index in _indices
    ]

    def matrices_cmp_test_data(self):
        data = [dict(shape=shape) for shape in self.shapes]
        return self.generate_tests(data)


class LaplacianSpectrumFinderCmpData(TestData):
    _indices = ["cat-00"]
    shapes = [
        TriangleMesh.from_file(_DATASET.get_filename(index)) for index in _indices
    ]

    tolerances = {"eigenvals_cmp": {"atol": 1e-1}}

    def eigenvals_cmp_test_data(self):
        data = [dict(shape=shape) for shape in self.shapes]
        return self.generate_tests(data)
