from geomstats.test.data import TestData

from geomfum.shape import TriangleMesh
from tests.utils import DATA_DIR


class LaplacianFinderCmpData(TestData):
    # TODO: use smaller meshes
    _filenames = ["cat-00.off"]
    shapes = [
        TriangleMesh.from_file(f"{DATA_DIR}/{filename}") for filename in _filenames
    ]

    def matrices_cmp_test_data(self):
        data = [dict(shape=shape) for shape in self.shapes]
        return self.generate_tests(data)


class LaplacianSpectrumFinderCmpData(TestData):
    # TODO: use smaller meshes
    # TODO: create default data
    _filenames = ["cat-00.off"]
    shapes = [
        TriangleMesh.from_file(f"{DATA_DIR}/{filename}") for filename in _filenames
    ]

    tolerances = {"eigenvals_cmp": {"atol": 1e-1}}

    def eigenvals_cmp_test_data(self):
        data = [dict(shape=shape) for shape in self.shapes]
        return self.generate_tests(data)
