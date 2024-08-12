import random

import numpy as np
from geomstats.test.data import TestData
from pyFM.mesh import TriMesh

from geomfun.shape import TriangleMesh
from tests.utils import DATA_DIR


class SpectralDescriptorCmpData(TestData):
    # TODO: use smaller meshes
    _filenames = ["cat-00.off"]
    shapes = [
        TriangleMesh.from_file(f"{DATA_DIR}/{filename}") for filename in _filenames
    ]
    pyfm_shapes = [TriMesh(mesh.vertices, mesh.faces) for mesh in shapes]

    def descriptor_cmp_test_data(self):
        data = []
        for domain in [None, np.random.uniform(size=random.randint(1, 3))]:
            data.extend(
                [
                    dict(shape=shape, pyfm_shape=pyfm_shape, domain=domain)
                    for shape, pyfm_shape in zip(self.shapes, self.pyfm_shapes)
                ]
            )
        return self.generate_tests(data)
