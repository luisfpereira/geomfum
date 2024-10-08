import random

import numpy as np
from geomstats.test.data import TestData
from pyFM.mesh import TriMesh

from geomfum.shape import TriangleMesh
from tests.utils import DATA_DIR


class SpectralDescriptorCmpData(TestData):
    trials = 1
    # TODO: use smaller meshes
    _filenames = ["cat-00.off"]
    shapes = [
        TriangleMesh.from_file(f"{DATA_DIR}/{filename}") for filename in _filenames
    ]
    pyfm_shapes = [TriMesh(mesh.vertices, mesh.faces) for mesh in shapes]

    def set_landmarks(self):
        n_landmarks = random.randint(1, 5)
        for shape, pyfm_shape in zip(self.shapes, self.pyfm_shapes):
            landmark_indices = np.random.choice(shape.n_vertices, size=n_landmarks)
            shape.set_landmarks(landmark_indices)
            pyfm_shape.landmark_indices = landmark_indices

    def set_spectrum(self, spectrum_size):
        spectrum_size = spectrum_size
        for shape, pyfm_shape in zip(self.shapes, self.pyfm_shapes):
            shape.laplacian.find_spectrum(spectrum_size=spectrum_size)
            shape.basis.use_k = spectrum_size

            pyfm_shape.eigenvalues = shape.basis.full_vals
            pyfm_shape.eigenvectors = shape.basis.full_vecs

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
