import random

import numpy as np
from geomstats.test.data import TestData

from tests.utils import MeshCollectionWithPyfm


class SpectralDescriptorCmpData(TestData):
    trials = 1

    _indices = ["cat-00"]
    shapes = MeshCollectionWithPyfm(_indices, target_reduction=0.95, recompute=False)

    def descriptor_cmp_test_data(self):
        data = []
        for domain in [None, np.random.uniform(size=random.randint(1, 3))]:
            data.extend(
                [
                    dict(shape_key=shape_key, domain=domain)
                    for shape_key in self.shapes.keys()
                ]
            )
        return self.generate_tests(data)
