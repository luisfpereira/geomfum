import numpy as _np
import scipy as _scipy
from numpy import geomspace, square

from . import sparse


def scatter_sum_1d(index, src, size=None):
    shape = None if size is None else (size, 1)

    dummy_indices = _np.zeros_like(index)

    return _np.array(
        _scipy.sparse.coo_matrix(
            (src, (index, dummy_indices)),
            shape=shape,
        ).todense()
    ).flatten()
