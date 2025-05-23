import scipy as _scipy


def to_dense(array):
    return array.todense()


def from_scipy_coo(array):
    return array


def from_scipy_csc(array):
    return array


def from_scipy_dia(array):
    return array


def coo_matrix(indices, values, shape=None, dtype=None, coalesce=False):
    if not coalesce:
        return _scipy.sparse.coo_matrix((values, indices), shape=shape, dtype=dtype)

    return _scipy.sparse.csr_matrix((values, indices), shape=shape, dtype=dtype).tocoo()


def csr_matrix(indices, values, shape=None, dtype=None, coalesce=False):
    # TODO: need to adapt to other arg combinations
    # TODO: expose other arguments
    if not coalesce:
        return _scipy.sparse.csr_matrix((values, indices), shape=shape, dtype=dtype)

    return _scipy.sparse.coo_matrix((values, indices), shape=shape, dtype=dtype).tocsr()


def csc_matrix(indices, values, shape=None, dtype=None, coalesce=False):
    if not coalesce:
        return _scipy.sparse.csc_matrix((values, indices), shape=shape, dtype=dtype)

    return _scipy.sparse.coo_matrix((values, indices), shape=shape, dtype=dtype).tocsc()


def dia_matrix(diagonals, offsets=0, shape=None, dtype=None):
    if shape is None:
        shape = (diagonals.shape[-1], diagonals.shape[-1])
    return _scipy.sparse.dia_matrix((diagonals, offsets), shape=shape)
