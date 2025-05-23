import torch as _torch

from . import sparse


def scatter_sum_1d(index, src, size=None):
    if size is None:
        size = index.max() + 1

    array = _torch.zeros(size, dtype=src.dtype)
    return _torch.scatter_add(array, -1, index, src)
