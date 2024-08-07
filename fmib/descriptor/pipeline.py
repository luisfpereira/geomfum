import abc

import numpy as np

from ._base import Descriptor
from .spectral import SpectralDescriptor


class Subsampler(abc.ABC):
    @abc.abstractmethod
    def __call__(self, array):
        pass


class ArangeSubsampler(Subsampler):
    def __init__(self, subsample_step=1, axis=-1):
        self.subsample_step = subsample_step
        self.axis = axis

    def __call__(self, array):
        indices = np.arange(0, array.shape[self.axis], self.subsample_step)
        slc = [slice(None)] * array.ndim
        slc[self.axis] = indices

        return array[tuple(slc)]


class Normalizer(abc.ABC):
    @abc.abstractmethod
    def __call__(self, shape, array):
        pass


class L2InnerNormalizer(Normalizer):
    def __call__(self, shape, array):
        coeff = np.sqrt(
            np.einsum("np,np->p", array, shape.basis.mass_matrix @ array),
        )
        return array / coeff


class DescriptorPipeline:
    # steps: descriptor, subsampler, normalizer
    def __init__(self, steps):
        self.steps = steps

    def _update_descr(self, current, new):
        if current is None:
            return new
        return np.c_[current, new]

    def apply(self, shape):
        descr = None
        for step in self.steps:
            if isinstance(step, SpectralDescriptor):
                descr = self._update_descr(descr, step(shape.basis))

            elif isinstance(step, Descriptor):
                descr = self._update_descr(descr, step(shape))

            elif isinstance(step, Subsampler):
                descr = step(descr)

            elif isinstance(step, Normalizer):
                descr = step(shape, descr)

            else:
                raise ValueError("Unkown step type.")

        return descr
