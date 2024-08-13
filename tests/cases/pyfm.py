import numpy as np
import pytest
from geomstats.test.test_case import TestCase


class SpectralDescriptorCmpCase(TestCase):
    """Laplacian finder comparison.

    Notes
    -----
    Needs: `descriptor`, `pyfm_descriptor`
    """

    # TODO: add also descriptor batch?

    def test_descriptor_cmp(self, shape, pyfm_shape, atol, domain=None):
        descr = self.descriptor(shape.basis, domain=domain)
        pyfm_descr = self.pyfm_descriptor(pyfm_shape, domain=domain)

        self.assertAllClose(descr, pyfm_descr.T, atol=atol)


class WeightedFactorCmpCase(TestCase):
    # TODO: find bug without underscore

    def _random_fmap_matrix(self):
        return np.random.uniform(size=(self.spectrum_size_b, self.spectrum_size_a))

    def _test_factor_value(self, fmap_matrix, atol):
        # TODO: identify bug when `test_factor_value`
        value = self.factor(fmap_matrix)
        pyfm_value = self.pyfm_factor(fmap_matrix)
        self.assertAllClose(value, pyfm_value, atol=atol)

    @pytest.mark.random
    def test_factor_value_random(self, atol):
        return self._test_factor_value(self._random_fmap_matrix(), atol)

    def _test_factor_gradient(self, fmap_matrix, atol):
        gradient = self.factor.gradient(fmap_matrix)
        pyfm_gradient = self.pyfm_factor.gradient(fmap_matrix)

        self.assertAllClose(gradient, pyfm_gradient, atol=atol)

    @pytest.mark.random
    def test_factor_gradient_random(self, atol):
        return self._test_factor_gradient(self._random_fmap_matrix(), atol)
