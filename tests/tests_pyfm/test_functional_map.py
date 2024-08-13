import pytest
from geomstats.test.parametrizers import DataBasedParametrizer
from pyFM.optimize.base_functions import (
    LB_commutation,
    LB_commutation_grad,
    descr_preservation,
    descr_preservation_grad,
    op_commutation,
    op_commutation_grad,
)

from geomfum.functional_map import (
    LBCommutativityEnforcing,
    OperatorCommutativityEnforcing,
    SpectralDescriptorPreservation,
    WeightedFactor,
)
from tests.cases.pyfm import WeightedFactorCmpCase

from .data.functional_map import WeightedFactorCmpTestData


class PyfmDescrPreservation(WeightedFactor):
    def __init__(self, descr1_red, descr2_red, weight=1.0):
        super().__init__(weight=weight)
        self.descr1_red = descr1_red
        self.descr2_red = descr2_red

    def __call__(self, fmap_matrix):
        return self.weight * descr_preservation(
            fmap_matrix, self.descr1_red, self.descr2_red
        )

    def gradient(self, fmap_matrix):
        return descr_preservation_grad(fmap_matrix, self.descr1_red, self.descr2_red)


class PyfmLBCommutation(WeightedFactor):
    def __init__(self, ev_sqdiff, weight=1.0):
        super().__init__(weight=weight)
        self.ev_sqdiff = ev_sqdiff

    def __call__(self, fmap_matrix):
        return LB_commutation(fmap_matrix, self.ev_sqdiff)

    def gradient(self, fmap_matrix):
        return LB_commutation_grad(fmap_matrix, self.ev_sqdiff)


class PyfmOpCommutation(WeightedFactor):
    def __init__(self, op1, op2, weight=1.0):
        super().__init__(weight=weight)
        self.op1 = op1
        self.op2 = op2

    def __call__(self, fmap_matrix):
        return op_commutation(fmap_matrix, self.op1, self.op2)

    def gradient(self, fmap_matrix):
        return op_commutation_grad(fmap_matrix, self.op1, self.op2)


@pytest.fixture(
    scope="class",
    params=[
        "spectral_descriptor_preservation",
        "lb_commutativity",
        "operator_commutativity-multiplication",
        "operator_commutativity-orientation",
    ],
)
def factors(request):
    factor_type = request.param

    testing_data = request.cls.testing_data

    shape_a, shape_b = testing_data.shape_a, testing_data.shape_b
    descr_a, descr_b = testing_data.generate_random_descriptors()

    if factor_type == "spectral_descriptor_preservation":
        sdescr_a = shape_a.basis.project(descr_a)
        sdescr_b = shape_b.basis.project(descr_b)

        factor = SpectralDescriptorPreservation(sdescr_a, sdescr_b)
        pyfm_factor = PyfmDescrPreservation(sdescr_a.T, sdescr_b.T)

    elif factor_type == "lb_commutativity":
        factor = LBCommutativityEnforcing.from_bases(shape_a.basis, shape_b.basis)
        pyfm_factor = PyfmLBCommutation(factor.vals_sqdiff)

    elif factor_type == "operator_commutativity-multiplication":
        factor = OperatorCommutativityEnforcing.from_multiplication(
            shape_a.basis, descr_a[0], shape_b.basis, descr_b[0]
        )
        pyfm_factor = PyfmOpCommutation(factor.oper_a, factor.oper_b)

    elif factor_type == "operator_commutativity-orientation":
        factor = OperatorCommutativityEnforcing.from_orientation(
            shape_a, descr_a[0], shape_b, descr_b[0]
        )
        pyfm_factor = PyfmOpCommutation(factor.oper_a, factor.oper_b)
    else:
        raise ValueError(f"Unkown factor type: {factor_type}")

    request.cls.factor = factor
    request.cls.pyfm_factor = pyfm_factor

    request.cls.spectrum_size_a = shape_a.basis.spectrum_size
    request.cls.spectrum_size_b = shape_b.basis.spectrum_size


@pytest.mark.usefixtures("factors")
class TestWeightedFactorCmp(WeightedFactorCmpCase, metaclass=DataBasedParametrizer):
    testing_data = WeightedFactorCmpTestData()
