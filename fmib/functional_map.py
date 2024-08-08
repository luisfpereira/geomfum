import abc

import numpy as np

import fmib.linalg


class WeightedFactor(abc.ABC):
    """
    Parameters
    ----------
    weight : float
        Weight of the factor.
    """

    def __init__(self, weight):
        self.weight = weight

    @abc.abstractmethod
    def __call__(self, fmap_matrix):
        pass

    @abc.abstractmethod
    def gradient(self, fmap_matrix):
        pass


class SpectralDescriptorPreservation(WeightedFactor):
    # TODO: update docstrings
    """
    Parameters
    ----------
    descr1 :
        (K1,p) descriptors on first basis
    descr2 :
        (K2,p) descriptros on second basis
    """

    def __init__(self, sdescr_a, sdescr_b, weight=1.0):
        super().__init__(weight)
        self.sdescr_a = sdescr_a
        self.sdescr_b = sdescr_b

    def __call__(self, fmap_matrix):
        """
        Compute the descriptor preservation constraint

        Parameters
        ----------
        fmap_matrix      :
            (K2,K1) Functional map

        Returns
        -------
        energy : float
            descriptor preservation squared norm
        """
        return (
            self.weight
            * 0.5
            * np.square(fmap_matrix @ self.sdescr_a - self.sdescr_b).sum()
        )

    def gradient(self, fmap_matrix):
        """
        Compute the gradient of the descriptor preservation constraint

        Parameters
        ----------
        C      :
            (K2,K1) Functional map

        Returns
        -------
        gradient : np.ndarray
            gradient of the descriptor preservation squared norm
        """
        return (
            self.weight
            * (fmap_matrix @ self.sdescr_a - self.sdescr_b)
            @ self.sdescr_a.T
        )


class LBCommutativityEnforcing(WeightedFactor):
    """

    Parameters
    ----------
    ev_sqdiff :
        (K2,K1) [normalized] matrix of squared eigenvalue differences
    """

    def __init__(self, vals_sqdiff, weight=1.0):
        super().__init__(weight)
        self.vals_sqdiff = vals_sqdiff

    @staticmethod
    def from_bases(basis_a, basis_b, weight=1.0):
        vals_sqdiff = np.square(basis_a.vals[None, :] - basis_b.vals[:, None])
        vals_sqdiff /= vals_sqdiff.sum()
        return LBCommutativityEnforcing(vals_sqdiff, weight=weight)

    def __call__(self, fmap_matrix):
        """
        Compute the LB commutativity constraint

        Parameters
        ---------------------
        C      :
            (K2,K1) Functional map

        Returns
        ---------------------
        energy : float
            (float) LB commutativity squared norm
        """
        return self.weight * 0.5 * (np.square(fmap_matrix) * self.vals_sqdiff).sum()

    def gradient(self, fmap_matrix):
        """
        Compute the gradient of the LB commutativity constraint

        Parameters
        ---------------------
        C         :
            (K2,K1) Functional map

        Returns
        ---------------------
        gradient : np.ndarray
            (K2,K1) gradient of the LB commutativity squared norm
        """
        return self.weight * fmap_matrix * self.vals_sqdiff


class OperatorCommutativityEnforcing(WeightedFactor):
    """

    Parameters
    ----------
    op1 :
        (K1,K1) operator on first basis
    op2 :
        (K2,K2) descriptros on second basis
    """

    def __init__(self, oper_a, oper_b, weight=1.0):
        super().__init__(weight)
        self.oper_a = oper_a
        self.oper_b = oper_b

    def __new__(cls, oper_a, oper_b, weight=1.0):
        """

        Parameters
        ----------
        oper_a : array-like, shape=[..., K1, K1]
            operator on first basis
        oper_b : array-like, shape=[..., K2, K2]
            (K2,K2) descriptros on second basis
        """
        if oper_a.ndim > 2:
            factors = [
                OperatorCommutativityEnforcing(oper_a_, oper_b_)
                for oper_a_, oper_b_ in zip(oper_a, oper_b)
            ]
            return FactorSum(factors, weight=weight)

        return super().__new__(cls)

    @staticmethod
    def compute_multiplication_operator(basis, descr):
        """
        Compute the multiplication operators associated with the descriptors

        Parameters
        ----------
        descr : array-like, shape=[..., n_vertices]

        Returns
        -------
        operators : array-like, shape=[..., spectrum_size, spectrum_size]
        """
        pinv = basis.vecs.T @ basis.mass_matrix
        return pinv @ fmib.linalg.columnwise_scaling(descr, basis.vecs)

    @staticmethod
    def compute_orientation_operator(shape, descr, reversing=False, normalize=False):
        """
        Compute orientation preserving or reversing operators associated to each descriptor.

        Parameters
        ----------
        reversing : bool
            whether to return operators associated to orientation inversion instead
                    of orientation preservation (return the opposite of the second operator)
        normalize : bool
            whether to normalize the gradient on each face. Might improve results
                    according to the authors

        Returns
        -------
        list_op : list
            (n_descr,) where term i contains (D1,D2) respectively of size (k1,k1) and
            (k2,k2) which represent operators supposed to commute.
        """
        # Precompute the inverse of the eigenvectors matrix
        pinv = shape.basis.vecs.T @ shape.basis.mass_matrix  # (k1,n)

        # Compute the gradient of each descriptor
        grads = shape.face_valued_gradient(descr)
        if normalize:
            grads = fmib.linalg.normalize(grads)

        # Compute the operators in reduced basis
        sign = -1 if reversing else 1.0

        orients = shape.face_orientation_operator(grads)
        if descr.ndim > 1:
            return np.stack(
                [sign * pinv @ orient @ shape.basis.vecs for orient in orients]
            )

        return sign * pinv @ orients @ shape.basis.vecs

    @classmethod
    def from_multiplication(cls, basis_a, descr_a, basis_b, descr_b, weight=1.0):
        """

        Parameters
        ----------
        descr_a : array-like, shape=[..., n_vertices]
        descr_b : array-like, shape=[..., n_vertices]
        """
        oper_a = cls.compute_multiplication_operator(basis_a, descr_a)
        oper_b = cls.compute_multiplication_operator(basis_b, descr_b)
        return OperatorCommutativityEnforcing(oper_a, oper_b, weight=weight)

    @classmethod
    def from_orientation(
        cls,
        shape_a,
        descr_a,
        shape_b,
        descr_b,
        reversing_a=False,
        reversing_b=False,
        normalize=False,
        weight=1.0,
    ):
        """

        Parameters
        ----------
        descr_a : array-like, shape=[..., n_vertices]
        descr_b : array-like, shape=[..., n_vertices]
        """
        oper_a = cls.compute_orientation_operator(
            shape_a, descr_a, reversing=reversing_a, normalize=normalize
        )
        oper_b = cls.compute_orientation_operator(
            shape_b, descr_b, reversing=reversing_b, normalize=normalize
        )
        return OperatorCommutativityEnforcing(oper_a, oper_b, weight=weight)

    def __call__(self, fmap_matrix):
        """
        Compute the operator commutativity constraint.
        Can be used with descriptor multiplication operator

        Parameters
        ---------------------
        C   :
            (K2,K1) Functional map

        Returns
        ---------------------
        energy : float
            (float) operator commutativity squared norm
        """
        return (
            self.weight
            * 0.5
            * np.square(fmap_matrix @ self.oper_a - self.oper_b @ fmap_matrix).sum()
        )

    def gradient(self, fmap_matrix):
        """
        Compute the gradient of the operator commutativity constraint.
        Can be used with descriptor multiplication operator

        Parameters
        ---------------------
        C   :
            (K2,K1) Functional map

        Returns
        ---------------------
        gradient : np.ndarray
            (K2,K1) gradient of the operator commutativity squared norm
        """
        return self.weight * (
            self.oper_b.T @ (self.oper_b @ fmap_matrix - fmap_matrix @ self.oper_a)
            - (self.oper_b @ fmap_matrix - fmap_matrix @ self.oper_a) @ self.oper_a.T
        )


class FactorSum(WeightedFactor):
    def __init__(self, factors, weight=1.0):
        super().__init__(weight=weight)
        self.factors = factors

    def __call__(self, fmap_matrix):
        return self.weight * np.sum([factor(fmap_matrix) for factor in self.factors])

    def gradient(self, fmap_matrix):
        return self.weight * np.sum(
            [factor.gradient(fmap_matrix) for factor in self.factors], axis=0
        )
