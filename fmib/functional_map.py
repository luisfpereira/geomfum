import abc

import numpy as np


class WeightedFactor(abc.ABC):
    """
    Parameters
    ----------
    fmap_shape : tuple(int, int)
        Assumes square if ``None``.
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

    def __init__(self, sdescr_a, sdescr_b, weight=1.0):
        super().__init__(weight)
        self.sdescr_a = sdescr_a
        self.sdescr_b = sdescr_b

    def __call__(self, fmap_matrix):
        """
        Compute the descriptor preservation constraint

        Parameters
        ---------------------
        fmap_matrix      :
            (K2,K1) Functional map
        descr1 :
            (K1,p) descriptors on first basis
        descr2 :
            (K2,p) descriptros on second basis

        Returns
        ---------------------
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
        ---------------------
        C      :
            (K2,K1) Functional map
        descr1 :
            (K1,p) descriptors on first basis
        descr2 :
            (K2,p) descriptros on second basis

        Returns
        ---------------------
        gradient : np.ndarray
            gradient of the descriptor preservation squared norm
        """
        return (
            self.weight
            * (fmap_matrix @ self.sdescr_a - self.sdescr_b)
            @ self.sdescr_a.T
        )


class CommutativityEnforcing(WeightedFactor):
    def __init__(self, vals_sqdiff, weight=1.0):
        super().__init__(weight)
        self.vals_sqdiff = vals_sqdiff

    @staticmethod
    def from_bases(basis_a, basis_b, weight=1.0):
        vals_sqdiff = np.square(basis_a.vals[None, :] - basis_b.vals[:, None])
        vals_sqdiff /= vals_sqdiff.sum()
        return CommutativityEnforcing(vals_sqdiff, weight=weight)

    def __call__(self, fmap_matrix):
        """
        Compute the LB commutativity constraint

        Parameters
        ---------------------
        C      :
            (K2,K1) Functional map
        ev_sqdiff :
            (K2,K1) [normalized] matrix of squared eigenvalue differences

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
        ev_sqdiff :
            (K2,K1) [normalized] matrix of squared eigenvalue differences

        Returns
        ---------------------
        gradient : np.ndarray
            (K2,K1) gradient of the LB commutativity squared norm
        """
        return self.weight * fmap_matrix * self.vals_sqdiff


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
