"""Optimization of the functional map with a forward pass."""

import abc

import numpy as np


class ForwardFunctionalMap(abc.ABC):
    """Class for the forward pass of the functional map.

    Parameters
    ----------
    lmbda : FLoat.
        weight of the mask.
    resolvant_gamma: Float.
        resolvant of the regularized functional map.
    bijective: Bool.
        whether we compute the map inboth the directions.
    """

    def __init__(self, lmbda=1e3, resolvent_gamma=1, bijective=True):
        super(ForwardFunctionalMap, self).__init__()
        self.lmbda = lmbda
        self.resolvent_gamma = resolvent_gamma
        self.bijective = bijective

    def _compute_functional_map(self, sdescr_a, sdescr_b, mask):
        """Compute the functional map between two shapes.

        Args
        -------
        sdescr_a : array-like, shape=[..., spectrum_size_a]
            Spectral descriptors on first basis.
        sdescr_b : array-like, shape=[..., spectrum_size_b]
            Spectral descriptors on second basis.
        mask: array-like, shape=[..., spectrum_size_a, spectrum_size_b]
            Mask for the functional map.

        Returns
        -------
            fmap : array-like, shape=[..., spectrum_size_a, spectrum_size_b]
                Functional map from shape a to shape b.
        """
        AA_aa = sdescr_a.T @ sdescr_a  # [B, K, K]
        AA_ba = sdescr_b.T @ sdescr_a  # [B, K, K]

        C_i = []
        for i in range(mask.shape[0]):
            if self.lmbda == 0:
                C = np.linalg.inv(AA_aa) @ AA_ba[i, :].reshape(-1, 1)
            else:
                MASK_i = np.diag(mask[i, :].flatten())
                C = np.linalg.inv(AA_aa + self.lmbda * MASK_i) @ AA_ba[i, :].reshape(
                    -1, 1
                )
            C_i.append(C.T)

        Cab = np.concatenate(C_i, axis=0)
        return Cab

    def __call__(self, mesh_a, mesh_b, descr_a, descr_b):
        """Compute the functional map between two shapes.

        Args
        -------
        mesh_a : TriangleMesh or dict
        The first shape, either as a TriangleMesh object or a dictionary containing 'basis', 'evals', and 'pinv'.
        mesh_b : TriangleMesh or dict
        The second shape, either as a TriangleMesh object or a dictionary containing 'basis', 'evals', and 'pinv'.
        descr_a : array-like, shape=[D, ...]
        Spectral descriptors on the first shape.
        descr_b : array-like, shape=[D, ...]
        Spectral descriptors on the second shape.

        Returns
        -------
        fmap_12 : array-like, shape[spectrum_size_a, spectrum_size_b]
        Functional map from shape a to shape b.
        fmap_21: array-like, shape=[spectrum_size_b, spectrum_size_a] or None
        Functional map from shape b to shape a if bijective, otherwise None.
        """
        sdescr_a = mesh_a.basis.project(descr_a)
        sdescr_b = mesh_b.basis.project(descr_b)
        mask = self._compute_mask(
            mesh_a.basis.vals, mesh_b.basis.vals, self.resolvent_gamma
        )
        fmap_12 = self._compute_functional_map(sdescr_a, sdescr_b, mask)

        if self.bijective:
            mask = self._compute_mask(
                mesh_b.basis.vals, mesh_a.basis.vals, self.resolvent_gamma
            )
            fmap_21 = self._compute_functional_map(sdescr_b, sdescr_a, mask)
        else:
            fmap_21 = None
        return fmap_12, fmap_21

    def _compute_mask(self, evals_a, evals_b, resolvant_gamma):
        """Compute the mask for the functional map.

        Args
        -------
        evals_a : array-like, shape=[..., spectrum_size_a]
            Eigenvalues of the first shape.
        evals_b : array-like, shape=[..., spectrum_size_b]
            Eigenvalues of the second shape.
        resolvant_gamma : float
            Resolvent of the regularized functional map.

        Returns
        -------
        mask : array-like, shape=[..., spectrum_size_a, spectrum_size_b]
            Mask for the functional map.
        """
        scaling_factor = max(np.max(evals_a), np.max(evals_b))
        evals_a, evals_b = evals_a / scaling_factor, evals_b / scaling_factor
        evals_gamma_a = np.power(evals_a, resolvant_gamma)[np.newaxis, :]
        evals_gamma_b = np.power(evals_b, resolvant_gamma)[:, np.newaxis]

        M_re = evals_gamma_b / (np.square(evals_gamma_b) + 1) - evals_gamma_a / (
            np.square(evals_gamma_a) + 1
        )
        M_im = 1 / (np.square(evals_gamma_b) + 1) - 1 / (np.square(evals_gamma_a) + 1)
        return np.square(M_re) + np.square(M_im)
