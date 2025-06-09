"""Optimization of the functional map with a forward pass."""

import abc

import geomstats.backend as gs
import torch.nn as nn

import geomfum.backend as xgs
import geomfum.linalg as la
import torch


class ForwardFunctionalMap(abc.ABC, nn.Module):
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
        """Compute the functional map between two shapes, supporting batching."""
        if sdescr_a.ndim == 3:  # Batched
            AA_aa = sdescr_a.transpose(2, 1) @ sdescr_a  # [B, K, F]
            AA_ba = sdescr_b.transpose(2, 1) @ sdescr_a  # [B, K, F]
            C_i = []
            for i in range(mask.shape[1]):
                if self.lmbda == 0:
                    C = gs.linalg.inv(AA_aa) @ AA_ba @ AA_ba[:, [i], :].transpose(1, 2)
                else:
                    MASK_i = gs.concatenate(
                        [
                            xgs.diag(mask[bs, i, :].flatten()).unsqueeze(0)
                            for bs in range(mask.shape[0])
                        ],
                        0,
                    )

                    C = torch.bmm(
                        gs.linalg.inv(AA_aa + self.lmbda * MASK_i),
                        AA_ba[:, [i], :].transpose(2, 1),
                    )
                C_i.append(C.transpose(1, 2))
            Cab = gs.concatenate(C_i, axis=1)
        else:  # Not batched
            AA_aa = sdescr_a.T @ sdescr_a  # [B, K, K]
            AA_ba = sdescr_b.T @ sdescr_a  # [B, K, K]

            C_i = []
            for i in range(mask.shape[0]):
                if self.lmbda == 0:
                    C = gs.linalg.inv(AA_aa) @ AA_ba[i, :].reshape(-1, 1)
                else:
                    MASK_i = xgs.diag(mask[i, :].flatten())
                    C = gs.linalg.inv(AA_aa + self.lmbda * MASK_i) @ AA_ba[
                        i, :
                    ].reshape(-1, 1)
                C_i.append(C.T)

            Cab = gs.concatenate(C_i, axis=0)
            print(Cab.shape)
        return Cab

    def forward(self, mesh_a, mesh_b, descr_a, descr_b):
        """Compute the functional map between two shapes, supporting batching."""
        # Handle mesh_a
        if isinstance(mesh_a, dict):
            evals_a = mesh_a["evals"]
            sdescr_a = (mesh_a["pinv"] @ descr_a.transpose(2, 1)).transpose(2, 1)

        else:
            evals_a = mesh_a.basis.vals
            sdescr_a = mesh_a.basis.project(descr_a)

        # Handle mesh_b
        if isinstance(mesh_b, dict):
            evals_b = mesh_b["evals"]
            sdescr_b = (mesh_b["pinv"] @ descr_b.transpose(2, 1)).transpose(2, 1)
        else:
            evals_b = mesh_b.basis.vals
            sdescr_b = mesh_b.basis.project(descr_b)

        mask = self._compute_mask(evals_a, evals_b, self.resolvent_gamma)
        fmap_12 = self._compute_functional_map(sdescr_a, sdescr_b, mask)

        if self.bijective:
            mask = self._compute_mask(evals_b, evals_a, self.resolvent_gamma)
            fmap_21 = self._compute_functional_map(sdescr_b, sdescr_a, mask)
        else:
            fmap_21 = None
        return fmap_12, fmap_21

    def _compute_mask(self, evals_a, evals_b, resolvant_gamma):
        """Compute the mask for the functional map, supporting batching."""
        # evals_a: [B, Ka] or [Ka], evals_b: [B, Kb] or [Kb]
        # Output: [B, Ka, Kb] or [Ka, Kb]

        evals_a = gs.array(evals_a)
        evals_b = gs.array(evals_b)
        # Determine if batched
        if evals_a.ndim == 2 and evals_b.ndim == 2:
            # Batched
            mask = []
            for i in range(evals_a.shape[0]):
                # Not batched

                scaling_factor = max(max(evals_a[i]), max(evals_b[i]))
                evals_a[i], evals_b[i] = (
                    evals_a[i] / scaling_factor,
                    evals_b[i] / scaling_factor,
                )
                evals_gamma_a = gs.power(evals_a[i], resolvant_gamma)[None, :]
                evals_gamma_b = gs.power(evals_b[i], resolvant_gamma)[:, None]
                M_re = evals_gamma_b / (
                    xgs.square(evals_gamma_b) + 1
                ) - evals_gamma_a / (xgs.square(evals_gamma_a) + 1)
                M_im = 1 / (xgs.square(evals_gamma_b) + 1) - 1 / (
                    xgs.square(evals_gamma_a) + 1
                )
                mask.append(xgs.square(M_re) + xgs.square(M_im))
            mask = gs.stack(mask, axis=0)
            return mask
        else:
            # Not batched
            scaling_factor = max(max(evals_a), max(evals_b))
            evals_a, evals_b = evals_a / scaling_factor, evals_b / scaling_factor
            evals_gamma_a = gs.power(evals_a, resolvant_gamma)[None, :]
            evals_gamma_b = gs.power(evals_b, resolvant_gamma)[:, None]
            M_re = evals_gamma_b / (xgs.square(evals_gamma_b) + 1) - evals_gamma_a / (
                xgs.square(evals_gamma_a) + 1
            )
            M_im = 1 / (xgs.square(evals_gamma_b) + 1) - 1 / (
                xgs.square(evals_gamma_a) + 1
            )
            return xgs.square(M_re) + xgs.square(M_im)
