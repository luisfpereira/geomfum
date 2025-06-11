"""Losses for functional maps."""

import torch
import torch.nn as nn


class LossManager:
    """
    Manages a list of loss functions and their weights for model training.

    Parameters
    ----------
    losses : list of (nn.Module, float) or list of nn.Module
        List of (loss_module, weight) tuples, or just loss modules (weight=1.0).
    """

    def __init__(self, losses):
        self.losses = losses

    def compute_loss(self, outputs):
        """Compute the total loss and a dictionary of individual losses."""
        total_loss = 0
        loss_dict = {}
        for loss_fn in self.losses:
            # Get required input keys for this loss
            required_keys = getattr(loss_fn, "required_inputs", None)
            if required_keys is not None:
                args = [outputs[k] for k in required_keys]
                loss_value = loss_fn(*args)
            else:
                # fallback: pass the whole dict
                loss_value = loss_fn(outputs)
            name = loss_fn.__class__.__name__
            loss_dict[name] = loss_value.item()
            total_loss += loss_value
        return total_loss, loss_dict


######################LOSS IMPLEMENTATIONS ############################


class SquaredFrobeniusLoss(nn.Module):
    """
    Computes the mean squared Frobenius norm between two input tensors.

    Parameters
    ----------
    a : torch.Tensor
        First input tensor (vector or matrix).
    b : torch.Tensor
        Second input tensor (vector or matrix), must be broadcastable to the shape of `a`.

    Returns
    -------
    torch.Tensor
        Scalar tensor representing the mean squared Frobenius norm between `a` and `b`.
    """

    def forward(self, a, b):
        """Forward pass."""
        return torch.mean(torch.sum(torch.abs(a - b) ** 2, dim=(-2, -1)))


class OrthonormalityLoss(nn.Module):
    """
    Computes the orthonormality error of a functional map by measuring the mean squared Frobenius norm between C^T C and the identity matrix.

    Parameters
    ----------
    fmap12 : torch.Tensor
        Functional map tensor of shape (batch_size, dim_out, dim_in).

    Returns
    -------
    torch.Tensor
        Scalar tensor representing the weighted mean squared Frobenius norm between C^T C and the identity matrix.
    """

    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    required_inputs = ["fmap12", "fmap21"]

    def forward(self, fmap12, fmap21):
        """Forward pass."""
        metric = SquaredFrobeniusLoss()
        eye = (
            torch.eye(fmap12.shape[1], device=fmap12.device)
            .unsqueeze(0)
            .expand(fmap12.shape[0], -1, -1)
        )
        return self.weight * (
            metric(torch.bmm(fmap12.transpose(1, 2), fmap12), eye)
            + metric(torch.bmm(fmap21.transpose(1, 2), fmap21), eye)
        )


class BijectivityLoss(nn.Module):
    """
    Computes the bijectivity error of two functional maps by measuring the mean squared Frobenius norm between fmap12 fmap21 and the identity matrix, and between fmap21 fmap12 and the identity matrix.

    Parameters
    ----------
    fmap12 : torch.Tensor
        Functional map tensor from shape 1 to shape 2 of shape (batch_size, dim_out, dim_in).
    fmap21 : torch.Tensor
        Functional map tensor from shape 2 to shape 1 of shape (batch_size, dim_in, dim_out).

    Returns
    -------
    torch.Tensor
        Scalar tensor representing the weighted mean squared Frobenius norm between fmap12 fmap21 and the identity matrix, and between fmap21 fmap12 and the identity matrix.
    """

    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    required_inputs = ["fmap12", "fmap21"]

    def forward(self, fmap12, fmap21):
        """Forward pass."""
        metric = SquaredFrobeniusLoss()
        eye = (
            torch.eye(fmap12.shape[1], device=fmap12.device)
            .unsqueeze(0)
            .expand(fmap12.shape[0], -1, -1)
        )
        return self.weight * metric(torch.bmm(fmap12, fmap21), eye) + metric(
            torch.bmm(fmap21, fmap12), eye
        )


class LaplacianCommutativityLoss(nn.Module):
    """
    Computes the Laplacian commutativity error of a functional map by measuring the discrepancy between the action of the Laplacian eigenvalues and the functional map.

    Parameters
    ----------
    fmap12 : torch.Tensor
        Functional map tensor from source to target shape, of shape (batch_size, dim_out, dim_in).
    source : dict
        Dictionary containing source shape information, must include key "evals" with eigenvalues tensor of shape (batch_size, dim_in).
    target : dict
        Dictionary containing target shape information, must include key "evals" with eigenvalues tensor of shape (batch_size, dim_out).

    Returns
    -------
    torch.Tensor
        Scalar tensor representing the weighted squared Frobenius norm of the Laplacian commutativity error.
    """

    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    required_inputs = ["fmap12", "source", "target"]

    def forward(self, fmap12, source, target):
        """Forward pass."""
        metric = SquaredFrobeniusLoss()
        return self.weight * metric(
            torch.einsum("abc,ac->abc", fmap12, source["evals"]),
            torch.einsum("ab,abc->abc", target["evals"], fmap12),
        )


class Fmap_Supervision(nn.Module):
    """
    Computes the Laplacian commutativity error of a functional map by measuring the discrepancy between the action of the Laplacian eigenvalues and the functional map.

    Parameters
    ----------
    fmap12 : torch.Tensor
        Functional map tensor from source to target shape, of shape (batch_size, dim_out, dim_in).
    fmap12_sup : torch.Tensor
        Supervised functional map tensor from source to target shape, of shape (batch_size, dim_out, dim_in).

    Returns
    -------
    torch.Tensor
        Scalar tensor representing the weighted squared Frobenius norm of the Laplacian commutativity error.
    """

    def __init__(self, weight=1):
        super().__init__()
        self.weight = weight

    required_inputs = ["fmap12", "fmap12_sup"]

    def forward(self, fmap12, fmap12_sup):
        """Forward pass."""
        metric = SquaredFrobeniusLoss()
        return self.weight * metric(fmap12, fmap12_sup)


class GeodesicError(nn.Module):
    """Computes the accuracy of a correspondence by measuring the mean of the geodesic distancees between points of the predicted permuted target and the ground truth target."""

    def __init__(self):
        super().__init__()

    required_inputs = ["p2p21", "source", "target"]

    def _compute_geodesic_loss(self, p2p, source_dist, source_corr, target_corr):
        """Compute the geodesic loss for batched inputs."""
        return torch.mean(source_dist[p2p.long()[target_corr], source_corr])

    def forward(self, p2p21, source, target):
        """Forward pass."""
        if p2p21.ndim == 2:
            loss = 0
            for b in range(p2p21.shape[0]):
                p2p = p2p21[b]
                source_dist = source["distances"][b]
                target_corr = target["corr"][b]
                source_corr = source["corr"][b]
                loss += self._compute_geodesic_loss(
                    p2p, source_dist, source_corr, target_corr
                )
            loss /= p2p21.shape[0]
        else:
            loss = self._compute_geodesic_loss(
                p2p21, source["distances"], source["corr"], target["corr"]
            )

        return loss
