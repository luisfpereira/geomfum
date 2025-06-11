"""Trainer for Deep Functional Maps (DFM) using PyTorch."""

import logging

import torch
from tqdm import tqdm
import torch.nn as nn

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DeepFunctionalMapTrainer:
    """Trainer for Deep Functional Maps (DFM) using PyTorch.

    Parameters
    ----------
    model : nn.Module
        The model to be trained, typically a subclass of nn.Module.
    Loss : LossManager
        Loss manager that computes the loss during training.
    train_loader : DataLoader
        DataLoader for the training dataset.
    val_loader : DataLoader
        DataLoader for the validation dataset.
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters.
    epochs : int, optional
        Number of epochs to train the model (default is 100).
    device : str, optional
        Device to run the training on, either "cuda" or "cpu" (default is "cuda").
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        train_loss_manager=None,
        val_loss_manager=None,
        optimizer=None,
        epochs=100,
        device="cpu",
        checkpoint_path=None,
        monitor_metric="loss",
        mode="min",
    ):
        self.model = model
        self.train_loss_manager = train_loss_manager
        self.val_loss_manager = val_loss_manager

        if self.val_loss_manager is None:
            self.val_loss_manager = train_loss_manager

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=1e-3, weight_decay=1e-5
            )
        self.epochs = epochs
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.monitor_metric = monitor_metric
        self.mode = mode

    def train_one_epoch(self, epoch):
        """Train the model for one epoch."""
        self.model.train()
        running_loss = 0.0
        with tqdm(
            total=len(self.train_loader),
            desc=f"Epoch {epoch + 1}/{self.epochs} (Train)",
            unit="batch",
        ) as pbar:
            for batch_idx, pair in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                outputs = self.model(pair["source"], pair["target"])
                outputs.update({"source": pair["source"], "target": pair["target"]})
                loss, loss_dict = self.train_loss_manager.compute_loss(outputs)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                pbar.update(1)
        avg_train_loss = running_loss / len(self.train_loader)
        return avg_train_loss

    def validate(self, return_metrics=False):
        """Validate the model on the validation dataset."""
        self.model.eval()
        val_loss = 0.0
        metrics_sum = {}
        with torch.no_grad():
            with tqdm(
                total=len(self.val_loader), desc="Validation", unit="batch"
            ) as pbar:
                for batch_idx, pair in enumerate(self.val_loader):
                    outputs = self.model(pair["source"], pair["target"])
                    outputs.update({"source": pair["source"], "target": pair["target"]})
                    loss, loss_dict = self.val_loss_manager.compute_loss(outputs)
                    val_loss += loss.item()
                    for k, v in loss_dict.items():
                        metrics_sum[k] = metrics_sum.get(k, 0.0) + v
                    pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                    pbar.update(1)
        avg_val_loss = val_loss / len(self.val_loader)
        avg_metrics = {k: v / len(self.val_loader) for k, v in metrics_sum.items()}
        if return_metrics:
            return avg_val_loss, avg_metrics
        return avg_val_loss

    def train(self):
        """Train the model for the specified number of epochs."""
        best_metric = float("inf") if self.mode == "min" else -float("inf")
        for epoch in range(self.epochs):
            logging.info(f"Epoch [{epoch + 1}/{self.epochs}] - Training")
            avg_train_loss = self.train_one_epoch(epoch)
            logging.info(
                f"Epoch [{epoch + 1}/{self.epochs}] - Average Training Loss: {avg_train_loss:.4f}"
            )

            avg_val_loss, val_metrics = self.validate(return_metrics=True)
            logging.info(
                f"Epoch [{epoch + 1}/{self.epochs}] - Average Validation Loss: {avg_val_loss:.4f}"
            )

            metric_value = val_metrics.get(self.monitor_metric, avg_val_loss)
            improved = (
                metric_value < best_metric
                if self.mode == "min"
                else metric_value > best_metric
            )
            if self.checkpoint_path and improved:
                best_metric = metric_value
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "best_metric": best_metric,
                    },
                    self.checkpoint_path,
                )
                logging.info(
                    f"Checkpoint saved at epoch {epoch + 1} with {self.monitor_metric}: {metric_value:.4f}"
                )
