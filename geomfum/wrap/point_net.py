"""Wrapper for the PointNet model.

References
----------
..PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation (https://arxiv.org/abs/1612.00593).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geomfum.descriptor import Descriptor


class PointNetDescriptor(Descriptor):
    """Descriptor representing the output of PointNet."""

    def __init__(
        self, n_features=128, device=torch.device("cpu"), feature_transform=False
    ):
        super(PointNetDescriptor, self).__init__()
        self.model = PointNet(k=n_features, feature_transform=feature_transform).to(
            device
        )
        self.n_features = n_features
        self.device = device

    def __call__(self, mesh):
        """Process the point cloud data using PointNet."""
        with torch.no_grad():
            point_cloud = torch.tensor(
                mesh.vertices, dtype=torch.float32, device=self.device
            )
            if point_cloud.ndimension() == 2:
                point_cloud = point_cloud.unsqueeze(0)
            self.features = self.model(point_cloud.transpose(2, 1))

        return self.features[0].T.cpu().numpy()

    def load_from_path(self, path):
        """Load model parameters from the provided file path.

        Parameters
        ----------
        path : String
            Path to the saved model parameters
        """
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {path}")
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")

    def load(self, premodel):
        """Load model parameters from a pre-trained model.

        Parameters
        ----------
        premodel: Dict
            State dictionary containing model parameters.
        """
        self.model.load_state_dict(premodel)

    def save(self, path):
        """Save model parameters to the specified file path.

        Parameters
        ----------
        path : String
            Path to save the model parameters
        """
        torch.save(self.model.state_dict(), path)


class PointNetfeat(nn.Module):
    """PointNet feature extraction module."""

    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNetfeat, self).__init__()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv41 = torch.nn.Conv1d(128, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 1024, 1)

        self.dense1 = torch.nn.Linear(1024, 256)
        self.dense2 = torch.nn.Linear(256, 256)

        self.global_feat = global_feat
        self.feature_transform = feature_transform

    def forward(self, x):
        """
        Forward pass of the PointNet feature extraction module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_features, num_points).

        Returns
        -------
        tuple
            A tuple containing:
            - torch.Tensor: Concatenated global and local features.
            - None: Placeholder for transformation matrix (not used).
            - None: Placeholder for feature transformation matrix (not used).
        """
        n_pts = x.size()[2]
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv4(x)))
        x = F.relu((self.conv41(x)))
        x = F.relu((self.conv5(x)))
        pointfeat = x
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = F.relu((self.dense1(x)))
        x = F.relu((self.dense2(x)))

        trans_feat = None
        trans = None
        x = x.view(-1, 256, 1).repeat(1, 1, n_pts)
        return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNet(nn.Module):
    """PointNet model for point cloud segementation."""

    def __init__(self, k=128, feature_transform=False):
        super(PointNet, self).__init__()
        self.k = k
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1280, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv2c = torch.nn.Conv1d(256, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, self.k, 1)
        self.m = nn.Dropout(p=0.3)

    def forward(self, x):
        """
        Forward pass of the PointNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_features, num_points).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, num_points, k).
        """
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        x = F.relu((self.conv2c(x)))
        x = self.m(x)
        x = self.conv3(x)
        x = x.transpose(2, 1).contiguous()
        x = x.view(batchsize, n_pts, self.k)
        return x
