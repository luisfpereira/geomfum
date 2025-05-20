"""
Wrap for PointNet feature extractor.

References
----------
    Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017). PointNet: Deep learning on point sets for 3D classification and segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 652-660).
    Qi, C. R., Su, H., Yi, L., & Guibas, L. J. (2017). PointNet++: Deep hierarchical feature learning on point sets in a metric space. In Advances in neural information processing systems (pp. 5098-5108).
    https://github.com/riccardomarin/Diff-FMaps by Riccardo Marin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from geomfum.descriptor.learned import BaseFeatureExtractor


class PointnetFeatureExtractor(BaseFeatureExtractor):
    """Feature extractor using PointNet architecture.

    Parameters
    ----------
    n_features : int
        Number of output features per point.
    conv_channels : list of int
        Channels for convolution layers in the feature extractor.
    mlp_dims : list of int
        Hidden dimensions for the global MLP.
    head_channels : list of int
        Convolutional layers in the head network.
    dropout : float
        Dropout probability.
    device : torch.device or str
        Device on which the model is allocated.
    """

    def __init__(
        self,
        n_features=128,
        conv_channels=[64, 64, 128],
        mlp_dims=[512, 256, 128],
        head_channels=[256, 128],
        dropout=0.3,
        device=None,
    ):
        self.device = device or torch.device("cpu")
        self.model = PointNet(
            conv_channels=conv_channels,
            mlp_dims=mlp_dims,
            head_channels=head_channels,
            out_features=n_features,
            dropout=dropout,
        ).to(self.device)

    def __call__(self, shape):
        """Extract point-wise features from a shape.

        Parameters
        ----------
        shape : object
            An object with a `vertices` attribute of shape (N, 3).

        Returns
        -------
        torch.Tensor
            Feature tensor of shape (1, N, n_features).
        """
        vertices = torch.tensor(shape.vertices, dtype=torch.float32).to(self.device)
        vertices = vertices.unsqueeze(0).transpose(2, 1).contiguous()
        features = self.model(vertices)
        return features

    def load_from_path(self, path):
        """Load model parameters from the provided file path.

        Args:
            path (str): Path to the saved model parameters
        """
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {path}")
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")

    def load(self, premodel):
        """Load model parameters from a pre-trained model.

        Args:
            premodel (dict): State dictionary containing model parameters
        """
        self.model.load_state_dict(premodel)

    def save(self, path):
        """Save model parameters to the specified file path.

        Args:
        path (str): Path to the saved model parameters
        """
        torch.save(self.model.state_dict(), path)


class PointNetfeat(nn.Module):
    """PointNet local and global feature extractor.

    Parameters
    ----------
    conv_channels : list of int
        List of output dimensions for each 1D convolution layer.
    mlp_dims : list of int
        List of hidden dimensions for the global MLP layers.
    """

    def __init__(
        self, conv_channels=[64, 64, 128, 128, 1024], mlp_dims=[1024, 256, 256]
    ):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        in_channels = 3

        for out_channels in conv_channels:
            self.conv_layers.append(nn.Conv1d(in_channels, out_channels, 1))
            in_channels = out_channels

        self.mlp = nn.Sequential(
            nn.Linear(conv_channels[-1], mlp_dims[0]),
            nn.ReLU(),
            nn.Linear(mlp_dims[0], mlp_dims[1]),
            nn.ReLU(),
            nn.Linear(mlp_dims[1], mlp_dims[2]),
            nn.ReLU(),
        )

        self.output_dim = mlp_dims[-1] + conv_channels[-1]

    def forward(self, x):
        """Forward pass of the PointNet feature extractor.

        Parameters
        ----------
        x : torch.Tensor
            Input point cloud of shape (B, 3, N), where B is the batch size and N is the number of points.

        Returns
        -------
        torch.Tensor
            Concatenated global and point-wise features of shape (B, output_dim, N).
        """
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        point_features = x
        global_features = torch.max(x, 2, keepdim=False)[0]
        global_features = self.mlp(global_features)
        global_features = global_features.unsqueeze(2).repeat(1, 1, x.shape[2])
        return torch.cat([global_features, point_features], dim=1)


class PointNet(nn.Module):
    """Full PointNet model with feature head.

    Parameters
    ----------
    feature_transform : bool
        Not used currently. Placeholder for future extension.
    conv_channels : list of int
        Output dimensions of initial PointNet convolution layers.
    mlp_dims : list of int
        Hidden dimensions of global MLP applied to global features.
    head_channels : list of int
        Output dimensions of the feature head layers.
    out_features : int
        Final number of output features per point.
    dropout : float
        Dropout rate applied before the final layer.
    """

    def __init__(
        self,
        feature_transform=False,
        conv_channels=[64, 64, 128, 128, 1024],
        mlp_dims=[1024, 256, 256],
        head_channels=[512, 256, 256],
        out_features=128,
        dropout=0.3,
    ):
        super().__init__()
        self.feat = PointNetfeat(conv_channels, mlp_dims)

        head = []
        in_ch = self.feat.output_dim
        for out_ch in head_channels:
            head.append(nn.Conv1d(in_ch, out_ch, 1))
            head.append(nn.ReLU())
            in_ch = out_ch

        self.head = nn.Sequential(*head)
        self.dropout = nn.Dropout(dropout)
        self.final_conv = nn.Conv1d(in_ch, out_features, 1)

    def forward(self, x):
        """Forward pass of the PointNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input point cloud of shape (B, 3, N).

        Returns
        -------
        torch.Tensor
            Per-point feature embeddings of shape (B, N, out_features).
        """
        x = self.feat(x)
        x = self.head(x)
        x = self.dropout(x)
        x = self.final_conv(x)
        return x.transpose(1, 2).contiguous()
