"""
Implementation of the DeltaConv feature extractor for 3D shapes.

References
----------
..DeltaConv: Anisotropic Operators for Geometric Deep Learning on Point Clouds, Ruben Wiersma, Ahmad Nasikun, Elmar Eisemann, and Klaus Hildebrandt. SIGGRAPH2022
"""

import torch
import torch.nn as nn

import geomfum.backend as xgs

import torch
from torch.nn import Sequential as Seq, Dropout, LeakyReLU, Linear
from torch_geometric.nn import global_max_pool
import torch.nn as nn
import torch.optim as optim

import deltaconv

# from deltaconv.models import DeltaNetBase
# from deltaconv.nn import MLP
import torch.nn.functional as F
from geomfum.descriptor.learned import BaseFeatureExtractor


class DeltaConvFeatureExtractor(BaseFeatureExtractor, nn.Module):
    """Feature extractor that uses DeltaConv for geometric deep learning on 3D point clouds.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels (e.g., 3 for xyz). Default is 3.
    out_channels : int
        Number of output feature channels. Default is 128.
    hidden_channels : int
        Number of hidden channels in the network. Default is 128.
    n_block : int
        Number of DiffusionNet blocks. Default is 4.
    """

    def __init__(self, in_channels=3, out_channels=128, hidden_channels=128, n_block=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_block = n_block
