"""Implementation of the learned descriptor.

The learned descriptor is a descriptor that uses a neural network to compute features.
"""

import abc

import geomstats.backend as gs

from geomfum._registry import FeatureExtractorRegistry, WhichRegistryMixins
from geomfum.descriptor._base import Descriptor
import torch.nn as nn


class BaseFeatureExtractor(abc.ABC):
    """Base class for feature extractor."""


class FeatureExtractor(WhichRegistryMixins):
    """Feature extractor."""

    _Registry = FeatureExtractorRegistry


class LearnedDescriptor(Descriptor, abc.ABC, nn.Module):
    """Learned descriptor.

    Parameters
    ----------
    n_features : number of features
        Number of features to compute.
    feature_extractor: Fature Extractor
        Feature extractor to use.
    """

    def __init__(self, feature_extractor=None):
        super().__init__()
        self.feature_extractor = feature_extractor
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor.from_registry(
                which="diffusionnet"
            )

    def forward(self, shape):
        """Compute descriptor.

        Parameters
        ----------
        shape : Shape.
            Shape.
        """
        features = self.feature_extractor(shape)
        features = features.squeeze().T.double()

        return features
