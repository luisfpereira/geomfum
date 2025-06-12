"""Implementation of the learned descriptor.

The learned descriptor is a descriptor that uses a neural network to compute features.
"""

import abc

import torch

from geomfum._registry import FeatureExtractorRegistry, WhichRegistryMixins
from geomfum.descriptor._base import Descriptor


class BaseFeatureExtractor(abc.ABC):
    """Base class for feature extractor."""


class FeatureExtractor(WhichRegistryMixins):
    """Feature extractor."""

    _Registry = FeatureExtractorRegistry


class LearnedDescriptor(Descriptor, abc.ABC):
    """Learned descriptor.

    Parameters
    ----------
    feature_extractor: Feature Extractor
        Feature extractor to use.
    """

    def __init__(self, feature_extractor=None):
        super().__init__()
        self.feature_extractor = feature_extractor
        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor.from_registry(
                which="diffusionnet"
            )

    def __call__(self, shape):
        """Compute descriptor.

        Parameters
        ----------
        shape : Shape.
            Shape.
        """
        with torch.no_grad():
            features = self.feature_extractor(shape)
        features = features.squeeze().T.cpu().numpy()
        return features
