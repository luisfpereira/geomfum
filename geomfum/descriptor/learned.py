"""Implementation of the learned descriptor.

The learned descriptor is a descriptor that uses a neural network to compute features.
"""

import abc

import geomstats.backend as gs
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

    def __call__(self, shape):
        """Compute descriptor.

        Parameters
        ----------
        shape : Shape.
            Shape.
        """
        if isinstance(shape, dict):
            features = self.feature_extractor(shape)
            features = features.transpose(2, 1).double()
        else:
            with torch.no_grad():
                features = self.feature_extractor(shape)
            features = gs.asarray(features.squeeze().T).double()

        return features

    def load_from_path(self, path):
        """Load model parameters from the provided file path.

        Args
        ----------
        path:  str
            Path to the model file.
        """
        self.feature_extractor.load_from_path(path)

    def save(self, path):
        """Save model parameters to the provided file path.

        Args
        -----------
        path:  str
            Path to save the model file.
        """
        self.feature_extractor.save(path)
