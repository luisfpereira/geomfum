"""
This module contains the implementation of the learned descriptor.
The learned descriptor is a spectral descriptor that uses a neural network to compute features.
"""
import geomfum.wrap as _wrap
from geomfum.descriptor._base import Descriptor


from geomfum._registry import WhichRegistryMixins, FeatureExtractorRegistry
import abc
import torch


class BaseFeatureExtractor(abc.ABC):
    """Base class for feature extractor."""
    
        
class FeatureExtractor(WhichRegistryMixins):
    """Feature extractor"""
    _Registry = FeatureExtractorRegistry
    
class LearnedDescriptor( Descriptor, abc.ABC):
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

    def __call__(self, shape):
        """Compute descriptor.

        Parameters
        ----------
        shape : Shape.
            Shape.
        """
        with torch.no_grad():
            if self.feature_extractor is None:
                features = shape.vertices
                print("Warning: No feature extractor provided. Using vertices as features.")
            else:
                features = self.feature_extractor(shape)
        features = features.squeeze().T.cpu().numpy()
        return features

