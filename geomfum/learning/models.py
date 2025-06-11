"""Models for learning features for functional maps.

References
----------
.. "Deep Geometric Functional Maps: Robust Feature Learning for Shape Correspondence" by Nicolas Donati, Abhishek Sharma, Maks Ovsjanikov.
.. "Deep Functional Maps: Structured Prediction for Dense Shape Correspondence" by O. Litany, T. Remez, E. Rodola, A. Bronstein, M. Bronstein.
..
"""

import abc

import torch.nn as nn

from geomfum.convert import P2pFromFmConverter
from geomfum.descriptor.learned import FeatureExtractor, LearnedDescriptor
from geomfum.forward_functional_map import ForwardFunctionalMap
from geomfum.refine import ProperRefiner
import torch
from geomfum.basis import EigenBasis


class BaseModel(abc.ABC, nn.Module):
    """Base class for all models."""


class FMNet(BaseModel):
    """Functional Map Network Model.

    Parameters
    ----------
    feature_extractor : FeatureExtractor
        Feature extractor to use for the descriptors.
    fmap_module : ForwardFunctionalMap
        Functional map module to use for the forward pass.
    """

    def __init__(
        self,
        feature_extractor=FeatureExtractor.from_registry(which="diffusionnet"),
        fmap_module=ForwardFunctionalMap(),
        converter=P2pFromFmConverter(),
    ):
        super(FMNet, self).__init__()

        self.feature_extractor = feature_extractor
        self.descriptors_module = LearnedDescriptor(
            feature_extractor=self.feature_extractor
        )
        self.fmap_module = fmap_module
        self.converter = converter

    def forward(self, mesh_a, mesh_b):
        """Compute the functional map between two shapes.

        Args
        -------
        mesh_a : TriangleMesh or dict
            The first shape, either as a TriangleMesh object or a dictionary containing 'basis', 'evals', and 'pinv'.
        mesh_b : TriangleMesh or dict
            The second shape, either as a TriangleMesh object or a dictionary containing 'basis', 'evals', and 'pinv'.

        Returns
        -------
            fmap12 : array-like, shape=[..., spectrum_size_a, spectrum_size_b]
                Functional map from shape a to shape b.
            fmap21 : array-like, shape=[..., spectrum_size_b, spectrum_size_a]
                Functional map from shape b to shape a.
        """
        desc_a = self.descriptors_module(mesh_a)
        desc_b = self.descriptors_module(mesh_b)

        fmap12, fmap21 = self.fmap_module(mesh_a, mesh_b, desc_a, desc_b)

        if not self.training:
            if desc_a.dim() == 3:
                p2p12 = []
                p2p21 = []
                for i in range(desc_a.shape[0]):
                    basis_a = EigenBasis(
                        mesh_a["evals"][i].cpu(), mesh_a["evecs"][i].cpu()
                    )
                    basis_b = EigenBasis(
                        mesh_b["evals"][i].cpu(), mesh_b["evecs"][i].cpu()
                    )

                    p2p12.append(self.converter(fmap12[i].cpu(), basis_a, basis_b))
                    p2p21.append(self.converter(fmap21[i].cpu(), basis_b, basis_a))
                p2p12 = torch.stack(p2p12, dim=0)
                p2p21 = torch.stack(p2p21, dim=0)
            else:
                basis_a = (
                    EigenBasis(mesh_a["evals"], mesh_a["evecs"])
                    if isinstance(mesh_a, dict)
                    else mesh_a.basis
                )
                basis_b = (
                    EigenBasis(mesh_b["evals"], mesh_b["evecs"])
                    if isinstance(mesh_b, dict)
                    else mesh_b.basis
                )

                p2p21 = self.converter(fmap12, basis_a, basis_b)
                p2p12 = self.converter(fmap21, basis_b, basis_a)
            return {
                "fmap12": fmap12,
                "fmap21": fmap21,
                "p2p12": p2p12,
                "p2p21": p2p21,
            }
        else:
            return {"fmap12": fmap12, "fmap21": fmap21}


class ProperMapNet(BaseModel):
    """Proper Functional Map Network Model.

    Parameters
    ----------
    feature_extractor : FeatureExtractor
        Feature extractor to use for the descriptors.
    fmap_module : ForwardFunctionalMap
        Functional map module to use for the forward pass.
    """

    def __init__(
        self,
        feature_extractor=FeatureExtractor.from_registry(which="diffusionnet"),
        fmap_module=ForwardFunctionalMap(),
        refiner=ProperRefiner(),
    ):
        super(ProperMapNet, self).__init__()

        self.feature_extractor = feature_extractor
        self.descriptors_module = LearnedDescriptor(
            feature_extractor=self.feature_extractor
        )
        self.fmap_module = fmap_module
        self.refiner = refiner

    def forward(self, mesh_a, mesh_b):
        """Compute the functional map between two shapes.

        Args
        -------
        mesh_a : TriangleMesh or dict
            The first shape, either as a TriangleMesh object or a dictionary containing 'basis', 'evals', and 'pinv'.
        mesh_b : TriangleMesh or dict
            The second shape, either as a TriangleMesh object or a dictionary containing 'basis', 'evals', and 'pinv'.

        Returns
        -------
            fmap12 : array-like, shape=[..., spectrum_size_a, spectrum_size_b]
                Functional map from shape a to shape b.
            fmap21 : array-like, shape=[..., spectrum_size_b, spectrum_size_a]
                Functional map from shape b to shape a.
        """
        desc_a = self.descriptors_module(mesh_a)
        desc_b = self.descriptors_module(mesh_b)
        fmap12, fmap21 = self.fmap_module(mesh_a, mesh_b, desc_a, desc_b)

        fmap12 = self.refiner(fmap12, mesh_a.basis, mesh_b.basis)

        if self.fmap_module.bijective:
            fmap21 = self.refiner(fmap21, mesh_b.basis, mesh_a.basis)

        return {"fmap12": fmap12, "fmap21": fmap21}
