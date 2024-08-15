"""Spectral descriptors."""

from geomfum._registry import (
    HeatKernelSignatureRegistry,
    WaveKernelSignatureRegistry,
    WhichRegistryMixins,
)


class HeatKernelSignature(WhichRegistryMixins):
    """Heat kernel signature."""

    _Registry = HeatKernelSignatureRegistry


class WaveKernelSignature(WhichRegistryMixins):
    """Wave kernel signature."""

    _Registry = WaveKernelSignatureRegistry
