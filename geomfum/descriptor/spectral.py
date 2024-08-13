"""Spectral descriptors."""

from geomfum._registry import (
    HeatKernelSignatureRegistry,
    WaveKernelSignatureRegistry,
)


class HeatKernelSignature:
    """Heat kernel signature.

    Parameters
    ----------
    which : str
        One of: pyfm
    """

    def __new__(cls, which="pyfm", **kwargs):
        """Create new instance."""
        return HeatKernelSignatureRegistry.MAP[which](**kwargs)


class WaveKernelSignature:
    """Wave kernel signature.

    Parameters
    ----------
    which : str
        One of: pyfm
    """

    def __new__(cls, which="pyfm", **kwargs):
        """Create new instance."""
        return WaveKernelSignatureRegistry.MAP[which](**kwargs)
