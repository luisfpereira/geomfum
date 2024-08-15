"""Spectral descriptors."""

from geomfum._registry import (
    HeatKernelSignatureRegistry,
    WaveKernelSignatureRegistry,
)


class HeatKernelSignature:
    """Heat kernel signature."""

    def __init__(self):
        raise ValueError(HeatKernelSignatureRegistry.only_from_registry())

    @staticmethod
    def from_registry(which="pyfm", **kwargs):
        """Instantiate registered implementation.

        Parameters
        ----------
        which : str
            A registered implementation.

        Returns
        -------
        obj : BaseHeatKernelSignature
            Instantiated object.
        """
        return HeatKernelSignatureRegistry.get(which)(**kwargs)


class WaveKernelSignature:
    """Wave kernel signature."""

    def __init__(self):
        raise ValueError(HeatKernelSignatureRegistry.from_wrap_msg())

    @staticmethod
    def from_registry(which="pyfm", **kwargs):
        """Instantiate registered implementation.

        Parameters
        ----------
        which : str
            A registered implementation.

        Returns
        -------
        obj : BaseWaveKernelSignature
            Instantiated object.
        """
        return WaveKernelSignatureRegistry.get(which)(**kwargs)
