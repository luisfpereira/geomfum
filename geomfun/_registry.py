import abc

# TODO: add requires


class WhichRegistry(abc.ABC):
    @classmethod
    def register(cls, which, Obj):
        """Register.

        Parameters
        ----------
        which : str
            One of: robust, pyfm
        Obj :
        """
        cls.MAP[which] = Obj


class MeshWhichRegistry(abc.ABC):
    @classmethod
    def register(cls, mesh, which, Obj):
        """Register.

        Parameters
        ----------
        mesh : bool
            If mesh or point cloud.
        which : str
            One of: robust, pyfm
        Obj : LaplacianFinder
        """
        # TODO: update doctrings
        cls.MAP[(mesh, which)] = Obj


class LaplacianFinderRegistry(MeshWhichRegistry):
    """Laplacian finder registry."""

    MAP = {}


register_laplacian_finder = LaplacianFinderRegistry.register


class HeatKernelSignatureRegistry(WhichRegistry):
    MAP = {}


register_heat_kernel_signature = HeatKernelSignatureRegistry.register


class WaveKernelSignatureRegistry(WhichRegistry):
    MAP = {}


register_wave_kernel_signature = WaveKernelSignatureRegistry.register
