import abc

# TODO: add requires
import inspect


class Registry(abc.ABC):
    @classmethod
    def register(cls, key, Obj):
        """Register.

        Parameters
        ----------
        which : str
            Key.
        Obj : class
            Object to register.
        """
        cls.MAP[key] = Obj

    @classmethod
    def get(cls, key):
        """Get register object.

        Parameters
        ----------
        which : str
            Key.

        Returns
        -------
        Obj : class
            Registered object.
        """
        return cls.MAP[key]

    @classmethod
    def list_available(cls):
        """List register keys.

        Returns
        -------
        keys : list
            Registered keys.
        """
        return list(cls.MAP.keys())

    @classmethod
    def only_from_registry(cls):
        """Message for no internal implementation.

        Returns
        -------
        msg : str
            Message for no internal implementation with available
            implementations.
        """
        sign = str(inspect.signature(cls.get))[1:-1]
        return (
            f"No internal implementation. Use`.from_registry({sign}, **kwargs)`. "
            "Available implementations: "
            f"{', '.join([str(elem) for elem in cls.list_available()])}."
        )


class WhichRegistry(Registry, abc.ABC):
    @classmethod
    def register(cls, which, Obj):
        """Register.

        Parameters
        ----------
        which : str
            Key.
        Obj : class
            Object to register.
        """
        return super().register(which, Obj)

    @classmethod
    def get(cls, which):
        """Get register object.

        Parameters
        ----------
        which : str
            Key.

        Returns
        -------
        Obj : class
            Registered object.
        """
        return super().get(which)


class MeshWhichRegistry(Registry, abc.ABC):
    @classmethod
    def register(cls, mesh, which, Obj):
        """Register.

        Parameters
        ----------
        mesh : bool
            Whether a mesh or point cloud.
        which : str
            Key.
        Obj : class
            Object to register.
        """
        return super().register((mesh, which), Obj)

    @classmethod
    def get(cls, mesh, which):
        """Get register object.

        Parameters
        ----------
        mesh : bool
            Whether a mesh or point cloud.
        which : str
            Key.

        Returns
        -------
        Obj : class
            Registered object.
        """
        return super().get((mesh, which))


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
