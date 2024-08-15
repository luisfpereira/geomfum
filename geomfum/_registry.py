import abc
import importlib
import inspect


class Registry(abc.ABC):
    @classmethod
    def register(cls, key, Obj, requires=()):
        """Register.

        Parameters
        ----------
        which : str
            Key.
        Obj : class
            Object to register.
        requires : str or tuple
            Required packages.
        """
        if isinstance(requires, str):
            requires = [requires]

        for package_name in requires:
            if importlib.util.find_spec(package_name) is None:
                missing_package = package_name
                break
        else:
            missing_package = None

        cls.MAP[key] = (Obj, missing_package)

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
        Obj, missing_package = cls.MAP[key]
        if missing_package:
            raise ModuleNotFoundError(missing_package)

        return Obj

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
    def register(cls, which, Obj, requires=()):
        """Register.

        Parameters
        ----------
        which : str
            Key.
        Obj : class
            Object to register.
        requires : str or tuple
            Required packages.
        """
        return super().register(which, Obj, requires)

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
    def register(cls, mesh, which, Obj, requires=()):
        """Register.

        Parameters
        ----------
        mesh : bool
            Whether a mesh or point cloud.
        which : str
            Key.
        Obj : class
            Object to register.
        requires : str or tuple
            Required packages.
        """
        return super().register((mesh, which), Obj, requires)

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
