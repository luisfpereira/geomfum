class LaplacianFinderRegistry:
    """Laplacian finder registry."""

    MAP = {}

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
        cls.MAP[(mesh, which)] = Obj


register_laplacian_finder = LaplacianFinderRegistry.register
