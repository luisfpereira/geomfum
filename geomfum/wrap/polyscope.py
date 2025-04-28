"""Wraps polyscope functions."""

import polyscope as ps

from geomfum.plot import ShapePlotter


class PsMeshPlotter(ShapePlotter):
    """Plotting object to display meshes."""

    # NB: for now assumes only one mesh is plotted

    def __init__(self, colormap="viridis", backend=""):
        super().__init__()

        self.colormap = colormap

        self._plotter = ps
        self._name = "Mymesh"

        self._plotter.init(backend)

    def add_mesh(self, mesh):
        """Add mesh to plot.

        Parameters
        ----------
        mesh : TriangleMesh
            Mesh to be plotted.
        """
        self._plotter.register_surface_mesh(self._name, mesh.vertices, mesh.faces)
        return self

    def set_vertex_scalars(self, scalars, name="scalars"):
        """Set vertex scalars on mesh.

        Parameters
        ----------
        scalars : array-like
            Value at each vertex.
        name : str
            Scalar field name.
        """
        ps.get_surface_mesh(self._name).add_scalar_quantity(
            name,
            scalars,
            defined_on="vertices",
            cmap=self.colormap,
            enabled=True,
        )
        return self

    def show(self):
        """Display plot."""
        self._plotter.show()
