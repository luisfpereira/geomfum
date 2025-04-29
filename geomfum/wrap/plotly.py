"""Wraps plotly functions."""

import plotly.graph_objects as go

from geomfum.plot import ShapePlotter
from geomfum.shape.convert import to_go_mesh3d

# TODO: add pointcloud alternatives/ enable to plot pointclouds


class PlotlyMeshPlotter(ShapePlotter):
    """Plotting object to display meshes."""

    # NB: for now assumes only one mesh is plotted

    def __init__(self, colormap="viridis"):
        self.colormap = colormap

        self._plotter = self.fig = go.Figure(
            data=[],
            layout=go.Layout(scene=dict(aspectmode="data")),
        )

    def add_mesh(self, mesh, **kwargs):
        """Add mesh to plot.

        Parameters
        ----------
        mesh : TriangleMesh
            Mesh to be plotted.
        """
        mesh3d = to_go_mesh3d(mesh)
        mesh3d.update(colorscale=self.colormap, **kwargs)

        self._plotter.update(data=[mesh3d])

        hover_text = [f"Index: {index}" for index in range(len(mesh.vertices))]
        self._plotter.data[0]["text"] = hover_text

        return self

    def set_vertex_scalars(self, scalars, name="scalars"):
        """Set vertex scalars on mesh.

        Parameters
        ----------
        scalars : array-like
            Value at each vertex.
        name : str
            Ignored.
        """
        data = self._plotter.data[0]
        data["intensity"] = scalars
        data["colorscale"] = self.colormap

        self._plotter.update(data=[data])

        return self

    def show(self):
        """Display plot."""
        self._plotter.show()
