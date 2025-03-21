import polyscope as ps
from geomfum.plot import ShapePlotter

class PolyscopeMeshPlotter(ShapePlotter):

    def __init__(self, colormap='viridis'):
        self.colormap = colormap
        self.fig = None
        ps.init()

    def plot(self, mesh):
        """
        Plot the mesh using Polyscope.
        """
        ps_mesh = ps.register_surface_mesh(
            "Mesh",
            mesh.vertices,
            mesh.faces,
            color=(0.5, 0.5, 0.5),  # Default gray color
        )
        ps_mesh.set_color_map(self.colormap)
        self.fig = ps
        return self.fig

    def plot_function(self, mesh, function):
        """
        Plot the mesh with a scalar function using Polyscope.
        """
        ps_mesh = ps.register_surface_mesh(
            "Mesh with Function",
            mesh.vertices,
            mesh.faces,
            scalar_function=function,
        )
        ps_mesh.set_color_map(self.colormap)
        self.fig = ps
        return self.fig

    def show(self):
        """
        Show the Polyscope visualization.
        """
        ps.show()