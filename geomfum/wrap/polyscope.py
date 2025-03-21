import polyscope as ps
from geomfum.plot import ShapePlotter

class PolyscopeMeshPlotter(ShapePlotter):

    def __init__(self,colormap='viridis',name='Mymesh'):
        self.colormap = colormap
        self.fig = None
        self.name=name
        ps.init()

    def plot(self, mesh):
        """
        Plot the mesh using Polyscope.
        """
        ps_mesh = ps.register_surface_mesh(self.name,mesh.vertices,mesh.faces)
        self.fig = ps
        return self.fig

    def plot_function(self, mesh, function):
        """
        Plot the mesh with a scalar function using Polyscope.
        """
        ps_mesh = ps.register_surface_mesh(
            self.name,
            mesh.vertices,
            mesh.faces,
        )
        ps_mesh.add_scalar_quantity("my_scalar", function, defined_on='vertices', cmap=self.colormap)

        self.fig = ps
        return self.fig

    def show(self):
        """
        Show the Polyscope visualization.
        """
        ps.show()