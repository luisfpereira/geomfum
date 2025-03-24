import polyscope as ps
from geomfum.plot import ShapePlotter

class PolyscopeMeshPlotter(ShapePlotter):

    def __init__(self,colormap='viridis',name='Mymesh'):
        self.colormap = colormap
        self.fig = None
        self.name=name

    def plot(self, mesh):
        """
        Plot the mesh using Polyscope.
        """
        ps.init()
        ps.register_surface_mesh(self.name,mesh.vertices,mesh.faces)
        self.fig = ps
        return self.fig

    def plot_function(self, mesh, function):
        """
        Plot the mesh with a scalar function using Polyscope.
        """
        ps.init()
        ps.register_surface_mesh(
            self.name,
            mesh.vertices,
            mesh.faces,
        )
        ps.get_surface_mesh(self.name).add_scalar_quantity("function", function, defined_on='vertices', cmap=self.colormap, enabled=True)

        self.fig = ps
        return self.fig

    def show(self):
        """
        Show the Polyscope visualization.
        """
        self.fig.show()