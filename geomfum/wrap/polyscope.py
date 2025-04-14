
"""
This is the wrap to implement polyscope functions
"""
import polyscope as ps
from geomfum.plot import ShapePlotter

class PolyscopeMeshPlotter(ShapePlotter):

    def __init__(self,colormap='viridis',name='Mymesh'):
        self.colormap = colormap
        self.fig = None
        self.name=name

    def plot(self, mesh):
        ps.init()
        ps.register_surface_mesh(self.name,mesh.vertices,mesh.faces)
        self.fig = ps
        return self.fig

    def plot_function(self, mesh, function):
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
        self.fig.show()