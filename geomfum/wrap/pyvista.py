"""
This is the wrap to implement pyvista functions
"""


import pyvista as pv
from geomfum.plot import ShapePlotter
import numpy as np
from geomfum.shape.convert import to_pv_polydata



class PyvistaMeshPlotter(ShapePlotter):

    def __init__(self, colormap='viridis'):
        self.colormap = colormap
        self.fig=None
    def plot(self, mesh):

        # here i call the plotter fig to be consistent with the plotly plotter
        mesh_polydata = to_pv_polydata(mesh)
        self.fig = pv.Plotter()
        self.fig.add_mesh(mesh_polydata, cmap=self.colormap, show_edges=False)
        return self.fig

    def show(self):
        self.fig.show()