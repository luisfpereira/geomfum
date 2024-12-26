import pyvista as pv
from geomfum.plot import ShapePlotter
import numpy as np


class PyvistaMeshPlotter(ShapePlotter):

    def plot(self, vertices, faces):

        faces_formatted = np.hstack([[len(face), *face] for face in faces])
        mesh = pv.PolyData(vertices, faces_formatted)
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, show_edges=False)
        plotter.show()