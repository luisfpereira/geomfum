
import plotly.graph_objects as go
from geomfum.plot import ShapePlotter

class PlotlyMeshPlotter(ShapePlotter):

    def plot(self, mesh):
        vertices=mesh.vertices
        faces=mesh.faces
        x, y, z = vertices[:,0], vertices[:,1], vertices[:,2]
        f1,f2,f3= faces[:,0], faces[:,1], faces[:,2]
        #project the error on the lbo basis
        fig = go.Figure(data=[go.Mesh3d(x=x,y=y,z=z, i=f1, j=f2, k=f3)])
        fig.show()

    def plot_function(self, mesh, function):
        #TODO: add parameters
        #TODO: add assertion
        

        vertices=mesh.vertices
        faces=mesh.faces
        x, y, z = vertices[:,0], vertices[:,1], vertices[:,2]
        f1,f2,f3= faces[:,0], faces[:,1], faces[:,2]
        fig = go.Figure(data=[go.Mesh3d(x=x,y=y,z=z, i=f1, j=f2, k=f3,
                                        intensity = function, 
                                        colorscale = 'viridis',   #TODO: MAKE THIS A PARAMETER
        )])
        fig.show()


