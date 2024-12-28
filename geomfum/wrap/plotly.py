
import plotly.graph_objects as go
from geomfum.plot import ShapePlotter
from geomfum.shape.convert import to_go_mesh3d


class PlotlyMeshPlotter(ShapePlotter):
    
    def __init__(self, colormap='viridis'):
        self.colormap = colormap
        self.fig = None

    def plot(self, mesh):

        mesh3d= to_go_mesh3d(mesh)
        #update adding plotter properties
        mesh3d.update(colorscale=self.colormap)
        self.fig = go.Figure(data=[mesh3d])

        return self.fig

    def plot_function(self, mesh, function):
        

        vertices=mesh.vertices
        faces=mesh.faces
        x, y, z = vertices[:,0], vertices[:,1], vertices[:,2]
        f1,f2,f3= faces[:,0], faces[:,1], faces[:,2]
        self.fig = go.Figure(data=[go.Mesh3d(x=x,y=y,z=z, i=f1, j=f2, k=f3,
                                        intensity = function, 
                                        colorscale  = self.colormap,   
        )])
        
        return self.fig

    def show(self):
        self.fig.show()    

