import numpy as np
# TODO: decide where its best to put this functions since they are related to the import of plotly and pyvista
import plotly.graph_objects as go
import pyvista as pv



def to_go_mesh3d(mesh):    
    """
    Convert a TriangleMesh object to a plotly Mesh3d object.
    """
    x, y, z = mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2]
    f1, f2, f3 = mesh.faces[:, 0], mesh.faces[:, 1], mesh.faces[:, 2]
    
    return go.Mesh3d(x=x, y=y, z=z, i=f1, j=f2, k=f3)


def to_pv_polydata(mesh):
    """
    Convert a TriangleMesh object to a PyVista PolyData object.
    """
    return pv.PolyData.from_regular_faces(points=mesh.vertices, faces=mesh.faces)