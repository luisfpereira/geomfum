import pyvista as pv


def to_pv_polydata(mesh):
    """Convert a TriangleMesh object to a PyVista PolyData object."""
    return pv.PolyData.from_regular_faces(points=mesh.vertices, faces=mesh.faces)
