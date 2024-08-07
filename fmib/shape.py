import abc


class Shape(abc.ABC):
    def __init__(self):
        # TODO: create automated way for computing this?
        # TODO: should this be handled as e.g. laplacian.<>
        self.basis = None


class TriangleMesh(Shape):
    def __init__(self, vertices, faces):
        super().__init__()
        self.vertices = vertices
        self.faces = faces


class PointCloud(Shape):
    def __init__(self, points):
        super().__init__()
        self.points = points
