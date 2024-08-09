"""Laplacian-related algorithms."""

from ._base import LaplacianFinder, LaplacianSpectrumFinder
from ._registry import register_laplacian_finder
from .mesh import (
    IglMeshLaplacianFinder,
    PyfmMeshLaplacianFinder,
    RobustMeshLaplacianFinder,
)
from .point_cloud import (
    RobustPointCloudLaplacianFinder,
)

register_laplacian_finder(True, "robust", RobustMeshLaplacianFinder)
register_laplacian_finder(True, "pyfm", PyfmMeshLaplacianFinder)
register_laplacian_finder(True, "igl", IglMeshLaplacianFinder)
register_laplacian_finder(False, "robust", RobustPointCloudLaplacianFinder)
