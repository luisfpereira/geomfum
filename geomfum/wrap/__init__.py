from geomfum._registry import (
    register_heat_kernel_signature,
    register_laplacian_finder,
    register_wave_kernel_signature,
)

from .igl import IglMeshLaplacianFinder
from .pyfm import (
    PyfmHeatKernelSignature,
    PyfmMeshLaplacianFinder,
    PyfmWaveKernelSignature,
)
from .robust_laplacian import RobustMeshLaplacianFinder, RobustPointCloudLaplacianFinder

register_laplacian_finder(True, "pyfm", PyfmMeshLaplacianFinder, requires="pyFM")
register_laplacian_finder(
    True, "robust", RobustMeshLaplacianFinder, requires="robust_laplacian"
)
register_laplacian_finder(True, "igl", IglMeshLaplacianFinder, requires="igl")
register_laplacian_finder(
    False, "robust", RobustPointCloudLaplacianFinder, requires="robust_laplacian"
)


register_heat_kernel_signature("pyfm", PyfmHeatKernelSignature, requires="pyFM")
register_wave_kernel_signature("pyfm", PyfmWaveKernelSignature, requires="pyFM")
