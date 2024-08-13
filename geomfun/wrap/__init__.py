from geomfun._registry import (
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

register_laplacian_finder(True, "pyfm", PyfmMeshLaplacianFinder)
register_laplacian_finder(True, "robust", RobustMeshLaplacianFinder)
register_laplacian_finder(True, "igl", IglMeshLaplacianFinder)
register_laplacian_finder(False, "robust", RobustPointCloudLaplacianFinder)


register_heat_kernel_signature("pyfm", PyfmHeatKernelSignature)
register_wave_kernel_signature("pyfm", PyfmWaveKernelSignature)
