import pyvista as pv
from polpo.preprocessing import Map
from polpo.preprocessing.mesh.conversion import DataFromPv
from polpo.preprocessing.mesh.decimation import PvDecimate
from polpo.preprocessing.mesh.io import PvReader

from geomfum.dataset import NotebooksDataset
from geomfum.shape import TriangleMesh


def MeshDecimationPipeline(target_reduction=0.95):
    """Create pipeline to load (decimated) meshes given filenames."""
    steps = (
        [
            PvReader(),
            lambda mesh: mesh
            if isinstance(mesh, pv.PolyData)
            else mesh.extract_surface(),
        ]
        + (
            [PvDecimate(target_reduction=target_reduction)]
            if target_reduction is not None
            else []
        )
        + [DataFromPv()]
    )
    return Map(steps, force_iter=True)


def get_meshes_from_indices(indices, target_reduction=0.95):
    dataset = NotebooksDataset()

    _filenames = [dataset.get_filename(index) for index in indices]

    return [
        TriangleMesh(vertices, faces)
        for vertices, faces in MeshDecimationPipeline(
            target_reduction=target_reduction
        )(_filenames)
    ]
