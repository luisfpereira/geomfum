"""Shape dataset for PyTorch."""

import hashlib
import itertools
import os
import random

import geomstats.backend as gs
import numpy as np
import scipy
import torch
from torch.utils.data import Dataset

import geomfum.backend as xgs
from geomfum.metric.mesh import HeatDistanceMetric
from geomfum.shape.mesh import TriangleMesh


def hash_arrays(arrs):
    """Compute a hash for a list of numpy arrays."""
    running_hash = hashlib.sha1()
    for arr in arrs:
        if arr is not None:
            arr = np.ascontiguousarray(arr)
            binarr = arr.view(np.uint8)
            running_hash.update(binarr)
    return running_hash.hexdigest()


class ShapeDataset(Dataset):
    """ShapeDataset for loading and preprocessing shape data.

    Parameters
    ----------
    shape_dir : str
        Path to the directory containing the shapes.
    spectral : bool
        Whether to compute the spectral features.
    distances : bool
        Whether to compute geodesic distance matrices.
    k : int
        Number of eigenvectors to use for the spectral features.
    device : torch.device, optional
        Device to move the data to.
    """

    def __init__(self, shape_dir, spectral=True, distances=False, k=200, device=None):
        self.shape_dir = shape_dir
        self.shape_files = sorted(
            [f for f in os.listdir(shape_dir) if f.endswith(".off")]
        )  # off but we can accept also other kind of files TODO: generalize

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.spectral = spectral
        self.k = k
        self.distances = distances
        # Preload meshes (or their important features) into memory
        self.meshes = {}
        self.corrs = {}
        for filename in self.shape_files:
            mesh = TriangleMesh.from_file(os.path.join(self.shape_dir, filename))
            # preprocess
            if spectral:
                mesh.laplacian.find_spectrum(spectrum_size=self.k, set_as_basis=True)
                mesh.basis.use_k = self.k
            if distances:
                mat_subfolder = os.path.join(self.shape_dir, "dist")
                mat_filename = filename.replace(".off", ".mat")
                mesh.mat_dist_path = os.path.join(mat_subfolder, mat_filename)

            self.meshes[filename] = mesh
            corr_filename = filename.replace(
                ".off", ".vts"
            )  # Assuming correspondence files are .txt
            if os.path.exists(os.path.join(self.shape_dir, "corr", corr_filename)):
                self.corrs[filename] = np.loadtxt(
                    os.path.join(self.shape_dir, "corr", corr_filename)
                ).astype(np.int32)
            else:
                self.corrs[filename] = np.arange(mesh.vertices.shape[0])

    def __getitem__(self, idx):
        """Retrieve a data sample by index.

        Parameters
        ----------
        idx : int
            Index of the item to retrieve.
        data : dict
            Dictionary containing mesh data, including vertices, faces, correspondence information,
            and optionally spectral features (eigenvalues, eigenvectors, pseudoinverse) and geodesic distances,
            depending on dataset configuration.
        """
        filename = self.shape_files[idx]
        mesh = self.meshes[filename]

        # the datas are stored in dictionaries
        data = {
            "vertices": xgs.to_torch(mesh.vertices).to(self.device),
            "faces": xgs.to_torch(mesh.faces).to(self.device),
            "corr": self.corrs[filename],
            "id": str(hash_arrays(arrs=(mesh.vertices, mesh.faces))),
        }
        if self.spectral:
            mesh.use_k = self.k
            data.update(
                {
                    "evals": xgs.to_torch(mesh.basis.vals).to(self.device),
                    "evecs": xgs.to_torch(mesh.basis.vecs).to(self.device),
                    "pinv": xgs.to_torch(mesh.basis.pinv).to(self.device),
                }
            )
        if self.distances:
            dist_path = mesh.mat_dist_path
            if os.path.exists(dist_path):
                mesh.geod_distance_matrix = scipy.io.loadmat(dist_path)["D"]
            else:
                metric = HeatDistanceMetric.from_registry(mesh, which="pp3d")
                mesh.geod_distance_matrix = metric.dist_matrix()
                os.makedirs(os.path.dirname(dist_path), exist_ok=True)
                scipy.io.savemat(
                    dist_path,
                    {"D": gs.to_numpy(mesh.geod_distance_matrix)},
                )
            data.update({"distances": xgs.to_torch(mesh.geod_distance_matrix)})

        return data

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.shape_files)


class PairsDataset(Dataset):
    """
    Dataset of pairs of shapes.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset or list
        Preloaded dataset or list of shape data objects.
    pair_mode : str, optional
        Strategy to generate pairs. Options: 'all', 'random'. Default is 'all'.
    n_pairs : int, optional
        Number of random pairs to generate if pair_mode is 'random'. Default is 100.
    device : torch.device, optional
        Device to move the data to. If None, uses CUDA if available, else CPU.
    """

    def __init__(self, dataset=None, pair_mode="all", n_pairs=100, device=None):
        # Preload meshes
        self.shape_data = dataset
        self.pair_mode = pair_mode
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Depending on pair_mode, choose the appropriate strategy
        if pair_mode == "all":
            self.pairs = self.generate_all_pairs()
        elif pair_mode == "random":
            self.pairs = self.generate_random_pairs(
                n_pairs
            )  # You can specify the number of pairs
        else:
            raise ValueError(f"Unsupported pair_mode: {pair_mode}")

    def generate_all_pairs(self):
        """Generate all possible pairs of shapes."""
        return list(itertools.permutations(range(self.shape_data.__len__()), 2))

    def generate_random_pairs(self, n_pairs=100):
        """Generate random pairs of shapes."""
        return random.sample(
            list(itertools.combinations(range(self.shape_data.__len__()), 2)), n_pairs
        )

    def generate_category_based_pairs(self, category_dict):
        """Generate pairs based on a specific category."""
        pairs = []
        for category, filenames in category_dict.items():
            pairs.extend(itertools.combinations(range(self.shape_data.__len__()), 2))
        return pairs

    def __getitem__(self, idx):
        """Get item by index.

        Parameters
        ----------
        idx : int
            Index of the item to retrieve.

        Returns
        -------
        data: dict
            Dictionary containing the source and target shapes.
        """
        src_idx, tgt_idx = self.pairs[idx]

        return {"source": self.shape_data[src_idx], "target": self.shape_data[tgt_idx]}

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.pairs)
