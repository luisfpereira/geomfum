"""Shape dataset for PyTorch."""

import itertools
import os
import random

import geomstats.backend as gs
import numpy as np
import scipy
import torch
from torch.utils.data import Dataset

import geomfum.backend as xgs
from geomfum.metric.mesh import ScipyGraphShortestPathMetric
from geomfum.shape.mesh import TriangleMesh


class ShapeDataset(Dataset):
    """ShapeDataset for loading and preprocessing shape data.

    Parameters
    ----------
    dataset_dir : str
        Path to the directory containing the dataset. We assume the dataset directory to have a subfolder shapes, for shapes, corr, for correspondences and dist, for chaced distance matrices.
    spectral : bool
        Whether to compute the spectral features.
    distances : bool
        Whether to compute geodesic distance matrices. For computational reasons, these are not computed on the fly, but rather loaded from a precomputed .mat file.
    k : int
        Number of eigenvectors to use for the spectral features.
    device : torch.device, optional
        Device to move the data to.
    """

    def __init__(
        self,
        dataset_dir,
        spectral=False,
        distances=False,
        k=200,
        device=None,
    ):
        self.dataset_dir = dataset_dir
        self.shape_dir = os.path.join(dataset_dir, "shapes")
        all_shape_files = sorted([f for f in os.listdir(self.shape_dir)])
        self.shape_files = all_shape_files

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
            base_name, _ = os.path.splitext(filename)

            # preprocess
            if spectral:
                mesh.laplacian.find_spectrum(spectrum_size=200, set_as_basis=True)
                mesh.basis.use_k = self.k
            if distances:
                mat_subfolder = os.path.join(self.dataset_dir, "dist")
                mat_filename = base_name + ".mat"
                mesh.mat_dist_path = os.path.join(mat_subfolder, mat_filename)

            self.meshes[filename] = mesh
            corr_filename = base_name + ".vts"

            if os.path.exists(os.path.join(self.dataset_dir, "corr", corr_filename)):
                self.corrs[filename] = (
                    np.loadtxt(
                        os.path.join(self.dataset_dir, "corr", corr_filename)
                    ).astype(np.int32)
                    - 1
                )
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
            "corr": self.corrs[filename],
        }
        if self.distances:
            dist_path = mesh.mat_dist_path
            if os.path.exists(dist_path):
                mat_contents = scipy.io.loadmat(dist_path)
                if "D" in mat_contents:
                    geod_distance_matrix = mat_contents["D"]
                else:
                    metric = ScipyGraphShortestPathMetric(mesh)
                    geod_distance_matrix = metric.dist_matrix()
                    mat_contents["D"] = gs.to_numpy(geod_distance_matrix)
                    os.makedirs(os.path.dirname(dist_path), exist_ok=True)
                    scipy.io.savemat(dist_path, mat_contents)
            else:
                metric = ScipyGraphShortestPathMetric(mesh)
                geod_distance_matrix = metric.dist_matrix()
                os.makedirs(os.path.dirname(dist_path), exist_ok=True)
                scipy.io.savemat(
                    dist_path,
                    {"D": gs.to_numpy(geod_distance_matrix)},
                )

            data.update({"dist_matrix": gs.array(geod_distance_matrix)})

        mesh.vertices = xgs.to_device(mesh.vertices, self.device)
        mesh.faces = xgs.to_device(mesh.faces, self.device)
        mesh.basis.full_vals = xgs.to_device(mesh.basis.full_vals, self.device)
        mesh.basis.full_vecs = xgs.to_device(mesh.basis.full_vecs, self.device)
        mesh.laplacian._mass_matrix = xgs.to_device(
            mesh.laplacian._mass_matrix, self.device
        )

        data.update({"mesh": mesh})

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
