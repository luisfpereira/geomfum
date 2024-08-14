"""Conversion between pointwise and functional maps."""

import scipy
from sklearn.neighbors import NearestNeighbors


class P2pFromFmConverter:
    """Pointwise map from functional map.

    Parameters
    ----------
    n_neighbors : int
         Ignored if ``neighbor_finder`` is not None.
    """

    def __init__(self, neighbor_finder=None, use_adjoint=False):
        if neighbor_finder is None:
            neighbor_finder = NearestNeighbors(
                n_neighbors=1, leaf_size=40, algorithm="kd_tree", n_jobs=1
            )
        if neighbor_finder.n_neighbors > 1:
            raise ValueError("Expects `n_neighors = 1`.")

        self.neighbor_finder = neighbor_finder
        self.use_adjoint = use_adjoint

    def __call__(self, fmap_matrix, basis_a, basis_b):
        """Convert functional map.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.

        Returns
        -------
        p2p : array-like, shape=[n_vertices_b]
        """
        k2, k1 = fmap_matrix.shape

        if self.use_adjoint:
            emb1 = basis_a.full_vecs[:, :k1]
            emb2 = basis_b.full_vecs[:, :k2] @ fmap_matrix

        else:
            emb1 = basis_a.full_vecs[:, :k1] @ fmap_matrix.T
            emb2 = basis_b.full_vecs[:, :k2]

        self.neighbor_finder.fit(emb1)
        _, p2p_21 = self.neighbor_finder.kneighbors(emb2)

        # TODO: check shape
        return p2p_21[:, 0]


class FmFromP2pConverter:
    """Functional map from pointwise map."""

    # TODO: add subsampling
    def __init__(self, use_area=False):
        self.use_area = use_area

    def __call__(self, p2p, basis_a, basis_b):
        """Convert point to point map.

        Parameters
        ----------
        p2p : array-like, shape=[n_vertices_b]

        Returns
        -------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        """
        evects1_pb = basis_a.vecs[p2p, :]

        if self.use_area:
            # TODO: give access to mass_matrix to basis?
            return basis_b.vecs.T @ (basis_b._shape.mass_matrix @ evects1_pb)

        return scipy.linalg.lstsq(basis_b.vecs, evects1_pb)[0]
