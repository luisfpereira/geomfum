import abc


class Basis(abc.ABC):
    pass


class EigenBasis(Basis):
    # TODO: add use_k and add full_vals, full_vecs (improve naming?)
    def __init__(self, vals, vecs):
        self.vals = vals
        self.vecs = vecs

    @property
    def spectrum_size(self):
        return len(self.vals)

    def truncate(self, spectrum_size):
        return EigenBasis(self.vals[:spectrum_size], self.vecs[:spectrum_size])


class LaplaceEigenBasis(EigenBasis):
    def __init__(self, vals, vecs, laplace_matrix, mass_matrix):
        super().__init__(vals, vecs)
        self.laplace_matrix = laplace_matrix
        self.mass_matrix = mass_matrix

    def truncate(self, spectrum_size):
        return LaplaceEigenBasis(
            self.vals[:spectrum_size],
            self.vecs[:spectrum_size],
            self.laplace_matrix,
            self.mass_matrix,
        )

    def project(self, array):
        """Project on the eigenbasis.

        Parameters
        ----------
        array : array-like, shape=[n_vertices, p]
            Array to project.

        Returns
        -------
        projected_func : array-like, shape=[spectrum_size, p]
            Projected array.
            (k,p) or (k,) projected function
        """
        return self.vecs.T @ (self.mass_matrix @ array)
