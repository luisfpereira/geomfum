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
    def __init__(self, shape, vals, vecs):
        self._shape = shape
        super().__init__(vals, vecs)

    def truncate(self, spectrum_size):
        return LaplaceEigenBasis(
            self._shape,
            self.vals[:spectrum_size],
            self.vecs[:spectrum_size],
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
        # TODO: make it proper
        return self.vecs.T @ (self._shape.mass_matrix @ array)
