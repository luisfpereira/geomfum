import abc


class BaseLaplacianFinder(abc.ABC):
    """Algorithm to find the Laplacian."""

    @abc.abstractmethod
    def __call__(self, shape):
        """Apply algorithm.

        Parameters
        ----------
        shape : Shape
            Shape.

        Returns
        -------
        laplace_matrix : array-like, shape=[n_vertices, n_vertices]
            Laplace matrix.
        mass_matrix : array-like, shape=[n_vertices, n_vertices]
            Diagonal lumped mass matrix.
        """
