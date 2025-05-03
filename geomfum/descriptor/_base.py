import abc


class Descriptor(abc.ABC):
    pass


class SpectralDescriptor(Descriptor, abc.ABC):
    """Spectral descriptor.

    Parameters
    ----------
    domain : callable or array-like, shape=[n_domain]
        Method to compute domain points (``f(basis, n_domain)``) or
        domain points.
    use_landmarks : bool
        Whether to use landmarks.
    """

    def __init__(self, domain, use_landmarks=False):
        super().__init__()
        self.domain = domain
        self.use_landmarks = use_landmarks

    @abc.abstractmethod
    def __call__(self, shape, domain=None):
        """Compute descriptor.

        Parameters
        ----------
        shape : Shape.
            Shape.
        domain : array-like, shape=[n_domain]
            Domain points for computation.
        """
