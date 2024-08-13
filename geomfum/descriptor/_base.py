import abc


class Descriptor(abc.ABC):
    pass


class SpectralDescriptor(Descriptor, abc.ABC):
    """Spectral descriptor.

    Parameters
    ----------
    n_domain : int
        Number of domain points. Ignored if ``domain`` is not a callable.
    domain : callable or array-like, shape=[n_domain]
        Method to compute domain points (``f(basis, n_domain)``) or
        domain points.
    use_landmarks : bool
        Whether to use landmarks.
    """

    # TODO: make general implementation

    def __init__(self, n_domain, domain, use_landmarks=False):
        self.n_domain = n_domain
        self.domain = domain
        self.use_landmarks = use_landmarks

    @abc.abstractmethod
    def __call__(self, basis, domain=None):
        """Compute descriptor.

        Parameters
        ----------
        basis : Eigenbasis.
            Basis.
        domain : array-like, shape=[n_domain]
            Domain points for computation.
        """
