import abc
import geomstats.backend as gs


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


class DistanceFromLandmarksDescriptor(Descriptor):
    """Distance from landmarks descriptor. A simple descriptor that returns the distance from landmarks as a function on the shape."""

    def __call__(self, shape):
        """Compute descriptor.

        Parameters
        ----------
        shape : Shape.
            Shape.

        Returns
        -------
        descriptor : array-like, shape=[n_landmarks]
            Descriptor values.
        """
        if not hasattr(shape, "landmark_indices"):
            raise AttributeError(
                "shape object does not have 'landmark_indices' attribute"
            )

        if shape.metric is None:
            raise ValueError("shape is not equipped with metric")
        distances_list = shape.metric.dist_from_source(shape.landmark_indices)[0]
        distances = gs.stack(distances_list)
        return distances
