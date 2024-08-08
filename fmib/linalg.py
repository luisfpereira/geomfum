import numpy as np


def _prefix_with_ellipsis(string):
    return f"...{string}"


def normalize(array, axis=-1):
    # TODO: handle norm zero?
    return array / np.linalg.norm(array, axis=1, keepdims=True)


def columnwise_scaling(vec, mat):
    """Columnwise scaling.

    Parameters
    ----------
    vec : array-like, shape=[..., n]
        Vector of scalings.
    mat :array-like, shape=[..., n, k]
        Matrix.

    Returns
    -------
    scaled_mat : array-like, shape=[..., n, k]
    """
    first_term = "n"
    second_term = "nk"
    rhs = "nk"

    if vec.ndim > 1:
        first_term = _prefix_with_ellipsis(first_term)
        rhs = _prefix_with_ellipsis(rhs)
    if mat.ndim > 2:
        second_term = _prefix_with_ellipsis(second_term)
        rhs = _prefix_with_ellipsis(rhs)

    return np.einsum(f"{first_term},{second_term}->{rhs}", vec, mat)
