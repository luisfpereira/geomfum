import scipy


class ScipyEigsh:
    def __init__(
        self,
        spectrum_size=6,
        sigma=None,
        which="LM",
    ):
        self.spectrum_size = spectrum_size
        self.sigma = sigma
        self.which = which

    def __call__(self, A, M=None):
        return scipy.sparse.linalg.eigsh(
            A, k=self.spectrum_size, M=M, sigma=self.sigma, which=self.which
        )
