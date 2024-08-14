"""Functional map upsampling machinery."""

import abc

from geomfum.convert import FmFromP2pConverter, P2pFromFmConverter


class Upsampler(abc.ABC):
    """Functional map upsampler."""


class ZoomOut(Upsampler):
    """Zoomout algorithm.

    Parameters
    ----------
    nit : int
        Number of iterations.
    step : int or tuple[2, int]
        How much to increase each basis per iteration.
    p2p_from_fm_converter : P2pFromFmConverter
        Pointwise map from functional map.
    fm_from_p2p_converter : FmFromP2pConverter
        Functional map from pointwise map.

    References
    ----------
    .. [MRRSWO2019] Simone Melzi, Jing Ren, Emanuele Rodolà, Abhishek Sharma,
        Peter Wonka, and Maks Ovsjanikov. “ZoomOut: Spectral Upsampling
        for Efficient Shape Correspondence.” arXiv, September 12, 2019.
        http://arxiv.org/abs/1904.07865
    """

    def __init__(
        self,
        nit=None,
        step=1,
        p2p_from_fm_converter=None,
        fm_from_p2p_converter=None,
    ):
        if p2p_from_fm_converter is None:
            p2p_from_fm_converter = P2pFromFmConverter()

        if fm_from_p2p_converter is None:
            fm_from_p2p_converter = FmFromP2pConverter()

        self.nit = nit
        self.step = step
        self.p2p_from_fm_converter = p2p_from_fm_converter
        self.fm_from_p2p_converter = fm_from_p2p_converter

    @property
    def step(self):
        """How much to increase each basis per iteration.

        Returns
        -------
        step : tuple[2, int]
            Step.
        """
        return self._step_a, self._step_b

    @step.setter
    def step(self, step):
        """Set step.

        Parameters
        ----------
        step : int or tuple[2, int]
            How much to increase each basis per iteration.
        """
        if isinstance(step, int):
            self._step_a = self._step_b = step
        else:
            self._step_a, self._step_b = step

    def iter(self, fmap_matrix, basis_a, basis_b):
        """Upsampler iteration.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        basis_a : Eigenbasis.
            Basis.
        basis_b: Eigenbasis.
            Basis.

        Returns
        -------
        fmap_matrix : array-like, shape=[new_spectrum_size_b, new_spectrum_size_a]
            Upsampled functional map matrix.
        """
        k2, k1 = fmap_matrix.shape
        new_k1, new_k2 = k1 + self._step_a, k2 + self._step_b

        p2p_21 = self.p2p_from_fm_converter(fmap_matrix, basis_a, basis_b)
        return self.fm_from_p2p_converter(
            p2p_21, basis_a.truncate(new_k1), basis_b.truncate(new_k2)
        )

    def __call__(self, fmap_matrix, basis_a, basis_b):
        """Apply upsampler.

        Parameters
        ----------
        fmap_matrix : array-like, shape=[spectrum_size_b, spectrum_size_a]
            Functional map matrix.
        basis_a : Eigenbasis.
            Basis.
        basis_b: Eigenbasis.
            Basis.

        Returns
        -------
        fmap_matrix : array-like, shape=[new_spectrum_size_b, new_spectrum_size_a]
            Upsampled functional map matrix.
        """
        # TODO: make it general?
        k2, k1 = fmap_matrix.shape

        nit = self.nit
        if nit is None:
            nit = min(
                (k1 - basis_a.full_spectrum_size) // self._step_a,
                (k2 - basis_b.full_spectrum_size) // self._step_b,
            )
        else:
            msg = []
            if k1 + nit * self._step_a > basis_a.full_spectrum_size:
                msg.append("`basis_a`")
            if k2 + nit * self._step_b > basis_b.full_spectrum_size:
                msg.append("`basis_b`")

            if msg:
                raise ValueError(f"Not enough eigenvectors on {', '.join(msg)}.")

        nit = self.nit

        # TODO: bring subsampling

        new_fmap_matrix = fmap_matrix
        for _ in range(nit):
            new_fmap_matrix = self.iter(new_fmap_matrix, basis_a, basis_b)

        return new_fmap_matrix
