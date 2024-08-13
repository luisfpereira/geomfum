import abc
import logging

from geomfum.convert import FmFromP2pConverter, P2pFromFmConverter


class Upsampler(abc.ABC):
    # TODO: add iter?
    pass


class ZoomOut(Upsampler):
    """Zoomout algorithm."""

    def __init__(
        self,
        n_iter=None,
        step=1,
        p2p_from_fm_converter=None,
        fm_from_p2p_converter=None,
    ):
        if p2p_from_fm_converter is None:
            p2p_from_fm_converter = P2pFromFmConverter()

        if fm_from_p2p_converter is None:
            fm_from_p2p_converter = FmFromP2pConverter(use_area=True)

        self.n_iter = n_iter
        self.p2p_from_fm_converter = p2p_from_fm_converter
        self.fm_from_p2p_converter = fm_from_p2p_converter

        # TODO: add step setter?
        if isinstance(step, int):
            step_a = step_b = step
        else:
            step_a, step_b = step

        self._step_a = step_a
        self._step_b = step_b

    def iter(self, fmap_matrix, basis_a, basis_b):
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
        """
        # TODO: can we make it general?
        k2, k1 = fmap_matrix.shape

        n_iter = self.n_iter
        if n_iter is None:
            n_iter = min(
                (k1 - basis_a.full_spectrum_size) // self._step_a,
                (k2 - basis_a.fullspectrum_size) // self._step_b,
            )
        else:
            msg = ""
            if k1 + n_iter * self._step_a > basis_a.full_spectrum_size:
                msg += "`basis_a`"
            if k2 + n_iter * self._step_b > basis_b.full_spectrum_size:
                msg += "`basis_b`"

            if msg:
                raise ValueError(f"Not enough eigenvectors on {msg.join(', ')}.")

        n_iter = self.n_iter

        # TODO: bring subsampling

        new_fmap_matrix = fmap_matrix
        for _ in range(n_iter):
            new_fmap_matrix = self.iter(new_fmap_matrix, basis_a, basis_b)

        return new_fmap_matrix
