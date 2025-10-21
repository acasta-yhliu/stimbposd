import numpy as np
from stimbposd.dem_to_matrices import detector_error_model_to_check_matrices

try:
    from ldpc.bplsd_decoder import BpLsdDecoder

    def create_decoder(pcm, priors, **kwargs):
        return BpLsdDecoder(pcm=pcm, error_channel=list(priors), **kwargs)
except ImportError:
    from ldpc.lsd import bplsd_decoder

    def create_decoder(pcm, priors, **kwargs):
        return bplsd_decoder(parity_check_matrix=pcm, channel_probs=priors, **kwargs)


import stim

from stimbposd.config import (
    DEFAULT_MAX_BP_ITERS,
    DEFAULT_BP_METHOD,
    DEFAULT_LSD_ORDER,
    DEFAULT_LSD_METHOD,
    DEFAULT_LSD_SCHEDULE,
)


class BPLSD:
    def __init__(
        self,
        model: stim.DetectorErrorModel,
        max_bp_iters: int = DEFAULT_MAX_BP_ITERS,
        bp_method: str = DEFAULT_BP_METHOD,
        lsd_order: int = DEFAULT_LSD_ORDER,
        lsd_method: str = DEFAULT_LSD_METHOD,
        schedule: str = DEFAULT_LSD_SCHEDULE,
        **bplsd_kwargs,
    ):
        f"""Class for decoding stim circuits using parallel belief propagation and ordered statistics decoding (BP+LSD).
        This class uses BP+LSD decoder as a subroutine. For more information on the options and 
        implementation of the BP+OSD subroutine, see the documentation of the LDPC library: https://roffe.eu/software/ldpc/index.html.
        Additional keyword arguments are passed to the ``bplsd_decoder`` class of the ldpc Python package.

        Parameters
        ----------
        model : stim.DetectorErrorModel
            The detector error model of the stim circuit to be decoded
        max_bp_iters : int, optional
            The maximum number of iterations of belief propagation to be used, by default {DEFAULT_MAX_BP_ITERS}
        bp_method : str, optional
            The BP method. Currently three methods are implemented: 1) "product_sum": product sum updates;
            2) "min_sum": min-sum updates; 3) "min_sum_log": min-sum log updates, by default {DEFAULT_BP_METHOD}
        lsd_order : int, optional
            The LSD order, by default {DEFAULT_LSD_ORDER}
        lsd_method : str, optional
            The LSD method. Currently three methods are available: 'LSD_0', 'LSD_E', 'LSD_CS', by default {DEFAULT_LSD_METHOD}
        schedule : str, optional
            The LSD schedule, by default {DEFAULT_LSD_SCHEDULE}
        """
        self._matrices = detector_error_model_to_check_matrices(
            model, allow_undecomposed_hyperedges=True
        )
        self.num_detectors = model.num_detectors
        self.num_errors = model.num_errors
        h_shape = self._matrices.check_matrix.shape
        max_lsd_order = h_shape[1] - h_shape[0]
        self._bplsd = create_decoder(
            pcm=self._matrices.check_matrix,
            max_iter=max_bp_iters,
            bp_method=bp_method,
            priors=self._matrices.priors,
            lsd_order=min(lsd_order, max_lsd_order),
            lsd_method=lsd_method,
            schedule=schedule,
            input_vector_type="syndrome",
            **bplsd_kwargs,
        )

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Decode the syndrome and return a prediction of which observables were flipped

        Parameters
        ----------
        syndrome : np.ndarray
            A single shot of syndrome data. This should be a binary array with a length equal to the
            number of detectors in the `stim.Circuit` or `stim.DetectorErrorModel`. E.g. the syndrome might be
            one row of shot data sampled from a `stim.CompiledDetectorSampler`.

        Returns
        -------
        np.ndarray
            A binary numpy array `predictions` which predicts which observables were flipped.
            Its length is equal to the number of observables in the `stim.Circuit` or `stim.DetectorErrorModel`.
            `predictions[i]` is 1 if the decoder predicts observable `i` was flipped and 0 otherwise.
        """
        corr = self._bplsd.decode(syndrome)
        return (self._matrices.observables_matrix @ corr) % 2

    def decode_batch(
        self,
        shots: np.ndarray,
        *,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
    ) -> np.ndarray:
        """
        Decode a batch of shots of syndrome data. This is just a helper method, equivalent to iterating over each
        shot and calling `BPOSD.decode` on it.

        Parameters
        ----------
        shots : np.ndarray
            A binary numpy array of dtype `np.uint8` or `bool` with shape `(num_shots, num_detectors)`, where
            here `num_shots` is the number of shots and `num_detectors` is the number of detectors in the `stim.Circuit` or `stim.DetectorErrorModel`.

        Returns
        -------
        np.ndarray
            A 2D numpy array `predictions` of dtype bool, where `predictions[i, :]` is the output of
            `self.decode(shots[i, :])`.
        """
        if bit_packed_shots:
            shots = np.unpackbits(shots, axis=1, bitorder="little")[
                :, : self.num_detectors
            ]
        predictions = np.zeros(
            (shots.shape[0], self._matrices.observables_matrix.shape[0]), dtype=bool
        )
        for i in range(shots.shape[0]):
            predictions[i, :] = self.decode(shots[i, :])
        if bit_packed_predictions:
            predictions = np.packbits(predictions, axis=1, bitorder="little")
        return predictions
