import pathlib

from sinter import Decoder, CompiledDecoder
import numpy as np

import stim

from stimbposd.bp_lsd import BPLSD
from stimbposd.config import (
    DEFAULT_MAX_BP_ITERS,
    DEFAULT_BP_METHOD,
    DEFAULT_LSD_ORDER,
    DEFAULT_LSD_METHOD,
    DEFAULT_LSD_SCHEDULE,
)


class SinterCompiledDecoder_BPLSD(CompiledDecoder):
    def __init__(self, decoder: "BPLSD"):
        self.decoder = decoder

    def decode_shots_bit_packed(
        self,
        *,
        bit_packed_detection_event_data: "np.ndarray",
    ) -> "np.ndarray":
        return self.decoder.decode_batch(
            shots=bit_packed_detection_event_data,
            bit_packed_shots=True,
            bit_packed_predictions=True,
        )


class SinterDecoder_BPLSD(Decoder):
    def __init__(
        self,
        max_bp_iters: int = DEFAULT_MAX_BP_ITERS,
        bp_method: str = DEFAULT_BP_METHOD,
        lsd_order: int = DEFAULT_LSD_ORDER,
        lsd_method: str = DEFAULT_LSD_METHOD,
        schedule: str = DEFAULT_LSD_SCHEDULE,
        **bplsd_kwargs,
    ):
        f"""Class for decoding stim circuits with sinter using parallel belief propagation and ordered statistics decoding (BP+LSD).
        This class uses BP+LSD decoder as a subroutine. For more information on the options and 
        implementation of the BP+LSD subroutine, see the documentation of the LDPC library: https://roffe.eu/software/ldpc/index.html.
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
        self.max_bp_iters = max_bp_iters
        self.bp_method = bp_method
        self.lsd_order = lsd_order
        self.lsd_method = lsd_method
        self.schedule = schedule
        self.bplsd_kwargs = bplsd_kwargs

    def compile_decoder_for_dem(
        self, *, dem: stim.DetectorErrorModel
    ) -> CompiledDecoder:
        bposd = BPLSD(
            model=dem,
            max_bp_iters=self.max_bp_iters,
            bp_method=self.bp_method,
            osd_order=self.lsd_order,
            osd_method=self.lsd_method,
            schedule=self.schedule,
            **self.bplsd_kwargs,
        )
        return SinterCompiledDecoder_BPLSD(bposd)

    def decode_via_files(
        self,
        *,
        num_shots: int,
        num_dets: int,
        num_obs: int,
        dem_path: pathlib.Path,
        dets_b8_in_path: pathlib.Path,
        obs_predictions_b8_out_path: pathlib.Path,
        tmp_dir: pathlib.Path,
    ) -> None:
        """Performs decoding by reading problems from, and writing solutions to, file paths.
        Args:
            num_shots: The number of times the circuit was sampled. The number of problems
                to be solved.
            num_dets: The number of detectors in the circuit. The number of detection event
                bits in each shot.
            num_obs: The number of observables in the circuit. The number of predicted bits
                in each shot.
            dem_path: The file path where the detector error model should be read from,
                e.g. using `stim.DetectorErrorModel.from_file`. The error mechanisms
                specified by the detector error model should be used to configure the
                decoder.
            dets_b8_in_path: The file path that detection event data should be read from.
                Note that the file may be a named pipe instead of a fixed size object.
                The detection events will be in b8 format (see
                https://github.com/quantumlib/Stim/blob/main/doc/result_formats.md ). The
                number of detection events per shot is available via the `num_dets`
                argument or via the detector error model at `dem_path`.
            obs_predictions_b8_out_path: The file path that decoder predictions must be
                written to. The predictions must be written in b8 format (see
                https://github.com/quantumlib/Stim/blob/main/doc/result_formats.md ). The
                number of observables per shot is available via the `num_obs` argument or
                via the detector error model at `dem_path`.
            tmp_dir: Any temporary files generated by the decoder during its operation MUST
                be put into this directory. The reason for this requirement is because
                sinter is allowed to kill the decoding process without warning, without
                giving it time to clean up any temporary objects. All cleanup should be done
                via sinter deleting this directory after killing the decoder.
        """
        dem = stim.DetectorErrorModel.from_file(dem_path)
        bposd = BPLSD(
            model=dem,
            max_bp_iters=self.max_bp_iters,
            bp_method=self.bp_method,
            lsd_order=self.lsd_order,
            lsd_method=self.lsd_method,
            schedule=self.schedule,
            **self.bplsd_kwargs,
        )
        shots = stim.read_shot_data_file(
            path=dets_b8_in_path,
            format="b8",
            num_detectors=dem.num_detectors,
            bit_packed=False,
        )
        predictions = bposd.decode_batch(shots)
        stim.write_shot_data_file(
            data=predictions,
            path=obs_predictions_b8_out_path,
            format="b8",
            num_observables=dem.num_observables,
        )
