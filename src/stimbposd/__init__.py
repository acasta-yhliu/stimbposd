__version__ = "0.1.0"

from typing import Dict
from sinter import Decoder

from stimbposd.bp_osd import BPOSD  # noqa: F401
from stimbposd.bp_lsd import BPLSD  # noqa: F401
from stimbposd.sinter_bp_osd import SinterDecoder_BPOSD  # noqa: F401
from stimbposd.sinter_bp_lsd import SinterDecoder_BPLSD  # noqa: F401
from stimbposd.dem_to_matrices import detector_error_model_to_check_matrices  # noqa: F401


def sinter_decoders() -> Dict[str, Decoder]:
    return {"bposd": SinterDecoder_BPOSD(), "bplsd": SinterDecoder_BPLSD()}
