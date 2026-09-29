"""RADIal CalibrationTable loader (torch): the azimuth beamformer of the production
signal-processing chain in dyrad/preprocessing/radial/preprocess.py, without importing it
(it pulls in DBReader).

The trainer builds its measured azimuth PSF (`Runner._az_response_table`) from the three
tensors of `DemuxOperator`: the per-azimuth steering matrix `calib_mat`, the virtual-array
Hamming taper `hamming` and the azimuth rows `az_bins_deg` (the table's 751 rows, sorted
-75..+75 deg; trainer azimuth grids snap onto these rows).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dyrad.constants import EL_IDX
from dyrad.preprocessing.radial import RADIAL_CALIB


class DemuxOperator:
    """The CalibrationTable's azimuth rows, steering matrix and taper, as torch tensors.

    The table (RADIAL_CALIB) holds the per-azimuth steering matrix "Signal", the Hamming
    taper "H" and "Azimuth_table"; the trainer constructs DemuxOperator without a path.
    """

    def __init__(self, calib_path: Path | str = RADIAL_CALIB, device: str = "cpu"):
        calib = np.load(calib_path, allow_pickle=True).item()
        self.az_bins_deg = torch.from_numpy(calib["Azimuth_table"].astype(np.float64))  # [A]
        self.calib_mat = torch.from_numpy(
            calib["Signal"][:, :, EL_IDX].astype(np.complex128)
        ).to(device)  # [A,192]
        self.hamming = torch.from_numpy(calib["H"][0].astype(np.float64)).to(device)  # [192]
