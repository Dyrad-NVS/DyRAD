"""RADIal preprocessing: raw recordings to RAD tensors, poses, ego velocity, and training windows.

The locations of the RADIal inputs outside a recording are defined here once.
"""

from dyrad.paths import ROOT

#: The RADIal repository clone (see the README).
RADIAL_REPO = ROOT / "data" / "radial_processed" / "RADIal_repo"
#: Import root of the repository's DBReader package (`from DBReader import SyncReader`).
RADIAL_DBREADER = RADIAL_REPO / "DBReader"
#: Beamforming calibration: per-azimuth steering matrix, Hamming taper and azimuth rows.
RADIAL_CALIB = RADIAL_REPO / "SignalProcessing" / "CalibrationTable.npy"
#: CAN database of the recordings (Vehicle_Speed, ID 0x3E9).
RADIAL_DBC = RADIAL_DBREADER / "examples" / "can_database.dbc"
#: The authors' object annotations of every recording.
RADIAL_LABELS = ROOT / "data" / "radial_raw" / "ready_to_use" / "labels_CVPR.csv"
