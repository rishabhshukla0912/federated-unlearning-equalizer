"""Fixed settings used in the paper (Table III)."""
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

CLIENT_SNR_DB = {
    "clean": 30, "snr20": 20, "snr15": 15, "snr10": 10, "snr5": 5,
    "fading20": 20, "fading10": 10, "fading5": 5, "corrupted": 5,
}
ALL_CLIENTS = list(CLIENT_SNR_DB)
FORGET_CLIENT = "snr20"
RETAINED = [c for c in ALL_CLIENTS if c != FORGET_CLIENT]

NUM_SUBCARRIERS = 1024
WINDOW_SIZE = 9
CENTER = WINDOW_SIZE // 2
NUM_FEATURES = 6          # Re/Im of y, h and the MMSE estimate z
NUM_CLASSES = 256         # 256-QAM

# Frame partitions of each client file (Protocol v2, Sec. V-G of the paper).
PARTS = {
    "train": (0, 150),        # frames 0-149
    "val": (190, 40),         # frames 190-229, round selection and ascent guard
    "pilot": (230, 80),       # frames 230-309, seed-52 pilots only
    "test": (2960, 40),       # frames 2960-2999, final evaluation only
}
# The original runs used frames 150-189 as the test split.
ORIGINAL_TEST = (150, 40)

# Mild IQ imbalance on the forget client (Sec. V-C).
IQ_GAIN_DB = 1.5
IQ_PHASE_DEG = 8.0

# Gates and margins (Sec. V).
UTILITY_GATE_PCT = 12.83
SIGNAL_GATE_ASR_PCT = 90.0
EQUIV_DELTA_PP = 2.0
