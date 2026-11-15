"""Data loading and preprocessing for BCI Competition IV 2a (MOABB: BNCI2014_001).

All design choices (band, time window, channels, evaluation protocol) are justified in docs/design_decisions.md.
"""

import warnings
from pathlib import Path

from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
from moabb.utils import set_download_dir

# Deprecation warning raised inside MOABB.
warnings.filterwarnings("ignore", message="Montage name 'standard_1005' is deprecated")

# Datasets are downloaded to <repo>/data (.gitignore)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(exist_ok=True)
set_download_dir(str(DATA_DIR))

# 9 subjects, 2 sessions of 288 trials, 4 classes, 22 EEG channels at 250 Hz.
DATASET = BNCI2014_001()

# Mu + beta band, where motor imagery causes ERD.
FMIN, FMAX = 8, 30

# Band-pass filter + epoching (t = 2-6 s) + EEG channels only.
PARADIGM = MotorImagery(n_classes=4, fmin=FMIN, fmax=FMAX)

# Cross-session protocol: train on day 1, test on day 2.
TRAIN_SESSION, TEST_SESSION = "0train", "1test"


def load_subject(subject):
    """Load the band-pass filtered epochs of one subject (both sessions).

    Parameters
    ----------
    subject : int
        Subject number, from 1 to 9.

    Returns
    -------
    X : ndarray, shape (n_trials, n_channels, n_times)
        Filtered EEG epochs: 22 channels, 4 s at 250 Hz.
    y : ndarray, shape (n_trials,)
        Class label of each trial: 'left_hand', 'right_hand', 'feet', 'tongue'.
    metadata : pandas.DataFrame
        One row per trial, with the subject, session and run it comes from.
    """
    X, y, metadata = PARADIGM.get_data(dataset=DATASET, subjects=[subject])
    return X, y, metadata


def load_train_test(subject):
    """Load one subject and split it into training and test sessions.

    Parameters
    ----------
    subject : int
        Subject number, from 1 to 9.

    Returns
    -------
    X_train, y_train : ndarray
        Epochs and labels from session 1 (training).
    X_test, y_test : ndarray
        Epochs and labels from session 2 (test).
    """
    X, y, metadata = load_subject(subject)
    is_train = (metadata["session"] == TRAIN_SESSION).to_numpy()
    is_test = (metadata["session"] == TEST_SESSION).to_numpy()
    return X[is_train], y[is_train], X[is_test], y[is_test]