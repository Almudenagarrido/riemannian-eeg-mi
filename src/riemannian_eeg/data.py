"""Data loading and preprocessing."""

import warnings
from pathlib import Path
from moabb.utils import set_download_dir
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

# MOABB internally uses an MNE montage name deprecated in MNE 1.13.
warnings.filterwarnings("ignore", message="Montage name 'standard_1005' is deprecated")

# Download directory: <repo>/data
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(exist_ok=True)
set_download_dir(str(DATA_DIR))

# Dataset: BCI Competition IV 2a (MOABB: BNCI2014_001), a standard public
# benchmark for 4-class motor imagery.
#   - 9 subjects
#   - 2 sessions per subject (different days)
#   - 288 trials per session
#   - Balanced classes: left hand, right hand, feet, tongue
#   - 22 EEG + 3 EOG channels
#   - Sampled at 250 Hz
# Ref: Tangermann et al. (2012), "Review of the BCI Competition IV",
#      Frontiers in Neuroscience 6:55. doi:10.3389/fnins.2012.00055
DATASET = BNCI2014_001()

# Frequency band: 8-30 Hz
# At rest, the motor cortex shows synchronized oscillations in two rhythms:
#   - mu:   8-13 Hz
#   - beta: 13-30 Hz
# Imagining a movement activates the corresponding motor area, which makes
# these oscillations lose amplitude: Event-Related Desynchronization (ERD).
# This band keeps both rhythms and removes slow drifts (< 8 Hz) and
# muscle noise (> 30 Hz). Standard choice for CSP.
# To isolate the effect of the classifier (Euclidean vs Riemannian) and not
# of the preprocessing, the same fixed band is used for all subjects and all
# methods.
# Possible extension: subject-specific bands, e.g. a filter bank as in
# FBCSP (Ang et al., 2008), with bands selected on training data only.
# Refs: Pfurtscheller & Lopes da Silva (1999), Clin. Neurophysiol. 110(11),
#       doi:10.1016/S1388-2457(99)00141-8;
#       Guger, Ramoser & Pfurtscheller (2000), IEEE Trans. Rehabil. Eng. 8(4);
#       Ang et al. (2008), "Filter Bank Common Spatial Pattern (FBCSP) in
#       Brain-Computer Interface", IJCNN 2008.
FMIN, FMAX = 8, 30

# Paradigm: band-pass filtering + epoching in a single object.
# - Time window: MOABB default for this dataset, t = 2-6 s of each trial
#   (cue appears at t = 2 s, imagery lasts until t = 6 s) -> 4 s per trial.
#   The 0-2 s fixation period could serve as a reference to select
#   subject-specific bands (see the frequency band extension above).
# - Channels: only the 22 EEG channels are kept. The 3 EOG channels record
#   eye movements, not motor cortex activity; the dataset rules state they
#   are provided for artifact processing and must not be used for
#   classification.
#   Ref: Brunner et al. (2008), "BCI Competition 2008 - Graz data set A".
PARADIGM = MotorImagery(n_classes=4, fmin=FMIN, fmax=FMAX)

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

# Evaluation protocol: cross-session, i.e. the official BCI Competition IV split.
# - Train on session 1 ('0train') and test on session 2 ('1test'),
#   with one model per subject.
# - The two sessions were recorded on different days: cap placement,
#   electrode impedance and the subject's state change between them.
#   This is the realistic BCI use case (calibrate one day, use it another).
#   Randomly mixing trials from both days would make train and test too
#   similar and give optimistic results.
# - Riemannian methods are reported to be robust to this session-to-session
#   variability, which makes this protocol especially relevant to compare
#   them against the Euclidean baseline.
# - It also makes our results comparable with the competition and literature.
# - Validation data (only needed for SPDNet hyperparameters) must be taken
#   from the training session; the test session is only used at the end.
# Refs: Tangermann et al. (2012), Front. Neurosci. 6:55;
#       Congedo, Barachant & Bhatia (2017), "Riemannian geometry for
#       EEG-based brain-computer interfaces; a primer and a review",
#       Brain-Computer Interfaces 4(3).
TRAIN_SESSION, TEST_SESSION = "0train", "1test"


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