"""Data loading and preprocessing."""

from pathlib import Path
from moabb.utils import set_download_dir
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery


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