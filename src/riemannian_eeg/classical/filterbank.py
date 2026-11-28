"""Filter bank tangent space: one covariance per frequency sub-band.

A single 8-30 Hz covariance sums the power of all frequencies, so a power
change confined to one sub-band (e.g. mu, 8-12 Hz) is diluted by the others.
Here each sub-band gets its own covariance, recentering and tangent space,
and the tangent vectors are concatenated, so the classifier can weight the
bands that matter for each subject.

Sub-bands are filtered by MOABB on the continuous signal, before epoching,
which avoids the edge artefacts of filtering short, already-cut epochs.
"""
from typing import Final, Self, TypeAlias

import numpy as np
from numpy.typing import NDArray
from moabb.paradigms import FilterBankMotorImagery
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline

from riemannian_eeg.classical.features import Labels, SessionRecentering, covariances
from riemannian_eeg.classical.pipelines import METRIC, make_lr_cv
from riemannian_eeg.data import DATASET, TEST_SESSION, TRAIN_SESSION

# (n_trials, n_channels, n_times, n_bands)
FilterBankEpochs: TypeAlias = NDArray[np.float64]

# Six 4 Hz sub-bands covering mu and beta (MOABB's default filter bank).
BANDS: Final[tuple[tuple[float, float], ...]] = (
    (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32),
)

FB_PARADIGM: Final[FilterBankMotorImagery] = FilterBankMotorImagery(
    n_classes=4, filters=[list(band) for band in BANDS]
)


def load_train_test_fb(
    subject: int,
) -> tuple[FilterBankEpochs, Labels, FilterBankEpochs, Labels]:
    """Load one subject filtered in every sub-band, split by session.

    Same trials, time window and sessions as ``riemannian_eeg.data.load_train_test``,
    with an extra last axis for the sub-band.

    Returns
    -------
    X_train, y_train, X_test, y_test
        Epochs of shape (n_trials, n_channels, n_times, n_bands) and their labels.
    """
    X, y, metadata = FB_PARADIGM.get_data(dataset=DATASET, subjects=[subject])
    is_train = (metadata["session"] == TRAIN_SESSION).to_numpy()
    is_test = (metadata["session"] == TEST_SESSION).to_numpy()
    return X[is_train], y[is_train], X[is_test], y[is_test]


class FilterBankTangentSpace(BaseEstimator, TransformerMixin):
    """Per-band covariance -> (recentering) -> tangent space, concatenated.

    Parameters
    ----------
    recenter : bool
        Recenter each session to the identity, band by band, before the
        tangent space (see ``SessionRecentering``).
    """

    def __init__(self, recenter: bool = True):
        self.recenter = recenter

    def _band_covariances(self, X: FilterBankEpochs, band: int) -> NDArray[np.float64]:
        C = covariances(X[..., band])
        return SessionRecentering().transform(C) if self.recenter else C

    def fit(self, X: FilterBankEpochs, y: Labels | None = None) -> Self:
        self.tangent_spaces_ = [
            TangentSpace(metric=METRIC).fit(self._band_covariances(X, b))
            for b in range(X.shape[-1])
        ]
        return self

    def transform(self, X: FilterBankEpochs) -> NDArray[np.float64]:
        """Return the concatenated tangent vectors, shape (n_trials, n_bands * 253)."""
        return np.concatenate(
            [ts.transform(self._band_covariances(X, b)) for b, ts in enumerate(self.tangent_spaces_)],
            axis=1,
        )


def make_fb_rc_ts_lrcv() -> Pipeline:
    """Filter bank + recentering + tangent space + cross-validated logistic regression."""
    return make_pipeline(FilterBankTangentSpace(recenter=True), make_lr_cv())


FB_PIPELINES = {
    "FB+RC+TS+LRCV": make_fb_rc_ts_lrcv,
}
