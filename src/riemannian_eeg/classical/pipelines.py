"""Classical Riemannian classifiers (MDM, tangent space), as scikit-learn pipelines on raw epochs.

Every pipeline starts from the epochs (n_trials, n_channels, n_times), so all
fitted state (class means, reference point of the tangent space) is learned in
``fit`` on the training session only and reused unchanged in ``predict``.

The ``RC`` variants first recenter each session to the identity with its own
(unlabelled) Riemannian mean, which needs the whole test session at once.
"""
from collections.abc import Callable

import numpy as np
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer

from riemannian_eeg.classical.features import SessionRecentering, covariances

# Affine-invariant Riemannian metric (AIRM) for distances, means and log maps.
METRIC = "riemann"


def _covariance_step() -> FunctionTransformer:
    return FunctionTransformer(covariances)


def make_lr_cv() -> LogisticRegressionCV:
    """Logistic regression whose regularisation C is chosen by 5-fold CV on the training data only."""
    return LogisticRegressionCV(Cs=np.logspace(-4, 2, 13), cv=5, scoring="accuracy", max_iter=5000)


def make_mdm() -> Pipeline:
    """Minimum Distance to Mean.

    Fits one Riemannian (Fréchet) mean per class and assigns each trial to the
    class whose mean is closest in Riemannian distance.
    """
    return make_pipeline(_covariance_step(), MDM(metric=METRIC))


def make_ts_lr() -> Pipeline:
    """Tangent space + logistic regression.

    Maps each covariance to the tangent space at the Riemannian mean of the
    training trials (253 features for 22 channels) and classifies the
    resulting vectors with a multinomial logistic regression.
    """
    return make_pipeline(
        _covariance_step(),
        TangentSpace(metric=METRIC),
        LogisticRegression(max_iter=1000),
    )


def make_rc_mdm() -> Pipeline:
    """MDM after recentering each session to the identity (see ``SessionRecentering``)."""
    return make_pipeline(_covariance_step(), SessionRecentering(), MDM(metric=METRIC))


def make_rc_ts_lr() -> Pipeline:
    """Tangent space + logistic regression after recentering each session."""
    return make_pipeline(
        _covariance_step(),
        SessionRecentering(),
        TangentSpace(metric=METRIC),
        LogisticRegression(max_iter=1000),
    )


def make_rc_ts_lrcv() -> Pipeline:
    """Recentering + tangent space + logistic regression with C tuned by CV on train."""
    return make_pipeline(
        _covariance_step(),
        SessionRecentering(),
        TangentSpace(metric=METRIC),
        make_lr_cv(),
    )


# Name -> factory. Factories (not fitted objects) so each subject gets a fresh model.
PIPELINES: dict[str, Callable[[], Pipeline]] = {
    "MDM": make_mdm,
    "TS+LR": make_ts_lr,
    "RC+MDM": make_rc_mdm,
    "RC+TS+LR": make_rc_ts_lr,
    "RC+TS+LRCV": make_rc_ts_lrcv,
}
