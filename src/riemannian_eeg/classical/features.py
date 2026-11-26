"""Covariance features: each EEG trial becomes one SPD matrix.

A band-pass filtered, zero-mean trial carries its class information in the
power of each channel and in how channels co-vary, which is exactly what the
covariance matrix summarises.
"""
from typing import Self, TypeAlias

import numpy as np
from numpy.typing import NDArray
from pyriemann.geometry.base import invsqrtm
from pyriemann.geometry.mean import mean_riemann
from sklearn.base import BaseEstimator, TransformerMixin

# Array types used across the classical Riemannian package.
Epochs: TypeAlias = NDArray[np.float64]       # (n_trials, n_channels, n_times)
Labels: TypeAlias = NDArray[np.str_]          # (n_trials,)
CovMatrices: TypeAlias = NDArray[np.float64]  # (n_trials, n_channels, n_channels), SPD


def covariances(X: Epochs) -> CovMatrices:
    """Centre each trial and compute its sample covariance matrix.

    Each trial is centred with its own mean, so the transform uses no
    information from other trials (no train/test leakage).

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_times)
        EEG epochs.

    Returns
    -------
    C : ndarray, shape (n_trials, n_channels, n_channels)
        Sample covariance matrix of each trial, C = X Xᵀ / (T - 1).
    """
    X = X - X.mean(axis=2, keepdims=True)
    n_times = X.shape[2]
    return X @ np.transpose(X, axes=(0, 2, 1)) / (n_times - 1)


def check_spd(C: CovMatrices) -> dict[str, float | bool]:
    """Check that every matrix is symmetric positive definite.

    Parameters
    ----------
    C : ndarray, shape (n_trials, n_channels, n_channels)
        Covariance matrices.

    Returns
    -------
    report : dict
        ``symmetric`` (bool), ``min_eigenvalue`` (must be > 0) and
        ``max_condition_number`` (largest / smallest eigenvalue, worst trial).
    """
    eigvals = np.linalg.eigvalsh(C)  # (n_trials, n_channels), ascending
    return {
        "symmetric": bool(np.allclose(C, np.transpose(C, axes=(0, 2, 1)))),
        "min_eigenvalue": float(eigvals.min()),
        "max_condition_number": float((eigvals[:, -1] / eigvals[:, 0]).max()),
    }


class SessionRecentering(BaseEstimator, TransformerMixin):
    """Riemannian recentering of a whole session to the identity.

    Each call to ``transform`` computes the Riemannian mean M of the matrices
    it receives and maps every matrix C to M^{-1/2} C M^{-1/2}. By affine
    invariance this preserves all Riemannian distances within the session,
    but moves its centre to the identity, so sessions recorded on different
    days become aligned with each other.

    It is unsupervised (labels are never used) and stateless: ``fit`` learns
    nothing, so the test session is recentered with its own mean, not with
    the training one. It therefore assumes that ``transform`` receives one
    complete session at a time, which is how ``evaluate`` calls it.
    """

    def fit(self, C: CovMatrices, y: Labels | None = None) -> Self:
        return self

    def transform(self, C: CovMatrices) -> CovMatrices:
        M_isqrt = invsqrtm(mean_riemann(C))
        return M_isqrt @ C @ M_isqrt
