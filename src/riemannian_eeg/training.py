"""How SPDNet is trained and evaluated on BCI IV 2a (train on day 1, test on day 2).

The network and its layers come from riemannian_eeg.spdnet. This module only adds the
experimental protocol:

    choose_epochs  ->  uses day 1 only: decides how many epochs to train
    final_test     ->  trains on all of day 1 for that many epochs, evaluates once on day 2

Options (see Config):
    recenter="session"  day 2 is re-centered with its own Riemannian mean instead of the mean of
                        day 1 (unsupervised adaptation: uses the day-2 matrices, never their labels)
    batchnorm=True      Riemannian batch norm after each BiMap (SPDBatchNorm, Brooks et al., 2019)
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.covariance import covariances
from pyriemann.utils.mean import mean_riemann
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from riemannian_eeg.data import load_train_test
from riemannian_eeg.spdnet import ReEig, SPDNet, StiefelParameter, StiefelSGD

DTYPE = torch.float64
N_CLASSES = 4


@dataclass
class Config:
    mode: str = "riemann"       # "riemann", or "euclid" for the ablation
    sizes: tuple = (16, 8)      # output size of each BiMap: 22 -> 16 -> 8
    eps: float = 1e-4           # ReEig threshold
    lr_stiefel: float = 0.1     # learning rate of the BiMap weights
    lr: float = 1e-2            # learning rate of the linear classifier (Adam)
    weight_decay: float = 1e-3  # penalty on large weights of the linear classifier
    epochs: int = 200           # maximum number of epochs
    batch_size: int = 32
    val_fraction: float = 0.2   # part of day 1 kept for validation
    recenter: str = "train"     # re-center day 2 with the mean of "train" (day 1) or its own "session"
    batchnorm: bool = False     # Riemannian batch norm after each BiMap


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_subject(subject):
    """Covariance matrices (OAS) and labels (0 to 3) of day 1 (train) and day 2 (test)."""
    X_train, y_train, X_test, y_test = load_train_test(subject)
    encoder = LabelEncoder().fit(y_train)  # 'feet', 'left_hand', ... -> 0, 1, 2, 3
    return (covariances(X_train.astype(np.float64), estimator="oas"), encoder.transform(y_train),
            covariances(X_test.astype(np.float64), estimator="oas"), encoder.transform(y_test))


def recenter(C, M):
    """C_i -> M^{-1/2} C_i M^{-1/2} for every trial: M is moved to the identity."""
    M_isqrt = invsqrtm(M)
    return M_isqrt @ C @ M_isqrt


def to_torch(C, y=None):
    C = torch.tensor(C, dtype=DTYPE)
    return C if y is None else (C, torch.tensor(y, dtype=torch.long))


# ---------------------------------------------------------------------------
# Model and optimisers
# ---------------------------------------------------------------------------

def build_model(cfg, n_channels):
    """SPDNet: [BiMap -> (SPDBatchNorm) -> ReEig] x len(sizes) -> LogEig -> Linear."""
    return SPDNet([n_channels, *cfg.sizes], n_classes=N_CLASSES, batchnorm=cfg.batchnorm,
                  eps=cfg.eps)


def stiefel_weights(model):
    return [p for p in model.parameters() if isinstance(p, StiefelParameter)]


def other_weights(model):
    return [p for p in model.parameters() if not isinstance(p, StiefelParameter)]


def build_optimisers(model, cfg):
    """Adam for the linear classifier, and for the BiMap weights:

    riemann: StiefelSGD without momentum (projection + step + QR retraction: W stays on Stiefel)
    euclid:  plain SGD, W <- W - lr * gradient (ablation: W can leave the Stiefel manifold)
    """
    adam = torch.optim.Adam(other_weights(model), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.mode == "riemann":
        bimap = StiefelSGD(stiefel_weights(model), lr=cfg.lr_stiefel, momentum=0.0)
    elif cfg.mode == "euclid":
        bimap = torch.optim.SGD(stiefel_weights(model), lr=cfg.lr_stiefel, momentum=0.0)
    else:
        raise ValueError(f"unknown mode {cfg.mode!r}")
    return [adam, bimap]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@torch.no_grad()
def loss_and_accuracy(model, C, y):
    model.eval()  # batch norm uses its running mean (no effect without batch norm)
    scores = model(C)
    loss = F.cross_entropy(scores, y).item()
    accuracy = (scores.argmax(dim=1) == y).double().mean().item()
    return loss, accuracy


def train(C, y, cfg, seed, epochs, C_val=None, y_val=None):
    """Train a new SPDNet on (C, y) for `epochs` epochs.

    If validation data are given, the validation loss is measured after every epoch.
    Returns the trained model and a table with one row per epoch.
    """
    torch.manual_seed(seed)  # same seed -> same initial weights (riemann and euclid start equal)
    model = build_model(cfg, n_channels=C.shape[-1])
    optimisers = build_optimisers(model, cfg)
    shuffler = torch.Generator().manual_seed(seed)  # same seed -> same mini-batches

    history = []
    for epoch in range(1, epochs + 1):
        model.train()  # batch norm uses the mean of each mini-batch (set back by loss_and_accuracy)
        batches = torch.randperm(len(y), generator=shuffler).split(cfg.batch_size)
        for batch in batches:
            model.zero_grad()
            loss = F.cross_entropy(model(C[batch]), y[batch])
            loss.backward()
            for optimiser in optimisers:
                optimiser.step()

        row = {"epoch": epoch}
        row["train_loss"], row["train_acc"] = loss_and_accuracy(model, C, y)
        if C_val is not None:
            row["val_loss"], row["val_acc"] = loss_and_accuracy(model, C_val, y_val)
        history.append(row)

        if not math.isfinite(row["train_loss"]):  # training diverged (possible in the ablation)
            break

    return model, pd.DataFrame(history)


# ---------------------------------------------------------------------------
# Experiment protocol
# ---------------------------------------------------------------------------

def choose_epochs(C_train, y_train, cfg, seed):
    """Decide how long to train, using day 1 only.

    1. Split day 1: 80 % fit / 20 % validation, with the same class proportions in both.
    2. Re-center both parts with the Riemannian mean of the fit part only
       (the validation part must look like unseen data).
    3. Train on the fit part, measuring the validation loss after every epoch.
    4. Best epoch = lowest validation loss.
    """
    fit, val = train_test_split(np.arange(len(y_train)), test_size=cfg.val_fraction,
                                stratify=y_train, random_state=seed)
    M = mean_riemann(C_train[fit])
    C_fit, y_fit = to_torch(recenter(C_train[fit], M), y_train[fit])
    C_val, y_val = to_torch(recenter(C_train[val], M), y_train[val])

    _, history = train(C_fit, y_fit, cfg, seed, cfg.epochs, C_val, y_val)

    finite = history[np.isfinite(history["val_loss"])]
    best_epoch = int(finite.loc[finite["val_loss"].idxmin(), "epoch"]) if len(finite) else 1
    return best_epoch, history


def final_test(C_train, y_train, C_test, y_test, cfg, seed, epochs):
    """Train on all of day 1 for `epochs` epochs, then evaluate once on day 2.

    Day 1 is re-centered with its own Riemannian mean. Day 2 is re-centered with the mean of
    day 1 (cfg.recenter == "train") or with its own mean (cfg.recenter == "session").
    The labels of day 2 are only used here, to compute the final accuracy.
    """
    M_train = mean_riemann(C_train)
    M_test = mean_riemann(C_test) if cfg.recenter == "session" else M_train
    C_tr, y_tr = to_torch(recenter(C_train, M_train), y_train)
    C_te = to_torch(recenter(C_test, M_test))

    model, _ = train(C_tr, y_tr, cfg, seed, epochs)

    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(C_te), dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(np.mean(predictions == y_test)),
        "kappa": float(cohen_kappa_score(y_test, predictions)),
        "probabilities": probabilities,  # (n_test_trials, 4), for the ensemble over seeds
        "reeig_activity": reeig_activity(model, C_tr),
        "orthogonality_error": orthogonality_error(model),
    }


# ---------------------------------------------------------------------------
# Diagnostics saved with the results
# ---------------------------------------------------------------------------

@torch.no_grad()
def reeig_activity(model, C):
    """Fraction of eigenvalues below eps (i.e. changed by ReEig) at each ReEig layer.

    Close to 0: ReEig almost never acts and the network is almost linear.
    """
    model.eval()
    fractions = []
    X = C
    for layer in model.spd_layers:
        if isinstance(layer, ReEig):
            eigenvalues = torch.linalg.eigvalsh(X)
            fractions.append((eigenvalues < layer.eps).double().mean().item())
        X = layer(X)
    return fractions


@torch.no_grad()
def orthogonality_error(model):
    """Largest entry of |W^T W - I| over the BiMap layers: 0 if W is on the Stiefel manifold."""
    errors = []
    for W in stiefel_weights(model):
        I = torch.eye(W.shape[1], dtype=W.dtype)
        errors.append((W.T @ W - I).abs().max().item())
    return max(errors)
