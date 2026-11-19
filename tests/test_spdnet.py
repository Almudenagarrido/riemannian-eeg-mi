"""Checks that the SPDNet layers respect the geometry of the SPD manifold."""

import pytest
import torch
from torch import nn

from riemannian_eeg.spdnet import (
    BiMap,
    LogEig,
    ReEig,
    SPDBatchNorm,
    SPDNet,
    StiefelSGD,
    SymVectorize,
    airm_distance,
    karcher_mean,
    sym_exp,
    sym_log,
    sym_pow,
)

DT = torch.float64


def random_spd(batch, n, seed=0, cond=10.0):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(batch, n, n, generator=g, dtype=DT))
    s = torch.exp(torch.rand(batch, n, generator=g, dtype=DT) * torch.log(torch.tensor(cond)))
    return Q @ torch.diag_embed(s) @ Q.transpose(-1, -2)


def is_spd(X, tol=0.0):
    sym = torch.allclose(X, X.transpose(-1, -2), atol=1e-10)
    return sym and bool((torch.linalg.eigvalsh(X) > tol).all())


# --- spectral functions ---------------------------------------------------


def test_log_exp_are_inverse():
    X = random_spd(5, 6)
    assert torch.allclose(sym_exp(sym_log(X)), X, atol=1e-10)
    assert torch.allclose(sym_pow(X, 0.5) @ sym_pow(X, 0.5), X, atol=1e-10)
    assert torch.allclose(sym_pow(X, -0.5) @ X @ sym_pow(X, -0.5), torch.eye(6, dtype=DT), atol=1e-10)


@pytest.mark.parametrize("fn", [sym_log, sym_exp, lambda X: sym_pow(X, 0.5), lambda X: ReEig(0.5)(X)])
def test_daleckii_krein_gradient(fn):
    """Our backward pass matches numerical finite differences."""
    X = random_spd(3, 4, seed=1).requires_grad_()
    # Inputs to gradcheck must be symmetric-perturbation friendly: symmetrize inside.
    assert torch.autograd.gradcheck(lambda A: fn(0.5 * (A + A.transpose(-1, -2))), (X,))


def test_gradient_with_repeated_eigenvalues_is_finite():
    """Equal eigenvalues (e.g. the identity) must not produce NaNs."""
    X = torch.eye(4, dtype=DT).repeat(2, 1, 1).requires_grad_()
    sym_log(X).sum().backward()
    assert torch.isfinite(X.grad).all()


# --- layers -----------------------------------------------------------------


def test_bimap_keeps_spd_and_reduces_size():
    layer = BiMap(8, 3)
    W = layer.W.detach()
    assert torch.allclose(W.T @ W, torch.eye(3, dtype=DT), atol=1e-12)
    Y = layer(random_spd(10, 8))
    assert Y.shape == (10, 3, 3) and is_spd(Y)


def test_reeig_clamps_eigenvalues():
    eps = 0.5
    Y = ReEig(eps)(random_spd(10, 5, cond=100.0) * 0.1)
    assert is_spd(Y)
    assert torch.linalg.eigvalsh(Y).min() >= eps - 1e-10


def test_logeig_and_vectorize_preserve_norm():
    X = random_spd(4, 5)
    S = LogEig()(X)
    v = SymVectorize(5)(S)
    assert v.shape == (4, 15)
    assert torch.allclose(v.norm(dim=-1), torch.linalg.matrix_norm(S), atol=1e-10)


def test_logeig_distance_at_identity_is_airm():
    """Through the identity, the tangent-space distance equals the Riemannian distance."""
    X = random_spd(1, 5)[0]
    I = torch.eye(5, dtype=DT)
    assert torch.allclose(torch.linalg.matrix_norm(sym_log(X)), airm_distance(I, X), atol=1e-10)


# --- Riemannian mean and batch norm ---------------------------------------------


def test_karcher_mean_is_affine_equivariant():
    """Mean(A X_i A^T) = A Mean(X_i) A^T (the arithmetic mean has this property too) and
    Mean(X_i^{-1}) = Mean(X_i)^{-1} (the arithmetic mean does not)."""
    X = random_spd(6, 4, seed=2)
    M = karcher_mean(X, n_iter=30)
    A = torch.randn(4, 4, dtype=DT, generator=torch.Generator().manual_seed(3)) + 3 * torch.eye(4, dtype=DT)
    assert torch.allclose(karcher_mean(A @ X @ A.T, n_iter=30), A @ M @ A.T, atol=1e-8)
    assert torch.allclose(karcher_mean(torch.linalg.inv(X), n_iter=30), torch.linalg.inv(M), atol=1e-8)


def test_batchnorm_centers_batch_at_identity():
    bn = SPDBatchNorm(4, n_iter=30)
    Y = bn(random_spd(16, 4, seed=4, cond=50.0))
    assert is_spd(Y)
    assert torch.allclose(karcher_mean(Y, n_iter=30), torch.eye(4, dtype=DT), atol=1e-8)


def test_batchnorm_running_mean_stays_spd_and_eval_runs():
    bn = SPDBatchNorm(4)
    for seed in range(5):
        bn(random_spd(8, 4, seed=seed))
    assert is_spd(bn.running_mean)
    bn.eval()
    assert is_spd(bn(random_spd(3, 4, seed=10)))


# --- optimization on the Stiefel manifold -----------------------------------------


@pytest.mark.parametrize("batchnorm", [False, True])
def test_spdnet_learns_and_stays_on_stiefel(batchnorm):
    """Two synthetic classes whose covariances differ in a few directions."""
    torch.manual_seed(0)
    n, n_per_class = 10, 60
    base = random_spd(2 * n_per_class, n, seed=5, cond=3.0)
    boost = torch.ones(n, dtype=DT)
    boost[:2] = 6.0
    D = torch.diag(boost.sqrt())
    X = torch.cat([base[:n_per_class], D @ base[n_per_class:] @ D])
    y = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class)]).long()

    model = SPDNet([n, 6, 4], n_classes=2, batchnorm=batchnorm)
    opt = StiefelSGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(150):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()

    model.eval()
    acc = (model(X).argmax(1) == y).double().mean().item()
    assert acc > 0.9
    for layer in model.spd_layers:
        if isinstance(layer, BiMap):
            W = layer.W.detach()
            assert torch.allclose(W.T @ W, torch.eye(W.shape[1], dtype=DT), atol=1e-10)
