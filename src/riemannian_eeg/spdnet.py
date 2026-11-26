"""SPDNet: neural network layers that operate on the manifold of SPD matrices.

Implemented from scratch in PyTorch, following:

- Huang & Van Gool (2017), "A Riemannian Network for SPD Matrix Learning" (BiMap, ReEig, LogEig).
- Ionescu, Vantzos & Sminchisescu (2015), "Matrix Backpropagation for Deep Networks with
  Structured Layers" (gradients through eigendecompositions).
- Brooks et al. (2019), "Riemannian batch normalization for SPD neural networks" (SPDBatchNorm).

The theory behind each layer is explained in docs/03_redes_neuronales_variedades.md.

Input: a batch of SPD matrices (e.g. EEG covariance matrices), shape (batch, n, n).
Double precision (torch.float64) is recommended: eigendecompositions are sensitive to rounding.

Usage
-----
Covariances should be well conditioned and of order 1 (e.g. shrinkage estimator + division by
the mean trace of the training set), otherwise the ReEig threshold eps is meaningless::

    import torch
    from torch import nn
    from riemannian_eeg.spdnet import SPDNet, StiefelSGD

    C_train = torch.as_tensor(covs, dtype=torch.float64)   # (n_trials, 22, 22)
    y_train = torch.as_tensor(labels)                       # integers 0..3

    model = SPDNet([22, 16, 8], n_classes=4, batchnorm=True)
    optimizer = StiefelSGD(model.parameters(), lr=0.05, momentum=0.9)
    for epoch in range(n_epochs):
        for idx in torch.randperm(len(C_train)).split(32):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(C_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()

    model.eval()   # SPDBatchNorm then uses its running mean
    predictions = model(C_test).argmax(dim=1)

The architecture and training hyperparameters must be chosen on the training session only.
"""

import math

import torch
from torch import nn

# ---------------------------------------------------------------------------
# Spectral functions f(X) = U diag(f(s)) U^T with a custom backward pass
# ---------------------------------------------------------------------------


def _sym(A):
    return 0.5 * (A + A.transpose(-1, -2))


class _EigFunction(torch.autograd.Function):
    """Applies f to the eigenvalues of a symmetric matrix: f(X) = U diag(f(s)) U^T.

    Backward (Daleckii-Krein formula): for a symmetric upstream gradient G,
        dL/dX = U (K o (U^T G U)) U^T,
        K_ij = (f(s_i) - f(s_j)) / (s_i - s_j)   if s_i != s_j,
        K_ij = f'(s_i)                           if s_i == s_j.
    This is the closed form of the matrix backpropagation of Ionescu et al. (2015) and avoids
    the 1 / (s_i - s_j) instability of differentiating U directly.
    """

    @staticmethod
    def forward(ctx, X, f, fprime):
        s, U = torch.linalg.eigh(X)
        fs = f(s)
        ctx.save_for_backward(s, U, fs)
        ctx.fprime = fprime
        return U @ torch.diag_embed(fs) @ U.transpose(-1, -2)

    @staticmethod
    def backward(ctx, G):
        s, U, fs = ctx.saved_tensors
        ds = s.unsqueeze(-1) - s.unsqueeze(-2)
        dfs = fs.unsqueeze(-1) - fs.unsqueeze(-2)
        fp = ctx.fprime(s)
        # Where two eigenvalues (almost) coincide, the divided difference tends to f'(s).
        close = ds.abs() < 1e-10 * s.abs().amax(-1, keepdim=True).unsqueeze(-1).clamp_min(1.0)
        fp_mid = 0.5 * (fp.unsqueeze(-1) + fp.unsqueeze(-2))
        K = torch.where(close, fp_mid, dfs / torch.where(close, torch.ones_like(ds), ds))
        Ut = U.transpose(-1, -2)
        grad = U @ (K * (Ut @ _sym(G) @ U)) @ Ut
        return grad, None, None


def sym_apply(X, f, fprime):
    """f(X) for a batch of symmetric matrices, differentiable with the Daleckii-Krein formula."""
    return _EigFunction.apply(X, f, fprime)


def sym_log(X):
    """Matrix logarithm of SPD matrices: the map from the manifold to the tangent space at I."""
    return sym_apply(X, torch.log, torch.reciprocal)


def sym_exp(X):
    """Matrix exponential of symmetric matrices: the map back from the tangent space at I."""
    return sym_apply(X, torch.exp, torch.exp)


def sym_pow(X, t):
    """X^t for SPD matrices (t = 1/2: square root, t = -1/2: inverse square root)."""
    return sym_apply(X, lambda s: s.pow(t), lambda s: t * s.pow(t - 1))


def sym_rectify(X, eps):
    """Clamps the eigenvalues from below: max(eps I, X). Keeps the output SPD."""
    return sym_apply(
        X,
        lambda s: s.clamp_min(eps),
        lambda s: (s > eps).to(s.dtype),
    )


# ---------------------------------------------------------------------------
# Riemannian (affine-invariant) operations on SPD matrices
# ---------------------------------------------------------------------------


def geodesic(A, B, t):
    """Point at time t on the affine-invariant geodesic from A (t=0) to B (t=1):
    A^{1/2} (A^{-1/2} B A^{-1/2})^t A^{1/2}.
    """
    A_sqrt, A_isqrt = sym_pow(A, 0.5), sym_pow(A, -0.5)
    return A_sqrt @ sym_pow(A_isqrt @ B @ A_isqrt, t) @ A_sqrt


def karcher_mean(X, n_iter=1, init=None):
    """Riemannian (Frechet/Karcher) mean of a batch of SPD matrices, shape (batch, n, n).

    Fixed-point iteration: M <- M^{1/2} exp( mean_i log(M^{-1/2} X_i M^{-1/2}) ) M^{1/2}.
    Each step maps the batch to the tangent space at M, averages there (where the space is
    flat) and maps the average back to the manifold. Starts from the arithmetic mean.
    """
    M = X.mean(dim=0) if init is None else init
    for _ in range(n_iter):
        M_sqrt, M_isqrt = sym_pow(M, 0.5), sym_pow(M, -0.5)
        T = sym_log(M_isqrt @ X @ M_isqrt).mean(dim=0)
        M = M_sqrt @ sym_exp(T) @ M_sqrt
    return M


def airm_distance(A, B):
    """Affine-invariant Riemannian distance ||log(A^{-1/2} B A^{-1/2})||_F."""
    A_isqrt = sym_pow(A, -0.5)
    s = torch.linalg.eigvalsh(A_isqrt @ B @ A_isqrt)
    return torch.linalg.vector_norm(torch.log(s), dim=-1)


# ---------------------------------------------------------------------------
# Stiefel manifold: parameters with orthonormal columns and their optimizer
# ---------------------------------------------------------------------------


class StiefelParameter(nn.Parameter):
    """A parameter W in St(n, p) = {W in R^{n x p} : W^T W = I_p}.

    Plain gradient descent would break the constraint, so these parameters are updated by
    StiefelSGD with a Riemannian gradient step followed by a retraction.
    """


def stiefel_project(W, G):
    """Projects G onto the tangent space of the Stiefel manifold at W: G - W sym(W^T G)."""
    return G - W @ _sym(W.transpose(-1, -2) @ G)


def stiefel_retract(A):
    """QR retraction: the Q factor of A, with signs fixed so that diag(R) > 0 (unique Q)."""
    Q, R = torch.linalg.qr(A)
    signs = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return Q * signs.unsqueeze(-2)


class StiefelSGD(torch.optim.Optimizer):
    """SGD with momentum that keeps StiefelParameter's on the Stiefel manifold.

    For a StiefelParameter W with Euclidean gradient G:
        1. Riemannian gradient:   xi = G - W sym(W^T G)        (projection onto T_W St)
        2. momentum:              m <- beta * proj_W(m) + xi   (projection as vector transport)
        3. retraction:            W <- qf(W - lr * m)          (back onto the manifold)
    Any other parameter (e.g. the final linear classifier) gets ordinary SGD with momentum.
    """

    def __init__(self, params, lr=1e-2, momentum=0.0, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, beta, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if isinstance(p, StiefelParameter):
                    xi = stiefel_project(p, p.grad)
                    if beta > 0:
                        m = state.get("momentum")
                        m = xi if m is None else beta * stiefel_project(p, m) + xi
                        state["momentum"] = m
                    else:
                        m = xi
                    p.copy_(stiefel_retract(p - lr * m))
                else:
                    d = p.grad if wd == 0 else p.grad + wd * p
                    if beta > 0:
                        m = state.get("momentum")
                        m = d.clone() if m is None else beta * m + d
                        state["momentum"] = m
                        d = m
                    p.add_(d, alpha=-lr)
        return loss


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


class BiMap(nn.Module):
    """Bilinear mapping X -> W^T X W, with W in St(n_in, n_out) and n_out <= n_in.

    Since W has full column rank, the output is again SPD (of smaller size). With n_out = n_in
    it is a congruence; with n_out < n_in it projects onto a lower-dimensional SPD manifold.
    For EEG covariances, each column of W is a spatial filter (as in CSP), and W^T X W is the
    covariance of the filtered "virtual electrodes", here learned end-to-end.
    """

    def __init__(self, n_in, n_out, dtype=torch.float64):
        super().__init__()
        if n_out > n_in:
            raise ValueError("BiMap needs n_out <= n_in so that W^T X W stays SPD.")
        W = stiefel_retract(torch.randn(n_in, n_out, dtype=dtype))
        self.W = StiefelParameter(W)

    def forward(self, X):
        return self.W.transpose(-1, -2) @ X @ self.W

    def extra_repr(self):
        return f"{self.W.shape[0]} -> {self.W.shape[1]}"


class ReEig(nn.Module):
    """Eigenvalue rectification max(eps I, X): the non-linearity of SPDNet.

    Like a ReLU on the eigenvalues, but with threshold eps > 0, so the output stays SPD.
    It also prevents the eigenvalues from collapsing towards 0 after several BiMap layers.
    """

    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, X):
        return sym_rectify(X, self.eps)

    def extra_repr(self):
        return f"eps={self.eps}"


class LogEig(nn.Module):
    """Matrix logarithm U log(S) U^T: flattens the manifold onto the tangent space at I.

    The output is a symmetric matrix (a vector space), so an ordinary Euclidean classifier can
    be used afterwards. Euclidean distances between log-matrices are Log-Euclidean distances.
    """

    def forward(self, X):
        return sym_log(X)


class SymVectorize(nn.Module):
    """Upper triangle of a symmetric matrix, off-diagonal entries scaled by sqrt(2).

    The scaling makes the Euclidean norm of the vector equal to the Frobenius norm of the
    matrix, so no information is duplicated and the geometry of the tangent space is kept.
    Output dimension: n (n + 1) / 2.
    """

    def __init__(self, n):
        super().__init__()
        rows, cols = torch.triu_indices(n, n)
        self.register_buffer("rows", rows)
        self.register_buffer("cols", cols)
        weights = torch.where(rows == cols, 1.0, math.sqrt(2.0)).to(torch.float64)
        self.register_buffer("weights", weights)

    def forward(self, S):
        return S[..., self.rows, self.cols] * self.weights.to(S.dtype)


class SPDBatchNorm(nn.Module):
    """Riemannian batch normalization for SPD matrices (Brooks et al., 2019).

    Euclidean batch norm subtracts the batch mean and adds a learned bias. On the SPD manifold:
        centering:  X_i -> M^{-1/2} X_i M^{-1/2}   (parallel transport from the batch mean M to I)
        biasing:    X_i -> B^{1/2}  X_i B^{1/2}    (parallel transport from I to a learned B)
    M is the Karcher mean of the batch while training, and a running mean (updated along the
    geodesic) at inference time.

    Simplification: B is parametrized as B = exp(S) with S symmetric, so that it can be learned
    with ordinary gradient descent instead of Riemannian optimization on the SPD manifold.
    """

    def __init__(self, n, momentum=0.1, n_iter=1, dtype=torch.float64):
        super().__init__()
        self.momentum = momentum
        self.n_iter = n_iter
        self.bias_log = nn.Parameter(torch.zeros(n, n, dtype=dtype))
        self.register_buffer("running_mean", torch.eye(n, dtype=dtype))

    def forward(self, X):
        if self.training:
            M = karcher_mean(X, n_iter=self.n_iter)
            with torch.no_grad():
                self.running_mean.copy_(geodesic(self.running_mean, M.detach(), self.momentum))
        else:
            M = self.running_mean
        M_isqrt = sym_pow(M, -0.5)
        B_sqrt = sym_exp(0.5 * _sym(self.bias_log))
        return B_sqrt @ (M_isqrt @ X @ M_isqrt) @ B_sqrt


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------


class SPDNet(nn.Module):
    """SPDNet classifier: [BiMap -> (SPDBatchNorm) -> ReEig] x k -> LogEig -> vectorize -> Linear.

    Parameters
    ----------
    dims : list of int
        Matrix sizes, e.g. [22, 16, 8]: input 22 x 22, then BiMap to 16 x 16 and to 8 x 8.
    n_classes : int
        Number of output classes (4 for BNCI2014_001).
    batchnorm : bool
        If True, inserts SPDBatchNorm after each BiMap (SPDNetBN, Brooks et al. 2019).
    eps : float
        ReEig threshold.

    The forward pass returns logits; train with nn.CrossEntropyLoss and StiefelSGD.
    """

    def __init__(self, dims, n_classes, batchnorm=False, eps=1e-4, dtype=torch.float64):
        super().__init__()
        layers = []
        for n_in, n_out in zip(dims[:-1], dims[1:]):
            layers.append(BiMap(n_in, n_out, dtype=dtype))
            if batchnorm:
                layers.append(SPDBatchNorm(n_out, dtype=dtype))
            layers.append(ReEig(eps))
        self.spd_layers = nn.Sequential(*layers)
        self.logeig = LogEig()
        self.vectorize = SymVectorize(dims[-1])
        self.classifier = nn.Linear(dims[-1] * (dims[-1] + 1) // 2, n_classes, dtype=dtype)

    def features(self, X):
        """Tangent-space features before the classifier, shape (batch, n (n + 1) / 2)."""
        return self.vectorize(self.logeig(self.spd_layers(X)))

    def forward(self, X):
        return self.classifier(self.features(X))
