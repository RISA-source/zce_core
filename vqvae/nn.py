"""
vqvae/nn.py
===========
Pure-numpy neural network primitives.

No PyTorch, no autograd. Every forward pass and backward pass
written explicitly. This is intentional — it makes the gradient
flow through the straight-through estimator transparent and
verifiable.

Layers:
    Linear      — y = Wx + b
    ReLU        — y = max(0, x)
    LayerNorm   — normalize across features

Loss:
    mse_loss    — mean squared error
    commitment_loss — ||z_e - sg(e)||^2

Straight-through estimator:
    Used in the quantizer. Forward: use quantized value.
    Backward: pass gradient through as if quantization didn't happen.
    Implemented as: grad_in = grad_out (copy gradient unchanged).
"""

import numpy as np


class Linear:
    """Fully connected layer: y = xW + b."""

    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator):
        # He initialization
        scale = np.sqrt(2.0 / in_dim)
        self.W = rng.normal(0, scale, (in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)
        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        # Cache for backward
        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        self.dW = self._x.T @ grad_out
        self.db = grad_out.sum(axis=0)
        return grad_out @ self.W.T

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    """ReLU activation: y = max(0, x)."""

    def __init__(self):
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = (x > 0)
        return x * self._mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self._mask


class LayerNorm:
    """
    Layer normalization across the last dimension.
    y = (x - mean) / sqrt(var + eps) * gamma + beta
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        self.gamma = np.ones(dim,  dtype=np.float32)
        self.beta  = np.zeros(dim, dtype=np.float32)
        self.eps   = eps
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta  = np.zeros_like(self.beta)
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var  = x.var(axis=-1,  keepdims=True)
        xhat = (x - mean) / np.sqrt(var + self.eps)
        self._cache = (x, xhat, mean, var)
        return self.gamma * xhat + self.beta

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        x, xhat, mean, var = self._cache
        N = x.shape[-1]
        self.dgamma = (grad_out * xhat).sum(axis=0)
        self.dbeta  = grad_out.sum(axis=0)
        dxhat = grad_out * self.gamma
        std_inv = 1.0 / np.sqrt(var + self.eps)
        dx = (1.0/N) * std_inv * (
            N * dxhat
            - dxhat.sum(axis=-1, keepdims=True)
            - xhat * (dxhat * xhat).sum(axis=-1, keepdims=True)
        )
        return dx

    def params(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]


class Sigmoid:
    """Sigmoid: y = 1 / (1 + exp(-x))."""

    def __init__(self):
        self._y = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._y = 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
        return self._y

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self._y * (1 - self._y)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def mse_loss(pred: np.ndarray, target: np.ndarray):
    """MSE loss and gradient."""
    diff = pred - target
    loss = (diff ** 2).mean()
    grad = 2 * diff / diff.size
    return loss, grad


def bce_loss(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7):
    """Binary cross-entropy loss and gradient."""
    pred  = np.clip(pred, eps, 1 - eps)
    loss  = -(target * np.log(pred) + (1 - target) * np.log(1 - pred)).mean()
    grad  = (-(target / pred) + (1 - target) / (1 - pred)) / pred.size
    return loss, grad


# ---------------------------------------------------------------------------
# SGD with momentum optimizer
# ---------------------------------------------------------------------------

class SGD:
    """SGD with momentum. Applied to a list of (param, grad) pairs."""

    def __init__(self, lr: float = 1e-3, momentum: float = 0.9,
                 weight_decay: float = 1e-4):
        self.lr           = lr
        self.momentum     = momentum
        self.weight_decay = weight_decay
        self._velocity    = {}

    def step(self, param_groups: list):
        """param_groups: list of (param_array, grad_array) pairs."""
        for i, (p, g) in enumerate(param_groups):
            if i not in self._velocity:
                self._velocity[i] = np.zeros_like(p)
            if self.weight_decay > 0:
                g = g + self.weight_decay * p
            v = self._velocity[i]
            v[:] = self.momentum * v + g
            p -= self.lr * v

    def zero_grad(self, param_groups: list):
        for p, g in param_groups:
            g[:] = 0


class Adam:
    """Adam optimizer."""

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8,
                 weight_decay: float = 1e-4):
        self.lr           = lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.eps          = eps
        self.weight_decay = weight_decay
        self._m   = {}
        self._v   = {}
        self._t   = 0

    def step(self, param_groups: list):
        self._t += 1
        t = self._t
        for i, (p, g) in enumerate(param_groups):
            if i not in self._m:
                self._m[i] = np.zeros_like(p)
                self._v[i] = np.zeros_like(p)
            if self.weight_decay > 0:
                g = g + self.weight_decay * p
            m = self._m[i]; v = self._v[i]
            m[:] = self.beta1 * m + (1 - self.beta1) * g
            v[:] = self.beta2 * v + (1 - self.beta2) * g * g
            mhat = m / (1 - self.beta1 ** t)
            vhat = v / (1 - self.beta2 ** t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def zero_grad(self, param_groups: list):
        for p, g in param_groups:
            g[:] = 0
