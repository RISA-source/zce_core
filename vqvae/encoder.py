"""
vqvae/encoder.py
================
The encoder network: maps continuous input to a latent vector
that will be quantized into a ZCE token.

Architecture:
    input_dim → hidden → hidden/2 → latent_dim

The encoder learns to produce latent vectors that are close to
codebook entries (base ZCE tokens). The quantizer then snaps
each latent to the nearest codebook entry.

This is the continuous-to-discrete interface. The encoder learns
what structure is worth naming — it learns which regions of the
input space should map to the same discrete token.
"""

import numpy as np
from .nn import Linear, ReLU, LayerNorm


class Encoder:
    """
    Encoder: input → latent vector.

    input_dim  : dimension of input (e.g. 784 for MNIST 28x28)
    hidden_dim : hidden layer width
    latent_dim : output dimension (must match codebook vector size)
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 rng: np.random.Generator):
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.fc1   = Linear(input_dim,    hidden_dim, rng)
        self.relu1 = ReLU()
        self.ln1   = LayerNorm(hidden_dim)

        self.fc2   = Linear(hidden_dim,   hidden_dim // 2, rng)
        self.relu2 = ReLU()
        self.ln2   = LayerNorm(hidden_dim // 2)

        self.fc3   = Linear(hidden_dim // 2, latent_dim, rng)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (batch, input_dim)
        returns: (batch, latent_dim)
        """
        h = self.fc1.forward(x)
        h = self.ln1.forward(h)
        h = self.relu1.forward(h)

        h = self.fc2.forward(h)
        h = self.ln2.forward(h)
        h = self.relu2.forward(h)

        z = self.fc3.forward(h)
        return z

    def backward(self, grad_z: np.ndarray) -> np.ndarray:
        """
        grad_z: (batch, latent_dim) — gradient from quantizer
        returns: gradient w.r.t. input
        """
        g = self.fc3.backward(grad_z)

        g = self.relu2.backward(g)
        g = self.ln2.backward(g)
        g = self.fc2.backward(g)

        g = self.relu1.backward(g)
        g = self.ln1.backward(g)
        g = self.fc1.backward(g)

        return g

    def params(self):
        """All (param, grad) pairs for optimizer."""
        return (
            self.fc1.params() + self.ln1.params() +
            self.fc2.params() + self.ln2.params() +
            self.fc3.params()
        )