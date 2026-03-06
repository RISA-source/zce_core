"""
vqvae/decoder.py
================
The decoder network: maps a codebook vector (quantized latent)
back to the input space.

Architecture:
    latent_dim → hidden/2 → hidden → input_dim

The decoder learns to reconstruct the input from a single
discrete token. The quality of reconstruction is what forces
the encoder to use the codebook entries meaningfully.

If reconstruction is bad, the encoder gets gradient signal
pushing it to use codebook entries that actually capture
the input's structure.
"""

import numpy as np
from .nn import Linear, ReLU, LayerNorm, Sigmoid


class Decoder:
    """
    Decoder: codebook vector → reconstructed input.

    latent_dim  : input dimension (codebook vector size)
    hidden_dim  : hidden layer width
    output_dim  : dimension of reconstructed output
    output_range: 'sigmoid' for [0,1] outputs (images),
                  'linear' for unbounded outputs
    """

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int,
                 rng: np.random.Generator, output_range: str = 'sigmoid'):
        self.latent_dim    = latent_dim
        self.hidden_dim    = hidden_dim
        self.output_dim    = output_dim
        self.output_range  = output_range

        self.fc1   = Linear(latent_dim,      hidden_dim // 2, rng)
        self.relu1 = ReLU()
        self.ln1   = LayerNorm(hidden_dim // 2)

        self.fc2   = Linear(hidden_dim // 2, hidden_dim,      rng)
        self.relu2 = ReLU()
        self.ln2   = LayerNorm(hidden_dim)

        self.fc3   = Linear(hidden_dim,      output_dim,      rng)
        self.sig   = Sigmoid() if output_range == 'sigmoid' else None

    def forward(self, z_q: np.ndarray) -> np.ndarray:
        """
        z_q: (batch, latent_dim) — quantized latent
        returns: (batch, output_dim)
        """
        h = self.fc1.forward(z_q)
        h = self.ln1.forward(h)
        h = self.relu1.forward(h)

        h = self.fc2.forward(h)
        h = self.ln2.forward(h)
        h = self.relu2.forward(h)

        out = self.fc3.forward(h)
        if self.sig is not None:
            out = self.sig.forward(out)
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """
        grad_out: gradient of loss w.r.t. decoder output
        returns: gradient w.r.t. z_q (quantized latent)
        """
        g = grad_out
        if self.sig is not None:
            g = self.sig.backward(g)

        g = self.fc3.backward(g)
        g = self.relu2.backward(g)
        g = self.ln2.backward(g)
        g = self.fc2.backward(g)

        g = self.relu1.backward(g)
        g = self.ln1.backward(g)
        g = self.fc1.backward(g)

        return g

    def params(self):
        p = (
            self.fc1.params() + self.ln1.params() +
            self.fc2.params() + self.ln2.params() +
            self.fc3.params()
        )
        if self.sig is not None:
            p = p + self.sig.params() if hasattr(self.sig, 'params') else p
        return p