"""
vqvae/vqvae.py
==============
The complete VQ-VAE model.

Combines:
    Encoder     — continuous input → latent vector
    Quantizer   — latent vector → discrete ZCE token + quantized vector
    Decoder     — quantized vector → reconstructed input

The key insight: the quantizer is the bridge between the neural
world (continuous latent vectors) and the ZCE world (discrete tokens).

Forward pass:
    x → z_e = Encoder(x)
    z_q, tokens, vq_losses = Quantizer(z_e)
    x_hat = Decoder(z_q)
    loss = recon_loss(x_hat, x) + vq_losses['total']

Backward pass:
    grad_recon → Decoder.backward → grad_z_q
    grad_z_q → Quantizer.backward (straight-through) → grad_z_e
    grad_z_e → Encoder.backward

ZCE integration:
    tokens: (batch,) int — these are ZCE token identities
    Each token identity maps to a codebook entry (depth=0 symbol)
    The SDTS engine receives these token streams and builds
    higher-depth vocabulary on top.

Spatial VQ-VAE (for images):
    For images, the encoder produces a grid of latents:
    (batch, H/stride, W/stride, latent_dim)
    Each spatial position gets its own token.
    A 28x28 image with stride 4 → 7x7 = 49 tokens per image.
    This is a token STREAM, which the SDTS engine can process.
"""

import numpy as np
from .encoder import Encoder
from .decoder import Decoder
from .quantizer import VectorQuantizer
from .nn import mse_loss, bce_loss, Adam


class VQVAE:
    """
    Complete VQ-VAE model.

    input_dim    : input dimension (e.g. 784 for MNIST)
    hidden_dim   : hidden layer width (e.g. 256)
    latent_dim   : codebook vector dimension (e.g. 64)
    n_tokens     : number of codebook entries (ZCE base vocab size)
    beta         : commitment loss weight
    output_range : 'sigmoid' for image inputs [0,1]
    """

    def __init__(
        self,
        input_dim    : int   = 784,
        hidden_dim   : int   = 256,
        latent_dim   : int   = 64,
        n_tokens     : int   = 32,
        beta         : float = 0.25,
        output_range : str   = 'linear',
        seed         : int   = 42,
    ):
        self.input_dim   = input_dim
        self.hidden_dim  = hidden_dim
        self.latent_dim  = latent_dim
        self.n_tokens    = n_tokens
        self.beta        = beta

        rng = np.random.default_rng(seed)

        self.encoder   = Encoder(input_dim, hidden_dim, latent_dim, rng)
        self.quantizer = VectorQuantizer(n_tokens, latent_dim, beta, rng=rng)
        self.decoder   = Decoder(latent_dim, hidden_dim, input_dim,
                                 rng, output_range)

        # Optimizer
        self.optimizer = Adam(lr=1e-3, weight_decay=1e-4)

        # Training state
        self.step_count = 0
        self.loss_history = []

    def forward(self, x: np.ndarray) -> dict:
        """
        Full forward pass.

        x: (batch, input_dim)
        returns: dict with all outputs and losses
        """
        # Encode
        z_e = self.encoder.forward(x)

        # Quantize
        z_q, tokens, vq_losses = self.quantizer.forward(z_e)

        # Decode (straight-through: use z_q but gradients flow through z_e)
        # z_q_st = z_e + (z_q - z_e) — same value as z_q, but gradient = grad of z_e
        z_q_st = z_e + (z_q - z_e)   # straight-through trick
        x_hat  = self.decoder.forward(z_q_st)

        # Reconstruction loss
        recon_loss, grad_recon = mse_loss(x_hat, x)

        total_loss = recon_loss + vq_losses['total_vq_loss']

        return {
            'x_hat'      : x_hat,
            'z_e'        : z_e,
            'z_q'        : z_q,
            'tokens'     : tokens,
            'recon_loss' : recon_loss,
            'vq_losses'  : vq_losses,
            'total_loss' : total_loss,
            'grad_recon' : grad_recon,
            '_z_q_st'    : z_q_st,
        }

    def backward(self, result: dict) -> None:
        """
        Full backward pass from result dict.
        """
        # Gradient through decoder
        grad_z_q_st = self.decoder.backward(result['grad_recon'])

        # Straight-through: gradient passes to encoder unchanged
        # grad_z_q_st is gradient w.r.t. z_q_st = z_e + (z_q - z_e)
        # Since d(z_q_st)/d(z_e) = I (identity), grad flows to z_e directly
        grad_z_e = self.quantizer.backward(grad_z_q_st)

        # Gradient through encoder
        self.encoder.backward(grad_z_e)

    def step(self, x: np.ndarray) -> dict:
        """
        One training step: forward + backward + optimizer update.

        x: (batch, input_dim)
        returns: metrics dict
        """
        # Collect all params before zeroing
        all_params = (
            self.encoder.params() +
            self.decoder.params() +
            self.quantizer.codebook_params()
        )

        # Zero gradients
        self.optimizer.zero_grad(all_params)

        # Forward
        result = self.forward(x)

        # Backward
        self.backward(result)

        # Optimizer step
        self.optimizer.step(all_params)

        self.step_count += 1

        metrics = {
            'step'             : self.step_count,
            'total_loss'       : result['total_loss'],
            'recon_loss'       : result['recon_loss'],
            'embedding_loss'   : result['vq_losses']['embedding_loss'],
            'commitment_loss'  : result['vq_losses']['commitment_loss'],
            'codebook_util'    : self.quantizer.codebook_utilization(),
            'perplexity'       : self.quantizer.perplexity(),
        }
        self.loss_history.append(metrics)
        return metrics

    def encode_to_tokens(self, x: np.ndarray) -> np.ndarray:
        """
        Encode input to discrete token indices.
        x: (batch, input_dim)
        returns: (batch,) int — ZCE token identities
        """
        z_e = self.encoder.forward(x)
        _, tokens, _ = self.quantizer.forward(z_e)
        return tokens

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """
        Reconstruct input through the full VQ-VAE.
        x: (batch, input_dim)
        returns: (batch, input_dim) reconstructed
        """
        result = self.forward(x)
        return result['x_hat']

    def token_to_vector(self, token_id: int) -> np.ndarray:
        """Get the codebook vector for a token identity."""
        return self.quantizer.token_to_vector(token_id)

    def summary(self) -> str:
        lines = [
            "VQVAE Summary:",
            f"  input_dim:  {self.input_dim}",
            f"  hidden_dim: {self.hidden_dim}",
            f"  latent_dim: {self.latent_dim}",
            f"  n_tokens:   {self.n_tokens}  (ZCE base vocab size)",
            f"  beta:       {self.beta}",
            f"  steps:      {self.step_count}",
        ]
        if self.loss_history:
            last = self.loss_history[-1]
            lines += [
                f"  last total_loss:  {last['total_loss']:.4f}",
                f"  last recon_loss:  {last['recon_loss']:.4f}",
                f"  codebook_util:    {last['codebook_util']:.2%}",
                f"  perplexity:       {last['perplexity']:.1f} / {self.n_tokens}",
            ]
        return "\n".join(lines)