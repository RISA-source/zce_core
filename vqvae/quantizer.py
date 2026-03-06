"""
vqvae/quantizer.py
==================
The Vector Quantizer: the core of VQ-VAE.

This is where continuous latent vectors become discrete ZCE tokens.

Forward pass:
    1. Given encoder output z_e (batch, latent_dim)
    2. For each z_e[i], find the nearest codebook entry e_k
       by Euclidean distance
    3. Replace z_e[i] with e_k (the quantized vector z_q)
    4. Return z_q and the token indices

Backward pass (straight-through estimator):
    The quantization operation argmin is not differentiable.
    The straight-through estimator bypasses this:
    - Gradient flows through z_q → z_e unchanged
    - As if quantization never happened
    - Encoder gets gradient from reconstruction loss
    - Codebook gets gradient from commitment loss

Loss components:
    embedding_loss  = ||sg(z_e) - e_k||^2  (moves codebook toward encoder)
    commitment_loss = beta * ||z_e - sg(e_k)||^2 (keeps encoder near codebook)

    where sg(·) = stop-gradient

Codebook update strategy:
    Option 1: gradient descent on embedding_loss (used here)
    Option 2: exponential moving average (more stable, added as option)

ZCE Connection:
    Each codebook entry corresponds to one ZCE token identity.
    codebook[k] = the vector representation of token with identity k.
    The SDTS engine receives the token INDEX k, not the vector.
    Token depth = 0 for all base codebook tokens.
    The SDTS engine builds depth-1, depth-2... symbols on top.
"""

import numpy as np


class VectorQuantizer:
    """
    Vector Quantizer with straight-through gradient estimator.

    n_embeddings : number of codebook entries (= ZCE vocabulary size)
    embedding_dim: dimension of each codebook vector
    beta         : commitment loss weight (default 0.25, per VQ-VAE paper)
    ema          : use exponential moving average for codebook update
    ema_decay    : EMA decay rate (if ema=True)
    """

    def __init__(self, n_embeddings: int, embedding_dim: int,
                 beta: float = 0.25,
                 ema: bool = True,
                 ema_decay: float = 0.99,
                 rng: np.random.Generator = None):
        self.n_embeddings  = n_embeddings
        self.embedding_dim = embedding_dim
        self.beta          = beta
        self.ema           = ema
        self.ema_decay     = ema_decay

        if rng is None:
            rng = np.random.default_rng(42)

        # Codebook: (n_embeddings, embedding_dim)
        # Initialized with uniform noise — small scale
        # Larger init forces encoder to spread out, prevents collapse
        self.embeddings = rng.normal(0, 1.0,
            (n_embeddings, embedding_dim)
        ).astype(np.float32)

        # Gradient for non-EMA update
        self.d_embeddings = np.zeros_like(self.embeddings)

        # EMA state (if ema=True)
        if ema:
            self._ema_cluster_size = np.ones(n_embeddings, dtype=np.float32)
            self._ema_dw           = self.embeddings.copy()

        # Cache for backward
        self._z_e    = None
        self._z_q    = None
        self._tokens = None

        # Usage tracking — which tokens are actually used
        self.usage_counts = np.zeros(n_embeddings, dtype=np.int64)

    def forward(self, z_e: np.ndarray) -> tuple:
        """
        z_e: (batch, embedding_dim)
        returns: (z_q, tokens, losses)
            z_q   : (batch, embedding_dim) — quantized latent
            tokens : (batch,) int — token index for each sample
            losses : dict with 'embedding_loss', 'commitment_loss', 'total'
        """
        batch = z_e.shape[0]

        # Compute distances: ||z_e - e_k||^2
        # = ||z_e||^2 - 2 z_e·e_k + ||e_k||^2
        z_sq  = (z_e ** 2).sum(axis=1, keepdims=True)          # (batch, 1)
        e_sq  = (self.embeddings ** 2).sum(axis=1, keepdims=True).T  # (1, K)
        dot   = z_e @ self.embeddings.T                         # (batch, K)
        dists = z_sq - 2 * dot + e_sq                          # (batch, K)

        # Nearest codebook entry
        tokens = dists.argmin(axis=1)   # (batch,)
        z_q    = self.embeddings[tokens]  # (batch, embedding_dim)

        # Track usage
        for t in tokens:
            self.usage_counts[t] += 1

        # Losses
        # stop-gradient of z_e for embedding loss: move codebook toward encoder
        embedding_loss  = ((z_q - z_e.copy()) ** 2).mean()
        # stop-gradient of z_q for commitment loss: keep encoder near codebook
        commitment_loss = self.beta * ((z_e - z_q.copy()) ** 2).mean()
        total_vq_loss   = embedding_loss + commitment_loss

        losses = {
            'embedding_loss'  : float(embedding_loss),
            'commitment_loss' : float(commitment_loss),
            'total_vq_loss'   : float(total_vq_loss),
        }

        # Cache for backward
        self._z_e    = z_e
        self._z_q    = z_q
        self._tokens = tokens
        self._dists  = dists

        # EMA update (if enabled)
        if self.ema:
            self._ema_update(z_e, tokens)

        return z_q, tokens, losses

    def restart_dead_entries(self, z_e: np.ndarray, threshold: int = 1):
        """
        Restart dead codebook entries (entries with very low usage).
        Pick a random encoder output and assign it to the dead entry.
        This is the standard fix for codebook collapse.
        """
        rng = np.random.default_rng()
        for k in range(self.n_embeddings):
            if self.usage_counts[k] < threshold:
                # Assign a random encoder output to this entry
                idx = rng.integers(0, len(z_e))
                self.embeddings[k] = z_e[idx] + rng.normal(0, 0.01, z_e.shape[1])

    def _ema_update(self, z_e: np.ndarray, tokens: np.ndarray):
        """
        Exponential moving average update of codebook.
        More stable than gradient descent on embedding_loss.
        Per VQ-VAE-2 paper (Razavi et al. 2019).
        """
        # Count how many encoder outputs map to each codebook entry
        one_hot = np.zeros((len(tokens), self.n_embeddings), dtype=np.float32)
        one_hot[np.arange(len(tokens)), tokens] = 1.0

        # EMA of cluster sizes
        n = one_hot.sum(axis=0)  # (K,)
        self._ema_cluster_size = (
            self.ema_decay * self._ema_cluster_size +
            (1 - self.ema_decay) * n
        )

        # EMA of sum of encoder outputs per cluster
        dw = one_hot.T @ z_e   # (K, D)
        self._ema_dw = (
            self.ema_decay * self._ema_dw +
            (1 - self.ema_decay) * dw
        )

        # Update codebook: normalize by cluster size
        n_stable = self._ema_cluster_size[:, None]
        n_stable = np.where(n_stable > 1e-5, n_stable, 1e-5)
        self.embeddings = self._ema_dw / n_stable

    def backward(self, grad_z_q: np.ndarray) -> tuple:
        """
        Straight-through estimator.

        grad_z_q: (batch, embedding_dim) — gradient from decoder
        returns: (grad_z_e, grad_codebook)

        The straight-through estimator passes grad_z_q directly
        to z_e, as if the quantization didn't happen. The encoder
        receives the decoder's gradient signal.

        Codebook gradient (for non-EMA mode):
            d embedding_loss / d e_k = 2 * (e_k - z_e[assigned to k]) / N
        """
        # Straight-through: grad flows to encoder unchanged
        grad_z_e = grad_z_q.copy()

        # Commitment loss gradient to encoder (additional signal)
        # d commitment_loss / d z_e = 2 * beta * (z_e - z_q) / N
        grad_z_e += (
            2 * self.beta *
            (self._z_e - self._z_q) / self._z_e.shape[0]
        )

        # Codebook gradient (used only if not EMA)
        if not self.ema:
            batch = self._z_e.shape[0]
            self.d_embeddings[:] = 0
            for i, k in enumerate(self._tokens):
                self.d_embeddings[k] += 2 * (
                    self.embeddings[k] - self._z_e[i]
                ) / batch

        return grad_z_e

    def codebook_params(self):
        """For non-EMA optimizer update."""
        if not self.ema:
            return [(self.embeddings, self.d_embeddings)]
        return []

    def token_to_vector(self, token_id: int) -> np.ndarray:
        """Get the codebook vector for a token identity."""
        return self.embeddings[token_id].copy()

    def active_tokens(self, min_uses: int = 1) -> list:
        """Return token IDs that have been used at least min_uses times."""
        return [i for i, c in enumerate(self.usage_counts) if c >= min_uses]

    def codebook_utilization(self) -> float:
        """Fraction of codebook entries that have been used."""
        used = (self.usage_counts > 0).sum()
        return used / self.n_embeddings

    def perplexity(self) -> float:
        """
        Codebook perplexity — measures how uniformly the codebook is used.
        Maximum = n_embeddings (uniform use).
        Minimum = 1 (all samples map to one token).
        Higher is better — means the codebook is well-utilized.
        """
        counts = self.usage_counts.astype(np.float64)
        total  = counts.sum()
        if total == 0:
            return 1.0
        probs = counts / total
        probs = probs[probs > 0]
        entropy = -(probs * np.log(probs)).sum()
        return float(np.exp(entropy))

    def reset_usage(self):
        """Reset usage counts (call between epochs if needed)."""
        self.usage_counts[:] = 0
