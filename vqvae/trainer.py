"""
vqvae/trainer.py
================
Training loop for the VQ-VAE.

Also handles the integration point with the SDTS Understanding Engine:
after training, the trainer feeds token streams from the encoder
into the SDTS engine, which builds a vocabulary on top.

Data:
    We don't need real MNIST here. We generate synthetic data
    that has the right statistical structure to test the system:
    - Clustered data: samples from K Gaussian clusters
    - Pattern data: inputs with recurring sub-patterns
    - Geometric data: simple shapes as binary vectors

    The goal is to verify that:
    1. VQ-VAE trains and loss decreases
    2. Codebook utilization is high (all tokens used)
    3. Token streams from similar inputs are similar
    4. SDTS engine finds structure in the token streams

Real MNIST (if available via file upload) can be plugged in
by replacing the data generator with a loader.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .vqvae import VQVAE


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def make_clustered_data(
    n_samples   : int   = 1000,
    input_dim   : int   = 64,
    n_clusters  : int   = 8,
    noise       : float = 0.1,
    seed        : int   = 42,
) -> tuple:
    """
    Generate data from K Gaussian clusters.
    Each cluster is a prototype vector + noise.
    Returns (data, labels).

    This tests whether VQ-VAE correctly identifies the K clusters
    as distinct tokens. Expected: codebook utilization ~K/n_tokens.
    """
    rng = np.random.default_rng(seed)
    prototypes = rng.normal(0, 1, (n_clusters, input_dim)).astype(np.float32)
    # Normalize prototypes to [0,1]
    prototypes = (prototypes - prototypes.min()) / (prototypes.max() - prototypes.min() + 1e-8)

    labels = rng.integers(0, n_clusters, n_samples)
    data   = prototypes[labels] + rng.normal(0, noise, (n_samples, input_dim)).astype(np.float32)
    data   = np.clip(data, 0, 1)
    return data, labels, prototypes


def make_pattern_data(
    n_samples   : int   = 1000,
    input_dim   : int   = 64,
    n_patterns  : int   = 4,
    seed        : int   = 42,
) -> tuple:
    """
    Generate data with recurring structural patterns.
    Each sample = one of N_patterns base patterns, with position
    and scale variation.

    This tests whether VQ-VAE discovers patterns that the SDTS
    engine can then find recurring and build symbols from.
    """
    rng = np.random.default_rng(seed)
    segment = input_dim // n_patterns

    # Create base patterns (e.g. different frequency sinusoids)
    patterns = []
    for k in range(n_patterns):
        t = np.linspace(0, 2 * np.pi, input_dim)
        p = (np.sin((k+1) * t) + 1) / 2   # [0,1]
        patterns.append(p.astype(np.float32))

    labels = rng.integers(0, n_patterns, n_samples)
    noise  = rng.normal(0, 0.05, (n_samples, input_dim)).astype(np.float32)
    data   = np.array([patterns[l] for l in labels]) + noise
    data   = np.clip(data, 0, 1)
    return data, labels, np.array(patterns)


def make_binary_shapes(
    n_samples : int = 1000,
    size      : int = 8,
    seed      : int = 42,
) -> tuple:
    """
    Generate binary 8x8 images of simple shapes:
    0: empty, 1: horizontal line, 2: vertical line,
    3: diagonal, 4: block, 5: cross, 6: border, 7: checkerboard

    This tests the visual/spatial tokenization case —
    closest to real image VQ-VAE.
    """
    rng = np.random.default_rng(seed)
    templates = np.zeros((8, size*size), dtype=np.float32)

    grid = np.zeros((size, size), dtype=np.float32)

    def shape(idx, grid):
        g = np.zeros((size, size), dtype=np.float32)
        mid = size // 2
        if idx == 0: pass  # empty
        elif idx == 1: g[mid, :] = 1
        elif idx == 2: g[:, mid] = 1
        elif idx == 3: np.fill_diagonal(g, 1)
        elif idx == 4: g[mid-1:mid+2, mid-1:mid+2] = 1
        elif idx == 5: g[mid, :] = 1; g[:, mid] = 1
        elif idx == 6: g[0,:]=g[-1,:]=g[:,0]=g[:,-1]=1
        elif idx == 7:
            for i in range(size):
                for j in range(size):
                    if (i+j) % 2 == 0: g[i,j] = 1
        return g.flatten()

    for i in range(8):
        templates[i] = shape(i, grid)

    labels = rng.integers(0, 8, n_samples)
    noise  = rng.normal(0, 0.05, (n_samples, size*size)).astype(np.float32)
    data   = templates[labels] + noise
    data   = np.clip(data, 0, 1)
    return data, labels, templates


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Trains a VQ-VAE and integrates the learned token vocabulary
    with the SDTS Understanding Engine.
    """

    def __init__(self, model: VQVAE, verbose: bool = True):
        self.model   = model
        self.verbose = verbose

    def train(
        self,
        data        : np.ndarray,
        n_epochs    : int   = 50,
        batch_size  : int   = 32,
        log_every   : int   = 10,
        labels      : np.ndarray = None,
    ) -> list:
        """
        Train the VQ-VAE.

        data: (N, input_dim)
        returns: loss history
        """
        N = len(data)
        rng = np.random.default_rng(0)

        if self.verbose:
            print("=" * 60)
            print("VQ-VAE TRAINING")
            print("=" * 60)
            print(f"  Samples:    {N}")
            print(f"  Input dim:  {data.shape[1]}")
            print(f"  Epochs:     {n_epochs}")
            print(f"  Batch size: {batch_size}")
            print(f"  Tokens:     {self.model.n_tokens}")
            print()

        history = []

        for epoch in range(1, n_epochs + 1):
            idx = rng.permutation(N)
            epoch_losses = []

            for i in range(0, N, batch_size):
                batch = data[idx[i:i+batch_size]]
                metrics = self.model.step(batch)
                epoch_losses.append(metrics)

            # Restart dead codebook entries every 10 epochs
            # threshold=2: restart entries used <2 times in last epoch
            if epoch % 10 == 0 and epoch > 10:
                sample_idx = rng.integers(0, N, min(128, N))
                sample_data = data[sample_idx]
                z_e = self.model.encoder.forward(sample_data)
                self.model.quantizer.restart_dead_entries(z_e, threshold=2)
            # Reset usage counts at end of epoch for next epoch's tracking
            if epoch % 10 == 0:
                self.model.quantizer.reset_usage()

            avg = {
                k: np.mean([m[k] for m in epoch_losses])
                for k in ['total_loss', 'recon_loss',
                          'embedding_loss', 'commitment_loss',
                          'codebook_util', 'perplexity']
            }
            avg['epoch'] = epoch
            history.append(avg)

            if self.verbose and epoch % log_every == 0:
                print(f"  Epoch {epoch:3d}/{n_epochs}  "
                      f"loss={avg['total_loss']:.4f}  "
                      f"recon={avg['recon_loss']:.4f}  "
                      f"util={avg['codebook_util']:.0%}  "
                      f"ppl={avg['perplexity']:.1f}")

        if self.verbose:
            print()
            print(self.model.summary())
            print()

        return history

    def evaluate_tokenization(
        self,
        data    : np.ndarray,
        labels  : np.ndarray,
    ) -> dict:
        """
        Evaluate how well the VQ-VAE tokenizes the data.

        Checks:
        1. Token consistency: do same-class inputs get same tokens?
        2. Token diversity: are different classes using different tokens?
        3. Reconstruction quality: how well can we reconstruct from tokens?

        Returns a results dict.
        """
        tokens = self.model.encode_to_tokens(data)
        recon  = self.model.reconstruct(data)
        recon_err = ((recon - data) ** 2).mean()

        n_classes = len(np.unique(labels))

        # Per-class token distribution
        class_tokens = {}
        for c in range(n_classes):
            mask = (labels == c)
            class_tokens[c] = tokens[mask]

        # Token consistency: for each class, what's the most common token?
        # Consistency = fraction of samples that get that most common token
        consistency_scores = []
        dominant_tokens    = []
        for c in range(n_classes):
            toks = class_tokens[c]
            if len(toks) == 0:
                continue
            counts    = np.bincount(toks, minlength=self.model.n_tokens)
            dominant  = counts.argmax()
            score     = counts[dominant] / len(toks)
            consistency_scores.append(score)
            dominant_tokens.append(dominant)

        avg_consistency = np.mean(consistency_scores)
        n_unique_dominant = len(set(dominant_tokens))

        results = {
            'recon_mse'         : float(recon_err),
            'avg_consistency'   : float(avg_consistency),
            'n_unique_tokens'   : int((np.bincount(tokens, minlength=self.model.n_tokens) > 0).sum()),
            'n_classes'         : n_classes,
            'n_classes_with_unique_dominant': int(n_unique_dominant),
            'class_tokens'      : class_tokens,
            'dominant_tokens'   : dominant_tokens,
            'all_tokens'        : tokens,
        }
        return results

    def to_zce_token_streams(
        self,
        data         : np.ndarray,
        base_identity_offset : int = 17,
    ) -> list:
        """
        Convert data to ZCE token streams for SDTS engine.

        base_identity_offset: start identity for VQ-VAE tokens in ZCE vocab
            (17 = first available after math domain tokens 5..16,
             or just use 5 if starting fresh without math domain)

        Returns list of lists of ZCEToken objects.
        """
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from core import ZCEToken

        raw_tokens = self.model.encode_to_tokens(data)

        # Each sample → one token (or a sequence for spatial models)
        # For now: one token per sample, identity = offset + codebook_index
        streams = []
        for t in raw_tokens:
            token = ZCEToken(identity=base_identity_offset + t, depth=0)
            streams.append([token])
        return streams