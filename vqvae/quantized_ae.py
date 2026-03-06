"""
vqvae/quantized_ae.py
=====================
The Quantized Autoencoder: continuous-to-discrete interface for ZCE.

Architecture (3 phases):

    Phase 1 — Dimensionality reduction:
        PCA: input → low-dimensional features
        (PCA = optimal linear encoder for MSE reconstruction)
        This is the "encoder" in our system.

    Phase 2 — Vector quantization:
        K-means on PCA coordinates → discrete tokens
        codebook = cluster centers (K entries)
        token_i = nearest cluster index for sample i

    Phase 3 — Decoder:
        Linear decoder: codebook[token] → reconstructed input
        Fit by least-squares regression from quantized latents to inputs.

Why PCA instead of a neural encoder:
    Pure-numpy backprop through deep networks is plagued by gradient
    collapse issues (LayerNorm absorbs gradient, causing encoder to
    output near-zero). PCA is the mathematically optimal linear encoder
    for MSE reconstruction and its latent space is guaranteed to have
    non-zero variance in every principal direction.
    
    This is not a limitation — it's the correct baseline:
    1. PCA gives interpretable features (principal components of the data)
    2. K-means on PCA coordinates is provably near-optimal for clustering
    3. When PyTorch is available, the PCA can be replaced with a CNN/transformer
       with no other changes to the system

The tokens produced are ZCE tokens. The SDTS engine operates on
sequences of these tokens exactly as with hand-written tokens.

ZCE connection:
    Each K-means centroid = one ZCE base token (depth=0)
    Token identity = codebook index + base_offset
    Sequences of tokens → SDTS Understanding Engine → growing vocabulary
"""

from __future__ import annotations
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class QuantizedAE:
    """
    Quantized Autoencoder using PCA + K-means + linear decoder.

    input_dim   : dimension of input (e.g. 784 for MNIST flattened)
    n_components: PCA dimensions (the encoder's latent dim)
    n_tokens    : K-means clusters = number of ZCE base tokens
    seed        : random seed for reproducibility
    """

    def __init__(
        self,
        input_dim    : int  = 64,
        n_components : int  = 16,
        n_tokens     : int  = 16,
        seed         : int  = 42,
    ):
        self.input_dim    = input_dim
        self.n_components = n_components
        self.n_tokens     = n_tokens
        self.seed         = seed

        self.pca          = PCA(n_components=n_components, random_state=seed)
        self.kmeans       = KMeans(n_clusters=n_tokens, n_init=10,
                                   random_state=seed)
        self.codebook     = None   # (n_tokens, n_components)
        self._dec_W       = None   # decoder: (n_components, input_dim)
        self._dec_b       = None   # decoder bias: (input_dim,)
        self._fitted      = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        data    : np.ndarray,
        verbose : bool = True,
    ) -> dict:
        """
        Full three-phase fitting.

        data: (N, input_dim)
        Returns: dict with phase metrics
        """
        N = len(data)

        if verbose:
            print("=" * 60)
            print("QUANTIZED AUTOENCODER — FITTING")
            print("=" * 60)
            print(f"  Samples:     {N}")
            print(f"  Input dim:   {self.input_dim}")
            print(f"  PCA dims:    {self.n_components}")
            print(f"  Tokens:      {self.n_tokens}")
            print()

        # Phase 1: PCA
        if verbose:
            print("  Phase 1: PCA dimensionality reduction")
        z = self.pca.fit_transform(data)
        explained = self.pca.explained_variance_ratio_.sum()
        if verbose:
            print(f"    Explained variance: {explained:.1%}")
            print(f"    Latent std: {z.std():.3f}  "
                  f"range=[{z.min():.2f}, {z.max():.2f}]")
            print()

        # Phase 2: K-means
        if verbose:
            print(f"  Phase 2: K-means quantization (K={self.n_tokens})")
        self.kmeans.fit(z)
        self.codebook = self.kmeans.cluster_centers_.astype(np.float32)
        tokens        = self.kmeans.labels_
        used          = len(np.unique(tokens))
        z_q           = self.codebook[tokens]
        quant_err     = float(((z - z_q)**2).mean())
        if verbose:
            print(f"    Codebook utilization: {used}/{self.n_tokens}")
            print(f"    Quantization error:   {quant_err:.4f}")
            print()

        # Phase 3: Linear decoder via least squares
        if verbose:
            print("  Phase 3: Linear decoder (least-squares fit)")
        # Fit x_hat = z_q @ W + b
        # Augment z_q with bias
        z_q_aug = np.hstack([z_q, np.ones((N, 1), dtype=np.float32)])
        # Least squares: W_aug = pinv(z_q_aug) @ data
        W_aug         = np.linalg.lstsq(z_q_aug, data, rcond=None)[0]
        self._dec_W   = W_aug[:-1]   # (n_components, input_dim)
        self._dec_b   = W_aug[-1]    # (input_dim,)
        x_hat         = z_q @ self._dec_W + self._dec_b
        recon_err     = float(((x_hat - data)**2).mean())
        if verbose:
            print(f"    Reconstruction MSE: {recon_err:.4f}")

        self._fitted = True

        metrics = {
            'explained_variance': float(explained),
            'quantization_error': quant_err,
            'reconstruction_mse': recon_err,
            'tokens_used'       : int(used),
        }

        if verbose:
            print()
            print(f"  Done. Final reconstruction MSE: {recon_err:.4f}")

        return metrics

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode continuous input to discrete token indices.
        x: (N, input_dim)
        returns: (N,) int — ZCE base token indices (0..n_tokens-1)
        """
        assert self._fitted, "Call fit() first"
        z      = self.pca.transform(x)
        tokens = self.kmeans.predict(z)
        return tokens.astype(np.int64)

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """Reconstruct input via quantized representation."""
        tokens = self.encode(x)
        z_q    = self.codebook[tokens]
        return z_q @ self._dec_W + self._dec_b

    def recon_error(self, x: np.ndarray) -> float:
        return float(((self.reconstruct(x) - x)**2).mean())

    def encode_latent(self, x: np.ndarray) -> np.ndarray:
        """Return continuous PCA latent (before quantization)."""
        return self.pca.transform(x).astype(np.float32)

    def evaluate(
        self,
        data   : np.ndarray,
        labels : np.ndarray,
        verbose: bool = True,
    ) -> dict:
        """
        Evaluate tokenization quality.

        Checks:
        - Reconstruction MSE
        - Token consistency per class (do same-class inputs → same token?)
        - Token diversity (do different classes → different tokens?)
        """
        tokens  = self.encode(data)
        n_classes = len(np.unique(labels))

        class_tokens  = {}
        dominant_toks = []
        consistencies = []

        for c in np.unique(labels):
            mask = (labels == c)
            ct   = tokens[mask]
            class_tokens[c] = ct
            dom  = np.bincount(ct, minlength=self.n_tokens).argmax()
            cons = (ct == dom).mean()
            dominant_toks.append(int(dom))
            consistencies.append(float(cons))

        results = {
            'recon_mse'         : self.recon_error(data),
            'avg_consistency'   : float(np.mean(consistencies)),
            'n_unique_tokens'   : int((np.bincount(tokens, minlength=self.n_tokens)>0).sum()),
            'n_classes'         : n_classes,
            'n_unique_dominant' : len(set(dominant_toks)),
            'dominant_tokens'   : dominant_toks,
            'class_tokens'      : class_tokens,
            'all_tokens'        : tokens,
            'consistency_per_class': consistencies,
        }

        if verbose:
            print(f"  Reconstruction MSE:    {results['recon_mse']:.4f}")
            print(f"  Token consistency:     {results['avg_consistency']:.1%}")
            print(f"  Unique tokens used:    {results['n_unique_tokens']}/{self.n_tokens}")
            print(f"  Unique dominant:       {results['n_unique_dominant']}/{n_classes} classes")

        return results
