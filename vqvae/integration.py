"""
vqvae/integration.py
====================
The integration of QuantizedAE with the SDTS Understanding Engine.

Full pipeline:
    Raw input (continuous)
        ↓
    QuantizedAE (PCA + K-means)
        ↓
    Discrete token stream (ZCE base tokens, depth=0)
        ↓
    SDTS Understanding Engine
        ↓
    Growing ZCE vocabulary (depth-1, depth-2, ...)
        ↓
    Compressed representation = "understanding"

This is the Symbol Grounding experiment:
    Do the symbols the SDTS engine discovers correspond to
    real structure in the continuous input?
"""

from __future__ import annotations
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Vocabulary, ZCEToken
from understanding_engine import UnderstandingEngine
from .quantized_ae import QuantizedAE


class ZCEInterface:
    """
    Full continuous-to-symbolic interface.
    Wraps a QuantizedAE and an SDTS Understanding Engine.
    """

    def __init__(
        self,
        input_dim      : int  = 64,
        n_components   : int  = 16,
        n_codebook     : int  = 16,
        min_frequency  : int  = 3,
        seed           : int  = 42,
    ):
        self.input_dim  = input_dim
        self.n_codebook = n_codebook

        self.qae = QuantizedAE(
            input_dim    = input_dim,
            n_components = n_components,
            n_tokens     = n_codebook,
            seed         = seed,
        )

        self.engine = UnderstandingEngine(
            min_frequency   = min_frequency,
            min_savings     = 0,
            max_pattern_len = 4,
        )

        self._base_offset = 5
        self._fitted      = False

    def _install_base_tokens(self):
        """Install one ZCE base token per codebook entry."""
        for i in range(self.n_codebook):
            name = f"T{i:02d}"
            n    = self._base_offset + i
            if not self.engine.vocabulary.has(name):
                self.engine.vocabulary.define(
                    name        = name,
                    components  = [],
                    description = f"Codebook token {i}",
                    identity    = n,
                )

    def fit(
        self,
        data    : np.ndarray,
        verbose : bool = True,
    ) -> dict:
        """Fit the QuantizedAE and install base tokens."""
        metrics = self.qae.fit(data, verbose=verbose)
        self._install_base_tokens()
        self._fitted = True
        return metrics

    def encode_to_streams(self, data: np.ndarray) -> list:
        """
        Encode continuous data to ZCE token streams.
        Each sample → one token (one-token stream).
        """
        raw_tokens = self.qae.encode(data)
        return [[ZCEToken(identity=self._base_offset + int(t), depth=0)]
                for t in raw_tokens]

    def run_sdts_cycles(
        self,
        data     : np.ndarray,
        n_cycles : int  = 5,
        verbose  : bool = True,
    ) -> dict:
        """Run SDTS compression cycles on encoded data."""
        streams = self.encode_to_streams(data)

        def total_dl(ss):
            return sum(sum(t.bit_length for t in s) for s in ss)

        results = {
            'vocab_sizes'        : [len(self.engine.vocabulary)],
            'description_lengths': [total_dl(streams)],
            'new_symbols'        : [],
        }

        if verbose:
            print("=" * 60)
            print("SDTS COMPRESSION CYCLES")
            print("=" * 60)
            print(f"Streams: {len(streams)}, Initial DL: {total_dl(streams):,} bits")
            print()

        for cycle in range(1, n_cycles + 1):
            for s in streams:
                self.engine.observe(s)
            new_syms = self.engine.compress_cycle(max_new_symbols=5, clear_after=True)
            streams  = [self.engine.rewrite(s) for s in streams]
            dl       = total_dl(streams)

            results['vocab_sizes'].append(len(self.engine.vocabulary))
            results['description_lengths'].append(dl)
            results['new_symbols'].append([s.name for s in new_syms])

            if verbose:
                prev  = results['description_lengths'][-2]
                saved = prev - dl
                pct   = 100 * saved / prev if prev > 0 else 0
                print(f"  Cycle {cycle}: {prev:,} → {dl:,} bits "
                      f"({saved:,} saved = {pct:.1f}%)  "
                      f"+{len(new_syms)} symbols")

        if verbose:
            print()
            print(self.engine.vocabulary.pretty_print())
            violations = self.engine.check_invariants()
            if violations:
                for v in violations:
                    print(f"  ✗ {v}")
            else:
                print("All invariants hold. ✓")

        return results

    def full_pipeline(
        self,
        data       : np.ndarray,
        labels     : np.ndarray = None,
        n_cycles   : int        = 5,
        verbose    : bool       = True,
    ) -> dict:
        """Complete pipeline: fit + evaluate + SDTS cycles."""

        if verbose:
            print("=" * 60)
            print("ZCE INTERFACE — FULL PIPELINE")
            print("=" * 60)

        fit_metrics = self.fit(data, verbose=verbose)

        eval_results = {}
        if labels is not None:
            if verbose:
                print("\nTokenization Evaluation:")
                print("-" * 40)
            eval_results = self.qae.evaluate(data, labels, verbose=verbose)
            if verbose:
                print()
                for c, (dom, cons) in enumerate(
                        zip(eval_results['dominant_tokens'],
                            eval_results['consistency_per_class'])):
                    print(f"  Class {c} → T{dom:02d}  ({cons:.0%})")
                print()

        sdts_results = self.run_sdts_cycles(data, n_cycles, verbose)
        violations   = self.engine.check_invariants()

        return {
            'fit_metrics'  : fit_metrics,
            'eval_results' : eval_results,
            'sdts_results' : sdts_results,
            'invariant_ok' : len(violations) == 0,
            'final_vocab'  : self.engine.vocabulary,
        }
