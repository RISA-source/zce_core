"""
zce_core/engine/compressor.py
==============================
THE COMPRESSION ENGINE.

The objective: find the shortest self-locating symbol for a
recurring structure.

This is the learning mechanism — the admissible ℳ from Definition 2.
It observes token streams, finds recurring subsequences, and
proposes new vocabulary symbols that compress those subsequences.

The compression criterion: a new symbol is worth defining if the
ZCE stream using it is strictly shorter than the stream without it.

This connects to the Complexity Conjecture:
    K_Σ(f, ε) → K(f) + c_Σ as ε → 0
The compressor is the mechanism that drives K_Σ downward.

Engineering note on the three unsolved problems:
    Problem 1 (search efficiency): currently uses frequency-based
        candidate generation. This is a placeholder. Better search
        strategies (beam search, wake-sleep, evolutionary) will be
        explored empirically.
    Problem 2 (discretization threshold): controlled by min_frequency
        parameter. Needs empirical tuning.
    Problem 3 (cross-domain vocabulary): not addressed here.
        This compressor works within a single token domain.
"""

from __future__ import annotations
from collections import Counter, defaultdict
from typing import Iterator
from core import ZCEToken, Vocabulary, Symbol


# ---------------------------------------------------------------------------
# A compression candidate — a proposed new symbol
# ---------------------------------------------------------------------------

class CompressionCandidate:
    """
    A proposed new symbol to add to the vocabulary.

    Represents the finding: "this subsequence of tokens appears
    frequently enough that naming it would shorten the stream."
    """

    def __init__(
        self,
        pattern     : tuple[int, ...],   # sequence of identity values
        frequency   : int,
        savings     : int,               # bits saved if this is defined
        vocabulary  : Vocabulary,
    ) -> None:
        self.pattern   = pattern
        self.frequency = frequency
        self.savings   = savings
        self._vocab    = vocabulary

    @property
    def component_identities(self) -> list[int]:
        return list(self.pattern)

    @property
    def proposed_depth(self) -> int:
        """The depth this symbol would have if defined."""
        if not self.pattern:
            return 0
        depths = []
        for n in self.pattern:
            sym = self._vocab.lookup_by_identity(n)
            if sym:
                depths.append(sym.token.depth)
        return (max(depths) + 1) if depths else 1

    def __repr__(self) -> str:
        names = []
        for n in self.pattern:
            sym = self._vocab.lookup_by_identity(n)
            names.append(sym.name if sym else str(n))
        return (
            f"Candidate([{' ; '.join(names)}], "
            f"freq={self.frequency}, savings={self.savings}b)"
        )


# ---------------------------------------------------------------------------
# The Compressor
# ---------------------------------------------------------------------------

class Compressor:
    """
    Finds recurring patterns in token streams and proposes
    new vocabulary symbols that would compress those streams.

    The core loop (called by the Understanding Engine):
        1. Observe a token stream
        2. Find recurring subsequences (n-grams of token identities)
        3. Score each by compression savings
        4. Return candidates above threshold
    """

    def __init__(
        self,
        vocabulary      : Vocabulary,
        min_frequency   : int   = 2,     # Problem 2: discretization threshold
        min_savings     : int   = 1,     # minimum bits saved to be worthwhile
        max_pattern_len : int   = 8,     # maximum pattern length to consider
    ) -> None:
        self._vocab          = vocabulary
        self.min_frequency   = min_frequency
        self.min_savings     = min_savings
        self.max_pattern_len = max_pattern_len

        # Observation buffer — accumulated token streams
        self._observations: list[list[int]] = []

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def observe(self, tokens: list[ZCEToken]) -> None:
        """
        Add a token stream to the observation buffer.
        Observations accumulate until compress() is called.
        """
        self._observations.append([t.identity for t in tokens])

    def clear_observations(self) -> None:
        """Clear the observation buffer."""
        self._observations = []

    # ------------------------------------------------------------------
    # Compression search — Problem 1 territory
    # ------------------------------------------------------------------

    def find_candidates(self) -> list[CompressionCandidate]:
        """
        Find all compression candidates in the current observation buffer.

        Strategy (current — frequency-based, Problem 1 placeholder):
            For each n-gram length from 2 to max_pattern_len:
                Count frequencies of all n-grams across all observations
                Score each by compression savings
                Keep those above threshold

        Returns candidates sorted by savings (best first).
        """
        if not self._observations:
            return []

        # Count n-gram frequencies across all observations
        ngram_counts: Counter = Counter()

        for obs in self._observations:
            for length in range(2, min(self.max_pattern_len + 1, len(obs) + 1)):
                for start in range(len(obs) - length + 1):
                    ngram = tuple(obs[start:start + length])
                    ngram_counts[ngram] += 1

        # Score candidates
        candidates: list[CompressionCandidate] = []

        for ngram, freq in ngram_counts.items():
            if freq < self.min_frequency:
                continue

            # Skip patterns containing undefined tokens
            valid = all(self._vocab.has_identity(n) for n in ngram)
            if not valid:
                continue

            # Skip patterns that are already a single defined symbol
            if len(ngram) == 1:
                continue

            # Compute compression savings
            # Current cost: sum of bit lengths of individual tokens
            current_cost = sum(
                ZCEToken(
                    identity=n,
                    depth=self._vocab.lookup_by_identity(n).token.depth
                ).bit_length
                for n in ngram
            )

            # Proposed depth for the new symbol
            depths = [
                self._vocab.lookup_by_identity(n).token.depth
                for n in ngram
            ]
            new_depth = max(depths) + 1

            # The new symbol would need a new identity — use a placeholder
            # identity to estimate its bit length
            # (actual identity assigned at define time)
            estimated_new_identity = self._vocab._next_learned_identity
            new_token_cost = ZCEToken(
                identity=estimated_new_identity,
                depth=new_depth
            ).bit_length

            # Savings per occurrence
            savings_per = current_cost - new_token_cost
            # Total savings: (savings per use * frequency) - define overhead
            # Define overhead: cost of writing the new symbol's token once
            # (the DEFINE meta-token + the new symbol label).
            # Conservative estimate: 2 * new_token_cost (once to define, once overhead)
            define_overhead = new_token_cost * 2
            total_savings = (savings_per * freq) - define_overhead

            if total_savings >= self.min_savings:
                candidates.append(CompressionCandidate(
                    pattern    = ngram,
                    frequency  = freq,
                    savings    = total_savings,
                    vocabulary = self._vocab,
                ))

        # Sort by savings descending
        candidates.sort(key=lambda c: c.savings, reverse=True)
        return candidates

    def best_candidate(self) -> CompressionCandidate | None:
        """Return the single best compression candidate, or None."""
        candidates = self.find_candidates()
        return candidates[0] if candidates else None

    # ------------------------------------------------------------------
    # Apply a candidate — actually define the new symbol
    # ------------------------------------------------------------------

    def apply_candidate(
        self,
        candidate   : CompressionCandidate,
        name        : str = None,
    ) -> Symbol:
        """
        Define the new symbol represented by a compression candidate.

        This calls vocabulary.define() which is the DEFINE meta-transition.
        The depth is computed automatically by the vocabulary.
        """
        if name is None:
            # Auto-generate a name based on components
            parts = []
            for n in candidate.pattern:
                sym = self._vocab.lookup_by_identity(n)
                parts.append(sym.name if sym else str(n))
            name = "_".join(parts)
            # Make unique if needed
            base_name = name
            counter = 1
            while self._vocab.has(name):
                name = f"{base_name}_{counter}"
                counter += 1

        return self._vocab.define(
            name       = name,
            components = candidate.component_identities,
            description = f"Learned by compression (freq={candidate.frequency})",
        )

    # ------------------------------------------------------------------
    # Rewrite a token stream using the current vocabulary
    # ------------------------------------------------------------------

    def rewrite(self, tokens: list[ZCEToken]) -> list[ZCEToken]:
        """
        Rewrite a token stream replacing known patterns with their
        defined symbols where possible.

        This is greedy — finds the longest matching defined symbol
        at each position.
        """
        identities = [t.identity for t in tokens]
        result_ids: list[int] = []
        pos = 0

        while pos < len(identities):
            # Try to match longest pattern first
            best_match_len = 1
            best_match_id  = identities[pos]

            # Check all symbols to find one whose components match here
            for sym_id, sym in self._vocab._symbols.items():
                if not sym.components:
                    continue
                comp = sym.components
                end  = pos + len(comp)
                if end <= len(identities) and identities[pos:end] == comp:
                    if len(comp) > best_match_len:
                        best_match_len = len(comp)
                        best_match_id  = sym_id

            result_ids.append(best_match_id)
            pos += best_match_len

        # Convert back to tokens
        result = []
        for n in result_ids:
            sym = self._vocab.lookup_by_identity(n)
            if sym:
                result.append(sym.token)
        return result

    # ------------------------------------------------------------------
    # Description length of a stream
    # ------------------------------------------------------------------

    def description_length(self, tokens: list[ZCEToken]) -> int:
        """Total bit length of a token stream."""
        return sum(t.bit_length for t in tokens)

    def __repr__(self) -> str:
        return (
            f"Compressor("
            f"observations={len(self._observations)}, "
            f"min_freq={self.min_frequency})"
        )
