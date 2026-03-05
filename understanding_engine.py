"""
zce_core/understanding_engine.py
=================================
THE UNDERSTANDING ENGINE.

This is the top-level system. It implements the core loop:

    observe → measure symbol-gap → find shortest self-locating symbol
            → add to vocab → repeat

All five conceptual subsystems unify under one objective:
    find self-locating symbols for recurring structure.

    Surprise    = you don't have a symbol for this yet
    Compression = find the shortest symbol that covers it
    Relation    = two things share a symbol or one references the other
    Question    = you have a gap where a symbol should be
    Vocab       = the accumulated set of symbols found so far

This engine does not decide WHAT counts as a symbol.
The compression objective decides that automatically.

The engine is the implementation of:
    "Understanding is finding the shortest self-locating symbol
     for a recurring structure."
"""

from __future__ import annotations
from typing import Any, Callable
from core   import ZCEToken, FrozenDecoder, Vocabulary, Symbol, BaseTokens
from engine import TransitionCalculus, ExecutionState, Compressor


class UnderstandingEngine:
    """
    The top-level ZCE understanding engine.

    Usage
    -----
    engine = UnderstandingEngine()

    # Register actions for base symbols
    engine.register_action("ADD", my_add_fn)

    # Observe token streams (raw experience)
    engine.observe([tok1, tok2, tok3, ...])

    # Run one compression cycle — find and define new symbols
    new_symbols = engine.compress_cycle()

    # Execute a token stream
    result = engine.execute([tok1, tok2], initial={"x": 5})

    # Inspect understanding
    print(engine.vocabulary.pretty_print())
    print(f"Current depth: {engine.vocabulary.max_depth}")
    print(f"Description length: {engine.description_length_of(tokens)}")
    """

    def __init__(
        self,
        min_frequency   : int = 2,
        min_savings     : int = 1,
        max_pattern_len : int = 8,
    ) -> None:
        # The frozen core
        self._vocab    = Vocabulary()
        self._decoder  = FrozenDecoder()

        # The engine layer
        self._calculus  = TransitionCalculus(self._vocab)
        self._compressor = Compressor(
            vocabulary      = self._vocab,
            min_frequency   = min_frequency,
            min_savings     = min_savings,
            max_pattern_len = max_pattern_len,
        )

        # Statistics
        self._cycles_run        : int = 0
        self._total_observations: int = 0
        self._total_defined     : int = 0

    # ------------------------------------------------------------------
    # Registration — connect actions to vocabulary symbols
    # ------------------------------------------------------------------

    def register_action(
        self,
        name_or_identity: str | int,
        fn: Callable[[ExecutionState], ExecutionState],
    ) -> None:
        """Register an executable function for a named vocabulary symbol."""
        if isinstance(name_or_identity, str):
            sym = self._vocab.lookup_by_name(name_or_identity)
            if sym is None:
                raise ValueError(f"No symbol named {name_or_identity!r}")
            identity = sym.token.identity
        else:
            identity = name_or_identity
        self._calculus.register(identity, fn)

    def define_base(
        self,
        name        : str,
        identity    : int,
        fn          : Callable,
        description : str = "",
    ) -> Symbol:
        """
        Add a new base primitive to the vocabulary.
        Use this to extend V₀ before learning begins.
        identity must be in range [5, BaseTokens.BASE_MAX].
        """
        if identity in BaseTokens.IMMUTABLE:
            raise ValueError(f"Identity {identity} is immutable")
        sym = self._vocab.define(
            name        = name,
            components  = [],       # base primitive — no components
            action      = fn,
            description = description,
            identity    = identity,
        )
        self._calculus.register(sym.token.identity, fn)
        return sym

    # ------------------------------------------------------------------
    # Observation — raw experience enters here
    # ------------------------------------------------------------------

    def observe(self, tokens: list[ZCEToken]) -> None:
        """
        Feed a token stream to the engine as an observation.
        Observations accumulate until compress_cycle() is called.
        """
        self._compressor.observe(tokens)
        self._total_observations += 1

    def observe_bits(self, bits: list[int]) -> None:
        """Decode a bit stream and observe the resulting tokens."""
        self._decoder.clear()
        self._decoder.feed(bits)
        tokens = self._decoder.decode_all()
        self.observe(tokens)

    # ------------------------------------------------------------------
    # Compression cycle — the core learning step
    # ------------------------------------------------------------------

    def compress_cycle(
        self,
        max_new_symbols : int = 10,
        clear_after     : bool = True,
    ) -> list[Symbol]:
        """
        Run one compression cycle:
            1. Find the best compression candidates
            2. Define the top candidates as new vocabulary symbols
            3. Optionally clear the observation buffer

        Returns the list of newly defined symbols.

        This is the admissible learning mechanism ℳ from Definition 2.
        """
        candidates = self._compressor.find_candidates()
        new_symbols: list[Symbol] = []

        for candidate in candidates[:max_new_symbols]:
            # Check we haven't already defined this exact pattern
            existing = self._find_existing_for_pattern(candidate.pattern)
            if existing:
                continue

            sym = self._compressor.apply_candidate(candidate)
            new_symbols.append(sym)
            self._total_defined += 1

        if clear_after:
            self._compressor.clear_observations()

        self._cycles_run += 1
        return new_symbols

    def _find_existing_for_pattern(
        self, pattern: tuple[int, ...]
    ) -> Symbol | None:
        """Check if a symbol already exists for exactly this component pattern."""
        for sym in self._vocab._symbols.values():
            if tuple(sym.components) == pattern:
                return sym
        return None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        tokens        : list[ZCEToken],
        initial_state : dict[str, Any] = None,
    ) -> ExecutionState:
        """
        Execute a token stream and return the final state.

        The Readability Theorem guarantees: the computation is
        fully recoverable from the token sequence. Reading is executing.
        """
        state = ExecutionState(initial_state or {})
        return self._calculus.execute_stream(tokens, state)

    # ------------------------------------------------------------------
    # Rewrite — compress a stream using current vocabulary
    # ------------------------------------------------------------------

    def rewrite(self, tokens: list[ZCEToken]) -> list[ZCEToken]:
        """
        Rewrite a token stream using the current vocabulary.
        Replaces known patterns with their defined symbols.
        The rewritten stream is shorter (or equal) in bit length.
        """
        return self._compressor.rewrite(tokens)

    # ------------------------------------------------------------------
    # Description length — tracks understanding progress
    # ------------------------------------------------------------------

    def description_length_of(self, tokens: list[ZCEToken]) -> int:
        """
        Compute the description length of a token stream.
        As the vocabulary grows, this should decrease for
        streams the engine has observed before.

        Decreasing description length = increasing understanding.
        """
        return self._compressor.description_length(tokens)

    # ------------------------------------------------------------------
    # Surprise — how unknown is this token stream?
    # ------------------------------------------------------------------

    def surprise(self, tokens: list[ZCEToken]) -> float:
        """
        Measure surprise: the fraction of token identities in the
        stream that have no symbol in the current vocabulary.

        surprise = 0.0  -> fully known (zero surprise)
        surprise = 1.0  -> completely unknown

        "Surprise = you don't have a symbol for this yet."
        """
        if not tokens:
            return 0.0
        unknown = sum(
            1 for t in tokens
            if not self._vocab.has_identity(t.identity)
        )
        return unknown / len(tokens)

    # ------------------------------------------------------------------
    # Vocabulary access
    # ------------------------------------------------------------------

    @property
    def vocabulary(self) -> Vocabulary:
        return self._vocab

    @property
    def decoder(self) -> FrozenDecoder:
        return self._decoder

    @property
    def calculus(self) -> TransitionCalculus:
        return self._calculus

    # ------------------------------------------------------------------
    # Invariant checking
    # ------------------------------------------------------------------

    def check_invariants(self) -> list[str]:
        """
        Check all system invariants. Returns violations (empty = healthy).

        Checks:
            - Vocabulary depth invariants
            - No immutable symbols redefined
            - Frozen decoder state consistency
        """
        violations = self._vocab.check_invariants()

        # Check DEFINE is still in base vocab and immutable
        define_sym = self._vocab.lookup_by_name("DEFINE")
        if define_sym is None:
            violations.append("DEFINE symbol missing from vocabulary")
        elif not define_sym.is_base:
            violations.append("DEFINE symbol is no longer marked as base")
        elif define_sym.token.identity != BaseTokens.DEFINE:
            violations.append(
                f"DEFINE identity changed: "
                f"expected {BaseTokens.DEFINE}, "
                f"got {define_sym.token.identity}"
            )

        return violations

    # ------------------------------------------------------------------
    # Statistics and inspection
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "vocab_size"          : len(self._vocab),
            "max_depth"           : self._vocab.max_depth,
            "cycles_run"          : self._cycles_run,
            "total_observations"  : self._total_observations,
            "total_defined"       : self._total_defined,
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"UnderstandingEngine("
            f"vocab={s['vocab_size']}, "
            f"depth={s['max_depth']}, "
            f"cycles={s['cycles_run']})"
        )
