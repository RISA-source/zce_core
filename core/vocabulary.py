"""
zce_core/core/vocabulary.py
===========================
THE VOCABULARY. GROWS MONOTONICALLY. NEVER SHRINKS.

The vocabulary is the accumulated set of self-locating symbols
found so far. It is the system's understanding made concrete.

Key invariants (enforced, never violated):
    1. Every symbol has a unique identity N
    2. Every symbol's depth M is correct:
           depth(s) = max(depth(component) for component in s.components) + 1
           depth(s) = 0 for base primitives
    3. No symbol is ever removed (monotonic growth)
    4. DEFINE is always in the base vocabulary (immutable)
    5. Base primitives (depth=0) are immutable

The vocabulary is the implementation of the Self-Extension
property (E) from SDTS. New symbols are defined via define(),
which corresponds to the DEFINE meta-transition in the theory.

The Frozen Decoder Property is maintained here:
    - define() adds data to the vocabulary
    - define() does NOT modify the decoder
    - The decoder reads the same three-step operation regardless
      of how large the vocabulary grows
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .token import ZCEToken


# ---------------------------------------------------------------------------
# A symbol entry in the vocabulary
# ---------------------------------------------------------------------------

@dataclass
class Symbol:
    """
    One entry in the vocabulary.

    token       : the ZCE token identifying this symbol
    name        : human-readable name (for readability — does not
                  affect any computation, purely for inspection)
    components  : list of token identities that compose this symbol
                  (empty for base primitives)
    action      : what this symbol does when executed
                  (callable, or None for abstract symbols)
    is_base     : True for symbols in V₀ — immutable primitives
    description : human-readable description of what this symbol does
    metadata    : any additional data the engine wants to store
    """
    token       : ZCEToken
    name        : str
    components  : list[int]         = field(default_factory=list)
    action      : Any               = field(default=None)
    is_base     : bool              = field(default=False)
    description : str               = field(default="")
    metadata    : dict              = field(default_factory=dict)

    def __repr__(self) -> str:
        comp = f" <- {self.components}" if self.components else ""
        return f"Symbol({self.name!r}, {self.token}{comp})"


# ---------------------------------------------------------------------------
# Base vocabulary token identities
# These are fixed. They correspond to N values (identity) in ZCE tokens.
# ---------------------------------------------------------------------------

class BaseTokens:
    """
    The immutable base vocabulary identities.
    These N values are reserved and cannot be redefined.

    The DEFINE token is the meta-transition. Its presence in the
    base vocabulary is what makes the Frozen Decoder Property provable:
    the mechanism for extension is itself immutable.
    """
    DEFINE  = 0   # meta-transition: extend the vocabulary
    SET     = 1   # set a slot to a value
    ADD     = 2   # add two slots
    JIF     = 3   # conditional jump (immutable)
    JMP     = 4   # unconditional jump (immutable)
    # 5..N are available for user-defined base primitives
    # Convention: base primitives use N=5..63
    # Learned symbols use N=64 and above

    IMMUTABLE = {DEFINE, JIF, JMP}   # cannot be redefined even by DEFINE
    BASE_MAX  = 7   # N=5 is DEFINE_OP, 6-7 reserved for future base                    # highest N reserved for base vocab


# ---------------------------------------------------------------------------
# The Vocabulary
# ---------------------------------------------------------------------------

class Vocabulary:
    """
    The accumulated set of self-locating symbols.

    Usage
    -----
    vocab = Vocabulary()
    vocab.define(name="MUL", components=[1, 2], action=mul_fn, depth=1)
    sym = vocab.lookup_by_name("MUL")
    sym = vocab.lookup_by_identity(64)
    """

    def __init__(self) -> None:
        # Map from identity (N) -> Symbol
        self._symbols: dict[int, Symbol] = {}
        # Map from name -> identity (N) for fast name lookup
        self._names: dict[str, int] = {}
        # Track the next available identity for learned symbols
        self._next_learned_identity: int = BaseTokens.BASE_MAX + 1
        # Total number of DEFINE operations applied
        self._define_count: int = 0

        # Install the base vocabulary
        self._install_base()

    # ------------------------------------------------------------------
    # Base vocabulary installation
    # ------------------------------------------------------------------

    def _install_base(self) -> None:
        """
        Install V₀ — the immutable base primitives.
        Called once at construction. Never called again.
        """
        base_symbols = [
            Symbol(
                token       = ZCEToken(identity=BaseTokens.DEFINE, depth=0),
                name        = "DEFINE",
                components  = [],
                is_base     = True,
                description = "Meta-transition: extend the vocabulary. IMMUTABLE.",
            ),
            Symbol(
                token       = ZCEToken(identity=BaseTokens.SET, depth=0),
                name        = "SET",
                components  = [],
                is_base     = True,
                description = "Set slot[dst] = value.",
            ),
            Symbol(
                token       = ZCEToken(identity=BaseTokens.ADD, depth=0),
                name        = "ADD",
                components  = [],
                is_base     = True,
                description = "Set slot[dst] = slot[a] + slot[b].",
            ),
            Symbol(
                token       = ZCEToken(identity=BaseTokens.JIF, depth=0),
                name        = "JIF",
                components  = [],
                is_base     = True,
                description = "If slot[cond] == 0: jump to offset. IMMUTABLE.",
            ),
            Symbol(
                token       = ZCEToken(identity=BaseTokens.JMP, depth=0),
                name        = "JMP",
                components  = [],
                is_base     = True,
                description = "Unconditional jump to offset. IMMUTABLE.",
            ),
        ]
        for sym in base_symbols:
            self._symbols[sym.token.identity] = sym
            self._names[sym.name] = sym.token.identity

    # ------------------------------------------------------------------
    # DEFINE — the only way to add symbols
    # ------------------------------------------------------------------

    def define(
        self,
        name        : str,
        components  : list[int],
        action      : Any    = None,
        description : str    = "",
        metadata    : dict   = None,
        identity    : int    = None,   # if None, auto-assigned
    ) -> Symbol:
        """
        Add a new symbol to the vocabulary. This is the DEFINE operation.

        The depth is computed automatically:
            depth = max(depth(component) for component in components) + 1
            depth = 0 if no components (should not happen for learned symbols)

        Parameters
        ----------
        name        : human-readable name
        components  : list of identity values (N) of component symbols
        action      : callable implementing this symbol's behavior
        description : human-readable description
        metadata    : arbitrary additional data
        identity    : explicit identity N (for base vocab extensions only)
                      if None, the next available learned identity is used

        Returns
        -------
        The newly created Symbol.

        Raises
        ------
        ValueError  if name already exists, if identity is reserved/taken,
                    or if any component is unknown.
        """
        # Validate name
        if name in self._names:
            raise ValueError(f"Symbol {name!r} already defined")

        # Validate and resolve components
        for comp_id in components:
            if comp_id not in self._symbols:
                raise ValueError(
                    f"Component identity {comp_id} not in vocabulary "
                    f"(cannot define {name!r})"
                )

        # Compute depth — this is the core invariant
        if components:
            max_component_depth = max(
                self._symbols[c].token.depth for c in components
            )
            depth = max_component_depth + 1
        else:
            # Base primitive being added to V₀ — depth 0
            depth = 0

        # Assign identity
        if identity is not None:
            # Explicit identity — validate it
            if identity in BaseTokens.IMMUTABLE:
                raise ValueError(
                    f"Cannot define symbol with identity {identity}: "
                    f"that identity is immutable (DEFINE/JIF/JMP)"
                )
            if identity in self._symbols:
                raise ValueError(
                    f"Identity {identity} already taken by "
                    f"{self._symbols[identity].name!r}"
                )
            n = identity
        else:
            # Auto-assign next available learned identity
            n = self._next_learned_identity
            self._next_learned_identity += 1

        # Create the token
        token = ZCEToken(identity=n, depth=depth)

        # Create the symbol
        sym = Symbol(
            token       = token,
            name        = name,
            components  = list(components),
            action      = action,
            is_base     = False,
            description = description,
            metadata    = metadata or {},
        )

        # Register
        self._symbols[n] = sym
        self._names[name] = n
        self._define_count += 1

        return sym

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup_by_identity(self, identity: int) -> Symbol | None:
        """Return the symbol with the given identity N, or None."""
        return self._symbols.get(identity)

    def lookup_by_name(self, name: str) -> Symbol | None:
        """Return the symbol with the given name, or None."""
        n = self._names.get(name)
        if n is None:
            return None
        return self._symbols.get(n)

    def lookup_by_token(self, token: ZCEToken) -> Symbol | None:
        """Return the symbol for a given ZCE token, or None."""
        return self._symbols.get(token.identity)

    def has(self, name: str) -> bool:
        return name in self._names

    def has_identity(self, identity: int) -> bool:
        return identity in self._symbols

    # ------------------------------------------------------------------
    # Depth queries
    # ------------------------------------------------------------------

    def symbols_at_depth(self, depth: int) -> list[Symbol]:
        """Return all symbols at a given depth, sorted by identity."""
        return sorted(
            [s for s in self._symbols.values() if s.token.depth == depth],
            key=lambda s: s.token.identity
        )

    @property
    def max_depth(self) -> int:
        """The current maximum depth in the vocabulary."""
        if not self._symbols:
            return 0
        return max(s.token.depth for s in self._symbols.values())

    # ------------------------------------------------------------------
    # Stream operations — the vocabulary IS the understanding
    # ------------------------------------------------------------------

    def to_token_stream(self) -> list[ZCEToken]:
        """
        Return the vocabulary as a ZCE token stream,
        ordered by depth then identity.
        This stream IS the accumulated understanding.
        It is decodable by the frozen decoder.
        """
        return sorted(
            [s.token for s in self._symbols.values()],
            key=lambda t: (t.depth, t.identity)
        )

    # ------------------------------------------------------------------
    # Invariant checking
    # ------------------------------------------------------------------

    def check_invariants(self) -> list[str]:
        """
        Check all vocabulary invariants.
        Returns a list of violation strings (empty = all good).
        """
        violations: list[str] = []

        for n, sym in self._symbols.items():
            # Identity consistency
            if sym.token.identity != n:
                violations.append(
                    f"{sym.name}: token identity {sym.token.identity} "
                    f"!= key {n}"
                )

            # Depth correctness
            if sym.components:
                expected_depth = max(
                    self._symbols[c].token.depth
                    for c in sym.components
                    if c in self._symbols
                ) + 1
                if sym.token.depth != expected_depth:
                    violations.append(
                        f"{sym.name}: depth {sym.token.depth} "
                        f"!= expected {expected_depth}"
                    )

            # Component existence
            for c in sym.components:
                if c not in self._symbols:
                    violations.append(
                        f"{sym.name}: component {c} not in vocabulary"
                    )

            # Base immutability
            if n in BaseTokens.IMMUTABLE and not sym.is_base:
                violations.append(
                    f"Identity {n} is immutable but symbol "
                    f"{sym.name!r} is not marked as base"
                )

        return violations

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._symbols)

    def __repr__(self) -> str:
        return (
            f"Vocabulary("
            f"size={len(self)}, "
            f"max_depth={self.max_depth}, "
            f"defines={self._define_count})"
        )

    def pretty_print(self) -> str:
        """Human-readable vocabulary listing, ordered by depth."""
        lines = ["Vocabulary:"]
        for depth in range(self.max_depth + 1):
            syms = self.symbols_at_depth(depth)
            if syms:
                lines.append(f"  Depth {depth}:")
                for s in syms:
                    comp = f" <- [{', '.join(str(c) for c in s.components)}]" \
                           if s.components else ""
                    base = " [BASE]" if s.is_base else ""
                    lines.append(
                        f"    {s.token}  {s.name!r}{comp}{base}"
                    )
        return "\n".join(lines)
