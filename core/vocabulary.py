"""
zce_core/core/vocabulary.py
===========================
THE VOCABULARY. GROWS MONOTONICALLY. NEVER SHRINKS.

Key invariants (enforced, never violated):
    1. Every symbol has a unique identity N
    2. Every symbol's depth is correct:
           depth(s) = max(depth(component) for component in components) + 1
           depth(s) = 0 for base primitives
    3. No symbol is ever removed (monotonic growth)
    4. DEFINE is always in the base vocabulary (immutable)
    5. Base primitives (depth=0) are immutable
    6. No two symbols share an identity — ever

IDENTITY ASSIGNMENT:
    Base vocab (core):   N = 0..4  (DEFINE, SET, ADD, JIF, JMP)
    Base vocab (domain): N = 5..N  via define(identity=N, components=[])
    Learned symbols:     N = next_learned_identity, auto-incremented

    next_learned_identity starts at 256.
    This reserves N=0..255 for hand-written base vocabularies.
    Learned symbols get N=256, 257, 258...

    Why 256? It's large enough for any reasonable domain base vocab
    (256 primitive operations), and still gives compact learned
    identities since N=256 -> bit_length = 256+depth+2 which is
    large. BUT: in the math domain, base tokens have N=5..16
    and the first learned token has N=256+0=256.

    Wait — that would make learned tokens VERY expensive (258+ bits).
    Better approach: track the high-water mark of explicitly assigned
    identities and start learned symbols just above that.

    ACTUAL STRATEGY: next_learned_identity = max(all assigned N) + 1.
    This is updated every time define() is called with an explicit identity.
    Learned symbols always get N just above the highest explicitly assigned N.
    
    For math domain: base N goes up to 16 (COMMA). So first learned = 17.
    A learned token at N=17, depth=1: bit_length = 17+1+2 = 20 bits.
    Three base tokens (avg N=10, depth=0): 3*(10+0+2) = 36 bits.
    Savings = 36-20 = 16 bits per use. Compression appears clearly.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from .token import ZCEToken


@dataclass
class Symbol:
    """One entry in the vocabulary."""
    token       : ZCEToken
    name        : str
    components  : list[int]  = field(default_factory=list)
    action      : Any        = field(default=None)
    is_base     : bool       = field(default=False)
    description : str        = field(default="")
    metadata    : dict       = field(default_factory=dict)
    use_count   : int        = field(default=0)

    def __repr__(self) -> str:
        comp = f" <- {self.components}" if self.components else ""
        return f"Symbol({self.name!r}, {self.token}{comp})"


class BaseTokens:
    """Reserved identity constants for the CORE base vocabulary."""
    DEFINE  = 0
    SET     = 1
    ADD     = 2
    JIF     = 3
    JMP     = 4

    IMMUTABLE = {DEFINE, JIF, JMP}
    # Core base vocab uses N=0..4.
    # Domain base vocabs use N=5..N via define(identity=N).
    # Learned symbols start just above the highest assigned identity.
    CORE_MAX = 4


class Vocabulary:
    """
    The accumulated set of self-locating symbols.
    Grows monotonically. Never shrinks.
    """

    def __init__(self) -> None:
        self._symbols   : dict[int, Symbol] = {}
        self._names     : dict[str, int]    = {}
        # High-water mark: the highest identity explicitly assigned so far.
        # Learned symbols start at _hwm + 1.
        self._hwm       : int = BaseTokens.CORE_MAX
        self._define_count : int = 0
        self._install_base()

    # ------------------------------------------------------------------
    # Base vocabulary
    # ------------------------------------------------------------------

    def _install_base(self) -> None:
        base = [
            (BaseTokens.DEFINE, "DEFINE", "Meta-transition: extend vocabulary. IMMUTABLE."),
            (BaseTokens.SET,    "SET",    "Set slot[dst] = value."),
            (BaseTokens.ADD,    "ADD",    "Set slot[dst] = slot[a] + slot[b]."),
            (BaseTokens.JIF,    "JIF",    "If slot[cond]==0: jump. IMMUTABLE."),
            (BaseTokens.JMP,    "JMP",    "Unconditional jump. IMMUTABLE."),
        ]
        for n, name, desc in base:
            sym = Symbol(
                token       = ZCEToken(identity=n, depth=0),
                name        = name,
                components  = [],
                is_base     = True,
                description = desc,
            )
            self._symbols[n] = sym
            self._names[name] = n
        # _hwm stays at CORE_MAX = 4 until domain base vocab is added

    # ------------------------------------------------------------------
    # DEFINE — the only way to add symbols
    # ------------------------------------------------------------------

    def define(
        self,
        name        : str,
        components  : list[int],
        action      : Any  = None,
        description : str  = "",
        metadata    : dict = None,
        identity    : int  = None,
    ) -> Symbol:
        """
        Add a new symbol. This is the DEFINE meta-transition.

        Depth = max(depth(component) for c in components) + 1,
                or 0 if components is empty (base primitive).

        Identity assignment:
            If identity is specified explicitly: use that N.
                Also update the high-water mark so learned symbols
                don't collide with it.
            If identity is None (learned symbol):
                Use _hwm + 1, increment _hwm.

        Invariant: no two symbols ever share an identity.
        """
        if name in self._names:
            raise ValueError(f"Symbol {name!r} already defined")

        for c in components:
            if c not in self._symbols:
                raise ValueError(
                    f"Component identity {c} not in vocabulary "
                    f"(defining {name!r})"
                )

        depth = (max(self._symbols[c].token.depth for c in components) + 1
                 if components else 0)

        # Assign identity
        if identity is not None:
            if identity in BaseTokens.IMMUTABLE:
                raise ValueError(f"Identity {identity} is immutable")
            if identity in self._symbols:
                raise ValueError(
                    f"Identity {identity} already taken by "
                    f"{self._symbols[identity].name!r}"
                )
            n = identity
            # Update high-water mark — learned symbols must stay above this
            if n > self._hwm:
                self._hwm = n
        else:
            # Auto-assign: first integer above the high-water mark
            self._hwm += 1
            n = self._hwm

        sym = Symbol(
            token       = ZCEToken(identity=n, depth=depth),
            name        = name,
            components  = list(components),
            action      = action,
            is_base     = False,
            description = description,
            metadata    = metadata or {},
        )
        self._symbols[n] = sym
        self._names[name] = n
        self._define_count += 1
        return sym

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    def record_use(self, identity: int) -> None:
        if identity in self._symbols:
            self._symbols[identity].use_count += 1

    def usage_sorted(self) -> list[Symbol]:
        return sorted(
            self._symbols.values(),
            key=lambda s: s.use_count,
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup_by_identity(self, identity: int) -> Symbol | None:
        return self._symbols.get(identity)

    def lookup_by_name(self, name: str) -> Symbol | None:
        n = self._names.get(name)
        return self._symbols.get(n) if n is not None else None

    def lookup_by_token(self, token: ZCEToken) -> Symbol | None:
        return self._symbols.get(token.identity)

    def has(self, name: str) -> bool:
        return name in self._names

    def has_identity(self, identity: int) -> bool:
        return identity in self._symbols

    # ------------------------------------------------------------------
    # Depth queries
    # ------------------------------------------------------------------

    def symbols_at_depth(self, depth: int) -> list[Symbol]:
        return sorted(
            [s for s in self._symbols.values() if s.token.depth == depth],
            key=lambda s: s.token.identity,
        )

    @property
    def max_depth(self) -> int:
        if not self._symbols:
            return 0
        return max(s.token.depth for s in self._symbols.values())

    @property
    def next_learned_identity(self) -> int:
        """The N that the next auto-assigned symbol will receive."""
        return self._hwm + 1

    # ------------------------------------------------------------------
    # Stream
    # ------------------------------------------------------------------

    def to_token_stream(self) -> list[ZCEToken]:
        return sorted(
            [s.token for s in self._symbols.values()],
            key=lambda t: (t.depth, t.identity),
        )

    # ------------------------------------------------------------------
    # Invariant checking
    # ------------------------------------------------------------------

    def check_invariants(self) -> list[str]:
        violations = []
        seen_identities: set[int] = set()
        for n, sym in self._symbols.items():
            # Identity uniqueness
            if n in seen_identities:
                violations.append(f"Duplicate identity N={n}")
            seen_identities.add(n)
            # Token identity matches key
            if sym.token.identity != n:
                violations.append(
                    f"{sym.name}: token.identity {sym.token.identity} != key {n}"
                )
            # Component existence and depth correctness
            if sym.components:
                missing = [c for c in sym.components if c not in self._symbols]
                if missing:
                    violations.append(
                        f"{sym.name}: missing components {missing}"
                    )
                else:
                    expected_depth = (
                        max(self._symbols[c].token.depth for c in sym.components) + 1
                    )
                    if sym.token.depth != expected_depth:
                        violations.append(
                            f"{sym.name}: depth {sym.token.depth} "
                            f"!= expected {expected_depth}"
                        )
            # Immutability
            if n in BaseTokens.IMMUTABLE and not sym.is_base:
                violations.append(
                    f"Identity {n} is immutable but symbol {sym.name!r} "
                    f"is not marked as base"
                )
        return violations

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._symbols)

    def __repr__(self) -> str:
        return (f"Vocabulary(size={len(self)}, "
                f"max_depth={self.max_depth}, "
                f"hwm={self._hwm})")

    def pretty_print(self) -> str:
        lines = ["Vocabulary:"]
        for depth in range(self.max_depth + 1):
            syms = self.symbols_at_depth(depth)
            if not syms:
                continue
            lines.append(f"  Depth {depth}:")
            for s in syms:
                comp_names = (
                    [self._symbols[c].name for c in s.components
                     if c in self._symbols]
                    if s.components else []
                )
                base  = "  [BASE]" if s.is_base else ""
                comp  = f"  =  {' ; '.join(comp_names)}" if comp_names else ""
                uses  = f"  (used {s.use_count}x)" if s.use_count else ""
                lines.append(
                    f"    {s.token}  {s.name:<22} "
                    f"[{s.token.bit_length}b]{base}{comp}{uses}"
                )
        return "\n".join(lines)