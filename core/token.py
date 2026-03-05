"""
zce_core/core/token.py  — THE FROZEN PRIMITIVE. THIS FILE DOES NOT CHANGE.

ZCE token: [N zeros][M ones][terminating zero]

Where:
    N = identity  (what this symbol IS) — non-negative integer
    M = depth + 1 (where it was CONSTRUCTED) — positive integer >= 1

The M field encodes depth+1 so that M is always >= 1. This guarantees
the format is unambiguously self-delimiting: zeros end at first 1,
ones end at next 0, that 0 is the terminator. No ambiguity in any stream.

API uses 'depth' (0-indexed). Internally M = depth + 1.
bit_length = N + (depth+1) + 1 = N + depth + 2.

Examples (N=identity, depth):
    (0, 0) -> M=1 -> [1,0]           2 bits
    (1, 0) -> M=1 -> [0,1,0]         3 bits
    (0, 1) -> M=2 -> [1,1,0]         3 bits
    (1, 1) -> M=2 -> [0,1,1,0]       4 bits
    (2, 3) -> M=4 -> [0,0,1,1,1,1,0] 7 bits
    (N, d) ->        N + d + 2 bits
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator

MAX_IDENTITY = 65535
MAX_DEPTH    = 65535


@dataclass(frozen=True)
class ZCEToken:
    """
    Immutable ZCE token.

    identity : int — what this symbol IS (N zeros in stream)
    depth    : int — structural depth (encodes as depth+1 ones in stream)

    Stream format: [N zeros][(depth+1) ones][terminating zero]
    bit_length = N + depth + 2

    The +1 offset ensures M >= 1 always, making the format
    unambiguously self-delimiting with no edge cases.
    """
    identity : int
    depth    : int

    def __post_init__(self) -> None:
        if not isinstance(self.identity, int) or self.identity < 0:
            raise ValueError(f"identity must be non-negative int, got {self.identity!r}")
        if not isinstance(self.depth, int) or self.depth < 0:
            raise ValueError(f"depth must be non-negative int, got {self.depth!r}")
        if self.identity > MAX_IDENTITY:
            raise ValueError(f"identity {self.identity} > MAX_IDENTITY {MAX_IDENTITY}")
        if self.depth > MAX_DEPTH:
            raise ValueError(f"depth {self.depth} > MAX_DEPTH {MAX_DEPTH}")

    # --- Encoding ---

    def to_bits(self) -> list[int]:
        """
        Encode as [N zeros][(depth+1) ones][terminating zero].
        M = depth + 1 >= 1 always.
        """
        bits: list[int] = []
        bits.extend([0] * self.identity)       # N zeros — identity
        bits.extend([1] * (self.depth + 1))    # depth+1 ones — depth marker
        bits.append(0)                          # terminating zero
        return bits

    @property
    def bit_length(self) -> int:
        """N + (depth+1) + 1 = N + depth + 2."""
        return self.identity + self.depth + 2

    # --- Decoding: THE FROZEN THREE-STEP OPERATION ---

    @classmethod
    def from_bits(cls, bits: list[int], pos: int = 0) -> tuple[ZCEToken, int]:
        """
        Decode one token. Returns (token, new_pos).

        The frozen decoder — three operations, always the same:
            1. Count leading zeros -> N (identity)
            2. Count following ones -> depth+1  =>  depth = ones - 1
            3. Read terminating zero

        Since M = depth+1 >= 1, there is always at least one '1' bit.
        This makes the boundary between zeros and ones unambiguous.
        """
        length = len(bits)
        if pos >= length:
            raise ValueError(f"Stream exhausted at pos {pos}")

        # Step 1: Count identity zeros (stop at first 1)
        n = 0
        while pos < length and bits[pos] == 0:
            n += 1
            pos += 1
            if n > MAX_IDENTITY:
                raise ValueError(f"Identity overflow N > {MAX_IDENTITY}")

        # Must find at least one 1 (M >= 1 guaranteed)
        if pos >= length:
            raise ValueError(
                f"Stream ended at {pos} after {n} zeros: expected at least one '1'"
            )
        if bits[pos] != 1:
            raise ValueError(
                f"Expected '1' at pos {pos} (after {n} identity zeros), got {bits[pos]}"
            )

        # Step 2: Count ones -> M, then depth = M - 1
        m = 0
        while pos < length and bits[pos] == 1:
            m += 1
            pos += 1
            if m > MAX_DEPTH + 1:
                raise ValueError(f"Depth overflow M > {MAX_DEPTH+1}")

        depth = m - 1  # M = depth + 1, so depth = M - 1

        # Step 3: Read terminating zero
        if pos >= length:
            raise ValueError(
                f"Stream ended at {pos}: expected terminating zero"
            )
        if bits[pos] != 0:
            raise ValueError(
                f"Expected terminating zero at {pos}, got {bits[pos]}"
            )
        pos += 1

        return cls(identity=n, depth=depth), pos

    @classmethod
    def stream_from_bits(cls, bits: list[int]) -> Iterator[ZCEToken]:
        """Decode all tokens from a bit list."""
        pos = 0
        while pos < len(bits):
            token, pos = cls.from_bits(bits, pos)
            yield token

    # --- Representation ---

    def __repr__(self) -> str:
        bits = ''.join(str(b) for b in self.to_bits())
        return f"ZCEToken(N={self.identity}, M={self.depth}, bits={bits!r})"

    def __str__(self) -> str:
        return f"[N={self.identity} D={self.depth}]"

    def __lt__(self, other: ZCEToken) -> bool:
        if self.depth != other.depth:
            return self.depth < other.depth
        return self.identity < other.identity

    def __le__(self, other: ZCEToken) -> bool:
        return self == other or self < other
