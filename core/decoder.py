"""
zce_core/core/decoder.py
========================
THE FROZEN DECODER. THIS FILE DOES NOT CHANGE.

The decoder has exactly three operations:
    1. Count leading zeros  -> identity N
    2. Count following ones -> depth M
    3. Read terminating zero

That is all. These three operations process every token ever defined,
at any depth, in any vocabulary, now and in the future.

This is the Frozen Decoder Property (Theorem 4):
    D(V*) = D(V₀)
The execution substrate for any extended vocabulary is identical
to the substrate for the base vocabulary alone.

The decoder is stateless between tokens. It does not remember
what it decoded before. It does not look ahead. It does not
consult the vocabulary. It just counts.

DO NOT ADD STATE TO THIS CLASS BETWEEN TOKEN READS.
DO NOT ADD LOOKAHEAD.
DO NOT MAKE DECODING DEPEND ON VOCABULARY CONTENTS.
"""

from __future__ import annotations
from typing import Iterator
from .token import ZCEToken


class FrozenDecoder:
    """
    The frozen decoder. Instantiate once. Use forever.

    The decoder operates on a bit buffer. Feed bits to it and
    call next_token() to decode one token at a time.

    The decoder is deliberately simple. Its simplicity IS the
    Frozen Decoder Property — there is nothing here that could
    need updating.
    """

    def __init__(self) -> None:
        self._bits: list[int] = []
        self._pos: int = 0
        self._tokens_decoded: int = 0

    # ------------------------------------------------------------------
    # Feed bits into the decoder
    # ------------------------------------------------------------------

    def feed(self, bits: list[int]) -> None:
        """Append bits to the internal buffer."""
        self._bits.extend(bits)

    def feed_bytes(self, data: bytes) -> None:
        """
        Append bytes to the internal buffer, MSB first.
        Standard byte-to-bit conversion.
        """
        for byte in data:
            for shift in range(7, -1, -1):
                self._bits.append((byte >> shift) & 1)

    # ------------------------------------------------------------------
    # Decode one token — THE FROZEN THREE-STEP OPERATION
    # ------------------------------------------------------------------

    def next_token(self) -> ZCEToken | None:
        """
        Decode the next token from the buffer.
        Returns None if the buffer is exhausted or contains only padding zeros.

        The three steps — count zeros, count ones, read terminator —
        are implemented exactly once here and nowhere else.
        """
        if self._pos >= len(self._bits):
            return None

        # Skip trailing padding zeros (byte-alignment padding at end of stream)
        # Padding is all-zeros with no following ones — not a valid token.
        remaining = self._bits[self._pos:]
        if all(b == 0 for b in remaining):
            self._pos = len(self._bits)
            return None

        try:
            token, new_pos = ZCEToken.from_bits(self._bits, self._pos)
        except ValueError:
            # Stream ended or padding — stop gracefully
            self._pos = len(self._bits)
            return None
        self._pos = new_pos
        self._tokens_decoded += 1
        return token

    # ------------------------------------------------------------------
    # Decode all tokens remaining in the buffer
    # ------------------------------------------------------------------

    def decode_all(self) -> list[ZCEToken]:
        """Decode all tokens currently in the buffer."""
        tokens: list[ZCEToken] = []
        while True:
            tok = self.next_token()
            if tok is None:
                break
            tokens.append(tok)
        return tokens

    def __iter__(self) -> Iterator[ZCEToken]:
        """Iterate over all tokens in the buffer."""
        while True:
            tok = self.next_token()
            if tok is None:
                return
            yield tok

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    @property
    def bits_remaining(self) -> int:
        return len(self._bits) - self._pos

    @property
    def tokens_decoded(self) -> int:
        return self._tokens_decoded

    @property
    def is_exhausted(self) -> bool:
        return self._pos >= len(self._bits)

    def reset(self) -> None:
        """Reset the decoder position. Buffer contents are preserved."""
        self._pos = 0
        self._tokens_decoded = 0

    def clear(self) -> None:
        """Clear the buffer entirely."""
        self._bits = []
        self._pos = 0
        self._tokens_decoded = 0

    # ------------------------------------------------------------------
    # Encode a sequence of tokens to bits (inverse of decoding)
    # ------------------------------------------------------------------

    @staticmethod
    def encode(tokens: list[ZCEToken]) -> list[int]:
        """
        Encode a sequence of tokens to a bit list.
        This is the inverse of decode_all().
        Static method — encoding does not require decoder state.
        """
        bits: list[int] = []
        for token in tokens:
            bits.extend(token.to_bits())
        return bits

    @staticmethod
    def encode_to_bytes(tokens: list[ZCEToken]) -> bytes:
        """
        Encode tokens to bytes, padding the last byte with zeros.
        """
        bits = FrozenDecoder.encode(tokens)
        # Pad to byte boundary
        remainder = len(bits) % 8
        if remainder:
            bits.extend([0] * (8 - remainder))
        result = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            result.append(byte)
        return bytes(result)

    def __repr__(self) -> str:
        return (
            f"FrozenDecoder("
            f"bits_remaining={self.bits_remaining}, "
            f"tokens_decoded={self.tokens_decoded})"
        )
