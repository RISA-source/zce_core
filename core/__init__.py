"""
zce_core/core/__init__.py
=========================
The frozen core. Everything exported from here is immutable in principle.

Imports from this package:
    ZCEToken      — the self-describing binary token
    FrozenDecoder — the three-operation decoder
    Vocabulary    — the monotonically growing symbol store
    BaseTokens    — the reserved identity constants
    Symbol        — a single vocabulary entry
"""

from .token      import ZCEToken, MAX_IDENTITY, MAX_DEPTH
from .decoder    import FrozenDecoder
from .vocabulary import Vocabulary, Symbol, BaseTokens

__all__ = [
    "ZCEToken",
    "MAX_IDENTITY",
    "MAX_DEPTH",
    "FrozenDecoder",
    "Vocabulary",
    "Symbol",
    "BaseTokens",
]
