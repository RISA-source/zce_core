"""
zce_core/domains/math/grammar.py
=================================
The mathematical expression grammar for ZCE.

This defines:
    - The base vocabulary V₀ for math expressions
    - The expression data structure (tree)
    - Tokenizer: expression -> ZCE token stream
    - Detokenizer: ZCE token stream -> expression string

Base vocabulary tokens (identity N, depth 0):
    N=5:  NUM     — a numeric literal
    N=6:  VAR     — a variable (x, y, z...)
    N=7:  ADD     — addition operator
    N=8:  SUB     — subtraction operator
    N=9:  MUL     — multiplication operator
    N=10: DIV     — division operator
    N=11: POW     — power/exponent operator
    N=12: NEG     — unary negation
    N=13: EQ      — equality
    N=14: LPAR    — left parenthesis (grouping marker)
    N=15: RPAR    — right parenthesis (grouping marker)
    N=16: COMMA   — argument separator

Note: N=0..4 are reserved by the core base vocab (DEFINE, SET, ADD, JIF, JMP).
Math base vocab starts at N=5 to avoid collision.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core import Vocabulary, ZCEToken, BaseTokens


# ---------------------------------------------------------------------------
# Math token identity constants
# ---------------------------------------------------------------------------

class MathTokens:
    NUM   = 5
    VAR   = 6
    ADD   = 7
    SUB   = 8
    MUL   = 9
    DIV   = 10
    POW   = 11
    NEG   = 12
    EQ    = 13
    LPAR  = 14
    RPAR  = 15
    COMMA = 16

    # For convenience: map from name to identity
    NAME_TO_ID = {
        'NUM': 5, 'VAR': 6, 'ADD': 7, 'SUB': 8, 'MUL': 9,
        'DIV': 10, 'POW': 11, 'NEG': 12, 'EQ': 13,
        'LPAR': 14, 'RPAR': 15, 'COMMA': 16,
    }
    ID_TO_NAME = {v: k for k, v in NAME_TO_ID.items()}


# ---------------------------------------------------------------------------
# Expression tree — the internal representation
# ---------------------------------------------------------------------------

@dataclass
class Expr:
    """
    An algebraic expression node.

    kind     : 'num', 'var', 'add', 'sub', 'mul', 'div', 'pow', 'neg', 'eq'
    value    : for 'num': the numeric value; for 'var': the variable name
    children : sub-expressions
    """
    kind     : str
    value    : Any          = None
    children : list[Expr]   = field(default_factory=list)

    def __str__(self) -> str:
        if self.kind == 'num':
            return str(self.value)
        if self.kind == 'var':
            return str(self.value)
        if self.kind == 'neg':
            return f"(-{self.children[0]})"
        if self.kind == 'add':
            return f"({self.children[0]} + {self.children[1]})"
        if self.kind == 'sub':
            return f"({self.children[0]} - {self.children[1]})"
        if self.kind == 'mul':
            return f"({self.children[0]} * {self.children[1]})"
        if self.kind == 'div':
            return f"({self.children[0]} / {self.children[1]})"
        if self.kind == 'pow':
            return f"({self.children[0]}^{self.children[1]})"
        if self.kind == 'eq':
            return f"({self.children[0]} = {self.children[1]})"
        return f"Expr({self.kind})"

    # Convenience constructors
    @staticmethod
    def num(v) -> Expr:
        return Expr('num', value=v)

    @staticmethod
    def var(name: str) -> Expr:
        return Expr('var', value=name)

    @staticmethod
    def add(a: Expr, b: Expr) -> Expr:
        return Expr('add', children=[a, b])

    @staticmethod
    def sub(a: Expr, b: Expr) -> Expr:
        return Expr('sub', children=[a, b])

    @staticmethod
    def mul(a: Expr, b: Expr) -> Expr:
        return Expr('mul', children=[a, b])

    @staticmethod
    def div(a: Expr, b: Expr) -> Expr:
        return Expr('div', children=[a, b])

    @staticmethod
    def pow(base: Expr, exp: Expr) -> Expr:
        return Expr('pow', children=[base, exp])

    @staticmethod
    def neg(a: Expr) -> Expr:
        return Expr('neg', children=[a])

    @staticmethod
    def eq(a: Expr, b: Expr) -> Expr:
        return Expr('eq', children=[a, b])


# ---------------------------------------------------------------------------
# Tokenizer: Expr -> list[ZCEToken]
# Produces a flat prefix-order token stream from an expression tree.
# Prefix order: operator first, then operands left-to-right.
# This is equivalent to S-expression / Lisp notation.
# ---------------------------------------------------------------------------

KIND_TO_ID = {
    'num':  MathTokens.NUM,
    'var':  MathTokens.VAR,
    'add':  MathTokens.ADD,
    'sub':  MathTokens.SUB,
    'mul':  MathTokens.MUL,
    'div':  MathTokens.DIV,
    'pow':  MathTokens.POW,
    'neg':  MathTokens.NEG,
    'eq':   MathTokens.EQ,
}


def expr_to_tokens(expr: Expr, vocab: Vocabulary) -> list[ZCEToken]:
    """
    Convert an expression to a ZCE token stream.
    Uses prefix (preorder) traversal: operator before operands.

    This is the canonical ZCE representation of an expression.
    The stream IS the expression — no separate parse needed.
    (Readability Theorem: the token stream is the computation.)
    """
    tokens: list[ZCEToken] = []
    _tokenize(expr, vocab, tokens)
    return tokens


def _tokenize(expr: Expr, vocab: Vocabulary, out: list[ZCEToken]) -> None:
    n = KIND_TO_ID.get(expr.kind)
    if n is None:
        raise ValueError(f"Unknown expression kind: {expr.kind!r}")
    sym = vocab.lookup_by_identity(n)
    if sym is None:
        raise ValueError(
            f"Token N={n} ({expr.kind}) not in vocabulary. "
            f"Did you call install_math_vocab()?"
        )
    out.append(sym.token)
    vocab.record_use(n)
    for child in expr.children:
        _tokenize(child, vocab, out)


def tokens_to_expr_str(tokens: list[ZCEToken], vocab: Vocabulary) -> str:
    """
    Convert a ZCE token stream back to a human-readable expression string.
    This demonstrates the Readability Theorem in the math domain.
    """
    names = []
    for tok in tokens:
        sym = vocab.lookup_by_identity(tok.identity)
        names.append(sym.name if sym else f"?{tok.identity}")
    return " ".join(names)


# ---------------------------------------------------------------------------
# Install the math base vocabulary into a Vocabulary instance
# ---------------------------------------------------------------------------

def install_math_vocab(vocab: Vocabulary) -> None:
    """
    Install the math base vocabulary into an existing Vocabulary.
    Must be called before tokenizing any expressions.

    This extends V₀ with math-specific primitives.
    These are base symbols (depth=0, no components).
    """
    math_base = [
        (MathTokens.NUM,   "NUM",   "Numeric literal placeholder"),
        (MathTokens.VAR,   "VAR",   "Variable placeholder"),
        (MathTokens.ADD,   "ADD_M", "Addition operator"),
        (MathTokens.SUB,   "SUB",   "Subtraction operator"),
        (MathTokens.MUL,   "MUL",   "Multiplication operator"),
        (MathTokens.DIV,   "DIV",   "Division operator"),
        (MathTokens.POW,   "POW",   "Power operator"),
        (MathTokens.NEG,   "NEG",   "Unary negation"),
        (MathTokens.EQ,    "EQ",    "Equality"),
        (MathTokens.LPAR,  "LPAR",  "Left parenthesis"),
        (MathTokens.RPAR,  "RPAR",  "Right parenthesis"),
        (MathTokens.COMMA, "COMMA", "Argument separator"),
    ]
    for n, name, desc in math_base:
        if not vocab.has_identity(n) and not vocab.has(name):
            vocab.define(
                name        = name,
                components  = [],        # base primitive — no components
                description = desc,
                identity    = n,
            )

    # Update KIND_TO_ID to use actual vocab token identities
    # (they should match MathTokens since we specified identity=n)


# ---------------------------------------------------------------------------
# Smart symbol naming — generate human-readable names for learned symbols
# ---------------------------------------------------------------------------

# Maps common token patterns to human-readable math names
_PATTERN_NAMES = {
    # 2-token patterns
    (MathTokens.MUL, MathTokens.NUM)  : "COEFF",        # coeff * ...
    (MathTokens.MUL, MathTokens.VAR)  : "MUL_VAR",
    (MathTokens.ADD, MathTokens.MUL)  : "ADD_MUL",
    (MathTokens.POW, MathTokens.VAR)  : "POW_VAR",
    (MathTokens.VAR, MathTokens.NUM)  : "VAR_EXP",       # var, exponent

    # 3-token patterns — core math primitives
    (MathTokens.MUL, MathTokens.NUM, MathTokens.VAR)                    : "LINEAR_TERM",   # a*x
    (MathTokens.POW, MathTokens.VAR, MathTokens.NUM)                    : "VAR_POWER",     # x^n
    (MathTokens.MUL, MathTokens.VAR, MathTokens.NUM)                    : "VAR_POW_TERM",
    (MathTokens.ADD, MathTokens.MUL, MathTokens.NUM)                    : "POLY_TERM",
    (MathTokens.MUL, MathTokens.NUM, MathTokens.NUM)                    : "CONST_PRODUCT",

    # 4-token patterns — compound math structures
    (MathTokens.MUL, MathTokens.NUM, MathTokens.VAR, MathTokens.NUM)   : "SCALED_VAR_N",  # a*x^n (partial)
    (MathTokens.MUL, MathTokens.NUM, MathTokens.POW, MathTokens.VAR)   : "COEFF_POW",
    (MathTokens.ADD, MathTokens.MUL, MathTokens.NUM, MathTokens.VAR)   : "ADD_LINEAR",    # +a*x
    (MathTokens.POW, MathTokens.MUL, MathTokens.NUM, MathTokens.VAR)   : "BINOMIAL_SQ_INNER",
}

_name_counter: dict[str, int] = {}


def suggest_name(pattern: tuple[int, ...], vocab: Vocabulary) -> str:
    """
    Suggest a human-readable name for a learned symbol
    based on its token pattern.
    """
    # Check exact pattern match
    if pattern in _PATTERN_NAMES:
        base = _PATTERN_NAMES[pattern]
    else:
        # Build name from component names
        parts = []
        for n in pattern[:4]:  # limit to first 4 for readability
            sym = vocab.lookup_by_identity(n)
            if sym:
                # Shorten known names
                short = {
                    'MUL': 'MUL', 'ADD_M': 'ADD', 'SUB': 'SUB',
                    'POW': 'POW', 'NUM': 'N', 'VAR': 'V',
                    'DIV': 'DIV', 'NEG': 'NEG', 'EQ': 'EQ',
                }.get(sym.name, sym.name[:6])
                parts.append(short)
        if len(pattern) > 4:
            parts.append(f'+{len(pattern)-4}')
        base = '_'.join(parts) if parts else 'SYM'

    # Ensure uniqueness
    if base not in _name_counter:
        _name_counter[base] = 0
        return base
    else:
        _name_counter[base] += 1
        return f"{base}_{_name_counter[base]}"
