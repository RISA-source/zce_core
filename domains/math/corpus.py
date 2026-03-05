"""
zce_core/domains/math/corpus.py
================================
The expression corpus — the training data.

Generates a diverse set of algebraic expressions that a compression
engine can learn from. The corpus is designed so that recurring
structures at different depths are present:

Depth-1 patterns (should be discovered):
    - x^2          (VAR POW NUM)
    - a*x          (NUM MUL VAR)
    - a*x^2        (NUM MUL VAR POW NUM)

Depth-2 patterns (should be discovered):
    - a*x^2 + b*x  (quadratic body without constant)
    - (a+b)^2      (binomial squared)
    - a^2 + b^2    (sum of squares)

Depth-3 patterns (should be discovered):
    - a*x^2 + b*x + c  (full quadratic)
    - (a+b)*(a-b)      (difference of squares)

The corpus contains many instances of each pattern with varying
coefficients and variables. This is realistic: real math involves
the same structural patterns applied to different specific values.
"""

from __future__ import annotations
import random
from .grammar import Expr


def make_corpus(n_expressions: int = 200, seed: int = 42) -> list[Expr]:
    """
    Generate a corpus of algebraic expressions.
    Returns a list of Expr trees.
    """
    rng = random.Random(seed)
    corpus: list[Expr] = []

    generators = [
        # Weight, generator function
        (8,  lambda: _linear(rng)),
        (8,  lambda: _quadratic(rng)),
        (5,  lambda: _binomial_squared(rng)),
        (5,  lambda: _diff_of_squares(rng)),
        (4,  lambda: _sum_of_squares(rng)),
        (4,  lambda: _cubic(rng)),
        (3,  lambda: _simple_equation(rng)),
        (3,  lambda: _power_expr(rng)),
        (2,  lambda: _nested_binomial(rng)),
        (2,  lambda: _rational(rng)),
    ]

    total_weight = sum(w for w, _ in generators)
    for _ in range(n_expressions):
        r = rng.random() * total_weight
        cumul = 0
        for weight, gen in generators:
            cumul += weight
            if r < cumul:
                try:
                    corpus.append(gen())
                except Exception:
                    corpus.append(_linear(rng))  # fallback
                break

    return corpus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vars(rng: random.Random) -> str:
    return rng.choice(['x', 'y', 'z', 't', 'n'])

def _coeff(rng: random.Random, max_val: int = 9) -> int:
    return rng.randint(1, max_val)

def _small_int(rng: random.Random) -> int:
    return rng.randint(0, 5)


# ---------------------------------------------------------------------------
# Expression generators
# ---------------------------------------------------------------------------

def _linear(rng: random.Random) -> Expr:
    """a*x + b"""
    a = _coeff(rng)
    b = _small_int(rng)
    v = _vars(rng)
    term = Expr.mul(Expr.num(a), Expr.var(v))
    if b == 0:
        return term
    return Expr.add(term, Expr.num(b))


def _quadratic(rng: random.Random) -> Expr:
    """a*x^2 + b*x + c"""
    a = _coeff(rng)
    b = _coeff(rng)
    c = _small_int(rng)
    v = _vars(rng)
    x_sq   = Expr.pow(Expr.var(v), Expr.num(2))
    ax_sq  = Expr.mul(Expr.num(a), x_sq)
    bx     = Expr.mul(Expr.num(b), Expr.var(v))
    ax_bx  = Expr.add(ax_sq, bx)
    if c == 0:
        return ax_bx
    return Expr.add(ax_bx, Expr.num(c))


def _binomial_squared(rng: random.Random) -> Expr:
    """(a*x + b)^2"""
    a = _coeff(rng)
    b = _coeff(rng)
    v = _vars(rng)
    inner = Expr.add(Expr.mul(Expr.num(a), Expr.var(v)), Expr.num(b))
    return Expr.pow(inner, Expr.num(2))


def _diff_of_squares(rng: random.Random) -> Expr:
    """(a*x + b)*(a*x - b)"""
    a = _coeff(rng)
    b = _coeff(rng)
    v = _vars(rng)
    ax   = Expr.mul(Expr.num(a), Expr.var(v))
    plus  = Expr.add(ax, Expr.num(b))
    minus = Expr.sub(ax, Expr.num(b))
    return Expr.mul(plus, minus)


def _sum_of_squares(rng: random.Random) -> Expr:
    """a^2 + b^2 (with variables)"""
    v1 = _vars(rng)
    v2 = _vars(rng)
    a  = _coeff(rng)
    b  = _coeff(rng)
    sq1 = Expr.pow(Expr.mul(Expr.num(a), Expr.var(v1)), Expr.num(2))
    sq2 = Expr.pow(Expr.mul(Expr.num(b), Expr.var(v2)), Expr.num(2))
    return Expr.add(sq1, sq2)


def _cubic(rng: random.Random) -> Expr:
    """a*x^3 + b*x^2 + c*x + d"""
    a = _coeff(rng)
    b = _coeff(rng)
    c = _coeff(rng)
    d = _small_int(rng)
    v = _vars(rng)
    x_cu  = Expr.pow(Expr.var(v), Expr.num(3))
    ax_cu = Expr.mul(Expr.num(a), x_cu)
    x_sq  = Expr.pow(Expr.var(v), Expr.num(2))
    bx_sq = Expr.mul(Expr.num(b), x_sq)
    cx    = Expr.mul(Expr.num(c), Expr.var(v))
    body  = Expr.add(Expr.add(ax_cu, bx_sq), cx)
    if d == 0:
        return body
    return Expr.add(body, Expr.num(d))


def _simple_equation(rng: random.Random) -> Expr:
    """a*x + b = c"""
    a = _coeff(rng)
    b = _small_int(rng)
    c = _coeff(rng, 20)
    v = _vars(rng)
    lhs = Expr.add(Expr.mul(Expr.num(a), Expr.var(v)), Expr.num(b))
    return Expr.eq(lhs, Expr.num(c))


def _power_expr(rng: random.Random) -> Expr:
    """x^n for various n"""
    v = _vars(rng)
    n = rng.randint(2, 5)
    return Expr.pow(Expr.var(v), Expr.num(n))


def _nested_binomial(rng: random.Random) -> Expr:
    """(a*x + b)*(c*x + d)"""
    a = _coeff(rng)
    b = _coeff(rng)
    c = _coeff(rng)
    d = _coeff(rng)
    v = _vars(rng)
    f1 = Expr.add(Expr.mul(Expr.num(a), Expr.var(v)), Expr.num(b))
    f2 = Expr.add(Expr.mul(Expr.num(c), Expr.var(v)), Expr.num(d))
    return Expr.mul(f1, f2)


def _rational(rng: random.Random) -> Expr:
    """(a*x + b) / (c*x + d)"""
    a = _coeff(rng)
    b = _coeff(rng)
    c = _coeff(rng)
    d = _coeff(rng)
    v = _vars(rng)
    num = Expr.add(Expr.mul(Expr.num(a), Expr.var(v)), Expr.num(b))
    den = Expr.add(Expr.mul(Expr.num(c), Expr.var(v)), Expr.num(d))
    return Expr.div(num, den)
