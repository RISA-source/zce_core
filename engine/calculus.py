"""
zce_core/engine/calculus.py
===========================
THE TRANSITION CALCULUS.

Three fundamental operations:
    Sequential  : u ; v    — do u then v
    Parallel    : u || v   — do u and v on separate state components
    Conditional : P ? u : v — do u if P holds, else v

These are the composition operators of SDTS.
The algebraic structure is a Kleene algebra with tests,
extended with parallel composition and the DEFINE meta-transition.

Every compound transition expressible in the system is a finite
composition of these three operations applied to vocabulary symbols.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any
from core import ZCEToken, Vocabulary, Symbol


# ---------------------------------------------------------------------------
# Execution state — the slots that transitions operate on
# ---------------------------------------------------------------------------

class ExecutionState:
    """
    The mutable state that transitions act on.
    Implemented as a dictionary of named slots.
    Slots can hold any Python value — integers, floats, strings, tensors.
    """

    def __init__(self, initial: dict[str, Any] = None) -> None:
        self._slots: dict[str, Any] = dict(initial or {})
        self._history: list[tuple[str, Any, Any]] = []   # (slot, old, new)

    def get(self, key: str, default: Any = None) -> Any:
        return self._slots.get(key, default)

    def set(self, key: str, value: Any) -> None:
        old = self._slots.get(key)
        self._slots[key] = value
        self._history.append((key, old, value))

    def copy(self) -> ExecutionState:
        new = ExecutionState(self._slots.copy())
        return new

    def __repr__(self) -> str:
        return f"ExecutionState({self._slots})"


# ---------------------------------------------------------------------------
# A transition — the unit of computation
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    """
    A named transition in the Transition Calculus.

    token  : the ZCE token identifying this transition
    symbol : the vocabulary symbol this transition corresponds to
    fn     : the function implementing this transition
             fn(state: ExecutionState) -> ExecutionState
    """
    token  : ZCEToken
    symbol : Symbol
    fn     : Callable[[ExecutionState], ExecutionState]

    def apply(self, state: ExecutionState) -> ExecutionState:
        """Execute this transition on a state."""
        return self.fn(state)

    def __repr__(self) -> str:
        return f"Transition({self.symbol.name!r}, {self.token})"


# ---------------------------------------------------------------------------
# The three calculus operations
# ---------------------------------------------------------------------------

class TransitionCalculus:
    """
    Implements the three fundamental composition operations
    and provides an execution engine for composed transitions.
    """

    def __init__(self, vocabulary: Vocabulary) -> None:
        self._vocab = vocabulary
        # Registry of executable transitions
        self._transitions: dict[int, Transition] = {}
        self._install_base_transitions()

    def _install_base_transitions(self) -> None:
        """Install transitions for base vocabulary symbols."""

        # SET: state['dst'] = value
        def set_fn(state: ExecutionState) -> ExecutionState:
            dst = state.get('__arg_dst')
            val = state.get('__arg_val')
            state.set(str(dst), val)
            return state

        # ADD: state['dst'] = state['a'] + state['b']
        def add_fn(state: ExecutionState) -> ExecutionState:
            dst = state.get('__arg_dst')
            a   = state.get('__arg_a')
            b   = state.get('__arg_b')
            state.set(str(dst), state.get(str(a), 0) + state.get(str(b), 0))
            return state

        # NOP: identity transition
        def nop_fn(state: ExecutionState) -> ExecutionState:
            return state

        for name, fn in [("SET", set_fn), ("ADD", add_fn),
                         ("DEFINE", nop_fn), ("JIF", nop_fn), ("JMP", nop_fn)]:
            sym = self._vocab.lookup_by_name(name)
            if sym:
                self._transitions[sym.token.identity] = Transition(
                    token=sym.token, symbol=sym, fn=fn
                )

    def register(self, identity: int, fn: Callable) -> None:
        """Register an executable function for a vocabulary symbol."""
        sym = self._vocab.lookup_by_identity(identity)
        if sym is None:
            raise ValueError(f"No symbol with identity {identity}")
        self._transitions[identity] = Transition(
            token=sym.token, symbol=sym, fn=fn
        )

    # ------------------------------------------------------------------
    # The three fundamental operations
    # ------------------------------------------------------------------

    def sequential(
        self,
        t1: Transition,
        t2: Transition,
    ) -> Callable[[ExecutionState], ExecutionState]:
        """
        Sequential composition: u ; v
        δ(s, u;v) = δ(δ(s,u), v)
        Do t1 then t2.
        """
        def composed(state: ExecutionState) -> ExecutionState:
            state = t1.apply(state)
            state = t2.apply(state)
            return state
        return composed

    def parallel(
        self,
        t1: Transition,
        t2: Transition,
        key1: str,
        key2: str,
    ) -> Callable[[ExecutionState], ExecutionState]:
        """
        Parallel composition: u || v
        δ((s1,s2), u||v) = (δ(s1,u), δ(s2,v))
        Apply t1 to the component of state keyed by key1,
        apply t2 to the component keyed by key2.
        """
        def composed(state: ExecutionState) -> ExecutionState:
            s1 = ExecutionState({key1: state.get(key1)})
            s2 = ExecutionState({key2: state.get(key2)})
            s1 = t1.apply(s1)
            s2 = t2.apply(s2)
            state.set(key1, s1.get(key1))
            state.set(key2, s2.get(key2))
            return state
        return composed

    def conditional(
        self,
        predicate: Callable[[ExecutionState], bool],
        t_true:  Transition,
        t_false: Transition,
    ) -> Callable[[ExecutionState], ExecutionState]:
        """
        Conditional transition: P ? u : v
        δ(s, P?u:v) = δ(s,u) if P(s) else δ(s,v)
        """
        def composed(state: ExecutionState) -> ExecutionState:
            if predicate(state):
                return t_true.apply(state)
            else:
                return t_false.apply(state)
        return composed

    # ------------------------------------------------------------------
    # Execute a token stream
    # ------------------------------------------------------------------

    def execute_stream(
        self,
        tokens: list[ZCEToken],
        initial_state: ExecutionState = None,
    ) -> ExecutionState:
        """
        Execute a sequence of ZCE tokens on a state.

        This is the core execution loop. It reads tokens in order
        and applies the corresponding transition to the state.

        The Readability Theorem holds here: every token is named,
        every transition is recoverable from its token, every
        intermediate state is determined by the token sequence.
        """
        state = initial_state or ExecutionState()

        for token in tokens:
            transition = self._transitions.get(token.identity)
            if transition is None:
                # Unknown token — check if it has a composed definition
                sym = self._vocab.lookup_by_identity(token.identity)
                if sym and sym.action:
                    # Symbol has an action registered directly
                    result = sym.action(state)
                    if result is not None:
                        state = result
                # else: silently skip (forward compatible)
            else:
                state = transition.apply(state)

        return state

    def get_transition(self, identity: int) -> Transition | None:
        return self._transitions.get(identity)

    def __repr__(self) -> str:
        return f"TransitionCalculus(transitions={len(self._transitions)})"
