"""
zce_core/main.py
================
Demonstration of the ZCE Understanding Engine.

Shows the core loop:
    observe → compress → define → repeat

And demonstrates:
    - Vocabulary growing correctly by depth
    - Description length decreasing as vocabulary grows
    - Frozen decoder working unchanged throughout
    - All theoretical invariants holding
"""

from core   import ZCEToken, FrozenDecoder, Vocabulary, BaseTokens
from engine import TransitionCalculus, ExecutionState
from understanding_engine import UnderstandingEngine


def demo_basic_encoding():
    print("=" * 60)
    print("DEMO 1: ZCE Token Encoding")
    print("=" * 60)
    print()
    print("Format: [N zeros][M ones][terminating zero]")
    print("N = identity (what it IS), M = depth (where it came FROM)")
    print()

    pairs = [(0,0), (1,0), (0,1), (1,1), (2,3), (5,2), (0,5)]
    for n, m in pairs:
        t    = ZCEToken(n, m)
        bits = ''.join(str(b) for b in t.to_bits())
        print(f"  ({n:2d}, {m:2d})  ->  {bits:<15}  [{t.bit_length} bits]")
    print()


def demo_frozen_decoder():
    print("=" * 60)
    print("DEMO 2: Frozen Decoder")
    print("=" * 60)
    print()
    print("Three operations. Always the same. Never change.")
    print("  1. Count leading zeros  -> identity N")
    print("  2. Count following ones -> depth M")
    print("  3. Read terminating zero")
    print()

    tokens  = [ZCEToken(2, 0), ZCEToken(0, 3), ZCEToken(1, 1), ZCEToken(4, 2)]
    decoder = FrozenDecoder()
    for t in tokens:
        decoder.feed(t.to_bits())

    print("Encoded stream:")
    all_bits = []
    for t in tokens:
        all_bits.extend(t.to_bits())
    print(f"  {''.join(str(b) for b in all_bits)}")
    print()

    print("Decoded tokens:")
    decoder.reset()
    for tok in decoder:
        print(f"  N={tok.identity}, M={tok.depth}  ->  {tok}")
    print()


def demo_vocabulary_growth():
    print("=" * 60)
    print("DEMO 3: Vocabulary Growth by Depth")
    print("=" * 60)
    print()
    print("Depth is computed automatically.")
    print("Shallower symbols are always simpler than deeper ones.")
    print()

    v    = Vocabulary()
    add  = v.lookup_by_name("ADD")
    set_ = v.lookup_by_name("SET")

    # Depth 1 symbols
    mul = v.define("MUL",  [add.token.identity, add.token.identity],
                   description="Multiplication (via repeated ADD)")
    sub = v.define("SUB",  [set_.token.identity, add.token.identity],
                   description="Subtraction")

    # Depth 2 symbol
    norm2 = v.define("NORM2", [mul.token.identity, add.token.identity],
                     description="Sum of squares (a² + b²)")

    # Depth 3 symbol
    dist = v.define("DIST", [norm2.token.identity, mul.token.identity],
                    description="Distance metric")

    print(v.pretty_print())
    print()
    print(f"Token stream (ordered by depth, then identity):")
    for tok in v.to_token_stream():
        sym = v.lookup_by_token(tok)
        print(f"  {tok}  {sym.name:<10}  [{tok.bit_length} bits]")
    print()


def demo_compression_learning():
    print("=" * 60)
    print("DEMO 4: Compression Learning")
    print("=" * 60)
    print()
    print("Observe patterns → find recurring structure → define symbols")
    print("Description length decreases as vocabulary grows.")
    print()

    engine = UnderstandingEngine(min_frequency=2, min_savings=0)
    add    = engine.vocabulary.lookup_by_name("ADD")
    set_   = engine.vocabulary.lookup_by_name("SET")

    # A pattern that repeats
    pattern = [add.token, set_.token, add.token]

    # Initial description length
    dl_before = engine.description_length_of(pattern)
    print(f"Pattern:            {[str(t) for t in pattern]}")
    print(f"Initial desc len:   {dl_before} bits")
    print(f"Initial vocab size: {len(engine.vocabulary)}")
    print()

    # Observe it multiple times
    for _ in range(5):
        engine.observe(pattern)

    # Run compression
    new_syms = engine.compress_cycle()
    print(f"Compression cycle complete.")
    print(f"New symbols defined: {len(new_syms)}")
    for sym in new_syms:
        print(f"  {sym}")
    print()

    # Rewrite and measure
    rewritten = engine.rewrite(pattern)
    dl_after  = engine.description_length_of(rewritten)
    print(f"Rewritten:          {[str(t) for t in rewritten]}")
    print(f"New desc len:       {dl_after} bits")
    print(f"Savings:            {dl_before - dl_after} bits")
    print(f"Final vocab size:   {len(engine.vocabulary)}")
    print(f"Max depth:          {engine.vocabulary.max_depth}")
    print()

    # Check all invariants hold
    violations = engine.check_invariants()
    if violations:
        print(f"INVARIANT VIOLATIONS: {violations}")
    else:
        print("All invariants hold. ✓")
    print()


def demo_readability_theorem():
    print("=" * 60)
    print("DEMO 5: Readability Theorem")
    print("=" * 60)
    print()
    print("Same token stream + same initial state = same result.")
    print("Reading the computation IS executing it.")
    print()

    engine = UnderstandingEngine()
    add    = engine.vocabulary.lookup_by_name("ADD")

    def add_fn(state: ExecutionState) -> ExecutionState:
        a = state.get("a", 0)
        b = state.get("b", 0)
        state.set("result", a + b)
        return state

    engine.register_action("ADD", add_fn)

    tokens = [add.token]
    state  = {"a": 15, "b": 27}

    results = []
    for i in range(3):
        r = engine.execute(tokens, state.copy())
        results.append(r.get("result"))
        print(f"  Execution {i+1}: result = {r.get('result')}")

    assert all(r == results[0] for r in results), "Readability violated!"
    print()
    print(f"All three executions gave {results[0]}. Readability holds. ✓")
    print()


def demo_frozen_decoder_property():
    print("=" * 60)
    print("DEMO 6: Frozen Decoder Property (Theorem 4)")
    print("=" * 60)
    print()
    print("The decoder does not change as the vocabulary grows.")
    print("D(V*) = D(V₀)")
    print()

    engine  = UnderstandingEngine(min_frequency=2, min_savings=0)
    decoder = engine.decoder
    add     = engine.vocabulary.lookup_by_name("ADD")
    set_    = engine.vocabulary.lookup_by_name("SET")

    # Test token
    test_token = ZCEToken(7, 3)

    def decode_test():
        decoder.clear()
        decoder.feed(test_token.to_bits())
        return decoder.next_token()

    r1 = decode_test()
    print(f"Before vocab growth: decoded N={r1.identity}, M={r1.depth}")

    # Grow the vocabulary significantly
    for i in range(10):
        for _ in range(3):
            engine.observe([add.token, set_.token])
        engine.compress_cycle()

    print(f"Vocabulary grew to {len(engine.vocabulary)} symbols, "
          f"depth {engine.vocabulary.max_depth}")

    r2 = decode_test()
    print(f"After vocab growth:  decoded N={r2.identity}, M={r2.depth}")

    assert r1.identity == r2.identity and r1.depth == r2.depth
    print()
    print("Decoder unchanged. Frozen Decoder Property holds. ✓")
    print()


if __name__ == "__main__":
    demo_basic_encoding()
    demo_frozen_decoder()
    demo_vocabulary_growth()
    demo_compression_learning()
    demo_readability_theorem()
    demo_frozen_decoder_property()

    print("=" * 60)
    print("All demonstrations complete.")
    print("Run tests: python tests/test_all.py")
    print("=" * 60)
