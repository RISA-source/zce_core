"""
zce_core/tests/test_all.py
Complete test suite — organized by theorem.
Run: python tests/test_all.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from core        import ZCEToken, FrozenDecoder, Vocabulary, BaseTokens, Symbol
from engine      import TransitionCalculus, ExecutionState, Compressor
from understanding_engine import UnderstandingEngine


# ===========================================================================
# Token Tests — Theorem 1
# New encoding: bit_length = N + depth + 2, M = depth+1 >= 1 always
# ===========================================================================

class TestZCEToken(unittest.TestCase):

    def test_valid_token_zero_zero(self):
        t = ZCEToken(0, 0)
        self.assertEqual(t.identity, 0)
        self.assertEqual(t.depth, 0)

    def test_valid_token_arbitrary(self):
        t = ZCEToken(3, 5)
        self.assertEqual(t.identity, 3)
        self.assertEqual(t.depth, 5)

    def test_negative_identity_raises(self):
        with self.assertRaises(ValueError):
            ZCEToken(-1, 0)

    def test_negative_depth_raises(self):
        with self.assertRaises(ValueError):
            ZCEToken(0, -1)

    def test_overflow_identity_raises(self):
        from core.token import MAX_IDENTITY
        with self.assertRaises(ValueError):
            ZCEToken(MAX_IDENTITY + 1, 0)

    def test_encoding_zero_zero(self):
        """(0,0) -> M=1 -> [1,0]  (2 bits, minimum)"""
        t = ZCEToken(0, 0)
        self.assertEqual(t.to_bits(), [1, 0])
        self.assertEqual(t.bit_length, 2)

    def test_encoding_one_zero(self):
        """(1,0) -> M=1 -> [0,1,0]"""
        t = ZCEToken(1, 0)
        self.assertEqual(t.to_bits(), [0, 1, 0])

    def test_encoding_zero_one(self):
        """(0,1) -> M=2 -> [1,1,0]"""
        t = ZCEToken(0, 1)
        self.assertEqual(t.to_bits(), [1, 1, 0])

    def test_encoding_one_one(self):
        """(1,1) -> M=2 -> [0,1,1,0]"""
        t = ZCEToken(1, 1)
        self.assertEqual(t.to_bits(), [0, 1, 1, 0])

    def test_encoding_two_three(self):
        """(2,3) -> M=4 -> [0,0,1,1,1,1,0]"""
        t = ZCEToken(2, 3)
        self.assertEqual(t.to_bits(), [0, 0, 1, 1, 1, 1, 0])

    def test_bit_length_formula(self):
        """bit_length == N + depth + 2 always"""
        for n in range(6):
            for d in range(6):
                t = ZCEToken(n, d)
                self.assertEqual(t.bit_length, n + d + 2,
                                 f"bit_length wrong for ({n},{d})")

    def test_minimum_token_is_two_bits(self):
        """Minimum possible token (0,0) uses 2 bits."""
        t = ZCEToken(0, 0)
        self.assertEqual(t.bit_length, 2)

    def test_encode_decode_roundtrip(self):
        """Encoding then decoding returns the original token."""
        for n in range(8):
            for d in range(8):
                original = ZCEToken(n, d)
                bits = original.to_bits()
                recovered, pos = ZCEToken.from_bits(bits, 0)
                self.assertEqual(recovered.identity, n,
                                 f"Identity mismatch ({n},{d})")
                self.assertEqual(recovered.depth, d,
                                 f"Depth mismatch ({n},{d})")
                self.assertEqual(pos, len(bits),
                                 f"Position mismatch ({n},{d})")

    def test_decode_stream(self):
        """Multiple tokens including depth=0 adjacent to depth>0."""
        tokens = [ZCEToken(1, 0), ZCEToken(0, 2), ZCEToken(3, 1)]
        bits = []
        for t in tokens:
            bits.extend(t.to_bits())
        decoded = list(ZCEToken.stream_from_bits(bits))
        self.assertEqual(len(decoded), len(tokens))
        for orig, dec in zip(tokens, decoded):
            self.assertEqual(orig.identity, dec.identity)
            self.assertEqual(orig.depth, dec.depth)

    def test_decode_stream_with_depth_zero_adjacent(self):
        """Two depth=0 tokens adjacent decode correctly."""
        tokens = [ZCEToken(2, 0), ZCEToken(1, 0), ZCEToken(0, 0)]
        bits = []
        for t in tokens:
            bits.extend(t.to_bits())
        decoded = list(ZCEToken.stream_from_bits(bits))
        self.assertEqual(len(decoded), 3)
        for o, d in zip(tokens, decoded):
            self.assertEqual(o.identity, d.identity)
            self.assertEqual(o.depth, d.depth)

    def test_decode_missing_ones_raises(self):
        """A bit stream with no ones raises (M >= 1 always required)."""
        # All zeros with no ones — not a valid ZCE token
        bits = [0, 0, 0]  # just zeros, no ones
        with self.assertRaises(ValueError):
            ZCEToken.from_bits(bits, 0)

    def test_token_is_immutable(self):
        t = ZCEToken(2, 3)
        with self.assertRaises((AttributeError, TypeError)):
            t.identity = 99  # type: ignore

    def test_ordering_depth_first(self):
        self.assertLess(ZCEToken(5, 0), ZCEToken(1, 3))

    def test_ordering_identity_within_depth(self):
        self.assertLess(ZCEToken(1, 2), ZCEToken(3, 2))


# ===========================================================================
# Frozen Decoder Tests — Theorem 4
# ===========================================================================

class TestFrozenDecoder(unittest.TestCase):

    def test_basic_decode(self):
        decoder = FrozenDecoder()
        t = ZCEToken(2, 3)
        decoder.feed(t.to_bits())
        result = decoder.next_token()
        self.assertIsNotNone(result)
        self.assertEqual(result.identity, 2)
        self.assertEqual(result.depth, 3)

    def test_exhausted_returns_none(self):
        decoder = FrozenDecoder()
        decoder.feed(ZCEToken(0, 1).to_bits())  # [1,1,0] — valid token
        decoder.next_token()                     # consume it
        result = decoder.next_token()            # now exhausted
        self.assertIsNone(result)

    def test_decode_all(self):
        tokens  = [ZCEToken(i, j) for i in range(3) for j in range(3)]
        decoder = FrozenDecoder()
        for t in tokens:
            decoder.feed(t.to_bits())
        decoded = decoder.decode_all()
        self.assertEqual(len(decoded), len(tokens))

    def test_encode_decode_bytes_roundtrip(self):
        tokens  = [ZCEToken(1, 2), ZCEToken(3, 1), ZCEToken(0, 4)]
        encoded = FrozenDecoder.encode_to_bytes(tokens)
        decoder = FrozenDecoder()
        decoder.feed_bytes(encoded)
        decoded = decoder.decode_all()
        # First 3 decoded tokens should match originals
        for orig, dec in zip(tokens, decoded[:len(tokens)]):
            self.assertEqual(orig.identity, dec.identity)
            self.assertEqual(orig.depth, dec.depth)

    def test_decoder_state_independence(self):
        """Same token decodes identically regardless of decoder instance state."""
        d1 = FrozenDecoder()
        d2 = FrozenDecoder()
        t  = ZCEToken(5, 10)
        d1.feed(t.to_bits())
        d2.feed(t.to_bits())
        r1 = d1.next_token()
        r2 = d2.next_token()
        self.assertEqual(r1.identity, r2.identity)
        self.assertEqual(r1.depth,    r2.depth)

    def test_bits_remaining_decreases(self):
        decoder = FrozenDecoder()
        tokens  = [ZCEToken(1, 1), ZCEToken(2, 0)]
        for t in tokens:
            decoder.feed(t.to_bits())
        before = decoder.bits_remaining
        decoder.next_token()
        after = decoder.bits_remaining
        self.assertLess(after, before)


# ===========================================================================
# Vocabulary Tests — Theorem 0 + Theorem 5
# ===========================================================================

class TestVocabulary(unittest.TestCase):

    def test_base_vocab_installed(self):
        v = Vocabulary()
        for name in ["DEFINE", "SET", "ADD", "JIF", "JMP"]:
            self.assertIsNotNone(v.lookup_by_name(name))

    def test_base_symbols_are_depth_zero(self):
        v = Vocabulary()
        for name in ["DEFINE", "SET", "ADD", "JIF", "JMP"]:
            sym = v.lookup_by_name(name)
            self.assertEqual(sym.token.depth, 0)

    def test_define_computes_depth_correctly(self):
        v    = Vocabulary()
        add  = v.lookup_by_name("ADD")
        set_ = v.lookup_by_name("SET")
        sym  = v.define("COMBO", [add.token.identity, set_.token.identity])
        self.assertEqual(sym.token.depth, 1)

    def test_depth_chains(self):
        v   = Vocabulary()
        add = v.lookup_by_name("ADD")
        s1  = v.define("OP1", [add.token.identity, add.token.identity])
        self.assertEqual(s1.token.depth, 1)
        s2  = v.define("OP2", [s1.token.identity, add.token.identity])
        self.assertEqual(s2.token.depth, 2)
        s3  = v.define("OP3", [s2.token.identity, s1.token.identity])
        self.assertEqual(s3.token.depth, 3)

    def test_vocabulary_grows_monotonically(self):
        v   = Vocabulary()
        add = v.lookup_by_name("ADD")
        before = len(v)
        v.define("OP1", [add.token.identity])
        self.assertGreater(len(v), before)

    def test_duplicate_name_raises(self):
        v   = Vocabulary()
        add = v.lookup_by_name("ADD")
        v.define("OP1", [add.token.identity])
        with self.assertRaises(ValueError):
            v.define("OP1", [add.token.identity])

    def test_immutable_tokens_cannot_be_redefined(self):
        v   = Vocabulary()
        add = v.lookup_by_name("ADD")
        for immutable_id in BaseTokens.IMMUTABLE:
            with self.assertRaises(ValueError):
                v.define("ATTEMPT", [add.token.identity], identity=immutable_id)

    def test_unknown_component_raises(self):
        v = Vocabulary()
        with self.assertRaises(ValueError):
            v.define("BAD", [99999])

    def test_invariants_pass_fresh(self):
        v = Vocabulary()
        self.assertEqual(v.check_invariants(), [])

    def test_invariants_pass_after_defines(self):
        v    = Vocabulary()
        add  = v.lookup_by_name("ADD")
        set_ = v.lookup_by_name("SET")
        v.define("OP1", [add.token.identity])
        v.define("OP2", [set_.token.identity, add.token.identity])
        v.define("OP3", [
            v.lookup_by_name("OP1").token.identity,
            v.lookup_by_name("OP2").token.identity,
        ])
        self.assertEqual(v.check_invariants(), [])

    def test_stream_ordered_by_depth_then_identity(self):
        v    = Vocabulary()
        add  = v.lookup_by_name("ADD")
        set_ = v.lookup_by_name("SET")
        v.define("D2", [
            v.define("D1", [add.token.identity]).token.identity,
            set_.token.identity,
        ])
        stream = v.to_token_stream()
        for i in range(len(stream) - 1):
            self.assertLessEqual(stream[i], stream[i+1])


# ===========================================================================
# Transition Calculus Tests — Theorem 2 (Readability)
# ===========================================================================

class TestTransitionCalculus(unittest.TestCase):

    def setUp(self):
        self.vocab = Vocabulary()
        self.calc  = TransitionCalculus(self.vocab)

        def set_action(state):
            state.set("result", state.get("value", 0))
            return state
        def add_action(state):
            state.set("result", state.get("a", 0) + state.get("b", 0))
            return state

        self.calc.register(BaseTokens.SET, set_action)
        self.calc.register(BaseTokens.ADD, add_action)

    def test_sequential_composition(self):
        t_set = self.calc.get_transition(BaseTokens.SET)
        t_add = self.calc.get_transition(BaseTokens.ADD)
        composed = self.calc.sequential(t_set, t_add)
        state = ExecutionState({"value": 5, "a": 3, "b": 4})
        result = composed(state)
        self.assertEqual(result.get("result"), 7)

    def test_conditional_true_branch(self):
        t_set = self.calc.get_transition(BaseTokens.SET)
        t_add = self.calc.get_transition(BaseTokens.ADD)
        pred  = lambda s: s.get("flag", 0) == 1
        cond  = self.calc.conditional(pred, t_set, t_add)
        state = ExecutionState({"flag": 1, "value": 42, "a": 1, "b": 2})
        self.assertEqual(cond(state).get("result"), 42)

    def test_conditional_false_branch(self):
        t_set = self.calc.get_transition(BaseTokens.SET)
        t_add = self.calc.get_transition(BaseTokens.ADD)
        pred  = lambda s: s.get("flag", 0) == 1
        cond  = self.calc.conditional(pred, t_set, t_add)
        state = ExecutionState({"flag": 0, "value": 42, "a": 10, "b": 20})
        self.assertEqual(cond(state).get("result"), 30)

    def test_execute_stream(self):
        tokens = [ZCEToken(BaseTokens.ADD, 0)]
        state  = ExecutionState({"a": 7, "b": 8})
        result = self.calc.execute_stream(tokens, state)
        self.assertEqual(result.get("result"), 15)

    def test_readability_same_stream_same_result(self):
        """Theorem 2: same tokens + same state = same result, always."""
        tokens = [ZCEToken(BaseTokens.ADD, 0)]
        results = set()
        for _ in range(3):
            r = self.calc.execute_stream(tokens, ExecutionState({"a": 3, "b": 4}))
            results.add(r.get("result"))
        self.assertEqual(len(results), 1)
        self.assertIn(7, results)


# ===========================================================================
# Compressor Tests — Theorem 3 (Learnability) in practice
# ===========================================================================

class TestCompressor(unittest.TestCase):

    def setUp(self):
        self.vocab = Vocabulary()
        self.add   = self.vocab.lookup_by_name("ADD")
        self.set_  = self.vocab.lookup_by_name("SET")

    def test_single_occurrence_not_a_candidate(self):
        c = Compressor(self.vocab, min_frequency=2)
        c.observe([self.add.token, self.set_.token])
        self.assertEqual(len(c.find_candidates()), 0)

    def test_rewrite_uses_manually_defined_symbols(self):
        """After defining a symbol, rewrite replaces matching sequences."""
        compound = self.vocab.define(
            "COMPOUND", [self.add.token.identity, self.set_.token.identity]
        )
        c = Compressor(self.vocab)
        stream    = [self.add.token, self.set_.token, self.add.token, self.set_.token]
        rewritten = c.rewrite(stream)
        # Should replace both pairs with compound
        identities = [t.identity for t in rewritten]
        self.assertIn(compound.token.identity, identities)

    def test_description_length_decreases_with_deep_compound(self):
        """
        DL decreases when replacing deep-token sequences with compounds.
        At depth 2, savings are substantial (see theory doc Section 8).
        """
        c = Compressor(self.vocab)
        # Build depth-1 symbols
        d1a = self.vocab.define("D1A", [self.add.token.identity, self.add.token.identity])
        d1b = self.vocab.define("D1B", [self.set_.token.identity, self.add.token.identity])
        # Build depth-2 symbol combining them
        d2  = self.vocab.define("D2", [d1a.token.identity, d1b.token.identity])

        # A stream using d1a and d1b repeatedly
        stream = [d1a.token, d1b.token] * 4
        before = c.description_length(stream)
        rewritten = c.rewrite(stream)
        after = c.description_length(rewritten)
        self.assertLess(after, before,
            f"DL should decrease: {before} -> {after}")

    def test_compressor_finds_candidate_with_deep_tokens(self):
        """Compressor finds candidates when patterns use expensive tokens."""
        # Build depth-1 symbols first
        d1a = self.vocab.define("D1A2", [self.add.token.identity, self.add.token.identity])
        d1b = self.vocab.define("D1B2", [self.set_.token.identity, self.set_.token.identity])

        c = Compressor(self.vocab, min_frequency=2, min_savings=0)
        pattern = [d1a.token, d1b.token]
        for _ in range(4):
            c.observe(pattern)

        candidates = c.find_candidates()
        self.assertGreater(len(candidates), 0,
            "Should find candidates for repeated deep-token pattern")

    def test_apply_candidate_adds_symbol(self):
        d1a = self.vocab.define("XA", [self.add.token.identity, self.add.token.identity])
        d1b = self.vocab.define("XB", [self.set_.token.identity, self.add.token.identity])
        c = Compressor(self.vocab, min_frequency=2, min_savings=0)
        for _ in range(4):
            c.observe([d1a.token, d1b.token])
        candidates = c.find_candidates()
        if candidates:
            before = len(self.vocab)
            c.apply_candidate(candidates[0], name="LEARNED_X")
            self.assertGreater(len(self.vocab), before)


# ===========================================================================
# Understanding Engine Integration Tests
# ===========================================================================

class TestUnderstandingEngine(unittest.TestCase):

    def setUp(self):
        self.engine = UnderstandingEngine(min_frequency=2, min_savings=0)

    def test_base_vocab_present(self):
        for name in ["DEFINE", "SET", "ADD", "JIF", "JMP"]:
            self.assertIsNotNone(self.engine.vocabulary.lookup_by_name(name))

    def test_observe_and_compress_cycle(self):
        add  = self.engine.vocabulary.lookup_by_name("ADD")
        set_ = self.engine.vocabulary.lookup_by_name("SET")
        d1a  = self.engine.vocabulary.define("IA", [add.token.identity, add.token.identity])
        d1b  = self.engine.vocabulary.define("IB", [set_.token.identity, add.token.identity])
        before = len(self.engine.vocabulary)
        for _ in range(4):
            self.engine.observe([d1a.token, d1b.token])
        self.engine.compress_cycle()
        self.assertGreaterEqual(len(self.engine.vocabulary), before)

    def test_surprise_known(self):
        add = self.engine.vocabulary.lookup_by_name("ADD")
        self.assertEqual(self.engine.surprise([add.token]), 0.0)

    def test_surprise_unknown(self):
        unknown = ZCEToken(9999, 0)
        self.assertEqual(self.engine.surprise([unknown]), 1.0)

    def test_invariants_hold(self):
        self.assertEqual(self.engine.check_invariants(), [])
        add  = self.engine.vocabulary.lookup_by_name("ADD")
        set_ = self.engine.vocabulary.lookup_by_name("SET")
        d1   = self.engine.vocabulary.define("IC", [add.token.identity, set_.token.identity])
        d2   = self.engine.vocabulary.define("ID", [set_.token.identity, add.token.identity])
        for _ in range(3):
            self.engine.observe([d1.token, d2.token])
        self.engine.compress_cycle()
        self.assertEqual(self.engine.check_invariants(), [])

    def test_frozen_decoder_unchanged_after_growth(self):
        """Frozen Decoder Property: decoder behavior unchanged as vocab grows."""
        d = self.engine.decoder
        add  = self.engine.vocabulary.lookup_by_name("ADD")
        set_ = self.engine.vocabulary.lookup_by_name("SET")
        d1 = self.engine.vocabulary.define("JA", [add.token.identity, add.token.identity])
        d2 = self.engine.vocabulary.define("JB", [set_.token.identity, add.token.identity])
        for _ in range(5):
            self.engine.observe([d1.token, d2.token])
            self.engine.compress_cycle()

        test_token = ZCEToken(7, 3)
        d.clear()
        d.feed(test_token.to_bits())
        result = d.next_token()
        self.assertEqual(result.identity, 7)
        self.assertEqual(result.depth, 3)

    def test_stats_track(self):
        stats = self.engine.stats()
        self.assertEqual(stats["cycles_run"], 0)
        add = self.engine.vocabulary.lookup_by_name("ADD")
        for _ in range(3):
            self.engine.observe([add.token])
        self.engine.compress_cycle()
        stats2 = self.engine.stats()
        self.assertEqual(stats2["cycles_run"], 1)
        self.assertEqual(stats2["total_observations"], 3)


# ===========================================================================
# Theoretical Property Tests
# ===========================================================================

class TestTheoreticalProperties(unittest.TestCase):

    def test_theorem0_no_continuous_dimension(self):
        """Continuous identity raises — Theorem 0 forces discrete."""
        with self.assertRaises((ValueError, TypeError)):
            ZCEToken(1.5, 0)  # type: ignore

    def test_theorem1_minimum_token_is_two_bits(self):
        """Theorem 1 minimality: minimum self-delimiting token with M>=1 is 2 bits."""
        t = ZCEToken(0, 0)
        self.assertEqual(t.bit_length, 2)

    def test_theorem1_bit_length_formula(self):
        """Theorem 1: bit_length = N + depth + 2."""
        for n in range(10):
            for d in range(10):
                t = ZCEToken(n, d)
                self.assertEqual(t.bit_length, n + d + 2)

    def test_theorem1_m_always_at_least_one(self):
        """Theorem 1 uniqueness: M >= 1 guaranteed, stream is unambiguous."""
        for n in range(5):
            for d in range(5):
                bits = ZCEToken(n, d).to_bits()
                ones_count = sum(1 for b in bits[n:] if b == 1)
                self.assertGreaterEqual(ones_count, 1,
                    f"Token ({n},{d}) has no ones — M < 1, encoding is ambiguous")

    def test_theorem4_define_in_base_vocab(self):
        """Theorem 4: DEFINE in V₀, immutable."""
        v   = Vocabulary()
        sym = v.lookup_by_name("DEFINE")
        self.assertIsNotNone(sym)
        self.assertTrue(sym.is_base)
        self.assertEqual(sym.token.identity, BaseTokens.DEFINE)
        self.assertIn(BaseTokens.DEFINE, BaseTokens.IMMUTABLE)

    def test_theorem5_four_properties(self):
        """Theorem 5: all four SDTS properties present."""
        engine = UnderstandingEngine()
        v      = engine.vocabulary

        # N — named
        self.assertIsInstance(v.lookup_by_name("ADD").name, str)
        # D — discrete
        self.assertIsInstance(v.lookup_by_name("ADD").token.identity, int)
        # L — learnable
        self.assertIsInstance(engine._compressor, Compressor)
        # E — self-extension
        self.assertTrue(v.lookup_by_name("DEFINE").is_base)

    def test_readability_theorem_determinism(self):
        """Theorem 2: same token stream + same state = same result always."""
        engine = UnderstandingEngine()
        add    = engine.vocabulary.lookup_by_name("ADD")
        engine.register_action("ADD", lambda s: (s.set("r", s.get("a",0)+s.get("b",0)), s)[1])
        tokens = [add.token]
        results = set()
        for _ in range(5):
            r = engine.execute(tokens, {"a": 10, "b": 20})
            results.add(r.get("r"))
        self.assertEqual(len(results), 1)
        self.assertIn(30, results)

    def test_depth_invariant_monotone(self):
        """Theorem 0: depth can only increase as we define new symbols."""
        v   = Vocabulary()
        add = v.lookup_by_name("ADD")
        depths = [add.token.depth]  # 0
        s1 = v.define("L1", [add.token.identity, add.token.identity])
        depths.append(s1.token.depth)  # 1
        s2 = v.define("L2", [s1.token.identity, add.token.identity])
        depths.append(s2.token.depth)  # 2
        # Depths should be non-decreasing
        for i in range(len(depths)-1):
            self.assertLessEqual(depths[i], depths[i+1])


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestZCEToken,
        TestFrozenDecoder,
        TestVocabulary,
        TestTransitionCalculus,
        TestCompressor,
        TestUnderstandingEngine,
        TestTheoreticalProperties,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"\n{'='*60}")
    print(f"Tests run: {result.testsRun}  |  "
          f"Failures: {len(result.failures)}  |  "
          f"Errors: {len(result.errors)}")
    print(f"{'PASSED' if result.wasSuccessful() else 'FAILED'}")
    print(f"{'='*60}")
    sys.exit(0 if result.wasSuccessful() else 1)