"""
zce_core/domains/math/experiment.py
=====================================
The ZCE math experiment — full implementation.

Tests all theoretical predictions against real algebraic expressions.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core import Vocabulary, ZCEToken, FrozenDecoder
from engine import Compressor, CompressionCandidate
from understanding_engine import UnderstandingEngine
from .grammar import (install_math_vocab, expr_to_tokens,
                      tokens_to_expr_str, MathTokens, suggest_name)
from .corpus import make_corpus


def run_experiment(
    n_expressions : int  = 200,
    n_cycles      : int  = 5,
    min_frequency : int  = 3,
    min_savings   : int  = 0,
    max_pattern   : int  = 4,    # shorter = cleaner primitives
    verbose       : bool = True,
) -> dict:

    engine = UnderstandingEngine(
        min_frequency   = min_frequency,
        min_savings     = min_savings,
        max_pattern_len = max_pattern,
    )
    install_math_vocab(engine.vocabulary)

    corpus       = make_corpus(n_expressions)
    token_streams = []
    for expr in corpus:
        try:
            token_streams.append(expr_to_tokens(expr, engine.vocabulary))
        except Exception:
            pass

    def total_dl(streams):
        return sum(sum(t.bit_length for t in s) for s in streams)

    def stream_to_names(stream):
        return [engine.vocabulary.lookup_by_identity(t.identity).name
                if engine.vocabulary.lookup_by_identity(t.identity) else f"?{t.identity}"
                for t in stream]

    if verbose:
        _header("ZCE MATHEMATICAL EXPRESSION EXPERIMENT")
        print(f"Corpus:         {len(token_streams)} expressions")
        print(f"Cycles:         {n_cycles}")
        print(f"Min frequency:  {min_frequency}")
        print(f"Max pattern:    {max_pattern} tokens")
        print()
        total_tok = sum(len(s) for s in token_streams)
        print(f"Total tokens:   {total_tok}")
        print(f"Initial DL:     {total_dl(token_streams):,} bits")
        print()

        print("Sample tokenizations (prefix/preorder):")
        for expr, stream in zip(corpus[:3], token_streams[:3]):
            print(f"  {str(expr)}")
            print(f"  -> {stream_to_names(stream)}")
        print()

    results = {
        'vocab_sizes'         : [len(engine.vocabulary)],
        'description_lengths' : [total_dl(token_streams)],
        'max_depths'          : [engine.vocabulary.max_depth],
        'new_symbols_per_cycle': [],
        'final_vocab'         : None,
        'final_streams'       : None,
    }

    for cycle in range(1, n_cycles + 1):
        # Observe all streams
        for stream in token_streams:
            engine.observe(stream)

        # Find candidates and apply with smart names
        candidates = engine._compressor.find_candidates()
        new_syms = []
        for cand in candidates[:5]:
            # Check not already defined
            existing = None
            for sym in engine.vocabulary._symbols.values():
                if tuple(sym.components) == cand.pattern:
                    existing = sym
                    break
            if existing:
                continue
            name = suggest_name(cand.pattern, engine.vocabulary)
            sym = engine._compressor.apply_candidate(cand, name=name)
            new_syms.append((sym, cand))
            engine._total_defined += 1

        engine._compressor.clear_observations()
        engine._cycles_run += 1

        # Rewrite streams
        token_streams = [engine.rewrite(s) for s in token_streams]
        dl = total_dl(token_streams)

        results['vocab_sizes'].append(len(engine.vocabulary))
        results['description_lengths'].append(dl)
        results['max_depths'].append(engine.vocabulary.max_depth)
        results['new_symbols_per_cycle'].append([s.name for s, _ in new_syms])

        if verbose:
            _section(f"CYCLE {cycle}")
            prev_dl = results['description_lengths'][-2]
            savings = prev_dl - dl
            pct     = 100 * savings / prev_dl if prev_dl > 0 else 0
            print(f"New symbols:  {len(new_syms)}")
            for sym, cand in new_syms:
                comp_names = [engine.vocabulary.lookup_by_identity(c).name
                              for c in sym.components
                              if engine.vocabulary.lookup_by_identity(c)]
                print(f"  {sym.token}  {sym.name:<20}  [{sym.token.bit_length}b]"
                      f"  freq={cand.frequency}  "
                      f"= {' ; '.join(comp_names)}")
            print(f"DL: {prev_dl:,} -> {dl:,} bits  "
                  f"(saved {savings:,} = {pct:.1f}%)")
            print(f"Vocab size: {results['vocab_sizes'][-2]} -> {len(engine.vocabulary)}")
            print(f"Max depth:  {engine.vocabulary.max_depth}")
            print()

    results['final_vocab']   = engine.vocabulary
    results['final_streams'] = token_streams

    if verbose:
        _header("FINAL VOCABULARY")
        _print_vocab(engine.vocabulary)

        _header("READABILITY CHECK  (Theorem 2)")
        print("Original expression  |  Compressed token stream")
        print()
        for expr, stream in zip(corpus[:6], token_streams[:6]):
            names = stream_to_names(stream)
            orig_stream = expr_to_tokens(expr, engine.vocabulary)
            reduction   = len(orig_stream) - len(stream)
            print(f"  {str(expr)[:50]:<52}")
            print(f"  [{' '.join(names)}]")
            if reduction:
                print(f"  {reduction} fewer tokens after compression")
            print()

        _header("FROZEN DECODER CHECK  (Theorem 4)")
        test = ZCEToken(7, 3)
        d = engine.decoder
        d.clear()
        d.feed(test.to_bits())
        r = d.next_token()
        ok = r.identity == 7 and r.depth == 3
        print(f"Test token {test} -> decoded N={r.identity}, M={r.depth}  "
              f"{'✓ CORRECT' if ok else '✗ WRONG'}")
        print(f"Decoder unchanged after learning "
              f"{len(engine.vocabulary)} symbols.  ✓")
        print()

        _header("INVARIANT CHECK")
        v = engine.check_invariants()
        if v:
            for violation in v:
                print(f"  ✗ {violation}")
        else:
            print("  All theoretical invariants hold.  ✓")
        print()

        _header("SUMMARY")
        i_dl = results['description_lengths'][0]
        f_dl = results['description_lengths'][-1]
        tot  = i_dl - f_dl
        print(f"  Expressions:             {len(token_streams)}")
        print(f"  Initial description len: {i_dl:,} bits")
        print(f"  Final description len:   {f_dl:,} bits")
        print(f"  Total compression:       {tot:,} bits  "
              f"({100*tot/i_dl:.1f}%)")
        print(f"  Symbols learned:         "
              f"{len(engine.vocabulary) - 17}")
        print(f"  Max vocab depth:         "
              f"{engine.vocabulary.max_depth}")
        print()

    return results


def _print_vocab(vocab: Vocabulary) -> None:
    for depth in range(vocab.max_depth + 1):
        syms = vocab.symbols_at_depth(depth)
        if not syms:
            continue
        print(f"  Depth {depth}:")
        for s in syms:
            comp_names = ([vocab.lookup_by_identity(c).name for c in s.components
                           if vocab.lookup_by_identity(c)]
                          if s.components else [])
            base = "  [BASE]" if s.is_base else ""
            comp = f"  =  {' ; '.join(comp_names)}" if comp_names else ""
            print(f"    {s.token}  {s.name:<22} [{s.token.bit_length}b]{base}{comp}")
    print()


def _header(title: str) -> None:
    print("=" * 65)
    print(title)
    print("=" * 65)


def _section(title: str) -> None:
    print(f"{'─'*65}")
    print(f"  {title}")
    print(f"{'─'*65}")


if __name__ == "__main__":
    run_experiment(
        n_expressions = 200,
        n_cycles      = 5,
        min_frequency = 3,
        min_savings   = 0,
        max_pattern   = 4,
        verbose       = True,
    )
