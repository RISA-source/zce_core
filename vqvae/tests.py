"""ZCE continuous-to-discrete interface — complete test suite."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vqvae.quantized_ae import QuantizedAE
from vqvae.integration import ZCEInterface
from vqvae.trainer import make_clustered_data, make_binary_shapes

_ok=[0]; _fail=[0]
def check(label,cond,note=""):
    sym="✓" if cond else "✗"
    print(f"  {sym}  {label}"+(f"  [{note}]" if note else ""))
    if cond: _ok[0]+=1
    else: _fail[0]+=1
    return cond
def sec(t): print(f"\n{'═'*60}\n  {t}\n{'═'*60}\n")

def test_qae_phases():
    sec("1. QUANTIZED AE — THREE PHASES")
    data,labels,_=make_clustered_data(200,32,4,noise=0.05,seed=0)
    qae=QuantizedAE(input_dim=32,n_components=8,n_tokens=4,seed=0)
    m=qae.fit(data,verbose=True); print()
    check("Explained variance > 80%", m['explained_variance']>0.8, f"{m['explained_variance']:.1%}")
    check("All tokens used",          m['tokens_used']==4,         f"{m['tokens_used']}/4")
    check("Reconstruction MSE < 0.1", m['reconstruction_mse']<0.1, f"{m['reconstruction_mse']:.4f}")
    t1=qae.encode(data); t2=qae.encode(data)
    check("Deterministic",   (t1==t2).all())
    check("Tokens in range", (t1>=0).all() and (t1<4).all())

def test_clustered():
    sec("2. CLUSTERED DATA — N_TOKENS = N_CLUSTERS")
    n=6
    data,labels,_=make_clustered_data(300,32,n,noise=0.05,seed=10)
    qae=QuantizedAE(input_dim=32,n_components=8,n_tokens=n,seed=10)
    qae.fit(data,verbose=False)
    r=qae.evaluate(data,labels,verbose=True); print()
    check("Reconstruction MSE < 0.1",  r['recon_mse']<0.1,           f"{r['recon_mse']:.4f}")
    check("Token consistency > 90%",   r['avg_consistency']>0.9,      f"{r['avg_consistency']:.1%}")
    check("All clusters → own token",  r['n_unique_dominant']==n,      f"{r['n_unique_dominant']}/{n}")
    check("All tokens used",           r['n_unique_tokens']==n,        f"{r['n_unique_tokens']} tokens")

def test_shapes():
    sec("3. BINARY SHAPES — VISUAL GROUNDING (N_TOKENS = N_SHAPES)")
    shape_names=['empty','h-line','v-line','diagonal','block','cross','border','checker']
    data,labels,_=make_binary_shapes(400,8,seed=20)
    qae=QuantizedAE(input_dim=64,n_components=16,n_tokens=8,seed=20)
    qae.fit(data,verbose=False)
    r=qae.evaluate(data,labels,verbose=True); print()
    for c,(dom,cons) in enumerate(zip(r['dominant_tokens'],r['consistency_per_class'])):
        print(f"  {shape_names[c]:<12} → T{dom:02d}  ({cons:.0%})")
    print()
    check("Reconstruction MSE < 0.1",     r['recon_mse']<0.1,           f"{r['recon_mse']:.4f}")
    check("Token consistency > 90%",      r['avg_consistency']>0.9,      f"{r['avg_consistency']:.1%}")
    check("All 8 shapes → unique tokens", r['n_unique_dominant']==8,     f"{r['n_unique_dominant']}/8")

def test_sdts_on_streams():
    sec("4. SDTS COMPRESSION ON TOKEN STREAMS")
    from core import ZCEToken
    from understanding_engine import UnderstandingEngine
    engine=UnderstandingEngine(min_frequency=3,min_savings=0,max_pattern_len=4)
    for i,name in enumerate(['TA','TB','TC','TD']):
        engine.vocabulary.define(name=name,components=[],identity=5+i)
    def tok(n): return ZCEToken(identity=int(n),depth=0)
    rng=np.random.default_rng(42)
    streams=[]
    for _ in range(60):
        s=[tok(5),tok(6),tok(5),tok(6)]
        for _ in range(rng.integers(0,2)): s=[tok(int(rng.integers(5,9)))]+s
        streams.append(s)
    for _ in range(40):
        s=[tok(7),tok(8),tok(7)]
        for _ in range(rng.integers(0,2)): s=s+[tok(int(rng.integers(5,9)))]
        streams.append(s)
    def dl(ss): return sum(sum(t.bit_length for t in s) for s in ss)
    init=dl(streams)
    for s in streams: engine.observe(s)
    new_syms=engine.compress_cycle(max_new_symbols=5,clear_after=True)
    streams=[engine.rewrite(s) for s in streams]
    final=dl(streams)
    print(f"  Initial DL: {init:,}  →  Final DL: {final:,}  ({100*(init-final)/init:.1f}% saved)")
    print(f"  New symbols: {[s.name for s in new_syms]}")
    print()
    check("SDTS finds patterns",    len(new_syms)>0,    f"{len(new_syms)} symbols")
    check("DL decreases",           final<init,          f"{init}→{final}")
    check("Invariants hold",        len(engine.check_invariants())==0)

def test_full_pipeline():
    sec("5. FULL PIPELINE: QAE + SDTS INTEGRATION")
    data,labels,_=make_clustered_data(300,32,4,noise=0.05,seed=30)
    iface=ZCEInterface(input_dim=32,n_components=8,n_codebook=4,min_frequency=2,seed=30)
    results=iface.full_pipeline(data=data,labels=labels,n_cycles=3,verbose=True)
    check("Invariants hold",       results['invariant_ok'])
    check("Base tokens installed", len(results['final_vocab'])>=4+5, f"size={len(results['final_vocab'])}")
    ev=results['eval_results']
    check("Consistency > 90%",    ev['avg_consistency']>0.9,  f"{ev['avg_consistency']:.1%}")
    check("All classes tokenized", ev['n_unique_dominant']==4, f"{ev['n_unique_dominant']}/4")

def test_frozen_decoder():
    sec("6. FROZEN DECODER — INVARIANT ACROSS VOCAB GROWTH")
    from core import ZCEToken
    iface=ZCEInterface(input_dim=32,n_components=8,n_codebook=4,seed=40)
    test_tok=ZCEToken(7,2)
    def decode():
        d=iface.engine.decoder; d.clear(); d.feed(test_tok.to_bits())
        r=d.next_token(); return (r.identity,r.depth)
    before=decode()
    check("Decoder correct before fit", before==(7,2), f"{before}")
    data,_,_=make_clustered_data(200,32,4,seed=40)
    iface.fit(data,verbose=False)
    iface.run_sdts_cycles(data,n_cycles=3,verbose=False)
    after=decode()
    check("Decoder unchanged after vocab growth", after==before, f"before={before} after={after}")
    check("Frozen Decoder Property",              after==(7,2),  f"vocab={len(iface.engine.vocabulary)}")

def test_symbol_grounding():
    sec("7. SYMBOL GROUNDING — SHAPES GET UNIQUE TOKENS")
    shape_names=['empty','h-line','v-line','diagonal','block','cross','border','checker']
    data,labels,_=make_binary_shapes(800,8,seed=42)
    qae=QuantizedAE(input_dim=64,n_components=16,n_tokens=8,seed=42)
    qae.fit(data,verbose=False)
    r=qae.evaluate(data,labels,verbose=False)
    print("  Shape → ZCE Token (symbol grounding):")
    for c,(dom,cons) in enumerate(zip(r['dominant_tokens'],r['consistency_per_class'])):
        print(f"  T{dom:02d}  ←  {shape_names[c]:<12}  ({cons:.0%})")
    print()
    check("Each shape → unique token", r['n_unique_dominant']==8, f"{r['n_unique_dominant']}/8")
    check("Consistency > 90%",         r['avg_consistency']>0.9,  f"{r['avg_consistency']:.1%}")

if __name__=="__main__":
    print("\n╔"+"═"*58+"╗")
    print("║  ZCE CONTINUOUS-TO-DISCRETE INTERFACE — TEST SUITE     ║")
    print("╚"+"═"*58+"╝")
    test_qae_phases()
    test_clustered()
    test_shapes()
    test_sdts_on_streams()
    test_full_pipeline()
    test_frozen_decoder()
    test_symbol_grounding()
    total=_ok[0]+_fail[0]
    print(f"\n{'═'*60}")
    print(f"  Results: {_ok[0]}/{total} passed")
    print("  ALL TESTS PASSED ✓" if _fail[0]==0 else f"  {_fail[0]} FAILED ✗")
    print(f"{'═'*60}\n")
