# Like calibrate_steps.py, but measures the CS 349E assn2 engine (HF transformers, cs349e-assn2/system/engine.py) on
# one real GPU instead of Mocker. Writes the same steps_<config>.json that interference_packing.StepLatency reads, and
# the raw measurements to steps_<config>_bench.json.
#
# How the assn2 engine differs from the vLLM engine Mocker simulates, and how that maps onto the step model:
#   - Prefill is one forward pass over the whole prompt (no chunking, no prefix cache): chunk = max_context, so every
#     prompt is one chunk, and only s = 0 is measured. g(n) is then measured directly at each n, and b (the n^2 term)
#     comes from a global a*n + b*n^2/2 + c fit, so interpolating between measured n stays quadratic.
#   - In MIXED (colocated) mode a prefill blocks the whole decode batch, so a prefill costs every decoding request its
#     full duration. That is the stall the solver's interference model already charges.
#   - Decode: B synthetic caches of length L admitted into a DecodeBatch, then DECODE_STEPS steps timed one by one.
#     Fit: step = f(B) + k * B * Lmid (Lmid: the mean context over the timed steps).
# Also records admission cost (DecodeBatch.admit copies the whole batch's KV), which the solver does not model.
#
# Usage, from soldr/, with the assn2 venv (it has torch + transformers):
#   ../cs349e-assn2/.venv/bin/python calibrate_harness.py qwen3-8b_a100x4 [--refit] [--gpu 0]
import argparse, json, os, statistics, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # as run_experiment.py
import numpy as np

ASSN2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cs349e-assn2")
DECODE_L = (512, 722, 1651, 2048, 4000, 6000)
DECODE_B = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96)
MAX_DECODE_TOKENS = 120_000          # B * L cap (~17.7 GB of KV) so a 40 GB A100 never OOMs
DECODE_FIT_TOKENS = 48_000         # decode fit region (B * L); see fit()
DECODE_WARMUP, DECODE_STEPS = 5, 40
PREFILL_N = (64, 128, 256, 512, 722, 1024, 1651, 2048, 3072, 4000, 4096, 6144, 8000)
PREFILL_REPS = 5


def measure(gpu, log=print):
    sys.path.insert(0, ASSN2)
    import torch
    from system.engine import Engine, Mode, make_prompt, synthetic_cache
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    engine = Engine(Mode.MIXED, device, max_slots=max(DECODE_B))
    sync = torch.cuda.synchronize

    # Warm-up, as worker._warmup does: the first prefill and decode step pay one-time CUDA setup
    engine.prefill(make_prompt(512, 0)); sync()

    prefill = []
    for n in PREFILL_N:
        ts = []
        for rep in range(PREFILL_REPS + 1):
            prompt = make_prompt(n, 1000 + rep)
            sync(); t = time.perf_counter()
            _, cache = engine.prefill(prompt)
            sync(); ts.append((time.perf_counter() - t) * 1e3)
            del cache
        ttft = statistics.median(ts[1:])
        prefill.append(dict(s=0, n=n, ttft=ttft, all=ts[1:]))
        log(f"prefill n {n:5d}  {ttft:8.2f} ms")

    decode = []
    for L in DECODE_L:
        cache = synthetic_cache(L, device, seed=L)
        for B in DECODE_B:
            if B * (L + DECODE_WARMUP + DECODE_STEPS) > MAX_DECODE_TOKENS:
                continue
            batch = engine.batch
            admit = []
            for i in range(B):
                sync(); t = time.perf_counter()
                batch.admit(i, cache, first_token=100, max_new=DECODE_WARMUP + DECODE_STEPS + 2)
                sync(); admit.append((time.perf_counter() - t) * 1e3)
            for _ in range(DECODE_WARMUP):
                batch.step()
            ts = []
            for _ in range(DECODE_STEPS):
                sync(); t = time.perf_counter()
                batch.step()                     # its .item() per slot syncs too, as in the real worker
                sync(); ts.append((time.perf_counter() - t) * 1e3)
            for i in range(B):
                batch.evict(i)
            tpot = statistics.median(ts)
            Lmid = L + DECODE_WARMUP + DECODE_STEPS / 2
            decode.append(dict(B=B, L=L, Lmid=Lmid, tpot=tpot, admit_ms_last=admit[-1], admit_ms_mean=float(np.mean(admit))))
            log(f"decode  B {B:3d}  L {L:5d}  step {tpot:7.2f} ms   admit (last) {admit[-1]:6.2f} ms")
            torch.cuda.empty_cache()
        del cache
    return dict(decode=decode, prefill=prefill)


def fit(decode, prefill, chunk):
    # Decode: f(B) per batch size + shared k per KV token of the B decoding requests
    Bs = sorted({r["B"] for r in decode})
    # The HF engine's step is a hinge: flat (~50 ms, launch-bound) until the batch holds ~20k KV tokens, then ~linear
    # in B * L. A free f(B) plus shared k fits that well, but then f(B) slopes *down* and StepLatency's B <= 8 linear
    # f1 + s (B - 1) gets s < 0 (negative step times at large B). So fit the solver's own linear form,
    # step = f1 + s (B - 1) + k B L with s, k >= 0, on the operating region B * L <= DECODE_FIT_TOKENS, and write
    # decode_f = f1 + s (B - 1) so StepLatency recovers f1 and s exactly. The free fit is kept as decode_f_free.
    X = np.array([[r["B"] == B for B in Bs] + [r["B"] * r["Lmid"]] for r in decode], float)
    y = np.array([r["tpot"] for r in decode])
    free, *_ = np.linalg.lstsq(X, y, rcond=None)
    op = [r for r in decode if r["B"] * r["L"] <= DECODE_FIT_TOKENS]
    Xo = np.array([[1, r["B"] - 1, r["B"] * r["Lmid"]] for r in op], float)
    (f1, s, k), *_ = np.linalg.lstsq(Xo, np.array([r["tpot"] for r in op]), rcond=None)
    if s < 0:
        (f1, k), *_ = np.linalg.lstsq(Xo[:, [0, 2]], np.array([r["tpot"] for r in op]), rcond=None)
        s = 0.0
    assert k >= 0
    fB = [f1 + s * (B - 1) for B in Bs]
    pred = np.array([f1 + s * (r["B"] - 1) + k * r["B"] * r["Lmid"] for r in decode])
    decode_rel = np.abs(pred - y) / y
    decode_rel = dict(all=decode_rel, op=decode_rel[[r["B"] * r["L"] <= DECODE_FIT_TOKENS for r in decode]])
    # Prefill: global a*n + b*n^2/2 + c for b, then g(n) = measured - b*n^2/2 (exact at each measured n)
    n = np.array([r["n"] for r in prefill], float)
    t = np.array([r["ttft"] for r in prefill])
    (a, b, c), *_ = np.linalg.lstsq(np.c_[n, n ** 2 / 2, np.ones_like(n)], t, rcond=None)
    gn = t - b * n ** 2 / 2
    prefill_rel = np.abs(a * n + b * n ** 2 / 2 + c - t) / t
    steps = dict(chunk=chunk, decode_B=[int(B) for B in Bs], decode_f=[float(x) for x in fB], decode_k=float(k),
                 prefill_n=[int(x) for x in n], prefill_g=[float(x) for x in gn], prefill_b=float(b),
                 decode_f_free=[float(x) for x in free[:-1]], decode_k_free=float(free[-1]),
                 decode_fit_tokens=DECODE_FIT_TOKENS)
    return steps, dict(a=float(a), b=float(b), c=float(c)), decode_rel, prefill_rel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="e.g. qwen3-8b_a100x4 (params_<config>.json)")
    ap.add_argument("--refit", action="store_true", help="fit the saved measurements again, without the GPU")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    params_path = f"params_{args.config}.json"
    params = json.load(open(params_path))
    out, bench_path = f"steps_{args.config}.json", f"steps_{args.config}_bench.json"
    if args.refit:
        bench = json.load(open(bench_path))
    else:
        bench = dict(config=args.config, source="cs349e-assn2 engine", gpu=args.gpu) | measure(args.gpu)
        json.dump(bench, open(bench_path, "w"), indent=1)

    steps, abc, decode_rel, prefill_rel = fit(bench["decode"], bench["prefill"], params["chunk"])
    json.dump(dict(config=args.config) | steps, open(out, "w"), indent=1)

    # Refresh the cache_packing-style params from the same measurements: a, b, c from the prefill fit (one chunk, so c
    # is the per-pass overhead), decode_ms = a lone decode at a typical context
    f1 = float(np.interp(1, steps["decode_B"], steps["decode_f"]))
    params |= abc | dict(decode_ms=max(1, round(f1 + steps["decode_k"] * 1000)))
    json.dump(params, open(params_path, "w"), indent=1)

    print(f"wrote {out}, {bench_path}, updated {params_path}")
    s = steps["decode_f"][1] - steps["decode_f"][0] if len(steps["decode_B"]) > 1 else 0.0
    print(f"  decode:  {len(bench['decode'])} runs; linear fit on B*L <= {DECODE_FIT_TOKENS}: error median "
          f"{np.median(decode_rel['op']):.1%}, max {decode_rel['op'].max():.1%} (all runs: max "
          f"{decode_rel['all'].max():.1%});  f1 {f1:.2f} ms, s {s:.3f} ms, k {steps['decode_k'] * 1e3:.3f} us/token")
    print(f"  prefill: {len(bench['prefill'])} runs, quadratic fit error median {np.median(prefill_rel):.1%}, "
          f"max {prefill_rel.max():.1%};  a {abc['a']:.4g} b {abc['b']:.3g} c {abc['c']:.2f}")
    print(f"  decode_ms -> {params['decode_ms']}")
