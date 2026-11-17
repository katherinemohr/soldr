# Regenerates steps_<config>.json, the step-level latency model of Mocker's engine that interference_packing's
# StepLatency reads, for an LLM + hardware configuration. Runs two microbenchmarks on a single Mocker worker set up as
# calibrate.mocker_engine(params_<config>.json) does, and fits them by least squares (as tpot_sim.StepModel does):
#   decode: B requests decoding at context ~L (B - 1 in the background, then a probe); the probe's TPOT is the step
#     time with B decodes. Fit: tpot = f(B) + k * B * (L + 300), f(B) free per batch size and one slope k per KV token
#     (L + 300: the context grows as the requests decode).
#   prefill: a prefill of n new tokens after a cached prefix of s tokens, alone on the engine; its TTFT. Fit on single
#     chunks (n <= chunk): ttft = g(n) + b * n * (s + n / 2), g(n) free per chunk size and one attention slope b.
#     Longer prefills (several chunks) are only used to check the chunked prediction StepLatency makes from the fit.
# Needs params_<config>.json (see calibrate.MockerLatency) for the engine: chunk, kv_block, kv_capacity, max_context
# and aic (model and system for Mocker's AIC timing model). Its a, b, c (calibrate.calibrate) only space the decode
# benchmark's probe after the background prefills; if they're off, the spacing is doubled until it holds.
# Writes steps_<config>.json and the measurements, steps_<config>_bench.json (--refit fits that file again without
# rerunning Mocker). Takes about a minute.
# Usage, from the repo root: python calibrate_steps.py <config> [--out steps.json] [--refit]
#   e.g. .venv/bin/python calibrate_steps.py qwen3-30b-a3b_h100x4
import argparse, bisect, json
from math import ceil
import numpy as np
from calibrate import MockerLatency, run_mocker

DECODE_L = (1024, 8192, 32768, 131072)
DECODE_B = (1, 2, 4, 8, 16, 32, 64, 128)
BACKGROUND_OSL, PROBE_OSL = 3000, 200
PREFILL_S = (0, 8192, 32768, 98304)
PREFILL_N = (256, 512, 1024, 2048, 4096, 6144, 8192, 12288, 16384, 32768, 65536, 131072)


class Hashes:
    # Fresh block hashes, so no two prompts share a prefix unless a benchmark wants them to
    def __init__(self, blk):
        self.blk, self.next = blk, 0

    def __call__(self, tokens):
        n = ceil(tokens / self.blk)
        self.next += n
        return list(range(self.next - n, self.next))


def decode_bench(p, log=print):
    hashes, out = Hashes(p.kv_block), []
    req = lambda t, isl, osl: {"timestamp": t, "input_length": isl, "output_length": osl, "hash_ids": hashes(isl)}
    for L in DECODE_L:
        for B in DECODE_B:
            if B * (L + BACKGROUND_OSL) > 0.9 * p.kv_capacity or L + BACKGROUND_OSL > p.max_context:
                continue
            # The probe arrives once the B - 1 background prefills are done. The first row (t = 0) anchors Mocker's
            # clock, which starts at the first request, when there's no background
            t0 = 1.5 * p.prefill_ms(L) * (B - 1) + 200
            while True:
                rows = [req(0, p.kv_block, 1)] + [req(0, L, BACKGROUND_OSL) for _ in range(B - 1)] \
                    + [req(t0, L, PROBE_OSL)]
                recs = run_mocker(rows, 1, p.kv_block, p)
                q = recs[-1]
                if max(r["first_token_ms"] for r in recs[:-1]) < q["arrival_time_ms"]:
                    break
                t0 *= 2
            # ... and the background still decodes until the probe is done, so it decodes with B - 1 others throughout
            assert all(r["last_token_ms"] >= q["last_token_ms"] for r in recs[1:-1]), \
                f"background finished before the probe (B {B}, L {L}): raise BACKGROUND_OSL"
            tpot = (q["last_token_ms"] - q["first_token_ms"]) / (q["output_length"] - 1)
            out.append(dict(B=B, L=L, tpot=tpot, readmit=sum(r["readmission_count"] for r in recs)))
            log(f"decode  B {B:3d}  L {L:6d}  TPOT {tpot:7.3f} ms")
    return out


def prefill_bench(p, log=print):
    hashes, out = Hashes(p.kv_block), []
    for s in PREFILL_S:
        for n in PREFILL_N:
            if s + n > p.max_context - 16384:
                continue
            # The first request caches the prefix (at least one block, which also anchors Mocker's clock); the second,
            # a minute later and alone on the engine, prefills the rest
            h, first = hashes(s + n), max(s, p.kv_block)
            rows = [{"timestamp": 0, "input_length": first, "output_length": 1, "hash_ids": h[:first // p.kv_block]},
                    {"timestamp": 60_000, "input_length": s + n, "output_length": 1, "hash_ids": h}]
            warm, q = run_mocker(rows, 1, p.kv_block, p)
            assert warm["last_token_ms"] < q["arrival_time_ms"], f"prefix prefill not done in a minute (s {s}, n {n})"
            hit = q["reused_input_tokens"]
            out.append(dict(s=hit, n=s + n - hit, ttft=q["ttft_ms"]))
            log(f"prefill s {hit:6d}  n {s + n - hit:6d}  TTFT {q['ttft_ms']:9.2f} ms")
    return out


def fit(decode, prefill, chunk):
    # Decode: f(B) per batch size + shared k per KV token of the B decoding requests
    Bs = sorted({r["B"] for r in decode})
    X = np.array([[r["B"] == B for B in Bs] + [r["B"] * (r["L"] + 300)] for r in decode], float)
    y = np.array([r["tpot"] for r in decode])
    sol, *_ = np.linalg.lstsq(X, y, rcond=None)
    fB, k, decode_err = sol[:-1], sol[-1], np.abs(X @ sol - y).max()
    # Prefill: single chunks; g(n) per chunk size + shared attention slope b
    one = [r for r in prefill if r["n"] <= chunk]
    ns = sorted({r["n"] for r in one})
    X = np.array([[r["n"] == n for n in ns] + [r["n"] * (r["s"] + r["n"] / 2)] for r in one], float)
    y = np.array([r["ttft"] for r in one])
    sol, *_ = np.linalg.lstsq(X, y, rcond=None)
    gn, b, prefill_err = sol[:-1], sol[-1], np.abs(X @ sol - y).max()
    steps = dict(chunk=chunk, decode_B=[int(B) for B in Bs], decode_f=[float(x) for x in fB], decode_k=float(k),
                 prefill_n=[int(n) for n in ns], prefill_g=[float(x) for x in gn], prefill_b=float(b))
    return steps, decode_err, prefill_err


def interp(xs, ys, x):
    # As in interference_packing: piecewise linear, linear extrapolation at both ends
    i = min(max(bisect.bisect_left(xs, x), 1), len(xs) - 1)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def chunked_errors(steps, prefill):
    # StepLatency's prediction for prefills of several chunks, from the single-chunk fit
    errs = []
    for r in prefill:
        if r["n"] <= steps["chunk"]:
            continue
        t, done = 0.0, r["s"]
        while done < r["s"] + r["n"]:
            n = min(r["s"] + r["n"] - done, steps["chunk"])
            t += interp(steps["prefill_n"], steps["prefill_g"], n) + steps["prefill_b"] * n * (done + n / 2)
            done += n
        errs.append((t - r["ttft"]) / r["ttft"])
    return np.array(errs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="LLM + hardware configuration, e.g. qwen3-30b-a3b_h100x4 (params_<config>.json)")
    ap.add_argument("--out", help="output file (default steps_<config>.json)")
    ap.add_argument("--refit", action="store_true", help="fit the saved measurements again, without running Mocker")
    args = ap.parse_args()
    p = MockerLatency.load(args.config)
    out = args.out or f"steps_{args.config}.json"
    bench_path = out.removesuffix(".json") + "_bench.json"
    if args.refit:
        bench = json.load(open(bench_path))
    else:
        bench = dict(config=args.config, decode=decode_bench(p), prefill=prefill_bench(p))
        json.dump(bench, open(bench_path, "w"), indent=1)
    steps, decode_err, prefill_err = fit(bench["decode"], bench["prefill"], p.chunk)
    json.dump(dict(config=args.config) | steps, open(out, "w"), indent=1)
    readmits = sum(r["readmit"] for r in bench["decode"])
    chunked = chunked_errors(steps, bench["prefill"])
    print(f"wrote {out} (measurements in {bench_path})")
    print(f"  decode:  {len(bench['decode'])} runs, max fit error {decode_err:.2f} ms"
          + (f", {readmits} readmissions (preemptions) in the runs" if readmits else ""))
    print(f"  prefill: {sum(r['n'] <= p.chunk for r in bench['prefill'])} single-chunk runs, max fit error "
          f"{prefill_err:.2f} ms; chunked prediction on {len(chunked)} longer prefills: median error "
          f"{np.median(chunked):+.1%}, max |error| {np.abs(chunked).max():.1%}")
