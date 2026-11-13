# Rolling interference solve of the first 32 Weka sessions (all their requests, or the first n) on 4 caches, vs
# Mocker's router. Both measured in Mocker; Soldr's schedule replayed per cache at its start times.
# Usage, from the repo root: python roll32.py n|all window step limit_s out.json
#   e.g. caffeinate -i .venv/bin/python -u roll32.py all 32 16 30 roll32_all.json   (caffeinate: CP-SAT's time limit
#   is wall clock, so a sleeping Mac stretches windows far past it)
import json, sys, time
import numpy as np
from cache_packing import from_weka, default_latency as p
from calibrate import run_mocker
from validate import base_rows, agentic_rows
from interference_packing import solve_rolling

n = None if sys.argv[1] == "all" else int(sys.argv[1])
window, step, limit, out = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
K, C, dt, path = 32, 4, p.decode_ms, "traces/weka_256k_062126.jsonl"
trace6, blk = from_weka(path, sessions=K, n=n, stagger_s=10)
trace1, _ = from_weka(path, sessions=K, n=n, stagger_s=10, ms_per_step=1)
N = len(trace1)
print(f"{N} requests of {K} sessions, {C} caches; window {window}, step {step}, {limit:g} s per window", flush=True)

t = time.time()
vals = solve_rolling(trace1, blk, C, p.kv_capacity, window, step, limit, log=lambda s: print(s, flush=True))
solve_s = time.time() - t
print(f"solved in {solve_s:.0f} s", flush=True)

rows = base_rows(trace6)
meas = {k: [None] * N for k in ("ttft", "tpot", "e2e")}
for c in range(C):
    ix = sorted((j for j in range(N) if vals[j]["route"] == c), key=lambda j: (vals[j]["start"], j))
    for j, q in zip(ix, run_mocker([rows[j] | {"timestamp": vals[j]["start"]} for j in ix], 1, blk, p)):
        wait = vals[j]["start"] - vals[j]["release"]
        meas["ttft"][j], meas["e2e"][j] = wait + q["ttft_ms"], wait + q["e2e_latency_ms"]
        meas["tpot"][j] = (q["last_token_ms"] - q["first_token_ms"]) / (q["output_length"] - 1) \
            if q["output_length"] > 1 else None
pred = dict(ttft=[vals[j]["prefill_end"] - vals[j]["release"] for j in range(N)],
            e2e=[vals[j]["end"] - vals[j]["release"] for j in range(N)],
            tpot=[vals[j]["decode"] / (u.osl - 1) if u.osl > 1 else None for j, u in enumerate(trace1)])
router = []
for _ in range(2):
    recs = run_mocker(agentic_rows(trace6), C, blk, p)
    router.append(dict(ttft=[q["ttft_ms"] for q in recs], e2e=[q["e2e_latency_ms"] for q in recs],
                       tpot=[(q["last_token_ms"] - q["first_token_ms"]) / (q["output_length"] - 1)
                             if q["output_length"] > 1 else None for q in recs]))


def stats(xs):
    xs = np.array([x for x in xs if x is not None])
    return float(xs.mean()), float(np.percentile(xs, 90)), float(np.percentile(xs, 99))


res = dict(N=N, window=window, step=step, limit=limit, solve_s=solve_s,
           router={k: np.mean([stats(r[k]) for r in router], 0).tolist() for k in ("ttft", "tpot", "e2e")},
           soldr={k: stats(meas[k]) for k in ("ttft", "tpot", "e2e")},
           predicted={k: stats(pred[k]) for k in ("ttft", "tpot", "e2e")},
           router_e2e_runs=[stats(r["e2e"])[0] for r in router], vals={str(j): v for j, v in vals.items()})
json.dump(res, open(out, "w"))
print(f"\nMeasured in Mocker (ms): mean / P90 / P99; router averaged over {len(router)} runs "
      f"(mean e2e {', '.join(f'{x:.0f}' for x in res['router_e2e_runs'])})")
print(f"{'':22s} {'TTFT':>22s} | {'TPOT':>18s} | {'e2e':>24s}")
for name in ("router", "soldr", "predicted"):
    s = res[name]
    print(f"{name:22s} " + " | ".join(" ".join(f"{x:7.1f}" if k != "tpot" else f"{x:5.2f}" for x in s[k])
                                      for k in ("ttft", "tpot", "e2e")))
for k in ("ttft", "tpot", "e2e"):
    e = np.array([a - b for a, b in zip(pred[k], meas[k]) if a is not None])
    print(f"predicted - measured {k.upper()} per request: median {np.median(e):.2f}, MAE {np.abs(e).mean():.2f} ms")
