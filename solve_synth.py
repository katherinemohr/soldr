# Solve the synthetic P-D trace (synth_pd.py) with the rolling solver on 4 workers, colocated, with
# each request free to split (topology "split", no pools), or on dedicated pools ("2p2d": workers 0-1 only prefill,
# 2-3 only decode; the solver still picks each request's prefill and decode worker and its timing). Describes how
# disaggregated the schedule is: how many requests split, and how each worker's time divides between prefills and
# decodes. A dedicated-pool schedule is also replayed in Mocker's disagg topology, pinned to its workers and
# submitted at its starts.
# Usage, from the repo root: python solve_synth.py colocated|split|<P>p<D>d window step limit_s out.json [n]
#   e.g. caffeinate -i .venv/bin/python -u solve_synth.py split 32 16 30 synth_split.json
#   n: solve only the first n requests
import json, re, sys, time
import numpy as np
from cache_packing import default_latency as p
from interference_packing import solve_rolling, KV_BANDWIDTH_GBPS
from synth_pd import TRACE, measured, run_mocker_pd, synthetic_trace

C = 4


def predicted(trace, vals):
    out = dict(ttft=[], tpot=[], e2e=[])
    for j, u in enumerate(trace):
        x = vals[j]
        split = x.get("droute", x["route"]) != x["route"]
        first = x["decode"] // u.osl if split else 0
        out["ttft"].append((x["decode_start"] + first if split else x["prefill_end"]) - x["release"])
        out["tpot"].append((x["decode"] - first) / (u.osl - 1) if u.osl > 1 else np.nan)
        out["e2e"].append(x["end"] - x["release"])
    return {k: np.array(v, float) for k, v in out.items()}


def describe(trace, vals):
    N = len(trace)
    d = [vals[j].get("droute", vals[j]["route"]) for j in range(N)]
    r = [vals[j]["route"] for j in range(N)]
    split = [a != b for a, b in zip(r, d)]
    print(f"  requests split (decode on another worker than prefill): {sum(split)}/{N}")
    end = max(x["end"] for x in vals.values())
    grid = np.arange(0, end, 10.0)
    print(f"  {'worker':6s} {'prefills':>8s} {'decodes':>8s} | {'busy prefilling':>15s} {'busy decoding':>13s} "
          f"{'both at once':>12s} | {'mean decodes when decoding':>26s}")
    for c in range(C):
        pf = np.zeros(len(grid), bool); dec = np.zeros(len(grid))
        for j in range(N):
            x = vals[j]
            if r[j] == c:
                pf |= (grid >= x["start"]) & (grid < x["prefill_end"])
            if d[j] == c:
                ds = x.get("decode_start", x["prefill_end"])
                dec += (grid >= ds) & (grid < x["end"])
        span = len(grid)
        print(f"  {c:6d} {sum(a == c for a in r):8d} {sum(b == c for b in d):8d} | {pf.mean():15.0%} "
              f"{(dec > 0).mean():13.0%} {(pf & (dec > 0)).mean():12.0%} | {dec[dec > 0].mean() if dec.any() else 0:26.1f}")


def summary(name, m):
    print(f"  {name:34s} TTFT {np.nanmean(m['ttft']):7.1f} (P90 {np.nanpercentile(m['ttft'], 90):7.1f}) | "
          f"TPOT {np.nanmean(m['tpot']):5.2f} (P90 {np.nanpercentile(m['tpot'], 90):5.2f}) | "
          f"e2e {np.nanmean(m['e2e']):7.1f} (P90 {np.nanpercentile(m['e2e'], 90):7.1f})")


# Replays a dedicated-pool schedule (prefill workers 0..P-1, decode workers P..C-1) in Mocker's PxD topology, each
# request pinned to its workers and submitted at its solved start; latencies measured from its release
def mocker_pinned(trace, vals, P):
    N = len(trace)
    pins = [(vals[j]["route"], vals[j]["droute"] - P) for j in range(N)]
    submit = [float(vals[j]["start"]) for j in range(N)]
    recs = run_mocker_pd(trace, p.kv_block, P, C - P, p, KV_BANDWIDTH_GBPS, submit=submit, pins=pins)
    m = measured(recs)
    wait = np.array([vals[j]["start"] - vals[j]["release"] for j in range(N)], float)
    return dict(ttft=m["ttft"] + wait, tpot=m["tpot"], e2e=m["e2e"] + wait)


if __name__ == "__main__":
    topology, window, step, limit, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    n = int(sys.argv[6]) if len(sys.argv) > 6 else TRACE["n"]
    pools = P = None
    if m := re.fullmatch(r"(\d+)p(\d+)d", topology):
        P = int(m[1])
        assert P + int(m[2]) == C
        pools = (list(range(P)), list(range(P, C)))
    trace = synthetic_trace(TRACE["n"], TRACE["rate"], TRACE["isl"], TRACE["osl"], TRACE["seed"])[:n]
    print(f"{len(trace)} synthetic requests, {C} workers, topology {topology}; window {window}, step {step}, "
          f"{limit:g} s per window", flush=True)
    t = time.time()
    vals = solve_rolling(trace, p.kv_block, C, p.kv_capacity, window, step, limit,
                         topology="colocated" if topology == "colocated" else "split", pools=pools,
                         log=lambda s: print(s, flush=True))
    print(f"solved in {time.time() - t:.0f} s")
    json.dump(dict(topology=topology, window=window, step=step, limit=limit, vals={str(j): v for j, v in vals.items()}),
              open(out, "w"))
    describe(trace, vals)
    summary("predicted by the model", predicted(trace, vals))
    if P is not None:
        summary(f"measured in Mocker {topology.upper()}, pinned", mocker_pinned(trace, vals, P))
