# Solve a CS 349E assn2 trace with the rolling interference solver and write a schedule that the assn2 harness can
# replay (cs349e-assn2/scripts/run_schedule.py).
#
# Arrivals are rebuilt exactly as assn2's Controller._arrival_times does (random.Random(seed).expovariate(rate) per
# request), so the solver and the harness see the same arrival pattern. Rates are the sweep's points
# (sweep_cs349e.py: RANGES, --points 10), and --rate must be one of them unless --any-rate is given.
#
# Everything is in ms. Predictions use the harness's metric definitions (system/metrics.py):
#   TTFT = prefill end - arrival   (token_times[0] is prefill_end_t, also when P and D are split)
#   TPOT = (last token - prefill end) / (osl - 1)   (so a split request's KV transfer counts toward TPOT)
#
# Usage, from soldr/:
#   .venv/bin/python assn2.py convert                                 # -> traces/assn2/<trace>_r<rate>_n300_s1234.jsonl
#   .venv/bin/python assn2.py rates                                   # list the sweep points
#   .venv/bin/python assn2.py solve chat 6.333 --n 10                 # -> schedules/chat_colocated_r6.333_n10_s1234.json
#   .venv/bin/python assn2.py solve chat 6.333 --topology 2p2d --window 16 --step 8 --limit 20
#   .venv/bin/python assn2.py compare schedules/<x>.json ../cs349e-assn2/results/<run>.jsonl [baseline.jsonl]
import argparse, json, os, random, re, sys, time
from math import ceil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSN2 = os.path.normpath(os.path.join(HERE, "..", "cs349e-assn2"))
CONFIG = "qwen3-8b_a100x4"
BLK = 64                       # block size for the solver's hash ids; no two assn2 prompts share a block
WORKERS = 4
FORMAT = "soldr-assn2-schedule/1"
# KV handoff (measured on the 16-request 2p2d runs, first send per P-D pair excluded: NCCL builds that pair's
# communicator lazily, ~300 ms once): transfer + admit = 84 ms + 0.0015 ms/token for all three prompt lengths, i.e. ~1.5
# decode steps of waiting for the decode worker's polls, plus the admit, plus bytes at ~98 GB/s
KV_GBPS = 98
HANDOFF_STEPS = 1.5
DP_SCALE = 0.5                 # assn2's MIXED engine: a prefill waits for the in-flight decode step (see interference_packing)

# constants.TRACES max_decode_slots: the harness admits at most this many requests per GPU (DecodeBatch.has_room).
# Every request in a trace has the same isl + osl, and the padded batch holds B x max length <= B x (isl + osl)
# tokens, so a token capacity of slots x (isl + osl) is exactly that slot cap. It is below the physical KV capacity
# (params kv_capacity, measured on a 40 GB A100) for all three traces.
MAX_SLOTS = {"chat": 96, "summarize": 48, "extract": 24}

# Copied from cs349e-assn2/sweep_cs349e.py (not imported: that module needs the assn2 environment)
RANGES = {"chat": (3, 9), "summarize": (2, 8), "extract": (2, 5)}
POINTS = 10


def sweep_rates(trace, points=POINTS):
    lo, hi = RANGES[trace]
    return [round(lo + (hi - lo) * i / (points - 1), 3) for i in range(points)]


def load_assn2_trace(name):
    path = os.path.join(ASSN2, "traces", f"{name}.jsonl")
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip() and not l.lstrip().startswith("//")]


def arrival_times_s(n, rate, seed):
    # Same as Controller._arrival_times
    rng, t, out = random.Random(seed), 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate)
        out.append(t)
    return out


def build(name, rate, n, seed):
    from cache_packing import Request
    rows = load_assn2_trace(name)[:n]
    arrivals = arrival_times_s(len(rows), rate, seed)
    trace, nxt = [], 0
    for r, t in zip(rows, arrivals):
        k = ceil(r["prompt_len"] / BLK)
        trace.append(Request(r["prompt_len"], r["output_len"], round(t * 1000), list(range(nxt, nxt + k))))
        nxt += k
    return rows, arrivals, trace


def convert(args):
    # Write each assn2 trace as a mooncake-style jsonl per sweep rate: the Poisson arrivals the harness's Controller will
    # draw (seed), in ms, plus the exact arrival_s and rid. Prompts are random per request, so no blocks are shared:
    # hash_ids are unique 512-token blocks, like from_mooncake's block size.
    out_dir = os.path.join(HERE, "traces", "assn2")
    os.makedirs(out_dir, exist_ok=True)
    for name in RANGES:
        rows = load_assn2_trace(name)[:args.n]
        for rate in sweep_rates(name):
            arrivals, nxt = arrival_times_s(len(rows), rate, args.seed), 0
            path = os.path.join(out_dir, f"{name}_r{rate:g}_n{len(rows)}_s{args.seed}.jsonl")
            with open(path, "w") as f:
                for j, (r, t) in enumerate(zip(rows, arrivals)):
                    k = ceil(r["prompt_len"] / 512)
                    f.write(json.dumps(dict(timestamp=round(t * 1000), input_length=r["prompt_len"],
                                            output_length=r["output_len"], hash_ids=list(range(nxt, nxt + k)),
                                            rid=j, arrival_s=t)) + "\n")
                    nxt += k
            print(f"wrote {path}")


def trace_steps(name, rate, n_decode, colocated, n=None, log=print):
    # Per-solve decode model from calibrate_traces.py's staggered-batch measurements (steps_<config>_traces_bench.json).
    # The measured step is a hinge in B: flat (launch-bound, ~52 ms) until the batch's KV copy dominates, then ~linear.
    # Each trace has one (isl, osl), so the context term is folded in (k = 0) and the step is
    #   max(FLAT_MS, t0 + delta (B - 1))
    # with FLAT_MS the median below the hinge and (t0, delta) a line through the points above it (see
    # interference_packing.FLAT_MS). The admit + evict stall and the step used to count overlapped steps are taken at
    # this solve's operating point: rate / n_decode arrivals per GPU, each held for (osl - 1) steps (Little's law,
    # solved as a fixed point; it saturates at the slot cap when the decode GPUs can't keep up).
    from interference_packing import StepLatency
    tb = json.load(open(f"steps_{CONFIG}_traces_bench.json"))["traces"][name]
    rows, osl, slots = tb["rows"], tb["osl"], tb["slots"]
    cap = min(slots, n / n_decode) if n else slots               # a short trace never fills more than n / n_decode
    Bm = np.array([r["B"] for r in rows], float)
    ym = np.array([r["step"] for r in rows])
    step = lambda B: np.interp(B, Bm, ym)
    event = lambda B: np.interp(B, Bm, [r["admit"] + r["evict"] for r in rows])
    flat = float(np.median(ym[ym <= ym.min() * 1.08]))
    above = ym > flat * 1.08
    delta, icpt = np.polyfit(Bm[above] - 1, ym[above], 1)
    model = lambda B: np.maximum(flat, icpt + delta * (B - 1))
    steps = StepLatency(CONFIG)
    steps.f1, steps.s, steps.k = float(icpt), float(delta), 0.0
    lam = rate / n_decode / 1e3                                   # arrivals per ms per decode GPU
    rho = lam * steps.prefill_ms(tb["isl"]) if colocated else 0   # share of time a colocated GPU spends prefilling
    B = 1.0
    for _ in range(200):
        tpot = (step(B) + lam * step(B) * event(B)) / max(1 - rho, 0.05)
        B = 0.5 * B + 0.5 * min(cap, max(1.0, lam * (osl - 1) * tpot))
    err = np.abs(model(Bm) - ym) / ym
    ev, typ = float(event(B)), float(step(B))
    admit = float(np.interp(B, Bm, [r["admit"] for r in rows]))
    fit = dict(trace=name, op_B=float(B), flat_ms=flat, lin_t0=float(icpt), lin_delta=float(delta), event_ms=ev,
               step_at_op_ms=typ, admit_ms=admit, err_max=float(err.max()), err_median=float(np.median(err)), rho=float(rho),
               saturated=bool(B >= slots - 0.5), op_cap=float(cap))
    log(f"  decode model: step max({flat:.1f}, {icpt:.1f} + {delta:.3f} (B - 1)) ms, error median {np.median(err):.1%} "
        f"max {err.max():.1%}; operating batch {B:.1f}/GPU{' (SATURATED at the slot cap)' if fit['saturated'] else ''}, "
        f"step there {typ:.1f} ms, admit+evict stall {ev:.1f} ms")
    return steps, flat, typ, ev, fit


def topology_ranks(topology):
    if topology == "colocated":
        return "colocated", None, list(range(WORKERS)), list(range(WORKERS))
    m = re.fullmatch(r"(\d+)p(\d+)d", topology)
    assert m and int(m[1]) + int(m[2]) == WORKERS and int(m[1]) > 0 and int(m[2]) > 0, topology
    P = int(m[1])
    pools = (list(range(P)), list(range(P, WORKERS)))
    return "split", pools, *pools


def solve(args):
    from cache_packing import default_latency as p
    import interference_packing
    from interference_packing import solve_rolling
    interference_packing.DP_SCALE = DP_SCALE
    interference_packing.PD_FUSED = False          # a MIXED prefill pauses every decode on its GPU
    if not args.any_rate and args.rate not in sweep_rates(args.trace):
        sys.exit(f"{args.rate} is not a sweep point for {args.trace}: {sweep_rates(args.trace)} (--any-rate to allow)")
    rows, arrivals, trace = build(args.trace, args.rate, args.n, args.seed)
    N = len(trace)
    mem = min(p.kv_capacity, MAX_SLOTS[args.trace] * max(u.isl + u.osl for u in trace))
    solver_topology, pools, prefill_ranks, decode_ranks = topology_ranks(args.topology)
    window = args.window or N
    steps, flat_ms, typ_ms, event_ms, fit = trace_steps(args.trace, args.rate, len(decode_ranks),
                                                        solver_topology == "colocated", n=N)
    interference_packing.FLAT_MS, interference_packing.DD_STEP_MS = flat_ms, typ_ms
    # Colocated: the stall is charged per overlapping prefill (schedule-aware). Split: prefills are elsewhere, so the
    # decode GPU's expected stall per step (arrivals per step x stall) is folded into t0
    if solver_topology == "colocated":
        interference_packing.EVENT_MS = event_ms
    else:
        interference_packing.EVENT_MS = 0.0
        stall = args.rate / len(decode_ranks) / 1e3 * typ_ms * event_ms     # expected stall per decode step
        steps.f1 += stall
        interference_packing.FLAT_MS += stall
        interference_packing.KV_BANDWIDTH_GBPS = KV_GBPS
        interference_packing.TRANSFER_FIXED_MS = HANDOFF_STEPS * typ_ms + fit["admit_ms"]
    step = args.step or window
    print(f"{args.trace} r{args.rate:g} seed {args.seed}: {N} requests, topology {args.topology}; "
          f"window {window}, step {step}, {args.limit:g} s per window", flush=True)
    policy = None
    if args.policy == "rr":
        # Controller._dispatch_loop: prefill rank rid % P, decode rank rid % D, dispatched on arrival, FCFS per worker
        P, D = len(prefill_ranks), len(decode_ranks)
        policy = lambda g: (prefill_ranks[g % P], decode_ranks[g % D], g - P if g >= P else None)
        interference_packing.DECODE_MAY_WAIT = True
    t = time.time()
    vals = solve_rolling(trace, BLK, WORKERS, mem, window, step, args.limit,
                         step_latency=steps, topology=solver_topology, pools=pools, policy=policy, log=lambda s: print(" ", s, flush=True))
    solve_s = time.time() - t

    reqs = []
    for j, (r, u) in enumerate(zip(rows, trace)):
        x = vals[j]
        droute = x.get("droute", x["route"])
        ttft = x["prefill_end"] - x["release"]
        tpot = (x["end"] - x["prefill_end"]) / (u.osl - 1) if u.osl > 1 else None
        reqs.append(dict(rid=j, prompt_len=r["prompt_len"], output_len=r["output_len"],
                         arrival_s=arrivals[j], release_ms=x["release"], start_ms=x["start"],
                         prefill_rank=x["route"], decode_rank=droute,
                         pred=dict(prefill_end_ms=x["prefill_end"], end_ms=x["end"],
                                   decode_start_ms=x.get("decode_start", x["prefill_end"]),
                                   ttft_ms=ttft, tpot_ms=tpot, e2e_ms=x["end"] - x["release"])))
        assert x["release"] == u.arrival and x["start"] >= x["release"]
        assert x["route"] in prefill_ranks and droute in decode_ranks
    out = dict(format=FORMAT, trace=args.trace, rate=args.rate, seed=args.seed, n_requests=N,
               topology=args.topology, prefill_ranks=prefill_ranks, decode_ranks=decode_ranks,
               solver=dict(policy=args.policy, mem_tokens=mem, window=window, step=step, limit_s=args.limit, steps_config=CONFIG, solve_s=solve_s, dp_scale=DP_SCALE,
                           steps=dict(f1=steps.f1, s=steps.s, k=steps.k), event_ms=event_ms, pd_fused=False, flat_ms=interference_packing.FLAT_MS,
                           dd_step_ms=typ_ms, transfer_fixed_ms=interference_packing.TRANSFER_FIXED_MS,
                           kv_gbps=interference_packing.KV_BANDWIDTH_GBPS,
                           decode_fit=fit),
               requests=reqs)
    path = args.out or os.path.join(HERE, "schedules",
                                    f"{args.trace}_{args.topology}_r{args.rate:g}_n{N}_s{args.seed}"
                                    + ("" if args.policy == "opt" else f"_{args.policy}") + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    summarize_pred(out)
    print(f"solved in {solve_s:.0f} s; wrote {path}")
    return path


def slos(trace):
    # constants.TRACES (TTFT_MULT 3, TPOT_MULT 2), in ms
    return {"chat": (203.4, 97.2), "summarize": (423.9, 96.8), "extract": (1086.5, 97.3)}[trace]


def summarize_pred(s):
    ttft = np.array([r["pred"]["ttft_ms"] for r in s["requests"]])
    tpot = np.array([r["pred"]["tpot_ms"] for r in s["requests"]], float)
    ts, ps = slos(s["trace"])
    ok = np.mean((ttft <= ts) & (tpot <= ps)) * 100
    per = np.bincount([r["prefill_rank"] for r in s["requests"]], minlength=WORKERS)
    print(f"  predicted: TTFT p50/p90 {np.percentile(ttft, 50):.0f}/{np.percentile(ttft, 90):.0f} ms, "
          f"TPOT p50/p90 {np.percentile(tpot, 50):.1f}/{np.percentile(tpot, 90):.1f} ms, SLO attainment {ok:.1f}%; "
          f"prefills per rank {per.tolist()}")


def load_results(path):
    with open(path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    header = next(l for l in lines if l.get("_type") == "header")
    recs = {r["rid"]: r for r in lines if r.get("_type") != "header" and "rid" in r}
    return header, recs


def compare(args):
    s = json.load(open(args.schedule))
    ts, ps = slos(s["trace"])

    def stats(label, ttft, tpot):
        ttft, tpot = np.array(ttft, float) * 1e3, np.array(tpot, float) * 1e3
        ok = np.mean((ttft <= ts) & (tpot <= ps)) * 100
        print(f"  {label:26s} TTFT p50/p90 {np.percentile(ttft, 50):6.0f}/{np.percentile(ttft, 90):6.0f} ms   "
              f"TPOT p50/p90 {np.percentile(tpot, 50):5.1f}/{np.percentile(tpot, 90):5.1f} ms   SLO {ok:5.1f}%")

    print(f"{s['trace']} r{s['rate']:g} {s['topology']} n{s['n_requests']}  (SLO TTFT {ts} ms, TPOT {ps} ms)")
    reqs = s["requests"]
    stats("solver prediction", [r["pred"]["ttft_ms"] / 1e3 for r in reqs], [r["pred"]["tpot_ms"] / 1e3 for r in reqs])
    header, recs = load_results(args.results)
    missing = [r["rid"] for r in reqs if r["rid"] not in recs]
    if missing:
        print(f"  WARNING: {len(missing)} scheduled requests missing from results: {missing[:10]}")
    got = [r for r in reqs if r["rid"] in recs]
    stats("harness, solver schedule", [recs[r["rid"]]["ttft"] for r in got], [recs[r["rid"]]["tpot"] for r in got])
    if args.baseline:
        bh, brecs = load_results(args.baseline)
        b = list(brecs.values())
        stats("harness, round robin", [r["ttft"] for r in b], [r["tpot"] for r in b])
    # Per request: predicted - measured
    for m in ("ttft", "tpot"):
        e = np.array([r["pred"][f"{m}_ms"] - recs[r["rid"]][m] * 1e3 for r in got if r["pred"][f"{m}_ms"] is not None])
        rel = e / np.array([recs[r["rid"]][m] * 1e3 for r in got if r["pred"][f"{m}_ms"] is not None])
        print(f"  {m.upper()} predicted - measured: median {np.median(e):+.1f} ms ({np.median(rel):+.1%}), "
              f"P10 {np.percentile(e, 10):+.1f}, P90 {np.percentile(e, 90):+.1f} ms")
    # Did the harness follow the schedule? (placement, and prefill start vs scheduled start)
    t0 = min(recs[r["rid"]]["arrival_t"] - r["arrival_s"] for r in got)
    wrong = sum(recs[r["rid"]].get("prefill_rank", r["prefill_rank"]) != r["prefill_rank"] or
                recs[r["rid"]].get("decode_rank", r["decode_rank"]) != r["decode_rank"] for r in got)
    lag = [(recs[r["rid"]]["prefill_start_t"] - t0) * 1e3 - r["start_ms"] for r in got
           if recs[r["rid"]].get("prefill_start_t")]
    print(f"  placement mismatches {wrong}/{len(got)}; prefill start - scheduled start: median {np.median(lag):+.1f} ms, "
          f"P90 {np.percentile(lag, 90):+.1f} ms")
    # Each GPU's first prefill pays one-time kernel setup (the harness warms up with an 8-token prompt)
    first = {}
    for r in sorted(got, key=lambda r: recs[r["rid"]]["prefill_start_t"]):
        first.setdefault(r["prefill_rank"], r["rid"])
    warm = [r for r in got if r["rid"] not in first.values()]
    e = np.array([(r["pred"]["ttft_ms"] - recs[r["rid"]]["ttft"] * 1e3) / (recs[r["rid"]]["ttft"] * 1e3) for r in warm])
    print(f"  TTFT predicted - measured, excluding each GPU's first prefill ({len(warm)} requests): median "
          f"{np.median(e):+.1%}, mean |error| {np.mean(np.abs(e)):.1%}")
    for m in ("ttft", "tpot"):
        e = np.array([(r["pred"][f"{m}_ms"] - recs[r["rid"]][m] * 1e3) / (recs[r["rid"]][m] * 1e3) for r in got])
        print(f"  {m.upper()} mean |error| {np.mean(np.abs(e)):.1%}")
    xs = [recs[r["rid"]]["transfer_s"] * 1e3 for r in got if recs[r["rid"]].get("transfer_s")]
    if xs:
        print(f"  KV transfer (measured): median {np.median(xs):.1f} ms, P90 {np.percentile(xs, 90):.1f} ms "
              f"for {s['requests'][0]['prompt_len']} tokens")
    # End-to-end: last token of the batch, from the first arrival
    span = lambda R: max(R[r]["done_t"] for r in R) - min(R[r]["arrival_t"] for r in R)
    pred_span = (max(r["pred"]["end_ms"] for r in reqs) - min(r["release_ms"] for r in reqs)) / 1e3
    print(f"  makespan: predicted {pred_span:.2f} s, harness {span(recs):.2f} s"
          + (f", round robin {span(brecs):.2f} s" if args.baseline else ""))


def rank(args):
    # Which topology is best for this trace slice: the solver's predictions vs. the harness (its schedule, and round
    # robin), ranked by SLO attainment, then mean end-to-end latency (arrival to last token)
    ts, ps = slos(args.trace)
    res = os.path.join(ASSN2, "results")
    rows = {}
    for topo in ("colocated", "1p3d", "2p2d", "3p1d"):
        name = f"{args.trace}_{topo}_r{args.rate:g}_n{args.n}_s{args.seed}"
        sp = os.path.join(HERE, "schedules", name + ".json")
        if not os.path.exists(sp):
            continue
        s = json.load(open(sp))
        pred = [(r["pred"]["ttft_ms"], r["pred"]["tpot_ms"], r["pred"]["e2e_ms"]) for r in s["requests"]]
        out = {"predicted": pred}
        if os.path.exists(sp.replace(".json", "_rr.json")):
            r_ = json.load(open(sp.replace(".json", "_rr.json")))
            out["pred_rr"] = [(r["pred"]["ttft_ms"], r["pred"]["tpot_ms"], r["pred"]["e2e_ms"]) for r in r_["requests"]]
        for arm in ("sched", "rr"):
            p = os.path.join(res, f"{arm}_{name}.jsonl")
            if os.path.exists(p):
                _, recs = load_results(p)
                out[arm] = [(r["ttft"] * 1e3, r["tpot"] * 1e3, (r["done_t"] - r["arrival_t"]) * 1e3) for r in recs.values()]
        rows[topo] = out

    def stats(v):
        v = np.array(v, float)
        ok = np.mean((v[:, 0] <= ts) & (v[:, 1] <= ps)) * 100
        return ok, np.percentile(v[:, 0], 90), np.percentile(v[:, 1], 90), v[:, 2].mean()

    print(f"{args.trace} r{args.rate:g} n{args.n}  (SLO TTFT {ts} ms, TPOT {ps} ms)")
    print(f"  {'topology':10s} {'':10s} {'SLO %':>6s} {'TTFT p90':>9s} {'TPOT p90':>9s} {'mean e2e':>9s}")
    best = {}
    for topo, out in rows.items():
        for arm in ("predicted", "sched", "pred_rr", "rr"):
            if arm in out:
                ok, t90, p90, e2e = stats(out[arm])
                best.setdefault(arm, []).append(((-ok, e2e), topo))
                print(f"  {topo if arm == 'predicted' else '':10s} {arm:10s} {ok:6.1f} {t90:9.0f} {p90:9.1f} {e2e:9.0f}")
    for arm, v in best.items():
        print(f"  ranking by {arm:9s}: " + " > ".join(t for _, t in sorted(v)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rates")
    a = sub.add_parser("solve")
    a.add_argument("trace", choices=RANGES)
    a.add_argument("rate", type=float)
    a.add_argument("--n", type=int, default=300, help="first n requests (the sweep uses 300)")
    a.add_argument("--seed", type=int, default=1234)
    a.add_argument("--topology", default="colocated", help="colocated or <P>p<D>d, e.g. 2p2d")
    a.add_argument("--window", type=int, default=None, help="requests per solver window (default: all)")
    a.add_argument("--step", type=int, default=None, help="requests committed per window (default: window)")
    a.add_argument("--limit", type=float, default=30, help="CP-SAT seconds per window")
    a.add_argument("--out", default=None)
    a.add_argument("--any-rate", action="store_true")
    a.add_argument("--policy", choices=("opt", "rr"), default="opt",
                   help="opt: optimize routing and starts; rr: predict the harness's round robin (writes <name>_rr.json)")
    v = sub.add_parser("convert")
    v.add_argument("--n", type=int, default=300)
    v.add_argument("--seed", type=int, default=1234)
    k = sub.add_parser("rank")
    k.add_argument("trace", choices=RANGES); k.add_argument("rate", type=float)
    k.add_argument("--n", type=int, default=300); k.add_argument("--seed", type=int, default=1234)
    c = sub.add_parser("compare")
    c.add_argument("schedule"); c.add_argument("results"); c.add_argument("baseline", nargs="?")
    args = ap.parse_args()
    for k in ("schedule", "results", "baseline", "out"):
        if getattr(args, k, None):
            setattr(args, k, os.path.abspath(getattr(args, k)))
    os.chdir(HERE)          # params_/steps_ files are loaded by relative path
    sys.path.insert(0, HERE)
    if args.cmd == "rates":
        for t in RANGES:
            print(t, sweep_rates(t))
    elif args.cmd == "rank":
        rank(args)
    elif args.cmd == "convert":
        convert(args)
    elif args.cmd == "solve":
        solve(args)
    else:
        compare(args)
