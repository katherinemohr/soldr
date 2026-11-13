# Step 0: does the step-level model predict Mocker's per-request TTFT and TPOT, on Soldr's schedules and on Mocker's
# own routing? Old = Soldr's current model (fixed decode step, full-stop stalls) on the same decisions and cache hits.
import json, os, sys
from math import ceil
from calibrate import run_mocker, soldr_latencies, summarize
from cache_packing import from_weka, default_latency as p
from tpot_sim import StepModel, predict

HERE = os.path.dirname(os.path.abspath(__file__))
dt = p.decode_ms
m = StepModel()


def base_rows(trace):
    return [{"timestamp": (u.arrival or 0) * dt, "input_length": u.isl, "output_length": u.osl, "hash_ids": u.hash_ids}
            for u in trace]


def agentic_rows(trace):
    rows, session = base_rows(trace), []
    for j, u in enumerate(trace):
        session.append(j if u.prev is None else session[u.prev])
    deps = lambda u: [] if u.prev is None else [dict(request_id=str(u.prev), trigger="completion",
                                                     delay_ms=u.think * dt, relation="sequence")]
    return [r | dict(request_id=str(j), play_id=str(session[j]), session_id=str(session[j]), model=p.aic["model"],
                     not_before_ms=float(r["timestamp"]), dependencies=deps(u))
            for j, (r, u) in enumerate(zip(rows, trace))]


def scored(rows, recs):
    """Per request (measured, new, old) TTFT and TPOT in ms for one Mocker run."""
    new_ttft, new_tpot = predict(rows, recs, m)
    stall = predict.stall
    # soldr_latencies returns requests in Mocker's arrival order (a stable sort)
    order = sorted(range(len(recs)), key=lambda j: recs[j]["arrival_time_ms"])
    old_ttft, old_tpot = [None] * len(recs), [None] * len(recs)
    for j, a, b in zip(order, *soldr_latencies(rows, recs, p)):
        old_ttft[j], old_tpot[j] = a, b
    meas_ttft = [q["ttft_ms"] for q in recs]
    meas_tpot = [(q["last_token_ms"] - q["first_token_ms"]) / (q["output_length"] - 1) if q["output_length"] > 1
                 else None for q in recs]
    readmit = sum(q["readmission_count"] for q in recs)
    return dict(ttft=(meas_ttft, new_ttft, [x * dt for x in old_ttft]),
                tpot=(meas_tpot, new_tpot, [x * dt if x is not None else None for x in old_tpot]), readmit=readmit,
                stall=stall)


def soldr_schedule(trace, blk, route, start):
    """Soldr's decisions replayed as compare() does: one Mocker run per cache, submitted at Soldr's start times."""
    rows = base_rows(trace)
    out = dict(ttft=([], [], []), tpot=([], [], []), readmit=0, stall=[])
    for c in sorted(set(route)):
        ix = sorted((j for j in range(len(rows)) if route[j] == c), key=lambda j: (start[j], j))
        sub = [rows[j] | {"timestamp": start[j] * dt} for j in ix]
        s = scored(sub, run_mocker(sub, 1, blk, p))
        for k in ("ttft", "tpot"):
            for col, xs in zip(out[k], s[k]):
                col.extend(xs)
        out["readmit"] += s["readmit"]
        out["stall"].extend(s["stall"])
    return out


def report(name, s):
    print(f"\n{name}  (readmissions: {s['readmit']})")
    print(f"{'':6s} {'':5s} {'measured':>9s} {'new':>8s} {'old':>8s} | per-request error (pred - meas) ms: "
          f"{'new med':>8s} {'P10':>7s} {'P90':>7s} {'MAE':>7s} | {'old med':>8s} {'P10':>7s} {'P90':>7s} {'MAE':>7s}")
    for k in ("ttft", "tpot"):
        meas, new, old = s[k]
        keep = [i for i, x in enumerate(meas) if x is not None]
        cols = [[c[i] for i in keep] for c in (meas, new, old)]
        errs = []
        for c in cols[1:]:
            e = sorted(a - b for a, b in zip(c, cols[0]))
            errs.append((e[len(e) // 2], e[len(e) // 10], e[ceil(0.9 * len(e)) - 1],
                         sum(abs(x) for x in e) / len(e)))
        for stat in ("mean", 0.9):
            label = "mean" if stat == "mean" else "P90"
            vals = [summarize(c, stat) for c in cols]
            tail = (" | " + " ".join(f"{x:8.2f}" if i % 4 == 0 else f"{x:7.2f}" for i, x in
                                     enumerate(errs[0] + errs[1]))) if stat == "mean" else ""
            if k == "tpot":
                st = [x for x in s["stall"] if x is not None]
                tail += f"   [new: prefill stall {summarize(st, stat):.2f} ms/token, rest is batch + KV]"
            print(f"{k.upper():6s} {label:5s} " + " ".join(f"{v:9.2f}" if i == 0 else f"{v:8.2f}"
                                                         for i, v in enumerate(vals)) + tail)


if __name__ == "__main__":
    which = sys.argv[1:] or ["512"]
    if "512" in which:
        trace, blk = from_weka("traces/weka_256k_062126.jsonl", sessions=64, n=512, stagger_s=900 // 64)
        rows = agentic_rows(trace)
        report("512 requests, 4 caches, Mocker routing", scored(rows, run_mocker(rows, 4, blk, p)))
        sched = json.load(open("schedule_4c_64s_512r_14st_qwen3.json"))["requests"]
        assert [r["isl"] for r in sched] == [u.isl for u in trace]
        report("512 requests, 4 caches, Soldr P90-optimal schedule (no interference)",
               soldr_schedule(trace, blk, [r["cache"] for r in sched], [r["start"] for r in sched]))
    if "64" in which:
        saved = json.load(open(os.path.join(HERE, "sched_64.json")))
        for key, cfg in saved.items():
            trace, blk = from_weka("traces/weka_256k_062126.jsonl", sessions=cfg["sessions"], n=cfg["n"],
                                   stagger_s=900 // cfg["sessions"])
            trace = trace[:cfg["first"]]
            rows = agentic_rows(trace)
            report(f"({key}) 64 requests, {cfg['C']} caches, Mocker routing", scored(rows, run_mocker(rows, cfg["C"], blk, p)))
            for sch in ("A", "B"):
                d = cfg[sch]
                route = [d[str(j)][0] for j in range(len(trace))]
                start = [d[str(j)][1] for j in range(len(trace))]
                report(f"({key}) 64 requests, {cfg['C']} caches, Soldr schedule {sch}", soldr_schedule(trace, blk, route, start))
