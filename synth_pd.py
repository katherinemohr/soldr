# Generate a Synthetic trace with heavy P-D interference: single-turn requests, Poisson arrivals, large prompts with no prefix
# sharing (every prefill is several 8192-token chunks), and outputs long enough that decodes are running when the next
# prefills land. Compares Mocker's colocated topology (C aggregated workers, KV router) against P-D disaggregated
# splits of the same GPUs (Mocker's own disagg routing), both open loop at the trace's arrival times.
#
# Usage, from the repo root:
#   python synth_pd.py n rate isl_lo isl_hi osl_lo osl_hi [seed] [workers]
#   e.g. python synth_pd.py 300 1.5 16384 65536 256 1024
import json, os, sys, tempfile
import numpy as np
from math import ceil
from cache_packing import Request, default_latency as p
from calibrate import mocker_engine
from interference_packing import KV_BYTES_PER_TOKEN

BLK = p.kv_block
# The synthetic P-D trace of experiments 1-3 (synthetic_trace's arguments)
TRACE = dict(n=300, rate=1.25, isl=(32768, 98304), osl=(256, 1024), seed=0)


# n requests, Poisson arrivals at `rate` per second; isl and osl uniform in [lo, hi]. Times in ms. Hash ids are unique
# per block, so no request gets a cache hit.
def synthetic_trace(n, rate, isl, osl, seed=0):
    rng = np.random.default_rng(seed)
    arrivals = np.cumsum(rng.exponential(1000 / rate, n))
    arrivals -= arrivals[0]
    trace, next_hash = [], 0
    for t in arrivals:
        i, o = int(rng.integers(isl[0], isl[1] + 1)), int(rng.integers(osl[0], osl[1] + 1))
        blocks = ceil(i / BLK)
        trace.append(Request(i, o, int(round(t)), list(range(next_hash, next_hash + blocks))))
        next_hash += blocks
    return trace


def rows(trace):
    return [dict(timestamp=float(u.arrival), input_length=u.isl, output_length=u.osl, hash_ids=u.hash_ids,
                 request_id=str(j), play_id=str(j), session_id=str(j), model=p.aic["model"],
                 not_before_ms=float(u.arrival), dependencies=[]) for j, u in enumerate(trace)]


# Mocker engine args for one side of a P-D disaggregated deployment (copied from pd_packing): calibrate.mocker_engine
# for a prefill or decode worker, with the KV transfer at bw_gbps and the model's KV bytes per token
def mocker_pd_engine(p, worker_type, bw_gbps):
    e = mocker_engine(p)
    e["engine"]["worker_type"] = worker_type
    e["engine"]["timing_model"]["config"] = e["engine"]["timing_model"]["config"] | {"worker_type": worker_type}
    e["engine"]["kv_transfer_bandwidth"] = bw_gbps
    e["engine"]["kv_transfer_bytes_per_token"] = KV_BYTES_PER_TOKEN
    return e


# Replays a trace (times in ms) on num_prefill prefill + num_decode decode workers, in Mocker's agentic format (the
# one that carries request IDs, which pins need). With submit (ms per request) requests arrive then, open loop;
# otherwise later turns follow their predecessor, closed loop. pins maps each request to its (prefill, decode) worker,
# each numbered within its pool; it needs Dynamo built with patches/dynamo-replay-pins.patch. Returns Mocker's
# per-request records in trace order. (From pd_packing.run_mocker_pd, which takes traces in 0.1 ms units.)
def run_mocker_pd(trace, blk_size, num_prefill, num_decode, p, bw_gbps, submit=None, pins=None):
    os.environ.setdefault("DYN_LOG", "warn")
    from dynamo.replay import run_trace_replay

    session = []
    for j, u in enumerate(trace):
        session.append(j if u.prev is None else session[u.prev])
    rows = []
    for j, u in enumerate(trace):
        t = (u.arrival or 0) if submit is None else submit[j]
        deps = [] if u.prev is None or submit is not None else [
            dict(request_id=str(u.prev), trigger="completion", delay_ms=u.think, relation="sequence")]
        rows.append(dict(timestamp=t, input_length=u.isl, output_length=u.osl, hash_ids=u.hash_ids, request_id=str(j),
                         play_id=str(session[j]), session_id=str(session[j]), model=p.aic["model"],
                         not_before_ms=float(t), dependencies=deps))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "trace.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({"schema": "dynamo.agentic_mooncake", "version": 2, "block_size": blk_size,
                                "hash_id_scope": "local", "source": {"format": "soldr", "digest": "-"}}) + "\n")
            f.writelines(json.dumps(r) + "\n" for r in rows)
        if pins is not None:
            pin_path = os.path.join(tmp, "pins.json")
            with open(pin_path, "w") as f:
                json.dump(dict(prefill={str(j): pd[0] for j, pd in enumerate(pins)},
                               decode={str(j): pd[1] for j, pd in enumerate(pins)}), f)
            os.environ["DYN_REPLAY_PINS"] = pin_path
        try:
            report = run_trace_replay(path, prefill_engine_args=mocker_pd_engine(p, "prefill", bw_gbps),
                                      decode_engine_args=mocker_pd_engine(p, "decode", bw_gbps),
                                      num_prefill_workers=num_prefill, num_decode_workers=num_decode,
                                      router_mode="kv_router", trace_block_size=blk_size, capture_per_request=True,
                                      trace_format="agentic_mooncake", execution_model=p.aic["model"])
        finally:
            os.environ.pop("DYN_REPLAY_PINS", None)
    recs = {q["request_id"]: q for q in report.to_dict()["per_request"]}
    recs = [recs[r["request_id"]] for r in rows]
    if pins is not None:
        assert all((q["prefill_worker_idx"], q["decode_worker_idx"]) == tuple(pd) for q, pd in zip(recs, pins)), \
            "Mocker ignored the pins; is Dynamo built with patches/dynamo-replay-pins.patch?"
    return recs


# Replays the trace open loop in Mocker: colocated on `workers` aggregated workers (num_prefill None), or disaggregated
# on num_prefill prefill + num_decode decode workers. Returns per-request records in trace order.
def run(trace, workers=None, num_prefill=None, num_decode=None, bw_gbps=450):
    os.environ.setdefault("DYN_LOG", "warn")
    from dynamo.replay import run_trace_replay
    rs = rows(trace)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "trace.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({"schema": "dynamo.agentic_mooncake", "version": 2, "block_size": BLK,
                                "hash_id_scope": "local", "source": {"format": "soldr", "digest": "-"}}) + "\n")
            f.writelines(json.dumps(r) + "\n" for r in rs)
        common = dict(router_mode="kv_router", trace_block_size=BLK, capture_per_request=True,
                      trace_format="agentic_mooncake", execution_model=p.aic["model"])
        if num_prefill is None:
            report = run_trace_replay(path, extra_engine_args=mocker_engine(p), num_workers=workers, **common)
        else:
            report = run_trace_replay(path, prefill_engine_args=mocker_pd_engine(p, "prefill", bw_gbps),
                                      decode_engine_args=mocker_pd_engine(p, "decode", bw_gbps),
                                      num_prefill_workers=num_prefill, num_decode_workers=num_decode, **common)
    recs = {q["request_id"]: q for q in report.to_dict()["per_request"]}
    return [recs[r["request_id"]] for r in rs]


def metrics(recs):
    tpot = [(q["last_token_ms"] - q["first_token_ms"]) / (q["output_length"] - 1) for q in recs if q["output_length"] > 1]
    out = {}
    for k, xs in (("ttft", [q["ttft_ms"] for q in recs]), ("tpot", tpot), ("e2e", [q["e2e_latency_ms"] for q in recs])):
        out[k] = (float(np.mean(xs)), float(np.percentile(xs, 90)))
    return out


# Mocker's per-request TTFT, TPOT (over the osl - 1 tokens after the first; nan for osl 1) and e2e, in ms, as arrays
def measured(recs):
    return dict(ttft=np.array([q["ttft_ms"] for q in recs]), e2e=np.array([q["e2e_latency_ms"] for q in recs]),
                tpot=np.array([(q["last_token_ms"] - q["first_token_ms"]) / (q["output_length"] - 1)
                               if q["output_length"] > 1 else np.nan for q in recs]))


if __name__ == "__main__":
    n, rate = int(sys.argv[1]), float(sys.argv[2])
    isl, osl = (int(sys.argv[3]), int(sys.argv[4])), (int(sys.argv[5]), int(sys.argv[6]))
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    W = int(sys.argv[8]) if len(sys.argv) > 8 else 4
    trace = synthetic_trace(n, rate, isl, osl, seed)
    print(f"{n} requests at {rate:g}/s over {trace[-1].arrival / 1000:.0f} s, isl {isl}, osl {osl}, seed {seed}; "
          f"{W} GPUs (workers)", flush=True)
    print(f"{'topology':10s} {'TTFT mean':>9s} {'P90':>7s} | {'TPOT mean':>9s} {'P90':>6s} | {'e2e mean':>9s} {'P90':>8s}")
    for name, kw in [(f"{W} coloc", dict(workers=W))] + [(f"{k}P{W - k}D", dict(num_prefill=k, num_decode=W - k))
                                                        for k in range(1, W)]:
        m = metrics(run(trace, **kw))
        print(f"{name:10s} {m['ttft'][0]:9.1f} {m['ttft'][1]:7.1f} | {m['tpot'][0]:9.2f} {m['tpot'][1]:6.2f} | "
              f"{m['e2e'][0]:9.1f} {m['e2e'][1]:8.1f}", flush=True)
