import json
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from math import ceil

import numpy as np

MOCKER_ENGINE = {
    "engine": {
        "backend": "vllm", "block_size": 64, "num_gpu_blocks": 16384, "max_num_seqs": 256,
        "max_num_batched_tokens": 8192, "enable_prefix_caching": True, "enable_chunked_prefill": True,
        "max_model_len": 262144,
        "timing_model": {"type": "external", "provider": "aic", "config": {
            "model": "Qwen/Qwen3-30B-A3B", "system": "h100_sxm", "tp": 4, "moe_tp_size": 4, "moe_ep_size": 1,
            "backend": "vllm",
            "backend_version": "0.24.0", "worker_type": "aggregated", "gemm_quant_mode": "bfloat16",
            "fmha_quant_mode": "bfloat16", "kvcache_quant_mode": "bfloat16", "kv_block_size": 64,
            "estimation_mode": "op_level"}},
    },
    "num_gpu_blocks_is_explicit": True,
}
MOCKER_MODEL = MOCKER_ENGINE["engine"]["timing_model"]["config"]["model"]


@dataclass(frozen=True)
class MockerLatency:
    decode_ms: int # one timestep
    a: float
    b: float
    c: float
    chunk: int
    kv_block: int

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))

    def prefill_ms(self, isl, hit=0):
        hit = min(hit, isl - 1) // self.kv_block * self.kv_block
        return self.a * (isl - hit) + self.b * (isl ** 2 - hit ** 2) / 2 + self.c * ceil((isl - hit) / self.chunk)

    def __call__(self, isl, hit=0):
        return max(1, round(self.prefill_ms(isl, hit) / self.decode_ms))


def run_mocker(rows, workers, block_size):
    os.environ.setdefault("DYN_LOG", "warn")
    from dynamo.replay import run_trace_replay

    # Rows with dependencies are sessions, replayed closed-loop in Mocker's agentic format
    agentic = "dependencies" in rows[0]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "trace.jsonl")
        with open(path, "w") as f:
            if agentic:
                f.write(json.dumps({"schema": "dynamo.agentic_mooncake", "version": 2, "block_size": block_size,
                                    "hash_id_scope": "local", "source": {"format": "soldr", "digest": "-"}}) + "\n")
            f.writelines(json.dumps(r) + "\n" for r in rows)
        extra = dict(trace_format="agentic_mooncake", execution_model=MOCKER_MODEL,
                     agentic_lanes=len({r["play_id"] for r in rows})) if agentic else {}
        report = run_trace_replay(path, extra_engine_args=MOCKER_ENGINE, num_workers=workers, router_mode="kv_router",
                                  trace_block_size=block_size, capture_per_request=True, **extra)
    if agentic:
        recs = {q["request_id"]: q for q in report.to_dict()["per_request"]}
        return [recs[r["request_id"]] for r in rows]

    key = lambda arrival, isl, osl: (round(arrival), isl, osl)
    recs = defaultdict(list)
    for q in report.to_dict()["per_request"]:
        recs[key(q["arrival_time_ms"], q["input_length"], q["output_length"])].append(q)
    return [recs[key(r["timestamp"] - rows[0]["timestamp"], r["input_length"], r["output_length"])].pop() for r in rows]


# Mocker's decisions in Soldr's cost model
def soldr_ttft(rows, recs, prefill_latency):
    free, total = {}, 0
    # In the order Mocker sent the requests (in a closed-loop replay, this depends on Mocker's own timing)
    for r, q in sorted(zip(rows, recs), key=lambda rq: rq[1]["arrival_time_ms"]):
        arrival = int(q["arrival_time_ms"]) // prefill_latency.decode_ms
        c = q["decode_worker_idx"]
        start = max(arrival, free.get(c, arrival))
        free[c] = start + prefill_latency(r["input_length"], q["reused_input_tokens"])
        total += free[c] - arrival
    return total / len(rows)


def compare(trace, blk_size, solver, vars, prefill_latency, mocker_runs=3):
    dt = prefill_latency.decode_ms
    rows = [{"timestamp": u.arrival * dt, "input_length": u.isl, "output_length": u.osl, "hash_ids": u.hash_ids}
            for u in trace]
    route = [next(c for c, r in enumerate(rs) if solver.Value(r)) for rs in vars["routes"]]
    submit = [solver.Value(s) * dt for s in vars["starts"]]
    release = [solver.Value(r) * dt for r in vars["releases"]]
    soldr_mocker, mocker_end = [0.0] * len(rows), [0.0] * len(rows)
    for c in set(route):
        ix = sorted((j for j in range(len(rows)) if route[j] == c), key=lambda j: (submit[j], j))
        for j, q in zip(ix, run_mocker([rows[j] | {"timestamp": submit[j]} for j in ix], 1, blk_size)):
            soldr_mocker[j] = submit[j] - release[j] + q["ttft_ms"]
            mocker_end[j] = submit[j] + q["e2e_latency_ms"]
    # Where Mocker's timing departs from Soldr's: how much later each request ends, and the turns sent before
    # Mocker had finished their predecessor plus think time
    late = [mocker_end[j] - solver.Value(e) * dt for j, e in enumerate(vars["ends"])]
    early = [mocker_end[u.prev] + u.think * dt - submit[j] for j, u in enumerate(trace) if u.prev is not None]
    early = [x for x in early if x > 0]
    print(f"Mocker end - Soldr end (ms), min/quartiles/max: {' / '.join(f'{x:.0f}' for x in np.percentile(late, [0, 25, 50, 75, 100]))}; "
          f"{len(early)} turns sent before Mocker would release them, by {sum(early):.0f} ms in total")

    if any(u.prev is not None for u in trace):
        play = []
        for j, u in enumerate(trace):
            play.append(j if u.prev is None else play[u.prev])
        deps = lambda u: [] if u.prev is None else [dict(request_id=str(u.prev), trigger="completion",
                                                         delay_ms=u.think * dt, relation="sequence")]
        rows = [r | dict(request_id=str(j), play_id=str(play[j]), session_id=str(play[j]), model=MOCKER_MODEL,
                         not_before_ms=r["timestamp"] if u.prev is None else 0.0, dependencies=deps(u)) for j, (r, u) in enumerate(zip(rows, trace))]

    score = lambda recs: (soldr_ttft(rows, recs, prefill_latency), sum(q["ttft_ms"] for q in recs) / len(recs) / dt)
    mocker = [score(run_mocker(rows, len(vars["routes"][0]), blk_size)) for _ in range(mocker_runs)]
    soldr_soldr = sum(solver.Value(t) for t in vars["ttfts"]) / len(rows)
    table = [("Soldr decisions", soldr_soldr, sum(soldr_mocker) / len(rows) / dt),
             ("Mocker decisions", sum(s for s, _ in mocker) / mocker_runs, sum(m for _, m in mocker) / mocker_runs)]
    print(f"{'Avg TTFT':18s} {'Soldr':>7s} {'Mocker':>7s}")
    for name, soldr, mock in table:
        print(f"{name:18s} {soldr:7.1f} {mock:7.1f}")


def calibrate_ab():
    chunk = MOCKER_ENGINE["engine"]["max_num_batched_tokens"]
    rows, next_hash = [], 0
    for isl in [512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072, 196608, 245760]:
        for frac in [0, 0.25, 0.5, 0.75, 0.95]:
            hit, hashes = int(isl * frac) // 512 * 512, list(range(next_hash, next_hash + ceil(isl / 512)))
            next_hash += len(hashes)
            if hit:
                rows.append({"timestamp": 60_000 * len(rows), "input_length": hit, "output_length": 1,
                             "hash_ids": hashes[:hit // 512]})
            rows.append({"timestamp": 60_000 * len(rows), "input_length": isl, "output_length": 1, "hash_ids": hashes})
    x, y = [], []
    for r, q in zip(rows, run_mocker(rows, 1, 512)):
        new, hit = r["input_length"] - q["reused_input_tokens"], q["reused_input_tokens"]
        x.append([new, (r["input_length"] ** 2 - hit ** 2) / 2, ceil(new / chunk)])
        y.append(q["ttft_ms"])
    x, y = np.array(x), np.array(y)
    (a, b, _), *_ = np.linalg.lstsq(x / y[:, None], np.ones(len(y)), rcond=None)
    return float(a), float(b)


def calibrate(path, workers, out="mocker_params.json"):
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    recs = run_mocker(rows, workers, 512)
    target = sum(q["ttft_ms"] for q in recs) / len(recs)
    decode_ms = round(sum(q["last_token_ms"] - q["first_token_ms"] for q in recs)
                      / sum(q["output_length"] - 1 for q in recs))
    a, b = calibrate_ab()
    engine = MOCKER_ENGINE["engine"]
    p = MockerLatency(decode_ms, a, b, 0.0, engine["max_num_batched_tokens"], engine["block_size"])
    # Bisect for c
    lo, hi = 0.0, 100.0
    for _ in range(20):
        p = replace(p, c=(lo + hi) / 2)
        if soldr_ttft(rows, recs, p) * decode_ms < target:
            lo = p.c
        else:
            hi = p.c
    p = replace(p, c=round((lo + hi) / 2, 1))
    with open(out, "w") as f:
        json.dump(asdict(p), f, indent=1)
    print(f"{p} (Mocker mean TTFT {target:.1f} ms)")
