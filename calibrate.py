import json
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from itertools import islice
from math import ceil

import numpy as np

MOCKER_ENGINE = {
    "engine": {
        "backend": "vllm", "block_size": 64, "num_gpu_blocks": 16384, "max_num_seqs": 256,
        "max_num_batched_tokens": 8192, "enable_prefix_caching": True, "enable_chunked_prefill": True,
        "max_model_len": 131072,
        "timing_model": {"type": "external", "provider": "aic", "config": {
            "model": "meta-llama/Meta-Llama-3.1-8B", "system": "h200_sxm", "backend": "vllm",
            "backend_version": "0.24.0", "worker_type": "aggregated", "gemm_quant_mode": "bfloat16",
            "fmha_quant_mode": "bfloat16", "kvcache_quant_mode": "bfloat16", "kv_block_size": 64,
            "estimation_mode": "op_level"}},
    },
    "num_gpu_blocks_is_explicit": True,
}


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


def run_mocker(rows, workers):
    os.environ.setdefault("DYN_LOG", "warn")
    from dynamo.replay import run_trace_replay

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "trace.jsonl")
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        report = run_trace_replay(path, extra_engine_args=MOCKER_ENGINE, num_workers=workers, router_mode="kv_router",
                                  trace_block_size=512, capture_per_request=True)

    key = lambda arrival, isl, osl: (round(arrival), isl, osl)
    recs, hashes = defaultdict(list), {}
    for q in report.to_dict()["per_request"]:
        recs[key(q["arrival_time_ms"], q["input_length"], q["output_length"])].append(q)
    out = []
    for r in rows:
        k = key(r["timestamp"] - rows[0]["timestamp"], r["input_length"], r["output_length"])
        assert hashes.setdefault(k, r["hash_ids"]) == r["hash_ids"], "rows sharing arrival, isl and osl differ"
        out.append(recs[k].pop())
    return out


def soldr_ttft(rows, recs, prefill_latency):
    free, total = {}, 0
    for r, q in zip(rows, recs):
        arrival = (r["timestamp"] - rows[0]["timestamp"]) // prefill_latency.decode_ms
        c = q["decode_worker_idx"]
        start = max(arrival, free.get(c, arrival))
        free[c] = start + prefill_latency(r["input_length"], q["reused_input_tokens"])
        total += free[c] - arrival
    return total / len(rows)


def compare(path, solver, vars, prefill_latency, mocker_runs=3):
    with open(path) as f:
        rows = [json.loads(line) for line in islice(f, len(vars["starts"]))]
    dt = prefill_latency.decode_ms
    route = [next(c for c, r in enumerate(rs) if solver.Value(r)) for rs in vars["routes"]]
    submit = [max(r["timestamp"], rows[0]["timestamp"] + solver.Value(s) * dt) for r, s in zip(rows, vars["starts"])]
    soldr_mocker = [0.0] * len(rows)
    for c in set(route):
        ix = sorted((j for j in range(len(rows)) if route[j] == c), key=lambda j: (submit[j], j))
        for j, q in zip(ix, run_mocker([rows[j] | {"timestamp": submit[j]} for j in ix], workers=1)):
            soldr_mocker[j] = submit[j] - rows[j]["timestamp"] + q["ttft_ms"]

    score = lambda recs: (soldr_ttft(rows, recs, prefill_latency), sum(q["ttft_ms"] for q in recs) / len(recs) / dt)
    mocker = [score(run_mocker(rows, len(vars["routes"][0]))) for _ in range(mocker_runs)]
    soldr_soldr = sum(solver.Value(t) for t in vars["ttfts"]) / len(rows)
    table = [("Soldr decisions", soldr_soldr, sum(soldr_mocker) / len(rows) / dt),
             ("Mocker decisions", sum(s for s, _ in mocker) / mocker_runs, sum(m for _, m in mocker) / mocker_runs)]
    print(f"{'Avg TTFT (steps)':18s} {'Soldr':>7s} {'Mocker':>7s}")
    for name, soldr, mock in table:
        print(f"{name:18s} {soldr:7.1f} {mock:7.1f}")


def calibrate_ab():
    chunk = MOCKER_ENGINE["engine"]["max_num_batched_tokens"]
    rows, next_hash = [], 0
    for isl in [512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304, 122880]:
        for frac in [0, 0.25, 0.5, 0.75, 0.95]:
            hit, hashes = int(isl * frac) // 512 * 512, list(range(next_hash, next_hash + ceil(isl / 512)))
            next_hash += len(hashes)
            if hit:
                rows.append({"timestamp": 60_000 * len(rows), "input_length": hit, "output_length": 1,
                             "hash_ids": hashes[:hit // 512]})
            rows.append({"timestamp": 60_000 * len(rows), "input_length": isl, "output_length": 1, "hash_ids": hashes})
    x, y = [], []
    for r, q in zip(rows, run_mocker(rows, workers=1)):
        new, hit = r["input_length"] - q["reused_input_tokens"], q["reused_input_tokens"]
        x.append([new, (r["input_length"] ** 2 - hit ** 2) / 2, ceil(new / chunk)])
        y.append(q["ttft_ms"])
    x, y = np.array(x), np.array(y)
    (a, b, _), *_ = np.linalg.lstsq(x / y[:, None], np.ones(len(y)), rcond=None)
    return float(a), float(b)


def calibrate(path, workers, out="mocker_params.json"):
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    recs = run_mocker(rows, workers)
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
