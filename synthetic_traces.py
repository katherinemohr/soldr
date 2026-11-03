"""Generate controlled Mooncake-format workloads and optionally solve them.

Run from the repository root:
    .venv/bin/python synthetic_traces.py --solve
    .venv/bin/python synthetic_traces.py --solve --case four_prefixes --objective p90

Each trace has 16 requests: four waves of four, 1,200 ms apart, with
8,192 input tokens and 16 output tokens per request. The first wave seeds
the caches. No worker assignments or reuse decisions are prescribed.

same_prefix: all prompts identical.
distinct_prefixes: no shared blocks between requests.
four_prefixes: four independent 7,680-token prefixes, each followed by a
unique 512-token suffix. Request i belongs to group i % 4.

Mean TTFT is the default because p90 may leave some requests unoptimized.
Schedules are saved as schedule_synthetic_<case>_<objective>.json.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cache_packing as packing
from ortools.sat.python import cp_model


CASES = ("same_prefix", "distinct_prefixes", "four_prefixes")
BLOCK_SIZE = 512
NUM_REQUESTS = 16
NUM_WORKERS = 4
WAVE_MS = 1200
TRACE_DIR = Path("traces/synthetic")


def make_rows(case):
    rows = []
    for i in range(NUM_REQUESTS):
        if case == "same_prefix":
            group, hashes = 0, list(range(16))
        elif case == "distinct_prefixes":
            group, hashes = i, list(range(16 * i, 16 * (i + 1)))
        else:
            group = i % 4
            hashes = list(range(15 * group, 15 * (group + 1))) + [1000 + i]
        rows.append(dict(timestamp=(i // 4) * WAVE_MS,
                         input_length=8192, output_length=16,
                         hash_ids=hashes, prefix_group=group))
    return rows


def validate(trace, rows, block_size, case):
    """Check the overlap matrix, not just the JSON's shape."""
    assert len(trace) == NUM_REQUESTS
    for i, u in enumerate(trace):
        assert len(u.hash_ids) * block_size == u.isl
        for j in range(i):
            expected = 0
            if case == "same_prefix":
                expected = 8192
            elif case == "four_prefixes" and rows[i]["prefix_group"] == rows[j]["prefix_group"]:
                expected = 7680
            assert packing.shared_tokens(u, trace[j], block_size) == expected
    assert packing.default_latency(8192) * packing.default_latency.decode_ms < WAVE_MS


def solve_case(case, trace, rows, block_size, args):
    latency = packing.default_latency
    objective = "mean" if args.objective == "mean" else 0.90
    # Plenty of room beyond the arrivals, without the production-sized horizon.
    horizon = max(r.arrival for r in trace) + sum(latency(r.isl) + r.osl for r in trace) + 1
    model, variables = packing.build_model(
        trace, block_size, num_caches=NUM_WORKERS, mem=latency.kv_capacity,
        horizon=horizon, prefill_latency=latency, objective=objective,
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = args.time_limit
    solver.parameters.num_workers = 4
    solver.parameters.random_seed = 0
    status = solver.Solve(model)
    print(f"  {solver.StatusName(status)}; objective={solver.ObjectiveValue():g}; "
          f"bound={solver.BestObjectiveBound():g}")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("  No solution available; try a longer --time-limit.")
        return False

    routes = [next(c for c, var in enumerate(rs) if solver.Value(var))
              for rs in variables["routes"]]
    groups = defaultdict(Counter)
    for row, worker in zip(rows, routes):
        groups[row["prefix_group"]][worker] += 1
    print("  group -> {worker: request count}")
    for group, counts in sorted(groups.items()):
        print(f"    {group}: {dict(sorted(counts.items()))}")
    ttfts = [solver.Value(t) for t in variables["ttfts"]]
    hits = sum(solver.Value(v) for rs in variables["reuses"] for v in rs)
    print(f"  mean TTFT={sum(ttfts) / len(ttfts) * latency.decode_ms:.1f} ms; "
          f"requests reusing a prefix={hits}/{len(trace)}")
    if case == "four_prefixes":
        colocated = all(len(counts) == 1 for counts in groups.values())
        separate = colocated and len({next(iter(counts)) for counts in groups.values()}) == 4
        print(f"  Each prefix group stays on one worker: {colocated}")
        print(f"  Four groups use four different workers: {separate}")
        if status == cp_model.OPTIMAL and args.objective == "mean":
            assert separate, "Expected one distinct worker per prefix group for this controlled workload"
    packing.dump_schedule(f"synthetic_{case}_{args.objective}", trace, block_size,
                          solver, status, variables, latency)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=("all", *CASES), default="all")
    parser.add_argument("--solve", action="store_true", help="also run the cache-packing optimizer")
    parser.add_argument("--objective", choices=("mean", "p90"), default="mean")
    parser.add_argument("--time-limit", type=float, default=30, help="solver seconds per case")
    args = parser.parse_args()
    if args.time_limit <= 0:
        parser.error("--time-limit must be positive")
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    success = True
    for case in CASES if args.case == "all" else (args.case,):
        rows = make_rows(case)
        path = TRACE_DIR / f"{case}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        trace, block_size = packing.from_mooncake(path)
        validate(trace, rows, block_size, case)
        print(f"{case}: wrote and validated {path}", flush=True)
        if args.solve:
            success = solve_case(case, trace, rows, block_size, args) and success
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
