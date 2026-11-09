"""Generate controlled Mooncake-format workloads and optionally solve them.

Run from the repository root:
    .venv/bin/python synthetic_traces.py --solve
    .venv/bin/python synthetic_traces.py --solve --case four_prefixes --objective p90
    .venv/bin/python synthetic_traces.py --solve --case four_prefixes --metric ttft
    .venv/bin/python synthetic_traces.py --solve --wave-ms 1200
    .venv/bin/python synthetic_traces.py --solve --case bubbles --exclusive-prefill
    python3 visualization/visualize_schedule.py outputs/schedule_synthetic_bubble_*_aggregate_e2e.json --output visualization/decode_bubbles.html

The original three traces have 16 requests: four waves of four, 60 ms apart by default,
with 8,192 input tokens and 16 output tokens per request. Use --wave-ms
to change spacing (0 makes one simultaneous burst). The first wave
introduces the prefixes. No worker assignments or reuse decisions are prescribed.

same_prefix: all prompts identical.
distinct_prefixes: no shared blocks between requests.
four_prefixes: four independent 7,680-token prefixes, each followed by a
unique 512-token suffix. Request i belongs to group i % 4.

The bubbles cases use 2–3 distinct prompts on 1–2 workers, with fixed arrivals
chosen to show decode pauses, concurrent decoding, and an exact completion boundary.

Mean end-to-end latency is the default because p90 may leave some requests unoptimized.
Use --metric ttft to optimize time to first token instead.
Schedules are saved as outputs/schedule_synthetic_<case>_<objective>_<metric>.json.
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
WAVE_MS = 60
TRACE_DIR = Path("traces/synthetic")
# Bubble demos use distinct 8,192-token prompts. Times below are solver steps.
_prefill = packing.default_latency(8192)
_arrival = _prefill + 16
BUBBLES = {
    "bubble_01_repeated_pauses": (1, [(0, 100), (_arrival, 60), (2 * _arrival, 8)],
        "#1 interrupts #0, then joins its decode. #2 interrupts both. All three decode together afterward."),
    "bubble_02_one_pause": (1, [(0, 80), (_arrival, 8)],
        "#1's prefill pauses #0. Its solid bar aligns with #0's hatched bubble; both decode afterward."),
    "bubble_03_other_worker": (2, [(0, 80), (_arrival, 8)],
        "The same requests, now on two workers. #1's prefill does not interrupt #0's decode."),
    "bubble_04_exact_boundary": (1, [(0, 16), (_arrival, 8)],
        "#0 finishes exactly when #1 begins prefill. #0 must not extend through that prefill."),
}


def make_rows(case, wave_ms=WAVE_MS):
    if case in BUBBLES:
        return [dict(timestamp=arrival * packing.default_latency.decode_ms, input_length=8192,
                     output_length=output, hash_ids=list(range(16*i, 16*(i+1))), prefix_group=i)
                for i, (arrival, output) in enumerate(BUBBLES[case][1])]
    rows = []
    for i in range(NUM_REQUESTS):
        if case == "same_prefix":
            group, hashes = 0, list(range(16))
        elif case == "distinct_prefixes":
            group, hashes = i, list(range(16 * i, 16 * (i + 1)))
        else:
            group = i % 4
            hashes = list(range(15 * group, 15 * (group + 1))) + [1000 + i]
        rows.append(dict(timestamp=(i // 4) * wave_ms,
                         input_length=8192, output_length=16,
                         hash_ids=hashes, prefix_group=group))
    return rows


def validate(trace, rows, block_size, case):
    """Check the overlap matrix, not just the JSON's shape."""
    assert len(trace) == (len(BUBBLES[case][1]) if case in BUBBLES else NUM_REQUESTS)
    for i, u in enumerate(trace):
        assert len(u.hash_ids) * block_size == u.isl
        for j in range(i):
            expected = 0
            if case == "same_prefix":
                expected = 8192
            elif case == "four_prefixes" and rows[i]["prefix_group"] == rows[j]["prefix_group"]:
                expected = 7680
            assert packing.shared_tokens(u, trace[j], block_size) == expected


def solve_case(case, trace, rows, block_size, args):
    latency = packing.default_latency
    objective = "mean" if args.objective == "mean" else 0.90
    # Plenty of room beyond the arrivals, without the production-sized horizon.
    horizon = max(r.arrival for r in trace) + sum(latency(r.isl) + r.osl for r in trace) + 1
    model, variables = packing.build_model(
        trace, block_size, num_caches=BUBBLES[case][0] if case in BUBBLES else NUM_WORKERS, mem=latency.kv_capacity,
        horizon=horizon, prefill_latency=latency, objective=objective, metric=args.metric,
        exclusive_prefill=args.exclusive_prefill,
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
    latencies = [solver.Value(t) for t in variables["ttfts" if args.metric == "ttft" else "e2es"]]
    hits = sum(solver.Value(v) for rs in variables["reuses"] for v in rs)
    print(f"  mean {args.metric.upper()}={sum(latencies) / len(latencies) * latency.decode_ms:.1f} ms; "
          f"requests reusing a prefix={hits}/{len(trace)}")
    if case == "four_prefixes":
        colocated = all(len(counts) == 1 for counts in groups.values())
        separate = colocated and len({next(iter(counts)) for counts in groups.values()}) == 4
        print(f"  Each prefix group stays on one worker: {colocated}")
        print(f"  Four groups use four different workers: {separate}")
        # Only assert affinity in the original, noncontended experiment.
        # Under overlapping arrivals, splitting a group may be beneficial.
        cold_ms = latency(8192) * latency.decode_ms
        if status == cp_model.OPTIMAL and args.objective == "mean" and args.wave_ms > cold_ms:
            assert separate, "Expected one distinct worker per prefix group for this controlled workload"
    path = packing.dump_schedule(f"synthetic_{case}_{args.objective}", trace, block_size,
                                 solver, status, variables, latency)
    if case in BUBBLES:
        data = json.loads(path.read_text())
        data["description"] = (f"{BUBBLES[case][0]} worker(s); distinct prompts; starts and routes chosen by the solver. "
                               + (BUBBLES[case][2] if args.exclusive_prefill else "Overlap-allowed comparison."))
        for request, row in zip(data["requests"], rows):
            request["prefix_group"] = row["prefix_group"]
        path.write_text(json.dumps(data, indent=1) + "\n")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=("all", "bubbles", *CASES, *BUBBLES), default="all")
    parser.add_argument("--solve", action="store_true", help="also run the cache-packing optimizer")
    parser.add_argument("--objective", choices=("mean", "p90"), default="mean")
    parser.add_argument("--metric", choices=("ttft", "e2e"), default="e2e", help="latency to optimize (default: e2e)")
    parser.add_argument("--exclusive-prefill", action="store_true", help="pause decode using whole-prefill overlap constraints")
    parser.add_argument("--time-limit", type=float, default=30, help="solver seconds per case")
    parser.add_argument("--wave-ms", type=int, default=WAVE_MS,
                        help="milliseconds between waves; 0 for a simultaneous burst (default: 60)")
    args = parser.parse_args()
    if args.time_limit <= 0:
        parser.error("--time-limit must be positive")
    if args.wave_ms < 0:
        parser.error("--wave-ms must be nonnegative")
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    success = True
    cases = CASES if args.case == "all" else BUBBLES if args.case == "bubbles" else (args.case,)
    for case in cases:
        rows = make_rows(case, args.wave_ms)
        path = TRACE_DIR / f"{case}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        trace, block_size = packing.from_mooncake(path)
        validate(trace, rows, block_size, case)
        spacing = "" if case in BUBBLES else f" (wave spacing {args.wave_ms} ms)"
        print(f"{case}: wrote and validated {path}{spacing}", flush=True)
        if args.solve:
            success = solve_case(case, trace, rows, block_size, args) and success
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
