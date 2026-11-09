"""Bounded baseline/aggregate comparisons; run from repository root.

    .venv/bin/python benchmark_prefill_overlap.py --time-limit 30
    .venv/bin/python benchmark_prefill_overlap.py --cases conversation_128 --objective p90

Each combination is a single trial, with four search workers and seed zero.
Baseline and aggregate use identical inputs and horizons. Multithreaded timings
can vary. Each subprocess has a separate wall timeout including model building.
Default reports are outputs/benchmark_prefill_overlap_<metric>_<objective>.json.
ERROR/WALL_TIMEOUT runs make the command exit nonzero; a solver's UNKNOWN or
INFEASIBLE status is still a valid benchmark measurement.
"""
import argparse
import json
from math import ceil
from pathlib import Path
import platform
import subprocess
import sys
import time
import ortools
from ortools.sat.python import cp_model
import cache_packing as packing
from schedule_validation import check_exclusive_schedule
from synthetic_traces import CASES, make_rows


class FirstSolution(cp_model.CpSolverSolutionCallback):
    def __init__(self):
        super().__init__()
        self.seconds = None

    def on_solution_callback(self):
        if self.seconds is None:
            self.seconds = self.WallTime()


def run(case, metric, objective, exclusive, limit):
    if case in CASES:
        trace = [packing.Request(r['input_length'], r['output_length'], r['timestamp']//packing.default_latency.decode_ms,
                                 r['hash_ids']) for r in make_rows(case)]
        block_size = 512
    else:
        name, count = case.rsplit('_', 1)
        trace, block_size = packing.from_mooncake(f'traces/mooncake_{name}.jsonl', n=int(count))
    latency = packing.default_latency
    horizon = max(r.arrival for r in trace) + sum(latency(r.isl) + r.osl for r in trace) + 1
    started = time.perf_counter()
    model, variables = packing.build_model(trace, block_size, 4, latency.kv_capacity, horizon, latency,
                                           objective='mean' if objective == 'mean' else .9, metric=metric,
                                           exclusive_prefill=exclusive)
    build_seconds = time.perf_counter() - started
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = limit
    solver.parameters.num_workers = 4
    solver.parameters.random_seed = 0
    callback = FirstSolution()
    status = solver.Solve(model, callback)
    result = dict(case=case, requests=len(trace), output_tokens=sum(r.osl for r in trace), metric=metric,
                  objective=objective, mode='aggregate' if exclusive else 'baseline', horizon=horizon,
                  build_seconds=build_seconds, solve_seconds=solver.WallTime(), first_solution_seconds=callback.seconds,
                  status=solver.StatusName(status), variables=len(model.Proto().variables),
                  constraints=len(model.Proto().constraints), limit_seconds=limit, search_workers=4, seed=0,
                  bound=solver.BestObjectiveBound(),
                  environment=dict(python=platform.python_version(), ortools=ortools.__version__,
                                   machine=platform.machine()))
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        value = solver.ObjectiveValue()
        result.update(value=value, gap=(value-solver.BestObjectiveBound())/max(abs(value),1))
        if exclusive:
            check_exclusive_schedule(trace, variables, solver)
        for label, key in [('ttft', 'ttfts'), ('e2e', 'e2es')]:
            times = sorted(solver.Value(t)*latency.decode_ms for t in variables[key])
            result[f'mean_{label}_ms'] = sum(times)/len(times)
            result[f'p90_{label}_ms'] = times[ceil(.9*len(times))-1]
        packing.dump_schedule(f'overlap_benchmark_{case}_{objective}', trace, block_size, solver, status, variables, latency)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--time-limit', type=float, default=30)
    parser.add_argument('--cases', nargs='+', default=[*CASES, 'conversation_16', 'conversation_32', 'conversation_128'])
    parser.add_argument('--metric', choices=['ttft', 'e2e'], default='e2e')
    parser.add_argument('--objective', choices=['mean', 'p90'], default='mean')
    parser.add_argument('--modes', nargs='+', choices=['baseline', 'aggregate'], default=['baseline', 'aggregate'])
    parser.add_argument('--output', type=Path, help='report path (default includes metric and objective)')
    parser.add_argument('--single', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.time_limit <= 0:
        parser.error('--time-limit must be positive')
    if args.output is None:
        args.output = Path(f'outputs/benchmark_prefill_overlap_{args.metric}_{args.objective}.json')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.single:
        result = run(args.cases[0], args.metric, args.objective, args.modes[0]=='aggregate', args.time_limit)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        return
    results, failed = [], False
    for case in args.cases:
        for mode in args.modes:
            print(f'Running {case}: {mode}, {args.metric}/{args.objective}, {args.time_limit:g}s', flush=True)
            part = args.output.with_suffix('.part.json')
            cmd = [sys.executable, __file__, '--single', '--cases', case, '--metric', args.metric,
                   '--objective', args.objective, '--modes', mode, '--time-limit', str(args.time_limit), '--output', str(part)]
            started = time.perf_counter()
            try:
                completed = subprocess.run(cmd, capture_output=True, text=True, timeout=args.time_limit+90)
                if completed.returncode:
                    result = dict(case=case, mode=mode, metric=args.metric, objective=args.objective,
                                  status='ERROR', error=completed.stderr[-3000:])
                else:
                    result = json.loads(part.read_text())
            except subprocess.TimeoutExpired:
                result = dict(case=case, mode=mode, metric=args.metric, objective=args.objective,
                              status='WALL_TIMEOUT')
            failed |= result['status'] in ('ERROR', 'WALL_TIMEOUT')
            result['process_seconds'] = time.perf_counter()-started
            results.append(result)
            args.output.write_text(json.dumps(results, indent=2)+'\n')
            print(json.dumps(result), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
