"""Generate and solve four small decode-pause demonstrations, then build their viewer.

    .venv/bin/python make_bubble_traces.py
    open visualization/decode_bubbles.html

Uses the calibrated latency model, distinct prompts, and mean E2E optimization.
Only the number of workers is prescribed; starts and placements are solver choices.
The Mooncake traces are saved in traces/synthetic/bubbles/ and schedules in outputs/.
"""
import json
from pathlib import Path
import subprocess
import sys

from ortools.sat.python import cp_model
import cache_packing as packing


ROOT = Path(__file__).resolve().parent


def check_demo(condition, case, message):
    if not condition:
        raise RuntimeError(f'{case}: {message}. The calibrated timings may need different demo arrivals or lengths.')


def main():
    latency = packing.default_latency
    dt, prefill = latency.decode_ms, latency(8192)
    first, second = prefill + 16, 2 * prefill + 32
    cases = [
        ('01_repeated_pauses', 1, [(0, 100), (first, 60), (second, 8)],
         '#0 decodes for a long time. #1 interrupts it, then joins its decode. '
         '#2 interrupts BOTH #0 and #1. After #2 prefills, all three decode together.'),
        ('02_one_pause', 1, [(0, 80), (first, 8)],
         '#1 prefills while #0 is decoding. The hatched gap in #0 lines up with #1’s solid prefill. '
         'Both requests decode together afterward.'),
        ('03_other_worker', 2, [(0, 80), (first, 8)],
         'The same requests as “one pause,” now with two workers. The solver places them separately. '
         '#1’s prefill overlaps #0’s decode without interrupting it.'),
        ('04_exact_boundary', 1, [(0, 16), (first, 8)],
         '#0 completes exactly when #1 begins prefill. #0 has no pause and must not be extended '
         'through #1’s prefill. This demonstrates the final-decode-step boundary rule.'),
    ]
    trace_dir = ROOT / 'traces' / 'synthetic' / 'bubbles'
    trace_dir.mkdir(parents=True, exist_ok=True)
    schedules = []
    for case, workers, workload, description in cases:
        rows = [dict(timestamp=arrival * dt, input_length=8192, output_length=output,
                     hash_ids=list(range(i * 16, (i + 1) * 16)), prefix_group=i)
                for i, (arrival, output) in enumerate(workload)]
        trace_path = trace_dir / f'{case}.jsonl'
        trace_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        trace, block_size = packing.from_mooncake(trace_path)
        horizon = max(r.arrival for r in trace) + sum(latency(r.isl) + r.osl for r in trace) + 1
        model, variables = packing.build_model(trace, block_size, workers, latency.kv_capacity,
                                               horizon, latency, metric='e2e', objective='mean', exclusive_prefill=True)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10
        solver.parameters.num_workers = 1
        status = solver.Solve(model)
        if status != cp_model.OPTIMAL:
            raise RuntimeError(f'{case}: expected a proven optimum, got {solver.StatusName(status)}')
        # Demonstrations should emerge naturally from arrivals and worker count.
        check_demo(all(solver.Value(variables['starts'][i]) == arrival for i, (arrival, _) in enumerate(workload)),
                   case, 'a request did not start at arrival')
        config = f'bubbles_{case}_mean'
        schedule = packing.dump_schedule(config, trace, block_size, solver, status, variables, latency)
        data = json.loads(schedule.read_text())
        data['description'] = (f'{workers} worker(s); distinct 8,192-token prompts; each prefill takes '
                               f'{prefill * dt} ms. Starts and placements were chosen by the solver. ' + description)
        data['trace_source'] = str(trace_path)
        for request, row in zip(data['requests'], rows):
            request['prefix_group'] = row['prefix_group']
            check_demo(sum(b-a for a, b in request['decode_segments']) == request['osl'],
                       case, f"request #{request['id']} has incorrect decode work")
            for a, b in request['decode_segments']:
                for other in data['requests']:
                    if request['cache'] == other['cache']:
                        check_demo(b <= other['start'] or a >= other['prefill_end'],
                                   case, f"request #{request['id']} decodes during a prefill")
        a = data['requests'][0]
        if case == '01_repeated_pauses':
            check_demo(a['decode_pause'] == 2 * prefill and len(a['decode_segments']) == 3
                       and data['requests'][1]['decode_pause'] == prefill,
                       case, 'expected two pauses in #0 and one in #1')
        elif case == '02_one_pause':
            check_demo(a['decode_pause'] == prefill and len(a['decode_segments']) == 2,
                       case, 'expected one pause in #0')
        elif case == '03_other_worker':
            check_demo(a['cache'] != data['requests'][1]['cache'] and a['decode_pause'] == 0,
                       case, 'expected separate workers with no pause')
        else:
            check_demo(a['end'] == data['requests'][1]['start'] and a['decode_pause'] == 0,
                       case, 'expected #0 to finish exactly when #1 starts')
        schedule.write_text(json.dumps(data, indent=1) + '\n')
        schedules.append(schedule)
        print(f'{case}: OPTIMAL in {solver.WallTime():.3f}s')
        for r in data['requests']:
            print(f"  #{r['id']} worker {r['cache']}: prefill {r['start']*dt}–{r['prefill_end']*dt} ms; "
                  f"decode {[[a*dt,b*dt] for a,b in r['decode_segments']]}; pause {r['decode_pause']*dt} ms")
    subprocess.run([sys.executable, str(ROOT / 'visualization' / 'visualize_schedule.py'),
                    *map(str, schedules), '--output', str(ROOT / 'visualization' / 'decode_bubbles.html')], check=True)


if __name__ == '__main__':
    main()
