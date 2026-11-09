"""Independent work-conserving simulation checks for aggregate prefill blocking."""
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ortools.sat.python import cp_model
import cache_packing as packing


class FixedLatency:
    decode_ms = 1
    config = 'test'

    def __call__(self, isl, hit=0):
        return 2


def simulate_end(prefill_end, output, prefills):
    time, remaining = prefill_end, output
    while remaining:
        if not any(a <= time < b for a, b in prefills):
            remaining -= 1
        time += 1
    return time


def check_schedule(trace, variables, solver):
    value = solver.Value
    workers = [next(c for c, route in enumerate(rs) if value(route)) for rs in variables['routes']]
    for i, request in enumerate(trace):
        prefills = [(value(variables['starts'][j]), value(variables['prefill_ends'][j]))
                    for j in range(len(trace)) if workers[j] == workers[i]]
        expected = simulate_end(value(variables['prefill_ends'][i]), request.osl, prefills)
        assert value(variables['ends'][i]) == expected, (i, expected, value(variables['ends'][i]))
        assert value(variables['ttfts'][i]) == value(variables['prefill_ends'][i]) - value(variables['releases'][i])


class AggregateTests(unittest.TestCase):
    def make(self, outputs=(5, 3), arrivals=(0, 4), workers=1, memory=10000, metric='e2e'):
        trace = [packing.Request(512, output, arrival, [i]) for i, (output, arrival) in enumerate(zip(outputs, arrivals))]
        model, variables = packing.build_model(trace, 512, workers, memory, 40, FixedLatency(),
                                               metric=metric, exclusive_prefill=True)
        for i, arrival in enumerate(arrivals):
            model.Add(variables['starts'][i] == arrival)
        return trace, model, variables

    def solve(self, model):
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 5
        solver.parameters.num_workers = 1
        status = solver.Solve(model)
        return solver, status

    def test_interruptions_and_shared_decode(self):
        trace, model, variables = self.make(outputs=(7, 3, 1), arrivals=(0, 4, 7))
        solver, status = self.solve(model)
        self.assertEqual(status, cp_model.OPTIMAL)
        self.assertEqual([solver.Value(t) for t in variables['ends']], [13, 11, 10])
        check_schedule(trace, variables, solver)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(packing, '__file__', str(Path(tmp) / 'cache_packing.py')):
                packing.dump_schedule('test', trace, 512, solver, status, variables, FixedLatency())
            data = json.loads((Path(tmp) / 'outputs/schedule_test_aggregate_e2e.json').read_text())
            self.assertEqual(data['requests'][0]['decode_segments'], [[2, 4], [6, 7], [9, 13]])
            self.assertEqual(data['requests'][0]['decode_pause'], 4)

    def test_completion_cannot_drift_across_prefill(self):
        _, model, variables = self.make(outputs=(2, 1))
        model.Add(variables['ends'][0] == 6)
        _, status = self.solve(model)
        self.assertEqual(status, cp_model.INFEASIBLE)

    def test_other_worker_does_not_block(self):
        trace, model, variables = self.make(workers=2)
        model.Add(variables['routes'][1][1] == 1)
        solver, status = self.solve(model)
        self.assertEqual(status, cp_model.OPTIMAL)
        self.assertEqual(solver.Value(variables['ends'][0]), 7)
        check_schedule(trace, variables, solver)

    def test_paused_request_retains_memory(self):
        _, model, _ = self.make(memory=600)
        _, status = self.solve(model)
        self.assertEqual(status, cp_model.INFEASIBLE)

    def test_zero_output_and_ttft(self):
        trace, model, variables = self.make(outputs=(5, 0), metric='ttft')
        solver, status = self.solve(model)
        self.assertEqual(status, cp_model.OPTIMAL)
        self.assertEqual(solver.Value(variables['ends'][1]), 6)
        self.assertEqual(solver.ObjectiveValue(), 4)
        check_schedule(trace, variables, solver)

    def test_session_release_uses_paused_completion(self):
        trace = [packing.Request(512, 5, 0, [0]), packing.Request(512, 3, 4, [1]),
                 packing.Request(512, 1, None, [2], prev=0, think=2)]
        model, variables = packing.build_model(trace, 512, 1, 10000, 40, FixedLatency(), exclusive_prefill=True)
        model.Add(variables['starts'][0] == 0)
        model.Add(variables['starts'][1] == 4)
        solver, status = self.solve(model)
        self.assertEqual(status, cp_model.OPTIMAL)
        self.assertEqual(solver.Value(variables['releases'][2]), 11)
        self.assertEqual(solver.Value(variables['ends'][2]), 14)
        check_schedule(trace, variables, solver)

    def test_small_grid_against_independent_step_simulation(self):
        # Every completion is checked, even under a TTFT-only objective that
        # otherwise would permit spurious end-time slack.
        for second, third, outputs in itertools.product([2, 3, 4], [6, 8], itertools.product([0, 1, 4], repeat=3)):
            trace, model, variables = self.make(outputs=outputs, arrivals=(0, second, third), metric='ttft')
            solver, status = self.solve(model)
            self.assertEqual(status, cp_model.OPTIMAL)
            check_schedule(trace, variables, solver)


if __name__ == '__main__':
    unittest.main()
