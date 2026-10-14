from ortools.sat.python import cp_model

# ISL, OSL, arrival, hash_ids
trace = [(2, 2, 0, [0]), (4, 10, 1, [0])]

cap = 20
assert cap >= max([req[0] + req[1] for req in trace])

horizon = 20
assert horizon > sorted(map(lambda req: int(req[0] > 0) + req[1] + req[2], trace))[-1]

model = cp_model.CpModel()

starts = []
intervals = []
sizes = []
for idx, req in enumerate(trace):
    start = model.NewIntVar(req[2], horizon, f"start[{idx}]")
    duration = int(req[0] > 0) + req[1]
    interval = model.NewIntervalVar(start, duration, start + duration, f"interval[{idx}]")
    starts.append(start)
    intervals.append(interval)
    sizes.append(req[0] + req[1])

model.AddCumulative(intervals, sizes, cap)

waits = []
for idx, req in enumerate(trace):
    wait = model.NewIntVar(0, horizon, f"waits[{idx}]")
    model.Add(wait == starts[idx] - req[2])
    waits.append(wait)
model.minimize(sum(waits))

solver = cp_model.CpSolver()
status = solver.Solve(model)
print(solver.StatusName(status))

for idx, req in enumerate(trace):
    print(f"waits[{idx}]= ", solver.Value(waits[idx]))