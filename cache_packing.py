from ortools.sat.python import cp_model

# ISL, OSL, arrival, hash_ids
trace = [(2, 2, 0, [0]), (4, 10, 1, [0])]

cap = 20
assert cap >= max([req[0] + req[1] for req in trace])

horizon = 20
assert horizon > sorted(map(lambda req: int(req[0] > 0) + req[1] + req[2], trace))[-1]

model = cp_model.CpModel()

starts = [model.NewIntVar(req[2], horizon, f"start[{idx}]") for idx, req in enumerate(trace)]
ends = [model.NewIntVar(req[2], horizon, f"end[{idx}]") for idx, req in enumerate(trace)]

PROXIMAL = 10
# proximal[i][j] implies starts[i] <= starts[j] <= starts[i] + PROXIMAL
proximal = [[model.NewBoolVar(f"proximal[{i}{j}]") for j in range(len(trace))] for i in range(len(trace))]
for i in range(len(trace)):
    for j in range(len(trace)):
        if i == j:
            # Don't get a cache hit from yourself
            model.Add(proximal[i][j] == 0)
            continue
        model.Add(starts[i] <= starts[j]).OnlyEnforceIf(proximal[i][j])
        model.Add(starts[j] <= starts[i] + PROXIMAL).OnlyEnforceIf(proximal[i][j])

cache_hit = [model.NewBoolVar(f"reuse[{idx}]") for idx in range(len(trace))]
for idx, req in enumerate(trace):
    model.Add(cache_hit[idx] <= sum(proximal[i][idx] for i in range(len(trace))))

intervals = []
sizes = []
for idx, req in enumerate(trace):
    # Cache hit means entire prefill is skipped
    duration = int(req[0] > 0) * (1 - cache_hit[idx]) + req[1]
    interval = model.NewIntervalVar(starts[idx], duration, ends[idx], f"interval[{idx}]")
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