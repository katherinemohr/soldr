from math import ceil
from ortools.sat.python import cp_model

CACHE_DURATION = 10
BLK_SIZE = 1000 # No. of tokens per hash_id

# ISL, OSL, arrival, hash_ids
trace = [(2000, 200, 0, [0, 1]), (4000, 100, 1, [0, 3, 4, 5])]

for req in trace:
    assert ceil(req[0] / BLK_SIZE) == len(req[3])

# Approximate 200 prefill tokens ~ 1 decode token = 1 timestep
prefill_latency = lambda ctx: max(1, ctx // 200)

# Reduction in prefill time of r2 if it gets a cache hit from r1
def saving(r1, r2):
    count = 0
    for b1, b2 in zip(r1[3], r2[3]):
        if b1 != b2:
            break
        count += 1
    saved_toks = min(count * BLK_SIZE, r1[0], r2[0])
    return prefill_latency(r2[0]) - prefill_latency(r2[0] - saved_toks)

mem = 50000
assert mem >= max([req[0] + req[1] for req in trace])

horizon = 50000
assert horizon > sorted(map(lambda req: prefill_latency(req[0]) + req[1] + req[2], trace))[-1]

model = cp_model.CpModel()

starts = [model.NewIntVar(req[2], horizon, f"start[{idx}]") for idx, req in enumerate(trace)]
ends = [model.NewIntVar(req[2], horizon, f"end[{idx}]") for idx, req in enumerate(trace)]

# proximal[i][j] implies ends[i] <= starts[j] <= ends[i] + CACHE_DURATION
proximal = [[model.NewBoolVar(f"proximal[{i}{j}]") for j in range(len(trace))] for i in range(len(trace))]
for i in range(len(trace)):
    for j in range(len(trace)):
        if i == j:
            # Don't get a cache hit from yourself
            model.Add(proximal[i][j] == 0)
            continue
        model.Add(ends[i] <= starts[j]).OnlyEnforceIf(proximal[i][j])
        model.Add(starts[j] <= ends[i] + CACHE_DURATION).OnlyEnforceIf(proximal[i][j])

for i in range(len(trace)):
    model.AddAtMostOne(proximal[j][i] for j in range(len(trace)))

intervals = []
sizes = []
for idx in range(len(trace)):
    # Prefill latency for a full hit is 1
    prefill = model.NewIntVar(1, prefill_latency(trace[idx][0]), f"prefill[{idx}]")
    model.Add(prefill == prefill_latency(trace[idx][0]) - sum(proximal[i][idx] * saving(trace[i], trace[idx]) for i in range(len(trace))))
    duration = prefill + trace[idx][1]
    interval = model.NewIntervalVar(starts[idx], duration, ends[idx], f"interval[{idx}]")
    intervals.append(interval)
    sizes.append(trace[idx][0] + trace[idx][1])

model.AddCumulative(intervals, sizes, mem)

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