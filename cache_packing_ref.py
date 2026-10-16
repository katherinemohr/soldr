from math import ceil
from ortools.sat.python import cp_model

BLK_SIZE = 1000 # No. of tokens per hash_id

# ISL, OSL, arrival, hash_ids
trace = [(2000, 200, 0, [0, 1]), (4000, 100, 1, [0, 3, 4, 5])]

for req in trace:
    assert ceil(req[0] / BLK_SIZE) == len(req[3])

# Approximate 200 prefill tokens ~ 1 decode token = 1 timestep
prefill_latency = lambda ctx: max(1, ctx // 200)

# Tokens in the common prefix of r1 and r2's hash_ids
def shared_tokens(r1, r2):
    count = 0
    for b1, b2 in zip(r1[3], r2[3]):
        if b1 != b2:
            break
        count += 1
    return min(count * BLK_SIZE, r1[0], r2[0])

# Reduction in prefill time of r2 if it gets a cache hit from r1
def saving(r1, r2):
    saved_toks = shared_tokens(r1, r2)
    return prefill_latency(r2[0]) - prefill_latency(r2[0] - saved_toks)

mem = 50000
assert mem >= max([req[0] + req[1] for req in trace])

horizon = 50000
assert horizon > sorted(map(lambda req: prefill_latency(req[0]) + req[1] + req[2], trace))[-1]

model = cp_model.CpModel()

starts = [model.NewIntVar(req[2], horizon, f"start[{idx}]") for idx, req in enumerate(trace)]
ends = [model.NewIntVar(req[2], horizon, f"end[{idx}]") for idx, req in enumerate(trace)]

# reuse[i][j] implies ends[i] <= starts[j]
reuse = [[model.NewBoolVar(f"reuse[{i}{j}]") for j in range(len(trace))] for i in range(len(trace))]
for i in range(len(trace)):
    for j in range(len(trace)):
        if i == j:
            # Don't get a cache hit from yourself
            model.Add(reuse[i][j] == 0)
            continue
        model.Add(ends[i] <= starts[j]).OnlyEnforceIf(reuse[i][j])

for i in range(len(trace)):
    model.AddAtMostOne(reuse[j][i] for j in range(len(trace)))

intervals = []
sizes = []
for idx in range(len(trace)):
    # Prefill latency for a full hit is 1
    prefill = model.NewIntVar(1, prefill_latency(trace[idx][0]), f"prefill[{idx}]")
    model.Add(prefill == prefill_latency(trace[idx][0]) - sum(reuse[i][idx] * saving(trace[i], trace[idx]) for i in range(len(trace))))
    duration = prefill + trace[idx][1]
    interval = model.NewIntervalVar(starts[idx], duration, ends[idx], f"interval[{idx}]")
    intervals.append(interval)
    sizes.append(trace[idx][0] + trace[idx][1])

# Source i keeps its prefix in memory from its end until the last request that reuses it starts
for i in range(len(trace)):
    cached = model.NewBoolVar(f"cached[{i}]")
    keep = model.NewIntVar(0, horizon, f"keep[{i}]")
    evict = model.NewIntVar(0, horizon, f"evict[{i}]")
    for j in range(len(trace)):
        model.AddImplication(reuse[i][j], cached)
        model.Add(evict >= starts[j]).OnlyEnforceIf(reuse[i][j])
    intervals.append(model.NewOptionalIntervalVar(ends[i], keep, evict, cached, f"cache[{i}]"))
    sizes.append(max(shared_tokens(trace[i], trace[j]) for j in range(len(trace)) if j != i))

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