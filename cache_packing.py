from math import ceil
from ortools.sat.python import cp_model

BLK_SIZE = 1000 # No. of tokens per hash_id

# ISL, OSL, arrival, hash_ids
trace = [(2000, 200, 0, [0, 1]), (4000, 100, 1, [0, 3, 4, 5])]

for u in trace:
    assert ceil(u[0] / BLK_SIZE) == len(u[3])

# Approximate 200 prefill tokens ~ 1 decode token = 1 timestep
prefill_latency = lambda ctx: max(1, ctx // 200)

# Tokens in the common prefix of u and v's hash_ids
def shared_tokens(u, v):
    count = 0
    for b1, b2 in zip(u[3], v[3]):
        if b1 != b2:
            break
        count += 1
    return min(count * BLK_SIZE, u[0], v[0])

# Reduction in prefill time of v if it gets a cache hit from u
def saving(u, v):
    saved_toks = shared_tokens(u, v)
    return prefill_latency(v[0]) - prefill_latency(v[0] - saved_toks)

mem = 50000
assert mem >= max([u[0] + u[1] for u in trace])

horizon = 50000
assert horizon > sorted(map(lambda u: prefill_latency(u[0]) + u[1] + u[2], trace))[-1]

model = cp_model.CpModel()

starts = [model.NewIntVar(u[2], horizon, f"start[{i}]") for i, u in enumerate(trace)]
ends = [model.NewIntVar(u[2], horizon, f"end[{i}]") for i, u in enumerate(trace)]

# reuses[i][j] means j gets the largest possible cache hit from i.
# TODO: Allow partial hits
reuses = [[model.NewBoolVar(f"reuse[{i}{j}]") for j in range(len(trace))] for i in range(len(trace))]

# Don't get a cache hit from yourself
for i, u in enumerate(trace):
    model.Add(reuses[i][i] == False)

# To get a cache hit, you must start after your provider is done
for i in range(len(trace)):
    for j in range(len(trace)):
        model.Add(ends[i] <= starts[j]).OnlyEnforceIf(reuses[i][j])

# You may only get a cache hit from one prior request
for j in range(len(trace)):
    model.AddAtMostOne(reuses[i][j] for i in range(len(trace)))

intervals = []
sizes = []
for j, v in enumerate(trace):
    # Prefill latency for a full hit is 1
    prefill = model.NewIntVar(1, prefill_latency(v[0]), f"prefill[{j}]")
    model.Add(prefill == prefill_latency(v[0]) - sum(reuses[i][j] * saving(u, v) for i, u in enumerate(trace)))
    duration = prefill + v[1]
    intervals.append(model.NewIntervalVar(starts[j], duration, ends[j], f"lifetime[{j}]"))
    sizes.append(v[0] + v[1])

for i, u in enumerate(trace):
    # Time at which we evict u's prompt
    evict = model.NewIntVar(0, horizon, f"evict[{i}]")
    model.add(evict >= ends[i])

    # Is u providing a cache hit to anybody?
    is_provider = model.NewBoolVar(f"provider[{i}]")

    for j, v in enumerate(trace):
        # If v is getting a cache hit from u, u's prompt must not be evicted until v starts
        model.Add(evict >= starts[j]).OnlyEnforceIf(reuses[i][j])
        # ... and u is a provider
        model.AddImplication(reuses[i][j], is_provider)

    _evict = model.NewIntVar(0, horizon, f"_evict[{i}]")
    intervals.append(model.NewOptionalIntervalVar(ends[i], _evict, evict, is_provider, f"eviction[{i}]"))
    # TODO: This is an overestimate I think
    sizes.append(max(shared_tokens(u, v) for j, v in enumerate(trace) if j != i))

model.AddCumulative(intervals, sizes, mem)

waits = []
for i, u in enumerate(trace):
    wait = model.NewIntVar(0, horizon, f"waits[{i}]")
    model.Add(wait == starts[i] - u[2])
    waits.append(wait)
model.minimize(sum(waits))

solver = cp_model.CpSolver()
status = solver.Solve(model)
print(solver.StatusName(status))

for i, u in enumerate(trace):
    print(f"waits[{i}]= ", solver.Value(waits[i]))