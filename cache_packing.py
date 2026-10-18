import json
from itertools import islice
from math import ceil
from ortools.sat.python import cp_model

C = 4 # No. of caches
N = 256 # No. of requests
MEM = 1_000_000
HORIZON = 100_000

# Approximate 800 prefill tokens ~ 1 decode token = 1 timestep
# TODO: What is a good way to pick this?
prefill_latency = lambda ctx: max(1, ctx // 800)

# TODO: What is a good way to pick ms_per_step?
def from_mooncake(path, n=None, ms_per_step=40):
    with open(path) as f:
        rows = [json.loads(line) for line in islice(f, n)]
    t0 = rows[0]["timestamp"]
    trace = [(r["input_length"], r["output_length"], (r["timestamp"] - t0) // ms_per_step, r["hash_ids"]) for r in rows]
    return trace, 512

def from_weka(path, lanes, n=None, ms_per_step=15, idle_cap_s=300):
    plays = []
    with open(path) as f:
        for line in f:
            play = json.loads(line)
            # Ignore sessions with subagents
            if all(r["type"] != "subagent" for r in play["requests"]):
                plays.append(play)
            if len(plays) == lanes:
                break
    assert len(plays) == lanes, f"only {len(plays)} plays without subagents"
    blk_sizes = {p["block_size"] for p in plays}
    assert len(blk_sizes) == 1

    trace, offset = [], 0
    for play in plays:
        shift, prev_t = 0.0, 0.0
        for r in play["requests"]:
            idle = r["t"] - prev_t
            # Cap the gap between successive requests from a session. This
            # appears to be a standard transformation for Weka and helps
            # decrease the horizon for us.
            if idle > idle_cap_s:
                shift += idle - idle_cap_s
            arrival = round((r["t"] - shift) * 1000) // ms_per_step
            trace.append((r["in"], r["out"], arrival, [h + offset for h in r["hash_ids"]]))
            prev_t = r["t"]
        offset += 1 + max(h for r in play["requests"] for h in r["hash_ids"])

    trace.sort(key=lambda u: u[2])
    return trace[:n], blk_sizes.pop()

# (ISL, OSL, arrival, hash_ids)
trace, BLK_SIZE = from_weka("traces/weka_256k.jsonl", lanes=8, n=N)
assert len(trace) == N
assert MEM >= max([u[0] + u[1] for u in trace])
assert HORIZON > sorted(map(lambda u: prefill_latency(u[0]) + u[1] + u[2], trace))[-1]
for u in trace:
    assert ceil(u[0] / BLK_SIZE) == len(u[3])

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

model = cp_model.CpModel()

routes = [[model.NewBoolVar(f"route[{i},{c}]") for c in range(C)] for i in range(N)]

# For symmetry breaking
model.Add(routes[0][0] == True)

for i, u in enumerate(trace):
    model.AddExactlyOne(routes[i][c] for c in range(C))

starts = [model.NewIntVar(u[2], HORIZON, f"start[{i}]") for i, u in enumerate(trace)]
ends = [model.NewIntVar(u[2], HORIZON, f"end[{i}]") for i, u in enumerate(trace)]

# reuses[i][j] means j gets the largest possible cache hit from i.
# TODO: Allow partial hits
reuses = [[model.NewBoolVar(f"reuse[{i},{j}]") for j in range(N)] for i in range(N)]

# Don't get a cache hit from yourself
for i, u in enumerate(trace):
    model.Add(reuses[i][i] == False)

# To get a cache hit, you must start after your provider is done
for i in range(N):
    for j in range(N):
        model.Add(ends[i] <= starts[j]).OnlyEnforceIf(reuses[i][j])

# You may only get a cache hit from one provider
for j in range(N):
    model.AddAtMostOne(reuses[i][j] for i in range(N))

# If i provides j a cache hit and j is routed to c, then i must also be routed to c
for i in range(N):
    for j in range(N):
        for c in range(C):
            model.AddBoolOr([reuses[i][j].Not(), routes[j][c].Not(), routes[i][c]])

intervals = [[] for c in range(C)]
sizes = [[] for c in range(C)]
for j, v in enumerate(trace):
    # Prefill latency for a full hit is 1
    prefill = model.NewIntVar(1, prefill_latency(v[0]), f"prefill[{j}]")
    model.Add(prefill == prefill_latency(v[0]) - sum(reuses[i][j] * saving(u, v) for i, u in enumerate(trace)))
    duration = prefill + v[1]
    for c in range(C):
        intervals[c].append(model.NewOptionalIntervalVar(starts[j], duration, ends[j], routes[j][c], f"lifetime[{j},{c}]"))
        sizes[c].append(v[0] + v[1])

for i, u in enumerate(trace):
    # Time at which we evict u's prompt
    evict = model.NewIntVar(0, HORIZON, f"evict[{i}]")
    model.add(evict >= ends[i])

    # Is u providing a cache hit to anybody?
    is_provider = model.NewBoolVar(f"provider[{i}]")

    for j, v in enumerate(trace):
        # If v is getting a cache hit from u, u's prompt must not be evicted until v starts
        model.Add(evict >= starts[j]).OnlyEnforceIf(reuses[i][j])
        # ... and u is a provider
        model.AddImplication(reuses[i][j], is_provider)

    _evict = model.NewIntVar(0, HORIZON, f"_evict[{i}]")
    for c in range(C):
        # cond == is_provider AND routes[i][c]
        # It's annoying that CP-SAT doesn't let us write SAT formulas naturally...
        cond = model.NewBoolVar(f"cond[{i},{c}]")
        model.AddBoolOr([is_provider.Not(), routes[i][c].Not(), cond])
        model.AddImplication(cond, is_provider)
        model.AddImplication(cond, routes[i][c])

        intervals[c].append(model.NewOptionalIntervalVar(ends[i], _evict, evict, cond, f"eviction[{i},{c}]"))
        # TODO: This is an overestimate I think
        sizes[c].append(max(shared_tokens(u, v) for j, v in enumerate(trace) if j != i))

for c in range(C):
    model.AddCumulative(intervals[c], sizes[c], MEM)

# waits = []
# for i, u in enumerate(trace):
#     wait = model.NewIntVar(0, HORIZON, f"waits[{i}]")
#     model.Add(wait == starts[i] - u[2])
#     waits.append(wait)
# model.minimize(sum(waits))

ttfts = []
for i, u in enumerate(trace):
    ttft = model.NewIntVar(0, HORIZON, f"ttft[{i}]")
    model.Add(ttft == ends[i] - u[1] - u[2])
    ttfts.append(ttft)
model.minimize(sum(ttfts))

solver = cp_model.CpSolver()
status = solver.Solve(model)
print(solver.StatusName(status))

if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    avg_ttft = sum(solver.Value(ttfts[i]) for i in range(N)) / N
    print("Avg TTFT: ", avg_ttft)

    if status != cp_model.OPTIMAL:
        obj, bound = solver.ObjectiveValue(), solver.BestObjectiveBound()
        print(f"Gap: {(obj - bound) / max(abs(obj), 1):.1%}")