import json
from dataclasses import dataclass, replace
from itertools import islice
from math import ceil
from ortools.sat.python import cp_model
from calibrate import MockerLatency, compare

@dataclass
class Request:
    isl: int
    osl: int
    # Ignored in build_model for a request with a predecessor
    arrival: int
    hash_ids: list[int]
    # Previous request in the same session: this one is released think timesteps after it ends
    prev: int | None = None
    think: int = 0

default_latency = MockerLatency.load("mocker_params.json")

def from_mooncake(path, n=None, ms_per_step=default_latency.decode_ms):
    with open(path) as f:
        rows = [json.loads(line) for line in islice(f, n)]
    t0 = rows[0]["timestamp"]
    trace = [Request(r["input_length"], r["output_length"], (r["timestamp"] - t0) // ms_per_step, r["hash_ids"]) for r in rows]
    return trace, 512

def from_weka(path, lanes, n=None, ms_per_step=default_latency.decode_ms, idle_cap_s=300):
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
        shift, prev_t, prev_end = 0.0, 0.0, None
        for r in play["requests"]:
            idle = r["t"] - prev_t
            # Cap the gap between successive requests from a session. This
            # appears to be a standard transformation for Weka and helps
            # decrease the horizon for us.
            if idle > idle_cap_s:
                shift += idle - idle_cap_s
            arrival = round((r["t"] - shift) * 1000) // ms_per_step
            # Think time: from the previous request finishing in Weka to this one arriving
            prev, think = (None, 0) if prev_end is None else (len(trace) - 1, max(0, arrival - prev_end))
            trace.append(Request(r["in"], max(1, r["out"]), arrival, [h + offset for h in r["hash_ids"]], prev, think))
            prev_t = r["t"]
            prev_end = round((r["t"] - shift + r["api_time"]) * 1000) // ms_per_step
        offset += 1 + max(h for r in play["requests"] for h in r["hash_ids"])

    assert len(trace) >= n
    # A request's predecessor sorts before it, so it survives the truncation too
    order = sorted(range(len(trace)), key=lambda i: trace[i].arrival)[:n]
    pos = {i: k for k, i in enumerate(order)}
    trace = [replace(trace[i], prev=None if trace[i].prev is None else pos[trace[i].prev]) for i in order]
    return trace, blk_sizes.pop()

def shared_tokens(u, v, blk_size):
    count = 0
    for b1, b2 in zip(u.hash_ids, v.hash_ids):
        if b1 != b2:
            break
        count += 1
    return min(count * blk_size, u.isl, v.isl)

# Reduction in prefill time of v if it gets a cache hit from u
def saving(u, v, blk_size, prefill_latency=default_latency):
    return prefill_latency(v.isl) - prefill_latency(v.isl, shared_tokens(u, v, blk_size))

def build_model(trace, blk_size, num_caches, mem, horizon, prefill_latency):
    N, C = len(trace), num_caches
    assert mem >= max([u.isl + u.osl for u in trace])
    # Earliest possible release
    lbs = []
    for u in trace:
        lbs.append(u.arrival if u.prev is None else lbs[u.prev] + 1 + trace[u.prev].osl + u.think)
    assert horizon > max(lb + prefill_latency(u.isl) + u.osl for lb, u in zip(lbs, trace))
    for u in trace:
        assert ceil(u.isl / blk_size) == len(u.hash_ids)

    model = cp_model.CpModel()

    routes = [[model.NewBoolVar(f"route[{i},{c}]") for c in range(C)] for i in range(N)]

    # For symmetry breaking
    model.Add(routes[0][0] == True)

    for i, u in enumerate(trace):
        model.AddExactlyOne(routes[i][c] for c in range(C))

    starts = [model.NewIntVar(lbs[i], horizon, f"start[{i}]") for i in range(N)]
    ends = [model.NewIntVar(lbs[i], horizon, f"end[{i}]") for i in range(N)]
    prefill_ends = [model.NewIntVar(lbs[i] + 1, horizon, f"prefill_end[{i}]") for i in range(N)]

    # A session's first request is released at its arrival, and each later one think timesteps after its
    # predecessor ends. TTFT counts from the release.
    releases = [u.arrival if u.prev is None else ends[u.prev] + u.think for u in trace]
    for i in range(N):
        model.Add(starts[i] >= releases[i])

    # reuses[i][j] means j gets the largest possible cache hit from i.
    # TODO: Allow partial hits
    reuses = [[model.NewBoolVar(f"reuse[{i},{j}]") for j in range(N)] for i in range(N)]

    # Don't get a cache hit from yourself
    for i, u in enumerate(trace):
        model.Add(reuses[i][i] == False)

    # To get a cache hit, you must start after your provider's prefill is done
    for i in range(N):
        for j in range(N):
            model.Add(prefill_ends[i] <= starts[j]).OnlyEnforceIf(reuses[i][j])

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
    prefills = [[] for c in range(C)]
    for j, v in enumerate(trace):
        # Full cache hit costs 1 timestep
        prefill = model.NewIntVar(1, prefill_latency(v.isl), f"prefill[{j}]")
        model.Add(prefill == prefill_latency(v.isl) - sum(reuses[i][j] * saving(u, v, blk_size, prefill_latency) for i, u in enumerate(trace)))
        duration = prefill + v.osl
        model.Add(prefill_ends[j] == starts[j] + prefill)
        for c in range(C):
            intervals[c].append(model.NewOptionalIntervalVar(starts[j], duration, ends[j], routes[j][c], f"lifetime[{j},{c}]"))
            sizes[c].append(v.isl + v.osl)
            prefills[c].append(model.NewOptionalIntervalVar(starts[j], prefill, prefill_ends[j], routes[j][c], f"prefill[{j},{c}]"))

    for c in range(C):
        model.AddNoOverlap(prefills[c])

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
        for c in range(C):
            # cond == is_provider AND routes[i][c]
            # It's annoying that CP-SAT doesn't let us write SAT formulas naturally...
            cond = model.NewBoolVar(f"cond[{i},{c}]")
            model.AddBoolOr([is_provider.Not(), routes[i][c].Not(), cond])
            model.AddImplication(cond, is_provider)
            model.AddImplication(cond, routes[i][c])

            intervals[c].append(model.NewOptionalIntervalVar(ends[i], _evict, evict, cond, f"eviction[{i},{c}]"))
            # TODO: This is an overestimate I think
            sizes[c].append(max(shared_tokens(u, v, blk_size) for j, v in enumerate(trace) if j != i))

    for c in range(C):
        model.AddCumulative(intervals[c], sizes[c], mem)

    # waits = []
    # for i, u in enumerate(trace):
    #     wait = model.NewIntVar(0, horizon, f"waits[{i}]")
    #     model.Add(wait == starts[i] - u.arrival)
    #     waits.append(wait)
    # model.minimize(sum(waits))

    ttfts = []
    for i, u in enumerate(trace):
        ttft = model.NewIntVar(0, horizon, f"ttft[{i}]")
        model.Add(ttft == ends[i] - u.osl - releases[i])
        ttfts.append(ttft)
    model.minimize(sum(ttfts))

    return model, dict(routes=routes, starts=starts, ends=ends, reuses=reuses, releases=releases, ttfts=ttfts)

class GapLogger(cp_model.CpSolverSolutionCallback):
    def on_solution_callback(self):
        obj, bound = self.ObjectiveValue(), self.BestObjectiveBound()
        print(f"[{self.WallTime():.1f}s] obj={obj:.0f} bound={bound:.0f} gap={(obj - bound) / max(abs(obj), 1):.1%}")

def solve(model, vars):
    solver = cp_model.CpSolver()
    status = solver.Solve(model, GapLogger())
    print(solver.StatusName(status))

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        ttfts = vars["ttfts"]
        avg_ttft = sum(solver.Value(t) for t in ttfts) / len(ttfts)
        print("Avg TTFT: ", avg_ttft)

        if status != cp_model.OPTIMAL:
            obj, bound = solver.ObjectiveValue(), solver.BestObjectiveBound()
            print(f"Gap: {(obj - bound) / max(abs(obj), 1):.1%}")
    return solver, status

if __name__ == "__main__":
    num_requests = 256
    trace, blk_size = from_weka("traces/weka_256k.jsonl", lanes=8, n=num_requests)
    # trace, blk_size = from_mooncake("traces/mooncake_conversation.jsonl", n=num_requests)
    assert len(trace) == num_requests
    model, vars = build_model(trace, blk_size, num_caches=4, mem=16384*64, horizon=1_000_000, prefill_latency=default_latency)
    print("Built model, beginning solve...")
    solver, status = solve(model, vars)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        compare(trace, blk_size, solver, vars, default_latency)
