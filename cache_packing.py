import json
from dataclasses import dataclass
from itertools import islice
from math import ceil
from ortools.sat.python import cp_model
from calibrate import MockerLatency

@dataclass
class Request:
    isl: int
    osl: int
    arrival: int
    hash_ids: list[int]

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
        shift, prev_t = 0.0, 0.0
        for r in play["requests"]:
            idle = r["t"] - prev_t
            # Cap the gap between successive requests from a session. This
            # appears to be a standard transformation for Weka and helps
            # decrease the horizon for us.
            if idle > idle_cap_s:
                shift += idle - idle_cap_s
            arrival = round((r["t"] - shift) * 1000) // ms_per_step
            trace.append(Request(r["in"], r["out"], arrival, [h + offset for h in r["hash_ids"]]))
            prev_t = r["t"]
        offset += 1 + max(h for r in play["requests"] for h in r["hash_ids"])

    assert len(trace) >= n
    trace.sort(key=lambda u: u.arrival)
    return trace[:n], blk_sizes.pop()

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

# Steps of network time to ship a prompt of `isl` tokens to its decode worker
def transfer_steps(isl, transfer_bw):
    return 0 if not transfer_bw else -(-isl // transfer_bw)

def build_model(trace, blk_size, num_caches, mem, horizon, prefill_latency,
                num_prefill=None, num_decode=None, mem_prefill=None, mem_decode=None,
                transfer_bw=0, makespan_weight=1):
    N, C = len(trace), num_caches
    # Prefill and decode pools are sized independently. The defaults reproduce the
    # monolithic configuration: one prefill and one decode worker per cache slot,
    # sharing the single memory budget.
    P = C if num_prefill is None else num_prefill
    D = C if num_decode is None else num_decode
    mem_p = mem if mem_prefill is None else mem_prefill
    mem_d = mem if mem_decode is None else mem_decode
    assert mem_p >= max(u.isl for u in trace)
    assert mem_d >= max([u.isl + u.osl for u in trace])
    assert horizon > sorted(map(lambda u: prefill_latency(u.isl) + u.osl + u.arrival +
                                transfer_steps(u.isl, transfer_bw), trace))[-1]
    for u in trace:
        assert ceil(u.isl / blk_size) == len(u.hash_ids)

    model = cp_model.CpModel()

    # Prefill routing decides who computes the prompt, and therefore whose KV cache
    # can be reused by whom.
    proute = [[model.NewBoolVar(f"proute[{i},{p}]") for p in range(P)] for i in range(N)]
    # Decode routing only decides who pays the request's KV arena.
    droute = [[model.NewBoolVar(f"droute[{i},{d}]") for d in range(D)] for i in range(N)]

    # For symmetry breaking
    model.Add(proute[0][0] == True)
    model.Add(droute[0][0] == True)

    for i, u in enumerate(trace):
        model.AddExactlyOne(proute[i][p] for p in range(P))
        model.AddExactlyOne(droute[i][d] for d in range(D))

    pstarts = [model.NewIntVar(u.arrival, horizon, f"pstart[{i}]") for i, u in enumerate(trace)]
    prends = [model.NewIntVar(u.arrival, horizon, f"prend[{i}]") for i, u in enumerate(trace)]
    dstarts = [model.NewIntVar(u.arrival, horizon, f"dstart[{i}]") for i, u in enumerate(trace)]
    dends = [model.NewIntVar(u.arrival, horizon, f"dend[{i}]") for i, u in enumerate(trace)]

    # reuses[i][j] means j gets the largest possible cache hit from i.
    # TODO: Allow partial hits
    reuses = [[model.NewBoolVar(f"reuse[{i},{j}]") for j in range(N)] for i in range(N)]

    # Don't get a cache hit from yourself
    for i, u in enumerate(trace):
        model.Add(reuses[i][i] == False)

    # To get a cache hit, your provider's prefill must have finished. Its decode runs
    # on a different worker, so the provider's generation no longer blocks you.
    for i in range(N):
        for j in range(N):
            model.Add(prends[i] <= pstarts[j]).OnlyEnforceIf(reuses[i][j])

    # You may only get a cache hit from one provider
    for j in range(N):
        model.AddAtMostOne(reuses[i][j] for i in range(N))

    # If i provides j a cache hit and j is prefilled on p, then i must be too
    for i in range(N):
        for j in range(N):
            for p in range(P):
                model.AddBoolOr([reuses[i][j].Not(), proute[j][p].Not(), proute[i][p]])

    # Prefill pool: prompt KV while prefilling, plus retained prefixes for reuse.
    p_intervals = [[] for p in range(P)]
    p_sizes = [[] for p in range(P)]
    prefills = [[] for p in range(P)]
    # Decode pool: each request's whole KV arena, from its first decode to completion.
    d_intervals = [[] for d in range(D)]
    d_sizes = [[] for d in range(D)]
    for j, v in enumerate(trace):
        # Uncached prefill cost minus whatever hit is claimed
        raw_prefill = model.NewIntVar(0, prefill_latency(v.isl), f"raw_prefill[{j}]")
        model.Add(raw_prefill == prefill_latency(v.isl) - sum(reuses[i][j] * saving(u, v, blk_size, prefill_latency) for i, u in enumerate(trace)))
        # Full cache hit costs 1 timestep, so floor rather than constrain
        prefill = model.NewIntVar(1, prefill_latency(v.isl), f"prefill[{j}]")
        model.AddMaxEquality(prefill, [raw_prefill, model.NewConstant(1)])
        # Shipping the untouched prompt KV to the decode worker costs ceil(isl / transfer_bw)
        # steps. It is paid in full even on a cache hit: the hit saves prefill compute,
        # not the bytes that must land on the decode worker.
        tfer = transfer_steps(v.isl, transfer_bw)
        model.Add(prends[j] == pstarts[j] + prefill)
        # The KV cache must have landed on the decode worker before it can decode
        model.Add(dstarts[j] >= prends[j] + tfer)
        model.Add(dends[j] == dstarts[j] + v.osl)
        for p in range(P):
            # Prefill compute: the prompt occupies the worker for exactly the prefill
            prompt = model.NewOptionalIntervalVar(pstarts[j], prefill, prends[j], proute[j][p], f"prompt[{j},{p}]")
            prefills[p].append(prompt)
            # The prompt stays resident through the transfer, which is not compute
            prompt_mem = model.NewOptionalIntervalVar(pstarts[j], prefill + tfer, prends[j] + tfer, proute[j][p], f"prompt_mem[{j},{p}]")
            p_intervals[p].append(prompt_mem)
            p_sizes[p].append(v.isl)
        for d in range(D):
            # Held for the decode duration, but it pins the whole isl+osl arena
            d_intervals[d].append(model.NewOptionalIntervalVar(dstarts[j], v.osl, dends[j], droute[j][d], f"arena[{j},{d}]"))
            d_sizes[d].append(v.isl + v.osl)

    for p in range(P):
        model.AddNoOverlap(prefills[p])

    for i, u in enumerate(trace):
        # Time at which u's prompt stops occupying memory as a whole and, if it is a
        # provider, survives only as a retained prefix. That is after its own transfer.
        release = prends[i] + transfer_steps(u.isl, transfer_bw)
        evict = model.NewIntVar(0, horizon, f"evict[{i}]")
        model.Add(evict >= release)

        # Is u providing a cache hit to anybody?
        is_provider = model.NewBoolVar(f"provider[{i}]")

        for j, v in enumerate(trace):
            # If v is getting a cache hit from u, u's prompt must not be evicted until v's prefill is done.
            # Note the intervals below are half-open, so `evict >= starts[j]` would let u's prompt be freed
            # at the instant v begins prefilling, counting zero of the retained bytes.
            model.Add(evict >= prends[j]).OnlyEnforceIf(reuses[i][j])
            # ... and u is a provider
            model.AddImplication(reuses[i][j], is_provider)

        _evict = model.NewIntVar(0, horizon, f"_evict[{i}]")
        for p in range(P):
            # cond == is_provider AND proute[i][p]
            # It's annoying that CP-SAT doesn't let us write SAT formulas naturally...
            cond = model.NewBoolVar(f"cond[{i},{p}]")
            model.AddBoolOr([is_provider.Not(), proute[i][p].Not(), cond])
            model.AddImplication(cond, is_provider)
            model.AddImplication(cond, proute[i][p])

            p_intervals[p].append(model.NewOptionalIntervalVar(release, _evict, evict, cond, f"eviction[{i},{p}]"))
            # TODO: This is an overestimate I think
            p_sizes[p].append(max(shared_tokens(u, v, blk_size) for j, v in enumerate(trace) if j != i))

    for p in range(P):
        model.AddCumulative(p_intervals[p], p_sizes[p], mem_p)
    for d in range(D):
        model.AddCumulative(d_intervals[d], d_sizes[d], mem_d)

    # waits = []
    # for i, u in enumerate(trace):
    #     wait = model.NewIntVar(0, horizon, f"waits[{i}]")
    #     model.Add(wait == starts[i] - u.arrival)
    #     waits.append(wait)
    # model.minimize(sum(waits))

    # TTFT is the prefill-side metric: arrival to the prompt being ready. It stays
    # exposed because the runtime comparison consumes it.
    ttfts = []
    # Latency is the decode-side metric: arrival to the last token. Under PD the first
    # token lands only after the transfer, so this is what the objective scores.
    latencies = []
    for i, u in enumerate(trace):
        ttft = model.NewIntVar(0, horizon, f"ttft[{i}]")
        model.Add(ttft == prends[i] - u.arrival)
        ttfts.append(ttft)

        latency = model.NewIntVar(0, horizon, f"latency[{i}]")
        model.Add(latency == dends[i] - u.arrival)
        latencies.append(latency)

    # Nothing else rewards finishing a decode early, so the solver is free to park
    # generations at the end of the horizon. Penalise the makespan to prevent that.
    makespan = model.NewIntVar(0, horizon, "makespan")
    for i in range(N):
        model.Add(makespan >= dends[i])

    # Minimise N * (average latency + makespan_weight * makespan), kept integral
    model.minimize(sum(latencies) + N * makespan_weight * makespan)

    # routes/starts/ends are the prefill-side aliases that the runtime comparison
    # consumes; the PD solve keys are exposed alongside them.
    return model, dict(routes=proute, starts=pstarts, ends=prends,
                       reuses=reuses, ttfts=ttfts,
                       proute=proute, droute=droute,
                       pstarts=pstarts, prends=prends,
                       dstarts=dstarts, dends=dends,
                       latencies=latencies, makespan=makespan)

class GapLogger(cp_model.CpSolverSolutionCallback):
    def on_solution_callback(self):
        obj, bound = self.ObjectiveValue(), self.BestObjectiveBound()
        print(f"[{self.WallTime():.1f}s] obj={obj:.0f} bound={bound:.0f} gap={(obj - bound) / max(abs(obj), 1):.1%}")

def solve(model, vars):
    solver = cp_model.CpSolver()
    # Cap the run so it always ends with a bound to report
    solver.parameters.max_time_in_seconds = 5 * 60
    status = solver.Solve(model, GapLogger())
    print(solver.StatusName(status))

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        ttfts = vars["ttfts"]
        latencies = vars["latencies"]
        print("Avg TTFT: ", sum(solver.Value(t) for t in ttfts) / len(ttfts))
        print("Avg latency: ", sum(solver.Value(t) for t in latencies) / len(latencies))
        print("Makespan: ", solver.Value(vars["makespan"]))

        if status != cp_model.OPTIMAL:
            obj, bound = solver.ObjectiveValue(), solver.BestObjectiveBound()
            print(f"Gap: {(obj - bound) / max(abs(obj), 1):.1%}")
    return solver, status

if __name__ == "__main__":
    num_requests = 256
    trace, blk_size = from_mooncake("traces/mooncake_conversation.jsonl", n=num_requests)
    # trace, blk_size = from_weka("traces/weka_256k.jsonl", lanes=8, n=num_requests)
    assert len(trace) == num_requests
    model, vars = build_model(trace, blk_size, num_caches=4, mem=16384*64, horizon=1_000_000, prefill_latency=default_latency)
    print("Built model, beginning solve...")
    solver, status = solve(model, vars)
    # compare() replays the schedule through the Mocker, which needs ai-dynamo/aisimulate.
    # The pinned aisimulate-core version is no longer on crates.io, so it cannot be built;
    # the model is validated structurally until the runtime check is reinstated.
    # if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    #     compare(trace, blk_size, solver, vars, default_latency)
