"""Rolling-horizon solve of the cache_packing model.

The trace (sorted by arrival) is split into chunks of `chunk_size` requests. Each chunk is solved with CP-SAT
while everything scheduled in earlier chunks stays fixed. The fixed past shows up in a chunk's model in two ways:
  - its cache occupancy (lifetimes and eviction intervals) is a fixed memory profile on each cache, and
  - its prompts can still be reused. A chunk request j can take a hit from past request p only on p's cache, after
    p ends. Doing so extends p's eviction interval (or creates one, if p wasn't a provider yet) up to j's start, and
    the extension has to fit in the fixed profile.
The stitched schedule satisfies every constraint in cache_packing.build_model, so its objective is an upper bound
for that model. check_schedule verifies this independently of the chunk models.
"""
import bisect
import time
from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from cache_packing import default_prefill_latency, from_mooncake, from_weka, saving


class PrefixTrie:
    """Trie over hash_ids. members[node] are the request indices whose prompts pass through node, in increasing
    order (requests are inserted in trace order)."""

    def __init__(self, trace):
        self.children = [{}]
        self.members = [[]]
        self.paths = []
        for i, u in enumerate(trace):
            node, path = 0, []
            for h in u.hash_ids:
                nxt = self.children[node].get(h)
                if nxt is None:
                    nxt = len(self.children)
                    self.children[node][h] = nxt
                    self.children.append({})
                    self.members.append([])
                node = nxt
                self.members[node].append(i)
                path.append(node)
            self.paths.append(path)


def eviction_sizes(trace, blk_size, trie):
    """sizes in build_model's eviction intervals: max over v != u of shared_tokens(u, v), without the O(N^2) loop."""
    # Per node, the two largest isl among members, so each member can find the largest isl of the others
    top2 = []
    for ms in trie.members:
        best = sorted(((trace[i].isl, i) for i in ms), reverse=True)[:2]
        top2.append(best)
    sizes = []
    for i, u in enumerate(trace):
        size = 0
        for d, node in enumerate(trie.paths[i], start=1):
            other = next((isl for isl, k in top2[node] if k != i), None)
            if other is None:
                break
            size = max(size, min(d * blk_size, u.isl, other))
        sizes.append(size)
    return sizes


@dataclass
class Schedule:
    route: list[int]
    start: list[int]
    end: list[int]
    src: list[int]            # provider of each request's cache hit, -1 if none
    evict: list[int]          # eviction time of each provider's prompt, -1 if not a provider
    chunk_stats: list[dict] = field(default_factory=list)


def check_schedule(trace, blk_size, sched, num_caches, mem, horizon, csz, prefill_latency=default_prefill_latency):
    """Check sched against the constraints of build_model and return the sum of TTFTs."""
    N = len(trace)
    consumers = [[] for _ in range(N)]
    for j in range(N):
        assert 0 <= sched.route[j] < num_caches
        assert trace[j].arrival <= sched.start[j] and sched.end[j] <= horizon
        i = sched.src[j]
        sv = 0
        if i >= 0:
            assert i != j and sched.route[i] == sched.route[j] and sched.end[i] <= sched.start[j], j
            sv = saving(trace[i], trace[j], blk_size, prefill_latency)
            consumers[i].append(j)
        assert sched.end[j] - sched.start[j] == prefill_latency(trace[j].isl) - sv + trace[j].osl, j
    events = [[] for _ in range(num_caches)]
    for i, u in enumerate(trace):
        c = sched.route[i]
        events[c] += [(sched.start[i], u.isl + u.osl), (sched.end[i], -(u.isl + u.osl))]
        if consumers[i]:
            ev = sched.evict[i]
            assert ev >= sched.end[i] and all(ev >= sched.start[j] for j in consumers[i]), i
            if ev > sched.end[i]:
                events[c] += [(sched.end[i], csz[i]), (ev, -csz[i])]
    for c in range(num_caches):
        # Releases sort before acquisitions at the same time, matching half-open intervals
        load = 0
        for t, d in sorted(events[c], key=lambda e: (e[0], e[1])):
            load += d
            assert load <= mem, f"cache {c} over capacity at t={t}: {load}"
    return sum(sched.end[i] - u.osl - u.arrival for i, u in enumerate(trace))


def solve_chunked(trace, blk_size, chunk_size, num_caches=4, mem=1_000_000, horizon=100_000,
                  prefill_latency=default_prefill_latency, max_past_providers=8, lookback=None,
                  time_limit=10.0, budget=None, num_workers=8, verbose=True):
    """Solve the trace chunk by chunk, fixing the schedule of earlier chunks.

    max_past_providers: per request, how many earlier-chunk requests to offer as cache hit providers (the ones with
        the largest saving, most recent first). Requests in the same chunk are always all offered.
    lookback: if set, past providers whose prompt would have to be held from before (chunk's first arrival -
        lookback) aren't offered.
    time_limit: CP-SAT time limit per chunk, in seconds.
    budget: if set, a total wall-clock budget in seconds that replaces time_limit. Each chunk gets an equal share of
        what's left, so chunks that finish early leave more time for later ones.
    A chunk that ends without a solution is retried with twice the time limit, up to max_retries times.
    """
    max_retries = 3
    t_total = time.time()
    N, C = len(trace), num_caches
    assert all(u.arrival <= v.arrival for u, v in zip(trace, trace[1:])), "trace must be sorted by arrival"
    assert mem >= max(u.isl + u.osl for u in trace)

    trie = PrefixTrie(trace)
    csz = eviction_sizes(trace, blk_size, trie)
    pf = [prefill_latency(u.isl) for u in trace]
    D0 = [pf[j] + u.osl for j, u in enumerate(trace)]

    sched = Schedule([-1] * N, [-1] * N, [-1] * N, [-1] * N, [-1] * N)
    # Items placed on each cache so far: (lo, hi, size, owner, kind), kind 0 = lifetime, 1 = eviction interval
    placed = [[] for _ in range(C)]

    for k0 in range(0, N, chunk_size):
        t_chunk = time.time()
        chunk = list(range(k0, min(k0 + chunk_size, N)))
        in_chunk = set(chunk)
        arr0 = trace[chunk[0]].arrival

        # Candidate providers per chunk request. Savings are monotone in shared depth, so walk the trie path
        # deepest first.
        cands = {j: [] for j in chunk}
        for j in chunk:
            v = trace[j]
            seen = set()
            past = []
            for d in range(len(trie.paths[j]), 0, -1):
                if pf[j] - prefill_latency(v.isl - min(d * blk_size, v.isl)) <= 0:
                    break
                ms = trie.members[trie.paths[j][d - 1]]
                if len(ms) < 2:
                    continue
                lo = bisect.bisect_left(ms, k0)
                for i in ms[lo:bisect.bisect_left(ms, chunk[-1] + 1)]:
                    if i != j and i not in seen:
                        seen.add(i)
                        sv = saving(trace[i], v, blk_size, prefill_latency)
                        if sv > 0:
                            cands[j].append((i, sv))
                for idx in range(lo - 1, -1, -1):
                    if len(past) >= max_past_providers:
                        break
                    i = ms[idx]
                    if i in seen:
                        continue
                    seen.add(i)
                    base = sched.evict[i] if sched.evict[i] >= 0 else sched.end[i]
                    if lookback is not None and base < arr0 - lookback:
                        continue
                    sv = saving(trace[i], v, blk_size, prefill_latency)
                    if sv > 0:
                        past.append((i, sv))
            cands[j] += past

        past_srcs = {i for j in chunk for i, _ in cands[j] if i not in in_chunk}
        base = {p: (sched.evict[p] if sched.evict[p] >= 0 else sched.end[p]) for p in past_srcs}
        t_lo = min([arr0] + list(base.values()))
        last_fixed = max([hi for c in range(C) for _, hi, *_ in placed[c]], default=0)
        # A schedule that runs the chunk one request at a time after everything fixed is gone always fits
        H = min(horizon, max(trace[chunk[-1]].arrival, last_fixed) + sum(D0[j] for j in chunk))

        model = cp_model.CpModel()
        intervals = [[] for _ in range(C)]
        demands = [[] for _ in range(C)]

        # Fixed profile from earlier chunks, clipped to [t_lo, inf)
        for c in range(C):
            steps = {}
            for lo, hi, size, *_ in placed[c]:
                lo = max(lo, t_lo)
                if hi > lo:
                    steps[lo] = steps.get(lo, 0) + size
                    steps[hi] = steps.get(hi, 0) - size
            load, ts = 0, sorted(steps)
            for a, b in zip(ts, ts[1:]):
                load += steps[a]
                if load > 0:
                    intervals[c].append(model.NewFixedSizeIntervalVar(a, b - a, f"fixed[{c},{a}]"))
                    demands[c].append(load)

        routes = {j: [model.NewBoolVar(f"route[{j},{c}]") for c in range(C)] for j in chunk}
        starts = {j: model.NewIntVar(trace[j].arrival, H, f"start[{j}]") for j in chunk}
        ends = {j: model.NewIntVar(trace[j].arrival, H, f"end[{j}]") for j in chunk}
        for j in chunk:
            model.AddExactlyOne(routes[j])
        if k0 == 0:
            # For symmetry breaking
            model.Add(routes[0][0] == True)

        reuses = {}
        for j in chunk:
            for i, sv in cands[j]:
                x = reuses[i, j] = model.NewBoolVar(f"reuse[{i},{j}]")
                if i in in_chunk:
                    model.Add(ends[i] <= starts[j]).OnlyEnforceIf(x)
                    for c in range(C):
                        model.AddBoolOr([x.Not(), routes[j][c].Not(), routes[i][c]])
                else:
                    model.Add(starts[j] >= sched.end[i]).OnlyEnforceIf(x)
                    model.AddImplication(x, routes[j][sched.route[i]])
            model.AddAtMostOne(reuses[i, j] for i, _ in cands[j])

        consumers = {}
        for (i, j) in reuses:
            consumers.setdefault(i, []).append(j)

        prefills = {}
        for j in chunk:
            v = trace[j]
            prefill = prefills[j] = model.NewIntVar(1, pf[j], f"prefill[{j}]")
            model.Add(prefill == pf[j] - sum(reuses[i, j] * sv for i, sv in cands[j]))
            for c in range(C):
                intervals[c].append(model.NewOptionalIntervalVar(starts[j], prefill + v.osl, ends[j], routes[j][c],
                                                                 f"lifetime[{j},{c}]"))
                demands[c].append(v.isl + v.osl)

        # Eviction intervals of chunk providers
        for i in chunk:
            if i not in consumers:
                continue
            evict = model.NewIntVar(0, H, f"evict[{i}]")
            model.Add(evict >= ends[i])
            is_provider = model.NewBoolVar(f"provider[{i}]")
            for j in consumers[i]:
                model.Add(evict >= starts[j]).OnlyEnforceIf(reuses[i, j])
                model.AddImplication(reuses[i, j], is_provider)
            _evict = model.NewIntVar(0, H, f"_evict[{i}]")
            for c in range(C):
                cond = model.NewBoolVar(f"cond[{i},{c}]")
                model.AddBoolOr([is_provider.Not(), routes[i][c].Not(), cond])
                model.AddImplication(cond, is_provider)
                model.AddImplication(cond, routes[i][c])
                intervals[c].append(model.NewOptionalIntervalVar(ends[i], _evict, evict, cond, f"eviction[{i},{c}]"))
                demands[c].append(csz[i])

        # Extensions of past providers' eviction intervals, from where they currently end to the last reuse
        for p in past_srcs:
            ext = model.NewBoolVar(f"extend[{p}]")
            evict = model.NewIntVar(base[p], max(base[p], H), f"evict[{p}]")
            for j in consumers[p]:
                model.Add(evict >= starts[j]).OnlyEnforceIf(reuses[p, j])
                model.AddImplication(reuses[p, j], ext)
            _evict = model.NewIntVar(0, max(0, H - base[p]), f"_evict[{p}]")
            c = sched.route[p]
            intervals[c].append(model.NewOptionalIntervalVar(base[p], _evict, evict, ext, f"extension[{p}]"))
            demands[c].append(csz[p])

        for c in range(C):
            model.AddCumulative(intervals[c], demands[c], mem)

        ttfts = [ends[j] - trace[j].osl - trace[j].arrival for j in chunk]
        model.Minimize(sum(ttfts))

        limit = time_limit
        if budget is not None:
            chunks_left = -(-(N - k0) // chunk_size)
            # Keep some slack for building the remaining models
            limit = max(0.5, 0.9 * (budget - (time.time() - t_total)) / chunks_left)
        for attempt in range(max_retries + 1):
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = limit
            solver.parameters.num_workers = num_workers
            status = solver.Solve(model)
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                break
            if status != cp_model.UNKNOWN or attempt == max_retries:
                raise RuntimeError(f"chunk at request {k0}: {solver.StatusName(status)}")
            if verbose:
                print(f"chunk at request {k0}: no solution in {limit:.1f}s, retrying with {2 * limit:.1f}s", flush=True)
            limit *= 2

        # Commit the chunk
        for j in chunk:
            sched.route[j] = next(c for c in range(C) if solver.Value(routes[j][c]))
            sched.start[j] = solver.Value(starts[j])
            sched.end[j] = solver.Value(ends[j])
            sched.src[j] = next((i for i, _ in cands[j] if solver.Value(reuses[i, j])), -1)
            placed[sched.route[j]].append((sched.start[j], sched.end[j], trace[j].isl + trace[j].osl, j, 0))
        # Evict each provider's prompt as soon as its last consumer starts, rather than at the solver's evict value
        last_use = {}
        for j in chunk:
            i = sched.src[j]
            if i >= 0:
                last_use[i] = max(last_use.get(i, -1), sched.start[j])
        for i, t in last_use.items():
            c = sched.route[i]
            if i in in_chunk:
                ev = max(sched.end[i], t)
                placed[c].append((sched.end[i], ev, csz[i], i, 1))
            else:
                ev = max(base[i], t)
                placed[c].append((base[i], ev, csz[i], i, 1))
            sched.evict[i] = ev

        # Drop items that can no longer intersect a later chunk's window. A later chunk only looks back to its first
        # arrival or a past provider's eviction time, and with lookback set the latter is bounded.
        if lookback is not None:
            cutoff = trace[min(k0 + chunk_size, N - 1)].arrival - lookback
            placed = [[it for it in pl if it[1] > cutoff] for pl in placed]

        obj = sum(sched.end[j] - trace[j].osl - trace[j].arrival for j in chunk)
        stats = dict(first=k0, n=len(chunk), status=solver.StatusName(status), obj=obj,
                     bound=solver.BestObjectiveBound(), hits=sum(sched.src[j] >= 0 for j in chunk),
                     past_hits=sum(sched.src[j] >= 0 and sched.src[j] not in in_chunk for j in chunk),
                     wall=time.time() - t_chunk)
        sched.chunk_stats.append(stats)
        if verbose:
            print(f"[{k0 + len(chunk)}/{N}] {stats['status']:8s} avg ttft {obj / len(chunk):8.1f} "
                  f"(chunk bound {stats['bound'] / len(chunk):8.1f}) hits {stats['hits']}/{len(chunk)} "
                  f"({stats['past_hits']} from past) {stats['wall']:.1f}s, total {time.time() - t_total:.0f}s",
                  flush=True)

    total = check_schedule(trace, blk_size, sched, C, mem, horizon, csz, prefill_latency)
    if verbose:
        print(f"Feasible schedule for all {N} requests, avg TTFT {total / N:.2f}")
    return sched, total


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="traces/mooncake_conversation.jsonl")
    ap.add_argument("-n", type=int, default=None, help="number of requests (default: whole trace)")
    ap.add_argument("--chunk-size", type=int, default=100)
    ap.add_argument("--caches", type=int, default=4)
    ap.add_argument("--mem", type=int, default=1_000_000)
    ap.add_argument("--horizon", type=int, default=200_000)
    ap.add_argument("--max-past-providers", type=int, default=8)
    ap.add_argument("--lookback", type=int, default=None)
    ap.add_argument("--budget", type=float, default=600.0, help="total wall-clock budget in seconds")
    ap.add_argument("--time-limit", type=float, default=None, help="fixed per-chunk time limit, instead of --budget")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default="schedule.json", help="write the schedule to this JSON file")
    args = ap.parse_args()

    if "weka" in args.trace:
        trace, blk_size = from_weka(args.trace, lanes=8, n=args.n)
    else:
        trace, blk_size = from_mooncake(args.trace, n=args.n)
    sched, total = solve_chunked(trace, blk_size, args.chunk_size, num_caches=args.caches, mem=args.mem,
                                 horizon=args.horizon, max_past_providers=args.max_past_providers,
                                 lookback=args.lookback, num_workers=args.workers,
                                 **(dict(time_limit=args.time_limit) if args.time_limit else dict(budget=args.budget)))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(args=vars(args), total_ttft=total, avg_ttft=total / len(trace), route=sched.route,
                           start=sched.start, end=sched.end, src=sched.src, evict=sched.evict,
                           chunks=sched.chunk_stats), f)
