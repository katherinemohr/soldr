# Cache packing with prefill/decode interference modelling.
import heapq
import json
from bisect import bisect_left
from dataclasses import dataclass, field
from math import ceil
import numpy as np
from ortools.sat.python import cp_model
from cache_packing import Request, shared_tokens

# Coefficients are scaled by S and rounded to integers; each scaled sum is then matched to within S - 1
S = 100

# finds the intercept of x on a piecewise linear function defined by xs and ys
def interp(xs, ys, x):
    # Piecewise linear, extrapolating linearly at both ends
    i = min(max(bisect_left(xs, x), 1), len(xs) - 1)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


# Step-level latency model of Mocker's engine, from steps_<config>.json. decode_f is f(B) at the measured batch
# sizes and decode_k the cost per KV token of the decoding requests; prefill_g is g(n) at the measured chunk sizes and
# prefill_b the attention cost. Fitted by least squares on Mocker runs: decode-only batches (B up to 128, contexts up
# to 128k, max error 0.5 ms) and single prefills (up to 8192 new tokens after 0-98k cached, max error 0.2 ms).
class StepLatency:
    def __init__(self, config):
        with open(f"steps_{config}.json") as f:
            p = json.load(f)
        self.chunk, self.k, self.b = p["chunk"], p["decode_k"], p["prefill_b"]
        self.Bs, self.fB, self.ns, self.gn = p["decode_B"], p["decode_f"], p["prefill_n"], p["prefill_g"]
        # Assuming we are running at most B decodes at once
        # f(B) is linear enough up to B = 8: f(B) ~ f1 + s (B - 1)
        Bs = [B for B in self.Bs if B <= 8]
        self.s, intercept = np.polyfit(Bs, [interp(self.Bs, self.fB, B) for B in Bs], 1)
        self.f1 = intercept + self.s
        # self.f1 is cost of a decode running alone
        # self.s is the interference cost per additional decode running concurrently
        self._prefill = {}

    def prefill_ms(self, isl, hit=0):
        # Prefill alone on the engine: chunk steps of up to `chunk` new tokens each
        key = (isl, hit)
        if key not in self._prefill:
            t, done = 0.0, hit
            while done < isl:
                n = min(isl - done, self.chunk)
                # first part is latency for ffn layers, second part is for attention
                t += interp(self.ns, self.gn, n) + self.b * n * (done + n / 2)
                done += n
            self._prefill[key] = t
        return self._prefill[key]

    def __call__(self, isl, hit, blk_size):
        # As in Mocker: hits are block-aligned, and at least one token is computed
        return max(1, round(self.prefill_ms(isl, min(hit, isl - 1) // blk_size * blk_size)))


default_steps = StepLatency("qwen3-30b-a3b_h100x4")


# Context contains the information about the requests that the solver has
# already fully or partially solved.
# Every request can be in one of the four states:
#   1) uncommitted - all the decisions for the request are yet to be made
#   2) committed - cache and cache-hit source for the request is fixed, but timing is still being solved
#   3) started - cache, hit source and start timing for the request are fixed.
#   4) finished - all decisions are final, and the request is not solved in the current window anymore.
@dataclass
class Context:
    # finished_hit_sources[j] = (src, prefill_end, route, end, tokens, size)
    # A request j in the current window "can" get a cache hit of `tokens` from this finished request `src` on cache `route`; 
    #  `src`'s prefix of `size` tokens stays in memory from src's end until j starts.
    finished_hit_sources: dict = field(default_factory=dict)
    # (route, start, prefill_end) of finished requests' prefills that may still overlap with the current window.
    finished_prefills: list = field(default_factory=list)
    # (route, start, end, size) of finished requests' KV memory that may still overlap the current window.
    finished_kv_lifetimes: list = field(default_factory=list)
    # finished_interference[j] = (decode, prefill_slowdown, any)
    #   for a started request j: the interference from finished requests it overlaps with is carried over here. 
    #       decode  interference time added to j's decode
    #       prefill_slowdown  interference time added to each of j's prefill chunks
    #       any     whether any of them was decoding during j's prefill (forces j's "any decode active" on)
    # decode and prefill_slowdown are scaled by S like the model's terms.
    finished_interference: dict = field(default_factory=dict)
    # cache_free_at[j] = {cache: time}: a request j may only start on a particular `cache` after `time`
    cache_free_at: dict = field(default_factory=dict)
    # committed_route_and_hit[j] = (cache, src)
    #   a request j that is already commited to `cache` and with `src` as the cache provider.
    #   `src` can be a 
    #       1) request index in the current window
    #       2) "ext" - when the cache hit provider is a finished request (given by finished_hit_sources[j])
    #       3) None - when there was no cache hit provider (cache miss)
    committed_route_and_hit: dict = field(default_factory=dict)
    # committed_droute[j] = cache: the cache a committed request j decodes on 
    #   This is used when the 'topology' is split, so the solver can decide where to place
    #   both the prefill and decode of a request separately.
    committed_droute: dict = field(default_factory=dict)
    # fixed_start[j]: for a request j that is in `started` state, fixed_start[j] is its start time
    fixed_start: dict = field(default_factory=dict)


#  per-request constants used to calculate the interference and transfer times that affect the prefill
#  decode times, one list entry per request.
# `steps` is the StepLatency model
#  maxhit is each request's largest possible cache hit (max_hits).
def constants(trace, blk_size, steps, maxhit):
    # Average KV tokens a request holds while decoding (its prompt plus half its output), so its per-step cost is a
    # constant instead of growing with each token
    ctx = [u.isl + u.osl / 2 for u in trace]
    c = {}
    # Decode steps: number of decode steps. 
    # The decode takes decode_steps * t0 ms when there is no interference.
    c["decode_steps"] = [u.osl - 1 for u in trace]
    # t0: time to run one decode step when there is no interference
    c["t0"] = [steps.f1 + steps.k * x for x in ctx]
    # delta: The amount of interference this request adds when its decode is running at 
    # the same time as some other request.
    c["delta"] = [steps.s + steps.k * x for x in ctx]
    # chunks: number of chunk steps in its prefill, assuming its largest possible
    # hit, since the hit is a decision. Undercounts when the solver skips the hit
    c["chunks"] = [ceil((v.isl - min(h, v.isl - 1)) / steps.chunk) for v, h in zip(trace, maxhit)]
    # chunk_ms: average ms of the prefill part of one of its chunk steps, without a hit (again since the hit is a
    # decision). 
    c["chunk_ms"] = [steps(v.isl, 0, blk_size) / ceil(v.isl / steps.chunk) for v in trace]
    # kv_cache transfer time from one worker to another for this request
    c["kv_transfer_time"] = [transfer_ms(v.isl, t0) for v, t0 in zip(trace, c["t0"])]
    return c


# Constants for KV transfer calculations
KV_BYTES_PER_TOKEN = 2 * 48 * 4 * 128 * 2       # Qwen3-30B-A3B: 48 layers x 4 KV heads x 128 dims x (K, V) x bf16
KV_BANDWIDTH_GBPS = 450                         # Mocker's default (H100 NVLink)


def transfer_ms(isl, t0):
    return ceil(isl * KV_BYTES_PER_TOKEN / (KV_BANDWIDTH_GBPS * 1e9) * 1e3 + t0)


def max_hits(trace, shared, ctx):
    return [max([shared[i][j] for i in range(len(trace))] + ([ctx.finished_hit_sources[j][4]] if j in ctx.finished_hit_sources else []))
            for j in range(len(trace))]


# objective is "e2e" (mean end - release) or "ttft" (mean TTFT). 
# ctx: a Context when modelling one window of a longer trace (see solve_rolling).
# topology "colocated" (default) runs each request's prefill and decode on one worker, as before.
# "split" lets each request decode on a different worker than it prefills on 
# pools = (prefill workers, decode workers)
# restricts which workers each phase may use (e.g. for 2P2D: ([0, 1], [2, 3])); 
# None allows the solver to freely assign prefill decode to any workers.
def build_model(trace, blk_size, num_caches, mem, horizon, steps=default_steps, objective="e2e", ctx=None,
                topology="colocated", pools=None):
    N, C = len(trace), num_caches
    first_window = ctx is None
    ctx = ctx or Context()
    assert topology in ("colocated", "split") and (pools is None or topology == "split")
    colocated = topology == "colocated"
    assert mem >= max([u.isl + u.osl for u in trace])
    for u in trace:
        assert ceil(u.isl / blk_size) == len(u.hash_ids)

    model = cp_model.CpModel()

    # routes[i][c] = whether request i is routed to cache c (its prefill runs there)
    routes = [[model.NewBoolVar(f"route[{i},{c}]") for c in range(C)] for i in range(N)]
    # Symmetry breaking for first C requests (workers are interchangeable unless pools tell them apart)
    for j in range(min(N, C) if first_window and pools is None else 0):
        for c in range(j + 1, C):
            model.Add(routes[j][c] == False)

    # Every request gets routed to exactly 1 cache
    for i in range(N):
        model.AddExactlyOne(routes[i])

    # droutes[i][c] = whether request i decodes on cache c; for colocated this is same as routes[i][c].
    # splits[i] = whether i decodes on another cache than it prefills on (0 colocated)
    droutes, splits = routes, [0] * N
    if not colocated:
        droutes = [[model.NewBoolVar(f"droute[{i},{c}]") for c in range(C)] for i in range(N)]
        for i in range(N):
            model.AddExactlyOne(droutes[i])
            split = splits[i] = model.NewBoolVar(f"split[{i}]")
            for c in range(C):
                model.Add(routes[i][c] == droutes[i][c]).OnlyEnforceIf(split.Not())
                model.Add(routes[i][c] + droutes[i][c] <= 1).OnlyEnforceIf(split)
            for c in range(C):
                # if prefill and decode pools are fixed, restrict to those
                if pools is not None and c not in pools[0]:
                    model.Add(routes[i][c] == 0)
                if pools is not None and c not in pools[1]:
                    model.Add(droutes[i][c] == 0)

    # Start and end times of a request
    starts = [model.NewIntVar(0, horizon, f"start[{i}]") for i in range(N)]
    ends = [model.NewIntVar(0, horizon, f"end[{i}]") for i in range(N)]

    # prefill end time of a request
    prefill_ends = [model.NewIntVar(1, horizon, f"prefill_end[{i}]") for i in range(N)]

    # release time of a request is either when it arrives or think time after its parent request has ended
    releases = [u.arrival if u.prev is None else ends[u.prev] + u.think for u in trace]
    for i in range(N):
        model.Add(starts[i] >= releases[i])

    shared = [[shared_tokens(u, v, blk_size) if i != j else 0 for j, v in enumerate(trace)] for i, u in enumerate(trace)]

    # pair of requests i,j which share some tokens
    pairs = [(i, j) for i in range(N) for j in range(N) if shared[i][j] > 0]


    reuses = [[model.NewConstant(0)] * N for _ in range(N)]
    for i, j in pairs:
        # whether request j reuses tokens (gets a cache hit) from request i
        reuses[i][j] = model.NewBoolVar(f"reuse[{i},{j}]")
        # if j gets a cache hit from i, i's prefill must end before j starts
        model.Add(prefill_ends[i] <= starts[j]).OnlyEnforceIf(reuses[i][j])
        # if j gets a cache hit from i, both need to be routed to the same worker
        for c in range(C):
            model.AddBoolOr([reuses[i][j].Not(), routes[j][c].Not(), routes[i][c]])
    
    # hits from finished requests outside the current window of requests
    ext_reuses = {}
    # One possible source per request: ctx.finished_hit_sources
    for j, (key, pe, route, _, _, _) in ctx.finished_hit_sources.items():
        x = ext_reuses[j] = model.NewBoolVar(f"ext_reuse[{key},{j}]")
        model.AddImplication(x, routes[j][route])
        model.Add(starts[j] >= pe).OnlyEnforceIf(x)

    # request j gets a cache hit from atmost one other request
    for j in range(N):
        model.AddAtMostOne([reuses[i][j] for i in range(N)] + ([ext_reuses[j]] if j in ext_reuses else []))


    P = lambda isl, hit=0: steps(isl, hit, blk_size)
    q = constants(trace, blk_size, steps, max_hits(trace, shared, ctx))

    prefill_lens, prefill_slowdowns, decodes = [], [], []
    intervals, sizes, prefills = [[] for _ in range(C)], [[] for _ in range(C)], [[] for _ in range(C)]
    # decode_starts[j]: time at which j's decode starts. 
    #   if split it starts after the KV transfer 
    #   if colocated, it starts after the prefill end
    decode_starts = prefill_ends if colocated else \
        [model.NewIntVar(1, horizon, f"decode_start[{i}]") for i in range(N)]
    for j, v in enumerate(trace):
        # prefill latency
        prefill = model.NewIntVar(1, P(v.isl), f"prefill[{j}]")
        # saving is the prefill latency saved due to a cache hit
        saving = sum(reuses[i][j] * (P(v.isl) - P(v.isl, shared[i][j])) for i in range(N) if shared[i][j] > 0)
        if j in ext_reuses:
            saving += ext_reuses[j] * (P(v.isl) - P(v.isl, ctx.finished_hit_sources[j][4]))
        model.Add(prefill == P(v.isl) - saving)

        # prefill_slowdown: how much decodes on its cache lengthen its prefill (D->P interference)
        prefill_slowdown = model.NewIntVar(0, horizon, f"prefill_slowdown[{j}]")

        # decode is the total decode latency
        decode = model.NewIntVar(0, horizon, f"decode[{j}]")

        # prefill_duration is the total time spent in the prefill phase. prefill latency + prefill slowdown (interference)
        prefill_duration = model.NewIntVar(1, horizon, f"prefill_duration[{j}]")
        model.Add(prefill_duration == prefill + prefill_slowdown)
        model.Add(prefill_ends[j] == starts[j] + prefill_duration)
        if colocated:
            model.Add(ends[j] == prefill_ends[j] + decode)
            duration = model.NewIntVar(1, horizon, f"duration[{j}]")
            model.Add(duration == prefill_duration + decode)
            for c in range(C):
                intervals[c].append(model.NewOptionalIntervalVar(starts[j], duration, ends[j], routes[j][c],
                                                                 f"life[{j},{c}]"))
                sizes[c].append(v.isl + v.osl)
        else:
            # (P-D split): the decode starts after the transfer if split
            model.Add(decode_starts[j] == prefill_ends[j] + q["kv_transfer_time"][j] * splits[j])
            model.Add(ends[j] == decode_starts[j] + decode)
            # Memory: the decode worker holds isl + osl tokens from the start (colocated) or the prefill end
            # to the end; if split, the prefill worker holds the prompt's isl until the
            # transfer is done
            life_start = model.NewIntVar(0, horizon, f"life_start[{j}]")
            model.Add(life_start == starts[j]).OnlyEnforceIf(splits[j].Not())
            model.Add(life_start == prefill_ends[j]).OnlyEnforceIf(splits[j])
            duration = model.NewIntVar(1, horizon, f"duration[{j}]")
            model.Add(duration == ends[j] - life_start)
            held = model.NewIntVar(1, horizon, f"held[{j}]")
            model.Add(held == decode_starts[j] - starts[j])
            for c in range(C):
                intervals[c].append(model.NewOptionalIntervalVar(life_start, duration, ends[j], droutes[j][c],
                                                                 f"life[{j},{c}]"))
                sizes[c].append(v.isl + v.osl)
                on = model.NewBoolVar(f"held_on[{j},{c}]")
                model.AddBoolAnd([routes[j][c], splits[j]]).OnlyEnforceIf(on)
                model.AddBoolOr([routes[j][c].Not(), splits[j].Not(), on])
                intervals[c].append(model.NewOptionalIntervalVar(starts[j], held, decode_starts[j], on,
                                                                 f"held[{j},{c}]"))
                sizes[c].append(v.isl)
        for c in range(C):
            prefills[c].append(model.NewOptionalIntervalVar(starts[j], prefill_duration, prefill_ends[j], routes[j][c],
                                                            f"pf[{j},{c}]"))
        prefill_lens.append(prefill)
        prefill_slowdowns.append(prefill_slowdown)
        decodes.append(decode)

    # finished requests' prefills and lifetimes that may still overlap the window's
    for route, s, e in ctx.finished_prefills:
        prefills[route].append(model.NewIntervalVar(s, e - s, e, "fixed_pf"))
    for route, s, e, size in ctx.finished_kv_lifetimes:
        intervals[route].append(model.NewIntervalVar(s, e - s, e, "fixed_life"))
        sizes[route].append(size)

    # one prefill at a time per cache
    for c in range(C):
        model.AddNoOverlap(prefills[c])

    # Constraints based on Context that is carried in a rolling window solve
    # context: no start on a cache before its finished requests end (cache_free_at), and committed
    # decisions are fixed (committed_route_and_hit, fixed_start)
    for j, bound in ctx.cache_free_at.items():
        for c, t in bound.items():
            # if the context has a constraint on start time for request j on cache c,
            # add the constraint in the solver.
            #  This constraint ensures that a new request only starts after the 
            #  requests which have finished. If it started before it will interfere
            #  with a finished requests, which means the request should not have been finished.
            model.Add(starts[j] >= t).OnlyEnforceIf(routes[j][c])
            # same constraint if the decode is running on other worker
            if not colocated:
                model.Add(decode_starts[j] >= t).OnlyEnforceIf(droutes[j][c])
    # committed decode caches
    for j, c in ctx.committed_droute.items():
        model.Add(droutes[j][c] == 1)
    for j, (route, src) in ctx.committed_route_and_hit.items():
        model.Add(routes[j][route] == 1)
        # for request that was already committed, we fix its route and cache hit provider if it exists.
        for i in range(N):
            model.Add(reuses[i][j] == int(i == src))
        if j in ext_reuses:
            model.Add(ext_reuses[j] == int(src == "ext"))
    for j, t in ctx.fixed_start.items():
        # for requests that were already started in prev window, we fix their start time
        model.Add(starts[j] == t)



    # Assign session ids for every session, session[j] is the session id for request j
    session = []
    for u in trace:
        session.append(len(session) if u.prev is None else session[u.prev])
    

    # interference modelling
    #   we use pairwise interference, for any pair of requests (k, j) their
    #   interference is modelled in the solver by using the following variables
    #       same[k,j]  : bool                 
    #               -> both requests k and j are on the same cache
    #       k_prefills_first : bool       
    #               -> k's prefill comes before j's
    #       decoding_prefill_overlap[a, b] : bool
    #                -> a's decoding overlaps with b's prefill
    #       decode_prefill_overlap_len[b, a] : int
    #                -> time length of overlap between b's prefill and a's decode
    #       decode_decode_overlap_len[a, b] : int    
    #                -> time length of overlap between a's and b's decode
    #       decode_decode_interference[x, y] : int
    #                -> how much interference (time) does y's decode add to x's
    dd_terms, pd_terms, dp_terms, decode_prefill_overlaps = ([[] for _ in range(N)] for _ in range(4))
    # pair_terms[kind, x, y] = (coef, var): each interference term's coefficient and variable, for carrying it forward
    # when one side finishes (used for rolling solve). 
    #   kind "dd": y's decode slowing x's; "pd": prefill x inside decode y; "dp": decode x during prefill y
    pair_terms = {}

    # helper to create a var for "cache route lists x and y pick the same cache"
    def same_cache(x, y, name):
        same = model.NewBoolVar(name)
        for c in range(C):
            model.Add(x[c] == y[c]).OnlyEnforceIf(same)
            model.Add(x[c] + y[c] <= 1).OnlyEnforceIf(same.Not())
        return same

    for k in range(N):
        for j in range(k + 1, N):
            if session[k] == session[j]:
                continue

            if colocated:
                # request k and j are on the same cache/worker
                same = same_cache(routes[k], routes[j], f"same[{k},{j}]")

                k_prefills_first = model.NewBoolVar(f"k_prefills_first[{k},{j}]")
                model.AddImplication(same.Not(), k_prefills_first.Not())
                model.Add(prefill_ends[k] <= starts[j]).OnlyEnforceIf([same, k_prefills_first])
                model.Add(prefill_ends[j] <= starts[k]).OnlyEnforceIf([same, k_prefills_first.Not()])
                same_dd, orders = same, ((k, j, k_prefills_first), (j, k, k_prefills_first.Not()))
            else:
                #where decodes of k and j are on one cache - used to determine if there is dd interference between them
                same_dd, orders = same_cache(droutes[k], droutes[j], f"same_dd[{k},{j}]"), ((k, j, None), (j, k, None))

            for a, b, a_prefills_first in orders:
                # (split, a decode of osl 1 still runs one step on its decode worker; ignored, as osl 1 is rare)
                if q["decode_steps"][a] == 0:
                    continue
                decoding_prefill_overlap = model.NewBoolVar(f"decoding_prefill_overlap[{a},{b}]")
                decode_prefill_overlap_end = model.NewIntVar(0, horizon, f"decode_prefill_overlap_end[{b},{a}]")
                if colocated:
                    model.AddImplication(decoding_prefill_overlap, same)
                    model.AddImplication(decoding_prefill_overlap, a_prefills_first)
                    model.Add(starts[b] + 1 <= ends[a]).OnlyEnforceIf(decoding_prefill_overlap)
                    model.Add(starts[b] >= ends[a]).OnlyEnforceIf([same, a_prefills_first,
                                                                   decoding_prefill_overlap.Not()])
                    overlap_from = starts[b]
                else:
                    same_pd = same_cache(routes[b], droutes[a], f"same_pd[{b},{a}]")
                    overlap_from = model.NewIntVar(0, horizon, f"decode_prefill_overlap_start[{b},{a}]")
                    model.AddMaxEquality(overlap_from, [decode_starts[a], starts[b]])
                    model.AddImplication(decoding_prefill_overlap, same_pd)
                    model.Add(decode_prefill_overlap_end >= overlap_from + 1).OnlyEnforceIf(decoding_prefill_overlap)
                    model.Add(decode_prefill_overlap_end <= overlap_from).OnlyEnforceIf(
                        [same_pd, decoding_prefill_overlap.Not()])

                # P-D: b's prefill chunks inside a's decode part, adds interference to b
                model.AddMinEquality(decode_prefill_overlap_end, [ends[a], prefill_ends[b]])
                decode_prefill_overlap_len = model.NewIntVar(0, horizon, f"decode_prefill_overlap_len[{b},{a}]")
                model.Add(decode_prefill_overlap_len == decode_prefill_overlap_end - overlap_from).OnlyEnforceIf(
                    decoding_prefill_overlap)
                model.Add(decode_prefill_overlap_len == 0).OnlyEnforceIf(decoding_prefill_overlap.Not())
                coef = round(S * q["chunk_ms"][b] / (q["chunk_ms"][b] + q["t0"][a]))
                pd_terms[a].append(coef * decode_prefill_overlap_len)
                pair_terms["pd", b, a] = (coef, decode_prefill_overlap_len)
                # D->P: a's decode part is in each of b's chunk steps
                coef = round(S * q["delta"][a])
                dp_terms[b].append(coef * decoding_prefill_overlap)
                decode_prefill_overlaps[b].append(decoding_prefill_overlap)
                pair_terms["dp", a, b] = (coef, decoding_prefill_overlap)

            if q["decode_steps"][k] == 0 or q["decode_steps"][j] == 0:
                continue

            # D-D interference: k's decode and j's decode overlap, adds interference to both
            overlap_start = model.NewIntVar(0, horizon, f"overlap_start[{k},{j}]")
            model.AddMaxEquality(overlap_start, [decode_starts[k], decode_starts[j]])
            overlap_end = model.NewIntVar(0, horizon, f"overlap_end[{k},{j}]")
            model.AddMinEquality(overlap_end, [ends[k], ends[j]])
            decode_decode_overlap_len = model.NewIntVar(0, horizon, f"decode_decode_overlap_len[{k},{j}]")
            model.AddMaxEquality(decode_decode_overlap_len, [0, overlap_end - overlap_start])
            for x, y in ((k, j), (j, k)):
                cap = round(S * q["delta"][y] * min(q["decode_steps"][x], q["decode_steps"][y]))
                dd_capped = model.NewIntVar(0, cap, f"dd_capped[{x},{y}]")
                coef = round(S * q["delta"][y] / (q["t0"][x] + q["delta"][y]))
                model.AddMinEquality(dd_capped, [coef * decode_decode_overlap_len, cap])
                decode_decode_interference = model.NewIntVar(0, cap, f"decode_decode_interference[{x},{y}]")
                model.Add(decode_decode_interference == dd_capped).OnlyEnforceIf(same_dd)
                model.Add(decode_decode_interference == 0).OnlyEnforceIf(same_dd.Not())
                dd_terms[x].append(decode_decode_interference)
                pair_terms["dd", x, y] = (1, decode_decode_interference)

    # Add constraints so that the decode and prefill interference terms is accounted for
    # in the decode and prefill time variables defined previously.
    for j in range(N):
        # get the interference caused due to finished requests
        finished_decode, finished_prefill_slowdown, finished_any = ctx.finished_interference.get(j, (0, 0, False))
        # decode = raw decode time + p-d and d-d interference
        rhs = round(S * q["decode_steps"][j] * q["t0"][j]) + sum(dd_terms[j]) + sum(pd_terms[j]) + finished_decode
        rhs += round(S * q["t0"][j]) * splits[j]
        model.Add(S * decodes[j] >= rhs)
        model.Add(S * decodes[j] <= rhs + S - 1)
        if finished_any:
            any_decoding = 1
        elif decode_prefill_overlaps[j]:
            any_decoding = model.NewBoolVar(f"any_decoding[{j}]")
            model.AddMaxEquality(any_decoding, decode_prefill_overlaps[j])
        else:
            any_decoding = 0
        # prefill time = raw prefill time + d-p interference
        rhs = q["chunks"][j] * (round(S * (steps.f1 - steps.s)) * any_decoding + sum(dp_terms[j])
                                + finished_prefill_slowdown)
        model.Add(S * prefill_slowdowns[j] >= rhs)
        model.Add(S * prefill_slowdowns[j] <= rhs + S - 1)

    # Same as cache_packing: eviction intervals and cache memory
    evicts = []
    for i, u in enumerate(trace):
        evict = model.NewIntVar(0, horizon, f"evict[{i}]")
        model.Add(evict >= ends[i])
        is_provider = model.NewBoolVar(f"provider[{i}]")
        evicts.append(evict)
        for j in range(N):
            if shared[i][j] > 0:
                model.Add(evict >= starts[j]).OnlyEnforceIf(reuses[i][j])
                model.AddImplication(reuses[i][j], is_provider)
        _evict = model.NewIntVar(0, horizon, f"_evict[{i}]")
        for c in range(C):
            cond = model.NewBoolVar(f"cond[{i},{c}]")
            model.AddBoolOr([is_provider.Not(), routes[i][c].Not(), cond])
            model.AddImplication(cond, is_provider)
            model.AddImplication(cond, routes[i][c])
            intervals[c].append(model.NewOptionalIntervalVar(ends[i], _evict, evict, cond, f"eviction[{i},{c}]"))
            sizes[c].append(max(shared[i]))

    # Also add cache constraints for finshed request in context which may be cache providers 
    # to requests in current window
    users = {}
    for j, (key, pe, route, end, _, size) in ctx.finished_hit_sources.items():
        users.setdefault(key, (route, end, size, []))[3].append(j)
    for key, (route, end, size, js) in users.items():
        used = model.NewBoolVar(f"ext_used[{key}]")
        model.AddMaxEquality(used, [ext_reuses[j] for j in js])
        evict = model.NewIntVar(end, horizon, f"ext_evict[{key}]")
        for j in js:
            model.Add(evict >= starts[j]).OnlyEnforceIf(ext_reuses[j])
        length = model.NewIntVar(0, horizon, f"_ext_evict[{key}]")
        intervals[route].append(model.NewOptionalIntervalVar(end, length, evict, used, f"ext_eviction[{key}]"))
        sizes[route].append(size)
    for c in range(C):
        model.AddCumulative(intervals[c], sizes[c], mem)

    ttfts = []
    for i in range(N):
        ttft = model.NewIntVar(0, horizon, f"ttft[{i}]")
        if colocated:
            model.Add(ttft == prefill_ends[i] - releases[i])
        else:
            first_step = model.NewIntVar(0, horizon, f"first_step[{i}]")
            model.AddDivisionEquality(first_step, decodes[i], trace[i].osl)
            model.Add(ttft == prefill_ends[i] - releases[i]).OnlyEnforceIf(splits[i].Not())
            model.Add(ttft == decode_starts[i] + first_step - releases[i]).OnlyEnforceIf(splits[i])
        ttfts.append(ttft)
    # mean e2e by default. TTFT alone favours packing sessions onto the caches holding their prefixes, which
    # costs more in TPOT than it saves in TTFT at realistic load.
    if objective == "e2e":
        model.Minimize(sum(ends[i] - releases[i] for i in range(N)))
    else:
        model.Minimize(sum(ttfts))
    return model, dict(routes=routes, droutes=droutes, splits=splits, decode_starts=decode_starts,
                       starts=starts, ends=ends, prefill_ends=prefill_ends, reuses=reuses,
                       ext_reuses=ext_reuses, releases=releases, ttfts=ttfts, decodes=decodes,
                       prefill_slowdowns=prefill_slowdowns,
                       evicts=evicts, pair_terms=pair_terms)


# NEW: start times for a complete feasible hint. Simulates the closed loop under the model's own rules for the given
# routes and cache-hit sources (src: a request index, "ext" or None): prefills FIFO per cache by release, interference
# terms with the model's rounded coefficients, iterated to a fixed point. Starts are padded by `margin` ms so rounding
# can't make them infeasible. The memory constraint is not simulated.
# NEW (P-D split): droute is each request's decode cache (None: its prefill cache, colocated); a split request's decode
# starts after its transfer and runs one more step, as in build_model. full=True returns the simulated times
# (dict of arrays: starts, prefill_ends, decode_starts, ends, decodes, prefill_slowdowns) instead of the start list.
def simulate_starts(trace, blk_size, route, src, steps, margin, ctx=None, iters=400, droute=None, full=False):
    N = len(trace)
    ctx = ctx or Context()
    shared = [[shared_tokens(u, v, blk_size) if i != j else 0 for j, v in enumerate(trace)] for i, u in enumerate(trace)]
    P = lambda isl, hit=0: steps(isl, hit, blk_size)
    q = {key: np.array(val) for key, val in constants(trace, blk_size, steps, max_hits(trace, shared, ctx)).items()}
    hit = [ctx.finished_hit_sources[j][4] if s == "ext" else shared[s][j] if s is not None else 0 for j, s in enumerate(src)]
    prefill = np.array([P(v.isl, h) for v, h in zip(trace, hit)])
    session = []
    for u in trace:
        session.append(len(session) if u.prev is None else session[u.prev])
    route, session = np.array(route), np.array(session)
    droute = route if droute is None else np.array(droute)
    # Pairs that interact, from different sessions: (CHANGED for P-D split) a's decode on b's prefill cache (P-D and
    # D->P, [a, b]), and decodes on one cache (D-D)
    other = session[:, None] != session[None, :]
    inter_pd = (droute[:, None] == route[None, :]) & other
    inter_dd = (droute[:, None] == droute[None, :]) & other
    split = droute != route
    kv_transfer_time = np.where(split, q["kv_transfer_time"], 0)
    t0, delta, decode_steps = q["t0"], q["delta"], q["decode_steps"]
    cpd = np.round(S * q["chunk_ms"][None, :] / (q["chunk_ms"][None, :] + t0[:, None]))  # [a, b]: b's prefill, a's decode
    cdd = np.round(S * delta[None, :] / (t0[:, None] + delta[None, :]))                  # [x, y]: y's decode slows x
    cap = np.round(S * delta[None, :] * np.minimum.outer(decode_steps, decode_steps))
    cdp, cany = np.round(S * delta), round(S * (steps.f1 - steps.s))
    finished_terms = np.array([ctx.finished_interference.get(j, (0, 0, False)) for j in range(N)], float).reshape(N, 3)
    base = np.round(S * decode_steps * t0) + finished_terms[:, 0] + np.round(S * t0) * split
    fixed = {}
    for r, s, e in ctx.finished_prefills:
        fixed.setdefault(r, []).append((s, e))

    succ = [[] for _ in range(N)]
    for j, u in enumerate(trace):
        if u.prev is not None:
            succ[u.prev].append(j)
    decode, prefill_slowdown = np.ceil(base / S), np.zeros(N)
    for _ in range(iters):
        ends, starts, pe, ds = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
        # Closed loop: releases depend on predecessors' ends, so place requests in release order
        ready = [(u.arrival, j) for j, u in enumerate(trace) if u.prev is None]
        heapq.heapify(ready)
        free = {}
        while ready:
            rel, j = heapq.heappop(ready)
            c = route[j]
            src_pe = ctx.finished_hit_sources[j][1] if src[j] == "ext" else pe[src[j]] if src[j] is not None else -np.inf
            s = ceil(max(rel, free.get(c, -np.inf), src_pe, ctx.cache_free_at.get(j, {}).get(c, -np.inf))) + margin
            if split[j] and droute[j] in ctx.cache_free_at.get(j, {}):
                # NEW (P-D split): nor decode on its decode cache before that cache's finished requests end
                s = max(s, ceil(ctx.cache_free_at[j][droute[j]] - prefill[j] - prefill_slowdown[j] - kv_transfer_time[j])
                        + margin)
            # Skip past finished requests' prefills and fixed-start requests' prefills on the cache
            moved = j not in ctx.fixed_start
            if not moved:
                s = ctx.fixed_start[j]
            blocked = fixed.get(c, []) + [(ctx.fixed_start[i],
                                           ctx.fixed_start[i] + prefill[i] + prefill_slowdown[i] + margin)
                                          for i in ctx.fixed_start if route[i] == c and i != j]
            while moved:
                moved = False
                for fs, fe in blocked:
                    if s < fe and fs < s + prefill[j] + prefill_slowdown[j] + margin:
                        s, moved = fe + margin, True
            starts[j] = s
            pe[j] = s + prefill[j] + prefill_slowdown[j]
            ds[j] = pe[j] + kv_transfer_time[j]
            ends[j] = ds[j] + decode[j]
            free[c] = max(free.get(c, -np.inf), pe[j])
            for i in succ[j]:
                heapq.heappush(ready, (ends[j] + trace[i].think, i))
        # CHANGED for P-D split: a's decode [ds_a, end_a) overlaps b's prefill [start_b, pe_b), as in build_model
        overlap_from = np.maximum(ds[:, None], starts[None, :])
        overlap_to = np.minimum(ends[:, None], pe[None, :])
        decoding_prefill_overlap = inter_pd & (overlap_from < overlap_to) & (decode_steps[:, None] > 0)
        decode_prefill_overlap_len = np.where(decoding_prefill_overlap, overlap_to - overlap_from, 0)
        decode_decode_overlap_len = np.clip(np.minimum.outer(ends, ends) - np.maximum.outer(ds, ds), 0, None)
        decode_decode_overlap_len = np.where(inter_dd & (decode_steps[:, None] > 0) & (decode_steps[None, :] > 0),
                                             decode_decode_overlap_len, 0)
        np.fill_diagonal(decode_decode_overlap_len, 0)
        new_decode = np.ceil((base + np.minimum(cdd * decode_decode_overlap_len, cap).sum(1)
                              + (cpd * decode_prefill_overlap_len).sum(1)) / S)
        any_decoding = decoding_prefill_overlap.any(0) | (finished_terms[:, 2] > 0)
        new_prefill_slowdown = np.ceil(q["chunks"] * (cany * any_decoding
                                                      + (cdp[:, None] * decoding_prefill_overlap).sum(0)
                                                      + finished_terms[:, 1]) / S)
        change = max(np.abs(new_decode - decode).max(initial=0),
                     np.abs(new_prefill_slowdown - prefill_slowdown).max(initial=0))
        if change < 1:
            break
        # Damped, so the closed loop settles
        decode = np.ceil(0.5 * decode + 0.5 * new_decode)
        prefill_slowdown = np.ceil(0.5 * prefill_slowdown + 0.5 * new_prefill_slowdown)
    if full:
        return dict(starts=starts, prefill_ends=pe, decode_starts=ds, ends=ends, decodes=decode,
                    prefill_slowdowns=prefill_slowdown)
    return [int(s) for s in starts]


# NEW: a complete feasible hint for build_model (every proto variable), from routes and cache-hit sources. Fixes them
# with simulated starts and lets CP-SAT fill in the rest, widening the margin if that's infeasible. Returns
# (hint, objective), or (None, None). Partial hints (routes, starts, hits) leave CP-SAT searching for minutes beyond
# about 96 requests.
# starts: use these start times instead of simulating.
# NEW (P-D split): droute: decode caches, for topology "split" (None: colocated); pools as in build_model.
def full_hint(trace, blk_size, C, mem, horizon, route, src, steps=default_steps, limit=60, margins=(3, 10, 50),
              ctx=None, starts=None, droute=None, pools=None):
    N = len(trace)
    given = starts
    topology = "colocated" if droute is None else "split"
    for margin in margins[:1] if given else margins:
        starts = given or simulate_starts(trace, blk_size, route, src, steps, margin, ctx, droute=droute)
        m, v = build_model(trace, blk_size, C, mem, horizon, steps, ctx=ctx, topology=topology, pools=pools)
        for j in range(N):
            m.Add(v["routes"][j][route[j]] == 1)
            if droute is not None:
                m.Add(v["droutes"][j][droute[j]] == 1)
            if j not in (ctx.fixed_start if ctx else {}):
                m.Add(v["starts"][j] == starts[j])
            for i in range(N):
                m.Add(v["reuses"][i][j] == int(i == src[j]))
            if j in v["ext_reuses"]:
                m.Add(v["ext_reuses"][j] == int(src[j] == "ext"))
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = limit
        st = s.Solve(m)
        if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return list(s.ResponseProto().solution), s.ObjectiveValue()
    return None, None


# NEW: solves a whole trace (in ms) in windows that advance in solved time. Each window adds up to `window` new
# requests, those with the earliest estimated release among requests whose predecessor is solved (so at most one new
# turn per session), solves them with the uncommitted and still-running requests, then commits the `step` uncommitted
# requests that start first: their cache and cache hit are fixed, and their start once their predecessor is finished
# (fixing it earlier makes new requests wait for the predecessor to end, since slowing it would move the start;
# never fixing it lets committed requests move later and never finish). The fallback hint keeps the previous window's
# starts and places new requests after everything on their cache. Once it ends before every
# uncommitted request can start, it is finished and leaves the model, leaving constants (Context) for the requests that
# follow it, take a cache hit from it, or overlapped it. Windows advance in solved time rather than trace order since
# sessions drift from their recorded timing; in trace order, uncommitted requests can start long before committed
# ones end, so nothing finishes and windows grow. Each window starts from a complete hint: decisions from the previous
# window, and for new requests the cache of their cache-hit source (the session's latest prompt), or the cache with the
# fewest running requests. Returns per request dict(route, start, src, prefill_slowdown, decode, prefill_end, end,
# release), src a request index or None.
# NEW (P-D split): topology and pools as in build_model. With "split", results also carry each request's decode cache
# (droute) and decode start, and new requests are hinted colocated, so any split comes from the solver.
def solve_rolling(trace, blk_size, C, mem, window, step, limit, steps=default_steps, horizon=None, lookahead=30_000,
                  log=print, inspect=None, topology="colocated", pools=None):
    N = len(trace)
    split = topology == "split"
    # Everything run back to back, at 3x its uninterrupted decode time
    horizon = horizon or int(max(u.arrival or 0 for u in trace) + sum(
        u.think + steps(u.isl, 0, blk_size) + 3 * u.osl * (steps.f1 + steps.k * (u.isl + u.osl)) for u in trace))
    session = []
    for u in trace:
        session.append(len(session) if u.prev is None else session[u.prev])
    members = {}
    for j, s in enumerate(session):
        members.setdefault(s, []).append(j)
    # Cache-hit sources: earlier requests of the session sharing a prefix (hash ids differ across sessions)
    sources = {j: [(i, t) for i in members[session[j]] if i < j and (t := shared_tokens(trace[i], trace[j], blk_size)) > 0]
               for j in range(N)}
    size = [0] * N
    for j in range(N):
        for i, t in sources[j]:
            size[i] = max(size[i], t)

    vals, committed, started, finished, finished_interference, uncommitted = {}, set(), set(), set(), {}, []
    cache_free_at = {}                          # cache -> latest end of its finished requests
    last = None                                 # previous window: (index map, vars, solver)
    while True:
        # New requests: earliest estimated release among those whose predecessor is solved
        pool = [j for j in range(N) if j not in vals and (trace[j].prev is None or trace[j].prev in vals)]
        rel = {j: trace[j].arrival if trace[j].prev is None else vals[trace[j].prev]["end"] + trace[j].think for j in pool}
        if not pool and not uncommitted:
            return vals
        # Earliest any uncommitted request can start: its predecessor's earliest end + think, recursively. A request
        # ends no earlier than its start (fixed once committed) plus its prefill and decode with no interference, which
        # only adds time. Solved starts would be tighter but can move earlier, and finishing against them locks in
        # delays, since nothing may start on a finished request's cache before it ends.
        # Requests still in the pool count too: they may be released before the window's requests and added later.
        # Repeated until nothing changes, so a session can advance several turns per window: a fixed start
        # tightens its successor's bound, which can finish more requests
        while True:
            size_started = len(started)
            lb = {}
            for j in sorted(set(uncommitted) | set(pool) | (committed - started)):
                u = trace[j]
                if u.prev is None:
                    lb[j] = u.arrival
                    continue
                p = trace[u.prev]
                if u.prev in started:
                    hit = next((t for i, t in sources[u.prev] if i == vals[u.prev]["src"]), 0)
                    start = vals[u.prev]["start"]
                else:
                    hit = max([t for _, t in sources[u.prev]], default=0)
                    start = lb[u.prev]
                lb[j] = start + steps(p.isl, hit, blk_size) + (p.osl - 1) * (steps.f1 + steps.k * (p.isl + p.osl / 2)) \
                    + u.think - 1
            T = min(lb.values())
            # A request is finished once it ends before any request of another session can start (its own session's
            # later turns can't overlap it, and comparing against them would deadlock: its successor's bound comes from
            # its own earliest end)
            first = {}
            for j, t in lb.items():
                first[session[j]] = min(first.get(session[j], t), t)
            others = lambda s: min([t for s2, t in first.items() if s2 != s], default=float("inf"))
            newly = [j for j in started if j not in finished and vals[j]["end"] <= others(session[j])]
            # Two overlapping sessions can block each other: each one's next turn may (by its bound) start before the
            # other's running request ends, so neither finishes and the frontier stalls on a committed request that
            # can't start. Then compare against solved starts instead, and failing that finish the running request
            # that ends first. A later request can then wait for it, but only on its cache.
            if not newly and min(lb, key=lambda j: (lb[j], j)) in committed - started:
                est = {j: vals[j]["start"] if j in vals else t for j, t in lb.items()}
                first2 = {}
                for j, t in est.items():
                    first2[session[j]] = min(first2.get(session[j], t), t)
                newly = [j for j in started if j not in finished and vals[j]["end"] <= min(
                    [t for s2, t in first2.items() if s2 != session[j]], default=float("inf"))]
                running = [j for j in started if j not in finished]
                if not newly and running:
                    newly = [min(running, key=lambda j: (vals[j]["end"], j))]
            # Record the finished requests' interference on the committed requests they overlapped
            # (finished_interference). Only started ones keep it; see where the window's context is built
            if last is not None and newly:
                index, lv, ls = last
                for u in committed:
                    if u in finished or u in newly:
                        continue
                    d, x, a = finished_interference.get(u, (0, 0, False))
                    for f in newly:
                        # (CHANGED for P-D split: f and u can share a cache through either phase; terms are 0 when
                        # they share none)
                        if (not split and vals[f]["route"] != vals[u]["route"]) or f not in index or u not in index:
                            continue
                        fu, uu = index[f], index[u]
                        for key in (("dd", uu, fu), ("pd", fu, uu)):
                            if key in lv["pair_terms"]:
                                coef, var = lv["pair_terms"][key]
                                d += coef * ls.Value(var)
                        if ("dp", fu, uu) in lv["pair_terms"]:
                            coef, var = lv["pair_terms"]["dp", fu, uu]
                            x += coef * ls.Value(var)
                            a = a or bool(ls.Value(var))
                    finished_interference[u] = (d, x, a)
            for f in newly:
                finished.add(f)
                r, d = vals[f]["route"], vals[f].get("droute", vals[f]["route"])
                # NEW (P-D split): a split request is done on its prefill cache at its prefill end
                cache_free_at[r] = max(cache_free_at.get(r, 0), vals[f]["end"] if r == d else vals[f]["prefill_end"])
                cache_free_at[d] = max(cache_free_at.get(d, 0), vals[f]["end"])
                finished_interference.pop(f, None)
            # A committed request's start is fixed once its predecessor is finished: before that, new requests may slow
            # the predecessor, and a fixed start would leave them waiting until it ends
            started |= {j for j in committed if trace[j].prev is None or trace[j].prev in finished}
            if not newly and len(started) == size_started:
                break
        # Add requests released within `lookahead` of the frontier (at least one, so the solve always advances).
        # Without it, later sessions' first turns fill the window and are committed far ahead of the frontier, and
        # sit in the window until it catches up.
        near = sorted((j for j in pool if rel[j] <= T + lookahead), key=lambda j: (rel[j], j))
        new = near[:max(0, window - len(uncommitted))]
        if not new and not uncommitted:
            new = sorted(pool, key=lambda j: (rel[j], j))[:1]
        if inspect is not None:
            inspect.append(dict(lb=dict(lb), first=dict(first), session=session, started=set(started),
                                committed=set(committed), finished=set(finished), vals={j: dict(x) for j, x in vals.items()}))
        active = sorted((committed - finished) | set(uncommitted) | set(new))
        index = {g: l for l, g in enumerate(active)}

        # The window's trace and context
        local, ctx = [], Context()
        for g in active:
            u = trace[g]
            if u.prev is None or u.prev in index:
                local.append(Request(u.isl, u.osl, u.arrival, u.hash_ids, index.get(u.prev), u.think))
            else:
                local.append(Request(u.isl, u.osl, vals[u.prev]["end"] + u.think, u.hash_ids, None, 0))
            l = index[g]
            # One cache-hit source among finished requests: the one already chosen, else the largest hit (the
            # session's latest prompt, in practice)
            ext = [(t, i) for i, t in sources[g] if i in finished]
            if g in vals and vals[g]["src"] in finished:
                ext = [(t, i) for t, i in ext if i == vals[g]["src"]]
            if ext:
                t, i = max(ext)
                ctx.finished_hit_sources[l] = (i, vals[i]["prefill_end"], vals[i]["route"], vals[i]["end"], t, size[i])
            # A request that isn't started has to start after the finished requests on its cache end (cache_free_at), so
            # it won't overlap them: drop what it overlapped them by in the last solve (nonzero only after a forced
            # finish). A started one keeps its start, and so its overlaps.
            if g in finished_interference and g not in started:
                del finished_interference[g]
            if g in finished_interference:
                ctx.finished_interference[l] = finished_interference[g]
            if g in committed:
                src = vals[g]["src"]
                ctx.committed_route_and_hit[l] = (vals[g]["route"],
                                                  index[src] if src in index else "ext" if src is not None else None)
                if split:
                    ctx.committed_droute[l] = vals[g]["droute"]
            if g in started:
                ctx.fixed_start[l] = vals[g]["start"]
            elif cache_free_at:
                ctx.cache_free_at[l] = dict(cache_free_at)
        T0 = min([vals[g]["start"] for g in active if g in vals] + [rel[j] for j in new])
        for f in finished:
            if vals[f]["prefill_end"] > T0:
                ctx.finished_prefills.append((vals[f]["route"], vals[f]["start"], vals[f]["prefill_end"]))
            r, d = vals[f]["route"], vals[f].get("droute", vals[f]["route"])
            if vals[f]["end"] > T0 and r == d:
                ctx.finished_kv_lifetimes.append((r, vals[f]["start"], vals[f]["end"], trace[f].isl + trace[f].osl))
            elif vals[f]["end"] > T0:
                # NEW (P-D split): the decode cache holds it from the prefill end, the prefill cache until the transfer
                ctx.finished_kv_lifetimes.append((d, vals[f]["prefill_end"], vals[f]["end"],
                                                  trace[f].isl + trace[f].osl))
                if vals[f]["decode_start"] > T0:
                    ctx.finished_kv_lifetimes.append((r, vals[f]["start"], vals[f]["decode_start"], trace[f].isl))

        # Complete hint: previous decisions, else follow the cache-hit source, else the least busy cache
        # (NEW (P-D split): decode caches from previous decisions, else colocated; with pools, prefills go to the
        # least busy prefill cache and decodes stay there if it's also a decode cache, else go to the least busy one)
        route, src, running, drunning = [], [], [0] * C, [0] * C
        droute = [vals[g]["droute"] if g in vals else None for g in active] if split else None
        prefill_pool, decode_pool = pools or (range(C), range(C))
        for g in active:
            if g in vals:
                running[vals[g]["route"]] += 1
                drunning[vals[g].get("droute", vals[g]["route"])] += 1
        for g in active:
            l = index[g]
            if g in vals:
                s = vals[g]["src"]
                route.append(vals[g]["route"])
                src.append(index[s] if s in index else "ext" if s is not None and l in ctx.finished_hit_sources else None)
            else:
                cands = [(t, i) for i, t in sources[g] if i in index]
                if cands:
                    i = max(cands)[1]
                    route.append(route[index[i]])
                    src.append(index[i])
                elif l in ctx.finished_hit_sources:
                    route.append(ctx.finished_hit_sources[l][2])
                    src.append("ext")
                else:
                    route.append(min(prefill_pool, key=lambda c: running[c]))
                    src.append(None)
                running[route[-1]] += 1
        if split:
            for l, r in enumerate(route):
                if droute[l] is None:
                    droute[l] = r if r in decode_pool else min(decode_pool, key=lambda c: drunning[c])
                    drunning[droute[l]] += 1
        hint, _ = full_hint(local, blk_size, C, mem, horizon, route, src, steps, ctx=ctx, droute=droute, pools=pools)
        if hint is None:
            # Always feasible: the previous window's starts, and new requests after everything on their cache, so
            # they slow nothing down
            free, starts, ends = {}, [0] * len(active), [0] * len(active)
            for g in active:
                if g in vals:
                    for c in {vals[g]["route"], vals[g].get("droute", vals[g]["route"])}:
                        free[c] = max(free.get(c, 0), vals[g]["end"])
            for g in sorted(active, key=lambda g: (g not in vals, rel.get(g, 0), g)):
                l = index[g]
                if g in vals:
                    starts[l] = vals[g]["start"]
                    continue
                u = local[l]
                r = u.arrival if u.prev is None else None
                if r is None:
                    p = active[u.prev]
                    r = (vals[p]["end"] if p in vals else ends[u.prev]) + u.think
                c = route[l]
                d = droute[l] if split else c
                starts[l] = max(r, free.get(c, 0), free.get(d, 0), *ctx.cache_free_at.get(l, {}).values()) + 1
                h = ctx.finished_hit_sources[l][4] if src[l] == "ext" else shared_tokens(local[src[l]], u, blk_size) if src[l] is not None else 0
                e = starts[l] + steps(u.isl, h, blk_size) + ceil(u.osl * (steps.f1 + steps.k * (u.isl + u.osl)) + 1)
                if d != c:
                    # NEW (P-D split): the transfer and the extra decode step
                    e += transfer_ms(u.isl, steps.f1 + steps.k * (u.isl + u.osl)) + ceil(steps.f1 + steps.k * (u.isl + u.osl))
                free[c] = max(free.get(c, 0), e)
                free[d] = max(free.get(d, 0), e)
                ends[l] = e
            hint, _ = full_hint(local, blk_size, C, mem, horizon, route, src, steps, ctx=ctx, starts=starts,
                                droute=droute, pools=pools)
        model, v = build_model(local, blk_size, C, mem, horizon, steps, ctx=ctx, topology=topology, pools=pools)
        for i, x in enumerate(hint or []):
            model.AddHint(model.get_int_var_from_proto_index(i), x)
        if hint is None:
            for g in active:
                if g in vals:
                    model.AddHint(v["starts"][index[g]], vals[g]["start"])
                model.AddHint(v["routes"][index[g]][route[index[g]]], 1)
                if split:
                    model.AddHint(v["droutes"][index[g]][droute[index[g]]], 1)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = limit
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(f"window of {len(active)} requests: {solver.StatusName(status)}")
        obj, bound = solver.ObjectiveValue(), solver.BestObjectiveBound()
        V = solver.Value
        for g in active:
            l = index[g]
            s = next((active[i] for i in range(len(active)) if V(v["reuses"][i][l])), None)
            if s is None and l in v["ext_reuses"] and V(v["ext_reuses"][l]):
                s = ctx.finished_hit_sources[l][0]
            r = v["releases"][l]
            vals[g] = dict(route=next(c for c in range(C) if V(v["routes"][l][c])), start=V(v["starts"][l]), src=s,
                           prefill_slowdown=V(v["prefill_slowdowns"][l]), decode=V(v["decodes"][l]),
                           prefill_end=V(v["prefill_ends"][l]),
                           end=V(v["ends"][l]), release=r if isinstance(r, int) else V(r))
            if split:
                vals[g] |= dict(droute=next(c for c in range(C) if V(v["droutes"][l][c])),
                                decode_start=V(v["decode_starts"][l]))
        last = (index, v, solver)
        # Commit (cache and cache hit) the uncommitted requests that start first; with nothing left to add, commit
        # everything
        pending = sorted(uncommitted + new, key=lambda j: (vals[j]["start"], j))
        more = any(j not in vals for j in range(N))
        if more:
            # Within half the lookahead of the frontier, at least one
            commit = [j for j in pending if vals[j]["start"] <= T + lookahead / 2][:step] or pending[:1]
        else:
            commit = pending
        committed |= set(commit)
        uncommitted = [j for j in pending if j not in committed]
        log(f"Window at {T / 1000:.0f} s: {len(active)} requests ({len(new)} new, {len(active) - len(pending)} committed), "
            f"{solver.StatusName(status)} in {solver.WallTime():.1f} s, gap {(obj - bound) / max(obj, 1):.1%}"
            + ("" if hint else ", no complete hint") + f"; {len(vals)}/{N} solved")
        if not more and not uncommitted:
            return vals
