# Cache packing with prefill/decode interference modelling.
import json
from bisect import bisect_left
from dataclasses import dataclass, field
from math import ceil
import numpy as np
from ortools.sat.python import cp_model
from cache_packing import shared_tokens
from interference_packing_helper import RollingState, advance_states, build_window_context, session_info

# Scaling Coefficient Constant used to account for fractional latency steps.
S = 100

# D->P interference: the fraction of the concurrent decode step each prefill chunk pays. 1.0 for a vLLM-style engine
# that fuses a chunk and the decodes into one step (Mocker). An engine that runs a prefill alone and pauses the decodes
# (cs349e-assn2's MIXED mode) only waits for the in-flight decode step, on average half of it: 0.5.
DP_SCALE = 1.0

# P->D interference: how much of an overlapping prefill a decode loses. Fused (vLLM/Mocker): the chunk joins the decode
# step, so a decode loses chunk / (chunk + t0) of the overlap. Not fused (cs349e-assn2 MIXED): the prefill runs alone
# and every decode on the GPU waits for all of it.
PD_FUSED = True

# Batch-wide stall (ms) each request's arrival and departure cost the decodes on its GPU (cs349e-assn2: admit pads and
# concatenates the whole batch's KV, evict index_selects and compacts it). Charged to a decode once per later prefill
# that overlaps it on the same GPU: in steady state that counts one admit and one evict per neighbour.
EVENT_MS = 0.0

# Hinged decode step (cs349e-assn2's HF engine): a step costs max(FLAT_MS, t0 + delta (B - 1)), flat while launch-bound
# and linear once the batch's KV copy dominates. With FLAT_MS set, each decode is the larger of the two branches summed
# over its steps (D-D terms only in the linear one), and a prefill's D->P wait is at least DP_SCALE x FLAT_MS when
# anything decodes. 0 disables it (one linear branch, as before).
FLAT_MS = 0.0
DD_STEP_MS = 0.0
DECODE_MAY_WAIT = False

# finds the intercept of x on a piecewise linear function defined by xs and ys
def interp(xs, ys, x):
    # Piecewise linear, extrapolating linearly at both ends
    i = min(max(bisect_left(xs, x), 1), len(xs) - 1)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


# Step-level latency model obtained by running tpot_sim.py, loaded from steps_<config>.json. 
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
        # hits are block-aligned, and at least one token is computed
        return max(1, round(self.prefill_ms(isl, min(hit, isl - 1) // blk_size * blk_size)))


default_steps = StepLatency("qwen3-8b_a100x4")


# Context contains the information about the requests that the solver has
# already fully or partially solved.
# Every request can be in one of the four states:
#   1) uncommitted - all the decisions for the request are yet to be made
#   2) committed - cache and cache-hit source for the request is fixed, but timing is still being solved
#   3) started - cache, hit source and start timing for the request are fixed.
#   4) finished - all decisions are final, and the request is not solved in the current window anymore.
@dataclass
class Context:
    # finished_hit_sources[j] = (src, prefill_end, route, end, tokens, max_reused_prefix)
    # A request j in the current window "can" get a cache hit of `tokens` from the finished request `src` on cache `route`; 
    #  `src`'s prefix of `max_reused_prefix` tokens stays in memory from src's end until j starts.
    finished_hit_sources: dict = field(default_factory=dict)
    # (route, start, prefill_end) of finished requests' prefills that may still overlap with the current window.
    finished_prefills: list = field(default_factory=list)
    # (route, start, end, size) of finished requests' KV memory that may still overlap the current window.
    finished_kv_lifetimes: list = field(default_factory=list)
    # finished_interference[j] = (decode, prefill_slowdown, any, decode's P->D part)
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
KV_BYTES_PER_TOKEN = 2 * 36 * 8 * 128 * 2       # Qwen3-8B: 36 layers x 8 KV heads x 128 dims x (K, V) x bf16
KV_BANDWIDTH_GBPS = 300                         # A100 NVLink, one direction (Mocker-style default)
# Fixed part of a P->D handoff, in ms. None: one decode step (t0). cs349e-assn2 sets it from measurement: the decode
# worker only polls between steps, so a handoff waits ~1.5 steps, plus its own admit.
TRANSFER_FIXED_MS = None


def transfer_ms(isl, t0):
    fixed = t0 if TRANSFER_FIXED_MS is None else TRANSFER_FIXED_MS
    return ceil(isl * KV_BYTES_PER_TOKEN / (KV_BANDWIDTH_GBPS * 1e9) * 1e3 + fixed)


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
            if DECODE_MAY_WAIT:
                # a decode worker with no free slot leaves the handoff queued (only for predicting a fixed policy:
                # the harness admits as soon as it can, so an optimized schedule must not choose to wait)
                model.Add(decode_starts[j] >= prefill_ends[j] + q["kv_transfer_time"][j] * splits[j])
            else:
                model.Add(decode_starts[j] == prefill_ends[j] + q["kv_transfer_time"][j] * splits[j])
            model.Add(ends[j] == decode_starts[j] + decode)
            # Memory: the decode worker holds isl + osl tokens from the start (colocated) or the prefill end
            # to the end; if split, the prefill worker holds the prompt's isl until the
            # transfer is done
            life_start = model.NewIntVar(0, horizon, f"life_start[{j}]")
            model.Add(life_start == starts[j]).OnlyEnforceIf(splits[j].Not())
            # (a handoff queued for a decode slot is still held by the prefill worker: see `held`)
            model.Add(life_start == (decode_starts[j] if DECODE_MAY_WAIT else prefill_ends[j])).OnlyEnforceIf(splits[j])
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
                coef = round(S * q["chunk_ms"][b] / (q["chunk_ms"][b] + q["t0"][a])) if PD_FUSED else S
                pd_terms[a].append(coef * decode_prefill_overlap_len)
                pair_terms["pd", b, a] = (coef, decode_prefill_overlap_len)
                if EVENT_MS:
                    coef = round(S * EVENT_MS)
                    pd_terms[a].append(coef * decoding_prefill_overlap)
                    pair_terms["pdev", b, a] = (coef, decoding_prefill_overlap)
                # D->P: a's decode part is in each of b's chunk steps
                coef = round(S * DP_SCALE * q["delta"][a])
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
                # overlap ms -> overlapped steps of x: / (t0 + delta) by default, or / DD_STEP_MS (a typical step at
                # the operating point) when the step model is hinged and t0 is only the linear branch's intercept
                coef = round(S * q["delta"][y] / (DD_STEP_MS or (q["t0"][x] + q["delta"][y])))
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
        finished_decode, finished_prefill_slowdown, finished_any, finished_pd = \
            (tuple(ctx.finished_interference.get(j, ())) + (0, 0, False, 0)[len(ctx.finished_interference.get(j, ())):])
        # decode = raw decode time + p-d and d-d interference
        rhs = round(S * q["decode_steps"][j] * q["t0"][j]) + sum(dd_terms[j]) + sum(pd_terms[j]) + finished_decode
        rhs += round(S * q["t0"][j]) * splits[j]
        if FLAT_MS:
            # Flat branch: every step costs FLAT_MS, no D-D interference; P->D still applies
            flat = round(S * q["decode_steps"][j] * FLAT_MS) + sum(pd_terms[j]) + finished_pd
            flat += round(S * FLAT_MS) * splits[j]
            lin_v = model.NewIntVar(0, S * horizon, f"decode_lin[{j}]")
            flat_v = model.NewIntVar(0, S * horizon, f"decode_flat[{j}]")
            model.Add(lin_v == rhs)
            model.Add(flat_v == flat)
            both = model.NewIntVar(0, S * horizon, f"decode_hinge[{j}]")
            model.AddMaxEquality(both, [lin_v, flat_v])
            rhs = both
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
        rhs = q["chunks"][j] * (round(S * DP_SCALE * (steps.f1 - steps.s)) * any_decoding + sum(dp_terms[j])
                                + finished_prefill_slowdown)
        if FLAT_MS:
            lin_v = model.NewIntVar(0, S * horizon, f"pslow_lin[{j}]")
            model.Add(lin_v == rhs)
            flat_v = model.NewIntVar(0, S * horizon, f"pslow_flat[{j}]")
            model.Add(flat_v == q["chunks"][j] * round(S * DP_SCALE * FLAT_MS) * any_decoding)
            both = model.NewIntVar(0, S * horizon, f"pslow_hinge[{j}]")
            model.AddMaxEquality(both, [lin_v, flat_v])
            rhs = both
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
    for j, (key, pe, route, end, _, max_reused_prefix) in ctx.finished_hit_sources.items():
        users.setdefault(key, (route, end, max_reused_prefix, []))[3].append(j)
    for key, (route, end, max_reused_prefix, js) in users.items():
        used = model.NewBoolVar(f"ext_used[{key}]")
        model.AddMaxEquality(used, [ext_reuses[j] for j in js])
        evict = model.NewIntVar(end, horizon, f"ext_evict[{key}]")
        for j in js:
            model.Add(evict >= starts[j]).OnlyEnforceIf(ext_reuses[j])
        length = model.NewIntVar(0, horizon, f"_ext_evict[{key}]")
        intervals[route].append(model.NewOptionalIntervalVar(end, length, evict, used, f"ext_eviction[{key}]"))
        sizes[route].append(max_reused_prefix)
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


@dataclass
class PartialHint:
    # route[l]: the cache request l prefills on
    route: list
    # droute[l]: the cache it decodes on, with topology "split" (None: colocated)
    droute: list = None


# get a window's routing hint: use previous decisions, else follow the cache-hit source, else the least busy cache
def get_cache_routing_partial_hint(state, info, active, index, ctx, C, split, pools=None):
    solved_vars = state.solved_vars
    route, running, drunning = [], [0] * C, [0] * C
    droute = [solved_vars[g]["droute"] if g in solved_vars else None for g in active] if split else None
    prefill_pool, decode_pool = pools or (range(C), range(C))
    for g in active:
        if g in solved_vars:
            running[solved_vars[g]["route"]] += 1
            drunning[solved_vars[g].get("droute", solved_vars[g]["route"])] += 1
    for g in active:
        l = index[g]
        if g in solved_vars:
            route.append(solved_vars[g]["route"])
        else:
            cands = [(t, i) for i, t in info.sources[g] if i in index]
            if cands:
                i = max(cands)[1]
                route.append(route[index[i]])
            elif l in ctx.finished_hit_sources:
                route.append(ctx.finished_hit_sources[l][2])
            else:
                route.append(min(prefill_pool, key=lambda c: running[c]))
            running[route[-1]] += 1
    if split:
        for l, r in enumerate(route):
            if droute[l] is None:
                droute[l] = r if r in decode_pool else min(decode_pool, key=lambda c: drunning[c])
                drunning[droute[l]] += 1
    return PartialHint(route, droute)


def solve_rolling(trace, blk_size, C, mem, window_size, window_step, window_solving_time_limit,
                  step_latency=default_steps, horizon=None, window_lookahead_time_limit=30_000, log=print,
                  topology="colocated", pools=None, policy=None, warm_policy=None):
    # policy: None optimizes routing and starts. A fixed policy instead pins them and the solver only predicts the
    # outcome: policy(g) -> (prefill cache, decode cache, FCFS predecessor on that prefill cache or None). Each request
    # then starts at max(release, predecessor's prefill end): dispatched on arrival, served FCFS, never held.
    # warm_policy: optimize, but first solve the window under this fixed policy and hint every variable with that
    # solution, so the window's result is never worse (by the model) than following the policy.
    N = len(trace)
    split = topology == "split"

    # Everything run back to back, at 3x its uninterrupted decode time
    horizon = horizon or int(max(u.arrival or 0 for u in trace) + sum(
        u.think + step_latency(u.isl, 0, blk_size) + 3 * u.osl * (step_latency.f1 + step_latency.k * (u.isl + u.osl))
        for u in trace))
    
    info = session_info(trace, blk_size)

    state = RollingState()

    solved_vars = state.solved_vars

    while True:
        # New requests: earliest estimated release among those whose predecessor is solved
        pool = [j for j in range(N) if j not in solved_vars and (trace[j].prev is None or trace[j].prev in solved_vars)]
        # release time of each request in the pool
        rel = {j: trace[j].arrival if trace[j].prev is None else solved_vars[trace[j].prev]["end"] + trace[j].think
               for j in pool}

        # If all requests have been committed, return the schedule
        if not pool and not state.uncommitted:
            return solved_vars


        # `advance_states` updates the `RollingState` state from the previous window to the current one.
        time_frontier = advance_states(state, trace, info, pool, step_latency, blk_size, split)
        # time_frontier: the earliest time (ms) at which any request whose start isn't fixed yet can start. 
        # This moves ahead as requests start and finish.
        # The number of requests in the sliding window is limited both by `window_size`
        # and the time lookahead from this frontier (`window_lookahead_time_limit`).

        # Add requests released within `window_lookahead_time_limit` of the frontier.
        # Without it, later sessions' first turns fill the window and are committed far ahead of the
        # frontier, and sit in the window until it catches up.
        near = sorted((j for j in pool if rel[j] <= time_frontier + window_lookahead_time_limit),
                      key=lambda j: (rel[j], j))
        new = near[:max(0, window_size - len(state.uncommitted))]
        # add atleast one new request in the window, if the window is empty.
        # ensure progress
        if not new and not state.uncommitted:
            new = sorted(pool, key=lambda j: (rel[j], j))[:1]
        # Requests that are going to be solved in the current window
        active = sorted((state.committed - state.finished) | set(state.uncommitted) | set(new))

        # Build the window's trace and `Context` that will be passed to build model
        local_trace, ctx, global_index_to_local_index = build_window_context(state, trace, info, active, new, rel,
                                                                             split, Context())

        # Hint: the window's routing (previous decisions, else follow the cache-hit source, else the least busy cache)
        # and the previous windows' starts; CP-SAT finds everything else
        partial_hint = get_cache_routing_partial_hint(state, info, active, global_index_to_local_index, ctx, C, split,
                                                      pools)
        model, v = build_model(local_trace, blk_size, C, mem, horizon, step_latency, ctx=ctx, topology=topology,
                               pools=pools)
        def pin(model, v, policy):
            for g in active:
                l = global_index_to_local_index[g]
                r, d, pred = policy(g)
                model.Add(v["routes"][l][r] == 1)
                if split:
                    model.Add(v["droutes"][l][d] == 1)
                rel_l = v["releases"][l]
                if pred is None:
                    model.Add(v["starts"][l] == rel_l)
                elif pred in global_index_to_local_index:
                    model.AddMaxEquality(v["starts"][l], [rel_l, v["prefill_ends"][global_index_to_local_index[pred]]])
                else:
                    model.Add(v["starts"][l] == max(rel_l, solved_vars[pred]["prefill_end"]))

        def hint(model, v):
            for g in active:
                l = global_index_to_local_index[g]
                if g in solved_vars:
                    model.AddHint(v["starts"][l], solved_vars[g]["start"])
                model.AddHint(v["routes"][l][partial_hint.route[l]], 1)
                if split:
                    model.AddHint(v["droutes"][l][partial_hint.droute[l]], 1)

        if policy is not None:
            pin(model, v, policy)
        hint(model, v)
        warm_obj = None
        if warm_policy is not None:
            # Same build, so the same variables in the same order; the policy only adds constraints
            m0, v0 = build_model(local_trace, blk_size, C, mem, horizon, step_latency, ctx=ctx, topology=topology,
                                 pools=pools)
            pin(m0, v0, warm_policy)
            hint(m0, v0)
            s0 = cp_model.CpSolver()
            s0.parameters.max_time_in_seconds = window_solving_time_limit
            st0 = s0.Solve(m0)
            if st0 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                assert len(m0.Proto().variables) == len(model.Proto().variables)
                warm_obj = s0.ObjectiveValue()
                model.ClearHints()
                for i in range(len(model.Proto().variables)):
                    model.AddHint(model.GetIntVarFromProtoIndex(i), s0.Value(m0.GetIntVarFromProtoIndex(i)))
                log(f"  warm start: policy solution {s0.StatusName(st0)} in {s0.WallTime():.1f} s, objective {warm_obj:.0f}")
            else:
                log(f"  warm start: policy window {s0.StatusName(st0)}, optimizing without it")

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = window_solving_time_limit
        status = solver.Solve(model)
        if warm_obj is not None:
            log(f"  optimized objective {solver.ObjectiveValue():.0f} vs policy {warm_obj:.0f} "
                f"({solver.ObjectiveValue() / warm_obj - 1:+.1%})")
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(f"window of {len(active)} requests: {solver.StatusName(status)}")
        obj, bound = solver.ObjectiveValue(), solver.BestObjectiveBound()
        V = solver.Value

        for g in active:
            l = global_index_to_local_index[g]
            s = next((active[i] for i in range(len(active)) if V(v["reuses"][i][l])), None)
            if s is None and l in v["ext_reuses"] and V(v["ext_reuses"][l]):
                s = ctx.finished_hit_sources[l][0]
            r = v["releases"][l]
            solved_vars[g] = dict(route=next(c for c in range(C) if V(v["routes"][l][c])), start=V(v["starts"][l]),
                                  src=s, prefill_slowdown=V(v["prefill_slowdowns"][l]), decode=V(v["decodes"][l]),
                                  prefill_end=V(v["prefill_ends"][l]),
                                  end=V(v["ends"][l]), release=r if isinstance(r, int) else V(r))
            if split:
                solved_vars[g] |= dict(droute=next(c for c in range(C) if V(v["droutes"][l][c])),
                                       decode_start=V(v["decode_starts"][l]))
        state.last_window_state = (global_index_to_local_index, v, solver)
        # Commit the uncommitted requests that start first.
        # If with no new requests left to add, commit everything
        pending = sorted(state.uncommitted + new, key=lambda j: (solved_vars[j]["start"], j))
        more = any(j not in solved_vars for j in range(N))
        if more:
            # Commit window_step no of requests within half of window_lookahead_time_limit of the frontier.
            # atleast one request is always committed to ensure progress
            commit = [j for j in pending
                      if solved_vars[j]["start"] <= time_frontier + window_lookahead_time_limit / 2][:window_step] \
                or pending[:1]
        else:
            commit = pending
        state.committed |= set(commit)
        state.uncommitted = [j for j in pending if j not in state.committed]
        log(f"Window at {time_frontier / 1000:.0f} s: {len(active)} requests ({len(new)} new, "
            f"{len(active) - len(pending)} committed), {solver.StatusName(status)} in {solver.WallTime():.1f} s, "
            f"gap {(obj - bound) / max(obj, 1):.1%}; {len(solved_vars)}/{N} solved")
        if not more and not state.uncommitted:
            return solved_vars
