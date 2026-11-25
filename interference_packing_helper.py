# Helpers for interference_packing's rolling solve (solve_rolling)
from dataclasses import dataclass, field
from cache_packing import Request, shared_tokens


# what the rolling solve needs about a trace's sessions (built once by session_info)
@dataclass
class SessionInfo:
    # session[j]: the session id of request j (the index of its session's first turn)
    session: list
    # sources[j] = [(i, tokens), ...]: possible cache-hit sources of request j, any earlier request i in the same
    #   session which shares a prefix of `tokens` with it (hash ids differ across sessions)
    sources: dict
    # max_reused_prefix[i]: the largest prefix (tokens) any later request reuses from i, which i's KV keeps in memory
    #   after i ends (its eviction interval); passed on in finished_hit_sources once i is finished
    max_reused_prefix: list


def session_info(trace, blk_size):
    N = len(trace)
    # assign Session IDs to the individual session
    session = []
    for u in trace:
        session.append(len(session) if u.prev is None else session[u.prev])
    # Map from session id to all the requests in that session
    members = {}
    for j, s in enumerate(session):
        members.setdefault(s, []).append(j)
    sources = {j: [(i, t) for i in members[session[j]] if i < j and (t := shared_tokens(trace[i], trace[j], blk_size)) > 0]
               for j in range(N)}
    max_reused_prefix = [0] * N
    for j in range(N):
        for i, t in sources[j]:
            max_reused_prefix[i] = max(max_reused_prefix[i], t)
    return SessionInfo(session, sources, max_reused_prefix)


# The rolling solve's state, carried from window to window.
# `advance_states` moves requests' states forward between windows.
# `build_window_context` builds each window's Context to pass to `build_model` from the RollingState.
@dataclass
class RollingState:
    # solved_vars[j] = records the values of the variables solved for request j in the rolling window solve.
    #   This is also returned as the schedule found by the rolling window solve at the end.
    solved_vars: dict = field(default_factory=dict)

    # A request's state (see Context) moves from uncommitted -> committed -> started -> finished.

    # uncommitted = requests in the current window whose decisions (cache, cache-hit source, timing)
    #   are all still not solved.
    #   At the start of each window, new requests are added in the current window until uncommitted + new = window_size.
    uncommitted: list = field(default_factory=list)

    # committed = {j, ...}: requests whose cache (and decode cache) and cache-hit source are fixed
    #   their timing is still not solved. After each window, the `window_step`
    #   number of uncommitted requests that start first are committed.
    committed: set = field(default_factory=set)

    # started = {j, ...}: committed requests whose start is also fixed.
    #   Their end can still move with interference from new requests.
    #   A committed request is started once its previous turn (the parent request in its session) is finished.
    #   Before that, new requests may slow the previous turn and so push back its release.
    started: set = field(default_factory=set)

    # finished = {j, ...}: started requests that end before any request of another session can start, so nothing new
    #   can overlap them.
    #   They leave the windows, and their solved_vars are final.
    #   Some of the their information which is still required is carried in Context.
    finished: set = field(default_factory=set)

    # finished_interference[j] = (decode, prefill_slowdown, any, decode's P->D part)
    #   for a request j that isn't finished: finished_interference records the interference that finished requests caused on this request j,
    #   This is also passed on as ctx.finished_interference in `Context`
    #       decode  interference time added to j's decode
    #       prefill_slowdown  interference time added to each of j's prefill chunks
    #       any     whether any of them was decoding during j's prefill
    #   decode and prefill_slowdown are scaled by S like the model's terms as these can be fractions of the single step latency.
    finished_interference: dict = field(default_factory=dict)

    # map from a cache/worker id -> latest end of a finished requests on this cache/worker
    cache_free_at: dict = field(default_factory=dict)

    # previous window state: (index map, vars, solver)
    last_window_state: tuple = None


# Moves the RollingState forward between two windows, updating it in place. 
#   The started request that cannot overlap with anything are marked as finished.
#   The committed requests whose previous turn is finished is marked as started. 
# The function also returns a time_frontier: the earliest time (trace time, ms) at which any request whose start isn't fixed yet can start.
# solve_rolling adds new requests released within window_lookahead_time_limit of it.
def advance_states(state, trace, info, pool, step_latency, blk_size, split):
    solved_vars, committed, started, finished = state.solved_vars, state.committed, state.started, state.finished
    finished_interference, cache_free_at = state.finished_interference, state.cache_free_at
    session, sources = info.session, info.sources
    while True:
        size_started = len(started)
        lb = {}
        for j in sorted(set(state.uncommitted) | set(pool) | (committed - started)):
            u = trace[j]
            if u.prev is None:
                lb[j] = u.arrival
                continue
            p = trace[u.prev]
            if u.prev in started:
                hit = next((t for i, t in sources[u.prev] if i == solved_vars[u.prev]["src"]), 0)
                start = solved_vars[u.prev]["start"]
            else:
                hit = max([t for _, t in sources[u.prev]], default=0)
                start = lb[u.prev]
            lb[j] = start + step_latency(p.isl, hit, blk_size) \
                + (p.osl - 1) * (step_latency.f1 + step_latency.k * (p.isl + p.osl / 2)) + u.think - 1
        # The earliest any request whose start isn't fixed can start (see above)
        time_frontier = min(lb.values())
        # A request is finished once it ends before any request of another session can start (its own session's
        # later turns can't overlap it, and comparing against them would deadlock: its successor's bound comes from
        # its own earliest end)
        first = {}
        for j, t in lb.items():
            first[session[j]] = min(first.get(session[j], t), t)
        others = lambda s: min([t for s2, t in first.items() if s2 != s], default=float("inf"))
        newly = [j for j in started if j not in finished and solved_vars[j]["end"] <= others(session[j])]
        # Two overlapping sessions can block each other: each one's next turn may (by its bound) start before the
        # other's running request ends, so neither finishes and the frontier stalls on a committed request that
        # can't start. Then compare against solved starts instead, and failing that finish the running request
        # that ends first. A later request can then wait for it, but only on its cache.
        if not newly and min(lb, key=lambda j: (lb[j], j)) in committed - started:
            est = {j: solved_vars[j]["start"] if j in solved_vars else t for j, t in lb.items()}
            first2 = {}
            for j, t in est.items():
                first2[session[j]] = min(first2.get(session[j], t), t)
            newly = [j for j in started if j not in finished and solved_vars[j]["end"] <= min(
                [t for s2, t in first2.items() if s2 != session[j]], default=float("inf"))]
            running = [j for j in started if j not in finished]
            if not newly and running:
                newly = [min(running, key=lambda j: (solved_vars[j]["end"], j))]
        # Record the finished requests' interference on the committed requests they overlapped
        # (finished_interference). Only started ones keep it (see below)
        if state.last_window_state is not None and newly:
            index, lv, ls = state.last_window_state
            for u in committed:
                if u in finished or u in newly:
                    continue
                d, x, a, p = finished_interference.get(u, (0, 0, False, 0))
                for f in newly:
                    # (CHANGED for P-D split: f and u can share a cache through either phase; terms are 0 when
                    # they share none)
                    if ((not split and solved_vars[f]["route"] != solved_vars[u]["route"]) or f not in index
                            or u not in index):
                        continue
                    fu, uu = index[f], index[u]
                    for key in (("dd", uu, fu), ("pd", fu, uu), ("pdev", fu, uu)):
                        if key in lv["pair_terms"]:
                            coef, var = lv["pair_terms"][key]
                            d += coef * ls.Value(var)
                            if key[0] != "dd":          # the P->D part alone, for the flat decode branch
                                p += coef * ls.Value(var)
                    if ("dp", fu, uu) in lv["pair_terms"]:
                        coef, var = lv["pair_terms"]["dp", fu, uu]
                        x += coef * ls.Value(var)
                        a = a or bool(ls.Value(var))
                finished_interference[u] = (d, x, a, p)
        for f in newly:
            finished.add(f)
            r, d = solved_vars[f]["route"], solved_vars[f].get("droute", solved_vars[f]["route"])
            # a split request is done on its prefill cache at its prefill end
            cache_free_at[r] = max(cache_free_at.get(r, 0),
                                   solved_vars[f]["end"] if r == d else solved_vars[f]["prefill_end"])
            cache_free_at[d] = max(cache_free_at.get(d, 0), solved_vars[f]["end"])
            finished_interference.pop(f, None)
        # A committed request's start is fixed once its predecessor is finished: before that, new requests may slow
        # the predecessor, and a fixed start would leave them waiting until it ends
        started.update(j for j in committed if trace[j].prev is None or trace[j].prev in finished)
        if not newly and len(started) == size_started:
            break
    # A request that isn't started has to start after the finished requests on its cache end (cache_free_at), so
    # it won't overlap them: drop what it overlapped them by in the last solve (nonzero only after a forced
    # finish). A started one keeps its start, and so its overlaps.
    for g in [g for g in finished_interference if g not in started]:
        del finished_interference[g]
    return time_frontier


# Builds current window's trace and Context for build_model from RollingState. 
# Returns (local_trace, ctx, global_index_to_local_index): the window's trace for build_model, its
# Context, and the map from each active request's trace (global) index to its window (local) index.
def build_window_context(state, trace, info, active, new, rel, split, ctx):
    solved_vars, committed, started, finished = state.solved_vars, state.committed, state.started, state.finished
    global_index_to_local_index = {g: l for l, g in enumerate(active)}
    local_trace = []
    for g in active:
        u = trace[g]
        if u.prev is None or u.prev in global_index_to_local_index:
            local_trace.append(Request(u.isl, u.osl, u.arrival, u.hash_ids, global_index_to_local_index.get(u.prev),
                                       u.think))
        else:
            local_trace.append(Request(u.isl, u.osl, solved_vars[u.prev]["end"] + u.think, u.hash_ids, None, 0))
        l = global_index_to_local_index[g]
        # One cache-hit source among finished requests: the one already chosen, else the largest hit (the
        # session's latest prompt, in practice)
        ext = [(t, i) for i, t in info.sources[g] if i in finished]
        if g in solved_vars and solved_vars[g]["src"] in finished:
            ext = [(t, i) for t, i in ext if i == solved_vars[g]["src"]]
        if ext:
            t, i = max(ext)
            ctx.finished_hit_sources[l] = (i, solved_vars[i]["prefill_end"], solved_vars[i]["route"],
                                           solved_vars[i]["end"], t, info.max_reused_prefix[i])
        # (only started requests have one: advance_states drops the others')
        if g in state.finished_interference:
            ctx.finished_interference[l] = state.finished_interference[g]
        if g in committed:
            src = solved_vars[g]["src"]
            ctx.committed_route_and_hit[l] = (solved_vars[g]["route"],
                                              global_index_to_local_index[src] if src in global_index_to_local_index
                                              else "ext" if src is not None else None)
            if split:
                ctx.committed_droute[l] = solved_vars[g]["droute"]
        if g in started:
            ctx.fixed_start[l] = solved_vars[g]["start"]
        elif state.cache_free_at:
            ctx.cache_free_at[l] = dict(state.cache_free_at)
    T0 = min([solved_vars[g]["start"] for g in active if g in solved_vars] + [rel[j] for j in new])
    for f in finished:
        if solved_vars[f]["prefill_end"] > T0:
            ctx.finished_prefills.append((solved_vars[f]["route"], solved_vars[f]["start"],
                                          solved_vars[f]["prefill_end"]))
        r, d = solved_vars[f]["route"], solved_vars[f].get("droute", solved_vars[f]["route"])
        if solved_vars[f]["end"] > T0 and r == d:
            ctx.finished_kv_lifetimes.append((r, solved_vars[f]["start"], solved_vars[f]["end"],
                                              trace[f].isl + trace[f].osl))
        elif solved_vars[f]["end"] > T0:
            # the decode cache holds it from the prefill end, the prefill cache until the transfer
            ctx.finished_kv_lifetimes.append((d, solved_vars[f]["prefill_end"], solved_vars[f]["end"],
                                              trace[f].isl + trace[f].osl))
            if solved_vars[f]["decode_start"] > T0:
                ctx.finished_kv_lifetimes.append((r, solved_vars[f]["start"], solved_vars[f]["decode_start"],
                                                  trace[f].isl))
    return local_trace, ctx, global_index_to_local_index
