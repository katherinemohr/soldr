"""Time-indexed ILP for trace-latency routing with continuous batching and prefix (KV) caching.

Model (per worker w, steps k = 1..K of that worker's own engine clock; one model step = G engine iterations):
  Y[i,w,k] in {0,1}  request i has been prefilled on w by step k (nondecreasing in k; x[i,w] = Y[i,w,K]).
                     y = Y[k] - Y[k-1] is the start pulse. A request occupies n_i = ceil(o_i / G) consecutive
                     steps (prefill in the first, decode in the rest, no chunking, no preemption), so
                     act[i,w,k] = Y[k] - Y[k-n_i] and every batching quantity is a 2-term difference.
  u[w,k] in {0,1}    step k is non-empty; nonincreasing in k (empty steps WLOG sit at the tail, since idle
                     time is absorbed by slack in the step start times).
  S[w,k], E[w,k]     wall-clock start / end of step k (ms). E = S + d, S[k] >= E[k-1], S[k] >= r_i Y[i,w,k].
  d[w,k] = a * P[w,k] + G * (c * u[w,k] + b * sum_i act[i,w,k])        (Qwen2.5-32B, 2x H100 TP=2, below)
  P[w,k] = sum_i p_i y[i,w,k] - sum_{i,d} dS[i,d] q[i,d,w,k]           (uncached prefill tokens)
  capacity           sum_i act[i,w,k] <= B u[w,k]
  FCFS               chain G[i,w,k] = "all requests <= i routed to w have started by k" (continuous):
                     G[i] <= G[i-1], G[i] <= Y[i] + 1 - x[i], Y[i] <= G[i-1]       O(NWK), not O(N^2 WK)
  KV cache           Mooncake hash_ids are prefix-chained (a trie), so the cached prefix of i is the deepest
                     cached node on its chain, and only the depths D_i = {lcp(i,l) : l < i} matter (1-3 per
                     request on Mooncake). TTL of C steps, refreshed on every use (hit or miss), so presence
                     is a window sum with no recursion through hit variables:
                       pi[h,w,k] <= sum_{l contains h} (Y[l,w,k-1] - Y[l,w,k-1-C]),  pi <= 1
                       q[i,d,w,k] <= y[i,w,k],  q[i,d,w,k] <= pi[h_{i,d},w,k],  q[i,d] <= q[i,d-]
                     q and pi are continuous: they only shorten steps, so the optimum sits on their upper
                     bounds, which are integral once Y is. Caching adds no binaries.
  objective          mean e2e latency (default) or TTFT: C_i >= E[w,k+m-1] - H (1 - x[i,w] + Y[i,w,k-1])
                     ("if i starts on w at or after k it ends no earlier than E[k+m-1]"; m = n_i or 1).
  --work-conserving  adds arrival binaries al[i,w,k] (S[w,k] - r_i + eps <= H al) and
                     sum_j act[j,w,k] >= B (al + x - Y - 1): an arrived, queued request forces a full batch.
                     Without it the ILP may hold a request back (to win a cache hit), i.e. it optimizes
                     admission too, so its objective is optimistic for the real FCFS engine; every routing
                     is therefore re-scored with the step simulator below, which is work-conserving.
Horizon: K steps per worker and H ms (big-M) come from the baseline routings' simulated schedules (x slack);
the best baseline schedule is the MIP start. Workers are identical: symmetry broken by first-use order.

Latency coefficients: aisimulate op-level estimates (h100_sxm / vllm "current" data, ModelConfig(tp_size=2),
Qwen/Qwen2.5-32B-Instruct, backend.run_mixed). Prefill-only: 2k tok 106 ms, 8k 464 ms, 16k 947 ms;
decode-only at 8k (16k) context: 1 seq 14.3 (14.6) ms, 32 seqs 25.3 (36.2) ms. Hence c = 14 ms/iteration,
b = 0.5 ms per decoding sequence (Mooncake contexts are ~8-16k), a = 0.055 ms per prefill token (exact at 8k,
+20% at 2k, -3% at 16k). TTL: ~270k KV tokens/worker free after weights (256 KiB/token), minus ~100k in flight,
at ~10k new tokens/s/worker => ~18 s retention ~= 450 iterations at ~25 iterations/s.

Usage:
  ~/.venv/bin/python timeindexed_routing.py --trace traces/mooncake_conversation.jsonl --N 50 --N 100 \
      [--W 4] [--B 32] [--G 32] [--ttl-iters 448] [--start 3000] [--obj e2e|ttft] [--time-limit 300] [--log]
"""
import argparse, json, math, time
from collections import deque

BLOCK = 512                       # Mooncake hash block size (tokens)
A_MS, B_MS, C_MS = 0.055, 0.5, 14.0


def load(path, start, N):
    rs = [json.loads(l) for l in open(path)][start:start + N]
    t0 = rs[0]["timestamp"]
    return [dict(r=float(q["timestamp"] - t0), p=q["input_length"], o=q["output_length"], h=q["hash_ids"])
            for q in rs]


def lcp(a, b):
    k = 0
    while k < len(a) and k < len(b) and a[k] == b[k]:
        k += 1
    return k


def saved_tokens(p, depth):
    return min(BLOCK * depth, p - 1) if depth else 0


# ----------------------------------------------------------------------------- step simulator
class Worker:
    """FCFS continuous batching on one worker, G iterations per step, TTL-C prefix cache (refresh on use)."""

    def __init__(self, reqs, B, G, C):
        self.R, self.B, self.G, self.C = reqs, B, G, C
        self.q, self.act = deque(), []            # waiting request ids; [id, remaining steps]
        self.t, self.k, self.cache = 0.0, 0, {}   # end of last step, steps executed, block -> last step used

    def next_start(self):
        if self.act: return self.t
        if self.q: return max(self.t, self.R[self.q[0]]["r"])
        return math.inf

    def cached_depth(self, i, k):
        d = 0
        for h in self.R[i]["h"]:
            if self.cache.get(h, -math.inf) < k - self.C: break
            d += 1
        return d

    def step(self, out):
        t, k = self.next_start(), self.k + 1
        new = []
        while self.q and len(self.act) + len(new) < self.B and self.R[self.q[0]]["r"] <= t:
            new.append(self.q.popleft())
        P = 0
        for i in new:                               # hits are computed before any block of this step is stored
            s = saved_tokens(self.R[i]["p"], self.cached_depth(i, k))
            P += self.R[i]["p"] - s
            out[i].update(start=k, saved=s)
        for i in new:
            for h in self.R[i]["h"]: self.cache[h] = k
        self.act += [[i, math.ceil(self.R[i]["o"] / self.G)] for i in new]
        dur = A_MS * P + self.G * (C_MS + B_MS * len(self.act))
        end = t + dur
        for i in new: out[i]["ttft"] = end - self.R[i]["r"]
        for a in self.act:
            a[1] -= 1
            if a[1] == 0: out[a[0]]["e2e"] = end - self.R[a[0]]["r"]
        self.act = [a for a in self.act if a[1] > 0]
        self.t, self.k = end, k


def simulate(reqs, policy, W, B, G, ttl_iters):
    """Online replay. policy(i, workers) -> worker index, called at r_i after every worker has run all steps
    that start strictly before r_i. Returns per-request dicts and the per-worker step counts."""
    C = math.ceil(ttl_iters / G)
    ws = [Worker(reqs, B, G, C) for _ in range(W)]
    out = [dict(worker=None) for _ in reqs]
    for i, q in enumerate(reqs):
        for w in ws:
            while w.next_start() < q["r"]: w.step(out)
        wi = policy(i, ws); out[i]["worker"] = wi; ws[wi].q.append(i)
    for w in ws:
        while w.next_start() < math.inf: w.step(out)
    return out, [w.k for w in ws]


def summarize(out, reqs):
    n = len(out)
    return dict(e2e=sum(o["e2e"] for o in out) / n / 1000, ttft=sum(o["ttft"] for o in out) / n / 1000,
                hit=sum(o["saved"] for o in out) / sum(q["p"] for q in reqs),
                mk=max(q["r"] + o["e2e"] for q, o in zip(reqs, out)) / 1000)


def replay(reqs, starts, W, G, C, key):
    """Re-time a fixed ILP schedule (starts[i] = (w, k)) with the simulator's step semantics and the earliest
    feasible step starts. Every ILP solution must score no better than this (q and S slack only hurt)."""
    n = [math.ceil(q["o"] / G) for q in reqs]
    tot = 0.0
    for w in range(W):
        mine = [i for i in range(len(reqs)) if starts[i][0] == w]
        if not mine: continue
        K = max(starts[i][1] + n[i] - 1 for i in mine)
        cache, E, end = {}, 0.0, {}
        for k in range(1, K + 1):
            new = [i for i in mine if starts[i][1] == k]
            act = sum(1 for i in mine if starts[i][1] <= k < starts[i][1] + n[i])
            P = 0
            for i in new:
                d = 0
                for h in reqs[i]["h"]:
                    if cache.get(h, -math.inf) < k - C: break
                    d += 1
                P += reqs[i]["p"] - saved_tokens(reqs[i]["p"], d)
            for i in new:
                for h in reqs[i]["h"]: cache[h] = k
            S = max([E] + [reqs[i]["r"] for i in new])
            E = S + A_MS * P + G * ((C_MS if act else 0) + B_MS * act)
            end[k] = E
        for i in mine:
            tot += end[starts[i][1] + (n[i] if key == "e2e" else 1) - 1] - reqs[i]["r"]
    return tot / len(reqs) / 1000


def policies(reqs):
    def rr(i, ws): return i % len(ws)
    def jsq(i, ws): return min(range(len(ws)), key=lambda w: len(ws[w].act) + len(ws[w].q))
    def kv(i, ws):                                # Dynamo KV-router style: new prefill blocks + active blocks
        nb = math.ceil(reqs[i]["p"] / BLOCK)
        def cost(w):
            W_ = ws[w]
            load = sum(math.ceil(reqs[j]["p"] / BLOCK) for j in [a[0] for a in W_.act] + list(W_.q))
            return nb - W_.cached_depth(i, W_.k + 1) + load
        return min(range(len(ws)), key=cost)
    return dict(rr=rr, jsq=jsq, kv=kv)


# ----------------------------------------------------------------------------- ILP
def branch_depths(reqs):
    """D_i: distinct nonzero lcp(i, l) over earlier requests l < i (ascending)."""
    D = []
    for i, q in enumerate(reqs):
        s = set()
        for l in range(i):
            d = lcp(q["h"], reqs[l]["h"])
            if d: s.add(d)
        D.append(sorted(s))
    return D


def build(reqs, W, B, G, C, K, H, obj, wc, threads, log):
    import gurobipy as gp
    from gurobipy import GRB
    N = len(reqs)
    r = [q["r"] for q in reqs]; p = [q["p"] for q in reqs]
    n = [math.ceil(q["o"] / G) for q in reqs]
    Ki = [K - n[i] + 1 for i in range(N)]          # latest start step that still finishes within K
    assert min(Ki) >= 1, "horizon K shorter than the longest request"
    m = gp.Model("timeindexed"); m.Params.OutputFlag = 1 if log else 0; m.Params.Threads = threads

    Y = {(i, w, k): m.addVar(vtype=GRB.BINARY, name=f"Y[{i},{w},{k}]")
         for i in range(N) for w in range(W) for k in range(1, Ki[i] + 1)}
    def Yx(i, w, k):                                # Y with the k <= 0 and k >= Ki conventions folded in
        if k <= 0: return 0
        return Y[i, w, min(k, Ki[i])]
    x = {(i, w): Y[i, w, Ki[i]] for i in range(N) for w in range(W)}
    for (i, w, k), v in Y.items():
        if k > 1: m.addConstr(Y[i, w, k - 1] <= v)
    m.addConstrs((gp.quicksum(x[i, w] for w in range(W)) == 1 for i in range(N)), name="assign")
    m.addConstr(x[0, 0] == 1)                      # identical workers: worker w opens only after w-1
    for w in range(1, W):
        for i in range(1, N):
            m.addConstr(x[i, w] <= gp.quicksum(x[j, w - 1] for j in range(i)))

    u = m.addVars(W, range(1, K + 1), vtype=GRB.BINARY, name="u")
    S = m.addVars(W, range(1, K + 1), lb=0, ub=H, name="S")
    E = m.addVars(W, range(1, K + 1), lb=0, ub=H, name="E")
    Gc = m.addVars(N, W, range(1, K + 1), lb=0, ub=1, name="G")

    # --- cache: branch depths, trie nodes, presence pi, hit q
    D = branch_depths(reqs)
    dS = [[saved_tokens(p[i], d) - saved_tokens(p[i], dp) for d, dp in zip(D[i], [0] + D[i][:-1])]
          for i in range(N)]
    nodes = {reqs[i]["h"][d - 1] for i in range(N) for d in D[i]}
    hs = [set(q["h"]) for q in reqs]
    holders = {h: [l for l in range(N) if h in hs[l]] for h in nodes}
    pi = m.addVars(list(nodes), W, range(1, K + 1), lb=0, ub=1, name="pi")
    for h in nodes:
        for w in range(W):
            for k in range(1, K + 1):
                if k == 1: m.addConstr(pi[h, w, k] == 0); continue
                m.addConstr(pi[h, w, k] <= gp.quicksum(Yx(l, w, k - 1) - Yx(l, w, k - 1 - C) for l in holders[h]))
    q = {}
    for i in range(N):
        for j, d in enumerate(D[i]):
            h = reqs[i]["h"][d - 1]
            for w in range(W):
                for k in range(1, Ki[i] + 1):
                    v = q[i, j, w, k] = m.addVar(lb=0, ub=1, name=f"q[{i},{j},{w},{k}]")
                    m.addConstr(v <= Yx(i, w, k) - Yx(i, w, k - 1))
                    m.addConstr(v <= pi[h, w, k])
                    if j: m.addConstr(v <= q[i, j - 1, w, k])

    # --- batching, durations, clocks
    for w in range(W):
        for k in range(1, K + 1):
            act = gp.quicksum(Yx(i, w, k) - Yx(i, w, k - n[i]) for i in range(N))
            m.addConstr(act <= B * u[w, k], name=f"cap[{w},{k}]")
            if k < K: m.addConstr(u[w, k + 1] <= u[w, k])
            P = gp.quicksum(p[i] * (Yx(i, w, k) - Yx(i, w, k - 1)) for i in range(N) if k <= Ki[i]) \
                - gp.quicksum(dS[i][j] * q[i, j, w, k] for i in range(N) for j in range(len(D[i])) if k <= Ki[i])
            m.addConstr(E[w, k] == S[w, k] + A_MS * P + G * (C_MS * u[w, k] + B_MS * act), name=f"dur[{w},{k}]")
            if k > 1: m.addConstr(S[w, k] >= E[w, k - 1])
            for i in range(N):
                if r[i] > 0: m.addConstr(S[w, k] >= r[i] * Yx(i, w, k))
                # FCFS chain
                if i: m.addConstr(Gc[i, w, k] <= Gc[i - 1, w, k]); m.addConstr(Yx(i, w, k) <= Gc[i - 1, w, k])
                m.addConstr(Gc[i, w, k] <= Yx(i, w, k) + 1 - x[i, w])

    # --- work conservation (optional)
    nal = 0
    if wc:
        for i in range(N):
            for w in range(W):
                for k in range(1, Ki[i]):
                    al = m.addVar(vtype=GRB.BINARY, name=f"al[{i},{w},{k}]"); nal += 1
                    m.addConstr(S[w, k] - r[i] + 1e-3 <= H * al)
                    m.addConstr(gp.quicksum(Yx(j, w, k) - Yx(j, w, k - n[j]) for j in range(N))
                                >= B * (al + x[i, w] - Y[i, w, k] - 1))

    # --- completions and objective
    Cv = m.addVars(N, lb=0, ub=H, name="C")
    for i in range(N):
        mm = n[i] if obj == "e2e" else 1
        smax = saved_tokens(p[i], D[i][-1]) if D[i] else 0
        m.addConstr(Cv[i] >= r[i] + A_MS * (p[i] - smax) + G * mm * (C_MS + B_MS))   # isolated service
        for w in range(W):
            for k in range(1, Ki[i] + 1):
                m.addConstr(Cv[i] >= E[w, k + mm - 1] - H * (1 - x[i, w] + Yx(i, w, k - 1)))
    m.setObjective(gp.quicksum(Cv[i] - r[i] for i in range(N)) * (1.0 / N / 1000), GRB.MINIMIZE)
    m.update()
    sizes = dict(Y=len(Y), u=len(u), al=nal, SE=2 * len(S), Gc=len(Gc), pi=len(pi), q=len(q), C=N,
                 depths=sum(len(d) for d in D), nodes=len(nodes))
    return m, Y, u, x, Ki, sizes


def mip_start(Y, u, out, steps, reqs, W, Ki):
    """Seed Y/u from a simulated schedule, relabeling workers by first use (symmetry-breaking order)."""
    order = []
    for o in out:
        if o["worker"] not in order: order.append(o["worker"])
    order += [w for w in range(W) if w not in order]
    lab = {w: j for j, w in enumerate(order)}
    for (i, w, k), v in Y.items():
        v.Start = 1.0 if lab[out[i]["worker"]] == w and out[i]["start"] <= k else 0.0
    for (w, k), v in u.items():
        v.Start = 1.0 if k <= steps[order[w]] else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--N", type=int, action="append", required=True, help="trace length(s); repeat for a sweep")
    ap.add_argument("--W", type=int, default=4)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--G", type=int, default=32, help="engine iterations per model step")
    ap.add_argument("--ttl-iters", type=int, default=448, help="cache TTL in engine iterations")
    ap.add_argument("--start", type=int, default=3000)
    ap.add_argument("--obj", choices=["e2e", "ttft"], default="e2e")
    ap.add_argument("--work-conserving", action="store_true")
    ap.add_argument("--k-slack", type=float, default=1.25, help="K = slack * max baseline steps per worker")
    ap.add_argument("--h-slack", type=float, default=1.5, help="H = slack * max baseline makespan")
    ap.add_argument("--time-limit", type=float, default=300)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--build-only", action="store_true", help="report model size, skip the solve")
    ap.add_argument("--log", action="store_true")
    a = ap.parse_args()
    W, B, G = a.W, a.B, a.G
    C = math.ceil(a.ttl_iters / G)
    key = a.obj
    print(f"W={W} B={B} G={G} TTL={a.ttl_iters} it (C={C} steps) a={A_MS} b={B_MS} c={C_MS} ms  obj=mean {key}"
          f"  slice@{a.start}  wc={a.work_conserving}  limit {a.time_limit:.0f}s", flush=True)
    print(f"{'N':>5} {'K':>4} {'vars':>8} {'bin':>7} {'cons':>8} {'nnz':>9} {'build':>6} {'solve':>6} "
          f"{'status':>9} {'gap':>6} | {'ILP':>6} {'simILP':>6} {'G=1':>6} | {'rr':>6} {'jsq':>6} {'kv':>6} "
          f"| hit ILP/kv   (mean {key}, s)", flush=True)
    for N in a.N:
        reqs = load(a.trace, a.start, N)
        base = {nm: simulate(reqs, pol, W, B, G, a.ttl_iters) for nm, pol in policies(reqs).items()}
        bs = {nm: summarize(o, reqs) for nm, (o, _) in base.items()}
        K = math.ceil(a.k_slack * max(max(st) for _, st in base.values()))
        K = max(K, max(math.ceil(q["o"] / G) for q in reqs))
        H = a.h_slack * max(s["mk"] for s in bs.values()) * 1000
        t0 = time.time()
        m, Y, u, x, Ki, sz = build(reqs, W, B, G, C, K, H, a.obj, a.work_conserving, a.threads, a.log)
        tb = time.time() - t0
        best = min(bs, key=lambda nm: bs[nm][key])
        mip_start(Y, u, *base[best], reqs, W, Ki)
        row = f"{N:5d} {K:4d} {m.NumVars:8d} {m.NumBinVars:7d} {m.NumConstrs:8d} {m.NumNZs:9d} {tb:5.1f}s "
        tail = f"{bs['rr'][key]:6.2f} {bs['jsq'][key]:6.2f} {bs['kv'][key]:6.2f}"
        if a.build_only:
            print(row + f"{'':>6} {'built':>9} {'':>6} | {'':>6} {'':>6} {'':>6} | " + tail + f"  sizes {sz}", flush=True)
            continue
        m.Params.TimeLimit = a.time_limit
        t0 = time.time(); m.optimize(); ts = time.time() - t0
        from gurobipy import GRB
        st = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "timelimit"}.get(m.Status, str(m.Status))
        if m.SolCount:
            route = [max(range(W), key=lambda w: x[i, w].X) for i in range(N)]
            fixed = lambda i, ws: route[i]
            starts = [(route[i], min(k for k in range(1, Ki[i] + 1) if Y[i, route[i], k].X > 0.5)) for i in range(N)]
            rp = replay(reqs, starts, W, G, C, key)
            assert rp <= m.ObjVal + 1e-6, f"ILP objective {m.ObjVal} beats its own replay {rp}"
            so = summarize(simulate(reqs, fixed, W, B, G, a.ttl_iters)[0], reqs)
            s1 = summarize(simulate(reqs, fixed, W, B, 1, a.ttl_iters)[0], reqs)
            print(f"      replay of ILP schedule {rp:.3f}s vs ILP obj {m.ObjVal:.3f}s", flush=True)
            ilp = f"{m.ObjVal:6.2f} {so[key]:6.2f} {s1[key]:6.2f}"; gap = f"{100 * m.MIPGap:5.1f}%"
            hit = f"{100 * so['hit']:4.1f}%/{100 * bs['kv']['hit']:4.1f}%"
        else:
            ilp, gap, hit = f"{'none':>6} {'':>6} {'':>6}", "   n/a", ""
        k1 = {nm: summarize(simulate(reqs, pol, W, B, 1, a.ttl_iters)[0], reqs)[key] for nm, pol in policies(reqs).items()}
        print(row + f"{ts:5.1f}s {st:>9} {gap:>6} | {ilp} | {tail} | {hit}   G=1 base rr/jsq/kv "
              f"{k1['rr']:.2f}/{k1['jsq']:.2f}/{k1['kv']:.2f}  (MIP start: {best})", flush=True)
        print(f"      sizes {sz}", flush=True)


if __name__ == "__main__":
    main()
