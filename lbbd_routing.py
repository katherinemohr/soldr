"""LBBD: optimal (min-makespan) routing of a request trace across W identical workers, solved by
logic-based Benders decomposition (Hooker 2019, Sec. 5: job assignment + scheduling, min makespan).

Same model as ilp3_routing.py (fixed step latency L, B concurrent slots per worker, FCFS admission, no
preemption, no prefix reuse) but decomposed:

  master      assignment x_iw (binary), makespan M >= M_w, subproblem relaxation, Benders cuts.
  subproblem  per worker w, the FCFS schedule of the requests routed to it. Unlike Hooker's CP
              scheduling subproblem ours has no freedom at all: it is the Kiefer-Wolfowitz recursion,
              i.e. an O(N B) simulation that returns the exact makespan M_w*(J_w).

Two properties of the FCFS/B-slot recursion give the Benders cuts (J' below is an inclusion-minimal
subset of J_w with the same makespan, found by re-simulating with requests removed, as in the paper):
  monotone   J ⊆ J'  =>  M_w(J) <= M_w(J')        (adding a request never speeds anything up)
  Lipschitz  M_w(J ∪ S) <= M_w(J) + sum_{j in S} n_j  (sup-norm 1-Lipschitz of "insert + sort + age")
so for every worker w' (workers are identical, so one subproblem proof gives W cuts):
  (L) M_w' >= M*  - sum_{j in J'} n_j (1 - x_jw')                          dominates nogood cut (13)
  (E) M_w' >= r_i + (1/B) [ backlog_i - sum_{j<=i, j in J_w} n_j (1 - x_jw') + sum_{j>i} n_j x_jw' ]
      for the arrival index i maximising the RHS at x̄: the paper's energy relaxation R(t) at t=r_i,
      strengthened by the backlog (work still queued on w at r_i) of the current schedule, valid by
      monotonicity and L1 1-Lipschitzness. Its per-request rate n_j/B is the "right" rate for B slots.
  (D) optional (--cuts ...,delta): exact for any single removal from J', M_w' >= M* - sum d_j (1 - x_jw') - K z,
      z >= sum_{J'} (1 - x_jw') - 1, with d_j = M* - M(J' \ {j}) simulated and K from the Lipschitz bound.
      Needs a variable per cut, so bnc preallocates a pool (--zpool); it did not help in practice.
Subproblem relaxation in the master (Sec. 6, makespan): M_w >= r_i + (1/B) sum_{j>=i} n_j x_jw for every
distinct release time, and M_w >= (r_j + n_j) x_jw.

Findings (mooncake_conversation, slice 3000, B=4 W=2 L=50, 300 s): ilp3 solves N=25 instantly, reaches a
4.8% gap at N=50 and finds no feasible solution at N=100/200. LBBD/bnc matches ilp3's N=50 incumbent with
a 2.1% gap and returns 14% (N=100) and 2% (N=200) better-than-greedy routings with 9% / 1.3% gaps.
The bound is the weak part: it stalls at the optimal free-packing bound over W*B slots, and closing the
remaining JSQ-waiting gap (a 2000-step request arriving at step 240 waits 167 steps because three
same-batch fillers took the emptier slots) needs cuts whose slope is far below n_j, which the monotone +
Lipschitz logic cannot give. --seed-sa warm starts the master from a short annealing run.

Modes: --mode bnc  branch and check (Sec. 4): master solved once, cuts added as Gurobi lazy constraints
                   at every integer incumbent; repaired incumbents (x̄, true M_w) injected as heuristic
                   solutions.
       --mode lbbd classic Algorithm 1: re-solve the master MIP after each round of cuts.

Usage:
  python lbbd_routing.py --trace traces/mooncake_conversation.jsonl --N 100 [--N 200 ...]
         [--B 4] [--W 2] [--L 50] [--start 3000] [--time-limit 300] [--mode bnc|lbbd] [--cuts lip,energy]
         [--seed-sa 20] [--threads 0] [--log]
"""
import argparse, bisect, json, math, time
import gurobipy as gp
from gurobipy import GRB

from ilp3_routing import load, sim_fixed, greedy_route, makespan_of_route


# ----------------------------------------------------------------------------- subproblem (simulation)
def kw_states(J, r, n, B):
    """Kiefer-Wolfowitz recursion over the requests J (increasing index) on one B-slot FCFS worker.
    Returns the list of sorted remaining-work vectors V_i right after inserting each request i in J."""
    Wv = [0] * B; t = 0; out = []
    for j in J:
        d = r[j] - t; t = r[j]
        if d: Wv = [max(w - d, 0) for w in Wv]                     # age; stays sorted
        w0 = Wv.pop(0); bisect.insort(Wv, w0 + n[j])               # least-loaded slot takes the request
        out.append(Wv[:])
    return out


def makespan(J, r, n, B):
    """Worker makespan of request set J, evaluated as in ilp3 at the instance's last arrival r[-1]:
    max(last completion step, r[-1]). The fixed reference time matters: monotonicity and the
    Lipschitz bound hold for the remaining-work state at a fixed time, not for the raw completion."""
    Wv = [0] * B; t = 0
    for j in J:
        d = r[j] - t; t = r[j]
        if d: Wv = [max(w - d, 0) for w in Wv]
        w0 = Wv.pop(0); bisect.insort(Wv, w0 + n[j])
    return max(t + Wv[-1], r[-1])


def critical_subset(J, r, n, B, Mstar, order="early"):
    """Inclusion-minimal J' ⊆ J with makespan(J') == Mstar (Hooker's 'remove jobs one at a time and
    re-solve'). Because makespan is monotone in the set, the removable prefix is found by bisection;
    the remaining requests are then tried in the given order (early/late in the trace, big/small n)."""
    lo, hi = 0, len(J) - 1                                          # J[k:] keeps Mstar for all k <= lo
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if makespan(J[mid:], r, n, B) == Mstar: lo = mid
        else: hi = mid - 1
    keep = J[lo:]
    key = {"early": lambda j: j, "late": lambda j: -j, "big": lambda j: -n[j], "small": lambda j: n[j]}[order]
    for j in sorted(keep, key=key):
        cand = [k for k in keep if k != j]
        if makespan(cand, r, n, B) == Mstar: keep = cand
    return keep


# ----------------------------------------------------------------------------- Benders cuts
class Cuts:
    """Generates cuts from the schedule of one worker. Each cut is (const, coef, zinfo) meaning, for every
    worker w' (identical workers):   M_w' >= const + sum_j coef_j x_jw' [- K z],
    where zinfo = None, or (K, zconst, zcoef) with a fresh continuous z >= 0, z >= zconst + sum_j zcoef_j x_jw'."""

    def __init__(self, r, n, B, kinds, orders=("early",), energy_k=1):
        self.r, self.n, self.B, self.kinds = r, n, B, kinds
        self.orders, self.energy_k = orders, energy_k
        self.stats = {"lip": 0, "energy": 0, "nogood": 0, "delta": 0, "sims": 0}

    def for_worker(self, J, Mstar):
        r, n, B = self.r, self.n, self.B; cuts = []
        if not J: return cuts
        if self.kinds & {"lip", "nogood", "delta"}:
            seen = set()
            for order in self.orders:
                Jc = critical_subset(J, r, n, B, Mstar, order); self.stats["sims"] += len(J)
                if tuple(Jc) in seen: continue
                seen.add(tuple(Jc))
                if "lip" in self.kinds:                             # M >= M* - sum n_j (1 - x_j)
                    cuts.append((Mstar - sum(n[j] for j in Jc), {j: n[j] for j in Jc}, None)); self.stats["lip"] += 1
                if "nogood" in self.kinds:                          # M >= M* (1 - sum (1 - x_j))
                    cuts.append((Mstar * (1 - len(Jc)), {j: Mstar for j in Jc}, None)); self.stats["nogood"] += 1
                if "delta" in self.kinds and len(Jc) > 1:
                    # (D) exact for <= 1 removal: M >= M* - sum d_j (1 - x_j) - K z,  z >= sum (1 - x_j) - 1,
                    # d_j = M* - M(J' \ {j}) (simulated). Valid for |S| >= 2 by the Lipschitz bound provided
                    # K >= a_1 + a_2, the two largest n_j - d_j (also K = M* - r[-1] always suffices).
                    d = {j: Mstar - makespan([k for k in Jc if k != j], r, n, B) for j in Jc}; self.stats["sims"] += len(Jc)
                    a = sorted((n[j] - d[j] for j in Jc), reverse=True)
                    K = min(Mstar - r[-1], a[0] + a[1])
                    cuts.append((Mstar - sum(d.values()), dict(d), (K, len(Jc) - 1, {j: -1 for j in Jc}))); self.stats["delta"] += 1
        if "energy" in self.kinds:
            states = kw_states(J, r, n, B); N = len(n)
            fut = [0] * (len(J) + 1)
            for k in range(len(J) - 1, -1, -1): fut[k] = fut[k + 1] + n[J[k]]
            cand = sorted(range(len(J)), key=lambda k: -(r[J[k]] + (sum(states[k]) + fut[k + 1]) / B))
            used = set()
            for k in cand:                                           # best few, at distinct arrival times
                j = J[k]
                if r[j] in used: continue
                used.add(r[j]); backlog = sum(states[k])
                const = r[j] + (backlog - sum(n[jj] for jj in J[:k + 1])) / B
                coef = {jj: n[jj] / B for jj in J[:k + 1]}
                for jj in range(j + 1, N): coef[jj] = coef.get(jj, 0) + n[jj] / B
                cuts.append((const, coef, None)); self.stats["energy"] += 1
                if len(used) >= self.energy_k: break
        return cuts


# ----------------------------------------------------------------------------- master
def build_master(reqs, L, B, W, log, threads, zpool=0, prio=False):
    N = len(reqs)
    r = [math.ceil(a / L) for a, _, _ in reqs]; n = [o for _, _, o in reqs]
    m = gp.Model("routing_master"); m.Params.OutputFlag = 1 if log else 0; m.Params.Threads = threads
    x = m.addVars(N, W, vtype=GRB.BINARY, name="x")
    Mw = m.addVars(W, vtype=GRB.INTEGER, lb=0, name="Mw")
    M = m.addVar(vtype=GRB.INTEGER, lb=max(r[j] + n[j] for j in range(N)), name="M")
    m.addConstrs((x.sum(i, "*") == 1 for i in range(N)), name="assign")
    m.addConstrs((M >= Mw[w] for w in range(W)), name="mk")
    # symmetry: workers are identical; request i may open worker w only if w-1 is already in use
    m.addConstr(x[0, 0] == 1)
    for i in range(1, N):
        for w in range(1, W):
            m.addConstr(x[i, w] <= gp.quicksum(x[j, w - 1] for j in range(i)))
    # subproblem relaxation (Hooker Sec. 6): energy after each distinct release time, and head-of-line
    suffix = [0] * (N + 1)
    for j in range(N - 1, -1, -1): suffix[j] = suffix[j + 1] + n[j]
    rel = [i for i in range(N) if i == 0 or r[i] > r[i - 1]]
    for w in range(W):
        for i in rel:
            m.addConstr(Mw[w] >= r[i] + gp.quicksum(n[j] * x[j, w] for j in range(i, N)) / B, name=f"R{w}_{i}")
        for j in range(N):
            if r[j] + n[j] > r[N - 1]:                               # only bind when worth anything
                m.addConstr(Mw[w] >= (r[j] + n[j]) * x[j, w], name=f"H{w}_{j}")
    if prio:                                                         # branch in arrival order (prefix states)
        for i in range(N):
            for w in range(W): x[i, w].BranchPriority = N - i
    z = m.addVars(zpool, lb=0, name="z") if zpool else None          # pool for (D) cuts (lazy cuts cannot add vars)
    m.setObjective(M, GRB.MINIMIZE)
    m.update()
    return m, x, Mw, M, r, n, z


def add_cut(model, x, Mw, w2, cut, lazy, znext):
    """Add one cut for worker w2. znext: callable returning a fresh z var (or None if the pool is empty)."""
    const, coef, zinfo = cut
    lhs = Mw[w2] - gp.quicksum(c * x[j, w2] for j, c in coef.items())
    if zinfo is not None:
        K, zc, zcoef = zinfo; zv = znext()
        if zv is None: return 0
        lhs = lhs + K * zv
        zrow = zv - gp.quicksum(c * x[j, w2] for j, c in zcoef.items()) >= zc
        (model.cbLazy if lazy else model.addConstr)(zrow)
    (model.cbLazy if lazy else model.addConstr)(lhs >= const)
    return 1


def anneal(r, n, B, W, seed_route, seconds, seed=0):
    """Simulated annealing on the routing (single-request moves), used to warm start the master
    (Hooker Sec. 2: "generate a few Benders cuts in advance for heuristically chosen values of x")."""
    import random
    rng = random.Random(seed); N = len(n); t_end = time.time() + seconds
    def mk(route): return max(makespan(J, r, n, B) for J in per_worker(route, W))
    route = list(seed_route); cur = mk(route); best, best_route = cur, list(route); T = 0.02 * cur
    while time.time() < t_end:
        for _ in range(200):
            i = rng.randrange(N); old = route[i]; route[i] = (old + 1 + rng.randrange(W - 1)) % W if W > 1 else old
            v = mk(route)
            if v <= cur or rng.random() < math.exp((cur - v) / T): cur = v
            else: route[i] = old
            if cur < best: best, best_route = cur, list(route)
        T = max(0.5, T * 0.995)
    return best_route, best


def route_of(xv, N, W):
    return [max(range(W), key=lambda w: xv[i, w]) for i in range(N)]


def per_worker(route, W):
    Js = [[] for _ in range(W)]
    for i, w in enumerate(route): Js[w].append(i)
    return Js


def set_start(x, Mw, M, route, mks, N, W):
    for i in range(N):
        for w in range(W): x[i, w].Start = 1.0 if route[i] == w else 0.0
    for w in range(W): Mw[w].Start = mks[w]
    M.Start = max(mks)


# ----------------------------------------------------------------------------- solvers
def solve_bnc(m, x, Mw, M, z, r, n, B, W, cuts, time_limit, seed_route):
    """Branch and check: one master solve, Benders cuts as lazy constraints at each MIPSOL."""
    N = len(n); m.Params.LazyConstraints = 1; m.Params.TimeLimit = time_limit
    st = {"cb": 0, "violated": 0, "cuts": 0, "best": math.inf, "route": None, "pending": None, "tcb": 0.0, "zused": 0}
    mks0 = [makespan(J, r, n, B) for J in per_worker(seed_route, W)]
    st["best"], st["route"] = max(mks0), seed_route
    set_start(x, Mw, M, seed_route, mks0, N, W)
    cache = {}                                                       # worker job set -> cuts (dedupe)

    def znext():
        if z is None or st["zused"] >= len(z): return None
        st["zused"] += 1; return z[st["zused"] - 1]

    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            t0 = time.time(); st["cb"] += 1
            xv = model.cbGetSolution(x); mv = model.cbGetSolution(Mw)
            route = route_of(xv, N, W); Js = per_worker(route, W)
            mks = [makespan(J, r, n, B) for J in Js]
            for w in range(W):
                if mks[w] > mv[w] + 0.5:                            # subproblem value exceeds master's guess
                    st["violated"] += 1; key = tuple(Js[w])
                    if key in cache: cs = cache[key]                 # same set seen before: re-add its cuts
                    else: cs = cache[key] = cuts.for_worker(Js[w], mks[w])
                    for cut in cs:
                        for w2 in range(W): st["cuts"] += add_cut(model, x, Mw, w2, cut, True, znext)
            if max(mks) < st["best"]:                                # repaired incumbent, inject at next node
                st["best"], st["route"] = max(mks), route
                st["pending"] = (route, mks)
            st["tcb"] += time.time() - t0
        elif where == GRB.Callback.MIPNODE and st["pending"] is not None:
            if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                route, mks = st["pending"]; st["pending"] = None
                vs = [x[i, w] for i in range(N) for w in range(W)] + [Mw[w] for w in range(W)] + [M]
                vals = [1.0 if route[i] == w else 0.0 for i in range(N) for w in range(W)] + list(mks) + [max(mks)]
                model.cbSetSolution(vs, vals); model.cbUseSolution()

    m.optimize(cb)
    lb = math.ceil(m.ObjBound - 1e-6) if m.SolCount or m.Status == GRB.TIME_LIMIT else None
    return st, lb


def solve_lbbd(m, x, Mw, M, r, n, B, W, cuts, time_limit, seed_route, log=False):
    """Classic LBBD (Algorithm 1): master MIP re-solved to optimality after each round of cuts."""
    N = len(n); t_end = time.time() + time_limit
    st = {"iters": 0, "cuts": 0, "best": math.inf, "route": None, "tsub": 0.0}
    mks0 = [makespan(J, r, n, B) for J in per_worker(seed_route, W)]
    st["best"], st["route"] = max(mks0), seed_route
    set_start(x, Mw, M, seed_route, mks0, N, W)
    znext = lambda: m.addVar(lb=0, name="z")
    for w in range(W):                                               # warm start: cuts for the heuristic routing
        for cut in cuts.for_worker(per_worker(seed_route, W)[w], mks0[w]):
            for w2 in range(W): st["cuts"] += add_cut(m, x, Mw, w2, cut, False, znext)
    lb = 0
    while True:
        m.Params.TimeLimit = max(0.0, t_end - time.time())
        m.optimize(); st["iters"] += 1
        if m.SolCount == 0: break
        lb = max(lb, math.ceil(m.ObjBound - 1e-6))
        t0 = time.time()
        route = route_of({(i, w): x[i, w].X for i in range(N) for w in range(W)}, N, W)
        Js = per_worker(route, W); mks = [makespan(J, r, n, B) for J in Js]
        if max(mks) < st["best"]: st["best"], st["route"] = max(mks), route
        if log: print(f"  iter {st['iters']:4d}  LB {lb:6d}  sub {max(mks):6d}  best {st['best']:6d}  cuts {st['cuts']}", flush=True)
        if st["best"] <= lb or m.Status == GRB.TIME_LIMIT: break
        for w in range(W):
            if mks[w] > Mw[w].X + 0.5:
                for cut in cuts.for_worker(Js[w], mks[w]):
                    for w2 in range(W): st["cuts"] += add_cut(m, x, Mw, w2, cut, False, znext)
        set_start(x, Mw, M, st["route"], [makespan(J, r, n, B) for J in per_worker(st["route"], W)], N, W)
        st["tsub"] += time.time() - t0
    return st, lb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--N", type=int, action="append", required=True, help="trace length(s); repeat flag for a sweep")
    ap.add_argument("--B", type=int, default=4, help="max concurrent sequences per worker")
    ap.add_argument("--W", type=int, default=2, help="number of workers")
    ap.add_argument("--L", type=float, default=50, help="fixed step latency (ms)")
    ap.add_argument("--start", type=int, default=3000, help="first trace row of the slice")
    ap.add_argument("--time-limit", type=float, default=300)
    ap.add_argument("--mode", choices=["bnc", "lbbd"], default="bnc")
    ap.add_argument("--cuts", default="lip,energy", help="comma list of lip,energy,delta,nogood")
    ap.add_argument("--zpool", type=int, default=6000, help="bnc: pool of z variables for delta cuts")
    ap.add_argument("--prio", action="store_true", help="branch on requests in arrival order")
    ap.add_argument("--orders", default="early", help="comma list of removal orders for J': early,late,big,small")
    ap.add_argument("--energy-k", type=int, default=1, help="energy cuts per subproblem (distinct arrival times)")
    ap.add_argument("--seed-sa", type=float, default=0, help="seconds of simulated annealing to warm start the master (0 = greedy only)")
    ap.add_argument("--threads", type=int, default=0, help="Gurobi threads (0 = all)")
    ap.add_argument("--log", action="store_true", help="show Gurobi log / LBBD iterations")
    a = ap.parse_args()
    B, W, L = a.B, a.W, a.L; kinds = set(a.cuts.split(","))
    print(f"B={B} W={W} L={L}ms slice@{a.start}  mode={a.mode} cuts={sorted(kinds)}  time limit {a.time_limit:.0f}s per instance", flush=True)
    print(f"{'N':>6} {'vars':>7} {'cons':>7} {'solve':>7} {'status':>9} {'LB':>6} {'iters/cb':>8} {'cuts':>6} | "
          f"{'LBBD':>8} {'greedy':>8} {'1-worker':>8}  (makespan, s)", flush=True)
    for N in a.N:
        reqs = load(a.trace, a.start, N)
        g_route, g_mk = greedy_route(reqs, L, B, W); one_mk = max(sim_fixed(reqs, L, B)[1])
        m, x, Mw, M, r, n, z = build_master(reqs, L, B, W, a.log, a.threads, a.zpool if "delta" in kinds else 0, a.prio)
        cuts = Cuts(r, n, B, kinds, tuple(a.orders.split(",")), a.energy_k)
        t0 = time.time()
        if a.seed_sa > 0: g_route, _ = anneal(r, n, B, W, g_route, a.seed_sa)
        tl = max(1.0, a.time_limit - (time.time() - t0))                 # annealing time counts against the limit
        if a.mode == "bnc": st, lb = solve_bnc(m, x, Mw, M, z, r, n, B, W, cuts, tl, g_route); it = st["cb"]
        else: st, lb = solve_lbbd(m, x, Mw, M, r, n, B, W, cuts, tl, g_route, a.log); it = st["iters"]
        ts = time.time() - t0
        best = st["best"]; route = st["route"]
        assert makespan_of_route(reqs, route, L, B, W) == best                  # re-simulate with ilp3's simulator
        status = "optimal" if lb is not None and lb >= best else "timelimit"
        print(f"{N:6d} {m.NumVars:7d} {m.NumConstrs:7d} {ts:6.1f}s {status:>9} {lb if lb is not None else -1:6d} {it:8d} {st['cuts']:6d} | "
              f"{best * L / 1000:8.1f} {g_mk * L / 1000:8.1f} {one_mk * L / 1000:8.1f}   steps: best {best} greedy {g_mk}"
              f"  [{'cb' if a.mode == 'bnc' else 'sub'} time {st.get('tcb', st.get('tsub', 0)):.1f}s, sims {cuts.stats['sims']}, z used {st.get('zused', 0)}]", flush=True)


if __name__ == "__main__":
    main()
