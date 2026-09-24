"""ILP 3: optimal (min-makespan) routing of a request trace across W identical workers.

Built on ILP 1 (fixed step latency L, B concurrent slots per worker, FCFS admission, no preemption,
no prefix reuse). Per worker the Kiefer-Wolfowitz remaining-work recursion runs with n_i replaced by
n_i * x_iw, where x_iw is the routing binary. Objective: minimize makespan (in steps).

Usage:
  python ilp3_routing_standalone.py --trace mooncake_conversation.jsonl --N 100 [--N 200 ...]
         [--B 4] [--W 2] [--L 50] [--start 3000] [--time-limit 300] [--threads 0] [--log]

Requires: gurobipy (licensed). Output: one row per N with model size, solve status, MIP gap and the
makespan of the ILP routing vs. a greedy join-shortest-queue router vs. a single worker.
"""
import argparse, json, math, time
import gurobipy as gp
from gurobipy import GRB


def load(path, start, N):
    rs = [json.loads(l) for l in open(path)][start:start + N]
    t0 = rs[0]["timestamp"]
    return [(r["timestamp"] - t0, r["input_length"], r["output_length"]) for r in rs]


def sim_fixed(reqs, L, B):
    """B-slot FCFS continuous batching on one worker, global step clock of period L.
    Returns (start step, completion step) per request. Service = output_length steps."""
    r = [math.ceil(a / L) for a, _, _ in reqs]; n = [o for _, _, o in reqs]
    comps, s, c = [], [], []
    for i in range(len(reqs)):
        t = max(r[i], s[-1] if s else 0)
        act = sorted(x for x in comps if x > t)
        if len(act) >= B:
            t = act[len(act) - B]
        s.append(t); c.append(t + n[i]); comps.append(t + n[i])
    return s, c


def greedy_route(reqs, L, B, W):
    """Baseline: send each arrival to the worker where it would start earliest."""
    r = [math.ceil(a / L) for a, _, _ in reqs]; n = [o for _, _, o in reqs]
    comps = [[] for _ in range(W)]; last_s = [0] * W; route = []
    def start_on(w, i):
        t = max(r[i], last_s[w]); act = sorted(x for x in comps[w] if x > t)
        return act[len(act) - B] if len(act) >= B else t
    for i in range(len(reqs)):
        w = min(range(W), key=lambda w: start_on(w, i)); t = start_on(w, i)
        comps[w].append(t + n[i]); last_s[w] = t; route.append(w)
    return route, max(max(c) for c in comps if c)


def makespan_of_route(reqs, route, L, B, W):
    return max(max(sim_fixed([q for q, w in zip(reqs, route) if w == ww], L, B)[1])
               for ww in range(W) if any(w == ww for w in route))


def build(reqs, L, B, W, log=False, threads=0):
    N = len(reqs)
    r = [math.ceil(a / L) for a, _, _ in reqs]; n = [o for _, _, o in reqs]
    M = r[-1] + sum(n) + 1                                        # bound on any step index
    m = gp.Model("routing"); m.Params.OutputFlag = 1 if log else 0; m.Params.Threads = threads
    x = m.addVars(N, W, vtype=GRB.BINARY, name="x")
    m.addConstrs((x.sum(i, "*") == 1 for i in range(N)), name="assign")
    m.addConstr(x[0, 0] == 1)                                     # symmetry breaking (identical workers)

    def add_max(a, b, name):                                      # z = max(a, b), exact, one binary
        z = m.addVar(vtype=GRB.INTEGER, lb=-M, ub=M, name=name); d = m.addVar(vtype=GRB.BINARY, name=name + "_d")
        m.addConstr(z >= a); m.addConstr(z >= b)
        m.addConstr(z <= a + M * (1 - d)); m.addConstr(z <= b + M * d)
        return z

    def add_min(a, b, name):                                      # z = min(a, b), exact, one binary
        z = m.addVar(vtype=GRB.INTEGER, lb=-M, ub=M, name=name); d = m.addVar(vtype=GRB.BINARY, name=name + "_d")
        m.addConstr(z <= a); m.addConstr(z <= b)
        m.addConstr(z >= a - M * (1 - d)); m.addConstr(z >= b - M * d)
        return z

    T = m.addVar(vtype=GRB.INTEGER, lb=0, ub=M, name="makespan_steps")
    for w in range(W):
        Wv = m.addVars(N, B, vtype=GRB.INTEGER, lb=0, ub=M, name=f"W{w}")   # sorted remaining steps per slot
        m.addConstrs((Wv[0, k] == 0 for k in range(B)))
        for i in range(N):
            xi = Wv[i, 0] + n[i] * x[i, w]                        # least-loaded slot takes request i iff routed here
            V = []                                                # sorted re-insertion: V[k] = max(W[k], min(xi, W[k+1]))
            for k in range(B):
                if B == 1: V.append(xi)
                elif k == 0: V.append(add_min(xi, Wv[i, 1], f"V{w}_{i}_{k}"))
                elif k == B - 1: V.append(add_max(Wv[i, k], xi, f"V{w}_{i}_{k}"))
                else: V.append(add_max(Wv[i, k], add_min(xi, Wv[i, k + 1], f"Vm{w}_{i}_{k}"), f"V{w}_{i}_{k}"))
            if i + 1 < N:                                         # age by inter-release gap: W_{i+1} = (V - delta)^+
                for k in range(B):
                    m.addConstr(Wv[i + 1, k] == add_max(V[k] - (r[i + 1] - r[i]), 0, f"Wp{w}_{i}_{k}"))
            else:
                m.addConstr(T >= r[N - 1] + V[B - 1])             # worker w's last completion step
    m.setObjective(T, GRB.MINIMIZE)
    m.update()
    return m, x, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--N", type=int, action="append", required=True, help="trace length(s); repeat flag for a sweep")
    ap.add_argument("--B", type=int, default=4, help="max concurrent sequences per worker")
    ap.add_argument("--W", type=int, default=2, help="number of workers")
    ap.add_argument("--L", type=float, default=50, help="fixed step latency (ms)")
    ap.add_argument("--start", type=int, default=3000, help="first trace row of the slice")
    ap.add_argument("--time-limit", type=float, default=300)
    ap.add_argument("--threads", type=int, default=0, help="Gurobi threads (0 = all)")
    ap.add_argument("--log", action="store_true", help="show Gurobi log")
    a = ap.parse_args()
    B, W, L = a.B, a.W, a.L
    print(f"B={B} W={W} L={L}ms slice@{a.start}  time limit {a.time_limit:.0f}s per instance", flush=True)
    print(f"{'N':>6} {'vars':>9} {'bin':>8} {'cons':>9} {'build':>7} {'solve':>7} {'status':>9} {'gap':>6} | "
          f"{'ILP':>8} {'greedy':>8} {'1-worker':>8}  (makespan, s)", flush=True)
    for N in a.N:
        reqs = load(a.trace, a.start, N)
        _, g_mk = greedy_route(reqs, L, B, W); one_mk = max(sim_fixed(reqs, L, B)[1])
        t0 = time.time(); m, x, T = build(reqs, L, B, W, a.log, a.threads); tb = time.time() - t0
        m.Params.TimeLimit = a.time_limit
        t0 = time.time(); m.optimize(); ts = time.time() - t0
        st = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "timelimit"}.get(m.Status, str(m.Status))
        if m.SolCount:
            route = [max(range(W), key=lambda w: x[i, w].X) for i in range(N)]
            assert makespan_of_route(reqs, route, L, B, W) == round(T.X)   # re-simulate the ILP's routing
            ilp = f"{T.X * L / 1000:8.1f}"; gap = f"{100 * m.MIPGap:5.1f}%"
        else:
            ilp, gap = "    none", "   n/a"
        print(f"{N:6d} {m.NumVars:9d} {m.NumBinVars:8d} {m.NumConstrs:9d} {tb:6.1f}s {ts:6.1f}s {st:>9} {gap:>6} | "
              f"{ilp} {g_mk * L / 1000:8.1f} {one_mk * L / 1000:8.1f}", flush=True)


if __name__ == "__main__":
    main()
