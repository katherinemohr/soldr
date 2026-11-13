# Step-level model of Mocker's vLLM engine, calibrated from microbenchmarks (step_bench.json, prefill_bench.json).
# Step time = prefill part (if any prefill tokens) + decode part (if any decodes):
#   prefill: g(sum n) + b * sum n (s + n / 2), for chunks of n new tokens after s cached/done tokens
#   decode: f(B) + k * total KV tokens of the B decoding requests
import bisect, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNK, MAX_SEQS = 8192, 256


def interp(xs, ys, x):
    # Piecewise linear, linear extrapolation at both ends
    i = min(max(bisect.bisect_left(xs, x), 1), len(xs) - 1)
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


class StepModel:
    def __init__(self):
        dec = json.load(open(os.path.join(HERE, "step_bench.json")))["decode"]
        pre = json.load(open(os.path.join(HERE, "prefill_bench.json")))["single"]
        # Decode: one free f(B) per batch size + shared KV slope k (context ~ L + 300 generated)
        Bs = sorted({r["B"] for r in dec})
        X = np.array([[r["B"] == B for B in Bs] + [r["B"] * (r["L"] + 300)] for r in dec], float)
        sol, *_ = np.linalg.lstsq(X, np.array([r["tpot"] for r in dec]), rcond=None)
        self.Bs, self.fB, self.k = Bs, list(sol[:-1]), sol[-1]
        self.decode_fit_err = np.abs(X @ sol - [r["tpot"] for r in dec]).max()
        # Prefill: single-chunk points only; g(n) free per n, shared attention slope b
        one = [r for r in pre if r["n"] <= CHUNK]
        ns = sorted({r["n"] for r in one})
        X = np.array([[r["n"] == n for n in ns] + [r["n"] * (r["s"] + r["n"] / 2)] for r in one], float)
        sol, *_ = np.linalg.lstsq(X, np.array([r["ttft"] for r in one]), rcond=None)
        self.ns, self.gn, self.b = ns, list(sol[:-1]), sol[-1]
        self.prefill_fit_err = np.abs(X @ sol - [r["ttft"] for r in one]).max()

    def decode(self, B, kv):
        return interp(self.Bs, self.fB, B) + self.k * kv if B else 0.0

    def prefill(self, chunks):
        n = sum(c for c, _ in chunks)
        return interp(self.ns, self.gn, n) + self.b * sum(c * (s + c / 2) for c, s in chunks) if n else 0.0


def simulate(reqs, m):
    """reqs: dicts with arrival (ms), isl, osl, hit, on one worker. Returns per request (first_token_ms, last_token_ms)."""
    order = sorted(range(len(reqs)), key=lambda j: (reqs[j]["arrival"], j))
    done = [r["hit"] for r in reqs]          # prompt tokens computed or cached
    gen = [0] * len(reqs)                    # output tokens emitted
    first, last = [None] * len(reqs), [None] * len(reqs)
    stall = [0.0] * len(reqs)               # ms of prefill work in the steps a request decodes in
    running, nxt, t = [], 0, 0.0
    while nxt < len(order) or running:
        while nxt < len(order) and reqs[order[nxt]]["arrival"] <= t:
            nxt += 1
        waiting = [j for j in order[:nxt] if first[j] is None and j not in running]
        if not running and not waiting:
            t = reqs[order[nxt]]["arrival"]
            continue
        budget = CHUNK
        decodes = [j for j in running if first[j] is not None]
        budget -= len(decodes)
        chunks, prefilling = [], []
        # Running (partially prefilled) requests first, then waiting in arrival order
        for j in [j for j in running if first[j] is None] + waiting:
            if budget <= 0:
                break
            if j not in running:
                if len(running) >= MAX_SEQS:
                    break
                running.append(j)
            c = min(reqs[j]["isl"] - done[j], budget)
            chunks.append((c, done[j]))
            prefilling.append((j, c))
            budget -= c
        kv = sum(reqs[j]["isl"] + gen[j] for j in decodes)
        pf = m.prefill(chunks)
        t += pf + m.decode(len(decodes), kv)
        for j in decodes:
            stall[j] += pf
            gen[j] += 1
            if gen[j] == reqs[j]["osl"]:
                last[j] = t
                running.remove(j)
        for j, c in prefilling:
            done[j] += c
            if done[j] >= reqs[j]["isl"]:
                first[j], gen[j] = t, 1
                if reqs[j]["osl"] == 1:
                    last[j] = t
                    running.remove(j)
    simulate.stall = stall
    return list(zip(first, last))


def predict(rows, recs, m):
    """Mocker run (rows, recs) re-simulated per worker with Mocker's routing and cache hits. Returns ttfts, tpots in ms."""
    by_worker = {}
    for j, q in enumerate(recs):
        by_worker.setdefault(q["decode_worker_idx"], []).append(j)
    ttft, tpot = [None] * len(recs), [None] * len(recs)
    predict.stall = [None] * len(recs)   # per-token prefill stall, ms
    for js in by_worker.values():
        reqs = [dict(arrival=recs[j]["arrival_time_ms"], isl=rows[j]["input_length"], osl=rows[j]["output_length"],
                     hit=recs[j]["reused_input_tokens"]) for j in js]
        for j, r, (f, l) in zip(js, reqs, simulate(reqs, m)):
            ttft[j] = f - r["arrival"]
            tpot[j] = (l - f) / (r["osl"] - 1) if r["osl"] > 1 else None
        for j, r, st in zip(js, reqs, simulate.stall):
            predict.stall[j] = st / (r["osl"] - 1) if r["osl"] > 1 else None
    return ttft, tpot


if __name__ == "__main__":
    m = StepModel()
    print("decode f(B):", {B: round(f, 2) for B, f in zip(m.Bs, m.fB)}, f"k = {m.k:.3g} ms/KV token, "
          f"max fit error {m.decode_fit_err:.2f} ms")
    print("prefill g(n):", {n: round(g, 2) for n, g in zip(m.ns, m.gn)}, f"b = {m.b:.3g}, "
          f"max fit error {m.prefill_fit_err:.2f} ms")
    # Held-out: multi-chunk prefills, pairs, and the mixed (decode + prefill) benchmark
    pre = json.load(open(os.path.join(HERE, "prefill_bench.json")))
    for r in pre["single"]:
        if r["n"] > CHUNK:
            (f, _), = [x for x in simulate([dict(arrival=0, isl=r["s"] + r["n"], osl=1, hit=r["s"])], m)]
            print(f"prefill s={r['s']:6d} n={r['n']:6d}: measured {r['ttft']:7.1f} predicted {f:7.1f}")
    for r in pre["pair"]:
        f = simulate([dict(arrival=0, isl=r["n"], osl=1, hit=0)] * 2, m)[0][0]
        print(f"pair n={r['n']}: measured {r['ttfts'][0]:.1f} predicted {f:.1f}")
