# Per-trace decode calibration of the CS 349E assn2 engine (HF transformers) on one real GPU.
#
# calibrate_harness.py times B identical caches admitted at once. In a running worker the batch is staggered: requests
# entered at different times, so slot lengths are spread over [isl, isl + osl) and the batch is padded to the longest.
# The HF step copies the whole padded KV every step, so its cost follows B x S_max. This script measures that steady
# state for each trace's (isl, osl), on a fine B grid up to the trace's slot cap, and also times the two batch-wide
# copies a worker pays per request:
#   admit:  DecodeBatch.admit of a fresh isl-token cache into a batch of B - 1 (pads + concatenates every layer)
#   evict:  DecodeBatch.evict of the longest slot (index_select of every layer, then _compact, since S_max drops)
# Both stall every decode on the GPU. Writes steps_<config>_traces_bench.json; assn2.py fits from it per solve.
#
# Usage, from soldr/, with the assn2 venv:
#   ../cs349e-assn2/.venv/bin/python calibrate_traces.py qwen3-8b_a100x4 [--gpu 0]
import argparse, json, os, statistics, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ASSN2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cs349e-assn2")
TRACES = {"chat": (722, 185, 96), "summarize": (1651, 90, 48), "extract": (4000, 25, 24)}   # isl, osl, slots
B_GRID = (1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96)
WARMUP, STEPS, EVENT_REPS = 3, 20, 5


def measure(gpu, log=print):
    sys.path.insert(0, ASSN2)
    import torch
    from system.engine import Engine, Mode, make_prompt, synthetic_cache
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    engine = Engine(Mode.MIXED, device, max_slots=max(B_GRID))
    sync = torch.cuda.synchronize
    engine.prefill(make_prompt(512, 0)); sync()

    def timed(f):
        sync(); t = time.perf_counter(); f(); sync()
        return (time.perf_counter() - t) * 1e3

    out = {}
    for name, (isl, osl, slots) in TRACES.items():
        fresh = synthetic_cache(isl, device, seed=1)
        # Every slot's cache is a prefix of one long cache (a view: admit copies it), so the bench holds no extra KV
        full = synthetic_cache(isl + osl, device, seed=2)
        prefix = lambda L: tuple((k[:, :, :L], v[:, :, :L]) for k, v in full)
        rows = []
        for B in [b for b in B_GRID if b <= slots]:
            batch = engine.batch
            # Slot i has generated i * osl / B tokens: lengths spread evenly over [isl, isl + osl), longest first
            lens = [isl + (osl - 1) * (B - 1 - i) // max(B - 1, 1) if B > 1 else isl + osl // 2 for i in range(B)]
            for i, L in enumerate(lens):
                batch.admit(i, prefix(L), first_token=100, max_new=osl + WARMUP + STEPS + 4 * EVENT_REPS + 8)
            for _ in range(WARMUP):
                batch.step()
            step = statistics.median(timed(batch.step) for _ in range(STEPS))
            S = batch._S
            # Events: evict the longest slot (triggers _compact), then admit a fresh isl cache in its place
            ev, ad = [], []
            rid = B
            for _ in range(EVENT_REPS):
                longest = max(range(len(batch._rids)), key=lambda s: int(batch._lengths[s]))
                ev.append(timed(lambda: batch.evict(batch._rids[longest])))
                ad.append(timed(lambda: batch.admit(rid, fresh, first_token=100, max_new=osl)))
                rid += 1
                batch.step()
            for r in list(batch._rids):
                batch.evict(r)
            rows.append(dict(B=B, S=S, tokens=B * S, step=step, evict=statistics.median(ev), admit=statistics.median(ad)))
            log(f"{name:9s} B {B:3d}  S {S:5d}  B*S {B * S:6d}  step {step:7.2f} ms  admit {rows[-1]['admit']:6.2f}  "
                f"evict {rows[-1]['evict']:6.2f} ms")
            torch.cuda.empty_cache()
        out[name] = dict(isl=isl, osl=osl, slots=slots, rows=rows)
        del full, fresh
        torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    path = f"steps_{args.config}_traces_bench.json"
    bench = dict(config=args.config, source="cs349e-assn2 engine, staggered batches", gpu=args.gpu,
                 traces=measure(args.gpu))
    json.dump(bench, open(path, "w"), indent=1)
    print(f"wrote {path}")
