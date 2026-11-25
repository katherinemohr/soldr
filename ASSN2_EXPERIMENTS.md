# soldr ↔ CS 349E assn2: code reference

This code connects soldr's CP-SAT interference solver to the CS 349E assignment 2 serving harness, which runs
Qwen3-8B with HF transformers on 4× A100 40 GB. With it you can:

- **Calibrate:** measure the assn2 engine's prefill and decode costs on a real GPU.
- **Convert:** write the assn2 traces, with the exact Poisson arrivals the harness draws, in soldr's trace format.
- **Solve:** compute a schedule (an optimized one, a prediction of assn2's round robin, or an optimized one
  warm-started from round robin), for a colocated layout or for a prefill/decode split (1p3d, 2p2d or 3p1d: number
  of prefill GPUs, then decode GPUs).
- **Replay:** run a schedule on the real harness.
- **Compare:** check a schedule's predictions against measured results, and rank the four layouts.

## Setup

The assn2 code is private, on code.stanford.edu. soldr includes it as a git submodule at `cs349e-assn2/`, pinned to
the exact commit these tools were run with. You need access to `kgmohr/cs349e-assn2` on code.stanford.edu, with an
SSH key on your account.

```bash
git clone --recursive git@github.com:katherinemohr/soldr.git    # or, in an existing clone:
git submodule update --init
```

When soldr's pin moves, run `git submodule update` after pulling.

**Where soldr looks for the assn2 code** (`assn2_dir.py`), in order:

1. `$ASSN2_DIR`, if set.
2. The submodule at `soldr/cs349e-assn2`, if it is checked out.
3. A sibling checkout at `../cs349e-assn2`.

The assn2 traces, and the harness `results/` that `compare` and `rank` read, come from that directory.

**Virtual environments.** They are built with `uv` and are not in either repo:

- `soldr/.venv`: Python 3.12 with `requirements.txt`. It runs the solver; it needs ortools, not torch.
- `cs349e-assn2/.venv`: the assn2 `requirements.txt`. It holds torch and transformers, and runs the calibration
  scripts and the harness.

```bash
uv venv .venv --python 3.12 && uv pip install -p .venv -r requirements.txt                       # from soldr/
cd cs349e-assn2 && uv venv .venv && uv pip install -p .venv -r requirements.txt                   # harness
```

**Updating the assn2 code.** Commit and push in `cs349e-assn2/`, then commit the new pin in soldr
(`git add cs349e-assn2`).

## Files

| File | Role |
|---|---|
| `soldr/assn2_dir.py` | Resolves the assn2 checkout (see Setup). |
| `soldr/cs349e-assn2/` | Submodule: the assn2 harness. |
| `soldr/calibrate_harness.py` | Measures prefill (and a uniform-batch decode) on the assn2 engine; writes `steps_qwen3-8b_a100x4.json` (prefill curve read by `StepLatency`) and `steps_qwen3-8b_a100x4_bench.json`. |
| `soldr/calibrate_traces.py` | Measures decode per trace on staggered batches, plus admit/evict cost; writes `steps_qwen3-8b_a100x4_traces_bench.json`. |
| `soldr/params_qwen3-8b_a100x4.json` | Hardware parameters; `kv_capacity = 163840` tokens per GPU. |
| `soldr/assn2.py` | Driver: `rates`, `convert`, `solve`, `compare`, `rank`. |
| `soldr/interference_packing.py` | Solver; assn2-specific behavior is behind module globals (below). |
| `soldr/interference_packing_helper.py` | Rolling-window carry-forward of interference for finished requests. |
| `cs349e-assn2/system/schedule_controller.py` | `ScheduleController`: dispatches requests from a schedule instead of round robin. |
| `cs349e-assn2/scripts/run_schedule.py` | Runs `run_experiment.main()` with `ScheduleController`. |
| `cs349e-assn2/scripts/warm_launch.py` | Wrapper that builds every rank pair's NCCL communicator before the run starts. |

The provided assn2 files are not modified. Run the harness commands below from `soldr/cs349e-assn2/`, where `../schedules/` is soldr's schedules directory.

---

## `calibrate_traces.py`

For each trace it uses a fixed prompt length (isl), output length (osl) and slot cap: chat (722, 185, 96),
summarize (1651, 90, 48), extract (4000, 25, 24).

**Batch sizes.** Measurements run at each B in `1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96`,
up to the trace's slot cap.

**Batch setup.** B caches are admitted with lengths spread evenly over `[isl, isl+osl)`, longest first. Each one is a
prefix view of a single full-length cache, so the benchmark holds no extra KV memory.

**What it times at each B:**

- `step`: median of 20 `DecodeBatch.step()` calls, after 3 warm-up steps.
- `evict`: evicting the longest slot, which triggers `_compact`. Median of 5.
- `admit`: admitting a fresh isl-length cache into the batch. Median of 5.

**Output.** Each row records `B`, `S` (padded length), `tokens`, `step`, `evict` and `admit`. The output file is
`steps_<config>_traces_bench.json`.

```
cd soldr && cs349e-assn2/.venv/bin/python calibrate_traces.py qwen3-8b_a100x4 [--gpu 0]
```

---

## `assn2.py`

### Constants

| Name | Value | Meaning |
|---|---|---|
| `CONFIG` | `qwen3-8b_a100x4` | params/steps files to load |
| `BLK` | 64 | solver block size for `hash_ids` |
| `WORKERS` | 4 | GPUs |
| `RANGES`, `POINTS` | chat (3, 9), summarize (2, 8), extract (2, 5); 10 | sweep rates, copied from `sweep_cs349e.py` |
| `MAX_SLOTS` | chat 96, summarize 48, extract 24 | assn2 `max_decode_slots` |
| `KV_GBPS` | 98 | KV transfer bandwidth for split layouts |
| `HANDOFF_STEPS` | 1.5 | decode steps of polling included in the fixed handoff cost |
| `DP_SCALE` | 0.5 | share of a decode step a prefill waits for |
| `FORMAT` | `soldr-assn2-schedule/1` | schedule file format tag |

### Arrivals

`arrival_times_s(n, rate, seed)` reproduces assn2's `Controller._arrival_times`:
`random.Random(seed).expovariate(rate)`, accumulated.

`build()` takes the first n requests of `cs349e-assn2/traces/<trace>.jsonl` and turns them into solver `Request`s.
Arrival times are in ms, and each request gets its own unique `hash_ids`, since no two assn2 prompts share a prefix.

### Subcommands

```
.venv/bin/python assn2.py rates
.venv/bin/python assn2.py convert [--n 300] [--seed 1234]
.venv/bin/python assn2.py solve TRACE RATE [--n 300] [--seed 1234] [--topology colocated|1p3d|2p2d|3p1d]
                                [--window N] [--step N] [--limit 30] [--policy opt|rr|warm] [--out PATH] [--any-rate]
.venv/bin/python assn2.py compare SCHEDULE RESULTS [BASELINE]
.venv/bin/python assn2.py rank TRACE RATE [--n 300] [--seed 1234]
```

- **`rates`** prints the sweep points for each trace.
- **`convert`** writes one file per trace and sweep rate, `traces/assn2/<trace>_r<rate>_n<n>_s<seed>.jsonl`. Each
  line holds:
  - `timestamp`: arrival time in ms
  - `input_length`, `output_length`
  - `hash_ids`: unique 512-token blocks
  - `rid`
  - `arrival_s`: the exact float arrival time
- **`solve`** builds the decode model, solves, and writes
  `schedules/<trace>_<topology>_r<rate>_n<n>_s<seed>[_rr|_warm].json`. `RATE` must be a sweep point unless
  `--any-rate` is given. With no `--window`, the solver uses one window over all n requests.
- **`compare`** matches a schedule to a harness result (and optionally a round-robin baseline) and reports:
  - TTFT and TPOT p50/p90 and SLO attainment, for the prediction, the replayed schedule and the baseline
  - Prediction error (median, P10, P90 and mean |error|), with TTFT error also shown excluding each GPU's first
    prefill
  - Placement mismatches and how late each prefill started vs. its scheduled start
  - Measured KV transfer median and P90 (split layouts only)
  - Makespan
- **`rank`** looks up each topology's schedule files and harness results for a trace, rate and n. For each arm found
  it prints SLO %, TTFT p90, TPOT p90 and mean e2e, then ranks the topologies by SLO attainment, then mean e2e.

  | Arm | Source |
  |---|---|
  | `predicted` | `schedules/<name>.json` |
  | `sched` | `results/sched_<name>.jsonl` |
  | `pred_rr` | `schedules/<name>_rr.json` |
  | `rr` | `results/rr_<name>.jsonl` |
  | `pred_warm` | `schedules/<name>_warm.json` |
  | `warm` | `results/sched_<name>_warm.jsonl` |

### Decode model: `trace_steps(name, rate, n_decode, colocated, n)`

Reads the trace's rows from `steps_<CONFIG>_traces_bench.json` and fits a hinge:

- `flat` = median of the steps within 8% of the minimum.
- `(icpt, delta)` = a straight line, `polyfit`, through the steps more than 8% above `flat`.
- Step model: `max(flat, icpt + delta·(B−1))`. This is returned as a `StepLatency` with `f1 = icpt`, `s = delta`,
  `k = 0`.

Next it finds the **operating batch** B: the typical decode batch per decode GPU at this rate. B is a damped fixed
point of Little's law:

- `lam` = `rate / n_decode` (arrivals per decode GPU)
- `rho` = `lam · prefill_ms(isl)` (share of time spent prefilling; colocated only, else 0)
- `tpot` = `(step(B) + lam·step(B)·event(B)) / (1 − rho)`
- `B` = `lam·(osl−1)·tpot`, capped at `min(slots, n / n_decode)`

It returns the steps, `flat`, the step time at B, and admit+evict at B. It also returns a `fit` dict with `op_B`,
`flat_ms`, `lin_t0`, `lin_delta`, `event_ms`, `step_at_op_ms`, `admit_ms`, the fit errors, `rho`, `saturated` and
`op_cap`.

### `solve` setup

| Setting | Value |
|---|---|
| `interference_packing.DP_SCALE` | `0.5` |
| `PD_FUSED` | `False` (a colocated prefill stalls every decode on its GPU) |
| `FLAT_MS` | `flat` |
| `DD_STEP_MS` | step time at the operating batch |
| memory | `min(kv_capacity, MAX_SLOTS[trace] × (isl + osl))`, which enforces the slot cap |
| colocated | `EVENT_MS` = admit + evict at the operating batch |
| split (`<P>p<D>d`: prefill ranks `0..P−1`, decode ranks `P..3`) | `EVENT_MS = 0`. The expected stall per decode step, `rate / D / 1000 × step × event`, is added to `steps.f1` and `FLAT_MS`. `KV_BANDWIDTH_GBPS = 98`, `TRANSFER_FIXED_MS = 1.5 × step + admit`. |

The `--policy` option chooses what the solver does:

| `--policy` | What the solver does |
|---|---|
| `opt` (default) | Optimizes routing and start times. |
| `rr` | Pins assn2's round robin: prefill rank `rid % P`, decode rank `rid % D`, start = max(arrival, previous request's prefill end on that rank). Sets `DECODE_MAY_WAIT`. Predicts the outcome. |
| `warm` | Optimizes, warm-started from the `rr` solution of each window. Sets `DECODE_MAY_WAIT`. |

### Schedule file

```
format, trace, rate, seed, n_requests, topology, prefill_ranks, decode_ranks,
solver: { policy, mem_tokens, window, step, limit_s, steps_config, solve_s, dp_scale, steps{f1,s,k},
          event_ms, pd_fused, flat_ms, dd_step_ms, transfer_fixed_ms, kv_gbps, decode_fit },
requests: [ { rid, prompt_len, output_len, arrival_s, release_ms, start_ms, prefill_rank, decode_rank,
              pred: { prefill_end_ms, end_ms, decode_start_ms, ttft_ms, tpot_ms, e2e_ms } } ]
```

Predictions use the harness's metric definitions:

- TTFT = prefill end − arrival. This holds for split layouts too.
- TPOT = (end − prefill end) / (osl − 1). For split layouts this includes the KV transfer.

---

## `interference_packing.py`

### Module globals

| Global | Default | Effect when set |
|---|---|---|
| `DP_SCALE` | `1.0` | Scales the decode→prefill wait, the share of a decode step that a prefill waits for. |
| `PD_FUSED` | `True` | `False`: a prefill overlapping a decode stalls that decode for the full overlap. The P→D coefficient is `S` instead of `S·chunk/(chunk + t0)`. |
| `EVENT_MS` | `0.0` | Adds a per-prefill stall to each decode it overlaps: a `pdev` pair term of `round(S·EVENT_MS)` times the prefill/decode overlap indicator. |
| `FLAT_MS` | `0.0` | Hinged decode cost (see below). |
| `DD_STEP_MS` | `0.0` | Denominator of the decode–decode coefficient, `S·delta[y] / DD_STEP_MS`. The default uses `t0[x] + delta[y]`. |
| `DECODE_MAY_WAIT` | `False` | Split: `decode_start ≥ prefill_end + transfer` instead of `==`, and decode memory is held from `decode_start` instead of `prefill_end`. |
| `KV_BYTES_PER_TOKEN` | `2·36·8·128·2` | Qwen3-8B KV bytes per token. |
| `KV_BANDWIDTH_GBPS` | `300` | KV transfer bandwidth. `assn2.py` sets 98. |
| `TRANSFER_FIXED_MS` | `None` | Fixed handoff cost. `None` uses `t0`. |

### Hinged decode cost (`FLAT_MS > 0`)

**Decode.** A request's decode time is the larger of two branches:

- **linear:** the existing interference model, including decode–decode terms.
- **flat:** `decode_steps × FLAT_MS`, plus all prefill→decode terms (including finished requests' carried P→D part),
  plus one extra `FLAT_MS` for split requests.

**Prefill.** The prefill slowdown is similarly the larger of:

- the linear decode→prefill term, and
- `chunks × DP_SCALE × FLAT_MS × any_decoding`.

### `transfer_ms(isl, t0)`

```
ceil(isl × KV_BYTES_PER_TOKEN / (KV_BANDWIDTH_GBPS·1e9) × 1e3 + fixed)
```

Here `fixed` is `TRANSFER_FIXED_MS`, or `t0` when that is `None`.

### `solve_rolling(..., policy=None, warm_policy=None)`

**`policy(g)`** returns `(prefill cache, decode cache, FCFS predecessor or None)` for request g. For every active
request it pins:

- the routes, and the decode routes (split only)
- `start == release` if there is no predecessor
- `start == max(release, predecessor's prefill_end)` otherwise. If the predecessor is in the window this is an
  `AddMaxEquality`; if it was solved in an earlier window it is a constant.

**`warm_policy`** works per window:

1. Build the window's model a second time.
2. Pin the policy in that copy and solve it.
3. If it is feasible, replace the free model's hints with that solution's value for **every** variable. The two
   models come from the same `build_model` call, so their variables line up by index.
4. Solve the free model.

CP-SAT therefore starts from the policy's solution, so its result is never worse than the policy on the model's
objective. The log prints the policy's objective and the optimized objective.

**Shared helpers.** `pin()` and `hint()` apply the policy constraints and the default hints (previous windows' starts
and cache-routing partial hints) to either model.

**Rolling windows with a pinned policy.** With `policy` or `warm_policy`, the rolling-window carry-forward state
(`cache_free_at`, finished requests) can conflict with the forced FCFS starts and make a window infeasible. Use a
single window (no `--window`) with `rr` and `warm`.

### `interference_packing_helper.py`

The carry-forward record for a finished request is a 4-tuple `(decode, prefill_slowdown, any, pd)`. The `pd` field
is the request's non-decode–decode part of decode interference, used by the flat branch. The carry-forward key set
includes `("pdev", finished, uncommitted)` pairs.

---

## Harness side (`cs349e-assn2`)

### `system/schedule_controller.py`

`ScheduleController` subclasses the assn2 `Controller` and inherits everything except dispatch: arrivals, READY
handshake, HANDOFF relay, metrics and output.

- It loads a `soldr-assn2-schedule/1` file and checks that its arrivals match the controller's to within 1e-6 s.
- It sends each request at `start_ms` (after t0) to `prefill_rank`, tagged with `decode_rank`, in start order.
- TTFT is still measured from the Poisson arrival.

### `scripts/run_schedule.py`

```
torchrun --nproc_per_node=4 -m scripts.run_schedule --schedule ../schedules/<x>.json [--name NAME]
```

The schedule determines the run arguments: trace, rate, seed, n and topology. The script builds the matching
`run_experiment.py` arguments and runs it with `ScheduleController`. It writes `results/sched_<schedule name>.jsonl`
and prints the equivalent round-robin command.

### `scripts/warm_launch.py`

```
torchrun --nproc_per_node=4 -m scripts.warm_launch experiment ARGS                  # round robin
torchrun --nproc_per_node=4 -m scripts.warm_launch schedule --schedule <x>.json       # schedule replay
```

It wraps `transfer.init_transfer_group` so that, before any worker reports READY, every rank pair exchanges a
1024-float tensor. Each rank uses blocking send/recv in the same pair order, so the exchange can't deadlock. This
builds NCCL's lazily created 2-rank communicators before the clock starts. Then it calls `run_experiment.main()` or
`run_schedule.main()`.

### Environment for split layouts

The shell profile sets `NCCL_NET`, `NCCL_TUNER_CONFIG_PATH` and `LD_LIBRARY_PATH` for GCP's gIB network plugin,
which this single machine doesn't have. With them set, NCCL fails to initialize. Run with them unset:

```
TR="env -u NCCL_NET -u NCCL_TUNER_CONFIG_PATH -u LD_LIBRARY_PATH .venv/bin/torchrun --nproc_per_node=4 -m scripts.warm_launch"
$TR schedule --schedule ../schedules/extract_2p2d_r5_n32_s1234.json
$TR experiment --prefill 0,1 --decode 2,3 --trace extract --rate 5 --n-requests 32 --seed 1234 \
    --name rr_extract_2p2d_r5_n32_s1234
```

Layout flags for `experiment`:

| Layout | Flags |
|---|---|
| colocated | `--colocated 0,1,2,3` |
| 1p3d | `--prefill 0 --decode 1,2,3` |
| 2p2d | `--prefill 0,1 --decode 2,3` |
| 3p1d | `--prefill 0,1,2 --decode 3` |

`rank` looks for round-robin results named `rr_<trace>_<topology>_r<rate>_n<n>_s<seed>`.

---

## End-to-end example

```bash
cd soldr
cs349e-assn2/.venv/bin/python calibrate_harness.py qwen3-8b_a100x4      # prefill curve
cs349e-assn2/.venv/bin/python calibrate_traces.py qwen3-8b_a100x4       # per-trace decode
.venv/bin/python assn2.py convert

.venv/bin/python assn2.py solve extract 5 --n 32 --topology 2p2d --window 16 --step 8 --limit 30   # opt
.venv/bin/python assn2.py solve extract 5 --n 32 --topology 2p2d --policy rr
.venv/bin/python assn2.py solve extract 5 --n 32 --topology 2p2d --policy warm --limit 60

cd cs349e-assn2      # with $TR as above
$TR schedule --schedule ../schedules/extract_2p2d_r5_n32_s1234.json
$TR schedule --schedule ../schedules/extract_2p2d_r5_n32_s1234_warm.json
$TR experiment --prefill 0,1 --decode 2,3 --trace extract --rate 5 --n-requests 32 --seed 1234 \
    --name rr_extract_2p2d_r5_n32_s1234

cd ..
.venv/bin/python assn2.py compare schedules/extract_2p2d_r5_n32_s1234.json \
    cs349e-assn2/results/sched_extract_2p2d_r5_n32_s1234.jsonl \
    cs349e-assn2/results/rr_extract_2p2d_r5_n32_s1234.jsonl
.venv/bin/python assn2.py rank extract 5 --n 32
```
