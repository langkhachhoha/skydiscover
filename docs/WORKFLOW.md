# BLADE — runtime workflow

This file describes **what actually executes** when you run BLADE,
in chronological order, with every branch and failure-recovery path
called out. It complements [BLADE.md](BLADE.md), which covers the
theory (math, design choices, defaults). Read this when you want to
understand *why* the loop did what it did during a run.

Source of truth: [`levi/levi/blade/orchestrator.py`](../levi/levi/blade/orchestrator.py).

---

## 0. Vocabulary

* **Mutation model** — small, cheap LLM (default: Qwen3-30B). Drives
  the high-frequency mutation / crossover / repair / fallback-summary
  calls.
* **Paradigm model** — frontier reasoning LLM (default: GPT-5). Used
  rarely, only inside `_paradigm_shift`.
* **Embedder** — separate embedding endpoint (default:
  `text-embedding-3-small`). Encodes natural-language *descriptions*
  of programs, never the code itself.
* **Pool** — top-K bag of programs with description-embedding niching
  and family clustering. K=100 by default.
* **Monitor** — bookkeeping object. Tracks `eval_count`, `best_score`,
  three sliding-window signals, and exposes `is_stuck()`,
  `is_collapsing()`, `stagnation_level()`.
* **`eval_count`** — every *finished attempt* (accepted, rejected,
  parse-failed, LLM-errored, evaluator-errored) increments this by
  exactly one. It is the loop's heartbeat.

---

## 1. Run lifecycle (top-level)

```text
evolve_code_blade(...)
  └─ BladeOrchestrator(config).run()
       ├─ start_time = now()
       ├─ open ResilientProcessPool (n_eval_processes subprocesses)
       ├─ await _bootstrap_population()         # phase A — LEVI-style 2-phase init
       ├─ create_task(_pe_monitor())            # background paradigm-shift cron
       ├─ create_task(_meta_advice_monitor())   # background lessons-learnt cron
       ├─ await _main_loop()                    # phase B — generate / repair
       ├─ cancel _pe_monitor + _meta_advice_monitor
       └─ close pool, write snapshot.json + best.py, return BladeResult
```

Phase A is sequential and blocks until done. Phase B's main loop runs
in parallel with two background tasks: the PE monitor (fires paradigm
shifts on boundary crossings) and the meta-advice monitor (refreshes
the "lessons learnt" block on boundary crossings). If phase A fails
to produce *any* seeds, the first worker re-runs it as a safety net
(see §3 below).

---

## 2. Phase A — bootstrap (`_bootstrap_population`)

LEVI-style two-phase initial population. Goal: end up with a sizable,
*diverse* seed pool (up to `n_diverse_seeds × n_variants_per_seed`
candidates, default 5 × 20 = 100) before the evolutionary loop starts.
Mirrors `levi.init.diversifier.Diversifier`.

### 2.1 Phase 1 — diverse seeds (SEQUENTIAL, frontier model)

```text
diverse_seeds = []
if seed_program is provided:
    score, err = _evaluate_code(seed_program)
    if no err:
        _admit(seed_program, source="init")
        diverse_seeds.append((code, score, description))
    else:
        record_eval(reject)                     # advance eval_count
n_seeds = n_diverse_seeds (+1 if no seed_program — LEVI parity)
for i in range(n_seeds):
    if budget exhausted: break
    for attempt in 1..3:                        # up to 3 retries per slot
        prompt = build_diverse_seed_prompt(
            existing_seeds=[(c,s) for c,s,_ in diverse_seeds])  # frontier sees all prior seeds
        raw = await paradigm_lm(prompt,
                                temperature=init_diversity_temperature,
                                max_tokens=None)               # never cap frontier
        parsed = OutputParser.parse(raw)
        if not parsed.has_code or LLM raised: record_eval(reject); continue
        score, err = _evaluate_code(parsed.code)
        if err: error_buffer.append; record_eval(reject); continue
        description = _summarize_if_needed(parsed.code, parsed.description)
        _admit(code, source="init")
        diverse_seeds.append((code, score, description))
        break                                  # success → next seed
```

Seeds are generated **sequentially** (not parallel) because each
prompt must see *all* previously accepted seeds to push the model
toward a fundamentally different paradigm. This is the LEVI design
and we mirror it as-is.

### 2.2 Phase 2 — variants (PARALLEL, mutation model)

```text
if not diverse_seeds: skip phase 2
n_variants = n_variants_per_seed × len(diverse_seeds)
# Build all prompts up front — each variant sees 2 randomly sampled
# seeds (or all of them, if fewer than 2 exist) as inspirations.
prompts = []
for seed in diverse_seeds:
    for _ in range(n_variants_per_seed):
        inspirations = random.sample(diverse_seeds, min(2, len(diverse_seeds)))
        prompts.append(build_init_variant_prompt(inspirations))

# Fan out all of them at once.
async def one_variant(prompt):
    if budget exhausted: return
    raw = await mutation_lm(prompt, temperature=init_variant_temperature)
    parsed = OutputParser.parse(raw)
    if not parsed.has_code or LLM raised: record_eval(reject); return
    score, err = _evaluate_code(parsed.code)
    if err: error_buffer.append; record_eval(reject); return
    description = _summarize_if_needed(parsed.code, parsed.description)
    _admit(code, source="init")

await asyncio.gather(*(one_variant(p) for p in prompts))
```

After phase 2, the pool typically holds anywhere from ~5 programs
(only seeds survived) up to ~5 + 5×20 = 105 (everything passed). The
evolutionary loop in phase B starts immediately afterward.

**Failure mode:** if phase 1 produces zero seeds (all retries failed
for every slot), phase 2 is skipped and the pool is empty. The
safety net in `_generate_one` (§4) catches this.

---

## 3. Phase B — main loop (`_main_loop` + `_pe_monitor`)

Two coroutines run in parallel: the mutation-worker loop and the PE
monitor task. Paradigm shifts are scheduled by the PE monitor (not
inline in the worker loop), so the worker loop only owns mutation +
repair.

### 3.1 Worker loop (`_main_loop`)

Holds two task slots:

* `in_flight: set[Task]` — up to `n_workers` (default 4) mutation
  workers.
* `repair_task: Task | None` — at most **one** repair in flight.

Exit conditions (any one):

* `stop_event` set (a worker observed `_budget_exhausted()`),
* `_budget_exhausted()` directly (`budget_evals`, `budget_dollars`,
  `budget_seconds`, or `target_score`).

There is **no stall-detection / early-stop**. The loop runs until a
budget exit condition trips.

```text
while running:
    # (a) top up the worker pool
    while len(in_flight) < n_workers and budget OK:
        in_flight.add(create_task(_generate_one()))

    # (b) opportunistic repair if buffer has fresh errors AND none in flight
    if (repair_task is None or repair_task.done()) and error_buffer non-empty:
        repair_task = create_task(_repair_one())

    # (c) wait for ANY task in the wait_set to finish
    await asyncio.wait({in_flight ∪ repair_task?}, FIRST_COMPLETED)
    log any exceptions
    prune in_flight to non-done
```

### 3.2 PE monitor (`_pe_monitor`) — boundary-crossing gate

Background task that wakes every ~2 s and decides whether to fire a
paradigm shift. Adapted from LEVI's `_pe_monitor`, with one
modification: BLADE's `asyncio.gather` in bootstrap phase 2 and
paradigm fanout makes `eval_count` jump in bursts (e.g. 12 → 17
between two wake-ups), so an exact-modulo gate (`ec % interval == 0`)
would silently skip boundaries. We use a **boundary-crossing** gate
instead. Holds `_pe_lock` so at most one paradigm shift ever runs at
a time.

```text
while not stop_event and not budget_exhausted:
    await asyncio.sleep(2.0)
    ec = monitor.eval_count
    if ec > 0 and ec >= last_pe_eval_count + pe_cron_interval:
        last_pe_eval_count = ec            # advance gate so we don't immediately refire
        async with _pe_lock:
            pe_trigger_count += 1
            await _paradigm_shift()         # spends 1 + n_paradigm_variants evals
            last_pe_eval_count = max(last_pe_eval_count, monitor.eval_count)
                                            # snap past the variants we just ran
```

With `pe_cron_interval=N` you should see approximately
`(eval_count - phase0_evals) / N` triggers per run.

---

## 4. Worker step (`_generate_one`)

The heart of the throughput pipeline. Acquires one slot of
`self._semaphore` (= `n_workers`) and runs end-to-end:

```text
async with self._semaphore:
    if budget exhausted: stop_event.set(); return
    programs = pool.programs()

    # Cold-start safety net:
    if not programs:
        await _bootstrap_population()    # re-run the 2-phase init
        if still empty:
            monitor.record_eval(reject)  # avoid infinite empty returns
        return

    stuck = monitor.is_stuck()
    op    = "crossover" w.p. p_xover, else "mutate"
    temp  = llm_temperature_stuck if stuck else llm_temperature

    try:
        # 4.1 — build prompt
        if op == crossover and len(programs) >= 2:
            (p_a, p_b)  = selector.select_two_parents(programs, …)
            inspirations = selector.select_inspirations(programs, exclude=[p_a,p_b], …)
            prompt = build_crossover_prompt(…, inspirations)
            parent_score = max(p_a.score, p_b.score)
        else:
            parent       = selector.select_parent(programs, …)
            inspirations = selector.select_inspirations(programs, exclude=[parent], …)
            prompt = build_mutate_prompt(…, inspirations)
            parent_score = parent.score

        # 4.2 — call LLM
        raw = await mutation_lm(prompt, temperature=temp)

        # 4.3 — parse
        parsed = OutputParser.parse(raw)
        if not parsed.has_code:
            monitor.record_eval(-inf, accepted=False)   # NEW
            return

        # 4.4 — evaluate (subprocess pool, eval_timeout)
        score, err = _evaluate_code(parsed.code)
        if err is not None:
            error_buffer.append((parsed.code, parent_score, err))
            monitor.record_eval(score, accepted=False)
            return

        # 4.5 — summarize if description is too short
        description = _summarize_if_needed(parsed.code, parsed.description)

        # 4.6 — embed description, push to pool, update monitor
        await _admit(code=parsed.code, description=description,
                     score=score, source=op, parent_score=parent_score)
    except Exception:
        # NEW: any uncaught failure (provider 5xx, rate-limit, network)
        # counts as a reject so eval_count advances.
        monitor.record_eval(-inf, accepted=False)
```

**Crucial guarantee (after recent fixes):** every code path through
`_generate_one` either calls `_admit()` (which calls `record_eval`
internally) or calls `record_eval` directly. There is no longer any
silent early-return that fails to advance `eval_count`.

---

## 5. Paradigm shift (`_paradigm_shift`)

Fired by the PE monitor on cron-modulo + freshness (§3.2). LEVI-style
**two-step**: frontier seed + mutation fanout. Total eval cost per
trigger ≈ 1 + `n_paradigm_variants` (default 1 + 4 = 5).

### 5.1 Step 1 — frontier paradigm seed (1 LLM call)

```text
if len(pool) < paradigm_min_pool_size: return    # not enough representatives
stage = get_budget_stage(
    budget_progress=_budget_progress(),          # real budget fraction now
    stagnation=monitor.stagnation_level(),
)                                                 # → "early" | "mid" | "late"
reps      = pool.representatives(stage, n=3)
rep_pairs = [(p.description, p.score) for p in reps]
for r in reps: pool.mark_used(r)

prev_best = monitor.best_score
prompt    = build_paradigm_prompt(stage, …, recent_trials)

try:
    raw = await paradigm_lm(prompt, temperature=0.7,
                            max_tokens=None)      # NEVER cap (reasoning-heavy)
except Exception:
    record_eval(reject); return                   # no fanout — nothing to fan from

parsed = OutputParser.parse(raw)
if not parsed.has_code:
    paradigm_trials.append(ParadigmTrial(accepted=False, …))
    record_eval(reject); return                   # no fanout

score, err = _evaluate_code(parsed.code)
description = _summarize_if_needed(parsed.code, parsed.description)
accepted = False
if err is None:
    accepted, _ = await _admit(code, description, score, source="paradigm",
                               parent_score=prev_best)
else:
    error_buffer.append((parsed.code, prev_best, err))
    record_eval(score, accepted=False)

paradigm_trials.append(ParadigmTrial(accepted=accepted, score=score, delta=…))
if accepted:
    pool.reset_uses_after_paradigm()              # fresh paradigm gets clean novelty budget
```

### 5.2 Step 2 — mutation fanout (PARALLEL, n_paradigm_variants calls)

If the frontier seed produced usable, evaluable code (regardless of
whether it was *admitted* by the pool), fan out K variants from it:

```text
if frontier failed (err or no code) or budget_exhausted: skip fanout
base_code, base_score = parsed.code, score   # from step 1

async def one_paradigm_variant():
    if budget exhausted: return
    prompt = build_paradigm_variant_prompt(base_code, base_score)  # LEVI's VARIANT_GENERATION_PROMPT
    raw = await mutation_lm(prompt, temperature=paradigm_variant_temperature)
    parsed_v = OutputParser.parse(raw)
    if not parsed_v.has_code or LLM raised: record_eval(reject); return
    v_score, v_err = _evaluate_code(parsed_v.code)
    if v_err: error_buffer.append; record_eval(reject); return
    v_description = _summarize_if_needed(parsed_v.code, parsed_v.description)
    _admit(parsed_v.code, source="paradigm_variant", parent_score=base_score)

await asyncio.gather(*(one_paradigm_variant() for _ in range(n_paradigm_variants)))
```

**Why representatives are descriptions, not code:** the frontier
model reasons over *paradigms*, not implementations. Sending code
biases it toward edits-of-existing rather than genuine shifts.

**Why fanout uses the mutation model:** the frontier model is slow
and expensive; once we have one paradigm seed it's the small model's
job to spin off K nearby variants. Same as LEVI.

---

## 6. Repair (`_repair_one`)

Opportunistic background task. Picks the freshest entry from
`error_buffer` (max 64 retained, deque-style FIFO eviction) and asks
the mutation model to fix it.

```text
if not enable_repair or not error_buffer: return
broken_code, parent_score, error_msg = error_buffer.popleft()
prompt = build_repair_prompt(broken_code, parent_score, error_msg)
try:
    raw = await mutation_lm(prompt, temperature=0.4)
except Exception:
    monitor.record_eval(-inf, accepted=False)     # NEW
    return
parsed = OutputParser.parse(raw)
if not parsed.has_code:
    monitor.record_eval(-inf, accepted=False)     # NEW
    return
score, err = _evaluate_code(parsed.code)
if err is not None:
    monitor.record_eval(score, accepted=False)    # one-shot — no retry
    return
description = _summarize_if_needed(parsed.code, parsed.description)
await _admit(parsed.code, description, score, source="repair", parent_score)
```

**Repair is one-shot.** A repaired-but-still-broken candidate is
*not* re-queued — that would create unbounded retry loops. The
mutation model gets exactly one attempt per error.

---

## 6.5. Meta-advisor (`_meta_advice_monitor`)

LEVI-style background task that refreshes a short "lessons learnt"
paragraph every `meta_advice_interval` evaluations (default 50) and
injects it into subsequent mutate/crossover prompts. Mirrors LEVI's
`_generate_meta_advice` / `current_meta_advice` pattern.

### Trigger

Same boundary-crossing semantics as the PE monitor (see §3.2). Fires
when `eval_count >= last_meta_advice_eval_count + meta_advice_interval`.
Held under `_meta_advice_lock` so only one advisor call runs at a
time. Disabled when `enable_meta_advice=False`.

### Generation

```text
async def _generate_meta_advice():
    prompt = build_meta_advice_prompt(
        problem_description,
        function_signature,
        best_score          = monitor.best_score,
        n_evaluations       = monitor.eval_count,
        accept_rate         = monitor.acceptance_rate(),
        stagnation_level    = monitor.stagnation_level(),
        recent_errors       = error_buffer tail (5),
        previous_advice     = current_meta_advice,    # so the model refines, not restarts
    )
    raw = await mutation_lm(prompt,
                            temperature=meta_advice_temperature,    # 0.4
                            max_tokens=meta_advice_max_tokens)      # 400
    if raw.strip():
        current_meta_advice = raw.strip()[:1200]   # defensive cap
```

The advisor uses the **mutation (small) model**, not the frontier
model — this is a high-frequency, short-output call where the small
model is fine and the frontier model would be too expensive.

The output is plain prose; it does NOT go through `OutputParser`
(there is no code to extract). It's stored as-is in
`current_meta_advice`.

### Injection

In `_generate_one`, every mutate/crossover prompt build pulls advice
via `_pick_meta_advice()`:

```text
def _pick_meta_advice() -> str | None:
    if not enable_meta_advice or current_meta_advice is None:
        return None
    if random.random() >= meta_advice_inject_p:    # default 0.8
        return None
    return current_meta_advice
```

The 80% gating mirrors LEVI: it keeps some variance across prompts
even while the advice is fresh. The prompt builders
(`build_mutate_prompt`, `build_crossover_prompt`) inject the block as
`## Lessons learnt so far\n{advice}\n\n` between the parents and the
task instruction.

### Failure modes

* LLM call fails → log and KEEP previous advice (the old advice may
  still be useful next interval).
* Empty response → same: keep previous advice.
* The advisor never blocks worker progress — it runs in its own task.

### Snapshot

`snapshot.json` carries a top-level `meta_advice` block:

```json
"meta_advice": {
  "enabled": true,
  "interval": 50,
  "trigger_count": 3,
  "current": "Avoid …"
}
```

Use this to verify the advisor actually ran (`trigger_count > 0`)
and inspect what advice was active at run end.

---

## 7. Admit (`_admit`)

Where new programs enter the pool. Called from `_generate_one`,
`_repair_one`, `_paradigm_shift` (both step 1 and step 2 variants),
and `_bootstrap_population` (both phase 1 and phase 2 variants).

```text
embedding = await asyncio.to_thread(embedder.embed, description)
program = Program(code, description, score, embedding, source, created_at_eval)
accepted, reason = pool.add(program)            # 3-stage filter (niche / family / top-K)
monitor.record_eval(score=score,
                    accepted=accepted,
                    embedding=embedding if accepted else None)
return accepted, reason
```

The `monitor.record_eval` call here is what advances `eval_count` on
the happy path. The `accepted` flag is what feeds the monitor's
accept-rate window — rejected duplicates *do* increment `eval_count`
but count as a non-accept.

---

## 8. Where `eval_count` advances — exhaustive list

After the recent fixes, **every** of the following advances
`eval_count` by exactly 1:

| Path                                                  | Outcome                       |
| ----------------------------------------------------- | ----------------------------- |
| Bootstrap P1 seed eval ok → admit                     | accepted (init seed)          |
| Bootstrap P1 seed eval errored                        | reject + push to error_buffer |
| Bootstrap P1 LLM raised / parse miss                  | reject (score = −∞)           |
| Bootstrap P1 seed_program eval errored                | reject (score = −∞)           |
| Bootstrap P2 variant ok → admit                       | accepted or rejected (pool)   |
| Bootstrap P2 variant eval errored                     | reject + push to error_buffer |
| Bootstrap P2 LLM raised / parse miss                  | reject (score = −∞)           |
| Worker → LLM ok → parse ok → eval ok → admit          | accepted or rejected (pool)   |
| Worker → LLM ok → parse ok → eval errored             | reject + push to error_buffer |
| Worker → LLM ok → parse miss                          | reject (score = −∞)           |
| Worker → LLM call raised                              | reject (score = −∞)           |
| Worker → empty pool after re-bootstrap                | reject (score = −∞)           |
| Repair → LLM ok → parse ok → eval ok → admit          | accepted or rejected (pool)   |
| Repair → LLM ok → parse ok → eval errored             | reject (drop, one-shot)       |
| Repair → LLM ok → parse miss                          | reject (score = −∞)           |
| Repair → LLM call raised                              | reject (score = −∞)           |
| Paradigm step 1 → LLM ok → parse ok → eval ok → admit | accepted or rejected (pool)   |
| Paradigm step 1 → eval errored                        | reject + push to error_buffer |
| Paradigm step 1 → LLM raised / parse miss             | reject (score = −∞)           |
| Paradigm step 2 variant → LLM ok → eval ok → admit    | accepted or rejected (pool)   |
| Paradigm step 2 variant → eval errored                | reject + push to error_buffer |
| Paradigm step 2 variant → LLM raised / parse miss     | reject (score = −∞)           |

Paths that do **not** advance `eval_count` (by design):

* Bootstrap P1 budget exhausted before any LLM call → no work started.
* Bootstrap P2 budget exhausted before variant launches → skipped.
* Worker hit `_budget_exhausted()` at the top → no work was started.
* Pool `< paradigm_min_pool_size` at paradigm entry → no work was started.
* Paradigm step 2 skipped because step 1 failed (no base to fan from).
* `enable_repair=False` or empty error_buffer at repair entry → no
  work was started.

---

## 9. Termination

```text
_main_loop exits  →  cancel _pe_monitor (cooperative)
                  →  drain remaining workers + repair task
                  →  return BladeResult(best_program, best_score, …)
                  →  _save_snapshot writes snapshot.json + best.py
                  →  ResilientProcessPool shut down
```

Workers and repair are cancelled cooperatively at shutdown; in-flight
evaluations get to complete (cancelling mid-evaluation can corrupt
the subprocess pool). The PE monitor is cancelled immediately so it
stops scheduling new paradigm shifts.

---

## 10. What recent changes brought BLADE in line with LEVI

1. **LEVI-style 2-phase bootstrap.** `_bootstrap_population` runs
   phase 1 (frontier model generates `n_diverse_seeds` diverse seeds
   sequentially) then phase 2 (mutation model spins off
   `n_variants_per_seed` variants per seed, in parallel via
   `asyncio.gather`). Replaces the old one-shot draft prompt.
2. **LEVI-style paradigm shift.** `_paradigm_shift` is now two-step:
   step 1 frontier seed, step 2 K-way mutation fanout. Step 2 is
   skipped only if step 1 produced no usable code.
3. **LEVI-style PE trigger.** The `_pe_monitor` background task wakes
   every 2 s and fires on `ec > 0 and ec % interval == 0 and ec !=
   last_pe_eval_count`. After firing, `last_pe_eval_count` is bumped
   past the K+1 evals just spent. Replaces the inline `next_pe_at`
   counter that had a back-to-back-firing bug at small intervals.
4. **No more early-stop.** The stall watchdog (`stall_ticks >= 40 →
   break`) is gone. The loop runs to budget.
5. **Every failure path records an eval.** Bootstrap P1, bootstrap P2,
   workers, repair, paradigm step 1, paradigm step 2 — all code paths
   now call `record_eval` on parse misses, LLM exceptions, eval
   errors. `eval_count` advances per LEVI's semantics.
6. **Real budget-progress in stage routing.** `_paradigm_shift` now
   computes `budget_progress = min(eval_count/budget_evals, …)` and
   passes it to `get_budget_stage`, so the early/mid/late prompt
   routing actually reflects how far the run has come.

---

## 11. Reading a run

When you're staring at a snapshot or live logs and trying to figure
out what happened:

* **Pool size at end of phase A** → should be roughly
  `n_diverse_seeds + n_diverse_seeds × n_variants_per_seed` minus any
  rejections/dups. If it's much smaller, the mutation model is
  producing non-evaluable code; check the parse-miss / eval-error
  counters in the snapshot.
* **`monitor.eval_count` ≪ `total LLM calls`** → many failures
  (parse, eval, provider). Look at the worker exception log lines
  (`[BLADE] worker step failed`) and the parse-miss debug lines.
* **`monitor.best_score` plateaued, `pool_size` growing** → mutation
  is finding novel-but-equal-or-worse programs. Pool is doing its
  job; selector should start pulling under-used branches via the UCB
  novelty term.
* **`paradigm_trials` mostly `accepted=False`** → frontier paradigm
  shifts aren't beating the incumbent. Either the problem is mostly
  saturated, or the paradigm prompt isn't helpful here. Try a larger
  `pe_cron_interval` so frontier cost goes to compute that helps.
* **`paradigm_trials` count ≈ `eval_count / pe_cron_interval`** →
  cron is working as expected.
* **`paradigm_trials` count ≫ `eval_count / pe_cron_interval`** →
  shouldn't happen anymore (fixed); if you still see it, the
  back-to-back guard regressed.
