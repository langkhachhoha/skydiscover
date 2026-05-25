# WORKFLOW_BLADE — Luồng hoạt động của BLADE

> BLADE = **B**ehavior-**L**atent **A**daptive **D**iscovery **E**ngine.
> Mục tiêu: cùng một bài toán tối ưu mã nguồn (LEVI examples) nhưng thay 3 hệ
> con "nặng đô" của LEVI bằng các phiên bản nhẹ-và-mạnh hơn:
>
> | Hệ con | LEVI | BLADE |
> | --- | --- | --- |
> | Archive | CVT-MAP-Elites (centroid behavioural grid) | Top-K **Pool** với 2 lớp dedup + **quota niching** + **Hall-of-Fame** ([levi/levi/simple/pool.py](levi/levi/simple/pool.py)) |
> | Stagnation signal | PPS formula (Punctuated-Equilibrium PPS) | 3 sliding-window stats: accept-rate / plateau / diversity ([levi/levi/simple/monitor.py](levi/levi/simple/monitor.py)) |
> | Sampler | 4-D Thompson bandit (SAL) | UCB-style **Selector** + **paradigm-source boost** ([levi/levi/simple/selector.py](levi/levi/simple/selector.py)) |
>
> Frontier prompts (3-phase paradigm shift), error-archive self-repair, async
> producer/consumer giữ nguyên ý tưởng từ LEVI; nhưng paradigm prompt đã được
> viết lại BLADE-native (anchors có CODE + inspirations description-only) và
> được "vũ trang" với cross-family anchor selection + HoF backfill + stuck-aware
> stage/temperature routing.

Entry point: [scripts/run_blade.py](scripts/run_blade.py) → `levi.evolve_code_blade()`
([levi/levi/methods/blade.py](levi/levi/methods/blade.py))
→ `BladeOrchestrator.run()` ([levi/levi/blade/orchestrator.py](levi/levi/blade/orchestrator.py)).

---

## 0. Bản nâng cấp này giải quyết bốn vấn đề quan sát được

Run trước (3 giờ, circle_packing_rect) cho thấy:

- Pool 100 elite nhưng **chỉ 1 family** (toàn bộ collapse) — `mean_recent_diversity = 0.81`, `is_collapsing = True`.
- 21 paradigm trials, 9/21 accepted nhưng delta vs prev_best đều **âm** — paradigm bị evict ngay sau khi admit, không kịp fanout.
- AST signature 14-count log-vec: cross-paradigm cosine ≈ 0.95-0.99 → lớp AST **luôn pass**, dedup chỉ dựa description embedding một mình.
- Late-stage paradigm prompt yêu cầu "surgical fix on best anchor" trong khi pool đã stuck — đảm bảo paradigm-shift biến thành paradigm-stay.

Bốn component kiến trúc mới khắc phục bốn vấn đề trên (mỗi component có flag `enable_*` riêng để ablation):

| Vấn đề | Component | Toggle |
| --- | --- | --- |
| Pool collapse về 1 family | **A. Quota niching** — family cap kick-in từ admit đầu tiên (không defer) + family threshold 0.72→0.85 | `PoolConfig.enable_quota_niching` |
| AST gate luôn pass | **B. Bigram histogram** — (parent, child) node-type histogram thay 14-count log-vec | `PoolConfig.ast_mode ∈ {"bigram", "count14"}` |
| Paradigm-shift không phá vỡ stagnation | **C. Cross-family anchors + stuck-early routing + bumped temperature** | `paradigm_cross_family_anchors`, `paradigm_force_early_on_collapse` |
| Paradigm tốt bị loại sớm | **D. Hall-of-Fame + paradigm grace + selector boost** | `PoolConfig.enable_hall_of_fame`, `enable_paradigm_grace`, `SelectorConfig.enable_paradigm_boost` |

Defaults = **tất cả ON**. Để ablate component nào, set flag tương ứng = `False`
hoặc dùng cờ `--disable-*` của CLI. Snapshot.json ghi `ablation_flags` block đầy đủ
để post-hoc attribution.

---

## 1. Tổng quan luồng

```text
┌─────────────────────────────────────────────────────────────────────┐
│  run_blade.py (CLI)                                                  │
│      │   parse args (kèm --disable-* ablation toggles)              │
│      │   load problem.py, setup output dir                          │
│      ▼                                                               │
│  evolve_code_blade()  ──►  _setup_logging() + BladeConfig            │
│      │                                                               │
│      ▼                                                               │
│  BladeOrchestrator.run()                                             │
│      │                                                               │
│      ├── [start] _status_monitor (heartbeat 30s)                    │
│      │                                                               │
│      ├── PHASE 1 — Diverse seeds (SEQUENTIAL, frontier GPT-5)       │
│      │     for i in 1..n_diverse_seeds:                              │
│      │       build_diverse_seed_prompt → paradigm_lm → eval → admit │
│      │                                                               │
│      ├── PHASE 2 — Init variants (PARALLEL, mutation Qwen)          │
│      │     n_diverse_seeds × n_variants_per_seed prompts            │
│      │     asyncio.gather → build_init_variant_prompt → mutation_lm │
│      │                                                               │
│      ├── [start] _pe_monitor (mỗi 2s, kick paradigm-shift)          │
│      ├── [start] _meta_advice_monitor (mỗi 2s, refresh advice)      │
│      │                                                               │
│      └── MAIN LOOP — mutate / crossover / repair workers             │
│            ▲                                                         │
│            │  Mỗi N evals (pe_cron_interval) → paradigm shift:       │
│            │    1) ANCHORS = cross-family + HoF backfill             │
│            │    2) stage = stuck/collapse ? "early" : budget-route   │
│            │    3) frontier_temp = stuck ? 1.0 : 0.7                 │
│            │    4) n_paradigm_variants fanout (qwen, parallel)      │
│            │    5) seed admitted với paradigm grace (30 evals)       │
│            │                                                         │
│            │  Mỗi N evals (meta_advice_interval) → advisor refresh   │
│            │                                                         │
│            │  Pool.tick_eval() đồng bộ eviction clock                │
│            ▼                                                         │
│        budget exhausted → snapshot.json + best.py + HoF              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Cấu hình mặc định (GitHub Actions run)

Workflow: [.github/workflows/blade.yml](.github/workflows/blade.yml)

### Knobs cũ (giữ nguyên)

| Tham số | Giá trị mặc định | Ý nghĩa |
| --- | --- | --- |
| `mutation_model` | `openrouter/qwen/qwen3-30b-a3b-instruct-2507` | "Worker bee" — mọi LLM call tần suất cao |
| `paradigm_model` | `openrouter/openai/gpt-5` | Frontier — diverse seeds + paradigm shift |
| `embedding_model` | `openrouter/openai/text-embedding-3-small` | Description embedding cho Pool niching |
| `workers` | 4 | Số coroutine LLM song song trong main loop |
| `eval_processes` | 4 | Subprocess pool đánh giá code |
| `pe_interval` | 10 (workflow) / 50 (CLI default) | Mỗi N evals kích hoạt paradigm shift |
| `n_diverse_seeds` | 4 | Số seed đa dạng phase 1 |
| `n_variants_per_seed` | 20 | Variant/seed ở phase 2 (tổng init ≈ 4×20 = 80) |
| `n_paradigm_variants` | 4 | Fanout sau mỗi paradigm seed |
| `meta_advice_interval` | 50 | Cron refresh advice |
| `pool_k` (PoolConfig.K) | 100 | Target population size |
| `niche_cosine_threshold` | 0.92 | Description-embedding cosine **gắn cờ** trùng nghĩa (lớp 1) |
| `paradigm_n_anchors` | 3 | Số anchor representative (kèm code) cho frontier |
| `paradigm_n_inspirations` | 5 | Số inspiration description-only |
| `budget_seconds` | 10800 (3 giờ) | Wall-clock cap |

### Knobs mới (component A+B+C+D)

| Tham số | Default | Ý nghĩa |
| --- | --- | --- |
| `ast_mode` | **`"bigram"`** | Component B. AST signature implementation. `"count14"` reproduces legacy. |
| `structural_cosine_threshold` | **0.85** (đã hạ từ 0.97) | Lớp 2 AST — với bigram, cross-paradigm cosine ở 0.4-0.6 và paraphrase ở 0.9+ → 0.85 phân biệt sạch. |
| `family_cosine_threshold` | **0.85** (đã tăng từ 0.72) | Single-linkage gom cụm "họ" thuật toán. 0.72 quá lỏng, gộp mọi paraphrase vào 1 family. |
| `enable_quota_niching` | **True** | Component A. Family cap fire từ admit đầu (không defer) → ngăn pool collapse trong giai đoạn fill. |
| `target_n_families` | 5 | Mỗi family được cấp `ceil(K/N)` slots = 20 (với K=100, N=5). |
| `max_per_family` | 10 | Hard ceiling per family (hard cap nhỏ hơn quota → cap này fire trước). |
| `enable_paradigm_grace` | **True** | Component D. Paradigm-source program có `protected_until_eval = current + 30` → không bị evict trong grace window. |
| `paradigm_grace_evals` | 30 | Length grace window (≈ 1 fanout + vài rounds mutate/crossover). |
| `enable_hall_of_fame` | **True** | Component D. Side-store paradigm seeds + top-score-per-family ever. |
| `hof_size` | 30 | Capacity HoF. Paradigm-source entries không bao giờ evict. |
| `enable_paradigm_boost` | **True** | Component D. Selector boost cộng vào UCB priority cho paradigm-source. |
| `paradigm_boost` | 0.6 | Magnitude boost (additive, decay linear theo age). |
| `paradigm_exploit_window` | 25 | Bao lâu boost còn active sau `created_at_eval`. |
| `paradigm_cross_family_anchors` | **True** | Component C. Anchors = 1 top/family + HoF backfill (thay top-3-by-score). |
| `paradigm_force_early_on_collapse` | **True** | Component C. `is_stuck() OR is_collapsing()` → force stage="early". |
| `paradigm_temperature` | 0.7 | Frontier temperature healthy regime. |
| `paradigm_temperature_stuck` | **1.0** | Frontier temperature khi stuck/collapsing. |
| `paradigm_variant_temperature_stuck` | **1.0** | Variant fanout temperature khi stuck. |

---

## 3. Phase 1 — Diverse Seeds (sequential, frontier)

Code: `BladeOrchestrator._bootstrap_population()` phase 1, prompt builder
`build_diverse_seed_prompt` ([levi/levi/blade/prompts.py](levi/levi/blade/prompts.py)).

- **Model**: `paradigm_lm` (GPT-5).
- **Tuần tự**: mỗi prompt sau phải nhìn các seed đã chấp nhận trước đó để được
  "đẩy" theo hướng paradigm khác hẳn.
- **Retry**: tối đa 3 lần per seed (LLM lỗi / parse miss / eval fail đều retry).
- **Admit qua Pool.add**: nếu `ast_mode="bigram"`, structural signature được
  compute ngay khi admit và **không có HoF/grace gì đặc biệt** ở phase này
  (init không phải paradigm-source).
- **Prompt**: wrap LEVI's `DIVERSITY_SEED_PROMPT` + `OUTPUT_FORMAT_INSTRUCTION`.

Sau phase 1: log `[BLADE init] phase 1 done: K seeds admitted`.

## 4. Phase 2 — Init Variants (parallel, mutation)

Code: cùng method, phase 2. Prompt builder: `build_init_variant_prompt`.

- **Model**: `mutation_lm` (Qwen 30B).
- **Parallel**: `asyncio.gather` chạy `n_diverse_seeds × n_variants_per_seed`
  prompt đồng thời.
- Mỗi variant prompt lấy **2 seed ngẫu nhiên** làm inspiration (code + score).
- Mục tiêu: khai thác chiều sâu xung quanh từng paradigm — *giữ paradigm gốc*.
- **Quota niching đã active**: nếu phase 1 chỉ tạo 2 family thực sự khác nhau
  và phase 2 sinh 80 variants thì pool **không** chứa 80 variants — quota cap
  20/family sẽ evict yếu nhất. Đây là intent: pool đầu vào main-loop đã có
  cấu trúc đa dạng-bằng-cấu-trúc thay vì cào bằng theo score.

## 5. Main Loop — mutate / crossover / repair

Code: `_main_loop()` + `_generate_one()` + `_repair_one()`.

- **Workers**: tối đa `n_workers` coroutine `_generate_one` chạy song song
  (cap bằng `asyncio.Semaphore`).
- **Operator** chọn ngẫu nhiên theo trạng thái stagnation (`monitor.is_stuck()`):
  - Healthy: p_crossover = 0.30 → 70% mutate, 30% crossover.
  - Stuck: p_crossover = 0.70 → ngược lại.
- **Selector** (`levi.simple.Selector`) chọn parent (1 cho mutate, 2 cho
  crossover) bằng UCB priority **+ paradigm boost** (component D):

  ```text
  priority(p; S) = score_norm
                 + α · √(log(1+N) / (1+uses_count))           # UCB novelty
                 + β · exp(-age / τ)                          # recency
                 − γ · max cosine(p, q∈S)                     # diversity penalty
                 + paradigm_boost · max(0, 1 - age/window)    # NEW: D
                   for p.source ∈ {paradigm, paradigm_variant}
  ```

  Với `paradigm_boost=0.6` và `window=25`, một paradigm seed mới (age=0) được
  cộng thêm 0.6 priority — đủ để outrank các elite cao điểm hơn một chút và
  đảm bảo workers thực sự mutate/crossover quanh paradigm seed sau khi nó vừa
  được admit (thay vì để nó "chết" trong pool).

- **Inspirations**: thêm 2-3 program nữa từ pool, chỉ truyền `description + score`
  vào prompt.
- **Meta-advice**: ở 80% xác suất, chèn block "Lessons learnt so far".

### Mutate / Crossover prompt

Giữ nguyên (không đổi vì không phải nguồn gốc vấn đề). Xem
[levi/levi/blade/prompts.py](levi/levi/blade/prompts.py).

### Repair (one-shot)

Khi candidate raise exception ở `_evaluate_code`, BLADE đẩy
`(broken_code, parent_score, error_msg)` vào `error_buffer` (deque maxlen 64).
Main loop opportunistically `_repair_one()`. **One-shot**: nếu repair lại lỗi
thì drop, không loop. (Có thể tắt với `--no-repair`.)

## 6. Paradigm Shift 2.0 — Cross-family + Stuck-aware + HoF

Code: `_pe_monitor()` + `_paradigm_shift()` + `build_paradigm_prompt`.

### Cron trigger (không đổi)

Task chạy nền `await asyncio.sleep(2.0)`, mỗi lần wake kiểm tra
`eval_count >= last_pe_eval_count + pe_cron_interval`. Vì BLADE bootstrap
phase 2 và variant fanout submit nhiều eval qua `asyncio.gather`, dùng
**boundary-crossing** (không phải modulo) để không bị skip.
`_pe_lock` đảm bảo tối đa **một** paradigm shift in-flight.

### Stage routing — đã có cờ stuck-early (component C)

```python
stage = get_budget_stage(budget_progress, stagnation)
if paradigm_force_early_on_collapse and (is_stuck() or is_collapsing()):
    stage = "early"   # override route bất kể budget_progress
```

Lý do: legacy logic gửi search đã stuck vào `late` (do `budget_progress` đã cao)
— nhưng late-stage prompt yêu cầu "surgical fix on best anchor", chính xác là
điều **không** nên làm khi đã collapse. Forced-early ép frontier nghĩ paradigm
mới thay vì điều chỉnh paradigm hiện tại.

### Anchor selection — cross-family + HoF backfill (component C)

Legacy: `pool.representatives(stage, n=3)` → 3 best-by-score → **khi pool collapse,
3 anchors là 3 paraphrase của cùng paradigm**.

Mới: `pool.representatives_cross_family(n=3, include_hof=True)`:

1. Group programs by `family_id`, pick **1 top-score per family**.
2. Sort by score desc → strongest paradigm-anchor leads.
3. Nếu thiếu (pool ít family hơn `n`), backfill từ **Hall-of-Fame**: với mỗi
   HoF entry, chỉ pick nếu cosine vs các anchor đã chọn < `family_cosine_threshold`.
   → đảm bảo frontier luôn thấy N paradigm-distinct anchors ngay cả khi pool
   hiện tại đã collapse.

Inspirations cũng được top-up từ HoF nếu pool không đủ diversity.

### Temperature — stuck-aware (component C)

```python
paradigm_temp = 1.0 if (is_stuck() or is_collapsing()) else 0.7
variant_temp  = 1.0 if (is_stuck() or is_collapsing()) else 0.8
```

Ở 0.7, GPT-5 có xu hướng paraphrase best anchor; ở 1.0 nó thực sự rời family.
Bump variant temperature giúp variants thoát khỏi description-embedding niche
của paradigm seed (nếu không, niche-dedup gate sẽ giết hầu hết variants).

### Khung prompt (giữ nguyên về form, richer Strategy Log)

```text
# Algorithmic Paradigm Shift Challenge ({early|mid|late})

## Problem … / Function Signature …

## Archive Snapshot
The archive has evolved through {n_evaluations} evaluations and currently
contains {n_families} distinct behavioural families.

### Anchor representatives (code + description + score)
#### Anchor 1 (score=…)   ← từ family 0
_Description_: …
```python … ```

#### Anchor 2 (score=…)   ← từ family 1 (hoặc HoF backfill)
…

#### Anchor 3 (score=…)   ← từ family 2 (hoặc HoF backfill)
…

### Additional inspirations (description + score only)
1. (score=…) …
…

## Strategy Log (recent paradigm attempts) — description budget 360 chars
- [#1 early] ✓ score=… Δ=+… :: <full description, không cắt 160>
- [#2 mid]  ✗ score=… Δ=…  :: …
…

## Your Challenge: {stage-specific instructions}
{OUTPUT_FORMAT_INSTRUCTION}
```

`ParadigmTrial.render()` mới mặc định `max_desc_chars=360` (cũ 160) — với 160
chars mọi late-stage trial trông giống nhau ("A hybrid discrete-continuous
search first selects 21 centers…") và frontier kept proposing variations on
the same idea.

### Sau khi accept seed

Mutation model fanout `n_paradigm_variants` variants song song với
`build_paradigm_variant_prompt` ở `variant_temp`. Mỗi variant sinh ra được
admit qua `Pool.add` với `source="paradigm_variant"` → **được stamp grace +
HoF + selector boost** (component D).

## 7. Meta-Advisor (lessons learnt)

Không đổi. Code: `_meta_advice_monitor()` + `_generate_meta_advice()`.

- Cron boundary-crossing, mỗi `meta_advice_interval` eval.
- Model: mutation model, low-temp 0.4, cap 400 tokens.
- Output: 3-5 câu prescriptive, chèn vào 80% mutate/crossover prompt kế.

## 8. Pool 3.0 — Quota niching + Hall-of-Fame + paradigm grace

Đây là thay đổi cốt lõi so với Pool 2.0 (bản trước). Pool 3.0 không còn chế độ
"fill-first" / "at-capacity" rời rạc — thay vào đó:

### Niche dedup 2 lớp (giữ nguyên, nhưng AST mạnh hơn)

1. Compute embedding (có sẵn) + AST signature qua `compute_ast_signature(code, mode="bigram")`.
   Bigram = histogram của `(parent_node_type, child_node_type)` edges trên 40 node-types
   được chọn (`FunctionDef`, `For`, `If`, `BinOp`, `Compare`, `Subscript`, …),
   L2-normalised, dim = 40×40 = 1600.
2. **Niche check**: nếu description cosine ≥ 0.92 **VÀ** AST cosine ≥ 0.85
   với nearest neighbour → duplicate thật:
   - Điểm cao hơn ⇒ `replaced_duplicate` (inherit `uses_count` và
     `protected_until_eval` từ incumbent).
   - Không cao hơn ⇒ `dropped_duplicate`.
3. Nếu chỉ description ≥ 0.92 nhưng AST < 0.85 → admit (biến thể có cấu trúc khác).

### Quota niching — luôn enforce (component A)

Khác Pool 2.0 (defer family cap đến khi pool đầy), Pool 3.0 enforce cap **ngay
từ admit đầu tiên** khi `enable_quota_niching=True`:

```python
quota = min(max_per_family, ceil(K / target_n_families))   # = min(10, 20) = 10
if len(family_of_newcomer) > quota:
    evict the weakest non-protected member of that family
```

Nếu toàn bộ family hiện đang **protected** (paradigm grace cover all), cap được
*defer* — chấp nhận overshoot tạm thời thay vì giết paradigm đang cooling.

### Paradigm grace (component D)

Khi `program.source in {"paradigm", "paradigm_variant"}` và
`enable_paradigm_grace=True`, Pool stamp:

```python
program.protected_until_eval = current_eval + paradigm_grace_evals    # = +30
```

`current_eval` được orchestrator đồng bộ qua `pool.tick_eval(monitor.eval_count)`
sau mỗi `record_eval`. Trong grace window:

- Family cap **không evict** program đó.
- Global K cap **không evict** program đó (trừ trường hợp tất cả đều protected
  → cap defer; hoặc newcomer chính là weakest và pool ≥ K → drop newcomer).
- Niche dedup vẫn áp dụng — duplicate là chuyện khác với quota.

Khi `replaced_duplicate`, newcomer kế thừa `protected_until_eval` của incumbent
(không bao giờ làm ngắn grace).

### Hall-of-Fame (component D)

Side-store thuần read-only ngoài pool chính (`_hof: list[Program]`, capacity `hof_size=30`):

**Admit rule**:

- Mọi paradigm-source program ⇒ admit.
- Non-paradigm: admit nếu cosine vs **mọi** HoF entry hiện tại < `family_cosine_threshold`
  (i.e. nó đem lại family axis mới). Skip noise.

**Eviction rule**: chỉ evict non-paradigm. Paradigm-source entries **không bao
giờ bị evict** — chúng là long-term memory của các paradigm đã được frontier sinh ra.

Truy cập:

- `pool.hall_of_fame()` → snapshot read-only.
- `pool.representatives_cross_family(include_hof=True)` → backfill anchors.
- `snapshot.json["hall_of_fame"]` → dump artifact cuối run.

### Reasons trả về bởi `pool.add`

| Reason | Khi nào |
| --- | --- |
| `added` | Append clean |
| `replaced_duplicate` | Duplicate (cả niche+struct) score cao hơn |
| `dropped_duplicate` | Duplicate score thấp hơn |
| `replaced_family_weak` | Family cap evicted weakest non-protected (≠ newcomer) |
| `dropped_family_full` | Family cap fire và newcomer là weakest non-protected |
| `dropped_full` | Pool > K và newcomer là global weakest non-protected |
| `no_embedding` | Refuse (cần embedding) |

## 9. Vì sao Pool 3.0 + Paradigm 2.0 khắc phục được collapse

Đối chiếu run cũ (circle_packing_rect, 3 giờ):

| Chỉ số | LEVI baseline | BLADE Pool 2.0 | BLADE Pool 3.0 (kỳ vọng) |
| --- | --- | --- | --- |
| best_score | 2.6027 | 2.2745 (collapsed) | ≥ 2.6 |
| pool_size | 53 elites | 100 (full nhưng 1 family) | 100 với 4-6 families |
| paradigm_trials accepted | n/a | 9/21 (toàn delta âm) | similar count, delta dương khả thi |
| mean_recent_diversity | n/a | 0.81 (collapse) | < 0.7 |
| HoF entries | n/a | n/a | 5-15 paradigm seeds |

Các thay đổi đem lại:

1. **Bigram AST** — cross-paradigm cosine 0.95 → 0.69 (smoke test). Lớp AST
   bây giờ thực sự discriminate, niche dedup ngừng accept variants chỉ-vì-AST-trông-giống-nhau.
2. **Family threshold 0.72 → 0.85** — paraphrase cùng paradigm (cosine 0.75-0.85)
   không còn merge thành 1 family. Live runs show 4-6 family thay vì 1.
3. **Quota niching luôn enforce** — không còn cửa sổ "fill-first" để 1 paradigm
   thắng phase bootstrap rồi chiếm 100 slot.
4. **Paradigm cross-family anchors** — kể cả khi pool collapse, frontier vẫn
   thấy 3 paradigm-distinct anchors (HoF backfill). Trước đó 3 anchors là 3
   paraphrase của cùng paradigm → frontier không có thông tin để break out.
5. **Stuck → early** — stage routing fix bug "late-stage forbids rewrites we
   need". Khi stuck/collapse, prompt chuyển sang PARADIGM SHIFT mode.
6. **Paradigm grace 30 evals** — paradigm seed có 30 evals để fanout + xác minh
   trước khi quota cap có thể kill nó. Trước đó paradigm bị evict ngay khi
   admit (vì điểm thấp hơn best, family cap chọn nó làm weakest).
7. **Selector paradigm boost +0.6 decay-linear-25-evals** — workers thực sự
   chọn paradigm seed làm parent cho mutate/crossover trong 25 evals đầu. Trước
   đó paradigm có recency tốt nhưng score-normalized thấp → bị các elite
   cao điểm outrank → workers không bao giờ mutate trên paradigm.
8. **Hall-of-Fame** — paradigm seeds tốt nhất qua đời các epoch vẫn còn sống
   để backfill cho paradigm shift sau và frontier có chu kỳ memory dài hạn.

## 10. Ablation framework

Mọi component đều có toggle, chạy ablation bằng cờ CLI hoặc JSON advanced_options.

### Bảng ablation đề xuất

| Cấu hình | Ý nghĩa | CLI |
| --- | --- | --- |
| **Full** | Tất cả ON (default) | (no flag) |
| `-A` | Ablate quota niching | `--disable-quota-niching` |
| `-B` | Ablate bigram AST | `--ast-mode count14` |
| `-C` | Ablate cross-family + stuck-early | `--disable-cross-family-anchors --disable-force-early-on-collapse` |
| `-D` | Ablate HoF + grace + boost | `--disable-hall-of-fame --disable-paradigm-grace --disable-paradigm-boost` |
| **Legacy** | Tất cả OFF (≈ behavior Pool 2.0) | Tất cả `--disable-*` + `--ast-mode count14` + `--family-threshold 0.72` + `--structural-threshold 0.97` |

### Snapshot phản ánh cấu hình

`snapshot.json["ablation_flags"]` ghi rõ:

```json
{
  "ast_mode": "bigram",
  "enable_quota_niching": true,
  "enable_paradigm_grace": true,
  "enable_hall_of_fame": true,
  "enable_paradigm_boost": true,
  "paradigm_cross_family_anchors": true,
  "paradigm_force_early_on_collapse": true,
  "family_cosine_threshold": 0.85,
  "structural_cosine_threshold": 0.85,
  "target_n_families": 5,
  "max_per_family": 10,
  "paradigm_grace_evals": 30,
  "hof_size": 30,
  "paradigm_boost": 0.6,
  "paradigm_exploit_window": 25
}
```

Post-hoc attribution: chạy mỗi cấu hình 2-3 seeds, so `best_score` + `pool_size`
+ `num_families_final` + `paradigm_trials_with_positive_delta`.

## 11. Output artifacts

- **`snapshot.json`** — dump đầy đủ:
  - `monitor` stats (eval_count, best_score, plateau_steps, stagnation_level,
    accept_rate, mean_recent_diversity, is_stuck, is_collapsing).
  - `meta_advice` cuối + trigger_count.
  - `paradigm_trials` (idx, stage, accepted, score, delta, description).
  - `elites` (kèm `family_id`, `protected_until_eval`, `created_at_eval`,
    `uses_count`, `description`, `content`).
  - **`hall_of_fame`** (mới) — danh sách HoF entries sorted by score desc.
  - **`ablation_flags`** (mới) — phản ánh BladeConfig cho component A/B/C/D.
- **`best.py` / `best_program.py`** — chương trình điểm cao nhất.
- **`summary.json`** — `run_blade.py` thêm metadata model/budget.

## 12. Logging

Không đổi nhiều. Bổ sung:

- **`[BLADE PE] forcing stage=early (was X) — monitor flags stuck=… collapsing=…`**
  khi component C trigger.
- **`[BLADE PE] trigger #N at eval=… | stage=… | best=… | pool=… | families=…`**
  vẫn như cũ, nhưng `families` count bây giờ phản ánh family threshold mới
  (0.85) — kỳ vọng 2-6 thay vì 1-2 ở run cũ.
- Pool internal: `_log` paradigm grace stamps, HoF admits/evictions không
  emit log (giữ runtime quiet); kiểm tra qua snapshot cuối run.

## 13. Knobs CLI / workflow advanced_options

```text
# scripts/run_blade.py flags

# === Knobs cũ ===
--pool-k                       # PoolConfig.K (100)
--niche-threshold              # lớp 1: description cosine (0.92)
--structural-threshold         # lớp 2: AST cosine (0.85, đã hạ từ 0.97)
--family-threshold             # family clustering (0.85, đã tăng từ 0.72)
--max-per-family               # cap (10)
--paradigm-n-anchors           # anchors có code (3)
--paradigm-n-inspirations      # inspirations description-only (5)
--n-paradigm-variants          # fanout sau paradigm shift (4)
--n-diverse-seeds              # phase-1 sequential seeds (4-5)
--n-variants-per-seed          # phase-2 parallel variants/seed (20)
--pe-interval                  # cron N evals (50)
--meta-advice-interval         # cron N evals (50)
--no-repair / --no-meta-advice # tắt phụ trợ

# === Ablation toggles (mới) ===

# Component B — AST signature
--ast-mode {bigram, count14}   # bigram = production, count14 = legacy

# Component A — Quota niching
--disable-quota-niching        # ablate
--target-n-families            # quota = ceil(K/N) (5)

# Component D — Paradigm grace
--disable-paradigm-grace
--paradigm-grace-evals         # grace window (30)

# Component D — Hall of Fame
--disable-hall-of-fame
--hof-size                     # capacity (30)

# Component D — Selector boost
--disable-paradigm-boost
--paradigm-boost               # additive boost magnitude (0.6)
--paradigm-exploit-window      # window in evals (25)

# Component C — Paradigm shift
--disable-cross-family-anchors      # fall back to representatives(stage)
--disable-force-early-on-collapse   # let budget_progress alone route stage
--paradigm-temperature              # healthy regime (0.7)
--paradigm-temperature-stuck        # stuck/collapsing (1.0)
```

Trong `.github/workflows/blade.yml`, các knob nâng cao đi qua JSON
`advanced_options`. Key tương ứng theo pattern `snake_case`:

```json
{
  "ast_mode": "count14",
  "quota_niching_disabled": true,
  "hall_of_fame_disabled": true,
  "paradigm_grace_disabled": true,
  "paradigm_boost_disabled": true,
  "cross_family_anchors_disabled": true,
  "force_early_on_collapse_disabled": true
}
```

(ví dụ trên = ablate hết, run BLADE ở chế độ "legacy" gần Pool 2.0).

Đầy đủ keys được handle bởi step "Parse advanced options" trong workflow:
`problem_module, target_score, embedding_model, eval_processes, n_paradigm_variants,
paradigm_n_anchors, paradigm_n_inspirations, pool_k, niche_threshold,
structural_threshold, family_threshold, max_per_family, repair_disabled,
meta_advice_disabled, meta_advice_interval, ast_mode, quota_niching_disabled,
target_n_families, paradigm_grace_disabled, paradigm_grace_evals,
hall_of_fame_disabled, hof_size, paradigm_boost_disabled, paradigm_boost,
paradigm_exploit_window, cross_family_anchors_disabled,
force_early_on_collapse_disabled, paradigm_temperature, paradigm_temperature_stuck`.

## 14. Gợi ý A/B tiếp theo

Để khẳng định mỗi component thực sự đóng góp:

1. **A/B Pool architecture** (component A):
   - Full vs `--disable-quota-niching` → đo `num_families_final` và
     `mean_recent_diversity`. Kỳ vọng quota niching giữ 4-6 family, ablate
     thường về 1-2.

2. **A/B AST signature** (component B):
   - Full vs `--ast-mode count14 --structural-threshold 0.97` → đếm số
     "lỡ admit duplicate" qua bigram cosine của các cặp elite sau run.

3. **A/B paradigm shift** (component C):
   - Full vs `--disable-cross-family-anchors --disable-force-early-on-collapse`
     → đếm % paradigm trials có delta > 0.

4. **A/B paradigm protection** (component D):
   - Full vs `--disable-paradigm-grace --disable-paradigm-boost --disable-hall-of-fame`
     → tỷ lệ `paradigm_variant` sống trong pool cuối, average lifetime
     (created_at_eval → evicted_at) của paradigm seeds.

5. **Sweep family threshold**: `--family-threshold {0.72, 0.80, 0.85, 0.90}`
   ở chế độ full → đo `num_families_final`. Kỳ vọng 0.85 sweet spot.

Mỗi A/B nên dùng cùng `seed=0` (đã có trong `BladeConfig`) + 2-3 repeat để
loại bỏ noise khởi tạo. Output dir phân biệt qua `--output-dir` để snapshot
không bị overwrite.
