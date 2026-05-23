# WORKFLOW_BLADE — Luồng hoạt động của BLADE

> BLADE = **B**ehavior-**L**atent **A**daptive **D**iscovery **E**ngine.
> Mục tiêu: cùng một bài toán tối ưu mã nguồn (LEVI examples) nhưng thay 3 hệ con
> "nặng đô" của LEVI bằng các phiên bản nhẹ-và-mạnh hơn:
>
> | Hệ con | LEVI | BLADE |
> | --- | --- | --- |
> | Archive | CVT-MAP-Elites (centroid behavioural grid) | Top-K **Pool** với description-embedding niching ([levi/levi/simple/pool.py](levi/levi/simple/pool.py)) |
> | Stagnation signal | PPS formula (Punctuated-Equilibrium PPS) | 3 sliding-window stats: accept-rate / plateau / diversity ([levi/levi/simple/monitor.py](levi/levi/simple/monitor.py)) |
> | Sampler | 4-D Thompson bandit (SAL) | UCB-style **Selector** (novelty + recency − diversity penalty) ([levi/levi/simple/selector.py](levi/levi/simple/selector.py)) |
>
> Frontier prompts (3-phase paradigm shift), error-archive self-repair, async producer/consumer
> đều giữ nguyên từ LEVI để hai phương pháp so sánh fair-game ngân sách.

Entry point: [scripts/run_blade.py](scripts/run_blade.py) → `levi.evolve_code_blade()`
([levi/levi/methods/blade.py](levi/levi/methods/blade.py))
→ `BladeOrchestrator.run()` ([levi/levi/blade/orchestrator.py](levi/levi/blade/orchestrator.py)).

---

## 1. Tổng quan luồng

```
┌─────────────────────────────────────────────────────────────────────┐
│  run_blade.py (CLI)                                                  │
│      │   parse args, load problem.py, setup output dir              │
│      ▼                                                               │
│  evolve_code_blade()  ──►  _setup_logging() + BladeConfig            │
│      │                                                               │
│      ▼                                                               │
│  BladeOrchestrator.run()                                             │
│      │                                                               │
│      ├── [start] _status_monitor (heartbeat 30s)                    │
│      │                                                               │
│      ├── PHASE 1 — Diverse seeds (SEQUENTIAL, frontier model GPT-5) │
│      │     for i in 1..n_diverse_seeds:                              │
│      │       build_diverse_seed_prompt → paradigm_lm → eval → admit │
│      │                                                               │
│      ├── PHASE 2 — Init variants (PARALLEL, mutation model Qwen)    │
│      │     n_diverse_seeds × n_variants_per_seed prompts            │
│      │     asyncio.gather → build_init_variant_prompt → mutation_lm │
│      │                                                               │
│      ├── [start] _pe_monitor (mỗi 2s, kick paradigm-shift)          │
│      ├── [start] _meta_advice_monitor (mỗi 2s, refresh advice)      │
│      │                                                               │
│      └── MAIN LOOP — mutate / crossover / repair workers             │
│            ▲                                                         │
│            │  Mỗi N evals (pe_cron_interval) → paradigm shift:       │
│            │    1) frontier seed (gpt-5)                            │
│            │    2) n_paradigm_variants fanout (qwen, parallel)      │
│            │                                                         │
│            │  Mỗi N evals (meta_advice_interval) → advisor refresh   │
│            │  (mutation model viết "lessons learnt" cho prompts kế) │
│            ▼                                                         │
│        budget exhausted → snapshot.json + best.py                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Cấu hình mặc định (GitHub Actions run)

Workflow: [.github/workflows/blade.yml](.github/workflows/blade.yml)

| Tham số | Giá trị mặc định | Ý nghĩa |
| --- | --- | --- |
| `mutation_model` | `openrouter/qwen/qwen3-30b-a3b-instruct-2507` | "Worker bee" — mọi LLM call tần suất cao |
| `paradigm_model` | `openrouter/openai/gpt-5` | Frontier — chỉ dùng cho diverse seeds + paradigm shift |
| `embedding_model` | `openrouter/openai/text-embedding-3-small` | Description embedding cho Pool niching |
| `workers` | 4 | Số coroutine LLM song song trong main loop |
| `eval_processes` | 4 | Subprocess pool đánh giá code |
| `pe_interval` | **10** (workflow) / 50 (CLI default) | Mỗi N evals kích hoạt paradigm shift |
| `n_diverse_seeds` | 4 | Số seed đa dạng phase 1 |
| `n_variants_per_seed` | 20 | Variant/seed ở phase 2 (tổng init ≈ 4×20 = 80) |
| `n_paradigm_variants` | 4 | Fanout sau mỗi paradigm seed |
| `meta_advice_interval` | 50 | Cron refresh advice |
| `pool_k` (PoolConfig.K) | **100** | Số program tối đa trong pool |
| `niche_cosine_threshold` | 0.92 | Dedup ngữ nghĩa qua description embedding |
| `family_cosine_threshold` | 0.72 | Single-linkage gom cụm "họ" thuật toán |
| `max_per_family` | 8 | Một họ tối đa 8 chỗ trong pool |
| `budget_seconds` | 10800 (3 giờ) | Wall-clock cap |

---

## 3. Phase 1 — Diverse Seeds (sequential, frontier)

Code: `BladeOrchestrator._bootstrap_population()` phase 1, sử dụng prompt
`build_diverse_seed_prompt` ([levi/levi/blade/prompts.py:272](levi/levi/blade/prompts.py#L272)).

- **Model**: `paradigm_lm` (GPT-5).
- **Tuần tự** vì mỗi prompt sau phải nhìn thấy các seed đã chấp nhận trước đó để
  được "đẩy" theo hướng *paradigm khác hẳn*.
- **Retry**: tối đa 3 lần per seed (LLM lỗi / parse miss / eval fail đều retry).
- **Prompt**: wrap LEVI's `DIVERSITY_SEED_PROMPT`
  ([levi/levi/artifacts/code.py:23](levi/levi/artifacts/code.py#L23)) + dán
  `OUTPUT_FORMAT_INSTRUCTION` (yêu cầu trả về `## Description` và `## Code`).

Tóm tắt nội dung prompt diverse seed:

```
# {problem_title}
## Problem: {problem_description}
## Function signature: ```python {function_signature}```

## Existing diverse seeds (so far)
{existing_seeds_text}    # code + score của các seed trước đó

## Task
Design a SOLUTION that uses a fundamentally DIFFERENT algorithmic paradigm
than ANY of the seeds above. Identify which paradigm families are missing
(greedy / DP / graph / SA / gradient / brute-prune / …) and pick one of
the missing classes.

{OUTPUT_FORMAT_INSTRUCTION}
```

Sau phase 1: log dòng `[BLADE init] phase 1 done: K seeds admitted`.

## 4. Phase 2 — Init Variants (parallel, mutation)

Code: cùng method, phase 2. Prompt builder: `build_init_variant_prompt`.

- **Model**: `mutation_lm` (Qwen 30B).
- **Parallel**: `asyncio.gather` chạy `n_diverse_seeds × n_variants_per_seed` prompt
  đồng thời (mỗi prompt vẫn đi qua `_semaphore` của worker pool).
- Mỗi variant prompt lấy **2 seed ngẫu nhiên** làm inspiration (code + score).
- Mục tiêu: khai thác chiều sâu xung quanh từng paradigm — **giữ paradigm gốc**,
  chỉ tinh chỉnh hằng số / heuristic phụ / xử lý edge-case.

Prompt template ([levi/levi/blade/prompts.py:228](levi/levi/blade/prompts.py#L228)):

```
# Init Variant
## Problem … / Function signature …
## Inspirations (existing diverse seeds — code + score)
{inspirations_block}

## Your task
Produce a variant of one of the inspirations. Borrow mechanisms across them
but KEEP the core algorithmic paradigm intact — variant exploration only.

### Critical requirements
- Function signature MUST match exactly
- Standard libraries only (numpy, …)
- Include all imports; no placeholders

{OUTPUT_FORMAT_INSTRUCTION}
```

## 5. Main Loop — mutate / crossover / repair

Code: `_main_loop()` + `_generate_one()` + `_repair_one()`.

- **Workers**: tối đa `n_workers` coroutine `_generate_one` chạy song song
  (cap bằng `asyncio.Semaphore`).
- **Operator chọn ngẫu nhiên** theo trạng thái stagnation (`monitor.is_stuck()`):
  - Healthy: p_crossover = 0.30 → 70% mutate, 30% crossover.
  - Stuck: p_crossover = 0.70 → ngược lại.
- **Selector** (`levi.simple.Selector`) chọn parent (1 cho mutate, 2 cho crossover)
  bằng UCB-style: ưu tiên program ít được dùng (novelty bonus), điểm cao,
  có embedding xa các program hay được dùng.
- **Inspirations**: thêm 2-3 program nữa từ pool, chỉ truyền `description + score`
  vào prompt (không truyền code — tiết kiệm token, tăng diversity).
- **Meta-advice**: ở 80% xác suất, chèn block "Lessons learnt so far" do advisor
  sinh ra ngay trước phần `## Your task`.

### Mutate prompt ([levi/levi/blade/prompts.py:38](levi/levi/blade/prompts.py#L38))

```
# Mutate
## Problem … / Function signature …

## Parent solution
Score: {parent_score:.4f}
```python {parent_code} ```

## Inspirations (paradigm sketches from the archive — descriptions only)
{inspirations_block}

{meta_advice_block}
## Your task
Produce a mutated variant of the parent that is meaningfully different.
Treat the inspirations as ideas to draw from — do NOT copy their code.
Keep what works in the parent, change what doesn't.

{OUTPUT_FORMAT_INSTRUCTION}
```

### Crossover prompt ([levi/levi/blade/prompts.py:68](levi/levi/blade/prompts.py#L68))

Cùng template + 2 parent (A, B) đầy đủ code. Task:

> Produce a hybrid solution that combines the strongest mechanisms of both
> parents while fixing at least one weakness. Be structural, not stitched:
> do not paste A's branch into B's branch.

### Repair (one-shot)

Khi một candidate raise exception ở `_evaluate_code`, BLADE đẩy
`(broken_code, parent_score, error_msg)` vào `error_buffer` (deque maxlen 64).
Main loop opportunistically `_repair_one()`: mutation model nhận `build_repair_prompt`:

```
# Repair
## Problem … / Function signature …

## Broken candidate (parent score was {parent_score})
```python {broken_code} ```

## Error
``` {error_msg_last_1500_chars} ```

## Your task
Produce a corrected version of the candidate that addresses the error above.
Keep the algorithmic intent intact — only patch what's broken.
```

**One-shot**: nếu repair lại lỗi thì drop, không loop.

## 6. Punctuated Equilibrium (paradigm shift)

Code: `_pe_monitor()` + `_paradigm_shift()`.

- **Cron**: task chạy nền `await asyncio.sleep(2.0)`, mỗi lần wake kiểm tra
  `eval_count >= last_pe_eval_count + pe_cron_interval`. Vì BLADE bootstrap
  phase 2 và variant fanout submit nhiều eval qua `asyncio.gather`, dùng
  **boundary-crossing** (không phải modulo) để không bị skip.
- **Lock**: `_pe_lock` đảm bảo tối đa **một** paradigm shift in-flight.
- **Stage routing** (`get_budget_stage`): tuỳ `budget_progress` + `stagnation`,
  chọn 1 trong 3 prompt: `early` / `mid` / `late`
  ([levi/levi/equilibrium/prompts.py:22](levi/levi/equilibrium/prompts.py#L22)).
- **Bước 1** (frontier, GPT-5): lấy 3 đại diện từ pool (chỉ `description + score`,
  không truyền code → BLADE deviation so với LEVI), kèm "Strategy Log" — list
  các paradigm trial gần đây (đã accept hay từ chối, delta). Mục đích: ép GPT-5
  *thử paradigm mới* thay vì lặp lại.
- **Bước 2** (mutation, Qwen, song song): nếu seed paradigm chấp nhận được, fan
  ra `n_paradigm_variants` variant bằng `build_paradigm_variant_prompt`
  (wrap `VARIANT_GENERATION_PROMPT` của LEVI).

Tóm tắt prompt 3-stage:

| Stage | Khi nào | Yêu cầu chính |
| --- | --- | --- |
| **early** | budget_progress < ~0.4 hoặc stagnation cao sớm | "PARADIGM SHIFT" — chọn lớp paradigm CHƯA xuất hiện trong archive (DP/graph/SA/…), không retune |
| **mid** | trung gian | "SYNTHESIS" — kết hợp 2-3 cơ chế từ các region, fix 1-2 điểm yếu cụ thể |
| **late** | budget_progress cao + plateau dài | "REFINE" — siết hằng số, tối ưu hyperparam, sửa edge-case |

## 7. Meta-Advisor (lessons learnt)

Code: `_meta_advice_monitor()` + `_generate_meta_advice()`.

- **Cron**: cùng pattern boundary-crossing, mỗi `meta_advice_interval` eval.
- **Model**: mutation model (Qwen) — bài toán summarisation rẻ, low-temp 0.4,
  cap 400 tokens.
- **Input**: best_score, accept_rate, stagnation_level, 5 lỗi gần nhất,
  advice trước (để refine, không restart).
- **Output**: 3-5 câu prescriptive, chèn vào 80% mutate/crossover prompt kế.

Prompt ([levi/levi/blade/prompts.py:325](levi/levi/blade/prompts.py#L325)):

```
# Lessons-Learned Advisor
You are reviewing the last batch of attempts.
Output a SHORT (3-5 sentences) note that future mutation prompts will include verbatim.
Focus on what to AVOID and what to TRY next — concrete, prescriptive, code-shaped.

## Current state
- Best score so far: {best_score}
- Evaluations completed: {n_evaluations}
- Accept rate (last window): {accept_rate}
- Stagnation level: {stagnation_level} (0=fresh, 1=plateaued)

## Recent failure modes
{error_block}

## Previous advice (carried over so you can refine, not repeat)
{previous_advice_block}

## Your task
Write the new advice block. No preamble, no markdown headers.
Keep under 100 words. Do NOT restate the problem.
```

## 8. Pool — vì sao pool_size cuối cùng chỉ = 19 ?

> Cấu hình: `K=100`, `niche_cosine_threshold=0.92`, `family_cosine_threshold=0.72`,
> `max_per_family=8`.

Trong run đang xét (`log_blade/summary.json`):

```
best_score        = 2.5206
total_evaluations = 1032
pool_size         = 19          ← chỉ 19 / K=100
n_paradigm_trials = 15  (11 accepted)
elite source mix  = paradigm:8, init:5, mutate:4, crossover:1, repair:1
monitor: plateau_steps=490, stagnation_level=1.0, accept_rate=0.0
```

`K=100` là cận **trên**, KHÔNG phải mục tiêu phải đạt. Pool tự loại bỏ chương
trình ở **bốn cửa**, code: [levi/levi/simple/pool.py:97](levi/levi/simple/pool.py#L97):

1. **`no_embedding`** — description rỗng / embedder lỗi → reject im lặng.
2. **`dropped_duplicate` / `replaced_duplicate`** — semantic dedup. Nếu cosine
   description embedding ≥ 0.92 với hàng xóm gần nhất:
   - score cao hơn ⇒ thay thế (`replaced_duplicate`, **không** tăng size).
   - không cao hơn ⇒ drop.
3. **`replaced_family_weak`** — sau khi append, nếu family này đã chạm
   `max_per_family=8`, evict program **yếu nhất trong family** (lại **không**
   tăng size).
4. **`dropped_full`** — sau khi pass dedup + family-cap, nếu `len > K=100`, evict
   chương trình **toàn cục** có score thấp nhất (có thể chính là chương trình
   vừa add).

Với run circle_packing:

- 1032 evals nhưng `accept_rate=0.0` (last window) và `plateau_steps=490` cho
  thấy phần lớn candidate **trượt** ở cửa (2) hoặc (3): mutation model liên tục
  sinh ra các biến thể *giống nhau về mặt mô tả* của vài paradigm có sẵn.
- Embedder `text-embedding-3-small` chấm các variant cùng paradigm rơi vào dải
  cosine ~0.75-0.92, sát ngưỡng 0.92 → dễ trigger near-duplicate.
- Source distribution của 19 elite (paradigm 8, init 5, mutate 4, crossover 1,
  repair 1) cho thấy **paradigm shifts** đóng góp nhiều nhất; mutate/crossover
  của Qwen hầu như không "qua được lưới".

→ "Pool 19" là dấu hiệu Pool **đang làm đúng chức năng dedup** chứ không phải
bug. Nhưng nó cũng cho thấy mutation model không tạo ra description đủ khác.

## 9. Vì sao điểm BLADE (2.5206) thấp hơn LEVI (2.6027) ?

Đối chiếu hai summary (cùng problem `circle_packing`, cùng ngân sách 3 giờ):

| Chỉ số | LEVI | BLADE |
| --- | --- | --- |
| best_score | 2.6027 | 2.5206 (-0.082) |
| total_evaluations | **205** | **1032** (5×) |
| total_cost | $1.35 | $2.81 (2.1×) |
| archive / pool | 53 elites | 19 |
| runtime | 10800 s | 10803 s |

Một số nguyên nhân hợp lý (cần kiểm chứng thêm bằng run thử nghiệm):

1. **LEVI tiêu tốn nhiều ngân sách hơn cho mỗi eval** ($1.35 / 205 = $0.0066/eval)
   vs BLADE ($2.81 / 1032 = $0.0027/eval). LEVI có thể đã cho mỗi candidate dùng
   nhiều token suy nghĩ hơn, hoặc dùng frontier nhiều hơn — chất lượng/eval cao.
2. **LEVI đã tìm được best 2.6008 ngay từ Seed 3 trong phase 1**
   ([log_levi/run.txt:25](log_levi/run.txt#L25)). Toàn bộ pipeline sau đó về cơ
   bản chỉ tìm được paradigm shift ngang ngửa (2.6008 → 2.6027). BLADE cũng chạm
   ~2.52 sớm rồi plateau (`plateau_steps=490`, `stagnation_level=1.0`).
   → Cả hai stuck ở local optimum; LEVI may mắn hơn ở seed.
3. **BLADE dedup chặt hơn** (Pool cosine 0.92 + family 0.72 + max_per_family 8).
   Khi paradigm space đã bị "phủ" sớm, mutation calls sau đó hầu như đều bị
   drop làm `accept_rate → 0`, model không có tín hiệu phản hồi rõ ràng để
   chuyển hướng (advisor chỉ refresh mỗi 50 evals, không tác động lên Selector).
4. **PE cadence quá dày** so với chất lượng paradigm output. Workflow đặt
   `pe_interval=10` (gọi GPT-5 sau mỗi 10 evals) → 11/15 trial được accept,
   chiếm ưu thế trong pool (8/19 = 42% elites) nhưng nhiều shift "ngang
   side-grade" thay vì breakout. LEVI default PE interval lớn hơn.
5. **`n_diverse_seeds=4`** (BLADE workflow) vs `n_diverse_seeds=5` (LEVI workflow
   default). 1 seed ít hơn nghĩa là phase 1 mất 1 cơ hội tìm paradigm mạnh —
   đáng kể trên bài toán mà điểm tốt nhất đến từ một seed cụ thể.
6. **Repair có thể chưa kích hoạt nhiều**: trong elite cuối cùng chỉ có 1 program
   từ `source=repair`. Nếu mutation model sinh ra nhiều lỗi mà repair bị
   one-shot rồi drop, ta mất bài học đó.

(Tóm tắt: BLADE có nhiều eval hơn, nhưng quality-per-eval thấp hơn và pool dedup
chặt làm exploration hiệu quả thấp ở giai đoạn cuối. Không phải bug, là
trade-off cần tune.)

## 10. Output artifacts

- **`snapshot.json`** — dump đầy đủ: monitor stats, meta-advice cuối, danh sách
  `paradigm_trials` và tất cả `elites` (kèm code).
- **`best.py` / `best_program.py`** — chương trình điểm cao nhất.
- **`summary.json`** — `run_blade.py` thêm metadata model/budget.

## 11. Logging (sau bản cập nhật này)

Trước đây `run.txt` của BLADE gần như rỗng (chỉ có dòng cấu hình + kết quả cuối)
vì `evolve_code_blade` không gọi `_setup_logging()` như Levi.

Đã thêm vào [levi/levi/methods/blade.py](levi/levi/methods/blade.py) và
[levi/levi/blade/orchestrator.py](levi/levi/blade/orchestrator.py):

1. **`_setup_logging()`** — basicConfig giống Levi: `%H:%M:%S [INFO] message`.
2. **`_status_monitor`** — heartbeat mỗi 30 s:
   `[Status] Cost: $… | Evals: … | Clients in-flight: … | Eval in-flight: … | Pool: … | Best: … | Elapsed: …s`
3. **`_record_reject(source, score, error_msg)`** — helper duy nhất bọc
   `monitor.record_eval` cho mọi spot reject (parse_miss, LLM err, eval err,
   worker exception). Sinh dòng:
   `[Eval #N] {model:27s} ERROR (source): {msg[:80]}`
4. **`_admit`** — log dòng:
   `[Eval #N] {model:27s} {status:12s} | source: … | score: … | best: … | $cost` —
   `status` ∈ `{NEW BEST ★, accepted, rejected}`.
5. **In-flight counters** — wrap `_call` và `_evaluate_code` bằng try/finally
   tăng/giảm `_client_in_flight` và `_eval_in_flight`.
6. **PE trigger log** giàu hơn:
   `[BLADE PE] trigger #N at eval=… | stage=… | best=… | pool=… | families=…`
7. Thông báo phase đầu cuối + summary sau bootstrap:
   `[BLADE] bootstrap complete — pool=… best=… cost=$… evals=…`.

Sau khi merge, `run.txt` của BLADE sẽ có cùng cấu trúc với `log_levi/run.txt`.

---

## 12. Gợi ý thử nghiệm (không sửa code)

Nếu muốn cải thiện điểm BLADE, các tham số có khả năng tác động nhiều nhất:

- Tăng `n_diverse_seeds` (4 → 6-8): mua thêm cơ hội tìm paradigm mạnh ở phase 1.
- Tăng `pe_interval` (10 → 30-50): mỗi paradigm shift đắt (gọi GPT-5) và hiện
  side-grade nhiều — chạy thưa hơn để main loop có thời gian khai thác.
- Nới `niche_cosine_threshold` (0.92 → 0.95) hoặc `max_per_family` (8 → 16): cho
  pool giữ thêm biến thể, tăng dữ liệu cho Selector.
- Bật `target_score` lớn hơn 2.6 để run dừng ngay khi vượt LEVI thay vì plateau.
- Đổi `mutation_model` sang một model mạnh hơn nếu ngân sách cho phép — Qwen
  hiện đang là điểm nghẽn của accept_rate.
