# WORKFLOW_BLADE — Luồng hoạt động của BLADE

> BLADE = **B**ehavior-**L**atent **A**daptive **D**iscovery **E**ngine.
> Mục tiêu: cùng một bài toán tối ưu mã nguồn (LEVI examples) nhưng thay 3 hệ
> con "nặng đô" của LEVI bằng các phiên bản nhẹ-và-mạnh hơn:
>
> | Hệ con | LEVI | BLADE |
> | --- | --- | --- |
> | Archive | CVT-MAP-Elites (centroid behavioural grid) | Top-K **Pool** với 2 lớp dedup (description embedding + AST signature) ([levi/levi/simple/pool.py](levi/levi/simple/pool.py)) |
> | Stagnation signal | PPS formula (Punctuated-Equilibrium PPS) | 3 sliding-window stats: accept-rate / plateau / diversity ([levi/levi/simple/monitor.py](levi/levi/simple/monitor.py)) |
> | Sampler | 4-D Thompson bandit (SAL) | UCB-style **Selector** (novelty + recency − diversity penalty) ([levi/levi/simple/selector.py](levi/levi/simple/selector.py)) |
>
> Frontier prompts (3-phase paradigm shift), error-archive self-repair, async
> producer/consumer giữ nguyên ý tưởng từ LEVI; nhưng paradigm prompt đã được
> viết lại BLADE-native (anchors có CODE + inspirations description-only).

Entry point: [scripts/run_blade.py](scripts/run_blade.py) → `levi.evolve_code_blade()`
([levi/levi/methods/blade.py](levi/levi/methods/blade.py))
→ `BladeOrchestrator.run()` ([levi/levi/blade/orchestrator.py](levi/levi/blade/orchestrator.py)).

---

## 1. Tổng quan luồng

```text
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
│            │    1) anchors (3 code) + inspirations (5 desc) → gpt-5  │
│            │    2) n_paradigm_variants fanout (qwen, parallel)      │
│            │                                                         │
│            │  Mỗi N evals (meta_advice_interval) → advisor refresh   │
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
| `paradigm_model` | `openrouter/openai/gpt-5` | Frontier — diverse seeds + paradigm shift |
| `embedding_model` | `openrouter/openai/text-embedding-3-small` | Description embedding cho Pool niching |
| `workers` | 4 | Số coroutine LLM song song trong main loop |
| `eval_processes` | 4 | Subprocess pool đánh giá code |
| `pe_interval` | **10** (workflow) / 50 (CLI default) | Mỗi N evals kích hoạt paradigm shift |
| `n_diverse_seeds` | 4 | Số seed đa dạng phase 1 |
| `n_variants_per_seed` | 20 | Variant/seed ở phase 2 (tổng init ≈ 4×20 = 80) |
| `n_paradigm_variants` | 4 | Fanout sau mỗi paradigm seed |
| `meta_advice_interval` | 50 | Cron refresh advice |
| `pool_k` (PoolConfig.K) | **100** | Target population size — trong khi chưa đầy Pool nới các ràng buộc để fill đủ K |
| `niche_cosine_threshold` | 0.92 | Description-embedding cosine **gắn cờ** trùng nghĩa (lớp 1) |
| `structural_cosine_threshold` | **0.97** | AST-signature cosine **lớp 2** — phải đồng tình lớp 1 mới drop. *Mới thêm* |
| `family_cosine_threshold` | 0.72 | Single-linkage gom cụm "họ" thuật toán |
| `max_per_family` | **10** | Một họ tối đa 10 chỗ — và *chỉ enforce khi pool ≥ K* |
| `paradigm_n_anchors` | 3 | Số anchor representative (kèm code) cho frontier mỗi paradigm shift |
| `paradigm_n_inspirations` | 5 | Số inspiration description-only gửi kèm |
| `budget_seconds` | 10800 (3 giờ) | Wall-clock cap |

---

## 3. Phase 1 — Diverse Seeds (sequential, frontier)

Code: `BladeOrchestrator._bootstrap_population()` phase 1, prompt builder
`build_diverse_seed_prompt` ([levi/levi/blade/prompts.py:272](levi/levi/blade/prompts.py#L272)).

- **Model**: `paradigm_lm` (GPT-5).
- **Tuần tự**: mỗi prompt sau phải nhìn các seed đã chấp nhận trước đó để được
  "đẩy" theo hướng paradigm khác hẳn.
- **Retry**: tối đa 3 lần per seed (LLM lỗi / parse miss / eval fail đều retry).
- **Prompt**: wrap LEVI's `DIVERSITY_SEED_PROMPT`
  ([levi/levi/artifacts/code.py:23](levi/levi/artifacts/code.py#L23)) + dán
  `OUTPUT_FORMAT_INSTRUCTION` (yêu cầu trả về `## Description` và `## Code`).

Tóm tắt nội dung prompt diverse seed:

```text
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
- **Parallel**: `asyncio.gather` chạy `n_diverse_seeds × n_variants_per_seed`
  prompt đồng thời (mỗi prompt vẫn đi qua `_semaphore` của worker pool).
- Mỗi variant prompt lấy **2 seed ngẫu nhiên** làm inspiration (code + score).
- Mục tiêu: khai thác chiều sâu xung quanh từng paradigm — *giữ paradigm gốc*,
  chỉ tinh chỉnh hằng số / heuristic phụ / xử lý edge-case.

Prompt template ([levi/levi/blade/prompts.py:228](levi/levi/blade/prompts.py#L228)):

```text
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
- **Selector** (`levi.simple.Selector`) chọn parent (1 cho mutate, 2 cho
  crossover) bằng UCB-style.
- **Inspirations**: thêm 2-3 program nữa từ pool, chỉ truyền `description + score`
  vào prompt (không truyền code — tiết kiệm token, tăng diversity).
- **Meta-advice**: ở 80% xác suất, chèn block "Lessons learnt so far" do
  advisor sinh ra ngay trước phần `## Your task`.

### Mutate prompt ([levi/levi/blade/prompts.py:38](levi/levi/blade/prompts.py#L38))

```text
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
Main loop opportunistically `_repair_one()`. Mutation model nhận
`build_repair_prompt`:

```text
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

Code: `_pe_monitor()` + `_paradigm_shift()` + `build_paradigm_prompt`
([levi/levi/blade/prompts.py](levi/levi/blade/prompts.py)).

- **Cron**: task chạy nền `await asyncio.sleep(2.0)`, mỗi lần wake kiểm tra
  `eval_count >= last_pe_eval_count + pe_cron_interval`. Vì BLADE bootstrap
  phase 2 và variant fanout submit nhiều eval qua `asyncio.gather`, dùng
  **boundary-crossing** (không phải modulo) để không bị skip.
- **Lock**: `_pe_lock` đảm bảo tối đa **một** paradigm shift in-flight.
- **Stage routing** (`get_budget_stage`): tuỳ `budget_progress` + `stagnation`,
  chọn 1 trong 3 stage: `early` / `mid` / `late`.

### Prompt mới (BLADE-native, code-aware)

Khác bản trước (wrap LEVI template, chỉ truyền description):

1. **3 anchor representatives** kèm **toàn bộ code + description + score** —
   frontier có thể đọc *cơ chế thật*, không chỉ paraphrase.
2. **5 inspirations** chỉ description-only — mở rộng góc nhìn về archive mà
   không phình token.
3. **`n_families`** thay cho placeholder mơ hồ `n_regions` — đúng nghĩa
   `Pool.num_families()`.

Khung prompt:

```text
# Algorithmic Paradigm Shift Challenge ({early|mid|late})

## Problem … / Function Signature …

## Archive Snapshot
The archive has evolved through {n_evaluations} evaluations and currently
contains {n_families} distinct behavioural families.

### Anchor representatives (code + description + score)
#### Anchor 1 (score=…)
_Description_: …
```python … ```

#### Anchor 2 (score=…)
…

#### Anchor 3 (score=…)
…

### Additional inspirations (description + score only)
1. (score=…) …
2. (score=…) …
…

## Strategy Log (recent paradigm attempts)
- [#1 early] ✓ score=… Δ=+… :: …
- [#2 mid]  ✗ score=… Δ=…  :: …
…

## Your Challenge: {early=PARADIGM SHIFT | mid=SYNTHESIS | late=TARGETED REFINEMENT}
{stage-specific instructions}
{OUTPUT_FORMAT_INSTRUCTION}
```

| Stage | Trigger | Yêu cầu chính |
| --- | --- | --- |
| **early** | stagnation < 0.3 | PARADIGM SHIFT — chọn lớp paradigm CHƯA xuất hiện ở anchor/inspiration, structurally different |
| **mid** | 0.3 ≤ stagnation < 0.7 | SYNTHESIS — kết hợp 2-3 cơ chế từ các anchor, fix 1-2 điểm yếu |
| **late** | stagnation ≥ 0.7 | TARGETED REFINEMENT — siết hằng số / patch surgical trên anchor mạnh nhất |

Sau khi frontier trả về paradigm seed → mutation model fanout
`n_paradigm_variants` variants song song với `build_paradigm_variant_prompt`
(wrap `VARIANT_GENERATION_PROMPT` của LEVI).

## 7. Meta-Advisor (lessons learnt)

Code: `_meta_advice_monitor()` + `_generate_meta_advice()`.

- **Cron**: cùng pattern boundary-crossing, mỗi `meta_advice_interval` eval.
- **Model**: mutation model (Qwen) — bài toán summarisation rẻ, low-temp 0.4,
  cap 400 tokens.
- **Input**: best_score, accept_rate, stagnation_level, 5 lỗi gần nhất, advice
  trước (để refine, không restart).
- **Output**: 3-5 câu prescriptive, chèn vào 80% mutate/crossover prompt kế.

Prompt ([levi/levi/blade/prompts.py:325](levi/levi/blade/prompts.py#L325)):

```text
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

## 8. Pool 2.0 — fill-first, then niche+family

Đây là thay đổi quan trọng nhất so với bản trước. Pool hoạt động theo **hai
chế độ**:

### Chế độ "filling" (len(pool) < K)

Mục tiêu: lấp đầy pool tới K càng nhanh càng tốt, ưu tiên đa dạng.

1. Tính embedding (đã có) + AST signature (`compute_ast_signature(code)` —
   length-14 log-normalised vector: depth, cyclomatic, loop/branch/call
   counts, comprehension/comparison/subscript counts, …).
2. **Niche check 2 lớp**: nếu description cosine ≥ 0.92 *VÀ* AST cosine
   ≥ 0.97 với hàng xóm gần nhất → coi là duplicate thật:
   - Điểm cao hơn ⇒ `replaced_duplicate`.
   - Không cao hơn ⇒ `dropped_duplicate`.
3. Nếu **chỉ** description ≥ 0.92 nhưng AST < 0.97 → vẫn admit dưới dạng
   biến thể có cấu trúc khác. *Đây là điểm chốt: paraphrase nhưng code khác
   sẽ không bị giết oan.*
4. **Bỏ qua family cap**. Bỏ qua global K cap (pool chưa đầy mà).
5. Return `"added"`.

### Chế độ "at capacity" (len(pool) ≥ K)

Mục tiêu: giữ chất lượng, tránh một paradigm độc tài.

1. Niche check 2 lớp như trên.
2. Append, sau đó:
3. **Family cap**: `_enforce_family_cap()` — nếu family của newcomer vượt
   `max_per_family=10`, evict member yếu nhất của family đó. Nếu newcomer
   tự là weakest → `dropped_family_full`.
4. **Global K cap**: nếu sau bước 3 vẫn `len > K`, evict program điểm thấp
   nhất toàn cục. Nếu chính newcomer là worst → `dropped_full`.

Code: [levi/levi/simple/pool.py](levi/levi/simple/pool.py).

### Vì sao thiết kế này khắc phục được "pool_size=19"?

Trong run cũ, 1032 evals mà pool chỉ 19. Nguyên nhân:

- Niche threshold 0.92 trên text-embedding-3-small chấm các variant cùng
  paradigm rơi dải 0.75–0.92 → quá nhiều candidate bị giết oan ở cửa dedup.
- Family cap 8 enforce ngay từ đầu → khi 1 paradigm thắng được phase 1, mọi
  variant của nó chỉ giữ tối đa 8 chỗ → pool nghẹt sớm.

Với Pool 2.0:

- Lớp AST cho phép "cùng mô tả nhưng code khác cấu trúc" qua được — Qwen
  thường paraphrase y hệt nhau nhưng code khác hẳn ⇒ sống.
- Family cap bị defer tới khi pool đầy → phase bootstrap không bị siết.
- Cấu trúc reason `dropped_family_full` mới giúp log rõ ràng vì sao đào
  thải.

## 9. Vì sao điểm BLADE (2.5206) thấp hơn LEVI (2.6027) — và đã làm gì để gỡ

Đối chiếu run cũ (cùng problem `circle_packing`, cùng ngân sách 3 giờ):

| Chỉ số | LEVI | BLADE cũ | BLADE 2.0 (kỳ vọng) |
| --- | --- | --- | --- |
| best_score | 2.6027 | 2.5206 | ≥ 2.55 |
| total_evaluations | 205 | 1032 | tương tự |
| total_cost | $1.35 | $2.81 | tương tự + ít prompt token paradigm hơn |
| archive / pool | 53 elites | **19** | gần K=100 |
| runtime | 10800 s | 10803 s | tương tự |

Các thay đổi đã làm:

1. **Pool 2-lớp + fill-first** (mục 8) — giải quyết gốc rễ "pool=19".
2. **Paradigm prompt có code anchor + inspirations** (mục 6) — frontier nhìn
   thấy cơ chế thật, kết hợp được; trước đó chỉ thấy mô tả nên hay sinh ra
   side-grade.
3. **Default `max_per_family` 8 → 10** — nới thêm một chút khi pool đầy.
4. **`structural_cosine_threshold=0.97`** (mới) — lộ ra qua flag
   `--structural-threshold`; đặt > 1.0 để tắt lớp AST nếu muốn so sánh.
5. Surface thêm `--paradigm-n-anchors`, `--paradigm-n-inspirations` để tune.
6. **Bug fix: `parse_miss` storm ở mutation model** — xem mục 9.1.

### 9.1 Bug fix: `parse_miss (no code in output)` chiếm gần toàn bộ Phase 2

Trong run sau khi đã áp dụng các thay đổi trên, log Phase 2 cho thấy **30/32
variants liên tiếp bị reject** với lý do `parse_miss (no code in output)` —
nghĩa là output của Qwen mutation không chứa fenced code block nào.

**Giả thuyết đầu tiên** (sai): đổ lỗi cho format prompt — nghĩ rằng
`OUTPUT_FORMAT_INSTRUCTION` 2-section (`## Description` + `## Code`) quá khó
cho Qwen3-30B-A3B. Đã thử tạo `MUTATION_OUTPUT_FORMAT` mới gọn hơn.

**A/B test trên Qwen thật** (16 calls mỗi cấu hình, `scripts/test_mutation_format.py`):

| Cấu hình | has_code |
| --- | --- |
| OLD strict format, `max_tokens=1200` | 15/16 (93.8%) — **1 truncated** |
| OLD strict format, `max_tokens=4096` | 16/16 (100%) |
| OLD strict format, `max_tokens=None` | 16/16 (100%) |
| NEW gọn format, `max_tokens=1200` | 15/16 (93.8%) — vẫn truncated |
| NEW gọn format, `max_tokens=4096` | 16/16 (100%) |

Format prompt **không phải nguyên nhân** — cả hai format đều bị miss như
nhau ở mức `max_tokens=1200`. Thủ phạm thật là **`llm_max_tokens=1200`** —
prompt phase-2 chứa 2 seed program full (đến hàng ngàn token mỗi cái) cộng
với `## Description` prose mà model viết trước → response bị truncate
TRƯỚC khi kịp mở fence ` ```python `.

**Fix thật sự**: đổi default `BladeConfig.llm_max_tokens` từ `1200` → `None`
([levi/levi/blade/orchestrator.py:104](levi/levi/blade/orchestrator.py#L104)).
Khi `None`, `LM.acompletion` strip key `max_tokens` trước khi gọi litellm
([levi/levi/clients/lm.py:153](levi/levi/clients/lm.py#L153)) — provider
(OpenRouter / Qwen) tự dùng ceiling mặc định ≥ 4096, đủ rộng để Qwen luôn
hoàn thành cả description + code fence.

Prompt giữ nguyên `OUTPUT_FORMAT_INSTRUCTION` 2-section cho tất cả prompts.
Description luôn có sẵn → không cần `_summarize_if_needed` cho mutation
output → tiết kiệm 1 LLM call cho mỗi candidate, đồng thời pool description-
embedding niching nhận được mô tả "do model viết" thay vì "summary từ code".

`pe_interval=10` và `n_diverse_seeds=4` trong workflow đều là *LEVI parity*
chứ không phải bug:

- LEVI `PunctuatedEquilibriumConfig.interval = 10` mặc định
  ([levi/levi/config/models.py:111](levi/levi/config/models.py#L111)); BLADE
  workflow giữ cùng cadence.
- Khi không có `seed_program`, bootstrap **tự cộng thêm 1 seed** để bù
  ([levi/levi/blade/orchestrator.py:921](levi/levi/blade/orchestrator.py#L921)):
  `n_seeds = cfg.n_diverse_seeds + (0 if cfg.seed_program else 1)` → với
  `n_diverse_seeds=4` thực tế vẫn sinh **5 seed**, ngang LEVI.

Hai con số đó không cần đụng. Cải thiện gốc rễ kỳ vọng đến từ Pool 2.0 +
paradigm prompt code-aware.

## 10. Output artifacts

- **`snapshot.json`** — dump đầy đủ: monitor stats, meta-advice cuối, danh
  sách `paradigm_trials` và tất cả `elites` (kèm code).
- **`best.py` / `best_program.py`** — chương trình điểm cao nhất.
- **`summary.json`** — `run_blade.py` thêm metadata model/budget.

## 11. Logging

`evolve_code_blade` gọi `_setup_logging()` (giống Levi) và orchestrator có:

1. **`_status_monitor`** — heartbeat mỗi 30 s:
   `[Status] Cost: $… | Evals: … | Clients in-flight: … | Eval in-flight: … | Pool: … | Best: … | Elapsed: …s`
2. **`_record_reject(source, score, error_msg)`** — helper duy nhất bọc
   `monitor.record_eval` cho mọi spot reject. Log:
   `[Eval #N] {model:27s} ERROR (source): {msg[:80]}`.
3. **`_admit`** — log dòng:
   `[Eval #N] {model:27s} {status:12s} | source: … | score: … | best: … | $cost`
   với `status` ∈ `{NEW BEST ★, accepted, rejected}`.
4. **In-flight counters** — wrap `_call` và `_evaluate_code` bằng try/finally
   tăng/giảm `_client_in_flight` và `_eval_in_flight`.
5. **PE trigger log** giàu:
   `[BLADE PE] trigger #N at eval=… | stage=… | best=… | pool=… | families=…`.
6. Thông báo phase đầu cuối + summary sau bootstrap:
   `[BLADE] bootstrap complete — pool=… best=… cost=$… evals=…`.

## 12. Knobs CLI / workflow advanced_options

```text
# scripts/run_blade.py flags (mới + đã có)
--pool-k                       # PoolConfig.K (target 100)
--niche-threshold              # lớp 1: description cosine (0.92)
--structural-threshold         # lớp 2: AST cosine (0.97); >1.0 để tắt
--family-threshold             # family clustering (0.72)
--max-per-family               # cap khi đầy (10)
--paradigm-n-anchors           # anchors có code (3)
--paradigm-n-inspirations      # inspirations description-only (5)
--n-paradigm-variants          # fanout sau paradigm shift (4)
--n-diverse-seeds              # phase-1 sequential seeds (4-5)
--n-variants-per-seed          # phase-2 parallel variants/seed (20)
--pe-interval                  # cron N evals (50)
--meta-advice-interval         # cron N evals (50)
--no-repair / --no-meta-advice # tắt các phụ trợ
```

Trong `.github/workflows/blade.yml`, các knob nâng cao đi qua JSON
`advanced_options`, ví dụ:

```json
{
  "structural_threshold": 0.95,
  "max_per_family": 12,
  "paradigm_n_inspirations": 8,
  "pe_interval": 30
}
```

## 13. Gợi ý A/B tiếp theo

Để khẳng định Pool 2.0 + paradigm prompt mới thực sự cải thiện kết quả
(thay vì kết luận từ một run đơn lẻ):

1. **A/B Pool**: chạy 2 lần cùng seed, một lần `--structural-threshold 1.5`
   (tắt lớp AST, fallback về description-only như bản cũ) → so `pool_size`
   và `best_score` cuối run.
2. **A/B Paradigm**: tạm rollback `paradigm_n_inspirations=0` để xem có
   cần inspirations description-only không; hoặc `paradigm_n_anchors=1` vs
   `=3` để đo giá trị biên của việc cho frontier xem nhiều code anchor.
3. **A/B family cap**: `max_per_family=10` (default) vs `=100` (gần như
   tắt) để xem family cap có còn cần khi pool 2-lớp dedup đã chặt.

Mỗi A/B nên dùng cùng `seed=0` (đã có trong `BladeConfig`) để loại bỏ
noise khởi tạo.
