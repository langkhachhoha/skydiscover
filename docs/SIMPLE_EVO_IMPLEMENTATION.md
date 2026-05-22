# SIMPLE-EVO Implementation Notes

Phase A & B (modules không phụ thuộc evolution pipeline) đã được implement,
test, và live-validate qua OpenRouter. Phần này tổng kết: file layout,
config defaults sau khi tune, kết quả live test, workflow CI, và những
điểm còn mở.

> Design reference: [SIMPLE_EVO.md](SIMPLE_EVO.md)

---

## 1. File layout

```text
levi/levi/simple/
├── __init__.py        # Re-exports
├── embedder.py        # DescriptionEmbedder (text-embedding-3-small qua litellm)
├── parser.py          # OutputParser, OUTPUT_FORMAT_INSTRUCTION, fallback_summarize
├── pool.py            # Pool, PoolConfig, Program (top-K + niching + family cap)
├── monitor.py         # Monitor, MonitorConfig (3 sliding-window signals)
└── selector.py        # Selector, SelectorConfig (UCB-style priority)

levi/tests/simple/
├── __init__.py
├── _fake_embeddings.py    # deterministic unit-vectors cho unit tests
├── test_parser.py         # 8 tests (incl. header-less prose extraction)
├── test_pool.py           # 10 tests
├── test_monitor.py        # 8 tests
├── test_selector.py       # 6 tests
├── live_smoke.py          # manual live API test (4 paradigms, ~3¢/run)
└── format_compliance.py   # stress test 3 problems × 5 paradigms × 2 temps
```

Tổng cộng **32 unit tests pass** (chạy 4 giây), 1 live smoke test (4
paradigm calls), 1 format-compliance stress test (30 LLM calls). Toàn bộ
xanh.

---

## 2. Module responsibilities

### `embedder.py` (DescriptionEmbedder)

- Wraps `litellm.embedding` với default model
  `openrouter/openai/text-embedding-3-small` (1536-d).
- Tự L2-normalize trước khi cache → `cosine()` chỉ là dot product.
- LRU-ish cache theo SHA1 của description (insertion-order eviction).
- `embed_batch` batch litellm call, tận dụng cache cho từng phần tử, và
  sort response theo `index` field để defensively giữ thứ tự khớp input.

### `parser.py`

`OutputParser.parse(text) -> LLMOutput` thử 5 recognized output shapes,
theo thứ tự:

1. `## Description` + `## Code` headers (the requested format).
2. `## Description` header, no `## Code` header, code in a fence right after.
3. **Header-less prose-before-fence**: prose preceding the first fenced
   code block là description. Đây là branch quan trọng — thực tế
   gpt-4o-mini thường bỏ headers nhưng vẫn viết prose paragraph trước code.
4. Code fence only: description empty, caller invokes `fallback_summarize`.
5. Raw Python (`def`/`class`/`import`/`from` ở đầu): code only.

`OUTPUT_FORMAT_INSTRUCTION` yêu cầu prose paragraph 2-4 sentences covering
paradigm + data structures + distinguishing trick. Không ép tag format
cứng — đoạn văn tự nhiên cho embedding chất lượng tốt hơn.

`fallback_summarize(code, completion_fn)`: gọi mutation model sinh
description khi parser thấy code-only. Reuse cùng mutation model, không
thêm config.

### `pool.py`

Ba lớp selection pressure:

1. **Semantic dedup** (`niche_cosine_threshold = 0.92`): nếu cosine với
   bất kỳ program nào trong pool ≥ threshold, replace (nếu score cao hơn)
   hoặc drop.
2. **Family cap** (`family_cosine_threshold = 0.72`, `max_per_family = 8`):
   single-linkage clustering trên embedding. Khi family đầy, eviction
   weakest-in-family thay vì global lowest.
3. **Global top-K** (`K = 100`): khi pool vượt K, drop lowest-score.

`representatives(phase, n)` cho 3 phase frontier:

- `early`: greedy MMR với `λ=0.2` (diversity-bias).
- `mid`: anchor top-score + MMR pick các complement với `λ=0.5`.
- `late`: top-N by score thuần.

`recent_diversity(last=20)`: mean pairwise cosine của last-N programs để
feed `Monitor.diversity_window`.

`reset_uses_after_paradigm()`: zero `uses_count` cho tất cả programs sau
khi frontier accept một paradigm shift.

### `monitor.py`

Ba signals, đều sliding-window:

| Signal             | Maxlen      | Threshold                              | Triggers          |
| ------------------ | ----------- | -------------------------------------- | ----------------- |
| `plateau_steps`    | (counter)   | `stuck_plateau_threshold = 80`         | `is_stuck()`      |
| `accept_window`    | 50          | `stuck_accept_threshold = 0.08`        | `is_stuck()`      |
| `diversity_window` | 20          | `collapse_diversity_threshold = 0.78`  | `is_collapsing()` |

- `stagnation_level()` = `min(1, plateau_steps / plateau_max)` ∈ [0,1] →
  feeds `get_budget_stage()` để route frontier 3 phase.
- `is_stuck()` → switch advisor sang offensive mode + toggle operator mix.
- `is_collapsing()` → trigger diversity injection (forced crossover từ
  2 niches xa nhau nhất) ở pipeline layer (Phase C).

**Critical fix during testing**: ban đầu `record_eval` count "new best"
cho mọi candidate có score > best_score, dẫn đến eval đầu tiên (score
> −inf) reset plateau ngay → off-by-one. Sửa: chỉ count khi
`accepted=True` (rejects không thể là incumbent).

### `selector.py`

Priority cho 1 program `p` cho trước đã pick set `S`:

```text
priority(p; S) = score_norm(p)                                  # exploit
              + α · sqrt(log(1+N) / (1+uses_count(p)))          # UCB novelty
              + β · exp(−(N − created_at(p)) / τ)               # recency
              − γ · max_{q ∈ S} cosine(p, q)                    # diversity penalty
```

Defaults `(α, β, γ)`: `(0.5, 0.3, 0.4)` healthy, `(0.8, 0.5, 0.7)` stuck.
`τ = 30 evals`. `crossover_min_family_separation = 0.65`.

Methods:

- `select_parent(programs, n_total, stuck)`
- `select_two_parents(...)`: pick parent 1, then prefer parent 2 from a
  cross-family pool (cos < `crossover_min_family_separation`).
- `select_inspirations(programs, exclude, n_total, stuck, k=3)`:
  greedy batched pick, mỗi step update set picked → diversity penalty
  built in.

---

## 3. Config defaults sau tuning (live data driven)

Các threshold được tune dựa trên cosine distribution thực tế trên
text-embedding-3-small (live runs với coin-change, longest-path-DAG,
tsp-approx).

| Param                          | Default     | Lý do                                                                  |
| ------------------------------ | ----------- | ---------------------------------------------------------------------- |
| `niche_cosine_threshold`       | 0.92        | Paraphrases ≥ 0.92; distinct paradigms ≤ 0.78                          |
| `family_cosine_threshold`      | 0.72        | Cross-paradigm ~0.55, same-paradigm variants ~0.73–0.85                |
| `max_per_family`               | 8           | Cho phép vài variant per paradigm, nhưng cap                           |
| `collapse_diversity_threshold` | 0.78        | Same-paradigm-only window có mean ~0.78–0.85                           |
| `stuck_plateau_threshold`      | 80          | Plateau dài đáng kể                                                    |
| `stuck_accept_threshold`       | 0.08        | Cửa sổ 50 → < 4 accept = stuck                                         |
| `ucb_alpha_healthy/stuck`      | 0.5 / 0.8   | Stuck tăng exploration                                                 |
| `ucb_beta_healthy/stuck`       | 0.3 / 0.5   | Recency boost vừa phải                                                 |
| `ucb_gamma_healthy/stuck`      | 0.4 / 0.7   | Stuck tăng diversity force                                             |
| `recency_tau`                  | 30          | Half-life ~20 evals                                                    |

---

## 4. Live test results

### Live smoke (`tests/simple/live_smoke.py`)

Sinh 4 solutions cho bài coin change, hints khác paradigm:

1. dynamic programming (bottom-up)
2. dynamic programming (top-down memoization)  ← cùng họ với (1)
3. BFS over partial sums
4. greedy descent with backtracking

Pairwise cosine distribution sau khi parser được nâng cấp (chấp nhận
prose-before-fence):

| Pair                        | Cosine                                              |
| --------------------------- | --------------------------------------------------- |
| DP (bottom-up) vs DP (memo) | 0.728 — same family, sát threshold 0.72             |
| DP vs BFS                   | 0.635                                               |
| DP vs greedy                | 0.714                                               |
| DP-memo vs BFS              | 0.661                                               |
| DP-memo vs greedy           | 0.763 — boilerplate prefix bị inflate               |
| BFS vs greedy               | 0.772 — boilerplate prefix bị inflate               |

4 programs → 4 distinct families (đúng kỳ vọng). Pairwise cosine
cross-paradigm thường dưới 0.72 trừ khi prefix boilerplate trùng nhau.

### Format-compliance matrix (`tests/simple/format_compliance.py`)

30 LLM calls (3 problems × 5 paradigms × 2 temperatures):

| Outcome              | Count     | Share |
| -------------------- | --------- | ----- |
| Direct hits          | 30        | 100%  |
| Fallback summary     | 0         | 0%    |
| Hard failures        | 0         | 0%    |
| Boilerplate openers  | 4 / 30    | 13%   |
| Avg description size | 523 chars | —     |

Per-paradigm: dynamic-programming, bfs, branch-and-bound, greedy,
simulated-annealing — mỗi paradigm 6/6 direct hits, không cần fallback.
Per-temperature: T=0.4 và T=0.9 cùng 15/15 direct hits.

**Key finding**: parser nhánh "prose-before-fence" (case 3) là then chốt
— gpt-4o-mini thường bỏ `## Description` header nhưng vẫn viết paragraph
trước fence. Việc parser nhận shape này đã đẩy direct-hit rate từ 53%
lên 100%. Fallback model vẫn tồn tại như safety net cho output thực sự
chỉ có code.

---

## 5. CI workflow (`.github/workflows/_simple_evo.yml`)

Reusable workflow song song với `_levi.yml`:

- **Phase 1 — Offline unit tests** (always): `pytest tests/simple/ -v`.
- **Phase 2 — Live LLM smoke** (opt-in via `live_smoke: true`, cần
  `OPENROUTER_API_KEY`):
  - `live_smoke.py` (4 paradigm calls, ~3¢).
  - `format_compliance.py` (30 calls, ~3¢; fails khi hard-failure rate
    vượt 5%).
  - Upload `outputs/simple_evo_format_compliance.json` làm artifact.

Inputs: `live_smoke`, `mutation_model`, `embedding_model`,
`hard_failure_threshold`. Mặc định mutation = `openrouter/openai/gpt-4o-mini`,
embedding = `openrouter/openai/text-embedding-3-small`.

Trigger thủ công qua `workflow_dispatch` hoặc gọi từ workflow khác bằng
`workflow_call`.

---

## 6. Tích hợp với LEVI hiện tại (Phase C, D — chưa làm)

Các module này stand-alone, **chưa được wire vào pipeline LEVI**. Để
hoàn thiện SIMPLE-EVO end-to-end cần:

- **Operator layer**: 2 prompt template (mutate / crossover) sinh code
  với `OUTPUT_FORMAT_INSTRUCTION`. Reuse `levi/clients/lm.py` cho LLM
  call. Reuse `levi/pipeline/producer.py` error handling + self-repair
  loop (chỉ đổi parse step).
- **Frontier orchestrator**: wrap `levi/equilibrium/prompts.py`
  `PARADIGM_SHIFT_PROMPTS` (early/mid/late) với:
  - Routing input: `Monitor.stagnation_level()` thay cho PPS công thức.
  - Representatives: `Pool.representatives(phase, n=3)` chỉ inject
    `description + score`.
  - Recent trials log: maintain ring buffer của K=5 paradigm attempts,
    inject vào `{strategy_log_block}` slot có sẵn.
- **Method entry point**: `levi/methods/simple_evo.py` song song
  `levi/methods/levi.py`. Reuse error archive, repair, meta-advisor
  hiện tại của LEVI nguyên xi.

---

## 7. Open items

- **DP-DP cosine 0.73**: borderline với family_cosine_threshold 0.72.
  Sau Phase C/D nếu thấy DP variants leaks vào pool, hạ threshold xuống
  ~0.70 hoặc tăng `max_per_family` lên 10.
- **Boilerplate openers (13%)**: model vẫn thỉnh thoảng mở đầu "This
  solution employs...". Acceptable — phần sau vẫn nội dung-dense và
  embedding capture đúng paradigm.
- **Cross-paradigm cosine có thể bị inflate bởi prefix boilerplate**
  (vd cos = 0.77 giữa BFS-greedy do cả hai mở đầu giống). Nếu trở thành
  vấn đề, có thể strip leading boilerplate trước khi embed (regex
  preprocessor trong embedder).
- **`pe_cron_interval`**: chưa code phase frontier orchestrator nên
  chưa benchmark.
- **`uses_count` reset semantics**: đã có `reset_uses_after_paradigm()`
  nhưng chưa được orchestrator gọi.

---

## 8. Cách chạy local

### Unit tests (offline, ~4s)

```bash
cd levi
python -m pytest tests/simple/ -q
```

### Live smoke (cần `OPENAI_API_KEY` cho OpenRouter, ~3¢)

```bash
cd levi
python tests/simple/live_smoke.py
```

### Format-compliance matrix (~3¢, 30 LLM calls)

```bash
cd levi
python tests/simple/format_compliance.py
```

Exit code 0 khi hard-failure rate < 5%, ngược lại 2. JSON results
write tại `outputs/simple_evo_format_compliance.json`.
