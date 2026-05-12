# FORE — Implementation Log (v0 smoke run)

Đây là log những gì đã code và kết quả smoke test 2 iter trên `circle_packing`. Plan đầy đủ ở [FORE_METHOD_PLAN.md](FORE_METHOD_PLAN.md).

## 1. Tóm tắt nhanh

- Tổng số file mới: **10** (8 file Python + 2 template prompt).
- Tổng số file sửa: **3** (`config.py`, `route.py`, `cli.py`).
- Smoke test: ✅ chạy thông 2 iter, iter 1 nhảy 0.36 → 0.888, iter 2 child 0.762 với 2 lần retry (toolchain xử lý lỗi LLM hoạt động đúng).
- LLM emit `<fore_meta>` block đúng format ngay từ iter 1 (model: `openrouter/openai/gpt-5`).
- Cluster, verdict, POV diagnostics đều được sinh và lưu vào checkpoint.

## 2. Cây file đã tạo / sửa

```
skydiscover/
├── cli.py                                  (MOD: thêm 'fore' vào _SEARCH_CHOICES)
├── config.py                               (MOD: thêm FOREDatabaseConfig + register)
├── search/
│   ├── route.py                            (MOD: import + register database/controller)
│   └── fore/                               (NEW package)
│       ├── __init__.py                     (NEW)
│       ├── fertility.py                    (NEW — math core)
│       ├── descriptions.py                 (NEW — strategy block parsing + Jaccard)
│       ├── review.py                       (NEW — ReflectiveReviewer)
│       ├── database.py                     (NEW — FOREDatabase)
│       └── controller.py                   (NEW — FOREController)
├── context_builder/
│   └── fore/                               (NEW package)
│       ├── __init__.py                     (NEW)
│       ├── builder.py                      (NEW — FOREContextBuilder)
│       └── templates/
│           ├── diff_user_message.txt       (NEW)
│           └── full_rewrite_user_message.txt (NEW)
configs/
└── fore.yaml                               (NEW — starter config)
```

## 3. Trách nhiệm từng file

### 3.1 `search/fore/fertility.py` — POV math core
- `NIGPrior`: Normal-Inverse-Gamma prior `(mu_0, kappa_0, alpha_0, beta_0)`.
- `FertilityStats`: per-parent sufficient statistics cho `Δ⁺ = max(f(child) - f(parent), 0)`. Track `n`, `sum_delta_plus`, `sum_sq_delta_plus`, `negative_count`, cộng các structural prior input (`novelty_score`, `cluster_rarity`, `age_at_birth`).
- `posterior_t` → trả `(loc, scale, df)` của Student-t marginal posterior trên `mu` theo công thức ở §2.3 của plan.
- `sample_mu` → Thompson sampling bằng trick `t = Z / sqrt(W/df)` với `Z ~ Normal(0,1)`, `W ~ ChiSquared(df)` (`random.gauss` + `random.gammavariate`, không cần scipy).
- `fertility_multiplier(K, alpha, k_max)` → closed-form geometric series `(1 - alpha^K)/(1 - alpha)` (Lemma 1, §2.2).
- `pov_score(...)` → một mẫu Thompson của POV: `fitness + mult * max(mu_sample, 0) + structural_bonus_via_prior`.
- `expected_pov(...)` → dạng deterministic (dùng cho logging).
- Heavy-tail clipping trong `update_with_child` để posterior không bị các Δ⁺ outlier kéo đi.

### 3.2 `search/fore/descriptions.py` — Strategy description schema
- `StrategyDescription` dataclass với `strategy_label / description / hypothesis / diff_from_parent / verdict / cluster_id`.
- `parse_strategy_block(llm_response)` → regex `<fore_meta>(.*?)</fore_meta>`; chịu được JSON trong code fence; fallback rỗng nếu parse fail → không crash iteration.
- `compute_verdict(parent_fitness, child_fitness, parent_mean_delta_plus)` → bucket `improved / regressed / dead_end / stepping_stone` theo logic ở §4.2.
- `tokenize_strategy` + `jaccard` cho clustering on-the-fly (regex + set ops, không cần embedding ngoài).

### 3.3 `search/fore/review.py` — Reflective Reviewer
- `FertilityReview` dataclass với `effective_lineages / exhausted_lineages / embryonic_lineages / next_steps`, có `uses_remaining` để decay dần.
- `ReflectiveReviewer.generate` async: build prompt JSON-output, gọi `LLMPool`, parse JSON (tolerant: strip code fence, fallback `{...}` regex). Trả `None` khi fail để controller fallback an toàn.
- `render_markdown()` để builder inject thẳng vào prompt.

### 3.4 `search/fore/database.py` — FOREDatabase (~470 LOC)
Subclass `ProgramDatabase`. Trách nhiệm chính:

- **`add(program, iteration)`**:
  1. Đọc `metadata['fore']` (controller đã gắn vào).
  2. Assign cluster bằng Jaccard trên token của description (fallback dùng code token nếu description rỗng — tránh seed program rơi vào "phony cluster").
  3. Compute `novelty_score` (mean code-distance đến ≤30 neighbors qua `CodeDiversity`) và `cluster_rarity = 1 - size/total`.
  4. Nếu có `parent_id`, gọi `parent_stats.update_with_child(delta_normalized)` rồi `compute_verdict`.
  5. Cập nhật best, push improvement signal vào `_recent_improvements`.
  6. Enforce population cap qua eviction (protect best + initial + top expected-POV).

- **`sample(num_context_programs)`**:
  1. Thompson-sample POV cho mọi program trong pool.
  2. Lấy top-1 làm parent.
  3. Context: 1 sibling cùng cluster nhưng **khác verdict** (đại diện complementary failure mode) + (n-1) cross-cluster diverse, fallback nếu thiếu.
  4. Trả `parent_dict` với key = `_build_parent_label(...)` (block markdown giải thích tại sao parent này được chọn — đây là chỗ inject reasoning xuống LLM).

- **`detect_stagnation()`** — 3 trigger độc lập:
  1. Rate trigger: rolling improvement rate < `review_rate_threshold`.
  2. POV-floor trigger: median của top-10 POV (trung bình 3 Thompson sample) < `pov_floor`.
  3. All-cluster-exhausted: max cluster mean Δ⁺ < 1e-4.

- **`get_fertility_summary()`**: bảng per-cluster (size, mean_fitness, mean_delta_plus, negative_frac, label) — input chính cho `ReflectiveReviewer`.

- **`get_pov_diagnostics(top_k)`**: deterministic POV ranking — inject vào prompt cho LLM thấy bức tranh hiện tại.

- **`set_active_review` / `consume_review_for_prompt`**: quản lý lifecycle review (decrement `uses_remaining` mỗi lần inject).

### 3.5 `search/fore/controller.py` — FOREController
Rất gọn (~150 LOC). Subclass `DiscoveryController` và chỉ override `_run_iteration`:

1. `_maybe_run_review(iteration)` — nếu `detect_stagnation()` báo có và `can_run_review()` đúng → await `ReflectiveReviewer.generate(...)` → `database.set_active_review(review)`.
2. Đẩy `fore_review` (rendered markdown) và `fore_diagnostics` (POV top-3) vào `self._prompt_context`, để base controller pass vào builder. *(Không phải patch builder để đọc trực tiếp DB → giữ separation of concerns.)*
3. Gọi `super()._run_iteration(...)` để dùng nguyên toàn bộ logic gốc (parallel/sequential, retry, eval, parse diff).
4. Sau khi result trả về, `parse_strategy_block(result.llm_response)` → gắn vào `result.child_program_dict["metadata"]["fore"]` trước khi base controller làm `database.add(...)`. Vì `_process_iteration_result` đọc dict này để khởi tạo `Program`, database sẽ nhận đúng metadata.
5. Logging JSONL ở `fore_stats_<ts>.jsonl` (event types: `review_generated`, `review_failed`, `child_evaluated`).

### 3.6 `context_builder/fore/builder.py` — FOREContextBuilder
Subclass `DefaultContextBuilder`. Thêm 2 placeholder:

- `{fore_guidance}`: active review (rendered markdown) + POV diagnostics table.
- `{fore_meta_instructions}`: spec JSON block mà LLM phải emit (label / description / hypothesis / diff_from_parent).

Parent-selection label đã ở key của `parent_dict` (return từ `database.sample`), do đó được render bởi base `_format_current_program` — **không cần** patch ở builder.

### 3.7 Templates
- `diff_user_message.txt` & `full_rewrite_user_message.txt`: copy theo style AdaEvolve, chỉ khác ở chỗ thay `{search_guidance}` bằng `{fore_guidance}` và thêm section `# FORE strategy block (required)` cuối prompt.

### 3.8 `configs/fore.yaml`
Starter config với toàn bộ hyperparam có default (xem `FOREDatabaseConfig` ở `config.py`).

### 3.9 Wiring
- `config.py`: thêm `FOREDatabaseConfig` (dataclass) và `_DB_CONFIG_BY_TYPE["fore"] = FOREDatabaseConfig`.
- `route.py`: import `FOREDatabase`, `FOREController` + `register_database("fore", ...)`, `register_controller("fore", ...)`.
- `cli.py`: thêm `"fore"` vào `_SEARCH_CHOICES`.

## 4. Smoke test — circle_packing, 2 iterations

Command:
```bash
uv run skydiscover-run benchmarks/math/circle_packing/initial_program.py \
  benchmarks/math/circle_packing/evaluator.py \
  --config benchmarks/math/circle_packing/config.yaml \
  --search fore \
  --model openrouter/openai/gpt-5 \
  --iterations 2 \
  --output outputs/local/fore_smoke
```

### 4.1 Kết quả

| Iter | Parent | Child fitness | Verdict | Cluster | Strategy label | Retry |
|---|---|---|---|---|---|---|
| 0 | – | 0.364 (seed) | seed | 0 | unspecified | – |
| 1 | seed (POV-sampled) | **0.888** | improved | 1 | `hexagonal-shell` | 0 |
| 2 | iter-1 best | 0.762 | dead_end | 2 | `pairwise-projection` | 2 retries (lỗi `ptp` + invalid shape) |

Best program: **`sum_radii = 2.34` / target 2.635 / ratio 0.888** sau 2 iter — pipeline hoạt động đúng kỳ vọng.

### 4.2 Bằng chứng từng thành phần FORE đã hoạt động

**1. `<fore_meta>` được parse và lưu**:
```json
{
  "strategy_label": "hexagonal-shell",
  "description": "<...>",
  "hypothesis": "A near-regular hex lattice maximizes local density while pruning corners reduces edge penalties; the closed-form radii avoid the over-shrinking from proportional pairwise scaling, yielding a much larger sum of radii.",
  "diff_from_parent": "<...>",
  "verdict": "improved",
  "cluster_id": 1
}
```

**2. Verdict được compute đúng**:
- iter 1: parent fitness 0.36, child 0.89, parent_mean_delta_plus = 0 → `improved` (vì delta > threshold).
- iter 2: parent fitness 0.89, child 0.76, parent_mean_delta_plus = 0 (parent chưa có positive children được record trước iter này) → `dead_end` (delta < -threshold AND parent chưa có track record). ✅ đúng logic.

**3. Clustering hoạt động**:
- Seed → cluster 0 (fallback dùng code tokens vì không có description).
- `hexagonal-shell` → cluster 1 (mới, vì Jaccard với cluster 0 < threshold 0.55).
- `pairwise-projection` → cluster 2 (mới).

**4. Parent label được inject thật vào prompt**:
```
## PARENT SELECTION (FORE — Thompson sampling on Posterior Offspring Value)
This parent was chosen because its sampled POV score was 3.6878.
- Current fitness: 0.8880
- Children evaluated so far: 0 positive, 0 non-positive
- Empirical mean positive improvement (Δ+): 0.0000
- Strategy cluster: id=1, size=1, label='hexagonal-shell'
- This parent has no offspring history yet — Thompson sampling is exploring its potential.
- Original hypothesis: A near-regular hex lattice maximizes local density...
```

**5. FORE stats JSONL** (`outputs/local/fore_smoke/fore_stats_20260512_122200.jsonl`):
```jsonl
{"event":"child_evaluated","iteration":1,"strategy_label":"hexagonal-shell","fitness":0.888,...}
{"event":"child_evaluated","iteration":2,"strategy_label":"pairwise-projection","fitness":0.762,...}
```

**6. Stagnation review chưa kích hoạt** (đúng — chỉ có 2 iter, `review_window=12`, không đủ history). Trigger logic sẽ được test ở run dài hơn.

### 4.3 Sanity checks (offline, không cần LLM)

```python
from skydiscover.search.fore import FertilityStats, NIGPrior, pov_score
import random
prior = NIGPrior()

# Posterior convergence
s = FertilityStats()
for _ in range(50): s.update_with_child(0.3)
# sample_mu ~ 0.31 (close to 0.3) ✅

# Stepping-stone bonus (no observation yet)
s_empty = FertilityStats()
pov_score(0.5, s_empty, prior, k_remaining=50, rng=random.Random(0))
# ~1.7 (large because posterior is wide and Thompson sampled high tail) ✅

# Strategy-block parser
parse_strategy_block('foo <fore_meta>{"strategy_label":"hex","description":"x"}</fore_meta> bar')
# → StrategyDescription(strategy_label='hex', description='x', ...) ✅
```

## 5. Điểm cần lưu ý / công việc tiếp theo

1. **Stagnation review chưa được kích hoạt trong 2 iter** — cần chạy ≥ `review_window=12` iter để có data thật cho `detect_stagnation`. Khuyến nghị chạy 20-30 iter để xác nhận review trigger hoạt động end-to-end.
2. **`pov_floor` mặc định = 0.0** quá lỏng cho task này; sau ablation sẽ tune theo magnitude của fitness range.
3. **`delta_normalization` mặc định = 1.0** — vì fitness ở đây thuộc [0, 1], các Δ⁺ ~0.5 không bị scale bất thường. Với task khác (vd combined_score range > 1), nên expose qua YAML.
4. **Unit tests cho `fertility.py`** (`tests/test_fore_fertility.py`) chưa viết — đề xuất trong PR kế tiếp:
   - `test_posterior_convergence`
   - `test_thompson_explores_unseen_parents`
   - `test_stepping_stone_preferred`
   - `test_verdict_buckets_match_design`
5. **Checkpoint không lưu FORE-specific state** (`fertility`, `clusters`, `strategy`, `_active_review`) — base class chỉ lưu `programs/`. Khi resume, FORE state sẽ rỗng và phải rebuild từ `metadata['fore']` của programs. Cho v0 đủ dùng; PR sau nên override `save`/`load` như AdaEvolve.
6. **Parallel iteration mode (`max_parallel_iterations > 1`)** chưa kiểm tra race. Hiện tại `FertilityStats.update_with_child` không có lock — chấp nhận được với base controller pattern (sync `_process_iteration_result` ngoài semaphore) nhưng cần khẳng định bằng test khi muốn dùng parallel.
7. **CLI flag** `_SEARCH_CHOICES` hard-coded là technical debt sẵn có trong codebase — đã thêm `fore` cho khớp, nhưng nên refactor đọc từ registry.

## 6. Tổng kết

Toàn bộ thiết kế trong [FORE_METHOD_PLAN.md](FORE_METHOD_PLAN.md) đã được hiện thực hóa, modular hóa thành 1 package `search/fore/` + 1 package `context_builder/fore/`, không sửa logic nào của controller / base database / archive có sẵn (chỉ subclass + register). Smoke test 2 iter trên `circle_packing` xác nhận pipeline thông suốt: POV sampling chọn parent → label markdown được inject → LLM trả về code + `<fore_meta>` block → parse → cluster → verdict → log. Sẵn sàng cho run dài (20-100 iter) để đánh giá chất lượng review + baseline so sánh với AdaEvolve/OpenEvolve_native.
