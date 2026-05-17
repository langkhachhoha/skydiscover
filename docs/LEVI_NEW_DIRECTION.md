# Hướng nghiên cứu mới cho LEVI — năm cải tiến cho paper A

> **Đọc cho ai?** Người hoàn toàn chưa biết LEVI là gì cũng nên theo được.
> Tài liệu này đi từ "LEVI giải bài toán gì" → "tại sao đổi" → "năm cơ chế
> mới" → "code ở đâu để tune" → "chạy thế nào" → "đề xuất ablation cho paper".

---

## 1. LEVI 101 — framework hiện tại làm gì?

LEVI là một framework **tiến hoá code bằng LLM**. Bài toán giải:

> Cho một mô tả bài toán (vd. *circle packing*), một function signature, và
> một hàm chấm điểm `score_fn(code)`. Tìm đoạn code Python có điểm số cao
> nhất trong giới hạn ngân sách (số evaluations, $, hoặc thời gian).

### Pipeline cơ bản

1. **Init** — sinh vài seed program đa dạng (Phase 1) rồi nhân bản (Phase 2).
2. **CVT-MAP-Elites archive** — chia *behavior space* (đặc trưng AST như
  `loop_count`, `branch_count`…) thành ~50 ô (centroid), mỗi ô giữ một
   elite duy nhất theo score.
3. **Main loop** (producer ↔ consumer):
  - *Producer* lấy parent + inspirations từ archive, gọi LLM nhỏ → tạo
   candidate mới.
  - *Consumer* chạy `score_fn`, đẩy vào archive nếu đánh bại elite trong ô.
4. **Punctuated Equilibrium (PE)** — định kỳ (vd. mỗi 10 evals) một LLM
  **lớn** sinh ra một paradigm shift hoàn toàn khác. LLM nhỏ làm vài
   variant từ paradigm đó.
5. **SAL** (Stagnation-Adaptive Levi) — đo độ "stagnation" `s(t)` và điều
  chỉnh nhiều cơ chế: PE prompt staging, contrastive context, meta-advice
   offensive, Thompson bandit, Hard-PE.

### Điểm yếu thấy rõ khi chạy circle_packing


| Điểm yếu                    | Triệu chứng trong log                                                                                                                          |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **A. Stagnation đơn giản**  | Cũ chỉ là `max(plateau_ratio, budget_ratio)` — không phân biệt "đang dậm chân" với "ngân sách cao mà vẫn đang cải thiện".                      |
| **B. Bandit fragmented**    | Cũ có softmax-T ∈ {0.3, 0.7, 1.0, 1.2} → 4 arms mỗi model → posterior chia 4 lần, hội tụ chậm.                                                 |
| **C. Heavy model lãng phí** | Một PE chỉ tạo 1 paradigm + few variants, không học từ lịch sử các paradigm cũ.                                                                |
| **D. Code lỗi vứt đi**      | Mỗi candidate syntax-error hay runtime-error đều bị huỷ ngay.                                                                                  |
| **E. Local optimum**        | Khi đã hội tụ, không có cách nào để mở rộng archive sang vùng behavior mới — `n_centroids` cố định, mọi cell có elite rồi thì search kẹt cứng. |


Sáu cơ chế dưới đây giải quyết từng điểm yếu.

---

## 2. Tổng quan năm đóng góp


| #   | Tên                                             | Giải quyết | Mới hay cũ?               |
| --- | ----------------------------------------------- | ---------- | ------------------------- |
| 1   | **PPS** — Posterior-Plateau Stagnation          | A          | Cải tiến từ stagnation cũ |
| 2   | **AdaptiveRankSampler**                         | B          | Mới hoàn toàn             |
| 3   | **Phase-based PE prompts + Strategy History**   | C          | Cải tiến + mới            |
| 4   | **Code Error Repair** (light-model one-shot)    | D          | Mới                       |
| 5   | **Adaptive Island Expansion** (AdaEvolve-style) | E          | Mới                       |


Phần 3–7 dưới đây là từng cơ chế: **bài toán → công thức → tại sao đúng →
hyper-params → tham chiếu code → góc nhìn paper**.

---

## 3. PPS — Posterior-Plateau Stagnation

### Vấn đề

Giá trị stagnation cũ:

```text
s_old(t) = max( n_since_best / τ ,  budget_consumed / budget_total )
```

→ Lúc `t=0` đã có giá trị > 0 nếu budget cố định. Cuối run thì luôn ≈ 1 cho
dù model vẫn đang cải thiện. Quá thô.

### Công thức mới

Coi mỗi NEW BEST là một event Poisson với rate λ̂(t):

```text
p(t)       = min(1, n_since_best / τ)                       # plateau ∈ [0,1]
b(t)       = max ratio across dollars / evals / seconds      # budget ∈ [0,1]
B_total    = ngân sách của trục thống trị
B_used     = đã tiêu                  ;   B_rem = B_total - B_used
B_W        = ngân sách kể từ NEW BEST đầu tiên trong window (history)
k_W        = số NEW BEST trong window
λ̂(t)       = (k_W + 1) / (B_W + ε)                          # Laplace smoothing
survival   = exp(-λ̂ · B_rem)                                # P(không cải thiện trong B_rem)
posterior  = p(t) · survival                                # gating bằng plateau
α(t)       = b(t)²                                          # confidence trong λ̂
s(t)       = (1 - α) · p + α · posterior
# Safety floor: cuối run (b≥0.95, p≥0.95) thì s ≥ p
```

### Tại sao đẹp paper-wise?

- **Có nguồn gốc thống kê rõ ràng**: Poisson survival, hazard rate, Laplace
smoothing — tất cả đều là khái niệm chuẩn trong reliability / survival
analysis. Không phải heuristic ad-hoc.
- **Tự calibration**: nếu run đang cải thiện (k_W lớn), `posterior_stuck`
→ 0 → s(t) → 0 dù budget gần hết → KHÔNG panic-trigger PE.
- **Properties** (đã verify bằng test):
  - Lúc `t=0`, `α=0` → s(t) ≈ p(t).
  - Cuối run + đang cải thiện → s(t) ≈ 0.
  - Cuối run + dậm chân → s(t) ≈ p(t) ≈ 1.

### Code


| File                                            | Nội dung                                                                             |
| ----------------------------------------------- | ------------------------------------------------------------------------------------ |
| `levi/levi/pipeline/state.py`                   | `PipelineState.stagnation_depth(tau)`, `record_new_best()`, `new_best_history` deque |
| `levi/tests/test_pps_adaptive_rank.py::TestPPS` | 6 unit tests                                                                         |


### Hyper-params


| Tên       | Mặc định | Khi nào đổi                        |
| --------- | -------- | ---------------------------------- |
| `sal.tau` | 80       | Nhỏ hơn → PE / Hard-PE bắn sớm hơn |


---

## 4. AdaptiveRankSampler

### Vấn đề

Cũ dùng **softmax-temperature sampling** trên score. Phải fan-out
4 temperature → 4 bandit arms → posterior chia 4 lần. Lại thêm cảm giác
"nhồi nhét quá nhiều knob".

### Công thức mới — Zipfian rank sampling

```text
xếp các cell theo primary_score giảm dần
rank r(c)   = vị trí 0-based
β(t)        = max(β_min, β_max · (1 - s(t)))      # phụ thuộc stagnation
P(c) ∝ (r(c)+1)^(-β(t))
```

- `β` cao (= s thấp) → top rank gần như chắc chắn.
- `β` thấp (= s cao) → phân phối phẳng (gần uniform).

### Tại sao đẹp paper-wise?

- **Score-scale-invariant**: chỉ dùng rank, không dùng raw score → không
collapse khi gap quá lớn.
- **Một knob thôi**: `β` được derive từ `s(t)` (PPS đã viết) — không phải
là arm dimension. Bandit chỉ còn (model, prompt_id, llm_temperature).
- **Tổng quát**: β→0 đồng nghĩa uniform, β→∞ đồng nghĩa argmax. Một
sampler covers cả hai ends của explore/exploit spectrum.

### Code


| File                                                            | Nội dung                                                |
| --------------------------------------------------------------- | ------------------------------------------------------- |
| `levi/levi/pool/cvt_map_elites.py`                              | `AdaptiveRankSampler` class                             |
| `levi/levi/config/models.py`                                    | `SamplerModelPair.sampler = "adaptive_rank"` (mặc định) |
| `levi/tests/test_pps_adaptive_rank.py::TestAdaptiveRankSampler` | 5 unit tests                                            |


### Hyper-params


| Tên                            | Mặc định | Ý nghĩa                 |
| ------------------------------ | -------- | ----------------------- |
| `AdaptiveRankSampler.beta_max` | 2.0      | β khi s=0 (rất exploit) |
| `AdaptiveRankSampler.beta_min` | 0.2      | β khi s=1 (gần uniform) |


---

## 5. Phase-based PE prompts + Strategy History

### Vấn đề

Trước, PE luôn dùng một prompt "paradigm shift" duy nhất, không quan tâm
đến giai đoạn của run. Heavy model cũng không biết những paradigm trước
đã thử cái gì → đôi khi đề xuất lại cái đã fail.

### Giải pháp

1. **Three-stage prompts** dispatch theo `s(t)`:
  - `early` (s thấp): "PARADIGM SHIFT — pick a paradigm class NOT
   represented in archive."
  - `mid` (s trung): "SYNTHESIZE — borrow strengths from each region,
  fix one weakness."
  - `late` (s cao): "TARGETED REFINEMENT — surgical fix on the best
  solution; do not rewrite."
   Mỗi prompt được viết riêng phù hợp giai đoạn, không phải "copy paste +
   đổi tiêu đề". Xem `levi/levi/equilibrium/prompts.py`.
2. **Strategy History** (cơ chế MỚI):
  - Sau MỖI PE event, dùng **light model** (rẻ) tóm tắt paradigm-shift
    code thành **đúng hai dòng**:
    - `IDEA:` — một câu nêu họ thuật toán + tactic đặc trưng nó dùng
      cho bài toán hiện tại (vd. "Simulated annealing on overlap-penalty
      energy with logarithmic cooling and radius perturbation").
    - `QUALITY:` — một câu chỉ ra nơi ý tưởng này có khả năng hoạt động
      tốt và nơi nó dễ mất điểm trên *bài toán hiện tại* (vd. "Mạnh khi
      số hình tròn nhỏ và radius đa dạng; yếu khi nhiều hình đồng kích
      thước vì bị mắc kẹt ở cấu hình đối xứng").
  - Tổng output ≤ 60 từ. Mục tiêu duy nhất: heavy model biết "đã thử
    gì rồi, chất lượng ra sao" — không phân tích sâu hơn thế.
  - Light summariser nhận được đầy đủ **problem description + function
    signature + code**, nên `QUALITY` được nói theo ngữ cảnh bài toán
    cụ thể, không phải nhận xét chung chung về thuật toán.
  - Lưu vào `state.strategy_history: deque[StrategyRecord]` (maxlen=12).
    Mỗi record có `pe_event_id, stage, summary, best_before,
    paradigm_score, delta_score, accepted`.
  - PE event tiếp theo: render 8 records gần nhất vào prompt của heavy
    model dưới section `## Strategy Log (already tried in this run)`.
    Heavy model thấy header `### PE #N [stage] — Δ=…, score=…, accepted`
    và body 2-dòng thụt vào dưới. Có chỉ dẫn:
    > "Do NOT propose an approach whose summary appears above with Δ ≤ 0."

### Tại sao paper-wise?

- **Memory-augmented optimization**: heavy model có "long-term memory"
giá rẻ — chỉ ~60 từ light-model output mỗi PE.
- **Anti-loop**: tự nhiên tránh re-trying paradigm đã fail, không cần
hard-coded blacklist.
- **Trace-able**: paper có thể plot `strategy_history` để chứng minh
heavy model thực sự diverse hoá thay vì spam.

### Code


| File                                   | Nội dung                                                                               |
| -------------------------------------- | -------------------------------------------------------------------------------------- |
| `levi/levi/equilibrium/prompts.py`     | `PARADIGM_SHIFT_PROMPTS` (3 stage), `STRATEGY_SUMMARY_PROMPT`                          |
| `levi/levi/equilibrium/equilibrium.py` | `_summarize_strategy`, append record cuối `trigger()`                                  |
| `levi/levi/pipeline/state.py`          | `StrategyRecord`, `strategy_history`, `format_strategy_log`                            |
| `levi/levi/artifacts/code.py`          | `build_paradigm_shift_prompt(strategy_log_block=...)`, `build_strategy_summary_prompt` |
| `levi/tests/test_strategy_log.py`      | 6 unit tests                                                                           |


### Hyper-params


| Tên                                   | Mặc định | Ý nghĩa                                |
| ------------------------------------- | -------- | -------------------------------------- |
| `strategy_log.enabled`                | True     | Tắt → heavy không thấy log             |
| `strategy_log.max_entries`            | 8        | Render bao nhiêu record gần nhất       |
| `strategy_log.summariser_max_tokens`  | 150      | Đủ cho 2-dòng IDEA/QUALITY (~60 từ)    |
| `strategy_log.summariser_temperature` | 0.2      | Thấp để summary deterministic          |


---

## 6. Code Error Repair — one-shot light-model fix

### Vấn đề

Mỗi candidate bị error (syntax, runtime, score parse) bị huỷ ngay.
Đôi khi parent là elite rất mạnh, lỗi chỉ là 1 typo. Vứt đi → lãng phí
LLM call.

### Giải pháp

1. **Bounded buffer**: `state.error_buffer: deque[ErrorRecord]` (maxlen=64).
  Mỗi record giữ `code, parent_score, error_msg, parent_cell, source`.
2. **Consumer push**: khi `eval_result` có `error` hoặc `score_error`,
  push vào buffer (dedupe theo prefix 200 char). Bỏ qua nếu chính
   item này đã là repair (one-shot).
3. **Producer poll**: mỗi `repair_every_n=8` evaluations trong main loop,
  producer thay vì sample parent bình thường, gọi `state.fire_repair_if_due`:
  - Chọn **một** record theo phân phối **Zipfian rank-by-parent-score**
  (giống AdaptiveRank, β=1.5).
  - Gọi light model với `CODE_REPAIR_PROMPT` (chứa code lỗi + error_msg).
  - Đặt candidate trả về vào code_queue với `is_repair=True`,
  `sampler="repair"`.
4. **Single-attempt**: record đã pop khỏi buffer ngay khi sample. Repair
  fail → bỏ luôn, không retry.

### Tại sao paper-wise?

- **Cost-aware**: repair chỉ dùng light model (cheap). Phục hồi parent
mạnh → giá trị / cost cao.
- **Rank-by-parent-score**: ưu tiên cứu lỗi của parent gần elite (analogy
với AdaptiveRank — cùng family).
- **Sane bound**: `max_per_run=100` chống burst burnout.

### Code


| File                               | Nội dung                                                                            |
| ---------------------------------- | ----------------------------------------------------------------------------------- |
| `levi/levi/pipeline/state.py`      | `ErrorRecord`, `push_error_record`, `sample_error_for_repair`, `fire_repair_if_due` |
| `levi/levi/pipeline/producer.py`   | Repair branch ở đầu mỗi iteration                                                   |
| `levi/levi/pipeline/consumer.py`   | `_maybe_push_error` ở 2 error paths                                                 |
| `levi/levi/equilibrium/prompts.py` | `CODE_REPAIR_PROMPT`                                                                |
| `levi/levi/artifacts/code.py`      | `build_code_repair_prompt`                                                          |
| `levi/tests/test_code_repair.py`   | 10 unit tests                                                                       |


### Hyper-params


| Tên                          | Mặc định | Khi nào đổi                   |
| ---------------------------- | -------- | ----------------------------- |
| `code_repair.repair_every_n` | 8        | Nhỏ hơn → repair nhiều hơn    |
| `code_repair.beta`           | 1.5      | Cao hơn → ưu tiên parent mạnh |
| `code_repair.buffer_size`    | 64       | Lớn → giữ nhiều lỗi cũ        |
| `code_repair.max_per_run`    | 100      | Cap tổng                      |
| `code_repair.max_tokens`     | 4000     | Output cap light-model        |


---

## 7. Adaptive Island Expansion (AdaEvolve-style)

### Vấn đề

Hai bài toán đan xen ở CVT-MAP-Elites cố định:

1. Khi `s(t)` ≈ 1, heavy model có thể sinh paradigm shift điểm gần
  incumbent (vd. 0.95 vs 1.00) nhưng không qua được rule "strict better"
   → vứt đi, mất cơ hội thoát local optimum.
2. Archive tessellate **một lần** khi init rồi giữ cố định 50 cells. Khi
  mọi cell đã có elite + s(t) cao → mọi behavior space "có sẵn" đã được
   khám phá; mutation tiếp theo chỉ là cải tiến biên.

Một cơ chế duy nhất giải cả hai.

### Giải pháp — open a new island

Khi PE candidate (paradigm hoặc variant) **không** vượt incumbent của cell
gần nhất, gọi `_try_island_expansion`. Điều kiện:

```text
s(t)                         ≥ stagnation_threshold (0.7)
state.island_expansion_count < max_per_run (16)
pool.n_centroids             < max_total_centroids (200)
```

Hành động:

1. Tính `vec = pool._behavior_to_normalized_vector(behavior)` cho
  candidate.
2. `pool.add_as_new_cell(program, eval_result, behavior)`: append
  `vec` vào `pool._centroids` → cell mới với index `n_centroids - 1`,  
   sau đó seed cell đó với candidate.
3. Tăng `state.island_expansion_count` lên 1.

Sau bước này:

- Cell cũ (mà candidate ban đầu thuộc về) GIỮ NGUYÊN elite — không bị
thay thế.
- Cell mới là "đảo" cho candidate đó, sẵn sàng đón mutation tương lai
có behavior gần.

### Tại sao paper-wise?

- **Tương tự AdaEvolve** ở ý tưởng dynamic archive sizing, nhưng đơn giản
hơn nhiều: không cần re-cluster, không cần buffer behaviour vectors,
không cần cooldown / saturation gate.
- **Một cơ chế thay hai**: hồi trước cần (rescue threshold + adaptive-CVT
growth) → giờ chỉ một trigger duy nhất. Conceptual surface area giảm.
- **Non-destructive**: incumbent cũ luôn được giữ → archive không bị xói
mòn ngay cả khi rescue candidate sau này tệ hơn.
- **Trigger có nguyên lý**: dùng `s(t)` (đã principled từ PPS) — không
cần threshold ad-hoc thứ hai.
- **Hard-bounded growth**: `max_per_run=16` + `max_total_centroids=200`
→ cost / archive size đều có ceiling.

### Code


| File                                   | Nội dung                                                    |
| -------------------------------------- | ----------------------------------------------------------- |
| `levi/levi/pool/cvt_map_elites.py`     | `add_as_new_cell(program, result, behavior)`                |
| `levi/levi/equilibrium/equilibrium.py` | `_try_island_expansion(...)` + call sites trong `trigger()` |
| `levi/levi/pipeline/state.py`          | `island_expansion_count` counter                            |
| `levi/levi/config/models.py`           | `AdaptiveIslandConfig`                                      |
| `levi/tests/test_adaptive_island.py`   | 10 unit tests (primitive + gate + integration)              |


### Hyper-params


| Tên                                    | Mặc định | Ý nghĩa                         |
| -------------------------------------- | -------- | ------------------------------- |
| `adaptive_island.stagnation_threshold` | 0.7      | s(t) tối thiểu để expansion bắn |
| `adaptive_island.max_per_run`          | 16       | Số expansion tối đa mỗi run     |
| `adaptive_island.max_total_centroids`  | 200      | Hard ceiling tuyệt đối          |


---

## 9. Code map — chỗ nào cần tune?

Tất cả config có default hợp lý. Khi cần thí nghiệm, đây là chỗ chỉnh:


| File                                   | Class / function                                                                       | Khi nào sửa?                                                 |
| -------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `levi/levi/config/models.py`           | `SalConfig.tau`, `StrategyLogConfig.`*, `CodeRepairConfig.*`, `AdaptiveIslandConfig.*` | Tinh chỉnh hyper-params, ablation                            |
| `levi/levi/pipeline/state.py`          | `stagnation_depth(tau)`, `format_strategy_log`, `fire_repair_if_due`                   | Đổi công thức PPS / cách render strategy log / repair gating |
| `levi/levi/pool/cvt_map_elites.py`     | `AdaptiveRankSampler._compute_beta`, `add_as_new_cell`                                 | Đổi shape của β(s) hoặc cách mở island mới                   |
| `levi/levi/equilibrium/prompts.py`     | 3 paradigm prompts, `STRATEGY_SUMMARY_PROMPT`, `CODE_REPAIR_PROMPT`                    | Sửa wording của heavy/light prompts                          |
| `levi/levi/equilibrium/equilibrium.py` | `_try_island_expansion`, `_summarize_strategy`                                         | Đổi logic 2 cơ chế PE-side                                   |
| `levi/levi/pipeline/producer.py`       | Repair branch + `fire_repair_if_due` call                                              | Đổi cách chọn parent / repair scheduling                     |
| `levi/levi/pipeline/consumer.py`       | `_maybe_push_error`                                                                    | Đổi điều kiện đẩy error vào buffer                           |


---

## 10. Cách chạy

### Local (Python ≥ 3.12, đã `uv sync` trong `./levi`)

```bash
# Full run với mọi cải tiến bật mặc định
uv run python scripts/run_levi.py \
    --example-dir levi/examples/circle_packing \
    --evals 200 \
    --small-model "openrouter/qwen/qwen3-30b-a3b-instruct-2507" \
    --large-model "openrouter/openai/gpt-5"
```

### Ablation runs (tắt từng cơ chế)

```bash
# Baseline (chỉ PE phase-based + AdaptiveRank + PPS, tắt 3 cơ chế mới)
uv run python scripts/run_levi.py --example-dir levi/examples/circle_packing \
    --evals 200 \
    --no-strategy-log --no-code-repair --no-adaptive-island

# Chỉ Strategy Log
uv run python scripts/run_levi.py --example-dir levi/examples/circle_packing \
    --evals 200 --no-code-repair --no-adaptive-island

# Chỉ Code Repair
uv run python scripts/run_levi.py --example-dir levi/examples/circle_packing \
    --evals 200 --no-strategy-log --no-adaptive-island

# Chỉ Adaptive Island Expansion
uv run python scripts/run_levi.py --example-dir levi/examples/circle_packing \
    --evals 200 --no-strategy-log --no-code-repair
```

### Hyper-param sweep ví dụ

```bash
# Quét stagnation threshold của Adaptive Island Expansion
for S in 0.5 0.6 0.7 0.8 0.9; do
    uv run python scripts/run_levi.py --example-dir levi/examples/circle_packing \
        --evals 200 --island-stagnation $S \
        --output-dir outputs/sweep/island_s_$S
done

# Quét max_per_run của island
for K in 4 8 16 32; do
    uv run python scripts/run_levi.py --example-dir levi/examples/circle_packing \
        --evals 200 --island-max $K \
        --output-dir outputs/sweep/island_max_$K
done

# Quét beta của code repair
for B in 1.0 1.5 2.0 3.0; do
    uv run python scripts/run_levi.py --example-dir levi/examples/circle_packing \
        --evals 200 --code-repair-beta $B \
        --output-dir outputs/sweep/repair_beta_$B
done
```

### GitHub Actions (CI)

`.github/workflows/_levi.yml` đã expose tất cả flags trên thành
workflow_dispatch inputs. Chạy từ tab Actions → "LEVI (reusable)" → "Run
workflow", điền `example_dir` và tuỳ chọn knobs (mọi knob để trống = giữ
default).

---

## 11. Kế hoạch ablation cho paper

Một bảng đề xuất tối thiểu để rơi ra paper:


| Run | PE phase | Strategy Log | Code Repair | Adaptive Island | Mục tiêu                                      |
| --- | -------- | ------------ | ----------- | --------------- | --------------------------------------------- |
| A   | ✓        | ×            | ×           | ×               | Baseline mới (chỉ phase + AdaptiveRank + PPS) |
| B   | ✓        | ✓            | ×           | ×               | + Strategy Log                                |
| C   | ✓        | ✓            | ✓           | ×               | + Code Repair                                 |
| D   | ✓        | ✓            | ✓           | ✓               | Full (paper headline)                         |


Mỗi run nên chạy ≥ 3 seed (random), report mean ± std.

Thêm 2 sweep nhỏ:

- `island.stagnation_threshold` ∈ {0.5, 0.6, 0.7, 0.8, 0.9} với `max_per_run=16`.
- `code_repair.beta` ∈ {1.0, 1.5, 2.0, 3.0} với `repair_every_n=8`.

---

## 12. Paper hooks

### Title gợi ý

> **LEVI-X: Adaptive Island Expansion and Posterior-Plateau Stagnation
> for LLM-Driven Code Evolution**

### Abstract draft (cô đọng)

> We extend MAP-Elites code evolution along four axes: a survival-style
> Posterior-Plateau Stagnation signal (PPS) that calibrates "stuck"
> against the empirical NEW-BEST hazard rate; a parameter-free Zipfian
> rank sampler that collapses the explore/exploit knob into a single
> stagnation-driven exponent; a structured strategy-history log that
> lets each heavy paradigm-shift LLM call explicitly avoid retracing
> past attempts and is paired with a one-shot light-model repair
> sub-pipeline for broken offspring; and an Adaptive Island Expansion
> mechanism that opens a brand-new behaviour cell whenever a
> stagnation-era paradigm shift fails standard admission — unifying
> stagnation rescue and archive growth into a single non-destructive
> trigger. On circle packing we observe …

### Related work hooks

- **AdaEvolve** — chia sẻ idea dynamic archive sizing; Adaptive Island
Expansion là dạng tối giản: không re-cluster, không buffer behaviour
vectors, mỗi PE candidate tự định nghĩa centroid mới của mình.
- **MAP-Elites + Stagnation triggers** (FunSearch, AlphaEvolve etc.) —
PPS đẹp hơn nhờ Poisson hazard estimate.
- **Memory-augmented prompting** — Strategy Log analog với recent papers
về "scratchpad memory" cho LLM optimization, nhưng output có cấu trúc
4-label cố định giúp downstream LLM parse ổn định.

---

## 13. FAQ

**Q. Có còn HLS (Heavy-Light Synthesis) trong code không?**
Không. Đã gỡ hoàn toàn (toàn bộ blueprint-related code đã bị remove).

**Q. Strategy summariser dùng model nào mặc định?**
`mutation_models[0]` — model rẻ nhất bạn đã cấu hình. Có thể override
qua `StrategyLogConfig.summariser_model`.

**Q. Tại sao `summariser_max_tokens` là 150?**
Output mới chỉ có 2 dòng `IDEA` + `QUALITY` (≤ 60 từ). 150 tokens cho
phép một biên độ an toàn nhỏ cho dấu câu / từ dài mà vẫn rẻ. Trước đây
prompt sinh post-mortem 4-label dài hơn nên cần 400; phiên bản mới đã
gọn nên kéo xuống.

**Q. Khi `code_repair_max=100`, có khả năng quá ngân sách không?**
Một repair call ≈ một mutation call → 100 repair ≈ 100 mutation. Light
model nên rẻ. Nếu muốn cap chặt hơn, hạ thành 50 hoặc 30.

**Q. Adaptive Island Expansion có thể làm hỏng archive không?**
Không — incumbent của cell cũ luôn được GIỮ. Cell mới được mở ra ở
behavior vector của candidate đó, độc lập với cell cũ. Trong trường hợp
xấu nhất (ngân sách hết) archive chỉ đơn giản có thêm vài cell điểm
thấp; các cell cũ vẫn nguyên.

**Q. `n_centroids` có thay đổi qua lại không?**
Chỉ tăng, không bao giờ giảm. Hard cap `max_total_centroids=200`.

**Q. Test cho 3 cơ chế mới ở đâu?**

```text
levi/tests/test_strategy_log.py        # 7 tests
levi/tests/test_code_repair.py         # 10 tests
levi/tests/test_adaptive_island.py     # 10 tests
```

Tổng 27 tests; chạy:

```bash
cd levi && python -m pytest tests/test_strategy_log.py tests/test_code_repair.py \
    tests/test_adaptive_island.py -q
```

PPS + AdaptiveRank giữ test file riêng:
`levi/tests/test_pps_adaptive_rank.py` (11 tests).

---

## 14. Mở rộng tương lai (ngoài paper hiện tại)

- **Strategy Log → vector memory**: thay vì plaintext, embed mỗi
4-label summary và tìm similar khi prompt PE → memory dày đặc hơn.
- **Adaptive Island bidirectional**: cho phép *merge* hai cell quá gần
hoặc *evict* cell trống lâu → archive thực sự thích ứng cả hai chiều.
- **Code Repair self-distillation**: log mỗi repair thành công + lỗi gốc
→ fine-tune light model trên dataset đó.

