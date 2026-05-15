# Prompt Bank Architecture — Joint Bandit trên (sampler, model, prompt, temperature)

> Tài liệu này mô tả cơ chế **Prompt Bank** mới của Levi: mở rộng Thompson Beta-Bernoulli bandit hiện có sang không gian arm bốn chiều `(sampler, model, mutation_prompt_id, llm_temperature)`, nhằm khai thác đa dạng prompt + nhiệt độ LLM thay vì chỉ đa dạng (sampler, model) như bản gốc.
>
> Lý do tồn tại: khi pipeline mutation chạy nhiều `mutation_models` con với CÙNG một prompt và CÙNG một temperature, output của các model dễ "na ná" nhau → diversity thấp, archive bão hòa sớm. Prompt Bank phá đối xứng đó bằng cách ngẫu nhiên hoá cả prompt lẫn temperature ở mỗi lần sample, đồng thời để Thompson bandit dần dồn ngân sách cho những bộ `(prompt, temperature)` thực sự sinh ra offspring được accept và new-best.

---

## 1. Tóm tắt chiến thuật

1. **Bank prompt mutation** là một file JSON do người dùng nắm: `[ {"id": str, "text": str}, ... ]`. Mỗi entry là MỘT prompt template *hoàn chỉnh* — cùng mục đích (cải tiến parent), khác cách phát biểu.
2. **Bank temperature** là một file JSON: `[float, float, ...]`. Mỗi giá trị là một mức temperature dùng khi gọi LLM.
3. Khi bank được bật, Levi **cross-product expand** các sampler-model gốc với toàn bộ bank:
   - Trước:  `(sampler, model)` — bandit 2 chiều.
   - Sau:    `(sampler, model, prompt_id, llm_temperature)` — bandit 4 chiều.
   - Mỗi tổ hợp là một **arm độc lập**, có riêng cặp posterior `(α, β)` và `new_best_count`.
4. Mỗi lần worker sampler chọn một arm theo Thompson Beta-Bernoulli (đã có sẵn cho SAL Cơ chế D). Arm đó quyết định:
   - **Sampler nào** lấy parent từ CVT-MAP-Elites.
   - **Model nào** sinh code.
   - **Prompt nào** trong bank được dùng để dựng nội dung gửi LLM.
   - **Temperature nào** đưa vào `acompletion(...)`.
5. Sau khi offspring được evaluate, `update_bandit(...)` cập nhật α/β/new_best của ĐÚNG arm đó. Arm tốt sẽ dần được rút thường xuyên hơn — đặc biệt khi stagnation `s(t)` cao, công thức reweight `(1 + γ·nb)^(1+s)` khuếch đại ưu thế new-best.
6. `bandit_w_min` floor (≥ 0.05 mặc định) đảm bảo arm tệ vẫn còn xác suất tối thiểu để được thử → không collapse, vẫn có thám hiểm.

Hiệu quả mong đợi:

- **Diversity nguồn**: hai LLM khác nhau với hai prompt khác nhau và hai temperature khác nhau khó sinh ra code giống nhau hơn nhiều so với cùng (prompt, temp).
- **Khám phá có dẫn dắt**: bandit tự "học" cặp `(prompt, temp)` nào hợp với problem cụ thể, thay vì cố định bằng tay.
- **Compatibility**: cơ chế cũ vẫn chạy y nguyên khi `prompt_bank.enabled=False` (default).

---

## 2. Vị trí cắm vào pipeline — không xoá gì, chỉ thêm nhánh

### 2.1. Bức tranh tổng quan

```
                   ┌────────────────────────────────┐
                   │ LeviConfig._auto_wire_models() │
                   │  - Default arms:               │
                   │      (sampler × model × T)     │
                   │      4 arms                    │
                   │  - If prompt_bank.enabled:     │
                   │      EXPAND to                 │
                   │      (sampler × model × T      │
                   │       × prompt × llm_T)        │
                   │      N×M×P×T arms              │
                   └─────────────┬──────────────────┘
                                 │
                                 ▼
                ┌──────────────────────────────────┐
                │ CVTMAPElitesPool                 │
                │  register_sampler_model_pair()   │
                │  → mỗi arm là 1 SamplerModelCfg  │
                │    (α=1, β=1, new_best=0,        │
                │     prompt_id, llm_temperature)  │
                └─────────────┬────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────────────┐
        │ Producer worker (mỗi lần lặp)                        │
        │  1. arm = pool.get_weighted_sampler_config(s(t))    │
        │     → (sampler, model, prompt_id, llm_temperature)  │
        │  2. sample = pool.sample(sampler, ...)              │
        │  3. NẾU prompt_id ≠ None:                            │
        │       prompt = adapter.build_mutation_prompt_       │
        │                  from_template(parents, bank[pid])  │
        │     KHÁC:                                            │
        │       prompt = adapter.build_mutation_prompt(...)   │
        │  4. response = state.acompletion(                    │
        │         model,                                       │
        │         temperature = llm_temperature ?? default,    │
        │         ...)                                         │
        │  5. queue.put({content, arm_metadata})              │
        └─────────────┬───────────────────────────────────────┘
                      │
                      ▼
        ┌───────────────────────────────────────────────┐
        │ Consumer worker                                │
        │  1. evaluate code                              │
        │  2. pool.add() → accepted / rejected           │
        │  3. pool.update_bandit(                        │
        │        sampler, model,                         │
        │        accepted=…, is_new_best=…,              │
        │        mutation_prompt_id=…,                   │
        │        llm_temperature=…)                      │
        │     → ĐÚNG arm 4-chiều được cập nhật α/β/nb    │
        └───────────────────────────────────────────────┘
```

### 2.2. File-by-file chi tiết

| File | Vị trí | Vai trò mới | Cái cũ có bị xoá? |
|---|---|---|---|
| `levi/levi/config/models.py` | `PromptBankConfig`, `SamplerModelPair` | Định nghĩa schema bank + 2 trường mới `mutation_prompt_id`, `llm_temperature` trên arm. `_auto_wire_models()` cross-product expand khi bank bật. | **Không.** `SamplerModelPair` cũ vẫn tương thích: 2 trường mới mặc định `None`. Khi `prompt_bank.enabled=False`, đường auto-gen 4 cặp `softmax_T` cũ chạy y nguyên. |
| `levi/levi/config/__init__.py` | Export `PromptBankConfig` | Cho người dùng import từ `levi.config`. | Không xoá symbol nào. |
| `levi/levi/pool/cvt_map_elites.py` | `SamplerModelConfig`, `register_sampler_model_pair`, `get_weighted_sampler_config`, `update_bandit`, `get_bandit_stats` | Mỗi arm lưu thêm `mutation_prompt_id` + `llm_temperature`. `get_weighted_sampler_config` trả 4-tuple. `update_bandit` match theo cả 4 chiều (so sánh float với tolerance 1e-9). | **Không.** Khi cả hai trường mới là `None` ở cả arm và call site, hành vi quy về cũ. Thompson bandit logic (`bandit_w_min`, `(1+γ·nb)^(1+s)`) không đổi. |
| `levi/levi/artifacts/code.py` | `build_mutation_prompt_from_template(parents, template, ...)` | Phương thức MỚI: format template bằng `str.format_map` với `_SafeFmt` (missing placeholder → ""). | **Không.** `build_mutation_prompt(...)` cũ vẫn còn nguyên — producer chỉ chuyển nhánh khi arm có `mutation_prompt_id`. |
| `levi/levi/pipeline/producer.py` | `llm_producer(...)` | Lấy 4-tuple từ pool, load bank registry 1 lần, chọn nhánh prompt theo có/không `mutation_prompt_id`, override temperature từ arm, push arm metadata vào queue. | **Không.** Logic dựng `mutation_kwargs` cũ giữ nguyên; nhánh mới là phân nhánh có điều kiện. |
| `levi/levi/pipeline/consumer.py` | 4 call sites của `update_bandit` | Forward `mutation_prompt_id` + `llm_temperature` từ `item` payload. | **Không.** Khi `item` không có 2 khoá đó (`.get()` trả `None`), match-rule rơi về cũ. |
| `levi/levi/methods/levi.py` | `_run_async`, vòng register arm | Skip DSPy `prompt_opt` (với warning) khi bank bật. Register arm với 2 trường mới. | **Không xoá `prompt_opt`.** Chỉ bỏ qua khi đụng độ slot `## Output`. Người dùng có thể tắt bank, bật prompt_opt như cũ. |
| `levi/tests/test_sal.py` | Sửa unpack `name, model, _, _` | Thích nghi return-type mới của `get_weighted_sampler_config`. | Test logic không đổi. |
| `levi/examples/mutation_prompts.json` | MỚI | 5 prompt template dùng chung cho mọi example (đặt ở `examples/`, không bind riêng circle_packing). | — |
| `levi/examples/mutation_temperatures.json` | MỚI | `[0.5, 0.8, 1.1]`. Shared default cho tất cả example. | — |

### 2.3. Prompt cũ có bị xoá không?

**KHÔNG.** Cụ thể từng loại "prompt cũ":

1. **Mutation prompt mặc định trong `CodeAdapter.build_mutation_prompt(...)`**:
   Vẫn còn nguyên, vẫn dùng `PromptBuilder` + `OutputMode.FULL/DIFF`. Khi `prompt_bank.enabled=False` (default) hoặc khi một arm bất kỳ trong bank có `mutation_prompt_id=None`, producer rơi vào nhánh này.

2. **Override từ DSPy MIPROv2 (`config.prompt_overrides["mutation"]`)**:
   Code vẫn còn, vẫn được `build_mutation_prompt` đọc qua `set_custom_output(...)`. Chỉ bị **bỏ qua TẠM THỜI** ở runtime khi cả `prompt_opt.enabled=True` *và* `prompt_bank.enabled=True` — vì hai cơ chế tranh slot `## Output`. Levi log warning và chạy với bank. Nếu bạn tắt bank, prompt_opt sẽ chạy lại y nguyên.

3. **Paradigm-shift prompt (`PARADIGM_SHIFT_PROMPTS`, SAL Cơ chế A staging)**:
   Hoàn toàn không đụng tới. Bank chỉ thay phần mutation thường (nhánh `llm_producer`), không can thiệp pipeline `PunctuatedEquilibrium`.

4. **Init / diversity prompt, variant prompt, meta-advice prompt**:
   Không đụng tới.

Triết lý: bank là **đường thay thế tuỳ chọn cho slot mutation prompt**, không phải refactor cố định. Tắt cờ là pipeline chạy lại y như trước.

---

## 3. Chi tiết bandit — cách arm "tốt" được dùng nhiều

### 3.1. State của một arm

```python
@dataclass
class SamplerModelConfig:
    sampler_name: str
    model: ClientSpec
    weight: float = 1.0

    # MỚI — joint bandit dimensions
    mutation_prompt_id: Optional[str] = None
    llm_temperature:   Optional[float] = None

    # Thompson Beta-Bernoulli state (đã có sẵn cho SAL D)
    alpha: float = 1.0            # +1 mỗi lần offspring được accept
    beta:  float = 1.0            # +1 mỗi lần offspring bị reject / cascade-skip / error
    new_best_count: int = 0       # +1 mỗi lần offspring là NEW BEST
```

Tổng arm khi bank bật: `N_sampler × N_model × N_prompt × N_temperature`. Ví dụ trong demo circle_packing: 1 × 1 × 5 × 3 × 4 cặp softmax_T = 60 arms. Khuyến nghị giữ N_prompt ≤ 5, N_temperature ≤ 4 để bandit có đủ mẫu hội tụ trong vài trăm evals.

### 3.2. Rút arm theo Thompson — công thức nguyên thuỷ của SAL D, không sửa

Mỗi lần `get_weighted_sampler_config(stagnation=s(t))`:

1. Với mỗi arm `i`:
   ```
   θ_i ~ Beta(α_i, β_i)                        # posterior sample
   bonus_i = (1 + γ · new_best_count_i) ^ (1+s)
   raw_i   = θ_i · bonus_i
   ```
   `γ = bandit_new_best_gamma` (default 0.5). `s = stagnation_depth ∈ [0,1]`.

2. Chuẩn hoá: `p_i = raw_i / Σ raw_j`.

3. Floor + remix:
   ```
   w_i = w_min + (1 − N · w_min) · p_i
   ```
   `w_min = bandit_w_min` (default 0.05). Đảm bảo `w_i ≥ w_min`, `Σ w_i = 1`.

4. `idx = np.random.choice(N, p=w)` → chọn arm.

5. Trả về `(sampler_name, model, mutation_prompt_id, llm_temperature)`.

Ý nghĩa với prompt bank:

- Arm `(softmax_T0.3, qwen, surgical_local_refine, 0.5)` thắng nhiều → α tăng → θ tăng → được chọn nhiều hơn.
- Arm có lịch sử new-best (mỗi lần phá kỷ lục thì +1 `new_best_count`) được khuếch đại bởi số mũ `1+s` khi search bị stagnant — đẩy mạnh "double down" trên cặp đã từng break plateau.
- Arm tệ vẫn được rút với xác suất tối thiểu `w_min` → cơ chế thám hiểm sẵn có, không cần thêm code.

### 3.3. Cập nhật posterior

```python
pool.update_bandit(
    sampler_name = item["sampler"],
    model        = item["model"],
    mutation_prompt_id = item.get("mutation_prompt_id"),
    llm_temperature    = item.get("llm_temperature"),
    accepted = …,
    is_new_best = …,
)
```

Match-rule trong `update_bandit`:
- Tên sampler, model exact match.
- `mutation_prompt_id` exact (kể cả `None`).
- `llm_temperature`: nếu cả hai cùng `None` thì match; ngược lại `|a - b| ≤ 1e-9` (tolerance float).
- Không khớp → silently no-op (đúng tinh thần cũ: bundle adapter / paradigm-shift call không có arm tương ứng).

---

## 4. Định dạng template prompt — cái mà người dùng nắm

Mỗi entry trong `mutation_prompts.json`:

```json
{
  "id": "surgical_local_refine",
  "text": "# Mutation Task\n\n## Problem\n{problem_description}\n\n## Function Signature\n```python\n{function_signature}\n```\n\n{parents_block}\n\n{search_trajectory_block}\n\n{feedback_block}\n\n{meta_advice_block}\n\n## Your Task\nMake a SURGICAL improvement to v1...\n\n## Output\nReturn the COMPLETE improved file as a single ```python``` block ..."
}
```

### 4.1. Placeholders được hỗ trợ

`build_mutation_prompt_from_template` format template bằng `str.format_map` với `_SafeFmt` (missing key → `""`). Sáu placeholders chính thức:

| Placeholder | Giá trị | Khi nào rỗng |
|---|---|---|
| `{problem_description}` | `config.problem_description` | Không bao giờ rỗng. |
| `{function_signature}` | `config.function_signature` (không bọc ```python```) | Không bao giờ rỗng. |
| `{parents_block}` | Markdown `## v1 / v2 / v3` kèm score và code fence, ghép bằng `\n\n`. | Không bao giờ rỗng (luôn có v1). |
| `{search_trajectory_block}` | `## Search Trajectory` + bullet `Current best`, `Evaluations since last NEW BEST`, `Stagnation depth s(t)`, `Recurring failure modes`. | Rỗng khi SAL A.2 không bật hoặc không có dữ liệu. |
| `{feedback_block}` | `## Feedback` + bullet các failure feedback ngẫu nhiên từ parent. | Rỗng khi không có per-example feedback. |
| `{meta_advice_block}` | `## Meta-Advice` + text từ SAL Cơ chế C / meta-advisor. | Rỗng khi meta-advice tắt hoặc 20% bị bỏ ngẫu nhiên. |

Vì dùng `_SafeFmt`, bất kỳ placeholder lạ nào trong template cũng tự động thành `""`, không raise — an toàn cho ngày sau nếu bạn thêm placeholder riêng.

### 4.2. Mỗi entry là một FULL prompt — bao gồm cả Output

Đây là khác biệt cốt lõi so với `prompt_overrides` của DSPy (chỉ override mỗi block `## Output`). Lý do user chọn full template: mỗi entry trong bank có thể có cấu trúc section khác nhau, có thể bỏ qua block không cần, có thể thay đổi cả "Your Task" — cho diversity sâu hơn. Nhược điểm: phải tự viết phần Output rõ ràng (yêu cầu format code block + signature giữ nguyên), nếu quên LLM sẽ trả text loạn xạ và `extract_code` rỗng → offspring bị bỏ.

Đã có 5 ví dụ trong [examples/mutation_prompts.json](../levi/examples/mutation_prompts.json) bao quát các style: surgical refine, borrow from inspiration, aggressive rewrite, hyperparameter tuning, diversify initialization.

### 4.3. Temperature bank

File `mutation_temperatures.json` đơn giản: `[0.5, 0.8, 1.1]`. Khuyến nghị giữ range vừa phải — quá thấp (< 0.3) thì các arm gần như xác định, quá cao (> 1.5) thì code dễ vô nghĩa với code-LLM hiện tại.

Temperature trong bank là **LLM sampling temperature** (truyền vào `litellm.acompletion`), KHÁC `SamplerModelPair.temperature` cũ — cái cũ là tham số của `SoftmaxSampler` chọn cell trong CVT, không liên quan đến LLM. Cả hai vẫn cùng tồn tại trong một arm.

---

## 5. Kích hoạt — cần và đủ

```python
from levi.config import LeviConfig, BudgetConfig, PromptBankConfig

config = LeviConfig(
    problem_description="...",
    function_signature="def run_packing() -> tuple[np.ndarray, np.ndarray, float]:",
    seed_program="...",
    score_fn=score_fn,
    budget=BudgetConfig(seconds=10800),
    mutation_models=["openrouter/qwen/qwen3-30b-a3b-instruct-2507"],

    # MỚI — bật bank
    prompt_bank=PromptBankConfig(
        enabled=True,
        prompts_file="examples/mutation_prompts.json",
        temperatures_file="examples/mutation_temperatures.json",
        # replace_default_pairs=True (default): default auto-gen (4 softmax_T)
        # bị thay hoàn toàn bằng cross-product. Đặt False nếu muốn GHÉP THÊM.
    ),
)
```

Có thể inline thay vì file (tiện cho test):

```python
PromptBankConfig(
    enabled=True,
    prompts=[
        {"id": "v1", "text": "...{problem_description}..."},
        {"id": "v2", "text": "...{parents_block}..."},
    ],
    temperatures=[0.7, 1.0],
)
```

Khi bật bank, log đầu run sẽ in:

```
[Levi] Sampler-model pairs: 24       # ví dụ 4 base × 2 prompts × 3 temps
[Levi] prompt_bank is enabled — skipping DSPy prompt_opt to avoid override conflicts
```

---

## 6. Quan sát kết quả — đọc bandit stats

`CVTMAPElitesPool.get_bandit_stats()` (đã có sẵn, snapshot gọi cuối run) giờ trả thêm 2 trường:

```json
{
  "sampler": "softmax_T0.3",
  "model": "openrouter/qwen/qwen3-30b-a3b-instruct-2507",
  "mutation_prompt_id": "surgical_local_refine",
  "llm_temperature": 0.5,
  "alpha": 41.0,
  "beta": 9.0,
  "posterior_mean": 0.82,
  "new_best_count": 3
}
```

Cách dùng để viết paper:

- **Heatmap** `prompt_id × llm_temperature` với mean = `α/(α+β)` → "ô" nào sáng = cặp `(prompt, temp)` hiệu quả nhất.
- **Ablation**: chạy 3 cấu hình — bank-off (baseline cũ), bank-on-uniform (replace bandit bằng uniform), bank-on-thompson (bandit gốc) → đo new-best discovery rate.
- **Cumulative new-best per arm** theo thời gian → cho thấy bandit dồn ngân sách như thế nào.

---

## 7. Đối chiếu với SAL — bank kết hợp cơ chế nào?

| SAL cơ chế | Status với bank |
|---|---|
| **A. PE staging** | Không đụng — paradigm-shift dùng pipeline riêng. |
| **A.2. Trajectory ngữ cảnh trong mutation** | **Tương thích.** `{search_trajectory_block}` trong template tự nhận `best_score`, `evals_since_best`, `stagnation`, `top_failures`. |
| **B. Mutation context (best elite + far elite)** | **Tương thích.** Producer vẫn append extra inspirations vào parents, hiện ra trong `{parents_block}`. |
| **C. Meta-advice dual-mode** | **Tương thích.** `{meta_advice_block}` nhận text từ state.current_meta_advice. |
| **D. Thompson bandit** | **CHÍNH LÀ NỀN của bank.** Bank chỉ MỞ RỘNG arm space của cơ chế này từ 2D → 4D. Mọi tham số (`bandit_w_min`, `bandit_new_best_bonus`, prior `α=β=1`) tái dùng. |
| **E. Hard-PE** | Không đụng. |

---

## 8. Rủi ro và cách kiểm soát

| Rủi ro | Triệu chứng | Cách giảm |
|---|---|---|
| **Arm space bùng nổ** | 100+ arms, bandit không kịp hội tụ trong budget. | Giữ `N_prompt ≤ 5`, `N_temperature ≤ 4`. Bắt đầu nhỏ. |
| **Template viết sai placeholder** | LLM trả output không có `python` block → `extract_code` rỗng → silent skip. | Test placeholder bằng `build_mutation_prompt_from_template` ngoài run lớn. |
| **Tranh slot `## Output` giữa bank và DSPy** | Behaviour ambiguous nếu cả hai bật. | Levi tự skip prompt_opt, log warning. Muốn so sánh: chạy 2 run riêng. |
| **Float key mismatch khi update bandit** | Posterior không cập nhật, log không lỗi. | Tolerance 1e-9 đã được thêm. Tránh dùng số như `0.1 + 0.2` ở temperature; ưu tiên giá trị tròn từ file JSON. |
| **Arm tốt bị "starve" sớm do prior weak** | Một arm vô tình đầu run reject 3 lần → β=4 → khó comeback. | `bandit_w_min` floor đã giữ chỗ tối thiểu. Có thể tăng `bandit_alpha_prior` lên 2.0 nếu thực tế thấy collapse. |

---

## 9. Checklist tương thích ngược

- [x] `prompt_bank.enabled=False` (default) → `sampler_model_pairs` auto-gen y như cũ (4 cặp softmax_T per model).
- [x] `get_weighted_sampler_config` đổi return từ 2-tuple → 4-tuple. Đã sửa caller duy nhất (`producer.py`) và test (`test_sal.py`).
- [x] `update_bandit` thêm 2 kwargs optional, mặc định `None` → call site cũ vẫn chạy.
- [x] `build_mutation_prompt(...)` giữ nguyên signature.
- [x] `prompt_opt` không bị xoá; chỉ skip khi bank đồng thời bật.
- [x] 238 / 240 test pass (2 fail là pre-existing, đã verify bằng `git stash`).

---

## 10. File mới / file sửa — quick reference

**Mới**:
- `levi/examples/mutation_prompts.json` (shared default cho mọi example)
- `levi/examples/mutation_temperatures.json` (shared default cho mọi example)
- `docs/PROMPT_BANK_ARCHITECTURE.md` (tài liệu này)

**Sửa**:
- `levi/levi/config/models.py` — `PromptBankConfig`, `SamplerModelPair` (+2 trường), `_auto_wire_models` (cross-product).
- `levi/levi/config/__init__.py` — export `PromptBankConfig`.
- `levi/levi/pool/cvt_map_elites.py` — `SamplerModelConfig` (+2 trường), `register_sampler_model_pair` (+2 kwargs), `get_weighted_sampler_config` (return 4-tuple), `update_bandit` (match 4 chiều), `get_bandit_stats` (+2 fields).
- `levi/levi/artifacts/code.py` — thêm `build_mutation_prompt_from_template(...)`.
- `levi/levi/pipeline/producer.py` — load bank registry, chọn nhánh prompt, override temperature, push arm metadata.
- `levi/levi/pipeline/consumer.py` — 4 call sites `update_bandit` forward arm metadata.
- `levi/levi/methods/levi.py` — skip prompt_opt khi bank bật, register arm với 2 trường mới.
- `levi/tests/test_sal.py` — unpack 4-tuple.
