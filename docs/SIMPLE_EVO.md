# SIMPLE-EVO: A Simplified Successor to LEVI

> Design draft. Mục tiêu: giữ trọn điểm mạnh nhất của LEVI — **phối hợp song song giữa frontier model (lớn) và mutation model (nhỏ)** — đồng thời thay thế các thành phần đang quá phức tạp về toán (CVT-MAP-Elites, công thức PPS phụ thuộc new-best, bandit 4D) bằng các thành phần đơn giản hơn về mặt thuật toán nhưng đủ novel để claim đóng góp trong paper.

---

## 1. Nguyên tắc thiết kế

**P1 — Frontier ↔ Mutation collaboration là core, không đụng tới.** Đây là USP của LEVI: model lớn vẽ hướng, model nhỏ khai thác song song. SIMPLE-EVO chỉ đơn giản hoá *xung quanh* lõi này.

**P2 — Giữ nguyên các module đã hoạt động tốt trong LEVI**: error archive + one-shot repair ([state.py:489](../levi/levi/pipeline/state.py#L489), [producer.py:69-161](../levi/levi/pipeline/producer.py#L69-L161)); meta-advisor 2 mode defensive/offensive inject vào mutation prompt ([consumer.py:129-240](../levi/levi/pipeline/consumer.py#L129-L240)); 3 phase paradigm shift `early`/`mid`/`late` của frontier ([equilibrium/prompts.py:22-159](../levi/levi/equilibrium/prompts.py#L22-L159), route bằng `get_budget_stage`); async producer/consumer (PE non-blocking, mutation producers tiếp tục); variants generation song song từ paradigm shift code bằng *mutation* models qua `asyncio.gather()` ([equilibrium.py:916-919](../levi/levi/equilibrium/equilibrium.py#L916-L919)).

**P3 — Thay 3 thứ đang nặng nề**: Pool (CVT-MAP-Elites 1000 centroids, AST descriptor, Welford → top-K + cosine niching trên **embedding của text description**); Stagnation signal (công thức PPS phụ thuộc "new best" rate → 3 sliding-window signals, chỉ để route 3 phase và toggle operator mix, **không** dùng để trigger PE); Sampler/bandit (4D Thompson bandit → 2 operator mutate/crossover + selection có UCB-style novelty bonus).

**P4 — Frontier vẫn rẻ**: PE fire theo **cron mỗi N evals** như LEVI hiện tại — không event-driven. Monitor signals chỉ dùng để route 3 phase + toggle operator mix, không kéo thêm frontier call.

**P5 — Không thêm complexity mới**. Mọi thay đổi phải khiến code ngắn hơn, ít hyperparameter hơn, dễ ablate hơn.

---

## 2. Cái gì giữ nguyên từ LEVI

### 2.1 Error archive + self-repair (giữ nguyên 100%)

Đã có và hoạt động đúng. SIMPLE-EVO dùng lại không sửa:

| Component                 | File                 | Mô tả                                                                      |
| ------------------------- | -------------------- | -------------------------------------------------------------------------- |
| `ErrorRecord`             | `state.py:82-90`     | code, parent_score, error_msg (tail 4000 chars), parent_cell, source       |
| `error_buffer`            | `state.py:489`       | Deque maxlen=64, dedup by code prefix                                      |
| `_maybe_push_error`       | `consumer.py:71-104` | Push khi candidate fail và không phải repair-of-repair                     |
| `fire_repair_if_due`      | `state.py:906`       | Mỗi `repair_every_n` eval, max `max_per_run` lần                           |
| `sample_error_for_repair` | `state.py:943-980`   | Zipfian rank-by-parent-score, β=1.5                                        |
| Repair LLM call           | `producer.py:69-161` | **Mutation model** (light), one-shot. Fail lần 2 → discard, không re-queue |

Lý do giữ: đã đơn giản, đã chạy ổn, và là một advantage độc lập của LEVI mà nhiều EvoSearch khác không có.

### 2.2 Meta-advisor 2 mode (giữ kiến trúc, hiệu chỉnh trigger)

| Component                         | File                           | Mô tả                                                   |
| --------------------------------- | ------------------------------ | ------------------------------------------------------- |
| `META_ADVISOR_PROMPT` (defensive) | `consumer.py:129-163`          | Lessons từ failures. Focus prevention.                  |
| `META_ADVISOR_OFFENSIVE_PROMPT`   | `consumer.py:169-240`          | Strategic suggestions. Focus break-out.                 |
| `should_generate_meta_advice`     | `consumer.py:454-455`          | Trigger mỗi `meta_advice.interval` eval                 |
| `_generate_meta_advice`           | `consumer.py:555-616`          | Async task                                              |
| Mode switch                       | `consumer.py:564-573`          | Offensive khi `sal.enabled && s ≥ context_threshold`    |
| Injection                         | `producer.py:223-224, 287-290` | `state.current_meta_advice` → mutation prompt với p=0.8 |

**Thay đổi nhỏ trong SIMPLE-EVO**: điều kiện switch defensive ↔ offensive không dùng `stagnation depth s(t)` công thức PPS nữa, mà dùng `Monitor.is_stuck()` (xem §3.2). Prompt template và injection mechanism giữ nguyên.

Advisor vẫn dùng **mutation model** (config `meta_advice.model`, có thể trỏ tới model rẻ) — không tốn budget frontier.

### 2.3 Frontier 3 phase paradigm shift (giữ nguyên kiến trúc, đơn giản hoá routing)

LEVI đã có sẵn 3 prompt:

| Phase     | Trigger (LEVI hiện tại) | Tinh thần                                                       |
| --------- | ----------------------- | --------------------------------------------------------------- |
| **early** | `s(t) < 0.3`            | Algorithmic paradigm shift — chọn paradigm class chưa xuất hiện |
| **mid**   | `0.3 ≤ s(t) < 0.7`      | Synthesise mechanisms từ nhiều region, fix weaknesses           |
| **late**  | `s(t) ≥ 0.7`            | Surgical fix trên best solution, không rewrite                  |

3 prompt này đã rất tốt và là một trong những đóng góp quan trọng nhất của LEVI. **Giữ nguyên prompts.py:22-159**, **giữ nguyên `get_budget_stage`**.

**Thay đổi 1 — routing source**: không tính `s(t)` bằng công thức PPS phức tạp nữa, mà bằng `Monitor.stagnation_level()` trả về float [0,1] đơn giản dựa trên `plateau_steps / plateau_max`. Routing 3 phase không đổi.

**Thay đổi 2 — representatives theo phase (MAX 3, chỉ description + score, KHÔNG code)**:

Mỗi phase cần representatives khác nhau về tinh thần. Để frontier input ngắn và tránh "copy-paste paradigm", **chỉ inject description + score** (không inject code) cho representatives:

| Phase     | Cách chọn 3 representatives                                          | Tinh thần                                                           |
| --------- | -------------------------------------------------------------------- | ------------------------------------------------------------------- |
| **early** | 3 program cách xa nhau nhất về embedding (MMR λ=0.2, diversity-bias) | Frontier thấy paradigm class hiện có để né, sinh paradigm class mới |
| **mid**   | 1 top-score + 2 MMR-diverse (λ=0.5, balanced)                        | Frontier thấy best đang có + 2 góc nhìn khác để synthesise          |
| **late**  | Top-3 by score thuần (không MMR)                                     | Frontier surgical fix vào incumbent mạnh nhất                       |

**Thay đổi 3 — recent trials log (thay strategy log hiện tại của LEVI)**:

Mỗi lần PE accept một paradigm code, lưu `RecentTrial(description, score, delta_vs_prev_best, phase)`. Inject K=5 trials gần nhất vào frontier prompt qua slot `{strategy_log_block}` (đã có sẵn ở prompts.py:39). Format:

```text
[trial #12, mid] desc: "branch-and-bound with adaptive pruning"  score=0.72 (Δ=+0.03)
[trial #18, early] desc: "simulated annealing with custom neighborhood" score=0.58 (Δ=-0.11)
...
```

Không cần small model thống kê — description đã có sẵn từ output cũ (xem §3.1). Frontier nhìn thấy ý tưởng đã thử và kết quả, tự tránh lặp lại.

**Tại sao novel so với LEVI hiện tại**:

- LEVI dùng cluster occupied cells (KMeans) → representative code đầy đủ → input dài, frontier dễ "copy and tweak". SIMPLE-EVO chỉ inject description → frontier buộc phải suy nghĩ về ý tưởng, không thể copy code.
- LEVI strategy log đang dùng light model để summarize (equilibrium.py:422-483). SIMPLE-EVO bỏ summarizer call: description đã được LLM sinh ra cùng code → free.

### 2.4 Frontier ↔ Mutation parallelism (giữ nguyên 100%)

Đây là **điểm mạnh không được động đến**:

- `n_llm_workers` mutation producers chạy concurrent.
- `_pe_monitor` async task (runner.py:346-399) check trigger mỗi 2s.
- Khi frontier PE fire: frontier sinh 1 paradigm code (heavy model), **không block** mutation producers.
- Variants từ paradigm code được sinh bởi **mutation models in parallel** qua `asyncio.gather()` (equilibrium.py:916-919).
- Single eval queue: paradigm code + variants + mutation candidates đều compete fair trong cùng pipeline.

SIMPLE-EVO giữ y nguyên kiến trúc này. Đây là novelty của LEVI cần được highlight trong paper, không phải thứ để thay.

---

## 3. Cái gì thay (3 simplifications)

### 3.1 Pool: Top-K + Embedding Niching trên TEXT DESCRIPTION (thay CVT-MAP-Elites)

**Thay thế**: `cvt_map_elites.py` (961 dòng) + toàn bộ `behavior/extractor.py`, `behavior/features.py`.

**Key insight — ép LLM output có description**:

Mỗi LLM call (cả mutation lẫn frontier) đều bị **ép format output** thành 2 phần: một block `## Description` (2-4 câu mô tả idea: paradigm, key data structure, control flow, distinguishing trick) đứng trước, theo sau là một block `## Code` chứa Python code trong fenced block.

Description này là thứ được embed — không phải code. Lý do:

- Code có rất nhiều noise (whitespace, biến đổi tên, import order, …) → embedding code dễ "giống nhau giả" hoặc "khác nhau giả".
- Description capture được *ý tưởng* (greedy / BnB / SA / DP / ...). Embedding của description phản ánh *paradigm distance*, không phải *syntactic distance*.

**Fallback khi output thiếu description**:

- **Bước 1**: Thử extract description bằng regex giữa `## Description` và `## Code`.
- **Bước 2**: Nếu code chạy được nhưng description bị thiếu/empty/quá ngắn (< 20 ký tự): gọi **mutation model** (reuse, không thêm config mới) với prompt dạng *"Summarize this code in 2-3 sentences. Focus on: algorithmic paradigm, key data structures, distinguishing technique."* Output trở thành description.
- **Bước 3**: Nếu code không chạy được → vào error archive (LEVI hiện tại đã có) → KHÔNG cần embed.

**Embedding model**: `openai/text-embedding-3-small` (1536-d, $0.02/1M tokens). Description ~50-100 tokens → ~$0.000002 / call. K=100 pool → toàn run < $0.01 embedding cost. Cache theo `hash(description)`.

**Cơ chế pool mới**:

- Single list size K=100.
- Mỗi program: `(code, description, score, embedding_1536d, uses_count, created_at_eval, source)` với `source ∈ {init, mutate, repair, paradigm, variant}`.
- `add(program)`:
  1. Extract description (fallback summarize nếu cần).
  2. Compute embedding (cached).
  3. Find nearest existing program by cosine.
  4. If `cosine ≥ 0.95` (near-duplicate semantic): replace incumbent nếu score cao hơn, else drop.
  5. Else: append. If `len > K`: drop lowest-score.

**Tại sao novel**:

- LEVI dùng *structural* AST descriptor (math_operators, loop_nesting, ...). Hand-crafted, brittle, không capture được paradigm.
- SIMPLE-EVO dùng *semantic description embedding* — capture ý tưởng, không bị fool bởi syntactic variation. LLM tự khai báo mình đang làm paradigm gì, framework dùng đúng tín hiệu đó.
- Bỏ Voronoi tessellation, k-means++ centroid init, adaptive bounds Welford normalization.

**Tác động lên frontier (xem §2.3)**: representatives chỉ inject `description + score` (không code) → frontier input ngắn, không thể copy-paste paradigm.

### 3.2 Stagnation signal: 3 sliding-window stats (thay PPS công thức)

**Thay thế**: công thức PPS hiện tại phụ thuộc nhiều vào "new-best rate" — biến này quá sparse, đa số thời gian = 0 nên signal vô nghĩa.

**Cơ chế mới**:

```python
class Monitor:
    plateau_steps: int                # eval kể từ best score tăng lần cuối
    accept_window: deque[bool]        # maxlen=50, accept/reject của pool
    diversity_window: deque[float]    # maxlen=20, mean pairwise cosine recent adds

    def stagnation_level(self) -> float:  # ∈ [0, 1]
        return min(1.0, plateau_steps / PLATEAU_MAX)  # PLATEAU_MAX = 100

    def is_stuck(self) -> bool:
        return (plateau_steps > 80) or (mean(accept_window) < 0.08)
```

**Monitor signals KHÔNG trigger PE** (giữ nguyên LEVI: PE fire cron mỗi N evals). Chúng chỉ dùng để:

- `stagnation_level()` → input cho `get_budget_stage()` → chọn early/mid/late paradigm prompt mỗi khi cron PE fire.
- `is_stuck()` → switch advisor sang offensive mode (giữ trigger logic của LEVI nhưng đổi source).
- `is_stuck()` → toggle operator mix (mutate ↔ crossover) và temperature trong §3.3.

Lý do: frontier model phải rẻ. Cron schedule budget được; event-driven có rủi ro nổ chi phí khi run xui xẻo bị stuck sớm. LEVI hiện tại đang dùng cron và work — không sửa.

**Tại sao novel**:

- 3 signal **dense** (sample mỗi eval) thay vì 1 signal **sparse** (new-best events).
- Diversity signal khả thi nhờ embedding description (§3.1) — không có ở LEVI vì LEVI không embed.
- Bỏ công thức PPS multi-term với weights — chỉ 3 biến và 2 threshold.

### 3.3 Sampler: 2 operator + UCB-style selection (thay bandit 4D)

**Thay thế**: Thompson bandit trên (sampler × model × prompt_id × temperature), SAL Cơ chế D, NEW BEST bonus multiplicative.

**Cơ chế mới — operator chọn theo state**:

```text
operator = 'crossover' if random() < p_crossover else 'mutate'
where p_crossover = 0.3 default, 0.7 when is_stuck()

temperature = 1.1 if is_stuck() else 0.8
```

- **MUTATE**: 1 parent (selection bên dưới) + k=3 inspirations (cùng cơ chế selection).
- **CROSSOVER**: 2 parents (chọn từ 2 cluster embedding xa nhau nhất khi có thể) + 1-2 inspirations.
- **Inspirations chỉ gồm `description + score`** (không code, tương tự frontier ở §2.3) → mutation prompt ngắn, LLM không "copy and tweak" inspiration code mà phải apply ý tưởng từ description.

**Parent / inspiration selection — UCB-style với 3 thành phần**:

```text
priority(p) = score(p)                                              # exploit
            + α · sqrt( log(N_total) / (1 + uses_count(p)) )       # UCB novelty bonus
            + β · recency(p)                                        # boost program mới thêm
            − γ · max_cosine(p, already_picked)                     # diversity penalty (MMR-style)

recency(p) = exp( -(N_total - created_at_eval(p)) / τ )            # τ = 30 evals
```

Diễn giải 3 trọng số:

- **UCB novelty** (`α`): program nào ít được dùng → bonus cao. Giống UCB cổ điển. Khi `uses_count = 0` bonus rất lớn → buộc framework phải thử nó.
- **Recency** (`β`): program mới thêm vào pool (`created_at_eval` gần `N_total`) được boost → fresh ideas không bị starve bởi top scorers tích luỹ.
- **MMR diversity** (`γ`): khi chọn nhiều program cùng batch (parent + inspirations), penalize nếu giống cái đã pick → đảm bảo set inspiration đa dạng.

Default: `α = 0.5`, `β = 0.3`, `γ = 0.4` khi healthy; `α = 0.8`, `β = 0.5`, `γ = 0.7` khi `is_stuck` (thiên về exploration mạnh hơn).

**Tại sao novel**:

- Bỏ bandit 4D phức tạp — không claim adaptive sampler vì empirical bandit không tạo khác biệt thống kê đáng tin.
- UCB-style novelty bonus là chuẩn explore/exploit, dễ giải thích và ablate.
- Recency boost là cơ chế chống "old best lock-in" — vấn đề kinh điển khi top scorers đời đầu áp đảo top-K pool.

---

## 4. Kiến trúc tổng thể

```text
                ┌──────────────────────────────────────────────┐
                │  Frontier Model (heavy, ~5-15% budget)       │
                │                                              │
                │  ┌──────────┐                                │
                │  │ Init     │ once at t=0                    │
                │  │ (seeds)  │ — sinh N seed programs đa dạng │
                │  └────┬─────┘                                │
                │       │                                      │
                │  ┌────▼─────────────────────────────────┐    │
                │  │ Paradigm Shift (CRON, mỗi N evals)   │    │
                │  │   route theo stagnation_level():     │    │
                │  │     early → radical paradigm         │    │
                │  │     mid   → synthesis                │    │
                │  │     late  → surgical refine          │    │
                │  │   reps: 3 program (desc+score only)  │    │
                │  │   recent_trials: K=5 (desc+score+Δ)  │    │
                │  └────┬─────────────────────────────────┘    │
                │       │                                      │
                │  paradigm_code                               │
                └───────┼──────────────────────────────────────┘
                        │
                        │ (async, non-blocking)
                        ▼
   ┌──────────────────────────────────────────────────────────┐
   │  Mutation Models (light, ~85-95% budget) — N producers   │
   │                                                          │
   │   ┌─────────────┐   ┌─────────────────────────────┐      │
   │   │ Mutate/     │◄──┤ Selector (UCB-style)        │      │
   │   │ Crossover   │   │  - score + novelty + recency│      │
   │   └──────┬──────┘   │    − diversity penalty      │      │
   │          │          │  - inspirations: desc+score │      │
   │          │          └─────────────────────────────┘      │
   │          │                                               │
   │   ┌──────▼──────┐   ┌──────────────────────────────┐     │
   │   │ Variants    │◄──┤ Paradigm code (from frontier)│     │
   │   │ (parallel   │   │  → n variants in parallel    │     │
   │   │  gather)    │   │    via asyncio.gather        │     │
   │   └──────┬──────┘   └──────────────────────────────┘     │
   │          │                                               │
   │   ┌──────▼──────┐   ┌─────────────────────────────┐      │
   │   │ Self-repair │◄──┤ Error archive (LEVI)        │      │
   │   │ (one-shot)  │   │  Zipfian by parent score    │      │
   │   └──────┬──────┘   └─────────────────────────────┘      │
   │          │                                               │
   │   output format: ## Description + ## Code                │
   │   prompt-injected:                                       │
   │     - meta_advice text (defensive/offensive)             │
   └──────────┼───────────────────────────────────────────────┘
              │
              ▼
   ┌──────────────────────────────────────────────────────────┐
   │  Eval Queue (single, shared) — N eval consumers          │
   └────────────┬─────────────────────────────────────────────┘
                │
                ▼
   ┌──────────────────────────────────────────────────────────┐
   │  Pool (Top-K + DESCRIPTION-embedding niching)            │
   │   - K=100, cosine niche threshold 0.95                   │
   │   - openai/text-embedding-3-small (1536-d)               │
   │   - tag: source, uses_count, created_at_eval             │
   └────────────┬─────────────────────────────────────────────┘
                │
                ▼
   ┌──────────────────────────────────────────────────────────┐
   │  Monitor — 3 sliding-window signals (route-only)         │
   │   - plateau_steps, accept_rate, diversity                │
   │   - exposes: stagnation_level(), is_stuck()              │
   │   → routes 3 phase + toggles operator mix                │
   │   → KHÔNG trigger PE (PE = cron)                         │
   └──────────────────────────────────────────────────────────┘
```

---

## 5. Bảng so sánh với LEVI

| Thành phần                                                             | LEVI gốc                                       | SIMPLE-EVO                                                         | Status           |
| ---------------------------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------ | ---------------- |
| Frontier ↔ Mutation async parallelism                                  | ✓                                              | ✓                                                                  | **giữ**          |
| 3 phase paradigm shift (early/mid/late)                                | ✓                                              | ✓                                                                  | **giữ**          |
| PE trigger                                                             | Cron `eval_count % interval == 0`              | Cron `eval_count % interval == 0`                                  | **giữ**          |
| Variants from paradigm via `asyncio.gather` (mutation model)           | ✓                                              | ✓                                                                  | **giữ**          |
| Error archive + self-repair (one-shot, mutation model)                 | ✓                                              | ✓                                                                  | **giữ**          |
| Meta-advisor 2 mode (defensive/offensive, inject into mutation prompt) | ✓                                              | ✓                                                                  | **giữ**          |
| Single shared eval queue                                               | ✓                                              | ✓                                                                  | **giữ**          |
| Pool                                                                   | CVT-MAP-Elites, 1000 centroids, AST descriptor | Top-K=100 + cosine-niche trên DESCRIPTION (text-embedding-3-small) | **thay**         |
| Frontier representatives                                               | KMeans cluster occupied cells, full code       | 3 program, **desc + score only**, phase-specific selection         | **thay**         |
| Strategy log                                                           | Light model summary mỗi PE                     | Recent trials (desc + score + Δ), không cần summarizer             | **thay**         |
| Stagnation signal                                                      | PPS công thức + new-best rate                  | 3 sliding windows (route-only)                                     | **thay**         |
| Sampler                                                                | 4D Thompson bandit (SAL D) + NEW BEST bonus    | 2 ops (mutate/crossover) + state-toggled mix                       | **thay**         |
| Selection                                                              | AdaptiveRank/UCB/Uniform/Subscore              | UCB-style: score + novelty + recency − diversity                   | **thay**         |
| Inspiration content                                                    | Full code                                      | Description + score only                                           | **thay**         |
| Stagnation routing for 3 phase                                         | `s(t)` PPS formula                             | `Monitor.stagnation_level()` (plateau/PLATEAU_MAX)                 | **đơn giản hoá** |

---

## 6. Đóng góp có thể claim trong paper

1. **Description-conditioned LLM output + description-based niching**: ép LLM output `## Description` + `## Code`, embed description (không phải code) bằng `openai/text-embedding-3-small`. Capture *paradigm distance*, không bị fool bởi syntactic noise. Thay thế CVT-MAP-Elites và hand-crafted AST features.
2. **Dense sliding-window stagnation signals (route-only)**: 3 signal dày đặc (plateau, accept-rate, diversity) chỉ dùng để **route** 3 phase và toggle operator mix. PE trigger vẫn cron như LEVI — đảm bảo cost frontier không nổ.
3. **Phase-specific representatives + recent-trial log (description-only)**: representatives cho 3 phase chọn theo tiêu chí riêng (early=diverse, mid=top+diverse, late=top-3). Recent trials inject `desc + score + Δ` — không cần light-model summarizer.
4. **UCB-style selection với novelty + recency + diversity**: thay bandit 4D bằng selection rule một dòng có 3 thành phần explore (novelty bonus + recency boost − diversity penalty). Dễ ablate từng thành phần riêng.
5. **Preserved architectural advantages**: async frontier-mutation parallelism, 3-phase paradigm shift, error-archive self-repair, meta-advisor 2 mode — bốn thứ LEVI làm tốt và SIMPLE-EVO chứng minh là cần thiết qua ablation.

Paragraph viết được cho abstract:

> "SIMPLE-EVO retains LEVI's core advantage — async coordination between a frontier model (orchestrating 3-phase paradigm shifts: radical, synthesis, surgical, on a fixed cron schedule) and parallel mutation models (handling variants, repairs, and exploration) — while replacing four over-engineered subsystems. First, the behavioral archive replaces CVT-MAP-Elites with description-based semantic niching: every LLM call is forced to output a short paradigm description that is embedded via `text-embedding-3-small`, capturing algorithmic intent rather than syntactic surface. Second, the PPS stagnation formula driven by sparse new-best events is replaced by three dense sliding-window statistics (plateau length, acceptance rate, recent diversity) used only to route the three paradigm-shift phases and toggle operator mix. Third, frontier prompts receive at most three phase-specific representatives and a recent-trial log composed solely of descriptions, scores, and deltas — eliminating both full-code copy-paste tendencies and the auxiliary summarizer model. Fourth, the 4-dimensional Thompson bandit over (sampler × model × prompt × temperature) is replaced by two evolutionary operators (mutate/crossover) coupled with a UCB-style selection rule that combines a novelty bonus, a recency boost, and a diversity penalty. SIMPLE-EVO reduces core code by ~80% and hyperparameters from ~25 to ~10 while matching LEVI on [benchmarks]."

---

## 7. Hyperparameters

| Param                       | Default                          | Role                                              |
| --------------------------- | -------------------------------- | ------------------------------------------------- |
| `pool_K`                    | 100                              | Top-K pool size                                   |
| `niche_cosine_threshold`    | 0.95                             | Near-duplicate replace (trên description embed)   |
| `embedding_model`           | `openai/text-embedding-3-small`  | Embedding của description                         |
| `pe_cron_interval`          | 50                               | Mỗi N evals fire frontier PE (giữ giống LEVI)     |
| `n_representatives`         | 3                                | Reps inject vào frontier prompt (desc+score only) |
| `recent_trials_k`           | 5                                | Số trial gần nhất inject vào frontier prompt      |
| `plateau_max`               | 100                              | Denominator của `stagnation_level()`              |
| `stuck_plateau_threshold`   | 80                               | Switch advisor offensive + operator mix           |
| `stuck_accept_threshold`    | 0.08                             | Switch advisor offensive + operator mix           |
| `mmr_lambda_early_reps`     | 0.2                              | MMR λ cho early-phase representatives             |
| `mmr_lambda_mid_reps`       | 0.5                              | MMR λ cho mid-phase representatives               |
| `ucb_alpha_healthy/stuck`   | 0.5 / 0.8                        | UCB novelty bonus weight                          |
| `ucb_beta_healthy/stuck`    | 0.3 / 0.5                        | Recency boost weight                              |
| `ucb_gamma_healthy/stuck`   | 0.4 / 0.7                        | Diversity penalty weight                          |
| `recency_tau`               | 30 evals                         | Half-life của recency exp decay                   |
| `p_crossover_healthy/stuck` | 0.3 / 0.7                        | Operator mix                                      |
| `temperature_healthy/stuck` | 0.8 / 1.1                        | Mutation temp                                     |
| `desc_min_chars`            | 20                               | Threshold gọi fallback summary                    |

Frontier model, mutation model, advisor model, repair budget, eval budget — đều giữ nguyên config schema từ LEVI.

---

## 8. Lộ trình tích hợp

**Phase A — modules không phụ thuộc LLM** (test được ngay): `levi/simple/pool.py` (Top-K + description-embedding niching, có cache); `levi/simple/monitor.py` (3 sliding-window signals); `levi/simple/selector.py` (UCB-style novelty + recency + diversity); unit tests cho cả 3.

**Phase B — adapter cho LLM stack đã có**: `levi/simple/operator.py` gọi mutation client với 2 prompt templates (mutate / crossover), ép format `## Description` + `## Code`, fallback summarize khi thiếu desc; reuse `clients/`, `prompts/builder.py`, `pipeline/state.py`; reuse error archive + repair từ producer.py; reuse meta-advisor từ consumer.py (chỉ đổi nguồn switch defensive/offensive sang `Monitor.is_stuck()`).

**Phase C — frontier orchestrator**: `levi/simple/frontier.py` wrap `equilibrium/prompts.py` 3 phase prompts; PE trigger giữ cron mỗi N evals (giống LEVI); phase routing dùng `Monitor.stagnation_level()`; representatives chọn theo phase (xem §2.3); recent_trials log inject `{strategy_log_block}` slot; variants generation reuse `asyncio.gather()` pattern từ equilibrium.py:916-919.

**Phase D — pipeline entry point**: `levi/methods/simple_evo.py` song song với `levi.py`; config flag `method: simple_evo` cho A/B testing.

**Phase E — ablation**: ablate từng simplification riêng (description-embed vs code-embed, UCB-style vs uniform selection, novelty bonus α=0 vs default, recency boost β=0 vs default, reps with-code vs desc-only) để defend trong paper; benchmark side-by-side với LEVI trên cùng problems, cùng budget.

---

## 9. Các quyết định đã chốt và câu hỏi mở còn lại

**Đã chốt qua thảo luận**:

- **Embed cái gì**: description (text), không phải code. LLM bị ép output `## Description` + `## Code`.
- **Embedding model**: `openai/text-embedding-3-small`.
- **Fallback khi thiếu description**: reuse **mutation model** để summarize (không thêm config model mới).
- **Frontier cadence**: cron mỗi N evals (giống LEVI), KHÔNG event-driven.
- **Representatives selection per phase**: early=3 MMR-diverse (λ=0.2), mid=top + 2 MMR-diverse (λ=0.5), late=top-3 by score.
- **Recent trials injection**: K=5 trials, format `desc + score + Δ`, không cần light-model summarizer.
- **Selection rule**: UCB-style `score + α·novelty + β·recency − γ·diversity` thay cho bandit 4D.

**Còn mở (low-risk, có thể chốt khi code)**:

- **`uses_count` reset sau PE**: có nên reset = 0 cho mọi program ngay sau khi PE accept paradigm code không (để paradigm-fresh programs không bị penalty kế thừa)? Đề xuất: yes.
- **Source-tag bonus**: có nên cho paradigm/variant source bonus nhẹ `+0.05` vào priority trong K=20 eval đầu sau PE? Đề xuất: yes, để frontier output không bị bury bởi top scorers cũ.
- **`is_collapsing` action**: nếu thêm signal collapse (diversity < threshold), action có phải là forced crossover từ 2 cluster xa nhau nhất không? Đề xuất: yes, một step "diversity injection" trước khi đợi cron PE kế tiếp.
- **`pe_cron_interval`**: 50 evals đề xuất, nhưng cần A/B với LEVI để chọn cho từng benchmark family (BBO, AlphaEvolve, …).

