# Levi — Chi Tiết Kiến Trúc, Luồng Hoạt Động & Toán Học

## Mục lục

1. [Tổng Quan Hệ Thống](#1-tổng-quan-hệ-thống)
2. [Kiến Trúc Module](#2-kiến-trúc-module)
3. [Luồng Hoạt Động Chi Tiết](#3-luồng-hoạt-động-chi-tiết)
4. [CVT-MAP-Elites — Toán Học](#4-cvt-map-elites--toán-học)
5. [Sampling Strategies — Toán Học](#5-sampling-strategies--toán-học)
6. [Behavior Extraction & Normalization](#6-behavior-extraction--normalization)
7. [Punctuated Equilibrium](#7-punctuated-equilibrium)
8. [SAL — Stagnation-Adaptive Levi](#8-sal--stagnation-adaptive-levi)
9. [Pipeline Architecture](#9-pipeline-architecture)
10. [Budget & Cost Control](#10-budget--cost-control)
11. [Meta-Advice System](#11-meta-advice-system)
12. [Cascade Evaluation](#12-cascade-evaluation)
13. [Init Phase — Diversifier](#13-init-phase--diversifier)
14. [Artifact Adapters](#14-artifact-adapters)
15. [Cấu Hình Đầy Đủ](#15-cấu-hình-đầy-đủ)

---

## 1. Tổng Quan Hệ Thống

Levi là một **evolutionary code/prompt optimization engine** sử dụng LLM để tiến hóa chương trình (code hoặc prompt). Ý tưởng cốt lõi:

```
Input:  Problem description + Score function + Budget
Output: Best program/prompt found within budget
```

Levi kết hợp 3 lý thuyết chính:
- **MAP-Elites** (Quality-Diversity): duy trì một archive đa dạng thay vì chỉ 1 best solution
- **CVT (Centroidal Voronoi Tessellation)**: phân chia không gian hành vi thành các ô (cell) bằng k-means
- **Punctuated Equilibrium** (Sinh học tiến hóa): Tiến hóa xảy ra theo đợt — giai đoạn ổn định xen kẽ đột phá

### Đặc trưng thiết kế:
- **Async producer-consumer pipeline**: N producers gọi LLM song song, M consumers đánh giá song song
- **Multi-model**: Dùng model nhẹ (mutation) + model nặng (paradigm shift) kết hợp
- **Budget-aware**: Dừng theo $ / số eval / thời gian, chi phí được track realtime
- **Adaptive**: SAL điều chỉnh chiến lược dựa trên tín hiệu trì trệ

---

## 2. Kiến Trúc Module

```
levi/
├── methods/levi.py          # Entry point: evolve_code(), evolve_prompts()
├── config/models.py         # Pydantic config models (LeviConfig, SalConfig, ...)
├── pipeline/
│   ├── runner.py            # PipelineRunner: điều phối toàn bộ
│   ├── producer.py          # llm_producer: gọi LLM tạo ứng viên
│   ├── consumer.py          # eval_consumer: đánh giá + cập nhật archive
│   └── state.py             # PipelineState, BudgetTracker, ClientGate
├── pool/
│   └── cvt_map_elites.py    # CVTMAPElitesPool + Samplers (UCB, Softmax, Cyclic, ...)
├── behavior/
│   ├── extractor.py         # BehaviorExtractor: trích xuất vector hành vi
│   └── features.py          # AST-based feature functions
├── equilibrium/
│   ├── equilibrium.py       # PunctuatedEquilibrium: paradigm shift engine
│   └── prompts.py           # Prompt templates (early/mid/late)
├── init/
│   └── diversifier.py       # Init phase: đa dạng hóa ban đầu
├── artifacts/
│   ├── base.py              # ArtifactAdapter (abstract)
│   ├── code.py              # CodeAdapter: evolution code
│   └── prompt.py            # PromptAdapter: evolution prompt
├── selection/
│   └── component.py         # UCB/RoundRobin/Stagnation component selectors
├── prompts/
│   ├── builder.py           # Prompt construction
│   └── bundle.py            # Multi-component prompt bundles
├── clients/
│   ├── base.py              # ClientSpec, LM protocol
│   └── lm.py               # LiteLLM integration
└── utils/
    ├── resilient_pool.py    # Process pool cho eval
    └── evaluation.py        # Score coercion utilities
```

---

## 3. Luồng Hoạt Động Chi Tiết

### 3.1 High-Level Flow

```
evolve_code() / evolve_prompts()
    │
    ├─ 1. Build LeviConfig
    ├─ 2. Create BehaviorExtractor, CVTMAPElitesPool, PipelineState
    ├─ 3. [Optional] Prompt optimization (DSPy/MIPROv2)
    ├─ 4. INIT PHASE (Diversifier)
    │      ├─ Evaluate seed program
    │      ├─ Generate N diverse seeds (heavy model)
    │      ├─ Generate M variants per seed (light model)
    │      ├─ Evaluate all variants
    │      ├─ Build k-means centroids from behavior vectors
    │      └─ Populate archive (best per cell)
    ├─ 5. MAIN LOOP (PipelineRunner)
    │      ├─ N × llm_producer (async tasks)
    │      ├─ M × eval_consumer (async tasks)
    │      ├─ PE monitor (trigger every K evals)
    │      └─ Status monitor
    └─ 6. Return LeviResult (best_program, best_score, stats)
```

### 3.2 Detailed Main Loop — Một Vòng Lặp Producer-Consumer

```
┌─────────────── llm_producer ───────────────┐
│                                             │
│  1. pool.get_weighted_sampler_config()      │
│     → chọn (sampler_name, model)            │
│     → nếu SAL enabled: Thompson Bandit     │
│                                             │
│  2. pool.sample(sampler_name, n_parents)    │
│     → sampler.select_cells(elites, n)       │
│     → trả về parent + inspirations          │
│                                             │
│  3. [SAL Cơ chế B] nếu s(t) ≥ threshold:  │
│     → thêm global-best elite               │
│     → thêm farthest-behavior elite         │
│                                             │
│  4. artifact_adapter.build_mutation_prompt()│
│     → format prompt với parents, meta-advice│
│                                             │
│  5. state.acompletion(model, prompt)        │
│     → gọi LLM qua ClientGate               │
│     → track cost, concurrency              │
│                                             │
│  6. artifact_adapter.extract_candidate()    │
│     → parse code từ response                │
│                                             │
│  7. code_queue.put(candidate)               │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────── eval_consumer ──────────────┐
│                                             │
│  1. code_queue.get() → item                 │
│                                             │
│  2. state.try_start_evaluation()            │
│     → reserve 1 eval slot (budget check)   │
│                                             │
│  3. [Cascade] quick eval → threshold check │
│     → reject sớm nếu quick_score thấp      │
│                                             │
│  4. artifact_adapter.evaluate(content)      │
│     → chạy trong process pool              │
│     → timeout protection                   │
│                                             │
│  5. pool.add(program, eval_result)          │
│     → extract behavior → find cell         │
│     → if score > incumbent → accept!       │
│                                             │
│  6. pool.update_sampler(cell, success)      │
│  7. [SAL] pool.update_bandit(accepted)     │
│  8. state.record_score(...)                │
│                                             │
│  9. [Meta-advice] mỗi K evals:             │
│     → generate_meta_advice()               │
│                                             │
│  10. state.finish_evaluation()              │
└─────────────────────────────────────────────┘
```

---

## 4. CVT-MAP-Elites — Toán Học

### 4.1 Ý tưởng cốt lõi

MAP-Elites chia không gian hành vi (behavior space) thành N ô (cell). Mỗi ô giữ đúng 1 chương trình tốt nhất (elite). Thay vì chỉ tối ưu score, ta duy trì **đa dạng cấu trúc**.

### 4.2 CVT (Centroidal Voronoi Tessellation)

Thay vì grid cứng, CVT dùng k-means để chia không gian:

**Khởi tạo centroids** (2 chế độ):

1. **Uniform (mặc định cũ)**: Sample 10000 × N_dims điểm ∼ U[0,1], chạy k-means++ → N centroids
2. **Data-driven (mặc định mới)**: Dùng behavior vectors thực từ init phase, chạy k-means → centroids

```python
# set_centroids_from_data():
X = behavior_vectors   # shape (n_programs, n_dims)
kmeans = KMeans(n_clusters=N, init="k-means++")
centroids = kmeans.fit(X).cluster_centers_   # shape (N, n_dims)
labels = kmeans.labels_                       # gán mỗi program vào 1 cell
```

### 4.3 Gán Cell cho một Program

Cho program p với behavior vector **b** (đã normalize về [0,1]):

$$\text{cell}(p) = \arg\min_{i \in [N]} \| \mathbf{b} - \mathbf{c}_i \|^2$$

Trong đó **c**_i là centroid thứ i.

### 4.4 Acceptance Rule

```
if cell[i] is empty:
    accept (fill cell)
elif score(new) > score(incumbent):
    accept (replace incumbent)
else:
    reject
```

Điều kiện thay thế: \( f(\text{new}) > f(\text{current elite of cell}_i) \)

---

## 5. Sampling Strategies — Toán Học

Pool có nhiều sampler, mỗi sampler chọn cell khác nhau:

### 5.1 UCB Sampler (Upper Confidence Bound)

$$\text{UCB}_i = \underbrace{\frac{s_i}{n_i}}_{\text{exploitation}} + c \cdot \underbrace{\sqrt{\frac{\ln(T)}{n_i}}}_{\text{exploration}}$$

Trong đó:
- \( s_i \) = số lần sampling cell i tạo ra offspring được accept
- \( n_i \) = tổng số lần cell i được sample
- \( T \) = tổng số sample (toàn bộ)
- \( c = 2.0 \) (exploration constant)
- Cell chưa từng sample: UCB = ∞ (ưu tiên tuyệt đối)

### 5.2 Softmax Sampler

Chọn cell theo phân phối xác suất Boltzmann:

$$P(\text{cell}_i) = \frac{\exp\left(\frac{f_i^{\text{norm}} - 1}{T}\right)}{\sum_j \exp\left(\frac{f_j^{\text{norm}} - 1}{T}\right)}$$

Trong đó:
- \( f_i^{\text{norm}} = \frac{f_i - f_{\min}}{f_{\max} - f_{\min}} \in [0, 1] \)
- \( T \) = temperature (0.3 → greedy, 1.2 → uniform hơn)

### 5.3 Cyclic Annealing Sampler

Temperature biến đổi theo chu kỳ dựa trên budget progress:

$$T(t) = T_{\min} + (T_{\max} - T_{\min}) \cdot (1 - \text{cycle\_progress})$$

$$\text{cycle\_progress} = (p_{\text{budget}} \cdot n_{\text{cycles}}) \mod 1$$

Trong đó:
- \( p_{\text{budget}} \in [0,1] \) = phần budget đã tiêu
- \( n_{\text{cycles}} = 4 \) (mặc định)
- \( T_{\max} = 1.2, T_{\min} = 0.15 \)

**Hiệu ứng**: Lặp lại nhiều lần: exploration → exploitation → exploration → ...

### 5.4 Uniform Sampler

$$P(\text{cell}_i) = \frac{1}{|\text{occupied cells}|}$$

### 5.5 Subscore Sampler

Softmax theo một sub-metric cụ thể (ví dụ: accuracy, latency) thay vì primary score.

### 5.6 Weighted Sampler-Model Selection

Mỗi lượt produce, hệ thống chọn (sampler, model) pair:

**Legacy mode** (SAL disabled):
$$P(\text{pair}_i) = \frac{w_i}{\sum_j w_j}$$

**SAL mode** (Thompson Bandit — Cơ chế D):
$$\theta_i \sim \text{Beta}(\alpha_i, \beta_i)$$
$$\text{raw}_i = \theta_i \cdot (1 + \gamma \cdot \text{nb}_i)^{1+s}$$
$$w_i = w_{\min} + (1 - N \cdot w_{\min}) \cdot \frac{\text{raw}_i}{\sum_j \text{raw}_j}$$

Trong đó:
- \( \alpha_i, \beta_i \) = posterior Beta-Bernoulli (khởi tạo = 1,1)
- \( \gamma = 0.5 \) = bonus cho arm có NEW BEST
- \( \text{nb}_i \) = số lần arm i tạo ra NEW BEST
- \( s = s(t) \) = stagnation depth hiện tại
- \( w_{\min} = 0.05 \) = floor weight (đảm bảo mọi arm luôn có cơ hội)

---

## 6. Behavior Extraction & Normalization

### 6.1 Feature Extraction

Mỗi program được biến thành vector hành vi qua AST analysis:

| Feature | Ý nghĩa | Cách tính |
|---------|----------|-----------|
| `loop_count` | Số vòng lặp | Count(For, While) nodes |
| `branch_count` | Số nhánh điều kiện | Count(If) nodes |
| `math_operators` | Mật độ toán tử | Count(BinOp, UnaryOp) |
| `loop_nesting_max` | Độ lồng loop tối đa | Max depth of nested For/While |
| `ast_depth` | Chiều sâu AST | Recursive max depth |
| `cyclomatic_complexity` | Độ phức tạp McCabe | 1 + If + While + For + ExceptHandler + BoolOp |
| `code_length` | Độ dài ký tự | len(content) |
| `comprehension_count` | Số comprehension | ListComp + DictComp + SetComp + GeneratorExp |
| `range_max_arg` | Giới hạn range() | Max numeric arg in range() calls |
| + `score_keys` | Các sub-score | Lấy từ eval result dict |

### 6.2 Normalization: Welford's Online Z-Score + Sigmoid

Adaptive mode (mặc định):

**Bước 1**: Cập nhật running statistics bằng Welford's algorithm:
$$n \leftarrow n + 1$$
$$\delta = x - \mu$$
$$\mu \leftarrow \mu + \frac{\delta}{n}$$
$$\delta_2 = x - \mu$$
$$M_2 \leftarrow M_2 + \delta \cdot \delta_2$$

$$\sigma = \sqrt{\frac{M_2}{n-1}}, \quad \sigma \geq 0.1 \text{ (clamp)}$$

**Bước 2**: Z-score:
$$z = \frac{x - \mu}{\sigma}, \quad z \in [-10, 10] \text{ (clip)}$$

**Bước 3**: Sigmoid mapping to [0,1]:
$$\text{normalized} = \frac{1}{1 + e^{-z}}$$

**Hiệu ứng**: Feature values luôn nằm trong [0,1] bất kể scale ban đầu.

### 6.3 Fixed Bounds Mode (Deterministic)

$$\text{normalized} = \text{clip}\left(\frac{x - \text{min}}{\text{max} - \text{min}}, 0, 1\right)$$

---

## 7. Punctuated Equilibrium

### 7.1 Ý tưởng

Mô phỏng hiện tượng sinh học: tiến hóa = nhiều giai đoạn "trì trệ" (stasis) xen kẽ các "đột phá" (punctuation). Cụ thể: mỗi K evaluations, gọi model nặng (gpt-4o, deepseek-r1...) để tạo ra giải pháp **khác hẳn** cách tiếp cận hiện tại.

### 7.2 Algorithm

```
Every pe_interval evaluations:
  1. CLUSTER: k-means trên occupied centroids → n_clusters clusters
  2. SELECT REPRESENTATIVES: best elite per cluster (hoặc farthest-first nếu Hard-PE)
  3. BUILD PROMPT: chọn template (early/mid/late) theo stagnation
  4. PARADIGM SHIFT: gọi heavy model → generate 1 giải pháp mới
  5. EVALUATE: đánh giá paradigm solution
  6. VARIANTS: nếu paradigm valid → sinh n_variants biến thể (model nhẹ)
  7. INSERT: chèn tất cả vào archive
```

### 7.3 Prompt Staging (SAL Cơ chế A)

| Stagnation s(t) | Stage | Chiến lược prompt |
|-----------------|-------|-------------------|
| s < 0.3 | `early` | "Tìm paradigm KHÁC HẲN" — đa dạng tối đa |
| 0.3 ≤ s < 0.7 | `mid` | "Kết hợp ưu điểm" — synthesis |
| s ≥ 0.7 | `late` | "Sửa điểm yếu" — targeted refinement |

---

## 8. SAL — Stagnation-Adaptive Levi

SAL là hệ thống gồm 5 cơ chế (A–E), tất cả đọc từ **một tín hiệu duy nhất**: stagnation depth s(t).

### 8.1 Stagnation Signal

$$s(t) = \min\left(1, \frac{n_{\text{since\_best}}}{\tau}\right)$$

Trong đó:
- \( n_{\text{since\_best}} \) = số evals kể từ lần cuối tạo NEW BEST
- \( \tau = 80 \) = plateau length để saturate

| s(t) | Trạng thái |
|------|-----------|
| 0.0 | Đang improve liên tục |
| 0.5 | Bắt đầu trì trệ (40 evals không improve) |
| 1.0 | Hoàn toàn stuck (≥ τ evals không improve) |

### 8.2 Cơ chế A — PE Prompt Staging

Thay vì luôn dùng prompt "early" (tìm cách khác hẳn), điều chỉnh chiến lược PE theo mức trì trệ:
- Thấp → "Khám phá tự do" (đa dạng)
- Trung bình → "Tổng hợp ưu điểm" (recombination)
- Cao → "Sửa cụ thể" (exploitation tập trung)

Ngoài ra, inject context vào mutation prompt:
- `best_score`, `evals_since_best`, `top_failures`

### 8.3 Cơ chế B — Mutation Context Augmentation

Khi \( s(t) \geq \text{context\_threshold} \) (mặc định 0.5):

Producer thêm 2 elite vào danh sách inspirations:
1. **Global-best elite**: Giải pháp tốt nhất archive (làm "north star")
2. **Farthest-behavior elite**: Elite có behavior vector xa nhất từ parent

$$\text{far\_elite} = \arg\max_{e \neq \text{parent}} \| \mathbf{b}_e - \mathbf{b}_{\text{parent}} \|_2$$

**Hiệu ứng**: LLM thấy cả "đỉnh cao" và "giải pháp cấu trúc khác" → tạo ra offspring sáng tạo hơn.

### 8.4 Cơ chế C — Meta-Advice Dual Mode

| s(t) | Mode | Mục tiêu |
|------|------|----------|
| < threshold | **Defensive** | Liệt kê bugs, tránh lỗi lặp lại |
| ≥ threshold | **Offensive** | Đề xuất chiến lược mới, kỹ thuật đột phá |

Offensive prompt bao gồm:
- Best score + thời gian stuck
- Per-sampler accept counts (sampler nào đang work?)
- Yêu cầu 3 "lever" cụ thể (tên thuật toán, parameter, cấu trúc dữ liệu)

### 8.5 Cơ chế D — Thompson Beta-Bernoulli Bandit

Mỗi (sampler, model) pair = 1 arm trong multi-armed bandit:

**Prior**: \( \text{Beta}(1, 1) \) (uniform)

**Update** sau mỗi evaluation:
- Accepted offspring: \( \alpha_i \leftarrow \alpha_i + 1 \)
- Rejected offspring: \( \beta_i \leftarrow \beta_i + 1 \)
- NEW BEST: \( \text{nb}_i \leftarrow \text{nb}_i + 1 \)

**Selection** (Thompson Sampling):
$$\theta_i \sim \text{Beta}(\alpha_i, \beta_i)$$
$$\text{score}_i = \theta_i \cdot (1 + \gamma \cdot \text{nb}_i)^{1+s(t)}$$

Posterior mean of arm i: \( E[\theta_i] = \frac{\alpha_i}{\alpha_i + \beta_i} \)

**Hiệu ứng**: Arms có accept rate cao VÀ đã tạo NEW BEST được ưu tiên nhiều hơn khi stagnation tăng.

### 8.6 Cơ chế E — Hard Punctuated Equilibrium

**Trigger conditions** (tất cả phải đúng):
1. SAL enabled + mechanism E enabled
2. \( \text{hard\_pe\_count} < \text{max\_per\_run} \) (mặc định max = 2)
3. \( \text{consecutive\_pe\_no\_best} \geq 2 \) (2 PE liên tiếp thất bại)
4. \( s(t) \geq \text{hard\_pe\_threshold} \) (mặc định 0.8)

**Khác biệt so với PE thường**:
| Thuộc tính | PE thường | Hard-PE |
|-----------|-----------|---------|
| n_clusters | 3 | 6 |
| Representative selection | Max-score per cluster | Farthest-first traversal |
| Reasoning effort | None / config | "high" (forced) |

**Farthest-first traversal** (chọn đại diện):
1. Cluster 1: chọn elite có score cao nhất (anchor)
2. Cluster 2+: chọn elite có behavior vector **xa nhất** so với centroid của tất cả elite đã chọn

$$\text{next} = \arg\max_{e \in \text{cluster}_k} \left\| \mathbf{b}_e - \frac{1}{|\text{picked}|}\sum_{p \in \text{picked}} \mathbf{b}_p \right\|$$

---

## 9. Pipeline Architecture

### 9.1 Concurrency Model

```
PipelineRunner
├── N × llm_producer (asyncio tasks) ─→ code_queue ─→ M × eval_consumer
├── PE monitor (asyncio task, polled every 2s)
├── Status monitor (every 30s)
└── ClientGate (semaphore-based LLM concurrency control)
```

- `code_queue`: AsyncQueue với maxsize = 2 × n_eval_processes
- `archive_lock`: asyncio.Lock bảo vệ mọi mutation trên pool
- `stop_event`: Signal dừng tất cả workers

### 9.2 Budget Exhaustion Protocol

```
BudgetTracker.exhausted check:
  - dollars: total_cost >= budget.dollars?
  - evaluations: (eval_count + eval_in_flight) >= budget.evaluations?
  - seconds: elapsed >= budget.seconds?
  - target_score: best_score >= target?

Khi exhausted:
  1. try_start_evaluation() returns False
  2. acompletion() raises BudgetLimitReached
  3. Producer/Consumer set stop_event
  4. All tasks get cancelled gracefully
```

### 9.3 Serial Mode (Budget Tight)

Khi budget gần hết, ClientGate chuyển sang serial mode:
$$\text{serial} = \begin{cases} \text{true} & \text{if remaining} \leq \max(3 \cdot \text{EMA}, 0.03 \cdot \text{limit}, 0.05) \\ \text{true} & \text{if eval\_remaining} \leq 2 \\ \text{true} & \text{if time\_remaining} \leq 15s \\ \text{false} & \text{otherwise} \end{cases}$$

---

## 10. Budget & Cost Control

### 10.1 Cost Tracking

Mọi LLM call đều đi qua `ClientGate.acompletion()`:
1. Check budget (raise nếu hết)
2. Acquire semaphore (concurrency limit)
3. Gọi LLM
4. Record cost (EMA + accumulate)

```python
# EMA update:
ema = 0.8 * ema_prev + 0.2 * new_cost
```

### 10.2 BudgetConfig

```python
BudgetConfig(
    dollars=5.0,          # Max tổng chi phí ($)
    evaluations=200,      # Max số lượt eval
    seconds=600.0,        # Max wall-clock time
    target_score=0.95,    # Early-stop nếu đạt target
)
```

---

## 11. Meta-Advice System

### 11.1 Cơ chế

Mỗi `meta_advice.interval` evaluations (mặc định 50):
1. Thu thập metrics: accept/reject/error counts, top errors
2. Gọi LLM (model nhẹ) với prompt defensive/offensive
3. Kết quả = đoạn text "lessons learned"
4. Inject vào mutation prompt (80% probability) trong period tiếp theo

### 11.2 Prompt Template

**Defensive** (khi healthy): "Liệt kê lỗi thường gặp, cách fix"
**Offensive** (khi stagnant): "3 đòn bẩy cụ thể để phá plateau"

---

## 12. Cascade Evaluation

### 12.1 Ý tưởng

Đánh giá rẻ trước (quick_inputs), chỉ chạy full eval nếu vượt ngưỡng:

```
1. Run score_fn(candidate, quick_inputs) → quick_score
2. Preview target cell
3. Get incumbent's quick_score
4. If quick_score < incumbent * min_score_ratio → REJECT sớm
5. Else → chạy full evaluation
```

### 12.2 Tiết kiệm

- `min_score_ratio = 0.8`: Reject nếu quick score < 80% incumbent
- `quick_timeout = 30s`: Timeout riêng cho quick eval

---

## 13. Init Phase — Diversifier

### 13.1 Mục đích

Tạo archive ban đầu đa dạng để evolution không bị collapse vào 1 vùng.

### 13.2 Algorithm

```
Phase 1 — Diverse Seeds:
  for i in range(n_diverse_seeds):
    prompt = "Tạo cách giải KHÁC với N cách hiện có"
    seed[i] = heavy_model(prompt)
    evaluate(seed[i])

Phase 2 — Variants:
  for each seed:
    for j in range(n_variants_per_seed):
      prompt = "Biến thể nhẹ của seed + inspiration từ seed khác"
      variant = light_model(prompt)
      evaluate(variant)

Phase 3 — Populate Archive:
  behaviors = extract_all_behaviors(valid_programs)
  centroids, labels = kmeans(behaviors, n_centroids)
  for each cell:
    insert best_program_per_cell into archive
```

### 13.3 σ₀ Baseline (SAL)

Sau init, capture standard deviation của accepted scores:
$$\sigma_0 = \text{std}(\text{accepted scores in init window})$$

Dùng làm baseline cho diagnostic.

---

## 14. Artifact Adapters

### 14.1 CodeAdapter

- `make_program()`: Wrap code string → Program
- `evaluate()`: Chạy `score_fn(exec'd function, inputs)` trong subprocess
- `build_mutation_prompt()`: Format code parents + meta-advice + SAL context
- `extract_candidate()`: Parse ```python block từ LLM response
- `build_paradigm_shift_prompt()`: Template early/mid/late + representative code

### 14.2 PromptAdapter

- Tương tự CodeAdapter nhưng cho text prompt
- Hỗ trợ **PromptBundle** (multi-component prompt: system, user, few-shot...)
- Component selector quyết định mutate component nào

---

## 15. Cấu Hình Đầy Đủ

### 15.1 SalConfig — Activate SAL Mode

```python
from levi.config import SalConfig

sal = SalConfig(
    enabled=True,               # ← BẬT SAL

    # Stagnation signal
    tau=80,                     # Plateau length trước khi saturate
    sigma_window=30,            # Window cho std diagnostic

    # Per-mechanism toggles
    enable_a_pe_staging=True,   # PE prompt staging
    enable_b_mutation_ctx=True, # Mutation context augmentation
    enable_c_meta_advice=True,  # Offensive/defensive meta-advice
    enable_d_thompson=True,     # Thompson Bandit over arms
    enable_e_hard_pe=True,      # Hard-PE khi stuck nghiêm trọng

    # Cơ chế A thresholds
    pe_staging_mid_threshold=0.3,
    pe_staging_late_threshold=0.7,

    # Cơ chế B
    context_threshold=0.5,      # s(t) để trigger context augmentation

    # Cơ chế D — Bandit
    bandit_w_min=0.05,          # Floor weight per arm
    bandit_new_best_bonus=0.5,  # γ for NEW BEST bias
    bandit_alpha_prior=1.0,
    bandit_beta_prior=1.0,

    # Cơ chế E — Hard-PE
    hard_pe_threshold=0.8,
    hard_pe_max_per_run=2,
    hard_pe_n_clusters=6,
    hard_pe_reasoning_effort="high",
)
```

### 15.2 Ví dụ đầy đủ với SAL

```python
import levi
from levi.config import SalConfig, PunctuatedEquilibriumConfig

result = levi.evolve_code(
    "Optimize bin packing to minimize wasted space",
    function_signature="def pack(items, bin_capacity):",
    seed_program=seed,
    score_fn=my_scorer,
    model="openai/gpt-4o-mini",
    budget_dollars=5.0,
    sal=SalConfig(enabled=True),
    punctuated_equilibrium=PunctuatedEquilibriumConfig(
        enabled=True,
        interval=10,
        n_clusters=3,
    ),
)
```

---

## Appendix A — Tóm tắt toán học quan trọng

| Công thức | Dùng ở đâu |
|-----------|------------|
| \( s(t) = \min(1, n/\tau) \) | Stagnation depth — tín hiệu trung tâm SAL |
| \( \text{UCB} = \bar{x} + c\sqrt{\ln T / n} \) | Chọn cell (exploration/exploitation) |
| \( P_i \propto \exp((f_i^{norm}-1)/T) \) | Softmax sampling |
| \( \theta \sim \text{Beta}(\alpha, \beta) \) | Thompson Sampling — chọn arm |
| \( z = (x-\mu)/\sigma \rightarrow \text{sigmoid}(z) \) | Behavior normalization |
| \( \text{cell}(p) = \arg\min_i \|b - c_i\|^2 \) | Gán program vào cell (nearest centroid) |
| \( T(t) = T_{min} + (T_{max}-T_{min})(1-\text{cycle}) \) | Cyclic annealing temperature |

---

## Appendix B — Sơ đồ Data Flow

```
User API                    Internal Engine
─────────                   ───────────────
evolve_code()         ──→   LeviConfig
  │                            │
  ├─ score_fn                  ├─ BehaviorExtractor
  ├─ model specs               ├─ CVTMAPElitesPool (archive)
  ├─ budget                    ├─ PipelineState (budget + metrics)
  └─ problem desc              ├─ Diversifier (init)
                               ├─ PipelineRunner (main loop)
                               │   ├─ llm_producer × N
                               │   ├─ eval_consumer × M
                               │   └─ PunctuatedEquilibrium
                               └─ LeviResult
```

---

## Appendix C — File-Level Dependency Graph

```
levi.py (entry point)
  ├── config/models.py (LeviConfig, SalConfig, BudgetConfig, ...)
  ├── pipeline/runner.py (PipelineRunner)
  │     ├── pipeline/producer.py (llm_producer)
  │     ├── pipeline/consumer.py (eval_consumer, meta-advice)
  │     └── pipeline/state.py (PipelineState, BudgetTracker, ClientGate)
  ├── pool/cvt_map_elites.py (CVTMAPElitesPool, Samplers, Bandit)
  ├── behavior/extractor.py (BehaviorExtractor, FeatureVector)
  │     └── behavior/features.py (AST feature functions)
  ├── equilibrium/equilibrium.py (PunctuatedEquilibrium)
  │     └── equilibrium/prompts.py (prompt templates + staging)
  ├── init/diversifier.py (Diversifier: init phase)
  ├── artifacts/{base,code,prompt}.py (ArtifactAdapter hierarchy)
  ├── selection/component.py (UCB/RR/Stagnation selectors)
  ├── prompts/{builder,bundle}.py (prompt construction)
  └── clients/{base,lm}.py (LiteLLM client abstraction)
```
