# Levi — Giải thích bằng tiếng Việt (đọc 1 lần là hiểu)

Tài liệu này viết cho người đã chạy thử Levi và muốn hiểu rõ:
- Bộ máy Levi gồm những phần nào, ai làm gì?
- Khi nào dùng **mutation**, khi nào dùng **paradigm shift**?
- **Meta-advice** là gì, được tạo và dùng ra sao?
- Chọn parent (sampling) và lưu kho (archive) thế nào?

Cuối tài liệu là 1 hướng cải thiện thực tế nhất (không phá codebase, có khả năng ra paper) — bám vào mục tiêu mới của bạn: **tối ưu theo budget tiền/thời gian**, không phải theo số iteration.

---

## 1. Hình dung nhanh — Levi là cái gì?

Levi giống một **xưởng tiến hoá song song** với 2 loại thợ và 1 cái kệ:

```
                      ┌────────────────────────────────┐
                      │     KỆ ARCHIVE (CVT-MAP-Elites)│
                      │  N "ô" hành vi, mỗi ô giữ 1    │
                      │  ứng viên giỏi nhất (elite)    │
                      └────────────────────────────────┘
                          ▲          │
              chèn elite  │          │ rút parent ra
              mới vào ô   │          ▼
   ┌──────────────────────┴──┐    ┌─────────────────────────┐
   │   THỢ CHẤM ĐIỂM         │    │   THỢ ĐẺ ỨNG VIÊN       │
   │   (eval_consumer × N)   │    │   (llm_producer × N)    │
   │   - chạy code           │ ◄──┤   - chọn sampler        │
   │   - tính score          │    │   - build prompt        │
   │   - cập nhật kệ         │    │   - gọi MODEL NHỎ       │
   └─────────────────────────┘    │   - đẩy code vào queue  │
              ▲                   └─────────────────────────┘
              │
              │ thỉnh thoảng (interval=10 evals):
   ┌──────────┴──────────────────────────┐
   │   "ÔNG SẾP" PARADIGM SHIFT          │
   │   - cluster kệ thành n_clusters     │
   │   - lấy đại diện mỗi cluster        │
   │   - gọi MODEL LỚN (gpt-5...) đẻ     │
   │     ra cách giải KHÁC HẲN           │
   │   - chèn vào kệ                     │
   └─────────────────────────────────────┘
```

**Hai loại model** chạy song song:
| Vai trò | Model | Bao giờ chạy | Tỉ lệ chi phí |
|---|---|---|---|
| **Mutation** (rẻ) | qwen3-30b... | gần như mọi lúc | ~60% |
| **Paradigm shift** (đắt) | gpt-5 | định kỳ (mỗi 10 evals) | ~40% |

> Trong log circle_packing của bạn: 794 evals, $1.97 tổng. Khoảng $0.45 là init, $0.8 là PE (paradigm shift), còn lại là mutation.

---

## 2. Kệ Archive — chỗ lưu kết quả

### Ý tưởng

Không lưu “top-K điểm cao nhất” (sẽ bị clone hàng loạt), mà lưu **theo ô hành vi**: mỗi ô là một “phong cách thuật toán” khác nhau, ô nào cũng giữ ứng viên tốt nhất của ô đó.

### Cách tạo ô (CVT)

- Trích đặc trưng **hành vi** từ code: số vòng for, số nhánh if, độ lồng sâu, số phép toán, … (file [behavior/features.py](../levi/levi/behavior/features.py))
- Sau init phase, lấy đặc trưng của ~50 chương trình đầu tiên rồi chạy **KMeans** ra `n_centroids=50` tâm. Mỗi tâm = 1 ô.
- Khi có ứng viên mới → tính đặc trưng → tâm gần nhất = ô của nó.

### Chèn ứng viên

```
ứng viên mới ──► tính hành vi ──► ô gần nhất
                                      │
                       điểm > elite hiện tại?
                       ├── Có → thay thế (accepted, có thể là NEW BEST)
                       └── Không → bỏ (rejected)
```

> Ở circle_packing: kệ luôn 47 cells (đầy ngay sau init, không grow thêm) — nghĩa là cả run dài 3h, mọi thay đổi đều là **thay người trong ô có sẵn**.

---

## 3. Sampler — chọn parent kiểu gì?

Trước mỗi lần mutation, Levi phải chọn 1 elite trong kệ làm parent. Có 4 chiến lược (file [pool/cvt_map_elites.py](../levi/levi/pool/cvt_map_elites.py)):

| Sampler | Logic |
|---|---|
| **UCBSampler** | UCB1 — ô nào ít sample sẽ được ưu tiên (khám phá). |
| **SoftmaxSampler** | Xác suất ~ exp(score / T). Nhiệt T cao → đa dạng, T thấp → bám điểm cao. |
| **CyclicAnnealingSampler** | Đảo qua đảo lại T cao/thấp. |
| **UniformSampler** | Random đều. |
| **SubscoreSampler** | Sample theo metric phụ. |

**Mặc định**: Levi auto-sinh ra 4 cặp `(qwen, SoftmaxSampler@T)` với `T ∈ {0.3, 0.7, 1.0, 1.2}`. Cứ mỗi vòng producer rút 1 cặp theo `weight`, sau đó lấy parent từ sampler đó.

> Trong log: cả 2 lần NEW BEST đều rơi vào `qwen_T0.3` (nhiệt thấp = bám sát top). Các nhiệt cao chủ yếu sinh ra biến thể không hơn.

---

## 4. Mutation — chu trình thường xuyên nhất

**Producer** (`llm_producer`, file [pipeline/producer.py](../levi/levi/pipeline/producer.py)):

```
1. lấy lock kệ → chọn (sampler, model) → sample parent + inspirations
2. tuỳ chọn ghép thêm:
     - meta_advice  (xác suất 80%)
     - feedback từ failed test cases của parent
3. dựng prompt (full file hoặc diff)
4. gọi model NHỎ → nhận code mới
5. đẩy vào code_queue
```

**Consumer** (`eval_consumer`):

```
1. lấy item từ queue → gọi executor chạy (process pool, có timeout)
2. nếu cascade.enabled & có quick_inputs:
     - chấm điểm trên subset nhỏ trước
     - nếu thua threshold của elite cùng ô → bỏ (CASCADE SKIP, rẻ tiền)
3. điểm tốt thì:
     - tạo Program, lưu vào kệ
     - update sampler statistics (success rate của ô đó)
4. mỗi N evals → save snapshot.json
5. mỗi `meta_advice.interval=50` evals → trigger meta-advice (xem mục 6)
```

Toàn bộ chạy **bất đồng bộ**: `n_llm_workers=4` producer + `n_eval_processes=4` consumer chạy song song. Budget (dollar/eval/time) được kiểm tra trước mỗi lần gọi.

---

## 5. Paradigm Shift — “Ông sếp” bự định kỳ

**Khi nào fire?** Mặc định mỗi `interval=10` evals. Cụ thể trong [pipeline/runner.py](../levi/levi/pipeline/runner.py):

```python
if eval_count > 0 and eval_count % 10 == 0 and eval_count != last_pe_eval_count:
    trigger PE
```

(Trong thực tế PE tốn 4 eval của riêng nó + chờ LLM, nên khoảng cách thực tế thường là 40–70 evals, đúng như log của bạn: PE #1 ở eval 70, #2 ở 110, #3 ở 170, …)

**Quy trình PE** (file [equilibrium/equilibrium.py](../levi/levi/equilibrium/equilibrium.py)):

```
1. KMeans → cluster 47 elites thành n_clusters=3 nhóm hành vi
2. mỗi cluster lấy elite điểm cao nhất làm đại diện
3. gọi MODEL LỚN (gpt-5) với prompt "đây là 3 cách giải hiện tại,
                       hãy đề xuất 1 cách KHÁC HẲN" → 1 paradigm
4. chấm điểm paradigm → chèn vào kệ
5. nếu hợp lệ → gọi model nhỏ đẻ n_variants=3 biến thể quanh paradigm
6. chấm hết, chèn hết
```

**Mục tiêu**: thoát khỏi local optimum mà mutation nhỏ không nhìn ra.

> Trong log của bạn: 8 lần fire PE, chỉ 5 lần paradigm được nhận vào kệ, **không lần nào tạo NEW BEST**. Cả 2 NEW BEST đều đến từ mutation thường (eval 73, eval 396). Đây là tín hiệu rõ ràng PE “đắt mà không trả”.

---

## 6. Meta-advice — “bài học rút kinh nghiệm”

**Cứ mỗi `meta_advice.interval=50` evals**, consumer trigger:

```
1. gom thống kê 50 evals vừa rồi: số accepted/rejected/errors,
   các thông báo lỗi hay gặp
2. ghép với "bài học cũ" (lần meta-advice trước)
3. gọi 1 LLM (mặc định = model paradigm) với prompt yêu cầu:
   "Viết bài học ngắn 150-200 từ về cách tránh các lỗi trên"
4. lưu vào state.current_meta_advice
```

**Cách dùng**: producer ở mục 4 sẽ ghép `meta_advice` này vào prompt mutation với xác suất 80%. Mục tiêu là tránh sinh lại code mắc cùng một lỗi (e.g. "Overlap between circles 1 and 7", "operands could not be broadcast …").

> Meta-advice rất rẻ: log của bạn cho thấy mỗi lần chỉ ~$0.0002. Nó là phòng thủ thuần — chỉ nói “tránh lỗi”, không nói “học cái hay”.

---

## 7. Budget & cascade & snapshot — phần “quản gia”

| Thành phần | Chức năng |
|---|---|
| **BudgetTracker** | Giữ tổng cost, evals đã tiêu, thời gian đã trôi. Mọi worker đều check `budget_exhausted` trước khi làm. |
| **ClientGate** | Semaphore giới hạn concurrent LLM call. Khi budget cạn → tự đổi sang “serial mode”. |
| **CascadeConfig** | Cho phép chấm điểm 2 lớp: subset nhanh trước, nếu thua thì khỏi cần chấm full. (Mặc định tắt cho circle_packing.) |
| **ResilientProcessPool** | 1 ứng viên crash chỉ chết 1 worker, không kéo cả pipeline đổ. |
| **snapshot.json** | Mỗi 10 evals dump toàn bộ archive + score history + cost. Có thể resume bằng `resume_snapshot=load_json(...)`. |

---

## 8. Một bức tranh tổng cho 1 vòng đời

```
       ┌─────────── INIT PHASE (1 lần) ───────────┐
       │ 1. (nếu có) chấm seed_program             │
       │ 2. gpt-5 sinh n_diverse_seeds=4 hạt khác  │
       │    nhau về cấu trúc                       │
       │ 3. qwen sinh n_variants_per_seed=10/seed  │
       │ 4. chấm tất cả → KMeans tạo 50 centroids  │
       │ 5. lấp đầy archive                        │
       └───────────────────────┬───────────────────┘
                               │
                ┌──────────────▼──────────────────┐
                │      MAIN LOOP (async)          │
                │                                 │
                │   ┌──── mỗi eval (qwen) ────┐   │
                │   │ sample → mutate → eval  │   │
                │   │ → archive.add()         │   │
                │   └─────────────────────────┘   │
                │                                 │
                │   ┌── mỗi 50 evals (rẻ) ────┐   │
                │   │  meta-advice            │   │
                │   └─────────────────────────┘   │
                │                                 │
                │   ┌── mỗi 10 evals (đắt) ───┐   │
                │   │  paradigm shift (gpt-5) │   │
                │   │  + 3 variants (qwen)    │   │
                │   └─────────────────────────┘   │
                │                                 │
                │   stop khi: budget hết HOẶC     │
                │     best_score ≥ target_score   │
                └─────────────────────────────────┘
```

---

## 9. Đọc lại log circle_packing — Levi đang lãng phí ở đâu?

Tóm tắt số liệu run vừa rồi:

| Chỉ số | Giá trị |
|---|---|
| Budget | 10800s (3h), không giới hạn $/eval |
| Tổng evals | **794** |
| Tổng cost | **$1.976** |
| Best score | **2.6001944** |
| Archive size | 47 (không grow) |
| PE fire | 8 lần |
| Meta-advice | 13 lần |

Hai điểm rất đáng chú ý:

**(a) Best score chạm ở eval #396 (~50 phút), sau đó plateau hoàn toàn 400 evals nữa.**
```
Eval #73   → NEW BEST 2.5647   (sau ~20 phút)
Eval #396  → NEW BEST 2.6002   (sau ~50 phút)   ← đến đây là dừng
Eval #794  → vẫn 2.6002        (sau ~3h)
```
Nửa cuối run **không cải thiện gì** nhưng vẫn tiêu ~$1.0 và 2h.

**(b) Paradigm shift đắt nhưng không tạo bước nhảy.**
- 8 PE fire = ~$0.8 ≈ 40% tổng chi phí.
- 0/8 lần tạo NEW BEST. 3/8 lần paradigm bị reject thẳng.
- 5/8 lần “accepted” cũng chỉ chèn vào ô có sẵn, không phá kỷ lục.

→ Cảm nhận “Levi vẫn chưa tối ưu” của bạn có cơ sở định lượng rõ ràng: **một nửa budget bị tiêu vào giai đoạn không sinh ra giá trị mới**, và **chi phí PE đang không hoàn vốn**.

---

## 9.5. Đọc kĩ code — 5 “lỗ hổng tiềm năng” chưa được khai thác

Bốn quan sát ở mục 9 là **triệu chứng**. Khi đọc kỹ code, mình tìm ra **nguyên nhân** — và may mắn, mỗi cái đều sửa rất nhẹ.

### Lỗ hổng #1 — Prompt PE có 3 giai đoạn (early/mid/late) nhưng **chỉ early được dùng**

[equilibrium/prompts.py:146-152](../levi/levi/equilibrium/prompts.py#L146-L152):
```python
def get_budget_stage(budget_progress: float) -> str:
    """Always returns 'early' ..."""
    return "early"
```

Code có sẵn 3 prompt rất khác nhau (`early` = đập bỏ, làm cái mới; `mid` = lai ghép điểm mạnh; `late` = mổ xẻ điểm yếu cụ thể của best). Nhưng hàm trả về luôn “early”. Trong log của bạn, cả 8 PE — kể cả PE #8 khi archive đã chín — đều nhận prompt “hãy phát minh paradigm mới”. Đáng ra ở PE #6 trở đi nên là “tinh chỉnh điểm yếu cụ thể của best (2.6002)”.

### Lỗ hổng #2 — Mutation prompt **không bao giờ nhìn thấy best score**

[artifacts/code.py:114-137](../levi/levi/artifacts/code.py#L114-L137): `build_mutation_prompt` chỉ ghép `Problem + Signature + v1 (parent) + v2 (inspiration) + meta-advice`. Không có:
- `best_score_so_far`
- mã của elite tốt nhất (trừ khi sampler ngẫu nhiên chọn đúng nó)
- “score trajectory” gần đây

Hệ quả: model nhỏ làm mutation không biết “mình đang đứng cách kỷ lục bao xa”. Nó cứ sửa parent có sẵn, dễ rơi vào clone (như log cho thấy hàng loạt `score: 2.5415696271225667` lặp lại).

### Lỗ hổng #3 — Inspirations cùng sampler, không có “đối chứng”

[pipeline/producer.py:67-73](../levi/levi/pipeline/producer.py#L67-L73):
```python
n_parents = config.pipeline.n_parents + config.pipeline.n_inspirations  # = 1 + 1 = 2
sample = pool.sample(sampler_name, n_parents=n_parents, ...)
...
inspirations = [p for p in sample.inspirations if random.random() < 0.8]
```

Cả parent lẫn inspiration đều do **cùng 1 sampler** chọn. Với sampler softmax T=0.3 thì cả hai đều là elite điểm cao → context lặp lại, không có “con kém điểm khác hành vi để đối chiếu”. Producer cũng KHÔNG bao giờ chủ động ghép thêm **elite ở xa về hành vi**.

Đáng chú ý: hàm `pool.select_most_diverse(...)` đã tồn tại trong code (farthest-first traversal trên feature vector) — nhưng không được dùng trong producer.

### Lỗ hổng #4 — Meta-advice **chỉ phòng thủ**, không tấn công

[pipeline/consumer.py:34-65](../levi/levi/pipeline/consumer.py#L34-L65): prompt nói thẳng:
> “Focus ONLY on Failure Prevention. You do NOT see successful solutions. Your job is purely defensive.”

Trong 13 lần meta-advice của bạn, model chỉ học “tránh shape mismatch”, “tránh overlap”. Nó không bao giờ được cho biết “best score đã nhảy từ 2.564 → 2.600 nhờ chiến lược interior-point + Halton — hãy đẩy hướng đó”.

### Lỗ hổng #5 — Sampler weights tĩnh, không có bandit feedback

[pool/cvt_map_elites.py:643-655](../levi/levi/pool/cvt_map_elites.py#L643-L655): `get_weighted_sampler_config()` chọn theo weight cố định khai báo ở init. Levi auto-sinh 4 cặp `(qwen, softmax_T)` với `weight=1.0` đều nhau.

Trong log, **cả 2 NEW BEST đều rơi vào T=0.3**. T=1.2 (high-explore) gần như không tạo NEW BEST. Nhưng weight vẫn 1:1:1:1 từ đầu đến cuối. Đã có sẵn `update_sampler(name, cell, success)` — ghi đếm per-cell success — nhưng nó chỉ phục vụ UCB nội bộ, **không loop về lại weight giữa các sampler**.

### Còn 1 cờ chưa dùng

`pe_config.reasoning_effort` mặc định None → gpt-5 ở PE chạy reasoning mặc định. Code đã có nhánh `extras["reasoning_effort"] = ...` ([equilibrium/equilibrium.py:251-258](../levi/levi/equilibrium/equilibrium.py#L251-L258)). Khi tiền không quan trọng → bật `"high"` ở PE late-stage là “miễn phí” về độ rủi ro.

> **Kết**: Levi đang chạy với **prompt nghèo context, không có đối chứng, meta-advice phòng thủ thuần, weight cố định**, và **prompt PE “late” bị code chặn**. 5 điểm này không phải bug nghiêm trọng, chỉ là tiềm năng chưa khai thác. Mỗi cái sửa ~10–80 dòng. Đó là mảnh đất của hướng đề xuất bên dưới.

---

## 10. Hướng cải thiện đề xuất — *Stagnation-Adaptive Levi (SAL)*

> **Mục tiêu đúng của bạn** (đã sửa lại): tăng **chất lượng best score**, đặc biệt thoát local optimum (giảm cái “tail 400 evals không sinh ra gì” và để PE thật sự tạo bước nhảy). Tiền không quan trọng, **thời gian là ràng buộc**.

### Ý tưởng 1 câu
> Định nghĩa **một tín hiệu “độ đứng yên” $s(t) \in [0,1]$** rồi dùng nó để **điều khiển đồng thời 4 thứ**: chọn prompt PE, làm giàu context mutation, tái phân bổ weight sampler, và kích hoạt “PE phá rào” khi nặng quá.

Bốn cơ chế chia sẻ chung 1 tín hiệu — nên ablation rất sạch (bật/tắt từng cái) và mỗi cái độc lập về code.

### 10.1 Tín hiệu lõi — `stagnation depth` $s(t)$

Có 2 thành phần, hợp lại thành 1 số ∈ [0,1]:

$$s(t) = \underbrace{\min\!\left(1,\ \frac{n_{\text{since-best}}}{\tau}\right)}_{\text{độ dài plateau}} \cdot \underbrace{\exp\!\left(-\frac{\sigma_W(t)}{\sigma_0}\right)}_{\text{phương sai gần đây cạn}}$$

trong đó:
- $n_{\text{since-best}}$ = số eval từ lần NEW BEST gần nhất (đã có sẵn trong `state.score_history`).
- $\tau$ = ngưỡng plateau (ví dụ 80 evals).
- $\sigma_W(t)$ = độ lệch chuẩn của best-score trên cửa sổ $W$ eval cuối.
- $\sigma_0$ = chuẩn hoá ban đầu (ước lượng từ init phase).

Trực giác: $s\!\to\!0$ khi đang ăn điểm tốt; $s\!\to\!1$ khi cả hai điều xảy ra — *(a)* không có NEW BEST đã lâu *và* *(b)* score gần như đóng băng (phương sai sụt). Cách dùng tích (×) ép buộc CẢ HAI điều phải đúng — tránh dương tính giả khi mới chỉ một trong hai.

Tính $s(t)$ rẻ ($O(W)$), gọi mỗi 5–10 eval một lần. Lưu vào `state.stagnation_depth`.

### 10.2 Bốn cơ chế dùng chung tín hiệu này

#### **Cơ chế A — Sửa lỗ hổng #1: chọn prompt PE theo $s(t)$**

Thay [equilibrium/prompts.py:146-152](../levi/levi/equilibrium/prompts.py#L146-L152) một dòng:
```python
def get_budget_stage(budget_progress, stagnation=0.0):
    if stagnation < 0.3: return "early"   # đập bỏ, làm mới
    if stagnation < 0.7: return "mid"     # lai ghép
    return "late"                          # mổ xẻ điểm yếu cụ thể của best
```

Khi `s` cao, model PE được nhắc *“archive đã chín, hãy mổ xẻ điểm yếu của lời giải tốt nhất”* thay vì *“đập đi làm lại”*. Code sẵn rồi, chỉ bị tắt.

**Đính kèm**: nhét thêm vào prompt PE (~10 dòng trong `build_paradigm_shift_prompt`):
```
Best score: {best}
Evals since last best: {n_since_best}  
Stagnation depth: {s:.2f}
Per-example failures of current best: {top_3_failure_modes}
```

Để model gpt-5 thấy *“stuck ở 2.6002, các example còn fail là ...”*. Đây là tăng cường thông tin trực tiếp.

#### **Cơ chế B — Sửa lỗ hổng #2 & #3: context mutation có “đối chứng” khi $s$ cao**

Trong [pipeline/producer.py](../levi/levi/pipeline/producer.py), sửa khoảng dòng 65–75:
```python
parent = sample.parent
inspirations = list(sample.inspirations)

if s >= 0.5:
    # luôn nhét global-best (nếu nó không phải parent)
    best_elite = pool.best()
    if best_elite is not parent: inspirations.append(best_elite)
    # nhét 1 elite XA về hành vi (farthest-first)
    far = pool.select_most_diverse_from(parent, k=1)
    inspirations.extend(far)

parents = [parent] + inspirations[: min(3, len(inspirations))]
```

Tức là khi đang stuck, prompt mutation sẽ thấy:
- **v1**: parent từ sampler (như cũ)
- **v2**: global-best (luôn có, lỗ hổng #2 đã vá)
- **v3**: elite ở xa nhất về hành vi (đối chứng — lỗ hổng #3 đã vá)

Model nhỏ giờ có 1 context “tốt nhất + đối lập + parent”, dễ học “khoảng cách đến best là gì” hơn là chỉ thấy 2 elite na ná. Hàm `select_most_diverse` đã có sẵn, chỉ cần wrap thêm 5 dòng.

Có thể trộn thêm `best_score_so_far` vào header prompt (tăng 1 dòng).

#### **Cơ chế C — Sửa lỗ hổng #4: meta-advice 2 mặt (tấn công + phòng thủ)**

Hiện tại meta-advice mỗi 50 evals chỉ phòng thủ. Khi $s$ thấp (đang ăn điểm) → giữ nguyên prompt phòng thủ. Khi $s \ge 0.5$ → đổi sang prompt **tấn công**:

```
You are a strategist for a code-evolution system.
Recent best went from {best_prev} to {best_now}; stagnant for {n_since_best} evals.
Top-3 accepted improvements in this window came from these code patterns: {patterns}.
Current top failure modes on best: {top_failures}.

Suggest 3 specific algorithmic levers to explore next (NOT bug-fixes — strategic moves).
```

Để tạo `{patterns}`, đã có `score_history` ghi rõ accepted/rejected + sampler — chỉ cần thêm 1 tổng hợp nhỏ.

Meta-advice giờ là tín hiệu “đang đứng yên ở đâu, gợi ý hướng đi mới” thay vì chỉ “tránh lỗi cũ”. Cost vẫn ~$0.0002/lần (như log).

#### **Cơ chế D — Sửa lỗ hổng #5: Thompson sampling weight cho sampler**

Đây là **phần toán chính**. Thay weight cố định bằng **bandit Beta-Bernoulli** trên từng cặp `(sampler, model)`:

- State mỗi arm $i$: $(\alpha_i, \beta_i)$, khởi tạo $(1, 1)$ (Beta uniform prior).
- Reward định nghĩa **theo NEW BEST** (cho khớp với mục tiêu thoát local opt, không chỉ tỉ lệ accept):
$$r_i = \begin{cases} 1 & \text{nếu NEW BEST} \\ 0.3 & \text{nếu accepted thường} \\ 0 & \text{ngược lại} \end{cases}$$
- Cập nhật: $\alpha_i \mathrel{+}= r_i$, $\beta_i \mathrel{+}= (1 - r_i)$.
- Chọn arm: **Thompson sampling**, $\theta_i \sim \text{Beta}(\alpha_i, \beta_i)$, lấy $\arg\max_i \theta_i$.

Đặt weight ban đầu cho mỗi arm: $w_{\min} = 0.05$ (đảm bảo mọi arm vẫn có cơ hội). Hiệu chỉnh theo $s(t)$:

$$w_i = w_{\min} + (1 - w_{\min}) \cdot \frac{\theta_i^{1 + s(t)}}{\sum_j \theta_j^{1 + s(t)}}$$

Khi $s$ thấp ($\approx 0$): mũ = 1 → trộn nhẹ. Khi $s$ cao ($\to 1$): mũ = 2 → arm tốt nhất được nâng mạnh (vì đang stuck, cần ưu tiên hướng đã chứng minh ăn điểm).

Toán này:
- **Quy nạp Bayes chuẩn** (Beta-Bernoulli là conjugate prior cho phần thưởng nhị phân; soft-Bernoulli 0.3 vẫn hợp lệ).
- **Đảm bảo exploration**: $w_{\min}$ là floor; Thompson sampling tự nó cũng explore khi posterior chưa chắc.
- **Khớp với mục tiêu**: reward nặng cho NEW BEST → bandit ưu tiên cặp “hay tạo bước nhảy”, không phải cặp “hay accept lặt vặt”.

Cắm vào [pool/cvt_map_elites.py](../levi/levi/pool/cvt_map_elites.py): thêm `SamplerArm` (alpha, beta), sửa `get_weighted_sampler_config()` để dùng Thompson + reweight theo $s$. ~80 dòng.

#### **Cơ chế E (tuỳ chọn) — “PE phá rào” khi $s \ge 0.8$**

Khi 2 PE liên tiếp không tạo NEW BEST **và** $s \ge 0.8$:
- Tăng `n_clusters` $3 \to 6$ (đa dạng đại diện hơn).
- Chọn đại diện cluster bằng **farthest-first** thay vì max-score (đẩy gpt-5 đối diện những elite “lạ” chứ không phải elite điểm cao na ná).
- Bật `reasoning_effort="high"`.
- Dùng prompt `late` đã vá ở Cơ chế A.

Tiền không quan trọng, đây là “lá bài cuối” — kích hoạt thưa thớt nên không tốn nhiều.

### 10.3 Tóm lại đề xuất bằng 1 sơ đồ

```
                       ┌────────────────────────┐
                       │  TÍN HIỆU s(t)         │
                       │  (plateau × σ-collapse)│
                       └───────────┬────────────┘
                                   │
              ┌─────────┬──────────┼──────────┬──────────┐
              ▼         ▼          ▼          ▼          ▼
            [A]        [B]        [C]        [D]        [E]
       PE prompt   Mutation   Meta-advice  Sampler    Hard-PE
       early→mid   context     phòng thủ   weight     (n_clusters,
       →late       +best       ↔ tấn công  Thompson   farthest-rep,
                  +farthest                 (Beta)    reasoning=high)
```

Mỗi mũi tên đều **đọc trực tiếp dữ liệu sẵn có** (`score_history`, `pool.get_elites()`, `update_sampler` counts) — không phải thu thập thêm state mới.

### 10.4 Ablation và metric cho paper

| Setting | Prompt PE | Mutation ctx | Meta-advice | Sampler weight | Hard-PE |
|---|---|---|---|---|---|
| Baseline (Levi gốc) | early luôn | parent + 1 insp | phòng thủ | uniform | không |
| **+A** | A theo $s$ | — | — | — | — |
| **+A+B** | A | B | — | — | — |
| **+A+B+C** | A | B | C | — | — |
| **+A+B+C+D** | A | B | C | D (Thompson) | — |
| **Full (E)** | A | B | C | D | E |

Bài học: mỗi cơ chế *độc lập về cài đặt* nhưng *chia sẻ tín hiệu* — ablation kể được câu chuyện “thông tin → quyết định → bước nhảy”.

**Metrics chính** (đo trên cùng budget thời gian, tiền tự do):
1. **Best-score-cuối-run** (chủ chốt).
2. **Time-to-first-improvement-after-plateau** — xác suất phá local opt sau khi đã đứng.
3. **% PE tạo NEW BEST** (hiện 0/8 = 0%).
4. **Plateau-tail %** (hiện 50% — nửa run sau khi NEW BEST cuối).
5. **Behavior diversity của archive** (entropy occupancy) — kiểm tra D không “tham” quá.

**Benchmark đề xuất**:
- circle_packing (đã có sẵn dữ liệu baseline → so trực tiếp).
- 2 ADRS tasks (vd. transaction_scheduling, một bài graph).
- 3 seed, mỗi setting × benchmark.

Kỳ vọng định lượng (dựa vào pattern log):
- Cơ chế A + B đủ để tăng best score circle_packing từ 2.600 lên 2.61–2.62 (gap đến SOTA ~2.635).
- Cơ chế D giảm ~30–50% time-to-first-improvement nhờ Thompson bám T thấp hơn.
- Cơ chế E + late prompt: kỳ vọng có ít nhất 1 PE/run thực sự tạo NEW BEST (hiện 0%).

### 10.5 Lộ trình code (~2 tuần, không phá API)

| Ngày | Việc | File chính | LOC |
|---|---|---|---|
| 1 | Thêm `state.stagnation_depth` + hàm tính $s(t)$ + 1 unit test | [state.py](../levi/levi/pipeline/state.py) | ~40 |
| 2 | Cơ chế A: sửa `get_budget_stage` + nhét score-trajectory vào prompt | [prompts.py](../levi/levi/equilibrium/prompts.py), [code.py](../levi/levi/artifacts/code.py) | ~30 |
| 3 | Cơ chế B: gắn global-best + farthest-elite vào producer khi $s\ge 0.5$ | [producer.py](../levi/levi/pipeline/producer.py), [cvt_map_elites.py](../levi/levi/pool/cvt_map_elites.py) | ~50 |
| 4 | Cơ chế C: prompt meta-advice 2 chế độ + tổng hợp “top accepted patterns” | [consumer.py](../levi/levi/pipeline/consumer.py) | ~60 |
| 5–6 | Cơ chế D: Thompson Beta-Bernoulli bandit cho sampler weights | [cvt_map_elites.py](../levi/levi/pool/cvt_map_elites.py) | ~80 |
| 7 | Cơ chế E: hard-PE branch trong runner | [runner.py](../levi/levi/pipeline/runner.py), [equilibrium.py](../levi/levi/equilibrium/equilibrium.py) | ~40 |
| 8 | Cờ config thống nhất `levi_sal.{enabled, tau, sigma0, w_min, ...}` | [config/models.py](../levi/levi/config/models.py) | ~30 |
| 9–13 | Chạy ablation matrix (6 setting × 3 bench × 3 seed) | CI | — |
| 14 | Plot + draft workshop paper | — | — |

Tổng: **~330 dòng**, mỗi cơ chế có cờ riêng, mặc định OFF — đúng tinh thần “không phá codebase, dễ tích hợp”.

### 10.6 Vì sao hướng này là khả thi nhất

- **Mỗi cơ chế gắn vào 1 lỗ hổng đã chứng minh tồn tại** (mục 9.5), không phải lý thuyết suông.
- **Chia sẻ 1 tín hiệu** ($s(t)$) → câu chuyện paper gọn (“stagnation depth là chìa khoá”). Đúng kiểu workshop paper.
- **Toán nhỏ nhưng chuẩn** (Beta-Bernoulli + Thompson), không cần training, không cần thêm GPU.
- **Đối ngẫu với luận điểm gốc của Levi** (“diversity là vấn đề kiến trúc”): đề xuất bổ sung *“context và quyết định cũng là vấn đề kiến trúc — đặc biệt khi đang stuck”*.
- **Không động vào public API** (`evolve_code`, `LeviConfig` chỉ thêm trường con).

---

## 11. Tóm lại trong 5 dòng

- Levi = **kệ MAP-Elites theo hành vi** + **2 thợ song song** (mutation nhỏ rẻ, paradigm shift lớn đắt) + **gói lời khuyên rút kinh nghiệm** (meta-advice).
- Log circle_packing cho thấy điểm best đạt ở 13% thời gian, sau đó 87% budget không sinh thêm gì — và PE 0/8 lần tạo NEW BEST.
- Đọc kĩ code lộ ra 5 lỗ hổng chưa khai thác: prompt PE “late” bị code chặn, mutation prompt không có best score, inspirations không có đối chứng, meta-advice chỉ phòng thủ, sampler weights tĩnh.
- Đề xuất **Stagnation-Adaptive Levi (SAL)**: 1 tín hiệu lõi $s(t)$ điều khiển đồng thời 4 cơ chế (A–D), thêm cơ chế “phá rào” (E) khi quá nặng. Toán phụ trợ là **Thompson sampling Beta-Bernoulli** trên sampler weights, reward nặng cho NEW BEST.
- ~330 dòng code, không phá API, ablation 6-cell sạch, có dữ liệu kỳ vọng định lượng cụ thể từ chính log circle_packing của bạn.
