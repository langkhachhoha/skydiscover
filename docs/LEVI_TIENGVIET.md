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

## 10. Hướng cải thiện đề xuất (1 hướng khả thi nhất)

> **Mục tiêu bạn đã đặt lại**: tối ưu theo **tiền/thời gian** thay vì theo số iteration.
> Vậy bài toán đúng là: *“Mỗi đồng tiêu thêm, có còn tạo ra điểm số mới không?”*

### Tên gọi đề xuất: **Cost-Aware Levi** — bộ điều khiển ROI thời gian thực

#### Ý tưởng 1 dòng
> Quan sát “điểm tăng / chi phí” (ROI) trong cửa sổ thời gian gần đây, và **tự động** (a) giãn nhịp paradigm shift khi không hoàn vốn, (b) tái phân bổ trọng số sampler theo ROI, (c) dừng sớm khi ROI dự báo dưới ngưỡng.

#### Vì sao hướng này phù hợp?

- **Không phá codebase**: chỉ thêm 1 component mới (`ROIController`) trong [pipeline/state.py](../levi/levi/pipeline/state.py) + đọc thêm trong `_pe_monitor()` và `llm_producer`. ~200 dòng code, có cờ bật/tắt.
- **Trực tiếp khớp với mục tiêu mới (tối ưu theo cost/time)** — không phải “một sampler khác nữa”.
- **Có sẵn dữ liệu cho ablation**: log của bạn đã chứng minh tồn tại waste; chỉ cần lặp lại với controller bật và đo lại.
- **Câu chuyện paper rõ**: *“Knowing when to stop spending matters more than knowing how to spend”* — tự nhiên đối chiếu với chính luận điểm gốc của Levi (“diversity là vấn đề kiến trúc, không phải model”). Bài kết luận: *“cost-allocation cũng là vấn đề kiến trúc”*.

#### Cụ thể có 3 “nút” (knobs), tất cả đều dựa vào cùng 1 tín hiệu ROI

Định nghĩa **ROI gần đây** trong cửa sổ trượt $W$ giây hoặc $K$ đô-la:

$$\text{ROI}(t) = \frac{\Delta \text{best\_score trong cửa sổ}}{\Delta \text{cost trong cửa sổ}}$$

(Có sẵn `score_history` và `cumulative_cost` từng entry trong state — không cần thêm storage mới.)

**Knob A — Gate PE bằng ROI** (thay vì interval cố định)
```
trigger PE  ⇐  (eval_count - last_pe_eval_count ≥ min_gap)
        AND   (ROI gần đây < ε_pe  HOẶC  cell_entropy plateau)
```
PE chỉ fire khi mutation đang “đứng yên”. Khi mutation đang ăn điểm → khỏi tốn gpt-5.

**Knob B — Tái phân bổ sampler theo ROI**
Mỗi sampler-model-pair lưu thêm `cost_spent` và `best_score_delta_caused`. Sau mỗi cửa sổ, cập nhật weight ∝ EMA của (Δbest / $). Sampler nào hay tạo NEW BEST cho mỗi $ chi → nâng weight; sampler “rỗng” → giảm. Giữ floor để không xoá hẳn (vẫn explore).

**Knob C — Stop projection (kết thúc sớm)**
Khi ROI trượt $W$ lần liên tiếp dưới ngưỡng $\epsilon_{\text{stop}}$, *dự báo* rằng nửa budget còn lại có xác suất < $p$ tạo NEW BEST → fire stop event.
Bạn vẫn giữ snapshot, vẫn có “resume” nếu sau này muốn tiếp.

#### Ablation gọn, sạch cho paper

| Setting | PE trigger | Sampler weights | Stop |
|---|---|---|---|
| **Baseline** (Levi gốc) | fixed interval=10 | uniform | only when budget hit |
| **+A** | ROI-gated | uniform | budget |
| **+A+B** | ROI-gated | ROI-weighted | budget |
| **+A+B+C** (full) | ROI-gated | ROI-weighted | early-stop on flat ROI |

Chạy trên ít nhất 3 benchmark (circle_packing + 2 ADRS), 3 seed, **budget cố định theo $ và theo time** (chính là mục tiêu mới của bạn). Metrics:
1. Best-score-tại-budget (so sánh ở cùng $/time).
2. Cost-to-reach-X (tiền cần để chạm ngưỡng best score của baseline).
3. % chi phí dành cho PE.
4. Plateau-tail (% budget tiêu sau lần NEW BEST cuối).

Kết quả kỳ vọng (dựa vào pattern log circle_packing):
- Knob A: tiết kiệm ~30–40% chi phí PE mà không mất điểm.
- Knob B: tăng tốc đến điểm best (vì T=0.3 thường hot trên circle_packing, được nâng weight sớm).
- Knob C: cắt 30–50% tail không sinh giá trị → câu story chính của paper.

#### Lộ trình triển khai (~1–2 tuần)

1. **Ngày 1–2**: Thêm `ROIController` đọc từ `state.score_history`. Đưa cờ `cost_aware.enabled` vào [config/models.py](../levi/levi/config/models.py). Mặc định `False` (an toàn).
2. **Ngày 3**: Cắm Knob A vào `_pe_monitor()` (thay điều kiện `eval_count % interval == 0` bằng `controller.should_fire_pe()`).
3. **Ngày 4**: Cắm Knob B — chỉnh weight trong `pool.get_weighted_sampler_config()` theo controller. Giữ EMA, có floor.
4. **Ngày 5**: Cắm Knob C — emit `state.early_stop=True` khi ROI projection < ngưỡng; runner check ở `_wait_for_completion()`.
5. **Ngày 6–10**: Chạy ablation matrix (3 bench × 4 setting × 3 seed) trên CI; thu `summary.json`.
6. **Ngày 11–14**: Vẽ 4 plot chính, viết draft paper workshop (4–6 trang).

#### Vì sao không chọn các hướng trong [LEVI_RESEARCH_DIRECTIONS.md](LEVI_RESEARCH_DIRECTIONS.md)?

- **Direction 1 (CUSUM trigger cho PE)**: Tốt nhưng vẫn iteration-thinking. Cost-Aware Levi *bao* được nó (Knob A là superset của CUSUM + ROI-aware).
- **Direction 2 (warm-start centroids)**: Đòi hỏi matrix transfer n×n, khó kể chuyện gọn theo budget-thinking.
- **Direction 3 (LLM-proposed behavior axes)**: 4–6 tuần, rủi ro cao, không gắn trực tiếp với budget.

Cost-Aware Levi vừa là D1 mở rộng, vừa có 2 knob nữa, đúng tinh thần “tối ưu cost/time” bạn vừa nêu.

---

## 11. Tóm lại trong 5 dòng

- Levi = **kệ MAP-Elites theo hành vi** + **2 thợ song song** (mutation nhỏ rẻ, paradigm shift lớn đắt) + **gói lời khuyên rút kinh nghiệm** (meta-advice).
- Mutation chạy liên tục, PE fire định kỳ, meta-advice cập nhật mỗi 50 evals.
- Log circle_packing cho thấy điểm best đạt ở 13% thời gian, sau đó 87% budget không sinh thêm gì — và PE đang đốt 40% chi phí mà không tạo NEW BEST nào.
- Hướng cải thiện đề xuất: **Cost-Aware Levi** — gate PE bằng ROI, tái phân bổ sampler theo ROI, dừng sớm khi ROI cạn.
- Đây là 1–2 tuần code, không phá API, có ablation 4-cell sạch, và là cách tự nhiên nhất để chuyển bài toán từ “tối ưu theo iteration” sang “tối ưu theo $/giây” mà bạn vừa đặt.
