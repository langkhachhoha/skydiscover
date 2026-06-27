# SpecEvo: Speculative Scientific Discovery
### Báo cáo thiết kế framework (tái cấu trúc từ BLADE / "LiteEvo")

> Tài liệu này phân tích chi tiết framework cũ (mã nguồn `levi/levi/blade/`) và tái tổ chức
> nó thành 4 section theo hướng **speculative scientific discovery**, đặt tên mới là **SpecEvo**.
> Mọi cơ chế, con số, và prompt đều được trích **trực tiếp từ code thực tế** (không phải từ hình
> vẽ), nên có thể dùng làm xương sống cho method section của paper. Văn bản giải thích bằng tiếng
> Việt; tên component / section title / nguyên lý / prompt giữ nguyên tiếng Anh để bê thẳng vào paper.

---

## 0. Bối cảnh & câu chuyện trung tâm (đọc trước khi vào 4 section)

### 0.1. Một câu nguyên lý (the thesis)

> **Khám phá thuật toán *không đồng đều* về độ khó.** Phần lớn các bước cải tiến chỉ là *tinh
> chỉnh cục bộ* — "normal science" — mà một mô hình rẻ thừa sức đoán. Chỉ một *thiểu số* bước là
> *bước nhảy paradigm* thực sự khó. SpecEvo **suy đoán rẻ (speculate)** phần lớn các bước, và
> **chỉ leo thang (escalate)** lên mô hình frontier ở đúng những bước khó. Cơ chế trích dẫn được:
> **Speculate-then-Escalate**.

Đây là phát biểu giữ nguyên tinh thần trong [REFRAMING.md](REFRAMING.md): *mượn **ý tưởng** của
speculative decoding ("đoán rẻ trước, chỉ trả giá đắt khi đoán rẻ không đủ"), **không** mượn cơ
chế (draft/target/rejection sampling).* Discovery không có "phân phối target" để mà bằng — nên ta
**không** đổi tên cứng mọi thành phần thành proposer/verifier/rejection.

### 0.2. Hai mỏ neo trí tuệ (để câu chuyện "sâu", không chỉ là tiết kiệm tiền)

SpecEvo đứng trên giao của **hai** nguyên lý có sẵn trong văn liệu, và đó là điều khiến nó "có
nguyên tắc" chứ không phải một rổ kỹ thuật:

| Mỏ neo | Phát biểu | Ánh xạ vào SpecEvo |
|---|---|---|
| **Speculative decoding** (Leviathan et al., 2022) | Đừng trả giá đắt đồng đều cho mọi token; đoán rẻ, chỉ verify/đắt khi cần. | Mô hình rẻ đề xuất phần lớn các bước; frontier chỉ vào cuộc khi suy đoán rẻ cạn tác dụng. |
| **Kuhn — *Structure of Scientific Revolutions*** | Khoa học tiến hóa qua các giai đoạn dài **normal science** (tích lũy, cục bộ) bị ngắt quãng bởi các **paradigm shift** hiếm hoi (cách mạng). | Section 2 (Speculator) = normal science rẻ, ồ ạt. Section 3 (Navigator) = paradigm-scale reasoning đắt, hiếm. |

Chính chữ **"paradigm shift"** vốn đã nằm trong code (`paradigm_shift`, `paradigm class`), nên
khung Kuhn không phải gán ép — nó *đã* là ngôn ngữ của hệ thống. Khung speculative cho ta lý do
*kinh tế tính toán* cho hai tầng; khung Kuhn cho ta lý do *nhận thức luận* (epistemic) vì sao tiến
bộ khám phá vốn dĩ phân tầng như vậy. Hai khung củng cố lẫn nhau.

### 0.3. Bộ tên mới (đề xuất — đây là phần bạn yêu cầu đặt lại tên)

| Thành phần (tên cũ) | **Tên mới đề xuất** | Vai trò một câu | Lý do hợp "speculative" |
|---|---|---|---|
| Toàn framework (BLADE/LiteEvo) | **SpecEvo** | Speculative evolutionary discovery. | "Spec" = speculative; "Evo" = evolution. |
| Cơ chế lõi (để trích dẫn) | **Speculate-then-Escalate** | Đoán rẻ phần lớn, leo thang đắt khi cần. | Là danh từ riêng citable cho nguyên lý. |
| Small model (Section 2) | **Speculator** *(the Speculative Proposer)* | Mô hình rẻ liên tục tung ra hàng loạt *speculative refinements* theo nhiều hướng. | Nó "đoán" — fast, cheap, many, hữu ích kể cả khi sai (vì sinh evidence cho phase sau). |
| Frontier model (Section 3) | **Navigator** *(the Paradigm Navigator / Frontier Escalation)* | Mô hình đắt, hiếm, *điều hướng* search qua các move-class (synthesis / surgical / shift) khi suy đoán rẻ bế tắc. | Nó là nửa "Escalate"; "Navigator" vì nó chọn *hướng đi* ở ngã ba, không chỉ "shift". |
| Advice mode (Section 4) | **Advisor** *(the Meta-Advisor / Lessons Engine)* | Phản tư ở mức *quần thể*: chắt lọc bài học, lái suy đoán của Speculator. | Là "speculative prior" được học từ trajectory, rẻ hơn nhiều việc tự dò lại. |

> **Lưu ý dùng tên (theo cảnh báo trong REFRAMING.md):** đừng để reviewer tưởng ta tái hiện
> draft/target/rejection của speculative decoding. Vì thế **không** đặt tên "Draft model" /
> "Target model" / "Verifier". "Speculator" và "Navigator" giữ *tinh thần* mà không gợi cơ chế
> rejection-sampling.

### 0.4. Hai đính chính quan trọng so với hình vẽ & mô tả ban đầu của bạn

**(A) KHÔNG phải CVT-MAP-Elites.** Đây là điểm bạn yêu cầu làm rõ. Quần thể được quản lý bởi
`ClusterArchive` — docstring gọi là *"adaptive hybrid-behavior MAP-Elites"*. Khác biệt cốt lõi:

- **CVT-MAP-Elites** chia không gian hành vi thành các niche **cố định trước** (Centroidal Voronoi
  Tessellation tính một lần trên các điểm ngẫu nhiên *trước khi* search). Ranh giới niche **bất
  biến** suốt quá trình.
- **SpecEvo** thì: (1) descriptor hành vi là **học được, data-driven** (xem §1.3); (2) KMeans được
  fit **trên chính các chương trình đã khám phá**, và **re-fit định kỳ** mỗi `recluster_every = 30`
  admit. Tức là **ranh giới cụm bám theo search**, không áp đặt từ đầu.

→ **Trong paper nên viết:** *"an adaptive, periodically re-clustered MAP-Elites variant with a
learned hybrid behavior descriptor (AST + description-embedding), where niches are discovered
online by KMeans rather than fixed up-front as in CVT-MAP-Elites."* **Đừng** viết "CVT-MAP-Elites".

**(B) "Memory của frontier model" = Strategy Log.** Bạn nhớ đúng là có một bộ nhớ riêng cho
frontier. Trong code nó là `self.recent_trials` — một `deque(maxlen=6)` chứa 6 lần paradigm gần
nhất, mỗi dòng render dạng `[#idx] ✓/✗ score=… Δ=… :: description`. Nó được tiêm vào **mọi**
prompt paradigm dưới heading **`## Strategy Log`**, kèm chỉ thị thẳng *"Avoid any strategy whose
Strategy-Log entry has delta ≤ 0"*. Chi tiết ở §3.5.

---

## Section 1 — Init Phase: gieo mầm & dựng "bản đồ paradigm" ban đầu

**Mục tiêu nhận thức luận:** trước khi suy đoán rẻ có thể hữu ích, ta cần một **bộ khung paradigm
đa dạng và đã kiểm chứng** làm bệ phóng — vì "đoán rẻ cần ngữ cảnh tốt" (REFRAMING.md §5). Init
phase không phải thủ tục khởi tạo tầm thường; nó là bước *dựng không gian giả thuyết* để Speculator
có chỗ bám.

### 1.1. Đầu vào
Một `BladeConfig` gồm: `problem_description`, `function_signature`, `score_fn` (hàm fitness chạy
được), `fn_name`, `inputs`, và (tùy chọn) một `seed_program` do người dùng cấp.

### 1.2. Hai pha gieo mầm (đúng như code `_bootstrap_population`)

- **Phase 1 — Frontier gieo paradigm seeds (tuần tự).** Mô hình **frontier** (`paradigm_model`,
  mặc định `gpt-5`) viết `n_diverse_seeds = 5` seed. **Quan trọng:** sinh **tuần tự**, mỗi seed
  mới được *cho xem tất cả seed đã chấp nhận trước đó* (prompt `build_diverse_seed_prompt` → bọc
  `DIVERSITY_SEED_PROMPT`), nên mô hình bị đẩy về phía **paradigm mới** thay vì lặp lại. Mỗi seed
  có cơ chế retry tối đa 3 lần nếu parse lỗi / chạy lỗi.
  → *Vì sao frontier làm việc này, không phải small model?* Vì gieo *paradigm đa dạng* là bước
  "khó" điển hình — đúng chỗ đáng trả giá đắt.

- **Phase 2 — Speculator nở rộ variant (song song).** Với mỗi seed, mô hình **rẻ**
  (`mutation_model`, mặc định `qwen3-30b-a3b-instruct`) sinh **song song** `n_variants_per_seed = 20`
  biến thể (`build_init_variant_prompt`, mỗi prompt được tiêm 1–2 seed khác làm inspiration). Tức
  là sau init đã có tới ~`5 + 5×20 = 105` chương trình ứng viên (trừ các lần lỗi). → Đây là lần
  *speculation hàng loạt* đầu tiên: rẻ, rộng, đặt nền cho cluster.

### 1.3. Embedded Population — mã hóa mỗi chương trình thành vector hành vi

Mỗi `Program` được nhúng thành một **behavior vector** *hybrid* (đây chính là "AST vector +
Description Embedding → Individual representation" trong hình của bạn, nhưng có vài chi tiết hình
chưa thể hiện):

```
behavior_vec = [  AST features (14-d)  |  PCA(description embedding) (8-d)  ]   → 22-d
                 └ z-score (Welford) ┘   └────── z-score (Welford) ──────┘
```

- **AST half (14 chiều).** `compute_ast_features` đếm 14 đặc trưng cấu trúc: `ast_depth`,
  `cyclomatic_complexity`, `loop_count`, `loop_nesting_max`, `branch_count`, `function_def_count`,
  `comprehension_count`, `call_count`, `comparison_count`, `subscript_count`,
  `numeric_literal_count`, `math_op_count`, `import_count`, `code_length`. Tất cả được giảm chấn
  bằng `log1p` (để feature đếm-lớn như `code_length` không nhấn chìm feature đếm-nhỏ).
- **Embedding half (8 chiều).** Description (đoạn mô tả 2–4 câu kèm theo code) được embed bằng
  `text-embedding-3-small` (1536-d), rồi **PCA giảm xuống 8 chiều** (`embedding_dim = 8`), tính khi
  re-cluster.
- **Chuẩn hóa độc lập từng nửa** bằng thống kê Welford online *trước khi* ghép, để hai nửa không
  lệch thang nhau trong khoảng cách KMeans. Hai nửa có thể bật/tắt độc lập (`use_ast`,
  `use_embedding`) → đây là 2 trục ablation A1/A2.

> **Ý nghĩa sâu:** descriptor này nắm bắt *"chương trình này thuộc loại nào"* trên **hai trục bổ
> trợ**: *hình dạng tính toán* (AST — cấu trúc thuật toán) và *ý tưởng được phát biểu* (embedding —
> ngữ nghĩa mô tả). Hai bước cùng điểm fitness nhưng khác paradigm sẽ nằm ở hai cụm khác nhau →
> đúng tinh thần "diversity of paradigms" mà speculation cần.

### 1.4. Clustering & elitism (KMeans, không phải CVT)

- KMeans với `n_cells = 50` (cố định). Lần fit đầu khi đã đủ `min_admits_before_cluster = 16`
  chương trình; sau đó **re-cluster mỗi 30 admit** (warm-start từ centroid cũ).
- **Quy tắc nạp (admission):** một chương trình được nhận vào archive **iff điểm của nó > điểm của
  "cell incumbent"** (đại diện tốt nhất hiện thời của cụm nó rơi vào). Không có Hall-of-Fame, không
  grace period, không quota. Mỗi cụm chỉ giữ **một** phần tử tốt nhất (elitism) — đúng như "trong
  mỗi cụm chỉ giữ lại phần tử tốt nhất" bạn mô tả.
- **`force_add` (van thoát):** chỉ dùng cho paradigm mode = `shift` (xem §3) — nếu seed mới bị
  cụm từ chối, **đuổi chương trình yếu nhất toàn archive** và nạp seed vào. Lý do: khi đã bế tắc
  sâu, mất một slot yếu rẻ hơn việc bỏ lỡ cơ hội đổi paradigm.

> **Đính chính thuật ngữ cho hình:** ô "KMeans clustering" trong hình bạn để ở *giữa-cuối* pipeline
> (sau Try A/B/C). Thực tế clustering là **cơ chế thường trực** quản lý quần thể *xuyên suốt* (cả
> init lẫn main loop), không phải một bước rời ở cuối. Hình mới (§5) nên vẽ archive như một *vùng
> nền* mà mọi offspring chảy vào, kèm "re-cluster mỗi 30 admit".

---

## Section 2 — The **Speculator**: normal science rẻ, ồ ạt, động

> **Tên cũ "small model" → đề xuất "Speculator".** Đây là *engine of normal science*: nó không cố
> giải bài toán bằng một cú nhảy thiên tài, mà **liên tục tung ra hàng loạt suy đoán rẻ** theo nhiều
> hướng, để (a) thỉnh thoảng trúng một cải tiến, và (b) **sinh ra "evidence"** (admits, rejects,
> lỗi, độ lợi) nuôi cho Section 3 & 4. Giá trị của một suy đoán *không chỉ* nằm ở việc nó thắng —
> mà ở thông tin nó để lại.

### 2.1. Reframe đúng tinh thần bạn muốn
Bạn nói: *"tinh chỉnh code theo nhiều hướng khác nhau để khám phá, cung cấp thông tin hữu ích và
đầy đủ cho các phase sau."* Chính xác. Trong khung speculative, Speculator = **draft của normal
science**: rẻ, song song, đa hướng, và **động** (population thay đổi liên tục, không tĩnh).

### 2.2. "Try A / Try B / Try C" thực chất là gì
Trong hình, Try A/B/C trông như ba nhánh bí ẩn. Thực tế chúng chỉ là **các operator + template
prompt khác nhau** mà Speculator rút ngẫu nhiên ở mỗi bước (`_generate_one`):

- Tung đồng xu `p_crossover = 0.35`: **crossover** hay **mutate**.
- **Mutate** có **3 template** (rút *uniform* qua `PromptSampler`):
  1. `MUTATE_PROMPT_GENERAL` — cải tiến tổng quát.
  2. `MUTATE_PROMPT_FOCUSED_FIX` — sửa đúng **một** điểm yếu (surgical, cục bộ).
  3. `MUTATE_PROMPT_MECHANISM_SWAP` — thay **một** cơ chế (vd: greedy→Metropolis, fixed step→line
     search).
- **Crossover** có **2 template**: `CROSSOVER_PROMPT_STRUCTURAL` (lai cấu trúc) và
  `CROSSOVER_PROMPT_COMPONENT_SWAP` (ghép đúng một component từ "donor" vào "skeleton").
- **Targeted-mutate** (hướng dẫn): khi parent đã có *cached analysis* (xem §2.4) và đồng xu
  `p_targeted_mutate = 0.5` trúng → dùng `TARGETED_MUTATE_PROMPT`, bắt mô hình **chọn đúng một**
  đề xuất từ bản phân tích sẵn có thay vì bịa hướng mới.

> Mỗi prompt đều **bắt mô hình viết `## Analysis` trước** (Components / Strengths / Weaknesses /
> Plan) rồi mới viết code. Điều này ép Speculator *cam kết một giả thuyết* thay vì viết lại bừa —
> và bản Analysis này được tái dùng ở các pass sau. Đây là "speculation có lý lẽ", không phải nhiễu
> ngẫu nhiên.

### 2.3. Vòng lặp động & cách cluster "phán xử" offspring
- Main loop chạy `n_workers = 4` worker đồng thời, **liên tục cho tới khi hết budget**
  (`budget_dollars` / `budget_evals` / `budget_seconds` / `target_score` — chạm bất kỳ cái nào thì
  dừng).
- Mỗi offspring: parent được chọn bằng **Zipfian rank sampler** (xem 2.5) → sinh code → chấm điểm
  (process pool, sandbox) → gọi `archive.add()`. **Cluster chính là cơ chế chấp nhận:** offspring
  vào được archive *iff* nó vượt incumbent của cụm nó rơi vào; nếu không → `dropped_worse` (vẫn
  được ghi nhận là một eval + nuôi tín hiệu cho Advisor).
- **Re-cluster định kỳ** mỗi 30 admit: PCA + Welford + KMeans được fit lại, rồi *coalesce* (mỗi
  cụm giữ 1 tốt nhất). → "sau một số đánh giá nhất định sẽ recluster lại" mà bạn nói = **đếm theo
  admit, không phải theo eval**.

### 2.4. Analyzer — "analyse-then-mutate" (đề xuất 1 trong code)
Một background task chạy mỗi `analyzer_interval = 30` eval: duyệt archive theo điểm giảm dần, lấy
`analyzer_top_k = 3` chương trình **chưa có** analysis và gọi `mutation_lm` viết một bản review
ngắn (`ANALYSIS_PROMPT`: *algorithm summary → top-3 bottlenecks ranked by impact → 3 suggested
changes*, < 250 từ). Bản này **cache theo `id(parent)`**, chỉ bị xóa khi chương trình rời archive.
Chính sách *accumulating*: top-K đứng yên thì refresh #2 phân tích rank 4–6, #3 phân tích 7–9… cho
tới khi cả archive đều có analysis. Đây là "speculative prior cấp cá thể" để targeted-mutate bắn
trúng đích hơn.

### 2.5. Chọn cha mẹ: Zipfian rank sampler thích nghi theo stagnation
- Xác suất chọn chỉ phụ thuộc **hạng** (rank), không phụ thuộc khoảng cách điểm tuyệt đối →
  *scale-invariant*. `P(rank) ∝ rank^(−β)`.
- **β thích nghi theo stagnation:** `β = β_min + (1−s)(β_max − β_min)`, với `β_max = 2.0` (lúc
  tươi mới s=0 → exploit top-vài), `β_min = 0.3` (lúc bế tắc s=1 → trải rộng, explore đuôi).
- `select_two_parents` ưu tiên cha mẹ thứ hai ở **cụm khác** (phơi bày 2 paradigm cho crossover).
- `select_inspirations` lấy `k = 3` chương trình khác làm cảm hứng, **chỉ truyền description +
  score, không truyền code** → tránh sao chép, tránh phình token.

> **Ý nghĩa speculative:** ngay cả thuật điều khiển explore/exploit cũng *tự suy đoán* mức độ liều
> lĩnh dựa trên tín hiệu stagnation, thay vì cố định — rẻ và phản ứng nhanh.

---

## Section 3 — The **Navigator**: leo thang đắt, hiếm, có chủ đích (chi tiết)

> **Tên cũ "Frontier model / paradigm shift" → đề xuất "Navigator" (Frontier Escalation).** Đây là
> nửa **Escalate** của Speculate-then-Escalate. Nó **không chạy thường xuyên** — chỉ thức dậy theo
> nhịp, *verify* lại trạng thái quần thể, rồi quyết định một **move-class** tương xứng với mức bế
> tắc. Gọi là "Navigator" (không phải "paradigm shifter") vì *shift* chỉ là **một trong ba** hướng
> nó có thể chọn.

### 3.1. Khi nào Navigator thức dậy
Một background `_pe_monitor` kích hoạt **mỗi `pe_cron_interval = 50` eval** (tính từ *sau* init).
Nếu archive có ít hơn `paradigm_min_archive_size = 5` cụm chiếm dụng → bỏ qua (chưa đủ đại diện để
prompt có nghĩa). Mỗi lần thức dậy là **một** lần gọi frontier (đắt) + một đợt fanout rẻ.

### 3.2. "Verify lại quần thể" = đọc đại diện cụm + chọn move theo stagnation
Đúng như bạn mô tả ("sau mỗi lượt đánh giá cố định, verify lại quần thể; tùy stagnation mà chia số
cụm khác nhau để lấy code đại diện"). Cụ thể:

- **Stagnation** là tín hiệu lái mọi thứ: `stagnation = max(global, local)`, với
  `global = min(1, plateau_steps / 100)` (số eval kể từ NEW BEST gần nhất) và
  `local = min(1, admit_gap / 20)` (số eval kể từ admit gần nhất). Khoảng `[0,1]`.
- **Anchors** = đại diện **top-n cụm theo điểm** (truyền *full code + description + score*).
- **Inspirations** = các cụm *còn lại*, **chỉ truyền description + score** (để "không copy code và
  không quá nhiều context" như bạn nói).

### 3.3. Ba chế độ stagnation → ba action (đây là phần lõi, viết rõ)

Routing trong `_pick_paradigm_mode` (và bạn có thể ép cứng 1 mode qua ablation `paradigm_force_mode`):

| Stagnation `s` | Mode | Số anchors (cụm đại diện) | Action mô hình được yêu cầu | "First line" bắt buộc |
|---|---|---|---|---|
| `s ≤ 0.4` | **Synthesis** | 3 anchors gần điểm nhau + 5 inspiration | *Hybridise*: lấy 2–3 cơ chế cụ thể từ các anchor khác nhau, ghép thành một chương trình mạch lạc, vượt từng anchor. **Không** chỉ retune hằng số. | `MOVE: SYNTHESIS` |
| `0.4 < s ≤ 0.7` | **Surgical** | **1 anchor = champion** + 5 inspiration (chỉ description) | *Tune the champion*: một sửa đổi **cấu trúc cục bộ** chính xác (score-aware acceptance, repair tốt hơn, dùng slack/score signal sẵn có…). "Be the careful surgeon, not the wild inventor." | `MOVE: SURGICAL` |
| `s > 0.7` | **Shift** | 2 anchors | *Paradigm shift thật*: thiết kế **một lớp thuật toán khác hẳn** không xuất hiện trong anchor/inspiration. Cấm: chạy lại với hằng số mới, ghép sub-routine (đó là synthesis), đổi tên biến. Kèm **van thoát `force_on_drop`** (đuổi phần tử yếu nhất để nạp seed). | `MOVE: SHIFT` |

> **Vì sao map ngược trực giác (bế tắc sâu → Shift, bế tắc vừa → Surgical)?** Code ghi rõ kinh
> nghiệm: khi champion đã plateau, *xin frontier vá cục bộ tiếp hiếm khi thoát được*; cú **Shift**
> (paradigm class mới) mới là thứ phá vỡ plateau trong các run thực. Nên: **bế tắc sâu nhất ⇒ nhảy
> paradigm**; mức vừa (champion còn đà) ⇒ đánh bóng cục bộ để dồn lợi thế; còn tươi ⇒ tổng hợp các
> contender gần nhau. Đây là một quyết định thiết kế *phản trực giác nhưng có dữ liệu* — đáng nêu
> trong paper như một insight.

### 3.4. Sau khi frontier sinh seed: fanout rẻ (đẩy lại cho Speculator)
Seed paradigm (đắt) sau khi nạp được **nở `n_paradigm_variants = 4` biến thể** bằng mô hình rẻ,
với **nhiệt độ trải đều** từ exploit→explore (`[center−0.25, center+0.15]`) để 4 con không cùng
chỉnh đúng vài hằng số. → Một bước escalate đắt được **khuếch đại** thành nhiều suy đoán rẻ quanh
nó. Đây là nhịp "Escalate → Speculate" khép vòng.

### 3.5. Memory của Navigator = **Strategy Log** (bạn nhớ đúng có cái này)
`self.recent_trials` = `deque(maxlen=6)`: 6 paradigm gần nhất, mỗi dòng:
```
[#7] ✓ score=2.3610 Δ=+0.0042 :: <mô tả ngắn của paradigm đã thử>
[#6] ✗ score=n/a   Δ=n/a     :: <paradigm bị reject / parse-miss>
```
Block này được tiêm vào **mọi** prompt paradigm dưới `## Strategy Log (recent paradigm attempts)`,
kèm chỉ thị thẳng trong cả 3 mode: **"Avoid any strategy whose Strategy-Log entry has delta ≤ 0 —
that approach has already failed."** → Navigator **biết hướng nào đã thử và có hiệu quả không**,
nên không lặp lại ngõ cụt. (Đây đúng là "memory riêng cho frontier" bạn mô tả; nó *ngắn hạn, cuốn
chiếu* — 6 lần gần nhất — chứ không phải kho vĩnh viễn.)

### 3.6. Prompt đầy đủ của ba mode (để Section 3 thật cụ thể)

<details><summary><b>SYNTHESIS_PROMPT</b> (s ≤ 0.4)</summary>

```
# Paradigm Synthesis Challenge
## Problem … ## Function Signature …
## Archive Snapshot
The archive has run {n_evaluations} evaluations and currently occupies {n_cells} behavioural
cells. Stagnation level is {stagnation:.2f} (0 = just improved, 1 = stuck). The search is
**mildly stalled**: several anchors are close in score but no single mutation has combined
their strengths.
### Top anchors (close-in-score, full code) {anchor_block}
### Other paradigm inspirations (description + score only) {inspiration_block}
{strategy_log_block}
## Your Task
Your job is **synthesis**, not invention. Read the anchors and write ONE new program that
combines 2-3 concrete mechanisms drawn from different anchors into a structurally coherent
whole, beating each of them individually.
First, write a ## Analysis section. Its first line MUST be `MOVE: SYNTHESIS`. After that line,
include exactly these three sub-sections:
1. Component table. (Initialisation: from Anchor X … / Optimisation core: from Anchor Y … /
   Constraint repair: hybrid …)
2. Coherence note. (how borrowed components share data — variable layout, units, call ordering;
   avoid Frankenstein code)
3. Why this should beat all anchors. (one sentence per anchor)
Then write the program. Avoid any strategy whose Strategy-Log entry has delta ≤ 0 — that
approach has already failed. Do NOT just retune constants in one anchor.
### Critical requirements … (signature exact / imports / self-contained helpers)
```
</details>

<details><summary><b>PARADIGM_SHIFT_PROMPT</b> (s > 0.7)</summary>

```
# Paradigm Shift Challenge — Genuinely New Approach
## Archive Snapshot … Stagnation level is {stagnation:.2f} — the search is **moderately
stalled**, suggesting the current paradigm family has been mined out.
### Strongest paradigms currently in the archive (full code) {anchor_block}
### Other paradigm inspirations (description + score only) {inspiration_block}
{strategy_log_block}
## Your Task
Design a **fundamentally different algorithmic approach** — a paradigm class that does NOT
appear in any anchor or inspiration above. The new program's internal data structures, control
flow, and termination condition must all reflect the new paradigm.
Concrete forbidden moves:
- Re-running the same algorithm with new constants.
- Stitching a sub-routine from one anchor onto another (that is synthesis, not a shift).
- Renaming variables in an existing anchor.
First, write a ## Analysis section. Its first line MUST be `MOVE: SHIFT`. After that line,
include exactly these four sub-sections:
1. Paradigm name. (textbook name, e.g. "Lloyd relaxation", "Power diagram packing",
   "Lagrangian relaxation with subgradient ascent", "Branch-and-cut over a conflict graph")
2. Why this paradigm fits the problem. (two sentences; cite the specific problem feature)
3. Why current anchors miss it. (one sentence: what assumption the anchors share that this drops)
4. Risk. (one sentence: likeliest pitfall + how the code avoids it)
Then write the complete, runnable program. Avoid any strategy whose Strategy-Log entry has
delta ≤ 0. … (numpy/scipy only; implement the paradigm yourself)
```
</details>

<details><summary><b>SURGICAL_EXPLOIT_PROMPT</b> (0.4 < s ≤ 0.7)</summary>

```
# Surgical Exploit Challenge — Tune the Champion
## Archive Snapshot … Stagnation level is {stagnation:.2f} — the search is **deeply stalled**.
The same family of solutions has dominated for many evaluations, and previous paradigm attempts
have not produced improvements (see Strategy Log).
### Current champion (the ONLY anchor you target) {anchor_block}
### Top-ranked paradigm descriptions (for context only — code withheld) {inspiration_block}
{strategy_log_block}
## Your Task
A new paradigm will NOT help here — previous paradigm trials confirm that. What WILL help is a
**precise structural improvement** to the champion. Be the careful surgeon, not the wild inventor.
First, write a ## Analysis section. Its first line MUST be `MOVE: SURGICAL`. After that line,
four sub-sections:
1. Tightest constraint. (single mechanism most limiting the champion's score; cite exact
   function/loop/variable/update rule/repair step/acceptance rule/constant)
2. Structural fix. (exactly ONE local structural change; must change behaviour meaningfully —
   score-aware acceptance, better repair, use existing slack/score signal, targeted local polish)
3. Preservation list. (routines/constants/control-flow that stay unchanged)
4. Expected delta. (why this should improve score; main risk + guard: fallback/feasibility
   check/bounded step/deterministic randomness/NaN-Inf guard)
Then write the complete program. Implement exactly ONE structural, local fix. …
```
</details>

> *Lưu ý nhỏ về docstring:* trong comment cũ ở `prompts.py` có dòng mô tả routing "(low →
> synthesis, mid → shift, high → surgical)" — **đây là docstring lỗi thời, không khớp code**.
> Routing thực thi (trong `orchestrator._pick_paradigm_mode` + `BladeConfig`) là **low→synthesis,
> mid→surgical, high→shift** như bảng §3.3. Nên sửa lại docstring để khỏi gây nhầm khi viết paper.

---

## Section 4 — The **Advisor**: phản tư cấp quần thể theo **niche**, lái suy đoán (chi tiết + prompt)

> **Tên cũ "Advice mode" → giữ "Advisor" (Meta-Advisor / Lessons Engine).** Advisor tách thành
> section riêng vì nó làm việc ở **một tầng khác**: nó không sinh code, mà **đọc toàn bộ trajectory
> và viết ra "bài học"** rồi *tiêm ngược* vào prompt của Speculator. Trong khung speculative, Advisor
> là **prior học được** — thay vì để mỗi suy đoán rẻ tự dò lại, nó chắt lọc *"cái gì đang ăn / cái gì
> đã cạn / thử gì tiếp / tránh gì"* để các suy đoán sau **đoán trúng hơn**, tức tín hiệu *"khi nào
> đoán rẻ đã cạn tác dụng"* ở mức chiến lược (REFRAMING.md §5).
>
> **Ba thay đổi nền so với bản BLADE cũ** (đã implement trong code, đây là phần bạn yêu cầu sửa):
> 1. **Credit theo behavioural niche (archive cell), KHÔNG theo operator/template.** Operator chỉ là
>    template `PromptSampler` rút *uniform ngẫu nhiên* — một *process artifact* không mang tín hiệu
>    nhân quả về *loại thay đổi* nào work; nên Advisor không còn quan tâm admit thuộc source nào.
> 2. **"Improving vs saturated" quyết bởi *frontier của niche có dịch chuyển không*, KHÔNG bởi ngưỡng
>    Δ tuyệt đối.** Hằng số `1e-3` cũ là số tay-chỉnh cho một benchmark, không scale-invariant — bỏ hẳn.
> 3. **Lỗi được *tích luỹ thành tri thức* suốt run**, gom theo signature domain-agnostic — không còn
>    bảng keyword đặc thù circle-packing, không clear mỗi chu kỳ.

### 4.1. Đơn vị credit: niche + nội dung, không phải operator
- **Bỏ source/operator-tagging khỏi Advisor.** `mutate_focused_fix`, `crossover_component_swap`, … chỉ
  là template rút uniform; "cải tiến đến từ template X" là ngẫu nhiên, không actionable, và không phải
  *content*. (`source` vẫn giữ cho model-routing & run-log — chỉ **không** vào tín hiệu Advisor.)
- **Đơn vị mới = behavioural niche (cell của adaptive MAP-Elites) + nội dung mô tả.** "Hướng nào đang
  sinh lợi" được phát biểu bằng *thuật toán làm gì* (đọc từ description của incumbent), không bằng nhãn
  template.
- **LLM = verbalizer, không phải statistician.** Phần "đo" do vài counter rẻ, **xác định, tái lập được**
  (`_advisor_region_signals`); LLM chỉ chuyển bản đồ niche đó thành advice code-shaped.

### 4.2. Bốn tín hiệu vùng + cơ chế frontier-staleness (input)
Mỗi `meta_advice_interval = 50` eval, `_generate_meta_advice` đọc archive sống. Hai đại lượng rẻ,
**không ngưỡng**:

- **Frontier của một niche dịch chuyển lần cuối** = `created_at_eval` của *incumbent* cell đó (có sẵn
  trên `Program`; **bền vững qua re-cluster** vì gắn với object, không gắn với cell-id).
- **Độ "bận" của niche** = số *attempt* rơi vào cell trong cửa sổ (`_advisor_attempts_by_cell`), **đếm
  cả near-miss `dropped_worse`** — cell-id được archive gán *trước* khi so incumbent, nên near-miss vẫn
  có niche → **bỏ survivorship bias** của việc chỉ nhìn admit.

Với `window_start = eval_count − meta_advice_interval`, bốn bucket (trong `_advisor_region_signals`):

| Niche (đại diện bởi incumbent) | Điều kiện | Bucket |
|---|---|---|
| top-3 theo score (mỗi cái một cell khác → tự đa dạng) | — | **LEADING** |
| `incumbent.created_at_eval > window_start` (frontier vừa dịch) | ≤ 4 | **IMPROVING** |
| frontier **không** dịch trong cửa sổ **và** có ≥ 1 attempt (busy-but-stale) | top theo #attempt, ≤ 3 | **SATURATED** |
| occupied, không leader/improving, **ít attempt nhất** | ≤ 3 | **UNDER-EXPLORED** (nuôi TRY NEXT) |

> Đây là chỗ trả lời trực tiếp lo ngại *"dựa theo điểm có chuẩn không"*: ta **không** phân loại từng
> admit bằng độ lớn Δ; ta hỏi *"frontier của niche này có nhúc nhích không, và có đang bị cày không"*.
> Không ngưỡng, không phụ thuộc thang điểm benchmark. Tinh thần "effective / saturated / underexplored"
> mượn từ **SeaEvo's Strategic Landscape Navigation**, nhưng ở đây nó là một ước lượng *region-based,
> attempt-level, non-stationary* cụ thể thay vì định tính.

### 4.3. Error → tri thức tích luỹ (input, làm lại hoàn toàn)
- **Gom MỌI lỗi**, hai nhóm tự nhiên: *lỗi cấu trúc/code* (syntax, name/attr, type, shape…) và *lỗi
  không đạt yêu cầu hàm đánh giá* (timeout, lặp vô hạn, vi phạm ràng buộc, sai định dạng trả về…).
- **Key = `error_signature(msg)`**: lowercase, bỏ số & dấu câu, lấy ≤ 8 từ đầu → "Overlap between
  circles 0 and 2" và "… 3 and 5" gộp **một** entry đếm chung. *Recurring text chính là taxonomy* —
  không bảng keyword cứng, **portable** sang bài toán mới.
- **`_error_knowledge` gộp theo signature, persistent nhưng *bounded*** (`_ERROR_KNOWLEDGE_MAX = 24`):
  không reset mỗi chu kỳ, nhưng khi bảng đầy + có mode mới thì **evict mode hiếm nhất** (one-off cycle
  out, recurring sống & cộng dồn count). Vì output advice **cố định** (top-8, ví dụ cắt 160 ký tự) nên
  giữ một dict phình vô hạn là vô nghĩa — chỉ cần đủ entry để xếp hạng recurrence. **AVOID** đọc theo
  count giảm dần → ưu tiên failure mode tái diễn nhiều, bỏ qua lỗi lẻ một lần.

### 4.4. Output + cơ chế tiêm ngược
- Một block **đúng 4 mục, < 140 từ**: **WORKING** (cơ chế/cấu trúc mà IMPROVING niches & leaders chia
  sẻ — mô tả *thuật toán*, không nhãn template), **SATURATED** (niche busy-but-stale cần *giảm nhấn*,
  hoặc "none"), **TRY NEXT** (2–3 gợi ý *code-shaped*, xây trên WORKING, đẩy vào UNDER-EXPLORED, rời
  SATURATED), **AVOID** (anti-pattern từ tri thức lỗi tích luỹ, ưu tiên cái tái diễn).
- Advice (`current_meta_advice`) tiêm **verbatim** vào prompt mutate/crossover dưới `## Lessons learnt
  so far`, xác suất `meta_advice_inject_p = 0.35` mỗi prompt (~17 lần/50 eval — đủ nudge, không bão hòa,
  giữ đa dạng template). **Cố định** — bản này *không* dùng closed-loop đo-hiệu-lực-advice (giữ đơn giản
  theo yêu cầu).
- Cuối chu kỳ: **reset `_advisor_attempts_by_cell`** (mở cửa sổ mới), **không** đụng `_error_knowledge`
  (tích luỹ suốt run).
- Ablation `meta_advice_mode = "errors_only"`: bỏ hết region signals (success-side), chỉ giữ tri thức
  lỗi → đo đóng góp của phần success-side.

### 4.5. Prompt (excerpt) + hạn chế đã biết + mapping trước/sau

<details><summary><b>META_ADVICE_PROMPT</b> (region-based, verbatim từ code)</summary>

````
# Lessons-Learned Advisor

You are reviewing the search trajectory on this optimisation problem and writing a short
prescriptive note ... — it is to amplify what is working, name behavioural niches that have
*saturated* (still attract attempts but no longer improve), point at the next concrete thing to
try, and call out failure modes that are actually costing evaluations.

## Problem
{problem_description}
## Function signature
```python
{function_signature}
```
## Current state
- Best score so far: {best_score}
- Evaluations completed: {n_evaluations}
- Accept rate (last window): {accept_rate}
- Stagnation level: {stagnation_level} (0=fresh, 1=plateaued)

## Leaders — current niche champions (description + score)
{leaders_block}
## IMPROVING niches — frontier advanced this window (what is actually paying off)
{improving_block}
## SATURATED niches — many recent attempts but frontier stuck (mined out)
{saturated_block}
## Under-explored niches — few recent attempts (room to push)
{under_explored_block}
## Accumulated failure knowledge — every error seen so far, by recurrence
{error_knowledge_block}
## Previous advice (carried over so you can refine, not repeat)
{previous_advice_block}

## Your task
Write the new advice block using EXACTLY these four short sections, in this order, with these
literal headers and no other markdown:

WORKING: <... the concrete approach / structure / mechanism that the IMPROVING niches and leaders
share — read their descriptions and describe the algorithm itself, not any meta-process ...>
SATURATED: <... any behavioural niche / strategy family that keeps attracting attempts but no
longer improves — i.e. the SATURATED niches above. ... If no clear saturation, write "none".>
TRY NEXT: <2-3 short imperative suggestions ... Be code-shaped ... Build on WORKING, push into the
under-explored niches, and EXPLICITLY move away from SATURATED — do NOT propose more of the same.>
AVOID: <1-2 anti-patterns drawn from the accumulated failure knowledge above, prioritising the
most recurrent ones ...>

Total length: under 140 words. No preamble. No extra headers. No bullet characters ...
````
</details>

**Hạn chế đã biết (nên ghi honest trong paper):** `_advisor_attempts_by_cell` đếm theo `cell_id` *hiện
hành*; một lần re-cluster giữa chu kỳ có thể relabel cell-id → tín hiệu "bận" hơi nhoè giữa các nhãn.
Với re-cluster mỗi 30 admit và Advisor mỗi 50 eval (≤ ~1–2 lần/chu kỳ), nhoè bị chặn và chấp nhận được
cho một tín hiệu thô. Frontier-staleness thì *miễn nhiễm* vì bám `created_at_eval` của object incumbent.

**Mapping trước → sau (để viết được "what changed"):**

| | Trước (BLADE) | Sau (SpecEvo Advisor) |
|---|---|---|
| Đơn vị credit | operator/source (template ngẫu nhiên) | **niche (cell) + nội dung mô tả** |
| Mẫu đếm | chỉ admit (survivor) | **mọi attempt** (gồm near-miss `dropped_worse`) |
| Improving/Saturated | snapshot `\|Δ\| ≤ 1e-3` | **frontier dịch / busy-but-stale** (không ngưỡng) |
| Leading | top-3 score tĩnh | top-3 incumbents khác-niche (+ IMPROVING = rising) |
| Error | crash, 7 keyword đặc thù, clear mỗi chu kỳ | **mọi lỗi, signature domain-agnostic, gộp + bounded (≤24, evict lỗi hiếm)** |
| State / hàm | `_advisor_admits`, `_advisor_errors`, `classify_error` | `_advisor_region_signals`, `_advisor_attempts_by_cell`, `_error_knowledge`, `error_signature` |

---

## Section 5 — Đề xuất vẽ lại hình (làm rõ hướng speculative)

### 5.1. Vì sao hình cũ gây hiểu nhầm
- **Try A/B/C** trông như 3 nhánh đặc biệt — thực ra chỉ là các template operator rút ngẫu nhiên.
- **KMeans clustering** đặt ở giữa-cuối như một bước rời — thực ra là *cơ chế nền thường trực*.
- **Advice & Paradigm shift** bị trộn vào dòng chính, không thấy chúng là **hai tầng khác nhịp**
  (đắt/hiếm vs rẻ/liên tục).
- Không có gì thể hiện **tín hiệu stagnation** — thứ điều phối toàn bộ việc "khi nào leo thang".

### 5.2. Bố cục hình mới đề xuất (Speculate-then-Escalate)

Ý tưởng: **một vòng lặp rẻ ở dưới (Speculator) + một "thang máy" leo lên frontier ở trên
(Navigator), được kích bởi một đồng hồ Stagnation ở giữa; Advisor là bảng phản hồi bên cạnh bơm
"lessons" ngược vào vòng rẻ.** Archive là vùng nền mà mọi thứ chảy vào.

```
                ┌──────────────────────  ESCALATE (đắt, hiếm)  ───────────────────────┐
                │            NAVIGATOR  (frontier model · every 50 evals)             │
                │   stagnation gauge ──►  s≤0.4 SYNTHESIS · 0.4–0.7 SURGICAL · >0.7   │
                │   [≡ Strategy Log: 6 lần gần nhất, "avoid Δ≤0"]          SHIFT      │
                └───────▲────────────────────────────────────────────────┬───────────┘
   stagnation cao ─────┘ (leo thang khi suy đoán rẻ cạn)                  │ seed mới + 4 variant rẻ
                                                                          ▼
  ┌─ INIT PHASE ─┐   ┌──────────────────── SPECULATE (rẻ, liên tục) ─────────────────────┐
  │ frontier 5   │   │  SPECULATOR (small model · n_workers)                              │
  │ seeds  ───►  │──►│   pick parent (Zipfian, β←stagnation)                              │
  │ +20×/seed    │   │   ├ mutate ×3 templates  ├ crossover ×2  ├ targeted-mutate(analysis)│
  │ variants     │   │            │ offspring ──► evaluate ──► archive.add()              │
  └──────────────┘   └───────────────────────────────┬──────────────────────────────────┘
                                                      │ admits / rejects / errors / Δ
   ┌──────────── EMBEDDED POPULATION (adaptive MAP-Elites, KMeans, re-cluster /30 admits) ┐
   │  program → [AST 14d | PCA(desc-emb) 8d] = 22d → cell;  giữ best mỗi cell             │
   └───────────────────────────────────────────────┬───────────────────────────────────┘
                                                    │ top-K desc · improving/saturated · errors
                              ┌─────────────────────▼─────────────────────┐
                              │ ADVISOR (every 50 evals)                  │
                              │ WORKING · SATURATED · TRY NEXT · AVOID    │──┐ inject p=0.35
                              └───────────────────────────────────────────┘  │ "## Lessons"
                                          ▲                                   │
                                          └────────────── vào prompt Speculator ┘
```

### 5.3. Quy ước hình (để render đẹp)
- **Màu/độ cao = chi phí:** Speculator (xanh, ở dưới, mũi tên dày = nhiều lần gọi) ↔ Navigator
  (đỏ/tím, ở trên, mũi tên mảnh = hiếm). Trục dọc = "đắt dần".
- **Đồng hồ Stagnation** đặt ngay điểm nối: kim chỉ vùng synthesis/surgical/shift bằng đúng 3 màu
  (xanh→cam→tím như thanh "Synthesis/Surgical/Shift" trong hình cũ — giữ lại motif này, nó đẹp).
- **Archive** là dải nền ngang, vẽ các chấm nhiều màu gom cụm + nhãn "re-cluster /30 admits".
- **Hai memory** vẽ thành hai "thẻ ghi chú": Strategy Log (gắn Navigator) và Lessons (gắn Advisor)
  — và (mở rộng tuỳ chọn) có thể nối một mũi tên đứt từ các *SATURATED niches* của Advisor sang
  Navigator để nó tránh chính những niche đã được tuyên bố cạn.
- Bỏ nhãn "Try A/B/C"; thay bằng "operator templates ×(3 mutate / 2 crossover / targeted)".

> Tôi có thể render hình này thành một bản vẽ thực (SVG/HTML hoặc sơ đồ) nếu bạn muốn — chỉ cần nói.

---

## Section 6 — Bảng "vì sao mỗi component tồn tại" (khuôn REFRAMING: bất đối xứng X ⇒ buộc có Y)

| Component | Sự *không đồng đều* nào buộc nó ra đời | (KHÔNG nói "để tiết kiệm tiền") |
|---|---|---|
| **Two-tier (Speculator + Navigator)** | Độ khó các bước là *long-tailed*: đa số bước dễ, thiểu số rất khó. | Dùng frontier cho bước dễ là lãng phí; dùng model rẻ cho bước khó là vô vọng. |
| **Adaptive MAP-Elites (KMeans, hybrid descriptor)** | Suy đoán rẻ *trúng hơn hẳn* khi có kho lời giải đa dạng đã kiểm chứng làm bệ phóng. | Niche bám theo search (re-cluster) vì cấu trúc paradigm *chưa biết trước* — không thể áp đặt CVT từ đầu. |
| **Stagnation-routed 3-mode Navigator** | "Mức bế tắc" khác nhau cần *loại move* khác nhau, không phải cùng một cú. | Bế tắc sâu cần đổi paradigm; bế tắc vừa cần đánh bóng; còn tươi cần tổng hợp. |
| **Strategy Log (memory của Navigator)** | Mỗi cú escalate đắt; lặp lại ngõ cụt là phí kép. | Nhớ 6 lần gần nhất + "avoid Δ≤0" để mỗi cú đắt là một cú *mới*. |
| **Advisor (niche-based, 4-bucket)** | Cần tín hiệu *"khi nào suy đoán rẻ đã cạn ở mức chiến lược"* và *niche nào đáng khuếch đại / đã cạn*. | Credit theo niche + frontier-staleness (không operator, không ngưỡng Δ) là nơi tinh thần speculative lộ rõ mà không cần copy rejection rule. |

---

## Phụ lục — Bảng tham số mặc định (trích từ code, để viết Implementation Details)

| Nhóm | Tham số | Mặc định |
|---|---|---|
| Models | mutation / paradigm / embedding | `qwen3-30b-a3b-instruct` / `gpt-5` / `text-embedding-3-small` |
| Init | `n_diverse_seeds` / `n_variants_per_seed` | 5 / 20 |
| Archive | `n_cells` / AST dims / `embedding_dim` (PCA) / vector | 50 / 14 / 8 / 22-d |
| Archive | `min_admits_before_cluster` / `recluster_every` / `adaptive_recluster` | 16 / 30 admits / True |
| Main loop | `n_workers` / `p_crossover` | 4 / 0.35 |
| Operators | mutate templates / crossover templates / `p_targeted_mutate` | 3 / 2 / 0.5 |
| Analyzer | `analyzer_interval` / `analyzer_top_k` (accumulating) | 30 / 3 |
| Sampler | Zipfian `β_max → β_min` (theo stagnation) / `k` inspirations | 2.0 → 0.3 / 3 |
| Stagnation | `global = min(1, plateau/100)` · `local = min(1, admit_gap/20)` · `max(·,·)` | 100 / 20 |
| Navigator | `pe_cron_interval` / `paradigm_min_archive_size` / `n_paradigm_variants` | 50 / 5 / 4 |
| Navigator | synthesis ≤ `0.4` (3 anchors) · surgical ≤ `0.7` (1 anchor) · shift > `0.7` (2 anchors) | 0.4 / 0.7 |
| Advisor | `meta_advice_interval` / `inject_p` / `mode` (credit theo niche, không ngưỡng Δ) | 50 / 0.35 / `rich` |

---

### Việc nên làm tiếp khi viết paper
1. **Đổi nhãn nhất quán:** BLADE/LiteEvo → **SpecEvo**; small/frontier → **Speculator/Navigator**;
   nêu cơ chế **Speculate-then-Escalate**. (Memory dự án đang ghi tên cũ "LiteEvo" — cần thống nhất.)
2. **Sửa docstring routing lỗi thời** trong `prompts.py` (§3.6) để khỏi mâu thuẫn với code.
3. **Mở bài bằng quan sát bất đối xứng độ khó + Kuhn**, KHÔNG mở bằng "LLM đắt" (theo REFRAMING.md).
4. **Tránh tuyên bố "no-regret / phân phối y hệt"** — discovery không có phân phối target.
</content>
</invoke>
