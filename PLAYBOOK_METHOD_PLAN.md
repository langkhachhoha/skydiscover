# PLAYBOOK — Prose-Evolved Policy with Semantic-Cluster Context Filtering

Tài liệu thiết kế cho một search method mới trong SkyDiscover, tên là **Playbook**.
Đi kèm là plan code chi tiết bám sát codebase hiện tại.

## Triggers

Hai vấn đề quan sát được khi chạy AdaEvolve / default discovery:

- **P1 — Static system prompt**: System prompt do người viết cứng và không cập nhật khi search-landscape thay đổi (vừa tìm được vùng promising, vừa stagnate, vừa phát hiện constraint mới).
- **P2 — Context collapse**: Candidate được sinh ra ở cả giai đoạn đầu lẫn về sau **giống nhau ở phần kết quả (outcome)** vì các program tham khảo (context) đưa vào prompt **trùng lặp về cả code lẫn outcome**. Sampler hiện tại trong [base_database.py:272-290](skydiscover/search/base_database.py#L272-L290) chỉ ưu tiên fitness cao → top-k luôn cluster quanh cùng một local mode → LLM không có signal để escape.

Hai component đối ứng:

1. **Component A — Evolved Prose Policy** giải P1: markdown doc ngắn, LLM tự revise mỗi K iteration, inject vào system prompt. Có quality gate (probe + accept rule) đảm bảo không đi xuống.
2. **Component B — Semantic-Cluster Context Filter (SCCF)** giải P2: filter context programs trước khi đưa vào prompt, bằng **LLM-based semantic clustering** kết hợp với **MMR-based representative selection**. Mượn ý tưởng cốt lõi từ paper MPaGE (Ha et al., AAAI 2026, §4.4) — *clustering bằng LLM tốt hơn AST/embedding/K-Means cho việc nhóm heuristics theo logic chứ không theo bề mặt syntax*.

Hai component **độc lập nhau** về cài đặt (ablation từng cái riêng được), nhưng share một insight chung: *thông tin LLM nhận mỗi iteration (system prompt + context programs) quyết định chất lượng search nhiều hơn cơ chế selection*.

Cấu trúc tài liệu:

- §1 Vị trí trong codebase, mượn gì từ đâu.
- §2 Component A — Evolved Prose Policy.
- §3 Component B — SCCF: semantic clustering (MPaGE-style) + MMR.
- §4 Toán nhẹ: MMR objective, log-window scorer, retention rule.
- §5 Kiến trúc tổng thể với file paths cụ thể.
- §6 Plan từng file (skeleton, line-level cụ thể).
- §7 Prompt templates.
- §8 Experimental protocol (multi-seed, multi-benchmark).
- §9 Feasibility, rủi ro, fallback.
- §10 Outline paper + ablation.

---

## §1. Vị trí trong codebase và origin của ý tưởng

| Method | Object evolved | Channel | Math |
|---|---|---|---|
| [adaevolve/](skydiscover/search/adaevolve/) | Island/selection weights | Sampling | UCB |
| [evox/](skydiscover/search/evox/) | Search algorithm class (code) | Database hot-swap | Log-window scorer |
| [fore/](skydiscover/search/fore/) | Per-parent fertility belief | Selection | NIG / Thompson |
| **`playbook/` (mới)** | **System prompt (prose) + context selector (semantic clusters + MMR)** | **Prompt + retrieval** | **MMR + log-window scorer + retain/revert** |

→ Playbook là *prompt-level* + *retrieval-level*. **Orthogonal** với selection-level method (`adaevolve`, `fore`) và procedure-level (`evox`). Có thể compose sau này.

### 1.1 Mượn gì từ đâu (declared, để paper review không cãi)

- **Log-weighted window scorer** (§4.2) cho quality gate của Playbook: dạng giống `evox/utils/search_scorer.py:LogWindowScorer`, đổi đối tượng (policy version thay vì search-strategy version).
- **LLM-based semantic clustering** (§3.2) cho SCCF: copy paradigm trực tiếp từ paper MPaGE §4.4 + Appendix E.3 (clustering prompt), nhưng đổi mục đích — MPaGE dùng để **chọn parent** cho cross-cluster crossover; ta dùng để **chọn context programs** cho prompt.
- **Cross-cluster vs within-cluster sampling** (§3.3): mượn dạng `o ← Mutate(h) if U[0,1] < γ else Crossover(h, h')` của MPaGE eq.(9), nhưng adapt cho việc *chọn context programs* (không phải sinh con).
- **MMR objective** (§4.1) cho within-cluster representative pick: Carbonell & Goldstein (1998), không phải mượn từ method nào trong repo. Đây là phần ta đóng góp riêng.
- **Code-similarity Jaccard + structural features**: reuse trực tiếp [adaevolve/archive/diversity.py:CodeDiversity](skydiscover/search/adaevolve/archive/diversity.py#L55).
- **Outcome-similarity trên `Program.metrics`**: reuse trực tiếp [adaevolve/archive/diversity.py:MetricDiversity](skydiscover/search/adaevolve/archive/diversity.py#L191).

### 1.2 Khác biệt với MPaGE cần nói rõ trong paper

MPaGE là **multi-objective** framework với PFG (Pareto Front Grid) cho objective space và clustering cho parent selection. Playbook khác ở 3 chỗ:

1. **Single-objective setting**: không có PFG, không có Pareto front. SCCF chỉ giữ phần *semantic clustering + cross-cluster sampling*; bỏ phần grid.
2. **Mục đích clustering khác**: MPaGE clustering để **đa dạng parent → đa dạng offspring**. Playbook clustering để **đa dạng context → giảm outcome collapse**. Cùng cơ chế, channel khác.
3. **Có thêm prose policy layer**: orthogonal với clustering, không có trong MPaGE.

---

## §2. Component A — Evolved Prose Policy

### 2.1 Playbook là gì
Một file markdown ~200–400 từ inject vào prompt mỗi iteration. Cấu trúc 5 section cố định (LLM bắt buộc tuân thủ; reviser prompt enforce):

```markdown
## Current focus
<vùng search hiện tại nên nhắm vào, 2–3 câu>

## What's working
- <observation rút từ top programs gần nhất>

## What's exhausted / avoid
- <pattern đã thử nhiều lần mà không cải thiện>

## Suggested moves
- <move cụ thể, e.g. "try numerical relaxation of constraint X">

## Context selection hint
- <gợi ý loại reference programs nên ưu tiên — feed vào SCCF>
```

Section cuối là **cầu nối A→B**: hint được parse heuristically và truyền vào SCCF qua kwarg.

### 2.2 Lifecycle
1. **t=0**: bootstrap v0 bằng 1 LLM call ([bootstrap.py](skydiscover/search/playbook/bootstrap.py), §6.2).
2. **mỗi K iter** (default K=10): build window summary → reviser sinh v' → probe v' trên 3 random parent → accept/reject theo rule §4.3.
3. **end**: persist toàn bộ versions ra disk làm artifact.

### 2.3 Pseudo-code

```python
playbook = bootstrap(initial_program, problem_desc, llm)
store.record(playbook, born_iter=0)

for t in range(max_iterations):
    parent, context_programs = db.sample(  # SCCF inside, see §3
        num_context_programs=k,
        playbook_hint=store.current().context_hint,
    )
    prompt = context_builder.build_prompt(parent, context={
        "other_context_programs": {"": context_programs},
        "program_metrics": parent.metrics,
    })
    # `context_builder` injects playbook into both system message and user message
    child = await llm.generate(prompt["system"], prompt["user"])
    score = evaluator.evaluate(child)
    db.add(Program(solution=child, metrics=score, parent_id=parent.id))

    if t > 0 and t % K == 0:
        cur_score = scorer.score(store.current(), db, window=K)
        store.close_window(store.current(), last_iter=t, score=cur_score)

        proposed = reviser.revise(store.current(), summary(db, t, K), llm)
        proposed_score = probe(proposed, db, n_probes=3, llm, evaluator, builder)
        if accept(cur_score, proposed_score, epsilon=0.1):
            store.record(proposed, born_iter=t)
        else:
            store.record_rejected(proposed, attempted_at=t)
            # keep current playbook
```

### 2.4 Cost overhead
- Probe-evaluate proposed playbook: $3$ LLM calls + $3$ evaluator calls mỗi K iter.
- Reviser: $1$ LLM call mỗi K iter.
- Bootstrap: 1 LLM call một lần.
- Total trên 100 iter, K=10: bootstrap(1) + reviser(10) + probe(30) = 41 LLM calls extra ≈ 40% overhead.

Nếu overhead này quá cao, set `probe_n=0` → quality gate degrade về "luôn accept" (mất bảo đảm monotonicity nhưng vẫn chạy được). Đây là một dial trong config.

### 2.5 Ablations cho A
- **A1**: K ∈ {5, 10, 20}.
- **A2**: Retain/revert ON vs OFF (probe_n=0).
- **A3**: Static playbook (v0 cố định) — đo xem evolution có giá trị không.

---

## §3. Component B — Semantic-Cluster Context Filter (SCCF)

### 3.1 Vấn đề chính xác

Hàm `sample()` mặc định của các database hiện tại ([adaevolve/database.py:496](skydiscover/search/adaevolve/database.py#L496), [base_database.py:272](skydiscover/search/base_database.py#L272)) chọn context programs theo fitness (top-k weighted). Hệ quả:

- Đầu run: context = các program đầu tiên random, gần như cùng outcome profile.
- Cuối run: context = top-k by fitness, cluster quanh local mode.

Cả hai trường hợp, **outcome distribution của children gần như đồng nhất**. Quan sát này nhất quán với MPaGE §1: *"prior LLM-based methods tend to produce populations of algorithms with similar operational logic and slight representation difference"*.

### 3.2 Tier 1 — LLM-based semantic clustering (MPaGE §4.4)

Mỗi K iter (cùng nhịp với playbook revise), lấy **elite set** $\mathcal{E}$ = top-N programs by fitness (default N=20). Gọi 1 LLM call cluster chúng theo **semantic logic** dùng prompt giống MPaGE Appendix E.3.

Output: $\mathcal{E}$ được phân vào $m$ clusters $\{C_1, \ldots, C_m\}$, mỗi cluster chứa các program có *logic tương đương* dù code khác nhau.

Cache labels: ghi `cluster_id` vào `Program.metadata["playbook_cluster_id"]` (sẵn có field từ [base_database.py:47](skydiscover/search/base_database.py#L47)). Programs đã có nhãn từ lần cluster trước được skip — chỉ cluster lại khi (a) đã đủ K iter mới, hoặc (b) số program chưa-nhãn > 50% elite.

**Vì sao LLM clustering** (so với AST/embedding/K-means): MPaGE §F + Table 8 chứng minh empirically rằng AST similarity và embedding-based clustering fail khi LLM-generated code có cùng logic nhưng diverse syntax — chính xác là setting của chúng ta. Đây là backing rất mạnh để dùng trong paper.

**Fail-safe**: nếu LLM clustering parse fail (JSON malformed), fall back về K-means trên `Program.metrics` vector (cheap). Log warning, không crash run.

### 3.3 Tier 2 — Cross-cluster context selection

Khi `sample(num_context_programs=k)` được gọi với parent $p$:

1. Xác định cluster của parent: $c_p = $ cluster_id của $p$ (nếu $p \notin \mathcal{E}$ → mark $c_p = \perp$).
2. Roll $\gamma \sim U[0, 1]$:
   - **Trường hợp "exploitation"** ($\gamma < \gamma_{\text{local}}$, default $\gamma_{\text{local}} = 0.3$): tất cả $k$ context lấy từ cùng cluster $c_p$ — MMR within-cluster (§4.1) để refine logic hiện tại.
   - **Trường hợp "exploration"** (ngược lại): $k$ context lấy từ $m - 1$ clusters **khác** $c_p$. Số program rút từ cluster $C_j$ tỉ lệ với weight $w_j = \max_{q \in C_j} \tilde{f}(q)$ (cluster có top fitness cao được ưu tiên), nhưng đảm bảo ít nhất 1 cluster khác được represent.
3. Trong mỗi cluster đã chọn (cả 2 trường hợp), pick representative bằng **MMR** (§4.1) trên (fitness × outcome-distance).

Trường hợp $c_p = \perp$ (parent ngoài elite): mặc định coi như exploration, sample đều từ tất cả clusters.

**Coupling với Playbook hint**: nếu playbook context_hint có chứa metric name (e.g. "prefer high recall"), MMR within cluster sẽ **upweight metric đó** trong outcome distance vector. Parser hint: simple keyword match với metric keys; không match → ignore. Đây là light coupling A→B, ablate được (§3.6).

### 3.4 MMR within cluster

Đã định nghĩa trong §4.1. Greedy pick cho đến đủ size yêu cầu, O(|C| · k_C).

### 3.5 Tương thích với database hiện tại

SCCF **không** thay parent selection — vẫn dùng cách của database base (e.g. AdaEvolve UCB hoặc default top-k). Chỉ override **việc chọn context programs**. Lý do: muốn ablate SCCF orthogonal với selection method, và để tránh đụng UCB của AdaEvolve.

→ Cài đặt: `PlaybookDatabase` kế thừa `TopKDatabase` (đơn giản nhất) và chỉ override `sample()` để hook SCCF trước khi return.

### 3.6 Ablations cho B
- **B1**: Disable Tier 1 — context vẫn chọn theo fitness top-k (đo baseline collapse).
- **B2**: Disable Tier 2 — chỉ cluster, không cross-cluster sampling (random pick within elite).
- **B3**: $\gamma_{\text{local}} \in \{0, 0.3, 0.5, 0.7\}$.
- **B4**: MMR α (code:outcome weight) ∈ {0, 0.5, 1} — đo riêng đóng góp của code-sim vs outcome-sim.
- **B5**: Disable hint coupling A→B.
- **B6**: Replace LLM clustering với AST clustering (như paper Yao et al. 2025) — kì vọng kém hơn, validate MPaGE §F.

### 3.7 Cost overhead của SCCF
- LLM clustering: 1 LLM call mỗi K iter (cached). 100 iter, K=10 → 10 calls.
- MMR computation: O(|C| · k) ≈ O(20 · 4) = 80 ops mỗi sample. Negligible.
- Total cộng với A: ~50 LLM calls extra trên 100 iter (50% overhead). Vẫn khả thi với gpt-5.

---

## §4. Toán nhẹ

### 4.1 MMR objective

Cho parent $p$, pool ứng viên $C$, đã chọn $S$, fitness normalized $\tilde{f} \in [0, 1]$:

$$
\mathrm{MMR}(q \mid S) = (1-\lambda) \cdot \tilde{f}(q) - \lambda \cdot \max_{r \in S \cup \{p\}} \mathrm{sim}(q, r)
$$

Greedy: tại mỗi bước, chọn $q^* = \arg\max_{q \in C \setminus S} \mathrm{MMR}(q \mid S)$, thêm vào $S$, lặp.

$\mathrm{sim}$ là weighted combination:
$$
\mathrm{sim}(q, r) = \alpha \cdot \mathrm{sim}_{\text{code}}(q, r) + (1-\alpha) \cdot \mathrm{sim}_{\text{outcome}}(q, r)
$$

- $\mathrm{sim}_{\text{code}}$: $1 - $ distance từ `CodeDiversity.distance` ([adaevolve/archive/diversity.py:83](skydiscover/search/adaevolve/archive/diversity.py#L83)).
- $\mathrm{sim}_{\text{outcome}}$: $1 - $ distance từ `MetricDiversity.distance` ([adaevolve/archive/diversity.py:240](skydiscover/search/adaevolve/archive/diversity.py#L240)).
- Default $\lambda = 0.5$, $\alpha = 0.5$.

Greedy MMR đạt (1 - 1/e)-approximation cho submodular maximization → 1 dòng cite Carbonell & Goldstein 1998 trong paper.

### 4.2 Log-weighted window scorer (cho Playbook quality gate)

Phiên bản $v_i$ sống trong window $W_i = [a_i, b_i]$. Score:

$$
\mathrm{score}(v_i) = \frac{\mathrm{improvement}(W_i)}{\sqrt{|W_i|}} \cdot \big(1 + \log(1 + f_{\text{start}}(W_i))\big)
$$

trong đó $\mathrm{improvement}(W_i) = f^{\max}_{b_i} - f^{\max}_{a_i}$ và $f_{\text{start}}(W_i) = f^{\max}_{a_i}$.

Diễn giải:
- $\sqrt{|W_i|}$: penalize window dài (improvement chậm).
- $1 + \log(1 + f_{\text{start}})$: thưởng cho version cải thiện một solution đã mạnh sẵn.

Form giống `evox/utils/search_scorer.py`, đối tượng đổi từ search-strategy sang prose-policy.

### 4.3 Retention rule

Tại transition $i \to i+1$, proposed $v'$ được probe trên $n_p = 3$ random parent:

$$
\mathrm{score}_{\text{probe}}(v') = \frac{1}{n_p} \sum_{j=1}^{n_p} \big[f(\text{child}_j) - f(p_j)\big]
$$

trong đó $\text{child}_j$ sinh ra từ $p_j$ dưới prompt có inject $v'$, evaluator được gọi đầy đủ.

Accept rule:
$$
\mathrm{accept}(v') \iff \mathrm{score}_{\text{probe}}(v') \ge (1 - \epsilon) \cdot \widehat{\mathrm{score}}(v_i)
$$

với $\widehat{\mathrm{score}}(v_i)$ = EMA của improvement gần nhất dưới $v_i$, $\epsilon = 0.1$.

**Proposition 1 (informal)**. Với $\epsilon \to 0$ và $n_p \to \infty$, expected score của chain $\{v_i\}_{i \ge 0}$ là non-decreasing.

Proof: trivial — accept iff non-decreasing trong expectation. 2 dòng paper.

### 4.4 Cross-cluster sampling probability (Tier 2)

Cho parent $p$ trong cluster $c_p$, số context programs $k$, số clusters $m$:

- Với prob $\gamma_{\text{local}}$: tất cả $k$ rút từ $c_p$.
- Với prob $1 - \gamma_{\text{local}}$: rút theo phân phối $\pi_j \propto w_j = \max_{q \in C_j} \tilde{f}(q)$ trên $j \ne c_p$, có constraint *unique-cluster*: số program từ cluster $j$ ≤ $\lceil k / (m-1) \rceil$ để tránh dồn.

Đây không cần proof — chỉ là rule, paper trình bày dạng pseudo-code.

---

## §5. Kiến trúc tổng thể (file-level)

```
skydiscover/
  search/
    base_database.py                  ← KHÔNG SỬA
    default_discovery_controller.py   ← KHÔNG SỬA (subclass)
    registry.py                       ← KHÔNG SỬA
    route.py                          ← MODIFY: thêm 4 dòng đăng ký 'playbook'
    adaevolve/archive/diversity.py    ← KHÔNG SỬA (reuse CodeDiversity, MetricDiversity)
    topk/database.py                  ← KHÔNG SỬA (PlaybookDatabase kế thừa TopKDatabase)
    playbook/                         ← MỚI
      __init__.py
      controller.py        # PlaybookController(DiscoveryController)
      database.py          # PlaybookDatabase(TopKDatabase)
      playbook_store.py    # PlaybookVersion + PlaybookStore (state + persistence)
      reviser.py           # LLM-driven revise step
      bootstrap.py         # initial playbook generation
      scorer.py            # log-weighted window scorer (§4.2)
      retention.py         # probe + accept rule (§4.3)
      sccf.py              # SCCF orchestration: tier1+tier2
      clustering.py        # LLM semantic clustering with cache
      mmr.py               # MMR pick within cluster
      summary.py           # window summary builder for reviser
      config/
        bootstrap_sys_prompt.txt
        revise_sys_prompt.txt
        clustering_sys_prompt.txt   # adapted from MPaGE Appendix E.3
      templates/
        full_rewrite_user_message.txt   # = default + {{playbook}} block
        diff_user_message.txt
  context_builder/
    base.py                           ← KHÔNG SỬA
    default/                          ← KHÔNG SỬA
    playbook/                         ← MỚI
      __init__.py
      builder.py           # PlaybookContextBuilder(DefaultContextBuilder)
      templates/           # copy + tune từ default
  config.py                           ← MODIFY: thêm PlaybookDatabaseConfig
configs/playbook.yaml                 ← MỚI
tests/
  test_playbook_store.py              ← MỚI (no LLM)
  test_playbook_scorer.py             ← MỚI (no LLM)
  test_mmr.py                         ← MỚI (no LLM)
  test_sccf_integration.py            ← MỚI (LLM stubbed)
```

Số file mới: 18. Số file sửa: 2 (`route.py`, `config.py`).

---

## §6. Plan từng file (skeleton, code-ready)

### 6.1 `playbook/playbook_store.py`

State container, không phụ thuộc LLM. Test bằng pytest trực tiếp.

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json

@dataclass
class PlaybookVersion:
    id: int
    text: str
    born_iter: int
    closed_iter: Optional[int] = None
    parent_id: Optional[int] = None
    score: Optional[float] = None
    accepted: bool = True
    revision_metadata: dict = field(default_factory=dict)
    context_hint: str = ""              # parsed from "## Context selection hint"

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict): ...


class PlaybookStore:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.versions: list[PlaybookVersion] = []
        self._rejected: list[PlaybookVersion] = []
        self._ema_score: Optional[float] = None

    def current(self) -> PlaybookVersion: ...
    def record(self, text, born_iter, accepted=True) -> PlaybookVersion: ...
    def record_rejected(self, text, attempted_at) -> None: ...
    def close_window(self, version, last_iter, score) -> None: ...
    def update_ema(self, observed_score: float, alpha: float = 0.3) -> None: ...
    def ema_score(self) -> float: ...
    def window_iters(self, vid) -> tuple[int, int]: ...
    def _persist(self, v: PlaybookVersion) -> None:
        path = self.out_dir / f"playbook_v{v.id:03d}.json"
        path.write_text(json.dumps(v.to_dict(), indent=2))
```

### 6.2 `playbook/bootstrap.py`

```python
from pathlib import Path

def bootstrap_playbook(initial_program_code: str,
                       problem_description: str,
                       llm) -> str:
    """1 LLM call → playbook v0. Uses bootstrap_sys_prompt.txt."""
    sys_prompt = (Path(__file__).parent / "config/bootstrap_sys_prompt.txt").read_text()
    user_msg = (
        f"<problem>\n{problem_description}\n</problem>\n\n"
        f"<initial_program>\n{initial_program_code}\n</initial_program>"
    )
    resp = await llm.generate(sys_prompt, user_msg, temperature=0.5)
    return _extract_markdown(resp.text)


def _extract_markdown(s: str) -> str:
    """Robust: strip code fence if model wraps output in ```markdown ... ```"""
    ...
```

### 6.3 `playbook/clustering.py`

```python
import json
from pathlib import Path

CLUSTER_KEY = "playbook_cluster_id"
CLUSTER_VERSION_KEY = "playbook_cluster_version"  # increments each re-cluster


class SemanticClusterer:
    def __init__(self, llm, fallback_clusters: int = 3):
        self.llm = llm
        self.fallback_clusters = fallback_clusters
        self._cache_version: int = 0
        self._sys_prompt = (Path(__file__).parent / "config/clustering_sys_prompt.txt").read_text()

    async def cluster(self, elite: list[Program]) -> dict[str, int]:
        """Returns {program_id -> cluster_id}. Sets program.metadata[CLUSTER_KEY] too."""
        # Skip if all elite already have cluster_id from this version
        unclustered = [p for p in elite
                       if p.metadata.get(CLUSTER_VERSION_KEY) != self._cache_version]
        if not unclustered:
            return {p.id: p.metadata[CLUSTER_KEY] for p in elite}

        self._cache_version += 1

        # Build prompt with code snippets (truncated to 1500 chars each)
        code_blocks = "\n\n".join(
            f"<Code idx={i}>\n{p.solution[:1500]}\n</Code>"
            for i, p in enumerate(elite)
        )
        user_msg = code_blocks
        try:
            resp = await self.llm.generate(self._sys_prompt, user_msg, temperature=0.0)
            parsed = self._parse_json(resp.text)   # robust parser
            mapping = self._labels_from_parsed(parsed, elite)
        except Exception as e:
            logger.warning(f"LLM clustering failed ({e}), falling back to metric K-means")
            mapping = self._fallback_kmeans(elite)

        for p in elite:
            p.metadata[CLUSTER_KEY] = mapping[p.id]
            p.metadata[CLUSTER_VERSION_KEY] = self._cache_version
        return mapping

    def _parse_json(self, text: str) -> dict: ...
    def _labels_from_parsed(self, parsed: dict, elite: list[Program]) -> dict[str, int]: ...
    def _fallback_kmeans(self, elite: list[Program]) -> dict[str, int]: ...
```

Reference: MPaGE Appendix E.3 prompt format. Our `clustering_sys_prompt.txt` adapts that verbatim.

### 6.4 `playbook/mmr.py`

```python
from skydiscover.search.adaevolve.archive.diversity import CodeDiversity, MetricDiversity

class MMRSelector:
    def __init__(self, lam: float = 0.5, alpha: float = 0.5,
                 metric_keys: Optional[list[str]] = None):
        self.lam = lam
        self.alpha = alpha
        self.code_div = CodeDiversity()
        self.metric_div = MetricDiversity()

    def select(self,
               parent: Program,
               candidates: list[Program],
               k: int,
               hint_boost: Optional[dict[str, float]] = None) -> list[Program]:
        if not candidates:
            return []
        # Update MetricDiversity normalizers
        self.metric_div.update(candidates + [parent])
        f_norm = self._normalize_fitness(candidates)
        S = []
        pool = list(candidates)
        while len(S) < k and pool:
            best, best_score = None, -float("inf")
            for q in pool:
                sim_to_S = max((self._sim(q, r, hint_boost) for r in S + [parent]), default=0.0)
                score = (1 - self.lam) * f_norm[q.id] - self.lam * sim_to_S
                if score > best_score:
                    best, best_score = q, score
            S.append(best); pool.remove(best)
        return S

    def _sim(self, a: Program, b: Program, hint_boost) -> float:
        code_sim = 1.0 - self.code_div.distance(a, b)
        outcome_sim = 1.0 - self._weighted_metric_distance(a, b, hint_boost)
        return self.alpha * code_sim + (1 - self.alpha) * outcome_sim

    def _weighted_metric_distance(self, a, b, hint_boost) -> float:
        # If hint_boost provided, scale specific metric contributions
        ...

    def _normalize_fitness(self, candidates) -> dict[str, float]: ...
```

Unit-test fixture: 5 synthetic programs với metrics khác nhau → MMR(k=2) phải pick fitness top + diverse outcome, không phải hai program fitness top cùng cluster outcome.

### 6.5 `playbook/sccf.py` (orchestrator)

```python
class SCCF:
    """Semantic-Cluster Context Filter."""
    def __init__(self, clusterer: SemanticClusterer, mmr: MMRSelector,
                 gamma_local: float = 0.3,
                 elite_size: int = 20):
        self.clusterer = clusterer
        self.mmr = mmr
        self.gamma_local = gamma_local
        self.elite_size = elite_size

    async def maybe_recluster(self, db: ProgramDatabase) -> None:
        """Called by controller every K iter. Idempotent if cache fresh."""
        elite = db.get_top_programs(n=self.elite_size)
        await self.clusterer.cluster(elite)

    def select_context(self, parent: Program, db: ProgramDatabase,
                       k: int, hint: Optional[dict] = None) -> list[Program]:
        elite = db.get_top_programs(n=self.elite_size)
        clusters = self._group_by_cluster(elite)        # {cid: [Program]}
        c_p = parent.metadata.get(CLUSTER_KEY)
        if random.random() < self.gamma_local and c_p in clusters:
            pool = clusters[c_p]
            return self.mmr.select(parent, pool, k=k, hint_boost=hint)
        else:
            # cross-cluster: distribute k across clusters != c_p, weighted by max-fitness
            return self._cross_cluster_pick(parent, clusters, c_p, k, hint)

    def _cross_cluster_pick(...): ...
    def _group_by_cluster(...): ...
```

### 6.6 `playbook/database.py`

```python
from skydiscover.search.topk.database import TopKDatabase

class PlaybookDatabase(TopKDatabase):
    """Wrap default sampling with SCCF on the context side."""
    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.sccf: Optional[SCCF] = None         # controller injects it after creation
        self._current_hint: Optional[dict] = None  # controller updates each iter

    def sample(self, num_context_programs: int = 4, **kwargs):
        # Parent selection: unchanged (TopK behavior)
        parent, _default_context = super().sample(num_context_programs=num_context_programs)
        # Override context via SCCF, if attached
        if self.sccf is not None:
            actual_parent = parent if isinstance(parent, Program) else list(parent.values())[0]
            context = self.sccf.select_context(
                actual_parent, self, k=num_context_programs, hint=self._current_hint,
            )
            return parent, context
        return parent, _default_context
```

### 6.7 `playbook/scorer.py` + `retention.py`

```python
# scorer.py
import math

def window_score(version: PlaybookVersion, db: ProgramDatabase) -> float:
    a = version.born_iter
    b = version.closed_iter if version.closed_iter is not None else a + 1
    if b <= a:
        return float("nan")
    f_a = db.best_fitness_up_to(a)
    f_b = db.best_fitness_up_to(b)
    improvement = f_b - f_a
    f_start = max(f_a, 0.0)
    return (improvement / math.sqrt(b - a)) * (1.0 + math.log1p(f_start))
```

`best_fitness_up_to(t)` cần thêm vào `TopKDatabase`-side cache: maintain dict `{iter -> best_fitness_so_far}` trên mỗi `add()`. Cài trong `PlaybookDatabase`.

```python
# retention.py
async def probe_playbook(playbook_text: str,
                         db: ProgramDatabase,
                         builder, llm, evaluator,
                         n_probes: int = 3) -> float:
    """Run n_probes mini-iterations with `playbook_text` injected.
    Children are NOT added to main db (kept in a throwaway list)."""
    parents = db.sample_random_parents(n=n_probes)
    deltas = []
    for p in parents:
        ctx = build_context_with_playbook(builder, p, db, playbook_text)
        child_code = await llm.generate(ctx["system"], ctx["user"], temperature=0.5)
        child_metrics = await evaluator.evaluate(child_code)
        deltas.append(get_score(child_metrics) - get_score(p.metrics))
    return sum(deltas) / len(deltas)


def accept_proposed(current_ema: float, proposed: float, epsilon: float = 0.1) -> bool:
    if current_ema is None or math.isnan(current_ema):
        return True   # first revision always accepted
    return proposed >= (1.0 - epsilon) * current_ema
```

### 6.8 `playbook/reviser.py` + `summary.py`

```python
# summary.py
def build_window_summary(db: ProgramDatabase, t_now: int, K: int) -> dict:
    """Format last K iter into a dict feed-able to LLM reviser.
    Includes: top_programs (3, code+metrics), worst_program, fitness_trace,
    improvement_count, stagnation_count.
    """
    ...

# reviser.py
async def revise_playbook(current_text: str, summary: dict, llm) -> tuple[str, dict]:
    sys_prompt = (Path(__file__).parent / "config/revise_sys_prompt.txt").read_text()
    user_msg = render_revise_user_message(current_text, summary)
    resp = await llm.generate(sys_prompt, user_msg, temperature=0.4)
    new_text = extract_markdown_block(resp.text)
    # Validate structure: must contain all 5 required headers
    if not _has_required_sections(new_text):
        logger.warning("Reviser output missing sections; keeping current playbook")
        return current_text, {"failed_validation": True}
    return new_text, {"summary": summary, "raw": resp.text}
```

### 6.9 `playbook/controller.py`

```python
class PlaybookController(DiscoveryController):
    def __init__(self, controller_input, K: int = 10,
                 probe_n: int = 3,
                 elite_size: int = 20,
                 gamma_local: float = 0.3,
                 mmr_lambda: float = 0.5,
                 mmr_alpha: float = 0.5):
        super().__init__(controller_input)
        self.K = K
        self.probe_n = probe_n

        # Wire SCCF into the database
        clusterer = SemanticClusterer(llm=self.guide_llms.pick())
        mmr = MMRSelector(lam=mmr_lambda, alpha=mmr_alpha)
        self.sccf = SCCF(clusterer, mmr, gamma_local=gamma_local, elite_size=elite_size)
        assert isinstance(self.database, PlaybookDatabase), \
            "PlaybookController requires PlaybookDatabase"
        self.database.sccf = self.sccf

        self.store = PlaybookStore(Path(self.output_dir) / "playbook")

    async def run_discovery(self):
        # Bootstrap
        v0 = await bootstrap_playbook(
            self._initial_program_code(),
            self.config.context_builder.system_message,
            self.llms.pick(),
        )
        self.store.record(v0, born_iter=0)
        self._update_active_playbook(v0)

        # Main loop
        for t in range(self.config.max_iterations):
            await self._one_iteration(t)
            if t > 0 and t % self.K == 0:
                await self._meta_step(t)

        # Finalize
        self.store.close_window(self.store.current(),
                                last_iter=self.config.max_iterations,
                                score=window_score(self.store.current(), self.database))
        await super().finalize()

    async def _meta_step(self, t: int):
        # 1. Re-cluster elite (cached internally)
        await self.sccf.maybe_recluster(self.database)
        # 2. Score current playbook + close its window
        cur = self.store.current()
        cur_score = window_score(cur, self.database)
        self.store.close_window(cur, last_iter=t, score=cur_score)
        self.store.update_ema(cur_score)
        # 3. Revise + probe + accept
        summary = build_window_summary(self.database, t, self.K)
        proposed, meta = await revise_playbook(cur.text, summary, self.llms.pick())
        if proposed == cur.text:    # validation failed → no change
            return
        proposed_score = await probe_playbook(
            proposed, self.database, self.context_builder, self.llms.pick(),
            self.evaluator, n_probes=self.probe_n,
        )
        if accept_proposed(self.store.ema_score(), proposed_score, epsilon=0.1):
            self.store.record(proposed, born_iter=t)
            self._update_active_playbook(proposed)
        else:
            self.store.record_rejected(proposed, attempted_at=t)

    def _update_active_playbook(self, text: str):
        # Push into context builder + database hint
        self.context_builder.set_playbook(text)
        self.database._current_hint = parse_hint(text)
```

Lưu ý: `_one_iteration` reuse logic của parent — không cần override. Chỉ inject playbook trước iteration đầu, và refresh ở mỗi `_meta_step`.

### 6.10 `context_builder/playbook/builder.py`

```python
class PlaybookContextBuilder(DefaultContextBuilder):
    def __init__(self, config):
        super().__init__(config)
        self._playbook_text: str = ""

    def set_playbook(self, text: str) -> None:
        self._playbook_text = text

    def _get_system_message(self) -> str:
        base = super()._get_system_message()
        if not self._playbook_text:
            return base
        return f"{base}\n\n## Search policy (auto-evolved)\n{self._playbook_text}"
```

Đơn giản: chỉ prepend playbook vào system message. Không cần đổi user template.

### 6.11 `route.py` (MODIFY)

Thêm sau dòng 83:
```python
from skydiscover.search.playbook.controller import PlaybookController
from skydiscover.search.playbook.database import PlaybookDatabase

register_database("playbook", PlaybookDatabase)
register_controller("playbook", PlaybookController)
```

### 6.12 `config.py` (MODIFY)

Thêm class trước `_DB_CONFIG_BY_TYPE`:
```python
@dataclass
class PlaybookDatabaseConfig(DatabaseConfig):
    # Playbook lifecycle
    K: int = 10
    probe_n: int = 3
    bootstrap_temperature: float = 0.5
    revise_temperature: float = 0.4
    epsilon_accept: float = 0.1

    # SCCF
    elite_size: int = 20
    gamma_local: float = 0.3
    mmr_lambda: float = 0.5
    mmr_alpha: float = 0.5
    clustering_fallback_k: int = 3
```

Thêm vào `_DB_CONFIG_BY_TYPE`: `"playbook": PlaybookDatabaseConfig`.

### 6.13 Tests

- `test_playbook_store.py`: record/close/persist round-trip, EMA correct.
- `test_playbook_scorer.py`: 3 fixture window, kiểm tra eq (§4.2) ra đúng số.
- `test_mmr.py`: 5 synthetic programs với metric vector khác nhau, λ=0.7 → picks diverse outcome; λ=0 → picks fitness top.
- `test_sccf_integration.py`: stub LLM trả về fixed cluster mapping, SCCF với 3 cluster + γ=0 verify cross-cluster guarantee (k context không cùng cluster với parent).

---

## §7. Prompt templates

### 7.1 `config/bootstrap_sys_prompt.txt`

```
You are setting up a search policy for an iterative LLM-driven program discovery process.

Given a problem description and an initial program, write a SHORT markdown playbook
(200–400 words total) that future iterations will read before proposing new programs.
The playbook MUST use these five headers, in this exact order:

## Current focus
## What's working
## What's exhausted / avoid
## Suggested moves
## Context selection hint

Rules:
- Be concrete. Reference structural features of the problem or program.
- Avoid platitudes ("explore", "try different approaches").
- Each bullet should be actionable: a reader should know what code change to attempt.
- "Context selection hint": 1–2 short lines about WHICH kinds of reference programs
  would be most useful right now (e.g. "prefer programs with high recall but low
  precision", "avoid DP-based solutions"). This will be used by the retrieval system.
- If a section has nothing to say yet, write "(none yet — first iteration)".

Return ONLY the markdown. No code fences, no preamble.
```

### 7.2 `config/revise_sys_prompt.txt`

```
You are revising a search policy (playbook). You will see:
- The current playbook.
- A summary of the last K iterations: top 3 programs (code + metrics), worst program,
  fitness trace, improvement count, current stagnation length.

Produce a NEW playbook with the same five-section markdown structure (200–400 words).

Guidelines:
- Update "What's working" with patterns visible in top programs.
- Move now-exhausted ideas from "Suggested moves" into "What's exhausted / avoid".
- If stagnation_count is high, change "Current focus" to a different region.
- Each revision MUST change at least one bullet in either "What's working" or
  "Suggested moves". No rephrasing-only revisions.
- Update "Context selection hint" to reflect what reference programs would be most
  useful now.
- Avoid platitudes.

Return ONLY the markdown. No code fences.
```

### 7.3 `config/clustering_sys_prompt.txt` (adapted from MPaGE Appendix E.3)

```
You are an expert in code analysis. You will receive a numbered list of Python code
snippets, each tagged with an integer index.

Group the snippets into clusters where each cluster contains snippets implementing the
same underlying logic — even when their surface syntax (variable names, control flow
constructs, library calls, vectorization vs loops) differs significantly.

Two snippets belong to the same cluster iff a knowledgeable reader would describe
them with the same one-sentence summary of intent and method.

Return the result as a single JSON object on one line:
{"1": [0, 2, 4], "2": [1, 3], "3": [5]}

Keys are cluster IDs (strings); values are lists of snippet indices. Every input
index must appear in exactly one cluster. Do not include any text outside the JSON.
```

---

## §8. Experimental protocol

### 8.1 Setup
- **Benchmarks**: 3 từ [benchmarks/](benchmarks/) — đề nghị `math/circle_packing` (đã có và biết) + 2 cái nữa (sẽ pick sau khi quét repo, ưu tiên benchmark có ≥ 2 metric trong evaluator output để outcome-distance có nghĩa).
- **Seeds**: 3 seed/benchmark/method. Có thể nâng lên 5 nếu budget cho phép.
- **Budget**: 100 iter/run.
- **LLM**: gpt-5 qua openrouter (giống các config khác trong repo).

### 8.2 Methods

| Code | Method | Component A | Component B |
|---|---|---|---|
| B0 | Default (TopK) | – | – |
| B1 | AdaEvolve | – | – |
| B2 | FORE | – | – |
| M | Playbook (full) | ON | ON |
| M\A | Playbook \\ Policy | OFF (v0 only) | ON |
| M\B | Playbook \\ SCCF | ON | OFF |

### 8.3 Metrics

- **Final best fitness** (primary).
- **AUC of best-so-far curve** (secondary — early progress).
- **Outcome diversity @ iter 50**: mean pairwise distance trên outcome space của top-20 (= MPaGE's effective diversity proxy).
- **SWDI + CDI** (Shannon-Wiener + Cumulative Diversity Index) đã mô tả MPaGE D.3 + D.4 — code có thể adapt từ `hsevo` baseline.
- **Code diversity @ iter 50**: tương tự trên code space.
- Statistical: mean ± std cross-seed; paired Wilcoxon B0 vs M trên cùng seed/benchmark.

### 8.4 Ablations

- A1 (K), A2 (retain/revert), A3 (static playbook) — §2.5.
- B1 (no SCCF), B2 (no cross-cluster), B3 (γ_local), B4 (MMR α), B5 (no hint), B6 (AST clustering) — §3.6.

A1–A3, B1, B3, B4 là core. B2, B5, B6 nice-to-have.

### 8.5 Diversity figure (paper centerpiece)

Plot outcome-diversity-vs-iter cho B0 vs M trên 1 benchmark. Kỳ vọng:
- B0: drop nhanh, plateau thấp (mode collapse).
- M: giữ cao xuyên suốt.

Cộng với fitness-vs-iter curve để show không trade-off quality lấy diversity quá đắt.

### 8.6 Cluster visualization (interpretability figure)

Tại 3 thời điểm (t=10, 50, 100), visualize semantic clusters của elite top-20 dưới dạng diagram giống MPaGE Figure 8 (cluster ellipse + heuristic icons). Show evolution của cluster structure theo time — chứng minh rằng clustering thực sự bắt được semantic chứ không phải noise.

---

## §9. Feasibility, rủi ro, fallback

### 9.1 Effort estimate
- Code: ~1300–1500 LOC tổng (store 100, bootstrap 50, clustering 250, mmr 200, sccf 200, database 100, scorer 50, retention 150, reviser 100, controller 200, context_builder 50, tests 300).
- Prompt tuning: 3–4 day iteration trên revise + clustering prompts.
- Experiments: 9 run × 3 method core + ablations ≈ 30 run × 100 iter ≈ 3000 evaluator calls/method.
- Tổng: 2.5 tuần coding + 1 tuần thí nghiệm + 1.5 tuần viết paper.

### 9.2 Rủi ro

| Rủi ro | Mitigation |
|---|---|
| Playbook drift về platitude | reviser sys prompt cấm explicit + retention rule (§4.3) reject playbook không cải thiện |
| LLM ignore playbook | counterfactual probe trong retention chính là defense — nếu LLM ignore, score không tăng → reject |
| LLM clustering parse fail | Fall back K-means trên `Program.metrics` (`clustering.py:_fallback_kmeans`) — silent, không crash run |
| Outcome vector inconsistent (benchmark có 1 metric duy nhất) | MMR degrade về code-only (α=1); paper sẽ note limitation này |
| MMR α / λ sai → diversity quá ép → fitness drop | Ablation B3, B4 sẽ cho biết; default 0.5 chọn từ literature |
| Cluster count $m$ vô lý (e.g. 1 cluster cho 20 program) | clustering_sys_prompt enforce $m \in [2, 5]$; retry 1 lần nếu fail |
| Probe_n=3 quá ít, noise cao | Có thể nâng probe_n; ε=0.1 đã tính đến noise margin |
| Reviewer hỏi "single-objective why MPaGE-style?" | Paper section §1.2 đã viết rõ: ta mượn clustering technique, không phải full PFG framework |

### 9.3 Fallback hierarchy

Nếu method thật sự không tốt hơn baseline, vẫn còn 3 paper-able sub-stories:

1. **"Outcome collapse là vấn đề thật"**: figure outcome-diversity-vs-iter của B0 cho 3 benchmark + correlation với stagnation. Đây là một observation paper (workshop tier).
2. **"LLM clustering > AST/embedding cho LLM-generated code"**: replicate MPaGE §F trên benchmarks của SkyDiscover. Nếu hold, đây là một extended-validation paper.
3. **"Playbook retain/revert một mình"**: nếu chỉ Component A work, paper về quality-gated prose-policy evolution.

---

## §10. Outline paper + ablation

### 10.1 Title (tentative)
**"Playbook: Co-Evolving Search Policies and Semantically Clustered Context for LLM Program Discovery"**

### 10.2 Outline 8 trang

1. **Intro** (1 trang) — P1, P2. Two-component preview. Connection với MPaGE explicitly cited.
2. **Related** (0.75 trang) — AdaEvolve, EvoX, FORE, TusoAI, MPaGE positioned trong 2D map (object × channel). MMR + submodular trong IR.
3. **Method** (2 trang) — §2 (Playbook), §3 (SCCF), §4 (math). Pseudo-code chính.
4. **Experiments** (2.5 trang) — §8 setup, results table, diversity figure, ablation table, cluster visualization (§8.6).
5. **Analysis** (1 trang) — Qualitative read 2 playbook timelines (1 easy + 1 hard benchmark). Cluster evolution figure. Top section credit từ playbook diff.
6. **Discussion & limitations** (0.5 trang) — Compose với selection-level method (future), LLM dependence, single-metric benchmarks.
7. **Conclusion** (0.5 trang).

### 10.3 Đóng góp claim
1. **Prose-level co-evolution** với scoring + retention (EvoX-style scorer áp dụng cho prompt thay vì code).
2. **Semantic-Cluster Context Filter** — adapt MPaGE's LLM-based clustering từ multi-objective heuristic design sang **single-objective context retrieval**. Đây là một transfer khá tự nhiên nhưng chưa ai làm.
3. Empirical: improvement trên 3 benchmark với multi-seed; outcome-diversity-vs-iter figure validate P2; cluster visualization làm bằng chứng interpretability.

### 10.4 Threats to validity
- LLM-specific (gpt-5); generalization sang LLM khác chưa kiểm chứng.
- Benchmark coverage hạn chế (3 benchmark).
- Outcome distance dùng raw evaluator metrics → giả định metrics đại diện "outcome space"; benchmark có 1 metric duy nhất degrade SCCF về code-only.
- 3 seed có thể chưa đủ cho statistical significance trên benchmark khó; budget allowing thì 5 seed.

---

## TL;DR

Playbook là search method 2 component, mượn ý tưởng có chọn lọc từ EvoX (log-window scorer + retain/revert pattern) và MPaGE (LLM-based semantic clustering):

- **(A) Evolved prose policy** trong system prompt, revise mỗi K iter, có quality gate (probe + accept rule).
- **(B) Semantic-Cluster Context Filter** chọn reference programs qua 2 tier: (Tier 1) LLM cluster elite theo semantic logic — MPaGE-style; (Tier 2) cross-cluster sampling với MMR within-cluster trên fitness × code-and-outcome similarity.

Math: MMR objective + log-window scorer + accept rule. Không Bayesian, không bandit phức tạp.

Experiments: 3 benchmarks × 3 seeds × 6 method/ablation. Primary metric final fitness, secondary metric **outcome diversity** trực tiếp validate motivation P2. Diversity figure + cluster visualization là interpretability money shots.

Fallback hierarchy 3 tầng (§9.3) đảm bảo paper-able ngay cả khi method không SOTA.
