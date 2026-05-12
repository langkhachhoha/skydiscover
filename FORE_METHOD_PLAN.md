# FORE — Fertility-Oriented Reflective Evolution

Tài liệu này lên kế hoạch chi tiết để thêm một search method mới vào SkyDiscover, tên là **FORE** (Fertility-Oriented Reflective Evolution). Mục tiêu:

1. Trả lời câu hỏi nghiên cứu trọng tâm: *“nên mở rộng chương trình đang tốt nhất, hay chương trình có khả năng sinh ra hậu duệ tốt nhất?”* — bằng một bộ ước lượng **Posterior Offspring Value (POV)** dựa trên Bayesian + Thompson sampling. Đây là phần “toán sâu” khác biệt với AdaEvolve (UCB trên fitness improvement) và EvoX (co-evolve strategy).
2. Có cơ chế **Reflective Review** (LLM tổng kết & viết lại kế hoạch khi search bị kẹt) — lấy cảm hứng từ paradigm breakthrough của AdaEvolve và Strategic Landscape Navigation của TusoAI, nhưng tổ chức lại quanh **fertility map** thay vì paradigm list.
3. Mỗi candidate mang theo một **strategy description** ngắn (TusoAI-style) làm “first-class state” trong archive, dùng để (i) cluster lineage trong fertility map, (ii) làm tài liệu cho Reflective Review, (iii) đưa vào prompt cho con đời sau.

Tài liệu này được viết để có thể đem đi vừa code vừa viết paper. Cấu trúc:

- §1 Vị trí so với AdaEvolve, EvoX, TusoAI.
- §2 Mô hình toán: POV, Thompson sampling, regret bound.
- §3 Kiến trúc tổng thể trong codebase SkyDiscover.
- §4 Plan từng file (cái có sẵn để tái sử dụng, cái cần viết mới, ký + skeleton).
- §5 Stagnation Review pipeline.
- §6 Config + benchmark + cách chạy.
- §7 Feasibility, độ phức tạp, rủi ro.
- §8 Outline paper + thí nghiệm + ablation.

---

## §1. Vị trí của FORE so với các baseline trong codebase

| Trục | AdaEvolve | EvoX | TusoAI (paper) | **FORE (mới)** |
|---|---|---|---|---|
| Parent selection | UCB trên island, sampling theo *current fitness* + diversity | Co-evolve search strategy ngoài | Cluster theo strategy semantic, complementary retrieval | **Thompson sampling trên posterior offspring value** (POV). Parent được chọn theo *xác suất hậu duệ vượt global best*, không phải fitness hiện tại. |
| Stagnation handling | `ParadigmTracker` + `ParadigmGenerator` sinh ý tưởng đột phá | Switch sang search strategy mới khi stagnate | SLN tổng kết effective/saturated/underexplored families | **Reflective Review** với *fertility map* của lineage (effective/exhausted/embryonic), không chỉ paradigm cho program đơn lẻ. |
| Persistent strategy state | Có “changes summary” trong `metadata["changes"]` (1 dòng) | Không lưu cấp population | Strategy description + embedding (first-class) | **Strategy description + hypothesis + verdict** lưu trong `metadata`, được dùng cho cluster và review. Không bắt buộc dùng embedding ngoài (giản lược so với TusoAI). |
| Math model | Adaptive learning rate qua G | Black-box co-evolution | Probabilistic instruction sampling + Bayesian-style category update | **Normal-Inverse-Gamma posterior trên Δ-improvement, Thompson sampling, regret bound `O(sqrt(K log N))`**. |

Nhấn mạnh không copy:
- AdaEvolve dùng UCB ở mức *island*, FORE dùng Thompson ở mức *program* — cấp độ khác.
- TusoAI có Bayesian update cho *instruction category*; FORE Bayesian cho *parent fertility* (đối tượng khác).
- EvoX co-evolve search strategy; FORE giữ một meta-loop duy nhất nhưng review có cấu trúc.

---

## §2. Mô hình toán — Posterior Offspring Value & Thompson Sampling

### 2.1 Định nghĩa bài toán parent-selection

Cho archive $\mathcal{A}_t$ tại generation $t$. Mỗi program $p \in \mathcal{A}_t$ có fitness $f(p) \in \mathbb{R}$. Mục tiêu cuối cùng:
$$
P^* = \arg\max_{P \in \mathcal{P}} f(P)
$$
nhưng ta chỉ có budget $T$ generations. Ở mỗi step ta phải chọn 1 parent $p$ rồi LLM mutate ra child. Câu hỏi: chọn $p$ nào để tối đa hóa **best-found fitness sau $T$ generations**?

Định nghĩa **K-step descendant value** của $p$:
$$
V_K(p) \;=\; \mathbb{E}\!\left[\max_{q \in \mathrm{Desc}_{\le K}(p)} f(q) \;\Big|\; p\right]
$$
trong đó $\mathrm{Desc}_{\le K}(p)$ là tất cả hậu duệ trong tối đa $K$ bước mutate kể từ $p$. Greedy chọn $p^\star = \arg\max_p V_K(p)$ với $K$ = budget còn lại. Lưu ý $V_K$ phụ thuộc mạnh vào **đuôi phải** của phân phối improvement $\Delta = f(\text{child}) - f(p)$ chứ không phải $f(p)$ trực tiếp.

### 2.2 Mô hình branching đơn giản

Giả định tree branching trung bình $b$ con/iteration và improvement của một mutate trực tiếp $\Delta \sim \mathcal{D}_p$ với mean $\mu_p$, std $\sigma_p$. Một dạng ràng buộc dưới (Lemma 1) ta dùng trong paper:
$$
V_K(p) \;\ge\; f(p) + \mathbb{E}_{\Delta \sim \mathcal{D}_p}[\Delta^+] \cdot \mathrm{eff}(K, b)
$$
với $\Delta^+ = \max(\Delta, 0)$ và $\mathrm{eff}(K, b) = 1 + \sum_{k=1}^{K-1} \alpha^k$ là *fertility multiplier* (geometric series, $\alpha < 1$ điều chỉnh decay vì children có thể không productive bằng parent). Hệ quả: **fertility được drive bởi $\mu_p^+ := \mathbb{E}[\Delta^+]$ — kỳ vọng improvement dương — chứ không phải $f(p)$**. Đây là nền tảng định lượng cho idea “stepping stone”.

### 2.3 Posterior cho $\mu_p^+$

Coi $\Delta_p^{(1)}, \dots, \Delta_p^{(n_p)}$ là các quan sát historical improvement (lấy từ mọi mutation đã có $p$ làm parent, kể cả con-của-con nếu propagate). Trên các quan sát dương ($\Delta > 0$) ta đặt model:
$$
\Delta^+ \mid \mu, \tau \sim \mathcal{N}(\mu, \tau^{-1}), \quad
(\mu, \tau) \sim \text{NIG}(\mu_0, \kappa_0, \alpha_0, \beta_0)
$$
Với conjugate prior Normal-Inverse-Gamma, posterior cho $\mu_p$ là Student-$t$ với mean
$$
\hat\mu_{n_p} = \frac{\kappa_0 \mu_0 + n_p \bar{\Delta}^+_p}{\kappa_0 + n_p}, \quad
\hat\sigma_{n_p}^2 = \frac{2\beta_0 + S_p + \frac{\kappa_0 n_p}{\kappa_0+n_p}(\bar{\Delta}^+_p - \mu_0)^2}{(2\alpha_0 + n_p)(\kappa_0 + n_p)}
$$
trong đó $S_p = \sum_i (\Delta_p^{(i)+} - \bar{\Delta}^+_p)^2$.

### 2.4 Thompson Sampling cho parent

Mỗi iteration:
1. Với mỗi candidate parent $p$, lấy mẫu $\tilde\mu_p \sim t(\hat\mu_{n_p}, \hat\sigma_{n_p}, \nu_{n_p})$.
2. Tính $\widetilde{\mathrm{POV}}(p) = f(p) + \mathrm{eff}(K_{\text{remain}}, b)\cdot \tilde\mu_p^+$.
3. Chọn $p^\star = \arg\max_p \widetilde{\mathrm{POV}}(p)$.

Đây là **Thompson sampling trên một posterior tổng hợp**, không phải Thompson “raw” trên fitness. Quan trọng:

- Khi $n_p$ nhỏ (parent “embryonic”): posterior rộng → đôi khi $\tilde\mu_p$ rất lớn → được khám phá (stepping-stone bonus).
- Khi $n_p$ lớn và $\bar\Delta^+$ nhỏ (parent exhausted): posterior chặt quanh số nhỏ → ít được chọn dù $f(p)$ cao.
- Khi parent có $f(p)$ cao và $\bar\Delta^+$ cũng cao: rõ ràng được chọn (đúng kỳ vọng).
- Khi parent có $f(p)$ thấp nhưng $\bar\Delta^+$ rất cao (con thường tăng mạnh): được chọn — đây chính là “stepping stone” được nắm bắt bằng toán.

### 2.5 Regret bound

Với Thompson sampling trên $K$ candidate parents, $T$ generations, dưới giả định $\mu_p^+ \in [0, B]$ và prior bounded, regret kỳ vọng:
$$
\mathbb{E}[\mathrm{Regret}_T] \;=\; O\!\left(\sqrt{K T \log T}\right)
$$
(Russo & Van Roy 2014). Trong paper, ta state lại bound này cho setting evolutionary search (proof reduction: parent-selection ở mỗi step là một bandit; reward = realized $\Delta^+$ của child).

### 2.6 Vì sao không chỉ dùng UCB như AdaEvolve

UCB trên fitness improvement (như `MultiDimensionalAdapter`) làm việc ở mức *island*, không phải parent. Áp UCB cho từng parent dễ bị over-exploration vì parent count tăng nhanh. Thompson sampling tự nhiên scale, có Bayesian interpretation, dễ inject prior từ structural feature (xem 2.7).

### 2.7 Structural prior từ strategy description

Khi $n_p = 0$ (parent chưa có con), ta cần một prior tốt. Dùng:
$$
\mu_0(p) = w_1 \cdot \mathrm{novelty}(p) + w_2 \cdot \mathrm{rarity}(\mathrm{cluster}(p)) - w_3 \cdot \mathrm{age}(p)
$$
trong đó:
- $\mathrm{novelty}(p)$: trung bình code-distance đến top-$k$ neighbor (đã có sẵn `CodeDiversity` trong `skydiscover/search/adaevolve/archive/diversity.py`).
- $\mathrm{rarity}(\mathrm{cluster}(p))$: nghịch đảo số program trong strategy-cluster của $p$ (cluster bằng simple Jaccard trên description tokens — không cần embedding ngoài).
- $\mathrm{age}(p)$: số iteration kể từ khi sinh ra; phạt parent cũ vì xác suất search-space quanh nó đã được khám phá.

Phần này là chỗ đưa strategy description vào toán: description tham gia trực tiếp qua cluster rarity.

---

## §3. Kiến trúc — bản đồ tổng thể trong codebase

Tận dụng tối đa cấu trúc có sẵn:

```
skydiscover/
  search/
    base_database.py          ← (giữ nguyên) Program, ProgramDatabase
    default_discovery_controller.py  ← (giữ nguyên) — FORE controller subclass
    registry.py, route.py     ← MODIFY: đăng ký 'fore'
    adaevolve/
      archive/diversity.py    ← REUSE: CodeDiversity cho novelty prior
    fore/                     ← MỚI
      __init__.py
      fertility.py            ← Math: NIG posterior, Thompson sampling, POV
      descriptions.py         ← Parse/store strategy description từ LLM output
      review.py               ← LLM-driven Reflective Review (stagnation escape)
      database.py             ← FOREDatabase: fertility-aware archive
      controller.py           ← FOREController: orchestrate review + descriptions
      README.md
  context_builder/
    default/                  ← REUSE
    fore/                     ← MỚI
      __init__.py
      builder.py              ← FOREContextBuilder
      templates/
        full_rewrite_user_message.txt
        diff_user_message.txt
        review_user_message.txt
  config.py                   ← MODIFY: thêm FOREDatabaseConfig + register
configs/fore.yaml             ← MỚI
benchmarks/math/circle_packing  ← REUSE để smoke test
tests/test_fore_fertility.py  ← MỚI (unit test cho math)
```

Số file mới: ~12, sửa: 2. Toàn bộ giữ cùng pattern với `adaevolve/` để bạn (và reviewer code) dễ navigate.

---

## §4. Plan từng file

### 4.1 `skydiscover/search/fore/fertility.py` — MỚI (math core)

**Mục đích**: chứa toàn bộ phép toán Bayesian + Thompson sampling, tách hoàn toàn khỏi I/O, để test bằng pytest dễ. Không phụ thuộc LLM.

**Skeleton**:
```python
# fertility.py
from dataclasses import dataclass, field
from typing import List, Optional
import math, random

@dataclass
class NIGPrior:
    """Normal-Inverse-Gamma prior (mu_0, kappa_0, alpha_0, beta_0)."""
    mu_0: float = 0.0
    kappa_0: float = 1.0
    alpha_0: float = 2.0
    beta_0: float = 1.0

@dataclass
class FertilityStats:
    """Per-parent statistics for POV estimation.

    Maintains running sums to compute the posterior in O(1) per update.
    Only Δ⁺ = max(Δ, 0) observations are accumulated; sign info goes into
    `negative_count` so a parent that consistently produces regressions
    is penalized via a separate term.
    """
    n: int = 0                  # number of children evaluated
    sum_delta_plus: float = 0.0
    sum_sq_delta_plus: float = 0.0
    negative_count: int = 0
    # structural prior inputs (set once at insertion)
    novelty_score: float = 0.0  # in [0, 1]
    cluster_rarity: float = 0.0 # in [0, 1]
    age_at_birth: int = 0       # iteration when parent was added

    def update_with_child(self, child_delta: float) -> None:
        ...

    def posterior_t(self, prior: NIGPrior) -> tuple:
        """Return (loc, scale, df) for Student-t posterior on mu."""
        ...

    def sample_mu_plus(self, prior: NIGPrior, rng: random.Random) -> float:
        loc, scale, df = self.posterior_t(prior)
        # Sample from t(loc, scale, df) using a Normal-over-sqrt(Gamma) trick
        # (Section 2.3 of plan).
        ...

def fertility_multiplier(k_remaining: int, branching: float = 1.0, alpha: float = 0.7) -> float:
    """eff(K, b) = 1 + sum_{k=1..K-1} alpha^k  (clamped at K_max=20)."""
    K = min(max(k_remaining, 1), 20)
    return sum(alpha ** k for k in range(K))

def pov_score(
    fitness: float,
    stats: FertilityStats,
    prior: NIGPrior,
    k_remaining: int,
    rng: random.Random,
    w_novelty: float = 0.3,
    w_rarity: float = 0.2,
    w_age_penalty: float = 0.05,
    iteration: int = 0,
) -> float:
    """Thompson-sample one POV value for a parent."""
    # structural prior μ_0 only used implicitly via prior parameters? — see 2.7.
    structural_prior = (
        w_novelty * stats.novelty_score
        + w_rarity * stats.cluster_rarity
        - w_age_penalty * max(0, iteration - stats.age_at_birth)
    )
    effective_prior = NIGPrior(
        mu_0=prior.mu_0 + structural_prior,
        kappa_0=prior.kappa_0,
        alpha_0=prior.alpha_0,
        beta_0=prior.beta_0,
    )
    mu_sample = stats.sample_mu_plus(effective_prior, rng)
    return fitness + fertility_multiplier(k_remaining) * max(mu_sample, 0.0)
```

**Unit test** (`tests/test_fore_fertility.py`):
- Sanity: với 0 quan sát + prior trung tính, `sample_mu_plus` ra giá trị trong dải `[μ_0 ± vài sigma]`.
- Convergence: với 100 quan sát ~ N(0.5, 0.1), `posterior_t.loc → 0.5`.
- Stepping-stone case: parent A có fitness 0.3 nhưng `bar_delta = 0.4`; parent B có fitness 0.6 nhưng `bar_delta = 0.01`. Sau khi feed nhiều quan sát, A được chọn nhiều hơn dù fitness thấp hơn.

Độ khó: thấp. ~200 LOC.

### 4.2 `skydiscover/search/fore/descriptions.py` — MỚI

**Mục đích**: định nghĩa schema cho strategy description; parse từ LLM output; lưu vào `program.metadata`.

Schema (lưu plain dict trong `program.metadata["fore"]`):
```python
{
    "strategy_label": str,       # short tag, e.g. "hexagonal-shell"
    "description": str,          # 1-3 sentence what this candidate explores
    "hypothesis": str,           # why this *might* be better even if score doesn't show it yet
    "diff_from_parent": str,     # 1 line: what's different vs parent
    "verdict": str | None,       # filled after eval: "improved" | "regressed" | "stepping_stone" | "dead_end"
    "cluster_id": int,           # filled by FOREDatabase after insertion
}
```

**Skeleton**:
```python
# descriptions.py
import json, re
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

FORE_BLOCK_RE = re.compile(r"<fore_meta>(.*?)</fore_meta>", re.DOTALL)

@dataclass
class StrategyDescription:
    strategy_label: str = "unspecified"
    description: str = ""
    hypothesis: str = ""
    diff_from_parent: str = ""
    verdict: Optional[str] = None
    cluster_id: int = -1
    def to_dict(self) -> Dict[str, Any]: return asdict(self)

def parse_strategy_block(llm_response: str) -> StrategyDescription:
    """Extract <fore_meta> JSON block from LLM output (best-effort)."""
    m = FORE_BLOCK_RE.search(llm_response or "")
    if not m:
        return StrategyDescription()
    try:
        d = json.loads(m.group(1))
        return StrategyDescription(**{k: d.get(k, "") for k in
            ["strategy_label", "description", "hypothesis", "diff_from_parent"]})
    except Exception:
        return StrategyDescription()

def compute_verdict(parent_fitness: float, child_fitness: float,
                    parent_stats, threshold: float = 0.005) -> str:
    """Assign verdict from numeric outcome + parent's fertility stats.

    - improved : child > parent + threshold
    - regressed: child < parent - threshold AND parent has high mu+ (counts as bad)
    - stepping_stone: child ≈ parent OR slightly below, but in a structurally novel region
    - dead_end : child <<< parent and parent already has high negative_count
    """
    ...
```

Mấu chốt: ta **không** dùng vector embedding ngoài (như TusoAI dùng `text-embedding-3-small`) — chỉ dùng Jaccard-token-distance trên `description` + reuse `CodeDiversity`. Giữ FORE self-contained.

Độ khó: thấp. ~150 LOC.

### 4.3 `skydiscover/search/fore/review.py` — MỚI (Reflective Review)

**Mục đích**: khi search bị kẹt, gọi LLM để (i) đọc fertility map + danh sách strategy description gần đây, (ii) viết một **review** dạng cấu trúc: `effective_lineages`, `exhausted_lineages`, `embryonic_lineages`, `next_steps`. Output review là dữ liệu *persistent*: lưu vào `database._active_review` và inject vào prompt cho vài generation tiếp theo.

Khác paradigm AdaEvolve: paradigm là **idea cho 1 mutation step**; review là **bản đồ chiến lược cho 1 cửa sổ generations**, có references đến cluster cụ thể.

**Skeleton**:
```python
# review.py
import json, logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from skydiscover.llm.llm_pool import LLMPool

logger = logging.getLogger(__name__)

@dataclass
class FertilityReview:
    effective_lineages: List[Dict[str, Any]] = field(default_factory=list)
    exhausted_lineages: List[Dict[str, Any]] = field(default_factory=list)
    embryonic_lineages: List[Dict[str, Any]] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    iteration_generated: int = 0
    uses_remaining: int = 3
    def to_dict(self): ...

class ReflectiveReviewer:
    """Wraps an LLMPool to produce a FertilityReview from current archive state."""
    def __init__(self, llm_pool: LLMPool, system_message: str, evaluator_code: str = ""):
        self.llm_pool = llm_pool
        self.system_message = system_message
        self.evaluator_code = evaluator_code

    async def generate(
        self,
        fertility_summary: List[Dict[str, Any]],  # one row per cluster
        recent_attempts: List[Dict[str, Any]],
        global_best_score: float,
    ) -> Optional[FertilityReview]:
        prompt_user = self._build_prompt(fertility_summary, recent_attempts, global_best_score)
        # Use same pattern as ParadigmGenerator: structured JSON with retries.
        ...
```

Prompt design (template-driven, để dễ tinker mà không sửa code):
- INPUT: bảng fertility (per-cluster: size, mean Δ⁺, mean fitness, exemplar strategy_label).
- INPUT: 5–10 recent strategy descriptions với verdict.
- OUTPUT JSON: `{effective_lineages: [...], exhausted_lineages: [...], embryonic_lineages: [...], next_steps: [...]}`.

Trigger conditions (tính trong database, gọi review khi):
1. **Global rate**: improvement rate < `review_rate_threshold` trong cửa sổ `review_window`.
2. **POV variance crash**: max POV của top-10 parents giảm dưới `pov_floor` (signal: cả các parent “tiềm năng” cũng đã exhausted).
3. **All-cluster exhausted**: tất cả cluster có `mean Δ⁺ < epsilon` trong cửa sổ.

Hai trigger sau là **đóng góp mới** (không có trong AdaEvolve paradigm). Chúng map vào câu hỏi mở của user: "phát ra tín hiệu để thoát khỏi stagnation".

Độ khó: trung bình. ~300 LOC.

### 4.4 `skydiscover/search/fore/database.py` — MỚI

**Mục đích**: là `ProgramDatabase` chính cho FORE. Trách nhiệm:
- Lưu programs.
- Maintain `FertilityStats` per parent.
- Maintain strategy clusters (Jaccard trên token set của description + code).
- Sample parent qua Thompson trên POV.
- Detect stagnation triggers.
- Provide hooks cho `FOREController` đọc `_active_review`.

Reuse: `CodeDiversity` cho novelty score; `UnifiedArchive` *có thể* dùng cho mỗi cluster (optional ablation). Để keep simple, **start với flat archive + Jaccard cluster map**, không nhiều island.

**Skeleton**:
```python
# database.py
from typing import Dict, List, Tuple, Optional, Any
import logging, random, uuid
from skydiscover.config import DatabaseConfig
from skydiscover.search.base_database import Program, ProgramDatabase
from skydiscover.search.adaevolve.archive.diversity import CodeDiversity
from skydiscover.search.fore.fertility import (
    FertilityStats, NIGPrior, pov_score, fertility_multiplier,
)
from skydiscover.search.fore.descriptions import (
    StrategyDescription, parse_strategy_block, compute_verdict,
)
from skydiscover.search.fore.review import FertilityReview
from skydiscover.utils.metrics import get_score

logger = logging.getLogger(__name__)

class FOREDatabase(ProgramDatabase):
    def __init__(self, name: str, config: DatabaseConfig):
        super().__init__(name, config)
        self.prior = NIGPrior(
            mu_0=getattr(config, "prior_mu_0", 0.0),
            kappa_0=getattr(config, "prior_kappa_0", 2.0),
            alpha_0=getattr(config, "prior_alpha_0", 2.0),
            beta_0=getattr(config, "prior_beta_0", 0.5),
        )
        self.fertility: Dict[str, FertilityStats] = {}   # parent_id -> stats
        self.clusters: Dict[int, List[str]] = {}         # cluster_id -> [program_id]
        self.program_cluster: Dict[str, int] = {}        # program_id -> cluster_id
        self.diversity = CodeDiversity()
        self.population_size = getattr(config, "population_size", 80)
        self.cluster_similarity_threshold = getattr(config, "cluster_similarity_threshold", 0.55)
        self.k_remaining_init = getattr(config, "k_remaining", 100)
        self.k_neighbors = getattr(config, "k_neighbors", 5)

        # Review (Reflective)
        self._active_review: Optional[FertilityReview] = None
        self.review_rate_threshold = getattr(config, "review_rate_threshold", 0.1)
        self.review_window = getattr(config, "review_window", 12)
        self.pov_floor = getattr(config, "pov_floor", 0.0)
        self._recent_improvements: List[float] = []

        self._iteration = 0
        self._rng = random.Random(getattr(config, "random_seed", 42))

    # ---------- ProgramDatabase interface ----------
    def add(self, program: Program, iteration: Optional[int] = None, **kwargs) -> str:
        # 1. Compute novelty / cluster id
        # 2. Init FertilityStats for this program (n=0)
        # 3. If parent_id is known, update parent FertilityStats with Δ = f(child)-f(parent)
        # 4. Compute and store verdict on the *parent* update + on this program
        # 5. Push to programs dict, update best_program, save if configured
        # 6. Enforce population limit using fertility-aware eviction
        # 7. Update recent_improvements for stagnation tracking
        ...

    def sample(self, num_context_programs: Optional[int] = 4, **kwargs) -> Tuple[Dict[str, Program], Dict[str, List[Program]]]:
        # 1. Compute POV for all programs via Thompson sampling
        # 2. Pick top-1 as parent
        # 3. Pick context: 1 sibling (same cluster, complementary verdict) +
        #    (num-2) cross-cluster diverse exemplars
        # 4. Return dict-wrapped (key encodes 'fertility' label for prompt injection)
        ...

    # ---------- New (FORE-specific) ----------
    def detect_stagnation(self) -> Tuple[bool, str]:
        """Return (should_review, reason)."""
        # rule 1: rolling mean of recent_improvements < review_rate_threshold
        # rule 2: max POV of top-10 < pov_floor
        # rule 3: all-cluster mean Δ⁺ < epsilon
        ...

    def set_active_review(self, review: FertilityReview) -> None:
        self._active_review = review

    def consume_review_use(self) -> Optional[FertilityReview]:
        """Return current review and decrement its remaining uses."""
        ...

    def get_fertility_summary(self) -> List[Dict[str, Any]]:
        """Per-cluster summary used by ReflectiveReviewer prompt."""
        ...

    def get_recent_attempts(self, n: int = 10) -> List[Dict[str, Any]]:
        """Last n programs with their strategy description + verdict."""
        ...

    # ---------- Internals ----------
    def _assign_cluster(self, program: Program) -> int: ...
    def _evict_one(self) -> None: ...   # fertility-aware: protect high-POV + best
    def _record_improvement(self, child: Program, parent_id: Optional[str]) -> None: ...
```

**Sampling cụ thể** (cốt lõi):
```python
def sample(self, num_context_programs=4, **kwargs):
    if not self.programs:
        raise ValueError("Empty FORE database")
    pool = list(self.programs.values())
    # 1. Thompson
    scored = []
    for p in pool:
        stats = self.fertility[p.id]
        s = pov_score(
            fitness=get_score(p.metrics),
            stats=stats,
            prior=self.prior,
            k_remaining=max(1, self.k_remaining_init - self._iteration),
            rng=self._rng,
            iteration=self._iteration,
        )
        scored.append((s, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    parent = scored[0][1]
    parent_cluster = self.program_cluster[parent.id]

    # 2. Context: 1 sibling complementary + cross-cluster diverse
    siblings_same = [p for s, p in scored
                     if self.program_cluster[p.id] == parent_cluster and p.id != parent.id]
    sibling = siblings_same[0] if siblings_same else None
    cross = []
    seen = {parent_cluster}
    for s, p in scored:
        c = self.program_cluster[p.id]
        if c in seen: continue
        cross.append(p); seen.add(c)
        if len(cross) >= max(0, num_context_programs - 1):
            break

    context = []
    if sibling: context.append(sibling)
    context.extend(cross)

    parent_label = self._build_parent_label(parent, scored[0][0])
    return {parent_label: parent}, {"": context}
```

`_build_parent_label` trả về **một block prompt giống pattern `EXPLORE_LABEL/EXPLOIT_LABEL` của AdaEvolve** nhưng giải thích *vì sao parent này được chọn* (fitness, sampled μ⁺, novelty), ép LLM tận dụng đúng tính chất stepping stone.

Độ khó: trung bình. ~500 LOC tổng (kể cả docstring).

### 4.5 `skydiscover/search/fore/controller.py` — MỚI

**Mục đích**: subclass `DiscoveryController`. Trách nhiệm:
- Trước mỗi iteration: gọi `database.detect_stagnation()` → nếu cần, await `ReflectiveReviewer.generate()` rồi `database.set_active_review()`.
- Sau khi LLM trả về: gọi `parse_strategy_block(response)` để lấy `StrategyDescription`, gắn vào `program.metadata["fore"]` trước khi `database.add(...)`.
- Inject `_active_review` vào `_prompt_context` (đã có sẵn pattern này trong base controller).
- Logging JSONL như AdaEvolve để post-analysis (POV distribution, review events).

Phần lớn override chỉ là 2 hook: `_build_prompt` (đã có trong base — chỉ cần extra context key) và `_create_child_program` (gắn strategy desc). Vì vậy class này khá ngắn.

**Skeleton**:
```python
# controller.py
import logging
from typing import Optional, Dict, Any
from skydiscover.search.default_discovery_controller import (
    DiscoveryController, DiscoveryControllerInput,
)
from skydiscover.search.fore.review import ReflectiveReviewer
from skydiscover.search.fore.descriptions import parse_strategy_block, compute_verdict
from skydiscover.context_builder.fore.builder import FOREContextBuilder
from skydiscover.search.utils.discovery_utils import SerializableResult, load_evaluator_code

logger = logging.getLogger(__name__)

class FOREController(DiscoveryController):
    def __init__(self, controller_input: DiscoveryControllerInput):
        super().__init__(controller_input)
        self.context_builder = FOREContextBuilder(self.config)
        self.reviewer = ReflectiveReviewer(
            llm_pool=self.guide_llms,
            system_message=self.config.context_builder.system_message or "",
            evaluator_code=load_evaluator_code(self.evaluation_file),
        )
        self._review_cooldown = getattr(self.config.search.database, "review_cooldown", 20)
        self._last_review_iter = -10**9

    async def run_discovery(self, start_iteration, max_iterations, checkpoint_callback=None,
                            post_process_result=True, retry_times=3):
        # 1) Maybe trigger Reflective Review (every iter check, cheap)
        # 2) Then delegate to base loop for the heavy iteration work
        # We override _run_iteration only to add (a) review check, (b) description parsing.
        return await super().run_discovery(
            start_iteration, max_iterations, checkpoint_callback, post_process_result, retry_times
        )

    async def _run_iteration(self, iteration, retry_times=1):
        await self._maybe_run_review(iteration)
        result = await super()._run_iteration(iteration, retry_times=retry_times)
        # Parse strategy description from llm_response, attach onto child program dict
        if result and not result.error and result.llm_response:
            sd = parse_strategy_block(result.llm_response)
            cp = result.child_program_dict
            cp.setdefault("metadata", {})
            cp["metadata"]["fore"] = sd.to_dict()
        return result

    async def _maybe_run_review(self, iteration: int) -> None:
        if iteration - self._last_review_iter < self._review_cooldown:
            return
        should, reason = self.database.detect_stagnation()
        if not should:
            return
        review = await self.reviewer.generate(
            fertility_summary=self.database.get_fertility_summary(),
            recent_attempts=self.database.get_recent_attempts(10),
            global_best_score=self.database.get_program_proxy_score()
                if hasattr(self.database, "get_program_proxy_score") else 0.0,
        )
        if review:
            self.database.set_active_review(review)
            self._last_review_iter = iteration
            logger.info(f"FORE: Reflective Review triggered at iter {iteration} (reason={reason})")
```

Note: dùng `super().run_discovery` để tận dụng parallel/sequential loop có sẵn — không tự viết lại. Đây là chỗ giữ codebase **không bị rối** (đúng yêu cầu user).

Độ khó: thấp. ~250 LOC.

### 4.6 `skydiscover/context_builder/fore/builder.py` — MỚI

Subclass `DefaultContextBuilder`. Inject 3 thứ vào prompt:
1. **Parent fertility label** (từ key của parent_dict, đã có pattern AdaEvolve).
2. **Active Reflective Review** (nếu có) — format như block markdown.
3. **Hậu duệ verdict** của parent (vài dòng): xem các con của parent đã thử cái gì, verdict ra sao → giúp LLM tránh lặp.
4. **Yêu cầu output**: thêm chỉ dẫn “trả về `<fore_meta>{json}</fore_meta>` block bên cạnh code/diff” để parse description.

Template `full_rewrite_user_message.txt` (skeleton):
```
{system_prelude}

## CURRENT PROGRAM
{current_program}

## PARENT SELECTION REASONING
{parent_label}

{review_block}

## SIBLING VERDICTS (children of this parent)
{sibling_verdicts}

## OTHER CONTEXT PROGRAMS
{other_context_programs}

## INSTRUCTIONS
1. Propose a substantive improvement (or a deliberate stepping-stone exploration if review
   suggests this lineage is exhausted).
2. After your code/diff, append exactly one JSON block:
   <fore_meta>
   {{
     "strategy_label": "<short tag>",
     "description": "<1-3 sentences on the algorithmic idea>",
     "hypothesis": "<why this could be better, even if score may not show it immediately>",
     "diff_from_parent": "<one line summary of change>"
   }}
   </fore_meta>
```

Builder code chủ yếu là string composition + delegate vào `super().build_prompt` — ~150 LOC.

### 4.7 `skydiscover/config.py` — MODIFY

Thêm dataclass + đăng ký vào `_DB_CONFIG_BY_TYPE`:
```python
@dataclass
class FOREDatabaseConfig(DatabaseConfig):
    population_size: int = 80
    random_seed: int = 42
    # Prior
    prior_mu_0: float = 0.0
    prior_kappa_0: float = 2.0
    prior_alpha_0: float = 2.0
    prior_beta_0: float = 0.5
    # Cluster
    cluster_similarity_threshold: float = 0.55
    k_neighbors: int = 5
    # POV
    k_remaining: int = 100
    fertility_alpha: float = 0.7
    fertility_k_max: int = 20
    # Review triggers
    review_rate_threshold: float = 0.1
    review_window: int = 12
    pov_floor: float = 0.0
    review_cooldown: int = 20
    # Description
    require_fore_meta_block: bool = True
```
Thêm `"fore": FOREDatabaseConfig` vào `_DB_CONFIG_BY_TYPE`.

### 4.8 `skydiscover/search/route.py` — MODIFY

Thêm 2 dòng:
```python
from skydiscover.search.fore.database import FOREDatabase
from skydiscover.search.fore.controller import FOREController
register_database("fore", FOREDatabase)
register_controller("fore", FOREController)
```

### 4.9 `configs/fore.yaml` — MỚI

Sao chép cấu trúc `configs/adaevolve.yaml` và đổi `search.type: fore`, set các hyperparam ở §4.7. Cho user dùng làm starter:
```yaml
max_iterations: 100
checkpoint_interval: 10
llm:
  models:
    - { name: "openai/gpt-5", weight: 1.0 }
  api_base: "https://openrouter.ai/api/v1"
search:
  type: "fore"
  num_context_programs: 4
  database:
    population_size: 80
    cluster_similarity_threshold: 0.55
    review_rate_threshold: 0.1
    review_window: 12
    review_cooldown: 20
prompt:
  system_message: |
    <REPLACE WITH PROBLEM DESCRIPTION>
evaluator:
  timeout: 360
diff_based_generation: true
max_solution_length: 60000
```

### 4.10 `tests/test_fore_fertility.py` — MỚI

Pure-math tests, không cần LLM:
- `test_posterior_convergence`: feed 200 N(0.5, 0.1) → posterior mean ∈ [0.48, 0.52].
- `test_thompson_explores_when_unseen`: 2 parents, A có 0 obs, B có 50 obs mean=0; trong 1000 lần Thompson, A được chọn ≥ 200 lần.
- `test_stepping_stone_preferred`: A fitness=0.3, μ⁺=0.4 vs B fitness=0.6, μ⁺=0.01; sau 200 update mỗi parent, A trung bình có POV > B.
- `test_fertility_multiplier_monotone`.

---

## §5. Stagnation Review Pipeline (chi tiết)

Trigger (đọc trong `FOREDatabase.detect_stagnation`):
1. **Rate trigger**: `sum(recent_improvements[-W:]) / W < review_rate_threshold` với `W = review_window`.
2. **POV-floor trigger**: lấy 10 parent có POV cao nhất sau khi Thompson sample 5 lần và lấy median — nếu median < `pov_floor` thì kích hoạt.
3. **All-cluster exhausted**: với mỗi cluster, tính `mean Δ⁺` của các program trong đó từ `FertilityStats`; nếu max của các cluster < ε, kích hoạt.

Bất kỳ 1 trigger nào thỏa và cooldown đã qua → `FOREController` await `ReflectiveReviewer.generate()`.

Output review được inject vào prompt qua `FOREContextBuilder` trong `uses_remaining` generations tiếp (default 3). Sau khi hết, review được đẩy vào `tried_reviews` history (tương tự `tried_paradigms` của AdaEvolve) để tránh lặp.

**Vì sao thiết kế này khác AdaEvolve paradigm**:
- Paradigm là “1 idea cho 1 mutate”; review là “bản đồ cho 1 cửa sổ”.
- Review nhận **fertility map** làm input, do đó nó “biết” cluster nào đáng đầu tư.
- Output review có 4 trường (effective/exhausted/embryonic/next_steps) thay vì 5 trường của paradigm.

---

## §6. Cấu hình & cách chạy

Sau khi implement xong:
```bash
uv sync
uv run python scripts/install_benchmark_requirements.py benchmarks/math/circle_packing

# Smoke test 5 iter
uv run skydiscover-run benchmarks/math/circle_packing/initial_program.py \
  benchmarks/math/circle_packing/evaluator.py \
  --config configs/fore.yaml \
  --search fore \
  --model openrouter/openai/gpt-5 \
  --iterations 5 \
  --output outputs/local/fore_smoke
```

Kiểm tra output:
- `outputs/.../logs/*.log` có dòng `FOREController initialized` + `Reflective Review triggered`.
- `outputs/.../checkpoints/.../programs/*.json` có field `metadata.fore` chứa strategy description.

---

## §7. Feasibility & rủi ro

**Khả thi** — cao. Tổng LOC ước tính ~1.6k, thấp hơn AdaEvolve (~5k). Lý do:
- Toàn bộ orchestration (LLM call, eval, diff parsing, checkpointing, monitor) được kế thừa từ `DiscoveryController` và `ProgramDatabase`.
- Math thuần Python (NIG conjugate + Student-t sampling); chỉ cần `random` + `math`, không cần `scipy`.
- Reuse `CodeDiversity` cho novelty, không cần embedding model ngoài.
- Strategy description parse bằng regex + JSON, đã có pattern tương tự cho `extract_diffs`.

**Rủi ro / điểm cần test sớm**:
1. **LLM có chịu output `<fore_meta>` block không?** — Mitigation: prompt template nhấn mạnh; fallback parser luôn cho `StrategyDescription` rỗng nếu thiếu (không crash). Có thể fallback gọi 1 LLM call ngắn để sinh description nếu thiếu.
2. **Phân phối Δ⁺ thường lệch (heavy-tail)** — Student-t đã bền với điều này, nhưng có thể cần clip `Δ⁺` ở $B_{\max}$ trước khi update để giữ posterior ổn định. Đã thiết kế `update_with_child` có clip.
3. **Cluster bằng Jaccard có thể quá thô** — Acceptable trong v1; ablation v2 có thể dùng `MetricDiversity` hoặc embedding.
4. **Stagnation triggers có thể quá nhạy hoặc quá ì** — `review_cooldown` + 3 trigger có ngưỡng tunable; expose trong YAML.
5. **Parallel iteration**: base loop có chế độ `max_parallel_iterations > 1`. POV update có thể race; `FertilityStats.update_with_child` chỉ làm các phép cộng simple — không cần lock vì base controller `_process_iteration_result` chạy ngoài semaphore và atomic giữa các await (đã ghi rõ ở `_bounded_iteration`). Vẫn nên smoke test `max_parallel_iterations=1` trước.
6. **Checkpoint**: cần override `save`/`load` để serialize `fertility`, `clusters`, `program_cluster`, `_active_review`. Pattern này có sẵn ở `AdaEvolveDatabase.save/load` — copy structure.

---

## §8. Outline paper + thí nghiệm

### 8.1 Outline (8 sections)

1. **Introduction** — đặt câu hỏi “best program vs best-stepping-stone program”.
2. **Background** — FunSearch, AlphaEvolve, AdaEvolve (UCB ở island), EvoX (co-evolve), TusoAI (strategy memory). Nhấn vì sao chưa ai trả lời câu hỏi parent-selection ở mức program với mô hình Bayesian.
3. **Posterior Offspring Value** — §2 của plan này: định nghĩa $V_K$, Lemma 1, NIG posterior, structural prior từ description, Thompson sampling, regret bound.
4. **FORE Architecture** — §3 + Reflective Review.
5. **Reflective Review** — triggers, prompt schema, decay.
6. **Experiments**:
   - Benchmarks: `benchmarks/math/circle_packing` (2 variants), `benchmarks/math/heilbronn_triangles`, `benchmarks/math/minmax_distance` (đã có sẵn), thêm 1–2 ADRS benchmark cho “open-ended” gain (giống TusoAI).
   - Baselines: AdaEvolve, OpenEvolve_native, GEPA_native, TopK; nếu có budget thì cả Best-of-N.
   - Metric: best-so-far fitness (theo iteration + theo API cost), số iter để đạt mốc reference (như Figure 1 của TusoAI).
   - 3 seeds; báo cáo mean ± std.
7. **Ablations**:
   - **POV vs fitness-greedy**: vô hiệu Thompson, chọn theo $f(p)$ thuần. Kỳ vọng: degrade rõ trên Heilbronn (deceptive).
   - **No structural prior**: $w_1 = w_2 = w_3 = 0$. Kỳ vọng: cold-start chậm.
   - **No Reflective Review**: tắt trigger. Kỳ vọng: flat plateau dài hơn.
   - **No strategy description**: bỏ block, dùng `"changes"` đơn dòng như AdaEvolve. Kỳ vọng: review chất lượng kém, gain nhỏ.
   - **Hyperparam sweep**: `cluster_similarity_threshold ∈ {0.4, 0.55, 0.7}`, `review_window ∈ {6, 12, 24}`.
8. **Discussion** — limitations: Jaccard cluster thô, prior calibration phụ thuộc task; future: tích hợp embedding hoặc thay Student-t bằng Gaussian Process trên code embedding.

### 8.2 Câu hỏi nghiên cứu chính của paper

> **RQ1**: Liệu chọn parent theo posterior offspring value có outperform parent theo current fitness trên các task có deceptive landscape?
> **RQ2**: Stagnation review ở mức *fertility map* (cluster-level) có ổn định hơn paradigm breakthrough ở mức program đơn lẻ?
> **RQ3**: Strategy description có làm review hữu ích hơn (đo bằng % iterations trong cửa sổ post-review mà có improvement)?

Mỗi RQ map vào một ablation ở §8 phần 7.

### 8.3 Lý do định lượng để reviewer bị thuyết phục

- Thompson sampling regret là $O(\sqrt{KT\log T})$, *không phụ thuộc đặc tính fitness landscape* (frequentist guarantee qua Russo–Van Roy).
- Lemma 1 cho biết $V_K$ scale theo $\mathbb{E}[\Delta^+]$ — không phải $f$ — nên fitness-greedy *về mặt lý thuyết* dưới optimal khi $\Delta$-distribution lệch trái với fitness rank.
- Empirically: trong các tree-search literature, “virtual loss + fertility” lâu đời (POH-AlphaZero). FORE đem framework này về evolutionary LLM search lần đầu.

---

## §9. Lộ trình thực thi (đề xuất 4 PR)

| PR | Phạm vi | Test |
|---|---|---|
| 1 | `fertility.py` + `descriptions.py` + unit test math | `pytest tests/test_fore_fertility.py` xanh |
| 2 | `database.py` + register + minimal `controller.py` (chưa có review) + `fore.yaml` | Smoke 5 iter trên circle_packing chạy được, sinh checkpoint hợp lệ |
| 3 | `context_builder/fore/` + parse strategy block end-to-end | Verify `metadata.fore.description` xuất hiện trong checkpoint |
| 4 | `review.py` + trigger logic + reflective injection vào prompt | Force stagnation → quan sát log `Reflective Review triggered` + improvement curve sau review |

Mỗi PR ≤ ~600 LOC để dễ review.

---

## §10. Tóm lược

- Tận dụng `ProgramDatabase`, `DiscoveryController`, `DefaultContextBuilder`, `CodeDiversity`, `LLMPool`, checkpoint manager — *không* viết lại bất cứ thứ gì đã có.
- Đóng góp khoa học: (i) Posterior Offspring Value với prior có cấu trúc, (ii) Thompson sampling cho parent-selection trong LLM-driven evolutionary search, (iii) Reflective Review ở mức fertility map, (iv) Strategy description như first-class persistent state nhưng giữ self-contained (Jaccard, không embedding ngoài).
- Đóng góp engineering: 1 module mới `search/fore/`, 1 builder mới `context_builder/fore/`, 2 chỗ sửa nhỏ (config + route).
- Khả thi cao; rủi ro chính ở việc LLM tuân thủ format `<fore_meta>` (đã có fallback) và việc calibrate hyperparam stagnation (đã expose YAML).
