# Hướng dẫn chạy thực nghiệm trên server (SSH + tmux + conda)

Tài liệu này dành cho người **chưa từng dùng SSH / tmux / server**. Cứ làm theo
từng bước, mỗi lệnh copy-paste được luôn.

Quy trình làm việc tổng thể:

```
Máy của bạn (code + sửa code)  →  git push  →  GitHub  →  git pull  →  Server (chỉ chạy)
```

Trên server bạn **không sửa code**, chỉ `git pull` rồi chạy script.

---

## Mục lục

1. [Khái niệm 30 giây: SSH, tmux, conda](#1-khái-niệm-30-giây)
2. [Đăng nhập vào server bằng SSH](#2-đăng-nhập-vào-server-bằng-ssh)
3. [Lấy code về server](#3-lấy-code-về-server)
4. [Tạo file `.env` (chứa API key)](#4-tạo-file-env-chứa-api-key)
5. [Cài môi trường conda tên `minhhieu`](#5-cài-môi-trường-conda-tên-minhhieu)
6. [Chạy benchmark](#6-chạy-benchmark)
   — [6b. LLM-SRBench / LSR-Synth](#6b-llm-srbench--lsr-synth--dùng-script-riêng)
   — [6c. RelayEvolve + baseline cheap/strong](#6c-relayevolve--5-baseline-cheapstrong--dùng-run_relaysh)
7. [tmux — chạy rồi tắt máy vẫn không sao](#7-tmux--chạy-rồi-tắt-máy-vẫn-không-sao)
8. [Xem log / xem kết quả](#8-xem-log--xem-kết-quả)
9. [Lấy kết quả từ server về máy](#9-lấy-kết-quả-từ-server-về-máy)
10. [Bảng tra tham số đầy đủ](#10-bảng-tra-tham-số-đầy-đủ)
11. [Sự cố thường gặp](#11-sự-cố-thường-gặp)

---

## 1. Khái niệm 30 giây

| Thứ | Là gì | Vì sao cần |
|---|---|---|
| **SSH** | Lệnh để "vào" một máy tính khác qua mạng và gõ terminal trên máy đó | Server không có màn hình, chỉ vào được bằng SSH |
| **tmux** | Một "màn hình ảo" sống **bên trong server** | Khi bạn tắt laptop / rớt mạng, tiến trình chạy trong tmux **vẫn tiếp tục**. Nếu chạy trực tiếp, tắt SSH là job **chết** |
| **conda** | Trình quản lý môi trường Python | Cài thư viện vào một "hộp" riêng tên `minhhieu`, không đụng vào Python hệ thống |

Ý quan trọng nhất: **luôn chạy job dài trong tmux.**

---

## 2. Đăng nhập vào server bằng SSH

Trên terminal **máy của bạn** (Mac: mở app Terminal):

```bash
ssh <username>@<địa-chỉ-server>
```

Ví dụ: `ssh minhhieu@172.26.92.15` hoặc `ssh minhhieu@server.comp.nus.edu.sg`

- Lần đầu nó hỏi `Are you sure you want to continue connecting (yes/no)?` → gõ `yes` rồi Enter.
- Sau đó nó hỏi password → gõ password (**màn hình sẽ không hiện gì cả, kể cả dấu \***, đó là bình thường) rồi Enter.

Khi thấy dấu nhắc kiểu `minhhieu@server:~$` là bạn đang **ở trong server**. Từ giờ mọi lệnh gõ ra là chạy trên server.

Thoát ra: gõ `exit` hoặc bấm `Ctrl-D`.

### (Tuỳ chọn nhưng rất nên làm) Khỏi phải gõ password mỗi lần

Trên **máy của bạn**:

```bash
ssh-keygen -t ed25519            # Enter 3 lần, để mặc định hết
ssh-copy-id <username>@<địa-chỉ-server>   # gõ password lần cuối cùng
```

Từ lần sau `ssh <username>@<server>` sẽ vào thẳng, không hỏi password.

---

## 3. Lấy code về server

Sau khi đã SSH vào server:

```bash
# Kiểm tra có git chưa
git --version

# Clone repo (lần đầu tiên duy nhất)
cd ~
git clone --recurse-submodules <URL-repo-của-bạn> skydiscover
cd skydiscover
```

> `--recurse-submodules` là cần thiết vì repo có submodule (ALE-Bench).

**Những lần sau**, mỗi khi bạn đã push code mới từ máy mình:

```bash
cd ~/skydiscover
git pull
```

Nếu `git pull` báo lỗi vì server có thay đổi lặt vặt (file `.pyc`, output cũ...):

```bash
git checkout -- .      # bỏ mọi thay đổi cục bộ trên server
git pull
```

---

## 4. Tạo file `.env` (chứa API key)

File `.env` nằm trong `.gitignore` nên **không bao giờ được push lên GitHub**
(đó là chủ ý — API key không nên nằm trên GitHub). Vì vậy sau khi `git clone`,
trên server sẽ **không có** file này và bạn phải tự tạo nó bằng tay.

### Cách 1 (khuyến nghị): gõ trực tiếp trên server

```bash
cd ~/skydiscover
cat > .env <<'EOF'
OPENAI_API_KEY=sk-or-v1-....................
EOF
chmod 600 .env        # chỉ mình bạn đọc được file này
```

Thay `sk-or-v1-...` bằng key OpenRouter thật của bạn (lấy trên
https://openrouter.ai/keys). Lưu ý:

- Dùng `<<'EOF'` có dấu nháy đơn để shell không "ăn" ký tự đặc biệt trong key.
- Tên biến là `OPENAI_API_KEY`; script cũng chấp nhận `OPENROUTER_API_KEY`.
  Key bắt đầu bằng `sk-or-` được repo tự hiểu là key OpenRouter.

### Cách 2: copy file `.env` từ máy bạn sang server

Chạy lệnh này trên **máy của bạn** (KHÔNG phải trên server):

```bash
scp /Users/apple/Desktop/All/NUS_INTERNSHIP/skydiscover/.env \
    <username>@<địa-chỉ-server>:~/skydiscover/.env
```

### Kiểm tra key hoạt động

Trên server, sau khi đã cài môi trường ở bước 5:

```bash
cd ~/skydiscover
conda activate minhhieu
python scripts/test_openrouter_key.py
```

Thấy dòng `OpenRouter API key hoat dong binh thuong.` là ổn.

---

## 5. Cài môi trường conda tên `minhhieu`

### 5a. Nếu server chưa có conda

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
exec $SHELL -l          # nạp lại shell để lệnh `conda` có hiệu lực
conda --version         # phải in ra số phiên bản
```

> Nếu server không có `wget`, thay bằng
> `curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`.
> Nếu server là máy ARM (hiếm), đổi `x86_64` thành `aarch64`.

### 5b. Tạo môi trường (chạy 1 lệnh là xong)

```bash
cd ~/skydiscover
bash scripts/server/setup_env.sh
```

Script này sẽ: tạo env `minhhieu` với Python 3.12 → cài `skydiscover` (chế độ
editable) → cài các thư viện trong `scripts/server/requirements-server.txt`
(bao gồm toàn bộ dependency của `levi/`, tức phần BLADE) → cuối cùng **tự kiểm
tra import** và in `All core imports OK (skydiscover + levi).`

Mất khoảng 3–8 phút. Sau đó:

```bash
conda activate minhhieu
```

### 5c. Nếu bạn muốn tự gõ tay thay vì dùng script

Đây chính xác là những gì `setup_env.sh` làm:

```bash
conda create -y -n minhhieu python=3.12 pip
conda activate minhhieu
cd ~/skydiscover
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r scripts/server/requirements-server.txt
```

Python **3.12** là bắt buộc: `skydiscover` cần `>=3.10,<3.14`, còn `levi` cần
`>=3.11,<3.13` — chỉ 3.11/3.12 thoả cả hai, và 3.12 là bản đã kiểm chứng.

### 5d. Các benchmark "nặng" (chỉ cài khi cần)

Mặc định script **không** cài `torch`, `jax`, `triton` (rất nặng, vài GB). Chỉ
cài thêm nếu bạn chạy đúng nhóm benchmark đó:

```bash
bash scripts/server/setup_env.sh --extra math    # jax, optax, torch, numba, cvxpy, pymoo...
bash scripts/server/setup_env.sh --extra adrs    # torch, pandas, networkx<3.4
bash scripts/server/setup_env.sh --extra torch   # chỉ torch (ADRS/eplb, kernelbench)
```

Với các benchmark thông thường (`benchmarks/math/*`, `benchmarks/co_bench/*`),
script chạy benchmark sẽ **tự cài** `requirements.txt` riêng của benchmark đó
mỗi lần chạy — bạn không phải làm gì cả.

Các lệnh conda hữu ích khác:

```bash
conda env list                              # xem các env đang có
bash scripts/server/setup_env.sh --recreate # xoá và tạo lại env từ đầu
conda deactivate                            # thoát env
```

### 5f. Kiểm tra toàn bộ setup (miễn phí, không gọi API)

```bash
bash scripts/server/selftest.sh
```

Script này kiểm tra: conda/tmux/git đã có chưa, env `minhhieu` đúng Python
3.12 chưa, import được `skydiscover` + `levi` chưa, `.env` có key hợp lệ
chưa (và có bị để chế độ ai cũng đọc được không), repo có đủ file không,
`run_bench.sh` bắt lỗi tham số sai đúng không, và **chạy thử 2 tmux session
nháp** để chứng minh cơ chế chạy nền hoạt động. Không tốn một xu nào.

Cuối cùng in ra `PASS: N  FAIL: 0` là mọi thứ sẵn sàng. Có `FAIL` thì nó nói
luôn cần sửa gì. Chạy lại script này bất cứ lúc nào thấy có gì đó bất thường.

### 5e. Cài tmux (nếu server chưa có)

```bash
tmux -V                     # nếu in ra "tmux 3.x" là đã có, bỏ qua bước này
sudo apt-get install -y tmux                # nếu bạn có quyền sudo
conda install -y -c conda-forge tmux        # nếu KHÔNG có quyền sudo
```

---

## 6. Chạy benchmark

Tất cả chạy qua **một file duy nhất**: `scripts/server/run_bench.sh`.

Script này sao chép **nguyên xi** hai workflow GitHub Actions
(`.github/workflows/blade.yml` và `.github/workflows/baseline.yml`): cùng biến
môi trường, cùng bước cài dependency, cùng bước tải dataset, cùng dòng lệnh
cuối cùng, cùng giá trị mặc định. Mọi tham số bạn từng điền vào form "Run
workflow" trên GitHub đều có một cờ (flag) tương ứng ở đây, **cùng tên**.

### Xem toàn bộ tham số

```bash
./scripts/server/run_bench.sh --help
```

### Chạy BLADE (phương pháp của bạn)

```bash
cd ~/skydiscover
./scripts/server/run_bench.sh blade --tmux \
    --benchmark levi/examples/circle_packing \
    --evaluations 200 \
    --seed 1
```

### Chạy baseline (để so sánh)

```bash
./scripts/server/run_bench.sh baseline --tmux \
    --baseline openevolve_native \
    --benchmark-dir benchmarks/math/circle_packing \
    --iterations 100 \
    --model openrouter/openai/gpt-5 \
    --seed 1
```

### Thay đổi tham số — hoàn toàn tự do

Đổi benchmark, đổi model, đổi ablation, đổi ngân sách… chỉ cần thêm/bớt cờ:

```bash
# CO-Bench, giới hạn chi phí $5, dùng nhiều worker hơn
./scripts/server/run_bench.sh blade --tmux \
    --benchmark levi/examples/co_bench/tsp \
    --dollars 5 --workers 8 --cobench-timeout 10 --cobench-max-instances 0

# Chạy ablation A7 (bỏ crossover), seed 3
./scripts/server/run_bench.sh blade --tmux \
    --benchmark levi/examples/heilbronn_triangle \
    --evaluations 300 --ablation no_crossover --seed 3

# Chỉnh các knob cấp thấp bằng JSON (giống ô advanced_options trên GitHub)
./scripts/server/run_bench.sh blade --tmux \
    --benchmark levi/examples/circle_packing_rect --evaluations 300 \
    --advanced-options '{"n_cells":50,"p_targeted_mutate":0.5,"meta_advice_inject_p":0.35}'

# Baseline khác trên benchmark khác
./scripts/server/run_bench.sh baseline --tmux \
    --baseline evox --benchmark-dir benchmarks/co_bench/bin_packing_1d --iterations 150

# Baseline có trần chi phí: chạy tới 400 vòng nhưng tiêu đủ $3 là tự dừng
./scripts/server/run_bench.sh baseline --tmux \
    --baseline adaevolve --benchmark-dir benchmarks/math/circle_packing \
    --iterations 400 --dollars 3
```

### Giới hạn thời gian: `--timeout`

Trên GitHub Actions, baseline bị chặn cứng ở 3 tiếng (`timeout-minutes: 180`).
Server của bạn không có giới hạn đó, nên `run_bench.sh` **giữ nguyên mặc định
3 tiếng cho baseline** để kết quả so sánh được với các số bạn đã thu trước đây.

```bash
--timeout 3h        # mặc định của baseline (= 10800 giây)
--timeout 6h        # cho chạy lâu hơn
--timeout 0         # BỎ giới hạn — tận dụng server không bị cap
```

Chế độ `blade` mặc định **không** có timeout (đúng như `blade.yml`), vì BLADE đã
có sẵn ngân sách mềm `--seconds` / `--dollars` / `--evaluations`.

Khác biệt giữa hai loại giới hạn:

| Tiêu chí | `--seconds` (blade) | `--timeout` (cả hai) |
| --- | --- | --- |
| Cách dừng | BLADE tự dừng **êm**, ghi đầy đủ kết quả | Giết tiến trình bằng SIGTERM/SIGKILL |
| Exit status | 0 | **124** |
| Dùng khi | muốn kết thúc gọn gàng | muốn chặn cứng, phòng job treo |

Khi bị timeout, **kết quả dở dang vẫn được giữ nguyên** (log, cost_log,
checkpoint đã ghi tới thời điểm đó), và dòng cuối log ghi rõ:
`exit status : 124  (TIMED OUT after 10800s — partial results kept)`.

### Giới hạn tiền: `--dollars` (cả blade lẫn baseline)

`--timeout` chặn theo **thời gian**, `--dollars` chặn theo **tiền**. Baseline
giờ cũng có tham số này (giống `dollars` trong `blade.yml`), vừa là cờ của
`run_bench.sh` vừa là ô nhập trên form "Run workflow" của `baseline.yml`:

```bash
# Chạy tối đa 500 vòng, NHƯNG tiêu đủ $5 là tự dừng — cái nào đến trước thì thắng
./scripts/server/run_bench.sh baseline --tmux \
    --baseline evox --benchmark-dir benchmarks/co_bench/tsp \
    --iterations 500 --dollars 5
```

Cách nó hoạt động: mỗi lần gọi LLM, OpenRouter trả về `usage.cost` và
`skydiscover.llm.cost_tracker` cộng dồn vào `cost_log.totals.json`. Khi tổng
chạm ngưỡng, nó yêu cầu vòng lặp dừng **êm** — vòng lặp đang chạy dở được làm
nốt, best program + checkpoint được ghi đầy đủ, chấm điểm test cuối cùng vẫn
chạy, và **exit status vẫn là 0** (không phải 124 như timeout).

Vài điểm cần biết:

- **Đây là trần mềm.** Các request đã bay đi rồi vẫn hoàn thành, nên tổng cuối
  cùng thường **nhỉnh hơn** con số bạn đặt một chút (khoảng 1 vòng lặp; nếu
  `max_parallel_iterations` > 1 thì tối đa là bấy nhiêu vòng). Đặt $5 mà tiêu ra
  $5.12 là bình thường, không phải lỗi. Muốn chặn cứng tuyệt đối thì dùng thêm
  `--timeout`.
- Ngưỡng dựa trên chi phí do **nhà cung cấp báo về**, nên chỉ có tác dụng với
  OpenRouter (mọi model `openrouter/...` — tức là toàn bộ cấu hình mặc định ở đây).
- Bỏ trống hoặc để `0` = không giới hạn (y như trước khi có tham số này).
- Log in kèm tiến độ tiêu tiền ngay trên từng dòng iteration:
  `[cost=$1.2345/$5.00, llm_calls=87]`. Lúc dừng có dòng
  `💸 Spend budget reached: $5.0100 of $5.0000 ...`.
- Cuối `run.log` có thêm dòng `spend budget: $5` bên cạnh bảng `cost totals`.

### Thử trước khi chạy thật: `--dry-run`

**Rất nên dùng.** Nó in ra đúng dòng lệnh sẽ chạy rồi dừng lại, **không gọi API,
không tốn tiền**:

```bash
./scripts/server/run_bench.sh blade --dry-run \
    --benchmark levi/examples/circle_packing --evaluations 200 --ablation emb_only
```

Đọc dòng `>>> command:` để chắc chắn mọi tham số đúng ý, rồi bỏ `--dry-run`,
thêm `--tmux` và chạy thật.

### Chạy nhiều seed liên tiếp

```bash
for s in 1 2 3; do
  ./scripts/server/run_bench.sh blade --tmux --session blade_cp_seed$s \
      --benchmark levi/examples/circle_packing --evaluations 200 --seed $s
done
```

Ba job chạy **song song** trong ba tmux session. Nếu server yếu, giảm
`--workers` xuống, hoặc chạy tuần tự bằng cách bỏ `--tmux` trong vòng lặp và
bọc cả vòng lặp trong **một** tmux session.

### 6b. LLM-SRBench / LSR-Synth — dùng script riêng

LSR-Synth không phải một benchmark đơn lẻ mà là **một tập bài toán độc lập**
(10 bài mỗi domain trong cấu hình rút gọn, 129 bài nếu chạy full), nên nó có
script riêng chạy lần lượt từng bài, tự lưu checkpoint và **tự tiếp tục được nếu
bị ngắt giữa đường** (hết API, mất SSH, Ctrl-C).

Chỉ cần CPU — chấm điểm là BFGS của scipy + numpy, không dùng GPU.

#### 6b.1 Setup (một lần cho mỗi server)

**Cần gì so với env `minhhieu` đã có:** đúng **hai** thư viện, và chỉ cho bước
tải dataset một lần — `huggingface_hub` và `pyarrow`. Nếu env của bạn được tạo
trước khi LSR-Synth được thêm vào repo thì chưa có chúng; cài bằng cách chạy lại
script setup (idempotent, chỉ bổ sung phần thiếu):

```bash
conda activate minhhieu
bash scripts/server/setup_env.sh          # = pip install -e . + requirements-server.txt

# hoặc tối giản, chỉ hai gói đó:
pip install "huggingface_hub>=0.24" "pyarrow>=15.0"
```

Kiểm tra:

```bash
python -c "import numpy, scipy, huggingface_hub, pyarrow; print('deps OK')"
```

**Không cần dependency của repo LLM-SRBench gốc** (torch, transformers, vllm...).
Repo này không import repo đó: giao thức chấm điểm được viết lại trong
`benchmarks/llm_srbench/lsr_eval.py`, và nó chỉ dùng **numpy + scipy** — cả hai
đã nằm trong dependency nền của `skydiscover`. Dataset thì đọc thẳng từ parquet
trên HuggingFace.

Sau đó hai bước:

```bash
# 1. Tải dataset — ~9 MB, một lần duy nhất cho cả 4 domain
python benchmarks/llm_srbench/prepare_data.py
python benchmarks/llm_srbench/prepare_data.py --check     # xác nhận

# 2. Self-test: 16 kiểm tra, KHÔNG gọi API, không tốn tiền
bash scripts/server/selftest_lsr_synth.sh
```

Self-test báo thiếu gói nào thì nó nói rõ tên gói và câu lệnh cài (mục 1 của nó
là "Packages").

Self-test phải ra `16 passed, 0 failed` trước khi chạy thật. Nó kiểm tra:
packages, dataset toàn vẹn, cả 40 problem cho **điểm giống nhau trên cả hai
path** (baseline và SpecEvo), equation ground-truth recover đúng tham số, mọi
failure mode trả về 0 kèm lý do, timeout 30 s bắn đúng giờ, và dry-run runner.

#### 6b.2 Dataset: `git pull` **không** phải tải lại

Không. Dữ liệu nằm ở `benchmarks/llm_srbench/data/`, thư mục này **nằm trong
`.gitignore`** nên git không bao giờ chạm vào nó — `git pull` không xoá, không
ghi đè, không cần tải lại. Nó nằm nguyên trên đĩa server qua mọi lần pull.

`prepare_data.py` cũng **idempotent**: nếu dữ liệu đã đủ, nó thoát ngay mà không
tải và không ghi gì (~0.7 s):

```text
LSR-Synth data already prepared in .../benchmarks/llm_srbench/data — nothing to do.
```

Vì vậy `run_lsr_synth.sh` gọi nó trước **mỗi** lần chạy cũng không tốn gì — đó
chính là lý do đường fast-path đó tồn tại (và nó dùng file lock, nên 4 domain
launch cùng lúc cũng không đua nhau ghi `problems.json`).

Khi nào thật sự phải tải lại:

| tình huống | có tải lại? |
|---|---|
| `git pull` / `git checkout` / đổi branch | **không** |
| bạn tự xoá `benchmarks/llm_srbench/data/` | có — nhưng thường lấy từ cache HuggingFace (`~/.cache/huggingface/hub`), không cần mạng |
| clone mới sang máy/server khác | có, một lần |
| `--force` (ghi lại `.npz` đã có) | có |

Muốn bỏ qua bước chuẩn bị hoàn toàn khi đã biết dữ liệu sẵn sàng:
`run_lsr_synth.sh --no-install-deps`.

Server không có internet? Tải parquet ở máy khác rồi:
`python benchmarks/llm_srbench/prepare_data.py --from-local <dir>`.

#### 6b.3 Chạy

Một lệnh cho mỗi (method, domain). `--tmux` để job sống qua SSH disconnect:

```bash
# SpecEvo trên cả 4 domain, cấu hình rút gọn (10 bài/domain, 500 evals/bài)
for d in chem_react bio_pop_growth phys_osc matsci; do
    ./scripts/server/run_lsr_synth.sh --method specevo --domain "$d" --tmux
done

# Một baseline
./scripts/server/run_lsr_synth.sh --method openevolve_native --domain matsci --tmux
```

Method: `specevo` (alias `blade`), `openevolve_native`, `gepa_native`,
`adaevolve`, `evox`. Domain: `chem_react`, `bio_pop_growth`, `phys_osc`,
`matsci`.

**Full benchmark** — toàn bộ bài của domain thay vì 10 bài đầu. Repo chỉ commit
thư mục cho 10 bài đầu, phần còn lại script tự sinh khi bắt đầu chạy:

```bash
./scripts/server/run_lsr_synth.sh --method specevo --domain matsci --full --tmux
```

| domain | full | rút gọn |
|---|---|---|
| `chem_react` | 36 | 10 |
| `phys_osc` | 44 | 10 |
| `matsci` | 25 | 10 |
| `bio_pop_growth` | 24 | 10 |
| **tổng mỗi method** | **129** | 40 |

#### 6b.4 Đặt tên tmux session để dễ track

Giống các benchmark khác, session được sinh tự động theo run, nhưng bạn đổi được
bằng `--session`. Quy tắc mặc định:

```text
lsr_<method>_<domain>_seed<N>
```

Ví dụ, 8 job của một sweep 2 method × 4 domain sẽ tự có tên phân biệt sẵn:

```text
lsr_specevo_chem_react_seed1        lsr_openevolve_native_chem_react_seed1
lsr_specevo_bio_pop_growth_seed1    lsr_openevolve_native_bio_pop_growth_seed1
lsr_specevo_phys_osc_seed1          lsr_openevolve_native_phys_osc_seed1
lsr_specevo_matsci_seed1            lsr_openevolve_native_matsci_seed1
```

Chạy ablation thì tên có thêm ablation: `lsr_specevo-no_crossover_matsci_seed1`.
Dấu `.` và `:` bị đổi thành `_` (tmux không cho phép).

Đặt tên tay khi cần phân biệt hai lần chạy cùng cấu hình — ví dụ một lần full,
một lần rút gọn:

```bash
./scripts/server/run_lsr_synth.sh --method specevo --domain matsci --full \
    --session lsr_specevo_matsci_FULL --output-dir outputs/lsr_synth_full/specevo/matsci/seed1 --tmux
```

⚠️ Nếu đổi `--session` mà **không** đổi `--output-dir` thì hai job vẫn ghi vào
cùng một thư mục và sẽ phá nhau. Muốn hai run song song thật sự độc lập thì đổi
`--seed` (nó nằm trong đường dẫn output) hoặc đặt `--output-dir` riêng.

```bash
tmux ls                                        # xem tất cả session đang chạy
tmux attach -t lsr_specevo_matsci_seed1        # vào xem; detach: Ctrl-b rồi d
tmux kill-session -t lsr_specevo_matsci_seed1  # dừng (được handle sạch)
```

Job đã chạy rồi mà launch lại trùng tên thì script **từ chối** và nhắc cách
attach hoặc `--session NAME` khác, chứ không âm thầm chạy chồng lên.

#### 6b.5 Bảng tham số đầy đủ của `run_lsr_synth.sh`

**Bắt buộc**

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--method NAME` | `specevo` | `specevo` (alias `blade`), `openevolve_native`, `gepa_native`, `adaevolve`, `evox` |
| `--domain NAME` | **không có** | `chem_react`, `bio_pop_growth`, `phys_osc`, `matsci`. Phải truyền |

**Chọn bài**

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--problems N` | `10` | N bài đầu của domain. Nhận cả `all` (= toàn bộ) |
| `--full` | tắt | Viết tắt của `--problems all` (129 bài trên cả 4 domain) |
| `--problem-list LIST` | rỗng | Danh sách id cách nhau bằng dấu phẩy, ghi đè `--problems`. Prefix theo domain: `crk*`, `bpg*`, `po*`, `matsci*` — **id không liên tục** (matsci có 25 bài nhưng đánh số tới `matsci28`), nên lấy id thật bằng `--dry-run` |

Thư mục cho bài chưa có sẽ **tự sinh** khi bắt đầu chạy — không phải chạy
`generate_dirs.py` bằng tay.

**Ngân sách — tất cả áp dụng cho MỖI bài, không phải cho cả domain**

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--iterations N` | `500` | Số eval mỗi bài. **Chính xác** với baseline. Với `specevo` thì bootstrap của BLADE (`--n-diverse-seeds` × `--n-variants-per-seed`, ~105 eval ở mặc định) luôn chạy hết trước, nên giá trị dưới ~105 không thu nhỏ run — phải hạ hai knob bootstrap cùng lúc |
| `--dollars N` | rỗng (tắt) | Trần USD mỗi bài; search tự dừng êm ở ranh giới iteration kế tiếp. 10 bài × `--dollars 2` ≈ \$20/domain |
| `--seconds N` | rỗng (tắt) | Trần thời gian mỗi bài, **chỉ specevo** |
| `--problem-timeout N` | `7200` (2 h) | Trần thời gian thực cứng mỗi bài. Hết giờ thì search bị kill, **giữ lại những gì đã tìm được**, ghi kết quả rồi sang bài kế. `0` = tắt |
| `--eval-timeout N` | `30` | Giây cho mỗi hypothesis (fit BFGS + chấm 3 split). Đúng `T = 30s` của paper. SpecEvo nhận `N+60` cho timeout ngoài của nó, để cảnh báo 30 s của ta bắn trước và hypothesis bị cho 0 điểm thay vì worker bị kill |

Bài bị `--problem-timeout` cắt **không mất** — nhưng một method bị cắt ở điểm
khác các method còn lại thì không còn là so sánh công bằng, nên kiểm tra cột
`evaluations` trong `results.jsonl` trước khi kết luận.

**Dữ liệu và cách chấm điểm**

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--score-mode MODE` | `log_nmse` | Tín hiệu search. `log_nmse` = `log10(1 + 1/NMSE)`, đọc là "số decade NMSE dưới 1" (3.0 ⇒ NMSE 1e-3). `inv_nmse` = `1/(1+NMSE)` — thang cũ, bão hoà thành `1.0000` với mọi NMSE < 1e-4; chỉ dùng để reproduce run cũ. Cả hai đều đơn điệu theo NMSE nên xếp hạng như nhau |
| `--max-fit-points N` | `0` | `0` = fit trên **toàn bộ** 4000 điểm train, tức benchmark đúng như công bố. Khác `0` là smoke test và header sẽ in cảnh báo |

**Model**

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--model ID` | `openrouter/qwen/qwen3-30b-a3b-instruct-2507` | Model của baseline |
| `--mutation-model ID` | cùng qwen3-30b | SpecEvo Speculator (model nhỏ) |
| `--paradigm-model ID` | cùng qwen3-30b | SpecEvo Navigator (model mạnh). Ở thực nghiệm này cố tình đặt **bằng** Speculator thay vì tách nhỏ/frontier |
| `--embedding-model ID` | `openrouter/openai/text-embedding-3-small` | Embed description cho archive hành vi của SpecEvo |

**Hình dạng SpecEvo (baseline bỏ qua hết)**

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--workers N` | `4` | Số LLM worker song song. Đây là toàn bộ lý do SpecEvo nhanh hơn baseline ~4× về wall clock |
| `--eval-processes N` | `4` | Số process chấm điểm song song |
| `--pe-interval N` | `10` | Nhịp gọi Navigator (paradigm shift) |
| `--n-diverse-seeds N` | `5` | Số seed đa dạng ở pha 1 |
| `--n-variants-per-seed N` | `20` | Số biến thể mỗi seed ở pha 2 |
| `--ablation NAME` | `full` | `full`, `ast_only`, `emb_only`, `static_cells`, `no_meta_advice`, `meta_errors_only`, `no_targeted_mutate`, `no_crossover`, `no_paradigm`. Khác `full` thì tên run và output dir có kèm ablation |

**Điều khiển run**

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--seed N` | `1` | Nhãn seed; nằm trong đường dẫn output nên đây là cách chạy nhiều lần độc lập |
| `--output-dir DIR` | `outputs/lsr_synth/<method>/<domain>/seed<N>` | Không có timestamp — **đó chính là cơ chế resume** |
| `--fresh` | tắt | Xoá kết quả cũ của (method, domain, seed) và làm lại từ đầu. Không có cờ này thì run cũ được **tiếp tục** |
| `--status` | – | In tiến độ rồi thoát. Không gọi API, không tác dụng phụ |
| `--tmux` | tắt | Chạy nền trong tmux (sống qua SSH disconnect) |
| `--session NAME` | `lsr_<method>_<domain>_seed<N>` | Tên tmux session (mục 6b.4) |
| `--conda-env NAME` | `minhhieu` | Env conda cần activate |
| `--no-conda` | – | Dùng python đang active, không đụng conda |
| `--no-install-deps` | – | Bỏ bước chuẩn bị dataset (vốn chỉ mất ~0.7 s khi đã có dữ liệu) |
| `--dry-run` | – | Chỉ in lệnh sẽ chạy rồi thoát: không gọi API, không tốn tiền, không chạy search. (Nó **vẫn** tạo thư mục output và `run.log`; nhưng không sinh thư mục problem của benchmark, nên đây là cách an toàn để xem `--full` sẽ chọn đúng những bài nào) |
| `-h`, `--help` | – | Toàn bộ cờ, ngay trong terminal |

**Exit code:** `0` = xong cả domain, `3` = còn bài chưa xong (chạy lại để tiếp),
`2` = sai tham số.

Xem trước một cấu hình mà không tốn gì:

```bash
./scripts/server/run_lsr_synth.sh --method openevolve_native --domain matsci \
    --full --iterations 10 --dry-run
```

#### 6b.6 Ngân sách — đọc trước khi launch full sweep

~95% wall clock là **chờ LLM**, không phải chấm điểm: một completion của
qwen3-30b mất 15–25 s, còn fit BFGS + chấm một hypothesis chỉ 0.5–2 s.

| method | s/iteration | 500 iterations |
|---|---|---|
| `specevo` | ~3–6 (4 worker song song) | ~40–60 min ✓ vừa |
| `adaevolve` | ~15 | ~2 h — sát mép |
| `openevolve_native` | ~20 | ~2.8 h ✗ bị cắt ở trần 2 h |
| `evox` | ~110 | ~15 h ✗ bị cắt |

Baseline chạy **một iteration tại một thời điểm** theo đúng thiết kế của nó, còn
SpecEvo phát 4 completion song song — toàn bộ khoảng cách 4x ở trên chỉ là chỗ
đó. Nên **wall clock không phải trục so sánh công bằng giữa hai họ; eval budget
mới là.**

Với baseline chậm, hoặc nâng trần thời gian:

```bash
./scripts/server/run_lsr_synth.sh --method evox --domain chem_react \
    --problem-timeout 0 --tmux
```

hoặc — tốt hơn cho kết quả định công bố — cân bằng theo **tiền** thay vì theo
iteration, giống các thực nghiệm khác trong repo:

```bash
./scripts/server/run_lsr_synth.sh --method evox --domain chem_react \
    --dollars 2 --problem-timeout 21600 --tmux
```

Đo thử một bài trước để calibrate: `--problems 1 --iterations 20`.

Muốn nhanh hơn thì chạy **nhiều domain song song** thành nhiều process riêng
(output dir riêng) — mỗi search vẫn tuần tự y nguyên, nên không ảnh hưởng gì đến
tính so sánh được của kết quả.

#### 6b.7 Theo dõi

```bash
# Bảng tiến độ — không gọi API, an toàn gọi bất cứ lúc nào
./scripts/server/run_lsr_synth.sh --method specevo --domain chem_react --status

tail -f outputs/lsr_synth/specevo/chem_react/seed1/run.log
tmux attach -t lsr_specevo_chem_react_seed1        # detach: Ctrl-b rồi d
```

#### 6b.8 Ngắt và tiếp tục

Dừng: `Ctrl-C` trong pane, hoặc `tmux kill-session -t <session>`. Cả `SIGINT`,
`SIGTERM` và `SIGHUP` đều được xử lý: search bị terminate và bài đang chạy **bị
bỏ, không ghi vào kết quả**, nên không có gì nửa vời lọt vào `results.jsonl`.

⚠️ **Đừng `kill -9` runner.** `SIGKILL` không trap được, search sẽ sống sót và
tiếp tục đốt API credit. (Lần chạy sau phát hiện và diệt nó trước khi resume,
nhưng bạn đã trả tiền cho khoảng thời gian đó.)

Tiếp tục: **chạy lại đúng câu lệnh cũ.** Output dir suy ra từ (method, domain,
seed) và không có timestamp, nên bài nào có `.done` sẽ bị bỏ qua; baseline resume
từ checkpoint mới nhất và chỉ chạy đúng số iteration còn nợ; SpecEvo chạy lại bài
đó từ đầu (BLADE không resume giữa search) và attempt cũ được archive vào
`prev_attempt_<ts>/` chứ không bị xoá.

Exit code `3` = còn bài chưa xong, `0` = xong cả domain — dùng được trong loop:

```bash
until ./scripts/server/run_lsr_synth.sh --method specevo --domain chem_react; do
    sleep 60
done
```

Bỏ hết chạy lại từ đầu: `--fresh`.

#### 6b.9 Kết quả

```bash
python scripts/lsr_summarize.py outputs/lsr_synth                    # mọi method/domain
python scripts/lsr_summarize.py outputs/lsr_synth --csv lsr_synth.csv
python scripts/lsr_summarize.py outputs/lsr_synth/specevo/chem_react/seed1
```

Chương trình tìm được sẽ được **chấm lại từ đầu** trên hai tập test giữ riêng:
**ID** (in-domain) và **OOD** (out-of-domain, thời điểm muộn hơn / nhiệt độ,
biến dạng cao hơn). Không tập nào được dùng để dẫn dắt search — chỉ train NMSE.

Chi tiết đầy đủ: [`docs/LSR_SYNTH_GUIDE.md`](LSR_SYNTH_GUIDE.md) — §2.1 chế độ
full, §5.1 score mode (tại sao log hiện `score: 3.0` thay vì `0.999`), §5.2 định
nghĩa và công thức từng metric, §7 xử lý sự cố.

### 6c. RelayEvolve + 5 baseline cheap/strong — dùng `run_relay.sh`

**RelayEvolve** (bài "Relay, Don't Route") không chọn model cho *từng lời gọi*
mà chọn thời điểm **bàn giao cả quần thể**: model rẻ khám phá nhiều quỹ đạo
song song, một *relay bank* gọn (chất lượng + đa dạng) được cập nhật sau mỗi
block, *Relay Gain* (mức cải thiện biên của bank đó) vừa là reward cho bandit
Grow–Deepen vừa là tín hiệu dừng; khi Relay Gain bão hoà, tập seed đã được
curate sẽ khởi tạo **một** quần thể duy nhất cho model mạnh tinh chỉnh nốt
ngân sách.

Toàn bộ 6 method dưới đây chạy trên **cùng backend OpenEvolve** (MAP-Elites +
island), cùng evaluator, cùng prompt, cùng trần iteration và cùng trần tiền —
khác nhau **chỉ ở lịch dùng model**. Tất cả đều chạy **song song nhiều worker**
(mặc định 8 generation cùng lúc), không phải bản 1 luồng.

| `--method` | Là gì |
|---|---|
| `relayevolve` | Cheap khám phá nhiều quỹ đạo → Relay-Gain handoff → strong tinh chỉnh |
| `all_cheap` | Toàn bộ search dùng model nhỏ |
| `all_strong` | Toàn bộ search dùng model lớn |
| `fixed_switch` | Cheap trong một tiền tố cố định của ngân sách, rồi strong |
| `random` | Mỗi generation tung đồng xu độc lập giữa hai model |
| `bandit` | UCB hai tay {cheap, strong}, reward = mức cải thiện best-so-far thực tế |

#### 6c.1 Mặc định

| Thứ | Giá trị |
|---|---|
| Model lớn (strong) | `openrouter/moonshotai/kimi-k2` |
| Model nhỏ (cheap) | `openrouter/qwen/qwen3-30b-a3b-instruct-2507` |
| Iteration | **300** |
| Ngân sách | **$2 / run** — chạm ngưỡng là **bắt buộc dừng** (dừng êm, giữ nguyên kết quả) |
| Timeout mỗi eval | **150 s** |
| Worker song song | 8 |
| Retry | **tắt** (`--retries 1`) |

**Không retry.** Mặc định mỗi generation gọi model đúng **một lần**. Nếu model
sinh ra chương trình hỏng (parse lỗi, hoặc chấm ra `validity=0`), generation đó
coi như đã tiêu và search đi tiếp — thay vì thử lại 2–3 lần trong cùng một
iteration. Hai lợi ích: (1) nhanh hơn hẳn, vì retry chạy *tuần tự* bên trong một
worker và làm nghẽn slot đó; (2) kế toán trung thực — một generation luôn đúng
bằng một lời gọi LLM, nên trần 300 iteration cũng là trần 300 lời gọi. Muốn quay
lại hành vi cũ thì `--retries 3`.

Ba trần (iteration / tiền / thời gian) cùng có hiệu lực; **cái nào chạm trước
thì thắng**. Trần tiền là trần *mềm* ở mức một vòng lặp: các request đã bay đi
vẫn hoàn thành, nên tổng cuối có thể nhỉnh hơn $2 chút xíu (kiểu $2.03). Muốn
chặn cứng tuyệt đối thì thêm `--timeout`.

#### 6c.1b Chạy 300 iteration mất bao lâu, tốn bao nhiêu?

Đo thật trên `benchmarks/math/circle_packing`, 8 worker, `--retries 1`, mỗi
model 16 generation liên tiếp (OpenRouter, tháng 8/2026):

| Model | Trễ / generation | Chi phí / generation | Sinh code hỏng |
|---|---|---|---|
| `qwen3-30b-a3b-instruct` (cheap) | 31 s (19–62 s) | $0.0007 | 5/16 |
| `kimi-k2` (strong) | 74 s (37–179 s) | $0.0099 | 3/16 |

Với 8 worker, thời gian ≈ `300 × trễ / 8`. Cộng thêm ~20–50 % cho đuôi và
jitter của provider:

| Method | 300 generation mất | Tốn | Trần nào chạm trước |
|---|---|---|---|
| `all_cheap` | **20–30 phút** | ~$0.21 | iteration |
| `all_strong` | **30–50 phút** | **$2 → dừng ở ~200 gen** | **tiền** |
| `relayevolve` | **30–45 phút** | ~$1.6 | iteration |
| `fixed_switch` | **30–45 phút** | ~$1.6 | iteration |
| `random` (p=0.5) | **30–45 phút** | ~$1.6 | iteration |
| `bandit` | 25–50 phút | $1–2 | tuỳ arm nào thắng |

Cả 6 method của một benchmark, chạy **song song** trong 12 tmux session
(6 × 2 seed): khoảng **45–70 phút** và **~$16–20**. Nếu chạy tuần tự thì cỡ
6–8 tiếng. Bốn benchmark: nhân 4.

Vài lưu ý đọc bảng:

- **`all_strong` là method duy nhất bị tiền chặn** — $2 chỉ mua được ~200
  generation kimi-k2. Đây đúng là tình huống bài báo dựng ra (so sánh dưới trần
  chi phí chung), không phải lỗi cấu hình.
- Ngân sách cheap của RelayEvolve (`(1-0.85) × $2 = $0.30`) mua được ~430
  generation, nhiều hơn trần 150 generation của pha cheap. Nghĩa là pha cheap
  luôn kết thúc vì **Relay Gain bão hoà** hoặc chạm trần generation, chứ không
  phải vì hết tiền — đúng ý đồ thiết kế.
- Trễ của provider dao động mạnh (kimi-k2 từ 37 s tới 179 s cho cùng một
  prompt), nên coi các con số trên là khoảng ước lượng, không phải cam kết.
- Muốn nhanh hơn: tăng `--workers`. 16 worker gần như chia đôi thời gian, đổi
  lại nguy cơ bị OpenRouter rate-limit khi chạy nhiều job cùng lúc.
- Bảng trên đo trên `circle_packing`, nơi chấm một chương trình chỉ mất ~0.2 s
  nên `--eval-timeout` không ảnh hưởng gì. Các benchmark chấm nặng
  (`circle_packing_rect` chạy differential evolution, `prism`, `txn_scheduling`)
  thì thời gian chấm cộng thẳng vào trễ mỗi generation. Trần 150 s nghĩa là một
  chương trình treo giữ slot worker tối đa 150 s trước khi bị cắt — rộng rãi cho
  chương trình chậm nhưng hợp lệ, mà vẫn không để một vòng lặp vô hạn nuốt cả
  run.

#### 6c.2 Chạy một job

```bash
cd ~/skydiscover
./scripts/server/run_relay.sh --method relayevolve --tmux \
    --benchmark-dir benchmarks/math/circle_packing \
    --iterations 300 --dollars 2 --seed 1
```

Baseline thì chỉ đổi `--method`:

```bash
./scripts/server/run_relay.sh --method all_cheap     --tmux --benchmark-dir benchmarks/math/circle_packing --seed 1
./scripts/server/run_relay.sh --method fixed_switch  --tmux --benchmark-dir benchmarks/math/circle_packing --seed 1
./scripts/server/run_relay.sh --method random        --tmux --benchmark-dir benchmarks/math/circle_packing --seed 1
./scripts/server/run_relay.sh --method bandit        --tmux --benchmark-dir benchmarks/math/circle_packing --seed 1
```

Thử trước khi tốn tiền — in ra đúng dòng lệnh rồi thoát, **không gọi API**:

```bash
./scripts/server/run_relay.sh --method relayevolve --dry-run \
    --benchmark-dir benchmarks/math/circle_packing
```

#### 6c.3 Chạy đủ 6 method × 2 seed trên một benchmark

Mỗi lệnh là một tmux session riêng, chạy **song song**:

```bash
cd ~/skydiscover
BM=benchmarks/math/circle_packing
for m in relayevolve all_cheap all_strong fixed_switch random bandit; do
  for s in 1 2; do
    ./scripts/server/run_relay.sh --method "$m" --tmux \
        --benchmark-dir "$BM" \
        --iterations 300 --dollars 2 --eval-timeout 150 --workers 8 --seed "$s" \
        --session "relay_${m}_cp_seed${s}"
    sleep 2
  done
done
```

12 job × $2 = tối đa **$24** cho một benchmark. Nếu server yếu (hoặc bị
rate-limit), hạ `--workers` xuống 4, hoặc bỏ `--tmux` trong vòng lặp và bọc cả
vòng lặp trong **một** session để chạy tuần tự.

#### 6c.4 Chạy cả 4 benchmark của bài báo

```bash
cd ~/skydiscover
for bm in benchmarks/math/circle_packing \
          benchmarks/math/circle_packing_rect \
          benchmarks/ADRS/txn_scheduling \
          benchmarks/ADRS/prism; do
  tag=$(basename "$bm")
  for m in relayevolve all_cheap all_strong fixed_switch random bandit; do
    for s in 1 2; do
      ./scripts/server/run_relay.sh --method "$m" --tmux \
          --benchmark-dir "$bm" \
          --iterations 300 --dollars 2 --eval-timeout 150 --workers 8 --seed "$s" \
          --session "relay_${m}_${tag}_seed${s}"
      sleep 2
    done
  done
done
```

48 job. **Đừng phóng hết một lúc** trên một server nhỏ: mỗi job giữ 8 request
đồng thời, tức là tối đa 384 request song song tới OpenRouter. Chạy từng
benchmark một, hoặc kẹp `--workers 4`.

#### 6c.5 Theo dõi

```bash
tmux ls                                              # job nào đang sống
tmux attach -t relay_relayevolve_cp_seed1            # xem trực tiếp; thoát: Ctrl-b rồi d
tail -f outputs/server/<run-id>/run.log              # theo log

# Tiêu bao nhiêu tiền rồi
cat outputs/server/<run-id>/cost_log.totals.json

# Báo cáo relay: handoff ở generation nào, vì sao, seed nào được chọn,
# mỗi tier gọi bao nhiêu lần
cat outputs/server/<run-id>/relay_summary.json

# Đường cong cost-vs-score (mỗi generation một dòng JSON)
tail outputs/server/<run-id>/relay_progress.jsonl
```

Dòng log của mỗi generation đã kèm sẵn tiến độ tiêu tiền:
`[cost=$1.2345/$2.00, llm_calls=87]`. Lúc chạm trần có dòng
`💸 Spend budget reached: ...` rồi run dừng êm với **exit status 0**.

Với `relayevolve`, log còn có một dòng cho mỗi block:

```
Relay block 7: DEEPEN traj=2 (+5 gens) | gain=0.0182 rel=0.0231 | bank F=0.7914 | pool=41 | best=2.4188 | $0.1732
🔀 Relay handoff at generation 63: 8 seed(s), best=2.4188, spent $0.2043 (relay_gain_saturated)
```

#### 6c.5a Làm sao biết một run đã xong?

Mọi method — kể cả khi dừng vì **hết tiền** chứ không phải chạy hết generation —
đều kết thúc bằng đúng một khối như thế này:

```
========================================================
 [OK] RUN FINISHED — random on circle_packing (seed 1)
--------------------------------------------------------
 stopped because : generation budget spent
 best score      : 0.850185   (test-mode)
 generations     : 4 of 4
 llm calls       : cheap=2  strong=2
 cost            : $0.0128 of $0.30
 tokens          : in=6,930  out=8,060
 wall clock      : 1m 02s
 results in      : /tmp/relay_banner/random
========================================================
```

Dòng `stopped because` luôn có, và nói rõ **vì sao** run kết thúc:

| Giá trị | Nghĩa |
|---|---|
| `generation budget spent` | Chạy hết `--iterations`. Bình thường. |
| `dollar budget reached ($2.0143 of $2.00)` | **Hết tiền** — dừng sớm, số generation nhỏ hơn trần |
| `interrupted (signal or shutdown request)` | Bị `Ctrl-C` / `tmux kill-session` / hết `--timeout` |
| `stopped before the generation cap` | Vòng lặp kết thúc sớm vì lý do khác |

Vài điểm hay gây nhầm:

- **Đừng đọc số generation mà bỏ qua `stopped because`.** Một run `all_strong`
  dừng ở 198/300 vì hết $2 trông y hệt một run đã hội tụ nếu chỉ nhìn điểm số.
- Dòng `handoff` **chỉ hiện với method thực sự có bàn giao** (`relayevolve`,
  `fixed_switch`). Trước đây baseline in ra `handoff at generation None |
  reason: None`, đọc như bị lỗi — giờ không in nữa.
- Benchmark có thể tự in log của nó (`Optimization failed: DE failed`,
  `Results saved to /tmp/...`) trong lúc chấm điểm lần cuối. Đó là output của
  evaluator, **không phải** lỗi của run. Khối `=====` mới là dấu kết thúc.
- Chạy qua `run_relay.sh` thì sau khối trên còn một footer nữa của script với
  `exit status : 0` và đường dẫn kết quả — đó là dòng cuối cùng thật sự.

#### 6c.5b Quên tên session rồi thì xem kết quả kiểu gì?

**Không cần nhớ tên session.** Tên session chỉ để `tmux attach` lúc job đang
chạy; kết quả nằm trong thư mục run, và tên thư mục đã mã hoá sẵn
method + benchmark + seed + ngày giờ:

```
outputs/server/relay_<method>_<benchmark>_seed<N>_<YYYYmmdd-HHMMSS>/
```

Ba cách, từ nhanh tới chi tiết:

**1. Bảng tổng hợp mọi run** — cách hay dùng nhất:

```bash
python scripts/relay_summarize.py
```

```
method         benchmark              seed        score  gens     cost  cheap strong  handoff  run dir
------------------------------------------------------------------------------------------------------
relayevolve    circle_packing            1     2.536400   300   $1.612    150    150      151  outputs/server/relay_relayevolve_..._seed1_20260826-0913
all_strong     circle_packing            1     2.445300   198   $2.014      0    198        -  outputs/server/relay_all_strong_..._seed1_20260826-0913  [$]
...
[$] = the run stopped because it reached its dollar budget.
```

Lọc lại nếu nhiều quá:

```bash
python scripts/relay_summarize.py --benchmark circle_packing
python scripts/relay_summarize.py --method relayevolve --seed 1
```

**2. Bảng mean ± std theo seed** — đúng dạng bảng kết quả trong bài báo:

```bash
python scripts/relay_summarize.py --agg
```

```
=== circle_packing ===
method          n         mean        std         best   mean $
--------------------------------------------------------------
relayevolve     2     2.529100   0.010325     2.536400   $1.612
all_strong      2     2.441200   0.005800     2.445300   $2.014
```

Xuất ra CSV để vẽ hình: `--csv relay.csv`.

**3. Một run cụ thể, xem đầy đủ:**

```bash
python scripts/relay_summarize.py --path outputs/server/relay_relayevolve_..._seed1_20260826-0913
```

In ra tham số đã dùng, điểm test, chi phí / trần, token, thời gian, handoff ở
generation nào vì lý do gì, và **từng block Grow/Deepen với Relay Gain** —
đúng thứ cần để kiểm tra cơ chế relay có chạy như thiết kế không.

**Các cách khác đã có sẵn:**

```bash
tmux ls                                   # session nào CÒN sống (job đã xong thì mất tên)
./scripts/server/result.sh -l             # 20 run gần nhất, đánh số -1, -2, ...
./scripts/server/result.sh -2             # in kết quả của run gần thứ hai
ls -t outputs/server/ | head -20          # thô nhất, nhưng luôn đúng

# Còn nhớ mang máng tên session? collect_logs.sh tra ngược session -> run dir
./scripts/server/collect_logs.sh relay_relayevolve_cp_seed1
```

Lưu ý: job chạy xong thì tmux session **vẫn còn** (pane hiện
`=== run finished (exit N) ===` và chờ Enter), nên `tmux ls` vẫn thấy. Chỉ khi
bạn `kill-session` hoặc server reboot thì tên mới mất — lúc đó dùng cách 1.

#### 6c.6 File kết quả

| File | Nội dung |
|---|---|
| `best/best_program.py` | Chương trình tốt nhất, đã chấm lại ở chế độ `test` |
| `best/best_program_info.json` | Điểm số của nó |
| `relay_summary.json` | Method, model, handoff, seed, số call theo tier, tổng token/chi phí |
| `relay_progress.jsonl` | Mỗi generation: tier, phase, score, best-so-far, chi phí tích luỹ |
| `cost_log.jsonl` / `.totals.json` | Chi phí từng lời gọi LLM do OpenRouter báo về |
| `checkpoints/checkpoint_N/` | Toàn bộ quần thể tại generation N |
| `run.log` | Log đầy đủ + footer tóm tắt |
| `run_config.json` | Tham số đã dùng cho run này (để tra ngược khi quên) |

#### 6c.7 Bảng tham số `run_relay.sh`

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--method NAME` | `relayevolve` | Một trong 6 method ở bảng trên |
| `--benchmark-dir DIR` | `benchmarks/math/circle_packing` | Thư mục benchmark |
| `--iterations N` | `300` | Trần số generation |
| `--dollars N` | `2` | Trần chi phí USD (0 = tắt) |
| `--eval-timeout N` | `150` | Timeout mỗi lần chấm, giây |
| `--retries N` | `1` | Số lần gọi model mỗi generation. `1` = không retry |
| `--workers N` | `8` | Số generation chạy song song |
| `--seed N` | `1` | Seed; gắn vào tên run và output dir |
| `--strong-model ID` | `openrouter/moonshotai/kimi-k2` | Model lớn |
| `--cheap-model ID` | `openrouter/qwen/qwen3-30b-a3b-instruct-2507` | Model nhỏ |
| `--timeout SPEC` | tắt | Trần thời gian cứng (`3h` / `180m` / `600s` / `0`) |
| `--strong-reserve F` | `0.85` | Phần ngân sách dành cho model mạnh |
| `--block-size N` | `5` | `h` — số generation mỗi block Grow/Deepen |
| `--max-trajectories N` | `5` | Trần số quỹ đạo cheap |
| `--trajectory-horizon N` | `6` | Trần số block mỗi quỹ đạo |
| `--bank-size N` | `8` | `k` — kích thước relay bank = số seed bàn giao |
| `--relay-lambda F` | `0.5` | `λ` — cân bằng chất lượng vs đa dạng trong `F_C(S)` |
| `--epsilon-rel F` | `0.02` | Ngưỡng bão hoà Relay Gain |
| `--patience N` | `3` | Số block gain thấp liên tiếp trước khi handoff |
| `--curation MODE` | `full` | Ablation curation: `full` / `quality` / `diversity` / `random` |
| `--relay-control MODE` | `full` | Ablation cơ chế relay: `full` / `random` / `no_stop` / `random_no_stop` |
| `--switch-fraction F` | `0.5` | Điểm chuyển của `fixed_switch` |
| `--p-strong F` | `0.5` | `random`: xác suất chọn model mạnh |
| `--advanced-options JSON` | — | Override bất kỳ field nào của `search.database` |
| `--tmux` | tắt | Chạy nền trong tmux |
| `--session NAME` | = run id | Tên tmux session |
| `--output-dir DIR` | `outputs/server/<run-id>` | Nơi ghi kết quả |
| `--dry-run` | tắt | In lệnh rồi thoát, không gọi API |

#### 6c.8 Ablation (Figure 4 của bài báo)

```bash
# Cơ chế relay: bỏ bandit / bỏ luật dừng
for c in full random no_stop random_no_stop; do
  ./scripts/server/run_relay.sh --method relayevolve --tmux \
      --benchmark-dir benchmarks/math/circle_packing \
      --relay-control "$c" --session "relay_ctrl_${c}"
done

# Mục tiêu curation: đủ Q+D / chỉ Q / chỉ D / seed ngẫu nhiên
for c in full quality diversity random; do
  ./scripts/server/run_relay.sh --method relayevolve --tmux \
      --benchmark-dir benchmarks/math/circle_packing \
      --curation "$c" --session "relay_cur_${c}"
done

# Độ nhạy theo tỉ lệ chia ngân sách
for r in 0.65 0.75 0.85 0.95; do
  ./scripts/server/run_relay.sh --method relayevolve --tmux \
      --benchmark-dir benchmarks/math/circle_packing \
      --strong-reserve "$r" --session "relay_split_${r/./_}"
done
```

#### 6c.9 Chạy không qua tmux (test nhanh)

```bash
python scripts/run_relay.py --method relayevolve \
    --benchmark-dir benchmarks/math/circle_packing \
    --iterations 10 --dollars 0.3 --workers 4 --eval-timeout 150 --seed 1
```

Test offline (không gọi API, không tốn tiền), stub LLM nhưng chạy thật controller,
population, evaluator và vòng lặp song song:

```bash
python -m pytest tests/search/test_relay.py -q
```

#### 6c.10 Error rate theo 8 khoảng — `scripts/relay_error_rate.py`

Tỉ lệ generation sinh ra code hỏng, chia đều run thành 8 khoảng, gộp theo seed.

**Bước 1 — kéo log về máy.** Chạy trên **máy của bạn**, không phải trên server.
Glob `relay_*_circle_packing*_seed*_*` khớp đúng 2 task cần lấy
(`circle_packing` và `circle_packing_rect`), và 4 file `--include` giữ mỗi run
chỉ vài chục KB — `checkpoints/` bị bỏ hẳn:

```bash
mkdir -p ./outputs/server
rsync -avz --prune-empty-dirs \
    --include='*/' \
    --include='relay_progress.jsonl' --include='relay_summary.json' \
    --include='run_config.json'      --include='run.log' \
    --exclude='*' \
    '<username>@<server>:~/skydiscover/outputs/server/relay_*_circle_packing*_seed*_*' \
    ./outputs/server/
```

Đúng 20 thư mục (5 method × 2 task × 2 seed). Dấu nháy đơn quanh đường dẫn
remote là bắt buộc: glob phải để **shell trên server** khai triển, không phải
shell trên máy bạn.

Muốn chắc trước khi kéo thật thì thêm `--dry-run` vào `rsync`, hoặc đếm trước
trên server:

```bash
ls -d ~/skydiscover/outputs/server/relay_*_circle_packing*_seed*_* | wc -l
```

Chỉ một task, hoặc chỉ một seed:

```bash
# chỉ circle_packing (KHÔNG lấy _rect) — `_seed` chặn ngay sau tên task
'...:~/skydiscover/outputs/server/relay_*_circle_packing_seed*_*'
# chỉ circle_packing_rect
'...:~/skydiscover/outputs/server/relay_*_circle_packing_rect_seed*_*'
# chỉ seed 1
'...:~/skydiscover/outputs/server/relay_*_circle_packing*_seed1_*'
```

**Bước 2 — trích số:**

```bash
python scripts/relay_error_rate.py
```

```
### circle_packing  —  error rate (%) per 1/8 of the run
method                 bin1         bin2         bin3   ...        bin8
RelayEvolve        37.3±6.8     19.6±4.6     41.1±1.9   ...     8.3±4.9
All-cheap          37.8±7.7     28.1±2.5     25.7±6.6   ...     4.2±3.6
...
  RelayEvolve    generations [300, 300], bin sizes [38, 38, ...], overall 24.3%  (seed 1, 2 → padded to 3)
```

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--root DIR` | `outputs/server`, `outputs/relay` | Thư mục cần quét (lặp lại được) |
| `--bins N` | `8` | Số khoảng chia đều theo generation **đã chạy thật** |
| `--methods ...` | 5 baseline | Thêm `all_strong` nếu muốn đủ 6 |
| `--benchmarks ...` | `circle_packing circle_packing_rect` | Task cần lấy |
| `--seed-target N` | `3` | Nhân bản seed cuối cho đủ N seed khi tính mean/std. `0` = tắt |
| `--llm-failures M` | `exclude` | `exclude` (bỏ khỏi cả tử lẫn mẫu) / `error` / `ok` |
| `--detail` | tắt | In thêm 1 dòng mỗi run, kèm số lỗi theo loại |
| `--csv F` / `--json F` | — | Xuất ra để vẽ hình |

**Thế nào là "code lỗi"?** Đúng theo quy ước đã dùng cho ablation SpecEvo: một
generation bị tính là *lỗi* khi model có sinh ra code nhưng code đó không dùng
được — không parse được, quá dài, evaluator ném exception, hoặc chấm quá giờ.
Còn lời gọi API hỏng (rate limit, timeout mạng, worker chết) là nhiễu hạ tầng,
**không sinh ra code nào để đánh giá**, nên mặc định bị loại khỏi cả tử số lẫn
mẫu số và đếm riêng ở cột `api` của `--detail`.

Nguồn số liệu là `relay_progress.jsonl` (mỗi generation một dòng, có trường
`error`). Nếu file đó thiếu, script tự dựng lại từ `run.log` bằng các dòng
`Iteration N failed: ...` và đánh dấu `[from run.log]`.

---

---

## 7. tmux — chạy rồi tắt máy vẫn không sao

Khi bạn thêm cờ `--tmux`, script tự tạo một tmux session trên server rồi chạy
job trong đó, và in ra màn hình:

```
Started in tmux.
  session     : blade_levi_examples_circle_packing_full_seed1_20260723-180513
  output dir  : outputs/server/blade_levi_examples_circle_packing_full_seed1_20260723-180513
  log file    : outputs/server/.../run.log
```

Sau đó bạn có thể **thoát SSH, đóng laptop, mất mạng** — job vẫn chạy tiếp trên
server. Lần sau SSH vào lại, nó vẫn ở đó.

### Các lệnh tmux cần nhớ (chỉ 5 lệnh)

```bash
tmux ls                          # liệt kê các session đang chạy
tmux attach -t <tên-session>     # "chui vào" xem job đang chạy
                                 #   thoát ra: bấm Ctrl-b rồi thả tay, bấm d
tmux kill-session -t <tên>       # dừng hẳn một job
tmux kill-server                 # dừng TẤT CẢ (cẩn thận!)
```

> **Ctrl-b rồi d** là cách thoát đúng ("detach"). Nếu bạn bấm `Ctrl-c` thì bạn
> **giết job** chứ không phải thoát ra. Đây là lỗi phổ biến nhất của người mới.

Đặt tên session cho dễ nhớ bằng `--session`:

```bash
./scripts/server/run_bench.sh blade --tmux --session cp_seed1 --benchmark levi/examples/circle_packing
tmux attach -t cp_seed1
```

Khi job chạy xong, tmux session **vẫn còn** và hiện dòng
`=== run finished (exit 0) — press Enter to close this pane ===`. Đó là cố ý —
để bạn attach vào đọc kết quả. Số trong ngoặc là mã thoát: **0 = chạy trọn vẹn**,
khác 0 = job lỗi. Dọn session bằng cách attach vào rồi bấm **Enter**, hoặc
`tmux kill-session -t <tên>`.

> Lưu ý: khi đang attach mà bấm Enter thì pane đóng và session biến mất. Job
> lúc đó đã xong rồi nên không mất gì, nhưng nếu muốn giữ session lại để xem
> tiếp thì thoát bằng **Ctrl-b rồi d** thay vì Enter.

### Chạy không dùng tmux

Bỏ cờ `--tmux` thì job chạy ngay trước mắt bạn (foreground). Chỉ nên làm vậy
với `--dry-run` hoặc job rất ngắn, vì **đóng SSH là job chết**.

---

## 8. Xem log / xem kết quả

Mọi thứ nằm trong `outputs/server/<run-id>/`, trong đó `<run-id>` có dạng
`<mode>_<benchmark>_<ablation-hoặc-baseline>_seed<N>_<ngày-giờ>`.

> Nếu bạn phóng hai job **y hệt nhau trong cùng một giây**, run-id thứ hai sẽ
> tự thêm hậu tố `_2`, `_3`… nên hai job không bao giờ ghi đè hay trộn log vào
> nhau.

### Xem log đang chạy theo thời gian thực

```bash
tail -f outputs/server/<run-id>/run.log
```

Thoát khỏi `tail -f` bằng `Ctrl-c` (an toàn — nó chỉ dừng việc *xem*, job vẫn chạy).

Không nhớ tên run? Lấy cái mới nhất:

```bash
ls -t outputs/server/ | head -5                       # 5 run gần nhất
tail -f "outputs/server/$(ls -t outputs/server | head -1)/run.log"
```

### Các cách xem khác

```bash
less outputs/server/<run-id>/run.log      # cuộn xem cả file (thoát: q)
grep "Best" outputs/server/<run-id>/run.log        # lọc dòng có điểm tốt nhất
grep -E "Status|Best|cost" outputs/server/<run-id>/run.log | tail -20
tmux attach -t <session>                  # xem trực tiếp trong tmux
```

### Chi phí đã tiêu

```bash
cat outputs/server/<run-id>/cost_log.totals.json      # tổng USD, số token
wc -l outputs/server/<run-id>/cost_log.jsonl          # số lần gọi LLM
```

Phần cuối `run.log` cũng in sẵn bảng tổng kết chi phí + `exit status`
(`exit status : 0` = chạy xong bình thường; khác 0 = có lỗi).

#### Số token (input / output) — cả blade lẫn baseline

Baseline lấy token từ `cost_log.totals.json` (`total_prompt_tokens` /
`total_completion_tokens`). **SpecEvo/BLADE không đi qua wrapper đó**, nên nó tự
đếm và báo trong `summary.json` + mấy dòng cuối `run.log`:

```text
LLM calls         : 640
Input tokens      : 1843201
Output tokens     : 214883
Total tokens      : 2058084
Init input tokens : 402118  (init phase: 105 calls, 105 evals, $0.4821)
Init output tokens: 47330
```

Hai dòng `Init …` tách riêng **pha khởi tạo** — phase 1 (sinh diverse seed) +
phase 2 (sinh variants cho mỗi seed), tính tới đúng lúc bootstrap xong. Phần
của vòng lặp tiến hoá = tổng trừ init (init là một *tiền tố* của run, không
chồng lấn).

```bash
# Đọc thẳng bằng máy
python -c "import json,sys; s=json.load(open(sys.argv[1])); \
print(s['total_prompt_tokens'], s['total_completion_tokens'], s['init_usage'])" \
  outputs/server/<run-id>/summary.json
```

Cùng bộ số này cũng nằm trong `snapshot.json` và khối `final` của `snap.json`.
Với LSR-Synth, `results.jsonl` có thêm `prompt_tokens` / `completion_tokens` /
`init_usage` cho **mọi** method, nên so token giữa SpecEvo và baseline được.

Lưu ý: token **embedding** không nằm trong các con số này (cả hai phía đều
không đếm) — đây là token sinh văn bản.

### Kết quả chương trình tốt nhất

- Baseline: `outputs/server/<run-id>/run/best/` và `run/checkpoints/`
- BLADE: các file kết quả nằm thẳng trong `outputs/server/<run-id>/`

### Lấy nhanh điểm cuối cùng: `scripts/server/result.sh`

Thay cho việc `tail -f` rồi tự Ctrl-F tìm "New best", dùng script này. Nó tự
nhận biết log là BLADE hay baseline và làm đúng việc:

```bash
./scripts/server/result.sh                      # run mới nhất, tự động
./scripts/server/result.sh -2                   # run gần thứ 2 (không cần gõ tên log)
./scripts/server/result.sh --recent 3           # in luôn 3 run gần nhất, mỗi cái 1 header
./scripts/server/result.sh <đường-dẫn-run.log>  # chỉ định log cụ thể
./scripts/server/result.sh --list               # liệt kê các run gần đây (kèm số -1 -2 -3)
```

**Không cần gõ tên log** — tham chiếu theo độ mới: `-1` là mới nhất, `-2` là
trước đó… Chạy `--list` để xem danh sách đánh số. Mỗi lần in đều có dòng
`# log: ...` ở trên cho biết kết quả lấy từ file nào.

```bash
./scripts/server/result.sh -1          # = mặc định, run mới nhất
./scripts/server/result.sh -3 -B 1     # run gần thứ 3, kèm 1 dòng test score phía trên
./scripts/server/result.sh --recent 5  # 5 run gần nhất một thể, mỗi run có header + trạng thái
```

`--recent` tự nhận diện từng log riêng, nên một loạt trộn cả BLADE lẫn baseline
vẫn hiển thị đúng kiểu cho từng cái.

**Với baseline / CO-Bench** — nó tìm dòng `New best` **cuối cùng** và in kèm
mấy dòng phía trên (chính là dòng `Dev Score | Test Score | Overall`). Dòng có
dấu `>>` là dòng khớp keyword, các dòng còn lại là ngữ cảnh:

```bash
./scripts/server/result.sh                  # mặc định: New best cuối + 3 dòng trên
./scripts/server/result.sh -B 1 -A 0        # chỉ 1 dòng trên (đúng dòng test score)
./scripts/server/result.sh -n 3             # 3 lần New best cuối, không chỉ 1
./scripts/server/result.sh -n all           # tất cả các lần New best
```

**Với BLADE** — nó in **cả khối báo cáo cuối** (`Best score`, `Total cost`,
`Runtime`…) từ đầu báo cáo tới hết file, nên không bao giờ bị cắt cụt như
`tail -N`:

```bash
./scripts/server/result.sh path/to/blade/run.log   # tự nhận ra là BLADE
```

**Tìm keyword bất kỳ (linh hoạt hoàn toàn):** hai cờ `-B` (before, số dòng
phía trên) và `-A` (after, phía dưới) giống hệt `grep`, gõ dính hay tách đều
được (`-B3` hoặc `-B 3`):

```bash
./scripts/server/result.sh -k "Test Score" -B0 -A0      # chỉ các dòng Test Score
./scripts/server/result.sh -k "🌟" -B2                   # emoji cũng được
./scripts/server/result.sh -k "combined_score" -n all   # mọi lần xuất hiện
./scripts/server/result.sh --from "Runtime"             # in từ dòng 'Runtime' cuối tới hết
```

Các cờ chính:

| Cờ | Ý nghĩa | Mặc định |
| --- | --- | --- |
| `-N` (vd `-2`) | Chọn run gần thứ N thay vì gõ tên log | `-1` (mới nhất) |
| `-R, --recent N` | In N run gần nhất một thể, mỗi cái 1 header | — |
| `-k, --keyword RE` | Từ khoá cần tìm (regex) | tự nhận: `New best` cho baseline |
| `-B, --before N` | Số dòng in phía **trên** mỗi lần khớp | `3` |
| `-A, --after N` | Số dòng in phía **dưới** mỗi lần khớp | `1` |
| `-n, --num N` | Chỉ lấy N lần khớp **cuối** (`all` = tất cả) | `1` |
| `--from RE` | In từ dòng cuối khớp RE tới hết file (tail linh hoạt) | — |
| `--report` | Ép chế độ báo cáo BLADE (= `--from 'Best score'`) | — |
| `-f, --follow` | In xong rồi `tail -f` theo dõi tiếp | tắt |
| `-l, --list` | Liệt kê các run gần đây rồi thoát | — |

---

## 9. Lấy kết quả từ server về máy

Chạy trên **máy của bạn** (không phải trên server):

```bash
# Kéo về một run cụ thể
scp -r <username>@<server>:~/skydiscover/outputs/server/<run-id> ./ket_qua/

# Kéo về toàn bộ (dùng rsync, chỉ tải phần thay đổi — chạy lại nhiều lần rất nhanh)
rsync -avz <username>@<server>:~/skydiscover/outputs/server/ ./outputs/server/
```

Sau đó vẽ hình bằng các script sẵn có trên máy bạn (xem `scripts/plots/`).

---

## 10. Bảng tra tham số đầy đủ

### Tham số dùng chung

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--seed N` | `1` | Nhãn seed; chỉ đặt tên run/thư mục, **không** đổi cách chạy (giống hệt trên GitHub Actions) |
| `--cobench-timeout N` | `10` | CO-Bench: giới hạn giây cho mỗi instance |
| `--cobench-max-cases N` | tất cả | CO-Bench: số file test-case tối đa |
| `--cobench-max-instances N` | `3` | CO-Bench: số instance mỗi file (`0` = toàn bộ test set) |
| `--timeout SPEC` | `3h` cho baseline, **không giới hạn** cho blade | Trần thời gian thực. Hết giờ thì job bị SIGTERM (30 giây sau là SIGKILL) và exit status thành **124**. Nhận `10800`, `600s`, `180m`, `3h`; `0` hoặc `none` để tắt |
| `--tmux` | tắt | Chạy nền trong tmux |
| `--session NAME` | tự sinh | Tên tmux session |
| `--output-dir DIR` | `outputs/server/<run-id>` | Nơi lưu kết quả |
| `--run-id NAME` | tự sinh | Đặt tên run theo ý bạn |
| `--conda-env NAME` | `minhhieu` | Env conda cần kích hoạt |
| `--no-conda` | – | Không đụng vào conda, dùng python đang active |
| `--no-install-deps` | – | Bỏ qua bước tự cài dependency của benchmark (nhanh hơn khi chạy lại) |
| `--dry-run` | – | Chỉ in lệnh, không chạy, không tốn tiền |

### Tham số riêng của `blade` (khớp `blade.yml`)

| Cờ | Mặc định |
|---|---|
| `--benchmark PATH` | `levi/examples/circle_packing` |
| `--evaluations N` | rỗng (không giới hạn) |
| `--dollars N` | rỗng (tắt) |
| `--seconds N` | rỗng (tắt) |
| `--mutation-model ID` | `openrouter/qwen/qwen3-30b-a3b-instruct-2507` |
| `--paradigm-model ID` | `openrouter/openai/gpt-5` |
| `--workers N` | `4` |
| `--pe-interval N` | `10` |
| `--eval-timeout N` | `600` |
| `--n-diverse-seeds N` | `5` |
| `--n-variants-per-seed N` | `20` |
| `--ablation NAME` | `full` |
| `--advanced-options JSON` | rỗng |

**Các giá trị `--ablation` hợp lệ:** `full`, `ast_only`, `emb_only`,
`static_cells`, `no_meta_advice`, `meta_errors_only`, `no_targeted_mutate`,
`no_crossover`, `no_paradigm`.

**Các khoá dùng được trong `--advanced-options`:** `problem_module`,
`target_score`, `embedding_model`, `eval_processes`, `n_paradigm_variants`,
`paradigm_n_anchors`, `paradigm_n_inspirations`, `n_cells`, `recluster_every`,
`embedding_dim`, `meta_advice_disabled`, `meta_advice_interval`,
`meta_advice_mode` (`rich`|`errors_only`), `meta_advice_inject_p`,
`targeted_mutate_disabled`, `analyzer_interval`, `analyzer_top_k`,
`p_targeted_mutate`, `p_crossover`, `paradigm_synthesis_max_stagnation`,
`paradigm_surgical_max_stagnation`, `paradigm_synthesis_n_anchors`,
`paradigm_shift_n_anchors`, `paradigm_surgical_n_inspirations`,
`single_prompt_operators`, `paradigm_force_mode`.

**Chạy nhiều trục ablation cùng lúc.** `--ablation` chỉ nhận **một** tên, và
workflow `blade_ablation.yml` cũng chọn đúng một trục mỗi run. Muốn gộp nhiều
trục thì bật chúng qua `--advanced-options`, các khoá này kết hợp tự do:

| trục | khoá | tương đương cờ |
|---|---|---|
| A4 — bỏ Advisor | `"meta_advice_disabled": true` | `--no-meta-advice` |
| A6 — Speculator một prompt mỗi operator | `"single_prompt_operators": true` | `--single-prompt-operators` |
| A8 — Navigator khoá một mode | `"paradigm_force_mode": "reframe"` | `--paradigm-force-mode shift` |
| A5 — bỏ targeted-mutate | `"targeted_mutate_disabled": true` | `--no-targeted-mutate` |
| A7 — bỏ crossover | `"p_crossover": 0` | `--p-crossover 0` |

`paradigm_force_mode` nhận `reframe` (tên trong paper — script tự đổi sang tên
nội bộ `shift`), `synthesis`, hoặc `surgical`.

⚠️ **A6 một mình KHÔNG cho Speculator một prompt duy nhất.** Nó chỉ thu mỗi
operator về một template; lúc chạy vẫn còn ba đường prompt sống: mutate general,
crossover structural (xác suất `p_crossover` = 0.35) và targeted-mutate (xác
suất `p_targeted_mutate` = 0.5 khi parent đã có analysis). Muốn **đúng một**
prompt (`MUTATE_PROMPT_GENERAL`) thì phải tắt cả hai đường kia:

```bash
# Bỏ Advisor + Navigator chỉ reframe + Speculator đúng 1 prompt duy nhất
./scripts/server/run_bench.sh blade --tmux \
    --session eplb_noadv_reframe_single \
    --benchmark levi/examples/ADRS/eplb --seconds 10800 \
    --advanced-options '{"meta_advice_disabled":true,"paradigm_force_mode":"reframe","single_prompt_operators":true,"p_crossover":0,"targeted_mutate_disabled":true}'
```

Hai loại prompt vẫn còn sống sau cấu hình trên, **theo thiết kế**: prompt của
pha khởi tạo (`build_diverse_seed_prompt` / `build_init_variant_prompt` — không
có chúng thì không có quần thể ban đầu) và prompt fanout của Navigator
(`build_paradigm_variant_prompt`, dùng chung cho cả ba mode). Muốn tắt luôn
fanout thì đặt `"n_paradigm_variants": 0`.

Ngữ nghĩa của cấu hình này được khoá bằng test trong
`levi/tests/blade/test_lean_ablation_combo.py`, và đường JSON → cờ CLI được
`scripts/server/selftest.sh` (mục 5) kiểm tra.

Xem danh sách benchmark có sẵn: `ls levi/examples/` và `ls levi/examples/co_bench/`.

### Tham số riêng của `baseline` (khớp `baseline.yml`)

| Cờ | Mặc định |
|---|---|
| `--baseline NAME` | `evox` — chọn trong `openevolve_native`, `gepa_native`, `adaevolve`, `evox` |
| `--benchmark-dir DIR` | `benchmarks/math/circle_packing` |
| `--iterations N` | `100` |
| `--model ID` | `openrouter/openai/gpt-5` |
| `--dollars N` | rỗng (tắt) — trần chi phí USD, xem mục dưới |

Xem danh sách benchmark: `ls benchmarks/math/` và `ls benchmarks/co_bench/`.

---

## 11. Sự cố thường gặp

**`conda: command not found`**
→ Chưa cài conda hoặc chưa nạp lại shell. Làm lại bước 5a, nhớ `exec $SHELL -l`.

**`could not activate conda env 'minhhieu'`**
→ Chưa tạo env. Chạy `bash scripts/server/setup_env.sh`.

**`ERROR: no API key found`**
→ Chưa có file `.env`, hoặc file rỗng, hoặc bạn đang đứng sai thư mục. Kiểm tra:
```bash
cd ~/skydiscover && ls -la .env && head -c 20 .env
```

**Không chắc lỗi ở đâu**
→ Chạy `bash scripts/server/selftest.sh` trước tiên. Nó kiểm tra toàn bộ chuỗi
setup (conda → thư viện → `.env` → tmux) và chỉ đúng chỗ hỏng, hoàn toàn miễn phí.

**`tmux: command not found`**
→ Cài tmux theo bước 5e.

**`a tmux session named '...' already exists`**
→ Session cũ còn sót. Xem bằng `tmux ls`, rồi hoặc `tmux kill-session -t <tên>`,
hoặc chạy lại với `--session <tên-khác>`.

**Đóng SSH xong quay lại thì job biến mất**
→ Bạn quên cờ `--tmux`. Không có nó, job chết theo phiên SSH.

**Nhỡ bấm `Ctrl-c` trong lúc attach vào tmux**
→ Job đã bị giết. Chạy lại. Lần sau thoát bằng **Ctrl-b rồi d**.

**`ModuleNotFoundError` khi chạy một benchmark cụ thể**
→ Benchmark đó cần thư viện nặng. Chạy
`bash scripts/server/setup_env.sh --extra math` (hoặc `--extra adrs` / `--extra torch`),
hoặc kiểm tra chắc chắn bạn **không** truyền `--no-install-deps`.

**`git pull` báo conflict**
→
```bash
cd ~/skydiscover && git checkout -- . && git pull
```
(An toàn vì server không phải nơi bạn sửa code.)

**Server chậm / hết RAM khi chạy nhiều job**
→ Giảm `--workers` và `advanced_options.eval_processes`. Xem tài nguyên bằng
`htop` (hoặc `top`), xem dung lượng đĩa bằng `df -h`.

**Muốn biết job có còn chạy không**
```bash
tmux ls                                   # session còn sống?
tail -3 outputs/server/<run-id>/run.log   # log có tiến triển không?
```
