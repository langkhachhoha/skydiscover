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
